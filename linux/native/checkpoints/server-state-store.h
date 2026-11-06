#pragma once

#include "json.h"

#define XXH_INLINE_ALL
#include "vendor/hash/xxhash/xxhash.h"
#include "shared-cache-quota.h"
#include <memory>

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <deque>
#include <cstdio>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

// The metadata rename commits a snapshot. A partial blob is never indexed.
// Native context serialization owns the payload format; this layer checks all
// bytes before a payload can reach the engine's state loader.
struct server_state_store {
    struct entry {
        std::string name;
        uint64_t bytes = 0;
        std::string checksum;
        common_json metadata;
    };

    std::filesystem::path directory;
    std::string identity;
    uint64_t limit_bytes;
    std::unique_ptr<shared_cache_quota> shared_quota;
    std::list<entry> entries;
    // Only filesystem work runs in these jobs. GPU serialization stays on the
    // inference thread; completed entries are published by drain() on that thread.
    struct pending_job { common_json metadata; std::future<entry> future; };
    std::deque<pending_job> pending;

    void drain(bool wait = false) {
        while (!pending.empty()) {
            auto & job = pending.front().future;
            if (!wait && job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) break;
            try { entries.push_back(job.get()); }
            catch (const std::exception & e) { std::fprintf(stderr, "SSD commit failed: %s\n", e.what()); }
            pending.pop_front();
            prune();
        }
    }


    static std::string digest(const std::filesystem::path & path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { throw std::runtime_error("Cannot read state payload"); }
        XXH3_state_t state;
        XXH3_128bits_reset(&state);
        std::vector<char> buffer(4 * 1024 * 1024);
        while (in) {
            in.read(buffer.data(), buffer.size());
            XXH3_128bits_update(&state, buffer.data(), static_cast<size_t>(in.gcount()));
        }
        if (!in.eof()) { throw std::runtime_error("Incomplete state payload read"); }
        const XXH128_hash_t hash = XXH3_128bits_digest(&state);
        XXH128_canonical_t canonical;
        XXH128_canonicalFromHash(&canonical, hash);
        static constexpr char hex[] = "0123456789abcdef";
        std::string result;
        for (auto byte : canonical.digest) {
            result += hex[byte >> 4];
            result += hex[byte & 15];
        }
        return result;
    }

    static void sync_path(const std::filesystem::path & path, bool is_directory = false) {
#ifndef _WIN32
        const int fd = open(path.c_str(), O_RDONLY | (is_directory ? O_DIRECTORY : 0));
        if (fd < 0) { throw std::runtime_error("Cannot open state path for sync"); }
        const int result = fsync(fd);
        close(fd);
        if (result != 0) { throw std::runtime_error("Cannot sync state path"); }
#else
        (void) path;
        (void) is_directory;
#endif
    }

    server_state_store(const std::string & root, const std::string & id, uint64_t limit) :
        identity(id), limit_bytes(limit) {
        if (id.size() != 64 || id.find_first_not_of("0123456789abcdef") != std::string::npos || limit == 0) {
            throw std::invalid_argument("SSD cache needs a SHA-256 identity and positive byte limit");
        }
        if (const char * roots = std::getenv("LLAMA_CACHE_SSD_SHARED_ROOTS")) {
            const char * lock = std::getenv("LLAMA_CACHE_SSD_SHARED_LOCK");
            const char * cap = std::getenv("LLAMA_CACHE_SSD_SHARED_BYTES");
            if (!lock || !cap) throw std::invalid_argument("Incomplete shared SSD quota configuration");
            std::vector<std::filesystem::path> paths;
            for (const auto & p : quota_json::parse(roots)) paths.emplace_back(p.get<std::string>());
            shared_quota = std::make_unique<shared_cache_quota>(paths,lock,std::stoull(cap));
            shared_quota->recover_abandoned();
        }
        directory = std::filesystem::u8path(root) / identity;
        std::filesystem::create_directories(directory);
        std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
        std::vector<std::pair<std::filesystem::file_time_type, entry>> recovered;
        for (const auto & file : std::filesystem::directory_iterator(directory)) {
            if (file.path().extension() != ".json" || !file.is_regular_file()) { continue; }
            try {
                if (file.file_size() > 32 * 1024 * 1024) { continue; }
                std::ifstream stream(file.path());
                std::string text((std::istreambuf_iterator<char>(stream)), {});
                const auto value = common_json::parse(text);
                const auto name = file.path().stem().string();
                if (name.empty() || name.find_first_not_of("0123456789-") != std::string::npos ||
                    value.at("version") != 1 || value.at("identity") != identity) { continue; }
                entry cur{name, value.at("bytes").get<uint64_t>(), value.at("checksum").get<std::string>(), value.at("metadata")};
                if (cur.checksum.size() != 32 || cur.bytes == 0 ||
                    !std::filesystem::is_regular_file(payload(cur)) ||
                    std::filesystem::file_size(payload(cur)) != cur.bytes) { continue; }
                recovered.emplace_back(file.last_write_time(), std::move(cur));
            } catch (const std::exception &) { /* An incomplete entry is a cold miss. */ }
        }
        std::sort(recovered.begin(), recovered.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (auto & item : recovered) { entries.push_back(std::move(item.second)); }
        prune();
    }

    std::filesystem::path payload(const entry & item) const { return directory / (item.name + ".bin"); }

    bool valid(const entry & item) const {
        try {
            return std::filesystem::file_size(payload(item)) == item.bytes && digest(payload(item)) == item.checksum;
        } catch (const std::exception &) { return false; }
    }

    uint64_t size() const {
        uint64_t total = 0;
        for (const auto & item : entries) { total += item.bytes; }
        return total;
    }

    void erase(std::list<entry>::iterator it) {
        std::error_code ec;
        std::filesystem::remove(directory / (it->name + ".json"), ec);
        std::filesystem::remove(payload(*it), ec);
        entries.erase(it);
    }

    void touch(std::list<entry>::iterator it) {
        std::error_code ec;
        std::filesystem::last_write_time(directory / (it->name + ".json"), std::filesystem::file_time_type::clock::now(), ec);
        entries.splice(entries.end(), entries, it);
    }

    void prune() {
        uint64_t total = size();
        while (!entries.empty() && total > limit_bytes) {
            total -= entries.front().bytes;
            erase(entries.begin());
        }
    }

    // writer must close every output stream before returning its metadata.
    // The immutable .part payload is then checksummed/fsynced/committed off-thread.
    void save(const std::function<common_json(const std::string &)> & writer, uint64_t expected_bytes = 0) {
        drain();
        if (pending.size() >= 2) drain(true);
        static uint64_t sequence = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        entry item;
        item.name = std::to_string(stamp) + "-" + std::to_string(++sequence);
        const auto part = directory / (item.name + ".part");
        std::filesystem::path reservation;
        if (shared_quota) reservation = shared_quota->reserve(directory,item.name,expected_bytes);
        try {
            item.metadata = writer(part.string());
            item.bytes = std::filesystem::file_size(part);
            if (item.bytes == 0 || item.bytes > limit_bytes) throw std::runtime_error("State exceeds SSD cache capacity");
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(part, ec);
            if (shared_quota) shared_quota->release(reservation);
            throw;
        }
        const auto destination = payload(item);
        const auto meta_part = directory / (item.name + ".json.part");
        const auto meta_path = directory / (item.name + ".json");
        auto pending_metadata = item.metadata;
        pending.push_back({std::move(pending_metadata), std::async(std::launch::async,
            [item = std::move(item), part, destination, meta_part, meta_path,
             directory = directory, identity = identity, quota = shared_quota.get(), reservation]() mutable {
            try {
                item.checksum = digest(part);
                sync_path(part);
                std::filesystem::rename(part, destination);
                const common_json record = {{"version", 1}, {"identity", identity}, {"bytes", item.bytes},
                                            {"checksum", item.checksum}, {"metadata", item.metadata}};
                {
                    std::ofstream out(meta_part, std::ios::binary);
                    out << record.dump(); out.flush();
                    if (!out) throw std::runtime_error("Cannot write state metadata");
                }
                sync_path(meta_part);
                std::filesystem::rename(meta_part, meta_path);
                sync_path(directory, true);
                if (quota) quota->release(reservation);
                return item;
            } catch (...) {
                std::error_code ec;
                std::filesystem::remove(part, ec);
                std::filesystem::remove(meta_part, ec);
                if (!std::filesystem::exists(meta_path)) std::filesystem::remove(destination, ec);
                if (quota) quota->release(reservation);
                throw;
            }
        })});
    }
};
