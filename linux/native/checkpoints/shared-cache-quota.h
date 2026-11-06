#pragma once
#include "nlohmann/json.hpp"
using quota_json = nlohmann::ordered_json;
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <sstream>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

// One quota for model namespaces, sessions, committed KV, recurrent bundles,
// and in-flight writes. No payload hashing, inference state or GPU interaction.
struct shared_cache_quota {

    using path = std::filesystem::path;
    struct lock {
        int fd;
        explicit lock(const path & file) : fd(::open(file.c_str(), O_CREAT|O_RDWR|O_CLOEXEC, 0600)) {
            if (fd < 0 || flock(fd, LOCK_EX) != 0) {
                if(fd>=0) ::close(fd);
                throw std::runtime_error("Cannot lock shared SSD quota");
            }
        }
        ~lock() { flock(fd,LOCK_UN); ::close(fd); }
        lock(const lock&)=delete;
        lock& operator=(const lock&)=delete;
    };
    struct group {
        std::filesystem::file_time_type last_use{};
        std::vector<path> files;
        bool initialized=false;
    };
    std::vector<path> roots;
    path lock_path;
    uint64_t maximum;

    shared_cache_quota(std::vector<path> roots_, path lock_, uint64_t maximum_)
        : roots(std::move(roots_)), lock_path(std::move(lock_)), maximum(maximum_) {
        if (!maximum) throw std::invalid_argument("SSD quota must be positive");
        std::filesystem::create_directories(lock_path.parent_path());
    }
    static std::string process_start(int64_t pid) {
        std::ifstream in("/proc/"+std::to_string(pid)+"/stat");std::string text;
        if(!std::getline(in,text))return {};
        auto end=text.rfind(')');if(end==std::string::npos)return {};
        std::istringstream fields(text.substr(end+2));std::string value;
        for(int i=0;i<20;++i)if(!(fields>>value))return {};
        return value;
    }
    void recover_abandoned() {
        lock held(lock_path);
        for(const auto & root:roots) {
            if(!std::filesystem::exists(root) || std::filesystem::is_symlink(root))continue;
            std::vector<path> abandoned;
            for(const auto & item:std::filesystem::recursive_directory_iterator(root)) {
                if(item.is_symlink() || !item.is_regular_file() || !namespace_name(item.path().parent_path()) || item.path().extension()!=".reserve")continue;
                auto meta=read_json(item.path());
                if(meta.value("kind",std::string())!="shared-kv-reservation-v1")continue;
                auto start=process_start(meta.at("owner_pid").get<int64_t>());
                if(start.empty() || start!=meta.at("owner_start").get<std::string>())abandoned.push_back(item.path());
            }
            for(const auto & file:abandoned) {
                const auto stem=file.parent_path()/file.stem();
                for(const auto & suffix:{".part",".json.part"})std::filesystem::remove(stem.string()+suffix);
                if(!std::filesystem::exists(stem.string()+".json"))std::filesystem::remove(stem.string()+".bin");
                std::filesystem::remove(file);
            }
        }
    }
    static bool namespace_name(const path & p) {
        const auto s=p.filename().string();
        return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
    }
    static bool numeric_name(const std::string & s) {
        return !s.empty() && s.find_first_not_of("0123456789-")==std::string::npos;
    }
    static quota_json read_json(const path & file) {
        if(std::filesystem::file_size(file)>32*1024*1024) throw std::runtime_error("Oversized cache metadata");
        std::ifstream in(file); quota_json value; in>>value; return value;
    }
    static uint64_t bytes(const path & p) {
        std::error_code ec;
        if(std::filesystem::is_symlink(p,ec) || !std::filesystem::is_regular_file(p,ec)) return 0;
        const auto n=std::filesystem::file_size(p,ec); return ec ? 0 : n;
    }
    // Count reservations once, rather than charging both a reservation and its
    // already-written payload. Unknown/orphan files still consume quota.
    uint64_t inspect(std::map<std::string,group> * evictable=nullptr) const {
        std::set<path> files, accounted;
        for(const auto & root:roots) {
            if(!std::filesystem::exists(root) || std::filesystem::is_symlink(root)) continue;
            for(const auto & item:std::filesystem::recursive_directory_iterator(root)) {
                if(item.is_symlink() || !item.is_regular_file() || !namespace_name(item.path().parent_path())) continue;
                files.insert(std::filesystem::absolute(item.path()));
            }
        }
        uint64_t total=0;
        for(const auto & file:files) {
            if(file.extension()!=".reserve" || !numeric_name(file.stem().string())) continue;
            auto meta=read_json(file);
            const uint64_t expected=meta.at("reserved_bytes").get<uint64_t>();
            uint64_t actual=0;
            for(const auto & suffix:{".part",".bin",".json.part",".json"}) {
                path p=file.parent_path()/(file.stem().string()+suffix);
                actual+=bytes(p);accounted.insert(p);
            }
            total+=std::max(expected,actual)+bytes(file);accounted.insert(file);
        }
        for(const auto & file:files) {
            if(!accounted.count(file)) total+=bytes(file);
            if(!evictable || accounted.count(file) || file.extension()!=".json" || !numeric_name(file.stem().string())) continue;
            try {
                auto meta=read_json(file);
                if(meta.at("version")!=1 || meta.at("identity")!=file.parent_path().filename().string()) continue;
                const auto inner=meta.value("metadata",quota_json::object());
                const auto session=inner.value("session_id",file.stem().string());
                auto & g=(*evictable)[file.parent_path().string()+"/"+session];
                auto stamp=std::filesystem::last_write_time(file);
                if(!g.initialized || stamp>g.last_use) g.last_use=stamp;
                g.initialized=true;
                g.files.push_back(file); // Commit marker removed before data.
                g.files.push_back(file.parent_path()/(file.stem().string()+".bin"));
                g.files.push_back(file.parent_path()/(file.stem().string()+".bin.zst"));
            } catch(const std::exception &) { /* Unknown data counts but is never deleted. */ }
        }
        return total;
    }
    // Caller reserves before opening/truncating payload files. Reservation names
    // are native-generated, never accepted from the public API.
    path reserve(const path & directory, const std::string & name, uint64_t expected,
                 const std::set<path> & protected_files={}) {
        if(!namespace_name(directory) || !numeric_name(name) || !expected || expected>maximum)
            throw std::invalid_argument("Invalid SSD write reservation");
        const auto dest=std::filesystem::absolute(directory/(name+".reserve"));
        bool allowed=false;
        for(const auto & root:roots) {
            auto rel=dest.lexically_relative(std::filesystem::absolute(root));
            if(!rel.empty() && *rel.begin()!="..") allowed=true;
        }
        if(!allowed) throw std::invalid_argument("Reservation outside managed cache roots");
        lock held(lock_path);
        if(std::filesystem::exists(dest)) throw std::runtime_error("Duplicate SSD reservation");
        quota_json meta={{"kind","shared-kv-reservation-v1"},{"reserved_bytes",expected},{"owner_pid",int64_t(getpid())},{"owner_start",process_start(getpid())}};
        const auto encoded=meta.dump();
        const uint64_t required=expected+encoded.size();
        if(required>maximum) throw std::runtime_error("Snapshot exceeds shared SSD quota");
        std::map<std::string,group> groups;
        auto total=inspect(&groups);
        std::vector<group> ordered;
        for(auto & item:groups) ordered.push_back(std::move(item.second));
        std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){return a.last_use<b.last_use;});
        for(const auto & g:ordered) {
            if(total<=maximum-required) break;
            bool protected_group=false;
            for(const auto & f:g.files) if(protected_files.count(f)) protected_group=true;
            if(protected_group) continue;
            for(const auto & f:g.files) {
                if(std::filesystem::is_symlink(f)) continue;
                auto n=bytes(f);
                if(std::filesystem::remove(f)) total-=std::min(total,n);
            }
        }
        if(total>maximum-required) throw std::runtime_error("Shared SSD quota occupied by protected or pending data");
        const path temporary=dest.string()+".part";
        try {
            std::ofstream stream(temporary,std::ios::binary|std::ios::trunc);
            stream<<encoded;stream.close();
            if(!stream)throw std::runtime_error("Cannot write SSD reservation");
            int fd=::open(temporary.c_str(),O_RDONLY|O_CLOEXEC);
            if(fd<0)throw std::runtime_error("Cannot open SSD reservation for sync");
            int status=fsync(fd);::close(fd);
            if(status)throw std::runtime_error("Cannot sync SSD reservation");
            std::filesystem::rename(temporary,dest);
            fd=::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
            if(fd<0)throw std::runtime_error("Cannot sync SSD reservation directory");
            status=fsync(fd);::close(fd);
            if(status)throw std::runtime_error("Cannot sync SSD reservation directory");
        } catch(...) {
            std::filesystem::remove(temporary);std::filesystem::remove(dest);throw;
        }
        return dest;
    }
    void release(const path & reservation) {
        lock held(lock_path);
        std::filesystem::remove(reservation);
    }

};
