#pragma once
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>
#include <map>
#include <string>

namespace recurrent_checkpoint {
constexpr int64_t checkpoint_spacing = 32768;
constexpr size_t checkpoint_count = 6;
using ancestor_limits = std::map<std::string,int64_t>;

inline bool lineage_allows(const std::string & current, const ancestor_limits & ancestors,
                           const std::string & owner, int64_t position) {
    if(owner.empty() || position<=0)return false;
    if(owner==current)return true;
    const auto found=ancestors.find(owner);
    return found!=ancestors.end() && position<=found->second;
}

inline ancestor_limits inherited_prefixes(const std::string & current, const ancestor_limits & ancestors,
                                          int64_t kept, const std::vector<std::pair<std::string,int64_t>> & available) {
    ancestor_limits result;
    for(const auto & cp:available) {
        if(cp.second>kept || !lineage_allows(current,ancestors,cp.first,cp.second))continue;
        result[cp.first]=cp.first==current ? kept : std::min(kept,ancestors.at(cp.first));
    }
    return result;
}

inline std::set<std::string> retained_lru(const std::vector<std::pair<int64_t,std::string>> & ordered) {
    std::set<int64_t> positions;
    std::set<std::string> result;
    for(auto it=ordered.rbegin();it!=ordered.rend() && result.size()<checkpoint_count;++it) {
        if(positions.insert(it->first).second)result.insert(it->second);
    }
    return result;
}

// Production policy: preserve throughput by capturing only at an already safe
// execution boundary, never splitting a prefill batch to manufacture a point.
inline bool capture_due(int64_t previous, int64_t evaluated) {
    return previous >= 0 && evaluated >= previous && evaluated - previous >= checkpoint_spacing;
}
inline std::set<int64_t> retained_positions(const std::vector<int64_t> & committed) {
    std::set<int64_t, std::greater<int64_t>> sorted;
    for (const auto pos : committed) {
        if (pos <= 0) throw std::invalid_argument("checkpoint position must be positive");
        sorted.insert(pos);
    }
    std::set<int64_t> result;
    for (const auto pos : sorted) {
        if (result.size() == checkpoint_count) break;
        result.insert(pos);
    }
    return result;
}

// Called only after exact token-prefix and model/state identity verification.
// One token must remain for evaluation to regenerate next-token logits.
inline int64_t restore_position(const std::set<int64_t> & committed,
                                int64_t common_tokens, int64_t request_tokens,
                                int64_t resident_kv_tokens) {
    if (common_tokens < 0 || request_tokens <= 0 || resident_kv_tokens < 0) return 0;
    const auto limit = std::min({common_tokens, request_tokens - 1, resident_kv_tokens});
    auto it = committed.upper_bound(limit);
    return it == committed.begin() ? 0 : *--it;
}

inline std::vector<int64_t> boundaries(int64_t evaluated, int64_t next_evaluated) {
    if (evaluated < 0 || next_evaluated < evaluated) {
        throw std::invalid_argument("checkpoint progress must be monotone");
    }
    std::vector<int64_t> result;
    for (int64_t pos = (evaluated / checkpoint_spacing + 1) * checkpoint_spacing;
         pos <= next_evaluated; pos += checkpoint_spacing) result.push_back(pos);
    return result;
}
} // namespace recurrent_checkpoint
