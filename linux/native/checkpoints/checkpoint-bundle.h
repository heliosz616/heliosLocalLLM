#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cstring>

// Partial-state bundle; ordinary attention KV must never be serialized here.
struct recurrent_checkpoint_bundle {
    uint64_t n_tokens=0;
    uint64_t pos_min=0, pos_max=0;
    std::vector<uint8_t> tokens, target, draft, speculative;
    uint64_t size() const { return 8+7*8+tokens.size()+target.size()+draft.size()+speculative.size(); }
    static void write_u64(std::ostream & out,uint64_t v) {
        for(int i=0;i<8;++i) out.put(static_cast<char>(v>>(i*8)));
    }
    static uint64_t read_u64(std::istream & in) {
        uint64_t v=0;
        for(int i=0;i<8;++i){int c=in.get();if(c==EOF)throw std::runtime_error("Truncated recurrent bundle");v|=uint64_t(uint8_t(c))<<(i*8);}
        return v;
    }
    void write(const std::string & path) const {
        if(!n_tokens || target.empty() || tokens.size()%sizeof(int32_t) || pos_max+1!=n_tokens || pos_min>pos_max)
            throw std::runtime_error("Inconsistent recurrent checkpoint");
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        out.write("RCPKV001",8);
        for(auto n:{n_tokens,pos_min,pos_max,uint64_t(tokens.size()),uint64_t(target.size()),uint64_t(draft.size()),uint64_t(speculative.size())})write_u64(out,n);
        for(const auto * b:{&tokens,&target,&draft,&speculative})out.write(reinterpret_cast<const char*>(b->data()),b->size());
        out.close();if(!out)throw std::runtime_error("Checkpoint write failed");
    }
    static recurrent_checkpoint_bundle read(const std::string & path,uint64_t expected) {
        std::ifstream in(path,std::ios::binary);char magic[8];in.read(magic,8);
        if(!in || std::memcmp(magic,"RCPKV001",8))throw std::runtime_error("Unknown recurrent bundle");
        recurrent_checkpoint_bundle r;r.n_tokens=read_u64(in);r.pos_min=read_u64(in);r.pos_max=read_u64(in);
        uint64_t sizes[4];uint64_t remain=expected>=64?expected-64:0;
        for(auto & n:sizes){n=read_u64(in);if(n>remain)throw std::runtime_error("Invalid recurrent bundle extent");remain-=n;}
        if(remain || !r.n_tokens || !sizes[0] || sizes[0]%sizeof(int32_t) || !sizes[1] || r.pos_max+1!=r.n_tokens || r.pos_min>r.pos_max)
            throw std::runtime_error("Invalid recurrent bundle metadata");
        std::vector<uint8_t>* arrays[]={&r.tokens,&r.target,&r.draft,&r.speculative};
        for(int i=0;i<4;++i){arrays[i]->resize(sizes[i]);in.read(reinterpret_cast<char*>(arrays[i]->data()),sizes[i]);if(!in)throw std::runtime_error("Incomplete recurrent bundle");}
        if(in.peek()!=EOF)throw std::runtime_error("Unexpected recurrent bundle trailing data");
        return r;
    }
};
