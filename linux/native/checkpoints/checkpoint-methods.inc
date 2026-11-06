    static common_json checkpoint_ancestors_json(const recurrent_checkpoint::ancestor_limits & ancestors) {
        auto result=common_json::object();
        for(const auto & item:ancestors)result[item.first]=item.second;
        return result;
    }

    static recurrent_checkpoint::ancestor_limits checkpoint_ancestors_read(const common_json & meta) {
        recurrent_checkpoint::ancestor_limits result;
        if(!meta.contains("checkpoint_ancestors"))return result;
        const auto value=meta.at("checkpoint_ancestors");
        if(!value.is_object())throw std::runtime_error("Invalid checkpoint ancestry");
        for(const auto & item:value.items()) {
            const auto limit=item.value().get<int64_t>();
            if(item.key().empty() || limit<=0)throw std::runtime_error("Invalid checkpoint ancestor cutoff");
            result.emplace(item.key(),limit);
        }
        return result;
    }

    bool periodic_checkpoints_enabled() const {
        const char * enabled=std::getenv("LLAMA_RECURRENT_CHECKPOINTS");
        if(!enabled || std::string(enabled)!="1")return false;
        // Private same-process A/B only; production never configures this file.
        if(const char * control=std::getenv("LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL")) {
            std::ifstream in(control);int value=0;in>>value;return value==1;
        }
        return true;
    }

    void checkpoint_fork(server_slot & slot,int64_t kept) {
        if(kept>0 && kept>=slot.prompt.n_tokens())return;
        std::vector<std::pair<std::string,int64_t>> available;
        if(kept>0 && ssd_store)for(const auto & entry:ssd_store->entries) {
            try {
                if(entry.metadata.value("kind",std::string())=="recurrent-v1")
                    available.emplace_back(entry.metadata.at("kv_epoch").get<std::string>(),entry.metadata.at("n_tokens").get<int64_t>());
            }catch(const std::exception &){continue;}
        }
        auto inherited=recurrent_checkpoint::inherited_prefixes(slot.checkpoint_session,slot.checkpoint_ancestors,kept,available);
        slot.checkpoint_last=0;
        for(const auto & cp:available)if(recurrent_checkpoint::lineage_allows("",inherited,cp.first,cp.second))
            slot.checkpoint_last=std::max(slot.checkpoint_last,cp.second);
        slot.checkpoint_ancestors=std::move(inherited);
        slot.checkpoint_session=std::to_string(ggml_time_us())+"-"+std::to_string(slot.id);
        if(kept==0 || slot.checkpoint_family.empty())slot.checkpoint_family=slot.checkpoint_session;
    }

    void periodic_checkpoint_prune() {
        std::map<std::string,std::vector<std::pair<int64_t,std::string>>> entries;
        for(auto it=ssd_store->entries.begin();it!=ssd_store->entries.end();) {
            if(it->metadata.value("kind",std::string())!="recurrent-v1"){++it;continue;}
            try {
                const auto session=it->metadata.at("session_id").get<std::string>();
                const auto pos=it->metadata.at("n_tokens").get<int64_t>();
                if(session.empty() || pos<=0 || pos>n_ctx_slot())throw std::runtime_error("Invalid checkpoint identity/position");
                entries[session].emplace_back(pos,it->name);++it;
            }catch(const std::exception &){ssd_store->erase(it++);}
        }
        // A queued commit already occupies a retention slot, even before
        // drain() publishes it. Otherwise the seventh state could commit after
        // pruning and remain on SSD until another checkpoint operation.
        size_t pending_index=0;
        for(const auto & job:ssd_store->pending) {
            const auto & meta=job.metadata;
            if(meta.value("kind",std::string())=="recurrent-v1")
                entries[meta.at("session_id").get<std::string>()].emplace_back(
                    meta.at("n_tokens").get<int64_t>(),"pending:"+std::to_string(pending_index));
            ++pending_index;
        }
        std::map<std::string,std::set<std::string>> keep;
        for(const auto & p:entries)keep[p.first]=recurrent_checkpoint::retained_lru(p.second);
        for(auto it=ssd_store->entries.begin();it!=ssd_store->entries.end();) {
            if(it->metadata.value("kind",std::string())!="recurrent-v1"){++it;continue;}
            const auto session=it->metadata.at("session_id").get<std::string>();
            if(!keep[session].count(it->name))ssd_store->erase(it++);
            else ++it;
        }
    }

    size_t periodic_checkpoint_best(const server_tokens & tokens,size_t limit,const std::string & lineage,const recurrent_checkpoint::ancestor_limits & ancestors) const {
        if(!periodic_checkpoints_enabled() || !ssd_store || lineage.empty() || tokens.find_next_media_chunk(0).first)return 0;
        size_t best=0;
        for(const auto & entry:ssd_store->entries) {
            try {
                if(entry.metadata.value("kind",std::string())!="recurrent-v1")continue;
                const auto n=entry.metadata.at("n_tokens").get<int64_t>();
                if(!recurrent_checkpoint::lineage_allows(lineage,ancestors,entry.metadata.value("kv_epoch",std::string()),n))continue;
                if(n<=static_cast<int64_t>(best) || n>static_cast<int64_t>(limit))continue;
                const auto raw=base64::decode(entry.metadata.at("tokens").get<std::string>());
                if(raw.empty() || raw.size()%sizeof(llama_token))continue;
                llama_tokens packed(raw.size()/sizeof(llama_token));std::memcpy(packed.data(),raw.data(),raw.size());
                const auto prefix=server_tokens::deserialize(packed,mctx!=nullptr);
                if(prefix.size()==static_cast<size_t>(n) && prefix.get_common_prefix(tokens)==prefix.size())best=n;
            }catch(const std::exception &){continue;}
        }
        return best;
    }

    void periodic_checkpoint_capture(server_slot & slot) {
        if(!periodic_checkpoints_enabled() || !ssd_store || !slot.task || (!slot.task->params.cache_prompt && !std::getenv("LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL")) || slot.prompt.tokens.find_next_media_chunk(0).first)
            return; // Multimedia replay remains on the existing validated path.
        const auto count=slot.prompt.n_tokens();
        if(!recurrent_checkpoint::capture_due(slot.checkpoint_last,count))return;
        if(slot.checkpoint_session.empty())slot.checkpoint_session=std::to_string(ggml_time_us())+"-"+std::to_string(slot.id);
        if(slot.checkpoint_family.empty())slot.checkpoint_family=slot.checkpoint_session;
        const auto started=ggml_time_us();
        common_prompt_checkpoint cp{};
        const auto pos_min=llama_memory_seq_pos_min(llama_get_memory(ctx_tgt),slot.id);
        const auto pos_max=llama_memory_seq_pos_max(llama_get_memory(ctx_tgt),slot.id);
        if(pos_min<0 || pos_max+1!=count)return;
        cp.update_pos(count,pos_min,pos_max);
        cp.update_tgt(ctx_tgt,slot.id,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cp.update_dft(ctx_dft,slot.id,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        common_speculative_get_state(spec.get(),slot.id,cp.data_spec);
        recurrent_checkpoint_bundle bundle;
        bundle.n_tokens=count;bundle.pos_min=pos_min;bundle.pos_max=pos_max;
        const auto packed_tokens=slot.prompt.tokens.serialize();
        bundle.tokens.assign(packed_tokens.begin(),packed_tokens.end());
        bundle.target=std::move(cp.data_tgt);bundle.draft=std::move(cp.data_dft);bundle.speculative=std::move(cp.data_spec);
        const common_json meta={{"kind","recurrent-v1"},{"session_id",slot.checkpoint_family},{"kv_epoch",slot.checkpoint_session},
            {"n_tokens",count},{"pos_min",pos_min},{"pos_max",pos_max},
            {"tokens",base64::encode(reinterpret_cast<const char*>(bundle.tokens.data()),bundle.tokens.size())}};
        try {
            ssd_store->save([&](const std::string & path){bundle.write(path);return meta;},
                            bundle.size()+2*(meta.dump().size()+1024));
            slot.checkpoint_last=count;
            ssd_store->drain();periodic_checkpoint_prune();
            SLT_INF(slot,"recurrent checkpoint queued: tokens=%lld bytes=%llu capture_and_stage_ms=%.2f full_KV_bytes=0\n",
                    (long long)count,(unsigned long long)bundle.size(),(ggml_time_us()-started)/1000.0);
        } catch(const std::exception & e) {
            SLT_WRN(slot,"recurrent checkpoint unavailable: %s\n",e.what());
        }
    }

    bool periodic_checkpoint_restore(server_slot & slot,const server_tokens & tokens) {
        if(!periodic_checkpoints_enabled() || !ssd_store || slot.prompt.tokens.find_next_media_chunk(0).first || tokens.find_next_media_chunk(0).first || tokens.empty() || slot.prompt.tokens.empty())return false;
        const auto common=slot.prompt.tokens.get_common_prefix(tokens);
        const auto limit=std::min(common,tokens.size()-1);
        if(limit<recurrent_checkpoint::checkpoint_spacing)return false;
        ssd_store->drain(true);periodic_checkpoint_prune();
        auto selected=ssd_store->entries.end();int64_t best=0;
        for(auto it=ssd_store->entries.begin();it!=ssd_store->entries.end();++it) {
            try {
            if(it->metadata.value("kind",std::string())!="recurrent-v1")continue;
            const auto n=it->metadata.at("n_tokens").get<int64_t>();
            if(!recurrent_checkpoint::lineage_allows(slot.checkpoint_session,slot.checkpoint_ancestors,it->metadata.value("kv_epoch",std::string()),n))continue;
            if(n<=best || n>static_cast<int64_t>(limit))continue;
            const auto raw=base64::decode(it->metadata.at("tokens").get<std::string>());
            if(raw.empty() || raw.size()%sizeof(llama_token))continue;
            llama_tokens packed(raw.size()/sizeof(llama_token));std::memcpy(packed.data(),raw.data(),raw.size());
            const auto prefix=server_tokens::deserialize(packed,mctx != nullptr);
            if(prefix.size()!=static_cast<size_t>(n) || prefix.get_common_prefix(tokens)!=prefix.size())continue;
            best=n;selected=it;
            }catch(const std::exception &){continue;}
        }
        if(selected==ssd_store->entries.end())return false;
        // Preserve the existing near-tail RAM checkpoint for exact replays of
        // the current input. A coarse SSD ancestor must never replace it.
        if(common>=slot.prompt_input_tokens) {
            for(const auto & cp:slot.prompt.checkpoints) {
                if(cp.n_tokens>=best && cp.n_tokens<=static_cast<int64_t>(limit) &&
                   !cp.data_tgt.empty())return false;
            }
        }
        const auto started=ggml_time_us();
        try {
            if(!ssd_store->valid(*selected))throw std::runtime_error("Recurrent checkpoint checksum mismatch");
            auto b=recurrent_checkpoint_bundle::read(ssd_store->payload(*selected).string(),selected->bytes);
            if(b.n_tokens!=static_cast<uint64_t>(best))throw std::runtime_error("Recurrent checkpoint position mismatch");
            if(b.tokens.empty() || b.tokens.size()%sizeof(llama_token))throw std::runtime_error("Invalid checkpoint token extent");
            llama_tokens packed(b.tokens.size()/sizeof(llama_token));std::memcpy(packed.data(),b.tokens.data(),b.tokens.size());
            auto prefix=server_tokens::deserialize(packed,mctx != nullptr);
            if(prefix.size()!=b.n_tokens || prefix.get_common_prefix(tokens)!=prefix.size() ||
                    prefix.get_common_prefix(slot.prompt.tokens)!=prefix.size())throw std::runtime_error("Recurrent checkpoint prefix mismatch");
            common_prompt_checkpoint cp{};cp.update_pos(best,b.pos_min,b.pos_max);cp.id_task=-1;
            cp.data_tgt=std::move(b.target);cp.data_dft=std::move(b.draft);cp.data_spec=std::move(b.speculative);
            if(!cp.load_tgt(ctx_tgt,slot.id,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
               !cp.load_dft(ctx_dft,slot.id,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))throw std::runtime_error("Partial recurrent restore failed");
            common_speculative_set_state(spec.get(),slot.id,cp.data_spec);
            if(!llama_memory_seq_rm(llama_get_memory(ctx_tgt),slot.id,best,-1) ||
                (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft),slot.id,best,-1)))throw std::runtime_error("Cannot invalidate KV suffix after recurrent restore");
            checkpoint_fork(slot,best);
            slot.prompt.tokens.keep_first(best);slot.prompt_input_tokens=best;
            slot.prompt.checkpoints.clear();slot.prompt.checkpoints.push_back(std::move(cp));
            slot.checkpoint_last=best;
            // The new branch inherits ancestors only through this unchanged
            // KV prefix; later historical states are not compatible.
            ssd_store->touch(selected);
            SLT_INF(slot,"SSD recurrent-only restore: retained_GPU_KV_tokens=%lld checkpoint_bytes=%llu restore_ms=%.2f full_KV_read_bytes=0 replay_tokens=%zu\n",
                    (long long)best,(unsigned long long)selected->bytes,(ggml_time_us()-started)/1000.0,tokens.size()-best);
            return true;
        }catch(const std::exception & e){
            SLT_WRN(slot,"Recurrent restore failed; cold fallback: %s\n",e.what());
            ssd_store->erase(selected);
            slot.prompt_clear();slot.checkpoint_session.clear();slot.checkpoint_last=0;
            return false;
        }
    }
