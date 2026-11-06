#pragma once

#if defined(GGML_USE_HIP)
#include <map>
#include <mutex>

namespace ggml_hip_layer_f16 {
struct entry {
    void * data;
    int64_t elements;
    ggml_type original_type;
};
struct cache {
    int layer = -1;
    size_t bytes = 0, peak_bytes = 0, expanded_bytes = 0;
    uint64_t hits = 0, misses = 0;
    std::map<const void *,entry> weights;
};
static std::mutex mutex;
static std::map<const ggml_backend_cuda_context *,cache> caches;

static void clear_weights(cache & c) {
    for (auto & w : c.weights) CUDA_CHECK(cudaFree(w.second.data));
    c.weights.clear();
    c.bytes = 0;
}

static half * get(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, bool & fresh) {
    fresh = false;
    const char * enabled = getenv("GGML_HIP_LAYER_F16_CACHE");
    if (!enabled || strcmp(enabled,"1") || !getenv("GGML_CUDA_DISABLE_GRAPHS") ||
            !ggml_is_quantized(weight->type) || !ggml_is_contiguously_allocated(weight)) return nullptr;
    int layer = -1;
    if (sscanf(weight->name,"blk.%d.",&layer) != 1 || layer < 0) return nullptr;
    constexpr size_t limit = 1ull << 30;
    const size_t bytes = ggml_nelements(weight)*sizeof(half);
    if (bytes > limit) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    auto & c = caches[&ctx];
    if (c.layer != layer) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        clear_weights(c);
        c.layer = layer;
    }
    auto it = c.weights.find(weight->data);
    if (it != c.weights.end()) {
        GGML_ASSERT(it->second.elements == ggml_nelements(weight) && it->second.original_type == weight->type);
        ++c.hits;
        return (half *)it->second.data;
    }
    if (bytes > limit-c.bytes) return nullptr;
    void * data = nullptr;
    CUDA_CHECK(cudaMalloc(&data,bytes));
    c.weights.emplace(weight->data,entry{data,ggml_nelements(weight),weight->type});
    c.bytes += bytes;
    c.expanded_bytes += bytes;
    c.peak_bytes = std::max(c.peak_bytes,c.bytes);
    ++c.misses;
    fresh = true;
    return (half *)data;
}

static void release(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = caches.find(ctx);
    if (it == caches.end()) return;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
    auto & c = it->second;
    GGML_LOG_INFO("layer_f16_cache: device=%d hits=%llu misses=%llu peak_bytes=%zu expanded_bytes=%zu exact_existing_conversion=1\n",
            ctx->device,(unsigned long long)c.hits,(unsigned long long)c.misses,c.peak_bytes,c.expanded_bytes);
    clear_weights(c);
    caches.erase(it);
}
}
#endif
