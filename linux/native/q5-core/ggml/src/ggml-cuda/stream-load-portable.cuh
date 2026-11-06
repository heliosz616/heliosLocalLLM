#pragma once
// Preserve the value while retaining AMD's optional streaming-cache hint.
template <typename T>
static __device__ __forceinline__ T ggml_cuda_stream_load(const T * address) {
#if defined(GGML_USE_HIP)
    return __builtin_nontemporal_load(address);
#else
    return *address;
#endif
}
template <typename T>
static __device__ __forceinline__ void ggml_cuda_stream_store(T value, T * address) {
#if defined(GGML_USE_HIP)
    __builtin_nontemporal_store(value, address);
#else
    *address = value;
#endif
}
