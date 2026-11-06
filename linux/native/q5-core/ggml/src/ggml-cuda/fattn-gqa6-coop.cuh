#pragma once

static __device__ __forceinline__ void ggml_hip_gqa6_local_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Two waves share QK, then each wave computes half of the output dimensions.
template<int query_tiles, int key_tile, bool split_k>
static __global__ __launch_bounds__(64*query_tiles) void ggml_hip_gqa6_cooperative(ggml_hip_gqa6_args a) {
#if defined(RDNA4)
    using hf8 = __attribute__((ext_vector_type(8))) _Float16;
    using hf2 = __attribute__((ext_vector_type(2))) _Float16;
    using f8 = __attribute__((ext_vector_type(8))) float;
    constexpr int value_stride = key_tile;
    __shared__ half keys[key_tile][256];
    __shared__ half values[256][value_stride];
    __shared__ float partial_scores[query_tiles][256];
    const int tid = threadIdx.x, lane = tid%32, wave = tid/32;
    const int query_tile = wave/2, dim_half = wave%2;
    const int half_wave = lane/16, column = lane%16;
    const int tile = (split_k ? 0 : int(blockIdx.x)*query_tiles) + query_tile;
    const int query_row = tile*16 + column;
    const int token = query_row/6, head = blockIdx.y*6 + query_row%6;
    const int live_token = min(token, a.tokens - 1);
    const int query_tile_count = (a.tokens*6 + 15)/16;
    f8 sum[8] = {};
    float reference = -1.0e30f, denominator = 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    const int begin = split_k ? blockIdx.x*a.part_keys : 0;
    const int end = split_k ? min(begin + a.part_keys, a.cache) : a.cache;
    for (int start = begin; start < end; start += key_tile) {
        ggml_hip_gqa6_local_barrier();
        for (int load = tid; load < key_tile*16; load += 64*query_tiles) {
            const int row = (load/32)*2 + load%2;
            const int dim = ((load%32)/2)*16;
            half2 k[8] = {}, v[8] = {};
            if (start + row < end) {
                dequantize_V_q8_0<half, 16>(a.k + blockIdx.y*a.k_head + (start + row)*a.k_token, k, dim);
                dequantize_V_q8_0<half, 16>(a.v + blockIdx.y*a.v_head + (start + row)*a.v_token, v, dim);
            }
            const int swizzle = (row%8)*8;
            *(uint4 *)&keys[row][dim ^ swizzle] = *(const uint4 *)k;
            *(uint4 *)&keys[row][(dim + 8) ^ swizzle] = *(const uint4 *)(k + 4);
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int d = dim + e;
                const int value_swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                values[d][row ^ value_swizzle] = ((half *)v)[e];
            }
        }
        ggml_hip_gqa6_local_barrier();
        #pragma unroll 1
        for (int part = 0; part < key_tile/16; ++part) {
            f8 scores = {};
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int key_row = part*16 + column;
                const int dimension = dim_half*128 + g*16;
                const hf8 key = __builtin_bit_cast(hf8, *(const uint4 *)&keys[key_row][(dimension + 8*half_wave) ^ ((key_row%8)*8)]);
                hf8 query = {};
                if (tile < query_tile_count) {
                    const int64_t offset = (((int64_t(blockIdx.y)*query_tile_count + tile)*16 + dim_half*8 + g)*32 + lane)*8;
                    query = __builtin_bit_cast(hf8, *(const uint4 *)((const half *)a.q + offset));
                }
                scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(key, query, scores);
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = e*32 + lane;
                if (dim_half == 0) { partial_scores[query_tile][i] = scores[e]; }
            }
            ggml_hip_gqa6_local_barrier();
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = e*32 + lane;
                if (dim_half == 1) { scores[e] += partial_scores[query_tile][i]; partial_scores[query_tile][i] = scores[e]; }
            }
            ggml_hip_gqa6_local_barrier();
            float maximum = -INFINITY;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (dim_half == 0) { scores[e] = partial_scores[query_tile][e*32 + lane]; }
                const int pos = start + part*16 + 8*half_wave + e;
                const float bias = pos < end ? ggml_hip_gqa6_mask_bias(a, live_token, pos) : -INFINITY;
                scores[e] = scores[e]*(a.scale*log2e) + bias*log2e;
                maximum = fmaxf(maximum, scores[e]);
            }
            maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 32));
            if (__any_sync(0xffffffff, maximum > reference + 8.0f)) {
                const float next = fmaxf(reference, maximum), factor = exp2f(reference - next);
                #pragma unroll
                for (int g = 0; g < 8; ++g) { sum[g] *= factor; }
                denominator *= factor; reference = next;
            }
            uint32_t probabilities[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const half2 p = __floats2half2_rn(exp2f(scores[2*e] - reference), exp2f(scores[2*e + 1] - reference));
                probabilities[e] = __builtin_bit_cast(uint32_t, p);
                denominator = __builtin_amdgcn_fdot2(__builtin_bit_cast(hf2, p), hf2{1, 1}, denominator, false);
            }
            const hf8 p = __builtin_bit_cast(hf8, make_uint4(probabilities[0], probabilities[1], probabilities[2], probabilities[3]));
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int d = dim_half*128 + g*16 + column;
                const int swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                const hf8 value = __builtin_bit_cast(hf8, *(const uint4 *)&values[d][(part*16 + half_wave*8) ^ swizzle]);
                sum[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(value, p, sum[g]);
            }
        }
    }
    denominator += __shfl_xor(denominator, 16, 32);
    const float inv = split_k ? 1.0f : (denominator > 0.0f ? 1.0f/denominator : 0.0f);
    if (token < a.tokens) {
        const int64_t row = int64_t(token)*a.heads + head;
        float * out = a.output + (split_k ? row*a.parts + blockIdx.x : row)*256 + dim_half*128 + half_wave*8;
        if (split_k && dim_half == 0 && lane < 16) { a.meta[row*a.parts + blockIdx.x] = make_float2(reference/log2e, denominator); }
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            *(float4 *)(out + g*16) = make_float4(sum[g][0]*inv, sum[g][1]*inv, sum[g][2]*inv, sum[g][3]*inv);
            *(float4 *)(out + g*16 + 4) = make_float4(sum[g][4]*inv, sum[g][5]*inv, sum[g][6]*inv, sum[g][7]*inv);
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif
}
