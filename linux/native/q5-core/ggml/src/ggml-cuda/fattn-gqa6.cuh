#pragma once

#if defined(GGML_USE_HIP)

struct ggml_hip_gqa6_args {
    const char * q;
    const char * k;
    const char * v;
    const half * mask;
    float * output;
    int tokens;
    int cache;
    int heads;
    int64_t q_token;
    int64_t q_head;
    int64_t k_token;
    int64_t k_head;
    int64_t v_token;
    int64_t v_head;
    int64_t mask_token;
    float scale;
    float2 * meta;
    int parts;
    int part_keys;
    const int32_t * cell_pos = nullptr;
    const int32_t * tok_lo = nullptr;
    const int32_t * tok_hi = nullptr;
    const uint2 * v_codes = nullptr;
    const uint4 * v_scales = nullptr;
    int v_groups = 0;
};

static __device__ __forceinline__ float ggml_hip_gqa6_mask_bias(const ggml_hip_gqa6_args & a, int token, int key) {
    if (a.cell_pos) {
        const int32_t pos = a.cell_pos[key];
        return pos != INT32_MIN && pos >= a.tok_lo[token] && pos <= a.tok_hi[token] ? 0.0f : -INFINITY;
    }
    return __half2float(a.mask[int64_t(token)*a.mask_token + key]);
}

static __global__ void ggml_hip_pack_query_fragments(ggml_hip_gqa6_args a, half * out) {
    const int tiles = (a.tokens*6 + 15)/16;
    const int fragment = blockIdx.x*256 + threadIdx.x;
    if (fragment >= (a.heads/6)*tiles*16*32) { return; }
    const int lane = fragment%32;
    const int g = (fragment/32)%16;
    const int tile = (fragment/(32*16))%tiles;
    const int group = fragment/(32*16*tiles);
    const int row = tile*16 + lane%16;
    const int token = row/6, head = group*6 + row%6;
    uint4 result = make_uint4(0, 0, 0, 0);
    if (token < a.tokens) {
        const float * input = (const float *)(a.q + token*a.q_token + head*a.q_head) + g*16 + (lane/16)*8;
        const float4 lo = *(const float4 *)input;
        const float4 hi = *(const float4 *)(input + 4);
        result = make_uint4(__builtin_bit_cast(uint32_t, __floats2half2_rn(lo.x, lo.y)),
                            __builtin_bit_cast(uint32_t, __floats2half2_rn(lo.z, lo.w)),
                            __builtin_bit_cast(uint32_t, __floats2half2_rn(hi.x, hi.y)),
                            __builtin_bit_cast(uint32_t, __floats2half2_rn(hi.z, hi.w)));
    }
    *(uint4 *)(out + int64_t(fragment)*8) = result;
}

// Transpose eight keys without decoding or rounding any original Q8_0 byte.
static __global__ void ggml_hip_gqa6_pack_value(ggml_hip_gqa6_args a, uint2 * codes, uint4 * scales) {
    const int d = threadIdx.x;
    const int first = blockIdx.x*8;
    const int64_t group = int64_t(blockIdx.y)*a.v_groups + blockIdx.x;
    uint32_t words[2] = {};
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (first + t < a.cache) {
            const char * src = a.v + blockIdx.y*a.v_head + (first + t)*a.v_token + (d/32)*sizeof(block_q8_0);
            words[t/4] |= uint32_t(((const uint8_t *)src)[2 + d%32]) << (8*(t%4));
        }
    }
    codes[group*256 + d] = make_uint2(words[0], words[1]);
    if (d < 8) {
        uint32_t sw[4] = {};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (first + t < a.cache) {
                const char * src = a.v + blockIdx.y*a.v_head + (first + t)*a.v_token + d*sizeof(block_q8_0);
                sw[t/2] |= uint32_t(*(const uint16_t *)src) << (16*(t%2));
            }
        }
        scales[group*8 + d] = make_uint4(sw[0], sw[1], sw[2], sw[3]);
    }
}

struct ggml_hip_q8_value_fragment {
    int8_t codes[16];
    half scale;
};

// Request-local scratch only. Both products use the existing half2 Q8 decoder.
static __global__ void ggml_hip_gqa6_expand_exact_q8(
        ggml_hip_gqa6_args a, half * k_out, half * v_out) {
    const int64_t block = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t count = int64_t(a.cache)*(a.heads/6)*8;
    if (block >= count) return;
    const int dim = (block%8)*32;
    const int key = (block/8)%a.cache;
    const int head = block/(int64_t(a.cache)*8);
    half2 k[16], v[16];
    dequantize_V_q8_0<half, 16>(a.k + head*a.k_head + key*a.k_token, k, dim);
    dequantize_V_q8_0<half, 16>(a.k + head*a.k_head + key*a.k_token, k + 8, dim + 16);
    dequantize_V_q8_0<half, 16>(a.v + head*a.v_head + key*a.v_token, v, dim);
    dequantize_V_q8_0<half, 16>(a.v + head*a.v_head + key*a.v_token, v + 8, dim + 16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *(uint4 *)(k_out + block*32 + j*8) = *(const uint4 *)(k + j*4);
        *(uint4 *)(v_out + block*32 + j*8) = *(const uint4 *)(v + j*4);
    }
}

static __device__ __forceinline__ ggml_hip_q8_value_fragment ggml_hip_load_q8_value_fragment(
        const ggml_hip_gqa6_args & a, int key, int dim, int end) {
    ggml_hip_q8_value_fragment r = {};
    if (key < end) {
        const block_q8_0 * x = (const block_q8_0 *)(a.v + blockIdx.y*a.v_head + key*a.v_token);
        ggml_cuda_memcpy_1<16, 2>(r.codes, x[dim/32].qs + dim%32);
        r.scale = x[dim/32].d;
    }
    return r;
}

template <int n_waves, int key_tile, bool split_k, bool packed_query = false, bool packed_value = false,
          bool prefetch_value = false, bool temp_f16_tr = false>
static __global__ __launch_bounds__(32*n_waves) void ggml_hip_gqa6_attention(ggml_hip_gqa6_args a) {
#if defined(RDNA4)
    using hf8 = __attribute__((ext_vector_type(8))) _Float16;
    using hf2 = __attribute__((ext_vector_type(2))) _Float16;
    using f8 = __attribute__((ext_vector_type(8))) float;
    constexpr int key_stride = packed_query ? 256 : 264;
    constexpr int value_stride = split_k ? 16 : (packed_query ? 32 : 64);
    __shared__ half keys[key_tile][key_stride];
    __shared__ half values[256][value_stride];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int half_wave = lane / 16;
    const int column = lane % 16;
    const int query_row = wave * 16 + column;
    const int token = (split_k ? 0 : blockIdx.x * 64) + query_row / 6;
    const int head = blockIdx.y * 6 + query_row % 6;
    const int live_token = min(token, a.tokens - 1);
    const float * query = (const float *)(a.q + live_token*a.q_token + head*a.q_head);
    hf8 q[packed_query ? 1 : 16];
    if constexpr (!packed_query) {
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const float4 lo = *(const float4 *)(query + 16*g + 8*half_wave);
        const float4 hi = *(const float4 *)(query + 16*g + 8*half_wave + 4);
        const uint4 bits = make_uint4(
            __builtin_bit_cast(uint32_t, __floats2half2_rn(lo.x, lo.y)),
            __builtin_bit_cast(uint32_t, __floats2half2_rn(lo.z, lo.w)),
            __builtin_bit_cast(uint32_t, __floats2half2_rn(hi.x, hi.y)),
            __builtin_bit_cast(uint32_t, __floats2half2_rn(hi.z, hi.w)));
        q[g] = __builtin_bit_cast(hf8, bits);
    }
    }
    f8 sum[16] = {};
    float reference = -1.0e30f;
    float denominator = 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    const float score_scale = a.scale * log2e;

    const int begin = split_k ? blockIdx.x*a.part_keys : 0;
    const int end = split_k ? min(begin + a.part_keys, a.cache) : a.cache;
    ggml_hip_q8_value_fragment prefetched_v = {};
    if constexpr (prefetch_value) {
        static_assert(!split_k && !packed_query && !packed_value && key_tile == 2*n_waves,
                      "one exact original Q8 fragment per prefill thread");
        prefetched_v = ggml_hip_load_q8_value_fragment(a, begin + 2*wave + lane%2, (lane/2)*16, end);
    }
    for (int start = begin; start < end; start += key_tile) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        for (int load = tid; load < key_tile*16; load += 32*n_waves) {
            const int row = split_k ? load/16 : 2*wave + lane%2;
            const int dim = (split_k ? load%16 : lane/2)*16;
            half2 k[8] = {};
            half2 v[8] = {};
            if (start + row < end) {
                const char * kp = a.k + blockIdx.y*a.k_head + (start + row)*a.k_token;
                const char * vp = a.v + blockIdx.y*a.v_head + (start + row)*a.v_token;
                if constexpr (temp_f16_tr) {
                    *(uint4 *)k = *(const uint4 *)((const half *)kp + dim);
                    *(uint4 *)(k + 4) = *(const uint4 *)((const half *)kp + dim + 8);
                } else {
                    dequantize_V_q8_0<half, 16>(kp, k, dim);
                }
                if constexpr (!packed_value && !prefetch_value && !temp_f16_tr) {
                    dequantize_V_q8_0<half, 16>(vp, v, dim);
                }
            }
            if constexpr (prefetch_value) {
                const half2 d = __half2half2(prefetched_v.scale);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v[e] = d * make_half2(prefetched_v.codes[2*e], prefetched_v.codes[2*e + 1]);
                }
            }
            const int key_swizzle = packed_query ? (row%8)*8 : 0;
            *(uint4 *)&keys[row][dim ^ key_swizzle] = *(const uint4 *)k;
            *(uint4 *)&keys[row][(dim + 8) ^ key_swizzle] = *(const uint4 *)(k + 4);
            if constexpr (!packed_value && !temp_f16_tr) {
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int d = dim + e;
                    const int swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                    values[d][row ^ swizzle] = ((half *)v)[e];
                }
            }
        }
        if constexpr (temp_f16_tr) {
            static_assert(!split_k && !packed_query && !packed_value && !prefetch_value && key_tile == 2*n_waves,
                          "exact temporary F16 transposed-load prefill geometry");
            using i16x8 = __attribute__((ext_vector_type(8))) int16_t;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int idx = wave + r*n_waves;
                const int key_group = idx/8, dim_group = idx%8;
                // Each group of eight lanes loads eight neighboring dimensions
                // from eight distinct keys. The native load transposes them.
                const int key = min(start + key_group*8 + lane%8, end - 1);
                const int dim = dim_group*32 + (lane/8)*8;
                const half * src = (const half *)(a.v + blockIdx.y*a.v_head + key*a.v_token) + dim;
                const auto fragment = __builtin_amdgcn_global_load_tr_b128_v8i16(
                    (__attribute__((address_space(1))) i16x8 *)const_cast<half *>(src));
                const int d = dim_group*32 + (lane/8)*8 + lane%8;
                const int swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                *(uint4 *)&values[d][key_group*8 ^ swizzle] = __builtin_bit_cast(uint4, fragment);
            }
        }
        if constexpr (packed_value) {
            static_assert(!split_k && !packed_query, "packed value prefill geometry");
            for (int load = tid; load < key_tile*256/8; load += 32*n_waves) {
                const int d = load%256, row = (load/256)*8;
                uint4 result = {};
                if (start + row < end) {
                    const int64_t group = int64_t(blockIdx.y)*a.v_groups + (start + row)/8;
                    const uint2 codes = a.v_codes[group*256 + d];
                    const uint4 scales = a.v_scales[group*8 + d/32];
                    const uint32_t qw[2] = {codes.x, codes.y};
                    const uint32_t sw[4] = {scales.x, scales.y, scales.z, scales.w};
                    uint32_t out[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int8_t q0 = int8_t(qw[p/2] >> (16*(p%2)));
                        const int8_t q1 = int8_t(qw[p/2] >> (16*(p%2) + 8));
                        const half2 q = make_half2(q0, q1);
                        out[p] = __builtin_bit_cast(uint32_t, __hmul2(__builtin_bit_cast(half2, sw[p]), q));
                    }
                    result = make_uint4(out[0], out[1], out[2], out[3]);
                }
                const int swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                *(uint4 *)&values[d][row ^ swizzle] = result;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");

        // Issue the next original-byte V load before this tile's QK/softmax/PV work.
        // Its values are first consumed after the next LDS-only barrier.
        if constexpr (prefetch_value) {
            if (start + key_tile < end) {
                prefetched_v = ggml_hip_load_q8_value_fragment(
                    a, start + key_tile + 2*wave + lane%2, (lane/2)*16, end);
            }
        }

#pragma unroll 1
        for (int part = 0; part < key_tile / 16; ++part) {
            f8 scores = {};
#pragma unroll
            for (int g = 0; g < 16; g += 2) {
                hf8 fragment[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int key_row = part*16 + column;
                    const int swizzle = packed_query ? (key_row%8)*8 : 0;
                    fragment[j] = __builtin_bit_cast(hf8,
                        *(const uint4 *)&keys[key_row][(16*(g + j) + 8*half_wave) ^ swizzle]);
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if constexpr (packed_query) {
                        const int tiles = (a.tokens*6 + 15)/16;
                        const int tile = blockIdx.x*24 + wave;
                        hf8 query_fragment = {};
                        if (tile < tiles) {
                            const int64_t offset = (((int64_t(blockIdx.y)*tiles + tile)*16 + g + j)*32 + lane)*8;
                            query_fragment = __builtin_bit_cast(hf8, *(const uint4 *)((const half *)a.q + offset));
                        }
                        scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fragment[j], query_fragment, scores);
                    } else {
                        scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fragment[j], q[g + j], scores);
                    }
                }
            }
            float maximum = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int pos = start + part*16 + 8*half_wave + e;
                const float bias = pos < end ? ggml_hip_gqa6_mask_bias(a, live_token, pos) : -INFINITY;
                scores[e] = scores[e]*score_scale + bias*log2e;
                maximum = fmaxf(maximum, scores[e]);
            }
            maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 32));
            if (__any_sync(0xffffffff, maximum > reference + 8.0f)) {
                const float next = fmaxf(reference, maximum);
                const float factor = exp2f(reference - next);
#pragma unroll
                for (int g = 0; g < 16; ++g) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sum[g][e] *= factor;
                    }
                }
                denominator *= factor;
                reference = next;
            }
            uint32_t probabilities[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const half2 p = __floats2half2_rn(exp2f(scores[2*e] - reference), exp2f(scores[2*e + 1] - reference));
                probabilities[e] = __builtin_bit_cast(uint32_t, p);
                denominator = __builtin_amdgcn_fdot2(__builtin_bit_cast(hf2, p), hf2{1, 1}, denominator, false);
            }
            const hf8 p = __builtin_bit_cast(hf8,
                make_uint4(probabilities[0], probabilities[1], probabilities[2], probabilities[3]));
#pragma unroll
            for (int g = 0; g < 16; g += 2) {
                hf8 fragment[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int d = 16*(g + j) + column;
                    const int swizzle = ((d/16 + d%16)%(value_stride/8))*8;
                    const int key = (part*16 + 8*half_wave) ^ swizzle;
                    fragment[j] = __builtin_bit_cast(hf8, *(const uint4 *)&values[d][key]);
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sum[g + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fragment[j], p, sum[g + j]);
                }
            }
        }
    }
    denominator += __shfl_xor(denominator, 16, 32);
    const float inv = split_k ? 1.0f : (denominator > 0.0f ? 1.0f/denominator : 0.0f);
    if (token < a.tokens) {
        const int64_t row = (int64_t)token*a.heads + head;
        float * out = a.output + (split_k ? row*a.parts + blockIdx.x : row)*256 + 8*half_wave;
        if (split_k && lane < 16) {
            a.meta[row*a.parts + blockIdx.x] = make_float2(reference/log2e, denominator);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            *(float4 *)(out + 16*g) = make_float4(sum[g][0]*inv, sum[g][1]*inv, sum[g][2]*inv, sum[g][3]*inv);
            *(float4 *)(out + 16*g + 4) = make_float4(sum[g][4]*inv, sum[g][5]*inv, sum[g][6]*inv, sum[g][7]*inv);
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif
}

static __global__ __launch_bounds__(256) void ggml_hip_gqa6_merge(
        const float * parts, const float2 * metadata, float * output, int count) {
    __shared__ float2 meta[256];
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (d < count) {
        meta[d] = metadata[row*count + d];
    }
    __syncthreads();
    float maximum = meta[0].x;
    for (int i = 1; i < count; ++i) {
        maximum = fmaxf(maximum, meta[i].x);
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float weight = expf(meta[i].x - maximum);
        numerator += weight*parts[(int64_t(row)*count + i)*256 + d];
        denominator += weight*meta[i].y;
    }
    output[int64_t(row)*256 + d] = denominator > 0.0f ? numerator/denominator : 0.0f;
}

#include "fattn-gqa6-flydsl.cuh"
#include "fattn-gqa6-coop.cuh"
#include "fattn-gqa6-keys.cuh"
#include "fattn-gqa6-wave-pipeline.cuh"

static bool ggml_hip_try_gqa6(ggml_backend_cuda_context & ctx, ggml_tensor * output) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GQA6_ATTN");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    static const bool decode_enabled = [] {
        const char * value = std::getenv("GGML_HIP_GQA6_DECODE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    const ggml_tensor * q = output->src[0];
    const ggml_tensor * k = output->src[1];
    const ggml_tensor * v = output->src[2];
    const ggml_tensor * mask = output->src[3];
    const ggml_tensor * cell_pos = output->src[5];
    const ggml_tensor * tok_lo = output->src[6];
    const ggml_tensor * tok_hi = output->src[7];
    const bool derived = cell_pos && tok_lo && tok_hi;
    const bool derived_valid = derived && !mask &&
            cell_pos->type == GGML_TYPE_I32 && tok_lo->type == GGML_TYPE_I32 && tok_hi->type == GGML_TYPE_I32 &&
            ggml_is_contiguous(cell_pos) && ggml_is_contiguous(tok_lo) && ggml_is_contiguous(tok_hi) &&
            ggml_nelements(cell_pos) == k->ne[1] && ggml_nelements(tok_lo) == q->ne[1] && ggml_nelements(tok_hi) == q->ne[1];
    const bool explicit_valid = !cell_pos && !tok_lo && !tok_hi && mask &&
            mask->type == GGML_TYPE_F16 && mask->nb[0] == 2 && mask->ne[2] == 1 && mask->ne[3] == 1 &&
            mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1];
    float scale, max_bias, softcap;
    memcpy(&scale, output->op_params, sizeof(float));
    memcpy(&max_bias, (const float *)output->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *)output->op_params + 2, sizeof(float));
    const bool decode = decode_enabled && q->ne[1] >= 1 && q->ne[1] <= 8;
    if ((!decode && (!enabled || q->ne[1] < 64)) || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc) ||
            q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || v->type != GGML_TYPE_Q8_0 ||
            q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
            k->ne[1] <= 0 || k->ne[1] > INT_MAX - 2048 || q->ne[1] > INT_MAX || q->ne[2] > INT_MAX ||
            q->ne[2] != k->ne[2]*6 || k->ne[2] != v->ne[2] || k->ne[1] != v->ne[1] ||
            q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            q->nb[0] != 4 || k->nb[0] != sizeof(block_q8_0) || v->nb[0] != sizeof(block_q8_0) ||
            q->nb[1]%16 || q->nb[2]%16 ||
            (!derived_valid && !explicit_valid) ||
            output->src[4] || max_bias != 0.0f || softcap != 0.0f ||
            ggml_get_op_params_i32(output, 4) != 0 || !ggml_is_contiguous(output)) {
        return false;
    }
    ggml_hip_gqa6_args args{
        (const char *)q->data, (const char *)k->data, (const char *)v->data,
        mask ? (const half *)mask->data : nullptr, (float *)output->data,
        (int)q->ne[1], (int)k->ne[1], (int)q->ne[2],
        (int64_t)q->nb[1], (int64_t)q->nb[2], (int64_t)k->nb[1], (int64_t)k->nb[2],
        (int64_t)v->nb[1], (int64_t)v->nb[2], mask ? (int64_t)mask->nb[1]/2 : 0, scale, nullptr, 1, (int)k->ne[1],
        derived ? (const int32_t *)cell_pos->data : nullptr,
        derived ? (const int32_t *)tok_lo->data : nullptr,
        derived ? (const int32_t *)tok_hi->data : nullptr};
    if (ggml_hip_flydsl_attn::try_run(ctx,args)) return true;
    static const bool wave_pipeline = [] {
        const char *v=std::getenv("GGML_HIP_GQA6_WAVE_PIPELINE");
        return v&&v[0]=='1'&&v[1]==0;
    }();
    if(wave_pipeline&&!decode) {
        ggml_hip_gqa6_wave_pipeline<<<dim3((args.tokens+63)/64,k->ne[2]),800,0,ctx.stream()>>>(args);
        CUDA_CHECK(cudaGetLastError());return true;
    }
    static const bool cooperative = [] {
        const char * value = std::getenv("GGML_HIP_GQA6_COOPERATIVE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (cooperative) {
        const int tiles = (args.tokens*6 + 15)/16;
        const int fragments = (args.heads/6)*tiles*16*32;
        ggml_cuda_pool_alloc<half> packed(ctx.pool(), int64_t(fragments)*8);
        ggml_hip_pack_query_fragments<<<(fragments + 255)/256, 256, 0, ctx.stream()>>>(args, packed.get());
        args.q = (const char *)packed.get();
        if (decode) {
            args.part_keys = ((args.cache + 16*128 - 1)/(16*128))*16;
            args.parts = (args.cache + args.part_keys - 1)/args.part_keys;
            ggml_cuda_pool_alloc<float> partial(ctx.pool(), size_t(args.tokens)*args.heads*args.parts*256);
            ggml_cuda_pool_alloc<float2> meta(ctx.pool(), size_t(args.tokens)*args.heads*args.parts);
            args.output = partial.get(); args.meta = meta.get();
            const dim3 grid(args.parts, k->ne[2]);
            if (tiles == 1) {
                ggml_hip_gqa6_cooperative<1, 16, true><<<grid, 64, 0, ctx.stream()>>>(args);
            } else if (tiles == 2) {
                ggml_hip_gqa6_cooperative<2, 16, true><<<grid, 128, 0, ctx.stream()>>>(args);
            } else {
                ggml_hip_gqa6_cooperative<3, 16, true><<<grid, 192, 0, ctx.stream()>>>(args);
            }
            ggml_hip_gqa6_merge<<<args.tokens*args.heads, 256, 0, ctx.stream()>>>(partial.get(), meta.get(), (float *)output->data, args.parts);
        } else {
            static const bool wide = [] {
                int blocks10 = 0, blocks8 = 0;
                CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks10, ggml_hip_gqa6_cooperative<10, 16, false>, 640, 0));
                CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks8, ggml_hip_gqa6_cooperative<8, 16, false>, 512, 0));
                GGML_LOG_INFO("gqa6 cooperative occupancy: groups10=%d groups8=%d\n", blocks10, blocks8);
                return 10*blocks10 >= 8*blocks8;
            }();
            if (wide) {
                ggml_hip_gqa6_cooperative<10, 16, false><<<dim3((tiles + 9)/10, k->ne[2]), 640, 0, ctx.stream()>>>(args);
            } else {
                ggml_hip_gqa6_cooperative<8, 16, false><<<dim3((tiles + 7)/8, k->ne[2]), 512, 0, ctx.stream()>>>(args);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (decode) {
        const int requested_parts = 128;
        args.part_keys = ((args.cache + 16*requested_parts - 1)/(16*requested_parts))*16;
        args.parts = (args.cache + args.part_keys - 1)/args.part_keys;
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), size_t(q->ne[1])*q->ne[2]*args.parts*256);
        ggml_cuda_pool_alloc<float2> meta(ctx.pool(), size_t(q->ne[1])*q->ne[2]*args.parts);
        args.output = partial.get();
        args.meta = meta.get();
        const dim3 grid(args.parts, k->ne[2]);
        const int waves = (args.tokens*6 + 15)/16;
        if (waves == 1) {
            ggml_hip_gqa6_attention<1, 16, true><<<grid, 32, 0, ctx.stream()>>>(args);
        } else if (waves == 2) {
            ggml_hip_gqa6_attention<2, 16, true><<<grid, 64, 0, ctx.stream()>>>(args);
        } else {
            ggml_hip_gqa6_attention<3, 16, true><<<grid, 96, 0, ctx.stream()>>>(args);
        }
        ggml_hip_gqa6_merge<<<q->ne[1]*q->ne[2], 256, 0, ctx.stream()>>>(
            partial.get(), meta.get(), (float *)output->data, args.parts);
    } else {
        static const bool key_split = [] {
            const char * value = std::getenv("GGML_HIP_GQA6_KEY_SPLIT");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (key_split) {
            ggml_hip_gqa6_key_split<<<dim3((args.tokens + 15)/16,args.heads/6),384,0,ctx.stream()>>>(args);
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        static const bool temp_f16_tr = [] {
            const char * value = std::getenv("GGML_HIP_GQA6_TEMP_F16_TR");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (temp_f16_tr) {
            const size_t elements = size_t(args.cache)*(args.heads/6)*256;
            ggml_cuda_pool_alloc<half> k_f16(ctx.pool(), elements), v_f16(ctx.pool(), elements);
            const int64_t blocks = int64_t(args.cache)*(args.heads/6)*8;
            ggml_hip_gqa6_expand_exact_q8<<<(blocks + 127)/128, 128, 0, ctx.stream()>>>(args, k_f16.get(), v_f16.get());
            args.k = (const char *)k_f16.get(); args.v = (const char *)v_f16.get();
            args.k_token = args.v_token = 256*sizeof(half);
            args.k_head = args.v_head = int64_t(args.cache)*256*sizeof(half);
            ggml_hip_gqa6_attention<24, 48, false, false, false, false, true><<<dim3((q->ne[1] + 63)/64, k->ne[2]), 768, 0, ctx.stream()>>>(args);
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        static const bool prefetch_value = [] {
            const char * value = std::getenv("GGML_HIP_GQA6_PREFETCH_VALUE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (prefetch_value) {
            ggml_hip_gqa6_attention<24, 48, false, false, false, true><<<dim3((q->ne[1] + 63)/64, k->ne[2]), 768, 0, ctx.stream()>>>(args);
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        static const bool packed_value = [] {
            const char * value = std::getenv("GGML_HIP_GQA6_PACKED_VALUE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (packed_value) {
            args.v_groups = (args.cache + 7)/8;
            const size_t groups = size_t(args.heads/6)*args.v_groups;
            ggml_cuda_pool_alloc<uint2> codes(ctx.pool(), groups*256);
            ggml_cuda_pool_alloc<uint4> scales(ctx.pool(), groups*8);
            ggml_hip_gqa6_pack_value<<<dim3(args.v_groups, args.heads/6), 256, 0, ctx.stream()>>>(args, codes.get(), scales.get());
            args.v_codes = codes.get(); args.v_scales = scales.get();
            ggml_hip_gqa6_attention<24, 48, false, false, true><<<dim3((q->ne[1] + 63)/64, k->ne[2]), 768, 0, ctx.stream()>>>(args);
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        static const bool packed_query = [] {
            const char * value = std::getenv("GGML_HIP_GQA6_PACKED_QUERY");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        if (packed_query) {
            const int tiles = (args.tokens*6 + 15)/16;
            const int fragments = (args.heads/6)*tiles*16*32;
            ggml_cuda_pool_alloc<half> packed(ctx.pool(), int64_t(fragments)*8);
            ggml_hip_pack_query_fragments<<<(fragments + 255)/256, 256, 0, ctx.stream()>>>(args, packed.get());
            args.q = (const char *)packed.get();
            ggml_hip_gqa6_attention<24, 32, false, true><<<dim3((q->ne[1] + 63)/64, k->ne[2]), 768, 0, ctx.stream()>>>(args);
        } else {
            ggml_hip_gqa6_attention<24, 48, false><<<dim3((q->ne[1] + 63)/64, k->ne[2]), 768, 0, ctx.stream()>>>(args);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif
