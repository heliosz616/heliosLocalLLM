#pragma once
// Separate wave ownership of next-tile loading and current-tile WMMA work.
// Existing original-Q8 dequantization, F16 operands, FP32 accumulation,
// compact/dense masks and softmax arithmetic are retained without conversion.
static __global__ __launch_bounds__(800) void ggml_hip_gqa6_wave_pipeline(ggml_hip_gqa6_args a) {
#if defined(RDNA4)
    using hf8=__attribute__((ext_vector_type(8))) _Float16;
    using hf2=__attribute__((ext_vector_type(2))) _Float16;
    using f8=__attribute__((ext_vector_type(8))) float;
    constexpr int consumers=24,key_tile=16,value_stride=16;
    __shared__ half keys[2][16][264];
    __shared__ half values[2][256][16];
    __shared__ int ready[2],retired[2];
    const int tid=threadIdx.x,lane=tid%32,wave=tid/32;
    if(tid<2){ready[tid]=-1;retired[tid]=consumers;}
    __syncthreads();
    if(wave==consumers) {
        for(int tile=0,start=0;start<a.cache;start+=key_tile,++tile) {
            const int buffer=tile&1;
            if(lane==0) {
                while(__atomic_load_n(&retired[buffer],__ATOMIC_ACQUIRE)!=consumers)
                    __builtin_amdgcn_s_sleep(1);
                __atomic_store_n(&retired[buffer],0,__ATOMIC_RELAXED);
            }
            __syncwarp();
            for(int load=lane;load<key_tile*16;load+=32) {
                const int row=load/16,dim=(load%16)*16;
                half2 k[8]={},v[8]={};
                if(start+row<a.cache) {
                    dequantize_V_q8_0<half,16>(a.k+blockIdx.y*a.k_head+(start+row)*a.k_token,k,dim);
                    dequantize_V_q8_0<half,16>(a.v+blockIdx.y*a.v_head+(start+row)*a.v_token,v,dim);
                }
                *(uint4 *)&keys[buffer][row][dim]=*(const uint4 *)k;
                *(uint4 *)&keys[buffer][row][dim+8]=*(const uint4 *)(k+4);
                #pragma unroll
                for(int e=0;e<16;++e) {
                    const int d=dim+e,swizzle=((d/16+d%16)%2)*8;
                    values[buffer][d][row^swizzle]=((half *)v)[e];
                }
            }
            __threadfence_block();__syncwarp();
            if(lane==0)__atomic_store_n(&ready[buffer],tile,__ATOMIC_RELEASE);
        }
        return;
    }
    const int half_wave=lane/16,column=lane%16,query_row=wave*16+column;
    const int token=blockIdx.x*64+query_row/6,head=blockIdx.y*6+query_row%6;
    const int live_token=min(token,a.tokens-1);
    const float *query=(const float *)(a.q+live_token*a.q_token+head*a.q_head);
    hf8 q[16];
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

    f8 sum[16]={};float reference=-1.0e30f,denominator=0.0f;
    constexpr float log2e=1.4426950408889634f;
    const float score_scale=a.scale*log2e;
    const int end=a.cache,part=0;
    for(int tile=0,start=0;start<end;start+=key_tile,++tile) {
        const int buffer=tile&1;
        if(lane==0) {
            while(__atomic_load_n(&ready[buffer],__ATOMIC_ACQUIRE)!=tile)
                __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_block();__syncwarp();

            f8 scores = {};
#pragma unroll
            for (int g = 0; g < 16; g += 2) {
                hf8 fragment[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int key_row = part*16 + column;
                    const int swizzle = 0;
                    fragment[j] = __builtin_bit_cast(hf8,
                        *(const uint4 *)&keys[buffer][key_row][(16*(g + j) + 8*half_wave) ^ swizzle]);
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    
                        scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fragment[j], q[g + j], scores);
                    
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
                    fragment[j] = __builtin_bit_cast(hf8, *(const uint4 *)&values[buffer][d][key]);
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sum[g + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fragment[j], p, sum[g + j]);
                }
            }
        
        __threadfence_block();__syncwarp();
        if(lane==0)__atomic_fetch_add(&retired[buffer],1,__ATOMIC_RELEASE);
    }
    denominator+=__shfl_xor(denominator,16,32);
    const float inv=denominator>0.0f?1.0f/denominator:0.0f;
    if(token<a.tokens) {
        float *out=a.output+(int64_t(token)*a.heads+head)*256+8*half_wave;
        #pragma unroll
        for(int g=0;g<16;++g) {
            *(float4 *)(out+16*g)=make_float4(sum[g][0]*inv,sum[g][1]*inv,sum[g][2]*inv,sum[g][3]*inv);
            *(float4 *)(out+16*g+4)=make_float4(sum[g][4]*inv,sum[g][5]*inv,sum[g][6]*inv,sum[g][7]*inv);
        }
    }
#else
    GGML_UNUSED(a);NO_DEVICE_CODE;
#endif
}
