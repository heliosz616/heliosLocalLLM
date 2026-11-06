#pragma once
// Adapted from WHIRL attn_kg_impl, commit 0e35407b75dad7f503909cb72b962f3b97a5c9ec.
// SPDX-License-Identifier: Apache-2.0
// Upstream source and full license are preserved in
// experiments/r9700-reference/whirl-llm-source-pinned-0e35407b75da/.
// Adaptations: original ggml Q8_0 bytes, arbitrary tensor strides, full dense or
// compact visibility (including holes), attention-only output without gate fusion.
// Two waves split keys, exchange complete scores once per 32-key tile, and each
// computes half of the output dimensions. This does not split the QK reduction.
static __device__ __forceinline__ uint4 ggml_hip_keys_q8_8(const char * row, int dim) {
    const block_q8_0 * b = (const block_q8_0 *)row + dim/32;
    int8_t q[8]; ggml_cuda_memcpy_1<8,2>(q,b->qs+dim%32);
    half2 z[4]; const half2 scale=__half2half2(b->d);
#pragma unroll
    for(int e=0;e<4;e++)z[e]=scale*make_half2(q[2*e],q[2*e+1]);
    return *(const uint4 *)z;
}
static __global__ __launch_bounds__(384) void ggml_hip_gqa6_key_split(ggml_hip_gqa6_args a) {
#if defined(RDNA4)
    using h8=__attribute__((ext_vector_type(8))) _Float16;
    using f8=__attribute__((ext_vector_type(8))) float;
    __shared__ f8 scores_shared[2][12][32];
    __shared__ h8 values[2][2][16][32];
    const int lane=threadIdx.x%32,wid=threadIdx.x/32,w=wid&1;
    const int h=blockIdx.y*6+(wid/2),dh=w*128;
    const int token=blockIdx.x*16+lane%16,live=min(token,a.tokens-1);
    const float *qr=(const float *)(a.q+int64_t(live)*a.q_token+int64_t(h)*a.q_head);
    h8 q[16];
#pragma unroll
    for(int st=0;st<16;st++){
        const float4 lo=*(const float4 *)(qr+16*st+8*(lane/16));
        const float4 hi=*(const float4 *)(qr+16*st+8*(lane/16)+4);
        q[st]=h8{(_Float16)lo.x,(_Float16)lo.y,(_Float16)lo.z,(_Float16)lo.w,(_Float16)hi.x,(_Float16)hi.y,(_Float16)hi.z,(_Float16)hi.w};
    }
    f8 out[8]={};float maximum=-1e30f,denom=0.f;
    const int key_row=w*16+lane%16,key_dim=8*(lane/16);
    const bool loader=threadIdx.x<256;
    const int octet=threadIdx.x/64,dq=threadIdx.x%64;
    int raw[8];_Float16 scale[8];
#define KEYS_FETCH(K0) do { \
    _Pragma("unroll") for(int r=0;r<8;r++){ \
      const int key=min((K0)+octet*8+r,a.cache-1),dim=dq*4; \
      const block_q8_0 *b=(const block_q8_0 *)(a.v+int64_t(blockIdx.y)*a.v_head+int64_t(key)*a.v_token)+dim/32; \
      ggml_cuda_memcpy_1<4,2>(&raw[r],b->qs+dim%32); \
      scale[r]=__builtin_bit_cast(_Float16,b->d); \
    } \
} while(0)
#define KEYS_STORE(BUF) do { \
    _Pragma("unroll") for(int c=0;c<4;c++){h8 col; \
      _Pragma("unroll") for(int r=0;r<8;r++)col[r]=(_Float16)(int8_t)(raw[r]>>(8*c))*scale[r]; \
      const int dim=dq*4+c; \
      values[(BUF)][octet>>1][dim>>4][(dim&15)+16*(octet&1)]=col; \
    } \
} while(0)
    if(loader){KEYS_FETCH(0);KEYS_STORE(0);}
    int parity=0;
    constexpr float log2e=1.44269504088896340736f;
    for(int start=0;start<a.cache;start+=32,parity^=1){
        f8 own={};
        const int key=min(start+key_row,a.cache-1);
        const char *kr=a.k+int64_t(blockIdx.y)*a.k_head+int64_t(key)*a.k_token;
#pragma unroll
        for(int st=0;st<16;st++){
            if(st%4==0&&st>0)__builtin_amdgcn_sched_barrier(0);
            const h8 k=__builtin_bit_cast(h8,ggml_hip_keys_q8_8(kr,16*st+key_dim));
            own=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k,q[st],own);
        }
        scores_shared[parity][wid][lane]=own;
        __syncthreads();
        const bool more=start+32<a.cache;
        if(loader&&more)KEYS_FETCH(start+32);
        f8 s[2]={scores_shared[parity][wid&~1][lane],scores_shared[parity][wid|1][lane]};
        float mx=-INFINITY;
#pragma unroll
        for(int kt=0;kt<2;kt++){
#pragma unroll
            for(int v=0;v<8;v++){
                const int index=start+kt*16+8*(lane/16)+v;
                const float bias=index<a.cache?ggml_hip_gqa6_mask_bias(a,live,index):-INFINITY;
                s[kt][v]=s[kt][v]*a.scale+bias;
                mx=fmaxf(mx,s[kt][v]);
            }
        }
        mx=fmaxf(mx,__shfl_xor(mx,16,32));
        const float next=fmaxf(maximum,mx),alpha=__builtin_amdgcn_exp2f((maximum-next)*log2e);
        float sum=0;h8 p[2];
#pragma unroll
        for(int kt=0;kt<2;kt++){
#pragma unroll
            for(int v=0;v<8;v++){
                const float probability=__builtin_amdgcn_exp2f((s[kt][v]-next)*log2e);
                sum+=probability;p[kt][v]=(_Float16)probability;
            }
        }
        sum+=__shfl_xor(sum,16,32);denom=denom*alpha+sum;maximum=next;
        if(loader&&more)KEYS_STORE(parity^1);
        if(__builtin_amdgcn_ballot_w32(alpha!=1.f)){
#pragma unroll
            for(int dt=0;dt<8;dt++)out[dt]*=alpha;
        }
#pragma unroll
        for(int dt=0;dt<8;dt++){
            if(dt==4)__builtin_amdgcn_sched_barrier(0);
            out[dt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(values[parity][0][w*8+dt][lane],p[0],out[dt]);
            out[dt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(values[parity][1][w*8+dt][lane],p[1],out[dt]);
        }
    }
#undef KEYS_FETCH
#undef KEYS_STORE
    if(token>=a.tokens)return;
    const float inv=denom>0.f?1.f/denom:0.f;
    float *dst=a.output+int64_t(token)*a.heads*256+int64_t(h)*256;
#pragma unroll
    for(int dt=0;dt<8;dt++){
#pragma unroll
        for(int v=0;v<8;v++)dst[dh+16*dt+8*(lane/16)+v]=out[dt][v]*inv;
    }
#endif
}
