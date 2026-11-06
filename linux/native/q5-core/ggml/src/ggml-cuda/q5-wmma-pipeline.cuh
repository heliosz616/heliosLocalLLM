#pragma once
// Derived from WHIRL 0e35407b75dad7f503909cb72b962f3b97a5c9ec, gemv_wp.hip.
// Copyright the WHIRL authors. SPDX-License-Identifier: Apache-2.0
// License: experiments/r9700-reference/whirl-llm-source-pinned-0e35407b75da/LICENSE.
#if defined(GGML_USE_HIP)
namespace ggml_hip_q5_pipeline {
constexpr int WARP = 32;
constexpr int QT_Q4_K = 0, QT_Q5_K = 1, QT_Q6_K = 2, QT_Q8_0 = 3;
using wi2 = __attribute__((ext_vector_type(2))) int;
using wi8 = __attribute__((ext_vector_type(8))) int;
__device__ __forceinline__ wi8 wmma_iu8(int a0,int a1,int b0,int b1,wi8 c) {
#if defined(RDNA4)
    const wi2 a = {a0,a1}, b = {b0,b1};
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a,true,b,c,false);
#else
    return c;
#endif
}
template <int T, int TILES, int NV, int XDS> struct GU;

template <int TILES, int NV, int XDS, int QT> struct GU_K {
    static constexpr int bytes = QT == QT_Q4_K ? 144 : 176;
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 4; }
    struct P {
        int4 a, c;
        uint4 hd[TILES], qv[TILES];
        uint4 hb[QT == QT_Q5_K ? TILES : 1];
        float2 xs[XDS ? 1 : NV];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int u, int h, int t0, int ntm1, int lane) {
        const int b = u >> 2, j = u & 3;
        const int base = b * 256 + 64 * j;
        p.a = *(const int4 *)(xt + base + 16 * h);
        p.c = *(const int4 *)(xt + base + 32 + 16 * h);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * bytes;
            p.hd[tt] = *(const uint4 *)blk;
            p.qv[tt] = *(const uint4 *)(blk + (QT == QT_Q4_K ? 16 : 48) + 32 * j + 16 * h);
            if constexpr (QT == QT_Q5_K) p.hb[tt] = *(const uint4 *)(blk + 16 + 16 * h);
        }
        const int xi = base >> 5;
        if constexpr (XDS) {
            p.xs[0].x = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + (lane >> 4)];
        } else {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, ntm1);
                p.xs[v] = *(const float2 *)(xd + (size_t)t * nb32 + xi);
            }
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int j = u & 3;
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int one = 0x01010101;
        const wi8 xsl = wmma_iu8(p.a.z, p.a.w, one, one, wmma_iu8(p.a.x, p.a.y, one, one, z));
        const wi8 xsh = wmma_iu8(p.c.z, p.c.w, one, one, wmma_iu8(p.c.x, p.c.y, one, one, z));
        float x0v[NV], x1v[NV];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if constexpr (XDS) {
                x0v[v] = __shfl(p.xs[0].x, t0 + v, WARP);
                x1v[v] = __shfl(p.xs[0].x, 16 + t0 + v, WARP);
            } else {
                x0v[v] = p.xs[v].x;
                x1v[v] = p.xs[v].y;
            }
        }
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint4 hd = p.hd[tt];
            const float d = __half2float(__ushort_as_half((uint16_t)(hd.x & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((uint16_t)(hd.x >> 16)));
            const int sel = 16 * (j & 1);
            const uint32_t A = hd.y >> sel, B = hd.z >> sel, C = hd.w >> sel;
            int s0, m0, s1, m1;
            if constexpr (!J2) {
                s0 = A & 63;
                s1 = (A >> 8) & 63;
                m0 = B & 63;
                m1 = (B >> 8) & 63;
            } else {
                s0 = (C & 0xF) | (((A >> 6) & 3) << 4);
                s1 = ((C >> 8) & 0xF) | (((A >> 14) & 3) << 4);
                m0 = ((C >> 4) & 0xF) | (((B >> 6) & 3) << 4);
                m1 = ((C >> 12) & 0xF) | (((B >> 14) & 3) << 4);
            }
            const float ds0 = d * s0, ds1 = d * s1, dm0 = dmin * m0, dm1 = dmin * m1;
            const uint32_t q[4] = {p.qv[tt].x, p.qv[tt].y, p.qv[tt].z, p.qv[tt].w};
            int lo[4], hi[4];
            if constexpr (QT == QT_Q4_K) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] = q[k] & 0x0F0F0F0F;
                    hi[k] = (q[k] >> 4) & 0x0F0F0F0F;
                }
            } else {
                const uint32_t hb[4] = {p.hb[tt].x, p.hb[tt].y, p.hb[tt].z, p.hb[tt].w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] = (q[k] & 0x0F0F0F0F) | (((hb[k] >> (2 * j)) & 0x01010101) << 4);
                    hi[k] = ((q[k] >> 4) & 0x0F0F0F0F) | (((hb[k] >> (2 * j + 1)) & 0x01010101) << 4);
                }
            }
            const wi8 cl = wmma_iu8(p.a.z, p.a.w, lo[2], lo[3], wmma_iu8(p.a.x, p.a.y, lo[0], lo[1], z));
            const wi8 ch = wmma_iu8(p.c.z, p.c.w, hi[2], hi[3], wmma_iu8(p.c.x, p.c.y, hi[0], hi[1], z));
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const float x0 = x0v[v], x1 = x1v[v];
                const int dl = cl[v], dh = ch[v], sl = xsl[v], sh = xsh[v];
                acc[tt][v] += x0 * (ds0 * dl - dm0 * sl) + x1 * (ds1 * dh - dm1 * sh);
            }
        }
    }
};
template <int TILES, int NV, int XDS> struct GU<QT_Q4_K, TILES, NV, XDS> : GU_K<TILES, NV, XDS, QT_Q4_K> {};
template <int TILES, int NV, int XDS> struct GU<QT_Q5_K, TILES, NV, XDS> : GU_K<TILES, NV, XDS, QT_Q5_K> {};

__device__ __forceinline__ uint2 ld64a2(const uint8_t *p) {  // 2-byte aligned 8-byte load
    uint2 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 8);
    return v;
}
template <int TILES, int NV, int XDS> struct GU<QT_Q6_K, TILES, NV, XDS> {
    static __device__ __forceinline__ int units(int ncols) { return ncols / 256 * 8; }
    struct P {
        int2 a01, a23;
        uint2 l[TILES], hq[TILES], scw[TILES];
        uint32_t dd[TILES];
        float4 xs[XDS ? 1 : NV];
    };
    static __device__ __forceinline__ void fetch(P &p, const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32,
                                                 int u, int h, int t0, int ntm1, int lane) {
        const int b = u >> 3, n = (u >> 2) & 1, g = u & 3;
        const int base = b * 256 + 128 * n + 8 * g + 32 * h;
        p.a01 = *(const int2 *)(xt + base);
        p.a23 = *(const int2 *)(xt + base + 64);
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint8_t *blk = wr[tt] + (size_t)b * 210;
            p.l[tt] = ld64a2(blk + 64 * n + 8 * g + 32 * h);
            p.hq[tt] = ld64a2(blk + 128 + 32 * n + 8 * g);
            p.scw[tt] = ld64a2(blk + 192 + 8 * n);
            p.dd[tt] = *(const uint16_t *)(blk + 208);
        }
        const int xi = (b * 256 + 128 * n) >> 5;
        if constexpr (XDS) {
            // lane: token lane & 15, quarters 2 (lane >> 4) .. + 1
            p.xs[0].x = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + 2 * (lane >> 4)];
            p.xs[0].y = xd[(size_t)min(lane & 15, ntm1) * nb32 + xi + 2 * (lane >> 4) + 1];
        } else {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int t = min(t0 + v, ntm1);
                p.xs[v] = *(const float4 *)(xd + (size_t)t * nb32 + xi);
            }
        }
    }
    template <bool J2>
    static __device__ __forceinline__ void compute(const P &p, int u, int h, int t0, float (*acc)[NV]) {
        const int g = u & 3;
        const int is = g >> 1;
        float xq4[NV][4];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if constexpr (XDS) {
                xq4[v][0] = __shfl(p.xs[0].x, t0 + v, WARP);
                xq4[v][1] = __shfl(p.xs[0].y, t0 + v, WARP);
                xq4[v][2] = __shfl(p.xs[0].x, 16 + t0 + v, WARP);
                xq4[v][3] = __shfl(p.xs[0].y, 16 + t0 + v, WARP);
            } else {
                xq4[v][0] = p.xs[v].x;
                xq4[v][1] = p.xs[v].y;
                xq4[v][2] = p.xs[v].z;
                xq4[v][3] = p.xs[v].w;
            }
        }
        const wi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
        const int m0 = h == 0 ? -1 : 0, m1 = ~m0;
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt) {
            const uint2 l = p.l[tt], hq = p.hq[tt], scw = p.scw[tt];
            const float dd = __half2float(__ushort_as_half((uint16_t)p.dd[tt]));
            float scf[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t word = (is + 2 * q) < 4 ? scw.x : scw.y;
                scf[q] = (float)(int8_t)(word >> (8 * ((is + 2 * q) & 3)));
            }
            int wq[2][2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int qq = h + 2 * e;
                const uint32_t L[2] = {l.x, l.y}, H[2] = {hq.x, hq.y};
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const uint32_t v = ((L[k] >> (4 * e)) & 0x0F0F0F0Fu) | (((H[k] >> (2 * qq)) & 0x03030303u) << 4);
                    wq[e][k] = (int)(((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                }
            }
            const wi8 c0 = wmma_iu8(p.a01.x, p.a01.y, wq[0][0] & m0, wq[0][1] & m0, z);
            const wi8 c1 = wmma_iu8(p.a01.x, p.a01.y, wq[0][0] & m1, wq[0][1] & m1, z);
            const wi8 c2 = wmma_iu8(p.a23.x, p.a23.y, wq[1][0] & m0, wq[1][1] & m0, z);
            const wi8 c3 = wmma_iu8(p.a23.x, p.a23.y, wq[1][0] & m1, wq[1][1] & m1, z);
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                const int dv[4] = {c0[v], c1[v], c2[v], c3[v]};
                const float xsq[4] = {xq4[v][0], xq4[v][1], xq4[v][2], xq4[v][3]};
                float total = 0.f;
#pragma unroll
                for (int q = 0; q < 4; ++q) total += scf[q] * xsq[q] * (float)dv[q];
                acc[tt][v] += dd * total;
            }
        }
    }
};

// Each unit is one unchanged 34-byte Q8_0 block and its canonical activation block.
// A WMMA pair sums all 32 signed integer products before applying the F16 scales.
template <int TILES, int NV, int XDS> struct GU<QT_Q8_0,TILES,NV,XDS> {
    static __device__ __forceinline__ int units(int ncols) { return ncols/32; }
    struct P { int4 a; uint4 w[TILES]; uint16_t d[TILES]; float xs[XDS?1:NV]; };
    static __device__ __forceinline__ void fetch(P &p,const uint8_t *const *wr,
        const int8_t *xt,const float *xd,int nb32,int u,int h,int t0,int ntm1,int lane) {
        p.a=*(const int4 *)(xt+32*u+16*h);
        #pragma unroll
        for(int tt=0;tt<TILES;tt++) {
            const uint8_t *blk=wr[tt]+size_t(u)*34;
            // Original row and block boundaries guarantee only two-byte alignment.
            __builtin_memcpy(&p.w[tt],__builtin_assume_aligned(blk+2+16*h,2),16);
            p.d[tt]=*(const uint16_t *)blk;
        }
        if constexpr(XDS) p.xs[0]=xd[size_t(min(lane&15,ntm1))*nb32+u];
        else {
            #pragma unroll
            for(int v=0;v<NV;v++)p.xs[v]=xd[size_t(min(t0+v,ntm1))*nb32+u];
        }
    }
    template<bool J2> static __device__ __forceinline__ void compute(
        const P &p,int u,int h,int t0,float (*acc)[NV]) {
        const wi8 zero={0,0,0,0,0,0,0,0};
        float xscale[NV];
        #pragma unroll
        for(int v=0;v<NV;v++) {
            if constexpr(XDS)xscale[v]=__shfl(p.xs[0],t0+v,WARP);
            else xscale[v]=p.xs[v];
        }
        #pragma unroll
        for(int tt=0;tt<TILES;tt++) {
            const uint4 w=p.w[tt];
            const wi8 dot=wmma_iu8(p.a.z,p.a.w,(int)w.z,(int)w.w,
                                  wmma_iu8(p.a.x,p.a.y,(int)w.x,(int)w.y,zero));
            const float d=__half2float(__ushort_as_half(p.d[tt]));
            #pragma unroll
            for(int v=0;v<NV;v++)acc[tt][v]+=(d*xscale[v])*(float)dot[v];
        }
    }
};


#define GW2_STEP(S, GUARD)                                                                          \
    if (GUARD) {                                                                                    \
        U::fetch(p[(S + DEPTH) % RING], wr, xt, xd, nb32, w + 8 * min(kk + S + DEPTH, cnt - 1), h, t0, NT - 1, lane); \
        U::template compute<J2>(p[S % RING], w + 8 * (kk + S), h, t0, acc[S]);                            \
        _Pragma("unroll") for (int tt = 0; tt < TILES; ++tt)                                        \
            _Pragma("unroll") for (int v = 0; v < NV; ++v) asm volatile("" : "+v"(acc[S][tt][v])); \
    }

template <int T, int NT, int TILES, int XDS, bool J2, int DEPTH>
__device__ __forceinline__ void gw2_loop(const uint8_t *const *wr, const int8_t *xt, const float *xd, int nb32, int nu,
                                         int w, int h, int t0, int lane, float (*acc)[TILES][NT < 8 ? NT : 8]) {
    constexpr int NV = NT < 8 ? NT : 8;
    using U = GU<T, TILES, NV, XDS>;
    const int cnt = (nu - w + 7) >> 3;
    if (cnt <= 0) return;
    constexpr int RING = DEPTH == 1 ? 2 : 4;
    typename U::P p[RING];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) U::fetch(p[d], wr, xt, xd, nb32, w + 8 * min(d, cnt - 1), h, t0, NT - 1, lane);
    int kk = 0;
    for (; kk + 4 <= cnt; kk += 4) {
        GW2_STEP(0, true)
        GW2_STEP(1, true)
        GW2_STEP(2, true)
        GW2_STEP(3, true)
    }
    GW2_STEP(0, kk < cnt)
    GW2_STEP(1, kk + 1 < cnt)
    GW2_STEP(2, kk + 2 < cnt)
}

template <int T, int NT, int TILES, int XDS, int DEPTH = 1>
__device__ __forceinline__ void gw2_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                         const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                         const int *__restrict__ skip, int bid) {
    constexpr int ROWS = 16 * TILES;
    constexpr int NV = NT < 8 ? NT : 8;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);
    const int r0 = bid * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)min(n, NT - 1) * ncols;
    const int nb32 = ncols / 32;
    const int nu = GU<T, TILES, NV, XDS>::units(ncols);
    const int t0 = 8 * h;
    float acc[4][TILES][NV];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
            for (int v = 0; v < NV; ++v) acc[q][tt][v] = 0.f;
    if ((T == QT_Q4_K || T == QT_Q5_K) && (w & 3) >= 2)
        gw2_loop<T, NT, TILES, XDS, true, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, lane, acc);
    else
        gw2_loop<T, NT, TILES, XDS, false, DEPTH>(wr, xt, xd, nb32, nu, w, h, t0, lane, acc);
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = t0 + v;
            if (t < NT) red[w][t][tt * 16 + n] = (acc[0][tt][v] + acc[2][tt][v]) + (acc[1][tt][v] + acc[3][tt][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[4][t][nn]) + (red[2][t][nn] + red[6][t][nn])) +
                        ((red[1][t][nn] + red[5][t][nn]) + (red[3][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }
}

#undef GW2_STEP
// Consume the original cache: norm/GLU fusion may intentionally omit its F32 tensor.
static __global__ void unpack_inputs(const block_q8_1 *x,int8_t *q,float *ds,int k,int kp) {
    const int lane = threadIdx.x;
    const int qb = blockIdx.x;
    const block_q8_1 &v = x[size_t(blockIdx.y)*(kp/32)+qb];
    q[size_t(blockIdx.y)*k+qb*32+lane] = v.qs[lane];
    if (!lane) ds[size_t(blockIdx.y)*(k/32)+qb] = __low2float(v.ds);
}
template<int NT> static __global__ __launch_bounds__(256) void project(
        const uint8_t*w,const int8_t*q,const float*d,float*y,int k,int n) {
    gw2_impl<QT_Q5_K,NT,1,1,3>(w,uint64_t(k/256)*176,q,d,y,k,n,0,nullptr,blockIdx.x);
}
template<int QT,int NT> static __global__ __launch_bounds__(256) void project_aux(
        const uint8_t*w,const int8_t*q,const float*d,float*y,int k,int n) {
    const uint64_t row_bytes=QT==QT_Q6_K?uint64_t(k/256)*210:uint64_t(k/32)*34;
    gw2_impl<QT,NT,1,1,3>(w,row_bytes,q,d,y,k,n,0,nullptr,blockIdx.x);
}
template<int QT> static void launch_aux(const uint8_t*w,const int8_t*q,const float*d,float*y,int k,int n,int m,hipStream_t stream) {
#define AUX_PIPE_CASE(NT) case NT: project_aux<QT,NT><<<dim3(n/16),256,0,stream>>>(w,q,d,y,k,n);break;
    switch(m){AUX_PIPE_CASE(9) AUX_PIPE_CASE(10) AUX_PIPE_CASE(11) AUX_PIPE_CASE(12) AUX_PIPE_CASE(13) AUX_PIPE_CASE(14) AUX_PIPE_CASE(15) AUX_PIPE_CASE(16) default:GGML_ABORT("unsupported wide auxiliary width");}
#undef AUX_PIPE_CASE
}
static bool supported(ggml_backend_cuda_context &ctx,const ggml_tensor *w,const ggml_tensor *x,const ggml_tensor *y) {
    static const bool enabled = [] {
        const char *v = std::getenv("GGML_HIP_Q5_PIPELINE");
        return v && v[0] == '1' && v[1] == 0;
    }();
    static const bool wide16 = [] {
        const char *v=std::getenv("GGML_HIP_Q5_PIPELINE_WIDE16");
        return v && v[0]=='1' && v[1]==0;
    }();
    static const bool aux=[] {const char*v=std::getenv("GGML_HIP_WIDE_AUX_WMMA");return v&&v[0]=='1'&&v[1]==0;}();
    const bool measured_aux=aux&&wide16&&x->ne[1]>=9&&x->ne[1]<=16&&
        ((w->type==GGML_TYPE_Q8_0&&w->ne[1]>=4096)||
         (w->type==GGML_TYPE_Q6_K&&w->ne[0]==5120&&w->ne[1]==248320));
    if (!enabled || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc) ||
            (w->type != GGML_TYPE_Q5_K && !measured_aux) || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 ||
            !ggml_is_matrix(w) || !ggml_is_matrix(x) || !ggml_is_matrix(y) || w->view_src ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(y) ||
            w->ne[0] <= 0 || w->ne[0] > INT_MAX || w->ne[0]%256 || w->ne[1] <= 0 ||
            w->ne[1] > INT_MAX || w->ne[1]%16 || x->ne[1] < 1 || x->ne[1] > (wide16 ? 16 : 8) ||
            x->ne[0] != w->ne[0]) return false;
    return true;
}
static bool try_run(ggml_backend_cuda_context &ctx,const ggml_tensor *w,const ggml_tensor *x,ggml_tensor *y) {
    if (!supported(ctx,w,x,y)) return false;
    const int k = w->ne[0], n = w->ne[1], m = x->ne[1];
    ggml_cuda_pool_alloc<int8_t> codes(ctx.pool(),size_t(m)*k);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(),size_t(m)*(k/32));
    const int kp = GGML_PAD(k,MATRIX_ROW_PADDING);
    const ggml_tensor *key = x;
    while (key->view_src != nullptr && x->data == key->data) key = key->view_src;
    bool cached = false;
    void *q8 = ctx.q8_1_cache_get(key,ctx.curr_stream_no,size_t(m)*kp*sizeof(block_q8_1)/QK8_1,
        k,m,1,1,x->nb[1]/sizeof(float),x->nb[2]/sizeof(float),x->nb[3]/sizeof(float),cached);
    if (!cached) {
        GGML_ASSERT(ctx.q8_1_src_range_ok(x,x->data));
        quantize_row_q8_1_cuda((const float*)x->data,nullptr,q8,w->type,k,
            x->nb[1]/sizeof(float),x->nb[2]/sizeof(float),x->nb[3]/sizeof(float),kp,m,1,1,ctx.stream());
    }
    unpack_inputs<<<dim3(k/32,m),32,0,ctx.stream()>>>((const block_q8_1*)q8,codes.get(),scales.get(),k,kp);
    if(w->type!=GGML_TYPE_Q5_K) {
        if(w->type==GGML_TYPE_Q6_K)launch_aux<QT_Q6_K>((const uint8_t*)w->data,codes.get(),scales.get(),(float*)y->data,k,n,m,ctx.stream());
        else launch_aux<QT_Q8_0>((const uint8_t*)w->data,codes.get(),scales.get(),(float*)y->data,k,n,m,ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        static thread_local bool reported[2]={false,false};const int slot=w->type==GGML_TYPE_Q6_K?0:1;
        if(!reported[slot]) {GGML_LOG_INFO("wide_aux_wmma: original %s, canonical Q8 activation, FP32 scales/accumulation, rows9..16\n",ggml_type_name(w->type));reported[slot]=true;}
        return true;
    }
#define Q5_PIPE_CASE(NT) case NT: project<NT><<<dim3(n/16),256,0,ctx.stream()>>>((const uint8_t*)w->data,codes.get(),scales.get(),(float*)y->data,k,n); break;
    switch (m) {Q5_PIPE_CASE(1) Q5_PIPE_CASE(2) Q5_PIPE_CASE(3) Q5_PIPE_CASE(4) Q5_PIPE_CASE(5) Q5_PIPE_CASE(6) Q5_PIPE_CASE(7) Q5_PIPE_CASE(8) Q5_PIPE_CASE(9) Q5_PIPE_CASE(10) Q5_PIPE_CASE(11) Q5_PIPE_CASE(12) Q5_PIPE_CASE(13) Q5_PIPE_CASE(14) Q5_PIPE_CASE(15) Q5_PIPE_CASE(16)}
#undef Q5_PIPE_CASE
    CUDA_CHECK(cudaGetLastError());
    static thread_local bool reported = false;
    if (!reported) {
        GGML_LOG_INFO("q5_pipeline: original Q5, original Q8 activation arithmetic, FP32 scales/accumulation, depth=3\n");
        reported = true;
    }
    return true;
}
} // namespace ggml_hip_q5_pipeline
#endif
