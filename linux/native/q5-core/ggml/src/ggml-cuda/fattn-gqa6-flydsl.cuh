#pragma once
// Private adapter for pinned ROCm/aiter FlyDSL gfx1201 attention, Apache-2.0.
// The compiled artifact keeps FP32 accumulation/output and exact derived masks.
#include <map>
#include <cstring>
#include <cstdlib>
namespace ggml_hip_flydsl_attn {
static constexpr int BLOCK_M = 384;
static __global__ void expand_q8(ggml_hip_gqa6_args a, half *k_out, half *v_out) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int groups=a.heads/6;
    if(i>=int64_t(a.cache)*groups*8)return;
    const int dim=(i%8)*32, head=(i/8)%groups, key=i/(8*groups);
    half2 k[16],v[16];
    const char *kp=a.k+head*a.k_head+key*a.k_token;
    const char *vp=a.v+head*a.v_head+key*a.v_token;
    dequantize_V_q8_0<half,16>(kp,k,dim);dequantize_V_q8_0<half,16>(kp,k+8,dim+16);
    dequantize_V_q8_0<half,16>(vp,v,dim);dequantize_V_q8_0<half,16>(vp,v+8,dim+16);
#pragma unroll
    for(int j=0;j<4;j++){
        *(uint4*)(k_out+i*32+j*8)=*(const uint4*)(k+j*4);
        *(uint4*)(v_out+i*32+j*8)=*(const uint4*)(v+j*4);
    }
}
static __global__ void pack_query(ggml_hip_gqa6_args a,half *out) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int groups=a.heads/6;
    if(i>=int64_t(a.tokens)*a.heads*256)return;
    const int d=i%256,g=(i/256)%groups,row=i/(256*groups);
    const int token=row/6,head=g*6+row%6;
    out[i]=__float2half_rn(*(const float*)(a.q+token*a.q_token+head*a.q_head+d*4));
}
static __global__ void unpack_output(ggml_hip_gqa6_args a,const float *packed) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int groups=a.heads/6;
    if(i>=int64_t(a.tokens)*a.heads*256)return;
    const int d=i%256,head=(i/256)%a.heads,token=i/(256*a.heads);
    const int row=token*6+head%6,g=head/6;
    a.output[i]=packed[(int64_t(row)*groups+g)*256+d];
}
struct loaded_module { hipModule_t module=nullptr;hipFunction_t function=nullptr; };
static bool try_run(ggml_backend_cuda_context &ctx,const ggml_hip_gqa6_args&a) {
    static const bool enabled=[] {const char*v=std::getenv("GGML_HIP_FLYDSL_ATTN");return v&&v[0]=='1'&&v[1]==0;}();
    if(!enabled || !a.cell_pos || !a.tok_lo || !a.tok_hi || a.mask || a.heads!=24 ||
            a.tokens<64 || a.cache%32 || a.scale!=0.0625f)return false;
    static thread_local std::map<int,loaded_module> modules;
    auto &loaded=modules[ctx.device];
    if(!loaded.module){
        hipDeviceProp_t props;CUDA_CHECK(hipGetDeviceProperties(&props,ctx.device));
        if(std::strncmp(props.gcnArchName,"gfx1201",7)!=0)return false;
        const char * artifact = std::getenv("LLM_SERVICE_FLYDSL_ARTIFACT");
        if (!artifact || !*artifact) return false; // Optional generated artifact is not distributed.
        CUDA_CHECK(hipModuleLoad(&loaded.module, artifact));
        CUDA_CHECK(hipModuleGetFunction(&loaded.function,loaded.module,"flash_attn_func_kernel_0"));
        GGML_LOG_INFO("flydsl_attention: pinned gfx1201 module, original Q8, FP32 accumulation/output, exact position/range mask, block_m=384\n");
    }
    const size_t kv_count=size_t(a.cache)*(a.heads/6)*256,q_count=size_t(a.tokens)*a.heads*256;
    ggml_cuda_pool_alloc<half> k(ctx.pool(),kv_count),v(ctx.pool(),kv_count),q(ctx.pool(),q_count);
    ggml_cuda_pool_alloc<float> output(ctx.pool(),q_count);
    const int64_t blocks=int64_t(a.cache)*(a.heads/6)*8;
    expand_q8<<<(blocks+127)/128,128,0,ctx.stream()>>>(a,k.get(),v.get());
    pack_query<<<(q_count+255)/256,256,0,ctx.stream()>>>(a,q.get());
    void *qp=q.get(),*kp=k.get(),*vp=v.get(),*op=output.get();
    const int32_t *pos=a.cell_pos,*lo=a.tok_lo,*hi=a.tok_hi;
    const int flat_q=a.tokens*6,cache=a.cache;
    void *args[]={&qp,&kp,&vp,&op,&pos,&lo,&hi,(void*)&flat_q,(void*)&cache};
    CUDA_CHECK(hipModuleLaunchKernel(loaded.function,((flat_q+BLOCK_M-1)/BLOCK_M)*4,1,1,
        BLOCK_M*2,1,1,0,ctx.stream(),args,nullptr));
    unpack_output<<<(q_count+255)/256,256,0,ctx.stream()>>>(a,output.get());
    CUDA_CHECK(cudaGetLastError());return true;
}
} // namespace ggml_hip_flydsl_attn
