#pragma once
// Narrow, measured hipBLASLt dispatch for original Q5 -> F16 prefill operands.
// All resident Q5 tensors and conversion values are unchanged. FP32 accumulation.
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
namespace ggml_hip_q5_lt {
struct plan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algorithm;
    size_t workspace = 0;
    ~plan() {
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (desc) hipblasLtMatmulDescDestroy(desc);
        if (handle) hipblasLtDestroy(handle);
    }
};
static bool run(const half * a, const half * b, float * c, int n, int m, int k,
                void * workspace, size_t workspace_bytes, cudaStream_t stream) {
    const char * enabled = std::getenv("GGML_HIP_Q5_LT");
    if (!enabled || std::strcmp(enabled, "1")) return false;
    int selected = -1;
    if (m == 2048) {
        selected = n == 17408 && k == 5120 ? 2 : n == 5120 && k == 17408 ? 7 : -1;
    } else {
        const char * large = std::getenv("GGML_HIP_Q5_LT_LARGE");
        if (!large || std::strcmp(large, "1")) return false;
        if (m == 4096) selected = n == 17408 && k == 5120 ? 2 : n == 5120 && k == 17408 ? 6 : -1;
        if (m == 8192) selected = n == 17408 && k == 5120 ? 2 : n == 5120 && k == 17408 ? 4 : -1;
    }
    if (selected < 0) return false;
    int device; CUDA_CHECK(cudaGetDevice(&device));
    const char * exhaustive_env = std::getenv("GGML_HIP_Q5_LT_EXHAUSTIVE");
    const bool exhaustive = exhaustive_env && !std::strcmp(exhaustive_env,"1") && m == 2048 && n == 17408 && k == 5120;
    using key = std::tuple<int,int,int,int,bool>;
    static thread_local std::map<key,std::unique_ptr<plan>> plans;
    auto & p = plans[key(device,n,m,k,exhaustive)];
    if (!p) {
        auto next = std::make_unique<plan>();
        auto ok = [](hipblasStatus_t s) { return s == HIPBLAS_STATUS_SUCCESS; };
        if (!ok(hipblasLtCreate(&next->handle)) ||
            !ok(hipblasLtMatmulDescCreate(&next->desc,HIPBLAS_COMPUTE_32F,HIP_R_32F))) return false;
        hipblasOperation_t trans = HIPBLAS_OP_T;
        if (!ok(hipblasLtMatmulDescSetAttribute(next->desc,HIPBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans))) ||
            !ok(hipblasLtMatrixLayoutCreate(&next->a,HIP_R_16F,k,n,k)) ||
            !ok(hipblasLtMatrixLayoutCreate(&next->b,HIP_R_16F,k,m,k)) ||
            !ok(hipblasLtMatrixLayoutCreate(&next->c,HIP_R_32F,n,m,n))) return false;
        bool selected_exhaustive = false;
        if (exhaustive) {
            std::vector<int> indices{136326};
            std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
            const float alpha=1.0f,beta=0.0f;size_t bytes=0;
            if (ok(hipblaslt_ext::getAlgosFromIndex(next->handle,indices,algorithms)) && !algorithms.empty() &&
                    ok(hipblaslt_ext::matmulIsAlgoSupported(next->handle,next->desc,&alpha,next->a,next->b,&beta,
                        next->c,next->c,algorithms[0].algo,bytes)) && bytes <= workspace_bytes) {
                next->algorithm=algorithms[0].algo;next->workspace=bytes;selected_exhaustive=true;
                GGML_LOG_INFO("q5_lt_exhaustive: device=%d n=%d m=%d k=%d algorithm=136326 workspace=%zu fp32_acc=1\n",device,n,m,k,bytes);
            } else GGML_LOG_WARN("q5_lt_exhaustive: measured algorithm unavailable; falling back to retained heuristic\n");
        }
        if (!selected_exhaustive) {
            hipblasLtMatmulPreference_t pref;
            if (!ok(hipblasLtMatmulPreferenceCreate(&pref))) return false;
            bool ready = ok(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
            hipblasLtMatmulHeuristicResult_t results[32]; int count = 0;
            ready = ready && ok(hipblasLtMatmulAlgoGetHeuristic(next->handle,next->desc,next->a,next->b,next->c,next->c,pref,32,results,&count));
            hipblasLtMatmulPreferenceDestroy(pref);
            if (!ready || count <= selected || results[selected].state != HIPBLAS_STATUS_SUCCESS || results[selected].workspaceSize > workspace_bytes) return false;
            next->algorithm = results[selected].algo;
            next->workspace = results[selected].workspaceSize;
            GGML_LOG_INFO("q5_lt: device=%d n=%d m=%d k=%d measured_heuristic=%d workspace=%zu fp32_acc=1\n",device,n,m,k,selected,next->workspace);
        }
        p = std::move(next);
    }
    const float alpha = 1.0f,beta = 0.0f;
    const auto status = hipblasLtMatmul(p->handle,p->desc,&alpha,a,p->a,b,p->b,&beta,c,p->c,c,p->c,&p->algorithm,workspace,p->workspace,stream);
    if (status != HIPBLAS_STATUS_SUCCESS) GGML_ABORT("q5_lt execution failed: %d",int(status));
    return true;
}
}
