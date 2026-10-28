/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

#include "cuteDslInt4Gemv.h"
#include "common/cudaUtils.h"

// CUDA 12.0-12.7 AOT headers use cudaLibrary_t, which the runtime does not
// declare until 12.8. CUDA 11.4 headers use CUmodule, and CUlibrary does not exist.
#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    return static_cast<cudaError_t>(cuLibraryUnload(lib));
}
#endif // CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
#endif

#include "cutedsl_all.h"

#include <mutex>

namespace trt_edgellm
{
namespace plugins
{

namespace
{

// The compiled GEMV variants (must match build_cutedsl.py's int4_fp16_gemv
// registration): one exported function per M. X(prefix, M).
#define INT4_FP16_GEMV_VARIANTS(X)                                                                                     \
    X(int4_fp16_gemv_m1, 1)                                                                                            \
    X(int4_fp16_gemv_m2, 2)                                                                                            \
    X(int4_fp16_gemv_m3, 3)                                                                                            \
    X(int4_fp16_gemv_m4, 4)                                                                                            \
    X(int4_fp16_gemv_m5, 5)                                                                                            \
    X(int4_fp16_gemv_m6, 6)                                                                                            \
    X(int4_fp16_gemv_m7, 7)                                                                                            \
    X(int4_fp16_gemv_m8, 8)

// Module storage, loaded once.
#define DECL_MODULE(prefix, M) prefix##_Kernel_Module_t g_##prefix{};
INT4_FP16_GEMV_VARIANTS(DECL_MODULE)
#undef DECL_MODULE

bool gLoaded = false;
std::mutex gLoadMutex;

// GEMV group size is baked at 128 (matches build_cutedsl.py); scale rows = ceil(K/128).
constexpr int32_t kGemvGroupSize = 128;
constexpr int32_t kBN = 128; // fragment N-block (shared with the GEMM)
constexpr int32_t kBK = 64;  // fragment K-tile
} // namespace

bool cuteDslInt4GemvSupported(int32_t M)
{
    return M >= 1 && M <= kGemvMaxM;
}

bool cuteDslInt4GemvLoadModules()
{
    std::lock_guard<std::mutex> lock(gLoadMutex);
    if (gLoaded)
    {
        return true;
    }
    cudaFree(nullptr); // ensure a current context for the (driver-API on CUDA<12.8) loads
    (void) cudaGetLastError();
    bool ok = true;
#define LOAD_MODULE(prefix, M)                                                                                         \
    prefix##_Kernel_Module_Load(&g_##prefix);                                                                          \
    ok = ok && (g_##prefix.module != nullptr);
    INT4_FP16_GEMV_VARIANTS(LOAD_MODULE)
#undef LOAD_MODULE
    if (!ok || cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    gLoaded = true;
    return true;
}

// Marshal tensor structs and call the M-variant's generated wrapper.
#define SET_2D(t, ptr, d0, d1)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)

int32_t cuteDslInt4GemvLaunch(int32_t M, void const* A, void const* fragW, void const* scales, void* Out, int32_t N,
    int32_t K, cudaStream_t stream)
{
    if (!gLoaded)
    {
        return -1;
    }
    int32_t const qwRows = divUp(N, kBN) * divUp(K, kBK) * (kBK / 16) * (kBN / 64);
    int32_t const scaleRows = divUp(K, kGemvGroupSize);
#define DISPATCH(prefix, VM)                                                                                           \
    if (M == (VM))                                                                                                     \
    {                                                                                                                  \
        prefix##_Tensor_mA_t tA{};                                                                                     \
        SET_2D(tA, A, M, K);                                                                                           \
        prefix##_Tensor_mQW_t tW{};                                                                                    \
        SET_2D(tW, fragW, qwRows, 128);                                                                                \
        prefix##_Tensor_mScales_t tS{};                                                                                \
        SET_2D(tS, scales, scaleRows, N);                                                                              \
        prefix##_Tensor_mOut_t tOut{};                                                                                 \
        SET_2D(tOut, Out, M, N);                                                                                       \
        return cute_dsl_##prefix##_wrapper(&g_##prefix, &tA, &tW, &tS, &tOut, stream);                                 \
    }
    INT4_FP16_GEMV_VARIANTS(DISPATCH)
#undef DISPATCH
    return -1; // M outside [1, kGemvMaxM]
}

#undef SET_2D
#undef INT4_FP16_GEMV_VARIANTS

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
