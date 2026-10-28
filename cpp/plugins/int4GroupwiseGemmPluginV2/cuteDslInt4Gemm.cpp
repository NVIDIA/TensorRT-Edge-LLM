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

#include "cuteDslInt4Gemm.h"
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
// The compiled variant set (must match build_cutedsl.py's int4_fp16_gemm
// registration -- same 16 configs, same order). bN is pinned to 128; bK=64.
// This is the 16 variant subset of the full 4-tile x {2,3,4}-stage x
// {1,2,4,8,16}-split-K = 60 space: the 16 configs that match full 60-config
// autotune within noise (max regret <1.07x) on Orin (SM87) / Thor (SM110) /
// Spark (SM121), by greedy set-cover over clock-locked traces. Shrinks the
// plugin's per-shape autotune candidate set 60 -> 16.
// X(prefix, bM, bN, bK, stages, splitK).
#define INT4_FP16_GEMM_VARIANTS(X)                                                                                     \
    X(int4_fp16_gemm_16x128x64_s3_sk1, 16, 128, 64, 3, 1)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk1, 16, 128, 64, 4, 1)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk2, 16, 128, 64, 4, 2)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk4, 16, 128, 64, 4, 4)                                                              \
    X(int4_fp16_gemm_32x128x64_s2_sk1, 32, 128, 64, 2, 1)                                                              \
    X(int4_fp16_gemm_32x128x64_s3_sk2, 32, 128, 64, 3, 2)                                                              \
    X(int4_fp16_gemm_32x128x64_s3_sk4, 32, 128, 64, 3, 4)                                                              \
    X(int4_fp16_gemm_32x128x64_s4_sk1, 32, 128, 64, 4, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s2_sk1, 64, 128, 64, 2, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s3_sk1, 64, 128, 64, 3, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s3_sk4, 64, 128, 64, 3, 4)                                                              \
    X(int4_fp16_gemm_64x128x64_s4_sk2, 64, 128, 64, 4, 2)                                                              \
    X(int4_fp16_gemm_64x128x64_s4_sk4, 64, 128, 64, 4, 4)                                                              \
    X(int4_fp16_gemm_128x128x64_s2_sk1, 128, 128, 64, 2, 1)                                                            \
    X(int4_fp16_gemm_128x128x64_s2_sk4, 128, 128, 64, 2, 4)                                                            \
    X(int4_fp16_gemm_128x128x64_s4_sk1, 128, 128, 64, 4, 1)

// Module storage, loaded once.
#define DECL_MODULE(prefix, bM, bN, bK, st, sk) prefix##_Kernel_Module_t g_##prefix{};
INT4_FP16_GEMM_VARIANTS(DECL_MODULE)
#undef DECL_MODULE

bool gLoaded = false;
std::mutex gLoadMutex;

#define VARIANT_ROW(prefix, bM, bN, bK, st, sk) CuteDslInt4Variant{bM, bN, bK, st, sk},
CuteDslInt4Variant const kVariants[] = {INT4_FP16_GEMM_VARIANTS(VARIANT_ROW)};
#undef VARIANT_ROW
} // namespace

int32_t cuteDslInt4NumVariants()
{
    return static_cast<int32_t>(sizeof(kVariants) / sizeof(kVariants[0]));
}

CuteDslInt4Variant const& cuteDslInt4VariantAt(int32_t index)
{
    return kVariants[index];
}

bool cuteDslInt4VariantValid(CuteDslInt4Variant const& v, int32_t N, int32_t K)
{
    if (N <= 0 || K <= 0 || K % 64 != 0)
    {
        return false;
    }
    if (v.splitK != 1 && (divUp(K, v.bK) % v.splitK != 0))
    {
        return false;
    }
    return true;
}

int64_t cuteDslInt4RepackedWords(int32_t N, int32_t K, int32_t bN, int32_t bK)
{
    int32_t const kn = (bK / 16) * (bN / 64);
    return static_cast<int64_t>(divUp(N, bN)) * divUp(K, bK) * kn * 128;
}

int64_t cuteDslInt4LockCount(int32_t M, int32_t N, int32_t bM, int32_t bN)
{
    return static_cast<int64_t>(divUp(M, bM)) * divUp(N, bN);
}

bool cuteDslInt4GemmLoadModules()
{
    std::lock_guard<std::mutex> lock(gLoadMutex);
    if (gLoaded)
    {
        return true;
    }
    cudaFree(nullptr); // ensure a current context for the (driver-API on CUDA<12.8) loads
    (void) cudaGetLastError();
    bool ok = true;
#define LOAD_MODULE(prefix, bM, bN, bK, st, sk)                                                                        \
    prefix##_Kernel_Module_Load(&g_##prefix);                                                                          \
    ok = ok && (g_##prefix.module != nullptr);
    INT4_FP16_GEMM_VARIANTS(LOAD_MODULE)
#undef LOAD_MODULE
    if (!ok || cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    gLoaded = true;
    return true;
}

// Marshal tensor structs and call the variant's generated wrapper.
#define SET_2D(t, ptr, d0, d1)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)
#define SET_1D(t, ptr, d0)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
    } while (0)

int32_t cuteDslInt4GemmLaunch(CuteDslInt4Variant const& v, void const* A, void const* fragW, void const* scales,
    void* C, void* locks, int32_t M, int32_t N, int32_t K, int32_t swizzle, cudaStream_t stream)
{
    if (!gLoaded)
    {
        return -1;
    }
    int32_t const scaleRows = divUp(K, 128); // group_size == 128 for these variants
#define DISPATCH(prefix, bm, bn, bk, st, sk)                                                                           \
    if (v.bM == (bm) && v.bN == (bn) && v.bK == (bk) && v.stages == (st) && v.splitK == (sk))                          \
    {                                                                                                                  \
        int32_t const qwRows = static_cast<int32_t>(cuteDslInt4RepackedWords(N, K, bn, bk) / 128);                     \
        int32_t const nTiles = static_cast<int32_t>(cuteDslInt4LockCount(M, N, bm, bn));                               \
        prefix##_Tensor_mA_t tA{};                                                                                     \
        SET_2D(tA, A, M, K);                                                                                           \
        prefix##_Tensor_mQW_t tW{};                                                                                    \
        SET_2D(tW, fragW, qwRows, 128);                                                                                \
        prefix##_Tensor_mScales_t tS{};                                                                                \
        SET_2D(tS, scales, scaleRows, N);                                                                              \
        prefix##_Tensor_mC_t tC{};                                                                                     \
        SET_2D(tC, C, M, N);                                                                                           \
        prefix##_Tensor_mWorkspace_t tWs{};                                                                            \
        SET_2D(tWs, C, M, N);                                                                                          \
        prefix##_Tensor_mLocks_t tL{};                                                                                 \
        SET_1D(tL, locks, nTiles);                                                                                     \
        return cute_dsl_##prefix##_wrapper(&g_##prefix, &tA, &tW, &tS, &tC, &tWs, &tL, swizzle, stream);               \
    }
    INT4_FP16_GEMM_VARIANTS(DISPATCH)
#undef DISPATCH
    return -1;
}

#undef SET_2D
#undef SET_1D
#undef INT4_FP16_GEMM_VARIANTS

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
