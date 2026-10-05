/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuteDslGemmNvFp4Runner.h"

#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#define NVFP4_LOG(fmt, ...) LOG_ERROR("CuteDslGemmNvFp4Runner: " fmt, ##__VA_ARGS__)

namespace trt_edgellm
{
namespace kernels
{

#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED
gemm_blackwell_nvfp4_fp16_tn64_Kernel_Module_t CuteDslGemmNvFp4Runner::sModFp16Tn64{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED
gemm_blackwell_nvfp4_fp16_tn128_Kernel_Module_t CuteDslGemmNvFp4Runner::sModFp16Tn128{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED
gemm_blackwell_nvfp4_fp8_tn64_Kernel_Module_t CuteDslGemmNvFp4Runner::sModFp8Tn64{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED
gemm_blackwell_nvfp4_fp8_tn128_Kernel_Module_t CuteDslGemmNvFp4Runner::sModFp8Tn128{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED
gemm_blackwell_nvfp4_ws_fp16_tn64_Kernel_Module_t CuteDslGemmNvFp4Runner::sModWsFp16Tn64{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED
gemm_blackwell_nvfp4_ws_fp16_tn128_Kernel_Module_t CuteDslGemmNvFp4Runner::sModWsFp16Tn128{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED
gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t CuteDslGemmNvFp4Runner::sModWsFp8Tn64{};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED
gemm_blackwell_nvfp4_ws_fp8_tn128_Kernel_Module_t CuteDslGemmNvFp4Runner::sModWsFp8Tn128{};
#endif
#endif // CUTE_DSL_GEMM_NVFP4_ENABLED
std::mutex CuteDslGemmNvFp4Runner::sMutex{};
bool CuteDslGemmNvFp4Runner::sLoaded = false;

CuteDslGemmNvFp4Runner::CuteDslGemmNvFp4Runner(int32_t mmaTilerN)
    : mMmaTilerN(mmaTilerN)
{
    if (mmaTilerN != 64 && mmaTilerN != 128)
    {
        throw std::invalid_argument(
            "CuteDslGemmNvFp4Runner: mmaTilerN must be 64 or 128 (got " + std::to_string(mmaTilerN) + ")");
    }
}

CuteDslGemmNvFp4Runner::~CuteDslGemmNvFp4Runner() {}

bool CuteDslGemmNvFp4Runner::loadKernelModules()
{
#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (sLoaded)
    {
        return true;
    }
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED
    gemm_blackwell_nvfp4_fp16_tn64_Kernel_Module_Load(&sModFp16Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED
    gemm_blackwell_nvfp4_fp16_tn128_Kernel_Module_Load(&sModFp16Tn128);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED
    gemm_blackwell_nvfp4_fp8_tn64_Kernel_Module_Load(&sModFp8Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED
    gemm_blackwell_nvfp4_fp8_tn128_Kernel_Module_Load(&sModFp8Tn128);
#endif
    // Load the warp-specialised variants if the artifact carried them.
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED
    gemm_blackwell_nvfp4_ws_fp16_tn64_Kernel_Module_Load(&sModWsFp16Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED
    gemm_blackwell_nvfp4_ws_fp16_tn128_Kernel_Module_Load(&sModWsFp16Tn128);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED
    gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_Load(&sModWsFp8Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED
    gemm_blackwell_nvfp4_ws_fp8_tn128_Kernel_Module_Load(&sModWsFp8Tn128);
#endif
    sLoaded = true;
    return true;
#else
    return false;
#endif
}

void CuteDslGemmNvFp4Runner::unloadKernelModules()
{
#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (!sLoaded)
    {
        return;
    }
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED
    gemm_blackwell_nvfp4_fp16_tn64_Kernel_Module_Unload(&sModFp16Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED
    gemm_blackwell_nvfp4_fp16_tn128_Kernel_Module_Unload(&sModFp16Tn128);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED
    gemm_blackwell_nvfp4_fp8_tn64_Kernel_Module_Unload(&sModFp8Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED
    gemm_blackwell_nvfp4_fp8_tn128_Kernel_Module_Unload(&sModFp8Tn128);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED
    gemm_blackwell_nvfp4_ws_fp16_tn64_Kernel_Module_Unload(&sModWsFp16Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED
    gemm_blackwell_nvfp4_ws_fp16_tn128_Kernel_Module_Unload(&sModWsFp16Tn128);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED
    gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_Unload(&sModWsFp8Tn64);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED
    gemm_blackwell_nvfp4_ws_fp8_tn128_Kernel_Module_Unload(&sModWsFp8Tn128);
#endif
    sLoaded = false;
#endif
}

bool CuteDslGemmNvFp4Runner::canImplement(int32_t smVersion, int32_t mmaTilerN)
{
    // AOT variants were registered for SM 100/101/103/110 (see
    // kernelSrcs/build_cutedsl.py::KERNEL_VARIANTS). canImplement is currently
    // a coarse FP16-or-FP8 check — both output dtypes share the same AOT
    // shape registration so the answer is the same for either path. If the
    // FP16 / FP8 AOT artifact sets diverge, split this into two methods.
    if (smVersion != 100 && smVersion != 101 && smVersion != 103 && smVersion != 110)
    {
        return false;
    }
#if defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED) || defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED)
    if (mmaTilerN == 64)
    {
        return true;
    }
#endif
#if defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED)                                                          \
    || defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED)
    if (mmaTilerN == 128)
    {
        return true;
    }
#endif
    (void) mmaTilerN;
    return false;
}

#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
namespace
{

cudaError_t prepareLaunch(
    char const* pathLabel, int32_t M, int32_t N, int32_t K, void const* alpha, void* dCaller, void*& dPtr)
{
    if (M <= 0 || N <= 0 || K <= 0 || (N % 128) != 0 || (K % 64) != 0)
    {
        NVFP4_LOG("%s: invalid shape M=%d N=%d K=%d (N%%128=%d K%%64=%d)", pathLabel, M, N, K, (N % 128), (K % 64));
        return cudaErrorInvalidValue;
    }
    if (alpha == nullptr)
    {
        NVFP4_LOG("%s: alpha must point at one device-resident FP32 scalar", pathLabel);
        return cudaErrorInvalidValue;
    }
    dPtr = dCaller;
    return cudaSuccess;
}

} // namespace

#endif

cudaError_t CuteDslGemmNvFp4Runner::run(void const* a, void const* b, void const* aSF, void const* bSF, void* d,
    void const* alpha, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
    if (!sLoaded)
    {
        NVFP4_LOG("run: modules not loaded");
        return cudaErrorNotReady;
    }

    void* dPtr = nullptr;
    cudaError_t prepErr = prepareLaunch("run", M, N, K, alpha, d, dPtr);
    if (prepErr != cudaSuccess)
    {
        return prepErr;
    }

    void* aMut = const_cast<void*>(a);
    void* bMut = const_cast<void*>(b);
    void* sfaMut = const_cast<void*>(aSF);
    void* sfbMut = const_cast<void*>(bSF);
    void* alphaMut = const_cast<void*>(alpha);

    int64_t const m64 = static_cast<int64_t>(M);
    int64_t const n64 = static_cast<int64_t>(N);
    int64_t const k64 = static_cast<int64_t>(K);
#if defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED)                                                        \
    || defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED)
    int32_t const maxActiveClusters = trt_edgellm::getDeviceMultiProcessorCount();
#endif

    int32_t launchRet = 0;
    switch (mMmaTilerN)
    {
    case 64:
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED
        launchRet = cute_dsl_gemm_blackwell_nvfp4_ws_fp16_tn64_wrapper(
            &sModWsFp16Tn64, aMut, bMut, sfaMut, sfbMut, dPtr, alphaMut, m64, n64, k64, maxActiveClusters, stream);
        break;
#else
        return cudaErrorNotSupported;
#endif
    case 128:
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED
        launchRet = cute_dsl_gemm_blackwell_nvfp4_ws_fp16_tn128_wrapper(
            &sModWsFp16Tn128, aMut, bMut, sfaMut, sfbMut, dPtr, alphaMut, m64, n64, k64, maxActiveClusters, stream);
        break;
#else
        return cudaErrorNotSupported;
#endif
    default: return cudaErrorNotSupported;
    }
    if (launchRet != 0)
    {
        NVFP4_LOG("cute_dsl FP16 launch returned %d (tn=%d M=%d N=%d K=%d)", launchRet, mMmaTilerN, M, N, K);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
#else
    (void) a;
    (void) b;
    (void) aSF;
    (void) bSF;
    (void) d;
    (void) alpha;
    (void) M;
    (void) N;
    (void) K;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t CuteDslGemmNvFp4Runner::runFp8(void const* a, void const* b, void const* aSF, void const* bSF, void* d,
    void const* alpha, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
    if (!sLoaded)
    {
        NVFP4_LOG("runFp8: modules not loaded");
        return cudaErrorNotReady;
    }

    void* dPtr = nullptr;
    cudaError_t prepErr = prepareLaunch("runFp8", M, N, K, alpha, d, dPtr);
    if (prepErr != cudaSuccess)
    {
        return prepErr;
    }

    void* aMut = const_cast<void*>(a);
    void* bMut = const_cast<void*>(b);
    void* sfaMut = const_cast<void*>(aSF);
    void* sfbMut = const_cast<void*>(bSF);
    void* alphaMut = const_cast<void*>(alpha);

    int64_t const m64 = static_cast<int64_t>(M);
    int64_t const n64 = static_cast<int64_t>(N);
    int64_t const k64 = static_cast<int64_t>(K);
#if defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED)                                                         \
    || defined(CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED)
    int32_t const maxActiveClusters = trt_edgellm::getDeviceMultiProcessorCount();
#endif

    int32_t launchRet = 0;
    switch (mMmaTilerN)
    {
    case 64:
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED
        launchRet = cute_dsl_gemm_blackwell_nvfp4_ws_fp8_tn64_wrapper(
            &sModWsFp8Tn64, aMut, bMut, sfaMut, sfbMut, dPtr, alphaMut, m64, n64, k64, maxActiveClusters, stream);
        break;
#else
        return cudaErrorNotSupported;
#endif
    case 128:
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED
        launchRet = cute_dsl_gemm_blackwell_nvfp4_ws_fp8_tn128_wrapper(
            &sModWsFp8Tn128, aMut, bMut, sfaMut, sfbMut, dPtr, alphaMut, m64, n64, k64, maxActiveClusters, stream);
        break;
#else
        return cudaErrorNotSupported;
#endif
    default: return cudaErrorNotSupported;
    }
    if (launchRet != 0)
    {
        NVFP4_LOG("cute_dsl FP8 launch returned %d (tn=%d M=%d N=%d K=%d)", launchRet, mMmaTilerN, M, N, K);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
#else
    (void) a;
    (void) b;
    (void) aSF;
    (void) bSF;
    (void) d;
    (void) alpha;
    (void) M;
    (void) N;
    (void) K;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace kernels
} // namespace trt_edgellm
