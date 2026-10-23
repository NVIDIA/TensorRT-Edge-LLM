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

#include "cuteDslFp8BlockwiseGemmRunner.h"

#if defined(CUTE_DSL_FP8_BLOCKWISE_GEMM_ENABLED)
#include "common/cudaUtils.h"
#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_fp8_blockwise_gemm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

#include <cstddef>
#include <cstdint>

namespace trt_edgellm::kernels
{
namespace
{

constexpr int32_t kTargetSm{110};
constexpr int32_t kBlockSize{128};
constexpr size_t kTensorAlignment{16};

#if defined(CUTE_DSL_FP8_BLOCKWISE_GEMM_ENABLED)
detail::LazyKernelModule<fp8_blockwise_gemm_Kernel_Module_t> gFp8BlockwiseGemmModule{};

bool isAligned(void const* pointer) noexcept
{
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % kTensorAlignment == 0;
}

cudaError_t getDeviceInfo(int32_t& smVersion, int32_t& maxActiveClusters) noexcept
{
    try
    {
        smVersion = getSMVersion();
        maxActiveClusters = getDeviceMultiProcessorCount();
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
    return smVersion > 0 && maxActiveClusters > 0 ? cudaSuccess : cudaErrorInvalidDevice;
}

bool loadModule(cudaStream_t stream) noexcept
{
    return detail::ensureModuleLoaded<fp8_blockwise_gemm_Kernel_Module_Load, fp8_blockwise_gemm_Kernel_Module_Unload>(
        gFp8BlockwiseGemmModule, "fp8_blockwise_gemm", stream);
}
#endif

} // namespace

cudaError_t CuteDslFp8BlockwiseGemmRunner::prepare(cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_FP8_BLOCKWISE_GEMM_ENABLED)
    if (stream == nullptr)
    {
        return cudaErrorInvalidResourceHandle;
    }

    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = getDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    (void) maxActiveClusters;
    if (smVersion != kTargetSm)
    {
        return cudaErrorNotSupported;
    }
    return loadModule(stream) ? cudaSuccess : cudaErrorInitializationError;
#else
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

bool CuteDslFp8BlockwiseGemmRunner::isSupported(
    int32_t smVersion, int32_t numTokens, int32_t outFeatures, int32_t inFeatures) noexcept
{
#if defined(CUTE_DSL_FP8_BLOCKWISE_GEMM_ENABLED)
    return smVersion == kTargetSm && numTokens > 0 && outFeatures > 0 && outFeatures % kBlockSize == 0 && inFeatures > 0
        && inFeatures % kBlockSize == 0;
#else
    (void) smVersion;
    (void) numTokens;
    (void) outFeatures;
    (void) inFeatures;
    return false;
#endif
}

cudaError_t CuteDslFp8BlockwiseGemmRunner::run(
    CuteDslFp8BlockwiseGemmParams const& params, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_FP8_BLOCKWISE_GEMM_ENABLED)
    if (stream == nullptr || !isAligned(params.activation) || !isAligned(params.weight)
        || !isAligned(params.activationScales) || !isAligned(params.weightScales) || !isAligned(params.output))
    {
        return cudaErrorInvalidValue;
    }

    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = getDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    if (!isSupported(smVersion, params.numTokens, params.outFeatures, params.inFeatures))
    {
        return cudaErrorNotSupported;
    }
    if (!loadModule(stream))
    {
        return cudaErrorInitializationError;
    }

    int32_t const kBlocks = params.inFeatures / kBlockSize;
    int32_t const nBlocks = params.outFeatures / kBlockSize;
    fp8_blockwise_gemm_Tensor_a_t activation{
        const_cast<void*>(params.activation), {params.numTokens, params.inFeatures, 1}, {params.inFeatures, 1}};
    fp8_blockwise_gemm_Tensor_b_t weight{
        const_cast<void*>(params.weight), {params.outFeatures, params.inFeatures, 1}, {params.inFeatures, 1}};
    fp8_blockwise_gemm_Tensor_c_t output{
        params.output, {params.numTokens, params.outFeatures, 1}, {params.outFeatures, 1}};
    fp8_blockwise_gemm_Tensor_sfa_t activationScales{
        const_cast<float*>(params.activationScales), {params.numTokens, kBlocks, 1}, {kBlocks, 1}};
    fp8_blockwise_gemm_Tensor_sfb_t weightScales{
        const_cast<float*>(params.weightScales), {nBlocks, kBlocks, 1}, {kBlocks, 1}};

    int32_t const result = cute_dsl_fp8_blockwise_gemm_wrapper(&gFp8BlockwiseGemmModule.module, &activation, &weight,
        &output, &activationScales, &weightScales, maxActiveClusters, stream);
    return result == 0 ? cudaSuccess : cudaErrorUnknown;
#else
    (void) params;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace trt_edgellm::kernels
