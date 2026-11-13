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

#include "cuteDslNvfp4DualGemmRunner.h"
#include "kernels/cuteDslModuleLoader.h"

#include <limits>

#if defined(CUTE_DSL_DUAL_GEMM_NVFP4_ENABLED)
#define CUTE_DSL_CUDA_ERROR_CHECK(expression) trt_edgellm::detail::recordCuteDslCudaError(expression)
#include "cutedsl_gemm_nvfp4_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

namespace trt_edgellm::kernels
{
namespace
{
#if defined(CUTE_DSL_DUAL_GEMM_NVFP4_ENABLED)
constexpr int32_t kFP16_CLUSTER_SIZE = 1;
constexpr int32_t kFIXED_CLUSTER_SIZE = 2;
constexpr int32_t kMIXED_CLUSTER_SIZE = 4;

template <auto Load, auto Unload, typename Module>
cudaError_t prepareModule(std::shared_ptr<void>& module, char const* name)
{
    if (module)
    {
        return cudaSuccess;
    }
    cudaStream_t initStream{};
    auto const error = cudaStreamCreateWithFlags(&initStream, cudaStreamNonBlocking);
    if (error != cudaSuccess)
    {
        return error;
    }
    module = detail::acquireKernelModule<Load, Unload, Module>(name, initStream);
    auto const destroyError = cudaStreamDestroy(initStream);
    return module ? destroyError : cudaErrorInitializationError;
}

template <auto Launch, typename Module>
cudaError_t runQuantized(std::shared_ptr<void> const& handle, Nvfp4DualGemmParams const& p, cudaStream_t stream)
{
    if (!handle)
    {
        return cudaErrorInitializationError;
    }
    auto* module = static_cast<detail::LazyKernelModule<Module>*>(handle.get());
    detail::clearCuteDslCudaError();
    int32_t const result = Launch(&module->module, const_cast<void*>(p.activation), const_cast<void*>(p.packedWeight),
        const_cast<void*>(p.activationScale), const_cast<void*>(p.packedWeightScale), p.output,
        const_cast<void*>(p.alpha), static_cast<int64_t>(p.numTokens), static_cast<int64_t>(p.interSize) * 2,
        static_cast<int64_t>(p.hiddenSize), p.maxActiveClusters, stream, p.outputScale,
        const_cast<float*>(p.globalScale));
    auto const error = detail::takeCuteDslCudaError();
    return error != cudaSuccess ? error : (result == 0 ? cudaSuccess : cudaErrorLaunchFailure);
}
#endif
} // namespace

bool CuteDslNvfp4DualGemmRunner::canImplement(int32_t smVersion) noexcept
{
#if defined(CUTE_DSL_DUAL_GEMM_NVFP4_ENABLED)
    return smVersion == 100 || smVersion == 103 || smVersion == 110;
#else
    (void) smVersion;
    return false;
#endif
}

cudaError_t CuteDslNvfp4DualGemmRunner::prepare(
    int32_t& maxActiveClusters, bool quantizedOutput, int32_t tactic) noexcept
{
    maxActiveClusters = 0;
#if defined(CUTE_DSL_DUAL_GEMM_NVFP4_ENABLED)
    int32_t device{};
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
    {
        return error;
    }
    int32_t major{}, minor{};
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (error != cudaSuccess)
    {
        return error;
    }
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (error != cudaSuccess)
    {
        return error;
    }
    if (!canImplement(major * 10 + minor))
    {
        return cudaErrorNotSupported;
    }
    if (tactic < kDEFAULT_TACTIC || tactic > kNUM_QUANTIZED_TACTICS || (!quantizedOutput && tactic != kDEFAULT_TACTIC))
    {
        return cudaErrorInvalidValue;
    }
    if (!quantizedOutput)
    {
        error = prepareModule<dual_gemm_blackwell_nvfp4_fp16_Kernel_Module_Load,
            dual_gemm_blackwell_nvfp4_fp16_Kernel_Module_Unload, dual_gemm_blackwell_nvfp4_fp16_Kernel_Module_t>(
            mModules[kDEFAULT_TACTIC], "dual_gemm_blackwell_nvfp4_fp16");
    }
    else if (tactic <= kFIXED_2X1_TACTIC)
    {
        error = prepareModule<dual_gemm_blackwell_nvfp4_output_Kernel_Module_Load,
            dual_gemm_blackwell_nvfp4_output_Kernel_Module_Unload, dual_gemm_blackwell_nvfp4_output_Kernel_Module_t>(
            mModules[kFIXED_2X1_TACTIC], "dual_gemm_blackwell_nvfp4_output");
    }
    else if (tactic == kMIXED_2X2_TACTIC)
    {
        error = prepareModule<dual_gemm_blackwell_nvfp4_output_mix_ab_Kernel_Module_Load,
            dual_gemm_blackwell_nvfp4_output_mix_ab_Kernel_Module_Unload,
            dual_gemm_blackwell_nvfp4_output_mix_ab_Kernel_Module_t>(
            mModules[kMIXED_2X2_TACTIC], "dual_gemm_blackwell_nvfp4_output_mix_ab");
    }
    else
    {
        error = prepareModule<dual_gemm_blackwell_nvfp4_output_mix_b_Kernel_Module_Load,
            dual_gemm_blackwell_nvfp4_output_mix_b_Kernel_Module_Unload,
            dual_gemm_blackwell_nvfp4_output_mix_b_Kernel_Module_t>(
            mModules[kMIXED_4X1_TACTIC], "dual_gemm_blackwell_nvfp4_output_mix_b");
    }
    if (error != cudaSuccess)
    {
        return error;
    }
    int32_t smCount{};
    error = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess)
    {
        return error;
    }
    // The persistent grid is expressed in preferred-cluster slots. CUDA may
    // split each slot into fallback clusters without changing the logical grid.
    int32_t const clusterSize = !quantizedOutput
        ? kFP16_CLUSTER_SIZE
        : (tactic <= kFIXED_2X1_TACTIC ? kFIXED_CLUSTER_SIZE : kMIXED_CLUSTER_SIZE);
    maxActiveClusters = smCount / clusterSize;
    return maxActiveClusters > 0 ? cudaSuccess : cudaErrorNotSupported;
#else
    (void) quantizedOutput;
    (void) tactic;
    return cudaErrorNotSupported;
#endif
}

cudaError_t CuteDslNvfp4DualGemmRunner::run(Nvfp4DualGemmParams const& p, cudaStream_t stream) const noexcept
{
#if defined(CUTE_DSL_DUAL_GEMM_NVFP4_ENABLED)
    if (p.activation == nullptr || p.packedWeight == nullptr || p.activationScale == nullptr
        || p.packedWeightScale == nullptr || p.alpha == nullptr || p.output == nullptr || p.numTokens <= 0
        || p.interSize <= 0 || p.interSize > std::numeric_limits<int32_t>::max() / 2 || p.hiddenSize <= 0
        || p.interSize % 128 != 0 || p.hiddenSize % 64 != 0 || p.maxActiveClusters <= 0 || p.tactic < kDEFAULT_TACTIC
        || p.tactic > kNUM_QUANTIZED_TACTICS || (p.outputScale == nullptr && p.tactic != kDEFAULT_TACTIC))
    {
        return cudaErrorInvalidValue;
    }
    if (p.outputScale != nullptr)
    {
        if (p.globalScale == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        if (p.tactic <= kFIXED_2X1_TACTIC)
        {
            return runQuantized<cute_dsl_dual_gemm_blackwell_nvfp4_output_wrapper,
                dual_gemm_blackwell_nvfp4_output_Kernel_Module_t>(mModules[kFIXED_2X1_TACTIC], p, stream);
        }
        if (p.tactic == kMIXED_2X2_TACTIC)
        {
            return runQuantized<cute_dsl_dual_gemm_blackwell_nvfp4_output_mix_ab_wrapper,
                dual_gemm_blackwell_nvfp4_output_mix_ab_Kernel_Module_t>(mModules[kMIXED_2X2_TACTIC], p, stream);
        }
        return runQuantized<cute_dsl_dual_gemm_blackwell_nvfp4_output_mix_b_wrapper,
            dual_gemm_blackwell_nvfp4_output_mix_b_Kernel_Module_t>(mModules[kMIXED_4X1_TACTIC], p, stream);
    }
    if (!mModules[kDEFAULT_TACTIC])
    {
        return cudaErrorInitializationError;
    }
    auto* module = static_cast<detail::LazyKernelModule<dual_gemm_blackwell_nvfp4_fp16_Kernel_Module_t>*>(
        mModules[kDEFAULT_TACTIC].get());
    detail::clearCuteDslCudaError();
    int32_t const result = cute_dsl_dual_gemm_blackwell_nvfp4_fp16_wrapper(&module->module,
        const_cast<void*>(p.activation), const_cast<void*>(p.packedWeight), const_cast<void*>(p.activationScale),
        const_cast<void*>(p.packedWeightScale), p.output, const_cast<void*>(p.alpha), static_cast<int64_t>(p.numTokens),
        static_cast<int64_t>(p.interSize) * 2, static_cast<int64_t>(p.hiddenSize), p.maxActiveClusters, stream);
    auto const error = detail::takeCuteDslCudaError();
    return error != cudaSuccess ? error : (result == 0 ? cudaSuccess : cudaErrorLaunchFailure);
#else
    (void) p;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace trt_edgellm::kernels
