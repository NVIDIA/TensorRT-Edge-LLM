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

#include "cuteDslDualGemmRunner.h"

#if defined(CUTE_DSL_GEMM_BLACKWELL_ENABLED)
#include "kernels/contextAttentionKernels/cuteDslTensorDescriptors.h"
#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

namespace trt_edgellm
{
namespace kernels
{
namespace
{

#if defined(CUTE_DSL_GEMM_BLACKWELL_ENABLED)

detail::LazyKernelModule<dual_gemm_blackwell_decode_fp16_Kernel_Module_t> gDecodeModule{};
detail::LazyKernelModule<dual_gemm_blackwell_prefill_fp16_Kernel_Module_t> gPrefillModule{};
detail::LazyKernelModule<dual_gemm_blackwell_large_fp16_Kernel_Module_t> gLargeModule{};

enum class Variant
{
    kDecode,
    kPrefill,
    kLarge,
};

Variant selectVariant(int32_t numTokens) noexcept
{
    if (numTokens <= dual_gemm_blackwell::kDecodeMaxM)
    {
        return Variant::kDecode;
    }
    return numTokens <= dual_gemm_blackwell::kPrefillMaxM ? Variant::kPrefill : Variant::kLarge;
}

// params == nullptr only loads the module.
template <auto Wrapper, auto Loader, auto Unloader>
cudaError_t loadOrLaunchVariant(detail::LazyKernelModule<cutedsl::WrapperArgT<0, decltype(Wrapper)>>& state,
    char const* moduleName, DualGemmBlackwellParams const* params, cudaStream_t stream) noexcept
{
    static_assert(cutedsl::WrapperArity<decltype(Wrapper)>::value == 5,
        "Dual GEMM wrapper must take module, activation, packed weight, output, and stream.");
    if (!detail::ensureModuleLoaded<Loader, Unloader>(state, moduleName, stream))
    {
        return cudaErrorInitializationError;
    }
    if (params == nullptr)
    {
        return cudaSuccess;
    }

    // 3D ABI: (rows, columns, L=1) with strides (columns, 1, rows*columns).
    auto a = cutedsl::makeStridedTensor<cutedsl::WrapperArgT<1, decltype(Wrapper)>>(params->activation,
        {params->numTokens, params->hiddenSize, 1},
        {static_cast<int64_t>(params->hiddenSize), static_cast<int64_t>(params->numTokens) * params->hiddenSize});
    auto w = cutedsl::makeStridedTensor<cutedsl::WrapperArgT<2, decltype(Wrapper)>>(params->packedWeight,
        {2 * params->interSize, params->hiddenSize, 1},
        {static_cast<int64_t>(params->hiddenSize), static_cast<int64_t>(2) * params->interSize * params->hiddenSize});
    auto c = cutedsl::makeStridedTensor<cutedsl::WrapperArgT<3, decltype(Wrapper)>>(params->output,
        {params->numTokens, params->interSize, 1},
        {static_cast<int64_t>(params->interSize), static_cast<int64_t>(params->numTokens) * params->interSize});

    detail::clearCuteDslCudaError();
    int32_t const ret = Wrapper(&state.module, &a, &w, &c, stream);
    cudaError_t const recorded = detail::takeCuteDslCudaError();
    if (recorded != cudaSuccess)
    {
        return recorded;
    }
    return ret == 0 ? cudaSuccess : cudaErrorLaunchFailure;
}

cudaError_t loadOrLaunch(int32_t numTokens, DualGemmBlackwellParams const* params, cudaStream_t stream) noexcept
{
    cudaError_t result = cudaErrorInvalidValue;
    switch (selectVariant(numTokens))
    {
    case Variant::kDecode:
        result = loadOrLaunchVariant<cute_dsl_dual_gemm_blackwell_decode_fp16_wrapper,
            dual_gemm_blackwell_decode_fp16_Kernel_Module_Load, dual_gemm_blackwell_decode_fp16_Kernel_Module_Unload>(
            gDecodeModule, "dual_gemm_blackwell_decode_fp16", params, stream);
        break;
    case Variant::kPrefill:
        result = loadOrLaunchVariant<cute_dsl_dual_gemm_blackwell_prefill_fp16_wrapper,
            dual_gemm_blackwell_prefill_fp16_Kernel_Module_Load, dual_gemm_blackwell_prefill_fp16_Kernel_Module_Unload>(
            gPrefillModule, "dual_gemm_blackwell_prefill_fp16", params, stream);
        break;
    case Variant::kLarge:
        result = loadOrLaunchVariant<cute_dsl_dual_gemm_blackwell_large_fp16_wrapper,
            dual_gemm_blackwell_large_fp16_Kernel_Module_Load, dual_gemm_blackwell_large_fp16_Kernel_Module_Unload>(
            gLargeModule, "dual_gemm_blackwell_large_fp16", params, stream);
        break;
    }
    return result;
}

#endif // CUTE_DSL_GEMM_BLACKWELL_ENABLED

} // namespace

bool CuteDslDualGemmRunner::canImplement(int32_t smVersion) noexcept
{
#if defined(CUTE_DSL_GEMM_BLACKWELL_ENABLED)
    return smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110;
#else
    (void) smVersion;
    return false;
#endif
}

cudaError_t CuteDslDualGemmRunner::prepare(int32_t numTokens, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_GEMM_BLACKWELL_ENABLED)
    if (numTokens <= 0)
    {
        return cudaErrorInvalidValue;
    }
    return loadOrLaunch(numTokens, nullptr, stream);
#else
    (void) numTokens;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t CuteDslDualGemmRunner::run(DualGemmBlackwellParams const& params, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_GEMM_BLACKWELL_ENABLED)
    if (params.activation == nullptr || params.packedWeight == nullptr || params.output == nullptr
        || params.numTokens <= 0 || !dual_gemm_blackwell::isSupportedProblemShape(params.interSize, params.hiddenSize))
    {
        return cudaErrorInvalidValue;
    }
    return loadOrLaunch(params.numTokens, &params, stream);
#else
    (void) params;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace kernels
} // namespace trt_edgellm
