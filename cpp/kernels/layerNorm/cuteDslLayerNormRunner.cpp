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

#if defined(CUTE_DSL_LAYERNORM_ENABLED)

#include "cuteDslLayerNormRunner.h"

#include "common/logger.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <exception>

namespace trt_edgellm
{

detail::LazyKernelModule<layernorm_fp16_h4096_Kernel_Module_t> CuteDslLayerNormRunner::sFp16H4096Module{};
detail::LazyKernelModule<layernorm_fp16_h4097_Kernel_Module_t> CuteDslLayerNormRunner::sFp16H4097Module{};
detail::LazyKernelModule<layernorm_fp16_h5120_Kernel_Module_t> CuteDslLayerNormRunner::sFp16H5120Module{};
detail::LazyKernelModule<layernorm_fp16_h7168_Kernel_Module_t> CuteDslLayerNormRunner::sFp16H7168Module{};
detail::LazyKernelModule<layernorm_fp16_h8192_Kernel_Module_t> CuteDslLayerNormRunner::sFp16H8192Module{};
detail::LazyKernelModule<layernorm_bf16_h4096_Kernel_Module_t> CuteDslLayerNormRunner::sBf16H4096Module{};
detail::LazyKernelModule<layernorm_bf16_h4097_Kernel_Module_t> CuteDslLayerNormRunner::sBf16H4097Module{};
detail::LazyKernelModule<layernorm_bf16_h5120_Kernel_Module_t> CuteDslLayerNormRunner::sBf16H5120Module{};
detail::LazyKernelModule<layernorm_bf16_h7168_Kernel_Module_t> CuteDslLayerNormRunner::sBf16H7168Module{};
detail::LazyKernelModule<layernorm_bf16_h8192_Kernel_Module_t> CuteDslLayerNormRunner::sBf16H8192Module{};

namespace
{

#if !defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM) && !defined(CUTE_DSL_LAYERNORM_MULTI_ARCH_ENABLED)
#error "The linked LayerNorm artifact SM metadata is missing"
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM)
constexpr int32_t kARTIFACT_SM{CUTE_DSL_LAYERNORM_ARTIFACT_SM};
static_assert(kARTIFACT_SM == 80 || kARTIFACT_SM == 86 || kARTIFACT_SM == 87 || kARTIFACT_SM == 90
        || kARTIFACT_SM == 100 || kARTIFACT_SM == 101 || kARTIFACT_SM == 110 || kARTIFACT_SM == 120
        || kARTIFACT_SM == 121,
    "CUTE_DSL_LAYERNORM_ARTIFACT_SM must be one of 80, 86, 87, 90, 100, 101, 110, 120, or 121");
#endif

bool isArtifactSm(int32_t smVersion) noexcept
{
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM)
    return smVersion == kARTIFACT_SM;
#else
    switch (smVersion)
    {
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_80)
    case 80: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_86)
    case 86: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_87)
    case 87: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_90)
    case 90: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_100)
    case 100: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_101)
    case 101: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_110)
    case 110: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_120)
    case 120: return true;
#endif
#if defined(CUTE_DSL_LAYERNORM_ARTIFACT_SM_121)
    case 121: return true;
#endif
    default: return false;
    }
#endif
}

bool isSupportedDataType(nvinfer1::DataType dataType) noexcept
{
    return dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kBF16;
}

template <typename OutputTensor, typename XTensor, typename GammaTensor, typename BetaTensor, auto Wrapper,
    typename Module>
int32_t launchVariant(
    detail::LazyKernelModule<Module>& module, CuteDslLayerNormParams const& params, cudaStream_t stream) noexcept
{
    OutputTensor outputTensor{};
    outputTensor.data = params.output;
    outputTensor.dynamic_shapes[0] = params.rows;

    XTensor xTensor{};
    xTensor.data = const_cast<void*>(params.input);
    xTensor.dynamic_shapes[0] = params.rows;

    GammaTensor gammaTensor{};
    gammaTensor.data = const_cast<void*>(params.gamma);

    BetaTensor betaTensor{};
    betaTensor.data = const_cast<void*>(params.beta);

    int32_t const result = Wrapper(&module.module, &outputTensor, &xTensor, &gammaTensor, &betaTensor,
        static_cast<int64_t>(params.rows), params.epsilon, stream);
    if (result != 0)
    {
        LOG_ERROR("CuteDslLayerNormRunner: AOT wrapper failed with code %d for rows=%d H=%d dtype=%d", result,
            params.rows, params.hiddenSize, static_cast<int32_t>(params.dataType));
        return result;
    }
    cudaError_t const launchError = cudaGetLastError();
    if (launchError != cudaSuccess)
    {
        LOG_ERROR("CuteDslLayerNormRunner: kernel launch failed: %s (%s)", cudaGetErrorName(launchError),
            cudaGetErrorString(launchError));
        return -1;
    }
    return 0;
}

} // namespace

bool CuteDslLayerNormRunner::canImplement(
    int32_t rows, int32_t hiddenSize, int32_t smVersion, nvinfer1::DataType dataType) noexcept
{
    return rows >= kMinRows && rows <= kMaxRows && isSupportedHiddenSize(hiddenSize) && isArtifactSm(smVersion)
        && isSupportedDataType(dataType);
}

CuteDslLayerNormRunner::Variant CuteDslLayerNormRunner::selectVariant(
    int32_t hiddenSize, nvinfer1::DataType dataType) noexcept
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        switch (hiddenSize)
        {
        case 4096: return Variant::kFp16H4096;
        case 4097: return Variant::kFp16H4097;
        case 5120: return Variant::kFp16H5120;
        case 7168: return Variant::kFp16H7168;
        case 8192: return Variant::kFp16H8192;
        default: return Variant::kNone;
        }
    }
    if (dataType == nvinfer1::DataType::kBF16)
    {
        switch (hiddenSize)
        {
        case 4096: return Variant::kBf16H4096;
        case 4097: return Variant::kBf16H4097;
        case 5120: return Variant::kBf16H5120;
        case 7168: return Variant::kBf16H7168;
        case 8192: return Variant::kBf16H8192;
        default: return Variant::kNone;
        }
    }
    return Variant::kNone;
}

bool CuteDslLayerNormRunner::ensureKernelModule(Variant variant, cudaStream_t stream) noexcept
{
    switch (variant)
    {
    case Variant::kFp16H4096:
        return detail::ensureModuleLoaded<layernorm_fp16_h4096_Kernel_Module_Load,
            layernorm_fp16_h4096_Kernel_Module_Unload>(sFp16H4096Module, "layernorm_fp16_h4096", stream);
    case Variant::kFp16H4097:
        return detail::ensureModuleLoaded<layernorm_fp16_h4097_Kernel_Module_Load,
            layernorm_fp16_h4097_Kernel_Module_Unload>(sFp16H4097Module, "layernorm_fp16_h4097", stream);
    case Variant::kFp16H5120:
        return detail::ensureModuleLoaded<layernorm_fp16_h5120_Kernel_Module_Load,
            layernorm_fp16_h5120_Kernel_Module_Unload>(sFp16H5120Module, "layernorm_fp16_h5120", stream);
    case Variant::kFp16H7168:
        return detail::ensureModuleLoaded<layernorm_fp16_h7168_Kernel_Module_Load,
            layernorm_fp16_h7168_Kernel_Module_Unload>(sFp16H7168Module, "layernorm_fp16_h7168", stream);
    case Variant::kFp16H8192:
        return detail::ensureModuleLoaded<layernorm_fp16_h8192_Kernel_Module_Load,
            layernorm_fp16_h8192_Kernel_Module_Unload>(sFp16H8192Module, "layernorm_fp16_h8192", stream);
    case Variant::kBf16H4096:
        return detail::ensureModuleLoaded<layernorm_bf16_h4096_Kernel_Module_Load,
            layernorm_bf16_h4096_Kernel_Module_Unload>(sBf16H4096Module, "layernorm_bf16_h4096", stream);
    case Variant::kBf16H4097:
        return detail::ensureModuleLoaded<layernorm_bf16_h4097_Kernel_Module_Load,
            layernorm_bf16_h4097_Kernel_Module_Unload>(sBf16H4097Module, "layernorm_bf16_h4097", stream);
    case Variant::kBf16H5120:
        return detail::ensureModuleLoaded<layernorm_bf16_h5120_Kernel_Module_Load,
            layernorm_bf16_h5120_Kernel_Module_Unload>(sBf16H5120Module, "layernorm_bf16_h5120", stream);
    case Variant::kBf16H7168:
        return detail::ensureModuleLoaded<layernorm_bf16_h7168_Kernel_Module_Load,
            layernorm_bf16_h7168_Kernel_Module_Unload>(sBf16H7168Module, "layernorm_bf16_h7168", stream);
    case Variant::kBf16H8192:
        return detail::ensureModuleLoaded<layernorm_bf16_h8192_Kernel_Module_Load,
            layernorm_bf16_h8192_Kernel_Module_Unload>(sBf16H8192Module, "layernorm_bf16_h8192", stream);
    case Variant::kNone: break;
    }
    LOG_ERROR("CuteDslLayerNormRunner: cannot load unknown AOT variant %d", static_cast<int32_t>(variant));
    return false;
}

int32_t CuteDslLayerNormRunner::run(CuteDslLayerNormParams const& params, cudaStream_t stream) noexcept
{
    try
    {
        if (params.input == nullptr || params.gamma == nullptr || params.beta == nullptr || params.output == nullptr)
        {
            LOG_ERROR("CuteDslLayerNormRunner: input, gamma, beta, and output pointers must be non-null");
            return -1;
        }
        if (!std::isfinite(params.epsilon) || params.epsilon <= 0.0F)
        {
            LOG_ERROR("CuteDslLayerNormRunner: epsilon must be finite and positive, got %.9g", params.epsilon);
            return -1;
        }
        if (params.rows < kMinRows || params.rows > kMaxRows)
        {
            LOG_ERROR("CuteDslLayerNormRunner: rows=%d is outside the supported range [%d, %d]", params.rows, kMinRows,
                kMaxRows);
            return -1;
        }
        if (!isSupportedHiddenSize(params.hiddenSize))
        {
            LOG_ERROR(
                "CuteDslLayerNormRunner: H=%d is unsupported; expected one of "
                "{4096, 4097, 5120, 7168, 8192}",
                params.hiddenSize);
            return -1;
        }
        if (!isSupportedDataType(params.dataType))
        {
            LOG_ERROR("CuteDslLayerNormRunner: dtype=%d is unsupported; expected FP16 or BF16",
                static_cast<int32_t>(params.dataType));
            return -1;
        }

        Variant const variant = selectVariant(params.hiddenSize, params.dataType);
        if (variant == Variant::kNone)
        {
            LOG_ERROR("CuteDslLayerNormRunner: no AOT variant matches H=%d dtype=%d", params.hiddenSize,
                static_cast<int32_t>(params.dataType));
            return -1;
        }
        if (!ensureKernelModule(variant, stream))
        {
            return -1;
        }

        switch (variant)
        {
        case Variant::kFp16H4096:
            return launchVariant<layernorm_fp16_h4096_Tensor_output_t, layernorm_fp16_h4096_Tensor_x_t,
                layernorm_fp16_h4096_Tensor_gamma_t, layernorm_fp16_h4096_Tensor_beta_t,
                cute_dsl_layernorm_fp16_h4096_wrapper>(sFp16H4096Module, params, stream);
        case Variant::kFp16H4097:
            return launchVariant<layernorm_fp16_h4097_Tensor_output_t, layernorm_fp16_h4097_Tensor_x_t,
                layernorm_fp16_h4097_Tensor_gamma_t, layernorm_fp16_h4097_Tensor_beta_t,
                cute_dsl_layernorm_fp16_h4097_wrapper>(sFp16H4097Module, params, stream);
        case Variant::kFp16H5120:
            return launchVariant<layernorm_fp16_h5120_Tensor_output_t, layernorm_fp16_h5120_Tensor_x_t,
                layernorm_fp16_h5120_Tensor_gamma_t, layernorm_fp16_h5120_Tensor_beta_t,
                cute_dsl_layernorm_fp16_h5120_wrapper>(sFp16H5120Module, params, stream);
        case Variant::kFp16H7168:
            return launchVariant<layernorm_fp16_h7168_Tensor_output_t, layernorm_fp16_h7168_Tensor_x_t,
                layernorm_fp16_h7168_Tensor_gamma_t, layernorm_fp16_h7168_Tensor_beta_t,
                cute_dsl_layernorm_fp16_h7168_wrapper>(sFp16H7168Module, params, stream);
        case Variant::kFp16H8192:
            return launchVariant<layernorm_fp16_h8192_Tensor_output_t, layernorm_fp16_h8192_Tensor_x_t,
                layernorm_fp16_h8192_Tensor_gamma_t, layernorm_fp16_h8192_Tensor_beta_t,
                cute_dsl_layernorm_fp16_h8192_wrapper>(sFp16H8192Module, params, stream);
        case Variant::kBf16H4096:
            return launchVariant<layernorm_bf16_h4096_Tensor_output_t, layernorm_bf16_h4096_Tensor_x_t,
                layernorm_bf16_h4096_Tensor_gamma_t, layernorm_bf16_h4096_Tensor_beta_t,
                cute_dsl_layernorm_bf16_h4096_wrapper>(sBf16H4096Module, params, stream);
        case Variant::kBf16H4097:
            return launchVariant<layernorm_bf16_h4097_Tensor_output_t, layernorm_bf16_h4097_Tensor_x_t,
                layernorm_bf16_h4097_Tensor_gamma_t, layernorm_bf16_h4097_Tensor_beta_t,
                cute_dsl_layernorm_bf16_h4097_wrapper>(sBf16H4097Module, params, stream);
        case Variant::kBf16H5120:
            return launchVariant<layernorm_bf16_h5120_Tensor_output_t, layernorm_bf16_h5120_Tensor_x_t,
                layernorm_bf16_h5120_Tensor_gamma_t, layernorm_bf16_h5120_Tensor_beta_t,
                cute_dsl_layernorm_bf16_h5120_wrapper>(sBf16H5120Module, params, stream);
        case Variant::kBf16H7168:
            return launchVariant<layernorm_bf16_h7168_Tensor_output_t, layernorm_bf16_h7168_Tensor_x_t,
                layernorm_bf16_h7168_Tensor_gamma_t, layernorm_bf16_h7168_Tensor_beta_t,
                cute_dsl_layernorm_bf16_h7168_wrapper>(sBf16H7168Module, params, stream);
        case Variant::kBf16H8192:
            return launchVariant<layernorm_bf16_h8192_Tensor_output_t, layernorm_bf16_h8192_Tensor_x_t,
                layernorm_bf16_h8192_Tensor_gamma_t, layernorm_bf16_h8192_Tensor_beta_t,
                cute_dsl_layernorm_bf16_h8192_wrapper>(sBf16H8192Module, params, stream);
        case Variant::kNone: break;
        }
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslLayerNormRunner failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslLayerNormRunner failed: unknown error");
    }
    return -1;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_LAYERNORM_ENABLED)
