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

#if defined(CUTE_DSL_RMSNORM_ENABLED)

#include "cuteDslRmsNormRunner.h"

#include "common/logger.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <exception>

namespace trt_edgellm
{

detail::LazyKernelModule<rmsnorm_fp16_h4096_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H4096Wbc0Module{};
detail::LazyKernelModule<rmsnorm_fp16_h4096_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H4096Wbc1Module{};
detail::LazyKernelModule<rmsnorm_fp16_h5120_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H5120Wbc0Module{};
detail::LazyKernelModule<rmsnorm_fp16_h5120_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H5120Wbc1Module{};
detail::LazyKernelModule<rmsnorm_fp16_h7168_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H7168Wbc0Module{};
detail::LazyKernelModule<rmsnorm_fp16_h7168_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H7168Wbc1Module{};
detail::LazyKernelModule<rmsnorm_fp16_h8192_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H8192Wbc0Module{};
detail::LazyKernelModule<rmsnorm_fp16_h8192_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sFp16H8192Wbc1Module{};
detail::LazyKernelModule<rmsnorm_bf16_h4096_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H4096Wbc0Module{};
detail::LazyKernelModule<rmsnorm_bf16_h4096_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H4096Wbc1Module{};
detail::LazyKernelModule<rmsnorm_bf16_h5120_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H5120Wbc0Module{};
detail::LazyKernelModule<rmsnorm_bf16_h5120_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H5120Wbc1Module{};
detail::LazyKernelModule<rmsnorm_bf16_h7168_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H7168Wbc0Module{};
detail::LazyKernelModule<rmsnorm_bf16_h7168_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H7168Wbc1Module{};
detail::LazyKernelModule<rmsnorm_bf16_h8192_wbc0_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H8192Wbc0Module{};
detail::LazyKernelModule<rmsnorm_bf16_h8192_wbc1_Kernel_Module_t> CuteDslRmsNormRunner::sBf16H8192Wbc1Module{};

namespace
{

#if !defined(CUTE_DSL_RMSNORM_ARTIFACT_SM)
#error "CUTE_DSL_RMSNORM_ARTIFACT_SM must identify the linked RMSNorm artifact"
constexpr int32_t kARTIFACT_SM{0};
#else
constexpr int32_t kARTIFACT_SM{CUTE_DSL_RMSNORM_ARTIFACT_SM};
static_assert(kARTIFACT_SM == 80 || kARTIFACT_SM == 86 || kARTIFACT_SM == 87 || kARTIFACT_SM == 90
        || kARTIFACT_SM == 100 || kARTIFACT_SM == 101 || kARTIFACT_SM == 110 || kARTIFACT_SM == 120
        || kARTIFACT_SM == 121,
    "CUTE_DSL_RMSNORM_ARTIFACT_SM must be one of 80, 86, 87, 90, 100, 101, 110, 120, or 121");
#endif

bool isSupportedDataType(nvinfer1::DataType dataType) noexcept
{
    return dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kBF16;
}

template <typename XTensor, typename GammaTensor, typename OutputTensor, auto Wrapper, typename Module>
int32_t launchLegacyVariant(
    detail::LazyKernelModule<Module>& module, CuteDslRmsNormParams const& params, cudaStream_t stream) noexcept
{
    XTensor xTensor{};
    xTensor.data = const_cast<void*>(params.input);
    xTensor.dynamic_shapes[0] = params.rows;

    GammaTensor gammaTensor{};
    gammaTensor.data = const_cast<void*>(params.gamma);

    OutputTensor outputTensor{};
    outputTensor.data = params.output;
    outputTensor.dynamic_shapes[0] = params.rows;

    int32_t const result = Wrapper(
        &module.module, &xTensor, &gammaTensor, &outputTensor, params.rmsNormEps, params.weightBeforeCast, stream);
    if (result != 0)
    {
        LOG_ERROR("CuteDslRmsNormRunner: legacy AOT wrapper failed with code %d", result);
        return result;
    }
    cudaError_t const launchError = cudaGetLastError();
    if (launchError != cudaSuccess)
    {
        LOG_ERROR("CuteDslRmsNormRunner: legacy kernel launch failed: %s (%s)", cudaGetErrorName(launchError),
            cudaGetErrorString(launchError));
        return -1;
    }
    return 0;
}

template <typename XTensor, typename GammaTensor, typename OutputTensor, auto Wrapper, typename Module>
int32_t launchWbc1Variant(
    detail::LazyKernelModule<Module>& module, CuteDslRmsNormParams const& params, cudaStream_t stream) noexcept
{
    XTensor xTensor{};
    xTensor.data = const_cast<void*>(params.input);
    xTensor.dynamic_shapes[0] = params.rows;

    GammaTensor gammaTensor{};
    gammaTensor.data = const_cast<void*>(params.gamma);

    OutputTensor outputTensor{};
    outputTensor.data = params.output;
    outputTensor.dynamic_shapes[0] = params.rows;

    int32_t const result = Wrapper(&module.module, &xTensor, &gammaTensor, &outputTensor,
        static_cast<int64_t>(params.rows), params.rmsNormEps, stream);
    if (result != 0)
    {
        LOG_ERROR("CuteDslRmsNormRunner: WBC1 AOT wrapper failed with code %d", result);
        return result;
    }
    cudaError_t const launchError = cudaGetLastError();
    if (launchError != cudaSuccess)
    {
        LOG_ERROR("CuteDslRmsNormRunner: WBC1 kernel launch failed: %s (%s)", cudaGetErrorName(launchError),
            cudaGetErrorString(launchError));
        return -1;
    }
    return 0;
}

} // namespace

bool CuteDslRmsNormRunner::canImplement(
    int32_t rows, int32_t hiddenSize, int32_t smVersion, nvinfer1::DataType dataType) noexcept
{
    return rows >= kMinRows && rows <= kMaxRows && isSupportedHiddenSize(hiddenSize) && smVersion == kARTIFACT_SM
        && isSupportedDataType(dataType);
}

CuteDslRmsNormRunner::Variant CuteDslRmsNormRunner::selectVariant(
    int32_t hiddenSize, nvinfer1::DataType dataType, int32_t weightBeforeCast) noexcept
{
    bool const wbc1 = weightBeforeCast != 0;
    if (dataType == nvinfer1::DataType::kHALF)
    {
        switch (hiddenSize)
        {
        case 4096: return wbc1 ? Variant::kFp16H4096Wbc1 : Variant::kFp16H4096Wbc0;
        case 5120: return wbc1 ? Variant::kFp16H5120Wbc1 : Variant::kFp16H5120Wbc0;
        case 7168: return wbc1 ? Variant::kFp16H7168Wbc1 : Variant::kFp16H7168Wbc0;
        case 8192: return wbc1 ? Variant::kFp16H8192Wbc1 : Variant::kFp16H8192Wbc0;
        default: return Variant::kNone;
        }
    }
    if (dataType == nvinfer1::DataType::kBF16)
    {
        switch (hiddenSize)
        {
        case 4096: return wbc1 ? Variant::kBf16H4096Wbc1 : Variant::kBf16H4096Wbc0;
        case 5120: return wbc1 ? Variant::kBf16H5120Wbc1 : Variant::kBf16H5120Wbc0;
        case 7168: return wbc1 ? Variant::kBf16H7168Wbc1 : Variant::kBf16H7168Wbc0;
        case 8192: return wbc1 ? Variant::kBf16H8192Wbc1 : Variant::kBf16H8192Wbc0;
        default: return Variant::kNone;
        }
    }
    return Variant::kNone;
}

bool CuteDslRmsNormRunner::ensureKernelModule(Variant variant, cudaStream_t stream) noexcept
{
    switch (variant)
    {
    case Variant::kFp16H4096Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h4096_wbc0_Kernel_Module_Load,
            rmsnorm_fp16_h4096_wbc0_Kernel_Module_Unload>(sFp16H4096Wbc0Module, "rmsnorm_fp16_h4096_wbc0", stream);
    case Variant::kFp16H4096Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h4096_wbc1_Kernel_Module_Load,
            rmsnorm_fp16_h4096_wbc1_Kernel_Module_Unload>(sFp16H4096Wbc1Module, "rmsnorm_fp16_h4096_wbc1", stream);
    case Variant::kFp16H5120Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h5120_wbc0_Kernel_Module_Load,
            rmsnorm_fp16_h5120_wbc0_Kernel_Module_Unload>(sFp16H5120Wbc0Module, "rmsnorm_fp16_h5120_wbc0", stream);
    case Variant::kFp16H5120Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h5120_wbc1_Kernel_Module_Load,
            rmsnorm_fp16_h5120_wbc1_Kernel_Module_Unload>(sFp16H5120Wbc1Module, "rmsnorm_fp16_h5120_wbc1", stream);
    case Variant::kFp16H7168Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h7168_wbc0_Kernel_Module_Load,
            rmsnorm_fp16_h7168_wbc0_Kernel_Module_Unload>(sFp16H7168Wbc0Module, "rmsnorm_fp16_h7168_wbc0", stream);
    case Variant::kFp16H7168Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h7168_wbc1_Kernel_Module_Load,
            rmsnorm_fp16_h7168_wbc1_Kernel_Module_Unload>(sFp16H7168Wbc1Module, "rmsnorm_fp16_h7168_wbc1", stream);
    case Variant::kFp16H8192Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h8192_wbc0_Kernel_Module_Load,
            rmsnorm_fp16_h8192_wbc0_Kernel_Module_Unload>(sFp16H8192Wbc0Module, "rmsnorm_fp16_h8192_wbc0", stream);
    case Variant::kFp16H8192Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_fp16_h8192_wbc1_Kernel_Module_Load,
            rmsnorm_fp16_h8192_wbc1_Kernel_Module_Unload>(sFp16H8192Wbc1Module, "rmsnorm_fp16_h8192_wbc1", stream);
    case Variant::kBf16H4096Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h4096_wbc0_Kernel_Module_Load,
            rmsnorm_bf16_h4096_wbc0_Kernel_Module_Unload>(sBf16H4096Wbc0Module, "rmsnorm_bf16_h4096_wbc0", stream);
    case Variant::kBf16H4096Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h4096_wbc1_Kernel_Module_Load,
            rmsnorm_bf16_h4096_wbc1_Kernel_Module_Unload>(sBf16H4096Wbc1Module, "rmsnorm_bf16_h4096_wbc1", stream);
    case Variant::kBf16H5120Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h5120_wbc0_Kernel_Module_Load,
            rmsnorm_bf16_h5120_wbc0_Kernel_Module_Unload>(sBf16H5120Wbc0Module, "rmsnorm_bf16_h5120_wbc0", stream);
    case Variant::kBf16H5120Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h5120_wbc1_Kernel_Module_Load,
            rmsnorm_bf16_h5120_wbc1_Kernel_Module_Unload>(sBf16H5120Wbc1Module, "rmsnorm_bf16_h5120_wbc1", stream);
    case Variant::kBf16H7168Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h7168_wbc0_Kernel_Module_Load,
            rmsnorm_bf16_h7168_wbc0_Kernel_Module_Unload>(sBf16H7168Wbc0Module, "rmsnorm_bf16_h7168_wbc0", stream);
    case Variant::kBf16H7168Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h7168_wbc1_Kernel_Module_Load,
            rmsnorm_bf16_h7168_wbc1_Kernel_Module_Unload>(sBf16H7168Wbc1Module, "rmsnorm_bf16_h7168_wbc1", stream);
    case Variant::kBf16H8192Wbc0:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h8192_wbc0_Kernel_Module_Load,
            rmsnorm_bf16_h8192_wbc0_Kernel_Module_Unload>(sBf16H8192Wbc0Module, "rmsnorm_bf16_h8192_wbc0", stream);
    case Variant::kBf16H8192Wbc1:
        return detail::ensureModuleLoaded<rmsnorm_bf16_h8192_wbc1_Kernel_Module_Load,
            rmsnorm_bf16_h8192_wbc1_Kernel_Module_Unload>(sBf16H8192Wbc1Module, "rmsnorm_bf16_h8192_wbc1", stream);
    case Variant::kNone: break;
    }
    LOG_ERROR("CuteDslRmsNormRunner: cannot load unknown AOT variant %d", static_cast<int32_t>(variant));
    return false;
}

int32_t CuteDslRmsNormRunner::run(CuteDslRmsNormParams const& params, cudaStream_t stream) noexcept
{
    try
    {
        if (params.input == nullptr || params.gamma == nullptr || params.output == nullptr)
        {
            LOG_ERROR("CuteDslRmsNormRunner: null input, gamma, or output pointer");
            return -1;
        }
        if (!std::isfinite(params.rmsNormEps) || params.rmsNormEps <= 0.0F
            || (params.weightBeforeCast != 0 && params.weightBeforeCast != 1))
        {
            LOG_ERROR("CuteDslRmsNormRunner: invalid epsilon or weight-before-cast mode");
            return -1;
        }

        if (params.rows < kMinRows || params.rows > kMaxRows || !isSupportedHiddenSize(params.hiddenSize)
            || !isSupportedDataType(params.dataType))
        {
            LOG_ERROR("CuteDslRmsNormRunner: unsupported rows=%d H=%d dtype=%d; rows must be in [%d,%d]", params.rows,
                params.hiddenSize, static_cast<int32_t>(params.dataType), kMinRows, kMaxRows);
            return -1;
        }

        Variant const variant = selectVariant(params.hiddenSize, params.dataType, params.weightBeforeCast);
        if (variant == Variant::kNone)
        {
            LOG_ERROR("CuteDslRmsNormRunner: no AOT variant matches H=%d dtype=%d weight-before-cast=%d",
                params.hiddenSize, static_cast<int32_t>(params.dataType), params.weightBeforeCast);
            return -1;
        }
        if (!ensureKernelModule(variant, stream))
        {
            return -1;
        }

        switch (variant)
        {
        case Variant::kFp16H4096Wbc0:
            return launchLegacyVariant<rmsnorm_fp16_h4096_wbc0_Tensor_x_t, rmsnorm_fp16_h4096_wbc0_Tensor_gamma_t,
                rmsnorm_fp16_h4096_wbc0_Tensor_output_t, cute_dsl_rmsnorm_fp16_h4096_wbc0_wrapper>(
                sFp16H4096Wbc0Module, params, stream);
        case Variant::kFp16H4096Wbc1:
            return launchWbc1Variant<rmsnorm_fp16_h4096_wbc1_Tensor_x_t, rmsnorm_fp16_h4096_wbc1_Tensor_gamma_t,
                rmsnorm_fp16_h4096_wbc1_Tensor_output_t, cute_dsl_rmsnorm_fp16_h4096_wbc1_wrapper>(
                sFp16H4096Wbc1Module, params, stream);
        case Variant::kFp16H5120Wbc0:
            return launchLegacyVariant<rmsnorm_fp16_h5120_wbc0_Tensor_x_t, rmsnorm_fp16_h5120_wbc0_Tensor_gamma_t,
                rmsnorm_fp16_h5120_wbc0_Tensor_output_t, cute_dsl_rmsnorm_fp16_h5120_wbc0_wrapper>(
                sFp16H5120Wbc0Module, params, stream);
        case Variant::kFp16H5120Wbc1:
            return launchWbc1Variant<rmsnorm_fp16_h5120_wbc1_Tensor_x_t, rmsnorm_fp16_h5120_wbc1_Tensor_gamma_t,
                rmsnorm_fp16_h5120_wbc1_Tensor_output_t, cute_dsl_rmsnorm_fp16_h5120_wbc1_wrapper>(
                sFp16H5120Wbc1Module, params, stream);
        case Variant::kFp16H7168Wbc0:
            return launchLegacyVariant<rmsnorm_fp16_h7168_wbc0_Tensor_x_t, rmsnorm_fp16_h7168_wbc0_Tensor_gamma_t,
                rmsnorm_fp16_h7168_wbc0_Tensor_output_t, cute_dsl_rmsnorm_fp16_h7168_wbc0_wrapper>(
                sFp16H7168Wbc0Module, params, stream);
        case Variant::kFp16H7168Wbc1:
            return launchWbc1Variant<rmsnorm_fp16_h7168_wbc1_Tensor_x_t, rmsnorm_fp16_h7168_wbc1_Tensor_gamma_t,
                rmsnorm_fp16_h7168_wbc1_Tensor_output_t, cute_dsl_rmsnorm_fp16_h7168_wbc1_wrapper>(
                sFp16H7168Wbc1Module, params, stream);
        case Variant::kFp16H8192Wbc0:
            return launchLegacyVariant<rmsnorm_fp16_h8192_wbc0_Tensor_x_t, rmsnorm_fp16_h8192_wbc0_Tensor_gamma_t,
                rmsnorm_fp16_h8192_wbc0_Tensor_output_t, cute_dsl_rmsnorm_fp16_h8192_wbc0_wrapper>(
                sFp16H8192Wbc0Module, params, stream);
        case Variant::kFp16H8192Wbc1:
            return launchWbc1Variant<rmsnorm_fp16_h8192_wbc1_Tensor_x_t, rmsnorm_fp16_h8192_wbc1_Tensor_gamma_t,
                rmsnorm_fp16_h8192_wbc1_Tensor_output_t, cute_dsl_rmsnorm_fp16_h8192_wbc1_wrapper>(
                sFp16H8192Wbc1Module, params, stream);
        case Variant::kBf16H4096Wbc0:
            return launchLegacyVariant<rmsnorm_bf16_h4096_wbc0_Tensor_x_t, rmsnorm_bf16_h4096_wbc0_Tensor_gamma_t,
                rmsnorm_bf16_h4096_wbc0_Tensor_output_t, cute_dsl_rmsnorm_bf16_h4096_wbc0_wrapper>(
                sBf16H4096Wbc0Module, params, stream);
        case Variant::kBf16H4096Wbc1:
            return launchWbc1Variant<rmsnorm_bf16_h4096_wbc1_Tensor_x_t, rmsnorm_bf16_h4096_wbc1_Tensor_gamma_t,
                rmsnorm_bf16_h4096_wbc1_Tensor_output_t, cute_dsl_rmsnorm_bf16_h4096_wbc1_wrapper>(
                sBf16H4096Wbc1Module, params, stream);
        case Variant::kBf16H5120Wbc0:
            return launchLegacyVariant<rmsnorm_bf16_h5120_wbc0_Tensor_x_t, rmsnorm_bf16_h5120_wbc0_Tensor_gamma_t,
                rmsnorm_bf16_h5120_wbc0_Tensor_output_t, cute_dsl_rmsnorm_bf16_h5120_wbc0_wrapper>(
                sBf16H5120Wbc0Module, params, stream);
        case Variant::kBf16H5120Wbc1:
            return launchWbc1Variant<rmsnorm_bf16_h5120_wbc1_Tensor_x_t, rmsnorm_bf16_h5120_wbc1_Tensor_gamma_t,
                rmsnorm_bf16_h5120_wbc1_Tensor_output_t, cute_dsl_rmsnorm_bf16_h5120_wbc1_wrapper>(
                sBf16H5120Wbc1Module, params, stream);
        case Variant::kBf16H7168Wbc0:
            return launchLegacyVariant<rmsnorm_bf16_h7168_wbc0_Tensor_x_t, rmsnorm_bf16_h7168_wbc0_Tensor_gamma_t,
                rmsnorm_bf16_h7168_wbc0_Tensor_output_t, cute_dsl_rmsnorm_bf16_h7168_wbc0_wrapper>(
                sBf16H7168Wbc0Module, params, stream);
        case Variant::kBf16H7168Wbc1:
            return launchWbc1Variant<rmsnorm_bf16_h7168_wbc1_Tensor_x_t, rmsnorm_bf16_h7168_wbc1_Tensor_gamma_t,
                rmsnorm_bf16_h7168_wbc1_Tensor_output_t, cute_dsl_rmsnorm_bf16_h7168_wbc1_wrapper>(
                sBf16H7168Wbc1Module, params, stream);
        case Variant::kBf16H8192Wbc0:
            return launchLegacyVariant<rmsnorm_bf16_h8192_wbc0_Tensor_x_t, rmsnorm_bf16_h8192_wbc0_Tensor_gamma_t,
                rmsnorm_bf16_h8192_wbc0_Tensor_output_t, cute_dsl_rmsnorm_bf16_h8192_wbc0_wrapper>(
                sBf16H8192Wbc0Module, params, stream);
        case Variant::kBf16H8192Wbc1:
            return launchWbc1Variant<rmsnorm_bf16_h8192_wbc1_Tensor_x_t, rmsnorm_bf16_h8192_wbc1_Tensor_gamma_t,
                rmsnorm_bf16_h8192_wbc1_Tensor_output_t, cute_dsl_rmsnorm_bf16_h8192_wbc1_wrapper>(
                sBf16H8192Wbc1Module, params, stream);
        case Variant::kNone: break;
        }
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslRmsNormRunner failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslRmsNormRunner failed: unknown error");
    }
    return -1;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_RMSNORM_ENABLED)
