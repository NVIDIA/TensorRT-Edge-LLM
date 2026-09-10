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

#pragma once

#if defined(CUTE_DSL_RMSNORM_ENABLED)

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_rmsnorm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <NvInferRuntime.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

//! Per-launch parameters for RMSNorm over a flattened [rows, hiddenSize] tensor.
struct CuteDslRmsNormParams
{
    void const* input{};
    void const* gamma{};
    void* output{};
    int32_t rows{};
    int32_t hiddenSize{};
    float rmsNormEps{};
    int32_t weightBeforeCast{};
    nvinfer1::DataType dataType{nvinfer1::DataType::kHALF};
};

//! Dispatches the homogeneous FP16/BF16 CuTe DSL RMSNorm AOT variants.
class CuteDslRmsNormRunner
{
public:
    CuteDslRmsNormRunner() = delete;

    static constexpr int32_t kMinRows{1};
    static constexpr int32_t kMaxRows{65536};

    static constexpr bool isSupportedHiddenSize(int32_t hiddenSize) noexcept
    {
        return hiddenSize == 4096 || hiddenSize == 5120 || hiddenSize == 7168 || hiddenSize == 8192;
    }

    static bool canImplement(int32_t rows, int32_t hiddenSize, int32_t smVersion, nvinfer1::DataType dataType) noexcept;
    static int32_t run(CuteDslRmsNormParams const& params, cudaStream_t stream) noexcept;

private:
    enum class Variant : int32_t
    {
        kNone,
        kFp16H4096Wbc0,
        kFp16H4096Wbc1,
        kFp16H5120Wbc0,
        kFp16H5120Wbc1,
        kFp16H7168Wbc0,
        kFp16H7168Wbc1,
        kFp16H8192Wbc0,
        kFp16H8192Wbc1,
        kBf16H4096Wbc0,
        kBf16H4096Wbc1,
        kBf16H5120Wbc0,
        kBf16H5120Wbc1,
        kBf16H7168Wbc0,
        kBf16H7168Wbc1,
        kBf16H8192Wbc0,
        kBf16H8192Wbc1,
    };

    static Variant selectVariant(int32_t hiddenSize, nvinfer1::DataType dataType, int32_t weightBeforeCast) noexcept;
    static bool ensureKernelModule(Variant variant, cudaStream_t stream) noexcept;

    static detail::LazyKernelModule<rmsnorm_fp16_h4096_wbc0_Kernel_Module_t> sFp16H4096Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h4096_wbc1_Kernel_Module_t> sFp16H4096Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h5120_wbc0_Kernel_Module_t> sFp16H5120Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h5120_wbc1_Kernel_Module_t> sFp16H5120Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h7168_wbc0_Kernel_Module_t> sFp16H7168Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h7168_wbc1_Kernel_Module_t> sFp16H7168Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h8192_wbc0_Kernel_Module_t> sFp16H8192Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_fp16_h8192_wbc1_Kernel_Module_t> sFp16H8192Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h4096_wbc0_Kernel_Module_t> sBf16H4096Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h4096_wbc1_Kernel_Module_t> sBf16H4096Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h5120_wbc0_Kernel_Module_t> sBf16H5120Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h5120_wbc1_Kernel_Module_t> sBf16H5120Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h7168_wbc0_Kernel_Module_t> sBf16H7168Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h7168_wbc1_Kernel_Module_t> sBf16H7168Wbc1Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h8192_wbc0_Kernel_Module_t> sBf16H8192Wbc0Module;
    static detail::LazyKernelModule<rmsnorm_bf16_h8192_wbc1_Kernel_Module_t> sBf16H8192Wbc1Module;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_RMSNORM_ENABLED)
