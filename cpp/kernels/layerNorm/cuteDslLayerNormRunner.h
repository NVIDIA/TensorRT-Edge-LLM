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

#if defined(CUTE_DSL_LAYERNORM_ENABLED)

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_layernorm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <NvInferRuntime.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace trt_edgellm
{

//! Per-launch parameters for LayerNorm over a flattened [rows, hiddenSize] tensor.
//!
//! Input, gamma, and beta have one homogeneous FP16 or BF16 type. The kernel converts them to FP32 for the mean,
//! variance, normalization, and affine computation, then casts the final result to the input type.
struct CuteDslLayerNormParams
{
    void const* input{};
    void const* gamma{};
    void const* beta{};
    void* output{};
    int32_t rows{};
    int32_t hiddenSize{};
    float epsilon{};
    nvinfer1::DataType dataType{nvinfer1::DataType::kHALF};
};

//! Dispatches the homogeneous FP16/BF16 CuTe DSL LayerNorm AOT variants.
class CuteDslLayerNormRunner
{
public:
    CuteDslLayerNormRunner() = delete;

    static constexpr int32_t kMinRows{1};
    static constexpr int32_t kMaxRows{std::numeric_limits<int32_t>::max()};

    static constexpr bool isSupportedHiddenSize(int32_t hiddenSize) noexcept
    {
        return hiddenSize == 4096 || hiddenSize == 4097 || hiddenSize == 5120 || hiddenSize == 7168
            || hiddenSize == 8192;
    }

    static bool canImplement(int32_t rows, int32_t hiddenSize, int32_t smVersion, nvinfer1::DataType dataType) noexcept;
    static int32_t run(CuteDslLayerNormParams const& params, cudaStream_t stream) noexcept;

private:
    enum class Variant : int32_t
    {
        kNone,
        kFp16H4096,
        kFp16H4097,
        kFp16H5120,
        kFp16H7168,
        kFp16H8192,
        kBf16H4096,
        kBf16H4097,
        kBf16H5120,
        kBf16H7168,
        kBf16H8192,
    };

    static Variant selectVariant(int32_t hiddenSize, nvinfer1::DataType dataType) noexcept;
    static bool ensureKernelModule(Variant variant, cudaStream_t stream) noexcept;

    static detail::LazyKernelModule<layernorm_fp16_h4096_Kernel_Module_t> sFp16H4096Module;
    static detail::LazyKernelModule<layernorm_fp16_h4097_Kernel_Module_t> sFp16H4097Module;
    static detail::LazyKernelModule<layernorm_fp16_h5120_Kernel_Module_t> sFp16H5120Module;
    static detail::LazyKernelModule<layernorm_fp16_h7168_Kernel_Module_t> sFp16H7168Module;
    static detail::LazyKernelModule<layernorm_fp16_h8192_Kernel_Module_t> sFp16H8192Module;
    static detail::LazyKernelModule<layernorm_bf16_h4096_Kernel_Module_t> sBf16H4096Module;
    static detail::LazyKernelModule<layernorm_bf16_h4097_Kernel_Module_t> sBf16H4097Module;
    static detail::LazyKernelModule<layernorm_bf16_h5120_Kernel_Module_t> sBf16H5120Module;
    static detail::LazyKernelModule<layernorm_bf16_h7168_Kernel_Module_t> sBf16H7168Module;
    static detail::LazyKernelModule<layernorm_bf16_h8192_Kernel_Module_t> sBf16H8192Module;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_LAYERNORM_ENABLED)
