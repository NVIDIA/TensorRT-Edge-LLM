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

#pragma once

#include "common/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kDFlash2MaxBlockSize = 16;
constexpr int32_t kDFlash2ProductionConvKernelSize = 2;
constexpr int32_t kDFlash2ProductionConvGroupSize = 16;

//! Apply one side of DFlash2's per-token dynamic grouped depthwise convolution.
//!
//! hidden: [..., blockSize, hiddenSize]
//! delta:  [..., blockSize, kernelSize, hiddenSize / groupSize]
//! base:   [kernelSize, hiddenSize]
//! residual: optional FP32, same shape as hidden
//! output: same shape as hidden, activation dtype without residual and FP32 with residual
//!
//! A tap never reads across a logical block boundary. Production block sizes are in
//! [2, 16]; kernelSize=2 and groupSize=16 vectorize two adjacent channels.
//! Explicit K=1..4 shapes share the same single-launch backend; unsupported layouts
//! return cudaErrorInvalidValue rather than falling back to decomposed operations.
cudaError_t launchDFlash2GroupedDynamicConv(rt::Tensor const& hidden, rt::Tensor const& delta, rt::Tensor const& base,
    rt::OptionalInputTensor residual, rt::Tensor& output, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm
