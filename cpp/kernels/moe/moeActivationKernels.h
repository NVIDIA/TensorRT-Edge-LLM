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
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

// Kernel-level activation type IDs for moeActivation().
enum MoeActivationType : int32_t
{
    kMoeSwiGlu = 2,
    kMoeRelu2 = 4,
    kMoeGeGlu = 5,
};

/**
 * @brief Apply an FP16 or BF16 MoE activation.
 *
 * Supported activation types:
 * - kMoeSwiGlu: SwiGLU. The input is [numTokens, 2 * intermediateDim], with all gate values followed by all up values.
 * - kMoeRelu2: ReLU2. The input is [numTokens, intermediateDim].
 * - kMoeGeGlu: GeGLU. The input is [numTokens, 2 * intermediateDim], with all gate values followed by all up values.
 *
 * FP16 and BF16 support all activation types. The input and output data types must match.
 *
 * @param input Input tensor (FP16 or BF16, GPU).
 * @param output Output tensor [numTokens, intermediateDim] with the same data type as input.
 * @param numTokens Number of routed token slots.
 * @param intermediateDim Intermediate dimension.
 * @param activationType Activation type from MoeActivationType enum.
 * @param stream CUDA stream.
 *
 * @throws std::runtime_error If the tensor shapes, data types, devices, alignment, or activation type are invalid.
 */
void moeActivation(rt::Tensor const& input, rt::Tensor& output, int64_t numTokens, int64_t intermediateDim,
    int32_t activationType, cudaStream_t stream);

//! Apply SwiGLU/GeGLU when gate and up projections use independent per-expert global scales.
void moeScaledGatedActivation(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& projectionGlobalScales,
    rt::Tensor const& expertIndices, int64_t numTokens, int64_t intermediateDim, int32_t activationType,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
