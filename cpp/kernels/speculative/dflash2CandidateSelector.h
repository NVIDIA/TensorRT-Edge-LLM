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

constexpr int32_t kDFlash2MaxSelectorTopK = 16;
constexpr int32_t kDFlash2MaxSelectorRank = 256;

void launchDFlash2CandidateSelector(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor const& uniforms, rt::Tensor const& temperatures,
    rt::Tensor const& greedyMask, rt::Tensor& proposalTokenIds, rt::Tensor& proposalProbabilities, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
