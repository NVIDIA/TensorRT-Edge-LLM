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

/*
 * CUDA kernels for AlpamayoExpertRunner.
 */

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

void kvCacheReshapeRepeat(float* dst, half const* src, int32_t numLayers, int32_t numKVHeads, int32_t seqLen,
    int32_t headDim, int32_t numCandidates, cudaStream_t stream);

void buildPositionIds(int64_t* posIds, int32_t numCandidates, int32_t numTokens, int64_t basePos, cudaStream_t stream);

void fillTimestep(float* dst, int32_t numCandidates, float tVal, cudaStream_t stream);

void eulerUpdate(float* x, float const* v, float dt, int32_t n, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
