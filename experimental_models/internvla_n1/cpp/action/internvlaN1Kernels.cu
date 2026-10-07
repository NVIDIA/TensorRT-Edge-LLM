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

#include "action/internvlaN1Kernels.h"

namespace trt_edgellm
{
namespace internvla_n1
{

namespace
{

constexpr int32_t kBlockSize = 256;

__global__ void guidedEulerStepKernel(
    float* latents, float const* modelOutput, int64_t count, float guidanceScale, float dt)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }
    float const uncond = modelOutput[idx];
    float const cond = modelOutput[idx + count];
    float const pred = uncond + guidanceScale * (cond - uncond);
    latents[idx] += dt * pred;
}

__global__ void duplicateBatchKernel(float* dst, float const* src, int64_t count)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }
    float const value = src[idx];
    dst[idx] = value;
    dst[idx + count] = value;
}

int32_t gridFor(int64_t count)
{
    return static_cast<int32_t>((count + kBlockSize - 1) / kBlockSize);
}

} // namespace

void launchGuidedEulerStep(
    float* latents, float const* modelOutput, int64_t count, float guidanceScale, float dt, cudaStream_t stream)
{
    if (count <= 0)
    {
        return;
    }
    guidedEulerStepKernel<<<gridFor(count), kBlockSize, 0, stream>>>(latents, modelOutput, count, guidanceScale, dt);
}

void launchDuplicateBatch(float* dst, float const* src, int64_t count, cudaStream_t stream)
{
    if (count <= 0)
    {
        return;
    }
    duplicateBatchKernel<<<gridFor(count), kBlockSize, 0, stream>>>(dst, src, count);
}

} // namespace internvla_n1
} // namespace trt_edgellm
