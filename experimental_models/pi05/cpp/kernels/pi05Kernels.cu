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

#include "kernels/pi05Kernels.h"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace pi05
{

namespace
{

//! x <- x + dt * v, plus the next step's timestep, in one launch. Keeping the update
//! on the device removes the per-step host round trip, which is what lets the whole
//! denoise loop be captured into a single CUDA graph.
__global__ void eulerStepKernel(float* __restrict__ x, float const* __restrict__ v, float dt, int64_t count,
    float* __restrict__ timestep, float nextT, int32_t batch)
{
    int64_t const idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx < count)
    {
        x[idx] += dt * v[idx];
    }
    // One thread refreshes the scalar the next enqueue reads.
    if (timestep != nullptr && idx < batch)
    {
        timestep[idx] = nextT;
    }
}

} // namespace

void launchEulerStep(
    float* x, float const* v, float dt, int64_t count, float* timestep, float nextT, int32_t batch, cudaStream_t stream)
{
    constexpr int32_t kBlock = 256;
    auto const grid = static_cast<int32_t>((count + kBlock - 1) / kBlock);
    eulerStepKernel<<<grid, kBlock, 0, stream>>>(x, v, dt, count, timestep, nextT, batch);
}

} // namespace pi05
} // namespace trt_edgellm
