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

#include "action/gr00tKernels.h"

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

__global__ void eulerUpdateKernel(float* trajectory, float const* velocity, float dt, int32_t count)
{
    int32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    trajectory[index] += dt * velocity[index];
}

} // namespace

void launchEulerUpdate(float* trajectory, float const* velocity, float dt, int32_t count, cudaStream_t stream)
{
    if (count <= 0)
    {
        return;
    }
    constexpr int32_t kBlock = 256;
    int32_t const grid = (count + kBlock - 1) / kBlock;
    eulerUpdateKernel<<<grid, kBlock, 0, stream>>>(trajectory, velocity, dt, count);
}

} // namespace gr00t
} // namespace trt_edgellm
