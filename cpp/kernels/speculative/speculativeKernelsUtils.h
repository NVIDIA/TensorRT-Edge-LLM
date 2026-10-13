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

#include <cstddef>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

//! Workspace alignment used by speculative CUDA kernels.
constexpr size_t kSpeculativeWorkspaceAlignment{256U};

inline size_t alignSpeculativeWorkspaceSize(size_t size)
{
    return (size + kSpeculativeWorkspaceAlignment - 1U) & ~(kSpeculativeWorkspaceAlignment - 1U);
}

//! Keep a uniform inside [0, 1) so an inverse-CDF walk can never step past the last bucket.
__device__ __forceinline__ float clampUniform(float uniform)
{
    return fminf(fmaxf(uniform, 0.0F), 0.99999994F);
}

__device__ __forceinline__ float toFloat(float value)
{
    return value;
}

__device__ __forceinline__ float toFloat(__half value)
{
    return __half2float(value);
}

} // namespace kernel
} // namespace trt_edgellm
