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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace pi05
{

//! \brief Device-side flow-matching Euler update: x += dt * v, and stage the
//! next timestep scalar in the same launch.
//! \param x Current trajectory, FLOAT32 [count].
//! \param v Velocity from the action engine, FLOAT32 [count].
//! \param timestep Device scalar the next enqueue reads, FLOAT32 [batch]; null
//!        when the modulation is hoisted and the graph has no timestep input.
void launchEulerStep(float* x, float const* v, float dt, int64_t count, float* timestep, float nextT, int32_t batch,
    cudaStream_t stream);

} // namespace pi05
} // namespace trt_edgellm
