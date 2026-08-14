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
namespace internvla_n1
{

//! \brief Fused classifier-free-guidance blend and Euler update.
//!
//! The engine is evaluated on a doubled batch: the first half is conditioned on a null
//! `z_latents`, the second on the real one. This folds the guidance blend and the Euler step
//! into one pass so the trajectory state is read and written once per denoising step rather
//! than three times:
//!
//!     pred = uncond + guidanceScale * (cond - uncond)
//!     latents += dt * pred
//!
//! \param latents      In/out, `[count]` elements -- one trajectory batch, not the doubled one.
//! \param modelOutput  Engine output, `[2 * count]` elements, unconditional half first.
//! \param count        Elements in a single (undoubled) trajectory batch.
//! \param guidanceScale Guidance weight; 1.0 reduces the blend to the conditional branch.
//! \param dt           Euler coefficient for this step (negative -- see the scheduler).
void launchGuidedEulerStep(
    float* latents, float const* modelOutput, int64_t count, float guidanceScale, float dt, cudaStream_t stream);

//! \brief Duplicate a trajectory batch into the doubled layout the engine expects.
//!
//! Writes `src` into both halves of `dst`. The two halves differ only in their conditioning,
//! which is supplied through `z_latents`, not through the latents themselves.
void launchDuplicateBatch(float* dst, float const* src, int64_t count, cudaStream_t stream);

} // namespace internvla_n1
} // namespace trt_edgellm
