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

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace cosmos3
{
namespace kernel
{

//! \brief Fused elementwise linear combination: out[i] = c0*v0[i] + c1*v1[i] + c2*v2[i] + c3*v3[i].
//! v2/v3 may be nullptr (their terms are skipped). This single kernel implements every vector
//! operation of the UniPC flow-matching update (x0 conversion, UniC corrector, UniP predictor),
//! keeping the whole denoising state device-resident. All tensors are device FLOAT32 with
//! identical element counts.
void launchFusedCombine(rt::Tensor& out, float c0, rt::Tensor const& v0, float c1, rt::Tensor const& v1, float c2,
    rt::Tensor const* v2, float c3, rt::Tensor const* v3, cudaStream_t stream);

//! \brief Fill the device FLOAT32 tensor `out` with standard-normal noise from a counter-based
//! hash (seeded, order-independent, reproducible for a given seed regardless of grid
//! configuration).
void launchNormalNoiseFill(rt::Tensor& out, uint64_t seed, uint64_t offset, cudaStream_t stream);

//! \brief Compute the unified-3D mRoPE (T,H,W) position planes for the GEN sequence directly on
//! device. Video tokens are t-major over the [tDim, hp, wp] latent grid with fps-modulated *float*
//! temporal positions (double math, matching the reference); action tokens follow with H=W=0. Writes
//! the [B, 3, genLen] FLOAT32 plane layout `initializeMRopeCosSin` consumes (planes are identical
//! across batch). Replaces the former host loop + H2D copy.
void launchBuildMRopePositions(rt::Tensor& positions, int32_t batch, int32_t genLen, int32_t numVideoTokens, int32_t hp,
    int32_t wp, int32_t actionStartFrameOffset, bool videoFpsMod, double mediaOffset, double videoTps,
    double videoBaseTps, double actionTps, double actionBaseTps, cudaStream_t stream);

} // namespace kernel
} // namespace cosmos3
} // namespace trt_edgellm
