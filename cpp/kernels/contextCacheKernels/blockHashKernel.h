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
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace rt
{

//! Number of parallel chunks used by the chunked FNV-1a hash algorithm. Each chunk is hashed independently by one
//! thread, then all per-chunk digests are reduced into a final hash via sequential FNV-1a chaining. Both the GPU
//! kernel and CPU fallback use this constant to ensure bit-identical results.
inline constexpr int32_t kHASH_NUM_CHUNKS = 2048;

//! Number of CTAs to spread Phase 1 across multiple SMs.
inline constexpr int32_t kHASH_NUM_CTAS = 32;

//! Launch a multi-CTA parallel FNV-1a Phase 1 kernel on device data.
//!
//! The kernel splits the payload into @p numChunks equal chunks distributed across @p numCtas CTAs.
//! Each thread hashes its chunk independently and writes the 128-bit partial digest to @p partialsOut
//! as two consecutive uint64_t (hi, lo) per chunk. The caller must synchronize @p stream then perform
//! the Phase 2 reduction on CPU.
void launchFnv1aHashChunksKernel(uint8_t const* deviceData, size_t totalSize, int32_t numChunks, int32_t numCtas,
    uint64_t* partialsOut, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm
