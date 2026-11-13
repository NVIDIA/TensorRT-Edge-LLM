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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/** Host launcher: count slots per expert (shared-memory reduction). Padded num experts and experts-per-warp are derived
 * internally. */
void launchCountSlotsPerExpertKernel(int32_t const* topkIndices, int32_t* slotsPerExpertWorkspace, int32_t numTokens,
    int32_t topK, int32_t numExperts, cudaStream_t stream);

/** Host launcher: compute padded offsets (CUB BlockScan prefix sum). */
void launchComputePaddedOffsetsKernel(int32_t const* counts, int32_t* paddedCounts, int32_t* paddedOffsets,
    int32_t* numTokensPostPadded, int32_t numExperts, int32_t moeBlockSize, cudaStream_t stream);

/** Host launcher: build per-expert slot lists. */
void launchBuildSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts, cudaStream_t stream);

/** Host launcher: build ascending routed-slot lists per expert in the existing workspace. */
void launchBuildStableSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts, cudaStream_t stream);

/** Return whether the measured grouped stable-list kernel applies to this architecture and expert count. */
bool shouldUseGroupedStableSlotLists(int32_t smVersion, int32_t numExperts);

} // namespace kernel
} // namespace trt_edgellm
