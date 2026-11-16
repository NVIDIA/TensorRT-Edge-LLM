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

#include "internvlImagePlan.h"

#include <cuda_fp16.h>

namespace trt_edgellm::safety
{

inline constexpr unsigned kInternVLImageBlockWidth{16};
inline constexpr unsigned kInternVLImageBlockHeight{16};
inline constexpr unsigned kInternVLTileBlocksPerDimension{kInternVLTileSize / kInternVLImageBlockWidth};
static_assert(kInternVLTileSize % kInternVLImageBlockWidth == 0);
static_assert(kInternVLImageBlockWidth == kInternVLImageBlockHeight);

__global__ void internvlTransposeTiles(half const* canvas, InternVLTilePlan plan, half* destination)
{
    int32_t const tileColumn = static_cast<int32_t>(blockIdx.x / kInternVLTileBlocksPerDimension);
    int32_t const tileRow = static_cast<int32_t>(blockIdx.y / kInternVLTileBlocksPerDimension);
    int32_t const column
        = static_cast<int32_t>((blockIdx.x % kInternVLTileBlocksPerDimension) * blockDim.x + threadIdx.x);
    int32_t const row = static_cast<int32_t>((blockIdx.y % kInternVLTileBlocksPerDimension) * blockDim.y + threadIdx.y);
    int32_t const channel = static_cast<int32_t>(blockIdx.z);
    if (tileColumn >= plan.columns || tileRow >= plan.rows || column >= kInternVLTileSize || row >= kInternVLTileSize
        || channel >= kInternVLChannels)
    {
        return;
    }
    int32_t const tile = tileRow * plan.columns + tileColumn;
    int32_t const canvasX = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    int32_t const canvasY = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    std::size_t const sourceIndex
        = (static_cast<std::size_t>(canvasY) * plan.resizedWidth + canvasX) * kInternVLChannels + channel;
    std::size_t const destinationIndex = static_cast<std::size_t>(tile) * kInternVLTileValues
        + static_cast<std::size_t>(channel) * kInternVLTilePixels + row * kInternVLTileSize + column;
    destination[destinationIndex] = canvas[sourceIndex];
}

} // namespace trt_edgellm::safety
