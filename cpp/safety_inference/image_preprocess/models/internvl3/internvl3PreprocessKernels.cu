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

#include "internvl3PreprocessKernels.cuh"

#include "internvlImageKernels.cuh"

#include <cuda_runtime_api.h>

namespace trt_edgellm::safety
{

cudaError_t launchInternVLPreprocess(uint8_t const* source, int32_t width, int32_t height, int32_t rowStride,
    InternVLTilePlan const& plan, RgbNormalization normalization, half* scratch, half* destination,
    cudaStream_t stream) noexcept
{
    if (source == nullptr || scratch == nullptr || destination == nullptr || stream == nullptr || width <= 0
        || height <= 0 || rowStride < width * kInternVLChannels || plan.mainTiles <= 0 || plan.totalTiles <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const expectedTiles = static_cast<int64_t>(plan.columns) * plan.rows;
    if (plan.columns <= 0 || plan.rows <= 0 || expectedTiles != plan.mainTiles
        || plan.resizedWidth != static_cast<int64_t>(plan.columns) * kInternVLTileSize
        || plan.resizedHeight != static_cast<int64_t>(plan.rows) * kInternVLTileSize
        || plan.totalTiles != plan.mainTiles + (plan.thumbnail ? 1 : 0))
    {
        return cudaErrorInvalidValue;
    }
    std::size_t const canvasPixels = static_cast<std::size_t>(plan.resizedWidth) * plan.resizedHeight;
    auto status = launchResizeNormalizeRgb8(
        source, width, height, rowStride, plan.resizedWidth, plan.resizedHeight, normalization, scratch, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    half* thumbnail = scratch + canvasPixels * kInternVLChannels;
    if (plan.thumbnail)
    {
        status = launchResizeNormalizeRgb8(
            source, width, height, rowStride, kInternVLTileSize, kInternVLTileSize, normalization, thumbnail, stream);
        if (status != cudaSuccess)
        {
            return status;
        }
    }
    dim3 const tileGrid{static_cast<unsigned>(plan.columns) * kInternVLTileBlocksPerDimension,
        static_cast<unsigned>(plan.rows) * kInternVLTileBlocksPerDimension, kInternVLChannels};
    dim3 const imageThreads{kInternVLImageBlockWidth, kInternVLImageBlockHeight, 1};
    internvlTransposeTiles<<<tileGrid, imageThreads, 0, stream>>>(scratch, plan, destination);
    status = cudaPeekAtLastError();
    if (status == cudaSuccess && plan.thumbnail)
    {
        InternVLTilePlan const thumbnailPlan{1, 1, kInternVLTileSize, kInternVLTileSize, 1, false, 1, 0};
        dim3 const thumbnailTileGrid{
            kInternVLTileBlocksPerDimension, kInternVLTileBlocksPerDimension, kInternVLChannels};
        internvlTransposeTiles<<<thumbnailTileGrid, imageThreads, 0, stream>>>(
            thumbnail, thumbnailPlan, destination + static_cast<std::size_t>(plan.mainTiles) * kInternVLTileValues);
        status = cudaPeekAtLastError();
    }
    return status;
}

} // namespace trt_edgellm::safety
