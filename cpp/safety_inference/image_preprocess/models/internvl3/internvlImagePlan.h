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

#include <cmath>
#include <cstdint>
#include <limits>

namespace trt_edgellm::safety
{

inline constexpr int32_t kInternVLTileSize{448};
inline constexpr int32_t kInternVLChannels{3};
inline constexpr int32_t kInternVLTilePixels{kInternVLTileSize * kInternVLTileSize};
inline constexpr int32_t kInternVLTileValues{kInternVLChannels * kInternVLTilePixels};
inline constexpr int32_t kMaxInternVLTiles{64};

struct InternVLTilePlan
{
    int32_t columns{0};
    int32_t rows{0};
    int32_t resizedWidth{0};
    int32_t resizedHeight{0};
    int32_t mainTiles{0};
    bool thumbnail{false};
    int32_t totalTiles{0};
    std::size_t imageTokenCount{0};
};

struct TileGridCandidate
{
    int32_t columns{0};
    int32_t rows{0};
    int32_t blockCount{0};
    double aspectDifference{std::numeric_limits<double>::max()};
    bool valid{false};
};

inline bool isBetterInternVLGrid(
    TileGridCandidate const& candidate, TileGridCandidate const& best, int64_t imageArea) noexcept
{
    if (!best.valid || candidate.aspectDifference < best.aspectDifference)
    {
        return true;
    }
    if (candidate.aspectDifference > best.aspectDifference)
    {
        return false;
    }
    int64_t constexpr kTILE_AREA{static_cast<int64_t>(kInternVLTilePixels)};
    bool const candidateAboveThreshold = imageArea > kTILE_AREA * candidate.blockCount / 2;
    bool const bestAboveThreshold = imageArea > kTILE_AREA * best.blockCount / 2;
    if (candidateAboveThreshold != bestAboveThreshold)
    {
        return candidateAboveThreshold;
    }
    if (candidate.blockCount != best.blockCount)
    {
        return candidate.blockCount < best.blockCount;
    }
    return candidate.columns < best.columns;
}

inline bool computeInternVLTilePlan(int32_t width, int32_t height, int32_t minBlocks, int32_t maxMainBlocks,
    bool addThumbnail, InternVLTilePlan& output) noexcept
{
    output = {};
    if (width <= 0 || height <= 0 || minBlocks <= 0 || minBlocks > maxMainBlocks || maxMainBlocks > kMaxInternVLTiles)
    {
        return false;
    }
    double const inputAspect = static_cast<double>(width) / static_cast<double>(height);
    TileGridCandidate best{};
    for (int32_t blocks = minBlocks; blocks <= maxMainBlocks; ++blocks)
    {
        for (int32_t columns = 1; columns <= blocks; ++columns)
        {
            if (blocks % columns != 0)
            {
                continue;
            }
            int32_t const rows = blocks / columns;
            TileGridCandidate const candidate{columns, rows, blocks,
                std::abs(inputAspect - static_cast<double>(columns) / static_cast<double>(rows)), true};
            if (isBetterInternVLGrid(candidate, best, static_cast<int64_t>(width) * height))
            {
                best = candidate;
            }
        }
    }
    if (!best.valid || best.columns > std::numeric_limits<int32_t>::max() / kInternVLTileSize
        || best.rows > std::numeric_limits<int32_t>::max() / kInternVLTileSize)
    {
        return false;
    }
    bool const thumbnail = addThumbnail && best.blockCount > 1;
    output = {best.columns, best.rows, best.columns * kInternVLTileSize, best.rows * kInternVLTileSize, best.blockCount,
        thumbnail, best.blockCount + (thumbnail ? 1 : 0), 0};
    return true;
}

} // namespace trt_edgellm::safety
