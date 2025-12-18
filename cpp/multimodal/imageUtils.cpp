/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multimodal/imageUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

std::vector<std::pair<int64_t, int64_t>> getAllSupportedAspectRatios(int64_t minImageTiles, int64_t maxImageTiles)
{
    std::vector<std::pair<int64_t, int64_t>> aspectRatios;
    for (int64_t width = 1; width <= maxImageTiles; ++width)
    {
        for (int64_t height = 1; height <= maxImageTiles; ++height)
        {
            if (width * height <= maxImageTiles && width * height >= minImageTiles)
            {
                aspectRatios.emplace_back(width, height);
            }
        }
    }
    std::sort(aspectRatios.begin(), aspectRatios.end(),
        [](std::pair<int64_t, int64_t> const& a, std::pair<int64_t, int64_t> const& b) {
            return a.first * a.second < b.first * b.second;
        });
    return aspectRatios;
}

/*!
 * @brief Choose target resize (H, W) for InternVL/Phi-4-multimodal style vision frontends.
 *
 * Given an input image (height, width) and the allowed token range, this function:
 * 1) Converts token bounds to tile bounds (each tile produces 256 tokens; we also add a thumbnail, so subtract 1).
 * 2) Enumerates candidate aspect-ratio grids (tw, th) within [minTiles, maxTiles].
 * 3) Picks the grid whose aspect ratio tw/th is closest to the original width/height.
 * 4) Tie-breaker: if two grids are equally close by ratio, prefer the one whose pixel capacity
 *    (tw*th * blockImageSizeH * blockImageSizeW) is “sufficiently large” for the current image
 *    (area > 0.5 * blockPixelArea * tw*th), reducing excessive scaling/letterboxing.
 *
 * Return value is the resized (height, width): (th * blockImageSizeH, tw * blockImageSizeW).
 * Note: This version uses floating-point for readability; near-ties are rare in practice.
 */
std::tuple<int64_t, int64_t> computeBestBlockGridForResize(int64_t height, int64_t width,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t blockImageSizeH, int64_t blockImageSizeW)
{
    // -1 because we add a thumbnail image for each image
    int64_t const minImageTiles = std::max<int64_t>(1, minImageTokensPerImage / 256 - 1);
    int64_t const maxImageTiles = std::max<int64_t>(1, maxImageTokensPerImage / 256 - 1);
    auto const targetRatios = getAllSupportedAspectRatios(minImageTiles, maxImageTiles);
    double const aspectRatio = static_cast<double>(width) / static_cast<double>(height);
    int64_t const area = width * height;

    double bestRatioDiff = std::numeric_limits<double>::max();
    std::pair<int64_t, int64_t> bestRatio = {1, 1};
    for (auto const& ratio : targetRatios)
    {
        double const targetAspectRatio = static_cast<double>(ratio.first) / static_cast<double>(ratio.second);
        double const ratioDiff = std::abs(aspectRatio - targetAspectRatio);

        if (ratioDiff < bestRatioDiff)
        {
            bestRatioDiff = ratioDiff;
            bestRatio = ratio;
        }
        else if (ratioDiff == bestRatioDiff)
        {
            // Tie-breaker: prefer grids whose capacity better matches the current image area
            int64_t const baseBlockArea = blockImageSizeH * blockImageSizeW;
            int64_t const targetTileArea = ratio.first * ratio.second;
            int64_t const thresholdArea = (baseBlockArea / 2) * targetTileArea;
            if (area > thresholdArea)
            {
                bestRatio = ratio;
            }
        }
    }
    // return (height, width)
    return {bestRatio.second * blockImageSizeH, bestRatio.first * blockImageSizeW};
}
} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
