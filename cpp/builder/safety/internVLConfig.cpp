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

#include "builder/safety/internVLConfig.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace trt_edgellm::safety::builder
{
int64_t getInternVLTokensPerBlock(nlohmann::json const& modelConfig)
{
    auto const& vision = modelConfig.at("vision_config");
    auto const& image = vision.at("image_size");
    auto const& patch = vision.at("patch_size");
    if ((image.is_array() && image.size() != 2) || (patch.is_array() && patch.size() != 2))
    {
        throw std::invalid_argument("InternVL image_size and patch_size must be scalars or [height,width].");
    }
    if (!modelConfig.contains("downsample_ratio") || !modelConfig.at("downsample_ratio").is_number())
    {
        throw std::invalid_argument("Safety InternVL requires downsample_ratio in config.json; re-export the model.");
    }
    double const ratio = modelConfig.at("downsample_ratio").get<double>();
    if (!std::isfinite(ratio) || ratio <= 0 || ratio > 1)
    {
        throw std::invalid_argument("Safety InternVL downsample_ratio must be finite and in (0,1].");
    }
    double const inverseRatio = 1.0 / ratio;
    int64_t constexpr kMAX_DIMENSION = std::numeric_limits<int32_t>::max();
    if (!std::isfinite(inverseRatio) || inverseRatio > static_cast<double>(kMAX_DIMENSION))
    {
        throw std::invalid_argument("Safety InternVL pixel-shuffle scale exceeds the supported dimensions.");
    }
    // Match the exporter's int(1.0 / scale_factor), including truncation at non-integral reciprocals.
    int64_t const scale = static_cast<int64_t>(inverseRatio);
    int64_t tokensPerTile{1};
    for (int32_t axis = 0; axis < 2; ++axis)
    {
        auto const& imageDimension = image.is_array() ? image.at(axis) : image;
        auto const& patchDimension = patch.is_array() ? patch.at(axis) : patch;
        // Bound dimensions before converting JSON numbers or multiplying token counts.
        if (!imageDimension.is_number_integer() || !patchDimension.is_number_integer() || imageDimension <= 0
            || patchDimension <= 0 || imageDimension > kMAX_DIMENSION || patchDimension > kMAX_DIMENSION)
        {
            throw std::invalid_argument("Safety InternVL image/patch dimensions must be positive int32 values.");
        }
        int64_t const imageSize = imageDimension.get<int64_t>();
        int64_t const patchSize = patchDimension.get<int64_t>();
        if (imageSize % patchSize != 0)
        {
            throw std::invalid_argument("Safety InternVL image dimensions must be divisible by patch dimensions.");
        }
        int64_t const patchGrid = imageSize / patchSize;
        if (patchGrid < scale || patchGrid % scale != 0)
        {
            throw std::invalid_argument("Safety InternVL patch grid must be divisible by the pixel-shuffle scale.");
        }
        tokensPerTile *= patchGrid / scale;
    }
    return tokensPerTile;
}

} // namespace trt_edgellm::safety::builder
