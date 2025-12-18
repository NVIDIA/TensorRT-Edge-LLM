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

#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

//! Get all supported aspect ratios (width_ratio, height_ratio) bounded by tile counts
std::vector<std::pair<int64_t, int64_t>> getAllSupportedAspectRatios(int64_t minImageTiles, int64_t maxImageTiles);

//! Compute resized image size (height, width) based on token/tile constraints
std::tuple<int64_t, int64_t> computeBestBlockGridForResize(int64_t height, int64_t width,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t blockImageSizeH, int64_t blockImageSizeW);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
