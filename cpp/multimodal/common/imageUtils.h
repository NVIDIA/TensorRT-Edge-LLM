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

#include <algorithm>
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

/*!
 * @brief Compute the Qwen-family resize target with the 2D per-frame pixel budget (HF `smart_resize` parity,
 *        incl. Python-round()-compatible banker's rounding). Used by Qwen2-VL, Qwen2.5-VL, and Qwen3-Omni.
 * @return Tuple of (resizedHeight, resizedWidth); both are multiples of `patchSize * mergeSize`
 * @throws std::runtime_error if the aspect ratio exceeds `maxRatio`
 */
std::tuple<int64_t, int64_t> qwenSmartResize(int64_t height, int64_t width, int64_t patchSize, int64_t mergeSize,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t maxRatio = 200);

/*!
 * @brief Compute the Qwen3-VL / Qwen3.5 resize target with the 3D temporal-aware pixel budget: same as
 *        :func:`qwenSmartResize` except `tBar = ceilByFactor(numFrames, temporalPatchSize)` is folded into
 *        the budget. `isVideo` selects the modality explicitly (HF applies the temporal budget to any video;
 *        a 1-frame video still gets tBar = temporalPatchSize; only still images use tBar = 1), and the result
 *        is constrained to the engine token profile — infeasible inputs throw.
 * @return Tuple of (resizedHeight, resizedWidth)
 * @throws std::runtime_error if the aspect ratio exceeds `maxRatio`
 */
std::tuple<int64_t, int64_t> qwenSmartResize3D(int64_t numFrames, bool isVideo, int64_t height, int64_t width,
    int64_t patchSize, int64_t mergeSize, int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage,
    int64_t temporalPatchSize, int64_t maxRatio = 200);

//! Upper bound of visual segments (one per image or Qwen3-VL video temporal group). cu_seqlens holds
//! one more entry than this (the leading 0). Header-only: visual_build links this without edgellmCore.
inline int64_t maxCuSeqlenGroups(int64_t maxImageTokens)
{
    return std::max<int64_t>(1, maxImageTokens);
}

/*!
 * @brief Compute the Gemma4 resize target: rescale (up or down) so the pixel count fills the per-image
 *        patch budget while preserving aspect ratio; sides are floored to multiples of
 *        `poolingKernelSize * patchSize` with degenerate-side clamping.
 * @return Tuple of (resizedHeight, resizedWidth)
 * @throws std::runtime_error on non-positive input dims or an unsatisfiable budget
 */
std::tuple<int64_t, int64_t> gemma4ResizeTarget(
    int64_t height, int64_t width, int64_t maxImageTokensPerImage, int64_t poolingKernelSize, int64_t patchSize);

/*!
 * @brief Compute the Gemma4 Unified resize target: rescale so the pixel count fills the per-image patch
 *        budget while preserving aspect ratio; sides are floored to multiples of `modelPatchSize` and
 *        clamped to `min(maxPatchesPerImage, positionEmbeddingSize) * modelPatchSize`.
 * @return Tuple of (resizedHeight, resizedWidth)
 * @throws std::runtime_error on non-positive input dims or an unsatisfiable budget
 */
std::tuple<int64_t, int64_t> gemma4UnifiedResizeTarget(
    int64_t height, int64_t width, int64_t maxPatchesPerImage, int64_t modelPatchSize, int64_t positionEmbeddingSize);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
