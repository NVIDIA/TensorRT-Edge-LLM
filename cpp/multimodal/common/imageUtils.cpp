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

#include "multimodal/common/imageUtils.h"
#include "common/checkMacros.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

namespace
{

//! Banker's rounding (round-half-to-even) to match Python's round() used by the HF reference.
int64_t roundByFactor(int64_t value, int64_t factor)
{
    int64_t q = value / factor;
    int64_t r = value - q * factor;
    int64_t twoR = 2 * r;
    if (twoR > factor || (twoR == factor && (q & 1)))
        ++q;
    return q * factor;
}

int64_t floorByFactor(int64_t value, int64_t factor)
{
    return std::floor(static_cast<double>(value) / factor) * factor;
}

int64_t ceilByFactor(int64_t value, int64_t factor)
{
    return std::ceil(static_cast<double>(value) / factor) * factor;
}

// Takes double: the min-pixels branches pass the fractional product h*beta directly
// (HF: ceil(h*beta/f)*f — truncating to int first under-sizes ~3% of upscaled inputs).
int64_t ceilByFactorF(double value, int64_t factor)
{
    return static_cast<int64_t>(std::ceil(value / factor)) * factor;
}

} // namespace

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
    // -1 to reserve space for a potential thumbnail (always added for Phi4MM; skipped for single-block images in
    // InternVL)
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

std::tuple<int64_t, int64_t> qwenSmartResize(int64_t height, int64_t width, int64_t patchSize, int64_t mergeSize,
    int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage, int64_t maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    int64_t const factor = patchSize * mergeSize;
    int64_t const minPixels = minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = maxImageTokensPerImage * factor * factor;

    // Floating-point ratio: integer division truncates (2001/10 -> 200) and would accept ratios the
    // HF reference rejects.
    ELLM_CHECK(static_cast<double>(std::max(height, width)) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(static_cast<double>(std::max(height, width)) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        // Clamp to >= factor: a heavily-downscaled image must not yield a sub-factor (or zero) dimension.
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactorF(height * beta, factor);
        wBar = ceilByFactorF(width * beta, factor);
    }

    return {hBar, wBar};
}

std::tuple<int64_t, int64_t> qwenSmartResize3D(int64_t numFrames, bool isVideo, int64_t height, int64_t width,
    int64_t patchSize, int64_t mergeSize, int64_t minImageTokensPerImage, int64_t maxImageTokensPerImage,
    int64_t temporalPatchSize, int64_t maxRatio)
{
    // Mirrors HF Qwen3-VL smart_resize: 3D temporal-aware budget (t_bar = ceil(N/TPS)*TPS) — the base 2D body plus
    // numFrames counted into the budget — then constrains the result to the engine token profile (HF has no hard
    // cap; the engine does).
    int64_t const factor = patchSize * mergeSize;
    // The budget below is 3D (tBar*h*w) and tokens = budget / (temporalFactor * factor^2), so translating the
    // per-media token bounds into a pixel budget must include the temporal factor (1 for still images).
    int64_t const temporalFactor = isVideo ? temporalPatchSize : 1;
    int64_t const minPixels = minImageTokensPerImage * temporalFactor * factor * factor;
    int64_t const maxPixels = maxImageTokensPerImage * temporalFactor * factor * factor;

    // Floating-point ratio: integer division truncates (2001/10 -> 200) and would accept ratios the
    // HF reference rejects.
    ELLM_CHECK(static_cast<double>(std::max(height, width)) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(static_cast<double>(std::max(height, width)) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    // HF video smart_resize applies the temporal budget unconditionally (a 1-frame video still gets
    // t_bar = temporalPatchSize); only still images use t_bar = 1.
    int64_t const tBar = isVideo ? ceilByFactor(numFrames, temporalPatchSize) : 1;

    int64_t const budget = tBar * hBar * wBar;
    if (budget > maxPixels)
    {
        double const beta = std::sqrt(static_cast<double>(numFrames * height * width) / maxPixels);
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (budget < minPixels)
    {
        double const beta = std::sqrt(static_cast<double>(minPixels) / (numFrames * height * width));
        hBar = ceilByFactorF(height * beta, factor);
        wBar = ceilByFactorF(width * beta, factor);
    }
    // HF sizes by the raw frame count and has no hard cap; with a padded tBar > numFrames (odd frame counts)
    // either branch can exceed the engine's token profile, so re-shrink against tBar.
    bool engineForcedResize = false;
    if (tBar * hBar * wBar > maxPixels)
    {
        engineForcedResize = true;
        double const beta = std::sqrt(static_cast<double>(tBar * height * width) / maxPixels);
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    // Quantization (factor alignment, the max(factor, ...) clamps) can still land outside the profile: one
    // factor step moves the budget by whole multiples of tBar*factor^2, which can straddle the entire
    // [minPixels, maxPixels] window. Fall back to the factor-grid shape (hBar = p*factor, wBar = q*factor)
    // closest in aspect ratio that fits; ties prefer the budget closest to the violated bound.
    if (tBar * hBar * wBar > maxPixels || tBar * hBar * wBar < minPixels)
    {
        engineForcedResize = true;
        int64_t const unit = tBar * factor * factor;
        int64_t const pqMin = std::max<int64_t>(1, (minPixels + unit - 1) / unit);
        int64_t const pqMax = maxPixels / unit;
        ELLM_CHECK(pqMax >= 1 && pqMin <= pqMax,
            "no resized visual shape fits the engine profile (frames=" + std::to_string(numFrames) + ", "
                + std::to_string(height) + "x" + std::to_string(width) + "); reduce the frame count/resolution or "
                  "rebuild the visual engine with wider --minImageTokens/--maxImageTokens bounds.");
        int64_t const pqTarget = (tBar * hBar * wBar < minPixels) ? pqMin : pqMax;
        double const targetRatio = static_cast<double>(height) / static_cast<double>(width);
        double bestDist = 0.0;
        int64_t bestGap = 0;
        bool found = false;
        for (int64_t p = 1; p <= pqMax; ++p)
        {
            int64_t const qLo = std::max<int64_t>(1, (pqMin + p - 1) / p);
            int64_t const qHi = pqMax / p;
            if (qLo > qHi)
            {
                continue;
            }
            int64_t const q = std::clamp(static_cast<int64_t>(std::llround(p / targetRatio)), qLo, qHi);
            double const dist = std::abs(std::log(static_cast<double>(p) / static_cast<double>(q) / targetRatio));
            int64_t const gap = std::abs(p * q - pqTarget);
            if (!found || dist < bestDist - 1e-12 || (std::abs(dist - bestDist) <= 1e-12 && gap < bestGap))
            {
                found = true;
                bestDist = dist;
                bestGap = gap;
                hBar = p * factor;
                wBar = q * factor;
            }
        }
    }

    // The engine token profile is a hard cap HF's smart_resize lacks; when it forces a smaller/regridded
    // shape than HF would pick, the ViT sees different patches so output can drift from HF.
    LOG_WARNING_IF(engineForcedResize,
        "qwenSmartResize3D resized to %ldx%ld to fit the engine token profile (frames=%ld, src %ldx%ld); "
        "differs from HF smart_resize, output may drift. Rebuild the visual engine with wider "
        "--maxImageTokens to match HF.",
        hBar, wBar, numFrames, height, width);
    return {hBar, wBar};
}

std::tuple<int64_t, int64_t> gemma4ResizeTarget(
    int64_t height, int64_t width, int64_t maxImageTokensPerImage, int64_t poolingKernelSize, int64_t patchSize)
{
    ELLM_CHECK(height > 0 && width > 0, "Gemma4 image height/width must be positive");
    int64_t const maxPatches = maxImageTokensPerImage * poolingKernelSize * poolingKernelSize;
    double const totalPx = static_cast<double>(height) * static_cast<double>(width);
    double const targetPx = static_cast<double>(maxPatches) * patchSize * patchSize;
    double const factor = std::sqrt(targetPx / totalPx);
    double const idealHeight = factor * static_cast<double>(height);
    double const idealWidth = factor * static_cast<double>(width);
    int64_t const sideMult = poolingKernelSize * patchSize;
    auto floorBySideMult = [sideMult](double value) {
        return static_cast<int64_t>(std::floor(value / static_cast<double>(sideMult))) * sideMult;
    };
    auto roundBySideMult = [sideMult](double value) {
        return static_cast<int64_t>(std::round(value / static_cast<double>(sideMult))) * sideMult;
    };

    int64_t targetHeight = floorBySideMult(idealHeight);
    int64_t targetWidth = floorBySideMult(idealWidth);
    ELLM_CHECK(targetHeight != 0 || targetWidth != 0, "Gemma4 target image size rounded to 0x0");

    int64_t const maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult;
    if (targetHeight == 0)
    {
        targetHeight = sideMult;
        int64_t const maxWidth = std::min(maxSideLength, floorBySideMult(targetPx / targetHeight));
        targetWidth = std::clamp(roundBySideMult(idealWidth), sideMult, maxWidth);
    }
    else if (targetWidth == 0)
    {
        targetWidth = sideMult;
        int64_t const maxHeight = std::min(maxSideLength, floorBySideMult(targetPx / targetWidth));
        targetHeight = std::clamp(roundBySideMult(idealHeight), sideMult, maxHeight);
    }

    ELLM_CHECK(
        static_cast<double>(targetHeight) * targetWidth <= targetPx, "Gemma4 target image size exceeds patch budget");
    return {targetHeight, targetWidth};
}

std::tuple<int64_t, int64_t> gemma4UnifiedResizeTarget(
    int64_t height, int64_t width, int64_t maxPatchesPerImage, int64_t modelPatchSize, int64_t positionEmbeddingSize)
{
    ELLM_CHECK(height > 0 && width > 0, "Gemma4 Unified image dimensions must be positive");
    double const targetPixels = static_cast<double>(maxPatchesPerImage) * modelPatchSize * modelPatchSize;
    double const scale = std::sqrt(targetPixels / (static_cast<double>(height) * width));
    double const idealHeight = scale * height;
    double const idealWidth = scale * width;
    int64_t const sideMultiple = modelPatchSize;
    auto floorToMultiple = [sideMultiple](double value) {
        return static_cast<int64_t>(std::floor(value / sideMultiple)) * sideMultiple;
    };
    int64_t const maxSide = std::min(maxPatchesPerImage, positionEmbeddingSize) * sideMultiple;
    int64_t targetHeight = std::min(floorToMultiple(idealHeight), maxSide);
    int64_t targetWidth = std::min(floorToMultiple(idealWidth), maxSide);
    ELLM_CHECK(targetHeight != 0 || targetWidth != 0, "Gemma4 Unified resized image rounded to 0x0");
    if (targetHeight == 0)
    {
        targetHeight = sideMultiple;
        targetWidth
            = std::min(static_cast<int64_t>(std::floor(static_cast<double>(width) / height)) * sideMultiple, maxSide);
    }
    else if (targetWidth == 0)
    {
        targetWidth = sideMultiple;
        targetHeight
            = std::min(static_cast<int64_t>(std::floor(static_cast<double>(height) / width)) * sideMultiple, maxSide);
    }
    ELLM_CHECK(static_cast<double>(targetHeight) * targetWidth <= targetPixels,
        "Gemma4 Unified resized image exceeds per-image patch budget");
    return {targetHeight, targetWidth};
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
