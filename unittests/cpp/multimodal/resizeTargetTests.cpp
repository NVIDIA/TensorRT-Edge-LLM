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

//! Golden tests for the stateless resize-target functions in multimodal/imageUtils.h.
//! Regenerate the tables with unittests/resources/gen_resize_target_golden.py (see its docstring for
//! which values are HF-reference goldens and which are regression pins).

#include "multimodal/common/imageUtils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace iu = trt_edgellm::rt::imageUtils;

namespace
{

struct Golden2D
{
    int64_t h;
    int64_t w;
    int64_t hBar;
    int64_t wBar;
};

struct Golden3D
{
    int64_t t;
    int64_t h;
    int64_t w;
    int64_t hBar;
    int64_t wBar;
};

//! Qwen2/2.5-VL-style config: patchSize=14, mergeSize=2 (factor 28), builder defaults
//! minImageTokens=4, maxImageTokensPerImage=512.
constexpr int64_t k2DPatch = 14;
constexpr int64_t k2DMerge = 2;
constexpr int64_t k2DMinTok = 4;
constexpr int64_t k2DMaxTok = 512;

//! Qwen3-VL-style config: patchSize=16, mergeSize=2 (factor 32), temporalPatchSize=2,
//! minImageTokens=4, maxImageTokensPerImage=6144 (HF default).
constexpr int64_t k3DPatch = 16;
constexpr int64_t k3DMerge = 2;
constexpr int64_t k3DMinTok = 4;
constexpr int64_t k3DMaxTok = 6144;
constexpr int64_t k3DTemporal = 2;

std::tuple<int64_t, int64_t> smartResize2D(int64_t h, int64_t w)
{
    return iu::qwenSmartResize(h, w, k2DPatch, k2DMerge, k2DMinTok, k2DMaxTok);
}

std::tuple<int64_t, int64_t> smartResize3D(int64_t t, int64_t h, int64_t w)
{
    return iu::qwenSmartResize3D(t, /*isVideo=*/true, h, w, k3DPatch, k3DMerge, k3DMinTok, k3DMaxTok, k3DTemporal);
}

} // namespace

//! Goldens from HF transformers qwen2_vl image_processing smart_resize (see generator script).
//! Covers: up-scale (< minPixels), round-only, down-scale (> maxPixels), banker's-rounding ties
//! (42 -> 56 rounds q=1.5 up to even 2; 70 -> 56 rounds q=2.5 down to even 2), extreme aspect
//! ratio exactly at the 200 limit (14 x 2800), and square/non-square shapes.
TEST(QwenSmartResizeTest, MatchesHFReferenceGoldens)
{
    Golden2D const goldens[] = {
        {64, 64, 56, 56},
        {42, 42, 56, 56},
        {70, 70, 56, 56},
        {224, 224, 224, 224},
        {448, 644, 448, 644},
        {1080, 1920, 448, 840},
        {360, 480, 364, 476},
        {28, 5600, 28, 5600},
        {812, 1092, 532, 728},
        {1024, 1024, 616, 616},
        {700, 700, 616, 616},
        {56, 84, 56, 84},
    };
    for (auto const& g : goldens)
    {
        auto const [hBar, wBar] = smartResize2D(g.h, g.w);
        EXPECT_EQ(hBar, g.hBar) << "input " << g.h << "x" << g.w;
        EXPECT_EQ(wBar, g.wBar) << "input " << g.h << "x" << g.w;
    }
}

//! The target is a fixed point: feeding a computed target back returns it unchanged. This is what
//! makes the "query the target size -> pre-resize -> submit with do_resize=false" workflow coherent,
//! and it must hold for every already-conforming size.
TEST(QwenSmartResizeTest, TargetIsIdempotent)
{
    Golden2D const inputs[] = {
        {64, 64, 0, 0},
        {1080, 1920, 0, 0},
        {360, 480, 0, 0},
        {1024, 1024, 0, 0},
        {14, 2800, 0, 0},
    };
    for (auto const& g : inputs)
    {
        auto const [hBar, wBar] = smartResize2D(g.h, g.w);
        auto const [hBar2, wBar2] = smartResize2D(hBar, wBar);
        EXPECT_EQ(hBar2, hBar) << "not a fixed point for input " << g.h << "x" << g.w;
        EXPECT_EQ(wBar2, wBar) << "not a fixed point for input " << g.h << "x" << g.w;
    }
}

//! KNOWN DIVERGENCE from the HF reference, pinned deliberately: for a dimension below factor/2
//! (e.g. a 14px-tall strip with factor 28), HF's round() yields 0 and falls into the min-pixels
//! rescale branch (HF returns 28x812 here), while this implementation clamps the rounded dim to
//! >= factor first and therefore skips the rescale. Pre-existing behavior retained by the
//! extraction; revisit separately if HF parity for sub-factor inputs is ever required.
TEST(QwenSmartResizeTest, SubFactorDimDivergesFromHFByDesign)
{
    auto const [hBar, wBar] = smartResize2D(14, 2800);
    EXPECT_EQ(hBar, 28);
    EXPECT_EQ(wBar, 2800);
}

TEST(QwenSmartResizeTest, ThrowsOnExcessiveAspectRatio)
{
    EXPECT_THROW(smartResize2D(10, 2810), std::runtime_error);
    EXPECT_THROW(smartResize2D(2810, 10), std::runtime_error);
}

//! Goldens from HF transformers qwen3_vl video_processing smart_resize with the video pixel bounds
//! (HF's video defaults carry the temporalPatchSize factor: max_pixels = 16*16*2*2*2*6144, i.e.
//! maxTokens * temporalPatchSize * factor^2 — tokens = budget / (temporalPatchSize * factor^2)).
//! Covers: t exactly on/off temporalPatchSize multiples (tBar rounding), the beta-from-raw-frames
//! asymmetry (beta uses numFrames*h*w, not tBar), up/down-scale regimes, and 1-frame videos
//! (HF applies the temporal budget to any video: t=1 still gets tBar = temporalPatchSize).
TEST(Qwen3VLSmartResize3DTest, MatchesHFReferenceGoldens)
{
    Golden3D const goldens[] = {
        {1, 64, 64, 64, 64},
        {2, 224, 224, 224, 224},
        {3, 224, 224, 224, 224},
        {4, 224, 224, 224, 224},
        {8, 512, 512, 512, 512},
        {16, 720, 1280, 640, 1152},
        {3, 1080, 1920, 1088, 1920},
        {2, 480, 360, 480, 352},
        {5, 256, 256, 256, 256},
        {1, 1024, 1024, 1024, 1024},
        {32, 224, 224, 224, 224},
        {2, 800, 600, 800, 608},
        {7, 96, 96, 96, 96},
        {2, 2048, 2048, 2048, 2048},
    };
    for (auto const& g : goldens)
    {
        auto const [hBar, wBar] = smartResize3D(g.t, g.h, g.w);
        EXPECT_EQ(hBar, g.hBar) << "input t=" << g.t << " " << g.h << "x" << g.w;
        EXPECT_EQ(wBar, g.wBar) << "input t=" << g.t << " " << g.h << "x" << g.w;
    }
}

//! The 2D and 3D budgets diverge for multi-frame inputs — this distinction is why Qwen3-Omni
//! delegates to the 2D formula while Qwen3-VL uses the 3D one, and it must not be "unified" away.
TEST(Qwen3VLSmartResize3DTest, DivergesFrom2DForVideo)
{
    auto const [h2d, w2d] = iu::qwenSmartResize(720, 1280, k3DPatch, k3DMerge, k3DMinTok, k3DMaxTok);
    auto const [h3d, w3d] = smartResize3D(16, 720, 1280);
    EXPECT_EQ(h2d, 704);
    EXPECT_EQ(w2d, 1280);
    EXPECT_EQ(h3d, 640);
    EXPECT_EQ(w3d, 1152);
}

//! Regression pins for the Gemma4 budget-fill formula (values pinned from the pre-extraction
//! implementation, not an HF reference). Config: maxImageTokensPerImage=256, poolingKernelSize=4,
//! patchSize=14 (sideMult 56). Covers up-scale, down-scale, exact-budget, and the degenerate
//! thin-side clamp paths (100x6000 / 6000x100).
TEST(Gemma4ResizeTargetTest, MatchesPinnedGoldens)
{
    constexpr int64_t kMaxTok = 256;
    constexpr int64_t kPool = 4;
    constexpr int64_t kPatch = 14;
    Golden2D const goldens[] = {
        {224, 224, 896, 896},
        {448, 644, 728, 1064},
        {1080, 1920, 672, 1176},
        {64, 64, 896, 896},
        {100, 6000, 112, 6888},
        {6000, 100, 6888, 112},
        {896, 896, 896, 896},
        {500, 700, 728, 1008},
    };
    constexpr int64_t kSideMult = kPool * kPatch;
    constexpr int64_t kTargetPx = kMaxTok * kPool * kPool * kPatch * kPatch;
    for (auto const& g : goldens)
    {
        auto const [tH, tW] = iu::gemma4ResizeTarget(g.h, g.w, kMaxTok, kPool, kPatch);
        EXPECT_EQ(tH, g.hBar) << "input " << g.h << "x" << g.w;
        EXPECT_EQ(tW, g.wBar) << "input " << g.h << "x" << g.w;
        EXPECT_EQ(tH % kSideMult, 0);
        EXPECT_EQ(tW % kSideMult, 0);
        EXPECT_LE(tH * tW, kTargetPx);
    }
}

TEST(Gemma4ResizeTargetTest, ThrowsOnNonPositiveDims)
{
    EXPECT_THROW(iu::gemma4ResizeTarget(0, 100, 256, 4, 14), std::runtime_error);
    EXPECT_THROW(iu::gemma4ResizeTarget(100, 0, 256, 4, 14), std::runtime_error);
}

//! Regression pins for the Gemma4 Unified budget-fill formula (values pinned from the pre-extraction
//! implementation, not an HF reference). Config: maxPatchesPerImage=256, modelPatchSize=48,
//! positionEmbeddingSize=64. Covers up-scale, down-scale, exact-budget, the maxSide clamp
//! (100x6000 / 48x4800), and side-multiple flooring.
TEST(Gemma4UnifiedResizeTargetTest, MatchesPinnedGoldens)
{
    constexpr int64_t kMaxPatches = 256;
    constexpr int64_t kPatch = 48;
    constexpr int64_t kPosEmb = 64;
    Golden2D const goldens[] = {
        {224, 224, 768, 768},
        {448, 644, 624, 912},
        {1080, 1920, 576, 1008},
        {64, 64, 768, 768},
        {100, 6000, 96, 3072},
        {768, 768, 768, 768},
        {500, 700, 624, 864},
        {48, 4800, 48, 3072},
    };
    constexpr int64_t kTargetPx = kMaxPatches * kPatch * kPatch;
    for (auto const& g : goldens)
    {
        auto const [tH, tW] = iu::gemma4UnifiedResizeTarget(g.h, g.w, kMaxPatches, kPatch, kPosEmb);
        EXPECT_EQ(tH, g.hBar) << "input " << g.h << "x" << g.w;
        EXPECT_EQ(tW, g.wBar) << "input " << g.h << "x" << g.w;
        EXPECT_EQ(tH % kPatch, 0);
        EXPECT_EQ(tW % kPatch, 0);
        EXPECT_LE(tH * tW, kTargetPx);
    }
}

TEST(Gemma4UnifiedResizeTargetTest, ThrowsOnNonPositiveDims)
{
    EXPECT_THROW(iu::gemma4UnifiedResizeTarget(0, 100, 256, 48, 64), std::runtime_error);
    EXPECT_THROW(iu::gemma4UnifiedResizeTarget(100, 0, 256, 48, 64), std::runtime_error);
}

// --- Qwen3-VL profile-window regressions (narrow [min, max] engine windows the
// wide-profile HF goldens above cannot exercise) ---

namespace qwen_profile_window
{

// patchSize=16, mergeSize=2 -> factor=32; temporalPatchSize=2 (Qwen3-VL defaults).
std::tuple<int64_t, int64_t> resize3d(
    int64_t numFrames, bool isVideo, int64_t h, int64_t w, int64_t minTokens, int64_t maxTokens)
{
    return iu::qwenSmartResize3D(numFrames, isVideo, h, w, 16, 2, minTokens, maxTokens, 2);
}

// Visual tokens produced by a resized video: tBar * h * w / (temporalPatchSize * factor^2).
int64_t videoTokens(int64_t numFrames, int64_t h, int64_t w)
{
    int64_t const tBar = (numFrames + 1) / 2 * 2;
    return tBar * h * w / (2 * 32 * 32);
}

} // namespace qwen_profile_window

TEST(QwenSmartResize, WindowStraddlingQuantizationStaysInsideProfile)
{
    // The HF shrink quantizes in whole factor steps; with 146 frames one step moves the budget by more than the
    // whole [min, max] window, so a naive floor lands below the minimum and must fall back to a feasible shape.
    auto const [h, w] = qwen_profile_window::resize3d(146, true, 360, 640, 256, 512);
    int64_t const tokens = qwen_profile_window::videoTokens(146, h, w);
    EXPECT_GE(tokens, 256);
    EXPECT_LE(tokens, 512);
}

TEST(QwenSmartResize, OddFrameVideoStaysInsideProfile)
{
    // Odd frame counts pad tBar above numFrames, so the HF shrink (beta from raw frames) under-shrinks and the
    // tBar re-shrink must bring the result back under the cap.
    auto const [h, w] = qwen_profile_window::resize3d(145, true, 360, 640, 256, 512);
    int64_t const tokens = qwen_profile_window::videoTokens(145, h, w);
    EXPECT_GE(tokens, 256);
    EXPECT_LE(tokens, 512);

    auto const [h1, w1] = qwen_profile_window::resize3d(1, true, 1024, 1024, 4, 512);
    EXPECT_LE(qwen_profile_window::videoTokens(1, h1, w1), 512);
}

TEST(QwenSmartResize, SingleFrameVideoUsesTemporalBudget)
{
    // HF's video path applies t_bar = ceil(1/2)*2 = 2 even for a single frame; the image path uses t_bar = 1.
    // The two paths must diverge for the same pixel input.
    auto const video = qwen_profile_window::resize3d(1, true, 32, 96, 256, 512);
    auto const image = qwen_profile_window::resize3d(1, false, 32, 96, 256, 512);
    EXPECT_NE(video, image);
    int64_t const tokens = qwen_profile_window::videoTokens(1, std::get<0>(video), std::get<1>(video));
    EXPECT_GE(tokens, 256);
    EXPECT_LE(tokens, 512);
}

TEST(QwenSmartResize, ExtremeWideVideoStaysUnderMaxProfile)
{
    // The max(factor, ...) clamp on the short side pushes the budget back above the cap for extreme aspect
    // ratios; the result must still respect the hard profile maximum.
    auto const [h, w] = qwen_profile_window::resize3d(64, true, 100, 2000, 4, 512);
    int64_t const tokens = qwen_profile_window::videoTokens(64, h, w);
    EXPECT_GE(tokens, 4);
    EXPECT_LE(tokens, 512);
}

TEST(QwenSmartResize, MinEqualsMaxFindsExactBudget)
{
    // A zero-width profile window is satisfiable only by an exact factor-grid product; the fallback must find it
    // rather than reject the input.
    auto const [h, w] = qwen_profile_window::resize3d(1, true, 1024, 1024, 512, 512);
    EXPECT_EQ(qwen_profile_window::videoTokens(1, h, w), 512);
}

TEST(QwenSmartResize, FractionalAspectRatioRejected)
{
    // 2001:10 = 200.1 exceeds the 200 limit; integer division would truncate
    // it to 200 and wrongly accept the input (HF uses float division).
    EXPECT_THROW(qwen_profile_window::resize3d(2, true, 10, 2001, 4, 6144), std::runtime_error);
    EXPECT_THROW(iu::qwenSmartResize(10, 2001, 14, 2, 4, 512), std::runtime_error);
}

TEST(QwenSmartResize, NoFeasibleShapeThrows)
{
    // 4096 frames need more tokens than the profile maximum even at the smallest spatial shape.
    EXPECT_THROW(qwen_profile_window::resize3d(4096, true, 32, 32, 4, 512), std::runtime_error);
}

TEST(QwenSmartResize, CuSeqlenBoundCoversTemporalGroups)
{
    // 66 frames -> 33 temporal groups; the profile must accommodate one entry per group even when the per-image
    // token budget alone would suggest far fewer entries.
    int64_t const groups = (66 + 1) / 2;
    EXPECT_LE(groups, iu::maxCuSeqlenGroups(512));
    EXPECT_GE(iu::maxCuSeqlenGroups(1), 1);
}
