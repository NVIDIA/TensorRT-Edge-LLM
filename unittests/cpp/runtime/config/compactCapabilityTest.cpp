/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/config/compactCapability.h"

#include "kernels/compactPhase/compactPhaseSupport.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

//! Every SM a deployment can report, plus SMs no kernel family knows.
std::vector<int32_t> const kSms{0, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121, 999};

//! Two attention layers of one shape; the second reads the first's cache when kvShared.
struct AttentionShape
{
    int32_t numQHeads;
    int32_t numKVHeads;
    int32_t headDim;
    bool fp8KVCache{false};
    int32_t tpSize{1};
    bool boundedCapable{false};
    bool kvShared{false};
    bool contextMaskSelector{false};
};

LLMEngineConfig engineOf(AttentionShape const& shape)
{
    LLMEngineConfig config{};
    config.numAttentionHeads = shape.numQHeads;
    config.tpSize = shape.tpSize;
    config.maxKVCacheCapacity = 4096;
    config.kvCacheDtype = shape.fp8KVCache ? trt_edgellm::DataType::kFP8 : trt_edgellm::DataType::kHALF;
    config.contextMaskSelectorEnabled = shape.contextMaskSelector;
    KVLayerConfig const layer{shape.numKVHeads, shape.headDim, shape.boundedCapable ? 1024 : 0};
    config.kvLayerConfigs = {layer, layer};
    if (shape.kvShared)
    {
        config.kvSharingDonors = {-1, 0};
    }
    return config;
}

//! Attention (16 query, 8 KV heads, head dim 128) beside a Mamba2 SSD (@p mamba2) or gated_delta_net layer of
//! @p headDim and @p stateSize, and a causal_conv1d of width @p convKernel.
LLMEngineConfig hybridOf(bool mamba2, int32_t headDim, int32_t stateSize, int32_t convKernel = 4)
{
    LLMEngineConfig config = engineOf({16, 8, 128});
    config.numLinearAttnLayers = 1;
    config.recurrentStateNumGroups = mamba2 ? 1 : 0;
    config.recurrentStateHeadDim = headDim;
    config.recurrentStateSize = stateSize;
    config.convDim = 64;
    config.convKernel = convKernel;
    return config;
}

LLMEngineConfig ngramOf(int32_t ngramSize)
{
    LLMEngineConfig config = engineOf({16, 8, 128});
    config.ngramLayers = {0};
    config.ngramConvKernel = 4;
    config.ngramConvDim = 256;
    config.ngramSize = ngramSize;
    return config;
}

bool refuses(LLMEngineConfig const& config, int32_t sm, CompactLayerKind kind)
{
    std::vector<CompactLayerGap> const gaps = compactLayerGaps(config, sm);
    return std::any_of(gaps.begin(), gaps.end(), [kind](CompactLayerGap const& gap) { return gap.kind == kind; });
}

//! The SMs whose causal_conv1d and gated_delta_net kernel families exist; the artifact carries some of them.
std::vector<int32_t> const kRecurrentKernelSms{80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121};

//! The Blackwell datacenter SM the linked CuTe DSL artifact carries (SM100 on x86, SM110 on Thor), or 0.
[[maybe_unused]] int32_t artifactDatacenterSm()
{
    for (int32_t const sm : {100, 101, 103, 110})
    {
        if (cuteDslArtifactCarriesSm(sm))
        {
            return sm;
        }
    }
    return 0;
}

//! "SM<a>/SM<b>/..." of the SMs among @p sms the linked CuTe DSL artifact carries.
[[maybe_unused]] std::string artifactSms(std::vector<int32_t> const& sms)
{
    std::string listed;
    for (int32_t const sm : sms)
    {
        if (cuteDslArtifactCarriesSm(sm))
        {
            listed += (listed.empty() ? "SM" : "/SM") + std::to_string(sm);
        }
    }
    return listed;
}

} // namespace

// Given engines with one phase-aware layer kind each over many shapes: attention (GQA ratios 1 to 16, head dims 64
//       to 512 and 80, FP8 KV, TP 2, bounded-SWA-capable, KV-shared, context mask selector), gated_delta_net and
//       Mamba2 SSD shapes inside and outside the packed kernels' sets, causal_conv1d widths 1 to 9, and the n-gram
//       conv at dilations 1 and 3
// When each is checked on every known SM and on SMs 0, 75 and 999
// Then the check refuses a kind on an SM exactly when that kind's plugin predicate refuses the shape there
TEST(CompactCapabilityTest, EachKindFollowsItsPluginPredicateOnEverySm)
{
    std::vector<AttentionShape> const attentionShapes{{16, 8, 128}, {32, 2, 128}, {24, 2, 64, true}, {8, 1, 512},
        {16, 8, 80}, {16, 8, 128, false, 2}, {16, 8, 256, false, 1, true}, {16, 8, 128, true, 1, false, true},
        {16, 8, 128, false, 1, false, false, true}};
    std::vector<std::pair<int32_t, int32_t>> const recurrentShapes{
        {128, 128}, {64, 64}, {128, 64}, {64, 128}, {80, 128}, {80, 64}};
    for (int32_t const sm : kSms)
    {
        SCOPED_TRACE("SM" + std::to_string(sm));
        for (AttentionShape const& shape : attentionShapes)
        {
            AttentionLayerTraits layer;
            layer.numQHeads = shape.numQHeads;
            layer.numKVHeads = shape.numKVHeads;
            layer.headSize = shape.headDim;
            layer.smVersion = sm;
            layer.tpSize = shape.tpSize;
            layer.fp8KVCache = shape.fp8KVCache;
            layer.slidingWindow = shape.boundedCapable;
            layer.supportsBoundedKVCache = shape.boundedCapable;
            layer.contextMaskSelector = shape.contextMaskSelector;
            AttentionLayerTraits shared = layer;
            shared.kvShared = shape.kvShared;
            bool const plugin = attentionCompactSupport(layer).supported && attentionCompactSupport(shared).supported;
            EXPECT_EQ(refuses(engineOf(shape), sm, CompactLayerKind::kAttention), !plugin)
                << "attention Hq " << shape.numQHeads << " Hkv " << shape.numKVHeads << " D " << shape.headDim;
        }
        for (auto const& [headDim, stateSize] : recurrentShapes)
        {
            EXPECT_EQ(refuses(hybridOf(false, headDim, stateSize), sm, CompactLayerKind::kGatedDeltaNet),
                !gatedDeltaNetCompactSupport(headDim, stateSize, sm, false, false).supported)
                << "GDN " << headDim << " " << stateSize;
            EXPECT_EQ(refuses(hybridOf(true, headDim, stateSize), sm, CompactLayerKind::kMamba2Ssd),
                !mamba2SsdCompactSupport(headDim, stateSize, sm, false).supported)
                << "SSD " << headDim << " " << stateSize;
        }
        for (int32_t width = 1; width <= 9; ++width)
        {
            EXPECT_EQ(refuses(hybridOf(false, 128, 128, width), sm, CompactLayerKind::kCausalConv1d),
                !causalConv1dCompactSupport(sm, 64, width, 1, 64, false).supported)
                << "conv width " << width;
        }
        for (int32_t const ngramSize : {1, 3})
        {
            EXPECT_EQ(refuses(ngramOf(ngramSize), sm, CompactLayerKind::kNgramConv),
                !causalConv1dCompactSupport(sm, 256, 4, ngramSize, 256, false).supported)
                << "n-gram dilation " << ngramSize;
        }
    }
}

// Given the K = V = 128 gated_delta_net and the width-4 causal_conv1d, every SM their kernel families cover, and the
//       linked CuTe DSL artifact's gpu_arch list as CMake passes it
// When each predicate is checked on each SM
// Then the artifact is taken to carry exactly the SMs of that list; each predicate refuses every SM the artifact
//      was not built for, naming the artifact where its kernel family covers the SM, and never names the artifact on
//      an SM the artifact carries
TEST(CompactCapabilityTest, RecurrentKindsRunCompactOnlyOnSmsTheArtifactCarries)
{
#if defined(CUTE_DSL_GDN_ENABLED) && defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED) && defined(CUTE_DSL_ARTIFACT_SMS)
    std::vector<int32_t> const builtFor{CUTE_DSL_ARTIFACT_SMS};
    std::string const artifactGap = "the linked CuTe DSL artifact has no kernels for this SM";
    int32_t artifactRefusals = 0;
    for (int32_t const sm : kRecurrentKernelSms)
    {
        SCOPED_TRACE("SM" + std::to_string(sm));
        bool const carried = std::find(builtFor.begin(), builtFor.end(), sm) != builtFor.end();
        EXPECT_EQ(cuteDslArtifactCarriesSm(sm), carried);
        for (CompactPhaseSupport const support : {gatedDeltaNetCompactSupport(128, 128, sm, false, false),
                 causalConv1dCompactSupport(sm, 64, 4, 1, 64, false)})
        {
            bool const namesArtifact = support.reason != nullptr && support.reason == artifactGap;
            EXPECT_FALSE(support.supported && !carried);
            EXPECT_FALSE(carried && namesArtifact);
            artifactRefusals += namesArtifact ? 1 : 0;
        }
    }
    // SM80 to SM90 are in both kernel families, so an artifact without one of them is refused by name.
    bool const carriesSm8x = std::all_of(kRecurrentKernelSms.begin(), kRecurrentKernelSms.begin() + 5,
        [&builtFor](int32_t sm) { return std::find(builtFor.begin(), builtFor.end(), sm) != builtFor.end(); });
    if (!carriesSm8x)
    {
        EXPECT_GT(artifactRefusals, 0);
    }
#else
    GTEST_SKIP() << "this build lacks the CuTe DSL GDN or causal_conv1d kernels";
#endif
}

// Given an attention-only engine (16 query heads, 8 KV heads, head dim 128), a gated_delta_net hybrid and a Mamba2
//       hybrid (K = V = 128 GDN or head dim 64 / state size 128 SSD, each with a width-4 causal_conv1d), a
//       context-mask-selector engine, an n-gram engine, and Qwen4-Exp-like engines whose every attention layer is QSA
//       (16 query and 2 KV heads, head dim 256, indexer 128 or 256)
// When each is checked on the Blackwell datacenter SM the artifact carries (SM100 on x86, SM110 on Thor), SM80 to SM90
//      (SM87 is Orin), SM120 and on SMs 0, 75 and 999
// Then the QSA engines are refused on every SM, QSA being the only gap on that datacenter SM in a build with its GDN
//      and causal_conv1d kernels, even where a plain attention layer of the pool row's head dim would run; and, in a
//      build with every CuTe DSL kernel group, that datacenter SM runs the first three; SM80, SM86, SM87, SM89 and SM90
//      run the GDN hybrid exactly when the linked artifact carries them; SM87 runs attention and refuses the Mamba2
//      hybrid for SSD, naming its shape, why, and supported SMs; SM120 runs attention, causal_conv1d when the artifact
//      carries SM120, GDN only when the build also carries the SM120/121 GDN kernels, and SSD only when it carries the
//      SM120/121 SSD kernels; the unknown SMs refuse every kind, SM75 naming the causal_conv1d width, why, and the
//      supported SMs, which are the ones the artifact carries; and the mask-selector and n-gram engines are refused
//      on SM100
TEST(CompactCapabilityTest, KnownDeploymentsMatchTheKernelMatrix)
{
    for (int32_t const indexerHeadDim : {128, 256})
    {
        LLMEngineConfig qsa = hybridOf(false, 128, 128);
        qsa.kvLayerConfigs = engineOf({16, 2, 256 + indexerHeadDim}).kvLayerConfigs;
        qsa.indexerHeadDim = indexerHeadDim;
        for (int32_t const sm : kSms)
        {
            EXPECT_NE(compactExecutionGap(qsa, sm).find("QSA attention layers"), std::string::npos)
                << "SM" << sm << ": " << compactExecutionGap(qsa, sm);
        }
#if defined(CUTE_DSL_GDN_BLACKWELL_ENABLED) && defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
        int32_t const datacenterSm = artifactDatacenterSm();
        ASSERT_NE(datacenterSm, 0) << "the artifact has the Blackwell GDN group but no Blackwell datacenter SM";
        std::vector<CompactLayerGap> const onDatacenter = compactLayerGaps(qsa, datacenterSm);
        ASSERT_EQ(onDatacenter.size(), 1U) << compactExecutionGap(qsa, datacenterSm);
        EXPECT_EQ(onDatacenter.front().kind, CompactLayerKind::kQsaAttention);
#endif
    }
#if defined(CUTE_DSL_FMHA_ENABLED) && defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)                                         \
    && defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED) && defined(CUTE_DSL_GDN_BLACKWELL_ENABLED)                            \
    && defined(CUTE_DSL_SSD_BLACKWELL_ENABLED) && defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
    LLMEngineConfig const attention = engineOf({16, 8, 128});
    LLMEngineConfig const gdn = hybridOf(false, 128, 128);
    LLMEngineConfig const mamba2 = hybridOf(true, 64, 128);
    int32_t const datacenterSm = artifactDatacenterSm();
    for (LLMEngineConfig const& config : {attention, gdn, mamba2})
    {
        EXPECT_EQ(compactExecutionGap(config, datacenterSm), "") << "SM" << datacenterSm;
    }
    for (int32_t const sm : {80, 86, 87, 89, 90})
    {
        EXPECT_EQ(compactExecutionGap(gdn, sm).empty(), cuteDslArtifactCarriesSm(sm))
            << "SM" << sm << ": " << compactExecutionGap(gdn, sm);
    }
    EXPECT_EQ(compactExecutionGap(attention, 87), "");
#ifdef CUTE_DSL_SSD_SM12X_ENABLED
    std::string const ssdSms = "SM100/SM101/SM103/SM110/SM120/SM121";
#else
    std::string const ssdSms = "SM100/SM101/SM103/SM110";
#endif
    // The SSD gap comes first; an artifact without SM87 adds the causal_conv1d gap after it.
    std::string const ssdOnSm87
        = "no compact-execution kernels on SM87 for Mamba2 SSD layers (head dim 64, state size 128: no packed SSD "
          "prefill kernel for this SM, head dim and state size in this build; supported on "
        + ssdSms + ")";
    EXPECT_EQ(compactExecutionGap(mamba2, 87).substr(0, ssdOnSm87.size()), ssdOnSm87);
    EXPECT_EQ(refuses(mamba2, 87, CompactLayerKind::kCausalConv1d), !cuteDslArtifactCarriesSm(87));
    EXPECT_EQ(compactExecutionGap(attention, 120), "");
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    EXPECT_EQ(compactExecutionGap(gdn, 120), "");
#else
    EXPECT_TRUE(refuses(gdn, 120, CompactLayerKind::kGatedDeltaNet));
    EXPECT_EQ(refuses(gdn, 120, CompactLayerKind::kCausalConv1d), !cuteDslArtifactCarriesSm(120));
#endif
#ifdef CUTE_DSL_SSD_SM12X_ENABLED
    EXPECT_FALSE(refuses(mamba2, 120, CompactLayerKind::kMamba2Ssd));
#else
    EXPECT_TRUE(refuses(mamba2, 120, CompactLayerKind::kMamba2Ssd));
#endif
    for (int32_t const sm : {0, 75, 999})
    {
        EXPECT_TRUE(refuses(attention, sm, CompactLayerKind::kAttention)) << "SM" << sm;
        EXPECT_TRUE(refuses(gdn, sm, CompactLayerKind::kGatedDeltaNet)) << "SM" << sm;
        EXPECT_TRUE(refuses(gdn, sm, CompactLayerKind::kCausalConv1d)) << "SM" << sm;
        EXPECT_TRUE(refuses(mamba2, sm, CompactLayerKind::kMamba2Ssd)) << "SM" << sm;
    }
    std::vector<CompactLayerGap> const gdnOnSm75 = compactLayerGaps(gdn, 75);
    auto const convOnSm75 = std::find_if(gdnOnSm75.begin(), gdnOnSm75.end(),
        [](CompactLayerGap const& gap) { return gap.kind == CompactLayerKind::kCausalConv1d; });
    ASSERT_NE(convOnSm75, gdnOnSm75.end());
    EXPECT_EQ(convOnSm75->reason,
        "width 4: no packed-varlen causal_conv1d kernel for this SM, width and dilation in this build; supported on "
            + artifactSms(kRecurrentKernelSms));
    LLMEngineConfig maskSelector = attention;
    maskSelector.contextMaskSelectorEnabled = true;
    EXPECT_EQ(compactExecutionGap(maskSelector, 100),
        "no compact-execution kernels on SM100 for attention layers (layer 0: 16 query heads, 8 KV heads, head dim "
        "128: no compact path for the context mask selector)");
    EXPECT_TRUE(refuses(ngramOf(3), 100, CompactLayerKind::kNgramConv));
#else
    GTEST_SKIP() << "this build lacks a CuTe DSL kernel group the matrix covers";
#endif
}

// Given a gated_delta_net hybrid (attention with 16 query and 8 KV heads at head dim 128, a K = V = 128 GDN layer and a
//       width-4 causal_conv1d)
// When it is checked on SM120 and SM121 in a build with the SM120/121 GDN, FMHA and causal_conv1d kernels
// Then every layer kind runs compact on each of the two SMs the linked artifact carries, and the hybrid is refused on
//      the other
TEST(CompactCapabilityTest, GdnHybridRunsCompactOnBlackwellGeforce)
{
#if defined(CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED) && defined(CUTE_DSL_FMHA_ENABLED)                                  \
    && defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
    LLMEngineConfig const gdn = hybridOf(false, 128, 128);
    for (int32_t const sm : {120, 121})
    {
        EXPECT_EQ(compactExecutionGap(gdn, sm).empty(), cuteDslArtifactCarriesSm(sm))
            << "SM" << sm << ": " << compactExecutionGap(gdn, sm);
    }
#else
    GTEST_SKIP() << "this build lacks the SM120/121 GDN, FMHA or causal_conv1d kernels";
#endif
}

// Given a build whose CuTe DSL artifact defines the CTM decode macro (a multi-SM artifact such as sm_110_sm_120 does so
//      for its SM110 kernels),
// When CTM decode is asked for on SMs with and without CTM kernels,
// Then it is selected only on the CTM-only SMs, so decode on SM120 falls back to XQA instead of aborting in the
//      artifact's SM dispatcher.
TEST(CompactCapabilityTest, CtmDecodeIsSelectedOnlyOnSmsWithCtmKernels)
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    AttentionLayerTraits layer;
    layer.numQHeads = 16;
    layer.numKVHeads = 8;
    layer.headSize = 128;
    layer.tpSize = 1;
    for (int32_t const sm : {100, 101, 103, 110})
    {
        layer.smVersion = sm;
        EXPECT_TRUE(canUseCtmDecode(layer)) << "SM" << sm;
    }
    for (int32_t const sm : {80, 86, 87, 89, 90, 120, 121})
    {
        layer.smVersion = sm;
        EXPECT_FALSE(canUseCtmDecode(layer)) << "SM" << sm;
    }
#else
    GTEST_SKIP() << "this build lacks the CuTe DSL CTM decode kernels";
#endif
}
