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

// collectRopeConfig() turns a HuggingFace `config.json` into the rope variant and
// parameters the runtime builds its cos/sin cache from. A misread here is not a
// crash: the engine builds, inference runs, and only the output degrades, which
// makes it invisible to end-to-end tests that assert on pipeline success. These
// tests pin the dispatch, the precedence between competing keys, and the
// parameter arithmetic.

#include "runtime/llmRuntimeUtils.h"

#include "common/tensor.h"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <functional>
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Defaults declared by RopeConfig. Restated here so a test that asserts "the
//! default survived" fails when the declared default changes, rather than
//! silently tracking it.
constexpr float kDefaultRotaryTheta{100000.0F};
constexpr int32_t kDefaultMaxPositionEmbeddings{32768};
constexpr float kDefaultRotaryScale{1.0F};

Json withScaling(Json scaling)
{
    Json config;
    config["rope_scaling"] = std::move(scaling);
    return config;
}

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

// A hybrid export states `use_rope: false` for a model that has no positional
// encoding at all. The flag has to outrank rope_scaling, which such a config may
// still carry from the upstream checkpoint: applying rotary positions to a NoPE
// model corrupts every attention score.
//
// Given a config that says use_rope is false but still carries a rope_scaling section
// When the rope config is collected
// Then the model is NoPE and no scaling parameters are collected
TEST(RopeConfigTest, ExplicitUseRopeFalseOutranksRopeScaling)
{
    Json config = withScaling({{"rope_type", "longrope"}});
    config["use_rope"] = false;
    config["max_position_embeddings"] = 4096;

    auto const rope = collectRopeConfig(config);

    EXPECT_EQ(rope.type, RopeType::kNoRope);
    // The cache is still sized from the config even though no angle is applied.
    EXPECT_EQ(rope.maxPositionEmbeddings, 4096);
    // longrope parameters must not have been collected: kNoRope returns early,
    // and a populated longRope would mean the early return was skipped.
    EXPECT_FALSE(rope.longRope.has_value());
}

// `use_rope: true` is not an override; it means "rope applies", so dispatch
// continues into rope_scaling as usual.
TEST(RopeConfigTest, ExplicitUseRopeTrueStillDispatchesOnRopeScaling)
{
    Json config = withScaling({{"rope_type", "proportional"}});
    config["use_rope"] = true;

    EXPECT_EQ(collectRopeConfig(config).type, RopeType::kProportional);
}

TEST(RopeConfigTest, NopeRopeTypeUsesIdentityCache)
{
    EXPECT_EQ(collectRopeConfig(withScaling({{"rope_type", "nope"}})).type, RopeType::kNoRope);
}

// A checkpoint with no rope_scaling section is the common plain-Llama shape.
TEST(RopeConfigTest, AbsentRopeScalingFallsBackToDefaultType)
{
    Json config;
    config["rope_theta"] = 500000.0F;

    auto const rope = collectRopeConfig(config);

    EXPECT_EQ(rope.type, RopeType::kDefault);
    EXPECT_FLOAT_EQ(rope.rotaryTheta, 500000.0F);
}

TEST(RopeConfigTest, LongPositionPhaseIsAccurateThroughHostAndDeviceDispatch)
{
    constexpr int64_t kMaxPositions{32768};
    for (auto const type : {RopeType::kDefault, RopeType::kDynamic, RopeType::kProportional})
    {
        for (int64_t const rotaryDim : {64, 96, 128, 256, 512})
        {
            SCOPED_TRACE("rotary dimension " + std::to_string(rotaryDim));
            RopeConfig rope;
            rope.type = type;
            rope.maxPositionEmbeddings = kMaxPositions;
            rope.partialRotaryFactor = type == RopeType::kProportional ? 0.5F : 1.0F;
            Tensor cache({1, kMaxPositions, rotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
            ASSERT_TRUE(initializeRopeCosSinCache(cache, rope, nullptr));
            std::vector<float> host(static_cast<size_t>(cache.getShape().volume()));
            ASSERT_EQ(cudaMemcpy(host.data(), cache.rawPointer(), host.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
            // The first frequency is exactly one, isolating trig phase error from pow rounding.
            for (int64_t const position : {4096, 16383, 32767})
            {
                size_t const offset = static_cast<size_t>(position * rotaryDim);
                EXPECT_NEAR(host[offset], std::cos(static_cast<double>(position)), 1e-6);
                EXPECT_NEAR(host[offset + rotaryDim / 2], std::sin(static_cast<double>(position)), 1e-6);
                if (type == RopeType::kProportional)
                {
                    EXPECT_FLOAT_EQ(host[offset + rotaryDim / 2 - 1], 1.0F);
                    EXPECT_FLOAT_EQ(host[offset + rotaryDim - 1], 0.0F);
                }
            }
        }
    }
}

// Qwen2-VL-era checkpoints spell MRoPE as `type: "mrope"`. Qwen2.5-VL-era ones
// spell the same thing as `rope_type: "default"` plus an `mrope_section` array.
// Both must reach kMRope, or a VLM silently decodes with 1-D text positions.
//
// Given the two spellings of MRoPE that different checkpoint generations use
// When each is collected
// Then both reach kMRope
TEST(RopeConfigTest, MRopeRecognizedUnderBothCheckpointConventions)
{
    EXPECT_EQ(collectRopeConfig(withScaling({{"type", "mrope"}})).type, RopeType::kMRope);

    Json const modern = withScaling({{"rope_type", "default"}, {"mrope_section", Json::array({16, 24, 24})}});
    EXPECT_EQ(collectRopeConfig(modern).type, RopeType::kMRope);
}

// The negative half of the case above: without mrope_section, "default" means
// what it says. Without this, the MRoPE test would still pass if dispatch
// returned kMRope for every "default" config.
TEST(RopeConfigTest, DefaultWithoutMropeSectionIsNotMRope)
{
    EXPECT_EQ(collectRopeConfig(withScaling({{"rope_type", "default"}})).type, RopeType::kDefault);
}

// ---------------------------------------------------------------------------
// Llama-3 scaling
//
// This used to route to kDefault, which dropped `factor` and every band cutoff
// with it and left the model running unscaled RoPE. Llama-3.1 and 3.2 are
// supported checkpoints, and their CI coverage is export-only, so nothing
// downstream would have reported it.
// ---------------------------------------------------------------------------

Json makeLlama3Scaling()
{
    return Json{{"rope_type", "llama3"}, {"factor", 8.0F}, {"low_freq_factor", 1.0F}, {"high_freq_factor", 4.0F},
        {"original_max_position_embeddings", 8192}};
}

// Given a Llama-3.1 config declaring llama3 rope scaling
// When the rope config is collected
// Then it reaches kLlama3 and every scaling parameter is collected
TEST(RopeConfigTest, Llama3ScalingIsCollectedRatherThanRoutedToDefault)
{
    auto const rope = collectRopeConfig(withScaling(makeLlama3Scaling()));

    ASSERT_EQ(rope.type, RopeType::kLlama3);
    ASSERT_TRUE(rope.llama3.has_value());
    EXPECT_FLOAT_EQ(rope.llama3->factor, 8.0F);
    EXPECT_FLOAT_EQ(rope.llama3->lowFreqFactor, 1.0F);
    EXPECT_FLOAT_EQ(rope.llama3->highFreqFactor, 4.0F);
    EXPECT_EQ(rope.llama3->originalMaxPositionEmbeddings, 8192);
}

// Same both-spellings rule the YaRN branch follows: transformers moved this key
// inside rope_scaling, and older checkpoints keep it at the top level.
TEST(RopeConfigTest, Llama3AcceptsTheOriginalContextLengthFromEitherLocation)
{
    Json scaling = makeLlama3Scaling();
    scaling.erase("original_max_position_embeddings");
    Json topLevel = withScaling(scaling);
    topLevel["original_max_position_embeddings"] = 4096;

    EXPECT_EQ(collectRopeConfig(topLevel).llama3->originalMaxPositionEmbeddings, 4096);
}

// Each of these is a divisor or a band boundary. Defaulting a missing one builds
// a plausible cache for the wrong model, which is the failure mode this whole
// section replaces -- so they are refused rather than filled in.
TEST(RopeConfigTest, Llama3RejectsAnIncompleteOrDegenerateScalingSection)
{
    auto reject = [](std::function<void(Json&)> const& damage) {
        Json scaling = makeLlama3Scaling();
        damage(scaling);
        EXPECT_THROW(collectRopeConfig(withScaling(scaling)), std::runtime_error);
    };

    reject([](Json& s) { s.erase("original_max_position_embeddings"); });
    reject([](Json& s) { s.erase("factor"); });
    reject([](Json& s) { s.erase("low_freq_factor"); });
    reject([](Json& s) { s.erase("high_freq_factor"); });
    reject([](Json& s) { s["factor"] = 0.0F; });
    reject([](Json& s) { s["low_freq_factor"] = 0.0F; });
    reject([](Json& s) { s["high_freq_factor"] = -1.0F; });
    // Equal cutoffs collapse the blend span to nothing.
    reject([](Json& s) { s["high_freq_factor"] = s["low_freq_factor"]; });
    // Swapped cutoffs are the case an inequality check alone would admit. They do not divide by zero -- the two
    // wavelength comparisons simply cover the whole range between them, so the blend never runs and 35 bands are
    // scaled where 29 should have been, with 6 blended. The cache builds, and inference proceeds on wrong
    // frequencies.
    reject([](Json& s) {
        s["low_freq_factor"] = 4.0F;
        s["high_freq_factor"] = 1.0F;
    });
}

// The parsing above only proves the numbers survived. This checks they are used,
// against inverse frequencies computed independently from HF's `apply_rope_scaling`
// for Llama-3.1-8B-Instruct (rope_theta 500000, head_dim 128, factor 8,
// low/high 1/4, original context 8192).
//
// One probe per regime the algorithm defines, each placed at a position where
// getting that band's regime wrong is unmistakable -- the gap between the right
// answer and the plausible mistake is between 0.28 and 1.78, against a 1e-4
// tolerance:
//
//   dim  pos   regime     correct    if mis-binned
//     0     1  untouched   +0.5403      +0.9922
//    20   128  untouched   -0.5218      +0.9651
//    33  4096  blended     +0.2860      +0.0064
//    36  4096  scaled      +0.9496      -0.8305
//    40  4096  scaled      +0.9902      +0.4327
//
// The untouched probes sit at small positions on purpose: those bands carry the
// highest frequencies, and by position 4096 the angle is thousands of radians,
// where float32 argument reduction costs more than the assertion's tolerance.
// The lowest band (dim 63) is left out rather than padded in -- its frequency is
// so small that scaled and unscaled agree to 5e-5 at every position the cache
// holds, so a probe there would assert nothing.
//
// Given a Llama-3.1 rope config
// When its cos/sin cache is built
// Then each band holds cos(position * its Llama-3-scaled inverse frequency)
TEST(RopeConfigTest, Llama3ScaledInverseFrequenciesReachTheCosSinCache)
{
    constexpr int64_t kRotaryDim{128};
    constexpr int64_t kMaxPositions{4097};

    struct Probe
    {
        int64_t dim;
        int64_t position;
        double invFreq;
    };

    std::array<Probe, 5> const kProbes{{
        {0, 1, 1.000000000000e+00},
        {20, 128, 1.656044008099e-02},
        {33, 4096, 3.126937503841e-04},
        {36, 4096, 7.784655273932e-05},
        {40, 4096, 3.428102195953e-05},
    }};

    Json rawConfig = withScaling(makeLlama3Scaling());
    rawConfig["rope_theta"] = 500000.0F;
    rawConfig["max_position_embeddings"] = 131072;
    auto const rope = collectRopeConfig(rawConfig);
    ASSERT_EQ(rope.type, RopeType::kLlama3);

    Tensor cache({1, kMaxPositions, kRotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "test::ropeCache");
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(initializeRopeCosSinCache(cache, rope, stream));

    std::vector<float> host(static_cast<size_t>(cache.getShape().volume()));
    ASSERT_EQ(
        cudaMemcpyAsync(host.data(), cache.rawPointer(), host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    for (auto const& probe : kProbes)
    {
        SCOPED_TRACE("rotary dimension " + std::to_string(probe.dim));
        double const angle = static_cast<double>(probe.position) * probe.invFreq;
        auto const base = static_cast<size_t>(probe.position * kRotaryDim);
        EXPECT_NEAR(host[base + static_cast<size_t>(probe.dim)], static_cast<float>(std::cos(angle)), 1e-4F);
        EXPECT_NEAR(
            host[base + static_cast<size_t>(probe.dim + kRotaryDim / 2)], static_cast<float>(std::sin(angle)), 1e-4F);
    }
}

// ---------------------------------------------------------------------------
// YaRN
// ---------------------------------------------------------------------------

Json makeYarnScaling(float factor = 4.0F)
{
    return Json{{"rope_type", "yarn"}, {"factor", factor}, {"original_max_position_embeddings", 4096}};
}

TEST(RopeConfigTest, YarnCollectsItsInterpolationParameters)
{
    Json scaling = makeYarnScaling();
    scaling["beta_fast"] = 16.0F;
    scaling["beta_slow"] = 2.0F;

    auto const rope = collectRopeConfig(withScaling(scaling));

    ASSERT_EQ(rope.type, RopeType::kYarn);
    ASSERT_TRUE(rope.yarn.has_value());
    EXPECT_FLOAT_EQ(rope.yarn->factor, 4.0F);
    EXPECT_EQ(rope.yarn->originalMaxPositionEmbeddings, 4096);
    EXPECT_FLOAT_EQ(rope.yarn->betaFast, 16.0F);
    EXPECT_FLOAT_EQ(rope.yarn->betaSlow, 2.0F);
}

// original_max_position_embeddings moved into rope_scaling in transformers v5
// and sits at the top level in older configs. Both spellings have to resolve, or
// one generation of checkpoint stops loading.
TEST(RopeConfigTest, YarnAcceptsTheOriginalContextLengthFromEitherLocation)
{
    Json nested = withScaling(makeYarnScaling());
    EXPECT_EQ(collectRopeConfig(nested).yarn->originalMaxPositionEmbeddings, 4096);

    Json scaling = makeYarnScaling();
    scaling.erase("original_max_position_embeddings");
    Json topLevel = withScaling(scaling);
    topLevel["original_max_position_embeddings"] = 8192;
    EXPECT_EQ(collectRopeConfig(topLevel).yarn->originalMaxPositionEmbeddings, 8192);
}

// mscale compensates for the interpolation: YaRN stretches the positions, so the
// attention magnitude has to be scaled back to match. Most checkpoints leave it
// out, which makes the derivation -- not the explicit value -- the path almost
// every YaRN model takes.
//
// Given a YaRN config stating a factor but no attention_factor
// When the rope config is collected
// Then mscale is 0.1*ln(factor)+1, a factor of 1 leaves it at 1, and an explicit
//      attention_factor overrides both
TEST(RopeConfigTest, YarnDerivesTheAttentionScaleWhenTheConfigOmitsIt)
{
    auto const derived = collectRopeConfig(withScaling(makeYarnScaling(4.0F)));
    EXPECT_FLOAT_EQ(derived.yarn->mscale, 0.1F * std::log(4.0F) + 1.0F);

    // No extension, no compensation.
    auto const unscaled = collectRopeConfig(withScaling(makeYarnScaling(1.0F)));
    EXPECT_FLOAT_EQ(unscaled.yarn->mscale, 1.0F);

    // An explicit attention_factor wins over the derived one.
    Json explicitScale = makeYarnScaling(4.0F);
    explicitScale["attention_factor"] = 1.5F;
    EXPECT_FLOAT_EQ(collectRopeConfig(withScaling(explicitScale)).yarn->mscale, 1.5F);
}

// Both parameters are divisors or interpolation references in the cache
// computation; neither has a safe default.
TEST(RopeConfigTest, YarnRejectsAMissingFactorOrOriginalContextLength)
{
    Json noFactor = makeYarnScaling();
    noFactor.erase("factor");
    EXPECT_THROW(collectRopeConfig(withScaling(noFactor)), std::runtime_error);

    Json zeroFactor = makeYarnScaling(0.0F);
    EXPECT_THROW(collectRopeConfig(withScaling(zeroFactor)), std::runtime_error);

    Json noOriginal = makeYarnScaling();
    noOriginal.erase("original_max_position_embeddings");
    EXPECT_THROW(collectRopeConfig(withScaling(noOriginal)), std::runtime_error);
}

TEST(RopeConfigTest, DynamicAndLongRopeAndProportionalDispatchByName)
{
    EXPECT_EQ(collectRopeConfig(withScaling({{"type", "dynamic"}})).type, RopeType::kDynamic);
    EXPECT_EQ(collectRopeConfig(withScaling({{"type", "proportional"}})).type, RopeType::kProportional);
}

// Some checkpoints carry both spellings of the key. `type` is consulted first,
// so it decides; a refactor that reorders the lookups would flip the variant for
// those configs without any other symptom.
//
// Given a config carrying both the legacy `type` key and the current `rope_type`
// When the rope config is collected
// Then the legacy `type` key decides
TEST(RopeConfigTest, LegacyTypeKeyOutranksRopeTypeKey)
{
    Json const both = withScaling({{"type", "proportional"}, {"rope_type", "dynamic"}});

    EXPECT_EQ(collectRopeConfig(both).type, RopeType::kProportional);
}

// ---------------------------------------------------------------------------
// Proportional parameters
// ---------------------------------------------------------------------------

// Proportional rope stores the reciprocal of the config's `factor`: the config
// states how far positions are stretched, the runtime needs the multiplier that
// compresses them back. Inverting this scales positions the wrong way.
//
// Given a proportional config declaring a scaling factor
// When the rope config is collected
// Then the stored scale is that factor's reciprocal
TEST(RopeConfigTest, ProportionalScaleIsReciprocalOfConfiguredFactor)
{
    auto const rope = collectRopeConfig(withScaling({{"type", "proportional"}, {"factor", 4.0F}}));

    EXPECT_FLOAT_EQ(rope.rotaryScale, 0.25F);
}

// Guard on the division above. A zero or negative factor is not merely odd, it
// produces an infinite or sign-flipped scale that would propagate into the
// cos/sin cache as NaN.
TEST(RopeConfigTest, ProportionalRejectsNonPositiveFactor)
{
    EXPECT_THROW(collectRopeConfig(withScaling({{"type", "proportional"}, {"factor", 0.0F}})), std::runtime_error);
    EXPECT_THROW(collectRopeConfig(withScaling({{"type", "proportional"}, {"factor", -2.0F}})), std::runtime_error);
}

// partial_rotary_factor appears at both nesting levels across checkpoints. The
// rope_scaling copy is the more specific one and must win.
//
// Given partial_rotary_factor declared both inside rope_scaling and at the top level
// When the rope config is collected
// Then the nested one wins
TEST(RopeConfigTest, ProportionalPrefersNestedPartialRotaryFactorOverTopLevel)
{
    Json config = withScaling({{"type", "proportional"}, {"partial_rotary_factor", 0.25F}});
    config["partial_rotary_factor"] = 0.75F;

    EXPECT_FLOAT_EQ(collectRopeConfig(config).partialRotaryFactor, 0.25F);
}

TEST(RopeConfigTest, ProportionalFallsBackToTopLevelPartialRotaryFactor)
{
    Json config = withScaling({{"type", "proportional"}});
    config["partial_rotary_factor"] = 0.75F;

    EXPECT_FLOAT_EQ(collectRopeConfig(config).partialRotaryFactor, 0.75F);
}

// ---------------------------------------------------------------------------
// LongRope parameters
// ---------------------------------------------------------------------------

Json makeLongRopeConfig()
{
    Json config = withScaling({{"rope_type", "longrope"}, {"long_factor", Json::array({1.0F, 2.0F, 3.0F})},
        {"short_factor", Json::array({1.0F, 1.0F, 1.0F})}});
    config["original_max_position_embeddings"] = 4096;
    config["max_position_embeddings"] = 131072;
    return config;
}

// The two factor arrays and the original context length together define which
// of the two cos/sin caches a given position reads from.
TEST(RopeConfigTest, LongRopeCollectsBothFactorArraysAndOriginalContextLength)
{
    auto const rope = collectRopeConfig(makeLongRopeConfig());

    ASSERT_EQ(rope.type, RopeType::kLongRope);
    ASSERT_TRUE(rope.longRope.has_value());
    EXPECT_EQ(rope.longRope->longFactor, std::vector<float>({1.0F, 2.0F, 3.0F}));
    EXPECT_EQ(rope.longRope->shortFactor, std::vector<float>({1.0F, 1.0F, 1.0F}));
    EXPECT_EQ(rope.longRope->originalMaxPositionEmbeddings, 4096);
    EXPECT_EQ(rope.maxPositionEmbeddings, 131072);
}

// Each guard below fails the load rather than proceeding with a half-built
// LongRopeParams, which would index past the end of a factor array while
// filling the cache.
TEST(RopeConfigTest, LongRopeRequiresLongFactor)
{
    Json config = makeLongRopeConfig();
    config["rope_scaling"].erase("long_factor");

    EXPECT_THROW(collectRopeConfig(config), std::runtime_error);
}

TEST(RopeConfigTest, LongRopeRequiresShortFactor)
{
    Json config = makeLongRopeConfig();
    config["rope_scaling"].erase("short_factor");

    EXPECT_THROW(collectRopeConfig(config), std::runtime_error);
}

// Both arrays are indexed by the same rotary dimension, so unequal lengths mean
// one of them is read out of bounds.
TEST(RopeConfigTest, LongRopeRejectsMismatchedFactorLengths)
{
    Json config = makeLongRopeConfig();
    config["rope_scaling"]["short_factor"] = Json::array({1.0F, 1.0F});

    EXPECT_THROW(collectRopeConfig(config), std::runtime_error);
}

// original_max_position_embeddings is the threshold that selects the short or
// long cache. Defaulting it would put every position on one side of the switch.
TEST(RopeConfigTest, LongRopeRequiresOriginalMaxPositionEmbeddings)
{
    Json config = makeLongRopeConfig();
    config.erase("original_max_position_embeddings");

    EXPECT_THROW(collectRopeConfig(config), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

// Absent keys must leave the declared defaults intact rather than zeroing them:
// a theta of 0 makes every rotary angle degenerate.
//
// Given a config that states neither rope_theta nor max_position_embeddings
// When the rope config is collected
// Then the declared defaults survive
TEST(RopeConfigTest, AbsentThetaAndMaxPositionsRetainDeclaredDefaults)
{
    auto const rope = collectRopeConfig(withScaling({{"type", "default"}}));

    EXPECT_FLOAT_EQ(rope.rotaryTheta, kDefaultRotaryTheta);
    EXPECT_EQ(rope.maxPositionEmbeddings, kDefaultMaxPositionEmbeddings);
    EXPECT_FLOAT_EQ(rope.rotaryScale, kDefaultRotaryScale);
    EXPECT_FLOAT_EQ(rope.partialRotaryFactor, 1.0F);
}

// Documents deliberate leniency: an unrecognized rope_type is not rejected, it
// silently keeps kDefault. This is load-bearing for checkpoints carrying scaling
// schemes the runtime does not implement but can approximate, and it is also why
// adding a new variant means adding it to this dispatch, not just to the enum.
//
// The name here is deliberately synthetic. An earlier version used "yarn", which
// stopped being unrecognized the moment YaRN support landed, and this test is
// what reported that: a name gaining real support has to be a visible change,
// not a silent one. Keep the example a name nobody will implement.
//
// Given a rope_type this runtime does not implement
// When the rope config is collected
// Then it falls back to the default variant and the load succeeds
TEST(RopeConfigTest, UnrecognizedRopeTypeIsSilentlyTreatedAsDefault)
{
    EXPECT_EQ(collectRopeConfig(withScaling({{"rope_type", "not_a_real_scaling_scheme"}, {"factor", 4.0F}})).type,
        RopeType::kDefault);
}

} // namespace
