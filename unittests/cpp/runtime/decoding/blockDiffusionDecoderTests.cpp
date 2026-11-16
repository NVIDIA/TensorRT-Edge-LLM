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

// Block-diffusion decoding assembled around a substitute engine.

#include "profiling/timer.h"
#include "substituteEngine.h"
#include <tuple>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

// --------------------------------------------------------------------------
// Block-diffusion decoding.
//
// Not speculative: one engine, and the decoder denoises a whole canvas per
// forward instead of proposing and verifying. It is here because the registry
// picks it from the same base config field every other strategy is chosen by,
// and because assembly branches on isDiffusionBackbone in half a dozen places.
// --------------------------------------------------------------------------

//! Small enough that the per-canvas allocations stay tiny; the default is 256.
constexpr int32_t kCanvasLength{8};

//! Denoising passes the decoder may spend on one canvas before committing it.
constexpr int32_t kMaxDenoisingSteps{2};

Json makeDiffusionConfig()
{
    Json config = makeTinyVanillaConfig();
    // The registry selects BlockDiffusionDecoder from this role alone.
    config["engine_role"] = "dllm";
    // parseEngineConfig rejects a dllm engine without it, so it is not optional despite reading like a tuning knob.
    config["diffusion_unified_conditioning"] = true;

    Json diffusion;
    diffusion["canvas_length"] = kCanvasLength;
    diffusion["max_denoising_steps"] = kMaxDenoisingSteps;
    diffusion["stability_window"] = 1;
    diffusion["t_max"] = 0.8F;
    diffusion["t_min"] = 0.4F;
    diffusion["entropy_bound"] = 0.1F;
    diffusion["entropy_threshold"] = 0.005F;
    config["diffusion_config"] = diffusion;
    return config;
}

class BlockDiffusionAssemblyTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("blockDiffusionAssemblyTests", makeDiffusionConfig());
    }
};

// Given a base config whose engine role is dllm
// When the runtime is assembled
// Then the registry picks the diffusion decoder and the round's ceiling is the canvas, not draftSteps + 1
TEST_F(BlockDiffusionAssemblyTest, SelectsTheDiffusionStrategyAndSizesTheRoundToTheCanvas)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);

    // The canvas is the unit of progress here, the way `draftSteps + 1` is for the chain decoders. Reading it
    // from the deployment rather than restating the constant keeps this honest if the resolution changes.
    ASSERT_EQ(artifacts.deployment.maxAcceptedTokensPerRound(), kCanvasLength);
    ASSERT_TRUE(artifacts.deployment.base.isDiffusionBackbone);

    auto runtime = makeRuntime(std::move(artifacts));

    ASSERT_FALSE(std::filesystem::exists(mModelDir / "dllm.engine"));
    EXPECT_FALSE(runtime.hasDraftModel());
    // Not asserted through getSpeculativeDecodingStrategyName(): that reports the speculative decoder, and block
    // diffusion is the registry's default decoder rather than a speculative one. The behavioral test below is what
    // separates it from the vanilla decoder that occupies the same slot.
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "vanilla");
}

TEST_F(BlockDiffusionAssemblyTest, SizesSharedContextMemoryFromTheExecutorItWasGiven)
{
    auto engine = makeEngine();

    // Diffusion takes its own path through the allocation arithmetic in initializeCommon -- the sampling workspace
    // is sized from the canvas rather than the batch, and several vanilla terms drop to zero. The context-memory
    // contract is unchanged by any of that, which is the point of asserting it separately here.
    expectContextMemorySizedFromTheExecutor(*engine, kContextMemoryBytes);

    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
}

// Given a request for a canvas worth of tokens
// When it is generated
// Then it costs fewer forward passes than tokens
TEST_F(BlockDiffusionAssemblyTest, FillsACanvasInFewerForwardsThanTokens)
{
    using ::testing::_;
    using ::testing::AtMost;

    auto engine = makeEngine();
    auto& mock = *engine;

    // This is what makes it block diffusion rather than the vanilla decoder sitting in the same registry slot.
    // Autoregressive decoding emits exactly one token per forward, so kCanvasLength tokens cost kCanvasLength
    // forwards and cannot cost fewer. Diffusion denoises a whole canvas per pass, so its forward count is decoupled
    // from the token count. Stated as that inequality rather than an exact count: how many denoising passes a
    // canvas needs is a convergence property, not a contract.
    EXPECT_CALL(mock, execute(_)).Times(AtMost(kCanvasLength - 1));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", kCanvasLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_GT(response.outputIds[0].size(), 1U) << "a canvas round should commit more than one token";
}

TEST_F(BlockDiffusionAssemblyTest, FixedRoundProfilingFitsTheRequestedDenoisingWork)
{
    // Given a 300-step diffusion canvas and a one-round experiment, when profiling is enabled,
    // then the whole canvas and all denoising observations survive the preallocated timer pool.
    constexpr int32_t kDenoisingSteps = 300;
    Json config = makeDiffusionConfig();
    config["diffusion_config"]["max_denoising_steps"] = kDenoisingSteps;
    std::ofstream(mModelDir / "config.json") << config.dump();
    auto engine = makeEngine();
    ON_CALL(*engine, execute(::testing::_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.benchmark.maxDecodeRounds = 1;
    rt::LLMInferenceRuntime runtime{std::move(artifacts), options, mStream};
    auto request = makeGreedyRequest("a", kCanvasLength);
    request.diffusionMaxDenoisingSteps = kDenoisingSteps;
    request.ignoreEos = true;
    struct RestoreProfiling
    {
        bool previous;
        ~RestoreProfiling()
        {
            setProfilingEnabled(previous);
        }
    } restoreProfiling{getProfilingEnabled()};
    setProfilingEnabled(true);

    rt::LLMGenerationResponse response;
    bool const success = runtime.handleRequest(request, response, mStream);
    auto const denoise = gTimer.getTimingData(metrics::StageNames::kBLOCK_DIFFUSION_DENOISE);
    EXPECT_EQ(std::make_tuple(success, response.outputIds, response.decodeRoundCounts,
                  denoise ? denoise->getTotalRuns() : int64_t{0}),
        std::make_tuple(true, std::vector<std::vector<int32_t>>{std::vector<int32_t>(kCanvasLength, kAgreedToken)},
            std::vector<int32_t>{1}, int64_t{kDenoisingSteps}));
}

//! Forces effectiveMaxDenoisingSteps() to take forceAccept on the very first (and only) step, sidestepping the
//! probabilistic entropy-threshold/stability-window accept path entirely -- which pins the happy-path argmax
//! accept call shape without needing to model that machinery.
constexpr int32_t kOneStepCanvasLength{2};

Json makeOneStepDiffusionConfig()
{
    Json config = makeDiffusionConfig();
    config["diffusion_config"]["canvas_length"] = kOneStepCanvasLength;
    config["diffusion_config"]["max_denoising_steps"] = 1;
    return config;
}

class OneStepBlockDiffusionAssemblyTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("oneStepBlockDiffusionAssemblyTests", makeOneStepDiffusionConfig());
    }
};

// Given max_denoising_steps=1 (forcing an immediate argmax accept) and every canvas position genuinely agreeing
// on the same argmax token
// When one round completes
// Then decodeStep()'s forward-call sequence is exactly one denoise forward followed by one commit forward, and
// the whole canvas is accepted and committed
//
// denoiseDims() and diffusionCommitDims() share kDecodeProfile and the same selectLen==canvasLen, so profile and
// selectLen alone cannot tell a denoise forward from a commit forward -- contextMaskSelectorLen is the only
// InferenceDims field that differs between them (denoiseDims() sets it to `batch`, diffusionCommitDims() sets it
// to 0). Found by mutation testing a denoise-loop boundary change that skipped denoising but forced an extra round
// (still ending in one commit call): invisible to profile/selectLen/call-count assertions alone, caught only once
// this field was added.
TEST_F(OneStepBlockDiffusionAssemblyTest, DenoisesOnceThenCommitsOnceWhenEveryPositionGenuinelyAgrees)
{
    using ::testing::_;

    auto engine = makeEngine();
    auto& mock = *engine;
    std::vector<int32_t> contextMaskSelectorLens;
    ON_CALL(mock, prepare(_, _, _, _))
        .WillByDefault(
            [&, this](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                contextMaskSelectorLens.push_back(dims.contextMaskSelectorLen);
                return true;
            });
    ON_CALL(mock, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", kOneStepCanvasLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(contextMaskSelectorLens.size(), 3U) << "prefill, one denoise, one commit";
    EXPECT_GT(contextMaskSelectorLens[1], 0) << "denoiseDims() sets contextMaskSelectorLen=batch";
    EXPECT_EQ(contextMaskSelectorLens[2], 0) << "diffusionCommitDims() sets contextMaskSelectorLen=0";

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(
        response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kOneStepCanvasLength), ::testing::Each(kAgreedToken)))
        << "every canvas position genuinely agrees on the same argmax token, so the forced-accept step should "
           "commit the whole canvas";
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

} // namespace
