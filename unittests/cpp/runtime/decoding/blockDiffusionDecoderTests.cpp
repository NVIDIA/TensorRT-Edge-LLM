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

#include "substituteEngine.h"

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
// Then the registry picks the diffusion decoder and the round's ceiling is the canvas, not draftingStep + 1
TEST_F(BlockDiffusionAssemblyTest, SelectsTheDiffusionStrategyAndSizesTheRoundToTheCanvas)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);

    // The canvas is the unit of progress here, the way `draftingStep + 1` is for the chain decoders. Reading it
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
} // namespace
