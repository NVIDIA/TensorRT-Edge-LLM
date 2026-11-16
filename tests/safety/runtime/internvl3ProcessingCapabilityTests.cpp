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

#include "safety_inference/image_preprocess/models/internvl3/internvlImagePlan.h"
#include "safety_inference/runtime/processing/singleRequestTextResponse.h"
#include "safety_inference/runtime/resources/declaredResourceManager.h"
#include "safety_inference/tokenizer/models/internvl3/internvl3TokenizerCapability.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

namespace trt_edgellm::safety
{
namespace
{
constexpr ResourceOwnerId kTEST_PROCESSING_OWNER{3};

class InternVL3TokenizerTest : public testing::Test
{
protected:
    void SetUp() override
    {
        char path[]{"/tmp/internvl3-tokenizer-XXXXXX"};
        int const descriptor = mkstemp(path);
        ASSERT_NE(descriptor, -1);
        close(descriptor);
        mPath = path;
        std::ofstream output(mPath);
        output << R"({
          "model": {"type": "BPE", "byte_fallback": false,
                    "vocab": {"H": 0, "i": 1, "Ã": 2, "Hi": 3,
                              "s": 10, "y": 11, "t": 12, "e": 13, "m": 14,
                              "u": 15, "r": 16, "a": 17, "n": 18, "Ċ": 19,
                              "T": 20, "o": 21, "l": 22, "Ġ": 23,
                              "ĠT": 24, "ĠTo": 25, "ĠTol": 26, "ĠToll": 27}},
          "added_tokens": [
            {"content": "<img>", "id": 4, "special": true},
            {"content": "<IMG_CONTEXT>", "id": 5, "special": true},
            {"content": "</img>", "id": 6, "special": true},
            {"content": "<image>", "id": 7, "special": true},
            {"content": "<pad>", "id": 8, "special": true},
            {"content": "<eos>", "id": 9, "special": true},
            {"content": "<|im_start|>", "id": 28, "special": true},
            {"content": "<|im_end|>", "id": 29, "special": true}
          ]
        })";
        output.close();
        mArtifacts.path = mPath;
        mArtifacts.imageStartId = 4;
        mArtifacts.imageContextId = 5;
        mArtifacts.imageEndId = 6;
        mArtifacts.padTokenId = 8;
        mArtifacts.stopTokenIds[0] = 9;
        mArtifacts.stopTokenIds[1] = 8;
        mArtifacts.stopTokenCount = 2;
        mLimits.maxPromptBytes = 160;
        mLimits.maxPromptTokens = 64;
        mLimits.maxResponseBytes = 32;
        mLimits.vocabularySize = 30;
        mDeclaration.owner = kTEST_PROCESSING_OWNER;
        auto status = mTokenizer.declareResources(mArtifacts, mLimits, mDeclaration);
        ASSERT_TRUE(status.okStatus()) << status.message();
        status = mManager.prepareResources({mDeclaration});
        ASSERT_TRUE(status.okStatus()) << status.message();
        mResourcesPrepared = true;
        PreparedResources resources{};
        ASSERT_TRUE(mManager.resourcesFor(mDeclaration.owner, resources).okStatus());
        status = mTokenizer.initialize(mArtifacts, resources);
        ASSERT_TRUE(status.okStatus()) << status.message();
        mTokenizerInitialized = true;
    }

    void TearDown() override
    {
        if (mTokenizerInitialized)
        {
            EXPECT_TRUE(mTokenizer.deinitialize().okStatus());
        }
        mLifecycle = LifecycleState::kCLOSED;
        if (mResourcesPrepared)
        {
            EXPECT_TRUE(mManager.deinitialize().okStatus());
        }
        std::filesystem::remove(mPath);
    }

    std::filesystem::path mPath{};
    LifecycleState mLifecycle{LifecycleState::kINITIALIZATION};
    DeclaredResourceManager mManager{mLifecycle};
    InternVL3TokenizerCapability mTokenizer{};
    TokenizerArtifactSpec mArtifacts{};
    ProcessingLimits mLimits{};
    ResourceDeclaration mDeclaration{};
    bool mResourcesPrepared{false};
    bool mTokenizerInitialized{false};
};

// Given a two-block landscape or portrait image, when planning the fixed grid,
// then the selected orientation is deterministic and the requested thumbnail count is exact.
TEST(InternVL3ImagePlan, SelectsDeterministicFixedGrid)
{
    InternVLTilePlan landscape{};
    InternVLTilePlan portrait{};
    ASSERT_TRUE(computeInternVLTilePlan(896, 448, 2, 2, false, landscape));
    ASSERT_TRUE(computeInternVLTilePlan(448, 896, 2, 2, false, portrait));
    EXPECT_EQ(landscape.columns, 2);
    EXPECT_EQ(landscape.rows, 1);
    EXPECT_EQ(landscape.totalTiles, 2);
    EXPECT_EQ(portrait.columns, 1);
    EXPECT_EQ(portrait.rows, 2);
    EXPECT_EQ(portrait.totalTiles, 2);
}

// Given a tile limit beyond the admitted InternVL work bound, when planning an image,
// then the planner rejects it before entering an unbounded divisor search or overflowing resized dimensions.
TEST(InternVL3ImagePlan, RejectsUnboundedTileProfile)
{
    InternVLTilePlan plan{};
    EXPECT_FALSE(computeInternVLTilePlan(448, 448, 1, kMaxInternVLTiles + 1, true, plan));
    EXPECT_EQ(plan.totalTiles, 0);
}

// Given a valid byte-level tokenizer and one image marker, when encoding and expanding the prompt,
// then it emits the configured markers, exact image span, and no placeholder token.
TEST_F(InternVL3TokenizerTest, EncodesAndExpandsOneImagePlaceholder)
{
    char const prompt[]{"Hi<image>"};
    auto const encoded = mTokenizer.encodePrompt({prompt, sizeof(prompt) - 1});
    ASSERT_TRUE(encoded.status.okStatus());
    auto const expanded = mTokenizer.expandImagePlaceholder(encoded.tokens, 2);
    ASSERT_TRUE(expanded.status.okStatus());
    ASSERT_EQ(expanded.tokens.size, 5U);
    EXPECT_EQ(expanded.imageOffset, 2U);
    int32_t const expected[]{3, 4, 5, 5, 6};
    EXPECT_TRUE(std::equal(expanded.tokens.data, expanded.tokens.data + expanded.tokens.size, expected));
}

// Given a Qwen byte-level vocabulary containing a merged leading-space word, when encoding that word,
// then the pre-tokenizer keeps the space with the letters and emits the same single token as Standard EdgeLLM.
TEST_F(InternVL3TokenizerTest, KeepsLeadingSpaceWithQwenLetterPiece)
{
    char const prompt[]{" Toll<image>"};
    auto const encoded = mTokenizer.encodePrompt({prompt, sizeof(prompt) - 1});
    ASSERT_TRUE(encoded.status.okStatus()) << encoded.status.message();
    ASSERT_EQ(encoded.tokens.size, 2U);
    EXPECT_EQ(encoded.tokens.data[0], 27);
}

// Given separate empty-system and user image/text messages, when applying the InternVL chat template,
// then the tokenizer accepts the existing public request shape and preserves one expandable image marker.
TEST_F(InternVL3TokenizerTest, AppliesChatTemplateToStructuredMessages)
{
    LLMGenerationRequest::Request request;
    request.messages = {{"system", {{"text", ""}}}, {"user", {{"image", ""}, {"text", "Hi"}}}};
    auto const encoded = mTokenizer.encodeMessages(request);
    ASSERT_TRUE(encoded.status.okStatus()) << encoded.status.message();
    auto const expanded = mTokenizer.expandImagePlaceholder(encoded.tokens, 2);
    ASSERT_TRUE(expanded.status.okStatus()) << expanded.status.message();
    EXPECT_GT(expanded.imageOffset, 0U);
    EXPECT_EQ(mTokenizer.stopTokens().size, 2U);
    EXPECT_EQ(mTokenizer.stopTokens().data[0], 9);
    EXPECT_EQ(mTokenizer.stopTokens().data[1], 8);
}

// Given an artifact whose configured image-control ID disagrees with the tokenizer,
// when the capability initializes, then artifact validation rejects the mismatch before request processing.
TEST_F(InternVL3TokenizerTest, RejectsMismatchedImageControlId)
{
    LifecycleState lifecycle{LifecycleState::kINITIALIZATION};
    DeclaredResourceManager manager{lifecycle};
    InternVL3TokenizerCapability tokenizer;
    auto artifacts = mArtifacts;
    artifacts.imageContextId = artifacts.imageStartId;
    ResourceDeclaration declaration{};
    declaration.owner = kTEST_PROCESSING_OWNER;
    ASSERT_TRUE(tokenizer.declareResources(artifacts, mLimits, declaration).okStatus());
    ASSERT_TRUE(manager.prepareResources({declaration}).okStatus());
    PreparedResources resources{};
    ASSERT_TRUE(manager.resourcesFor(declaration.owner, resources).okStatus());
    EXPECT_EQ(tokenizer.initialize(artifacts, resources).code(), SafetyStatusCode::kInvalidArgument);
    EXPECT_TRUE(tokenizer.deinitialize().okStatus());
    lifecycle = LifecycleState::kCLOSED;
    EXPECT_TRUE(manager.deinitialize().okStatus());
}

// Given generated byte tokens followed by the configured stop token, when decoding into bounded caller storage,
// then valid UTF-8 is published and an incomplete sequence leaves the destination unchanged.
TEST_F(InternVL3TokenizerTest, DecodesBoundedUtf8BeforePublication)
{
    int32_t const valid[]{3, 9};
    char output[16]{'x'};
    std::size_t written{0};
    ASSERT_TRUE(mTokenizer.decodeResponse({valid, 2}, {output, sizeof(output)}, written).okStatus());
    EXPECT_EQ(written, 2U);
    EXPECT_STREQ(output, "Hi");
    int32_t const invalid[]{2, 9};
    std::fill(std::begin(output), std::end(output), 'x');
    EXPECT_FALSE(mTokenizer.decodeResponse({invalid, 2}, {output, sizeof(output)}, written).okStatus());
    EXPECT_EQ(written, 0U);
    EXPECT_EQ(output[0], 'x');
}

// Given an initialized tokenizer and a generated token ID outside the configured vocabulary,
// when decoding the response, then decoding fails without accessing token storage or publishing output.
TEST_F(InternVL3TokenizerTest, RejectsOutOfVocabularyGeneratedToken)
{
    std::array<int32_t, 2> const invalidIds{-1, mLimits.vocabularySize};
    for (auto const id : invalidIds)
    {
        char output[16]{};
        std::size_t written{7};
        EXPECT_FALSE(mTokenizer.decodeResponse({&id, 1}, {output, sizeof(output)}, written).okStatus());
        EXPECT_EQ(written, 0U);
    }
}

// Given an InternVL response whose generated tokens end with incomplete UTF-8,
// when response formatting fails, then a subsequent response can still be formatted successfully.
TEST_F(InternVL3TokenizerTest, RecoversAfterIncompleteUtf8Response)
{
    SingleRequestTextResponse formatter;
    LLMGenerationResponse response;
    response.outputTexts.resize(1);
    response.outputTexts.front().reserve(mLimits.maxResponseBytes + 1U);
    int32_t const invalid[]{2, 9};
    EXPECT_FALSE(formatter
            .formatResponse(
                mTokenizer, mLimits.maxResponseBytes, ModelInferenceResult{{invalid, std::size(invalid)}}, response)
            .okStatus());
    int32_t const valid[]{3, 9};
    ASSERT_TRUE(formatter
            .formatResponse(
                mTokenizer, mLimits.maxResponseBytes, ModelInferenceResult{{valid, std::size(valid)}}, response)
            .okStatus());
    EXPECT_EQ(response.outputTexts.front(), "Hi");
}

} // namespace
} // namespace trt_edgellm::safety
