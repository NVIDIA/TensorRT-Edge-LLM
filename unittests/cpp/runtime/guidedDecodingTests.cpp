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

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "guidedTokenizerFixture.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

rt::GuidedDecodingParams makeGuide(rt::GuideType type, std::string guide)
{
    rt::GuidedDecodingParams params;
    params.type = type;
    params.guide = std::move(guide);
    return params;
}

// ---------------------------------------------------------------- pre-check (D6 appendix)

//! Keywords XGrammar compiles and then ignores are the dangerous case: the output would
//! violate the schema while the API claims it cannot. Rejecting up front is the only defence,
//! because compilation itself succeeds.
TEST(GuidedDecodingValidationTest, RejectsSilentlyIgnoredKeywords)
{
    for (char const* schema : {
             R"({"type":"object","properties":{"a":{"type":"integer","multipleOf":5}}})",
             R"({"type":"array","items":{"type":"integer"},"uniqueItems":true})",
             R"({"type":"array","items":{"type":"integer"},"contains":{"const":9}})",
             R"({"type":"array","items":{"type":"integer"},"contains":{"const":9},"minContains":2})",
             R"({"type":"array","items":{"type":"integer"},"maxContains":2})",
         })
    {
        std::string failReason;
        EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << "schema should have been rejected: " << schema;
        EXPECT_FALSE(failReason.empty());
    }
}

//! patternProperties compiles into a grammar that rejects every document, including matching
//! ones, which would otherwise surface much later as an unsatisfiable mask.
TEST(GuidedDecodingValidationTest, RejectsOverRestrictiveKeywords)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"type":"object","patternProperties":{"^S_":{"type":"string"}}})"),
        failReason));
}

//! The check keys off the keyword alone rather than a sibling "type". vLLM looks at the type
//! first, so a schema like this one slips past its check entirely.
TEST(GuidedDecodingValidationTest, RejectsUnsupportedKeywordWithoutSiblingType)
{
    std::string failReason;
    EXPECT_FALSE(
        rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, R"({"multipleOf": 5})"), failReason));
}

//! Unsupported keywords must be found however deeply they are buried.
TEST(GuidedDecodingValidationTest, RejectsNestedUnsupportedKeyword)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema,
            R"({"type":"object","properties":{"outer":{"type":"array","items":{"type":"object",
                "properties":{"inner":{"type":"integer","multipleOf":3}}}}}})"),
        failReason));
}

//! An unknown `format` is ignored rather than rejected by XGrammar, so it belongs on the list;
//! the fourteen it really implements must still pass.
TEST(GuidedDecodingValidationTest, RejectsUnknownFormatButAcceptsSupportedOnes)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"type":"string","format":"totally-bogus"})"), failReason));

    for (char const* format : {"email", "date-time", "uuid", "ipv4", "uri"})
    {
        auto const schema = std::string(R"({"type":"string","format":")") + format + R"("})";
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << format << ": " << failReason;
    }
}

//! propertyNames was broken in v0.1.25 and works in v0.2.1. The blacklist tracks measurement
//! against the pinned version, so this one is deliberately allowed; if a version bump
//! regresses it, this test is the reminder to re-measure.
TEST(GuidedDecodingValidationTest, AcceptsKeywordsThatWorkOnThePinnedVersion)
{
    std::string failReason;
    for (char const* schema : {
             R"({"type":"object","propertyNames":{"pattern":"^[a-z]+$"}})",
             R"({"type":"object","properties":{"a":{"type":"integer","minimum":1,"maximum":9}}})",
             R"({"type":"string","minLength":3,"maxLength":5})",
             R"({"type":"array","items":{"type":"integer"},"minItems":2})",
             R"({"type":"object","properties":{"a":{"enum":["x","y"]}}})",
         })
    {
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << schema << ": " << failReason;
    }
}

//! Under "properties" (and $defs, definitions, ...) the keys are user-chosen names, not schema
//! keywords, so a field literally named "contains" must not be mistaken for the keyword.
TEST(GuidedDecodingValidationTest, AcceptsPropertyNamesThatCollideWithBlacklistedKeywords)
{
    std::string failReason;
    for (char const* schema : {
             R"({"type":"object","properties":{"contains":{"type":"string"}},"required":["contains"]})",
             R"({"type":"object","properties":{"multipleOf":{"type":"integer"},"format":{"type":"string"}}})",
             R"({"$defs":{"uniqueItems":{"type":"string"}},"type":"object"})",
             R"({"type":"object","properties":{"a":{"type":"string"}},"required":["contains","multipleOf"]})",
         })
    {
        EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, schema), failReason))
            << schema << ": " << failReason;
    }
}

//! The name-space exemption must not hide a real keyword one level deeper.
TEST(GuidedDecodingValidationTest, StillRejectsUnsupportedKeywordInsideANamedSubschema)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema,
            R"({"type":"object","properties":{"contains":{"type":"integer","multipleOf":5}}})"),
        failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(
        makeGuide(rt::GuideType::kJsonSchema, R"({"$defs":{"ok":{"type":"array","uniqueItems":true}}})"), failReason));
}

//! A guide big enough to stall compilation is a denial-of-service vector, since compilation is
//! synchronous and runs before any GPU work.
TEST(GuidedDecodingValidationTest, RejectsOversizedGuide)
{
    std::string const huge(limits::security::kMaxGuidedDecodingGuideBytes + 1, 'a');
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, huge), failReason));
}

//! Malformed JSON must be reported as such rather than reaching the compiler.
TEST(GuidedDecodingValidationTest, RejectsMalformedJsonSchema)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, R"({"type":)"), failReason));
}

//! Regex and EBNF are not JSON and must not be parsed as such.
TEST(GuidedDecodingValidationTest, AcceptsNonJsonGuides)
{
    std::string failReason;
    EXPECT_TRUE(
        rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, R"([a-z]+@[a-z]+\.[a-z]{2,3})"), failReason))
        << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" | "b")"), failReason))
        << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    EXPECT_TRUE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"(["yes","no"])"), failReason))
        << failReason;
}

//! choice carries a JSON array rather than a grammar, so the shape is checked up front; an
//! empty array would otherwise lower to an alternation with no branches.
TEST(GuidedDecodingValidationTest, RejectsMalformedChoiceList)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, "[\"a\""), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, "[]"), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"({"a":1})"), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kChoice, R"(["a",1])"), failReason));
}

//! Only json_object may carry an empty guide; for the rest an empty string means "unset" and
//! must never be read as "a grammar that accepts nothing".
TEST(GuidedDecodingValidationTest, RejectsEmptyGuideExceptJsonObject)
{
    std::string failReason;
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kRegex, ""), failReason));
    EXPECT_FALSE(rt::validateGuidedDecodingParams(makeGuide(rt::GuideType::kJsonSchema, ""), failReason));
}

// ---------------------------------------------------------------- request helper

TEST(GuidedDecodingRequestTest, DetectsGuidedDecodingInAnySlot)
{
    rt::LLMGenerationRequest request;
    request.requests.resize(2);
    EXPECT_FALSE(rt::hasGuidedDecoding(request));
    request.requests[1].guidedDecoding = makeGuide(rt::GuideType::kJsonObject, "");
    EXPECT_TRUE(rt::hasGuidedDecoding(request));
}

// ---------------------------------------------------------------- apply kernel (T4)

class ApplyTokenBitmaskTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }
    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! Pack a per-token allow list into the bit layout XGrammar produces: bit set = allowed.
    static std::vector<int32_t> packMask(std::vector<std::vector<int32_t>> const& allowedPerRow, int32_t vocabSize)
    {
        int32_t const words = (vocabSize + 31) / 32;
        std::vector<int32_t> packed(allowedPerRow.size() * words, 0);
        for (size_t row = 0; row < allowedPerRow.size(); ++row)
        {
            for (auto const tokenId : allowedPerRow[row])
            {
                packed[row * words + tokenId / 32] |= (1 << (tokenId % 32));
            }
        }
        return packed;
    }

    cudaStream_t mStream{};
};

//! Forbidden entries go to the sentinel; allowed entries must be left exactly as they were.
TEST_F(ApplyTokenBitmaskTest, MasksForbiddenTokensAndLeavesAllowedUntouched)
{
    constexpr int32_t kROWS = 2;
    constexpr int32_t kVOCAB = 8;
    int32_t const words = (kVOCAB + 31) / 32;

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    std::vector<float> hostLogits(kROWS * kVOCAB);
    for (size_t i = 0; i < hostLogits.size(); ++i)
    {
        hostLogits[i] = static_cast<float>(i) + 1.0F;
    }
    copyHostToDevice<float>(logits, hostLogits);

    auto const packed = packMask({{1, 3}, {0, 7}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1, 1});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(hostLogits.size());
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int32_t>> const allowed{{1, 3}, {0, 7}};
    for (int32_t row = 0; row < kROWS; ++row)
    {
        for (int32_t token = 0; token < kVOCAB; ++token)
        {
            bool const isAllowed = std::find(allowed[row].begin(), allowed[row].end(), token) != allowed[row].end();
            float const value = result[row * kVOCAB + token];
            if (isAllowed)
            {
                EXPECT_FLOAT_EQ(value, hostLogits[row * kVOCAB + token]) << "row " << row << " token " << token;
            }
            else
            {
                EXPECT_FLOAT_EQ(value, kMaskedLogitValue) << "row " << row << " token " << token;
            }
        }
    }
}

//! A cleared row flag is how unconstrained requests share a batch with constrained ones at
//! zero cost; the kernel must not touch those rows even though the bitmask holds stale bits.
TEST_F(ApplyTokenBitmaskTest, SkipsRowsWithClearedFlag)
{
    constexpr int32_t kROWS = 2;
    constexpr int32_t kVOCAB = 4;
    int32_t const words = (kVOCAB + 31) / 32;

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    std::vector<float> const hostLogits{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    copyHostToDevice<float>(logits, hostLogits);

    // Row 1's mask forbids everything, but its flag is clear, so it must survive intact.
    auto const packed = packMask({{0}, {}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1, 0});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(hostLogits.size());
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(result[0], 1.0F);
    for (int32_t token = 1; token < kVOCAB; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue);
    }
    for (int32_t token = 0; token < kVOCAB; ++token)
    {
        EXPECT_FLOAT_EQ(result[kVOCAB + token], hostLogits[kVOCAB + token]);
    }
}

//! The sentinel must stay finite after the sampler divides by temperature; -FLT_MAX would
//! overflow back to -inf and make the softmax denominator zero.
TEST_F(ApplyTokenBitmaskTest, MaskedValueSurvivesTemperatureScaling)
{
    constexpr float kLOWEST_TEMPERATURE = 0.01F;
    float const scaled = kMaskedLogitValue / kLOWEST_TEMPERATURE;
    EXPECT_TRUE(std::isfinite(scaled));
    EXPECT_LT(scaled, 0.0F);
}

//! A vocabulary that is not a multiple of 32 exercises the partial trailing word.
TEST_F(ApplyTokenBitmaskTest, HandlesVocabularyNotAlignedToWordBoundary)
{
    constexpr int32_t kROWS = 1;
    constexpr int32_t kVOCAB = 35;
    int32_t const words = (kVOCAB + 31) / 32;
    ASSERT_EQ(words, 2);

    rt::Tensor logits({kROWS, kVOCAB}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    std::vector<float> const hostLogits(kVOCAB, 5.0F);
    copyHostToDevice<float>(logits, hostLogits);

    auto const packed = packMask({{34}}, kVOCAB);
    rt::Tensor bitmask({kROWS, words}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "bitmask");
    copyHostToDevice<int32_t>(bitmask, packed);

    rt::Tensor rowNeedsMask({kROWS}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "rowNeedsMask");
    copyHostToDevice<int32_t>(rowNeedsMask, {1});

    applyTokenBitmask(logits, bitmask, rowNeedsMask, kROWS, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[34], 5.0F);
    for (int32_t token = 0; token < 34; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue) << "token " << token;
    }
}

// ---------------------------------------------------------------- reasoning gate seed

//! Prompt markers are authoritative. The request's thinking mode only resolves a prompt with no
//! marker, where the model may still open a reasoning block before the grammar starts.
TEST(GuidedDecodingReasoningGateTest, SeedsFromPromptMarkersAndThinkingMode)
{
    constexpr int32_t kSTART_THINK = 100;
    constexpr int32_t kEND_THINK = 101;
    constexpr int32_t kSTART_CHANNEL = 102;
    constexpr int32_t kEND_CHANNEL = 103;
    constexpr int32_t kTEXT = 7;
    std::vector<int32_t> const starts{kSTART_CHANNEL, kSTART_THINK};
    std::vector<int32_t> const ends{kEND_CHANNEL, kEND_THINK};

    // Template opened and closed the block: constrain from the first generated token. The end
    // marker is never generated, so a gate waiting for one would never open.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kTEXT, kSTART_THINK, kEND_THINK}, starts, ends, /*thinkingEnabled=*/true));

    // Template left the block open: the model is mid-reasoning.
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kSTART_THINK}, starts, ends, /*thinkingEnabled=*/false));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kSTART_CHANNEL}, starts, ends, /*thinkingEnabled=*/false));

    // With no marker, disabled thinking starts the grammar immediately. Enabled thinking waits
    // because a model-native reasoning block may still begin with the first generated token.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kTEXT, kTEXT}, starts, ends, /*thinkingEnabled=*/false));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kTEXT}, starts, ends, /*thinkingEnabled=*/true));

    // Explicit markers override the request mode.
    EXPECT_FALSE(rt::reasoningClosedInPrompt({kTEXT, kSTART_CHANNEL}, starts, ends, /*thinkingEnabled=*/false));
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kSTART_CHANNEL, kEND_CHANNEL}, starts, ends, /*thinkingEnabled=*/true));

    // Only the most recent marker counts; an earlier turn's block must not leak.
    EXPECT_TRUE(rt::reasoningClosedInPrompt(
        {kSTART_THINK, kEND_THINK, kTEXT, kSTART_THINK, kEND_THINK}, starts, ends, /*thinkingEnabled=*/true));
    EXPECT_FALSE(rt::reasoningClosedInPrompt(
        {kSTART_THINK, kEND_THINK, kTEXT, kSTART_THINK}, starts, ends, /*thinkingEnabled=*/false));

    // Marker families are independent: a channel block closes on the channel end marker.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({kSTART_CHANNEL, kEND_CHANNEL}, starts, ends, /*thinkingEnabled=*/true));
}

//! A model with no reasoning markers has no reasoning phase, so it must not sit behind a gate
//! that can never open. Absent tokens come back from the tokenizer as -1.
TEST(GuidedDecodingReasoningGateTest, ModelWithoutMarkersIsNeverGated)
{
    std::vector<int32_t> const absent{-1, -1};
    EXPECT_TRUE(rt::reasoningClosedInPrompt({7, 8, 9}, absent, absent, /*thinkingEnabled=*/true));
    EXPECT_TRUE(rt::reasoningClosedInPrompt({}, absent, absent, /*thinkingEnabled=*/false));

    // -1 must not match a padding or placeholder id that happens to be negative.
    EXPECT_TRUE(rt::reasoningClosedInPrompt({-1, 7}, absent, absent, /*thinkingEnabled=*/true));
    EXPECT_FALSE(rt::reasoningClosedInPrompt({-1, 7}, {-1, 100}, {-1, 101}, /*thinkingEnabled=*/true));
}

// ---------------------------------------------------------------- GuidedDecoder lifecycle

//! A tiny real tokenizer: enough vocabulary to express `{"a":1}` plus an EOS and a few merged
//! tokens, so the grammar machinery can be exercised end to end without loading a model.
class GuidedDecoderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = std::filesystem::temp_directory_path() / "edgellm_guided_decoder_test";
        std::filesystem::remove_all(mDir);
        std::filesystem::create_directories(mDir);

        guidedFixture::writeBpeTokenizer(mDir);
        ASSERT_TRUE(mTokenizer.loadFromHF(mDir));
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! No reduced-vocabulary map: output space is the identity over the full vocabulary.
    void initDecoder(rt::GuidedDecoder& decoder, int32_t maxBatchSize, int32_t maxRowsPerSlot = 1)
    {
        rt::Tensor emptyMap;
        decoder.initialize(maxBatchSize, maxRowsPerSlot, kVOCAB_SIZE, kVOCAB_SIZE, &mTokenizer, emptyMap, mStream);
    }

    static constexpr int32_t kVOCAB_SIZE = guidedFixture::kBPE_VOCAB_SIZE;
    static constexpr int32_t kEOS_ID = guidedFixture::kEOS;
    static constexpr int32_t kTHINK_END_ID = guidedFixture::kTHINK_END;

    std::filesystem::path mDir;
    tokenizer::Tokenizer mTokenizer;
    cudaStream_t mStream{};
};

TEST_F(GuidedDecoderTest, CapturesVerifyTopologyWithoutStaleChainMetadata)
{
    // Given a tree followed by a chain and an unpadded tree, when their drafts are captured,
    // then tokens remain exact, only the current tree exposes parents, and only a draft
    // captured with valid counts exposes them.
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1, 3);
    rt::Tensor hostTokens({3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    rt::Tensor hostParents({3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    rt::Tensor hostCounts({1}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    std::vector<int32_t> const values{4, 5, 6};
    std::vector<int32_t> const parents{-1, 0, 0};
    std::copy(values.begin(), values.end(), hostTokens.dataPointer<int32_t>());
    std::copy(parents.begin(), parents.end(), hostParents.dataPointer<int32_t>());
    hostCounts.dataPointer<int32_t>()[0] = 2;
    rt::Tensor tokens({3}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor topology({3}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor counts({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(
        tokens.rawPointer(), hostTokens.rawPointer(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        topology.rawPointer(), hostParents.rawPointer(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        counts.rawPointer(), hostCounts.rawPointer(), sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    std::vector<int32_t> observed;
    auto capture = [&](rt::SpecVerifyLayout layout, rt::Tensor const* validCounts) {
        decoder.captureDraft({tokens, validCounts, &topology, nullptr, 1, 3, 3, layout}, mStream);
        decoder.waitForDraftTopology();
        auto const* captured = decoder.hostDraftTokens();
        observed.insert(observed.end(), captured, captured + 3);
        auto const* capturedParents = decoder.hostDraftParentIds();
        auto const* capturedCounts = decoder.hostDraftValidCounts();
        for (int i = 0; i < 3; ++i)
            observed.push_back(capturedParents ? capturedParents[i] : -9);
        observed.push_back(capturedCounts ? capturedCounts[0] : -9);
    };
    capture(rt::SpecVerifyLayout::kTree, &counts);
    capture(rt::SpecVerifyLayout::kChain, &counts);
    capture(rt::SpecVerifyLayout::kTree, nullptr);
    EXPECT_EQ(observed, (std::vector<int32_t>{4, 5, 6, -1, 0, 0, 2, 4, 5, 6, -9, -9, -9, 2, 4, 5, 6, -1, 0, 0, -9}));
}

TEST_F(GuidedDecoderTest, CompilesEveryGuideType)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    struct Case
    {
        rt::GuideType type;
        char const* guide;
    };
    std::vector<Case> const cases{
        {rt::GuideType::kJsonObject, ""},
        {rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"},
        {rt::GuideType::kRegex, "a+"},
        {rt::GuideType::kEbnf, R"(root ::= "a" | "b")"},
        {rt::GuideType::kStructuralTag,
            R"({"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<f="],
                "tags":[{"begin":"<f=g>","content":{"type":"json_schema","json_schema":{"type":"object"}},
                "end":"</f>"}]}})"},
        {rt::GuideType::kChoice, R"(["a","b"])"},
    };
    for (auto const& testCase : cases)
    {
        std::string failReason;
        EXPECT_TRUE(decoder.prepareSlot(0, makeGuide(testCase.type, testCase.guide), failReason))
            << rt::guideTypeName(testCase.type) << ": " << failReason;
        EXPECT_TRUE(decoder.hasGrammar(0));
    }
}

//! A guide that cannot compile must fail only its own slot, leaving the rest of the batch to
//! run: compilation is the one input-driven step, so it is the one that must not be fatal.
TEST_F(GuidedDecoderTest, FailedCompilationIsIsolatedToItsSlot)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    EXPECT_FALSE(decoder.prepareSlot(1, makeGuide(rt::GuideType::kEbnf, "this is not valid ebnf ((("), failReason));
    EXPECT_FALSE(failReason.empty());

    EXPECT_TRUE(decoder.hasGrammar(0));
    EXPECT_FALSE(decoder.hasGrammar(1));
    EXPECT_TRUE(decoder.hasAnyGrammar());
}

//! The three-stage termination model: mid-rule, root complete but stop token not yet accepted,
//! and finally terminated. Filling a mask past the last stage aborts inside XGrammar, which is
//! why the runtime gates on isTerminated().
TEST_F(GuidedDecoderTest, TerminatesOnlyAfterAcceptingTheStopToken)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    // Spell out {"a":1 -- still mid-rule.
    for (int32_t token : {0, 2, 4, 2, 3, 5})
    {
        ASSERT_TRUE(decoder.advance(0, token)) << "token " << token;
        EXPECT_FALSE(decoder.isTerminated(0));
    }
    // Closing brace completes the root rule, but termination needs the stop token.
    ASSERT_TRUE(decoder.advance(0, 1));
    EXPECT_FALSE(decoder.isTerminated(0));

    ASSERT_TRUE(decoder.advance(0, kEOS_ID));
    EXPECT_TRUE(decoder.isTerminated(0));
}

TEST_F(GuidedDecoderTest, CompletedJsonFinishesWithoutWaitingForEos)
{
    // Given equivalent JSON-schema and EBNF guides that have not accepted EOS,
    // When the generated top-level JSON object becomes complete,
    // Then only the JSON response finishes with normal stop semantics.
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;
    ASSERT_TRUE(decoder.prepareSlot(
        1, makeGuide(rt::GuideType::kEbnf, R"(root ::= "{" "\"" "a" "\"" ":" "1" "}")"), failReason))
        << failReason;

    rt::DecodingInferenceContext context;
    context.initialize(/*batchSize=*/2, /*maxSeqLen=*/16, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    context.hasGuidedDecoding = true;
    context.finishedStates = {0, 0};
    context.currentGenerateLengths = {0, 0};
    context.guidedReasoningEnded = {1, 1};

    for (int32_t token : {0, 2, 4, 2, 3, 5, 1})
    {
        int32_t const outputIds[]{token, token};
        rt::advanceGuidedDecoding(decoder, context, outputIds, /*activeBatchSize=*/2);
        ++context.currentGenerateLengths[0];
        ++context.currentGenerateLengths[1];
    }

    EXPECT_EQ(context.finishedStates, (std::vector<int8_t>{1, 0}));
    EXPECT_EQ(context.slotStreams[0].terminalReason, rt::FinishReason::kEndId);
    EXPECT_EQ(context.slotStreams[1].terminalReason, rt::FinishReason::kNotFinished);
}

TEST_F(GuidedDecoderTest, CompleteValueThatCanGrowKeepsGenerating)
{
    // Given a top-level integer schema,
    // When `1` completes the value but more digits remain legal,
    // Then the response stays open and `12` is generated in full.
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonSchema, R"({"type":"integer"})"), failReason))
        << failReason;

    rt::DecodingInferenceContext context;
    context.initialize(/*batchSize=*/1, /*maxSeqLen=*/16, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    context.hasGuidedDecoding = true;
    context.finishedStates = {0};
    context.currentGenerateLengths = {0};
    context.guidedReasoningEnded = {1};

    for (int32_t token : {5, 6})
    {
        int32_t const outputIds[]{token};
        rt::advanceGuidedDecoding(decoder, context, outputIds, /*activeBatchSize=*/1);
        ++context.currentGenerateLengths[0];
        EXPECT_EQ(context.finishedStates[0], 0) << "after token " << token;
    }
    EXPECT_EQ(context.slotStreams[0].terminalReason, rt::FinishReason::kNotFinished);
}

TEST_F(GuidedDecoderTest, DeepIndentationStaysLegal)
{
    // Given a JSON-schema guide with a declared property,
    // When the key follows 17 whitespace characters (a newline plus 4-space indentation at depth 4),
    // Then the grammar accepts every token.
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int32_t> tokens{guidedFixture::kOPEN_BRACE};
    tokens.insert(tokens.end(), 17, guidedFixture::kSPACE);
    tokens.insert(tokens.end(),
        {guidedFixture::kQUOTE, guidedFixture::kA, guidedFixture::kQUOTE, guidedFixture::kCOLON, guidedFixture::kONE,
            guidedFixture::kCLOSE_BRACE});
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        EXPECT_TRUE(decoder.advance(0, tokens[i])) << "token " << i;
    }
    EXPECT_TRUE(decoder.hasCompletedJson(0));
}

//! The grammar must actually forbid things: a letter where a value belongs is rejected.
TEST_F(GuidedDecoderTest, RejectsTokensTheGrammarForbids)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    for (int32_t token : {0, 2, 4, 2, 3})
    {
        ASSERT_TRUE(decoder.advance(0, token)) << "token " << token;
    }
    EXPECT_FALSE(decoder.advance(0, 4)) << "'a' is not a legal start for an integer value";
}

//! Missing this reindexing is the failure mode where a constraint silently migrates onto a
//! different request; after an eviction slot 0 must carry what was slot 1's grammar.
TEST_F(GuidedDecoderTest, CompactSlotsMovesMatchersWithTheirRequests)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 3);

    std::string failReason;
    // Only slots 1 and 2 are constrained, and they are given different grammars so the test can
    // tell which matcher survived where.
    ASSERT_TRUE(decoder.prepareSlot(1, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a")"), failReason)) << failReason;
    ASSERT_TRUE(decoder.prepareSlot(2, makeGuide(rt::GuideType::kEbnf, R"(root ::= "b")"), failReason)) << failReason;
    ASSERT_FALSE(decoder.hasGrammar(0));

    // Evict old slot 0; old slot 1 -> new 0, old slot 2 -> new 1.
    decoder.compactSlots({-1, 0, 1});

    EXPECT_TRUE(decoder.hasGrammar(0));
    EXPECT_TRUE(decoder.hasGrammar(1));
    EXPECT_FALSE(decoder.hasGrammar(2));
    // New slot 0 is the "a"-only grammar; new slot 1 is the "b"-only one.
    EXPECT_TRUE(decoder.advance(0, 4)) << "new slot 0 should accept 'a'";
    EXPECT_FALSE(decoder.advance(1, 4)) << "new slot 1 should reject 'a'";
}

TEST_F(GuidedDecoderTest, ResetDropsEveryMatcher)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);
    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;
    ASSERT_TRUE(decoder.hasAnyGrammar());
    decoder.reset();
    EXPECT_FALSE(decoder.hasAnyGrammar());
    EXPECT_FALSE(decoder.hasGrammar(0));
}

//! Repeating a schema must hit the LRU rather than recompiling.
TEST_F(GuidedDecoderTest, RepeatedGuideReusesTheCompiledGrammar)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    auto const guide = makeGuide(
        rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})");
    ASSERT_TRUE(decoder.prepareSlot(0, guide, failReason)) << failReason;
    int64_t const cacheAfterFirst = decoder.cacheSizeBytes();
    ASSERT_GT(cacheAfterFirst, 0);

    ASSERT_TRUE(decoder.prepareSlot(1, guide, failReason)) << failReason;
    EXPECT_EQ(decoder.cacheSizeBytes(), cacheAfterFirst) << "an identical guide should not grow the cache";
}

//! The mask must mark exactly the legal continuations. At the very start of an object grammar
//! only `{` is legal, so the row has a single set bit.
TEST_F(GuidedDecoderTest, FillMasksMarksOnlyLegalTokens)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_TRUE(unsatisfiable.empty());

    // Apply the mask to a flat row and read back which entries survived.
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[0], 1.0F) << "'{' must be allowed at the start of an object";
    for (int32_t token = 1; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], kMaskedLogitValue) << "token " << token << " should be forbidden";
    }
}

//! ---------------------------------------------------------------------------------------------
//! Speculative verification: one mask per draft-chain node.
//! ---------------------------------------------------------------------------------------------

//! The schema `{"a": <integer>}` walks through four distinct grammar states, so the four verify
//! rows must carry four distinct masks. Filling them from one un-advanced matcher would repeat
//! the same mask, which is the whole failure mode this path exists to avoid.
TEST_F(GuidedDecoderTest, DraftChainGivesEachRowItsOwnMask)
{
    constexpr int32_t kRows = 4;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    // Node 0 is the already-committed token; nodes 1..3 spell `{`, `"`, `a`.
    std::vector<int32_t> const chain{/*root*/ 0, 0, 2, 4};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, chain.data(), /*parentIds=*/nullptr,
        /*validCounts=*/nullptr, /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto allowed = [&](int32_t row, int32_t token) { return out[row * kVOCAB_SIZE + token] == 1.0F; };

    EXPECT_TRUE(allowed(0, 0)) << "row 0 is the object start, so '{' must survive";
    EXPECT_FALSE(allowed(0, 4)) << "'a' cannot start the object";
    EXPECT_TRUE(allowed(1, 2)) << "after '{' a key must open with '\"'";
    EXPECT_FALSE(allowed(1, 0)) << "a second '{' is not a key";
    EXPECT_TRUE(allowed(2, 4)) << "the only declared property is 'a'";
    EXPECT_FALSE(allowed(2, 2)) << "the key name has to start before it can close";
    EXPECT_TRUE(allowed(3, 2)) << "after 'a' the key closes";
    EXPECT_FALSE(allowed(3, 4)) << "'aa' is not a declared property";

    // The point of the test: no two rows may be identical.
    for (int32_t lhs = 0; lhs < kRows; ++lhs)
    {
        for (int32_t rhs = lhs + 1; rhs < kRows; ++rhs)
        {
            bool same = true;
            for (int32_t token = 0; token < kVOCAB_SIZE && same; ++token)
            {
                same = allowed(lhs, token) == allowed(rhs, token);
            }
            EXPECT_FALSE(same) << "rows " << lhs << " and " << rhs << " carry the same mask";
        }
    }
}

//! A draft token the grammar refuses ends the walk. That node and every node under it are
//! unreachable, so their rows stay unmasked rather than being driven to the sentinel.
TEST_F(GuidedDecoderTest, DraftChainStopsAtTheFirstIllegalNode)
{
    constexpr int32_t kRows = 4;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    // `{` is legal, then `a` is not a key opener, so nodes 2 and 3 are dead.
    std::vector<int32_t> const chain{0, 0, 4, 0};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, chain.data(), /*parentIds=*/nullptr,
        /*validCounts=*/nullptr, /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty()) << "a dead draft node is normal, not a failed request";

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(out[0 * kVOCAB_SIZE + 1], kMaskedLogitValue) << "row 0 is masked";
    EXPECT_FLOAT_EQ(out[1 * kVOCAB_SIZE + 0], kMaskedLogitValue) << "row 1 is masked";
    for (int32_t row = 2; row < kRows; ++row)
    {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            EXPECT_FLOAT_EQ(out[row * kVOCAB_SIZE + token], 1.0F)
                << "row " << row << " is unreachable and must be left alone";
        }
    }
}

//! The walk is speculative: it must leave the matcher exactly where it found it, or the next
//! step starts from a grammar state the model never actually reached.
TEST_F(GuidedDecoderTest, DraftChainLeavesTheMatcherWhereItFoundIt)
{
    constexpr int32_t kRows = 4;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int32_t> const chain{0, 0, 2, 4};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;

    auto maskAfterOneWalk = [&]() {
        decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, chain.data(), /*parentIds=*/nullptr,
            /*validCounts=*/nullptr, /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
        rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
        copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
        decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> out(kRows * kVOCAB_SIZE);
        CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    };

    EXPECT_EQ(maskAfterOneWalk(), maskAfterOneWalk()) << "the second walk saw a different grammar state";
}

//! Speculative decoding commits a variable number of tokens per step, so the grammar advances
//! by that many at once and must land exactly where the committed text left it.
TEST_F(GuidedDecoderTest, AdvanceCommittedWalksEveryCommittedToken)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, /*maxRowsPerSlot=*/4);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int32_t> const committed{0, 2, 4}; // `{`, `"`, `a`
    int8_t reasoningEnded = 1;
    ASSERT_TRUE(decoder.advanceCommitted(0, committed.data(), static_cast<int32_t>(committed.size()), reasoningEnded));

    // After `{"a` the key can only close, so a second 'a' must be refused and '"' accepted.
    EXPECT_FALSE(decoder.advance(0, 4)) << "'aa' is not a declared property";
}

//! A token the grammar forbids mid-run fails the request rather than being skipped: the mask
//! should have made it impossible, so reaching it means the two sides disagree.
TEST_F(GuidedDecoderTest, AdvanceCommittedReportsARejectedToken)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, /*maxRowsPerSlot=*/4);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int32_t> const committed{0, 4}; // `{` then 'a', which cannot open a key
    int8_t reasoningEnded = 1;
    EXPECT_FALSE(decoder.advanceCommitted(0, committed.data(), static_cast<int32_t>(committed.size()), reasoningEnded));
}

//! Reasoning gate, the case a per-slot flag cannot express: the block closes partway through a
//! verify chain, so the rows before the separator are free and the rows after it are governed.
TEST_F(GuidedDecoderTest, ReasoningGateStartsAtTheSeparatorInsideAChain)
{
    constexpr int32_t kRows = 4;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    // Still reasoning on entry; node 2 closes the block, so row 2 is the first governed row.
    std::vector<int32_t> const chain{4, 4, kTHINK_END_ID, 0};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const stillReasoning{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, chain.data(), /*parentIds=*/nullptr,
        /*validCounts=*/nullptr, /*forcedFlags=*/nullptr, notSuppressed, stillReasoning, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t row = 0; row < 2; ++row)
    {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            EXPECT_FLOAT_EQ(out[row * kVOCAB_SIZE + token], 1.0F)
                << "row " << row << " is inside the reasoning block and must stay free";
        }
    }
    EXPECT_FLOAT_EQ(out[2 * kVOCAB_SIZE + 0], 1.0F) << "the separator's row opens the object";
    EXPECT_FLOAT_EQ(out[2 * kVOCAB_SIZE + 4], kMaskedLogitValue) << "'a' cannot start the object";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 2], 1.0F) << "after '{' a key opens";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 0], kMaskedLogitValue) << "a second '{' is not a key";
}

//! The separator is a delimiter, not constrained output: consuming it flips the flag but must
//! not advance the grammar, or the schema would have to start with `</think>`.
TEST_F(GuidedDecoderTest, AdvanceCommittedConsumesTheSeparatorWithoutFeedingIt)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, /*maxRowsPerSlot=*/4);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int32_t> const committed{4, 4, kTHINK_END_ID, 0}; // free text, separator, then `{`
    int8_t reasoningEnded = 0;
    ASSERT_TRUE(decoder.advanceCommitted(0, committed.data(), static_cast<int32_t>(committed.size()), reasoningEnded));
    EXPECT_EQ(reasoningEnded, 1) << "the separator must latch the flag";

    // Only `{` reached the grammar, so a key must open next.
    EXPECT_TRUE(decoder.advance(0, 2)) << "after '{' the key opens with '\"'";
}

//! choice has no XGrammar primitive behind it: it is lowered to an EBNF alternation, so the
//! test that matters is that only the first character of some alternative is legal at step 0.
TEST_F(GuidedDecoderTest, ChoiceAllowsOnlyTheFirstCharacterOfAnAlternative)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kChoice, R"(["a","b"])"), failReason)) << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        bool const allowed = (token == 4 || token == 8); // 'a' and 'b'
        EXPECT_FLOAT_EQ(result[token], allowed ? 1.0F : kMaskedLogitValue) << "token " << token;
    }
}

//! An alternative is embedded in the generated grammar as a literal, so any character that
//! terminates an EBNF literal must survive the round trip rather than truncate the branch.
TEST_F(GuidedDecoderTest, ChoiceEscapesQuotesInsideAnAlternative)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    std::string const choiceGuide = R"(["\"a\"","b"])";
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kChoice, choiceGuide), failReason)) << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        bool const allowed = (token == 2 || token == 8); // '"' and 'b'
        EXPECT_FLOAT_EQ(result[token], allowed ? 1.0F : kMaskedLogitValue) << "token " << token;
    }
}

//! The reasoning gate must open only on the thinking-end marker. `thinkingDone` also flips
//! when the first generated token is not a thinking marker, and that token is answer content:
//! opening on it would let it past the mask and out of the matcher's prefix, so the request
//! returns `<stray token>{...}` while the matcher believes it started from an empty prefix.
TEST_F(GuidedDecoderTest, ReasoningGateIgnoresTheFirstTokenHeuristic)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;

    rt::DecodingInferenceContext context;
    context.initialize(/*batchSize=*/1, /*maxSeqLen=*/4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    context.hasGuidedDecoding = true;
    context.enableThinking = true;
    context.finishedStates = {0};
    context.currentGenerateLengths = {1};
    // What the first-token heuristic does, and what it must not be allowed to imply here.
    context.thinkingDone = {1};
    context.guidedReasoningEnded = {0};

    auto applyAndRead = [&]() {
        rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
        copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 3.0F));
        rt::applyGuidedDecodingMask(decoder, context, logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> result(kVOCAB_SIZE);
        CUDA_CHECK(
            cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    };

    std::vector<float> const duringReasoning = applyAndRead();
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(duringReasoning[token], 3.0F) << "token " << token << " must be free while reasoning";
    }

    // The end marker is what opens the gate; `{` is then the only legal start of an object.
    context.guidedReasoningEnded = {1};
    std::vector<float> const afterReasoning = applyAndRead();
    EXPECT_FLOAT_EQ(afterReasoning[0], 3.0F) << "'{' must be allowed once the gate opens";
    for (int32_t token = 1; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(afterReasoning[token], kMaskedLogitValue) << "token " << token;
    }
}

//! A suppressed slot -- finished, or still inside its thinking block -- must be left alone even
//! though it carries a grammar. This is also what keeps a terminated matcher from being asked
//! for another mask, which XGrammar treats as a hard error.
TEST_F(GuidedDecoderTest, SuppressedSlotIsLeftUnconstrained)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kJsonObject, ""), failReason)) << failReason;

    std::vector<int8_t> const suppressed{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);

    rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 3.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[token], 3.0F) << "token " << token;
    }
}

//! A slot without a grammar shares the batch with a constrained one and must be untouched.
TEST_F(GuidedDecoderTest, MixedBatchLeavesUnconstrainedSlotsAlone)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int8_t> const suppressed{0, 0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/2, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);

    rt::Tensor logits({2, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(2 * kVOCAB_SIZE, 2.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/2, /*rowsPerSlot=*/1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> result(2 * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(result.data(), logits.rawPointer(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[0], 2.0F) << "'{' stays legal for the constrained slot";
    EXPECT_FLOAT_EQ(result[1], kMaskedLogitValue) << "'}' is illegal for the constrained slot";
    for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
    {
        EXPECT_FLOAT_EQ(result[kVOCAB_SIZE + token], 2.0F) << "unconstrained slot, token " << token;
    }
}

//! A grammar needing bytes the vocabulary cannot spell produces an all-zero row. Only we act
//! on this, because only we support a pruned base vocabulary where it is reachable.
TEST_F(GuidedDecoderTest, ReportsUnsatisfiableGrammarAsAnAllZeroRow)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    // 'z' exists in no token of this vocabulary, so nothing can ever satisfy the rule.
    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "zzz")"), failReason)) << failReason;

    std::vector<int8_t> const suppressed{0};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, suppressed, unsatisfiable, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    ASSERT_EQ(unsatisfiable.size(), 1U);
    EXPECT_EQ(unsatisfiable[0], 0);
}

//! ---------------------------------------------------------------------------------------------
//! Speculative verification over a branching draft tree.
//! ---------------------------------------------------------------------------------------------

//! `root ::= "a" ("1" "b" | "2" ",") "a"` forks after `a`, so nodes 2 and 3 are siblings standing
//! at the same grammar state but leading to different ones. Walking rows in index order instead
//! of depth-first would evaluate node 3 with node 2 still accepted, and `2` would be refused.
TEST_F(GuidedDecoderTest, DraftTreeGivesSiblingBranchesIndependentMasks)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    //  node:    0       1      2      3      4      5
    //  parent: -1       0      1      1      2      3
    //  token:  root    'a'    '1'    '2'    'b'    ','
    std::vector<int32_t> const tokens{0, 4, 5, 6, 8, 7};
    std::vector<int32_t> const parents{-1, 0, 1, 1, 2, 3};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(
        /*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(), /*forcedFlags=*/nullptr,
        notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto allowed = [&](int32_t row, int32_t token) { return out[row * kVOCAB_SIZE + token] == 1.0F; };

    EXPECT_TRUE(allowed(0, 4)) << "the grammar starts with 'a'";
    EXPECT_TRUE(allowed(1, 5)) << "after 'a' the fork admits '1'";
    EXPECT_TRUE(allowed(1, 6)) << "after 'a' the fork admits '2'";

    // The two forks must disagree. Sharing a mask here is what a sweep in row order produces.
    EXPECT_TRUE(allowed(2, 8)) << "the '1' branch continues with 'b'";
    EXPECT_FALSE(allowed(2, 7)) << "the '1' branch does not admit ','";
    EXPECT_TRUE(allowed(3, 7)) << "the '2' branch continues with ','";
    EXPECT_FALSE(allowed(3, 8)) << "the '2' branch does not admit 'b'";

    EXPECT_TRUE(allowed(4, 4)) << "both branches close on a trailing 'a'";
    EXPECT_TRUE(allowed(5, 4)) << "the sibling branch reached its own trailing 'a'";
}

//! A branch the grammar refuses takes its own subtree down and nothing else. Pruning by row
//! index instead of by subtree would silently kill the sibling that is still legal.
TEST_F(GuidedDecoderTest, DraftTreeDeadBranchLeavesItsSiblingAlone)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    // Node 2 drafts 'b', which cannot follow 'a'. Node 3 is its sibling and stays legal.
    std::vector<int32_t> const tokens{0, 4, 8, 6, 8, 7};
    std::vector<int32_t> const parents{-1, 0, 1, 1, 2, 3};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(
        /*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(), /*forcedFlags=*/nullptr,
        notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty()) << "a dead draft node is normal output, not a failed request";

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (out[row * kVOCAB_SIZE + token] != 1.0F)
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(rowUntouched(2)) << "the refused node is unreachable and must stay unmasked";
    EXPECT_TRUE(rowUntouched(4)) << "the refused node's child is unreachable too";
    EXPECT_FALSE(rowUntouched(3)) << "the sibling branch is still alive and must be masked";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 7], 1.0F) << "the sibling continues with ','";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 8], kMaskedLogitValue) << "the sibling does not admit 'b'";
    EXPECT_FALSE(rowUntouched(5)) << "the sibling's child is reachable and must be masked";
}

//! Every accepted node has to be rewound on the way back up, or the next sibling -- and the next
//! decode step -- start from a grammar state the model never reached.
TEST_F(GuidedDecoderTest, DraftTreeLeavesTheMatcherWhereItFoundIt)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    std::vector<int32_t> const tokens{0, 4, 5, 6, 8, 7};
    std::vector<int32_t> const parents{-1, 0, 1, 1, 2, 3};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;

    auto maskAfterOneWalk = [&]() {
        decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(),
            /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
        rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
        copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
        decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> out(kRows * kVOCAB_SIZE);
        CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    };

    EXPECT_EQ(maskAfterOneWalk(), maskAfterOneWalk()) << "the second walk saw a different grammar state";
}

//! The builder pads the block out to verifyTokenCount when it runs out of candidates. Padding carries
//! parent -1 and a stale token, so treating it as a node would both corrupt the walk and mask
//! rows the base model is not going to read.
TEST_F(GuidedDecoderTest, DraftTreeStopsAtTheValidNodeCount)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    // Only nodes 0..2 are real; 3..5 are padding whose parent is -1 and whose token is stale.
    std::vector<int32_t> const tokens{0, 4, 5, 4, 4, 4};
    std::vector<int32_t> const parents{-1, 0, 1, -1, -1, -1};
    std::vector<int32_t> const counts{3};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(
        /*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(), /*forcedFlags=*/nullptr,
        notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(out[0 * kVOCAB_SIZE + 8], kMaskedLogitValue) << "row 0 is a real node and is masked";
    EXPECT_FLOAT_EQ(out[2 * kVOCAB_SIZE + 8], 1.0F) << "row 2 is a real node continuing with 'b'";
    for (int32_t row = 3; row < kRows; ++row)
    {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            EXPECT_FLOAT_EQ(out[row * kVOCAB_SIZE + token], 1.0F)
                << "row " << row << " is padding and must be left alone";
        }
    }
}

//! D6 under a tree: reasoning state belongs to the path, not the slot. One branch crossing
//! `</think>` must not constrain a sibling that is still inside the thinking block -- which is
//! exactly what a single flag carried across the walk in node order would do.
TEST_F(GuidedDecoderTest, ReasoningGateDoesNotLeakAcrossSiblingBranches)
{
    constexpr int32_t kRows = 5;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    //  node:    0       1          2      3      4
    //  parent: -1       0          0      1      2
    //  token:  root  '</think>'   'a'    'a'    'b'
    // Node 1 closes the block; node 2 is its sibling and is still inside it.
    std::vector<int32_t> const tokens{0, kTHINK_END_ID, 4, 4, 8};
    std::vector<int32_t> const parents{-1, 0, 0, 1, 2};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int32_t> unsatisfiable;

    // Leave a mask behind in row 0 first. Rows are reused across steps and only their flags are
    // cleared, so a walk that consults a row it did not fill this step reads whatever the last
    // step wrote. Against a freshly zeroed buffer such a read denies every token and hides the
    // bug; against a dirty one it admits the sibling and constrains it.
    std::vector<int8_t> const reasoningEnded{1};
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(),
        /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
    ASSERT_TRUE(unsatisfiable.empty());

    std::vector<int8_t> const stillReasoning{0};
    decoder.fillMasksForDraftTree(
        /*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(), /*forcedFlags=*/nullptr,
        notSuppressed, stillReasoning, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (out[row * kVOCAB_SIZE + token] != 1.0F)
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(rowUntouched(0)) << "the block is still open at the root, so nothing is constrained";
    EXPECT_FALSE(rowUntouched(1)) << "the separator owns the first row the grammar governs";
    EXPECT_FLOAT_EQ(out[1 * kVOCAB_SIZE + 4], 1.0F) << "the grammar is still at its start state there";
    EXPECT_FALSE(rowUntouched(3)) << "past the separator the branch is governed by the grammar";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 5], 1.0F) << "after 'a' the fork admits '1'";
    EXPECT_FLOAT_EQ(out[3 * kVOCAB_SIZE + 4], kMaskedLogitValue) << "a second 'a' is not in the grammar";

    // The assertion this test exists for.
    EXPECT_TRUE(rowUntouched(2)) << "the sibling never saw the separator and must stay unconstrained";
    EXPECT_TRUE(rowUntouched(4)) << "the sibling's child is inside the thinking block as well";
}

//! EAGLE selects verify nodes by score alone, so a node can end up in the tree while its parent
//! did not. Such a node hangs off nothing: acceptance walks down from the root, so it is
//! unreachable and must be left unmasked rather than aborting the walk or attaching to the root.
TEST_F(GuidedDecoderTest, DraftTreeSkipsNodesWhoseParentMissedTheSelection)
{
    constexpr int32_t kRows = 4;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(
        decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" "b" | "2" ",") "a")"), failReason))
        << failReason;

    //  node:    0      1      2      3
    //  parent: -1      0     -1      1     <- node 2 is an orphan
    //  token:  root   'a'    'a'    '1'
    // The orphan carries a token the root's mask does admit, so attaching it to the root instead
    // of skipping it would mask its row -- without that, the grammar would reject it anyway and
    // the two behaviours would be indistinguishable.
    std::vector<int32_t> const tokens{0, 4, 4, 5};
    std::vector<int32_t> const parents{-1, 0, -1, 1};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(
        /*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(), /*forcedFlags=*/nullptr,
        notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (out[row * kVOCAB_SIZE + token] != 1.0F)
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_FALSE(rowUntouched(0)) << "the root is masked as usual";
    EXPECT_FALSE(rowUntouched(1)) << "a node with a real parent is masked as usual";
    EXPECT_TRUE(rowUntouched(2)) << "the orphan is unreachable and must stay unmasked";
    EXPECT_FALSE(rowUntouched(3)) << "the orphan must not take its sibling down with it";
}

// Given the grammar ("a" "1" | "2" ",") "b" ("a" | "2"), a verify tree whose root carries the grafted forced chain a, 1
//      beside the legal draft sibling 2, and bitmask rows left over from a walk of the same tree without flags
// When the masks are filled with the forced flags
// Then the root and the inner forced node get no mask row, the sibling branch is not visited, and the chain tail and
//      the draft nodes below it are masked from the grammar state the chain leads to
TEST_F(GuidedDecoderTest, DraftTreeSkipsTheMaskOfANodeWhoseChildIsForced)
{
    constexpr int32_t kRows = 7;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(
        0, makeGuide(rt::GuideType::kEbnf, R"(root ::= ("a" "1" | "2" ",") "b" ("a" | "2"))"), failReason))
        << failReason;

    //  node:    0      1      2      3      4      5      6
    //  parent: -1      0      1      2      0      4      3
    //  token:  root   'a'    '1'    'b'    '2'    ','    'a'
    //  forced:         x      x
    std::vector<int32_t> const tokens{0, 4, 5, 8, 6, 7, 4};
    std::vector<int32_t> const parents{-1, 0, 1, 2, 0, 4, 3};
    std::vector<int8_t> const forced{0, 1, 1, 0, 0, 0, 0};
    std::vector<int32_t> const counts{kRows};
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    // The walk without flags leaves the root's row admitting the sibling's '2', as a previous step would.
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(),
        /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, tokens.data(), parents.data(), counts.data(),
        forced.data(), notSuppressed, reasoningEnded, unsatisfiable, mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto allowed = [&](int32_t row, int32_t token) { return out[row * kVOCAB_SIZE + token] == 1.0F; };
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (!allowed(row, token))
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(rowUntouched(0)) << "the root's prediction is replaced by its forced child";
    EXPECT_TRUE(rowUntouched(1)) << "an inner chain node's prediction is replaced too";
    EXPECT_TRUE(rowUntouched(4)) << "the forced child's sibling is unreachable";
    EXPECT_TRUE(rowUntouched(5)) << "and so is the sibling's subtree";
    EXPECT_FALSE(rowUntouched(2)) << "the chain tail is masked";
    EXPECT_TRUE(allowed(2, 8)) << "after a, 1 the grammar continues with b";
    EXPECT_FALSE(allowed(2, 4)) << "the tail's mask is filled after the chain, not at the root state";
    EXPECT_TRUE(allowed(3, 4)) << "the draft node below the tail is gated and masked as usual";
    EXPECT_TRUE(allowed(3, 6));
    EXPECT_FALSE(allowed(3, 8));
    EXPECT_TRUE(allowed(6, kEOS_ID)) << "the grammar completes below it";
    EXPECT_FALSE(allowed(6, 4));
}

// Given the grammar ("a" "1" | "2" ",") "b" ("a" | "2"), a verify chain root, a, 1, b, {, { whose 1 and b are grafted
//      forced tokens and whose last two rows are padding, captured as a chain with its forced flags, and bitmask rows
//      left over from a walk of the chain root, 2, ",", b, a, { that leave a's row refusing 1
// When the masks are filled from the captured draft
// Then a's row and 1's row get no mask, 1 is accepted without a's row gating it, so b's row is masked from the state
//      after a, 1, b, and the padding rows past the { that state refuses stay unmasked
TEST_F(GuidedDecoderTest, DraftChainSkipsTheMaskOfANodeWhoseChildIsForced)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(
        0, makeGuide(rt::GuideType::kEbnf, R"(root ::= ("a" "1" | "2" ",") "b" ("a" | "2"))"), failReason))
        << failReason;

    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    std::vector<int32_t> const previousChain{0, 6, 7, 8, 4, 0};
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, previousChain.data(), /*parentIds=*/nullptr,
        /*validCounts=*/nullptr, /*forcedFlags=*/nullptr, notSuppressed, reasoningEnded, unsatisfiable, mStream);

    //  node:    0      1      2      3      4      5
    //  token:  root   'a'    '1'    'b'    '{'    '{'
    //  forced:                x      x
    rt::Tensor tokens({1, kRows}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "tokens");
    rt::Tensor forced({1, kRows}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "forced");
    copyHostToDevice<int32_t>(tokens, {0, 4, 5, 8, 0, 0});
    copyHostToDevice<int8_t>(forced, {0, 0, 1, 1, 0, 0});
    decoder.captureDraft(
        {tokens, nullptr, nullptr, nullptr, 1, kRows, kRows, rt::SpecVerifyLayout::kChain, false, &forced}, mStream);
    decoder.waitForDraftTopology();
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, decoder.hostDraftTokens(), decoder.hostDraftParentIds(),
        decoder.hostDraftValidCounts(), decoder.hostDraftForcedFlags(), notSuppressed, reasoningEnded, unsatisfiable,
        mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto allowed = [&](int32_t row, int32_t token) { return out[row * kVOCAB_SIZE + token] == 1.0F; };
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (!allowed(row, token))
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(rowUntouched(1)) << "a's prediction is replaced by its forced child";
    EXPECT_TRUE(rowUntouched(2)) << "so is 1's";
    EXPECT_TRUE(allowed(3, 4)) << "after a, 1, b the grammar continues with a or 2";
    EXPECT_TRUE(allowed(3, 6));
    EXPECT_FALSE(allowed(3, 8));
    EXPECT_TRUE(rowUntouched(4)) << "the grammar refuses '{' after a, 1, b";
    EXPECT_TRUE(rowUntouched(5));
}

// Given the grammar "a" ("1" | "2") ("{" | "b") ("{" | "b"), and a verify chain root, a, 1 followed by three padding
//      rows carrying the padding id {, which the grammar would accept after a, 1, captured as a chain with a node count
//      of three
// When the masks are filled from the captured draft
// Then the last node's row is masked and the padding rows stay unmasked
TEST_F(GuidedDecoderTest, DraftChainStopsAtTheCapturedNodeCount)
{
    constexpr int32_t kRows = 6;
    rt::GuidedDecoder decoder;
    initDecoder(decoder, /*maxBatchSize=*/1, kRows);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(
        0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" ("1" | "2") ("{" | "b") ("{" | "b"))"), failReason))
        << failReason;

    rt::Tensor tokens({1, kRows}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "tokens");
    rt::Tensor counts({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "counts");
    copyHostToDevice<int32_t>(tokens, {0, 4, 5, 0, 0, 0});
    copyHostToDevice<int32_t>(counts, {3});
    decoder.captureDraft({tokens, &counts, nullptr, nullptr, 1, kRows, kRows, rt::SpecVerifyLayout::kChain}, mStream);
    decoder.waitForDraftTopology();
    std::vector<int8_t> const notSuppressed{0};
    std::vector<int8_t> const reasoningEnded{1};
    std::vector<int32_t> unsatisfiable;
    decoder.fillMasksForDraftTree(/*activeBatchSize=*/1, kRows, decoder.hostDraftTokens(), decoder.hostDraftParentIds(),
        decoder.hostDraftValidCounts(), decoder.hostDraftForcedFlags(), notSuppressed, reasoningEnded, unsatisfiable,
        mStream);
    EXPECT_TRUE(unsatisfiable.empty());

    rt::Tensor logits({1, kRows, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
    copyHostToDevice<float>(logits, std::vector<float>(kRows * kVOCAB_SIZE, 1.0F));
    decoder.applyMask(logits, /*activeBatchSize=*/1, kRows, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<float> out(kRows * kVOCAB_SIZE);
    CUDA_CHECK(cudaMemcpy(out.data(), logits.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto allowed = [&](int32_t row, int32_t token) { return out[row * kVOCAB_SIZE + token] == 1.0F; };
    auto rowUntouched = [&](int32_t row) {
        for (int32_t token = 0; token < kVOCAB_SIZE; ++token)
        {
            if (!allowed(row, token))
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(allowed(2, 0)) << "after a, 1 the grammar continues with { or b";
    EXPECT_TRUE(allowed(2, 8));
    EXPECT_FALSE(allowed(2, 4));
    for (int32_t row = 3; row < kRows; ++row)
    {
        EXPECT_TRUE(rowUntouched(row)) << "row " << row << " is padding";
    }
}

//! ---------------------------------------------------------------------------------------------
//! Jump-forward support: the text a grammar forces, and the matcher moves the planner relies on.
//! ---------------------------------------------------------------------------------------------

//! Given a grammar, when the forced continuation is read, then it is the text every legal path
//! shares and stops where the paths diverge; a JSON schema compiled with free whitespace diverges
//! wherever whitespace may appear, so its forced spans are the literals in between; a slot with no
//! grammar, a completed root rule, or a terminated matcher reads empty, and nothing throws.
TEST_F(GuidedDecoderTest, ForcedContinuationIsTheTextEveryLegalPathShares)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 3);
    EXPECT_EQ(decoder.forcedContinuation(1), "") << "no grammar";

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;
    EXPECT_EQ(decoder.forcedContinuation(0), "{");
    ASSERT_TRUE(decoder.advance(0, 0));
    EXPECT_EQ(decoder.forcedContinuation(0), "") << "whitespace or the key may follow '{'";
    ASSERT_TRUE(decoder.advance(0, 2));
    EXPECT_EQ(decoder.forcedContinuation(0), "a\"") << "the only key is 'a'";

    ASSERT_TRUE(decoder.prepareSlot(1, makeGuide(rt::GuideType::kRegex, "ab+c"), failReason)) << failReason;
    EXPECT_EQ(decoder.forcedContinuation(1), "ab") << "after one 'b' the regex may repeat it or move on";

    ASSERT_TRUE(decoder.prepareSlot(2, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" "1")"), failReason))
        << failReason;
    EXPECT_EQ(decoder.forcedContinuation(2), "a1");
    ASSERT_TRUE(decoder.advance(2, 4));
    ASSERT_TRUE(decoder.advance(2, 5));
    EXPECT_EQ(decoder.forcedContinuation(2), "") << "the root rule is complete";
    ASSERT_TRUE(decoder.advance(2, kEOS_ID));
    ASSERT_TRUE(decoder.isTerminated(2));
    EXPECT_EQ(decoder.forcedContinuation(2), "") << "terminated";
}

//! Given forced tokens the grammar only partly admits, when they are accepted, then the count stops
//! at the first rejected token, the matcher stands after the accepted ones, and the slot keeps its
//! grammar.
TEST_F(GuidedDecoderTest, AcceptForcedStopsAtTheFirstRejectedToken)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" "1" "b")"), failReason))
        << failReason;

    std::vector<int32_t> const wrong{4, 8, 5}; // 'a', then 'b' where '1' belongs
    EXPECT_EQ(decoder.acceptForced(0, wrong.data(), static_cast<int32_t>(wrong.size())), 1);
    EXPECT_TRUE(decoder.hasGrammar(0));
    EXPECT_EQ(decoder.forcedContinuation(0), "1b") << "the matcher stands after 'a'";

    std::vector<int32_t> const rest{5, 8};
    EXPECT_EQ(decoder.acceptForced(0, rest.data(), static_cast<int32_t>(rest.size())), 2);
    EXPECT_EQ(decoder.forcedContinuation(0), "") << "the rule is complete";
    EXPECT_EQ(decoder.acceptForced(0, rest.data(), 0), 0);
}

//! Given a matcher advanced past a token, when it is rolled back, then the next mask is the one from
//! before the token, and a rollback of zero changes nothing.
TEST_F(GuidedDecoderTest, RollbackRestoresTheMask)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0,
        makeGuide(
            rt::GuideType::kJsonSchema, R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"),
        failReason))
        << failReason;

    std::vector<int8_t> const notSuppressed{0};
    std::vector<int32_t> unsatisfiable;
    auto maskNow = [&]() {
        decoder.fillMasks(/*activeBatchSize=*/1, /*rowsPerSlot=*/1, notSuppressed, unsatisfiable, mStream);
        rt::Tensor logits({1, kVOCAB_SIZE}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "logits");
        copyHostToDevice<float>(logits, std::vector<float>(kVOCAB_SIZE, 1.0F));
        decoder.applyMask(logits, /*activeBatchSize=*/1, /*rowsPerSlot=*/1, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return copyDeviceToHost<float>(logits);
    };

    std::vector<float> const atStart = maskNow();
    ASSERT_TRUE(decoder.advance(0, 0));
    std::vector<float> const afterBrace = maskNow();
    ASSERT_NE(atStart, afterBrace);

    decoder.rollback(0, 0);
    EXPECT_EQ(maskNow(), afterBrace);
    decoder.rollback(0, 1);
    EXPECT_EQ(maskNow(), atStart);
}

//! Given a grammar, when a text is tested for completing it, then the answer says whether the root
//! rule ends exactly there, rejected text answers no, and the matcher is where it was afterwards.
TEST_F(GuidedDecoderTest, CompletesGrammarReportsTheEndOfTheRootRuleAndRestoresTheMatcher)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 2);

    std::string failReason;
    ASSERT_TRUE(decoder.prepareSlot(0, makeGuide(rt::GuideType::kEbnf, R"(root ::= "a" "1")"), failReason))
        << failReason;
    EXPECT_TRUE(decoder.completesGrammar(0, "a1"));
    EXPECT_FALSE(decoder.completesGrammar(0, "a")) << "mid-rule";
    EXPECT_FALSE(decoder.completesGrammar(0, "ab")) << "rejected text";
    EXPECT_FALSE(decoder.completesGrammar(1, "a1")) << "no grammar";
    EXPECT_EQ(decoder.forcedContinuation(0), "a1") << "the checks must leave the matcher at the start";

    ASSERT_TRUE(decoder.advance(0, 4));
    EXPECT_TRUE(decoder.completesGrammar(0, "1"));
    EXPECT_FALSE(decoder.completesGrammar(0, ""));
    ASSERT_TRUE(decoder.advance(0, 5));
    EXPECT_TRUE(decoder.completesGrammar(0, "")) << "already complete";
}

//! Given the output vocabulary, when a token is asked whether another token extends its piece, then
//! a piece some longer token starts with is extendable, one nothing extends is not, and an unknown
//! id counts as extendable.
TEST_F(GuidedDecoderTest, IsExtendableFollowsTheVocabulary)
{
    rt::GuidedDecoder decoder;
    initDecoder(decoder, 1);
    EXPECT_TRUE(decoder.isExtendable(9)) << "' ' starts ' \"'";
    EXPECT_TRUE(decoder.isExtendable(2)) << "'\"' starts '\",' and '\":'";
    EXPECT_TRUE(decoder.isExtendable(15)) << "the two-byte prefix starts the full character";
    EXPECT_FALSE(decoder.isExtendable(14)) << "nothing starts with '\":'";
    EXPECT_FALSE(decoder.isExtendable(8)) << "nothing starts with 'b'";
    EXPECT_TRUE(decoder.isExtendable(kVOCAB_SIZE));
    EXPECT_TRUE(decoder.isExtendable(-1));
}

} // namespace
