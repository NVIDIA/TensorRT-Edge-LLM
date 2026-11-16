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

// The decode-only compact step (phase 8 with Np = 0, on the decode profile) on a real engine whose plugins run phase 8
// on this GPU (for example Qwen3-0.6B or Qwen3.5-0.8B built with maxBatchSize >= 4 and maxInputLen >= 512), named by
// EDGELLM_COMPACT_ENGINE_DIR; skipped without it.

#include "compactSessionDriver.h"

#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace
{

LLMGenerationRequest makeRequest(std::string const& text)
{
    LLMGenerationRequest request{};
    request.temperature = 1.0F;
    request.topP = 1.0F;
    request.topK = 1;
    request.maxGenerateLength = 12;
    request.ignoreEos = true;
    request.numLogprobs = 2;
    request.requests.resize(1);
    request.requests.front().messages.push_back(Message{"user", {Message::MessageContent{"text", text}}});
    return request;
}

//! A prompt of several hundred tokens: its first chunk ends on a page boundary before its end.
std::string longPrompt()
{
    std::string text;
    for (int32_t index = 0; index < 12; ++index)
    {
        text += "Rivers carve their valleys over thousands of years, carry sand and silt downstream, and build wide "
                "plains where the current slows near the sea. ";
    }
    return text + "Summarize the passage in one sentence.";
}

//! Empty when @p run matches @p reference under the trace checker's tie rule: a greedy divergence is accepted only
//! where the reference's top-2 logprob gap is below 0.1 and the run's token is one of the reference's top two.
std::string tieRuleViolation(LLMGenerationResponse const& run, LLMGenerationResponse const& reference)
{
    std::vector<int32_t> const& tokens = run.outputIds.front();
    std::vector<int32_t> const& expected = reference.outputIds.front();
    if (tokens == expected)
    {
        return {};
    }
    size_t const at
        = std::mismatch(tokens.begin(), tokens.end(), expected.begin(), expected.end()).first - tokens.begin();
    std::vector<LogprobEntry> const& top = reference.logprobs.front().at(at);
    float const gap = top.at(0).logprob - top.at(1).logprob;
    bool const inTop2 = tokens[at] == top[0].tokenId || tokens[at] == top[1].tokenId;
    return gap < 0.1F && inTop2 ? std::string{}
                                : "diverges at " + std::to_string(at) + ", reference top-2 gap " + std::to_string(gap);
}

class CompactSessionEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        char const* directory = std::getenv("EDGELLM_COMPACT_ENGINE_DIR");
        if (directory == nullptr)
        {
            GTEST_SKIP() << "Set EDGELLM_COMPACT_ENGINE_DIR to an engine whose plugins run phase 8 on this GPU.";
        }
        ASSERT_TRUE(std::filesystem::is_directory(directory)) << directory;
        mPlugin = loadEdgellmPluginLib();
        ASSERT_NE(mPlugin, nullptr);
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mRuntime = std::make_unique<LLMInferenceRuntime>(
            directory, "", std::unordered_map<std::string, std::string>{}, mStream);
        ASSERT_GE(mRuntime->maxBatchSize(), 4);
        ASSERT_TRUE(mRuntime->captureDecodingCUDAGraph(mStream));
    }

    void TearDown() override
    {
        mDriver.reset();
        mSession.reset();
        mRuntime.reset();
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        }
    }

    //! What the blocking path generates for @p text alone, before any session opens.
    LLMGenerationResponse reference(std::string const& text)
    {
        LLMGenerationResponse response;
        EXPECT_TRUE(mRuntime->handleRequest(makeRequest(text), response, mStream));
        return response;
    }

    void openCompactSession()
    {
        mSession = mRuntime->openSession(mStream, SessionOptions{/*allowCompactExecution=*/true});
        ASSERT_NE(mSession, nullptr);
        ASSERT_TRUE(mSession->capabilities().compact) << mSession->capabilities().reason;
        mDriver = std::make_unique<CompactDriver>(*mSession);
    }

    std::unique_ptr<void, DlDeleter> mPlugin;
    cudaStream_t mStream{};
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    std::unique_ptr<RuntimeSession> mSession;
    std::unique_ptr<CompactDriver> mDriver;
};

// Given a compact session on a real engine with A and D decoding, B's long prompt half prefilled (its first chunk ran
//       beside A's and D's decode rows) and C's short prompt admitted with a cancel token
// When C's cancel token is set and the step planned as C's final chunk beside A's decode row runs, B's prompt still
//      pending and D left out as a paused resident is, and the session then runs B's final chunk beside A's and D's
//      decode rows and decodes until no one is left
// Then that step runs as one compact step of A's decode row alone, C ends cancelled, the decode steps after the compact
//      steps replay the decode CUDA graphs captured before the session opened, and A, B and D generate what the
//      blocking path generates for each alone (diverging, if at all, only at a reference top-2 near-tie)
TEST_F(CompactSessionEngineTest, CancelledSoleChunkLeavesADecodeOnlyCompactStep)
{
    std::string const promptA{"Name the planets of the solar system in order."};
    std::string const promptB = longPrompt();
    std::string const promptC{"Name three colors."};
    std::string const promptD{"Describe how a lighthouse helps ships at night."};
    LLMGenerationResponse const referenceA = reference(promptA);
    LLMGenerationResponse const referenceB = reference(promptB);
    LLMGenerationResponse const referenceD = reference(promptD);
    ASSERT_NO_FATAL_FAILURE(openCompactSession());
    CompactDriver& driver = *mDriver;

    ResidentRef const a = driver.admit(makeRequest(promptA), 1);
    ResidentRef const d = driver.admit(makeRequest(promptD), 4);
    ASSERT_TRUE(driver.run({driver.rest(a), driver.rest(d)}).ok);
    ResidentRef const b = driver.admit(makeRequest(promptB), 2);
    auto const cancelC = std::make_shared<std::atomic<bool>>(false);
    ResidentRef const c = driver.admit(makeRequest(promptC), 3, cancelC);
    int32_t const page = mSession->capabilities().kvPageSize;
    ASSERT_GT(driver.promptLength(b), page) << "B's prompt must take more than one chunk";
    ASSERT_TRUE(driver.run({driver.chunk(b, page), driver.decode(a), driver.decode(d)}).ok);

    cancelC->store(true);
    StepResult const step = driver.run({driver.rest(c), driver.decode(a)});
    ASSERT_TRUE(step.ok) << "the decode-only compact step failed";
    EXPECT_TRUE(step.compact) << "the remaining decode rows did not run as a compact step";
    ASSERT_EQ(step.execution.size(), 1U);
    EXPECT_EQ(step.execution.front().identity.resident, a);
    EXPECT_EQ(step.execution.front().work, SequenceWork::kDecode);
    ASSERT_EQ(driver.outcomes().count(3), 1U) << "the cancelled prompt was not evicted by the step";
    EXPECT_EQ(driver.outcomes().at(3), FinishReason::kCancelled);

    ASSERT_TRUE(driver.run({driver.rest(b), driver.decode(a), driver.decode(d)}).ok);
    bool decodeGraphReplayed = false;
    while (!mSession->idle())
    {
        StepResult const decodeStep = driver.run(driver.decodeAll());
        ASSERT_TRUE(decodeStep.ok);
        decodeGraphReplayed = decodeGraphReplayed || decodeStep.graphLaunched;
    }
    EXPECT_TRUE(decodeGraphReplayed) << "no decode step after the compact steps replayed its captured CUDA graph";
    for (auto const& [requestId, expected] : {std::pair{RequestId{1}, &referenceA},
             std::pair{RequestId{2}, &referenceB}, std::pair{RequestId{4}, &referenceD}})
    {
        ASSERT_EQ(driver.outcomes().at(requestId), FinishReason::kLength) << "request " << requestId;
        std::string const violation = tieRuleViolation(driver.responses().at(requestId), *expected);
        EXPECT_TRUE(violation.empty()) << "request " << requestId << ": " << violation;
    }
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
