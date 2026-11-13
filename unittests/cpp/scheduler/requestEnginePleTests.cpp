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
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "scheduler/requestEngine.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{
namespace
{
using namespace std::chrono_literals;

LLMGenerationRequest makePleRequest(std::string const& prompt)
{
    LLMGenerationRequest request{};
    request.temperature = 0.0F;
    request.topP = 1.0F;
    request.topK = 1;
    request.maxGenerateLength = 64;
    request.requests.resize(1);
    Message message;
    message.role = "user";
    Message::MessageContent content;
    content.type = "text";
    content.content = prompt;
    message.contents.push_back(std::move(content));
    request.requests.front().messages.push_back(std::move(message));
    return request;
}

bool waitForToken(RequestHandle& handle)
{
    auto const deadline = std::chrono::steady_clock::now() + 120s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        auto chunk = handle.stream().waitPop(100ms);
        if (chunk && !chunk->tokenIds.empty())
        {
            return true;
        }
        if (handle.ready() || (chunk && chunk->finished))
        {
            return false;
        }
    }
    return false;
}

bool waitForCompletion(RequestHandle& handle)
{
    auto const deadline = std::chrono::steady_clock::now() + 120s;
    while (!handle.ready() && std::chrono::steady_clock::now() < deadline)
    {
        handle.stream().waitPop(100ms);
    }
    return handle.ready();
}

class RequestEnginePleTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        char const* directory = std::getenv("EDGELLM_PLE_IFB_ENGINE_DIR");
        if (directory == nullptr)
        {
            GTEST_SKIP() << "Set EDGELLM_PLE_IFB_ENGINE_DIR to a Gemma pooled-PLE engine built with batch size >= 2.";
        }
        mEngineDir = directory;
        ASSERT_TRUE(std::filesystem::is_directory(mEngineDir));
        mPlugin = loadEdgellmPluginLib();
        ASSERT_NE(mPlugin, nullptr);
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        }
    }

    void exercise(PleRuntimeConfig const& policy, std::vector<std::vector<int32_t>>& tokens)
    {
        // Context reuse selects full SWA storage, which supports seated admission.
        ContextCacheConfig contextConfig{};
        contextConfig.enabled = true;
        auto deploymentPolicy = policy;
        if (char const* file = std::getenv("EDGELLM_PLE_IFB_FILE"))
        {
            deploymentPolicy.filePath = file;
        }
        auto runtime = std::make_unique<LLMInferenceRuntime>(mEngineDir, "",
            std::unordered_map<std::string, std::string>{}, mStream, contextConfig, "", std::nullopt, deploymentPolicy);
        ASSERT_TRUE(runtime->supportsSeatedAdmission());
        ASSERT_GE(runtime->maxBatchSize(), 2);
        ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));
        EngineConfig config{};
        config.maxBatchSize = 2;
        RequestEngine engine(std::move(runtime), mStream, config);
        auto first = engine.submit(makePleRequest("List the integers from 1 to 200, one per line. Do not stop early."));
        ASSERT_TRUE(waitForToken(first));
        auto joiner = engine.submit(
            makePleRequest("Write a detailed explanation of how rain forms, with ten numbered sections."));
        ASSERT_TRUE(waitForToken(joiner));
        ASSERT_FALSE(first.ready());
        first.cancel();
        auto replacement = engine.submit(
            makePleRequest("Write a detailed explanation of how trees grow, with ten numbered sections."));
        ASSERT_TRUE(waitForToken(replacement));
        ASSERT_TRUE(waitForCompletion(joiner));
        ASSERT_TRUE(waitForCompletion(replacement));
        auto const joinedResponse = joiner.get();
        auto const replacementResponse = replacement.get();
        engine.shutdown(ShutdownMode::kDrain);
        EXPECT_THROW(first.get(), std::runtime_error);
        auto const metrics = engine.metrics();
        EXPECT_GE(metrics.admittedMidFlight, 2U);
        EXPECT_EQ(metrics.cancelled, 1U);
        EXPECT_EQ(metrics.completed, 2U);
        EXPECT_EQ(metrics.failed, 0U);
        ASSERT_EQ(joinedResponse.outputIds.size(), 1U);
        ASSERT_EQ(replacementResponse.outputIds.size(), 1U);
        ASSERT_FALSE(joinedResponse.outputIds.front().empty());
        ASSERT_FALSE(replacementResponse.outputIds.front().empty());
        tokens = {joinedResponse.outputIds.front(), replacementResponse.outputIds.front()};
        std::string const name
            = policy.residency == PleResidency::kRESIDENT ? "resident" : "cached_" + std::to_string(policy.cacheSlots);
        nlohmann::json const result{{"policy", name}, {"admitted_midflight", metrics.admittedMidFlight},
            {"cancelled", metrics.cancelled}, {"completed", metrics.completed}, {"output_ids", tokens}};
        RecordProperty(name, result.dump());
        std::cout << "PLE_IFB_RESULT " << result.dump() << '\n';
    }

    std::string mEngineDir;
    std::unique_ptr<void, DlDeleter> mPlugin;
    cudaStream_t mStream{};
};

TEST_F(RequestEnginePleTest, ResidentAndCachesPreserveTokensAcrossAdmissionAndCompaction)
{
    std::vector<std::vector<int32_t>> reference;
    ASSERT_NO_FATAL_FAILURE(exercise({PleResidency::kRESIDENT, 0}, reference));
    for (int32_t const slots : {513, 4097})
    {
        SCOPED_TRACE("cache_slots=" + std::to_string(slots));
        std::vector<std::vector<int32_t>> actual;
        ASSERT_NO_FATAL_FAILURE(exercise({PleResidency::kCACHED, slots}, actual));
        EXPECT_EQ(actual, reference);
    }
}

} // namespace
} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
