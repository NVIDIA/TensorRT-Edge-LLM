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

// An llm.backbone in the one-stage text graph, driven by MultiStageRuntime over a fake engine the way
// multistage_inference drives it.

#include "fakeEngineExecutor.h"
#include "substituteEngine.h"

#include "runtime/llmRunner.h"
#include "runtime/multiStage/multiStageRuntime.h"
#include "runtime/state/sharedResources.h"

#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;
namespace ms = rt::multiStage;

namespace
{

constexpr int64_t kMaxGenerateLength{24};
//! The tiny deployment's KV capacity, which the backbone stage is sized to.
constexpr int64_t kMaxSeq{64};

std::unique_ptr<rt::LLMInferenceRuntime> makeFakeRuntime(rt::DeploymentOptions const& deployment, cudaStream_t stream)
{
    return std::make_unique<rt::LLMInferenceRuntime>(
        makeVanillaArtifacts(deployment.engineDir, makeFakeEngine(), stream), deployment, stream);
}

//! The shipped llm.backbone declaration, built over the fake engine.
ms::StageRegistry const& testRunners()
{
    static ms::StageRegistry const sRunners{
        {rt::llmRunnerDeclaration(), [](ms::StageInitContext const& init) -> std::unique_ptr<ms::IStage> {
             return std::make_unique<rt::LlmRunner>(init, &makeFakeRuntime);
         }}};
    return sRunners;
}

class LlmRunnerTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("llmRunnerTests", makeTinyVanillaConfig());
    }

    //! The graph tensorrt-edgellm-build writes for a text-only build.
    void load(ms::MultiStageRuntime& runtime)
    {
        auto const input = [](char const* name, char const* from, char const* onMissing) {
            return Json{{"name", name}, {"from", from}, {"on_missing", onMissing}};
        };
        Json const text{{"id", "text"}, {"kind", "text"}, {"transport", "chunk_stream"}, {"terminal", true}};
        Json const config{{"schema_version", ms::MultiStageConfig::kSchemaVersion}, {"label", "backbone"},
            {"stages",
                Json::array({{{"id", "llm"}, {"runner", "llm.backbone"}, {"artifact", "."}, {"kind", "autoregressive"},
                    {"max_batch", kMaxBatchSize}, {"max_seq", kMaxSeq},
                    {"config", {{"max_generate_length", kMaxGenerateLength}}},
                    {"inputs",
                        Json::array(
                            {input("prompt", "request:messages", "fail"), input("sampling", "request:sampling", "skip"),
                                input("request", "request:llm_request", "skip")})},
                    {"emits", Json::array({text})}}})},
            {"terminals", Json::array({"llm:text"})}};
        std::ofstream(mModelDir / "multistage_config.json") << config.dump(2);
        runtime.load(mModelDir.string(), mShared, &testRunners());
    }

    //! A greedy request with its own generate budget, bound as multistage_inference binds a parsed request and
    //! borrowed by the runtime until take().
    ms::PortBindings payload(std::string const& prompt, int64_t maxGenerateLength)
    {
        mPrompts.push_back(prompt);
        mRequests.push_back(makeGreedyRequest(prompt, maxGenerateLength));
        ms::PortBindings bindings;
        bindings.bind("request:llm_request",
            ms::PortValue{&mRequests.back(), {1}, ms::PortKind::kRequest,
                static_cast<int32_t>(sizeof(rt::LLMGenerationRequest)), /*complete=*/true});
        bindings.bind("request:messages",
            ms::PortValue{mPrompts.back().data(), {static_cast<int64_t>(mPrompts.back().size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
        return bindings;
    }

    //! The tokens each request's text channel receives when the requests are served as multistage_inference serves
    //! a request file: as many as one cohort seats, and each one it refuses in a later cohort.
    std::vector<std::vector<int32_t>> serve(std::vector<std::pair<std::string, int64_t>> const& asks)
    {
        ms::MultiStageRuntime runtime;
        load(runtime);
        std::vector<std::vector<int32_t>> replies(asks.size());
        std::deque<size_t> pending;
        for (size_t index = 0; index < asks.size(); ++index)
        {
            pending.push_back(index);
        }
        while (!pending.empty())
        {
            std::vector<size_t> seated;
            std::vector<std::shared_ptr<rt::StreamChannel>> channels;
            std::deque<size_t> refused;
            for (auto const index : pending)
            {
                ms::Sinks sinks;
                sinks.text = rt::StreamChannel::create();
                std::string error;
                if (runtime.submit(payload(asks[index].first, asks[index].second), sinks, error) != nullptr)
                {
                    seated.push_back(index);
                    channels.push_back(sinks.text);
                }
                else
                {
                    refused.push_back(index);
                }
            }
            if (seated.empty())
            {
                ADD_FAILURE() << "a cohort of its own refused request " << pending.front();
                break;
            }
            auto const responses = runtime.run();
            EXPECT_EQ(responses.size(), seated.size());
            for (size_t member = 0; member < seated.size(); ++member)
            {
                EXPECT_TRUE(member < responses.size() && responses[member].ok);
                auto& reply = replies[seated[member]];
                while (auto chunk = channels[member]->tryPop())
                {
                    reply.insert(reply.end(), chunk->tokenIds.begin(), chunk->tokenIds.end());
                }
            }
            runtime.take();
            pending = std::move(refused);
        }
        return replies;
    }

    rt::SharedResources mShared;
    std::deque<std::string> mPrompts;
    std::deque<rt::LLMGenerationRequest> mRequests;
};

} // namespace

// Given the one-stage text graph over a fake engine that never decodes EOS, and two greedy requests with different
//       prompts and different max_generate_length
// When they are served as multistage_inference serves a request file
// Then each reply holds exactly as many tokens as its own max_generate_length
TEST_F(LlmRunnerTest, EachRequestStopsAtItsOwnGenerateLength)
{
    auto const replies = serve({{makePrompt(6, 0), 3}, {makePrompt(7, 1), 8}});
    ASSERT_EQ(replies.size(), 2U);
    EXPECT_EQ(replies[0].size(), 3U);
    EXPECT_EQ(replies[1].size(), 8U);
}
