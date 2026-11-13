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

#include "safetyRuntimeTestDoubles.h"
#include "safety_inference/runtime/model_inference/internvl3/internvl3VanillaInference.h"
#include "safety_inference/runtime/model_inference/multimodal/vlmRequestContext.h"
#include "safety_inference/runtime/processing/processingCapabilities.h"
#include "safety_inference/runtime/safetyRuntime.h"
#include "safety_inference/trt_runner/runnerResourceResolver.h"

#include <gtest/gtest.h>

namespace trt_edgellm::safety::runtime_test
{
namespace
{
//! Synthetic graph payloads exercise the real Runner's context/profile handoff, not model arithmetic.
class Processing : public llm::RequestOperations, public vlm::RequestOperations
{
public:
    std::string_view failStage{};
    std::vector<std::string> prepared{};
    std::vector<int32_t> decodeLengths{};
    std::vector<int32_t> decodeTokens{};
    std::vector<float> logits{};
    float visualOutput{0};
    std::size_t imageOffset{0};
    std::size_t imageCount{0};
    int32_t commits{0};

    SafetyStatus prepareVisualStageInputs(LMStageContext const& context, vlm::ImageInputView image) noexcept override
    {
        return fill(context, *static_cast<float const*>(image.data));
    }
    SafetyStatus preparePrefillInputs(LMStageContext const& context, TokenIdView prompt) noexcept override
    {
        return fill(context, static_cast<float>(prompt.data[0]));
    }
    SafetyStatus prepareLlmPrefillInputs(LMStageContext const& context, LMStageOutputs const& visual,
        std::size_t offset, std::size_t count) noexcept override
    {
        visualOutput = visual.context->profiles[0].resources->bindings[1].getFloat()[0];
        imageOffset = offset;
        imageCount = count;
        context.profiles[0].resources->bindings[0].getFloat()[0] += visualOutput;
        return SafetyStatus::ok();
    }
    SafetyStatus prepareDecodeInputs(LMStageContext const& context, int32_t token, int32_t length) noexcept override
    {
        decodeTokens.push_back(token);
        decodeLengths.push_back(length);
        return fill(context, static_cast<float>(token));
    }
    SafetyStatus selectGreedyTokenFromLogits(LMStageOutputs const& output, int32_t& token) noexcept override
    {
        logits.push_back(output.context->profiles[0].resources->bindings[1].getFloat()[0]);
        token = commits == 0 ? 7 : 9;
        return SafetyStatus::ok();
    }
    SafetyStatus commitKvCacheFromLMStage(LMStageOutputs const&) noexcept override
    {
        ++commits;
        return SafetyStatus::ok();
    }

private:
    SafetyStatus fill(LMStageContext const& context, float value) noexcept
    {
        prepared.push_back(context.plan.id);
        context.profiles[0].resources->bindings[0].getFloat()[0] = value;
        if (context.plan.id == failStage)
        {
            gBackend.executionResult = ErrorCode::kFAILED_EXECUTION;
        }
        return SafetyStatus::ok();
    }
};

SafetyStatus warmup(LMStageContext const& context, int64_t, void*)
{
    context.profiles[0].resources->bindings[0].getFloat()[0] = 1.0F;
    return SafetyStatus::ok();
}

class RequestRunnerIntegration : public testing::Test
{
protected:
    void SetUp() override
    {
        gBackend.reset();
    }
    void TearDown() override
    {
        // Test containment resets fake SDK objects while their borrowed recorder still exists.
        gBackend.reset();
    }

    void prepare()
    {
        static char const kENGINE[8]{};
        SafetyRuntimeInitSpec spec{{{kENGINE, sizeof(kENGINE), "/trusted/visual.so", {0}},
                                       {kENGINE, sizeof(kENGINE), "/trusted/llm.so", {0, 1}}},
            {{"visual_encode", 0, 0, {0}, {}}, {"prefill", 0, 1, {0}, {}}, {"decode", 0, 1, {1}, {}}}, {0, 0}};
        ASSERT_TRUE(mRuntime.initialize(spec).okStatus());
        ASSERT_TRUE(mResolver.prepare(mRuntime.contexts()).okStatus());
        // The model factory's tensor admission is covered separately. Here processing uses the fake
        // SDK's two tensors, resolved through the same adapter used by prepared model-local data.
        mData.spec.vocabSize = 100;
        mData.maxPrefillTokens = 4;
        mData.maxContextLength = 8;
        for (auto* stage : {&mData.prefill, &mData.decode})
        {
            stage->stage = stage == &mData.prefill ? "prefill" : "decode";
            stage->profileId = stage == &mData.prefill ? 0 : 1;
            RuntimeBorrowedResource resolved{};
            ASSERT_TRUE(mResolver.resolve({0, RuntimeBorrowedResourceKind::kCUDA_STREAM, stage->stage, 0, {}}, resolved)
                    .okStatus());
            stage->stream = resolved.stream;
            for (char const* name : {"input", "output"})
            {
                ASSERT_TRUE(
                    mResolver.resolve({0, RuntimeBorrowedResourceKind::kIO_BINDING, stage->stage, 0, name}, resolved)
                        .okStatus());
                stage->bindings.push_back(resolved.tensor);
            }
        }
        auto const& definition = *findModelInferenceDefinition(ModelInferenceId::kInternVL3_1B_Vlm_Vanilla);
        ASSERT_TRUE(createModelInference(definition, mData, mRuntime.contexts(), mProcessing, mProcessing, mInference)
                .okStatus());
        ASSERT_TRUE(mRuntime.prepare(warmup).okStatus());
    }

    ProcessingRequest request(bool eos = true)
    {
        return {{mPrompt.data(), mPrompt.size()}, {&mImage, sizeof(mImage)}, 1, 1, eos ? 4 : 2,
            {mStop.data(), mStop.size()}};
    }

    Recorder mRecorder{};
    SafetyRuntime mRuntime{mRecorder};
    std::vector<ModelLocalDataBinding> mObjects{};
    RunnerResourceResolver mResolver{mObjects};
    llm::ModelLocalData mData{};
    Processing mProcessing{};
    std::unique_ptr<ModelInference> mInference{};
    std::array<int32_t, 2> mPrompt{3, 4};
    std::array<int32_t, 1> mStop{9};
    std::array<int32_t, 4> mOutput{};
    float mImage{5.0F};
};

TEST_F(RequestRunnerIntegration, PreparedFlowUsesRunnerContextsAndStopsAtEosOrLength)
{
    // Given prepared Runner views, when processing runs visual/prefill/decode, then visual data,
    // tokens and KV progress reach the selected contexts and requests stop at EOS or the token limit.
    ASSERT_NO_FATAL_FAILURE(prepare());
    for (bool eos : {true, false})
    {
        mProcessing = Processing{};
        ModelInferenceResult result{};
        auto const executionBegin = gBackend.operations.size();
        auto input = request(eos);
        if (!eos)
        {
            input.stopTokenIds = {};
        }
        ASSERT_TRUE(mInference->run(mRuntime, input, {mOutput.data(), mOutput.size()}, result).okStatus());
        EXPECT_EQ(mProcessing.prepared, (std::vector<std::string>{"visual_encode", "prefill", "decode"}));
        EXPECT_EQ((std::tuple{mProcessing.visualOutput, mProcessing.imageOffset, mProcessing.imageCount}),
            (std::tuple{5.0F, std::size_t{1}, std::size_t{1}}));
        EXPECT_EQ(mProcessing.logits, (std::vector<float>{108.0F, 117.0F}));
        EXPECT_EQ(mProcessing.decodeTokens, (std::vector<int32_t>{7}));
        EXPECT_EQ(mProcessing.decodeLengths, (std::vector<int32_t>{2}));
        EXPECT_EQ((std::tuple{result.generatedTokens.size, result.contextLength, mProcessing.commits,
                      result.finishReason, mOutput[0], mOutput[1]}),
            (std::tuple{std::size_t{2}, 3, 2, eos ? FinishReason::kEndId : FinishReason::kLength, 7, 9}));
        std::vector<std::pair<int64_t, int64_t>> executed;
        for (auto index = executionBegin; index < gBackend.operations.size(); ++index)
        {
            auto const& operation = gBackend.operations[index];
            if (operation.name == "execute")
            {
                executed.emplace_back(operation.graph, operation.profile);
            }
        }
        EXPECT_EQ(executed, (std::vector<std::pair<int64_t, int64_t>>{{0, 0}, {1, 0}, {1, 1}}));
    }
    ASSERT_TRUE(mRuntime.deinitializeGraphs().okStatus());
    mInference.reset();
    ASSERT_TRUE(mRuntime.deinitializeResources().okStatus());
}

class RequestRunnerFailure : public RequestRunnerIntegration, public testing::WithParamInterface<int>
{
};

TEST_P(RequestRunnerFailure, FailedStageStopsSchedulingWithoutPublishingItsProgress)
{
    // Given a graph failure at any stage, when the real Runner reports it, then no later stage runs
    // and the result contains only previously committed progress.
    ASSERT_NO_FATAL_FAILURE(prepare());
    std::array<char const*, 3> const stages{"visual_encode", "prefill", "decode"};
    auto const failed = GetParam();
    mProcessing.failStage = stages[failed];
    auto const executions = gBackend.count("execute");
    ModelInferenceResult result{};
    mOutput.fill(-1);
    auto const status = mInference->run(mRuntime, request(), {mOutput.data(), mOutput.size()}, result);
    EXPECT_TRUE(status.fatalStatus());
    EXPECT_EQ(gBackend.count("execute") - executions, failed + 1);
    EXPECT_EQ(mProcessing.prepared, (std::vector<std::string>{stages.begin(), stages.begin() + failed + 1}));
    EXPECT_EQ((std::tuple{result.generatedTokens.size, result.contextLength, mProcessing.commits, result.finishReason,
                  mOutput[0], mOutput[1]}),
        (std::tuple{failed == 2 ? std::size_t{1} : std::size_t{0}, failed == 2 ? 2 : 0, failed == 2 ? 1 : 0,
            FinishReason::kError, failed == 2 ? 7 : -1, -1}));
}

INSTANTIATE_TEST_SUITE_P(VisualPrefillDecode, RequestRunnerFailure, testing::Values(0, 1, 2));
} // namespace
} // namespace trt_edgellm::safety::runtime_test
