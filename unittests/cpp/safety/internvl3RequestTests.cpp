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

#include "safety_inference/runtime/model_inference/internvl3/internvl3VanillaInference.h"
#include "safety_inference/runtime/model_inference/modelInferenceCatalog.h"
#include "safety_inference/runtime/processing/processingCapabilities.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

namespace trt_edgellm::safety
{
// CPU substitute for Runner-owned bindings; production request control treats this type as opaque.
struct LMStageContext
{
    LMStageId id;
    int32_t* input;
    int32_t* output;
};

namespace
{

enum class FailurePoint
{
    kNone,
    kVisualInput,
    kVisualRun,
    kPrefillInput,
    kVisualHandoff,
    kPrefillRun,
    kDecodeInput,
    kDecodeRun,
    kSampling,
    kKvCopy,
};

// Models prepared storage and processing; logits depend on the visual data actually handed to prefill.
class MockPreparedRuntime : public ModelInferenceRunner, public llm::RequestOperations, public vlm::RequestOperations
{
public:
    FailurePoint failure{FailurePoint::kNone};
    std::array<LMStageId, 8> stages{};
    std::size_t stageCount{0};
    std::array<int32_t, 4> embeddings{};
    std::array<int32_t, 2> visualFeatures{};
    std::array<int32_t, 8> kv{};
    int32_t kvLength{0};
    int32_t decodeToken{-1};
    int32_t logits{-1};
    std::array<int32_t, 2> imageStaging{};
    std::array<int32_t, 2> decodeInput{};
    LMStageContext visualContext{"visual_encode", imageStaging.data(), visualFeatures.data()};
    LMStageContext prefillContext{"prefill", embeddings.data(), &logits};
    LMStageContext decodeContext{"decode", decodeInput.data(), &logits};
    LMStageContext const* selectedContext{nullptr};
    bool preparing{false};

    SafetyStatus prepareVisualStageInputs(LMStageContext const& context, vlm::ImageInputView input) noexcept override
    {
        if (!preparing || selectedContext != &context || &context != &visualContext)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Visual context was not selected");
        }
        std::copy_n(static_cast<int32_t const*>(input.data), imageStaging.size(), context.input);
        return check(FailurePoint::kVisualInput);
    }

    SafetyStatus preparePrefillInputs(LMStageContext const& context, TokenIdView prompt) noexcept override
    {
        if (!preparing || selectedContext != &context || &context != &prefillContext)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Prefill context was not selected");
        }
        std::copy_n(prompt.data, prompt.size, context.input);
        return check(FailurePoint::kPrefillInput);
    }

    SafetyStatus prepareLlmPrefillInputs(LMStageContext const& context, LMStageOutputs const& outputs,
        std::size_t offset, std::size_t count) noexcept override
    {
        if (!preparing || selectedContext != &context || &context != &prefillContext
            || outputs.context != &visualContext)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Missing visual output");
        }
        std::copy_n(outputs.context->output, count, context.input + offset);
        return check(FailurePoint::kVisualHandoff);
    }

    SafetyStatus prepareDecodeInputs(LMStageContext const& context, int32_t token, int32_t length) noexcept override
    {
        if (!preparing || selectedContext != &context || &context != &decodeContext || length != kvLength)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Uncommitted KV length");
        }
        decodeToken = token;
        context.input[0] = token;
        context.input[1] = length;
        return check(FailurePoint::kDecodeInput);
    }

    SafetyStatus runLMStage(LMStageId stage, std::size_t slot, LMStageOutputs& outputs, LMStagePrepareCallback prepare,
        void* userData) noexcept
    {
        if (slot != 0 || stageCount == stages.size())
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Invalid stage slot");
        }
        outputs = {};
        selectedContext = stage == "visual_encode" ? &visualContext
            : stage == "prefill"                   ? &prefillContext
            : stage == "decode"                    ? &decodeContext
                                                   : nullptr;
        if (selectedContext == nullptr || prepare == nullptr)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Stage preparation is required");
        }
        preparing = true;
        auto const prepared = prepare(*selectedContext, userData);
        preparing = false;
        if (!prepared.okStatus())
        {
            return prepared;
        }
        stages[stageCount++] = stage;
        outputs.context = selectedContext;
        if (stage == "visual_encode")
        {
            for (std::size_t i = 0; i < visualFeatures.size(); ++i)
            {
                selectedContext->output[i] = selectedContext->input[i] + 1;
            }
            return check(FailurePoint::kVisualRun);
        }
        if (stage == "prefill")
        {
            logits = 0;
            for (auto value : embeddings)
            {
                logits += value;
            }
            return check(FailurePoint::kPrefillRun);
        }
        *selectedContext->output = selectedContext->input[0] + 1;
        return check(FailurePoint::kDecodeRun);
    }

    SafetyStatus selectGreedyTokenFromLogits(LMStageOutputs const& outputs, int32_t& token) noexcept override
    {
        if (outputs.context != selectedContext || outputs.context == nullptr)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Missing logits");
        }
        token = *outputs.context->output;
        return kvLength > 0 ? check(FailurePoint::kSampling) : SafetyStatus::ok();
    }

    SafetyStatus commitKvCacheFromLMStage(LMStageOutputs const&) noexcept override
    {
        if (kvLength == 0)
        {
            std::copy(embeddings.begin(), embeddings.end(), kv.begin());
            kvLength = static_cast<int32_t>(embeddings.size());
            return SafetyStatus::ok();
        }
        auto const status = check(FailurePoint::kKvCopy);
        if (status.okStatus())
        {
            kv[kvLength++] = decodeToken;
        }
        return status;
    }

private:
    SafetyStatus check(FailurePoint point) const noexcept
    {
        return failure == point ? SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Injected stage failure")
                                : SafetyStatus::ok();
    }
};

struct RequestFixture
{
    MockPreparedRuntime runtime;
    llm::RuntimeData data{runtime, "prefill", "decode", runtime.prefillContext, runtime.decodeContext, 4, 6, 100};
    ModelInferenceDefinition const& definition{
        *findModelInferenceDefinition(ModelInferenceId::kInternVL3_1B_Vlm_Vanilla)};
    InternVL3VanillaInference inference{definition, data, runtime, runtime.visualContext};
    std::array<int32_t, 4> prompt{{5, 6, 7, 8}};
    std::array<int32_t, 2> image{{10, 20}};
    std::array<int32_t, 4> output{{-1, -1, -1, -1}};
    ProcessingRequest request{{prompt.data(), prompt.size()}, {image.data(), sizeof(image)}, 1, 2, 3, {}};
    ModelInferenceResult result{};

    SafetyStatus run() noexcept
    {
        return inference.run(runtime, request, {output.data(), 3}, result);
    }

    std::vector<LMStageId> executedStages() const
    {
        return {runtime.stages.begin(), runtime.stages.begin() + runtime.stageCount};
    }

    std::vector<int32_t> generatedTokens() const
    {
        return {result.generatedTokens.data, result.generatedTokens.data + result.generatedTokens.size};
    }
};

class InternVL3RequestFlowTest : public ::testing::TestWithParam<int32_t>
{
};

TEST_P(InternVL3RequestFlowTest, VisualPrefillDecodePublishesCommittedProgress)
{
    // Given prepared InternVL3 resources, when visual/prefill/decode succeed,
    // then visual results feed prefill, token/KV progress commits, and EOS or length stops scheduling.
    RequestFixture fixture;
    int32_t const stop = GetParam();
    fixture.request.stopTokenIds = {&stop, 1};
    auto const status = fixture.run();
    EXPECT_TRUE(status.okStatus());
    EXPECT_EQ(fixture.result.generatedTokens.data, fixture.output.data());
    EXPECT_EQ(fixture.runtime.embeddings, (std::array<int32_t, 4>{{5, 11, 21, 8}}));
    bool const earlyStop = stop == 46;
    EXPECT_EQ(fixture.executedStages(),
        earlyStop ? (std::vector<LMStageId>{"visual_encode", "prefill", "decode"})
                  : (std::vector<LMStageId>{"visual_encode", "prefill", "decode", "decode"}));
    EXPECT_EQ(
        fixture.generatedTokens(), earlyStop ? (std::vector<int32_t>{45, 46}) : (std::vector<int32_t>{45, 46, 47}));
    EXPECT_EQ(fixture.result.contextLength, earlyStop ? 5 : 6);
    EXPECT_EQ(fixture.runtime.kv,
        earlyStop ? (std::array<int32_t, 8>{{5, 11, 21, 8, 45, 0, 0, 0}})
                  : (std::array<int32_t, 8>{{5, 11, 21, 8, 45, 46, 0, 0}}));
    EXPECT_EQ(fixture.output.back(), -1);
    EXPECT_EQ(fixture.result.finishReason, stop == -1 ? FinishReason::kLength : FinishReason::kEndId);
}

INSTANTIATE_TEST_SUITE_P(LengthAndEarlyOrFinalEOS, InternVL3RequestFlowTest, ::testing::Values(-1, 46, 47));

class InternVL3RequestFailureTest : public ::testing::TestWithParam<FailurePoint>
{
};

TEST_P(InternVL3RequestFailureTest, FailedStepStopsSchedulingWithoutPublishingProgress)
{
    // Given prepared InternVL3 resources, when a stage or its processing fails,
    // then scheduling stops and the failed step publishes no token or KV progress.
    RequestFixture fixture;
    fixture.runtime.failure = GetParam();
    auto const status = fixture.run();
    EXPECT_EQ(status.code(), SafetyStatusCode::kFailedExecution);
    EXPECT_TRUE(status.fatalStatus());
    bool const prefilled = GetParam() >= FailurePoint::kDecodeInput;
    std::vector<LMStageId> expected;
    if (GetParam() >= FailurePoint::kVisualRun)
    {
        expected.push_back("visual_encode");
    }
    if (GetParam() >= FailurePoint::kPrefillRun)
    {
        expected.push_back("prefill");
    }
    if (GetParam() >= FailurePoint::kDecodeRun)
    {
        expected.push_back("decode");
    }
    EXPECT_EQ(fixture.executedStages(), expected);
    EXPECT_EQ(fixture.generatedTokens(), prefilled ? (std::vector<int32_t>{45}) : std::vector<int32_t>{});
    EXPECT_EQ(fixture.result.contextLength, prefilled ? 4 : 0);
    EXPECT_EQ(fixture.output,
        prefilled ? (std::array<int32_t, 4>{{45, -1, -1, -1}}) : (std::array<int32_t, 4>{{-1, -1, -1, -1}}));
    EXPECT_EQ(fixture.result.finishReason, FinishReason::kError);
}

INSTANTIATE_TEST_SUITE_P(RunnerAndProcessing, InternVL3RequestFailureTest,
    ::testing::Values(FailurePoint::kVisualInput, FailurePoint::kVisualRun, FailurePoint::kPrefillInput,
        FailurePoint::kVisualHandoff, FailurePoint::kPrefillRun, FailurePoint::kDecodeInput, FailurePoint::kDecodeRun,
        FailurePoint::kSampling, FailurePoint::kKvCopy));

} // namespace
} // namespace trt_edgellm::safety
