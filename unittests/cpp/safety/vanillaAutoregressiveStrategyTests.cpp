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

#include "safety_inference/runtime/model_inference/decoding/vanillaAutoregressiveStrategy.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace trt_edgellm::safety
{
namespace
{

struct MockDecodeState
{
    struct LMStageOutputs
    {
        int32_t token;
        bool fail{false};
    };
    struct PendingStep
    {
        int32_t selectedTokenId{-1};
        int32_t kvLength{0};
    };

    int32_t kvLength{0};
    int32_t commits{0};
    std::array<LMStageId, 8> observedStages{};
    std::size_t calls{0};

    int64_t generatedTokenCapacity() const noexcept
    {
        return 3;
    }

    SafetyStatus prepareLMStageOutputs(LMStageId stage, LMStageOutputs const& outputs, PendingStep& step) noexcept
    {
        observedStages[calls++] = stage;
        step = {outputs.token, stage == "prefill" ? 5 : kvLength + 1};
        return outputs.fail ? SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected sampling failure")
                            : SafetyStatus::ok();
    }

    void commitLMStageOutputs(PendingStep const& step) noexcept
    {
        kvLength = step.kvLength;
        ++commits;
    }
};

using Strategy = VanillaAutoregressiveStrategy<MockDecodeState>;

std::vector<int32_t> tokens(Strategy const& strategy)
{
    auto const view = strategy.generatedTokenIds();
    return {view.data, view.data + view.size};
}

TEST(SafetyVanillaStrategyTest, PrefillThenBoundedDecode)
{
    // Given a prepared vanilla request, when outputs succeed,
    // then prefill runs once followed by decode until the generation limit.
    MockDecodeState state;
    std::array<int32_t, 4> output{{-1, -1, -1, -1}};
    Strategy strategy(state, "prefill", "decode", {output.data(), 3});
    ASSERT_TRUE(strategy.beginRequest(3, nullptr, 0).okStatus());
    for (int32_t token = 41; strategy.hasNextLMStage() && token < 44; ++token)
    {
        ASSERT_TRUE(strategy.acceptLMStageOutputs({token}).okStatus());
    }
    EXPECT_EQ((std::vector<LMStageId>(state.observedStages.begin(), state.observedStages.begin() + state.calls)),
        (std::vector<LMStageId>{"prefill", "decode", "decode"}));
    EXPECT_EQ(tokens(strategy), (std::vector<int32_t>{41, 42, 43}));
    EXPECT_EQ(state.kvLength, 7);
    EXPECT_EQ(output.back(), -1);
    EXPECT_FALSE(strategy.hasNextLMStage());
    EXPECT_EQ(strategy.finishReason(), FinishReason::kLength);
}

TEST(SafetyVanillaStrategyTest, StopTokenWinsOverLengthLimit)
{
    // Given a prepared vanilla request, when the final allowed token is a stop token,
    // then stop-token termination wins over the simultaneous length limit.
    MockDecodeState state;
    std::array<int32_t, 2> output{{-1, -1}};
    std::array<int32_t, 2> const stops{{17, 42}};
    Strategy strategy(state, "prefill", "decode", {output.data(), output.size()});
    ASSERT_TRUE(strategy.beginRequest(2, stops.data(), stops.size()).okStatus());
    ASSERT_TRUE(strategy.acceptLMStageOutputs({41}).okStatus());
    ASSERT_TRUE(strategy.acceptLMStageOutputs({42}).okStatus());
    EXPECT_EQ(tokens(strategy), (std::vector<int32_t>{41, 42}));
    EXPECT_EQ(strategy.finishReason(), FinishReason::kEndId);
    EXPECT_FALSE(strategy.hasNextLMStage());
}

class SafetyVanillaFailureTest : public ::testing::TestWithParam<bool>
{
};

TEST_P(SafetyVanillaFailureTest, FailedStepLeavesProgressUncommitted)
{
    // Given a prepared request with an accepted prefill, when runner or output processing fails,
    // then no further stage runs and the failed step commits no token or KV progress.
    MockDecodeState state;
    std::array<int32_t, 3> output{{-1, -1, -1}};
    Strategy strategy(state, "prefill", "decode", {output.data(), output.size()});
    ASSERT_TRUE(strategy.beginRequest(3, nullptr, 0).okStatus());
    ASSERT_TRUE(strategy.acceptLMStageOutputs({41}).okStatus());

    auto const failure = GetParam()
        ? strategy.acceptLMStageOutputs({42, true})
        : strategy.abortRequest(SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Injected execution failure"));
    EXPECT_EQ(failure.code(), GetParam() ? SafetyStatusCode::kCudaError : SafetyStatusCode::kFailedExecution);
    EXPECT_TRUE(failure.fatalStatus());
    for (int i = 0; strategy.hasNextLMStage() && i < 2; ++i)
    {
        (void) strategy.acceptLMStageOutputs({43});
    }
    EXPECT_EQ(state.calls, GetParam() ? 2U : 1U);
    EXPECT_EQ(state.commits, 1);
    EXPECT_EQ(state.kvLength, 5);
    EXPECT_EQ(tokens(strategy), (std::vector<int32_t>{41}));
    EXPECT_EQ(output[1], -1);
    EXPECT_FALSE(strategy.hasNextLMStage());
    EXPECT_EQ(strategy.finishReason(), FinishReason::kError);
}

INSTANTIATE_TEST_SUITE_P(RunnerAndOutputProcessing, SafetyVanillaFailureTest, ::testing::Bool());

enum class RequestEnd
{
    kLength,
    kStopToken,
    kNonfatalError,
};

class SafetyVanillaRequestLifetimeTest : public ::testing::TestWithParam<RequestEnd>
{
};

TEST_P(SafetyVanillaRequestLifetimeTest, NewRequestBorrowsExistingStorageWithFreshState)
{
    // Given a prepared buffer, when a request completes or ends with a nonfatal error,
    // then its objects are finished; new request objects borrow the same buffer with fresh progress.
    std::array<int32_t, 3> output{{-1, -1, -1}};
    {
        MockDecodeState state;
        Strategy strategy(state, "prefill", "decode", {output.data(), output.size()});
        int32_t const stop = 42;
        ASSERT_TRUE(strategy
                .beginRequest(2, GetParam() == RequestEnd::kStopToken ? &stop : nullptr,
                    GetParam() == RequestEnd::kStopToken ? 1 : 0)
                .okStatus());
        ASSERT_TRUE(strategy.acceptLMStageOutputs({41}).okStatus());
        if (GetParam() == RequestEnd::kNonfatalError)
        {
            auto const status = strategy.abortRequest(
                SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Injected request input error"));
            EXPECT_EQ(status.code(), SafetyStatusCode::kInvalidArgument);
            EXPECT_FALSE(status.fatalStatus());
        }
        else
        {
            ASSERT_TRUE(strategy.acceptLMStageOutputs({42}).okStatus());
        }
        EXPECT_FALSE(strategy.hasNextLMStage());
        EXPECT_EQ(strategy.beginRequest(3, nullptr, 0).code(), SafetyStatusCode::kInvalidState);
    }

    MockDecodeState state;
    Strategy strategy(state, "prefill", "decode", {output.data(), output.size()});
    EXPECT_EQ(output, (std::array<int32_t, 3>{{41, GetParam() == RequestEnd::kNonfatalError ? -1 : 42, -1}}));
    ASSERT_TRUE(strategy.beginRequest(3, nullptr, 0).okStatus());
    EXPECT_EQ(strategy.generatedTokenIds().data, output.data());
    EXPECT_EQ(strategy.generatedTokenIds().size, 0U);
    EXPECT_EQ(strategy.finishReason(), FinishReason::kNotFinished);
    EXPECT_EQ(strategy.nextLMStage(), "prefill");
    EXPECT_TRUE(strategy.hasNextLMStage());
    for (int32_t token = 42; token < 45 && strategy.hasNextLMStage(); ++token)
    {
        ASSERT_TRUE(strategy.acceptLMStageOutputs({token}).okStatus());
    }
    EXPECT_EQ(tokens(strategy), (std::vector<int32_t>{42, 43, 44}));
    EXPECT_EQ(output, (std::array<int32_t, 3>{{42, 43, 44}}));
    EXPECT_EQ(state.kvLength, 7);
    EXPECT_FALSE(strategy.hasNextLMStage());
    EXPECT_EQ(strategy.finishReason(), FinishReason::kLength);
}

INSTANTIATE_TEST_SUITE_P(CompletedAndRejectedRequests, SafetyVanillaRequestLifetimeTest,
    ::testing::Values(RequestEnd::kLength, RequestEnd::kStopToken, RequestEnd::kNonfatalError));

} // namespace
} // namespace trt_edgellm::safety
