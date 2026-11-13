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

#include "common/checkMacros.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>

using namespace trt_edgellm;
class TimerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Reset timer and enable profiling
        gTimer.select(std::make_shared<timer::Timer>());
        gTimer.reset();
        setProfilingEnabled(true);

        // Initialize CUDA for testing
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        setProfilingEnabled(false);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    cudaStream_t stream;
};

TEST_F(TimerTest, BasicStageProfile)
{
    {
        TIME_STAGE("test_stage", stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Pending timings are calculated lazily when getTimingData() is called
    auto timingData = gTimer.getTimingData("test_stage");
    ASSERT_TRUE(timingData.has_value());
    EXPECT_EQ(timingData->getTotalRuns(), 1);
    EXPECT_GT(timingData->getTotalGpuTimeMs(), 0.0f);
    EXPECT_GT(timingData->getAverageTimeMs(), 0.0f);
}

TEST_F(TimerTest, MultipleStageTiming)
{
    // Test multiple stages with different timing
    {
        TIME_STAGE("stage1", stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    {
        TIME_STAGE("stage2", stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    {
        TIME_STAGE("stage1", stream); // Second run of stage1
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Pending timings are calculated lazily when getTimingData() is called

    // Verify stage1 has 2 runs
    auto stage1Data = gTimer.getTimingData("stage1");
    ASSERT_TRUE(stage1Data.has_value());
    EXPECT_EQ(stage1Data->getTotalRuns(), 2);
    EXPECT_GT(stage1Data->getTotalGpuTimeMs(), 0.0f);
    EXPECT_GT(stage1Data->getAverageTimeMs(), 0.0f);

    // Verify stage2 has 1 run
    auto stage2Data = gTimer.getTimingData("stage2");
    ASSERT_TRUE(stage2Data.has_value());
    EXPECT_EQ(stage2Data->getTotalRuns(), 1);
    EXPECT_GT(stage2Data->getTotalGpuTimeMs(), 0.0f);
    EXPECT_GT(stage2Data->getAverageTimeMs(), 0.0f);
}

TEST_F(TimerTest, SpecDecodeGenerationMetricsAccumulateAcceptanceCounters)
{
    metrics::SpecDecodeGenerationMetrics metrics;
    metrics.recordRun(30, 120, 8, 10);
    metrics.recordRun(10, 30, 3, 5);

    EXPECT_EQ(metrics.getTotalRuns(), 2);
    EXPECT_EQ(metrics.totalIterations, 40);
    EXPECT_EQ(metrics.totalGeneratedTokens, 150);
    EXPECT_EQ(metrics.totalAcceptedDraftTokens, 11);
    EXPECT_EQ(metrics.totalProposedDraftTokens, 15);
}

TEST_F(TimerTest, TimerReset)
{
    // Create some timing data
    {
        TIME_STAGE("test_stage", stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Pending timings are calculated lazily when getTimingData() is called
    auto timingData = gTimer.getTimingData("test_stage");
    ASSERT_TRUE(timingData.has_value());
    EXPECT_EQ(timingData->getTotalRuns(), 1);

    // Reset timer
    gTimer.reset();

    // Verify data is cleared
    auto clearedData = gTimer.getTimingData("test_stage");
    EXPECT_FALSE(clearedData.has_value());
}

// Given two uncollected executions of a stage with a nested model forward
// When the results are read after both executions
// Then both samples retain their own round/shape and are not counted twice on later reads.
TEST_F(TimerTest, PreservesInvocationIdentityUntilCollection)
{
    gTimer.reserve(16);
    gTimer.beginRequest();
    for (int32_t round = 0; round < 2; ++round)
    {
        gTimer.setRound(round);
        auto stage = gTimer.startStage("proposal", stream);
        gTimer.setPastLengths({17 + round});
        auto forward = gTimer.startForward("draft", 1, 4, 4, stream);
    }
    auto const samples = gTimer.getSamples();
    ASSERT_EQ(samples.size(), 4U);
    EXPECT_EQ(samples[1].parentStage, "proposal");
    EXPECT_EQ(samples[1].round, 0);
    EXPECT_EQ(samples[1].pastLengths, std::vector<int32_t>{17});
    EXPECT_EQ(samples[3].round, 1);
    EXPECT_EQ(samples[3].pastLengths, std::vector<int32_t>{18});
    EXPECT_TRUE(samples[3].forward);
    EXPECT_EQ(samples[3].stageId, "forward/draft");
    EXPECT_EQ(samples[3].queryTokens, 4);
    EXPECT_EQ(gTimer.getTimingData("proposal")->getTotalRuns(), 2);
    EXPECT_EQ(gTimer.getSamples().size(), 4U);
}

// Given repeated draft and base forwards within a round, across rounds and across requests
// When their records are collected
// Then each forward has an ordinal within its request/round, while stages have no forward ordinal.
TEST_F(TimerTest, ForwardOrdinalsFollowRequestAndRoundBoundaries)
{
    gTimer.reserve(12);
    for (int32_t request = 0; request < 2; ++request)
    {
        gTimer.beginRequest();
        {
            auto prefill = gTimer.startForward("base", 1, 4, 1, stream);
        }
        for (int32_t round = 0; round < 2; ++round)
        {
            gTimer.setRound(round);
            auto stage = gTimer.startStage("proposal", stream);
            for (std::string const role : {"draft", "base"})
            {
                gTimer.setRound(round);
                auto forward = gTimer.startForward(role, 1, 1, 1, stream);
            }
        }
    }
    auto const& samples = gTimer.getSamples();
    std::vector<int32_t> ordinals;
    for (auto const& sample : samples)
    {
        ordinals.push_back(sample.forwardIndex);
    }
    EXPECT_EQ(ordinals, (std::vector<int32_t>{0, -1, 0, 1, -1, 0, 1, 0, -1, 0, 1, -1, 0, 1}));
}

// Given host-known prefill lengths followed by a forward whose lengths are computed only on device
// When both forwards are recorded in the same round
// Then the second reports unavailable lengths instead of inheriting the prefill shape.
TEST_F(TimerTest, DoesNotReusePastLengthsFromAnEarlierForward)
{
    gTimer.reserve(2);
    gTimer.beginRequest();
    gTimer.setPastLengths({17});
    {
        auto prefill = gTimer.startForward("draft", 1, 4, 1, stream);
    }
    {
        auto proposal = gTimer.startForward("draft", 1, 3, 1, stream);
    }
    std::vector<std::vector<int32_t>> pastLengths;
    for (auto const& sample : gTimer.getSamples())
    {
        pastLengths.push_back(sample.pastLengths);
    }
    EXPECT_EQ(pastLengths, (std::vector<std::vector<int32_t>>{{17}, {}}));
}

// Given an estimated sample capacity and a completed pending sample
// When profiling outlives the estimate, then resets for another experiment
// Then samples remain complete and reset preserves usable storage.
TEST_F(TimerTest, PoolCanGrowAndResetRetainsUsableStorage)
{
    timer::Timer local;
    local.reserve(1);
    {
        auto first = local.startStage("first", stream);
    }
    static_cast<void>(local.getSamples());
    {
        auto extra = local.startStage("extra", stream);
    }
    EXPECT_EQ(local.getSamples().size(), 2U);
    local.reset();
    {
        auto second = local.startStage("second", stream);
    }
    auto const samples = local.getSamples();
    ASSERT_EQ(samples.size(), 1U);
    EXPECT_EQ(samples.front().stageId, "second");
    EXPECT_FALSE(local.getTimingData("first").has_value());
}

// Given a long-lived parent stage with an underestimated event pool
// When nested forwards continue across many rounds
// Then all observations retain their round, shape and parent through pool growth.
TEST_F(TimerTest, NestedStagesOutliveInitialEventCapacity)
{
    timer::Timer local;
    local.reserve(2);
    constexpr int32_t rounds = 40;
    {
        auto parent = local.startStage("audio", stream);
        for (int32_t round = 0; round < rounds; ++round)
        {
            local.setRound(round);
            auto stage = local.startStage("code_predictor", stream);
            local.setPastLengths({round});
            auto forward = local.startForward("draft", 1, 1, 1, stream);
        }
    }
    auto const& samples = local.getSamples();
    ASSERT_EQ(samples.size(), 1U + 2U * rounds);
    EXPECT_EQ(local.getTimingData("audio")->getTotalRuns(), 1);
    EXPECT_EQ(local.getTimingData("code_predictor")->getTotalRuns(), rounds);
    for (int32_t round = 0; round < rounds; ++round)
    {
        auto const& stage = samples[1 + 2 * round];
        auto const& forward = samples[2 + 2 * round];
        EXPECT_EQ(stage.parentStage, "audio");
        EXPECT_EQ(forward.parentStage, "code_predictor");
        EXPECT_EQ(forward.round, round);
        EXPECT_EQ(forward.forwardIndex, 0);
        EXPECT_EQ(forward.pastLengths, std::vector<int32_t>{round});
    }
}

// Given an acceptance-history estimate smaller than a continuously admitted workload
// When more rounds complete than initially reserved
// Then all per-sequence acceptance records remain available instead of failing inference.
TEST_F(TimerTest, AcceptanceHistoryOutlivesInitialCapacity)
{
    timer::Timer local;
    local.reserve(1);
    for (int32_t round = 0; round < 40; ++round)
    {
        local.setRound(round);
        local.recordAcceptance(0, 3, 2, 3, 1, round + 4, true);
    }
    auto const& samples = local.getAcceptanceSamples();
    ASSERT_EQ(samples.size(), 40U);
    EXPECT_EQ(samples.back().round, 39);
    EXPECT_EQ(samples.back().baseMaterializedLength, 43);
}

// Given profiling with GPU timing disabled
// When a stage runs and its round records acceptance
// Then the acceptance record is kept without creating timing events or stage samples.
TEST_F(TimerTest, AcceptanceIsRecordedWithoutGpuTiming)
{
    timer::Timer local;
    local.setGpuTimingEnabled(false);
    local.reserve(8);
    local.setRound(0);
    {
        auto stage = local.startStage("decode_round", stream);
        local.recordAcceptance(0, 3, 2, 3, 1, 4, true);
    }
    EXPECT_EQ(local.capacity(), 0U);
    EXPECT_TRUE(local.getSamples().empty());
    ASSERT_EQ(local.getAcceptanceSamples().size(), 1U);
    EXPECT_EQ(local.getAcceptanceSamples()[0].acceptedDrafts, 2);
    EXPECT_EQ(local.getAcceptanceSamples()[0].bonusTokens, 1);
}

// Given two runtime-owned timers on one calling thread
// When profiling switches owners and later returns to the first
// Then each runtime retains only its own observations.
TEST_F(TimerTest, RuntimeOwnersDoNotMixSamples)
{
    auto first = std::make_shared<timer::Timer>();
    auto second = std::make_shared<timer::Timer>();
    first->reserve(2);
    second->reserve(2);
    gTimer.select(first);
    {
        auto stage = gTimer.startStage("first", stream);
    }
    gTimer.select(second);
    {
        auto stage = gTimer.startStage("second", stream);
    }
    EXPECT_FALSE(gTimer.getTimingData("first").has_value());
    gTimer.select(first);
    EXPECT_EQ(gTimer.getTimingData("first")->getTotalRuns(), 1);
    EXPECT_FALSE(gTimer.getTimingData("second").has_value());
}
