/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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