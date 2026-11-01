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

#include "profiling/requestProfiler.h"
#include <gtest/gtest.h>
#include <utility>

namespace
{
struct TestClock
{
    using time_point = std::chrono::steady_clock::time_point;
    static inline time_point current{};
    static inline int32_t reads{0};

    static time_point now()
    {
        ++reads;
        return current;
    }
};
using Profiler = trt_edgellm::metrics::LLMRequestProfiler<TestClock>;
} // namespace

TEST(RequestProfilerTest, DisabledCollectionDoesNotReadClockOrRecordMetrics)
{
    // Given collection is disabled, when phases run, then no clock reads or metric updates occur.
    TestClock::reads = 0;
    Profiler profiler;
    profiler.beginPrefill(false);
    profiler.finishPrefill(1);
    profiler.recordGeneration(4);
    auto const result = profiler.snapshot(8);
    EXPECT_EQ(TestClock::reads, 0);
    EXPECT_EQ(result.processedPromptTokens, -1);
    EXPECT_EQ(result.generatedTokens, 0);
    EXPECT_EQ(result.decodedTokens, 0);
    EXPECT_DOUBLE_EQ(result.promptSeconds, 0.0);
    EXPECT_DOUBLE_EQ(result.generationSeconds, 0.0);
}

TEST(RequestProfilerTest, UsesRuntimeCountsAndExcludesPrefillTokensFromDecodeThroughput)
{
    // Given either prefill sampling mode, when tokens arrive in a burst, then counts and timing match the phase.
    for (int32_t const prefillTokens : {0, 1})
    {
        TestClock::current = TestClock::time_point{};
        Profiler profiler;
        profiler.beginPrefill(true);
        TestClock::current += std::chrono::seconds(1);
        profiler.beginPrefill(true);
        TestClock::current += std::chrono::seconds(1);
        profiler.finishPrefill(prefillTokens);
        auto const prefill = profiler.snapshot(2);
        EXPECT_EQ(prefill.generatedTokens, prefillTokens);
        EXPECT_EQ(prefill.decodedTokens, 0);
        EXPECT_DOUBLE_EQ(prefill.promptSeconds, 2.0);
        EXPECT_DOUBLE_EQ(prefill.generationSeconds, 0.0);
        profiler.recordGeneration(prefillTokens);

        TestClock::current += std::chrono::milliseconds(250);
        profiler.recordGeneration(3);
        auto const generated = profiler.snapshot(2);
        EXPECT_EQ(generated.processedPromptTokens, 2);
        EXPECT_EQ(generated.generatedTokens, 3);
        EXPECT_EQ(generated.decodedTokens, 3 - prefillTokens);
        EXPECT_DOUBLE_EQ(generated.generationSeconds, 0.25);

        // No new tokens after cancellation: draining or emitting the terminal chunk must not extend timing.
        auto const reads = TestClock::reads;
        TestClock::current += std::chrono::seconds(10);
        profiler.recordGeneration(3);
        EXPECT_EQ(TestClock::reads, reads);
        EXPECT_DOUBLE_EQ(profiler.snapshot(2).generationSeconds, 0.25);
    }
}

TEST(RequestProfilerTest, InterleavedRequestsKeepTheirOwnTimingsAfterCompaction)
{
    // Given interleaved sequences, when slots move, then their timing and counts remain independent.
    TestClock::current = TestClock::time_point{};
    Profiler first;
    first.beginPrefill(true);
    TestClock::current += std::chrono::seconds(1);
    first.finishPrefill(1);
    Profiler second;
    second.beginPrefill(true);
    TestClock::current += std::chrono::seconds(2);
    second.finishPrefill(1);
    std::swap(first, second);
    TestClock::current += std::chrono::seconds(1);
    first.recordGeneration(4);
    second.recordGeneration(2);
    auto const firstResult = first.snapshot(7);
    auto const secondResult = second.snapshot(2);
    EXPECT_EQ(firstResult.processedPromptTokens, 7);
    EXPECT_EQ(firstResult.generatedTokens, 4);
    EXPECT_EQ(firstResult.decodedTokens, 3);
    EXPECT_DOUBLE_EQ(firstResult.promptSeconds, 2.0);
    EXPECT_DOUBLE_EQ(firstResult.generationSeconds, 1.0);
    EXPECT_EQ(secondResult.processedPromptTokens, 2);
    EXPECT_EQ(secondResult.generatedTokens, 2);
    EXPECT_EQ(secondResult.decodedTokens, 1);
    EXPECT_DOUBLE_EQ(secondResult.promptSeconds, 1.0);
    EXPECT_DOUBLE_EQ(secondResult.generationSeconds, 3.0);
}
