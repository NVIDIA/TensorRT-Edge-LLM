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

#include "runtime/embedding/pleReadPlan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

uint8_t sourceByte(size_t shard, int64_t offset)
{
    auto const value = static_cast<uint64_t>(offset);
    return static_cast<uint8_t>((value * 37) ^ (value >> 8) ^ (value >> 32) ^ (shard * 53));
}

void verifyReconstruction(std::vector<PleReadSpan> const& spans, std::vector<int64_t> const& ends, size_t alignment,
    size_t budget, std::vector<PleReadRange> const& plan)
{
    size_t destinationBytes = 0;
    for (auto const& span : spans)
    {
        destinationBytes = std::max(destinationBytes, span.destination + span.bytes);
    }
    std::vector<uint8_t> actual(destinationBytes, 0xAC);
    auto expected = actual;
    std::vector<int32_t> writes(destinationBytes, 0);
    std::vector<int32_t> expectedWrites(destinationBytes, 0);
    for (auto const& span : spans)
    {
        for (size_t byte = 0; byte < span.bytes; ++byte)
        {
            expected[span.destination + byte] = sourceByte(span.shard, span.offset + static_cast<int64_t>(byte));
            ++expectedWrites[span.destination + byte];
        }
    }
    for (size_t index = 0; index < plan.size(); ++index)
    {
        auto const& range = plan[index];
        ASSERT_LT(range.shard, ends.size());
        ASSERT_GE(range.offset, 0);
        ASSERT_GT(range.bytes, 0U);
        ASSERT_LE(range.bytes, budget);
        ASSERT_LE(range.offset, ends[range.shard]);
        ASSERT_LE(range.bytes, static_cast<size_t>(ends[range.shard] - range.offset));
        EXPECT_EQ(static_cast<size_t>(range.offset) % alignment, 0U);
        EXPECT_TRUE(
            range.bytes % alignment == 0 || range.offset + static_cast<int64_t>(range.bytes) == ends[range.shard]);
        ASSERT_FALSE(range.copies.empty());
        if (index > 0)
        {
            auto const& previous = plan[index - 1];
            ASSERT_LE(previous.shard, range.shard);
            if (previous.shard == range.shard)
            {
                EXPECT_LE(previous.offset + static_cast<int64_t>(previous.bytes), range.offset);
            }
        }
        for (auto const& copy : range.copies)
        {
            ASSERT_GT(copy.bytes, 0U);
            ASSERT_LE(copy.source, range.bytes);
            ASSERT_LE(copy.bytes, range.bytes - copy.source);
            ASSERT_LE(copy.destination, actual.size());
            ASSERT_LE(copy.bytes, actual.size() - copy.destination);
            for (size_t byte = 0; byte < copy.bytes; ++byte)
            {
                actual[copy.destination + byte]
                    = sourceByte(range.shard, range.offset + static_cast<int64_t>(copy.source + byte));
                ++writes[copy.destination + byte];
            }
        }
    }
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(writes, expectedWrites);
}

TEST(PleReadPlanTest, EmptyRequestsNeedNoReads)
{
    EXPECT_TRUE(planPleReads({}, {}, 4096, 65536).empty());
    EXPECT_TRUE(planPleReads({}, {0}, 1, 1).empty());
}

TEST(PleReadPlanTest, SmallUnsortedRowsShareOnePhysicalPage)
{
    std::vector<PleReadSpan> const spans{{0, 27, 5, 12}, {0, 3, 7, 0}, {0, 12, 2, 9}};
    auto const plan = planPleReads(spans, {64}, 32, 64);
    ASSERT_EQ(plan.size(), 1U);
    EXPECT_EQ(plan[0].offset, 0);
    EXPECT_EQ(plan[0].bytes, 32U);
    verifyReconstruction(spans, {64}, 32, 64, plan);
}

TEST(PleReadPlanTest, OverlappingSourcesPopulateIndependentDestinations)
{
    std::vector<PleReadSpan> const spans{{0, 6, 21, 0}, {0, 10, 8, 24}, {0, 6, 21, 35}};
    auto const plan = planPleReads(spans, {64}, 8, 16);
    ASSERT_EQ(plan.size(), 2U);
    verifyReconstruction(spans, {64}, 8, 16, plan);
}

TEST(PleReadPlanTest, AdjacentAlignedIntervalsMergeButGapsRemainUnread)
{
    std::vector<PleReadSpan> const spans{{0, 65, 2, 0}, {0, 17, 2, 3}, {0, 1, 2, 6}};
    auto const plan = planPleReads(spans, {128}, 16, 128);
    ASSERT_EQ(plan.size(), 2U);
    EXPECT_EQ(plan[0].offset, 0);
    EXPECT_EQ(plan[0].bytes, 32U);
    EXPECT_EQ(plan[1].offset, 64);
    EXPECT_EQ(plan[1].bytes, 16U);
    verifyReconstruction(spans, {128}, 16, 128, plan);
}

TEST(PleReadPlanTest, RowCrossingReadBoundariesIsReconstructedOnce)
{
    std::vector<PleReadSpan> const spans{{0, 7, 47, 5}};
    auto const plan = planPleReads(spans, {80}, 8, 16);
    ASSERT_EQ(plan.size(), 4U);
    EXPECT_EQ(plan.back().bytes, 8U);
    verifyReconstruction(spans, {80}, 8, 16, plan);
}

TEST(PleReadPlanTest, UnalignedShardTailIsClipped)
{
    std::vector<PleReadSpan> const spans{{0, 30, 5, 0}, {0, 18, 7, 8}};
    auto const plan = planPleReads(spans, {35}, 16, 16);
    ASSERT_EQ(plan.size(), 2U);
    EXPECT_EQ(plan[0].offset, 16);
    EXPECT_EQ(plan[0].bytes, 16U);
    EXPECT_EQ(plan[1].offset, 32);
    EXPECT_EQ(plan[1].bytes, 3U);
    verifyReconstruction(spans, {35}, 16, 16, plan);
}

TEST(PleReadPlanTest, NonPowerOfTwoAlignment)
{
    std::vector<PleReadSpan> const spans{{0, 5, 24, 0}};
    auto const plan = planPleReads(spans, {31}, 6, 12);
    ASSERT_EQ(plan.size(), 3U);
    verifyReconstruction(spans, {31}, 6, 12, plan);
}

TEST(PleReadPlanTest, MultipleShardsDoNotMerge)
{
    std::vector<PleReadSpan> const spans{{2, 0, 4, 12}, {0, 8, 6, 0}, {1, 1, 5, 7}};
    auto const plan = planPleReads(spans, {16, 7, 9}, 8, 32);
    ASSERT_EQ(plan.size(), 3U);
    EXPECT_EQ(plan[0].shard, 0U);
    EXPECT_EQ(plan[1].shard, 1U);
    EXPECT_EQ(plan[2].shard, 2U);
    verifyReconstruction(spans, {16, 7, 9}, 8, 32, plan);
}

TEST(PleReadPlanTest, SparseLargeOffsetsDoNotAllocateTheGap)
{
    int64_t const large = (int64_t{1} << 40);
    std::vector<PleReadSpan> const spans{{0, large + 13, 39, 0}, {0, 8, 2, 45}};
    auto const plan = planPleReads(spans, {large + 4096}, 64, 128);
    ASSERT_EQ(plan.size(), 2U);
    EXPECT_EQ(plan[1].offset, large);
    EXPECT_EQ(plan[1].bytes, 64U);
    verifyReconstruction(spans, {large + 4096}, 64, 128, plan);
}

TEST(PleReadPlanTest, AlignmentNearMaximumOffsetClipsWithoutOverflow)
{
    int64_t const end = std::numeric_limits<int64_t>::max();
    std::vector<PleReadSpan> const spans{{0, end - 5, 5, 0}};
    auto const plan = planPleReads(spans, {end}, 4096, 8192);
    ASSERT_EQ(plan.size(), 1U);
    EXPECT_EQ(plan[0].bytes, 4095U);
    verifyReconstruction(spans, {end}, 4096, 8192, plan);
}

TEST(PleReadPlanTest, InvalidAlignmentOrBudgetIsRejected)
{
    EXPECT_THROW(planPleReads({}, {}, 0, 16), std::invalid_argument);
    EXPECT_THROW(planPleReads({}, {}, 8, 0), std::invalid_argument);
    EXPECT_THROW(planPleReads({}, {}, 8, 7), std::invalid_argument);
    EXPECT_THROW(planPleReads({}, {}, 8, 12), std::invalid_argument);
    size_t const oversized = static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;
    EXPECT_THROW(planPleReads({}, {}, oversized, oversized), std::invalid_argument);
    EXPECT_THROW(planPleReads({}, {}, 1, oversized), std::invalid_argument);
}

TEST(PleReadPlanTest, InvalidSourceOrDestinationRangesAreRejected)
{
    EXPECT_THROW(planPleReads({}, {-1}, 1, 1), std::invalid_argument);
    EXPECT_THROW(planPleReads({{0, 0, 1, 0}}, {}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, -1, 1, 0}}, {8}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, 0, 0, 0}}, {8}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, 8, 1, 0}}, {8}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, 9, 1, 0}}, {8}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, 7, 2, 0}}, {8}, 1, 1), std::out_of_range);
    auto const maxOffset = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(planPleReads({{0, maxOffset - 1, 2, 0}}, {maxOffset}, 1, 1), std::out_of_range);
    EXPECT_THROW(planPleReads({{0, 0, 2, std::numeric_limits<size_t>::max()}}, {8}, 1, 1), std::overflow_error);
    EXPECT_THROW(planPleReads({{0, 0, std::numeric_limits<size_t>::max(), 0}}, {maxOffset}, 1, 1), std::out_of_range);
}

TEST(PleReadPlanTest, UnaddressablePlanSizeIsRejectedBeforeAllocation)
{
    auto const end = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(planPleReads({{0, 0, static_cast<size_t>(end), 0}}, {end}, 1, 1), std::length_error);
}

TEST(PleReadPlanTest, ReconstructionAcrossManyAlignmentsAndBudgets)
{
    std::vector<PleReadSpan> const spans{
        {1, 81, 40, 0}, {0, 7, 37, 43}, {1, 0, 3, 82}, {0, 16, 24, 91}, {0, 170, 7, 118}, {1, 118, 9, 130}};
    for (size_t const alignment : {1, 3, 8, 32, 64})
    {
        for (size_t const multiplier : {1, 2, 5})
        {
            SCOPED_TRACE(::testing::Message() << alignment << "/" << multiplier);
            size_t const budget = alignment * multiplier;
            auto const plan = planPleReads(spans, {183, 127}, alignment, budget);
            verifyReconstruction(spans, {183, 127}, alignment, budget, plan);
        }
    }
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
