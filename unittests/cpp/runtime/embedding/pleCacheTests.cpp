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

#include "runtime/embedding/pleCache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

static_assert(!std::is_copy_constructible_v<PleCache::RequestLease>);
static_assert(std::is_nothrow_move_constructible_v<PleCache::RequestLease>);
static_assert(!std::is_copy_constructible_v<PleCache::BatchReservation>);

std::vector<PleCache::RequestLease> complete(PleCache::BatchReservation& batch)
{
    for (auto const& miss : batch.misses())
    {
        batch.markReady(miss.slot);
    }
    return batch.commit();
}

TEST(PleCacheTest, ExactCapacitySharesMissesAndPreservesOrderedPadding)
{
    PleCache cache(100, 4);
    auto batch = cache.reserveBatch({{10, -1, 11, 10}, {11, 12, 11}});
    ASSERT_EQ(batch.misses().size(), 3);
    auto leases = complete(batch);
    ASSERT_EQ(leases.size(), 2);
    auto const& first = leases[0].mapping();
    auto const& second = leases[1].mapping();
    EXPECT_EQ(first.size(), 4);
    EXPECT_EQ(first[0], first[3]);
    EXPECT_EQ(first[1], 0);
    EXPECT_NE(first[0], 0);
    EXPECT_EQ(first[2], second[0]);
    EXPECT_EQ(second[0], second[2]);
    EXPECT_NE(second[1], first[0]);
    auto const saved = first;
    leases.clear();
    auto cached = cache.reserveBatch({{10, -1, 11, 10}});
    EXPECT_TRUE(cached.misses().empty());
    auto reused = cached.commit();
    EXPECT_EQ(reused[0].mapping(), saved);
}

TEST(PleCacheTest, CapacityAndRowFailuresDoNotMutateExistingCache)
{
    PleCache cache(100, 4);
    auto initial = cache.reserveBatch({{10, 11, 12}});
    auto leases = complete(initial);
    auto const original = leases[0].mapping();
    leases.clear();

    try
    {
        cache.reserveBatch({{0, 1, 0, -1}, {1, 2, 3, -1}});
        FAIL() << "The execution batch exceeds usable cache capacity";
    }
    catch (std::length_error const& error)
    {
        std::string const message = error.what();
        EXPECT_NE(message.find("requires 4 distinct valid rows"), std::string::npos);
        EXPECT_NE(message.find("3 usable slots"), std::string::npos);
    }
    EXPECT_THROW(cache.reserveBatch({{1}, {100}}), std::out_of_range);
    EXPECT_THROW(cache.reserveBatch({{1, -2}}), std::out_of_range);
    auto preserved = cache.reserveBatch({{10, 11, 12}});
    EXPECT_TRUE(preserved.misses().empty());
    auto after = preserved.commit();
    EXPECT_EQ(after[0].mapping(), original);
}

TEST(PleCacheTest, IndependentRequestLeasesSurviveAnotherRequestReleaseAndMove)
{
    PleCache cache(100, 3);
    auto batch = cache.reserveBatch({{7, 7}, {7, 8}});
    auto leases = complete(batch);
    auto const slot = leases[0].mapping()[0];
    leases[0].release();
    leases[0].release();
    EXPECT_THROW(leases[0].mapping(), std::logic_error);
    EXPECT_EQ(leases[1].mapping()[0], slot);
    EXPECT_THROW(cache.reserveBatch({{9}}), std::logic_error);
    auto remaining = std::move(leases[1]);
    leases.clear();
    EXPECT_EQ(remaining.mapping()[0], slot);
    EXPECT_THROW(cache.reserveBatch({{9}}), std::logic_error);
    remaining.release();
    auto cached = cache.reserveBatch({{7, 8}});
    EXPECT_TRUE(cached.misses().empty());
    auto again = cached.commit();
    EXPECT_EQ(again[0].mapping()[0], slot);
}

TEST(PleCacheTest, ProtectsLaterRequestHitBeforeEvictingForEarlierMiss)
{
    PleCache cache(100, 4);
    auto initial = cache.reserveBatch({{10, 11, 12}});
    auto initialLeases = complete(initial);
    auto const protectedSlot = initialLeases[0].mapping()[0];
    initialLeases.clear();

    auto batch = cache.reserveBatch({{0}, {10}});
    ASSERT_EQ(batch.misses().size(), 1);
    EXPECT_EQ(batch.misses()[0].row, 0);
    EXPECT_NE(batch.misses()[0].slot, protectedSlot);
    auto leases = complete(batch);
    EXPECT_NE(leases[0].mapping()[0], 0);
    EXPECT_EQ(leases[1].mapping()[0], protectedSlot);
}

TEST(PleCacheTest, ClockTerminatesWhenEveryUnpinnedRowNeedsASecondChance)
{
    PleCache cache(200, 3);
    for (int64_t row = 0; row < 100; row += 2)
    {
        auto batch = cache.reserveBatch({{row, row + 1}});
        ASSERT_EQ(batch.misses().size(), 2);
        auto leases = complete(batch);
        EXPECT_NE(leases[0].mapping()[0], leases[0].mapping()[1]);
        EXPECT_NE(leases[0].mapping()[0], 0);
        EXPECT_NE(leases[0].mapping()[1], 0);
    }
}

TEST(PleCacheTest, PartialFillRollbackRemovesNewRowsAndPreservesProtectedHits)
{
    PleCache cache(100, 4);
    auto initial = cache.reserveBatch({{1, 8, 9}});
    auto initialLeases = complete(initial);
    auto const hitSlot = initialLeases[0].mapping()[0];
    initialLeases.clear();
    {
        auto batch = cache.reserveBatch({{1, 2}, {2, 3}});
        ASSERT_EQ(batch.misses().size(), 2);
        EXPECT_THROW(batch.markReady(hitSlot), std::logic_error);
        EXPECT_THROW(batch.markReady(0), std::logic_error);
        EXPECT_THROW(batch.markReady(-1), std::logic_error);
        EXPECT_THROW(batch.markReady(4), std::logic_error);
        batch.markReady(batch.misses()[0].slot);
        EXPECT_THROW(batch.markReady(batch.misses()[0].slot), std::logic_error);
        EXPECT_THROW(batch.commit(), std::logic_error);
    }
    auto retry = cache.reserveBatch({{1, 2, 3}});
    ASSERT_EQ(retry.misses().size(), 2);
    EXPECT_EQ(retry.misses()[0].row, 2);
    EXPECT_EQ(retry.misses()[1].row, 3);
    auto leases = complete(retry);
    EXPECT_EQ(leases[0].mapping()[0], hitSlot);
}

TEST(PleCacheTest, MovedReservationRollsBackOnceAndAllowsRetry)
{
    PleCache cache(100, 2);
    {
        auto source = cache.reserveBatch({{8}});
        auto destination = std::move(source);
        destination.markReady(destination.misses()[0].slot);
        destination.rollback();
        destination.rollback();
        EXPECT_THROW(destination.commit(), std::logic_error);
    }
    auto retry = cache.reserveBatch({{8}});
    ASSERT_EQ(retry.misses().size(), 1);
    auto leases = complete(retry);
    EXPECT_NE(leases[0].mapping()[0], 0);
}

TEST(PleCacheTest, MoveAssignmentReleasesThePreviousOwner)
{
    PleCache first(100, 2);
    PleCache second(100, 2);
    auto firstBatch = first.reserveBatch({{8}});
    auto secondBatch = second.reserveBatch({{9}});
    firstBatch = std::move(secondBatch);
    EXPECT_THROW(secondBatch.commit(), std::logic_error);
    auto firstRetry = first.reserveBatch({{8}});
    ASSERT_EQ(firstRetry.misses().size(), 1);
    auto firstLeases = complete(firstRetry);
    auto secondLeases = complete(firstBatch);
    firstLeases[0] = std::move(secondLeases[0]);
    EXPECT_THROW(secondLeases[0].mapping(), std::logic_error);
    auto firstCached = first.reserveBatch({{8}});
    EXPECT_TRUE(firstCached.misses().empty());
    EXPECT_THROW(second.reserveBatch({{9}}), std::logic_error);
    firstLeases[0].release();
    auto secondCached = second.reserveBatch({{9}});
    EXPECT_TRUE(secondCached.misses().empty());
}

TEST(PleCacheTest, EvictionRequiresAReResolvedMapForTheNextBatch)
{
    PleCache cache(100, 3);
    auto initial = cache.reserveBatch({{10, 11}});
    auto initialLeases = complete(initial);
    auto const previousSlots = initialLeases[0].mapping();
    initialLeases.clear();
    auto replacement = cache.reserveBatch({{12, 13}});
    auto replacementLeases = complete(replacement);
    auto const replacementSlots = replacementLeases[0].mapping();
    EXPECT_NE(replacementSlots[0], replacementSlots[1]);
    for (int32_t const slot : replacementSlots)
    {
        EXPECT_NE(std::find(previousSlots.begin(), previousSlots.end(), slot), previousSlots.end());
    }
    replacementLeases.clear();
    auto next = cache.reserveBatch({{12, 10}});
    ASSERT_EQ(next.misses().size(), 1);
    EXPECT_EQ(next.misses()[0].row, 10);
    auto nextLeases = complete(next);
    EXPECT_EQ(nextLeases[0].mapping()[0], replacementSlots[0]);
    EXPECT_NE(nextLeases[0].mapping()[1], nextLeases[0].mapping()[0]);
    EXPECT_NE(std::find(previousSlots.begin(), previousSlots.end(), nextLeases[0].mapping()[1]), previousSlots.end());
}

TEST(PleCacheTest, Int64RowsAndZeroOnlyCapacityHaveDistinctPaddingSemantics)
{
    int64_t const largeRow = int64_t{1} << 40;
    PleCache cache(largeRow + 1, 2);
    auto batch = cache.reserveBatch({{largeRow, -1, largeRow}});
    ASSERT_EQ(batch.misses().size(), 1);
    EXPECT_EQ(batch.misses()[0].row, largeRow);
    auto leases = complete(batch);
    EXPECT_EQ(leases[0].mapping(), (std::vector<int32_t>{1, 0, 1}));

    PleCache zeroOnly(10, 1);
    EXPECT_THROW(zeroOnly.reserveBatch({{0}}), std::length_error);
    auto padded = zeroOnly.reserveBatch({{-1, -1}, {}});
    EXPECT_TRUE(padded.misses().empty());
    auto paddingLeases = padded.commit();
    EXPECT_EQ(paddingLeases[0].mapping(), (std::vector<int32_t>{0, 0}));
    EXPECT_TRUE(paddingLeases[1].mapping().empty());
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
