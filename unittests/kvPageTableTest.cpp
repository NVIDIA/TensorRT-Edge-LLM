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

#include "runtime/state/kvPageTable.h"

#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{
//! Read entry [slot][kOrV][j] out of a KVPageTable's host-visible K row.
//! kOrV == 0 reads the K row directly; kOrV == 1 reads the adjacent V row
//! (hostRow only exposes K, so the V half is read via the raw K pointer offset).
int32_t hostEntry(KVPageTable const& table, int32_t slot, int32_t kOrV, int32_t j, int32_t maxPagesPerSeq)
{
    int32_t const* kRow = table.hostRow(slot);
    return kOrV == 0 ? kRow[j] : kRow[maxPagesPerSeq + j];
}
} // namespace

TEST(KVPageTableTest, IdentityLayout)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 4;
    constexpr int32_t numPages = maxBatch * maxPagesPerSeq;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();

    for (int32_t b = 0; b < maxBatch; ++b)
    {
        for (int32_t j = 0; j < maxPagesPerSeq; ++j)
        {
            EXPECT_EQ(hostEntry(table, b, 0, j, maxPagesPerSeq), b * maxPagesPerSeq + j) << "b=" << b << " j=" << j;
            EXPECT_EQ(hostEntry(table, b, 1, j, maxPagesPerSeq), b * maxPagesPerSeq + j + numPages)
                << "b=" << b << " j=" << j;
        }
    }

    std::string error;
    EXPECT_TRUE(table.checkInvariants(error)) << error;
}

// Robustness: numPages may exceed maxBatch*maxPagesPerSeq (the
// active-capacity floor) when the pool has retention headroom. Identity K ids must still occupy
// exactly [0, floor) (unaffected); the derived V ids must shift by the real (larger) numPages, not
// the floor, so they land in the pool's actual V-half [numPages, numPages+floor).
TEST(KVPageTableTest, IdentityLayoutWithRetentionPagesBeyondFloor)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 4;
    constexpr int32_t floorPages = maxBatch * maxPagesPerSeq;
    constexpr int32_t numPages = floorPages + 5; // retention headroom beyond the floor

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();

    for (int32_t b = 0; b < maxBatch; ++b)
    {
        for (int32_t j = 0; j < maxPagesPerSeq; ++j)
        {
            int32_t const kId = b * maxPagesPerSeq + j;
            EXPECT_EQ(hostEntry(table, b, 0, j, maxPagesPerSeq), kId) << "b=" << b << " j=" << j;
            ASSERT_LT(kId, floorPages) << "identity K ids must stay within the floor";
            EXPECT_EQ(hostEntry(table, b, 1, j, maxPagesPerSeq), kId + numPages)
                << "V id must shift by the real numPages, not the floor: b=" << b << " j=" << j;
        }
    }

    std::string error;
    EXPECT_TRUE(table.checkInvariants(error)) << error;
}

TEST(KVPageTableTest, SetRowPartialCountLeavesSentinelTailInBothHalves)
{
    constexpr int32_t maxBatch = 2;
    constexpr int32_t maxPagesPerSeq = 4;
    constexpr int32_t numPages = 10;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const kIds = {3, 7};
    table.setRow(1, kIds.data(), static_cast<int32_t>(kIds.size()));

    // Populated entries: K verbatim, V = K + numPages.
    EXPECT_EQ(hostEntry(table, 1, 0, 0, maxPagesPerSeq), 3);
    EXPECT_EQ(hostEntry(table, 1, 1, 0, maxPagesPerSeq), 3 + numPages);
    EXPECT_EQ(hostEntry(table, 1, 0, 1, maxPagesPerSeq), 7);
    EXPECT_EQ(hostEntry(table, 1, 1, 1, maxPagesPerSeq), 7 + numPages);

    // Tail entries beyond count are the sentinel in both halves.
    EXPECT_EQ(hostEntry(table, 1, 0, 2, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 1, 2, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 0, 3, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 1, 3, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);

    std::string error;
    EXPECT_TRUE(table.checkInvariants(error)) << error;
}

TEST(KVPageTableTest, CheckInvariantsRejectsNegativeOutOfRangeId)
{
    constexpr int32_t maxBatch = 1;
    constexpr int32_t maxPagesPerSeq = 2;
    constexpr int32_t numPages = 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const kIds = {-2};
    table.setRow(0, kIds.data(), static_cast<int32_t>(kIds.size()));

    std::string error;
    EXPECT_FALSE(table.checkInvariants(error));
    EXPECT_FALSE(error.empty());
}

TEST(KVPageTableTest, CheckInvariantsRejectsIdAtOrAboveNumPages)
{
    constexpr int32_t maxBatch = 1;
    constexpr int32_t maxPagesPerSeq = 2;
    constexpr int32_t numPages = 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const kIds = {numPages};
    table.setRow(0, kIds.data(), static_cast<int32_t>(kIds.size()));

    std::string error;
    EXPECT_FALSE(table.checkInvariants(error));
    EXPECT_FALSE(error.empty());
}

TEST(KVPageTableTest, CheckInvariantsRejectsLiveEntryAfterSentinel)
{
    constexpr int32_t maxBatch = 1;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    // Fill a full row via setRow, then poke a live id back in behind a sentinel
    // by writing a shorter row first (leaves index 1 and 2 as sentinel), then
    // overwriting index 2 through a second, longer-count setRow call is not
    // representative of real usage -- instead exercise the interleave directly
    // via two setRow calls that would only happen through a bug: this asserts
    // checkInvariants catches it regardless of how it happened.
    std::vector<int32_t> const kIds = {1};
    table.setRow(0, kIds.data(), static_cast<int32_t>(kIds.size()));
    // Row is now [1, -1, -1]. Directly reintroduce a live id after the sentinel
    // via a second setRow spanning the full row width, with a live id in the
    // trailing position and a sentinel in the middle.
    std::vector<int32_t> const kIdsWithGap = {1, kUNUSED_PAGE_ENTRY, 2};
    table.setRow(0, kIdsWithGap.data(), static_cast<int32_t>(kIdsWithGap.size()));

    std::string error;
    EXPECT_FALSE(table.checkInvariants(error));
    EXPECT_FALSE(error.empty());
}

TEST(KVPageTableTest, CompactRowsMovesBindingsWithoutRenumberingPhysicalPages)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 12;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const row0{7, 8};
    std::vector<int32_t> const row1{3};
    std::vector<int32_t> const row2{10, 11};
    table.setRow(0, row0.data(), static_cast<int32_t>(row0.size()));
    table.setRow(1, row1.data(), static_cast<int32_t>(row1.size()));
    table.setRow(2, row2.data(), static_cast<int32_t>(row2.size()));

    table.compactRows({-1, 1, 0}, 2);

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), 10);
    EXPECT_EQ(hostEntry(table, 0, 0, 1, maxPagesPerSeq), 11);
    EXPECT_EQ(hostEntry(table, 0, 1, 0, maxPagesPerSeq), 10 + numPages);
    EXPECT_EQ(hostEntry(table, 1, 0, 0, maxPagesPerSeq), 3);
    EXPECT_EQ(hostEntry(table, 1, 1, 0, maxPagesPerSeq), 3 + numPages);
    EXPECT_EQ(hostEntry(table, 2, 0, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 2, 1, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);

    std::string error;
    EXPECT_TRUE(table.checkInvariants(error)) << error;
}

TEST(KVPageTableTest, CompactRowsRejectsInvalidMappings)
{
    KVPageTable table(/*maxBatch=*/3, /*maxPagesPerSeq=*/2, /*numPages=*/8);

    EXPECT_THROW(table.compactRows({0, 0}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({1}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({-1}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({0, 1, 2, 3}, 4), std::runtime_error);
}
