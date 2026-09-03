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

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <algorithm>
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

std::vector<int32_t> copyDeviceTable(KVPageTable const& table, int32_t maxBatch, int32_t maxPagesPerSeq)
{
    std::vector<int32_t> result(static_cast<size_t>(maxBatch) * 2 * maxPagesPerSeq);
    CUDA_CHECK(cudaMemcpy(
        result.data(), table.kernelView().rawPointer(), result.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return result;
}
} // namespace

TEST(KVPageTableTest, IdentityLayout)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 4;
    constexpr int32_t numPages = maxBatch * maxPagesPerSeq;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();
    EXPECT_TRUE(table.isIdentity());

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

// numPages may include extra retained pages beyond maxBatch*maxPagesPerSeq, the minimum active
// pages. Identity K ids still occupy [0, minimumActivePages), while V ids shift by the configured
// numPages into the physical V-half.
TEST(KVPageTableTest, IdentityLayoutWithExtraRetainedPages)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 4;
    constexpr int32_t minimumActivePages = maxBatch * maxPagesPerSeq;
    constexpr int32_t numPages = minimumActivePages + 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();

    for (int32_t b = 0; b < maxBatch; ++b)
    {
        for (int32_t j = 0; j < maxPagesPerSeq; ++j)
        {
            int32_t const kId = b * maxPagesPerSeq + j;
            EXPECT_EQ(hostEntry(table, b, 0, j, maxPagesPerSeq), kId) << "b=" << b << " j=" << j;
            ASSERT_LT(kId, minimumActivePages) << "identity K ids must stay within the minimum active pages";
            EXPECT_EQ(hostEntry(table, b, 1, j, maxPagesPerSeq), kId + numPages)
                << "V id must shift by the configured numPages: b=" << b << " j=" << j;
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

TEST(KVPageTableTest, SetRowsAppliesMixedRowsAndClearsEmptyRows)
{
    constexpr int32_t maxBatch = 3;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 12;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();
    std::vector<int32_t> const row0{8, 9};
    std::vector<int32_t> const row2{4};
    table.setRows({
        KVPageTableRowUpdate{0, row0.data(), static_cast<int32_t>(row0.size())},
        KVPageTableRowUpdate{1, nullptr, 0},
        KVPageTableRowUpdate{2, row2.data(), static_cast<int32_t>(row2.size())},
    });

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), 8);
    EXPECT_EQ(hostEntry(table, 0, 0, 1, maxPagesPerSeq), 9);
    EXPECT_EQ(hostEntry(table, 0, 1, 0, maxPagesPerSeq), 8 + numPages);
    EXPECT_EQ(hostEntry(table, 0, 0, 2, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 0, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 1, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 2, 0, 0, maxPagesPerSeq), 4);
    EXPECT_EQ(hostEntry(table, 2, 1, 0, maxPagesPerSeq), 4 + numPages);
    EXPECT_FALSE(table.isIdentity());

    std::string error;
    EXPECT_TRUE(table.checkInvariants(error)) << error;
}

TEST(KVPageTableTest, SetRowsRejectsInvalidBatchWithoutPartialMutation)
{
    constexpr int32_t maxBatch = 2;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const original{1};
    table.setRow(0, original.data(), static_cast<int32_t>(original.size()));
    std::vector<int32_t> const replacement{2, 3};
    std::vector<int32_t> const invalid{numPages};

    EXPECT_THROW(table.setRows({
                     KVPageTableRowUpdate{0, replacement.data(), static_cast<int32_t>(replacement.size())},
                     KVPageTableRowUpdate{1, invalid.data(), static_cast<int32_t>(invalid.size())},
                 }),
        std::runtime_error);

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), 1);
    EXPECT_EQ(hostEntry(table, 0, 0, 1, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
    EXPECT_EQ(hostEntry(table, 1, 0, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
}

TEST(KVPageTableTest, SetRowsRejectsDuplicateSlotsWithoutMutation)
{
    constexpr int32_t maxPagesPerSeq = 3;
    KVPageTable table(/*maxBatch=*/2, maxPagesPerSeq, /*numPages=*/8);
    std::vector<int32_t> const first{2};
    std::vector<int32_t> const second{5};

    EXPECT_THROW(table.setRows({
                     KVPageTableRowUpdate{0, first.data(), static_cast<int32_t>(first.size())},
                     KVPageTableRowUpdate{0, second.data(), static_cast<int32_t>(second.size())},
                 }),
        std::runtime_error);

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
}

TEST(KVPageTableTest, SetRowRejectsInvalidPageDescriptionsWithoutMutation)
{
    constexpr int32_t maxBatch = 1;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 5;

    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const original{1};
    table.setRow(0, original.data(), static_cast<int32_t>(original.size()));
    std::vector<int32_t> const negative{-2};
    std::vector<int32_t> const tooLarge{numPages};
    std::vector<int32_t> const gap{1, kUNUSED_PAGE_ENTRY, 2};

    EXPECT_THROW(table.setRow(0, negative.data(), static_cast<int32_t>(negative.size())), std::runtime_error);
    EXPECT_THROW(table.setRow(0, tooLarge.data(), static_cast<int32_t>(tooLarge.size())), std::runtime_error);
    EXPECT_THROW(table.setRow(0, gap.data(), static_cast<int32_t>(gap.size())), std::runtime_error);
    EXPECT_THROW(table.setRow(0, nullptr, 1), std::runtime_error);

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), 1);
    EXPECT_EQ(hostEntry(table, 0, 0, 1, maxPagesPerSeq), kUNUSED_PAGE_ENTRY);
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
    EXPECT_FALSE(table.isIdentity());

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

TEST(KVPageTableTest, CompactRowsRejectsInvalidMappingsWithoutMutation)
{
    constexpr int32_t maxPagesPerSeq = 2;
    KVPageTable table(/*maxBatch=*/3, maxPagesPerSeq, /*numPages=*/8);
    std::vector<int32_t> const original{6, 7};
    table.setRow(0, original.data(), static_cast<int32_t>(original.size()));

    EXPECT_THROW(table.compactRows({0, 0}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({1}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({-1}, 1), std::runtime_error);
    EXPECT_THROW(table.compactRows({0, 1, 2, 3}, 4), std::runtime_error);

    EXPECT_EQ(hostEntry(table, 0, 0, 0, maxPagesPerSeq), 6);
    EXPECT_EQ(hostEntry(table, 0, 0, 1, maxPagesPerSeq), 7);
}

TEST(KVPageTableTest, UploadUsesStableDeviceStorageAndSkipsCleanTable)
{
    constexpr int32_t maxBatch = 2;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = maxBatch * maxPagesPerSeq;
    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();
    void const* const deviceAddress = table.kernelView().rawPointer();

    EXPECT_TRUE(table.upload(/*stream=*/nullptr));
    EXPECT_FALSE(table.upload(/*stream=*/nullptr));
    EXPECT_EQ(table.kernelView().rawPointer(), deviceAddress);

    std::vector<int32_t> const device = copyDeviceTable(table, maxBatch, maxPagesPerSeq);
    EXPECT_EQ(device[0], 0);
    EXPECT_EQ(device[static_cast<size_t>(maxPagesPerSeq)], numPages);
}

TEST(KVPageTableTest, UploadCopiesOnlyDirtyRows)
{
    constexpr int32_t maxBatch = 2;
    constexpr int32_t maxPagesPerSeq = 2;
    constexpr int32_t numPages = maxBatch * maxPagesPerSeq;
    constexpr int32_t marker = -99;
    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    table.setIdentity();
    ASSERT_TRUE(table.upload(/*stream=*/nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> const markerRow(static_cast<size_t>(2 * maxPagesPerSeq), marker);
    int32_t* const device = table.kernelView().dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpy(
        device + 2 * maxPagesPerSeq, markerRow.data(), markerRow.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> const row0{2};
    table.setRow(0, row0.data(), static_cast<int32_t>(row0.size()));
    ASSERT_TRUE(table.upload(/*stream=*/nullptr));
    std::vector<int32_t> const uploaded = copyDeviceTable(table, maxBatch, maxPagesPerSeq);

    EXPECT_EQ(uploaded[0], 2);
    EXPECT_EQ(uploaded[static_cast<size_t>(maxPagesPerSeq)], 2 + numPages);
    size_t const row1Offset = static_cast<size_t>(2 * maxPagesPerSeq);
    EXPECT_TRUE(
        std::all_of(uploaded.begin() + row1Offset, uploaded.end(), [](int32_t value) { return value == marker; }));
}

TEST(KVPageTableTest, IdenticalRowUpdateDoesNotUpload)
{
    constexpr int32_t maxPagesPerSeq = 3;
    KVPageTable table(/*maxBatch=*/1, maxPagesPerSeq, /*numPages=*/8);
    std::vector<int32_t> const row{1, 5};

    table.setRow(/*slot=*/0, row.data(), static_cast<int32_t>(row.size()));
    ASSERT_TRUE(table.upload(/*stream=*/nullptr));

    table.setRow(/*slot=*/0, row.data(), static_cast<int32_t>(row.size()));
    EXPECT_FALSE(table.upload(/*stream=*/nullptr));
}

TEST(KVPageTableTest, BackToBackUploadPreservesStagingLifetime)
{
    constexpr int32_t maxBatch = 1;
    constexpr int32_t maxPagesPerSeq = 3;
    constexpr int32_t numPages = 8;
    KVPageTable table(maxBatch, maxPagesPerSeq, numPages);
    std::vector<int32_t> const first{1, 2};
    std::vector<int32_t> const second{5, 6, 7};

    table.setRow(0, first.data(), static_cast<int32_t>(first.size()));
    ASSERT_TRUE(table.upload(/*stream=*/nullptr));
    table.setRow(0, second.data(), static_cast<int32_t>(second.size()));
    ASSERT_TRUE(table.upload(/*stream=*/nullptr));

    std::vector<int32_t> const device = copyDeviceTable(table, maxBatch, maxPagesPerSeq);
    EXPECT_EQ(device[0], 5);
    EXPECT_EQ(device[1], 6);
    EXPECT_EQ(device[2], 7);
    EXPECT_EQ(device[static_cast<size_t>(maxPagesPerSeq)], 5 + numPages);
}
