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

#include "action/actionKvBatch.h"

#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "runtime/state/kvPageTable.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

namespace
{

class ActionKvBatchMaterializationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    rt::Tensor makeDeviceLengths(std::vector<int32_t> const& lengths)
    {
        rt::Tensor result({static_cast<int64_t>(lengths.size())}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(result, lengths);
        return result;
    }

    cudaStream_t mStream{};
};

TEST_F(ActionKvBatchMaterializationTest, PrefillTerminalSnapshotDeepCopiesItsLogicalRow)
{
    // Given a terminal prefix, when device lengths change, then capture preserves its length and prefix mapping.
    constexpr int32_t maxBatch{2};
    constexpr int32_t maxPagesPerSeq{3};
    constexpr int32_t numPages{12};
    int32_t const terminalLength = rt::kTOKENS_PER_PAGE - 1;
    std::vector<int32_t> const terminalPages{5};
    std::vector<int32_t> const survivorPages{7, 2};

    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, numPages);
    pageTable.setRow(0, terminalPages.data(), static_cast<int32_t>(terminalPages.size()));
    pageTable.setRow(1, survivorPages.data(), static_cast<int32_t>(survivorPages.size()));
    rt::ActionKvBatchCollector collector(maxBatch, maxPagesPerSeq + 1, numPages);
    collector.beginRequest({true, false}, {101, 202});

    rt::Tensor lengths = makeDeviceLengths({terminalLength, rt::kTOKENS_PER_PAGE + 9});
    collector.captureFinished(pageTable, lengths, {1, 0}, {0, 1}, {{0, 1}, {1, 1}}, mStream);
    std::vector<int32_t> const changedLengths{rt::kTOKENS_PER_PAGE + 41, 0};
    CUDA_CHECK(cudaMemcpyAsync(lengths.rawPointer(), changedLengths.data(), changedLengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    rt::ActionKvBatchView const batch = collector.materialize(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    ASSERT_EQ(batch.batchSize, 1);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[0], terminalLength);
    EXPECT_EQ(batch.mropeDeltas, (std::vector<int64_t>{101}));
    EXPECT_EQ(batch.pageTable.hostRow(0)[0], terminalPages[0]);
}

TEST_F(ActionKvBatchMaterializationTest, RejectsAMissingActionTerminalSnapshot)
{
    // Given an unfinished action sequence, when materializing, then its missing terminal snapshot is rejected.
    rt::KVPageTable pageTable(/*maxBatch=*/2, /*maxPagesPerSeq=*/2, /*numPages=*/8);
    pageTable.setIdentity();
    rt::ActionKvBatchCollector collector(2, 3, 8);
    collector.beginRequest({true, true}, {101, 202});

    rt::Tensor lengths = makeDeviceLengths({17, 23});
    collector.captureFinished(pageTable, lengths, /*finished=*/{1, 0}, /*originalIndices=*/{0, 1},
        /*residentRefs=*/{{0, 1}, {1, 1}}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    EXPECT_THROW(static_cast<void>(collector.materialize(mStream)), std::runtime_error);
}

TEST_F(ActionKvBatchMaterializationTest, MaterializesMixedActionRowsAfterMultipleLogicalEvictions)
{
    // Given mixed action/text slots, when they retire separately, then action rows retain original order and metadata.
    constexpr int32_t maxBatch{3};
    constexpr int32_t maxPagesPerSeq{3};
    constexpr int32_t numPages{18};
    std::vector<int32_t> const action0Pages{9, 1};
    std::vector<int32_t> const textPages{8, 3};
    std::vector<int32_t> const action2Pages{11, 5, 0};

    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, numPages);
    pageTable.setRow(0, action0Pages.data(), static_cast<int32_t>(action0Pages.size()));
    pageTable.setRow(1, textPages.data(), static_cast<int32_t>(textPages.size()));
    pageTable.setRow(2, action2Pages.data(), static_cast<int32_t>(action2Pages.size()));

    rt::ActionKvBatchCollector collector(maxBatch, maxPagesPerSeq + 1, numPages);
    collector.beginRequest({true, false, true}, {111, 222, 333});

    int32_t const action0Length = rt::kTOKENS_PER_PAGE + 7;
    rt::Tensor firstLengths
        = makeDeviceLengths({action0Length, rt::kTOKENS_PER_PAGE + 9, rt::kTOKENS_PER_PAGE * 2 + 17});
    collector.captureFinished(pageTable, firstLengths, /*finished=*/{1, 0, 0},
        /*originalIndices=*/{0, 1, 2}, /*residentRefs=*/{{0, 1}, {1, 1}, {2, 1}}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    rt::Tensor secondLengths = makeDeviceLengths({rt::kTOKENS_PER_PAGE + 9, rt::kTOKENS_PER_PAGE * 2 + 17});
    collector.captureFinished(pageTable, secondLengths, /*finished=*/{1, 0}, /*originalIndices=*/{1, 2},
        /*residentRefs=*/{{1, 1}, {2, 1}}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    int32_t const action2Length = rt::kTOKENS_PER_PAGE * 2 + 1;
    rt::Tensor thirdLengths = makeDeviceLengths({action2Length});
    collector.captureFinished(pageTable, thirdLengths, /*finished=*/{1}, /*originalIndices=*/{2},
        /*residentRefs=*/{{2, 1}}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    rt::ActionKvBatchView const batch = collector.materialize(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    EXPECT_EQ(batch.batchSize, 2);
    EXPECT_EQ(batch.mropeDeltas, (std::vector<int64_t>{111, 333}));
    ASSERT_EQ(batch.kvLengthsHost.getShape().volume(), 2);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[0], action0Length);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[1], action2Length);
    EXPECT_EQ(batch.pageTable.hostRow(0)[0], action0Pages[0]);
    EXPECT_EQ(batch.pageTable.hostRow(0)[1], action0Pages[1]);
    EXPECT_EQ(std::vector<int32_t>(batch.pageTable.hostRow(1), batch.pageTable.hostRow(1) + 2),
        (std::vector<int32_t>{action2Pages[0], action2Pages[1]}));
    EXPECT_EQ(batch.pageTable.hostRow(1)[2], action2Pages[2]);

    EXPECT_EQ(collector.originalRequestIndices(), (std::vector<int32_t>{0, 2}));

    collector.releaseMaterialized(mStream);
    collector.beginRequest({true, false, false}, {444, 555, 666});
    EXPECT_THROW(static_cast<void>(collector.materialize(mStream)), std::runtime_error)
        << "the next request must not reuse the previous request's frozen action rows";
}

TEST_F(ActionKvBatchMaterializationTest, SnapshotsResidentPageRowAfterLogicalCompaction)
{
    // Given a compacted logical batch, when capturing, then the resident mapping selects the prefix page row.
    constexpr int32_t maxBatch{3};
    constexpr int32_t maxPagesPerSeq{2};
    constexpr int32_t numPages{12};
    std::vector<int32_t> const retiredPages{0};
    std::vector<int32_t> const survivorPages{4, 5};

    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, numPages);
    pageTable.setRow(0, retiredPages.data(), static_cast<int32_t>(retiredPages.size()));
    pageTable.setRow(2, survivorPages.data(), static_cast<int32_t>(survivorPages.size()));

    rt::ActionKvBatchCollector collector(maxBatch, maxPagesPerSeq + 1, numPages);
    collector.beginRequest({false, true}, {101, 202});
    int32_t const terminalLength = rt::kTOKENS_PER_PAGE + 3;
    rt::Tensor lengths = makeDeviceLengths({terminalLength});

    collector.captureFinished(pageTable, lengths, /*finished=*/{1}, /*originalIndices=*/{1},
        /*residentRefs=*/{{2, 1}}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    rt::ActionKvBatchView const batch = collector.materialize(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    ASSERT_EQ(batch.batchSize, 1);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[0], terminalLength);
    EXPECT_EQ(batch.pageTable.hostRow(0)[0], survivorPages[0]);
    EXPECT_EQ(batch.pageTable.hostRow(0)[1], survivorPages[1]);
}

TEST_F(ActionKvBatchMaterializationTest, ManagedPrefixPinsRequireNoAdditionalLLMPages)
{
    // Given scrambled managed prefixes and a full LLM pool, when owners retire and action materializes,
    // then pins preserve all prefix IDs without acquiring replacement or tail pages.
    for (auto const& terminalLengths :
        std::vector<std::vector<int32_t>>{{127, 128}, {129, 300}, {300, 380}, {384, 384}})
    {
        constexpr int32_t numPages = 10;
        rt::ContextCacheManager pageManager(rt::kTOKENS_PER_PAGE, {numPages, 0, 0, 0}, 4);
        auto prefixLease = pageManager.acquireVanilla({}, (6) * rt::kTOKENS_PER_PAGE).lease;
        ASSERT_TRUE(prefixLease.has_value());
        std::array<std::vector<int32_t>, 2> const rows{{{2, 0, 1}, {5, 3, 4}}};
        rt::KVPageTable llmTable(2, 3, numPages);
        llmTable.setRow(0, rows[0].data(), 3);
        llmTable.setRow(1, rows[1].data(), 3);
        rt::ActionKvBatchCollector collector(2, 4, numPages, &pageManager);
        collector.beginRequest({true, true}, {7, -20});
        auto lengths = makeDeviceLengths(terminalLengths);
        collector.captureFinished(llmTable, lengths, {1, 1}, {0, 1}, {{0, 1}, {1, 1}}, mStream);
        prefixLease.reset();
        int32_t const livePages = (terminalLengths[0] + 127) / 128 + (terminalLengths[1] + 127) / 128;
        ASSERT_EQ(pageManager.pools().freeCount(rt::ResourceType::kBaseKvPage), numPages - livePages);
        auto pressureLease = pageManager.acquireVanilla({}, (numPages - livePages) * rt::kTOKENS_PER_PAGE).lease;
        ASSERT_TRUE(pressureLease.has_value());
        EXPECT_FALSE(pageManager.acquireVanilla({}, (1) * rt::kTOKENS_PER_PAGE).lease.has_value());

        auto const batch = collector.materialize(mStream);
        for (int32_t seq = 0; seq < 2; ++seq)
        {
            int32_t const count = (terminalLengths[seq] + 127) / 128;
            for (int32_t page = 0; page < count; ++page)
            {
                EXPECT_EQ(batch.pageTable.hostRow(seq)[page], rows[seq][page]);
                EXPECT_EQ(batch.pageTable.hostRow(seq)[page + 4], rows[seq][page] + numPages);
            }
            EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[seq], terminalLengths[seq]);
        }
        EXPECT_EQ(batch.mropeDeltas, (std::vector<int64_t>{7, -20}));
        EXPECT_TRUE(pageManager.records().lruToMru().empty());
        collector.releaseMaterialized(mStream);
        EXPECT_EQ(pageManager.pools().freeCount(rt::ResourceType::kBaseKvPage), livePages);
        pressureLease.reset();
        EXPECT_EQ(pageManager.pools().freeCount(rt::ResourceType::kBaseKvPage), numPages);
    }
}

TEST_F(ActionKvBatchMaterializationTest, ReleasesCapturedPinsWithoutMaterializing)
{
    // Given a captured prefix, when generation exits before materialization, then its pin is released.
    rt::ContextCacheManager pageManager(rt::kTOKENS_PER_PAGE, {3, 0, 0, 0}, 0);
    auto prefixLease = pageManager.acquireVanilla({}, (1) * rt::kTOKENS_PER_PAGE).lease;
    ASSERT_TRUE(prefixLease.has_value());
    rt::KVPageTable table(1, 1, 3);
    table.setRow(0, prefixLease->basePages().data(), 1);
    rt::ActionKvBatchCollector collector(1, 2, 3, &pageManager);
    collector.beginRequest({true}, {0});
    auto lengths = makeDeviceLengths({17});
    collector.captureFinished(table, lengths, {1}, {0}, {{0, 1}}, mStream);
    prefixLease.reset();
    ASSERT_EQ(pageManager.pools().freeCount(rt::ResourceType::kBaseKvPage), 2);
    collector.releaseMaterialized(mStream);
    EXPECT_EQ(pageManager.pools().freeCount(rt::ResourceType::kBaseKvPage), 3);
}

} // namespace
