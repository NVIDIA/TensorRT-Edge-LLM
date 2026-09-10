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

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

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
    constexpr int32_t maxBatch{2};
    constexpr int32_t maxPagesPerSeq{3};
    constexpr int32_t numPages{8};
    int32_t const terminalLength = rt::kTOKENS_PER_PAGE - 1;
    std::vector<int32_t> const terminalPages{5};
    std::vector<int32_t> const survivorPages{7, 2};

    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, numPages);
    pageTable.setRow(0, terminalPages.data(), static_cast<int32_t>(terminalPages.size()));
    pageTable.setRow(1, survivorPages.data(), static_cast<int32_t>(survivorPages.size()));
    rt::ActionKvBatchCollector collector(maxBatch, maxPagesPerSeq, numPages);
    collector.beginRequest({true, false}, {101, 202});

    rt::Tensor lengths = makeDeviceLengths({terminalLength, rt::kTOKENS_PER_PAGE + 9});
    collector.captureFinished(pageTable, lengths, {1, 0}, {0, 1}, mStream);
    pageTable.compactRows({-1, 0}, 1);
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
    EXPECT_EQ(std::vector<int32_t>(batch.pageTable.hostRow(0), batch.pageTable.hostRow(0) + 1), terminalPages);
}

TEST_F(ActionKvBatchMaterializationTest, RejectsAMissingActionTerminalSnapshot)
{
    rt::KVPageTable pageTable(/*maxBatch=*/2, /*maxPagesPerSeq=*/2, /*numPages=*/4);
    pageTable.setIdentity();
    rt::ActionKvBatchCollector collector(/*maxBatch=*/2, /*maxPagesPerSeq=*/2, /*numPages=*/4);
    collector.beginRequest({true, true}, {101, 202});

    rt::Tensor lengths = makeDeviceLengths({17, 23});
    collector.captureFinished(pageTable, lengths, /*finished=*/{1, 0}, /*originalIndices=*/{0, 1}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    EXPECT_THROW(static_cast<void>(collector.materialize(mStream)), std::runtime_error);
}

TEST_F(ActionKvBatchMaterializationTest, MaterializesMixedActionRowsAfterMultipleLogicalEvictions)
{
    constexpr int32_t maxBatch{3};
    constexpr int32_t maxPagesPerSeq{3};
    constexpr int32_t numPages{12};
    std::vector<int32_t> const action0Pages{9, 1};
    std::vector<int32_t> const textPages{8, 3};
    std::vector<int32_t> const action2Pages{11, 5, 0};

    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, numPages);
    pageTable.setRow(0, action0Pages.data(), static_cast<int32_t>(action0Pages.size()));
    pageTable.setRow(1, textPages.data(), static_cast<int32_t>(textPages.size()));
    pageTable.setRow(2, action2Pages.data(), static_cast<int32_t>(action2Pages.size()));

    rt::ActionKvBatchCollector collector(maxBatch, maxPagesPerSeq, numPages);
    collector.beginRequest({true, false, true}, {111, 222, 333});

    int32_t const action0Length = rt::kTOKENS_PER_PAGE + 7;
    rt::Tensor firstLengths
        = makeDeviceLengths({action0Length, rt::kTOKENS_PER_PAGE + 9, rt::kTOKENS_PER_PAGE * 2 + 17});
    collector.captureFinished(pageTable, firstLengths, /*finished=*/{1, 0, 0},
        /*originalIndices=*/{0, 1, 2}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    pageTable.compactRows({-1, 0, 1}, 2);
    rt::Tensor secondLengths = makeDeviceLengths({rt::kTOKENS_PER_PAGE + 9, rt::kTOKENS_PER_PAGE * 2 + 17});
    collector.captureFinished(pageTable, secondLengths, /*finished=*/{1, 0}, /*originalIndices=*/{1, 2}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    pageTable.compactRows({-1, 0}, 1);
    int32_t const action2Length = rt::kTOKENS_PER_PAGE * 2 + 1;
    rt::Tensor thirdLengths = makeDeviceLengths({action2Length});
    collector.captureFinished(pageTable, thirdLengths, /*finished=*/{1}, /*originalIndices=*/{2}, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    collector.completeCapture();

    rt::ActionKvBatchView const batch = collector.materialize(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    EXPECT_EQ(batch.batchSize, 2);
    EXPECT_EQ(batch.mropeDeltas, (std::vector<int64_t>{111, 333}));
    ASSERT_EQ(batch.kvLengthsHost.getShape().volume(), 2);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[0], action0Length);
    EXPECT_EQ(batch.kvLengthsHost.dataPointer<int32_t>()[1], action2Length);
    EXPECT_EQ(std::vector<int32_t>(batch.pageTable.hostRow(0), batch.pageTable.hostRow(0) + 2), action0Pages);
    EXPECT_EQ(std::vector<int32_t>(batch.pageTable.hostRow(1), batch.pageTable.hostRow(1) + 3), action2Pages);

    EXPECT_EQ(collector.originalRequestIndices(), (std::vector<int32_t>{0, 2}));

    collector.beginRequest({true, false, false}, {444, 555, 666});
    EXPECT_THROW(static_cast<void>(collector.materialize(mStream)), std::runtime_error)
        << "the next request must not reuse the previous request's frozen action rows";
}

} // namespace
