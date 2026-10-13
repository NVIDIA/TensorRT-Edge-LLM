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

#include "runtime/state/boundedSwaKVPageManager.h"

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_BATCH{3};
constexpr int32_t kMAX_SEQUENCE_LENGTH{2048};
constexpr int32_t kDEFAULT_WINDOW_SIZE{kTOKENS_PER_PAGE + 1};

HybridCacheManager::Config makeCacheConfig(int32_t windowSize)
{
    int64_t const fullPages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
    int64_t const swaPages = computeMinimumSwaPoolPages(kMAX_BATCH, windowSize);
    KVCacheManager::Config kvConfig{
        /*.numAttentionLayers=*/2,
        /*.maxBatchSize=*/kMAX_BATCH,
        /*.maxSequenceLength=*/kMAX_SEQUENCE_LENGTH,
        /*.layerConfigs=*/
        {
            KVLayerConfig{/*.numKVHeads=*/1, /*.headDim=*/8},
            KVLayerConfig{/*.numKVHeads=*/1, /*.headDim=*/8, /*.kvCacheCapacity=*/windowSize},
        },
        /*.kvCacheType=*/DataType::kHALF,
        /*.numPages=*/static_cast<int32_t>(fullPages),
        /*.numSwaPages=*/static_cast<int32_t>(swaPages),
        /*.useBoundedSwaKVCache=*/true,
    };
    MambaCacheManager::Config mambaConfig{
        /*.numRecurrentLayers=*/1,
        /*.maxBatchSize=*/kMAX_BATCH,
        /*.recurrentStateNumHeads=*/1,
        /*.recurrentStateHeadDim=*/2,
        /*.recurrentStateSize=*/3,
        /*.convDim=*/4,
        /*.convKernel=*/2,
    };
    return HybridCacheManager::Config{
        /*.layerTypes=*/{
            HybridCacheManager::LayerType::kAttention,
            HybridCacheManager::LayerType::kAttention,
            HybridCacheManager::LayerType::kMamba,
        },
        /*.kvConfig=*/std::move(kvConfig),
        /*.mambaConfig=*/std::move(mambaConfig),
        /*.maxBatchSize=*/kMAX_BATCH,
    };
}

std::vector<int32_t> mappedLogicalPages(KVPageTable const& table, int32_t slot)
{
    std::vector<int32_t> pages;
    for (int32_t logicalPage = 0; logicalPage < table.maxPagesPerSeq(); ++logicalPage)
    {
        if (table.hostRow(slot)[logicalPage] != kUNUSED_PAGE_ENTRY)
        {
            pages.push_back(logicalPage);
        }
    }
    return pages;
}

std::vector<uint8_t> copyResidentBytes(Tensor const& tensor, int32_t slot, cudaStream_t stream)
{
    size_t const slotBytes = static_cast<size_t>(tensor.getShape().volume() / kMAX_BATCH) * sizeof(half);
    std::vector<uint8_t> result(slotBytes);
    auto const* source = static_cast<uint8_t const*>(tensor.rawPointer()) + static_cast<size_t>(slot) * slotBytes;
    EXPECT_EQ(cudaMemcpyAsync(result.data(), source, slotBytes, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    return result;
}

class BoundedSwaKVPageManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        initialize(kDEFAULT_WINDOW_SIZE);
    }

    void TearDown() override
    {
        releaseResources();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    void initialize(int32_t windowSize,
        BoundedSwaKVPageManager::StreamSynchronizer synchronizer = BoundedSwaKVPageManager::StreamSynchronizer{})
    {
        mWindowSize = windowSize;
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(windowSize), mStream);
        KVCacheManager const& kvCache = mCache->getKVCacheManager();
        mBasePageTable = std::make_unique<KVPageTable>(
            kMAX_BATCH, computeMaxPagesPerSeq(kMAX_SEQUENCE_LENGTH), kvCache.numPages());
        mBasePageTable->setIdentity();
        mBasePageTable->upload(mStream);
        mSwaPageTable = std::make_unique<KVPageTable>(kMAX_BATCH, computeMaxPagesPerSeq(kMAX_SEQUENCE_LENGTH),
            kvCache.numPages(/*attnLayerIdx=*/1), KVPageTable::Mode::kSparseWindow);
        mSwaPageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        mManager = std::make_unique<BoundedSwaKVPageManager>(
            *mCache, *mBasePageTable, *mSwaPageTable, mStream, std::move(synchronizer));
    }

    void releaseResources()
    {
        if (mManager != nullptr)
        {
            EXPECT_EQ(mManager->shutdown(), SwaKVCacheStatus::kOk);
            mManager.reset();
        }
        mSwaPageTable.reset();
        mBasePageTable.reset();
        mCache.reset();
    }

    BoundedSwaKVPageManager::RequestHandle begin(
        std::vector<int32_t> const& inputLengths, std::vector<ResidentRef> residents = {})
    {
        Tensor reuseLengths({static_cast<int64_t>(inputLengths.size())}, rt::DeviceType::kCPU, DataType::kINT32,
            "swaManagerTestReuseLengths");
        std::fill_n(reuseLengths.dataPointer<int32_t>(), inputLengths.size(), 0);
        mCache->resetForNewSequences(reuseLengths, mStream);
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

        if (residents.empty())
        {
            for (int32_t slot = 0; slot < static_cast<int32_t>(inputLengths.size()); ++slot)
            {
                residents.push_back(ResidentRef{slot, 1});
            }
        }
        BoundedSwaKVPageManager::BeginRequestResult admitted = mManager->beginRequest(residents, mStream);
        EXPECT_EQ(admitted.status, SwaKVCacheStatus::kOk);
        EXPECT_TRUE(admitted.request.has_value());
        return std::move(*admitted.request);
    }

    void completePrefill(
        BoundedSwaKVPageManager::RequestHandle& request, std::vector<int32_t> const& effectiveInputLengths)
    {
        ASSERT_EQ(mManager->preparePrefill(request, effectiveInputLengths), SwaKVCacheStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ASSERT_EQ(mManager->completePrefill(request), SwaKVCacheStatus::kOk);
    }

    cudaStream_t mStream{};
    int32_t mWindowSize{};
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mSwaPageTable;
    std::unique_ptr<BoundedSwaKVPageManager> mManager;
};

TEST_F(BoundedSwaKVPageManagerTest, AdmissionIsAtomicAndReturnsPrivatePages)
{
    int32_t const totalPages = mSwaPageTable->numPages();
    int32_t const pagesPerSequence = mManager->privatePagesPerSequence();
    EXPECT_EQ(totalPages, kMAX_BATCH * pagesPerSequence);

    EXPECT_THROW((void) mManager->beginRequest(std::vector<ResidentRef>(kMAX_BATCH + 1), mStream), std::runtime_error);
    EXPECT_EQ(mManager->freePageCount(), totalPages);

    BoundedSwaKVPageManager::RequestHandle request = begin(std::vector<int32_t>(kMAX_BATCH, 1));
    EXPECT_EQ(mManager->freePageCount(), 0);
    BoundedSwaKVPageManager::BeginRequestResult concurrent = mManager->beginRequest({ResidentRef{0, 1}}, mStream);
    EXPECT_EQ(concurrent.status, SwaKVCacheStatus::kRequestFailed);
    EXPECT_FALSE(concurrent.request.has_value());
    EXPECT_EQ(mManager->freePageCount(), 0);

    EXPECT_EQ(mManager->finish(request), SwaKVCacheStatus::kOk);
    EXPECT_EQ(mManager->freePageCount(), totalPages);

    {
        BoundedSwaKVPageManager::RequestHandle abandoned = begin({kTOKENS_PER_PAGE + 1});
        ASSERT_EQ(mManager->preparePrefill(abandoned, {kTOKENS_PER_PAGE + 1}), SwaKVCacheStatus::kOk);
    }
    EXPECT_EQ(mManager->freePageCount(), totalPages);
}

TEST_F(BoundedSwaKVPageManagerTest, PrefillRetainsAbsoluteSparseWindowAcrossPageGeometries)
{
    constexpr int32_t kINPUT_LENGTH{3 * kTOKENS_PER_PAGE + 1};
    std::vector<int32_t> const windowSizes{64, kTOKENS_PER_PAGE, 2 * kTOKENS_PER_PAGE, kDEFAULT_WINDOW_SIZE};
    for (int32_t const windowSize : windowSizes)
    {
        SCOPED_TRACE(windowSize);
        releaseResources();
        initialize(windowSize);

        BoundedSwaKVPageManager::RequestHandle request = begin({kINPUT_LENGTH});
        ASSERT_EQ(mManager->preparePrefill(request, {kINPUT_LENGTH}), SwaKVCacheStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

        int32_t const retainedBegin = std::max(0, kINPUT_LENGTH - windowSize) / kTOKENS_PER_PAGE;
        int32_t const retainedEnd = (kINPUT_LENGTH + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
        std::vector<int32_t> expectedPages(static_cast<size_t>(retainedEnd - retainedBegin));
        std::iota(expectedPages.begin(), expectedPages.end(), retainedBegin);
        EXPECT_EQ(mappedLogicalPages(*mSwaPageTable, 0), expectedPages);

        SwaKVCacheState const& staged = mManager->state(request, 0);
        EXPECT_EQ(staged.exactResidentTokenCount, 0);
        EXPECT_EQ(staged.pendingEndpoint, std::optional<int32_t>{kINPUT_LENGTH});
        EXPECT_EQ(staged.retainedLogicalPageBegin, retainedBegin);
        EXPECT_EQ(staged.retainedLogicalPageEnd, retainedEnd);
        EXPECT_EQ(staged.pageBindings.size(), expectedPages.size());
        EXPECT_EQ(mSwaPageTable->lastUploadEntryCount(), 2 * expectedPages.size());

        ASSERT_EQ(mManager->completePrefill(request), SwaKVCacheStatus::kOk);
        EXPECT_EQ(mManager->state(request, 0).exactResidentTokenCount, kINPUT_LENGTH);
        EXPECT_FALSE(mManager->state(request, 0).pendingEndpoint.has_value());
        EXPECT_EQ(mManager->finish(request), SwaKVCacheStatus::kOk);
        EXPECT_EQ(mManager->freePageCount(), mSwaPageTable->numPages());
    }
}

TEST_F(BoundedSwaKVPageManagerTest, DecodeRotationUsesConstantDirtyPatchesAndRecyclesPages)
{
    constexpr int32_t kINPUT_LENGTH{3 * kTOKENS_PER_PAGE};
    constexpr int32_t kDECODE_STEPS{2 * kDEFAULT_WINDOW_SIZE + 1};
    BoundedSwaKVPageManager::RequestHandle request = begin({kINPUT_LENGTH});
    completePrefill(request, {kINPUT_LENGTH});
    ASSERT_EQ(mappedLogicalPages(*mSwaPageTable, 0), std::vector<int32_t>({1, 2}));
    PageId const firstStalePage = mSwaPageTable->hostRow(0)[1];
    std::optional<PageId> recycledPage;

    for (int32_t step = 0; step < kDECODE_STEPS; ++step)
    {
        int32_t const nextEndpoint = kINPUT_LENGTH + step + 1;
        ASSERT_EQ(mManager->prepareDecodeStep(request), SwaKVCacheStatus::kOk);
        SwaKVCacheState const& staged = mManager->state(request, 0);
        EXPECT_EQ(staged.pendingEndpoint, std::optional<int32_t>{nextEndpoint});
        EXPECT_LE(staged.newPageBindings.size(), 1U);
        EXPECT_LE(staged.pendingRetireBindings.size(), 1U);
        EXPECT_LE(mSwaPageTable->lastUploadEntryCount(), 2U);
        EXPECT_LE(mSwaPageTable->lastUploadRangeCount(), 2U);
        if (!staged.newPageBindings.empty() && staged.newPageBindings.front().logicalPage == 4)
        {
            recycledPage = staged.newPageBindings.front().page;
        }

        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ASSERT_EQ(mManager->completeDecodeStep(request), SwaKVCacheStatus::kOk);
        SwaKVCacheState const& completed = mManager->state(request, 0);
        EXPECT_EQ(completed.exactResidentTokenCount, nextEndpoint);
        EXPECT_FALSE(completed.pendingEndpoint.has_value());
        EXPECT_LE(completed.pageBindings.size(),
            static_cast<size_t>((mWindowSize + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE + 1));
    }

    ASSERT_TRUE(recycledPage.has_value());
    EXPECT_EQ(*recycledPage, firstStalePage);
    EXPECT_GT(mManager->state(request, 0).exactResidentTokenCount, kINPUT_LENGTH + 2 * mWindowSize);
    EXPECT_EQ(mManager->finish(request), SwaKVCacheStatus::kOk);
}

TEST_F(BoundedSwaKVPageManagerTest, CompactionPreservesResidentRowsAndReleasesDroppedState)
{
    std::vector<int32_t> const inputLengths{3 * kTOKENS_PER_PAGE, 3 * kTOKENS_PER_PAGE + 1, 3 * kTOKENS_PER_PAGE + 2};
    BoundedSwaKVPageManager::RequestHandle request = begin(inputLengths);
    completePrefill(request, inputLengths);

    int32_t const rowWidth = mSwaPageTable->maxPagesPerSeq();
    std::vector<int32_t> const oldBaseRow0(mBasePageTable->hostRow(0), mBasePageTable->hostRow(0) + rowWidth);
    std::vector<int32_t> const oldBaseRow2(mBasePageTable->hostRow(2), mBasePageTable->hostRow(2) + rowWidth);
    std::vector<int32_t> const oldRow0(mSwaPageTable->hostRow(0), mSwaPageTable->hostRow(0) + rowWidth);
    std::vector<int32_t> const oldRow1(mSwaPageTable->hostRow(1), mSwaPageTable->hostRow(1) + rowWidth);
    std::vector<int32_t> const oldRow2(mSwaPageTable->hostRow(2), mSwaPageTable->hostRow(2) + rowWidth);
    Tensor& recurrent = mCache->getMambaCacheManager().getRecurrentState(0);
    Tensor& convolution = mCache->getMambaCacheManager().getConvState(0);
    size_t const recurrentSlotBytes = static_cast<size_t>(recurrent.getShape().volume() / kMAX_BATCH) * sizeof(half);
    size_t const convolutionSlotBytes
        = static_cast<size_t>(convolution.getShape().volume() / kMAX_BATCH) * sizeof(half);
    for (int32_t slot = 0; slot < kMAX_BATCH; ++slot)
    {
        EXPECT_EQ(cudaMemsetAsync(static_cast<uint8_t*>(recurrent.rawPointer()) + slot * recurrentSlotBytes,
                      0x11 * (slot + 1), recurrentSlotBytes, mStream),
            cudaSuccess);
        EXPECT_EQ(cudaMemsetAsync(static_cast<uint8_t*>(convolution.rawPointer()) + slot * convolutionSlotBytes,
                      0x11 * (slot + 1), convolutionSlotBytes, mStream),
            cudaSuccess);
    }
    std::vector<uint8_t> const recurrent0 = copyResidentBytes(recurrent, 0, mStream);
    std::vector<uint8_t> const recurrent2 = copyResidentBytes(recurrent, 2, mStream);
    std::vector<uint8_t> const convolution0 = copyResidentBytes(convolution, 0, mStream);
    std::vector<uint8_t> const convolution2 = copyResidentBytes(convolution, 2, mStream);
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "swaManagerTestBatchMapping");

    ASSERT_EQ(mManager->beginBatchCompaction(request, {0, -1, 1}, 2, deviceMapping), SwaKVCacheStatus::kOk);
    ASSERT_EQ(mManager->compactBatch(request), SwaKVCacheStatus::kOk);

    EXPECT_TRUE(std::equal(oldRow0.begin(), oldRow0.end(), mSwaPageTable->hostRow(0)));
    EXPECT_TRUE(std::equal(oldRow2.begin(), oldRow2.end(), mSwaPageTable->hostRow(2)));
    EXPECT_TRUE(std::equal(oldBaseRow0.begin(), oldBaseRow0.end(), mBasePageTable->hostRow(0)));
    EXPECT_TRUE(std::equal(oldBaseRow2.begin(), oldBaseRow2.end(), mBasePageTable->hostRow(2)));
    EXPECT_TRUE(std::all_of(mBasePageTable->hostRow(1), mBasePageTable->hostRow(1) + rowWidth,
        [](int32_t page) { return page == kUNUSED_PAGE_ENTRY; }));
    EXPECT_TRUE(std::all_of(mSwaPageTable->hostRow(1), mSwaPageTable->hostRow(1) + rowWidth,
        [](int32_t page) { return page == kUNUSED_PAGE_ENTRY; }));
    EXPECT_FALSE(std::equal(oldRow1.begin(), oldRow1.end(), mSwaPageTable->hostRow(2)));
    EXPECT_EQ(mManager->state(request, 0).exactResidentTokenCount, inputLengths[0]);
    EXPECT_EQ(mManager->state(request, 1).exactResidentTokenCount, inputLengths[2]);
    EXPECT_EQ(mManager->state(request, 0).resident.slot, 0);
    EXPECT_EQ(mManager->state(request, 1).resident.slot, 2);
    EXPECT_EQ(copyResidentBytes(recurrent, 0, mStream), recurrent0);
    EXPECT_EQ(copyResidentBytes(recurrent, 2, mStream), recurrent2);
    EXPECT_EQ(copyResidentBytes(convolution, 0, mStream), convolution0);
    EXPECT_EQ(copyResidentBytes(convolution, 2, mStream), convolution2);
    EXPECT_EQ(copyResidentBytes(recurrent, 1, mStream), std::vector<uint8_t>(recurrentSlotBytes, 0));
    EXPECT_EQ(copyResidentBytes(convolution, 1, mStream), std::vector<uint8_t>(convolutionSlotBytes, 0));
    EXPECT_EQ(mManager->freePageCount(), mManager->privatePagesPerSequence());

    ASSERT_EQ(mManager->prepareDecodeStep(request), SwaKVCacheStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mManager->completeDecodeStep(request), SwaKVCacheStatus::kOk);
    EXPECT_EQ(mManager->state(request, 0).exactResidentTokenCount, inputLengths[0] + 1);
    EXPECT_EQ(mManager->state(request, 1).exactResidentTokenCount, inputLengths[2] + 1);
    EXPECT_EQ(mappedLogicalPages(*mSwaPageTable, 0), std::vector<int32_t>({2, 3}));
    EXPECT_EQ(mappedLogicalPages(*mSwaPageTable, 2), std::vector<int32_t>({2, 3}));

    EXPECT_EQ(mManager->finish(request), SwaKVCacheStatus::kOk);
    EXPECT_EQ(mManager->freePageCount(), mSwaPageTable->numPages());
}

TEST_F(BoundedSwaKVPageManagerTest, FailedSynchronizationQuarantinesPagesUntilShutdown)
{
    releaseResources();
    int32_t synchronizeCalls{};
    initialize(kDEFAULT_WINDOW_SIZE, [&](cudaStream_t stream) {
        ++synchronizeCalls;
        return synchronizeCalls == 1 ? cudaErrorUnknown : cudaStreamSynchronize(stream);
    });
    int32_t const totalPages = mSwaPageTable->numPages();
    {
        BoundedSwaKVPageManager::RequestHandle request = begin({3 * kTOKENS_PER_PAGE + 1});
        ASSERT_EQ(mManager->preparePrefill(request, {3 * kTOKENS_PER_PAGE + 1}), SwaKVCacheStatus::kOk);
    }

    EXPECT_EQ(synchronizeCalls, 1);
    EXPECT_LT(mManager->freePageCount(), totalPages);
    BoundedSwaKVPageManager::BeginRequestResult poisoned = mManager->beginRequest({ResidentRef{0, 1}}, mStream);
    EXPECT_EQ(poisoned.status, SwaKVCacheStatus::kPoisoned);
    EXPECT_FALSE(poisoned.request.has_value());
    EXPECT_EQ(mManager->shutdown(), SwaKVCacheStatus::kOk);
    EXPECT_EQ(synchronizeCalls, 2);
    EXPECT_EQ(mManager->freePageCount(), totalPages);
    EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

TEST_F(BoundedSwaKVPageManagerTest, FailedCompactionSynchronizationQuarantinesDroppedPages)
{
    releaseResources();
    int32_t synchronizeCalls{};
    initialize(kDEFAULT_WINDOW_SIZE, [&](cudaStream_t stream) {
        ++synchronizeCalls;
        return synchronizeCalls == 1 ? cudaErrorUnknown : cudaStreamSynchronize(stream);
    });
    int32_t const totalPages = mSwaPageTable->numPages();
    std::vector<int32_t> const inputLengths{3 * kTOKENS_PER_PAGE, 3 * kTOKENS_PER_PAGE + 1};
    BoundedSwaKVPageManager::RequestHandle request = begin(inputLengths);
    completePrefill(request, inputLengths);
    Tensor deviceMapping({2}, rt::DeviceType::kGPU, DataType::kINT32, "swaManagerFailedCompactionMapping");

    ASSERT_EQ(mManager->beginBatchCompaction(request, {0, -1}, 1, deviceMapping), SwaKVCacheStatus::kOk);
    EXPECT_EQ(mManager->compactBatch(request), SwaKVCacheStatus::kPoisoned);
    EXPECT_FALSE(request.valid());
    EXPECT_LT(mManager->freePageCount(), totalPages);
    EXPECT_EQ(mManager->shutdown(), SwaKVCacheStatus::kOk);
    EXPECT_EQ(synchronizeCalls, 2);
    EXPECT_EQ(mManager->freePageCount(), totalPages);
}

} // namespace
