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
#include "kernels/dart/dartGatherKernels.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr int32_t kKV_HALVES = 2;

//! V id of a K id: the sentinel maps to itself, a live id shifts by numPages.
int32_t deriveV(int32_t k, int32_t numPages)
{
    return k == kUNUSED_PAGE_ENTRY ? kUNUSED_PAGE_ENTRY : k + numPages;
}
} // namespace

KVPageTable::KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages)
    : KVPageTable(maxBatch, maxPagesPerSeq, numPages, Mode::kDense)
{
}

KVPageTable::KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages, Mode mode)
    : mMaxBatch(maxBatch)
    , mMaxPagesPerSeq(maxPagesPerSeq)
    , mNumPages(numPages)
    , mMode(mode)
{
    check::check(maxBatch > 0, "KVPageTable: maxBatch must be positive.");
    check::check(maxPagesPerSeq > 0, "KVPageTable: maxPagesPerSeq must be positive.");
    check::check(numPages > 0, "KVPageTable: numPages must be positive.");
    check::check(numPages <= kMAX_KV_POOL_PAGES,
        "KVPageTable: numPages exceeds the largest pool whose derived V page ids fit int32.");
    check::check(mode == Mode::kDense || mode == Mode::kSparseWindow, "KVPageTable: unsupported mode.");

    size_t const tableSize = static_cast<size_t>(maxBatch) * kKV_HALVES * maxPagesPerSeq;
    mHost.assign(tableSize, kUNUSED_PAGE_ENTRY);
    // Device storage is uninitialized until the first upload, so every row initially needs a copy.
    mDirtyRows.resize(static_cast<size_t>(maxBatch), 1U);
    mUploadedHost = mHost;
    mSparseActivePages.resize(static_cast<size_t>(maxBatch));
    mSparseLogicalPages.resize(static_cast<size_t>(maxBatch));
    mDevice = rt::Tensor(Coords{maxBatch, kKV_HALVES, maxPagesPerSeq}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "KVPageTable::kernelView");
    mUploadStaging = rt::Tensor(Coords{maxBatch, kKV_HALVES, maxPagesPerSeq}, DeviceType::kCPU,
        nvinfer1::DataType::kINT32, "KVPageTable::uploadStaging");
    CUDA_CHECK(cudaEventCreateWithFlags(&mUploadComplete, cudaEventDisableTiming));
}

KVPageTable::~KVPageTable() noexcept
{
    if (mUploadComplete != nullptr)
    {
        if (mUploadPending)
        {
            (void) cudaEventSynchronize(mUploadComplete);
        }
        (void) cudaEventDestroy(mUploadComplete);
    }
}

void KVPageTable::setIdentity()
{
    check::check(static_cast<int64_t>(mMaxBatch) * mMaxPagesPerSeq <= mNumPages,
        "KVPageTable::setIdentity: physical pool is smaller than the identity layout.");
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        mSparseActivePages[static_cast<size_t>(b)].clear();
        mSparseLogicalPages[static_cast<size_t>(b)].clear();
        size_t const rowOffset = static_cast<size_t>(b) * kKV_HALVES * mMaxPagesPerSeq;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = b * mMaxPagesPerSeq + j;
            setHostValue(rowOffset + j, k);
            setHostValue(rowOffset + mMaxPagesPerSeq + j, deriveV(k, mNumPages));
            if (mMode == Mode::kSparseWindow)
            {
                mSparseActivePages[static_cast<size_t>(b)].insert(k);
                mSparseLogicalPages[static_cast<size_t>(b)].emplace(j, k);
            }
        }
    }
    mIsIdentity = true;
}

void KVPageTable::setRow(int32_t slot, int32_t const* kPageIds, int32_t count)
{
    check::check(slot >= 0 && slot < mMaxBatch, "KVPageTable::setRow: slot out of range.");
    check::check(count >= 0 && count <= mMaxPagesPerSeq, "KVPageTable::setRow: count out of range.");
    check::check(count == 0 || kPageIds != nullptr, "KVPageTable::setRow: kPageIds must not be null.");

    std::unordered_set<int32_t> sparsePages;
    std::unordered_map<int32_t, int32_t> sparseLogicalPages;
    for (int32_t j = 0; j < count; ++j)
    {
        int32_t const k = kPageIds[j];
        bool const pageInRange = k >= 0 && k < mNumPages;
        check::check(pageInRange || (mMode == Mode::kSparseWindow && k == kUNUSED_PAGE_ENTRY),
            "KVPageTable::setRow: page id out of range.");
        if (mMode == Mode::kSparseWindow)
        {
            check::check(k == kUNUSED_PAGE_ENTRY || sparsePages.insert(k).second,
                "KVPageTable::setRow: duplicate sparse page id.");
            if (k != kUNUSED_PAGE_ENTRY)
            {
                sparseLogicalPages.emplace(j, k);
            }
        }
    }

    mIsIdentity = false;
    size_t const rowOffset = static_cast<size_t>(slot) * kKV_HALVES * mMaxPagesPerSeq;
    for (int32_t j = 0; j < count; ++j)
    {
        int32_t const k = kPageIds[j];
        setHostValue(rowOffset + j, k);
        setHostValue(rowOffset + mMaxPagesPerSeq + j, deriveV(k, mNumPages));
    }
    for (int32_t j = count; j < mMaxPagesPerSeq; ++j)
    {
        setHostValue(rowOffset + j, kUNUSED_PAGE_ENTRY);
        setHostValue(rowOffset + mMaxPagesPerSeq + j, kUNUSED_PAGE_ENTRY);
    }
    if (mMode == Mode::kSparseWindow)
    {
        mSparseActivePages[static_cast<size_t>(slot)] = std::move(sparsePages);
        mSparseLogicalPages[static_cast<size_t>(slot)] = std::move(sparseLogicalPages);
    }
}

void KVPageTable::setRows(std::vector<KVPageTableRowUpdate> const& updates)
{
    applyRows(updates.data(), updates.size());
}

void KVPageTable::applyRows(KVPageTableRowUpdate const* updates, size_t count)
{
    std::vector<uint8_t> slotsSeen(static_cast<size_t>(mMaxBatch), 0U);
    for (size_t updateIndex = 0; updateIndex < count; ++updateIndex)
    {
        KVPageTableRowUpdate const& update = updates[updateIndex];
        ELLM_CHECK(update.slot >= 0 && update.slot < mMaxBatch,
            "KVPageTable::setRows: slot out of range: " + std::to_string(update.slot));
        ELLM_CHECK(update.count >= 0 && update.count <= mMaxPagesPerSeq,
            "KVPageTable::setRows: count out of range for slot " + std::to_string(update.slot));
        ELLM_CHECK(update.count == 0 || update.kPageIds != nullptr,
            "KVPageTable::setRows: non-empty row has a null page-id pointer at slot " + std::to_string(update.slot));
        ELLM_CHECK(slotsSeen[static_cast<size_t>(update.slot)] == 0U,
            "KVPageTable::setRows: slot appears more than once: " + std::to_string(update.slot));
        slotsSeen[static_cast<size_t>(update.slot)] = 1U;

        std::unordered_set<int32_t> sparsePages;
        for (int32_t pageIndex = 0; pageIndex < update.count; ++pageIndex)
        {
            int32_t const pageId = update.kPageIds[pageIndex];
            bool const pageInRange = pageId >= 0 && pageId < mNumPages;
            ELLM_CHECK(pageInRange || (mMode == Mode::kSparseWindow && pageId == kUNUSED_PAGE_ENTRY),
                "KVPageTable::setRows: page id " + std::to_string(pageId) + " out of range [0, "
                    + std::to_string(mNumPages) + ") at slot " + std::to_string(update.slot) + ", index "
                    + std::to_string(pageIndex));
            ELLM_CHECK(
                mMode != Mode::kSparseWindow || pageId == kUNUSED_PAGE_ENTRY || sparsePages.insert(pageId).second,
                "KVPageTable::setRows: duplicate sparse page id at slot " + std::to_string(update.slot));
        }
    }

    for (size_t updateIndex = 0; updateIndex < count; ++updateIndex)
    {
        KVPageTableRowUpdate const& update = updates[updateIndex];
        setRow(update.slot, update.kPageIds, update.count);
    }
}

void KVPageTable::setEntry(int32_t slot, int32_t logicalPage, int32_t kPageId)
{
    check::check(slot >= 0 && slot < mMaxBatch, "KVPageTable::setEntry: slot out of range.");
    check::check(logicalPage >= 0 && logicalPage < mMaxPagesPerSeq, "KVPageTable::setEntry: logicalPage out of range.");
    check::check(kPageId >= 0 && kPageId < mNumPages, "KVPageTable::setEntry: kPageId out of range.");

    mIsIdentity = false;
    size_t const kIndex = static_cast<size_t>(slot) * kKV_HALVES * mMaxPagesPerSeq + static_cast<size_t>(logicalPage);
    int32_t const oldPageId = mHost[kIndex];
    if (mMode == Mode::kSparseWindow && oldPageId != kPageId)
    {
        auto& activePages = mSparseActivePages[static_cast<size_t>(slot)];
        check::check(
            activePages.find(kPageId) == activePages.end(), "KVPageTable::setEntry: duplicate active sparse page id.");
        if (oldPageId != kUNUSED_PAGE_ENTRY)
        {
            activePages.erase(oldPageId);
        }
        activePages.insert(kPageId);
        mSparseLogicalPages[static_cast<size_t>(slot)][logicalPage] = kPageId;
    }
    setHostValue(kIndex, kPageId);
    setHostValue(kIndex + mMaxPagesPerSeq, deriveV(kPageId, mNumPages));
}

void KVPageTable::clearEntry(int32_t slot, int32_t logicalPage)
{
    check::check(slot >= 0 && slot < mMaxBatch, "KVPageTable::clearEntry: slot out of range.");
    check::check(
        logicalPage >= 0 && logicalPage < mMaxPagesPerSeq, "KVPageTable::clearEntry: logicalPage out of range.");

    mIsIdentity = false;
    size_t const kIndex = static_cast<size_t>(slot) * kKV_HALVES * mMaxPagesPerSeq + static_cast<size_t>(logicalPage);
    int32_t const oldPageId = mHost[kIndex];
    if (mMode == Mode::kSparseWindow && oldPageId != kUNUSED_PAGE_ENTRY)
    {
        mSparseActivePages[static_cast<size_t>(slot)].erase(oldPageId);
        mSparseLogicalPages[static_cast<size_t>(slot)].erase(logicalPage);
    }
    setHostValue(kIndex, kUNUSED_PAGE_ENTRY);
    setHostValue(kIndex + mMaxPagesPerSeq, kUNUSED_PAGE_ENTRY);
}

bool KVPageTable::checkInvariants(std::string& error) const
{
    error.clear();
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        int32_t const* kRow = hostRow(b);
        int32_t const* vRow = kRow + mMaxPagesPerSeq;
        bool sawSentinel = false;
        std::unordered_set<int32_t> activePageIds;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = kRow[j];
            if (k == kUNUSED_PAGE_ENTRY)
            {
                if (vRow[j] != kUNUSED_PAGE_ENTRY)
                {
                    error = "KVPageTable: V page id does not match unused K entry at slot " + std::to_string(b)
                        + ", index " + std::to_string(j);
                    return false;
                }
                sawSentinel = true;
                continue;
            }
            if (k < 0 || k >= mNumPages)
            {
                error = "KVPageTable: page id " + std::to_string(k) + " out of range [0, " + std::to_string(mNumPages)
                    + ") at slot " + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
            if (vRow[j] != deriveV(k, mNumPages))
            {
                error = "KVPageTable: V page id " + std::to_string(vRow[j]) + " is not derived from K page id "
                    + std::to_string(k) + " at slot " + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
            if (mMode == Mode::kDense && sawSentinel)
            {
                error = "KVPageTable: live page id " + std::to_string(k) + " follows sentinel at slot "
                    + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
            if (mMode == Mode::kSparseWindow && !activePageIds.insert(k).second)
            {
                error = "KVPageTable: duplicate active page id " + std::to_string(k) + " at slot " + std::to_string(b)
                    + ", index " + std::to_string(j);
                return false;
            }
        }
    }
    return true;
}

bool KVPageTable::upload(cudaStream_t stream)
{
    if (std::none_of(mDirtyRows.begin(), mDirtyRows.end(), [](uint8_t dirty) { return dirty != 0U; }))
    {
        return false;
    }

    std::string error;
    ELLM_CHECK(checkInvariants(error), "KVPageTable::upload: " + error);

    if (mUploadPending)
    {
        CUDA_CHECK(cudaEventSynchronize(mUploadComplete));
        mUploadPending = false;
    }

    size_t const rowElements = static_cast<size_t>(kKV_HALVES * mMaxPagesPerSeq);
    int32_t* const staging = mUploadStaging.dataPointer<int32_t>();
    int32_t* const device = mDevice.dataPointer<int32_t>();
    size_t entryCount = 0;
    size_t rangeCount = 0;
    int32_t rangeBegin = 0;
    while (rangeBegin < mMaxBatch)
    {
        while (rangeBegin < mMaxBatch && mDirtyRows[static_cast<size_t>(rangeBegin)] == 0U)
        {
            ++rangeBegin;
        }
        if (rangeBegin == mMaxBatch)
        {
            break;
        }

        int32_t rangeEnd = rangeBegin + 1;
        while (rangeEnd < mMaxBatch && mDirtyRows[static_cast<size_t>(rangeEnd)] != 0U)
        {
            ++rangeEnd;
        }

        size_t const elementOffset = static_cast<size_t>(rangeBegin) * rowElements;
        size_t const elementCount = static_cast<size_t>(rangeEnd - rangeBegin) * rowElements;
        std::copy_n(mHost.data() + elementOffset, elementCount, staging + elementOffset);
        CUDA_CHECK(cudaMemcpyAsync(device + elementOffset, staging + elementOffset, elementCount * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        entryCount += elementCount;
        ++rangeCount;
        rangeBegin = rangeEnd;
    }

    CUDA_CHECK(cudaEventRecord(mUploadComplete, stream));
    mUploadPending = true;
    mUploadedHost = mHost;
    mDirtyIndices.clear();
    std::fill(mDirtyRows.begin(), mDirtyRows.end(), 0U);
    mHasUploaded = true;
    mLastUploadEntryCount = entryCount;
    mLastUploadRangeCount = rangeCount;
    return true;
}

void KVPageTable::uploadDirty(cudaStream_t stream)
{
    if (mMode == Mode::kDense)
    {
        std::string error;
        ELLM_CHECK(checkInvariants(error), "KVPageTable::uploadDirty: " + error);
    }

    bool const initializeDevice = !mHasUploaded;
    if (!initializeDevice && mDirtyIndices.empty())
    {
        mLastUploadEntryCount = 0;
        mLastUploadRangeCount = 0;
        return;
    }

    if (mUploadPending)
    {
        CUDA_CHECK(cudaEventSynchronize(mUploadComplete));
        mUploadPending = false;
    }
    if (initializeDevice)
    {
        CUDA_CHECK(cudaMemsetAsync(mDevice.rawPointer(), 0xFF, mHost.size() * sizeof(int32_t), stream));
    }

    int32_t* const staging = mUploadStaging.dataPointer<int32_t>();
    for (size_t const index : mDirtyIndices)
    {
        staging[index] = mHost[index];
    }

    size_t entryCount = 0;
    size_t rangeCount = 0;
    auto dirtyIt = mDirtyIndices.cbegin();
    while (dirtyIt != mDirtyIndices.cend())
    {
        size_t const begin = *dirtyIt;
        size_t end = begin + 1;
        ++dirtyIt;
        while (dirtyIt != mDirtyIndices.cend() && *dirtyIt == end)
        {
            ++end;
            ++dirtyIt;
        }
        size_t const count = end - begin;
        CUDA_CHECK(cudaMemcpyAsync(mDevice.dataPointer<int32_t>() + begin, staging + begin, count * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        entryCount += count;
        ++rangeCount;
    }

    for (size_t const index : mDirtyIndices)
    {
        mUploadedHost[index] = mHost[index];
    }
    mDirtyIndices.clear();
    std::fill(mDirtyRows.begin(), mDirtyRows.end(), 0U);
    mHasUploaded = true;
    mLastUploadEntryCount = entryCount;
    mLastUploadRangeCount = rangeCount;
    if (initializeDevice || rangeCount > 0)
    {
        CUDA_CHECK(cudaEventRecord(mUploadComplete, stream));
        mUploadPending = true;
    }
}

void KVPageTable::gatherRows(
    rt::Tensor& destination, rt::Tensor const& residentSlots, int32_t numRows, cudaStream_t stream) const
{
    ELLM_CHECK(numRows >= 0 && numRows <= mMaxBatch, "KVPageTable::gatherRows: row count is out of range.");
    ELLM_CHECK(residentSlots.getDataType() == nvinfer1::DataType::kINT32,
        "KVPageTable::gatherRows: resident slots must be INT32.");
    ELLM_CHECK(
        residentSlots.getShape().volume() >= numRows, "KVPageTable::gatherRows: resident slot tensor is too small.");
    ELLM_CHECK(
        destination.getDataType() == nvinfer1::DataType::kINT32, "KVPageTable::gatherRows: destination must be INT32.");
    int64_t const requiredBytes = static_cast<int64_t>(numRows) * 2 * mMaxPagesPerSeq * sizeof(int32_t);
    ELLM_CHECK(
        destination.getMemoryCapacity() >= requiredBytes, "KVPageTable::gatherRows: destination storage is too small.");
    ELLM_CHECK(
        destination.reshape({numRows, 2, mMaxPagesPerSeq}), "KVPageTable::gatherRows: destination reshape failed.");
    if (numRows == 0)
    {
        return;
    }
    int64_t const rowBytes = static_cast<int64_t>(2) * mMaxPagesPerSeq * sizeof(int32_t);
    kernel::gatherRows(destination.rawPointer(), mDevice.rawPointer(), residentSlots.dataPointer<int32_t>(), numRows,
        rowBytes, stream);
}

rt::Tensor const& KVPageTable::kernelView() const
{
    return mDevice;
}

rt::Tensor& KVPageTable::kernelView()
{
    return mDevice;
}

int32_t const* KVPageTable::hostRow(int32_t slot) const
{
    return mHost.data() + static_cast<size_t>(slot) * kKV_HALVES * mMaxPagesPerSeq;
}

void KVPageTable::setHostValue(size_t index, int32_t value)
{
    if (mHost[index] == value)
    {
        return;
    }

    mHost[index] = value;
    if (value != mUploadedHost[index])
    {
        mDirtyIndices.insert(index);
    }
    else
    {
        mDirtyIndices.erase(index);
    }

    size_t const rowElements = static_cast<size_t>(kKV_HALVES * mMaxPagesPerSeq);
    size_t const slot = index / rowElements;
    size_t const rowBegin = slot * rowElements;
    auto const firstDirty = mDirtyIndices.lower_bound(rowBegin);
    mDirtyRows[slot] = (firstDirty != mDirtyIndices.end() && *firstDirty < rowBegin + rowElements) ? 1U : 0U;
}

} // namespace rt
} // namespace trt_edgellm
