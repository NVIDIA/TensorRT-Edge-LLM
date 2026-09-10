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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"

#include <NvInferRuntimeBase.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace trt_edgellm::rt
{

ActionKvBatchCollector::ActionKvBatchCollector(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages)
    : mMaxBatch(maxBatch)
    , mPageTable(maxBatch, maxPagesPerSeq, numPages)
    , mHostLengths({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "ActionKvBatchCollector::mHostLengths")
    , mDeviceLengths({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "ActionKvBatchCollector::mDeviceLengths")
{
    ELLM_CHECK(maxBatch > 0, "Action KV snapshot capacity must be positive");
}

void ActionKvBatchCollector::beginRequest(std::vector<bool> const& actionSlots, std::vector<int64_t> const& mropeDeltas)
{
    ELLM_CHECK(static_cast<int32_t>(actionSlots.size()) <= mMaxBatch,
        "Action KV snapshot request exceeds configured batch capacity");
    ELLM_CHECK(actionSlots.size() == mropeDeltas.size(),
        "Action KV snapshot mask and MRoPE deltas must describe the same request batch");
    ELLM_CHECK(std::any_of(actionSlots.begin(), actionSlots.end(), [](bool action) { return action; }),
        "Action KV snapshot request must contain at least one action slot");
    mActionSlots = actionSlots;
    mRequestMropeDeltas = mropeDeltas;
    mPending.assign(actionSlots.size(), 0U);
    mByOriginalIndex.assign(actionSlots.size(), std::nullopt);
    mOriginalRequestIndices.clear();
    mDenseMropeDeltas.clear();
    mMaterialized = false;
    check::check(mHostLengths.reshape({mMaxBatch}) && mDeviceLengths.reshape({mMaxBatch}),
        "Action KV length tensor reshape failed");
}

void ActionKvBatchCollector::captureFinished(KVPageTable const& pageTable, Tensor const& deviceKvLengths,
    std::vector<int8_t> const& finished, std::vector<int32_t> const& originalIndices, cudaStream_t stream)
{
    ELLM_CHECK(!mMaterialized, "Action KV collector cannot change after materialization");
    ELLM_CHECK(finished.size() == originalIndices.size(),
        "Action KV snapshot finish state and original-index mapping must have the same size");
    ELLM_CHECK(
        pageTable.maxPagesPerSeq() == mPageTable.maxPagesPerSeq() && pageTable.numPages() == mPageTable.numPages(),
        "Action KV snapshot page table is incompatible with the action-local table");
    ELLM_CHECK(deviceKvLengths.getDeviceType() == DeviceType::kGPU
            && deviceKvLengths.getDataType() == nvinfer1::DataType::kINT32,
        "Action KV snapshot lengths must be an INT32 GPU tensor");
    ELLM_CHECK(deviceKvLengths.getShape().volume() >= static_cast<int64_t>(finished.size()),
        "Action KV snapshot length tensor does not cover the active batch");

    std::vector<uint8_t> seen(mActionSlots.size(), 0U);
    int32_t const* const deviceLengths = deviceKvLengths.dataPointer<int32_t>();
    int32_t* const hostLengths = mHostLengths.dataPointer<int32_t>();
    for (size_t slot = 0; slot < finished.size(); ++slot)
    {
        int32_t const originalIndex = originalIndices[slot];
        ELLM_CHECK(originalIndex >= 0 && originalIndex < static_cast<int32_t>(mActionSlots.size()),
            "Action KV snapshot original request index is out of range");
        size_t const original = static_cast<size_t>(originalIndex);
        ELLM_CHECK(seen[original] == 0U, "Action KV snapshot original-index mapping contains a duplicate");
        seen[original] = 1U;
        if (finished[slot] == 0 || !mActionSlots[original] || mPending[original] != 0U
            || mByOriginalIndex[original].has_value())
        {
            continue;
        }

        int32_t const* const row = pageTable.hostRow(static_cast<int32_t>(slot));
        SequenceSnapshot snapshot;
        snapshot.pageIds.assign(row, row + pageTable.maxPagesPerSeq());
        mByOriginalIndex[original] = std::move(snapshot);
        mPending[original] = 1U;
        CUDA_CHECK(cudaMemcpyAsync(
            hostLengths + originalIndex, deviceLengths + slot, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
}

void ActionKvBatchCollector::completeCapture()
{
    ELLM_CHECK(!mMaterialized, "Action KV collector cannot complete after materialization");
    int32_t const* const hostLengths = mHostLengths.dataPointer<int32_t>();
    for (size_t original = 0; original < mPending.size(); ++original)
    {
        if (mPending[original] == 0U)
        {
            continue;
        }
        ELLM_CHECK(mByOriginalIndex[original].has_value(), "Pending Action KV length has no matching page snapshot");
        SequenceSnapshot& snapshot = *mByOriginalIndex[original];
        snapshot.finalKvLength = hostLengths[original];
        ELLM_CHECK(snapshot.finalKvLength > 0, "Action KV snapshot length must be positive");
        int32_t const livePages = (snapshot.finalKvLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
        ELLM_CHECK(livePages <= static_cast<int32_t>(snapshot.pageIds.size()),
            "Action KV snapshot length exceeds its page-table row");
        snapshot.pageIds.resize(static_cast<size_t>(livePages));
        ELLM_CHECK(
            std::all_of(snapshot.pageIds.begin(), snapshot.pageIds.end(), [](int32_t pageId) { return pageId >= 0; }),
            "Action KV snapshot has an unmapped live page");
        mPending[original] = 0U;
    }
}

ActionKvBatchView ActionKvBatchCollector::materialize(cudaStream_t stream)
{
    ELLM_CHECK(!mMaterialized, "Action KV batch was already materialized");
    ELLM_CHECK(!mActionSlots.empty(), "Action KV snapshot request was not initialized");
    ELLM_CHECK(std::none_of(mPending.begin(), mPending.end(), [](uint8_t pending) { return pending != 0U; }),
        "Action KV snapshot has an unfinished length transfer");

    int32_t const actionCount = static_cast<int32_t>(std::count(mActionSlots.begin(), mActionSlots.end(), true));
    check::check(mHostLengths.reshape({actionCount}) && mDeviceLengths.reshape({actionCount}),
        "Action KV dense length tensor reshape failed");
    mOriginalRequestIndices.clear();
    mDenseMropeDeltas.clear();
    mOriginalRequestIndices.reserve(static_cast<size_t>(actionCount));
    mDenseMropeDeltas.reserve(static_cast<size_t>(actionCount));
    std::vector<KVPageTableRowUpdate> pageUpdates;
    pageUpdates.reserve(static_cast<size_t>(actionCount));
    int32_t* const denseLengths = mHostLengths.dataPointer<int32_t>();
    for (size_t original = 0; original < mActionSlots.size(); ++original)
    {
        if (!mActionSlots[original])
        {
            ELLM_CHECK(!mByOriginalIndex[original].has_value(), "A non-Action request produced an Action KV snapshot");
            continue;
        }
        ELLM_CHECK(mByOriginalIndex[original].has_value(),
            "Every Action request must produce exactly one terminal KV snapshot");
        SequenceSnapshot const& snapshot = *mByOriginalIndex[original];
        int32_t const denseIndex = static_cast<int32_t>(mOriginalRequestIndices.size());
        denseLengths[denseIndex] = snapshot.finalKvLength;
        pageUpdates.push_back({denseIndex, snapshot.pageIds.data(), static_cast<int32_t>(snapshot.pageIds.size())});
        mOriginalRequestIndices.push_back(static_cast<int32_t>(original));
        mDenseMropeDeltas.push_back(mRequestMropeDeltas[original]);
    }
    ELLM_CHECK(static_cast<int32_t>(mOriginalRequestIndices.size()) == actionCount,
        "Action KV snapshot count does not match the requested Action batch size");

    mPageTable.setRows(pageUpdates);
    mPageTable.upload(stream);
    CUDA_CHECK(cudaMemcpyAsync(mDeviceLengths.rawPointer(), mHostLengths.rawPointer(),
        static_cast<size_t>(actionCount) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mMaterialized = true;
    return ActionKvBatchView{mPageTable, mDeviceLengths, mHostLengths, mDenseMropeDeltas, actionCount};
}

std::vector<int32_t> const& ActionKvBatchCollector::originalRequestIndices() const noexcept
{
    return mOriginalRequestIndices;
}

} // namespace trt_edgellm::rt
