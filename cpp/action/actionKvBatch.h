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

#pragma once

#include "common/tensor.h"
#include "runtime/state/kvPageTable.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! Dense logical KV batch consumed by an action model after language generation completes.
struct ActionKvBatchView
{
    KVPageTable const& pageTable;
    Tensor const& kvLengthsDevice;
    Tensor const& kvLengthsHost;
    std::vector<int64_t> const& mropeDeltas;
    int32_t batchSize{};
};

//! Collects request-local terminal KV rows and materializes the final dense action batch.
class ActionKvBatchCollector
{
public:
    ActionKvBatchCollector(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages);

    void beginRequest(std::vector<bool> const& actionSlots, std::vector<int64_t> const& mropeDeltas);
    void captureFinished(KVPageTable const& pageTable, Tensor const& deviceKvLengths,
        std::vector<int8_t> const& finished, std::vector<int32_t> const& originalIndices, cudaStream_t stream);
    //! Consume queued length copies after the caller has synchronized the enqueue stream.
    void completeCapture();

    //! Validate completeness and materialize one dense, action-local logical KV batch.
    ActionKvBatchView materialize(cudaStream_t stream);

    std::vector<int32_t> const& originalRequestIndices() const noexcept;

private:
    struct SequenceSnapshot
    {
        int32_t finalKvLength{};
        std::vector<int32_t> pageIds;
    };

    int32_t mMaxBatch{};
    KVPageTable mPageTable;
    Tensor mHostLengths;
    Tensor mDeviceLengths;
    std::vector<bool> mActionSlots;
    std::vector<int64_t> mRequestMropeDeltas;
    std::vector<uint8_t> mPending;
    std::vector<std::optional<SequenceSnapshot>> mByOriginalIndex;
    std::vector<int32_t> mOriginalRequestIndices;
    std::vector<int64_t> mDenseMropeDeltas;
    bool mMaterialized{};
};

} // namespace trt_edgellm::rt
