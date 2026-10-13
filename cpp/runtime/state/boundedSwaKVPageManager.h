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

#include "common/pagedKvTypes.h"
#include "runtime/exec/scheduledStep.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class HybridCacheManager;
class KVPageTable;
class Tensor;

//! One absolute logical-to-physical binding in a bounded SWA window.
struct SwaPageBinding
{
    int32_t logicalPage{};
    PageId page{};
};

inline bool operator==(SwaPageBinding const& lhs, SwaPageBinding const& rhs) noexcept
{
    return lhs.logicalPage == rhs.logicalPage && lhs.page == rhs.page;
}

//! Request-local execution state for one bounded SWA sequence.
//!
//! Logical positions and the completed endpoint remain absolute. The compact physical bindings are only an execution
//! view and never identify a reusable prefix. Context reuse may seed or consume settled state in a later integration,
//! but it does not own window rotation.
struct SwaKVCacheState
{
    ResidentRef resident;
    int32_t pageSizeTokens{};
    int32_t windowSizeTokens{};
    int32_t exactResidentTokenCount{};
    int32_t retainedLogicalPageBegin{};
    int32_t retainedLogicalPageEnd{};
    std::deque<SwaPageBinding> pageBindings;
    std::vector<PageId> reservedPages;
    std::vector<SwaPageBinding> newPageBindings;
    std::vector<SwaPageBinding> pendingRetireBindings;
    std::optional<int32_t> pendingEndpoint;
};

enum class SwaKVCacheStatus : uint8_t
{
    kOk,
    kRequestFailed,
    kPoisoned,
};

//! Runtime-level allocator and execution-lifecycle owner for bounded SWA pages.
//!
//! KVCacheManager owns the device buffers. This class owns only page IDs, sparse-table mappings, and request-local
//! transition state. Pages staged for retirement remain leased until the existing CUDA completion point.
class BoundedSwaKVPageManager final
{
public:
    using StreamSynchronizer = std::function<cudaError_t(cudaStream_t)>;

    class RequestHandle final
    {
    public:
        RequestHandle(RequestHandle&& other) noexcept;
        RequestHandle& operator=(RequestHandle&& other) = delete;
        RequestHandle(RequestHandle const&) = delete;
        RequestHandle& operator=(RequestHandle const&) = delete;
        ~RequestHandle() noexcept;

        bool valid() const noexcept;

    private:
        friend class BoundedSwaKVPageManager;
        struct Impl;

        explicit RequestHandle(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> mImpl;
    };

    struct BeginRequestResult
    {
        SwaKVCacheStatus status{SwaKVCacheStatus::kRequestFailed};
        std::optional<RequestHandle> request;
    };

    BoundedSwaKVPageManager(HybridCacheManager& baseCache, KVPageTable& basePageTable, KVPageTable& swaPageTable,
        cudaStream_t stream, StreamSynchronizer synchronizer = {});
    ~BoundedSwaKVPageManager() noexcept;
    BoundedSwaKVPageManager(BoundedSwaKVPageManager const&) = delete;
    BoundedSwaKVPageManager& operator=(BoundedSwaKVPageManager const&) = delete;

    BeginRequestResult beginRequest(std::vector<ResidentRef> const& residents, cudaStream_t stream);
    SwaKVCacheStatus preparePrefill(RequestHandle& request, std::vector<int32_t> const& inputLengths);
    SwaKVCacheStatus completePrefill(RequestHandle& request);
    SwaKVCacheStatus prepareDecodeStep(RequestHandle& request);
    SwaKVCacheStatus completeDecodeStep(RequestHandle& request);
    SwaKVCacheStatus beginBatchCompaction(
        RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    SwaKVCacheStatus compactBatch(RequestHandle& request);
    SwaKVCacheStatus finish(RequestHandle& request);
    SwaKVCacheStatus shutdown() noexcept;

    int32_t freePageCount() const noexcept;
    int32_t privatePagesPerSequence() const noexcept;
    SwaKVCacheState const& state(RequestHandle const& request, int32_t slot) const;

private:
    RequestHandle::Impl& checkedImpl(RequestHandle& request) const;
    RequestHandle::Impl const& checkedImpl(RequestHandle const& request) const;
    bool advance(SwaKVCacheState& state, int32_t newExactResidentTokenCount);
    void bindNewPages(SwaKVCacheState const& state, int32_t slot);
    void retire(SwaKVCacheState& state, int32_t slot);
    void releaseState(SwaKVCacheState& state) noexcept;
    void clearAndRelease(std::unique_ptr<RequestHandle::Impl> request) noexcept;
    void abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept;
    SwaKVCacheStatus synchronize(RequestHandle& request);

    HybridCacheManager& mBaseCache;
    KVPageTable& mBasePageTable;
    KVPageTable& mSwaPageTable;
    int32_t mPageSize{};
    int32_t mWindowSize{};
    int32_t mPrivatePagesPerSequence{};
    std::vector<PageId> mFreePages;
    cudaStream_t mStream{};
    StreamSynchronizer mSynchronizer;
    bool mRequestActive{};
    bool mPoisoned{};
    std::unique_ptr<RequestHandle::Impl> mQuarantinedRequest;
};

} // namespace rt
} // namespace trt_edgellm
