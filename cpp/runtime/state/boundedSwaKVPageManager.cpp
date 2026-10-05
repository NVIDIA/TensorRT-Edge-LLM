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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/kvCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

struct SwaRetainedRange
{
    int32_t begin{};
    int32_t end{};
};

SwaRetainedRange retainedRange(int32_t endpoint, int32_t pageSize, int32_t windowSize) noexcept
{
    int64_t const retainedEnd = (static_cast<int64_t>(endpoint) + pageSize - 1) / pageSize;
    int64_t const windowStart = std::max<int64_t>(0, static_cast<int64_t>(endpoint) - windowSize);
    return SwaRetainedRange{static_cast<int32_t>(windowStart / pageSize), static_cast<int32_t>(retainedEnd)};
}

std::optional<SwaPageBinding> findBinding(SwaKVCacheState const& state, int32_t logicalPage) noexcept
{
    if (logicalPage < state.retainedLogicalPageBegin || logicalPage >= state.retainedLogicalPageEnd)
    {
        return std::nullopt;
    }
    size_t const index = static_cast<size_t>(logicalPage - state.retainedLogicalPageBegin);
    if (index >= state.pageBindings.size() || state.pageBindings[index].logicalPage != logicalPage)
    {
        return std::nullopt;
    }
    return state.pageBindings[index];
}

void validateCompactionMapping(std::vector<int32_t> const& oldToNew, int32_t oldBatchSize, int32_t newBatchSize)
{
    ELLM_CHECK(static_cast<int32_t>(oldToNew.size()) == oldBatchSize,
        "SWA KV cache compaction mapping must describe every active slot");
    ELLM_CHECK(newBatchSize >= 0 && newBatchSize <= oldBatchSize,
        "SWA KV cache compaction batch size is outside the active range");
    std::vector<uint8_t> destinations(static_cast<size_t>(newBatchSize), 0U);
    for (int32_t const destination : oldToNew)
    {
        ELLM_CHECK(destination >= -1 && destination < newBatchSize,
            "SWA KV cache compaction mapping contains an invalid destination");
        if (destination >= 0)
        {
            ELLM_CHECK(destinations[static_cast<size_t>(destination)] == 0U,
                "SWA KV cache compaction mapping contains a duplicate destination");
            destinations[static_cast<size_t>(destination)] = 1U;
        }
    }
    ELLM_CHECK(std::all_of(destinations.begin(), destinations.end(), [](uint8_t value) { return value != 0U; }),
        "SWA KV cache compaction mapping omits a destination");
}

} // namespace

struct BoundedSwaKVPageManager::RequestHandle::Impl
{
    enum class Phase : uint8_t
    {
        kAdmitted,
        kExecuting,
        kFinishing,
    };

    Impl(BoundedSwaKVPageManager& manager_, cudaStream_t stream_)
        : manager(&manager_)
        , stream(stream_)
    {
    }

    BoundedSwaKVPageManager* manager{};
    cudaStream_t stream{};
    std::vector<int32_t> inputLengths;
    std::vector<SwaKVCacheState> states;
    std::vector<int32_t> pendingCompactionMapping;
    int32_t pendingCompactionBatchSize{-1};
    Tensor const* pendingDeviceBatchMapping{};
    Phase phase{Phase::kAdmitted};
    bool deviceWorkPending{};
};

BoundedSwaKVPageManager::RequestHandle::RequestHandle(std::unique_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{
}

BoundedSwaKVPageManager::RequestHandle::RequestHandle(RequestHandle&& other) noexcept = default;

BoundedSwaKVPageManager::RequestHandle::~RequestHandle() noexcept
{
    if (mImpl != nullptr)
    {
        BoundedSwaKVPageManager* const manager = mImpl->manager;
        manager->abandon(std::move(mImpl));
    }
}

bool BoundedSwaKVPageManager::RequestHandle::valid() const noexcept
{
    return mImpl != nullptr;
}

BoundedSwaKVPageManager::BoundedSwaKVPageManager(HybridCacheManager& baseCache, KVPageTable& basePageTable,
    KVPageTable& swaPageTable, cudaStream_t stream, StreamSynchronizer synchronizer)
    : mBaseCache(baseCache)
    , mBasePageTable(basePageTable)
    , mSwaPageTable(swaPageTable)
    , mPageSize(kTOKENS_PER_PAGE)
    , mStream(stream)
    , mSynchronizer(std::move(synchronizer))
{
    KVCacheManager const& kvCache = mBaseCache.getKVCacheManager();
    std::optional<int32_t> const windowSize = kvCache.reducedKVCacheCapacity();
    ELLM_CHECK(kvCache.hasReducedKVCache() && windowSize.has_value(),
        "SWA KV cache manager requires a physically reduced KV pool");
    mWindowSize = *windowSize;
    int64_t const privatePages = computeSwaPrivatePagesPerSlot(mWindowSize, mPageSize, kSWA_REPLACEMENT_PAGES);
    ELLM_CHECK(privatePages > 0 && privatePages <= std::numeric_limits<int32_t>::max(),
        "SWA KV cache private-page reservation exceeds int32 range");
    mPrivatePagesPerSequence = static_cast<int32_t>(privatePages);

    ELLM_CHECK(kvCache.numPages() == mBasePageTable.numPages(),
        "SWA KV cache base pool and page table have different physical page counts");
    ELLM_CHECK(mSwaPageTable.maxPagesPerSeq() == mBasePageTable.maxPagesPerSeq(),
        "SWA and base page tables have different logical widths");
    ELLM_CHECK(static_cast<int64_t>(mSwaPageTable.numPages())
            >= static_cast<int64_t>(kvCache.getConfig().maxBatchSize) * mPrivatePagesPerSequence,
        "SWA physical pool cannot reserve every active sequence");
    for (int32_t layer = 0; layer < kvCache.numLayers(); ++layer)
    {
        KVLayerConfig const& layerConfig = kvCache.getLayerConfig(layer);
        if (isReducedKvCacheCapacity(layerConfig.kvCacheCapacity, kvCache.getConfig().maxSequenceLength))
        {
            ELLM_CHECK(kvCache.numPages(layer) == mSwaPageTable.numPages(),
                "SWA reduced layer and sparse page table have different physical page counts");
        }
    }

    mFreePages.reserve(static_cast<size_t>(mSwaPageTable.numPages()));
    for (PageId page = 0; page < mSwaPageTable.numPages(); ++page)
    {
        mFreePages.push_back(page);
    }
    if (!mSynchronizer)
    {
        mSynchronizer = [](cudaStream_t requestStream) { return cudaStreamSynchronize(requestStream); };
    }
}

BoundedSwaKVPageManager::~BoundedSwaKVPageManager() noexcept
{
    if (mRequestActive || mQuarantinedRequest != nullptr)
    {
        std::terminate();
    }
}

BoundedSwaKVPageManager::BeginRequestResult BoundedSwaKVPageManager::beginRequest(
    std::vector<ResidentRef> const& residents, cudaStream_t stream)
{
    ELLM_CHECK(stream == mStream, "SWA KV cache request stream differs from the manager construction stream");
    if (mPoisoned)
    {
        return BeginRequestResult{SwaKVCacheStatus::kPoisoned, std::nullopt};
    }
    if (mRequestActive)
    {
        return BeginRequestResult{SwaKVCacheStatus::kRequestFailed, std::nullopt};
    }

    int32_t const maxBatch = mBaseCache.getKVCacheManager().getConfig().maxBatchSize;
    int32_t const batchSize = static_cast<int32_t>(residents.size());
    ELLM_CHECK(batchSize > 0 && batchSize <= maxBatch, "SWA KV cache request batch size is outside the engine range");
    for (size_t i = 0; i < residents.size(); ++i)
    {
        ELLM_CHECK(residents[i].slot >= 0 && residents[i].slot < maxBatch,
            "SWA KV cache resident slot is outside the engine range");
        ELLM_CHECK(std::none_of(residents.begin(), residents.begin() + static_cast<ptrdiff_t>(i),
                       [&](ResidentRef const& resident) { return resident.slot == residents[i].slot; }),
            "SWA KV cache resident slots must be unique");
    }

    int64_t const requiredPages = static_cast<int64_t>(batchSize) * mPrivatePagesPerSequence;
    if (requiredPages > static_cast<int64_t>(mFreePages.size()))
    {
        return BeginRequestResult{SwaKVCacheStatus::kRequestFailed, std::nullopt};
    }

    auto impl = std::make_unique<RequestHandle::Impl>(*this, stream);
    impl->states.resize(static_cast<size_t>(batchSize));
    size_t const maxRetainedPages
        = static_cast<size_t>((static_cast<int64_t>(mWindowSize) + mPageSize - 1) / mPageSize + 1);
    for (size_t sequence = 0; sequence < impl->states.size(); ++sequence)
    {
        SwaKVCacheState& state = impl->states[sequence];
        state.resident = residents[sequence];
        state.pageSizeTokens = mPageSize;
        state.windowSizeTokens = mWindowSize;
        state.reservedPages.reserve(static_cast<size_t>(mPrivatePagesPerSequence));
        state.newPageBindings.reserve(maxRetainedPages);
        state.pendingRetireBindings.reserve(maxRetainedPages);
    }

    for (SwaKVCacheState& state : impl->states)
    {
        for (int32_t page = 0; page < mPrivatePagesPerSequence; ++page)
        {
            state.reservedPages.push_back(mFreePages.back());
            mFreePages.pop_back();
        }
    }
    mRequestActive = true;
    return BeginRequestResult{SwaKVCacheStatus::kOk, std::optional<RequestHandle>{RequestHandle(std::move(impl))}};
}

BoundedSwaKVPageManager::RequestHandle::Impl& BoundedSwaKVPageManager::checkedImpl(RequestHandle& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->manager == this,
        "SWA KV cache request handle is invalid or belongs to another manager");
    return *request.mImpl;
}

BoundedSwaKVPageManager::RequestHandle::Impl const& BoundedSwaKVPageManager::checkedImpl(
    RequestHandle const& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->manager == this,
        "SWA KV cache request handle is invalid or belongs to another manager");
    return *request.mImpl;
}

bool BoundedSwaKVPageManager::advance(SwaKVCacheState& state, int32_t newExactResidentTokenCount)
{
    ELLM_CHECK(!state.pendingEndpoint.has_value(), "SWA KV cache advancement must be completed before another advance");
    ELLM_CHECK(
        newExactResidentTokenCount >= state.exactResidentTokenCount, "SWA KV cache endpoint cannot move backward");

    int32_t const oldEndpoint = state.exactResidentTokenCount;
    if (newExactResidentTokenCount == oldEndpoint)
    {
        state.pendingEndpoint = newExactResidentTokenCount;
        return true;
    }

    SwaRetainedRange const retained = retainedRange(newExactResidentTokenCount, mPageSize, mWindowSize);
    if (retained.end > mSwaPageTable.maxPagesPerSeq())
    {
        return false;
    }
    int32_t const firstWritePage = oldEndpoint / mPageSize;
    bool const writesExistingPartialPage = oldEndpoint % mPageSize != 0;
    bool const singleToken = static_cast<int64_t>(newExactResidentTokenCount) == static_cast<int64_t>(oldEndpoint) + 1;

    if (singleToken)
    {
        int32_t const beginDelta = retained.begin - state.retainedLogicalPageBegin;
        int32_t const endDelta = retained.end - state.retainedLogicalPageEnd;
        if (beginDelta < 0 || beginDelta > 1 || endDelta < 0 || endDelta > 1)
        {
            return false;
        }

        std::optional<SwaPageBinding> const writeBinding = findBinding(state, firstWritePage);
        if (writeBinding.has_value() != writesExistingPartialPage)
        {
            return false;
        }
        bool const mapNewPage = !writeBinding.has_value();
        if ((mapNewPage && (endDelta != 1 || firstWritePage != state.retainedLogicalPageEnd))
            || (!mapNewPage && endDelta != 0))
        {
            return false;
        }
        if (!mapNewPage && beginDelta == 0)
        {
            state.pendingEndpoint = newExactResidentTokenCount;
            return true;
        }
        if (mapNewPage && state.reservedPages.empty())
        {
            return false;
        }
        if (beginDelta == 1
            && (state.pageBindings.empty() || state.pageBindings.front().logicalPage != state.retainedLogicalPageBegin))
        {
            return false;
        }

        ELLM_CHECK(state.newPageBindings.empty() && state.pendingRetireBindings.empty(),
            "SWA KV cache advancement has stale pending operations");
        if (mapNewPage)
        {
            PageId const destination = state.reservedPages.back();
            state.reservedPages.pop_back();
            SwaPageBinding const binding{firstWritePage, destination};
            state.pageBindings.push_back(binding);
            state.newPageBindings.push_back(binding);
        }
        if (beginDelta == 1)
        {
            state.pendingRetireBindings.push_back(state.pageBindings.front());
            state.pageBindings.pop_front();
        }
        state.retainedLogicalPageBegin = retained.begin;
        state.retainedLogicalPageEnd = retained.end;
        state.pendingEndpoint = newExactResidentTokenCount;
        return true;
    }

    if (state.retainedLogicalPageBegin > retained.begin || state.retainedLogicalPageEnd > retained.end)
    {
        return false;
    }
    int32_t const staleLogicalEnd = std::min(retained.begin, state.retainedLogicalPageEnd);
    int32_t const staleCount = staleLogicalEnd - state.retainedLogicalPageBegin;
    size_t const retainedCount = static_cast<size_t>(retained.end - retained.begin);

    std::deque<SwaPageBinding> nextBindings;
    std::vector<PageId> nextReservedPages;
    nextReservedPages.reserve(static_cast<size_t>(mPrivatePagesPerSequence));
    nextReservedPages.assign(state.reservedPages.begin(), state.reservedPages.end());
    std::vector<SwaPageBinding> newBindings;
    newBindings.reserve(retainedCount);
    std::vector<SwaPageBinding> staleBindings;
    staleBindings.reserve(static_cast<size_t>(staleCount));
    for (int32_t index = 0; index < staleCount; ++index)
    {
        staleBindings.push_back(state.pageBindings[static_cast<size_t>(index)]);
    }

    for (int32_t logicalPage = retained.begin; logicalPage < retained.end; ++logicalPage)
    {
        std::optional<SwaPageBinding> const binding = findBinding(state, logicalPage);
        bool const writesRetainedPartialPage = logicalPage == firstWritePage && writesExistingPartialPage;
        if (binding.has_value())
        {
            if (logicalPage >= firstWritePage && !writesRetainedPartialPage)
            {
                return false;
            }
            nextBindings.push_back(*binding);
            continue;
        }
        if (logicalPage < firstWritePage || nextReservedPages.empty())
        {
            return false;
        }
        PageId const destination = nextReservedPages.back();
        nextReservedPages.pop_back();
        SwaPageBinding const destinationBinding{logicalPage, destination};
        newBindings.push_back(destinationBinding);
        nextBindings.push_back(destinationBinding);
    }
    if (nextBindings.size() != retainedCount)
    {
        return false;
    }

    state.retainedLogicalPageBegin = retained.begin;
    state.retainedLogicalPageEnd = retained.end;
    state.pageBindings = std::move(nextBindings);
    state.reservedPages = std::move(nextReservedPages);
    state.newPageBindings = std::move(newBindings);
    state.pendingRetireBindings = std::move(staleBindings);
    state.pendingEndpoint = newExactResidentTokenCount;
    return true;
}

void BoundedSwaKVPageManager::bindNewPages(SwaKVCacheState const& state, int32_t slot)
{
    for (SwaPageBinding const& binding : state.newPageBindings)
    {
        mSwaPageTable.setEntry(slot, binding.logicalPage, binding.page);
    }
}

void BoundedSwaKVPageManager::retire(SwaKVCacheState& state, int32_t slot)
{
    ELLM_CHECK(state.pendingEndpoint.has_value(), "SWA KV cache completion has no prepared advance");
    size_t const leasedPageCount
        = state.pageBindings.size() + state.reservedPages.size() + state.pendingRetireBindings.size();
    ELLM_CHECK(leasedPageCount == static_cast<size_t>(mPrivatePagesPerSequence),
        "SWA KV cache transition does not own its complete private reservation");
    for (SwaPageBinding const& binding : state.pendingRetireBindings)
    {
        mSwaPageTable.clearEntry(slot, binding.logicalPage);
        state.reservedPages.push_back(binding.page);
    }
    state.exactResidentTokenCount = *state.pendingEndpoint;
    state.newPageBindings.clear();
    state.pendingRetireBindings.clear();
    state.pendingEndpoint.reset();
    ELLM_CHECK(state.reservedPages.size() >= static_cast<size_t>(kSWA_REPLACEMENT_PAGES),
        "SWA KV cache retirement failed to restore its replacement reservation");
}

SwaKVCacheStatus BoundedSwaKVPageManager::preparePrefill(
    RequestHandle& request, std::vector<int32_t> const& inputLengths)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kAdmitted && !impl.deviceWorkPending,
        "SWA KV cache prefill preparation requires a newly admitted request");
    ELLM_CHECK(inputLengths.size() == impl.states.size(),
        "SWA KV cache prefill lengths must describe every admitted sequence");
    for (int32_t const inputLength : inputLengths)
    {
        ELLM_CHECK(inputLength > 0
                && static_cast<int64_t>(inputLength)
                    <= static_cast<int64_t>(mSwaPageTable.maxPagesPerSeq()) * mPageSize,
            "SWA KV cache input length is outside the logical page-table range");
    }
    impl.inputLengths = inputLengths;
    for (size_t slot = 0; slot < impl.states.size(); ++slot)
    {
        int32_t const residentSlot = impl.states[slot].resident.slot;
        mSwaPageTable.setRow(residentSlot, nullptr, 0);
        ELLM_CHECK(advance(impl.states[slot], impl.inputLengths[slot]),
            "SWA KV cache could not prepare the admitted prefill range");
        bindNewPages(impl.states[slot], residentSlot);
    }
    mSwaPageTable.uploadDirty(impl.stream);
    impl.phase = RequestHandle::Impl::Phase::kExecuting;
    impl.deviceWorkPending = true;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::completePrefill(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "SWA KV cache prefill completion requires pending model work");
    for (size_t slot = 0; slot < impl.states.size(); ++slot)
    {
        ELLM_CHECK(impl.states[slot].pendingEndpoint == std::optional<int32_t>{impl.inputLengths[slot]},
            "SWA KV cache prefill endpoint differs from its prepared input length");
        retire(impl.states[slot], impl.states[slot].resident.slot);
    }
    // The caller invokes this only after the runtime's existing host-visible prefill completion point.
    impl.deviceWorkPending = false;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::prepareDecodeStep(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && !impl.deviceWorkPending,
        "SWA KV cache decode preparation requires terminal prior work");
    for (size_t slot = 0; slot < impl.states.size(); ++slot)
    {
        SwaKVCacheState& state = impl.states[slot];
        if (state.exactResidentTokenCount == std::numeric_limits<int32_t>::max()
            || !advance(state, state.exactResidentTokenCount + 1))
        {
            impl.phase = RequestHandle::Impl::Phase::kFinishing;
            return SwaKVCacheStatus::kRequestFailed;
        }
        bindNewPages(state, state.resident.slot);
    }
    mSwaPageTable.uploadDirty(impl.stream);
    impl.deviceWorkPending = true;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::completeDecodeStep(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "SWA KV cache decode completion requires pending model work");
    for (size_t slot = 0; slot < impl.states.size(); ++slot)
    {
        retire(impl.states[slot], impl.states[slot].resident.slot);
    }
    // Vanilla decoding has already reached its existing host-visible completion point.
    impl.deviceWorkPending = false;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::beginBatchCompaction(
    RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && !impl.deviceWorkPending,
        "SWA KV cache compaction preparation requires terminal model work");
    int32_t const oldBatchSize = static_cast<int32_t>(impl.states.size());
    validateCompactionMapping(oldToNew, oldBatchSize, newBatchSize);
    impl.pendingCompactionMapping = oldToNew;
    impl.pendingCompactionBatchSize = newBatchSize;
    impl.pendingDeviceBatchMapping = &deviceBatchMapping;
    ELLM_CHECK(deviceBatchMapping.reshape({oldBatchSize}), "SWA KV cache batch-mapping tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(deviceBatchMapping.rawPointer(), oldToNew.data(),
        static_cast<size_t>(oldBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, impl.stream));
    impl.deviceWorkPending = true;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::synchronize(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    if (!impl.deviceWorkPending)
    {
        return SwaKVCacheStatus::kOk;
    }
    cudaError_t const status = mSynchronizer(impl.stream);
    if (status != cudaSuccess)
    {
        mPoisoned = true;
        ELLM_CHECK(mQuarantinedRequest == nullptr, "SWA KV cache manager already owns a quarantined request");
        mQuarantinedRequest = std::move(request.mImpl);
        return SwaKVCacheStatus::kPoisoned;
    }
    impl.deviceWorkPending = false;
    return SwaKVCacheStatus::kOk;
}

SwaKVCacheStatus BoundedSwaKVPageManager::compactBatch(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "SWA KV cache compaction requires prepared pending work");
    ELLM_CHECK(impl.pendingCompactionBatchSize >= 0 && impl.pendingDeviceBatchMapping != nullptr,
        "SWA KV cache compaction is missing its authoritative mapping");

    std::vector<int32_t> const& oldToNew = impl.pendingCompactionMapping;
    int32_t const newBatchSize = impl.pendingCompactionBatchSize;
    int32_t const oldBatchSize = static_cast<int32_t>(impl.states.size());
    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        if (oldToNew[static_cast<size_t>(oldSlot)] < 0)
        {
            int32_t const residentSlot = impl.states[static_cast<size_t>(oldSlot)].resident.slot;
            mBasePageTable.setRow(residentSlot, nullptr, 0);
            mSwaPageTable.setRow(residentSlot, nullptr, 0);
            mBaseCache.clearResidentSlot(residentSlot, impl.stream);
        }
    }
    mBasePageTable.upload(impl.stream);
    mSwaPageTable.uploadDirty(impl.stream);
    mBaseCache.compactKVCacheLengths(*impl.pendingDeviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);

    SwaKVCacheStatus const syncStatus = synchronize(request);
    if (syncStatus != SwaKVCacheStatus::kOk)
    {
        return syncStatus;
    }

    std::vector<SwaKVCacheState> survivors(static_cast<size_t>(newBatchSize));
    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        int32_t const newSlot = oldToNew[static_cast<size_t>(oldSlot)];
        if (newSlot >= 0)
        {
            survivors[static_cast<size_t>(newSlot)] = std::move(impl.states[static_cast<size_t>(oldSlot)]);
        }
        else
        {
            releaseState(impl.states[static_cast<size_t>(oldSlot)]);
        }
    }
    impl.states = std::move(survivors);
    impl.pendingCompactionMapping.clear();
    impl.pendingCompactionBatchSize = -1;
    impl.pendingDeviceBatchMapping = nullptr;
    mBaseCache.setActiveBatchSize(newBatchSize);
    if (newBatchSize == 0)
    {
        impl.phase = RequestHandle::Impl::Phase::kFinishing;
    }
    return SwaKVCacheStatus::kOk;
}

void BoundedSwaKVPageManager::releaseState(SwaKVCacheState& state) noexcept
{
    size_t const leasedPageCount
        = state.pageBindings.size() + state.reservedPages.size() + state.pendingRetireBindings.size();
    if (leasedPageCount != static_cast<size_t>(mPrivatePagesPerSequence)
        || mFreePages.size() + leasedPageCount > mFreePages.capacity())
    {
        std::terminate();
    }
    for (SwaPageBinding const& binding : state.pageBindings)
    {
        mFreePages.push_back(binding.page);
    }
    for (PageId const page : state.reservedPages)
    {
        mFreePages.push_back(page);
    }
    for (SwaPageBinding const& binding : state.pendingRetireBindings)
    {
        mFreePages.push_back(binding.page);
    }
    state.pageBindings.clear();
    state.reservedPages.clear();
    state.newPageBindings.clear();
    state.pendingRetireBindings.clear();
    state.pendingEndpoint.reset();
}

void BoundedSwaKVPageManager::clearAndRelease(std::unique_ptr<RequestHandle::Impl> request) noexcept
{
    if (request == nullptr)
    {
        return;
    }
    try
    {
        for (SwaKVCacheState const& state : request->states)
        {
            mSwaPageTable.setRow(state.resident.slot, nullptr, 0);
        }
        mSwaPageTable.uploadDirty(request->stream);
    }
    catch (...)
    {
        mPoisoned = true;
        if (mQuarantinedRequest != nullptr)
        {
            std::terminate();
        }
        mQuarantinedRequest = std::move(request);
        return;
    }
    for (SwaKVCacheState& state : request->states)
    {
        releaseState(state);
    }
    mRequestActive = false;
}

SwaKVCacheStatus BoundedSwaKVPageManager::finish(RequestHandle& request)
{
    if (!request.valid())
    {
        return SwaKVCacheStatus::kOk;
    }
    SwaKVCacheStatus const syncStatus = synchronize(request);
    if (syncStatus != SwaKVCacheStatus::kOk)
    {
        return syncStatus;
    }
    clearAndRelease(std::move(request.mImpl));
    return mPoisoned ? SwaKVCacheStatus::kPoisoned : SwaKVCacheStatus::kOk;
}

void BoundedSwaKVPageManager::abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept
{
    if (request == nullptr)
    {
        return;
    }
    if (request->deviceWorkPending)
    {
        cudaError_t status{cudaErrorUnknown};
        try
        {
            status = mSynchronizer(request->stream);
        }
        catch (...)
        {
            status = cudaErrorUnknown;
        }
        if (status != cudaSuccess)
        {
            mPoisoned = true;
            if (mQuarantinedRequest != nullptr)
            {
                std::terminate();
            }
            mQuarantinedRequest = std::move(request);
            return;
        }
        request->deviceWorkPending = false;
    }
    clearAndRelease(std::move(request));
}

SwaKVCacheStatus BoundedSwaKVPageManager::shutdown() noexcept
{
    if (mQuarantinedRequest == nullptr)
    {
        return mRequestActive ? SwaKVCacheStatus::kRequestFailed : SwaKVCacheStatus::kOk;
    }
    cudaError_t status{cudaErrorUnknown};
    try
    {
        status = mSynchronizer(mQuarantinedRequest->stream);
    }
    catch (...)
    {
        status = cudaErrorUnknown;
    }
    if (status != cudaSuccess)
    {
        return SwaKVCacheStatus::kPoisoned;
    }
    mQuarantinedRequest->deviceWorkPending = false;
    std::unique_ptr<RequestHandle::Impl> request = std::move(mQuarantinedRequest);
    mPoisoned = false;
    clearAndRelease(std::move(request));
    return mPoisoned ? SwaKVCacheStatus::kPoisoned : SwaKVCacheStatus::kOk;
}

int32_t BoundedSwaKVPageManager::freePageCount() const noexcept
{
    return static_cast<int32_t>(mFreePages.size());
}

int32_t BoundedSwaKVPageManager::privatePagesPerSequence() const noexcept
{
    return mPrivatePagesPerSequence;
}

SwaKVCacheState const& BoundedSwaKVPageManager::state(RequestHandle const& request, int32_t slot) const
{
    RequestHandle::Impl const& impl = checkedImpl(request);
    ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.states.size()),
        "SWA KV cache state slot is outside the active batch");
    return impl.states[static_cast<size_t>(slot)];
}

} // namespace rt
} // namespace trt_edgellm
