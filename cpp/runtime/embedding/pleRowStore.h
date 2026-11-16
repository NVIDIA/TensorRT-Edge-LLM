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
#include "runtime/embedding/pleCache.h"
#include "runtime/embedding/pleIoBudget.h"
#include "runtime/embedding/pleTable.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Embedding table with resident or synchronous SSD-to-GPU cached storage. The stream outlives this store, and the
//! store outlives its leases. All engine consumers use this stream and release leases only after their GPU work
//! completes.
class PleRowStore
{
public:
    enum class Residency
    {
        kCACHED,
        kRESIDENT
    };

    //! Ordered row mapping and optional cache pins. The store outlives this handle.
    class RequestLease
    {
    public:
        RequestLease(RequestLease const&) = delete;
        RequestLease& operator=(RequestLease const&) = delete;
        RequestLease(RequestLease&& other) noexcept;
        RequestLease& operator=(RequestLease&& other) noexcept;
        std::vector<int32_t> const& mapping() const;
        void release() noexcept;

    private:
        friend class PleRowStore;
        explicit RequestLease(PleCache::RequestLease lease);
        explicit RequestLease(std::vector<int32_t> mapping);
        std::optional<PleCache::RequestLease> mCached;
        std::vector<int32_t> mMapping;
        bool mValid{true};
    };

    struct Stats
    {
        size_t uniqueRows{}; //!< Computed for cached admission; resident queries do not deduplicate.
        size_t misses{};
        size_t readCalls{};
        size_t usefulBytes{}; //!< Requested miss payload, excluding page amplification.
        size_t readBytes{};   //!< Bytes returned by pread; not physical SSD traffic for buffered I/O.
        size_t fillWaves{};
    };

    //! Resident mode requires slotCount == rowCount + 1 and finishes bounded preload before returning.
    PleRowStore(PleTable table, int32_t slotCount, cudaStream_t stream, PleIoConfig io = {},
        Residency residency = Residency::kCACHED, std::shared_ptr<PleIoBudget> ioBudget = nullptr);
    ~PleRowStore() noexcept;
    PleRowStore(PleRowStore const&) = delete;
    PleRowStore& operator=(PleRowStore const&) = delete;

    std::vector<RequestLease> prepare(std::vector<std::vector<int64_t>> const& requests);
    Tensor& pool();
    //! Counters for the most recent admitted prepare, including partial reads on failure.
    Stats const& stats() const noexcept
    {
        return mStats;
    }
    Stats const& preloadStats() const noexcept
    {
        return mPreloadStats;
    }

private:
    struct Resources;
    bool drain() noexcept;
    void fillRows(std::vector<PleCache::Miss> const& misses, PleCache::BatchReservation* batch);
    void preload();

    PleTable mTable;
    std::unique_ptr<PleCache> mCache;
    Residency mResidency;
    cudaStream_t mStream;
    size_t mStagingRows;
    size_t mReadBytes;
    size_t mPageBytes;
    std::unique_ptr<Resources> mResources;
    Stats mStats;
    Stats mPreloadStats;
    bool mPoisoned{false};
};

} // namespace rt
} // namespace trt_edgellm
