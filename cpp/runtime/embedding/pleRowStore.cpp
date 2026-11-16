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

#include "pleRowStore.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/pleScatter.h"
#include "pleFileUtils.h"
#include "pleReadExecutor.h"
#include "pleReadPlan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{
trt_edgellm::DataType tensorType(PleTable::DType dtype)
{
    switch (dtype)
    {
    case PleTable::DType::kFP8: return trt_edgellm::DataType::kFP8;
    case PleTable::DType::kFP16: return trt_edgellm::DataType::kHALF;
    case PleTable::DType::kBF16: return trt_edgellm::DataType::kBF16;
    case PleTable::DType::kFP32: return trt_edgellm::DataType::kFLOAT;
    }
    throw std::invalid_argument("Unsupported PLE storage dtype");
}

size_t slotOffset(size_t count, int64_t rowBytes)
{
    return (count * static_cast<size_t>(rowBytes) + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
}

struct PlannedRow
{
    PleCache::Miss miss;
    PleTable::RowLocation location;
};

} // namespace

struct PleRowStore::Resources
{
    struct Staging
    {
        Staging(int64_t rowBytes, std::unique_ptr<PleIoBudget::Reservation> reservation)
            : budget(std::move(reservation))
            , host({static_cast<int64_t>(slotOffset(budget->rows(), rowBytes) + budget->rows() * sizeof(int32_t))},
                  DeviceType::kCPU, trt_edgellm::DataType::kINT8)
            , device(host.getShape(), DeviceType::kGPU, trt_edgellm::DataType::kINT8)
        {
        }
        ~Staging()
        {
            reader.reset();
            for (PleFileHandle file : files)
            {
                closePleFile(file);
            }
        }
        // Buffers must be freed before their reservation, including resource quarantine after a failed drain.
        std::unique_ptr<PleIoBudget::Reservation> budget;
        Tensor host;
        Tensor device;
        std::vector<PleFileHandle> files;
        std::vector<int64_t> shardEnds;
        std::unique_ptr<PleReadExecutor> reader;
    };
    Resources(PleTable const& table, int32_t slots, std::unique_ptr<PleIoBudget::Reservation> reservation)
        : pool({slots, table.rowWidth()}, DeviceType::kGPU, tensorType(table.dtype()))
        , staging(std::make_unique<Staging>(table.sourceRowBytes(), std::move(reservation)))
    {
    }
    Tensor pool;
    std::unique_ptr<Staging> staging;
};

PleRowStore::PleRowStore(PleTable table, int32_t slotCount, cudaStream_t stream, PleIoConfig io, Residency residency,
    std::shared_ptr<PleIoBudget> ioBudget)
    : mTable(std::move(table))
    , mResidency(residency)
    , mStream(stream)
    , mStagingRows(0)
    , mReadBytes(io.readBytes)
    , mPageBytes(0)
{
    size_t const pageBytes = pleSystemPageSize();
    auto const maxBytes = std::numeric_limits<int64_t>::max();
    auto const sourceRowBytes = mTable.sourceRowBytes();
    auto const rowBytes = mTable.rowBytes();
    if (stream == nullptr || pageBytes == 0)
    {
        throw std::invalid_argument("PLE requires an explicit stream and a positive system page size");
    }
    mPageBytes = pageBytes;
    if (slotCount < 2 || rowBytes > maxBytes / slotCount)
    {
        throw std::invalid_argument("PLE pool byte size is invalid");
    }
    if (residency == Residency::kRESIDENT)
    {
        if (mTable.rowCount() >= std::numeric_limits<int32_t>::max() || slotCount != mTable.rowCount() + 1)
        {
            throw std::invalid_argument("Resident PLE requires exactly one slot per row plus padding");
        }
    }
    bool const cached = mResidency == Residency::kCACHED;
    size_t const maxRows = cached ? std::min(static_cast<size_t>(mTable.rowCount()), static_cast<size_t>(slotCount - 1))
                                  : static_cast<size_t>(mTable.rowCount());
    if (!ioBudget)
    {
        ioBudget = std::make_shared<PleIoBudget>();
    }
    auto reservation = ioBudget->reserve(io, static_cast<size_t>(sourceRowBytes), maxRows, mPageBytes, cached);
    mStagingRows = reservation->rows();
    mReadBytes = reservation->readBytes();
    if (cached)
    {
        // Rows are read on demand, so the source files stay a live dependency for the whole run.
        std::set<std::filesystem::path> files;
        for (auto const& shard : mTable.shards())
        {
            files.insert(shard.path);
        }
        if (files.size() == 1)
        {
            LOG_WARNING("Cached PLE reads '%s' throughout the runtime lifetime. Keep it accessible and unchanged.",
                files.begin()->c_str());
        }
        else
        {
            LOG_WARNING(
                "Cached PLE reads %zu files under '%s' throughout the runtime lifetime. Keep them accessible and "
                "unchanged.",
                files.size(), files.begin()->parent_path().c_str());
        }
        mCache = std::make_unique<PleCache>(mTable.rowCount(), slotCount);
    }
    mResources = std::make_unique<Resources>(mTable, slotCount, std::move(reservation));
    try
    {
        auto& staging = *mResources->staging;
        staging.files.reserve(mTable.shards().size());
        for (auto const& shard : mTable.shards())
        {
            staging.files.push_back(openPleFile(shard.path));
            staging.shardEnds.push_back(shard.byteOffset + shard.byteLength);
        }
        CUDA_CHECK(cudaMemsetAsync(mResources->pool.rawPointer(), 0, rowBytes, mStream));
        if (!drain())
        {
            throw std::runtime_error("PLE zero-slot initialization did not complete");
        }
        if (mResidency == Residency::kRESIDENT)
        {
            preload();
            mResources->staging.reset();
        }
        else
        {
            staging.reader = std::make_unique<PleReadExecutor>(staging.files, staging.budget->readers(), mReadBytes);
        }
    }
    catch (...)
    {
        mResources->staging->reader.reset();
        if (!drain())
        {
            (void) mResources.release();
        }
        throw;
    }
}

PleRowStore::~PleRowStore() noexcept
{
    if (mResources && mResources->staging)
    {
        mResources->staging->reader.reset();
    }
    if (mResources && !drain())
    {
        // An unconfirmed stream can still access the GPU pool and pinned staging.
        LOG_ERROR("PLE store quarantines its resource bundle because stream completion could not be established.");
        (void) mResources.release();
    }
}

bool PleRowStore::drain() noexcept
{
    cudaError_t const result = cudaStreamSynchronize(mStream);
    if (result != cudaSuccess)
    {
        mPoisoned = true;
        LOG_ERROR("PLE stream drain failed: %s", cudaGetErrorString(result));
    }
    return result == cudaSuccess;
}

Tensor& PleRowStore::pool()
{
    if (mPoisoned)
    {
        throw std::runtime_error("PLE store is poisoned after a CUDA completion failure");
    }
    return mResources->pool;
}

PleRowStore::RequestLease::RequestLease(PleCache::RequestLease lease)
    : mCached(std::move(lease))
{
}

PleRowStore::RequestLease::RequestLease(std::vector<int32_t> mapping)
    : mMapping(std::move(mapping))
{
}

PleRowStore::RequestLease::RequestLease(RequestLease&& other) noexcept
    : mCached(std::move(other.mCached))
    , mMapping(std::move(other.mMapping))
    , mValid(std::exchange(other.mValid, false))
{
}

PleRowStore::RequestLease& PleRowStore::RequestLease::operator=(RequestLease&& other) noexcept
{
    if (this != &other)
    {
        mCached = std::move(other.mCached);
        mMapping = std::move(other.mMapping);
        mValid = std::exchange(other.mValid, false);
    }
    return *this;
}

std::vector<int32_t> const& PleRowStore::RequestLease::mapping() const
{
    if (!mValid)
    {
        throw std::logic_error("PLE request lease has been released");
    }
    return mCached ? mCached->mapping() : mMapping;
}

void PleRowStore::RequestLease::release() noexcept
{
    mCached.reset();
    mValid = false;
}

std::vector<PleRowStore::RequestLease> PleRowStore::prepare(std::vector<std::vector<int64_t>> const& requests)
{
    (void) pool();
    std::vector<RequestLease> leases;
    leases.reserve(requests.size());
    if (mResidency == Residency::kRESIDENT)
    {
        mStats = {};
        for (auto const& request : requests)
        {
            std::vector<int32_t> mapping;
            mapping.reserve(request.size());
            for (int64_t row : request)
            {
                if (row < -1 || row >= mTable.rowCount())
                {
                    throw std::out_of_range("PLE row is outside table");
                }
                mapping.push_back(static_cast<int32_t>(row + 1));
            }
            leases.push_back(RequestLease(std::move(mapping)));
        }
        return leases;
    }
    auto batch = mCache->reserveBatch(requests);
    mStats = {};
    try
    {
        mStats.uniqueRows = batch.uniqueRows();
        mStats.misses = batch.misses().size();
        mStats.usefulBytes = mStats.misses * mTable.sourceRowBytes();
        fillRows(batch.misses(), &batch);
        auto cached = batch.commit();
        for (auto& lease : cached)
        {
            leases.push_back(RequestLease(std::move(lease)));
        }
        return leases;
    }
    catch (...)
    {
        (void) drain();
        batch.rollback();
        throw;
    }
}

void PleRowStore::fillRows(std::vector<PleCache::Miss> const& misses, PleCache::BatchReservation* batch)
{
    auto const rowBytes = mTable.sourceRowBytes();
    std::vector<PlannedRow> plan;
    plan.reserve(misses.size());
    for (auto const& miss : misses)
    {
        plan.push_back({miss, mTable.locate(miss.row)});
    }
    std::sort(plan.begin(), plan.end(), [](PlannedRow const& a, PlannedRow const& b) {
        return std::tie(a.location.shardIndex, a.location.byteOffset)
            < std::tie(b.location.shardIndex, b.location.byteOffset);
    });
    auto& resources = *mResources->staging;
    for (size_t start = 0; start < plan.size(); start += mStagingRows)
    {
        size_t const count = std::min(mStagingRows, plan.size() - start);
        auto* packed = static_cast<char*>(resources.host.rawPointer());
        auto* slots = reinterpret_cast<int32_t*>(packed + slotOffset(count, rowBytes));
        std::vector<PleReadSpan> spans;
        spans.reserve(count);
        for (size_t row = 0; row < count; ++row)
        {
            auto const& entry = plan[start + row];
            slots[row] = entry.miss.slot;
            spans.push_back({entry.location.shardIndex, entry.location.byteOffset, static_cast<size_t>(rowBytes),
                row * static_cast<size_t>(rowBytes)});
        }
        auto const reads = planPleReads(spans, resources.shardEnds, mPageBytes, mReadBytes);
        resources.reader->read(reads, packed, mStats.readCalls, mStats.readBytes);
        size_t const stagingBytes = slotOffset(count, rowBytes) + count * sizeof(int32_t);
        CUDA_CHECK(cudaMemcpyAsync(
            resources.device.rawPointer(), resources.host.rawPointer(), stagingBytes, cudaMemcpyHostToDevice, mStream));
        auto const* deviceSlots = reinterpret_cast<int32_t const*>(
            static_cast<char const*>(resources.device.rawPointer()) + slotOffset(count, rowBytes));
        if (mTable.sourceDtype() == mTable.dtype() && mTable.sourceScale() == 1.0)
        {
            CUDA_CHECK(kernel::launchPleScatter(resources.device.rawPointer(), deviceSlots,
                mResources->pool.rawPointer(), static_cast<int64_t>(count), rowBytes, mStream));
        }
        else
        {
            CUDA_CHECK(kernel::launchPleScatterConvert(resources.device.rawPointer(), deviceSlots,
                mResources->pool.rawPointer(), static_cast<int64_t>(count), mTable.rowWidth(),
                tensorType(mTable.sourceDtype()), static_cast<float>(mTable.sourceScale()), mStream));
        }
        if (!drain())
        {
            throw std::runtime_error("PLE fill wave did not complete");
        }
        ++mStats.fillWaves;
        for (size_t row = 0; row < count; ++row)
        {
            if (batch != nullptr)
            {
                batch->markReady(plan[start + row].miss.slot);
            }
        }
    }
}

void PleRowStore::preload()
{
    auto& staging = *mResources->staging;
    auto const sourceBytes = mTable.sourceRowBytes();
    bool const direct = mTable.sourceDtype() == mTable.dtype() && mTable.sourceScale() == 1.0;
    for (size_t shardIndex = 0; shardIndex < mTable.shards().size(); ++shardIndex)
    {
        auto const& shard = mTable.shards()[shardIndex];
        for (int64_t start = 0; start < shard.rowCount; start += static_cast<int64_t>(mStagingRows))
        {
            int64_t const count = std::min(static_cast<int64_t>(mStagingRows), shard.rowCount - start);
            size_t const bytes = static_cast<size_t>(count * sourceBytes);
            auto* host = static_cast<char*>(staging.host.rawPointer());
            size_t copied{0};
            while (copied < bytes)
            {
                ++mPreloadStats.readCalls;
                size_t const result = readPleFileAt(staging.files[shardIndex], host + copied, bytes - copied,
                    shard.byteOffset + start * sourceBytes + static_cast<int64_t>(copied), "Preloading PLE rows");
                if (result == 0)
                {
                    throw std::runtime_error("PLE shard ended during resident preload");
                }
                copied += result;
                mPreloadStats.readBytes += result;
            }
            auto* destination
                = static_cast<char*>(mResources->pool.rawPointer()) + (shard.rowBegin + start + 1) * mTable.rowBytes();
            void* upload = direct ? destination : staging.device.rawPointer();
            CUDA_CHECK(cudaMemcpyAsync(upload, host, bytes, cudaMemcpyHostToDevice, mStream));
            if (!direct)
            {
                CUDA_CHECK(kernel::launchPleConvert(staging.device.rawPointer(), destination, count, mTable.rowWidth(),
                    tensorType(mTable.sourceDtype()), static_cast<float>(mTable.sourceScale()), mStream));
            }
            if (!drain())
            {
                throw std::runtime_error("PLE resident preload did not complete");
            }
            mPreloadStats.uniqueRows += static_cast<size_t>(count);
            mPreloadStats.usefulBytes += bytes;
            ++mPreloadStats.fillWaves;
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
