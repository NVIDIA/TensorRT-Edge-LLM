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

#include "runtime/embedding/pleIoBudget.h"

#include "common/logger.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

constexpr size_t kSLOT_BYTES = sizeof(int32_t);
constexpr size_t kMAX_TENSOR_BYTES = static_cast<size_t>(std::numeric_limits<int64_t>::max());

size_t stagingBytes(size_t rows, size_t sourceRowBytes)
{
    if (rows > kMAX_TENSOR_BYTES / sourceRowBytes)
    {
        throw std::overflow_error("PLE I/O source staging exceeds tensor size range");
    }
    size_t const payload = rows * sourceRowBytes;
    if (payload > kMAX_TENSOR_BYTES - (kSLOT_BYTES - 1))
    {
        throw std::overflow_error("PLE I/O staging alignment exceeds tensor size range");
    }
    size_t const aligned = (payload + kSLOT_BYTES - 1) / kSLOT_BYTES * kSLOT_BYTES;
    if (rows > (kMAX_TENSOR_BYTES - aligned) / kSLOT_BYTES)
    {
        throw std::overflow_error("PLE I/O slot staging exceeds tensor size range");
    }
    return aligned + rows * kSLOT_BYTES;
}

} // namespace

PleIoBudget::Reservation::Reservation(std::shared_ptr<PleIoBudget> budget, size_t rows, size_t readBytes,
    size_t readers, size_t hostBytes, bool adjusted) noexcept
    : mBudget(std::move(budget))
    , mRows(rows)
    , mReadBytes(readBytes)
    , mReaders(readers)
    , mHostBytes(hostBytes)
    , mAdjusted(adjusted)
{
}

PleIoBudget::Reservation::~Reservation() noexcept
{
    std::lock_guard<std::mutex> lock(mBudget->mMutex);
    mBudget->mUsedBytes -= mHostBytes;
}

std::unique_ptr<PleIoBudget::Reservation> PleIoBudget::reserve(
    PleIoConfig const& io, size_t sourceRowBytes, size_t maxRows, size_t pageBytes, bool cached)
{
    size_t const payloadBytes = cached ? io.fillBytes : io.preloadBytes;
    if (sourceRowBytes == 0 || maxRows == 0 || pageBytes == 0 || io.readers == 0 || io.readers > 32
        || io.readBytes < pageBytes || io.readBytes % pageBytes != 0 || payloadBytes < sourceRowBytes)
    {
        throw std::invalid_argument("Invalid PLE I/O row count, payload, reader count or page-aligned read size");
    }
    size_t const requestedRows = std::min(maxRows, payloadBytes / sourceRowBytes);
    stagingBytes(requestedRows, sourceRowBytes);
    if (io.readBytes > kMAX_TENSOR_BYTES / io.readers)
    {
        throw std::overflow_error("PLE I/O reader buffers exceed tensor size range");
    }
    size_t const requestedReaders = cached ? io.readers : 0;
    size_t const requestedReadBytes = cached ? io.readBytes : 0;
    size_t const minimumStaging = stagingBytes(1, sourceRowBytes);
    if (cached && pageBytes > kMAX_TENSOR_BYTES - minimumStaging)
    {
        throw std::overflow_error("PLE I/O minimum working set exceeds tensor size range");
    }
    size_t const minimum = minimumStaging + (cached ? pageBytes : 0);
    auto budget = shared_from_this();
    std::unique_lock<std::mutex> lock(mMutex);
    size_t const available = kLIMIT_BYTES - mUsedBytes;
    if (minimum > available)
    {
        throw std::length_error("PLE I/O minimum working set requires " + std::to_string(minimum)
            + " bytes; available=" + std::to_string(available) + ", limit=" + std::to_string(kLIMIT_BYTES));
    }

    size_t readers = requestedReaders;
    size_t readBytes = requestedReadBytes;
    if (cached && readers * readBytes > available - minimumStaging)
    {
        readBytes = (available - minimumStaging) / readers / pageBytes * pageBytes;
        if (readBytes < pageBytes)
        {
            readBytes = pageBytes;
            readers = (available - minimumStaging) / pageBytes;
        }
    }
    size_t const stagingBudget = available - readers * readBytes;
    size_t rows = std::min(requestedRows, stagingBudget / (sourceRowBytes + kSLOT_BYTES));
    if (stagingBytes(rows, sourceRowBytes) > stagingBudget)
    {
        --rows;
    }
    size_t const hostBytes = stagingBytes(rows, sourceRowBytes) + readers * readBytes;
    bool const adjusted = rows != requestedRows || readers != requestedReaders || readBytes != requestedReadBytes;
    auto reservation = std::unique_ptr<Reservation>(
        new Reservation(std::move(budget), rows, readBytes, readers, hostBytes, adjusted));
    mUsedBytes += hostBytes;
    lock.unlock();
    if (adjusted)
    {
        LOG_WARNING(
            "PLE I/O budget adjusted rows %zu -> %zu, readers %zu -> %zu, read bytes %zu -> %zu; "
            "host bytes=%zu, available=%zu, limit=%zu. More batches or slower startup may result.",
            requestedRows, rows, requestedReaders, readers, requestedReadBytes, readBytes, hostBytes, available,
            kLIMIT_BYTES);
    }
    return reservation;
}

size_t PleIoBudget::usedBytes() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mUsedBytes;
}

} // namespace rt
} // namespace trt_edgellm
