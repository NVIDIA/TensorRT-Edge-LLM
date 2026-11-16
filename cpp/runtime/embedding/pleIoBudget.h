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

#include <cstddef>
#include <memory>
#include <mutex>

namespace trt_edgellm
{
namespace rt
{

struct PleIoConfig
{
    size_t fillBytes{4U << 20};     //!< Packed source payload per cached upload, excluding slot IDs.
    size_t preloadBytes{16U << 20}; //!< Sequential source payload per resident upload.
    size_t readBytes{1U << 20};     //!< Maximum buffered read size per reader.
    size_t readers{4};
};

//! Bounds the simultaneously live PLE I/O buffers within one runtime.
class PleIoBudget : public std::enable_shared_from_this<PleIoBudget>
{
public:
    static constexpr size_t kLIMIT_BYTES = 64U << 20;

    //! Must outlive the buffers charged to it. Destruction returns its bytes to the shared budget.
    class Reservation
    {
    public:
        ~Reservation() noexcept;
        Reservation(Reservation const&) = delete;
        Reservation& operator=(Reservation const&) = delete;
        Reservation(Reservation&&) = delete;
        Reservation& operator=(Reservation&&) = delete;

        size_t rows() const noexcept
        {
            return mRows;
        }
        size_t readBytes() const noexcept
        {
            return mReadBytes;
        }
        size_t readers() const noexcept
        {
            return mReaders;
        }
        size_t hostBytes() const noexcept
        {
            return mHostBytes;
        }
        bool adjusted() const noexcept
        {
            return mAdjusted;
        }

    private:
        friend class PleIoBudget;
        Reservation(std::shared_ptr<PleIoBudget> budget, size_t rows, size_t readBytes, size_t readers,
            size_t hostBytes, bool adjusted) noexcept;
        std::shared_ptr<PleIoBudget> mBudget;
        size_t mRows;
        size_t mReadBytes;
        size_t mReaders;
        size_t mHostBytes;
        bool mAdjusted;
    };

    //! Validates requested sizes before reducing batches and reader buffers to fit the remaining budget.
    std::unique_ptr<Reservation> reserve(
        PleIoConfig const& io, size_t sourceRowBytes, size_t maxRows, size_t pageBytes, bool cached);
    size_t usedBytes() const;

private:
    mutable std::mutex mMutex;
    size_t mUsedBytes{};
};

} // namespace rt
} // namespace trt_edgellm
