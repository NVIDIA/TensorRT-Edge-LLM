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

#include "safetyErrorRecorder.h"

#include <algorithm>
#include <mutex>

namespace trt_edgellm
{
namespace safety
{
namespace
{

constexpr char kEMPTY_DESCRIPTION[]{""};
constexpr char kREFCOUNT_UNDERFLOW[]{"Safety recorder reference count underflow"};

template <std::size_t Capacity>
void copyBounded(std::array<char, Capacity>& destination, char const* source) noexcept
{
    static_assert(Capacity > 0U, "Bounded text storage cannot be empty");
    std::size_t index{0U};
    if (source != nullptr)
    {
        while (index + 1U < Capacity && source[index] != 0)
        {
            destination[index] = source[index];
            ++index;
        }
    }
    destination[index] = 0;
}

} // namespace

SafetyErrorRecorder::SafetyErrorRecorder(DliErrorCallback callback, void* context) noexcept
    : nvinfer2::safe::ISafeRecorder(nvinfer2::safe::Severity::kERROR)
    , mCallback(callback)
    , mContext(context)
{
}

int32_t SafetyErrorRecorder::getNbErrors() const noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    return mNbErrors;
}

nvinfer1::ErrorCode SafetyErrorRecorder::getErrorCode(int32_t errorIdx) const noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    if (!validIndex(errorIdx))
    {
        return nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
    }
    return mRecordedErrors[static_cast<std::size_t>(errorIdx)].code;
}

SafetyErrorRecorder::ErrorDesc SafetyErrorRecorder::getErrorDesc(int32_t errorIdx) const noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    if (!validIndex(errorIdx))
    {
        return kEMPTY_DESCRIPTION;
    }
    return mRecordedErrors[static_cast<std::size_t>(errorIdx)].description.data();
}

bool SafetyErrorRecorder::copyErrorDescription(int32_t errorIdx, DliDescription& destination) const noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    if (!validIndex(errorIdx))
    {
        destination.fill(0);
        return false;
    }
    destination = mRecordedErrors[static_cast<std::size_t>(errorIdx)].description;
    return true;
}

bool SafetyErrorRecorder::hasOverflowed() const noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    return mOverflowed;
}

void SafetyErrorRecorder::clear() noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    mNbErrors = 0;
    mOverflowed = false;
}

void SafetyErrorRecorder::recordDiagnostic(nvinfer1::ErrorCode code, ErrorDesc description) noexcept
{
    std::lock_guard<std::mutex> const lock(mDiagnosticsMutex);
    if (mNbErrors < kMAX_DLI_ERRORS)
    {
        RecordedDliError& recorded = mRecordedErrors[static_cast<std::size_t>(mNbErrors)];
        ++mNbErrors;
        recorded.code = code;
        copyBounded(recorded.description, description);
    }
    else
    {
        mOverflowed = true;
    }
}

bool SafetyErrorRecorder::reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept
{
    recordDiagnostic(code, description);
    if (mCallback != nullptr)
    {
        mCallback(mContext, code, description);
    }
    return true;
}

SafetyErrorRecorder::RefCount SafetyErrorRecorder::incRefCount() noexcept
{
    return mRefCount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

SafetyErrorRecorder::RefCount SafetyErrorRecorder::decRefCount() noexcept
{
    auto count = mRefCount.load(std::memory_order_acquire);
    while (count != 0)
    {
        if (mRefCount.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return count - 1;
        }
    }
    reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, kREFCOUNT_UNDERFLOW);
    return 0;
}

SafetyErrorRecorder::RefCount SafetyErrorRecorder::referenceCount() const noexcept
{
    return mRefCount.load(std::memory_order_acquire);
}

bool SafetyErrorRecorder::validIndex(int32_t index) const noexcept
{
    return index >= 0 && index < mNbErrors;
}

} // namespace safety
} // namespace trt_edgellm
