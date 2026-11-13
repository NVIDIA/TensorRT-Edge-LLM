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

#include "safetyPluginRegistry.h"

#include "safetyCreatorRegistration.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <new>

namespace trt_edgellm
{
namespace safety
{
namespace
{
// One admitted creator slot is required for each Safety attention role: LLM and visual.
constexpr int32_t kNB_ADMITTED_CREATORS = 2;

//! Fixed-storage recorder for externally serialized process-wide plugin registration.
class ProcessRegistrationRecorder final : public nvinfer2::safe::ISafeRecorder
{
public:
    ProcessRegistrationRecorder() noexcept
        : ISafeRecorder(nvinfer2::safe::Severity::kERROR)
    {
    }

    int32_t getNbErrors() const noexcept override
    {
        return mNbErrors;
    }

    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mRecordedErrors[static_cast<std::size_t>(errorIdx)].code
                                    : nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
    }

    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mRecordedErrors[static_cast<std::size_t>(errorIdx)].description.data() : "";
    }

    bool hasOverflowed() const noexcept override
    {
        return mOverflowed;
    }

    void clear() noexcept override
    {
        mNbErrors = 0;
        mOverflowed = false;
    }

    bool reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept override
    {
        if (mNbErrors < kMAX_DLI_ERRORS)
        {
            auto& recorded = mRecordedErrors[static_cast<std::size_t>(mNbErrors)];
            ++mNbErrors;
            recorded.code = code;
            copyDescription(recorded.description, description);
        }
        else
        {
            mOverflowed = true;
        }
        return true;
    }

    RefCount incRefCount() noexcept override
    {
        return mRefCount.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    RefCount decRefCount() noexcept override
    {
        auto count = mRefCount.load(std::memory_order_acquire);
        while (count != 0)
        {
            if (mRefCount.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel, std::memory_order_acquire))
            {
                return count - 1;
            }
        }
        return 0;
    }

private:
    static constexpr int32_t kMAX_DLI_ERRORS{16};
    static constexpr std::size_t kDESCRIPTION_CAPACITY{ISafeRecorder::kMAX_SAFE_DESC_LENGTH + 1U};

    struct RecordedError
    {
        nvinfer1::ErrorCode code{nvinfer1::ErrorCode::kSUCCESS};
        std::array<char, kDESCRIPTION_CAPACITY> description{};
    };

    static void copyDescription(std::array<char, kDESCRIPTION_CAPACITY>& destination, ErrorDesc source) noexcept
    {
        std::size_t index{};
        if (source != nullptr)
        {
            while (index + 1U < destination.size() && source[index] != '\0')
            {
                destination[index] = source[index];
                ++index;
            }
        }
        destination[index] = '\0';
    }

    bool validIndex(int32_t index) const noexcept
    {
        return index >= 0 && index < mNbErrors;
    }

    std::array<RecordedError, static_cast<std::size_t>(kMAX_DLI_ERRORS)> mRecordedErrors{};
    int32_t mNbErrors{};
    bool mOverflowed{};
    std::atomic<RefCount> mRefCount{0};
};
} // namespace

nvinfer2::safe::ISafeRecorder& processRegistrationRecorder() noexcept
{
    alignas(ProcessRegistrationRecorder) static unsigned char sStorage[sizeof(ProcessRegistrationRecorder)];
    static auto* const sRecorder = new (sStorage) ProcessRegistrationRecorder();
    return *sRecorder;
}

nvinfer1::ErrorCode registerSafetyPlugins(nvinfer2::safe::ISafePluginRegistry& registry) noexcept
{
    using namespace nvinfer1;
    using namespace nvinfer2::safe;
    auto& recorder = processRegistrationRecorder();
    ISafePluginCreatorV3One* selected[kNB_ADMITTED_CREATORS]{
        &safetyAttentionCreator(AttentionRole::kLLM), &safetyAttentionCreator(AttentionRole::kVISUAL)};
    IPluginCreatorInterface* newlyRegistered[kNB_ADMITTED_CREATORS]{};
    int32_t newlyRegisteredCount{};
    return registerSafetyCreators(
        registry, recorder, selected, kNB_ADMITTED_CREATORS, newlyRegistered, newlyRegisteredCount);
}
} // namespace safety
} // namespace trt_edgellm
