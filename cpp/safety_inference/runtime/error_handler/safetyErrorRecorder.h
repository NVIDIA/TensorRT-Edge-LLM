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

#include <NvInferSafeRecorder.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>

namespace trt_edgellm
{
namespace safety
{

constexpr int32_t kMAX_DLI_ERRORS{16};
//! Total storage for one TensorRT Safety description, including its required NUL terminator.
constexpr std::size_t kDLI_DESCRIPTION_CAPACITY{nvinfer2::safe::ISafeRecorder::kMAX_SAFE_DESC_LENGTH + 1U};
using DliDescription = std::array<char, kDLI_DESCRIPTION_CAPACITY>;

//! No-throw sink used to translate a DLI diagnostic into runtime policy state.
using DliErrorCallback
    = void (*)(void* context, nvinfer1::ErrorCode code, nvinfer1::IErrorRecorder::ErrorDesc description) noexcept;

//! One fixed-storage TensorRT diagnostic retained by SafetyErrorRecorder.
struct RecordedDliError
{
    nvinfer1::ErrorCode code{nvinfer1::ErrorCode::kSUCCESS}; //!< Owned TensorRT error code.
    DliDescription description{};                            //!< Owned, NUL-terminated diagnostic text.
};

static_assert(std::is_standard_layout<RecordedDliError>::value, "RecordedDliError must have standard layout");

//! Fixed-capacity runtime recorder with optional fatal-error forwarding.
//!
//! Retained diagnostics, their count, and overflow state are protected by one mutex. The callback and context are
//! borrowed for the recorder lifetime, and the callback executes only after the diagnostics mutex is released.
class SafetyErrorRecorder final : public nvinfer2::safe::ISafeRecorder
{
public:
    //! Construct an empty recorder.
    //! @param callback Optional no-throw error sink; nullptr disables runtime forwarding.
    //! @param context Borrowed callback context; may be nullptr when callback is nullptr.
    SafetyErrorRecorder(DliErrorCallback callback, void* context) noexcept;

    ~SafetyErrorRecorder() noexcept = default;

    //! Return the retained diagnostic count.
    //! @return A thread-safe count between zero and kMAX_DLI_ERRORS.
    int32_t getNbErrors() const noexcept override;

    //! Return a retained TensorRT error code.
    //! @param errorIdx Zero-based retained diagnostic index.
    //! @return The stored code, or kUNSPECIFIED_ERROR for an invalid index.
    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override;

    //! Return recorder-owned diagnostic text.
    //! @param errorIdx Zero-based retained diagnostic index.
    //! @return The retained slot, or an empty string for an invalid index. The SDK pointer must not be retained across
    //! clear() followed by slot reuse.
    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override;

    //! Copy one diagnostic while holding the recorder lock.
    //! @param errorIdx Zero-based retained diagnostic index.
    //! @param destination Caller-owned fixed storage that remains stable after this call.
    //! @return True when errorIdx selected a retained diagnostic; otherwise destination is empty.
    bool copyErrorDescription(int32_t errorIdx, DliDescription& destination) const noexcept;

    //! Determine whether at least one diagnostic was dropped.
    //! @return True after fixed diagnostic storage has overflowed and until clear() is called.
    bool hasOverflowed() const noexcept override;

    //! Clear retained diagnostic bookkeeping without changing refcount or runtime terminal state.
    void clear() noexcept override;

    //! Retain and forward one TensorRT error.
    //! @param code TensorRT error classification.
    //! @param description Diagnostic text valid for the duration of this call.
    //! @return Always true because every DLI error is fatal to the affected operation.
    bool reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept override;

    //! Acquire one borrowed recorder reference.
    //! @return The incremented reference count.
    RefCount incRefCount() noexcept override;

    //! Release one borrowed recorder reference without deleting the recorder.
    //! @return The decremented count, or zero after reporting an attempted underflow.
    RefCount decRefCount() noexcept override;

    //! Snapshot the current borrower count.
    //! @return The current nonnegative reference count.
    RefCount referenceCount() const noexcept;

private:
    //! Retain one diagnostic or set the overflow flag.
    //! @param code TensorRT error classification.
    //! @param description Diagnostic text copied into owned fixed storage.
    void recordDiagnostic(nvinfer1::ErrorCode code, ErrorDesc description) noexcept;

    //! Validate an index while mDiagnosticsMutex is held.
    //! @param index Candidate retained diagnostic index.
    //! @return True only when the index selects a retained slot.
    bool validIndex(int32_t index) const noexcept;

    DliErrorCallback mCallback;
    void* mContext;
    mutable std::mutex mDiagnosticsMutex;
    std::array<RecordedDliError, static_cast<std::size_t>(kMAX_DLI_ERRORS)> mRecordedErrors{};
    int32_t mNbErrors{0};
    bool mOverflowed{false};
    std::atomic<RefCount> mRefCount{0};
};

} // namespace safety
} // namespace trt_edgellm
