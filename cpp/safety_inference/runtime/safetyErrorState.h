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

#include "error_handler/safetyErrorRecorder.h"
#include "safetyStatusUtils.h"

#include <NvInferSafeRuntime.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <string_view>

namespace trt_edgellm::safety
{

//! Owns or borrows a runtime recorder and retains the first terminal failure after recorder diagnostics are cleared.
class SafetyErrorState
{
public:
    //! Construct the runtime-owned state and its per-runtime recorder.
    SafetyErrorState() noexcept
        : mOwnedRecorder(&SafetyErrorState::forwardDliError, this)
        , mRecorder(mOwnedRecorder)
    {
    }

    //! Construct state around a recorder whose lifetime is managed by the caller.
    explicit SafetyErrorState(nvinfer2::safe::ISafeRecorder& recorder) noexcept
        : mOwnedRecorder(nullptr, nullptr)
        , mRecorder(recorder)
    {
    }

    SafetyErrorState(SafetyErrorState const&) = delete;
    SafetyErrorState& operator=(SafetyErrorState const&) = delete;

    bool fatal() const noexcept
    {
        {
            std::lock_guard<std::mutex> const lock(mMutex);
            if (mFailure.fatalStatus())
            {
                return true;
            }
        }
        return mRecorder.getNbErrors() != 0 || mRecorder.hasOverflowed();
    }

    SafetyStatus status() const noexcept
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return mFailure;
    }

    void setFatal(SafetyStatus const& status) noexcept
    {
        (void) latch(status);
    }

    nvinfer2::safe::ISafeRecorder& recorder() noexcept
    {
        return mRecorder;
    }

    bool observe() noexcept
    {
        if (fatal())
        {
            setFatal(SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety recorder reported a failure"));
            return false;
        }
        return true;
    }

    SafetyStatus reject(SafetyStatusCode code, char const* message) noexcept
    {
        auto const status = SafetyStatus::error(code, message);
        // A recorder may suppress a warning; suppression does not turn rejection into a terminal error.
        (void) mRecorder.reportWarn(status.message());
        return status;
    }

    bool updateStatus(SafetyStatus const& status, nvinfer2::safe::ErrorCode code) noexcept
    {
        if (!status.okStatus())
        {
            if (latch(status))
            {
                mRecorder.reportError(code, status.message());
            }
            return false;
        }
        return observe();
    }

    //! Backend failures are terminal at the owning layer; keep their original category and diagnostics.
    SafetyStatus propagateFailure(SafetyStatus const& status, nvinfer2::safe::ErrorCode code) noexcept
    {
        if (!status.okStatus() && latch(status))
        {
            if (mRecorder.getNbErrors() == 0 && !mRecorder.hasOverflowed())
            {
                mRecorder.reportError(code, status.message());
            }
        }
        return this->status();
    }

    SafetyStatus fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept
    {
        updateStatus(fromTrtError(code, message), code);
        return status();
    }

    SafetyStatus fail(nvinfer2::safe::ErrorCode code, std::string_view context, char const* detail) noexcept
    {
        std::array<char, 256> message{};
        size_t offset{0};
        auto append = [&message, &offset](std::string_view text) {
            auto const count = std::min(text.size(), message.size() - 1 - offset);
            std::copy_n(text.data(), count, message.data() + offset);
            offset += count;
        };
        append(context);
        append(": ");
        if (detail != nullptr)
        {
            while (offset < message.size() - 1 && *detail != '\0')
            {
                message[offset++] = *detail++;
            }
        }
        return fail(code, message.data());
    }

private:
    static void forwardDliError(
        void* context, nvinfer1::ErrorCode code, nvinfer1::IErrorRecorder::ErrorDesc description) noexcept
    {
        auto& state = *static_cast<SafetyErrorState*>(context);
        auto const message = description == nullptr ? std::string_view{} : std::string_view{description};
        (void) state.latch(fromTrtError(code, message, false));
    }

    bool latch(SafetyStatus const& status) noexcept
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        if (mFailure.fatalStatus())
        {
            return false;
        }
        mFailure = SafetyStatus::fatal(status.code(), status.message());
        return true;
    }

    mutable std::mutex mMutex;
    SafetyStatus mFailure{SafetyStatus::ok()};
    SafetyErrorRecorder mOwnedRecorder;
    nvinfer2::safe::ISafeRecorder& mRecorder;
};

} // namespace trt_edgellm::safety
