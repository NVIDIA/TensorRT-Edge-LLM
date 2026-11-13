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

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace trt_edgellm::safety
{

constexpr std::size_t kSAFETY_STATUS_DESCRIPTION_CAPACITY{256U};

//! Error category; SafetyStatus::fatalStatus() separately specifies whether the error is fatal.
enum class SafetyStatusCode
{
    kSuccess,          //!< Operation completed successfully.
    kInvalidArgument,  //!< Invalid input or configuration.
    kInvalidState,     //!< Operation is incompatible with the current lifecycle or runtime state.
    kFailedExecution,  //!< Execution failure or unclassified backend error.
    kCudaError,        //!< Error reported by a CUDA API.
    kFailedAllocation, //!< Failed to allocate required memory or resources.
};

//! One operation's result: error category, bounded diagnostic, and fatal flag.
//! The caller applies fatal status to the runtime lifecycle; constructing and copying diagnostics never allocates.
class SafetyStatus
{
public:
    static SafetyStatus ok() noexcept
    {
        return {};
    }

    static SafetyStatus error(SafetyStatusCode code, std::string_view message = {}) noexcept
    {
        return SafetyStatus(code, message, false);
    }

    static SafetyStatus error(SafetyStatusCode code, char const* message) noexcept
    {
        return error(code, message == nullptr ? std::string_view{} : std::string_view{message});
    }

    static SafetyStatus fatal(SafetyStatusCode code, std::string_view message = {}) noexcept
    {
        return SafetyStatus(code, message, true);
    }

    static SafetyStatus fatal(SafetyStatusCode code, char const* message) noexcept
    {
        return fatal(code, message == nullptr ? std::string_view{} : std::string_view{message});
    }

    bool okStatus() const noexcept
    {
        return mCode == SafetyStatusCode::kSuccess;
    }

    SafetyStatusCode code() const noexcept
    {
        return mCode;
    }

    char const* message() const noexcept
    {
        return mMessage.data();
    }

    bool fatalStatus() const noexcept
    {
        return mFatal;
    }

private:
    SafetyStatus() noexcept = default;

    SafetyStatus(SafetyStatusCode code, std::string_view message, bool fatal) noexcept
        : mCode(code == SafetyStatusCode::kSuccess ? SafetyStatusCode::kInvalidState : code)
        , mFatal(fatal)
    {
        // Truncation affects diagnostics only; the status code and fatal flag are retained.
        auto const count = std::min(message.size(), mMessage.size() - 1U);
        if (count != 0U)
        {
            std::copy_n(message.data(), count, mMessage.data());
        }
    }

    SafetyStatusCode mCode{SafetyStatusCode::kSuccess};
    std::array<char, kSAFETY_STATUS_DESCRIPTION_CAPACITY> mMessage{};
    bool mFatal{false};
};

} // namespace trt_edgellm::safety
