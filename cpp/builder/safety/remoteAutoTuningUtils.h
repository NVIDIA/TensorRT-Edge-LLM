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

#include "common/logger.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <string_view>

namespace trt_edgellm::safety::builder
{

inline bool validateRemoteAutoTuningConfig(std::string_view config) noexcept
{
    if (config.empty())
    {
        LOG_ERROR("Remote auto-tuning configuration is empty.");
        return false;
    }

    size_t const protocolEnd = config.find("://");
    if (protocolEnd == std::string_view::npos || protocolEnd == 0)
    {
        LOG_ERROR(
            "Invalid remote auto-tuning URI: expected protocol://username[:password]@hostname[:port]?[parameters].");
        return false;
    }
    if (!std::all_of(config.begin(), config.begin() + static_cast<std::ptrdiff_t>(protocolEnd), [](char ch) {
            auto const value = static_cast<unsigned char>(ch);
            return std::isalnum(value) != 0 || ch == '+' || ch == '-' || ch == '.';
        }))
    {
        LOG_ERROR("Remote auto-tuning protocol contains unsupported characters.");
        return false;
    }

    size_t const authorityBegin = protocolEnd + 3;
    size_t const authorityEnd = config.find_first_of("/?#", authorityBegin);
    std::string_view const authority = config.substr(authorityBegin, authorityEnd - authorityBegin);
    size_t const at = authority.rfind('@');
    if (at == std::string_view::npos || at == 0 || at + 1 == authority.size())
    {
        LOG_ERROR("Remote auto-tuning authority must contain a username and hostname.");
        return false;
    }
    return true;
}

//! An omitted URI selects local tuning for Safety proxy builds; the production Safety flow uses remote tuning.
inline bool validateSafetyAutoTuningConfig(std::string_view config) noexcept
{
    if (config.empty())
    {
        return true;
    }
    return validateRemoteAutoTuningConfig(config);
}

enum class TensorRTCpuOnlyOptionState
{
    kABSENT,
    kENABLED,
    kCONFLICTING
};

//! Detect conflicting CPU-only settings before adding the option required for remote tuning.
inline TensorRTCpuOnlyOptionState getTensorRTCpuOnlyOptionState(std::string_view internalOptions) noexcept
{
    auto state = TensorRTCpuOnlyOptionState::kABSENT;
    size_t pos = 0;
    while (pos < internalOptions.size())
    {
        while (pos < internalOptions.size() && std::isspace(static_cast<unsigned char>(internalOptions[pos])) != 0)
        {
            ++pos;
        }
        size_t const end = internalOptions.find_first_of(" \t\r\n", pos);
        std::string_view const option = internalOptions.substr(pos, end - pos);
        if (option == "--cpu_only" || option == "--cpu_only=1")
        {
            state = TensorRTCpuOnlyOptionState::kENABLED;
        }
        else if (option.rfind("--cpu_only=", 0) == 0)
        {
            // Reject conflicting duplicates rather than relying on TensorRT's option precedence.
            return TensorRTCpuOnlyOptionState::kCONFLICTING;
        }
        if (end == std::string_view::npos)
        {
            break;
        }
        pos = end + 1;
    }
    return state;
}

//! Process-scoped TensorRT CPU-only mode for Safety remote auto-tuning.
//!
//! TRT 11.3 lacks a public CPU-only configuration API. Follow TRT's trtexec
//! approach by setting TRT_INTERNAL_OPTIONS=--cpu_only=1 before builder creation.
//! Restore the caller's environment after the build.
//! TODO: Use the public CPU-only configuration API when available.
class ScopedTensorRTCpuOnlyMode
{
public:
    ScopedTensorRTCpuOnlyMode() = default;
    ScopedTensorRTCpuOnlyMode(ScopedTensorRTCpuOnlyMode const&) = delete;
    ScopedTensorRTCpuOnlyMode& operator=(ScopedTensorRTCpuOnlyMode const&) = delete;

    ~ScopedTensorRTCpuOnlyMode() noexcept
    {
        restore();
    }

    bool enable() noexcept
    {
        if (mActive)
        {
            return true;
        }
        char const* const current = std::getenv("TRT_INTERNAL_OPTIONS");
        mHadPreviousValue = current != nullptr;
        mPreviousValue = current != nullptr ? current : "";
        auto const state = getTensorRTCpuOnlyOptionState(mPreviousValue);
        if (state == TensorRTCpuOnlyOptionState::kCONFLICTING)
        {
            LOG_ERROR(
                "Remote auto-tuning requires CPU-only mode; remove conflicting --cpu_only values from "
                "TRT_INTERNAL_OPTIONS or set --cpu_only=1.");
            return false;
        }
        if (state == TensorRTCpuOnlyOptionState::kENABLED)
        {
            mActive = true;
            return true;
        }

        std::string updated = mPreviousValue;
        if (!updated.empty())
        {
            updated += ' ';
        }
        updated += "--cpu_only=1";
        if (setenv("TRT_INTERNAL_OPTIONS", updated.c_str(), 1) != 0)
        {
            LOG_ERROR("Failed to enable TensorRT CPU-only mode for remote auto-tuning (errno=%d).", errno);
            return false;
        }
        mChangedEnvironment = true;
        mActive = true;
        return true;
    }

private:
    void restore() noexcept
    {
        if (!mChangedEnvironment)
        {
            return;
        }
        int const result = mHadPreviousValue ? setenv("TRT_INTERNAL_OPTIONS", mPreviousValue.c_str(), 1)
                                             : unsetenv("TRT_INTERNAL_OPTIONS");
        if (result != 0)
        {
            LOG_WARNING("Failed to restore TRT_INTERNAL_OPTIONS after remote auto-tuning (errno=%d).", errno);
        }
        mChangedEnvironment = false;
    }

    std::string mPreviousValue;
    bool mHadPreviousValue{false};
    bool mChangedEnvironment{false};
    bool mActive{false};
};

} // namespace trt_edgellm::safety::builder
