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

#include "common/pagedKvTypes.h"
#include "runtime/exec/scheduledStep.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Optional engine-wide sliding-window context-cache configuration.
struct SwaCacheConfig
{
    int32_t windowSizeTokens{};
    int32_t replacementPageCount{kSWA_REPLACEMENT_PAGES};
};

inline bool operator==(SwaCacheConfig const& lhs, SwaCacheConfig const& rhs) noexcept
{
    return lhs.windowSizeTokens == rhs.windowSizeTokens && lhs.replacementPageCount == rhs.replacementPageCount;
}

//! One absolute logical-to-physical binding in a bounded SWA window.
struct SwaPageBinding
{
    int32_t logicalPage{};
    PageId page{};
};

inline bool operator==(SwaPageBinding const& lhs, SwaPageBinding const& rhs) noexcept
{
    return lhs.logicalPage == rhs.logicalPage && lhs.page == rhs.page;
}

enum class SwaRetentionMode : uint8_t
{
    //! Live state retains the replay page preceding the latest publishable endpoint.
    kLive,
    //! A full-input hit exposes only the history needed to replay the final complete page.
    kFullInputRewind,
};

//! The immutable final record page is a replay source rather than an active binding during exact-input rewind.
inline int32_t swaReservationPageCredit(SwaRetentionMode mode) noexcept
{
    return mode == SwaRetentionMode::kFullInputRewind ? 1 : 0;
}

//! Settled SWA state that can be stored immutably in a context-cache record.
struct SwaWindowState
{
    int32_t pageSizeTokens{};
    int32_t windowSizeTokens{};
    int32_t exactResidentTokenCount{};
    int32_t retainedLogicalPageBegin{};
    int32_t retainedLogicalPageEnd{};
    std::deque<SwaPageBinding> pageBindings;
    SwaRetentionMode retentionMode{SwaRetentionMode::kLive};
};

inline bool operator==(SwaWindowState const& lhs, SwaWindowState const& rhs) noexcept
{
    return lhs.pageSizeTokens == rhs.pageSizeTokens && lhs.windowSizeTokens == rhs.windowSizeTokens
        && lhs.exactResidentTokenCount == rhs.exactResidentTokenCount
        && lhs.retainedLogicalPageBegin == rhs.retainedLogicalPageBegin
        && lhs.retainedLogicalPageEnd == rhs.retainedLogicalPageEnd && lhs.pageBindings == rhs.pageBindings
        && lhs.retentionMode == rhs.retentionMode;
}

//! Request-local execution state. The inherited settled fields remain directly accessible to existing runtime code;
//! the additional members are transient and are never stored in a cache record.
struct SwaKVCacheState : SwaWindowState
{
    ResidentRef resident;
    std::vector<PageId> reservedPages;
    std::vector<SwaPageBinding> newPageBindings;
    std::vector<SwaPageBinding> pendingRetireBindings;
    std::optional<int32_t> pendingEndpoint;
};

//! Non-owning access to the bounded manager's settled per-sequence windows. The manager retains ownership and the
//! view is valid only for the synchronous completion call in which it is supplied.
class SwaWindowStateView
{
public:
    SwaWindowStateView() = default;
    SwaWindowStateView(SwaKVCacheState const* states, size_t size) noexcept
        : mStates(states)
        , mSize(size)
    {
    }

    size_t size() const noexcept
    {
        return mSize;
    }

    bool empty() const noexcept
    {
        return mSize == 0;
    }

    SwaWindowState const& operator[](size_t slot) const
    {
        return mStates[slot];
    }

private:
    SwaKVCacheState const* mStates{};
    size_t mSize{};
};

} // namespace rt
} // namespace trt_edgellm
