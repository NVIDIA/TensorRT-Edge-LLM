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

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

using RequestId = uint64_t;
using CacheSlot = int32_t;
using StepId = uint64_t;

struct ResidentRef
{
    CacheSlot slot{-1};
    uint64_t epoch{0};
};

constexpr bool operator==(ResidentRef const& lhs, ResidentRef const& rhs) noexcept
{
    return lhs.slot == rhs.slot && lhs.epoch == rhs.epoch;
}

constexpr bool operator!=(ResidentRef const& lhs, ResidentRef const& rhs) noexcept
{
    return !(lhs == rhs);
}

enum class SequenceWork : int32_t
{
    kContext,
    kDecode,
};

struct SequenceIdentity
{
    RequestId requestId{0};
    ResidentRef resident;
};

constexpr bool operator==(SequenceIdentity const& lhs, SequenceIdentity const& rhs) noexcept
{
    return lhs.requestId == rhs.requestId && lhs.resident == rhs.resident;
}

constexpr bool operator!=(SequenceIdentity const& lhs, SequenceIdentity const& rhs) noexcept
{
    return !(lhs == rhs);
}

struct HostTokenRange
{
    int32_t const* data{nullptr};
    int32_t extent{0};
    int32_t begin{0};
    int32_t count{0};
};

struct ScheduledSequence
{
    SequenceIdentity identity;
    SequenceWork work{SequenceWork::kContext};
    int32_t queryLength{0};
    int32_t pastLength{0};
    HostTokenRange queryTokens;
    bool selectLastTokenLogits{true};
};

//! Owning, pointer-free description of one executed sequence. Suitable for scheduler ownership
//! and cross-rank consensus; queryTokens remains rank-local execution materialization.
struct ScheduledSequenceDescriptor
{
    SequenceIdentity identity;
    SequenceWork work{SequenceWork::kContext};
    int32_t queryLength{0};
    int32_t pastLength{0};
    bool selectLastTokenLogits{true};
};

constexpr bool operator==(ScheduledSequenceDescriptor const& lhs, ScheduledSequenceDescriptor const& rhs) noexcept
{
    return lhs.identity == rhs.identity && lhs.work == rhs.work && lhs.queryLength == rhs.queryLength
        && lhs.pastLength == rhs.pastLength && lhs.selectLastTokenLogits == rhs.selectLastTokenLogits;
}

struct ScheduledStep
{
    StepId id{0};
    std::vector<ScheduledSequence> sequences;
};

} // namespace rt
} // namespace trt_edgellm
