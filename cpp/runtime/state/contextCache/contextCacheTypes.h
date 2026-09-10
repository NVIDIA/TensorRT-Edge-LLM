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
#include <exception>
#include <functional>

namespace trt_edgellm
{
namespace rt
{

//! Host identity of one K/V page pair local to one typed KV pool.
//!
//! For an attention backend this is the logical K-half page ID. The runtime adapter emits both the K ID and the
//! pool-absolute V ID derived from the pool-half offset when it builds the kernel's [K, V] page-table row. Base and
//! draft pools have independent ID spaces.
using PageId = int32_t;
//! Host cache-record identity.
using RecordId = uint64_t;

//! Reserved substrate taxonomy for future speculative-state backends.
//! The current record stores paged state directly; add a discriminator only when a second substrate exists.
enum class SpecStateStorageKind : uint8_t
{
    kPaged,
};

//! Immutable contract between a speculative decoder and context reuse.
struct SpecReuseContract
{
    bool ownsPagedSpecState{};
    int32_t futureDependencyTokens{};
};

enum class ResourceType : uint8_t
{
    //! One host-addressed K/V page pair produced by the base model.
    kBaseKvPage,
    //! One host-addressed K/V page pair produced by the speculative draft model.
    kDraftKvPage,
    //! One page-aligned recurrent-state checkpoint.
    kRecurrentSnapshot,
    //! One snapshot of the partial attention page paired with recurrent state.
    kPartialKvSnapshot,
};

//! Pool-local identity for one typed cache resource.
//!
//! Identical numeric indexes in different ResourceType pools identify different physical resources.
struct ResourceId
{
    ResourceType type{};
    int32_t index{};
};

inline bool operator==(ResourceId const& lhs, ResourceId const& rhs) noexcept
{
    return lhs.type == rhs.type && lhs.index == rhs.index;
}

//! Number of resources required from each typed pool by one atomic operation.
struct ResourceDemand
{
    int32_t baseKvPages{};
    int32_t draftKvPages{};
    int32_t recurrentSnapshotSlots{};
    int32_t partialKvSnapshotSlots{};

    int32_t get(ResourceType type) const noexcept
    {
        switch (type)
        {
        case ResourceType::kBaseKvPage: return baseKvPages;
        case ResourceType::kDraftKvPage: return draftKvPages;
        case ResourceType::kRecurrentSnapshot: return recurrentSnapshotSlots;
        case ResourceType::kPartialKvSnapshot: return partialKvSnapshotSlots;
        }
        std::terminate();
    }

    int32_t& get(ResourceType type) noexcept
    {
        switch (type)
        {
        case ResourceType::kBaseKvPage: return baseKvPages;
        case ResourceType::kDraftKvPage: return draftKvPages;
        case ResourceType::kRecurrentSnapshot: return recurrentSnapshotSlots;
        case ResourceType::kPartialKvSnapshot: return partialKvSnapshotSlots;
        }
        std::terminate();
    }

    bool isNonNegative() const noexcept
    {
        return baseKvPages >= 0 && draftKvPages >= 0 && recurrentSnapshotSlots >= 0 && partialKvSnapshotSlots >= 0;
    }
};

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::ResourceId>
{
    size_t operator()(trt_edgellm::rt::ResourceId const& resource) const noexcept
    {
        size_t const typeHash = hash<uint8_t>{}(static_cast<uint8_t>(resource.type));
        size_t const indexHash = hash<int32_t>{}(resource.index);
        return indexHash ^ (typeHash << 1U);
    }
};

} // namespace std
