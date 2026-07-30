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

#include "runtime/state/contextCache/contextCacheTypes.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Host-side allocation and reference accounting for all context-cache storage types.
//!
//! Each ResourceType has an independent fixed-capacity pool and circular free list. ResourcePools manages only IDs and
//! reference counts; the corresponding GPU buffers are allocated and bound by the runtime. An active reference pins a
//! resource for an in-flight request, while a cache reference retains it for a published CacheRecord. A resource
//! returns to its free list only when both counts reach zero.
//!
//! Multi-pool allocation is all-or-nothing. Valid releases do not allocate and cannot fail after their preconditions
//! have been checked.
class ResourcePools
{
public:
    explicit ResourcePools(ResourceDemand capacities);

    //! Atomically allocate the full typed demand with active=1 and cache=0, or return nullopt without mutation.
    std::optional<std::vector<ResourceId>> allocate(ResourceDemand const& demand);
    //! Fill caller-owned storage and commit without dynamic memory allocation.
    //! The output vector must already match the total demand; insufficient capacity returns false without mutation.
    bool allocateInto(ResourceDemand const& demand, std::vector<ResourceId>& resources);

    //! Pin an allocated resource for an active request.
    void addActiveRef(ResourceId const& resource);
    //! Drop an active pin, returning the resource to the free list only when both reference counts reach zero.
    void releaseActiveRef(ResourceId const& resource);
    //! Add published-record ownership to an allocated resource.
    void addCacheRef(ResourceId const& resource);
    //! Drop published ownership, returning the resource to the free list only when both reference counts reach zero.
    void releaseCacheRef(ResourceId const& resource);

    int32_t activeRefCount(ResourceId const& resource) const;
    int32_t cacheRefCount(ResourceId const& resource) const;

    int32_t freeCount(ResourceType type) const noexcept;
    int32_t capacity(ResourceType type) const noexcept;
    bool canAllocate(ResourceDemand const& demand) const noexcept;

private:
    struct Slot
    {
        int32_t activeRefCount{};
        int32_t cacheRefCount{};
        bool onFreeList{true};
    };

    struct Pool
    {
        std::vector<Slot> slots;
        std::vector<int32_t> freeList;
        size_t freeListHead{};
        size_t freeListCount{};
    };

    Pool& pool(ResourceType type) noexcept;
    Pool const& pool(ResourceType type) const noexcept;
    Slot& slot(ResourceId const& resource);
    Slot const& slot(ResourceId const& resource) const;
    void releaseIfUnused(ResourceId const& resource, Slot& resourceSlot) noexcept;

    std::array<Pool, 4> mPools;
};

} // namespace rt
} // namespace trt_edgellm
