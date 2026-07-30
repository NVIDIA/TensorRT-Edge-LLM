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

#include "runtime/state/contextCache/resourcePools.h"

#include "common/checkMacros.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace trt_edgellm
{
namespace rt
{
namespace
{

constexpr std::array<ResourceType, 4> kRESOURCE_TYPES{ResourceType::kBaseKvPage, ResourceType::kDraftKvPage,
    ResourceType::kRecurrentSnapshot, ResourceType::kPartialKvSnapshot};

size_t poolIndex(ResourceType type) noexcept
{
    return static_cast<size_t>(static_cast<uint8_t>(type));
}

size_t resourceDemandSize(ResourceDemand const& demand)
{
    ELLM_CHECK(demand.isNonNegative(), "Context cache resource demand must be non-negative");

    uint64_t totalResourceCount{};
    for (ResourceType const type : kRESOURCE_TYPES)
    {
        totalResourceCount += static_cast<uint64_t>(demand.get(type));
    }
    ELLM_CHECK(totalResourceCount <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
        "Context cache resource demand is too large");
    return static_cast<size_t>(totalResourceCount);
}

} // namespace

ResourcePools::ResourcePools(ResourceDemand capacities)
{
    ELLM_CHECK(capacities.isNonNegative(), "Context cache resource capacities must be non-negative");

    for (ResourceType const type : kRESOURCE_TYPES)
    {
        Pool& resourcePool = pool(type);
        int32_t const resourceCount = capacities.get(type);
        resourcePool.slots.resize(static_cast<size_t>(resourceCount));
        resourcePool.freeList.resize(static_cast<size_t>(resourceCount));
        resourcePool.freeListHead = 0;
        resourcePool.freeListCount = static_cast<size_t>(resourceCount);
        for (int32_t index = 0; index < resourceCount; ++index)
        {
            resourcePool.freeList[static_cast<size_t>(index)] = index;
        }
    }
}

std::optional<std::vector<ResourceId>> ResourcePools::allocate(ResourceDemand const& demand)
{
    size_t const totalResourceCount = resourceDemandSize(demand);
    if (!canAllocate(demand))
    {
        return std::nullopt;
    }

    std::vector<ResourceId> resources(totalResourceCount);
    if (!allocateInto(demand, resources))
    {
        return std::nullopt;
    }
    return resources;
}

bool ResourcePools::allocateInto(ResourceDemand const& demand, std::vector<ResourceId>& resources)
{
    size_t const totalResourceCount = resourceDemandSize(demand);
    ELLM_CHECK(
        resources.size() == totalResourceCount, "Context cache allocation output size does not match resource demand");
    if (!canAllocate(demand))
    {
        return false;
    }

    size_t outputIndex{};
    for (ResourceType const type : kRESOURCE_TYPES)
    {
        Pool const& resourcePool = pool(type);
        int32_t const resourceCount = demand.get(type);
        for (int32_t index = 0; index < resourceCount; ++index)
        {
            size_t const freeListIndex
                = (resourcePool.freeListHead + static_cast<size_t>(index)) % resourcePool.freeList.size();
            resources[outputIndex] = ResourceId{type, resourcePool.freeList[freeListIndex]};
            ++outputIndex;
        }
    }

    for (ResourceId const& resource : resources)
    {
        Pool& resourcePool = pool(resource.type);
        resourcePool.freeListHead = (resourcePool.freeListHead + 1) % resourcePool.freeList.size();
        --resourcePool.freeListCount;
        Slot& resourceSlot = resourcePool.slots[static_cast<size_t>(resource.index)];
        resourceSlot.activeRefCount = 1;
        resourceSlot.cacheRefCount = 0;
        resourceSlot.onFreeList = false;
    }

    return true;
}

void ResourcePools::addActiveRef(ResourceId const& resource)
{
    Slot& resourceSlot = slot(resource);
    ELLM_CHECK(!resourceSlot.onFreeList, "Cannot add an active reference to a free context cache resource");
    ELLM_CHECK(resourceSlot.activeRefCount < std::numeric_limits<int32_t>::max(),
        "Context cache active reference count overflow");
    ++resourceSlot.activeRefCount;
}

void ResourcePools::releaseActiveRef(ResourceId const& resource)
{
    Slot& resourceSlot = slot(resource);
    ELLM_CHECK(resourceSlot.activeRefCount > 0, "Context cache active reference count underflow");
    if (resourceSlot.activeRefCount == 1 && resourceSlot.cacheRefCount == 0 && !resourceSlot.onFreeList)
    {
        Pool const& resourcePool = pool(resource.type);
        ELLM_CHECK(
            resourcePool.freeListCount < resourcePool.freeList.size(), "Context cache resource free list is full");
    }
    --resourceSlot.activeRefCount;
    releaseIfUnused(resource, resourceSlot);
}

void ResourcePools::addCacheRef(ResourceId const& resource)
{
    Slot& resourceSlot = slot(resource);
    ELLM_CHECK(!resourceSlot.onFreeList, "Cannot add a cache reference to a free context cache resource");
    ELLM_CHECK(resourceSlot.cacheRefCount < std::numeric_limits<int32_t>::max(),
        "Context cache cache reference count overflow");
    ++resourceSlot.cacheRefCount;
}

void ResourcePools::releaseCacheRef(ResourceId const& resource)
{
    Slot& resourceSlot = slot(resource);
    ELLM_CHECK(resourceSlot.cacheRefCount > 0, "Context cache cache reference count underflow");
    if (resourceSlot.cacheRefCount == 1 && resourceSlot.activeRefCount == 0 && !resourceSlot.onFreeList)
    {
        Pool const& resourcePool = pool(resource.type);
        ELLM_CHECK(
            resourcePool.freeListCount < resourcePool.freeList.size(), "Context cache resource free list is full");
    }
    --resourceSlot.cacheRefCount;
    releaseIfUnused(resource, resourceSlot);
}

int32_t ResourcePools::activeRefCount(ResourceId const& resource) const
{
    return slot(resource).activeRefCount;
}

int32_t ResourcePools::cacheRefCount(ResourceId const& resource) const
{
    return slot(resource).cacheRefCount;
}

int32_t ResourcePools::freeCount(ResourceType type) const noexcept
{
    size_t const index = poolIndex(type);
    if (index >= mPools.size())
    {
        return 0;
    }
    return static_cast<int32_t>(mPools[index].freeListCount);
}

int32_t ResourcePools::capacity(ResourceType type) const noexcept
{
    size_t const index = poolIndex(type);
    if (index >= mPools.size())
    {
        return 0;
    }
    return static_cast<int32_t>(mPools[index].slots.size());
}

bool ResourcePools::canAllocate(ResourceDemand const& demand) const noexcept
{
    if (!demand.isNonNegative())
    {
        return false;
    }
    for (ResourceType const type : kRESOURCE_TYPES)
    {
        if (static_cast<size_t>(demand.get(type)) > pool(type).freeListCount)
        {
            return false;
        }
    }
    return true;
}

ResourcePools::Pool& ResourcePools::pool(ResourceType type) noexcept
{
    return mPools[poolIndex(type)];
}

ResourcePools::Pool const& ResourcePools::pool(ResourceType type) const noexcept
{
    return mPools[poolIndex(type)];
}

ResourcePools::Slot& ResourcePools::slot(ResourceId const& resource)
{
    size_t const typeIndex = poolIndex(resource.type);
    ELLM_CHECK(typeIndex < mPools.size() && resource.index >= 0
            && static_cast<size_t>(resource.index) < mPools[typeIndex].slots.size(),
        "Invalid context cache resource ID");
    return mPools[typeIndex].slots[static_cast<size_t>(resource.index)];
}

ResourcePools::Slot const& ResourcePools::slot(ResourceId const& resource) const
{
    size_t const typeIndex = poolIndex(resource.type);
    ELLM_CHECK(typeIndex < mPools.size() && resource.index >= 0
            && static_cast<size_t>(resource.index) < mPools[typeIndex].slots.size(),
        "Invalid context cache resource ID");
    return mPools[typeIndex].slots[static_cast<size_t>(resource.index)];
}

void ResourcePools::releaseIfUnused(ResourceId const& resource, Slot& resourceSlot) noexcept
{
    if (resourceSlot.activeRefCount == 0 && resourceSlot.cacheRefCount == 0 && !resourceSlot.onFreeList)
    {
        Pool& resourcePool = pool(resource.type);
        size_t const freeListTail
            = (resourcePool.freeListHead + resourcePool.freeListCount) % resourcePool.freeList.size();
        resourcePool.freeList[freeListTail] = resource.index;
        ++resourcePool.freeListCount;
        resourceSlot.onFreeList = true;
    }
}

} // namespace rt
} // namespace trt_edgellm
