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

#include "runtime/state/contextCache/evictionPlanner.h"

#include "common/checkMacros.h"
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/resourcePools.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{
namespace
{

constexpr std::array<ResourceType, 4> kRESOURCE_TYPES{ResourceType::kBaseKvPage, ResourceType::kDraftKvPage,
    ResourceType::kRecurrentSnapshot, ResourceType::kPartialKvSnapshot};

bool satisfiesDemand(ResourceDemand const& demand, ResourceDemand const& reclaimed, ResourcePools const& pools) noexcept
{
    for (ResourceType const type : kRESOURCE_TYPES)
    {
        int64_t const available
            = static_cast<int64_t>(pools.freeCount(type)) + static_cast<int64_t>(reclaimed.get(type));
        if (available < static_cast<int64_t>(demand.get(type)))
        {
            return false;
        }
    }
    return true;
}

} // namespace

EvictionPlan EvictionPlanner::plan(ResourceDemand const& demand, ResourcePools const& pools,
    CacheRecordStore const& records, std::optional<RecordId> mruRecord)
{
    ELLM_CHECK(demand.isNonNegative(), "Context cache eviction demand must be non-negative");
    if (pools.canAllocate(demand))
    {
        return EvictionPlan{true, {}, {}};
    }

    EvictionPlan plan;
    std::unordered_map<ResourceId, int32_t> simulatedCacheRefs;
    std::vector<RecordId> evictionOrder = records.lruToMru();
    if (mruRecord.has_value())
    {
        auto const hit = std::find(evictionOrder.begin(), evictionOrder.end(), *mruRecord);
        ELLM_CHECK(hit != evictionOrder.end(), "Context cache MRU override record does not exist");
        evictionOrder.erase(hit);
        evictionOrder.push_back(*mruRecord);
    }
    for (RecordId const recordId : evictionOrder)
    {
        plan.victims.push_back(recordId);
        for (ResourceId const& resource : records.get(recordId).resources())
        {
            auto simulatedRef = simulatedCacheRefs.find(resource);
            if (simulatedRef == simulatedCacheRefs.end())
            {
                simulatedRef = simulatedCacheRefs.emplace(resource, pools.cacheRefCount(resource)).first;
            }
            ELLM_CHECK(
                simulatedRef->second > 0, "Context cache eviction record releases a resource with no cache reference");

            --simulatedRef->second;
            if (simulatedRef->second == 0 && pools.activeRefCount(resource) == 0)
            {
                ++plan.reclaimed.get(resource.type);
            }
        }

        if (satisfiesDemand(demand, plan.reclaimed, pools))
        {
            plan.feasible = true;
            return plan;
        }
    }
    return EvictionPlan{};
}

} // namespace rt
} // namespace trt_edgellm
