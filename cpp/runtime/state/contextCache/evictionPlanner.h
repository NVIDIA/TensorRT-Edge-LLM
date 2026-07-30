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

#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class CacheRecordStore;
class ResourcePools;

//! A non-mutating eviction simulation result.
struct EvictionPlan
{
    bool feasible{};
    std::vector<RecordId> victims;
    ResourceDemand reclaimed;
};

//! Simulates record-level LRU eviction across all typed resource pools.
//!
//! Planning is read-only. It walks records from LRU to MRU, simulates each cache-reference release, and counts a
//! resource as reclaimed only when its simulated cache count and real active count both reach zero. This preserves
//! shared ancestors and returns the shortest victim prefix that satisfies the complete multi-pool demand.
class EvictionPlanner
{
public:
    //! Simulate cache-reference releases without modifying pools or records.
    //! mruRecord, when present, is visited after every other record so a newly confirmed hit affects this simulation
    //! without mutating real LRU state before acquisition succeeds.
    static EvictionPlan plan(ResourceDemand const& demand, ResourcePools const& pools, CacheRecordStore const& records,
        std::optional<RecordId> mruRecord = std::nullopt);
};

} // namespace rt
} // namespace trt_edgellm
