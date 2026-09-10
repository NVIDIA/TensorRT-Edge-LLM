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

#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheTypes.h"
#include "runtime/state/contextCache/reusePlan.h"
#include "runtime/state/contextCache/specReuseContract.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Host-only inputs required to plan speculative context reuse.
struct SpecReusePlanInput
{
    std::vector<BlockHash> const& inputFullBlockHashes;
    int32_t inputTokenCount{};
    int32_t pageSize{};
    ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
    BaseBlockIndex const& baseIndex;
    SpecStateIndex const& specIndex;
    CacheRecordStore const& records;
};

//! Speculative state carried by one active lease at publication time.
struct SpecLeaseStateView
{
    std::vector<PageId> const& pageBindings;
};

//! Host-only inputs required to describe publishable speculative state.
struct SpecPublishStateInput
{
    SpecLeaseStateView const& leaseState;
    std::vector<BlockHash> const& fullBlockHashes;
    int32_t residentStateLength{};
};

ReusePlan makeSpecReusePlan(SpecReusePlanInput const& input, SpecReuseContract const& contract);
std::optional<SpecPagedStateRecord> makeSpecPublishedState(
    SpecPublishStateInput const& input, SpecReuseContract const& contract);

} // namespace rt
} // namespace trt_edgellm
