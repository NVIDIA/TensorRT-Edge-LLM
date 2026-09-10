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
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheTypes.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class BaseBlockIndex;
//! Classification of a valid reuse plan; no value represents acquisition failure.
enum class ReusePlanKind : uint8_t
{
    kStandard,
    kNoReusablePrefix,
    kFullInputRewind,
};

//! Decoder path whose state and replay invariants produced a reuse plan.
enum class ReusePlanMode : uint8_t
{
    kVanilla,
    //! Exact atomic recurrent/conv and optional partial-KV checkpoint reuse.
    kHybrid,
    //! Hybrid base paired with a coherent MTP spec path at an exact recurrent/partial-KV checkpoint.
    kHybridMtp,
    //! Initial speculative implementation: greedy, non-hybrid EAGLE.
    kSpec,
};

//! Exact digest computed by the runtime for one stored hybrid candidate length.
struct HybridCheckpointCandidate
{
    int32_t exactLength{};
    BlockHash exactPrefixDigest{};
};

//! Recompute required at the boundary of an EAGLE draft-path hit.
enum class SpecReplayMode : uint8_t
{
    kNone,
    kFullPage,
};

//! Original coherent state boundary required by full-page replay.
struct SpecReplayDependency
{
    int32_t pathBlockCount{};
    BlockHash terminalHash{};
};

//! Side-effect-free proposal for binding cached pages and allocating request-private state.
//!
//! The free planning helpers expose deterministic metadata decisions for focused tests. ContextCacheManager builds
//! and consumes each plan inside one serialized acquire call, so runtime callers never retain a mutable plan across
//! metadata operations. The demand contains only resources that still need private allocation.
struct ReusePlan
{
    ReusePlanMode mode{ReusePlanMode::kVanilla};
    //! Longest coherent cache match before any decoder-required rewind/replay adjustment.
    int32_t matchedTokenLength{};
    //! Materialized state boundary from which this request will actually execute.
    int32_t reuseTokenLength{};
    std::vector<BlockHash> matchedBlockHashes;
    std::vector<PageId> basePageBindings;
    std::optional<RecordId> specRecord;
    std::vector<PageId> specPageBindings;
    std::optional<SpecReplayDependency> specReplayDependency;
    bool hybridHasAttention{false};
    std::optional<HybridCheckpointKey> hybridCheckpoint;
    std::optional<RecordId> hybridRecord;
    std::optional<int32_t> recurrentSnapshotBinding;
    std::optional<int32_t> partialKvSnapshotBinding;
    ResourceDemand demand;
    ReusePlanKind kind{ReusePlanKind::kStandard};
    SpecReplayMode specReplayMode{SpecReplayMode::kNone};
};

//! Build a base-model KV reuse plan for vanilla autoregressive decoding without mutating cache metadata.
//!
//! An exact block-aligned full-input match rewinds one full page so the caller can recompute the final token boundary.
ReusePlan makeVanillaReusePlan(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
    int32_t pageSize, BaseBlockIndex const& index,
    ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);

//! Build an exact hybrid/pure-recurrent reuse plan. Candidates may be unordered; the longest coherent checkpoint
//! strictly shorter than the input wins. A missing snapshot member makes that candidate a complete miss.
ReusePlan makeHybridReusePlan(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize, bool hasAttention,
    CacheRecordStore const& records, ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);

//! Build an exact hybrid+MTP reuse plan while retaining the successor-dependent boundary token privately.
ReusePlan makeHybridMtpReusePlan(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    CacheRecordStore const& records, ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);

} // namespace rt
} // namespace trt_edgellm
