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
#include "runtime/state/contextCache/contextCacheTypes.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class BaseBlockIndex;
class DraftPathIndex;
enum class SpecDecodeMode : int32_t;

//! Classification of a valid reuse plan; no value represents acquisition failure.
enum class ReusePlanKind : uint8_t
{
    kStandard,
    kNoReusablePrefix,
    kFullInputRewind,
};

//! Whether planning may consult reusable-state indices.
enum class LookupPolicy : uint8_t
{
    kUseCache,
    //! Build an explicit cold demand without reading lookup indices. Acquisition may still evict retained records to
    //! make that active demand feasible.
    kBypass,
};

//! Decoder path whose state and replay invariants produced a reuse plan.
enum class ReusePlanMode : uint8_t
{
    kVanilla,
    //! Exact atomic recurrent/conv and optional partial-KV checkpoint reuse.
    kHybrid,
    //! Exact atomic Hybrid+MTP endpoint state with a successor-token dependency.
    kHybridMtp,
    //! Initial speculative implementation: greedy, non-hybrid EAGLE.
    kSpecEagle,
};

//! Exact digest computed by the runtime for one stored hybrid candidate length.
struct HybridCheckpointCandidate
{
    int32_t exactLength{};
    BlockHash exactPrefixDigest{};
};

//! Exact Hybrid+MTP candidate identity computed from one consumer input.
struct HybridMtpCheckpointCandidate
{
    int32_t exactLength{};
    BlockHash exactPrefixDigest{};
};

//! Recompute required at the boundary of an EAGLE draft-path hit.
enum class SpecReplayMode : uint8_t
{
    kNone,
    kOneToken,
    kFullPage,
};

//! Original coherent draft boundary required by EAGLE full-page replay.
//!
//! Full-page replay binds one fewer page than the original draft match, but the last retained draft slot still
//! depends on the first token in this boundary. The dependency is revalidated without binding or pinning its page.
struct SpecReplayDependency
{
    BlockHash terminalHash{};
    int32_t pathBlockCount{};
};

//! Side-effect-free proposal for binding cached pages and allocating request-private state.
//!
//! Planning reads the base and optional draft indices but does not pin pages, touch LRU state, or evict records.
//! Planning and acquisition are separate phases, so ContextCacheManager revalidates every matched binding during
//! acquire() before it performs any mutation. This is a transactional stale-plan check, not thread synchronization;
//! callers must still serialize access to the manager. The demand contains only resources that still need private
//! allocation.
struct ReusePlan
{
    ReusePlanMode mode{ReusePlanMode::kVanilla};
    LookupPolicy lookupPolicy{LookupPolicy::kUseCache};
    CacheDomainId domain{};
    int32_t inputTokenCount{};
    int32_t reuseTokenLength{};
    std::vector<BlockHash> matchedBlockHashes;
    std::vector<PageId> basePageBindings;
    std::vector<PageId> baseCowSources;
    std::optional<DraftEngineSignature> draftSignature;
    std::optional<RecordId> draftRecord;
    std::vector<PageId> draftPageBindings;
    std::vector<PageId> draftCowSources;
    std::optional<SpecReplayDependency> specReplayDependency;
    bool hybridHasAttention{false};
    std::optional<RecurrentStateSchemaId> recurrentStateSchema;
    std::optional<HybridCheckpointKey> hybridCheckpoint;
    std::optional<HybridMtpCheckpointKey> hybridMtpCheckpoint;
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
ReusePlan makeVanillaReusePlan(CacheDomainId domain, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, int32_t pageSize, BaseBlockIndex const& index,
    LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

//! Build an exact hybrid/pure-recurrent reuse plan. Candidates may be unordered; the longest coherent checkpoint
//! strictly shorter than the input wins. A missing snapshot member makes that candidate a complete miss.
ReusePlan makeHybridReusePlan(CacheDomainId domain, RecurrentStateSchemaId schema,
    std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, int32_t pageSize, bool hasAttention, CacheRecordStore const& records,
    LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

//! Build an exact Hybrid+MTP endpoint reuse plan.
//!
//! A hit is accepted only from one coherent record whose exact endpoint is strictly shorter than the consumer input.
//! The successor token is not part of the lookup key; the successor-dependent boundary draft slot is recomputed at
//! restore, so any consumer whose prefix matches the checkpoint reuses it regardless of the following token.
ReusePlan makeHybridMtpReusePlan(CacheDomainId domain, RecurrentStateSchemaId schema,
    DraftEngineSignature draftSignature, std::vector<HybridMtpCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    CacheRecordStore const& records, LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

//! Build a speculative reuse plan without mutating cache metadata.
//!
//! The initial implementation accepts only SpecDecodeMode::kEAGLE. The caller must reject non-greedy or hybrid EAGLE
//! before planning. A hit requires one coherent draft record path; base-only state is intentionally ignored because it
//! cannot reconstruct historical EAGLE draft KV.
ReusePlan makeSpecReusePlan(SpecDecodeMode mode, CacheDomainId domain, DraftEngineSignature draftSignature,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    bool supportsOneTokenReplay, BaseBlockIndex const& baseIndex, DraftPathIndex const& draftIndex,
    CacheRecordStore const& records, LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

} // namespace rt
} // namespace trt_edgellm
