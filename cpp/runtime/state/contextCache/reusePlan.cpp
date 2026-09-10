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

#include "runtime/state/contextCache/reusePlan.h"

#include "common/checkMacros.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

int64_t validateAndCountInputPages(
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize)
{
    ELLM_CHECK(pageSize > 0, "Context cache page size must be positive");
    ELLM_CHECK(inputTokenCount >= 0, "Context cache input token count must be non-negative");
    int64_t const fullInputBlockCount = static_cast<int64_t>(inputTokenCount) / static_cast<int64_t>(pageSize);
    ELLM_CHECK(inputFullBlockHashes.size() <= static_cast<size_t>(fullInputBlockCount),
        "Context cache input has more full-block hashes than full token blocks");
    return (static_cast<int64_t>(inputTokenCount) + static_cast<int64_t>(pageSize) - 1)
        / static_cast<int64_t>(pageSize);
}

} // namespace

ReusePlan makeVanillaReusePlan(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
    int32_t pageSize, BaseBlockIndex const& index, ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == ContextCacheLookupPolicy::kUseCache || lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    if (inputTokenCount == 0)
    {
        return plan;
    }

    if (lookupPolicy == ContextCacheLookupPolicy::kBypass)
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    }
    BaseLookupResult lookup = index.lookupPrefix(inputFullBlockHashes);
    plan.matchedBlockHashes = std::move(lookup.matchedHashes);
    plan.basePageBindings = std::move(lookup.pageIds);
    plan.matchedTokenLength
        = static_cast<int32_t>(static_cast<int64_t>(plan.basePageBindings.size()) * static_cast<int64_t>(pageSize));

    bool const fullInputMatch = static_cast<int64_t>(plan.basePageBindings.size()) == totalInputPages;
    if (inputTokenCount % pageSize == 0 && fullInputMatch)
    {
        plan.matchedBlockHashes.pop_back();
        plan.basePageBindings.pop_back();
        plan.kind = ReusePlanKind::kFullInputRewind;
    }
    else if (plan.basePageBindings.empty())
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
    }

    int64_t const reusablePageCount = static_cast<int64_t>(plan.basePageBindings.size());
    plan.reuseTokenLength = static_cast<int32_t>(reusablePageCount * static_cast<int64_t>(pageSize));
    plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages - reusablePageCount);
    return plan;
}

ReusePlan makeHybridReusePlan(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize, bool hasAttention,
    CacheRecordStore const& records, ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == ContextCacheLookupPolicy::kUseCache || lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kHybrid;
    plan.hybridHasAttention = hasAttention;

    auto makeCold = [&]() {
        plan.kind = inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = hasAttention ? static_cast<int32_t>(totalInputPages) : 0;
        return plan;
    };
    if (inputTokenCount == 0 || lookupPolicy == ContextCacheLookupPolicy::kBypass)
    {
        return makeCold();
    }

    std::vector<HybridCheckpointCandidate> ordered = candidates;
    std::sort(ordered.begin(), ordered.end(),
        [](auto const& lhs, auto const& rhs) { return lhs.exactLength > rhs.exactLength; });
    for (HybridCheckpointCandidate const& candidate : ordered)
    {
        if (candidate.exactLength <= 0 || candidate.exactLength >= inputTokenCount)
        {
            continue;
        }
        HybridCheckpointKey const key{candidate.exactPrefixDigest, candidate.exactLength};
        std::optional<RecordId> const recordId = records.findHybrid(key);
        if (!recordId.has_value())
        {
            continue;
        }

        CacheRecord const& record = records.get(*recordId);
        size_t const fullBlockCount = static_cast<size_t>(candidate.exactLength / pageSize);
        bool const logicalPrefixMatches = record.logicalBlockHashes.size() == fullBlockCount
            && fullBlockCount <= inputFullBlockHashes.size()
            && std::equal(
                record.logicalBlockHashes.begin(), record.logicalBlockHashes.end(), inputFullBlockHashes.begin());
        bool const partial = candidate.exactLength % pageSize != 0;
        bool const snapshotSetComplete = record.recurrentSnapshotSlot.has_value()
            && record.partialKvSnapshotSlot.has_value() == (hasAttention && partial);
        bool const attentionPathComplete
            = hasAttention ? record.basePagePath.size() == fullBlockCount : record.basePagePath.empty();
        if (!logicalPrefixMatches || !snapshotSetComplete || !attentionPathComplete
            || (!hasAttention && record.partialKvSnapshotSlot.has_value()))
        {
            continue;
        }

        plan.hybridCheckpoint = key;
        plan.hybridRecord = *recordId;
        plan.recurrentSnapshotBinding = record.recurrentSnapshotSlot;
        plan.partialKvSnapshotBinding = record.partialKvSnapshotSlot;
        if (hasAttention)
        {
            plan.matchedBlockHashes = record.logicalBlockHashes;
            plan.basePageBindings = record.basePagePath;
        }
        plan.reuseTokenLength = candidate.exactLength;
        plan.matchedTokenLength = candidate.exactLength;
        plan.demand.baseKvPages
            = hasAttention ? static_cast<int32_t>(totalInputPages - static_cast<int64_t>(fullBlockCount)) : 0;
        return plan;
    }
    return makeCold();
}

ReusePlan makeHybridMtpReusePlan(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    CacheRecordStore const& records, ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == ContextCacheLookupPolicy::kUseCache || lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kHybridMtp;
    plan.hybridHasAttention = true;

    auto makeCold = [&]() {
        plan.kind = inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        // MTP is a speculative deployment: a cold request still runs the draft engine over the full input, so the draft
        // pool needs the same full-input reservation as the base pool (mirrors makeSpecReusePlan's cold demand).
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.draftKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    };
    if (inputTokenCount == 0 || lookupPolicy == ContextCacheLookupPolicy::kBypass)
    {
        return makeCold();
    }

    std::vector<HybridCheckpointCandidate> ordered = candidates;
    std::sort(ordered.begin(), ordered.end(),
        [](auto const& lhs, auto const& rhs) { return lhs.exactLength > rhs.exactLength; });
    for (HybridCheckpointCandidate const& candidate : ordered)
    {
        if (candidate.exactLength <= 0 || candidate.exactLength >= inputTokenCount)
        {
            continue;
        }
        HybridCheckpointKey const key{candidate.exactPrefixDigest, candidate.exactLength};
        std::optional<RecordId> const recordId = records.findHybrid(key);
        if (!recordId.has_value())
        {
            continue;
        }

        CacheRecord const& record = records.get(*recordId);
        // This is the local form of SpecReuseContract::futureDependencyTokens == 1; keep the boundary token private.
        // Hybrid MTP checkpoint arithmetic intentionally remains local to this shipping path.
        size_t const fullBlockCount = static_cast<size_t>((candidate.exactLength - 1) / pageSize);
        bool const logicalPrefixMatches = record.logicalBlockHashes.size() == fullBlockCount
            && fullBlockCount <= inputFullBlockHashes.size()
            && std::equal(
                record.logicalBlockHashes.begin(), record.logicalBlockHashes.end(), inputFullBlockHashes.begin());
        // MTP always publishes a partial page, so both snapshots must be present.
        bool const snapshotSetComplete
            = record.recurrentSnapshotSlot.has_value() && record.partialKvSnapshotSlot.has_value();
        bool const basePathComplete = record.basePagePath.size() == fullBlockCount;
        bool const draftPathComplete
            = record.specState.has_value() && record.specState->pagePath.size() == fullBlockCount;
        if (!logicalPrefixMatches || !snapshotSetComplete || !basePathComplete || !draftPathComplete)
        {
            continue;
        }

        plan.hybridCheckpoint = key;
        plan.hybridRecord = *recordId;
        plan.recurrentSnapshotBinding = record.recurrentSnapshotSlot;
        plan.partialKvSnapshotBinding = record.partialKvSnapshotSlot;
        plan.matchedBlockHashes = record.logicalBlockHashes;
        plan.basePageBindings = record.basePagePath;
        plan.specPageBindings = record.specState->pagePath;
        plan.reuseTokenLength = candidate.exactLength;
        plan.matchedTokenLength = candidate.exactLength;
        int32_t const privatePageCount = static_cast<int32_t>(totalInputPages - static_cast<int64_t>(fullBlockCount));
        plan.demand.baseKvPages = privatePageCount;
        plan.demand.draftKvPages = privatePageCount;
        return plan;
    }
    return makeCold();
}
} // namespace rt
} // namespace trt_edgellm
