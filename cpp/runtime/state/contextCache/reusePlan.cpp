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

ReusePlan makeVanillaReusePlan(CacheDomainId domain, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, int32_t pageSize, BaseBlockIndex const& index, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == LookupPolicy::kUseCache || lookupPolicy == LookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.domain = domain;
    plan.lookupPolicy = lookupPolicy;
    plan.inputTokenCount = inputTokenCount;
    if (inputTokenCount == 0)
    {
        return plan;
    }

    if (lookupPolicy == LookupPolicy::kBypass)
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    }
    BaseLookupResult lookup = index.lookupPrefix(domain, inputFullBlockHashes);
    plan.matchedBlockHashes = std::move(lookup.matchedHashes);
    plan.basePageBindings = std::move(lookup.pageIds);

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

ReusePlan makeHybridReusePlan(CacheDomainId domain, RecurrentStateSchemaId schema,
    std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, int32_t pageSize, bool hasAttention, CacheRecordStore const& records,
    LookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == LookupPolicy::kUseCache || lookupPolicy == LookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kHybrid;
    plan.lookupPolicy = lookupPolicy;
    plan.domain = domain;
    plan.inputTokenCount = inputTokenCount;
    plan.hybridHasAttention = hasAttention;
    plan.recurrentStateSchema = schema;

    auto makeCold = [&]() {
        plan.kind = inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = hasAttention ? static_cast<int32_t>(totalInputPages) : 0;
        return plan;
    };
    if (inputTokenCount == 0 || lookupPolicy == LookupPolicy::kBypass)
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
        HybridCheckpointKey const key{domain, candidate.exactPrefixDigest, candidate.exactLength, schema};
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
        bool const attentionPathComplete = hasAttention
            ? record.baseFullBlockCount == static_cast<int32_t>(fullBlockCount)
                && record.basePagePath.size() == fullBlockCount
            : record.baseFullBlockCount == 0 && record.basePagePath.empty();
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
        plan.demand.baseKvPages
            = hasAttention ? static_cast<int32_t>(totalInputPages - static_cast<int64_t>(fullBlockCount)) : 0;
        return plan;
    }
    return makeCold();
}

ReusePlan makeHybridMtpReusePlan(CacheDomainId domain, RecurrentStateSchemaId schema,
    DraftEngineSignature draftSignature, std::vector<HybridMtpCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    CacheRecordStore const& records, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == LookupPolicy::kUseCache || lookupPolicy == LookupPolicy::kBypass,
        "Hybrid+MTP context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kHybridMtp;
    plan.lookupPolicy = lookupPolicy;
    plan.domain = domain;
    plan.inputTokenCount = inputTokenCount;
    plan.hybridHasAttention = true;
    plan.recurrentStateSchema = schema;
    plan.draftSignature = draftSignature;

    auto makeCold = [&]() {
        plan.kind = inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.draftKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    };
    if (inputTokenCount == 0 || lookupPolicy == LookupPolicy::kBypass)
    {
        return makeCold();
    }

    std::vector<HybridMtpCheckpointCandidate> ordered = candidates;
    std::sort(ordered.begin(), ordered.end(),
        [](auto const& lhs, auto const& rhs) { return lhs.exactLength > rhs.exactLength; });
    for (HybridMtpCheckpointCandidate const& candidate : ordered)
    {
        if (candidate.exactLength <= 0 || candidate.exactLength >= inputTokenCount)
        {
            continue;
        }
        HybridMtpCheckpointKey const key{
            domain, candidate.exactPrefixDigest, candidate.exactLength, schema, draftSignature};
        std::optional<RecordId> const recordId = records.findHybridMtp(key);
        if (!recordId.has_value())
        {
            continue;
        }

        CacheRecord const& record = records.get(*recordId);
        // The boundary token (exactLength - 1) is always retained in a private partial page (see publish side); reserve
        // one fewer full block than exactLength/pageSize so page-aligned checkpoints keep their boundary private.
        size_t const fullBlockCount = static_cast<size_t>((candidate.exactLength - 1) / pageSize);
        bool const logicalPrefixMatches = record.logicalBlockHashes.size() == fullBlockCount
            && fullBlockCount <= inputFullBlockHashes.size()
            && std::equal(
                record.logicalBlockHashes.begin(), record.logicalBlockHashes.end(), inputFullBlockHashes.begin());
        bool const partial = true;
        bool const snapshotSetComplete
            = record.recurrentSnapshotSlot.has_value() && record.partialKvSnapshotSlot.has_value() == partial;
        bool const pagePathsComplete = record.baseFullBlockCount == static_cast<int32_t>(fullBlockCount)
            && record.pairedDraftFullBlockCount == static_cast<int32_t>(fullBlockCount)
            && record.basePagePath.size() == fullBlockCount && record.draftPagePath.size() == fullBlockCount;
        if (!logicalPrefixMatches || !snapshotSetComplete || !pagePathsComplete)
        {
            continue;
        }

        plan.hybridMtpCheckpoint = key;
        plan.hybridRecord = *recordId;
        plan.recurrentSnapshotBinding = record.recurrentSnapshotSlot;
        plan.partialKvSnapshotBinding = record.partialKvSnapshotSlot;
        plan.matchedBlockHashes = record.logicalBlockHashes;
        plan.basePageBindings = record.basePagePath;
        plan.draftPageBindings = record.draftPagePath;
        plan.reuseTokenLength = candidate.exactLength;
        int32_t const privatePageCount = static_cast<int32_t>(totalInputPages - static_cast<int64_t>(fullBlockCount));
        plan.demand.baseKvPages = privatePageCount;
        plan.demand.draftKvPages = privatePageCount;
        return plan;
    }
    return makeCold();
}

ReusePlan makeSpecReusePlan(SpecDecodeMode mode, CacheDomainId domain, DraftEngineSignature draftSignature,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, int32_t pageSize,
    bool supportsOneTokenReplay, BaseBlockIndex const& baseIndex, DraftPathIndex const& draftIndex,
    CacheRecordStore const& records, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(mode == SpecDecodeMode::kEAGLE, "Context cache speculative reuse currently supports only EAGLE");
    ELLM_CHECK(lookupPolicy == LookupPolicy::kUseCache || lookupPolicy == LookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages = validateAndCountInputPages(inputFullBlockHashes, inputTokenCount, pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kSpecEagle;
    plan.lookupPolicy = lookupPolicy;
    plan.domain = domain;
    plan.draftSignature = draftSignature;
    plan.inputTokenCount = inputTokenCount;
    if (inputTokenCount == 0)
    {
        return plan;
    }

    if (lookupPolicy == LookupPolicy::kBypass)
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.draftKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    }
    BaseLookupResult const baseLookup = baseIndex.lookupPrefix(domain, inputFullBlockHashes);
    ELLM_CHECK(baseLookup.pageIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache base prefix contains too many pages");
    int32_t const baseBlockCount = static_cast<int32_t>(baseLookup.pageIds.size());
    std::optional<DraftPathMatch> const draftMatch
        = draftIndex.lookupLongest(draftSignature, domain, inputFullBlockHashes, baseBlockCount);
    if (!draftMatch.has_value())
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.draftKvPages = static_cast<int32_t>(totalInputPages);
        return plan;
    }

    CacheRecord const& record = records.get(draftMatch->record);
    int32_t const pairedBlockCount = draftMatch->pathBlockCount;
    ELLM_CHECK(record.draftSignature == plan.draftSignature && pairedBlockCount > 0
            && pairedBlockCount <= baseBlockCount
            && static_cast<size_t>(pairedBlockCount) <= record.draftPagePath.size()
            && static_cast<size_t>(pairedBlockCount) <= record.logicalBlockHashes.size()
            && record.logicalBlockHashes[static_cast<size_t>(pairedBlockCount - 1)]
                == inputFullBlockHashes[static_cast<size_t>(pairedBlockCount - 1)],
        "Context cache draft path index does not describe a coherent record prefix");

    size_t const pairedCount = static_cast<size_t>(pairedBlockCount);
    plan.draftRecord = record.id;
    plan.matchedBlockHashes.assign(
        inputFullBlockHashes.begin(), inputFullBlockHashes.begin() + static_cast<std::ptrdiff_t>(pairedCount));
    plan.basePageBindings.assign(
        baseLookup.pageIds.begin(), baseLookup.pageIds.begin() + static_cast<std::ptrdiff_t>(pairedCount));
    plan.draftPageBindings.assign(
        record.draftPagePath.begin(), record.draftPagePath.begin() + static_cast<std::ptrdiff_t>(pairedCount));

    bool const fullInputMatch
        = inputTokenCount % pageSize == 0 && static_cast<int64_t>(pairedBlockCount) == totalInputPages;
    if (fullInputMatch || !supportsOneTokenReplay)
    {
        if (pairedBlockCount > 1)
        {
            plan.specReplayDependency = SpecReplayDependency{plan.matchedBlockHashes.back(), pairedBlockCount};
        }
        plan.matchedBlockHashes.pop_back();
        plan.basePageBindings.pop_back();
        plan.draftPageBindings.pop_back();
        plan.specReplayMode = SpecReplayMode::kFullPage;
        if (fullInputMatch)
        {
            plan.kind = ReusePlanKind::kFullInputRewind;
        }
        else if (plan.basePageBindings.empty())
        {
            plan.kind = ReusePlanKind::kNoReusablePrefix;
        }
    }
    else
    {
        plan.baseCowSources.push_back(plan.basePageBindings.back());
        plan.draftCowSources.push_back(plan.draftPageBindings.back());
        plan.specReplayMode = SpecReplayMode::kOneToken;
    }

    if (plan.basePageBindings.empty())
    {
        plan.draftRecord.reset();
    }
    int64_t const sharedPageCount = static_cast<int64_t>(plan.basePageBindings.size() - plan.baseCowSources.size());
    plan.reuseTokenLength = plan.specReplayMode == SpecReplayMode::kOneToken
        ? static_cast<int32_t>(static_cast<int64_t>(pairedBlockCount) * pageSize - 1)
        : static_cast<int32_t>(sharedPageCount * pageSize);
    int64_t const privatePageCount = totalInputPages - sharedPageCount;
    plan.demand.baseKvPages = static_cast<int32_t>(privatePageCount);
    plan.demand.draftKvPages = static_cast<int32_t>(privatePageCount);
    return plan;
}

} // namespace rt
} // namespace trt_edgellm
