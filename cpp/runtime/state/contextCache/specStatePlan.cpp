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

#include "runtime/state/contextCache/specStatePlan.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

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

ReusePlan makeSpecReusePlan(SpecReusePlanInput const& input, SpecReuseContract const& contract)
{
    ELLM_CHECK(input.lookupPolicy == ContextCacheLookupPolicy::kUseCache
            || input.lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    int64_t const totalInputPages
        = validateAndCountInputPages(input.inputFullBlockHashes, input.inputTokenCount, input.pageSize);

    ReusePlan plan;
    plan.mode = ReusePlanMode::kSpec;
    auto makeCold = [&]() {
        plan.kind = input.inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.draftKvPages = contract.ownsPagedSpecState ? static_cast<int32_t>(totalInputPages) : 0;
        return plan;
    };
    if (input.inputTokenCount == 0 || input.lookupPolicy == ContextCacheLookupPolicy::kBypass)
    {
        return makeCold();
    }

    BaseLookupResult const baseLookup = input.baseIndex.lookupPrefix(input.inputFullBlockHashes);
    ELLM_CHECK(baseLookup.pageIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache base prefix contains too many pages");
    int32_t const baseBlockCount = static_cast<int32_t>(baseLookup.pageIds.size());
    int32_t pairedBlockCount = baseBlockCount;
    if (contract.ownsPagedSpecState)
    {
        std::optional<SpecPagedStateMatch> const specMatch
            = input.specIndex.paged().lookupLongest(input.inputFullBlockHashes, baseBlockCount);
        if (!specMatch.has_value())
        {
            return makeCold();
        }
        else
        {
            CacheRecord const& record = input.records.get(specMatch->record);
            ELLM_CHECK(record.specState.has_value(),
                "Context cache spec path index does not describe a paged spec-state record");
            SpecPagedStateRecord const& paged = *record.specState;
            pairedBlockCount = specMatch->pathBlockCount;
            ELLM_CHECK(pairedBlockCount > 0 && pairedBlockCount <= baseBlockCount
                    && static_cast<size_t>(pairedBlockCount) <= paged.pagePath.size()
                    && static_cast<size_t>(pairedBlockCount) <= record.logicalBlockHashes.size()
                    && record.logicalBlockHashes[static_cast<size_t>(pairedBlockCount - 1)]
                        == input.inputFullBlockHashes[static_cast<size_t>(pairedBlockCount - 1)],
                "Context cache spec path index does not describe a coherent record prefix");
            plan.specRecord = record.id;
            plan.specPageBindings.assign(
                paged.pagePath.begin(), paged.pagePath.begin() + static_cast<std::ptrdiff_t>(pairedBlockCount));
        }
    }

    // A speculative lease is one coherent base/spec prefix. Base pages beyond the matched spec record cannot be
    // retained without creating a stitched state whose draft projection does not describe the same endpoint.
    size_t const pairedCount = static_cast<size_t>(pairedBlockCount);
    plan.matchedTokenLength = static_cast<int32_t>(static_cast<int64_t>(pairedBlockCount) * input.pageSize);
    plan.matchedBlockHashes.assign(input.inputFullBlockHashes.begin(),
        input.inputFullBlockHashes.begin() + static_cast<std::ptrdiff_t>(pairedCount));
    plan.basePageBindings.assign(
        baseLookup.pageIds.begin(), baseLookup.pageIds.begin() + static_cast<std::ptrdiff_t>(pairedCount));

    bool const fullInputMatch
        = input.inputTokenCount % input.pageSize == 0 && static_cast<int64_t>(pairedBlockCount) == totalInputPages;
    int32_t const specReplayPageCount = replayPages(contract, input.pageSize);
    int32_t const rewindPageCount = std::max(specReplayPageCount, fullInputMatch ? 1 : 0);
    ELLM_CHECK(rewindPageCount <= pairedBlockCount, "Context cache replay exceeds the coherent spec-state path");
    if (specReplayPageCount > 0 && pairedBlockCount > specReplayPageCount)
    {
        plan.specReplayDependency = SpecReplayDependency{
            pairedBlockCount, input.inputFullBlockHashes[static_cast<size_t>(pairedBlockCount - 1)]};
    }
    for (int32_t page = 0; page < rewindPageCount; ++page)
    {
        plan.matchedBlockHashes.pop_back();
        plan.basePageBindings.pop_back();
        if (contract.ownsPagedSpecState)
        {
            plan.specPageBindings.pop_back();
        }
    }
    if (specReplayPageCount > 0 && !plan.basePageBindings.empty())
    {
        plan.specReplayMode = SpecReplayMode::kFullPage;
    }
    if (fullInputMatch)
    {
        plan.kind = ReusePlanKind::kFullInputRewind;
    }
    else if (plan.basePageBindings.empty())
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
    }
    if (plan.basePageBindings.empty())
    {
        plan.specRecord.reset();
    }

    int64_t const reusedBasePageCount = static_cast<int64_t>(plan.basePageBindings.size());
    int64_t const reusedSpecPageCount = static_cast<int64_t>(plan.specPageBindings.size());
    plan.reuseTokenLength = static_cast<int32_t>(reusedBasePageCount * input.pageSize);
    plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages - reusedBasePageCount);
    plan.demand.draftKvPages
        = contract.ownsPagedSpecState ? static_cast<int32_t>(totalInputPages - reusedSpecPageCount) : 0;
    return plan;
}

std::optional<SpecPagedStateRecord> makeSpecPublishedState(
    SpecPublishStateInput const& input, SpecReuseContract const& contract)
{
    ELLM_CHECK(input.residentStateLength >= 0, "Context cache resident state length must be non-negative");
    if (!contract.ownsPagedSpecState)
    {
        return std::nullopt;
    }
    ELLM_CHECK(input.leaseState.pageBindings.size() == input.fullBlockHashes.size(),
        "Speculative published state requires one page binding per full-block hash");
    if (input.fullBlockHashes.empty())
    {
        return std::nullopt;
    }
    return SpecPagedStateRecord{input.leaseState.pageBindings};
}

} // namespace rt
} // namespace trt_edgellm
