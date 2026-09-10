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

#include "runtime/state/contextCache/cacheRecord.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

// Runtime page tables use -1 for an unused slot, while cache records may own only allocated pool pages.
bool hasNegativePage(std::vector<PageId> const& pages) noexcept
{
    for (PageId const page : pages)
    {
        if (page < 0)
        {
            return true;
        }
    }
    return false;
}

void validateSpecState(SpecPagedStateRecord const& state, size_t basePathSize)
{
    ELLM_CHECK(!state.pagePath.empty(), "Paged context-cache spec state requires at least one page");
    ELLM_CHECK(state.pagePath.size() == basePathSize, "Paged context-cache spec state must cover its full base path");
    ELLM_CHECK(!hasNegativePage(state.pagePath), "Context cache spec state page IDs must be non-negative");
}

void validateRecord(CacheRecord const& record)
{
    bool const hasHybridIdentity = record.exactCheckpointLength.has_value();
    ELLM_CHECK(record.key.fullBlockCount >= 0, "Context cache record full block count must be non-negative");
    ELLM_CHECK(record.key.fullBlockCount > 0 || hasHybridIdentity,
        "Context cache record must contain a full block or an exact hybrid checkpoint");
    ELLM_CHECK(record.logicalBlockHashes.size() == static_cast<size_t>(record.key.fullBlockCount),
        "Context cache record logical block count does not match its key");
    if (hasHybridIdentity)
    {
        ELLM_CHECK(*record.exactCheckpointLength > 0 && record.recurrentSnapshotSlot.has_value(),
            "Context cache hybrid checkpoint requires a positive exact length and recurrent snapshot");
    }
    else
    {
        ELLM_CHECK(record.key.terminalHash == record.logicalBlockHashes.back(),
            "Context cache record terminal hash does not match its logical path");
    }
    ELLM_CHECK((hasHybridIdentity && record.basePagePath.empty())
            || record.basePagePath.size() == record.logicalBlockHashes.size(),
        "Context cache record base page path must cover its logical path");
    ELLM_CHECK(!hasNegativePage(record.basePagePath), "Context cache record page IDs must be non-negative");
    if (record.specState.has_value())
    {
        validateSpecState(*record.specState, record.basePagePath.size());
    }
    ELLM_CHECK((!record.recurrentSnapshotSlot.has_value() || *record.recurrentSnapshotSlot >= 0)
            && (!record.partialKvSnapshotSlot.has_value() || *record.partialKvSnapshotSlot >= 0),
        "Context cache record snapshot slot IDs must be non-negative");
    ELLM_CHECK(!record.exactCheckpointLength.has_value() || *record.exactCheckpointLength >= 0,
        "Context cache record exact checkpoint length must be non-negative");
}

} // namespace

std::vector<ResourceId> CacheRecord::resources() const
{
    std::vector<ResourceId> recordResources;
    size_t const specPageCount = specState.has_value() ? specState->pagePath.size() : 0U;
    recordResources.reserve(basePagePath.size() + specPageCount + static_cast<size_t>(recurrentSnapshotSlot.has_value())
        + static_cast<size_t>(partialKvSnapshotSlot.has_value()));
    for (PageId const page : basePagePath)
    {
        recordResources.push_back(ResourceId{ResourceType::kBaseKvPage, page});
    }
    if (specState.has_value())
    {
        for (PageId const page : specState->pagePath)
        {
            recordResources.push_back(ResourceId{ResourceType::kDraftKvPage, page});
        }
    }
    if (recurrentSnapshotSlot.has_value())
    {
        recordResources.push_back(ResourceId{ResourceType::kRecurrentSnapshot, *recurrentSnapshotSlot});
    }
    if (partialKvSnapshotSlot.has_value())
    {
        recordResources.push_back(ResourceId{ResourceType::kPartialKvSnapshot, *partialKvSnapshotSlot});
    }
    return recordResources;
}

std::optional<HybridCheckpointKey> CacheRecord::hybridKey() const
{
    if (!exactCheckpointLength.has_value())
    {
        return std::nullopt;
    }
    return HybridCheckpointKey{key.terminalHash, *exactCheckpointLength};
}

CacheRecordStore::CacheRecordStore(int32_t maxRecords)
    : mMaxRecords(maxRecords)
{
    ELLM_CHECK(maxRecords >= 0, "Context cache maximum record count must be non-negative");
}

RecordInsertResult CacheRecordStore::insert(CacheRecord record)
{
    validateRecord(record);

    std::optional<HybridCheckpointKey> const hybridKey = record.hybridKey();
    if (hybridKey.has_value())
    {
        auto const hybridDuplicate = mHybridIndex.find(*hybridKey);
        if (hybridDuplicate != mHybridIndex.end())
        {
            touch(hybridDuplicate->second);
            return RecordInsertResult{hybridDuplicate->second, false};
        }
    }

    auto const duplicate = mExactIndex.find(record.key);
    if (duplicate != mExactIndex.end())
    {
        touch(duplicate->second);
        return RecordInsertResult{duplicate->second, false};
    }

    ELLM_CHECK(mNextId != 0, "Context cache record ID space is exhausted");
    RecordId const id = mNextId;
    RecordId const nextId = id == std::numeric_limits<RecordId>::max() ? RecordId{} : static_cast<RecordId>(id + 1);
    CacheRecordKey const key = record.key;
    record.id = id;

    mLru.push_back(id);
    auto const lruPosition = std::prev(mLru.end());
    auto insertedRecord = mRecords.end();
    bool exactInserted = false;
    bool hybridInserted = false;
    // Record storage, exact-key lookup, and the LRU node form one metadata transaction. The catch removes every earlier
    // insertion if a later container allocation fails.
    try
    {
        auto const recordInsertion = mRecords.emplace(id, Entry{std::move(record), lruPosition});
        ELLM_CHECK(recordInsertion.second, "Context cache record ID is not unique");
        insertedRecord = recordInsertion.first;
        auto const exactInsertion = mExactIndex.emplace(key, id);
        ELLM_CHECK(exactInsertion.second, "Context cache exact record key is not unique");
        exactInserted = true;
        if (hybridKey.has_value())
        {
            auto const hybridInsertion = mHybridIndex.emplace(*hybridKey, id);
            ELLM_CHECK(hybridInsertion.second, "Context cache hybrid checkpoint key is not unique");
            hybridInserted = true;
        }
    }
    catch (...)
    {
        if (hybridInserted)
        {
            mHybridIndex.erase(*hybridKey);
        }
        if (exactInserted)
        {
            mExactIndex.erase(key);
        }
        if (insertedRecord != mRecords.end())
        {
            mRecords.erase(insertedRecord);
        }
        mLru.pop_back();
        throw;
    }

    mNextId = nextId;
    return RecordInsertResult{id, true};
}

std::optional<RecordId> CacheRecordStore::find(CacheRecordKey const& key) const
{
    auto const record = mExactIndex.find(key);
    if (record == mExactIndex.end())
    {
        return std::nullopt;
    }
    return record->second;
}

std::optional<RecordId> CacheRecordStore::findHybrid(HybridCheckpointKey const& key) const
{
    auto const record = mHybridIndex.find(key);
    if (record == mHybridIndex.end())
    {
        return std::nullopt;
    }
    return record->second;
}

std::vector<int32_t> CacheRecordStore::hybridCandidateLengths(int32_t inputTokenCount) const
{
    ELLM_CHECK(inputTokenCount >= 0, "Context cache hybrid input token count must be non-negative");
    std::vector<int32_t> lengths;
    for (auto const& [key, record] : mHybridIndex)
    {
        (void) record;
        if (key.exactLength < inputTokenCount)
        {
            lengths.push_back(key.exactLength);
        }
    }
    std::sort(lengths.begin(), lengths.end(), std::greater<int32_t>{});
    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    return lengths;
}

CacheRecord const& CacheRecordStore::get(RecordId id) const
{
    auto const record = mRecords.find(id);
    ELLM_CHECK(record != mRecords.end(), "Context cache record ID does not exist");
    return record->second.record;
}

bool CacheRecordStore::contains(RecordId id) const noexcept
{
    return mRecords.find(id) != mRecords.end();
}

void CacheRecordStore::setSpecState(RecordId id, SpecPagedStateRecord state)
{
    auto const record = mRecords.find(id);
    ELLM_CHECK(record != mRecords.end(), "Context cache record ID does not exist");
    if (record->second.record.specState.has_value())
    {
        ELLM_CHECK(*record->second.record.specState == state, "Context cache record already has different spec state");
        touch(id);
        return;
    }

    CacheRecord upgraded = record->second.record;
    upgraded.specState = state;
    validateRecord(upgraded);

    record->second.record.specState = std::move(state);
    touch(id);
}

void CacheRecordStore::touch(RecordId id)
{
    auto const record = mRecords.find(id);
    ELLM_CHECK(record != mRecords.end(), "Context cache record ID does not exist");
    mLru.splice(mLru.end(), mLru, record->second.lruPosition);
}

CacheRecord CacheRecordStore::erase(RecordId id)
{
    auto const record = mRecords.find(id);
    ELLM_CHECK(record != mRecords.end(), "Context cache record ID does not exist");
    auto const exact = mExactIndex.find(record->second.record.key);
    ELLM_CHECK(exact != mExactIndex.end() && exact->second == id, "Context cache record indexes are inconsistent");

    std::optional<HybridCheckpointKey> const hybridKey = record->second.record.hybridKey();
    if (hybridKey.has_value())
    {
        auto const hybrid = mHybridIndex.find(*hybridKey);
        ELLM_CHECK(hybrid != mHybridIndex.end() && hybrid->second == id,
            "Context cache hybrid checkpoint index is inconsistent");
        mHybridIndex.erase(hybrid);
    }

    CacheRecord erased = std::move(record->second.record);
    mExactIndex.erase(exact);
    mLru.erase(record->second.lruPosition);
    mRecords.erase(record);
    return erased;
}

std::vector<RecordId> CacheRecordStore::lruToMru() const
{
    return std::vector<RecordId>(mLru.begin(), mLru.end());
}

size_t CacheRecordStore::size() const noexcept
{
    return mRecords.size();
}

int32_t CacheRecordStore::maxRecords() const noexcept
{
    return mMaxRecords;
}

} // namespace rt
} // namespace trt_edgellm
