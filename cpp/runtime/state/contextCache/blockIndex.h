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
#include "runtime/state/contextCache/contextCacheTypes.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct CacheRecord;

//! Lookup key for one canonical base-model KV page.
//!
//! The domain separates incompatible model/configuration state while the chained block hash identifies the logical
//! prefix within that domain.
struct BaseBlockKey
{
    CacheDomainId domain{};
    BlockHash hash{};
};

inline bool operator==(BaseBlockKey const& lhs, BaseBlockKey const& rhs) noexcept
{
    return lhs.domain == rhs.domain && lhs.hash == rhs.hash;
}

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::BaseBlockKey>
{
    size_t operator()(trt_edgellm::rt::BaseBlockKey const& key) const noexcept
    {
        constexpr size_t kHASH_COMBINE_CONSTANT = static_cast<size_t>(0x9E3779B97F4A7C15ULL);
        size_t const domainHash = hash<trt_edgellm::rt::Hash128>{}(key.domain);
        size_t const blockHash = hash<trt_edgellm::rt::Hash128>{}(key.hash);
        return domainHash ^ (blockHash + kHASH_COMBINE_CONSTANT + (domainHash << 6U) + (domainHash >> 2U));
    }
};

} // namespace std

namespace trt_edgellm
{
namespace rt
{

struct BaseLookupResult
{
    std::vector<PageId> pageIds;
    std::vector<BlockHash> matchedHashes;
};

struct BaseInsertResult
{
    PageId canonicalPage{};
    bool inserted{};
};

//! Lookup identity for one coherent EAGLE draft path at a complete block boundary.
struct DraftPathKey
{
    DraftEngineSignature draftSignature{};
    CacheDomainId domain{};
    BlockHash terminalHash{};
};

inline bool operator==(DraftPathKey const& lhs, DraftPathKey const& rhs) noexcept
{
    return lhs.draftSignature == rhs.draftSignature && lhs.domain == rhs.domain && lhs.terminalHash == rhs.terminalHash;
}

//! One record candidate that owns a coherent draft path through pathBlockCount.
struct DraftPathMatch
{
    RecordId record{};
    int32_t pathBlockCount{};
};

inline bool operator==(DraftPathMatch const& lhs, DraftPathMatch const& rhs) noexcept
{
    return lhs.record == rhs.record && lhs.pathBlockCount == rhs.pathBlockCount;
}

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::DraftPathKey>
{
    size_t operator()(trt_edgellm::rt::DraftPathKey const& key) const noexcept
    {
        constexpr size_t kHASH_COMBINE_CONSTANT = static_cast<size_t>(0x9E3779B97F4A7C15ULL);
        size_t combined = hash<trt_edgellm::rt::Hash128>{}(key.draftSignature);
        size_t const domainHash = hash<trt_edgellm::rt::Hash128>{}(key.domain);
        combined ^= domainHash + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U);
        size_t const terminalHash = hash<trt_edgellm::rt::Hash128>{}(key.terminalHash);
        combined ^= terminalHash + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U);
        return combined;
    }
};

} // namespace std

namespace trt_edgellm
{
namespace rt
{

//! Maps logical base-model blocks to their canonical physical KV pages.
//!
//! The index is a lookup structure, not an owner: ResourcePools tracks page references and CacheRecordStore tracks the
//! records that keep pages resident. The forward map supports longest-prefix matching; the reverse map removes the
//! exact logical mapping when the last cache reference to a page is evicted. Lookup has no ownership or LRU side
//! effects, and publication uses first-committer-wins semantics.
class BaseBlockIndex
{
public:
    //! Return the longest contiguous prefix present in the index without changing recency or ownership.
    BaseLookupResult lookupPrefix(CacheDomainId domain, std::vector<BlockHash> const& hashes) const;
    std::optional<PageId> lookup(BaseBlockKey const& key) const;
    //! First committer wins: an existing key returns its canonical page without replacing it.
    BaseInsertResult insert(BaseBlockKey key, PageId proposedPage);
    //! Remove the exact reverse mapping for a physical page; an absent page is a no-op.
    void erasePage(PageId page);
    size_t size() const noexcept;

private:
    std::unordered_map<BaseBlockKey, PageId> mForward;
    std::unordered_map<PageId, BaseBlockKey> mReverse;
};

//! Maps EAGLE draft block boundaries to complete record-owned paths.
//!
//! Unlike BaseBlockIndex, draft pages are not independently canonicalized. Every match identifies one CacheRecord so
//! callers always consume a coherent path from one producer rather than stitching pages from unrelated records.
class DraftPathIndex
{
public:
    //! Register every paired full-block boundary in one EAGLE-capable record.
    void insert(CacheRecord const& record);
    //! Register boundaries beginning at firstBlockCount for a same-signature extension of an indexed record.
    void insertFrom(CacheRecord const& record, int32_t firstBlockCount);
    //! Return the longest boundary at or below maxBlockCount without changing ownership or recency.
    std::optional<DraftPathMatch> lookupLongest(DraftEngineSignature signature, CacheDomainId domain,
        std::vector<BlockHash> const& hashes, int32_t maxBlockCount) const;
    bool contains(DraftPathKey const& key, DraftPathMatch const& match) const;
    //! Remove only the entries owned by record; a base-only record is a no-op.
    void erase(CacheRecord const& record);

private:
    std::unordered_map<DraftPathKey, std::vector<DraftPathMatch>> mForward;
};

} // namespace rt
} // namespace trt_edgellm
