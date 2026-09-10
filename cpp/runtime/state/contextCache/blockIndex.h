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

//! One record candidate that owns a coherent paged spec-state path through pathBlockCount.
struct SpecPagedStateMatch
{
    RecordId record{};
    int32_t pathBlockCount{};
};

inline bool operator==(SpecPagedStateMatch const& lhs, SpecPagedStateMatch const& rhs) noexcept
{
    return lhs.record == rhs.record && lhs.pathBlockCount == rhs.pathBlockCount;
}

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
    BaseLookupResult lookupPrefix(std::vector<BlockHash> const& hashes) const;
    std::optional<PageId> lookup(BlockHash const& hash) const;
    //! First committer wins: an existing key returns its canonical page without replacing it.
    BaseInsertResult insert(BlockHash hash, PageId proposedPage);
    //! Remove the exact reverse mapping for a physical page; an absent page is a no-op.
    void erasePage(PageId page);
    size_t size() const noexcept;

private:
    std::unordered_map<BlockHash, PageId> mForward;
    std::unordered_map<PageId, BlockHash> mReverse;
};

//! Maps paged speculative block boundaries to complete record-owned paths.
//!
//! Unlike BaseBlockIndex, spec pages are not independently canonicalized. Every match identifies one CacheRecord so
//! callers always consume a coherent path from one producer rather than stitching pages from unrelated records.
class SpecPagedStateIndex
{
public:
    //! Register every paired full-block boundary in one paged spec-state record.
    void insert(CacheRecord const& record);
    //! Return the longest boundary at or below maxBlockCount without changing ownership or recency.
    std::optional<SpecPagedStateMatch> lookupLongest(std::vector<BlockHash> const& hashes, int32_t maxBlockCount) const;
    bool contains(BlockHash const& terminalHash, SpecPagedStateMatch const& match) const;
    //! Remove only the entries owned by record; a base-only record is a no-op.
    void erase(CacheRecord const& record);

private:
    std::unordered_map<BlockHash, std::vector<SpecPagedStateMatch>> mForward;
};

//! Wrapper for speculative state lookup substrates.
class SpecStateIndex
{
public:
    SpecPagedStateIndex const& paged() const noexcept;
    SpecPagedStateIndex& paged() noexcept;

private:
    SpecPagedStateIndex mPaged;
};

} // namespace rt
} // namespace trt_edgellm
