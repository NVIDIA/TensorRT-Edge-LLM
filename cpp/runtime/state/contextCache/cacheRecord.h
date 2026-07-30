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
#include <cstdint>
#include <functional>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Publication family occupying the exact-record key namespace.
enum class CacheRecordKind : uint8_t
{
    kStandard,
    kHybrid,
    kHybridMtp,
};

//! Exact lookup identity for one published sequence-state checkpoint.
//!
//! The terminal chained hash plus full-block count identifies the logical endpoint inside one compatibility domain.
struct CacheRecordKey
{
    CacheDomainId domain{};
    BlockHash terminalHash{};
    int32_t fullBlockCount{};
    CacheRecordKind kind{CacheRecordKind::kStandard};
};

inline bool operator==(CacheRecordKey const& lhs, CacheRecordKey const& rhs) noexcept
{
    return lhs.domain == rhs.domain && lhs.terminalHash == rhs.terminalHash && lhs.fullBlockCount == rhs.fullBlockCount
        && lhs.kind == rhs.kind;
}

//! Exact identity of an atomic recurrent checkpoint.
struct HybridCheckpointKey
{
    CacheDomainId domain{};
    BlockHash exactPrefixDigest{};
    int32_t exactLength{};
    RecurrentStateSchemaId schema{};
};

inline bool operator==(HybridCheckpointKey const& lhs, HybridCheckpointKey const& rhs) noexcept
{
    return lhs.domain == rhs.domain && lhs.exactPrefixDigest == rhs.exactPrefixDigest
        && lhs.exactLength == rhs.exactLength && lhs.schema == rhs.schema;
}

//! Exact identity of one coherent Hybrid+MTP endpoint checkpoint.
//!
//! The successor-dependent final MTP draft slot is recomputed at restore from the checkpoint's saved boundary hidden
//! state, so the following token is NOT part of identity. A consumer reuses exact length N whenever its prefix matches,
//! regardless of which token follows the checkpoint.
struct HybridMtpCheckpointKey
{
    CacheDomainId domain{};
    BlockHash exactPrefixDigest{};
    int32_t exactLength{};
    RecurrentStateSchemaId schema{};
    DraftEngineSignature draftSignature{};
};

inline bool operator==(HybridMtpCheckpointKey const& lhs, HybridMtpCheckpointKey const& rhs) noexcept
{
    return lhs.domain == rhs.domain && lhs.exactPrefixDigest == rhs.exactPrefixDigest
        && lhs.exactLength == rhs.exactLength && lhs.schema == rhs.schema && lhs.draftSignature == rhs.draftSignature;
}

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::CacheRecordKey>
{
    size_t operator()(trt_edgellm::rt::CacheRecordKey const& key) const noexcept
    {
        constexpr size_t kHASH_COMBINE_CONSTANT = static_cast<size_t>(0x9E3779B97F4A7C15ULL);
        size_t combined = hash<trt_edgellm::rt::Hash128>{}(key.domain);
        size_t const terminalHash = hash<trt_edgellm::rt::Hash128>{}(key.terminalHash);
        combined ^= terminalHash + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U);
        size_t const fullBlockCountHash = hash<int32_t>{}(key.fullBlockCount);
        combined ^= fullBlockCountHash + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U);
        size_t const kindHash = hash<uint8_t>{}(static_cast<uint8_t>(key.kind));
        combined ^= kindHash + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U);
        return combined;
    }
};

template <>
struct hash<trt_edgellm::rt::HybridMtpCheckpointKey>
{
    size_t operator()(trt_edgellm::rt::HybridMtpCheckpointKey const& key) const noexcept
    {
        constexpr size_t kHASH_COMBINE_CONSTANT = static_cast<size_t>(0x9E3779B97F4A7C15ULL);
        size_t combined = hash<trt_edgellm::rt::Hash128>{}(key.domain);
        auto combine
            = [&](size_t value) { combined ^= value + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U); };
        combine(hash<trt_edgellm::rt::Hash128>{}(key.exactPrefixDigest));
        combine(hash<int32_t>{}(key.exactLength));
        combine(hash<trt_edgellm::rt::Hash128>{}(key.schema));
        combine(hash<trt_edgellm::rt::Hash128>{}(key.draftSignature));
        return combined;
    }
};

template <>
struct hash<trt_edgellm::rt::HybridCheckpointKey>
{
    size_t operator()(trt_edgellm::rt::HybridCheckpointKey const& key) const noexcept
    {
        constexpr size_t kHASH_COMBINE_CONSTANT = static_cast<size_t>(0x9E3779B97F4A7C15ULL);
        size_t combined = hash<trt_edgellm::rt::Hash128>{}(key.domain);
        auto combine
            = [&](size_t value) { combined ^= value + kHASH_COMBINE_CONSTANT + (combined << 6U) + (combined >> 2U); };
        combine(hash<trt_edgellm::rt::Hash128>{}(key.exactPrefixDigest));
        combine(hash<int32_t>{}(key.exactLength));
        combine(hash<trt_edgellm::rt::Hash128>{}(key.schema));
        return combined;
    }
};

} // namespace std

namespace trt_edgellm
{
namespace rt
{

//! Complete reusable state retained at one publication endpoint.
//!
//! A record stores a full base path, at most one signature-bound draft path, and any recurrent or partial-page
//! snapshots. The accepted draft path may end before the base path. ContextCacheManager gives each listed resource one
//! cache reference, so evicting a branch releases only that record's ownership while shared ancestors remain resident
//! through other records.
struct CacheRecord
{
    RecordId id{};
    CacheRecordKey key{};
    std::vector<BlockHash> logicalBlockHashes;
    std::vector<PageId> basePagePath;
    std::optional<DraftEngineSignature> draftSignature;
    std::vector<PageId> draftPagePath;
    std::optional<int32_t> recurrentSnapshotSlot;
    std::optional<int32_t> partialKvSnapshotSlot;
    int32_t baseFullBlockCount{};
    int32_t pairedDraftFullBlockCount{};
    std::optional<int32_t> exactCheckpointLength;
    std::optional<BlockHash> exactCheckpointDigest;
    std::optional<RecurrentStateSchemaId> recurrentStateSchema;

    std::vector<ResourceId> resources() const;
    std::optional<HybridCheckpointKey> hybridKey() const;
    std::optional<HybridMtpCheckpointKey> hybridMtpKey() const;
};

struct RecordInsertResult
{
    RecordId id{};
    bool inserted{};
};

//! Owns complete cache records and their exact-key and record-level LRU indexes.
//!
//! The store owns host metadata but does not modify ResourcePools references or base/draft lookup indices; those
//! cross-component updates belong to ContextCacheManager. Insertion updates the record map, its mode-appropriate
//! exact index, and LRU as one transaction. An exact duplicate reuses the existing record and promotes it to MRU.
class CacheRecordStore
{
public:
    explicit CacheRecordStore(int32_t maxRecords);
    CacheRecordStore(CacheRecordStore const&) = delete;
    CacheRecordStore& operator=(CacheRecordStore const&) = delete;
    CacheRecordStore(CacheRecordStore&&) = delete;
    CacheRecordStore& operator=(CacheRecordStore&&) = delete;

    //! Insert with the strong exception guarantee. An exact duplicate is touched, not replaced.
    //! This store does not enforce maxRecords; the manager may hold one transient extra during atomic publication.
    RecordInsertResult insert(CacheRecord record);
    std::optional<RecordId> find(CacheRecordKey const& key) const;
    std::optional<RecordId> findHybrid(HybridCheckpointKey const& key) const;
    std::optional<RecordId> findHybridMtp(HybridMtpCheckpointKey const& key) const;
    //! Ready checkpoint lengths in descending order, excluding exact-input and longer endpoints.
    std::vector<int32_t> hybridCandidateLengths(
        CacheDomainId domain, RecurrentStateSchemaId schema, int32_t inputTokenCount) const;
    //! Ready Hybrid+MTP endpoint lengths in descending order, excluding exact-input and longer endpoints.
    std::vector<int32_t> hybridMtpCandidateLengths(CacheDomainId domain, RecurrentStateSchemaId schema,
        DraftEngineSignature draftSignature, int32_t inputTokenCount) const;
    CacheRecord const& get(RecordId id) const;
    bool contains(RecordId id) const noexcept;
    //! Replace the optional coherent draft state without changing base identity or record ownership.
    void setDraftState(
        RecordId id, DraftEngineSignature signature, std::vector<PageId> draftPagePath, int32_t pairedFullBlockCount);
    void touch(RecordId id);
    CacheRecord erase(RecordId id);
    std::vector<RecordId> lruToMru() const;
    size_t size() const noexcept;
    int32_t maxRecords() const noexcept;

private:
    struct Entry
    {
        CacheRecord record;
        std::list<RecordId>::iterator lruPosition;
    };

    RecordId mNextId{1};
    int32_t mMaxRecords{};
    std::list<RecordId> mLru;
    std::unordered_map<RecordId, Entry> mRecords;
    std::unordered_map<CacheRecordKey, RecordId> mExactIndex;
    std::unordered_map<HybridCheckpointKey, RecordId> mHybridIndex;
    std::unordered_map<HybridMtpCheckpointKey, RecordId> mHybridMtpIndex;
};

} // namespace rt
} // namespace trt_edgellm
