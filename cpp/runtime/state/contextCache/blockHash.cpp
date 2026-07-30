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

#include "runtime/state/contextCache/blockHash.h"

#include "common/checkMacros.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{
namespace
{

using FnvValue = __uint128_t;

// 128-bit Fowler-Noll-Vo (FNV-1a) starts from the standard offset basis and consumes one byte at a time:
//     state = (state XOR byte) * FNV_prime modulo 2^128.
// Unsigned 128-bit overflow supplies the modulo operation. Integers below are serialized least-significant byte first,
// and field tags plus presence markers make the byte stream unambiguous. FNV-1a is deterministic and inexpensive but
// is not collision-resistant; BlockHash equality has no secondary token comparison in this layer.
constexpr uint64_t kFNV_OFFSET_HI = 0x6C62272E07BB0142ULL;
constexpr uint64_t kFNV_OFFSET_LO = 0x62B821756295C58DULL;
constexpr FnvValue kFNV_OFFSET = (static_cast<FnvValue>(kFNV_OFFSET_HI) << 64U) | static_cast<FnvValue>(kFNV_OFFSET_LO);
constexpr FnvValue kFNV_PRIME
    = (static_cast<FnvValue>(1) << 88U) | (static_cast<FnvValue>(1) << 8U) | static_cast<FnvValue>(0x3B);

enum class Tag : uint8_t
{
    kParent = 1,
    kTokens,
    kMedia,
    kMediaEntry,
    kMediaContentDigest,
    kMediaPlacementDigest,
    kMediaOffset,
    kMediaItemOrder,
    kMediaModality,
    kAdapter,
    kAdapterId,
    kAdapterGeneration,
    kPositionDigest,
    kCustomEmbeddingDigest,
    kIsolationDigest,
    kAbsent,
    kPresent,
    kOpaqueIdentity,
};

void fnvByte(FnvValue& state, uint8_t value) noexcept
{
    state ^= static_cast<FnvValue>(value);
    state *= kFNV_PRIME;
}

void fnvTag(FnvValue& state, Tag tag) noexcept
{
    fnvByte(state, static_cast<uint8_t>(tag));
}

void fnvU32(FnvValue& state, uint32_t value) noexcept
{
    // Fixed little-endian serialization keeps the hash independent of host byte order.
    constexpr size_t kBYTE_COUNT = sizeof(value);
    constexpr size_t kBITS_PER_BYTE = 8U;
    for (size_t index = 0; index < kBYTE_COUNT; ++index)
    {
        fnvByte(state, static_cast<uint8_t>(value >> (kBITS_PER_BYTE * index)));
    }
}

void fnvI32(FnvValue& state, int32_t value) noexcept
{
    fnvU32(state, static_cast<uint32_t>(value));
}

void fnvU64(FnvValue& state, uint64_t value) noexcept
{
    constexpr size_t kBYTE_COUNT = sizeof(value);
    constexpr size_t kBITS_PER_BYTE = 8U;
    for (size_t index = 0; index < kBYTE_COUNT; ++index)
    {
        fnvByte(state, static_cast<uint8_t>(value >> (kBITS_PER_BYTE * index)));
    }
}

void fnvHash(FnvValue& state, Hash128 const& hash) noexcept
{
    fnvU64(state, hash.hi);
    fnvU64(state, hash.lo);
}

void fnvPresence(FnvValue& state, bool present) noexcept
{
    fnvTag(state, present ? Tag::kPresent : Tag::kAbsent);
}

void fnvOptionalHash(FnvValue& state, Tag tag, std::optional<Hash128> const& hash) noexcept
{
    fnvTag(state, tag);
    fnvPresence(state, hash.has_value());
    if (hash.has_value())
    {
        fnvHash(state, *hash);
    }
}

BlockHash finishHash(FnvValue state) noexcept
{
    constexpr size_t kHALF_WIDTH = 64U;
    return BlockHash{static_cast<uint64_t>(state >> kHALF_WIDTH), static_cast<uint64_t>(state)};
}

} // namespace

Hash128 hashOpaqueIdentity(std::string_view bytes)
{
    FnvValue state = kFNV_OFFSET;
    fnvTag(state, Tag::kOpaqueIdentity);
    fnvU64(state, static_cast<uint64_t>(bytes.size()));
    for (char const value : bytes)
    {
        fnvByte(state, static_cast<uint8_t>(value));
    }
    return finishHash(state);
}

BlockHash hashBlock(BlockHash parent, int32_t const* tokens, size_t count, BlockKeyExtras const& extras)
{
    ELLM_CHECK(tokens != nullptr || count == 0U, "Cannot hash context cache tokens from a null pointer");

    FnvValue state = kFNV_OFFSET;
    fnvTag(state, Tag::kParent);
    fnvHash(state, parent);

    fnvTag(state, Tag::kTokens);
    fnvU64(state, static_cast<uint64_t>(count));
    for (size_t index = 0; index < count; ++index)
    {
        fnvI32(state, tokens[index]);
    }

    fnvTag(state, Tag::kMedia);
    fnvU64(state, static_cast<uint64_t>(extras.media.size()));
    for (MediaSpanKey const& media : extras.media)
    {
        fnvTag(state, Tag::kMediaEntry);
        fnvTag(state, Tag::kMediaContentDigest);
        fnvHash(state, media.contentDigest);
        fnvTag(state, Tag::kMediaPlacementDigest);
        fnvHash(state, media.placementDigest);
        fnvTag(state, Tag::kMediaOffset);
        fnvI32(state, media.offsetWithinBlock);
        fnvTag(state, Tag::kMediaItemOrder);
        fnvI32(state, media.itemOrder);
        fnvTag(state, Tag::kMediaModality);
        fnvByte(state, media.modality);
    }

    fnvTag(state, Tag::kAdapter);
    fnvPresence(state, extras.adapter.has_value());
    if (extras.adapter.has_value())
    {
        fnvTag(state, Tag::kAdapterId);
        fnvHash(state, extras.adapter->id);
        fnvTag(state, Tag::kAdapterGeneration);
        fnvU64(state, extras.adapter->generation);
    }

    fnvOptionalHash(state, Tag::kPositionDigest, extras.positionDigest);
    fnvOptionalHash(state, Tag::kCustomEmbeddingDigest, extras.customEmbeddingDigest);
    fnvOptionalHash(state, Tag::kIsolationDigest, extras.isolationDigest);

    return finishHash(state);
}

std::vector<BlockHash> hashFullBlocks(
    int32_t const* tokens, size_t tokenCount, int32_t pageSize, std::vector<BlockKeyExtras> const& extrasPerBlock)
{
    ELLM_CHECK(pageSize > 0, "Context cache page size must be positive");

    size_t const pageTokenCount = static_cast<size_t>(pageSize);
    size_t const blockCount = tokenCount / pageTokenCount;
    ELLM_CHECK(extrasPerBlock.empty() || extrasPerBlock.size() == blockCount,
        "Context cache block extras count must match the number of full blocks");
    ELLM_CHECK(tokens != nullptr || blockCount == 0U, "Cannot hash context cache tokens from a null pointer");

    std::vector<BlockHash> hashes;
    hashes.reserve(blockCount);
    BlockHash parent = kCHAIN_ROOT;
    BlockKeyExtras const emptyExtras;
    for (size_t block = 0; block < blockCount; ++block)
    {
        BlockKeyExtras const& extras = extrasPerBlock.empty() ? emptyExtras : extrasPerBlock[block];
        parent = hashBlock(parent, tokens + block * pageTokenCount, pageTokenCount, extras);
        hashes.push_back(parent);
    }
    return hashes;
}

BlockHash hashExactPrefix(
    int32_t const* tokens, size_t tokenCount, int32_t pageSize, std::vector<BlockKeyExtras> const& extrasPerBlock)
{
    ELLM_CHECK(pageSize > 0, "Context cache page size must be positive");
    ELLM_CHECK(tokens != nullptr || tokenCount == 0U, "Cannot hash context cache tokens from a null pointer");

    size_t const pageTokenCount = static_cast<size_t>(pageSize);
    size_t const touchedBlockCount = (tokenCount + pageTokenCount - 1U) / pageTokenCount;
    ELLM_CHECK(extrasPerBlock.empty() || extrasPerBlock.size() == touchedBlockCount,
        "Context cache block extras count must match the number of touched blocks");

    BlockHash parent = kCHAIN_ROOT;
    BlockKeyExtras const emptyExtras;
    for (size_t block = 0; block < touchedBlockCount; ++block)
    {
        size_t const offset = block * pageTokenCount;
        size_t const count = std::min(pageTokenCount, tokenCount - offset);
        BlockKeyExtras const& extras = extrasPerBlock.empty() ? emptyExtras : extrasPerBlock[block];
        parent = hashBlock(parent, tokens + offset, count, extras);
    }
    return parent;
}

} // namespace rt
} // namespace trt_edgellm
