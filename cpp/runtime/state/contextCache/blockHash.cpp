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

// Integers are serialized least-significant byte first, and field tags plus presence markers make the byte stream
// unambiguous. FNV-1a is deterministic and inexpensive but is not collision-resistant; BlockHash equality has no
// secondary token comparison in this layer.
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
    kMediaToken = 19, // 18 is the opaque-identity tag (common/identityHash.cpp).
};

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

} // namespace

BlockHash hashBlock(BlockHash parent, int32_t const* tokens, size_t count, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash)
{
    ELLM_CHECK(tokens != nullptr || count == 0U, "Cannot hash context cache tokens from a null pointer");

    FnvValue state = kFNV_OFFSET;
    fnvTag(state, Tag::kParent);
    fnvHash(state, parent);

    fnvTag(state, Tag::kTokens);
    fnvU64(state, static_cast<uint64_t>(count));
    for (size_t index = 0; index < count; ++index)
    {
        if (perPositionMediaHash != nullptr && perPositionMediaHash[index] != Hash128{})
        {
            fnvTag(state, Tag::kMediaToken);
            fnvHash(state, perPositionMediaHash[index]);
        }
        else
        {
            fnvI32(state, tokens[index]);
        }
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

std::vector<BlockHash> hashFullBlocks(int32_t const* tokens, size_t tokenCount, int32_t pageSize,
    std::vector<BlockKeyExtras> const& extrasPerBlock, Hash128 const* perPositionMediaHash)
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
        Hash128 const* blockMediaHash
            = perPositionMediaHash != nullptr ? perPositionMediaHash + block * pageTokenCount : nullptr;
        parent = hashBlock(parent, tokens + block * pageTokenCount, pageTokenCount, extras, blockMediaHash);
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
