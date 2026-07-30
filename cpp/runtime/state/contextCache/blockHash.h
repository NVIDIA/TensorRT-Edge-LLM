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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Two-word representation used for stable context-cache identities.
struct Hash128
{
    uint64_t hi{};
    uint64_t lo{};
};

inline bool operator==(Hash128 const& lhs, Hash128 const& rhs) noexcept
{
    return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}

inline bool operator!=(Hash128 const& lhs, Hash128 const& rhs) noexcept
{
    return !(lhs == rhs);
}

//! Logical identity of one complete token block and all prefix state that produced it.
//!
//! Each block includes its parent's hash, so equal BlockHash values identify equal chained prefixes rather than only
//! equal tokens in the current block. Production chains contain complete blocks; hashFullBlocks() leaves a partial tail
//! private to its request.
using BlockHash = Hash128;
//! Compatibility namespace for every model/configuration property that changes the produced reusable state.
using CacheDomainId = Hash128;
//! Identity of every draft-specific input that can change stored speculative KV state.
//!
//! The producer must change this digest when draft weights, engine/configuration, KV schema/layout/dtype, adapter
//! state, or the EAGLE conditioning contract changes. It is stable across requests and excludes sampling-only policy.
using DraftEngineSignature = Hash128;
//! Identity of the recurrent/conv tensor layout and dtype captured in one exact checkpoint.
using RecurrentStateSchemaId = Hash128;

struct AdapterKey
{
    Hash128 id{};
    uint64_t generation{};
};

struct MediaSpanKey
{
    Hash128 contentDigest{};
    Hash128 placementDigest{};
    int32_t offsetWithinBlock{};
    int32_t itemOrder{};
    uint8_t modality{};
};

struct BlockKeyExtras
{
    //! Non-token inputs whose content or placement changes the state produced for this block.
    std::vector<MediaSpanKey> media;
    std::optional<AdapterKey> adapter;
    std::optional<Hash128> positionDigest;
    std::optional<Hash128> customEmbeddingDigest;
    std::optional<Hash128> isolationDigest;
};

inline constexpr BlockHash kCHAIN_ROOT{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};

//! Deterministically hash exact opaque bytes for adapter, deployment, media, or isolation identity.
//! This uses the same non-cryptographic 128-bit FNV-1a primitive as block chaining.
Hash128 hashOpaqueIdentity(std::string_view bytes);

//! Hash one caller-supplied token block and its non-token identity into its parent chain.
//!
//! The implementation uses deterministic 128-bit Fowler-Noll-Vo (FNV-1a) over a tagged byte stream. It is a
//! non-cryptographic hash; this layer does not perform a secondary token comparison when two hashes are equal.
BlockHash hashBlock(BlockHash parent, int32_t const* tokens, size_t count, BlockKeyExtras const& extras = {});

//! Hash each complete token block in order, chaining from kCHAIN_ROOT and excluding any partial tail.
std::vector<BlockHash> hashFullBlocks(
    int32_t const* tokens, size_t tokenCount, int32_t pageSize, std::vector<BlockKeyExtras> const& extrasPerBlock = {});

//! Hash an exact token prefix, including a partial final block when present.
//!
//! When extras are supplied, one entry is required for every touched block (`ceil(tokenCount / pageSize)`). An aligned
//! prefix returns the same digest as its final complete BlockHash; an empty prefix returns kCHAIN_ROOT.
BlockHash hashExactPrefix(
    int32_t const* tokens, size_t tokenCount, int32_t pageSize, std::vector<BlockKeyExtras> const& extrasPerBlock = {});

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::Hash128>
{
    size_t operator()(trt_edgellm::rt::Hash128 const& value) const noexcept
    {
        constexpr uint64_t kFOLD_MULTIPLIER = 0x9E3779B97F4A7C15ULL;
        return static_cast<size_t>(value.hi ^ (value.lo * kFOLD_MULTIPLIER));
    }
};

} // namespace std
