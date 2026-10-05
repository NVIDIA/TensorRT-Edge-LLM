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

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_a16_blackwell_moe
{

//! Backend selection for Nvfp4A16BlackwellMoePlugin (plugin attribute `backend`).
enum class Backend : int32_t
{
    kAuto = 0,    //!< token-count policy below
    kDecode = 1,  //!< force the CUDA-core decode kernels
    kPrefill = 2, //!< force the tcgen05 grouped GEMM
};

//! Token (MMA-N) tiles baked into the nvfp4_a16_blackwell_moe AOT group. The
//! tile doubles as the per-expert row padding granularity of the permuted
//! activation buffer.
enum class TokenTile : int32_t
{
    kTn8 = 8,
    kTn16 = 16,
    kTn32 = 32,
    kTn64 = 64,
    kTn128 = 128,
};

inline constexpr int32_t kLargestTokenTile{128};

//! Upper bound on token tiles per grouped-GEMM launch: the kernel stages
//! tile_group_idx into shared memory once per CTA (MAX_TOKEN_TILES in
//! nvfp4_a16_blackwell_moe_gemm.py). validShape rejects profiles whose padded
//! row count at the selected tile would exceed it (T > ~19k at E=128, topK=6).
inline constexpr int32_t kMaxTokenTiles{1024};

// ---------------------------------------------------------------------------
// Production policy for Thor SM110 and the Nemotron 3.5 Lightning shape
// (E=128, topK=6, H=2688, I=1856, FP16), sealed from a Marlin-vs-Blackwell
// plugin benchmark on Jetson AGX Thor (MAXN, CUDA-graph replay, cold L2):
//   * decode kernels only at T=1: every routed row is its own CUDA-core GEMV,
//     so from T=2 the grouped GEMM (each expert streamed once) wins.
//   * token tile = per-expert padding granularity of the grouped path. Small
//     tiles avoid pad rows, large tiles avoid re-streaming hot experts; the
//     thresholds below are the best worst case over uniform and skewed routing.
//   * the GEMM streams each 4 KB weight row tile as one 2 KB-row TMA box over
//     the pre-swizzled layout: 32-byte TMA rows cap at ~230 GB/s on Thor,
//     64 B+ rows reach 258-270 GB/s.
// Values are policy hints, not support gates; the runner validates shapes
// independently.
// ---------------------------------------------------------------------------
inline constexpr int32_t kDecodeMaxTokens{1};
//! Activation dtypes of the CUDA-core kernels (part of the JIT key, like the
//! decode constants below and their EDGELLM_MOE_DECODE_* overrides, all fixed
//! when the plugin compiles its bundle at engine build).
enum class DecodeDtype : int32_t
{
    kFP16 = 0,
    kBF16 = 1,
};

//! Decode FC1 split-K: FC1 has only 90 row tiles at T=1 (2.25 waves of 40
//! resident CTAs on 20 SMs); split-K 2 doubles the CTA count and was the best
//! of {1,2,4,8} in the engine (decode step 11.41 ms vs 11.57 / 11.49 / 11.63).
//! The fp32 partials cost one ~3 us reduce; decode stays deterministic.
inline constexpr int32_t kDecodeFc1SplitK{2};
//! Largest FC1 split-K the benchmark override may select; the decode workspace
//! is sized for it so the size recorded at engine build never depends on the
//! environment.
inline constexpr int32_t kDecodeFc1MaxSplitK{8};
static_assert(kDecodeFc1SplitK >= 1 && kDecodeFc1SplitK <= kDecodeFc1MaxSplitK, "sealed FC1 split-K out of range");
inline constexpr int32_t kDecodeFc2SplitK{8};
//! Decode FC2 slots whose weight tiles are staged into shared memory before the
//! kernel's PDL wait. Sealed to 0: the larger shared-memory carve-out (18 KB per
//! staged slot for Nemotron) keeps FC2 from co-residing with the shared-expert
//! GEMV on TensorRT's auxiliary stream, which cost more than the staging saved
//! (decode step 11.44 / 11.51 / 11.52 ms for 0 / 1 / 2 slots). The path stays
//! available through EDGELLM_MOE_DECODE_FC2_PREFETCH or the runner parameter
//! and is exercised by the unit tests.
inline constexpr int32_t kDecodeFc2PrefetchSlots{0};
inline constexpr int32_t kDecodeFc2MaxPrefetchSlots{2};

constexpr Backend resolveBackend(Backend const requested, int32_t const numTokens) noexcept
{
    if (requested != Backend::kAuto)
    {
        return requested;
    }
    return numTokens <= kDecodeMaxTokens ? Backend::kDecode : Backend::kPrefill;
}

constexpr TokenTile selectTokenTile(int32_t const numTokens) noexcept
{
    // Expected rows per expert is T * topK / E (0.75 at T=16, 12 at T=256 for
    // the Nemotron shape); the tile is the padding granularity, so small tiles
    // remove the pad rows FC1 reads/writes while weights dominate the bytes.
    if (numTokens <= 16)
    {
        return TokenTile::kTn8;
    }
    if (numTokens <= 32)
    {
        return TokenTile::kTn16;
    }
    if (numTokens <= 256)
    {
        return TokenTile::kTn32;
    }
    if (numTokens <= 2048)
    {
        return TokenTile::kTn64;
    }
    return TokenTile::kTn128;
}

//! Conservative padded-row capacity of the permuted activation buffer for a
//! given tile: every expert may waste tile-1 rows. Multiple of `tile`.
constexpr int64_t maxRowsPadded(
    int64_t const numTokens, int64_t const topK, int64_t const numExperts, int64_t const tile) noexcept
{
    int64_t const rows = numTokens * topK + numExperts * (tile - 1);
    return (rows + tile - 1) / tile * tile;
}

} // namespace nvfp4_a16_blackwell_moe
} // namespace kernel
} // namespace trt_edgellm
