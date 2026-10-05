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

//! Header-only CPU reference for the QSA indexer pipeline (see
//! cpp/kernels/qsaIndexer/qsaIndexerKernels.h for the numerics contract). Deliberately kept
//! local to unittests/cpp/kernels/qsaIndexer instead of unittests/support/references.{h,cpp}.
//! All stages take/return FP32 host arrays that represent half-rounded device values;
//! reductions run in double so the reference is strictly more precise than the kernels.

#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <numeric>
#include <random>
#include <vector>

namespace qsa_ref
{

constexpr int32_t kNumHeads = 4;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kRatio = 4;
constexpr int32_t kTopk = 512;
constexpr int32_t kWidth = 2051;
constexpr int32_t kRotaryDim = 64;
constexpr int32_t kIndexQkWidth = (kNumHeads + 1) * kHeadDim; // 640
constexpr int32_t kQRowWidth = kNumHeads * kHeadDim;          // 512

inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

//! Round every element through FP16 (the values a device half buffer would hold).
inline std::vector<float> halfRound(std::vector<float> const& src)
{
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = __half2float(__float2half(src[i]));
    }
    return dst;
}

//! Neox partial rope on one 128-dim head held in double: pair (i, i+32) rotated with
//! cos = cosSin[pos][i], sin = cosSin[pos][32+i]; dims [64, 128) pass through.
inline void applyNeoxRope(std::vector<double>& head, std::vector<float> const& cosSin, int32_t pos)
{
    float const* cs = cosSin.data() + static_cast<int64_t>(pos) * kRotaryDim;
    std::vector<double> in(head.begin(), head.begin() + kRotaryDim);
    for (int32_t i = 0; i < kRotaryDim / 2; ++i)
    {
        double const c = cs[i];
        double const s = cs[kRotaryDim / 2 + i];
        head[i] = in[i] * c - in[i + kRotaryDim / 2] * s;
        head[i + kRotaryDim / 2] = in[i + kRotaryDim / 2] * c + in[i] * s;
    }
}

//! Gemma-norm one 128-dim head: y = x / sqrt(mean(x^2) + eps) * (1 + w).
inline void applyGemmaNorm(std::vector<double>& head, std::vector<float> const& w, float eps)
{
    double sumSq = 0.0;
    for (int32_t d = 0; d < kHeadDim; ++d)
    {
        sumSq += head[d] * head[d];
    }
    double const inv = 1.0 / std::sqrt(sumSq / kHeadDim + static_cast<double>(eps));
    for (int32_t d = 0; d < kHeadDim; ++d)
    {
        head[d] = head[d] * inv * (1.0 + static_cast<double>(w[d]));
    }
}

//! K1a reference: qNormed [B, S, 4, 128] (pre-final-cast FP32 values; padding rows zero).
//! indexQk holds half-rounded FP32 values [B, S, 640].
inline std::vector<float> refQPrep(std::vector<float> const& indexQk, std::vector<float> const& cosSin,
    std::vector<int32_t> const& contextLengths, std::vector<float> const& wQ, float eps, int32_t batchSize,
    int32_t seqLen)
{
    std::vector<float> out(static_cast<size_t>(batchSize) * seqLen * kQRowWidth, 0.0f);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t t = 0; t < contextLengths[b]; ++t)
        {
            int64_t const row = static_cast<int64_t>(b) * seqLen + t;
            for (int32_t h = 0; h < kNumHeads; ++h)
            {
                float const* src = indexQk.data() + row * kIndexQkWidth + h * kHeadDim;
                std::vector<double> head(src, src + kHeadDim);
                applyGemmaNorm(head, wQ, eps);
                applyNeoxRope(head, cosSin, t);
                for (int32_t d = 0; d < kHeadDim; ++d)
                {
                    out[(row * kNumHeads + h) * kHeadDim + d] = static_cast<float>(head[d]);
                }
            }
        }
    }
    return out;
}

//! K1b reference: kbar [B, ceilDiv(S, 4), 128] (pre-final-cast FP32; invalid blocks zero).
//! The FP32 fixed-order mean and the intermediate FP16 cast follow the kernel bit-for-bit.
inline std::vector<float> refKCompress(std::vector<float> const& indexQk, std::vector<float> const& cosSin,
    std::vector<int32_t> const& contextLengths, std::vector<float> const& wK, float eps, int32_t batchSize,
    int32_t seqLen)
{
    int32_t const numBlocks = ceilDiv(seqLen, kRatio);
    std::vector<float> out(static_cast<size_t>(batchSize) * numBlocks * kHeadDim, 0.0f);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t g = 0; g < numBlocks; ++g)
        {
            if (g * kRatio + kRatio - 1 >= contextLengths[b])
            {
                continue; // incomplete block stays zero
            }
            std::vector<double> head(kHeadDim);
            for (int32_t d = 0; d < kHeadDim; ++d)
            {
                // FP32 mean in FIXED sequential order, then the contract-critical FP16 round-trip.
                float acc = 0.0f;
                for (int32_t j = 0; j < kRatio; ++j)
                {
                    int64_t const row = static_cast<int64_t>(b) * seqLen + g * kRatio + j;
                    acc = acc + indexQk[row * kIndexQkWidth + kQRowWidth + d];
                }
                acc = acc * 0.25f;
                head[d] = __half2float(__float2half(acc));
            }
            applyGemmaNorm(head, wK, eps);
            applyNeoxRope(head, cosSin, g * kRatio);
            for (int32_t d = 0; d < kHeadDim; ++d)
            {
                out[(static_cast<int64_t>(b) * numBlocks + g) * kHeadDim + d] = static_cast<float>(head[d]);
            }
        }
    }
    return out;
}

//! K2 reference for one query row: double-precision logits [numBlocks] from half-rounded
//! qNormed/kbar. Masked (g >= (t+1)/4) and padded rows get -FLT_MAX like the kernel.
inline std::vector<double> refLogitsRow(std::vector<float> const& qNormed, std::vector<float> const& kbar,
    std::vector<int32_t> const& contextLengths, int32_t batchSize, int32_t seqLen, int32_t b, int32_t t)
{
    (void) batchSize;
    int32_t const numBlocks = ceilDiv(seqLen, kRatio);
    std::vector<double> logits(numBlocks, -static_cast<double>(FLT_MAX));
    int32_t const contextLen = contextLengths[b];
    if (t >= contextLen || contextLen == 0)
    {
        return logits;
    }
    int32_t const numVisible = (t + 1) / kRatio;
    int64_t const row = static_cast<int64_t>(b) * seqLen + t;
    for (int32_t g = 0; g < numVisible; ++g)
    {
        double sum = 0.0;
        for (int32_t h = 0; h < kNumHeads; ++h)
        {
            double dot = 0.0;
            for (int32_t d = 0; d < kHeadDim; ++d)
            {
                dot += static_cast<double>(qNormed[(row * kNumHeads + h) * kHeadDim + d])
                    * kbar[(static_cast<int64_t>(b) * numBlocks + g) * kHeadDim + d];
            }
            sum += std::max(0.0, dot); // relu PER HEAD, then sum over heads
        }
        logits[g] = sum / std::sqrt(static_cast<double>(kHeadDim)); // scale AFTER the sum
    }
    return logits;
}

//! Block ids of one row sorted by descending logit; ties keep ascending id (matches the
//! stable cub radix sort over ascending-id-filled values).
inline std::vector<int32_t> refSortedIds(std::vector<double> const& logitsRow)
{
    std::vector<int32_t> ids(logitsRow.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(
        ids.begin(), ids.end(), [&logitsRow](int32_t a, int32_t b) { return logitsRow[a] > logitsRow[b]; });
    return ids;
}

//! K4 reference for one row: expanded token indices [kWidth]. Padded rows (t >= L or L == 0)
//! are all -1 and never read sortedIds.
inline std::vector<int32_t> refExpandRow(std::vector<int32_t> const& sortedIds, int32_t t, int32_t contextLen)
{
    std::vector<int32_t> out(kWidth, -1);
    if (t >= contextLen || contextLen == 0)
    {
        return out;
    }
    int32_t const numVisible = (t + 1) / kRatio;
    int32_t const numSelected = std::min(kTopk, numVisible);
    int32_t const tailCount = (t + 1) % kRatio;
    for (int32_t i = 0; i < numSelected; ++i)
    {
        for (int32_t j = 0; j < kRatio; ++j)
        {
            out[i * kRatio + j] = sortedIds[i] * kRatio + j;
        }
    }
    for (int32_t j = 0; j < tailCount; ++j) // causal tail, ALWAYS appended
    {
        out[numSelected * kRatio + j] = numVisible * kRatio + j;
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// Paged-pool / decode references (see QsaIndexerPoolState in qsaIndexerKernels.h for the
// addressing contract these mirror).
// ---------------------------------------------------------------------------------------

constexpr int32_t kTokensPerPage = 128;

//! Element offset of (page, rowInPage, head, col) in a pool [2*numPages, 128, numKVHeads,
//! poolHeadDim].
inline int64_t poolOffset(
    int32_t page, int32_t rowInPage, int32_t head, int32_t col, int32_t numKVHeads, int32_t poolHeadDim)
{
    return (static_cast<int64_t>(page) * kTokensPerPage + rowInPage) * numKVHeads * poolHeadDim
        + static_cast<int64_t>(head) * poolHeadDim + col;
}

//! Page id of `token` on `plane` (0 = K, 1 = V) in a table [batchSize, 2, maxPagesPerSeq].
inline int32_t pageOf(
    std::vector<int32_t> const& pageTable, int32_t maxPagesPerSeq, int32_t b, int32_t plane, int32_t token)
{
    return pageTable[(static_cast<int64_t>(b) * 2 + plane) * maxPagesPerSeq + token / kTokensPerPage];
}

//! Permuted NON-IDENTITY page table [batchSize, 2, maxPagesPerSeq]: the first pagesUsed
//! slots of every sequence get distinct page ids drawn from independently shuffled
//! K-plane / V-plane permutations of [0, numPages) (re-shuffled if the draw came out
//! sorted, so the mapping is never the identity); V ids are pre-offset +numPages; all
//! remaining slots are -1 (kUNUSED_PAGE_ENTRY). Requires batchSize * pagesUsed <= numPages
//! and numPages >= 2.
inline std::vector<int32_t> makePermutedPageTable(
    int32_t batchSize, int32_t maxPagesPerSeq, int32_t pagesUsed, int32_t numPages, uint32_t seed)
{
    std::vector<int32_t> table(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq, -1);
    std::mt19937 rng(seed);
    std::vector<int32_t> kPerm(numPages);
    std::vector<int32_t> vPerm(numPages);
    std::iota(kPerm.begin(), kPerm.end(), 0);
    std::iota(vPerm.begin(), vPerm.end(), 0);
    do
    {
        std::shuffle(kPerm.begin(), kPerm.end(), rng);
    } while (std::is_sorted(kPerm.begin(), kPerm.end()));
    do
    {
        std::shuffle(vPerm.begin(), vPerm.end(), rng);
    } while (std::is_sorted(vPerm.begin(), vPerm.end()));
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t p = 0; p < pagesUsed; ++p)
        {
            table[(static_cast<int64_t>(b) * 2 + 0) * maxPagesPerSeq + p] = kPerm[b * pagesUsed + p];
            table[(static_cast<int64_t>(b) * 2 + 1) * maxPagesPerSeq + p] = vPerm[b * pagesUsed + p] + numPages;
        }
    }
    return table;
}

//! B1 phase-1 reference: q-prep of ONE decode row [640] at an explicit rope position;
//! returns [4, 128] pre-final-cast FP32 values.
inline std::vector<float> refQPrepDecodeRow(float const* indexQkRow, std::vector<float> const& cosSin, int32_t position,
    std::vector<float> const& wQ, float eps)
{
    std::vector<float> out(kQRowWidth);
    for (int32_t h = 0; h < kNumHeads; ++h)
    {
        std::vector<double> head(indexQkRow + h * kHeadDim, indexQkRow + (h + 1) * kHeadDim);
        applyGemmaNorm(head, wQ, eps);
        applyNeoxRope(head, cosSin, position);
        for (int32_t d = 0; d < kHeadDim; ++d)
        {
            out[h * kHeadDim + d] = static_cast<float>(head[d]);
        }
    }
    return out;
}

//! K1b reference over four explicit half-rounded raw key rows [4][128] (the pool K-tails
//! of tokens firstToken .. firstToken + 2 plus the completing token's key from indexQk);
//! returns kbar [128] pre-final-cast FP32 values.
//! The fixed-order FP32 mean and the intermediate FP16 cast follow the kernel bit-for-bit.
inline std::vector<float> refKCompressGroup(std::array<float const*, kRatio> const& rawK,
    std::vector<float> const& cosSin, int32_t firstToken, std::vector<float> const& wK, float eps)
{
    std::vector<double> head(kHeadDim);
    for (int32_t d = 0; d < kHeadDim; ++d)
    {
        float acc = rawK[0][d];
        acc = acc + rawK[1][d];
        acc = acc + rawK[2][d];
        acc = acc + rawK[3][d];
        acc = acc * 0.25f;
        head[d] = __half2float(__float2half(acc));
    }
    applyGemmaNorm(head, wK, eps);
    applyNeoxRope(head, cosSin, firstToken);
    std::vector<float> out(kHeadDim);
    for (int32_t d = 0; d < kHeadDim; ++d)
    {
        out[d] = static_cast<float>(head[d]);
    }
    return out;
}

//! B2 reference: double-precision logits [numVisible] of one decode query row [4, 128]
//! against densely gathered kbar rows [numVisible, 128] (both half-rounded FP32).
inline std::vector<double> refLogitsDecodeRow(
    std::vector<float> const& qRow, std::vector<float> const& kbarRows, int32_t numVisible)
{
    std::vector<double> logits(numVisible);
    for (int32_t g = 0; g < numVisible; ++g)
    {
        double sum = 0.0;
        for (int32_t h = 0; h < kNumHeads; ++h)
        {
            double dot = 0.0;
            for (int32_t d = 0; d < kHeadDim; ++d)
            {
                dot += static_cast<double>(qRow[h * kHeadDim + d]) * kbarRows[static_cast<int64_t>(g) * kHeadDim + d];
            }
            sum += std::max(0.0, dot); // relu PER HEAD, then sum over heads
        }
        logits[g] = sum / std::sqrt(static_cast<double>(kHeadDim)); // scale AFTER the sum
    }
    return logits;
}

//! Monotone orderable key of an FP32 logit: the exact total order the device radix select
//! and cub's float radix sort use (orders -0.0f < +0.0f; NaNs by payload).
inline uint32_t orderKey(float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

//! Top-k block ids of logits[0:numVisible) under the radix order, stable descending —
//! ascending block id among equal keys, i.e. cub::DeviceSegmentedRadixSort descending over
//! ascending-id-filled values (the prefill tie behavior the decode select must match).
inline std::vector<int32_t> refTopKBlocks(std::vector<float> const& logits, int32_t numVisible, int32_t k)
{
    std::vector<int32_t> ids(numVisible);
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(
        ids.begin(), ids.end(), [&logits](int32_t a, int32_t b) { return orderKey(logits[a]) > orderKey(logits[b]); });
    ids.resize(std::min(k, numVisible));
    return ids;
}

} // namespace qsa_ref
