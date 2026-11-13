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

#include "ngramRowIds.h"
#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace rt
{

namespace
{

constexpr uint64_t kSignBit = uint64_t{1} << 63;

//! One token of the history, indexed over the concatenation of the carried-in context and this
//! call's tokens.
inline int32_t historyAt(int32_t const* tokenHistory, int32_t const* tokenIds, int32_t contextLen, int32_t index)
{
    return index < contextLen ? tokenHistory[index] : tokenIds[index - contextLen];
}

//! One entry of HF's `_shift_right_ignore_eos`.
//!
//! HF derives the segment start from a cumulative maximum over EOS positions. Per position that is
//! equivalent to the local test below: reaching back \p shift tokens is allowed only while the whole
//! span stays inside the current segment, so a single EOS anywhere in the skipped span, or running
//! off the front, substitutes EOS. Keeping it local is what lets every position be computed
//! independently.
inline int32_t shiftedToken(int32_t const* tokenHistory, int32_t const* tokenIds, int32_t contextLen, int32_t index,
    int32_t shift, int32_t eosTokenId)
{
    if (index - shift < 0)
    {
        return eosTokenId;
    }
    for (int32_t probe = index - shift; probe < index; ++probe)
    {
        if (historyAt(tokenHistory, tokenIds, contextLen, probe) == eosTokenId)
        {
            return eosTokenId;
        }
    }
    return historyAt(tokenHistory, tokenIds, contextLen, index - shift);
}

//! See ngramRemainder in the header.
//!
//! The negative branch cannot fire for Qwen3.8-Flash-Next: HF caps each multiplier at
//! (2^63 - 1) / vocabSize, so every token * multiplier product stays below 2^63, and an XOR of values
//! whose bit 63 is clear leaves it clear. That rests on two invariants outside this code -- token IDs
//! staying in range, and the checkpoint keeping HF's multiplier construction -- so the branch is kept
//! rather than dropped as dead: a different seed derivation or vocabulary reaches it, and C++ `%`
//! would then select a different row instead of failing.
inline int64_t remainderImpl(uint64_t bits, int64_t divisor)
{
    uint64_t const modulus = static_cast<uint64_t>(divisor);
    uint64_t remainder = bits % modulus;
    if ((bits & kSignBit) != 0U)
    {
        // The value is bits - 2^64. Subtracting 2^64 mod divisor keeps the arithmetic unsigned,
        // where the wraparound is defined; forming the negative value would overflow int64.
        uint64_t const wrap = (~uint64_t{0} % modulus + 1U) % modulus;
        remainder = (remainder + modulus - wrap) % modulus;
    }
    return static_cast<int64_t>(remainder);
}

//! Row IDs of a single position. \p index is the position inside the concatenated history.
inline void rowIdsForPosition(int32_t const* tokenHistory, int32_t const* tokenIds, int32_t contextLen, int32_t index,
    NGramRowIdParams const& params, int64_t* rowIds)
{
    uint64_t mixed
        = static_cast<uint64_t>(shiftedToken(tokenHistory, tokenIds, contextLen, index, 0, params.eosTokenId))
        * params.layerMultipliers[0];

    // Orders share one accumulator: the hash of order n is the hash of order n-1 folded with one
    // more shifted token, which is what makes HF's per-order recomputation redundant.
    for (int32_t ngram = 2; ngram <= params.ngramSize; ++ngram)
    {
        int32_t const order = ngram - 1;
        mixed
            ^= static_cast<uint64_t>(shiftedToken(tokenHistory, tokenIds, contextLen, index, order, params.eosTokenId))
            * params.layerMultipliers[order];

        int32_t const headBegin = (ngram - 2) * params.headsPerNgram;
        for (int32_t head = headBegin; head < headBegin + params.headsPerNgram; ++head)
        {
            rowIds[head] = remainderImpl(mixed, params.headVocabSizes[head]) + params.headOffsets[head];
        }
    }
}

void validate(int32_t batchSize, int32_t seqLen, NGramRowIdParams const& params)
{
    ELLM_CHECK(batchSize > 0, "n-gram row IDs require a positive batch size");
    ELLM_CHECK(seqLen > 0, "n-gram row IDs require a positive sequence length");
    ELLM_CHECK(params.ngramSize >= 2, "n-gram row IDs require ngramSize >= 2");
    ELLM_CHECK(params.headsPerNgram > 0, "n-gram row IDs require a positive headsPerNgram");
    ELLM_CHECK(params.ngramHeads == (params.ngramSize - 1) * params.headsPerNgram,
        "ngramHeads must equal (ngramSize - 1) * headsPerNgram");
    ELLM_CHECK(params.layerMultipliers != nullptr && params.headVocabSizes != nullptr && params.headOffsets != nullptr,
        "n-gram row IDs require all three constant arrays");
}

} // namespace

int64_t ngramRemainder(uint64_t bits, int64_t divisor)
{
    ELLM_CHECK(divisor > 0, "ngramRemainder requires a positive divisor");
    return remainderImpl(bits, divisor);
}

void computeNgramRowIds(int32_t const* tokenIds, int32_t const* tokenHistory, int64_t* rowIds,
    int32_t* nextTokenHistory, int32_t batchSize, int32_t seqLen, NGramRowIdParams const& params)
{
    validate(batchSize, seqLen, params);
    int32_t const contextLen = params.ngramSize - 1;

    for (int32_t sequence = 0; sequence < batchSize; ++sequence)
    {
        int32_t const* const history = tokenHistory + sequence * contextLen;
        int32_t const* const tokens = tokenIds + sequence * seqLen;
        for (int32_t position = 0; position < seqLen; ++position)
        {
            int64_t const rowOffset = (static_cast<int64_t>(sequence) * seqLen + position) * params.ngramHeads;
            rowIdsForPosition(history, tokens, contextLen, contextLen + position, params, rowIds + rowOffset);
        }
    }

    if (nextTokenHistory != nullptr)
    {
        // After every read above, so that nextTokenHistory may alias tokenHistory.
        for (int32_t sequence = 0; sequence < batchSize; ++sequence)
        {
            for (int32_t slot = 0; slot < contextLen; ++slot)
            {
                nextTokenHistory[sequence * contextLen + slot] = historyAt(
                    tokenHistory + sequence * contextLen, tokenIds + sequence * seqLen, contextLen, seqLen + slot);
            }
        }
    }
}
} // namespace rt
} // namespace trt_edgellm
