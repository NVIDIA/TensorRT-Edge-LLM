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
namespace rt
{

//! Constants of one hashed n-gram embedding instance (Qwen3.8-Flash-Next PLE).
//!
//! The three arrays are checkpoint buffers, stored under
//! `...ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}`, read from
//! host memory.
struct NGramRowIdParams
{
    int32_t ngramSize{};                //!< Longest n-gram order, 3 for Qwen3.8-Flash-Next.
    int32_t headsPerNgram{};            //!< Hash heads per order, 8.
    int32_t ngramHeads{};               //!< (ngramSize - 1) * headsPerNgram, 16. Orders below 2 have no head.
    int32_t eosTokenId{};               //!< Doubles as the segment separator and as the out-of-segment filler.
    uint64_t const* layerMultipliers{}; //!< [ngramSize] odd multipliers, one per position in the n-gram.
    int64_t const* headVocabSizes{};    //!< [ngramHeads] pairwise-coprime primes, one modulus per head.
    int64_t const* headOffsets{};       //!< [ngramHeads] start row of each head's slice of the table.
};

//! \brief `torch.remainder` for the int64 value whose bit pattern is \p bits, i.e. the result takes
//! the sign of \p divisor and is non-negative for a positive one.
//!
//! Exposed because it is the semantic the row IDs are most likely to get wrong silently: C++ `%`
//! truncates toward zero and yields a negative remainder for a negative dividend, which would land
//! on a different table row instead of failing.
//!
//! \param[in] bits Two's-complement bit pattern of the hash value
//! \param[in] divisor Modulus, must be positive
int64_t ngramRemainder(uint64_t bits, int64_t divisor);

//! \brief Row IDs into the PLE n-gram table for one step.
//!
//! For each position the bigram hash mixes the token with its predecessor and the trigram hash also
//! mixes in the token before that, each multiplied by its position's multiplier and XOR-folded. Every
//! head then reduces that hash modulo its own prime and shifts it into its slice of the table.
//!
//! Lookback stops at an EOS: a position whose span back to the source token crosses one uses EOS in
//! its place, which is also what makes the two EOS of an initial \p tokenHistory the correct
//! sequence start.
//!
//! \param[in] tokenIds [batchSize, seqLen] INT32 token IDs, each in [0, unigram vocab size)
//! \param[in] tokenHistory [batchSize, ngramSize - 1] INT32 tokens preceding this call; fill with
//!                         eosTokenId at the start of a sequence
//! \param[out] rowIds [batchSize, seqLen, ngramHeads] INT64 row IDs into the n-gram table
//! \param[out] nextTokenHistory [batchSize, ngramSize - 1] history to pass to the next call, may be
//!                              null, and may alias \p tokenHistory
//! \param[in] batchSize Number of sequences
//! \param[in] seqLen Tokens per sequence in this call; may be shorter than the history
//! \param[in] params Hash constants, in host memory
void computeNgramRowIds(int32_t const* tokenIds, int32_t const* tokenHistory, int64_t* rowIds,
    int32_t* nextTokenHistory, int32_t batchSize, int32_t seqLen, NGramRowIdParams const& params);
} // namespace rt
} // namespace trt_edgellm
