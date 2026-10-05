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

#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/residentSlotPool.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class KVPageTable;

//! @brief Per-request debug dumper for 4-layer numeric validation.
//!
//! When both environment variables below are set, the LLM runtime dumps, for
//! every inference round (round 0 = prefill, round r = decode step r), the
//! last-token logits, the per-layer combined KV cache (only the layers named
//! in the env var), the per-sequence valid lengths, and the per-round generated
//! token ids. Everything is buffered across rounds and written as a single
//! safetensors file at request end so the comparison tool can reconcile it
//! against the PyTorch golden.
//!
//!   EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS  number of leading decoder layers k
//!                                       -> dumps layers 0..k-1.
//!   EDGELLM_DUMP_LOGITS_KVCACHE_DIR     output directory for the dump file.
//!
//! The two variables are XOR-coupled: setting exactly one is an error.
//!
//! Per-layer tensors are gathered from resident slots into execution order. The comparison tool
//! slices each sequence to its valid length in PyTorch using the dumped context_lengths.
//!
//! Safetensors layout (single file, all rounds):
//!   round_{r}.logits               [activeBatch, vocab]                          (native dtype)
//!   round_{r}.layer_{i}.kv         [activeBatch, 2, kvHeads, maxSeqLen, headDim] (native dtype)
//!                                  dim 1: 0 = key, 1 = value
//!   round_{r}.context_lengths      [activeBatch]                                 int32
//!   round_{r}.generated_token_ids  [activeBatch]                                 int32
//!
//! Scope: the base model under vanilla or speculative decoding. Hooked from
//! ``runBaseModelPrefill`` (round 0), ``VanillaDecoder::decodeStep``, and -- via
//! ``decoder_utils::dumpSpecRound`` -- a speculative decoder's verification step. A speculative
//! round commits a variable number of tokens per sequence, so its ``context_lengths`` differ
//! between rows of the same round; the comparison tool pairs each row with the golden round of
//! the same length rather than by round index.
//!
//! Optionally also drives teacher-forcing: when ``EDGELLM_FORCE_TOKENS_FILE`` is set the
//! dumper overrides each step's sampled token with the golden's (see applyForcedTokens()),
//! so the run follows the golden token-for-token. Only ever active alongside a dump.
class LayerDebugger
{
public:
    //! @brief Build a dumper from the environment, or return nullptr when disabled.
    //! @return A dumper if both env vars are set; nullptr if neither is set.
    //! @throws std::runtime_error if exactly one of the two env vars is set (XOR
    //!         violation) or the layer spec is empty / malformed.
    static std::unique_ptr<LayerDebugger> fromEnv();

    //! @brief Accumulate one round's tensors into the in-memory buffer.
    //!
    //! Synchronises @p stream first, so the KV cache and logits are final.
    //! @param cacheManager       Base-model KV cache manager.
    //! @param pageTable          Full-capacity base-model KV page table.
    //! @param swaPageTable       Independent sparse page table for reduced SWA layers, or nullptr.
    //! @param logits             Device logits tensor [activeBatch, vocab].
    //! @param validLengths       Per-sequence valid KV/sequence length this round.
    //! @param originalIndices    Execution row -> original request row mapping used for reporting.
    //! @param residentRefs       Execution row -> persistent resident slot mapping used to gather state.
    //! @param generatedTokenIds  Host int32 [activeBatch] tokens sampled this round
    //!                           (may be nullptr to skip).
    //! @param activeBatchSize    Number of active sequences this round.
    //! @param stream             CUDA stream.
    void dumpRound(HybridCacheManager& cacheManager, KVPageTable const& pageTable, KVPageTable const* swaPageTable,
        Tensor const& logits, std::vector<int32_t> const& validLengths, std::vector<int32_t> const& originalIndices,
        std::vector<ResidentRef> const& residentRefs, int32_t const* generatedTokenIds, int32_t activeBatchSize,
        cudaStream_t stream);

    //! @brief Record how much of each sequence was restored from the context cache instead of
    //! executed. Call once from prefill, before the first dumpRound().
    //!
    //! Indexed by *original* request row, which is why it survives batch compaction: prefill runs
    //! before any sequence can finish, so there the active slot and the original row coincide.
    //!
    //! A request that reuses a cached prefix only executes the suffix after it, so the runtime's
    //! token list counts fewer tokens than the cache actually holds. Every dumped
    //! ``context_lengths`` adds this back, which is what keeps the dump comparable to a golden
    //! that prefilled the whole sequence.
    void setReusedPrefixLengths(std::vector<int32_t> lengths);

    //! @brief Write all buffered rounds to a single safetensors file.
    //! @param stream CUDA stream (forwarded to the safetensors writer).
    void flush(cudaStream_t stream);

    //! @brief Teacher-forcing: overwrite each active sequence's sampled token with the forced
    //! one for this step. No-op unless ``EDGELLM_FORCE_TOKENS_FILE`` was set at construction.
    //!
    //! Call *after* ``dumpRound`` so the dump still records the model's own sampled token; the
    //! forced token (if any) is what the caller then commits. This decouples the numeric
    //! comparison from greedy argmax stability — a near-tie argmax flip no longer diverges the
    //! two sides, while the dump still surfaces where the runtime *would* have diverged.
    //! Sequences are addressed across the whole run, not per request: the force-tokens file
    //! holds one line per sequence in request order, so a run that issues several requests (the
    //! context-reuse validation sends one per shared-prefix prompt) still lines up with a golden
    //! that batched them all.
    //! @param genLengths      Per-sequence count of tokens generated so far (== the index to force).
    //! @param originalIndices ``context.batchIndexMapping``; see dumpRound().
    //! @param tokenIds        Host array [activeBatchSize] of sampled tokens, overwritten in place.
    //! @param activeBatchSize Number of active sequences.
    void applyForcedTokens(std::vector<int32_t> const& genLengths, std::vector<int32_t> const& originalIndices,
        int32_t* tokenIds, int32_t activeBatchSize);

    //! @brief True when teacher-forcing tokens were supplied.
    bool hasForcedTokens() const noexcept
    {
        return !mForcedTokens.empty();
    }

    //! @brief Teacher-forcing for a speculative round: trim the acceptance so the tokens this
    //! round commits are the golden's.
    //!
    //! A speculative round commits several tokens at once, so overwriting them the way
    //! applyForcedTokens() does would leave their KV entries describing the tokens the draft
    //! actually proposed. Only slots ``[0, acceptLength - 1)`` have a committed cache entry --
    //! the last accepted token is the bonus token, whose entry is written next round -- so on the
    //! first slot @p j that disagrees with the golden the acceptance is trimmed to ``j + 1``,
    //! which drops that slot's cache entry, and only then is the token replaced.
    //!
    //! Call *before* the KV-cache commit, unlike applyForcedTokens().
    //! @param genLengths       Per-sequence count of tokens generated so far.
    //! @param originalIndices  ``context.batchIndexMapping``; see dumpRound().
    //! @param acceptLengths    Host [activeBatchSize] accept lengths, trimmed in place.
    //! @param acceptedTokenIds Host [activeBatchSize, maxAcceptDepth] tokens, overwritten in place.
    //! @param ownTokens        Out: each sequence's own token at the slot that ends up last, i.e.
    //!                         what it would have committed there without forcing.
    //! @param activeBatchSize  Number of active sequences.
    //! @param maxAcceptDepth   Row stride of @p acceptedTokenIds.
    //! @return true if any sequence was trimmed (the caller must then push the arrays back).
    bool applyForcedAcceptance(std::vector<int32_t> const& genLengths, std::vector<int32_t> const& originalIndices,
        int32_t* acceptLengths, int32_t* acceptedTokenIds, std::vector<int32_t>& ownTokens, int32_t activeBatchSize,
        int32_t maxAcceptDepth);

private:
    LayerDebugger(std::set<int32_t> layers, std::string dir, std::vector<std::vector<int32_t>> forcedTokens);

    //! @brief Read per-sequence forced token ids from ``EDGELLM_FORCE_TOKENS_FILE`` (one line per
    //! sequence, whitespace-separated ids), or an empty vector when the env var is unset. Emits a
    //! warning when forcing is enabled, since it overrides the model's own sampled tokens.
    static std::vector<std::vector<int32_t>> readForcedTokensFromEnv();

    //! @brief Original request row for an active slot, via ``context.batchIndexMapping``.
    static int32_t originalRow(std::vector<int32_t> const& originalIndices, int32_t slot);

    //! @brief This request's first row in the force-tokens file, claimed on first use.
    //!
    //! First use is always prefill, where the active batch is still the full request, so the
    //! claim covers every one of its sequences even if some finish later.
    int32_t forcedRowBase(int32_t activeBatchSize);

    std::set<int32_t> mLayers;                       //!< Absolute decoder-layer indices to dump KV for.
    std::string mDir;                                //!< Output directory.
    int32_t mRequestIdx{0};                          //!< Per-process request index (unique filenames).
    int32_t mRound{0};                               //!< Next round index to assign.
    std::vector<Tensor> mTensors;                    //!< Accumulated tensors across rounds.
    std::vector<std::vector<int32_t>> mForcedTokens; //!< Teacher-forcing tokens (empty = disabled).
    std::vector<int32_t> mReusedPrefix;              //!< Context-cache prefix by original row (empty = none).
    int32_t mForcedRowBase{-1};                      //!< Force-tokens row of this request's first sequence.
};

} // namespace rt
} // namespace trt_edgellm
