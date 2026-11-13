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
#include "runtime/config/llmEngineConfig.h"
#include "runtime/config/pleRuntimeConfig.h"
#include "runtime/embedding/pleRowStore.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/preprocess/ngramRowIds.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class PleIoBudget;

//! Runtime preprocessor for the Qwen4-Exp hashed n-gram embedding (PLE).
//!
//! Hashes each token's n-grams into table row IDs and turns them into a row map into the shared
//! embedding row pool. Row IDs depend on token IDs alone, so none of this needs a hidden state and
//! the whole mapping can happen before the engine runs; the lookup itself is a graph operation.
//!
//! The table is the largest tensor in the model -- 47.7 GiB in the released checkpoint -- while a
//! forward reads ``ngramHeads`` rows per token out of it. It therefore stays in the checkpoint and
//! reaches the device through PleRowStore, either resident or as a row cache over SSD.
//!
//! Unrelated to Gemma4EmbeddingPreprocessor despite the shared "PLE" name.
class Qwen4ExpNgramPreprocessor
{
public:
    Qwen4ExpNgramPreprocessor(LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen,
        TensorMap& tensorMap, cudaStream_t stream, PleRuntimeConfig const& pleConfig = {},
        std::filesystem::path const& checkpointDir = {}, std::shared_ptr<PleIoBudget> pleIoBudget = nullptr);
    ~Qwen4ExpNgramPreprocessor() noexcept;

    //! \brief Restore one slot's token history from the tokens already consumed for its request.
    //!
    //! The hash reaches back ``ngramContextLength`` tokens, so a chunk that does not start a
    //! sequence has to know what preceded it. That history is not separate state -- it is the tail
    //! of the tokens already consumed -- so re-deriving it here keeps a reused prefix and a
    //! chunked prefill correct without a checkpoint of its own.
    //!
    //! \param[in] residentSlot Resident state-pool row of the sequence, as in ``state_indices``
    //! \param[in] consumedTokens Tokens of this request that precede the current call, may be null
    //!                           when \p consumedCount is 0
    //! \param[in] consumedCount Number of such tokens; fewer than the context length pads with EOS,
    //!                          which is also what marks the start of a sequence
    void seedHistory(int32_t residentSlot, int32_t const* consumedTokens, int32_t consumedCount);

    //! \brief Start one slot's token history from scratch, as the beginning of a document.
    //! \param[in] residentSlot Resident state-pool row of the sequence
    void resetHistory(int32_t residentSlot);

    //! \brief Hash the step's tokens and upload the resulting row map.
    //!
    //! The entry-padded token layout is what lets one contiguous map serve the token-major
    //! binding: every sequence occupies \p queryWidth rows, so ``[numSequences, queryWidth, heads]``
    //! and ``[numSequences * queryWidth, heads]`` are the same bytes.
    //!
    //! \param[in] hostTokenIds [numSequences, queryWidth] INT32 token IDs in pinned host memory,
    //!                         right padded per sequence
    //! \param[in] hostQueryLengths [numSequences] INT32 valid token count per sequence, or null
    //!                             when every sequence contributes all \p queryWidth tokens
    //! \param[in] residentSlots [numSequences] INT32 state-pool row per sequence, in execution
    //!                          order; null means the identity mapping
    //! \param[in] numSequences Number of active sequences
    //! \param[in] queryWidth Padded token count per sequence
    //! \param[in] stream CUDA stream for the upload
    //! \param[in] hostTreeParents [numSequences, queryWidth] INT32 parent node per token (-1 for the
    //!                            root), or null for a chain. A tree hashes each node over its
    //!                            ancestors and leaves the history unchanged.
    void embed(int32_t const* hostTokenIds, int32_t const* hostQueryLengths, int32_t const* residentSlots,
        int32_t numSequences, int32_t queryWidth, cudaStream_t stream, int32_t const* hostTreeParents = nullptr);

    //! Reshape the bound row map for an execution or CUDA-graph capture shape.
    void reshapeOutput(int64_t physicalTokens);

    //! \brief The row map bound for the current step.
    //!
    //! Everything that produces it runs on the host and is bit-reproducible, unlike the engine's
    //! own output past the first MoE. That makes it the one probe that can answer "did chunking or
    //! padding change anything" by exact equality rather than by tolerance.
    //!
    //! \return The bound [physicalTokens, ngramHeads] INT32 tensor
    Tensor const& rowMap() const;

    //! Counters for the most recent step; meaningful in cached residency only.
    PleRowStore::Stats const& stats() const;

private:
    struct PooledResources;
    bool drainPooled() noexcept;

    //! Drop the previous step's row pins, once the device has finished reading them.
    void releaseLeases();

    //! This resident slot's ``ngramContextLength`` preceding tokens.
    int32_t* historyForSlot(int32_t residentSlot);

    //! Fold ``count`` newly consumed tokens into one resident slot's history.
    void advanceHistory(int32_t residentSlot, int32_t const* tokens, int32_t count);

    LLMEngineConfig mConfig{};
    std::vector<uint64_t> mLayerMultipliers{}; //!< Hash constants, carried in the engine config
    std::vector<int64_t> mHeadVocabSizes{};    //!< rather than re-derived, so the hash matches the
    std::vector<int64_t> mHeadOffsets{};       //!< values the table was trained against.
    NGramRowIdParams mParams{};

    int32_t mStatePoolRows{0}; //!< Resident slots the history is allocated for.
    //! [statePoolRows, ngramContextLength] preceding tokens, indexed the way the engine's
    //! convolution state is: by resident slot, so an eviction never has to renumber it.
    std::vector<int32_t> mTokenHistory{};
    std::vector<int32_t> mActiveHistory{};         //!< The step's histories gathered into execution order.
    std::vector<int32_t> mNodeHistory{};           //!< Tree steps: each node's preceding tokens.
    std::vector<std::vector<int64_t>> mRequests{}; //!< Per-sequence row IDs handed to the row store.
    std::vector<int64_t> mRowIds{};                //!< [maxTokens, ngramHeads] hash output.
    std::unique_ptr<PooledResources> mPooled;
    cudaStream_t mStream{};
};

} // namespace rt
} // namespace trt_edgellm
