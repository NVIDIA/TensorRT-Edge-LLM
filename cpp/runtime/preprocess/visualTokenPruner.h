/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "runtime/exec/scheduledStep.h"
#include "runtime/state/pipelineIO.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Visual-token pruning configuration (runtime-side; nothing is required in the exported
//! engine config — the prune operates on runtime buffers only).
struct VisualPrunerConfig
{
    bool enabled{false};
    //! Pruning algorithm name, resolved through the pruner registry.
    //! Built-in: "dart" (default, duplication-aware; paper: "Stop Looking for Important Tokens
    //! in Multimodal Language Models: Duplication Matters More").
    std::string algorithm{"dart"};
    //! Fraction of visual tokens to remove (0.25 = keep 75%).
    float reductionRatio{0.25F};
    //! Skip pruning when the request has fewer visual tokens than this (accuracy and
    //! break-even guard: tiny images gain nothing from pruning).
    int32_t minVisualTokens{16};

    // --- DART-specific parameters (ignored by other pruners) ---
    int32_t pivotImageTokens{4};
    int32_t pivotTextTokens{4};
};

//! One contiguous run of visual tokens (one image, or one video-frame block) with its
//! per-image retention quota. Pruning within each span independently — rather than over one
//! global candidate pool — guarantees every image keeps its proportional share of tokens
//! (at least one), so a low-information image can never be starved by the others.
struct ImageSpan
{
    int32_t begin{0};        //!< First sequence position of the span (inclusive)
    int32_t end{0};          //!< One past the last sequence position of the span (exclusive)
    int32_t targetTokens{0}; //!< Visual tokens to retain from this span (1 <= target <= end - begin)
};

//! Guard-checked, modality-partitioned view of one batch-1 prefill request, prepared by
//! VisualTokenPruner::pruneForPrefill and handed to the algorithm hook.
struct PruneRequest
{
    //! Non-owning [origLen, hiddenSize] FP16 GPU view of the assembled input embeddings.
    Tensor const* embeds{nullptr};
    //! Positions of visual tokens in the sequence (ascending, non-empty).
    std::vector<int32_t> const* imagePositions{nullptr};
    //! Positions of all non-visual tokens in the sequence (ascending).
    std::vector<int32_t> const* textPositions{nullptr};
    //! Contiguous visual spans (one per image) with per-span retention quotas.
    std::vector<ImageSpan> const* imageSpans{nullptr};
    //! Total number of visual tokens to retain — the sum of the per-span quotas
    //! (1 <= targetImageTokens < imagePositions->size()).
    int32_t targetImageTokens{0};
    //! Unpruned prefill length (== imagePositions->size() + textPositions->size()).
    int32_t origLen{0};
};

//! Abstract prefill-time visual-token pruner.
//!
//! The non-virtual pruneForPrefill() / pruneBatchForPrefill() own everything algorithms must
//! not diverge on: the enablement guards (minVisualTokens, target-is-a-reduction), modality
//! partitioning, and the target computation. Every subclass that shortens a request must finish
//! by calling compactToKeepList(), which records the ordered slot-local subset and owns the buffer compaction:
//! all text tokens are always kept; kept tokens retain their original absolute RoPE positions;
//! decode continues at the position after the unpruned sequence (matching the HF DART
//! reference).
//!
//! Batching: prune() is always invoked with a single-slot view (PruneRequest.embeds is that
//! slot's contiguous [len, hidden] plane; all positions are slot-local), so algorithms are
//! batch-agnostic. In the batched flow compactToKeepList() records the slot's keep list
//! instead of compacting immediately; once every slot is selected, the base repacks all
//! batch planes to the new (pruned) row pitch in one pass. Consequently batched pruning
//! requires the algorithm to route its result through compactToKeepList(). Direct rewrites and
//! token merging are unsupported by the MR1 protocol.
//!
//! Instances are created through createVisualTokenPruner() and reused across requests, so
//! per-request device work buffers should be preallocated in the constructor. The caller is
//! responsible for the runtime gates (fresh KV cache, mRoPE engine, ...) and
//! for shrinking the context lengths to the returned pruned lengths.
class VisualTokenPruner
{
public:
    virtual ~VisualTokenPruner() = default;

    //! Algorithm name (matches the registry key).
    virtual char const* name() const noexcept = 0;

    //! Prune the assembled prefill inputs for a batch-1 request.
    //!
    //! May synchronize `stream` (algorithm-dependent); on return the compaction kernels are
    //! enqueued on `stream` and the PipelineIO tensors are reshaped to the pruned length.
    //!
    //! \param hostTokenIds The request's expanded token ids (image placeholders already
    //!                     inserted); visual positions are `tokenId == imageTokenId`.
    //! \param io Pipeline buffers; `inputsEmbeds` must currently be [1, origLen, hiddenSize].
    //! \param origLen The unpruned prefill length (== hostTokenIds.size()).
    //! \param stream CUDA stream all device work runs on.
    //! \return The pruned length P (< origLen), or origLen when pruning is skipped
    //!         (no/too-few visual tokens, or the target keep count is not a reduction).
    int32_t pruneForPrefill(
        std::vector<int32_t> const& hostTokenIds, PipelineIO& io, int32_t origLen, cudaStream_t stream);

    //! Prune the assembled prefill inputs for a batch of requests. The batch size is taken
    //! from `io.inputsEmbeds` shape [batch, maxLen, hiddenSize].
    //!
    //! Selects per slot (slots without enough visual tokens are left unpruned), then repacks
    //! every batch plane of `inputsEmbeds` / `deepstackEmbeds` to the new maximum length and
    //! gathers each pruned slot's mRoPE rows. `effectiveLens` is updated in place to the
    //! per-slot pruned lengths and `prunedTokensOut` receives the per-slot removed counts.
    //! Returns `maxLen` unchanged when nothing was pruned, or when any slot's token ids don't
    //! cover its effective length (chunked continuation — modality partitioning is impossible).
    //!
    //! \param hostTokenIds Per-slot expanded token ids (size >= batch; slot i must satisfy
    //!                     hostTokenIds[i].size() == effectiveLens[i]).
    //! \param io Pipeline buffers; `inputsEmbeds` must currently be [batch, maxLen, hiddenSize].
    //! \param effectiveLens Per-slot prefill lengths (size >= batch); updated in place.
    //! \param maxLen Current padded prefill length (== max of effectiveLens).
    //! \param prunedTokensOut Per-slot number of removed tokens (resized to batch).
    //! \param stream CUDA stream all device work runs on.
    //! \return The new padded prefill length (== max of the updated effectiveLens).
    int32_t pruneBatchForPrefill(std::vector<std::vector<int32_t>> const& hostTokenIds, PipelineIO& io,
        std::vector<int32_t>& effectiveLens, int32_t maxLen, std::vector<int32_t>& prunedTokensOut,
        cudaStream_t stream);

    //! Compact the auxiliary prompt inputs that downstream consumers use to re-embed the
    //! prompt — required when the pruned request continues into a speculative-decoding draft
    //! prefill, which re-embeds host token ids and re-inserts the raw visual feature rows.
    //! (Deepstack features are not compacted: no draft strategy consumes them.)
    //!
    //! Must be called right after a pruning pass on the same request. Per slot (using the keep
    //! lists recorded by that pass): compacts `hostTokenIds[i]` in place, and gathers the kept
    //! visual feature rows into a pruner-owned buffer, rebinding the reference.
    //! Feature rows are indexed by the running image-token count over the packed batch grid,
    //! so the compacted grid's k-th image token is served by the original ordinal of the k-th
    //! kept one. This assumes ordinal 0 is the batch's first image token, i.e. embedding runs
    //! with zero multimodal base offsets — guaranteed by the fresh-KV-cache gate on pruning
    //! (prefix reuse would start the running count at a nonzero base offset). No-op when
    //! nothing was pruned.
    //!
    //! \param hostTokenIds Per-slot expanded token ids (compacted in place).
    //! \param batch Number of active slots.
    //! \param prunedTokens Per-slot removed counts from the pruning pass (validated against
    //!                     the recorded keep lists).
    //! \param visualFeatures Raw visual feature rows, one row per image token in the request
    //!                       ([totalImageTokens, dim] — Qwen-VL packing); rebound on return.
    //! \param stream CUDA stream the gathers run on.
    void compactAuxiliaryInputs(std::vector<std::vector<int32_t>>& hostTokenIds, int32_t batch,
        std::vector<int32_t> const& prunedTokens, OptionalInputTensor& visualFeatures, cudaStream_t stream);

    //! Preallocate the compactAuxiliaryInputs() buffers to their upper bound (feature rows are
    //! bounded by maxBatchSize x maxSupportedInputLength), so no allocation happens on the
    //! prefill path. Call once at setup when the deployment will use auxiliary compaction
    //! (spec decode); without this call the buffers grow lazily on first use instead — the
    //! feature plane is too large to always reserve for deployments that never need it.
    void preallocateAuxiliaryBuffers();

    std::vector<int32_t> const& keepStartOffsets() const noexcept
    {
        return mKeepStartOffsets;
    }

    std::vector<int32_t> const& concatenatedKeepIndices() const noexcept
    {
        return mConcatenatedKeepIndices;
    }

    VisualPrunerConfig const& config() const noexcept
    {
        return mConfig;
    }

protected:
    //! Preallocates the compaction buffers shared by every algorithm.
    //! \throws std::runtime_error on invalid config.
    VisualTokenPruner(VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig);

    //! The algorithm hook. Implementations that shorten a request must delegate buffer updates
    //! to compactToKeepList(). Return the pruned length, or req.origLen to skip pruning this request.
    virtual int32_t prune(PruneRequest const& req, PipelineIO& io, cudaStream_t stream) = 0;

    //! Shared compaction for subset-selection algorithms: keep all text tokens plus
    //! `retainedImageIndices` (any order, no duplicates, subset of *req.imagePositions),
    //! gather-compact inputsEmbeds / every deepstackEmbeds plane / the mRoPE cos-sin rows,
    //! and reshape to the pruned length. Returns the pruned length (or req.origLen when the
    //! retained set is not an actual reduction).
    int32_t compactToKeepList(
        PipelineIO& io, std::vector<int32_t> const& retainedImageIndices, PruneRequest const& req, cudaStream_t stream);

private:
    //! Guards + modality partitioning + algorithm invocation for one slot. `embedsView` is the
    //! slot's contiguous [origLen, hiddenSize] plane; positions handed to prune() are slot-local.
    int32_t selectForSlot(std::vector<int32_t> const& hostTokenIds, Tensor const& embedsView, PipelineIO& io,
        int32_t origLen, cudaStream_t stream);

    //! Repack every batch plane of inputsEmbeds / deepstackEmbeds from row pitch `oldMaxLen`
    //! to `newMaxLen` (dropping each pruned slot's removed rows) and gather pruned slots'
    //! mRoPE rows in place. Consumes mSlotKeepLists.
    void executeBatchCompaction(PipelineIO& io, std::vector<int32_t> const& oldLens,
        std::vector<int32_t> const& newLens, int32_t oldMaxLen, int32_t newMaxLen, cudaStream_t stream);
    void publishKeepMap(std::vector<int32_t> const& lengths, int32_t batch);

    VisualPrunerConfig mConfig;
    int32_t mImageTokenId{-1};
    int32_t mHiddenSize{0};
    int32_t mRotaryDim{0};
    int32_t mMaxKVCacheCapacity{0};
    int32_t mMaxBatchSize{1};
    int32_t mMaxInputLength{0};

    //! Batched-flow state: while set, compactToKeepList() validates and records the current
    //! slot's keep list into mSlotKeepLists instead of compacting immediately. All batched-flow
    //! vectors are sized/reserved once in the constructor so the prefill hot path never
    //! allocates: the outer mSlotKeepLists is fixed at mMaxBatchSize and the inner buffers
    //! (bounded by maxSupportedInputLength) are cleared and reused across requests.
    bool mDeferCompaction{false};
    int32_t mCurrentSlot{0};
    std::vector<std::vector<int32_t>> mSlotKeepLists;
    std::vector<int32_t> mKeepStartOffsets;
    std::vector<int32_t> mConcatenatedKeepIndices;
    std::vector<int32_t> mOldLens; //!< per-request scratch (reused)
    std::vector<int32_t> mNewLens; //!< per-request scratch (reused)

    //! compactAuxiliaryInputs state. Sized up front by preallocateAuxiliaryBuffers() on
    //! deployments that use auxiliary compaction; otherwise grown lazily on first use (the
    //! feature planes are too large to always reserve for deployments that never need them).
    std::vector<int32_t> mFeatureKeepHost; //!< global kept feature ordinals (reused)
    Tensor mFeatureIdxDevice;              //!< device mirror of mFeatureKeepHost
    Tensor mFeatureIdxPinned;              //!< pinned staging for the H2D upload
    Tensor mCompactVisualFeatures;         //!< owned compacted visual feature rows

    std::vector<int32_t> mImagePositions;  //!< per-request scratch (reused)
    std::vector<int32_t> mTextPositions;   //!< per-request scratch (reused)
    std::vector<ImageSpan> mImageSpans;    //!< per-request scratch (reused)
    std::vector<int32_t> mKeepIndicesHost; //!< final sorted keep list (text + retained images)
    Tensor mKeepIdxDevice;                 //!< [maxBatchSize * maxKVCacheCapacity] INT32, one region per slot
    Tensor mKeepIdxHost;                   //!< pinned mirror (per-slot regions avoid H2D reuse races)

    //! Gather scratch (large enough for one [maxInputLen, hiddenSize] FP16 plane and for the
    //! [maxKVCacheCapacity, rotaryDim] FP32 rope plane).
    Tensor mGatherScratch;
};

//! Factory signature for VisualTokenPruner implementations.
using VisualPrunerFactory
    = std::function<std::unique_ptr<VisualTokenPruner>(VisualPrunerConfig const&, LLMEngineConfig const&)>;

//! Register a pruning algorithm under `name` (case-sensitive). The built-in "dart" is
//! registered automatically; call this to plug in a custom algorithm before constructing
//! the runtime. Re-registering a name replaces the previous factory.
void registerVisualPruner(std::string const& name, VisualPrunerFactory factory);

//! Instantiate the pruner named by `config.algorithm`.
//! \throws std::runtime_error if the name is not registered or the config is invalid.
std::unique_ptr<VisualTokenPruner> createVisualTokenPruner(
    VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig);

//! Names of all registered pruning algorithms (for CLI help / validation).
std::vector<std::string> registeredVisualPrunerNames();

} // namespace rt
} // namespace trt_edgellm
