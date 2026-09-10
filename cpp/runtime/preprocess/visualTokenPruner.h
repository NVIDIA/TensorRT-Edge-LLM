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

//! Abstract prefill-time visual-token pruner (batch 1).
//!
//! The non-virtual pruneForPrefill() owns everything algorithms must not diverge on: the
//! enablement guards (minVisualTokens, target-is-a-reduction), modality partitioning, and the
//! target computation. Subclasses implement prune() — with full freedom over what "pruning"
//! means (subset selection, token merging, per-image quotas, ...) — and typically finish by
//! calling compactToKeepList(), which owns the invariant-laden buffer compaction: all text
//! tokens are always kept; kept tokens retain their original absolute RoPE positions; decode
//! continues at the position after the unpruned sequence (matching the HF DART reference).
//!
//! Instances are created through createVisualTokenPruner() and reused across requests, so
//! per-request device work buffers should be preallocated in the constructor. The caller is
//! responsible for the runtime gates (fresh KV cache, mRoPE engine, no spec decode, ...) and
//! for shrinking the context lengths to the returned pruned length.
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

    VisualPrunerConfig const& config() const noexcept
    {
        return mConfig;
    }

protected:
    //! Preallocates the compaction buffers shared by every algorithm.
    //! \throws std::runtime_error on invalid config.
    VisualTokenPruner(VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig);

    //! The algorithm hook. Implementations own the buffer update: either delegate to
    //! compactToKeepList() (subset-selection algorithms) or perform a custom rewrite of the
    //! PipelineIO buffers (e.g. token merging), upholding the compaction invariants described
    //! on the class. Return the pruned length, or req.origLen to skip pruning this request.
    virtual int32_t prune(PruneRequest const& req, PipelineIO& io, cudaStream_t stream) = 0;

    //! Shared compaction for subset-selection algorithms: keep all text tokens plus
    //! `retainedImageIndices` (any order, no duplicates, subset of *req.imagePositions),
    //! gather-compact inputsEmbeds / every deepstackEmbeds plane / the mRoPE cos-sin rows,
    //! and reshape to the pruned length. Returns the pruned length (or req.origLen when the
    //! retained set is not an actual reduction).
    int32_t compactToKeepList(
        PipelineIO& io, std::vector<int32_t> const& retainedImageIndices, PruneRequest const& req, cudaStream_t stream);

private:
    VisualPrunerConfig mConfig;
    int32_t mImageTokenId{-1};
    int32_t mHiddenSize{0};
    int32_t mRotaryDim{0};
    int32_t mMaxKVCacheCapacity{0};

    std::vector<int32_t> mImagePositions;  //!< per-request scratch (reused)
    std::vector<int32_t> mTextPositions;   //!< per-request scratch (reused)
    std::vector<ImageSpan> mImageSpans;    //!< per-request scratch (reused)
    std::vector<int32_t> mKeepIndicesHost; //!< final sorted keep list (text + retained images)
    Tensor mKeepIdxDevice;                 //!< [maxKVCacheCapacity] INT32
    Tensor mKeepIdxHost;                   //!< pinned mirror

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
