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
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/state/sharedResources.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class AsyncHostStagingFence
{
public:
    AsyncHostStagingFence() = default;
    AsyncHostStagingFence(AsyncHostStagingFence const&) = delete;
    AsyncHostStagingFence& operator=(AsyncHostStagingFence const&) = delete;
    AsyncHostStagingFence(AsyncHostStagingFence&& other) noexcept;
    AsyncHostStagingFence& operator=(AsyncHostStagingFence&& other) noexcept;
    ~AsyncHostStagingFence();

    void wait();
    void record(cudaStream_t stream);

private:
    void release() noexcept;

    cudaEvent_t mEvent{nullptr};
    bool mPending{false};
};

//! Persistent copies of the prefill-time input embeddings and engine
//! hidden_states output, used by streaming consumers that run concurrently
//! with the base model's decode loop.
//!
//! The base model's `inputsEmbeds` and `outputHiddenStates` tensors are
//! reshaped to `{B, 1, H}` and overwritten by every decode step. This struct
//! retains the `{B, prefillLen, H}` view as it stood at the end of prefill,
//! so consumers reading these buffers do not race with decode writes.
struct StreamingPrefillBuffers
{
    Tensor inputEmbeds;        //!< Prefill-time layer-0 input embeddings.
    Tensor engineHiddenStates; //!< Prefill-time engine hidden_states output.

    //! Allocate on first call (sized to the worst case `{maxBatch, maxSeq, hiddenSize}`),
    //! reshape to the current request's `{batch, prefillLen, hiddenSize}`, and copy
    //! from the live PipelineIO buffers on `stream`. Subsequent calls reuse the same
    //! allocation. Must be invoked after prefill and before the first decode step on
    //! the same stream so the copies precede any overwrite of `outputHiddenStates`.
    void populateFromPrefill(Tensor const& liveInputEmbeds, Tensor const& liveEngineHiddenStates, int32_t batch,
        int32_t prefillLen, int32_t hiddenSize, int32_t maxBatch, int32_t maxSeq, cudaStream_t stream);
};

//! All tensors flowing through the inference pipeline.
//! POINTER STABILITY INVARIANT: After buildTensorMap() is called, this struct
//! must not be moved, and deepstackEmbeds must not be resized. TensorMap holds
//! Tensor* pointers into these members — any reallocation invalidates them.
struct PipelineIO
{
    // Always present
    Tensor inputsEmbeds;
    Tensor outputLogits;
    Tensor selectTokenIndices;
    Tensor phaseIsEncoder;      //!< DiffusionGemma invocation phase selector, [1] INT32
    Tensor contextMaskSelector; //!< DiffusionGemma context-mask selector, [0] or [batch] INT32
    Tensor contextLengths;      //!< GPU
    Tensor hostContextLengths;  //!< CPU (pinned, [maxBatch] INT32)
    Tensor
        hostSelectTokenIndices; //!< CPU (pinned, [maxBatch, 1] INT64) — pairs with selectTokenIndices for H2D staging
    Tensor hostPhaseIsEncoder;  //!< CPU pinned scalar paired with phaseIsEncoder

    // Ragged ABI metadata. Device and pinned-host tensors are allocated at
    // maximum capacity before TensorMap construction and only reshaped while
    // executing a step, preserving every bound address across CUDA Graphs.
    Tensor positions;
    Tensor queryStartOffsets;
    Tensor queryLengths;
    Tensor pastLengths;
    Tensor attentionSequenceLengths;
    Tensor stateIndices;
    Tensor logitsIndices;
    Tensor raggedKVPageTable;
    Tensor raggedSwaKVPageTable;
    Tensor hostPositions;
    Tensor hostQueryStartOffsets;
    Tensor hostQueryLengths;
    Tensor hostPastLengths;
    Tensor hostAttentionSequenceLengths;
    Tensor hostStateIndices;
    Tensor hostLogitsIndices;
    //! Gemma4 Unified block IDs, [physical_tokens] INT32; empty for other models.
    Tensor visionBlockIds;

    // Multimodal (resize deepstackEmbeds BEFORE buildTensorMap, never after)
    std::vector<Tensor> deepstackEmbeds;
    Tensor mropeCosSin;       //!< Resident-slot MRoPE source cache.
    Tensor mropeActiveCosSin; //!< Active-row MRoPE preprocessing scratch.
    Tensor raggedRopeCosSin;
    Tensor raggedRopeCosSinSliding;
    Tensor raggedRopeCosSinFull;

    // Spec decode
    Tensor baseHiddenStates;
    Tensor draftHiddenStatesIn;
    Tensor draftHiddenStatesOut;

    //! Engine accept-layer output: the Qwen3-Omni Talker's feed on both
    //! pipelines. Bound to `hidden_states` on the vanilla path, and to
    //! `accept_hidden_states` on a SpecDecode base, where `hidden_states` is
    //! instead the draft's post-norm feed in `baseHiddenStates`. Binding the
    //! latter here would hand the Talker a post-final-norm tensor — degraded
    //! audio rather than an error.
    Tensor outputHiddenStates;

    //! Per-request copies of `inputsEmbeds` / `outputHiddenStates` that
    //! streaming consumers (e.g. the Qwen3-Omni Talker) read while the base
    //! model's decode loop overwrites the live buffers. Populated by
    //! `LLMInferenceRuntime` only when streaming output is enabled for the
    //! request; otherwise the buffers stay empty (no allocation cost).
    StreamingPrefillBuffers streamingPrefill;

    // SpecDecode engine-bound tensors (empty for vanilla LLM runtime).
    //! Packed proposal attention mask, [physical_tokens, divUp(proposalSize, 32)] INT32.
    //! Written by proposal/verify input preparation kernels; consumed by the base and draft
    //! engines via the `kAttentionMask` binding.
    Tensor packedAttentionMask;
    //! SpecDecode position IDs, [physical_tokens] INT32.
    //! Written by proposal/verify input preparation kernels; consumed by the base and draft
    //! engines via the `kAttentionPosId` binding.
    Tensor specDecodePositionIds;
    //! Shape-only execution-phase carrier. Plugins branch on its extent and never read its payload.
    Tensor executionPhaseMarker;
    //! Shape-only [N_context] INT32 carrier. Its payload is never initialized or read.
    Tensor contextSequenceCountCarrier;
    //! Shape-only runtime skip-softmax override carrier (data never read); bound
    //! with shape [S] where S comes from LLMEngineConfig::skipSoftmaxScaleOverride.
    Tensor skipSoftmaxScale;
    //! DDTree parent node ids, [physical_tokens] INT32. Runtime-owned
    //! metadata for tree attention and hybrid state plugin bindings.
    Tensor specTreeParentIds;
    //! DDTree depth per node, [physical_tokens] INT32. Runtime-owned
    //! metadata for tree attention and hybrid state plugin bindings.
    Tensor specTreeDepths;

    //! Build PipelineIO for the vanilla single-engine LLM runtime
    //! (basic I/O tensors, deepstack embeds, MRope cos/sin cache).
    static PipelineIO createForLLM(LLMEngineConfig const& cfg, cudaStream_t stream);

    //! Build PipelineIO for a two-engine speculative-decoding runtime
    //! (basic I/O, hidden states, deepstack embeds, MRope cos/sin cache).
    //!
    //! `hasAcceptHiddenOutput` must say whether the base engine actually exposes
    //! the `accept_hidden_states` binding: allocating regardless would make
    //! `outputHiddenStates.isEmpty()` stop meaning "nothing will fill this", and
    //! the Talker would be handed uninitialised memory instead of failing.
    //! `hasTreeMetadataInputs` similarly reflects the base or draft engine ABI.
    //! Some engines retain these optional bindings even when the selected
    //! runtime policy uses a linear proposal.
    static PipelineIO createForSpecDecode(DeploymentConfig const& bundle, int32_t maxRuntimeBatchSize,
        cudaStream_t stream, bool hasAcceptHiddenOutput, bool hasTreeMetadataInputs);

    //! Stage and asynchronously upload one already-validated ragged batch.
    void uploadRaggedMetadata(RaggedExecutionBatch const& batch, cudaStream_t stream);

    //! Stage and asynchronously upload active sequence-to-resident-slot indices.
    //! A null residentRefs uses identity mapping for runtimes without resident-slot indirection.
    void uploadStateIndices(std::vector<ResidentRef> const* residentRefs, int32_t numSequences, cudaStream_t stream);

    //! Protect reusable pinned step metadata and ragged token snapshots before CPU reuse.
    void waitForStepHostStaging();
    void recordStepHostUploads(cudaStream_t stream);

private:
    AsyncHostStagingFence mRaggedMetadataUploadFence;
    AsyncHostStagingFence mStepHostUploadFence;
};

void allocateBasicIO(PipelineIO& io, int32_t maxBatch, int32_t vocabSize);

void allocateDeepstackEmbeds(PipelineIO& io, int32_t numFeatures, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize,
    nvinfer1::DataType dtype);

void allocateSpecDecodeHiddenStates(PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t baseHiddenDim,
    int32_t draftHiddenDim, nvinfer1::DataType dtype, bool allocateDraftHiddenStates);

void allocateMRope(
    PipelineIO& io, int32_t residentRows, int32_t activeRows, int32_t maxKVCacheCapacity, int32_t rotaryDim);

void prepareTextOnlyMRope(PipelineIO& io, LLMEngineConfig const& cfg, int32_t activeRows, cudaStream_t stream);

//! Gather the current ragged step's token-aligned RoPE inputs after metadata upload.
void prepareRaggedRope(PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t physicalTokens,
    int32_t numSequences, cudaStream_t stream);

//! Publish active-row MRoPE preprocessing output into its resident-slot rows.
void scatterActiveMRopeToResident(
    PipelineIO& io, RaggedExecutionBatch const& batch, LLMEngineConfig const& cfg, cudaStream_t stream);

//! Gather resident page-table rows into the stable active-step binding after state-index upload.
void prepareRaggedKVPageTable(PipelineIO& io, KVPageTable const& pageTable, int32_t numSequences, cudaStream_t stream);

//! Gather bounded-SWA resident rows into its stable active-step binding after state-index upload.
void prepareRaggedSwaKVPageTable(
    PipelineIO& io, KVPageTable const& pageTable, int32_t numSequences, cudaStream_t stream);

//! Upload one validated execution batch and derive every token-major engine binding backed by persistent resources.
//! MRoPE resident rows must already have been initialized or published by the caller.
void prepareRaggedExecutionBindings(PipelineIO& io, SharedResources& resources, LLMEngineConfig const& cfg,
    RaggedExecutionBatch const& batch, int32_t kvCacheIndex, cudaStream_t stream);

//! Populate a TensorMap from PipelineIO + SharedResources for engine binding.
//!
//! This is the critical glue function that wires all allocated tensors into the
//! name-to-pointer map consumed by TensorRegistry::bindAll().
//!
//! @param map          Output map to populate.
//! @param io           Pipeline I/O tensors.
//! @param res          Shared resources (KV caches, RoPE pool, LoRA, zero buffer).
//! @param cfg          Engine configuration.
//! @param kvCacheIndex Index into res.cacheManagers for the target engine.
void buildTensorMap(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex);

//! Populate a TensorMap for a DiffusionGemma unified-backbone engine.
//!
//! This keeps DiffusionGemma-only phase/canvas bindings out of the default
//! autoregressive tensor-map path while still sharing the common KV/RoPE/state
//! bindings with standard LLM engines.
void buildTensorMapForDiffusionBackbone(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex);

//! Rebind DiffusionGemma unified-backbone tensors for the current denoise,
//! prefill, or commit step. Self-conditioning feedback is hidden-size state
//! ping-ponged by the block-diffusion decoder.
void bindDiffusionUnifiedBackboneTensors(TensorMap& map, PipelineIO& io, Tensor& logits, Tensor& canvasIds,
    Tensor& prevSelfConditioningEmbeds, Tensor& nextSelfConditioningEmbeds, Tensor& selfConditioningTemperature);

//! Rebind only the DiffusionGemma self-conditioning tensors that ping-pong
//! between denoise steps. Static unified-backbone bindings are established by
//! bindDiffusionUnifiedBackboneTensors().
void bindDiffusionUnifiedBackboneSelfConditioningTensors(
    TensorMap& map, Tensor& prevSelfConditioningEmbeds, Tensor& nextSelfConditioningEmbeds);

//! Populate a TensorMap for a SpecDecode draft engine. Delegates to `buildTensorMap`
//! with `kvCacheIndex=1` for the common bindings, then patches in draft-engine-
//! specific bindings (base/draft hidden states in+out, packed proposal attention
//! mask, proposal position IDs).
//!
//! Preconditions: `io` must have been constructed via `PipelineIO::createForSpecDecode`
//! for an EAGLE/MTP-style draft path where draftHiddenStatesIn/Out are
//! populated alongside baseHiddenStates, packedAttentionMask, and
//! specDecodePositionIds. DFlash uses its own draft TensorMap.
//!
//! @param map Output map for the draft engine's bindings.
//! @param io  Pipeline I/O (must be the SpecDecode-flavoured one).
//! @param res Shared resources.
//! @param cfg Draft engine configuration.
void buildTensorMapForSpecDecodeDraft(TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg);

//! Populate a TensorMap for a Gemma4 MTP assistant draft engine.
//!
//! Unlike EAGLE/MTP draft engines, Gemma4 assistant engines do not own a draft
//! KV cache. Their `past_key_values_*` bindings are zero-copy aliases to the
//! base target KV cache selected by `draftCfg.gemma4MTPKVSharingMap`.
void buildTensorMapForGemma4MTPDraft(
    TensorMap& map, PipelineIO& io, SharedResources& res, DeploymentConfig const& bundle);

} // namespace rt
} // namespace trt_edgellm
