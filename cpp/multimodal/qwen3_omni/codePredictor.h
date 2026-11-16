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

#include "common/executionPhase.h"
#include "common/tensor.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "sampler/sampling.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

/*!
 * @brief Publish one step's token-major layout to the engine's ragged bindings.
 *
 * Every execute() reads the offsets, past lengths and state indices this writes; they are not
 * step-invariant, so binding them once per prepare() leaves the attention kernels addressing
 * whatever the previous step left behind. The talker steps its own engine through this too.
 */
void prepareRaggedStep(rt::RaggedExecutionBatch& batch, rt::PipelineIO& io, rt::SharedResources& resources,
    rt::LLMEngineConfig const& config, std::vector<int32_t> const& queryLengths,
    std::vector<int32_t> const& pastLengths, int32_t batchSize, int32_t queryWidth, rt::ExecutionPhase phase,
    cudaStream_t stream);

/*!
 * @brief Residual codebook prediction, one engine behind the talker.
 *
 * One frame is one sequence: the engine runs a two-row prefill and then either
 * `numRvqLayers - 1` decode steps or, with speculation on, draft/verify rounds. The KV cache
 * holds a single frame, so the talker resets it at the start of every frame and no state
 * crosses a frame boundary.
 *
 * This owns its engine, its weights and every workspace it touches. Two tensors belong
 * to the talker's codec vocabulary and are only borrowed: the code_0 embedding table
 * and the per-frame codec-hidden buffer whose middle rows are filled here for the
 * talker's residual. Both arrive in Binding.
 *
 * The small_to_mtp projection needs the CuTe DSL GEMM module, which the talker loads before
 * preallocate() and unloads on destruction.
 */
class CodePredictor
{
public:
    /*!
     * @brief What the talker lends, checked once at load.
     *
     * Deployment-scoped, not per-request: the two tensors are borrowed and must stay
     * alive, with at least this capacity, for as long as frames are driven.
     */
    struct Binding
    {
        int32_t talkerHiddenSize{0};
        int32_t numCodesPerFrame{0};
        int32_t maxBatchSize{0};
        //! [codecVocabSize, talkerHiddenSize] table code_0 is looked up in. Which table
        //! that is depends on the checkpoint, which is why the talker picks it.
        rt::Tensor const* codeZeroEmbedTable{nullptr};
        //! [maxBatchSize, numCodesPerFrame, talkerHiddenSize]. Rows 1..numRvqLayers-1 are
        //! written here; rows 0 and numRvqLayers stay the talker's to fill.
        rt::Tensor* codecHiddens{nullptr};
    };

    //! One codec frame's inputs. Everything is borrowed for the duration of the call.
    struct FramePlan
    {
        int32_t activeBatchSize{0};
        //! Frames decoded since the batch's prefill, the same for every row; keys the offsets of the
        //! frame's non-speculative draws.
        int32_t frameIdx{-1};
        //! Pinned host INT32, at least [activeBatchSize]: the talker's code_0, uploaded asynchronously.
        rt::Tensor const* codeZero{nullptr};
        //! [activeBatchSize, talkerHiddenSize] FP16 device view of the talker's last hidden state.
        rt::Tensor const* talkerLastHidden{nullptr};
        //! Only temperature/topK/topP are read; batch and vocabulary are re-derived per call.
        SamplingParams const* sampling{nullptr};
    };

    //! Load the engine and claim every workspace. @p checkpointDir may be empty.
    //! @p specVerifySize is the verify window the deployment asked for; 0, or anything this
    //! engine cannot honour, leaves the autoregressive loop in place.
    //! Throws on failure, with a message saying what failed.
    void preallocate(std::string const& artifactDir, std::string const& checkpointDir, int32_t maxBatch,
        int32_t specVerifySize, cudaStream_t stream);

    //! Declare the talker's half of the contract, which this checks against the artifacts.
    void bind(Binding binding);

    size_t requiredContextMemory() const noexcept
    {
        return static_cast<size_t>(mContextMemoryBytes);
    }

    /*!
     * @brief Hand the engine an arena the talker owns, releasing the one claimed here.
     *
     * Legal only because both engines run on the talker's stream: TensorRT forbids giving
     * one context's device memory to a context executing concurrently.
     *
     * @return false when @p memory is smaller than requiredContextMemory().
     */
    bool setContextMemory(rt::Tensor& memory);

    //! Zero the KV lengths for @p batchSize sequences. One frame is one sequence.
    void resetKvCache(int32_t batchSize, cudaStream_t stream);

    /*!
     * @brief Generate codes 1..numRvqLayers for one codec frame.
     *
     * @param outputCodesPerBatch [activeBatchSize][numCodesPerFrame], code_0 first. Rows keep
     *                            their capacity across frames, so reuse the same vector.
     * @return false when an engine call fails; the talker must then abandon the frame.
     */
    bool runFrame(FramePlan const& plan, std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream);

    //! Capture one decode graph per batch size. Leaves the KV cache empty, ready for a frame.
    bool captureDecodeGraphs(cudaStream_t stream);

    //! Report and reset the speculation counters accumulated since the last call. No-op
    //! while speculation is off or no round has run.
    void logAndResetSpecStats();

    int32_t numRvqLayers() const noexcept
    {
        return mNumRvqLayers;
    }

    int32_t codebookSize() const noexcept
    {
        return mCodebookSize;
    }

    //! Residual codebook table @p layer, [codebookSize, talkerHiddenSize], 0-based over the
    //! numRvqLayers residual codebooks. Borrowed; the talker reads them for its own lookups.
    rt::Tensor const& codecEmbedTable(int32_t layer) const;

private:
    void loadEngine();
    void loadWeights();
    void allocateWorkspaces();

    //! Project a [batch, talkerHiddenSize] view through small_to_mtp into @p output, [batch, hiddenSize].
    //! Only for artifacts that carry the projection; without it the talker's width is already the code predictor's.
    void projectToCp(rt::Tensor const& source, rt::Tensor& output, int32_t batchSize, cudaStream_t stream);

    //! Two-row prefill scored through head 0; leaves the device head index at 0.
    bool executePrefillStep(
        rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    //! Bind the step-invariant decode tensors once per frame so every step hash-hits the graph.
    bool prepareDecodeBindings(int32_t activeBatchSize, cudaStream_t stream);

    // ---- speculative decoding over RVQ depth; all inert while mSpec.verifySize == 0 ----

    //! Decide the verify window this engine can actually serve and claim its workspace.
    void resolveSpeculation(int32_t requested);
    //! Score \p rows hidden rows, each against the lm_head its selector plane names.
    void applyRowLmHeads(
        __half const* hiddens, int32_t hiddenPlane, int32_t headPlane, int32_t rows, cudaStream_t stream);
    //! Publish this round's five row selectors in one upload.
    void uploadRowSelectors(int32_t activeBatchSize, int32_t const* hostM, int32_t n, cudaStream_t stream);
    //! Embed the committed code and \p n drafts, each through the table its depth names.
    void buildVerifyInput(int32_t activeBatchSize, int32_t n, cudaStream_t stream);
    //! One engine call over \p seqLen positions appended to each batch's committed context.
    bool executeVerifyStep(int32_t activeBatchSize, int32_t seqLen, cudaStream_t stream);
    //! Save the raw codec embeddings of freshly committed codes for the talker's residual.
    void storeCodecHiddensForAccepted(rt::Tensor const& codeIds, int32_t activeBatchSize, int32_t stride,
        int32_t const* hostM, int32_t offset, cudaStream_t stream, int32_t const* counts = nullptr);
    //! Draft/verify/accept rounds until every batch holds numRvqLayers codes.
    bool runSpecDecodeFrame(FramePlan const& plan, int32_t activeBatchSize,
        std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream);

    // ---- Deployment ----
    std::string mArtifactDir;
    std::string mCheckpointDir;
    cudaStream_t mStream{nullptr};
    int32_t mMaxBatch{1};
    Binding mBinding;
    bool mBound{false};

    // ---- Artifact shape ----
    int32_t mNumRvqLayers{0};     //!< Residual codebooks this engine drives.
    int32_t mNumCodesPerFrame{0}; //!< numRvqLayers + 1, code_0 included.
    int32_t mCodebookSize{0};     //!< Rows of one lm_head; the sampling vocabulary.
    int32_t mHiddenSize{0};
    //! Talker-side width the artifacts were exported for; the talker's own hidden size
    //! must equal it.
    int32_t mTalkerHiddenSize{0};

    // ---- Engine and per-step execution state ----
    LLMEngineConfig mEngineConfig;
    std::unique_ptr<EngineExecutor> mExec;
    std::unique_ptr<SharedResources> mResources;
    std::unique_ptr<PipelineIO> mPipelineIO;
    std::unique_ptr<StepPreparer> mStepPreparer;
    rt::RaggedExecutionBatch mRagged;
    //! Per-step ragged lengths, sized at maxBatch so a frame never allocates.
    std::vector<int32_t> mStepQueryLengths;
    std::vector<int32_t> mStepPastLengths;
    int32_t mPastLen{0};
    TensorMap mTensorMap;
    rt::Tensor mExecContextMemory;
    int64_t mContextMemoryBytes{0};

    // ---- Weights, read-only after preallocate() ----
    struct Weights
    {
        //! Stacked [numRvqLayers, codebookSize, hiddenSize]. The engine gathers the active head
        //! from the device index, which is what keeps the decode bindings step-invariant.
        rt::Tensor lmHeads;
        std::vector<rt::Tensor> codecEmbeddingTables;
        rt::Tensor smallToMtpWeight;
        rt::Tensor smallToMtpBias;
        bool useSmallToMtpProjection{false};
    };
    Weights mWeights;

    // ---- Per-frame buffers: prefill, autoregressive decode, and the verify pass's scoring ----
    struct FrameBuffers
    {
        rt::Tensor lmHeadIdx; //!< Device index of the active lm_head.
        rt::Tensor prefillInput;
        rt::Tensor codecIds;
        rt::Tensor rawCodecEmbed;
        rt::Tensor codecEmbed;
        rt::Tensor projectedHidden;
        rt::Tensor hiddenStates;
        rt::Tensor logits;
        rt::Tensor selectedIndices;
        rt::Tensor samplingWorkspace;
        rt::Tensor hostSelectedCodeIds;
        //! Pinned [numRvqLayers - 1, maxBatch]: step k writes row k, one sync drains the frame.
        rt::Tensor hostGenCodes;
        rt::Tensor hostReuseKVCacheLengths;
    };
    FrameBuffers mFrame;

    // ---- Speculation over RVQ depth; its buffers exist only when resolveSpeculation() keeps it on ----
    //! A device tensor and the pinned host tensor it is uploaded from.
    struct Upload
    {
        rt::Tensor device;
        rt::Tensor host;
    };

    struct Speculation
    {
        //! Verify window in use, and what the engine's generation profile admits in physical
        //! tokens. The token-major ABI carries the window in the row count, not a sequence axis.
        int32_t verifySize{0};
        int32_t genProfileMaxTokens{1};

        // Device buffers, at the widest shape the verify window admits, so a round reshapes down and never
        // allocates.
        //! [maxBatch, cpHidden]: per row, the hidden state that produced its last committed code, which every
        //! draft of the next round reads.
        rt::Tensor draftHidden;
        rt::Tensor verifyCodeIds;     //!< [maxBatch, K] committed code then drafts
        rt::Tensor verifyInput;       //!< [maxBatch * K, cpHidden]
        rt::Tensor verifyHidden;      //!< [maxBatch * K, cpHidden]
        rt::Tensor logitsFp16;        //!< [maxBatch * K, codebook] grouped-head output
        rt::Tensor draftProbs;        //!< [maxBatch * (K - 1), codebook]
        rt::Tensor targetProbs;       //!< [maxBatch * K, codebook]
        rt::Tensor draftTokenIds;     //!< [maxBatch, K - 1]
        rt::Tensor acceptedTokenIds;  //!< [maxBatch, K]
        rt::Tensor acceptLength;      //!< [maxBatch]
        rt::Tensor rawCodecEmbedRows; //!< [maxBatch * K, talkerHidden] pre-projection gather

        //! Five selector planes of stride maxBatch * K: draft hidden/head, verify hidden/head,
        //! verify codec table. One upload per round covers all of them.
        Upload rowIdx;
        //! Two planes of the same stride: destination row in the talker's codec-hidden buffer,
        //! and which RVQ table the code is embedded through.
        Upload storeIdx;
        Upload tablePtrs;       //!< One codec table pointer per RVQ depth, without the talker table
        Upload proposalLengths; //!< [maxBatch]

        // Pinned host state of a round: the accept results read back, and per-row bookkeeping.
        rt::Tensor hostAcceptLength;
        rt::Tensor hostAcceptedTokenIds;
        rt::Tensor hostM;       //!< [maxBatch] codes committed so far this frame, per batch
        rt::Tensor hostBaseLen; //!< [maxBatch] KV rows the verify pass appends onto

        // Sampling randomness and the verify binding.
        rt::Tensor uniformPool;
        int64_t uniformPoolSize{0};
        uint64_t randomOffset{0};
        //! Shape the verify tensors are currently bound at; rebinding changes the executor's
        //! binding hash, so an unchanged round must skip it to stay on its captured graph.
        //! Any other prepare() on this executor invalidates it back to -1.
        int32_t verifyBoundBatch{-1};
        int32_t verifyBoundSeqLen{-1};

        // Counters for logAndResetSpecStats().
        uint64_t rounds{0};
        uint64_t proposed{0};
        uint64_t codes{0};
    };
    Speculation mSpec;
};

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
