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
#include "kernels/speculative/batchEvictKernels.h" // KVLayerInfo
#include <NvInferRuntime.h>                        // nvinfer1::DataType
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kEagleMaxAcceptedPathLength{32};

// Disable clang-format to explicitly format the documentation.
// clang-format off

//! The kernel will prepare required inputs to execute the eagle prefill step.
//! Inputs:
//!     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine, shape [batch].
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output, shape [batch].
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or datatypes are incorrect
void prepareEaglePrefillInputs(rt::Tensor const& sequenceContextLengths,
    rt::Tensor& selectTokenIndices,  cudaStream_t stream);

//! The kernel will prepare required inputs to execute the eagle draft proposal step.
//! In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
//! MISC inputs to execute the draft proposal step.
//! Inputs:
//!     draftTreeMask [GPU, Int8]: unpacked draft tree mask denote the relationship between the draft tree nodes.
//!         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//!     draftTreeLength [GPU, Int32]: Real length of the draft tree.
//!     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     packedDraftTreeMask [GPU, Int32]: Packed tree mask where each flag takes 1 bit.
//!     tensorPositionIndices [GPU, Int32]: Positional indices of draft tree nodes among the sequence.
//!     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//!     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void prepareEagleDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
    rt::Tensor const& sequenceStartIndices, rt::Tensor& packedDraftTreeMask, rt::Tensor& tensorPositionIndices,
    rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths, cudaStream_t stream);

//! The kernel will prepare required inputs to execute the eagle accept decode token step.
//! Since we reuse the logic of tree attention, instead of draft tree mask, we will prepare casual mask and corresponding
//! position indices of each accepted token.
//! Inputs:
//!     sequenceStartIndices [GPU, Int32]: The start indices of the first accepted token, shape [batch].
//!     acceptedTokenNums [GPU, Int32]: Number of accepted tokens for each batch, shape [batch].
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     packedTreeMask [GPU, Int32]: Packed casual tree mask where each flag takes 1 bit.
//!     tensorPositionIndices [GPU, Int32]: Positional indices of accepted tokens among the sequence.
//!     selectTokenIndices [GPU, Int64]: Denote the position (always the last one) to gather the hidden states and logits.
//!     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void prepareEagleAcceptDecodeTokenInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor const& acceptedTokenNums, rt::Tensor& packedTreeMask,
    rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths,
    cudaStream_t stream);

//! The kernel will prepare required inputs to execute the eagle draft proposal step.
//! In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
//! MISC inputs to execute the draft proposal step.
//! Inputs:
//!     baseTreeDecodingMask [GPU, Int8]: unpacked base tree decoding mask denotes the relationship between the base tree decoding nodes.
//!         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//!     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     packedBaseTreeDecodingMask [GPU, Int32]: Packed base tree decoding mask where each flag takes 1 bit.
//!     tensorPositionIndices [GPU, Int32]: Positional indices of base tree decoding nodes among the sequence.
//!     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//!     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void prepareEagleBaseTreeDecodingInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
    rt::Tensor& packedBaseTreeDecodingMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream);

//! Per-head-dim-group batched commit of accepted tokens into the KVCache.
//!
//! One launch covers every layer in a single head-dim group. Each layer's storage
//! is read through `deviceLayerInfos[i]`, which carries the per-layer device pointer,
//! `numKVHeads`, and `maxSeqLen`. Layers with fewer KV heads than the group's
//! `maxKVHeads` early-exit on the extra grid columns.
//!
//! Callers with hybrid (heterogeneous head-dim) cache layouts should invoke this once
//! per `HybridCacheManager::KVHeadDimGroupView`. Uniform models invoke it once total.
//!
//! Inputs:
//!     acceptedIndices [GPU, Int32]: Accepted indices, shape [batch, max-depth].
//!     acceptLengths [GPU, Int32]: Accept lengths, shape [batch].
//!     kvCacheLengths [GPU, Int32]: KVCache lengths, shape [batch].
//!     deviceLayerInfos: Device-resident array of `KVLayerInfo` (size == numLayers) with each
//!         entry's `data` pointing to a per-layer paged pool
//!         [2, numPages, kTOKENS_PER_PAGE, numKVHeads_i, headDim].
//!     numLayers: Number of KV layers in this head-dim group.
//!     headDim: Head dimension shared by all layers in this group (currently 64 or 128).
//!     maxKVHeads: Largest `numKVHeads` across the group's layers (used for grid sizing).
//!     activeBatchSize: Number of active sequences in the batch.
//!     maxDepth: max-depth of acceptedIndices (acceptedIndices.shape[1]).
//!     kvCacheType: Storage dtype of the per-layer buffers (kHALF or kFP8).
//!     stream: CUDA stream to execute the kernel.
//!     pageTable: Required device page table [batch, 2, maxPagesPerSeq], with K rows followed by
//!         V rows and V page id equal to K page id + numPages. An unmapped or out-of-plane source
//!         resolves to zeros; an unmapped or out-of-plane destination skips the write.
//!     numPages: Number of physical pages in each KV plane.
//!     maxPagesPerSeq: Row stride of `pageTable`.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void eagleBaseCommitKVCache(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, KVLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t headDim,
    int32_t maxKVHeads, int32_t activeBatchSize, int32_t maxDepth, nvinfer1::DataType kvCacheType,
    cudaStream_t stream, int32_t const* pageTable, int32_t numPages, int32_t maxPagesPerSeq);

//! In-place compact the hidden-state buffer to keep only the accepted tokens.
//!
//! Updates `hiddenState` inplace from [batch, verify-tree-size, hidden-dim] to
//! [batch, max-accept-depth, hidden-dim]. Safe because max-accept-depth << verify-tree-size,
//! so output positions never overwrite unread input data. Layer-agnostic.
//!
//! Inputs:
//!     acceptedIndices [GPU, Int32]: Accepted indices, shape [batch, max-depth].
//!     acceptLengths [GPU, Int32]: Accept lengths, shape [batch].
//!     hiddenState [GPU, Half]: Hidden state, shape [batch, num-tokens, base-hidden-dim].
//!     stream: CUDA stream to execute the kernel.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void eagleBaseAssembleHiddenState(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor& hiddenState, cudaStream_t stream);

//! The kernel will initialize the draft table for a new round of drafting.
//! First-level draft logits are translated to target-vocabulary token ids. During eagle spec-decode draft tree
//! construction, we build multiple full data tables to record the complete description of a draft tree. The full table will contain
//! the root node so the full-table size is (1 + draft-topK + total-draft-round x draft-topK x draft-topK).
//! Inputs:
//!     selectedIndices [GPU, Int32]: Draft-vocabulary selected token ids, shape [batch, draftTopK].
//!     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK].
//!     rootTokens [GPU, Int32]: Committed tokens selected by base model to act as the root token of the draft tree [batch].
//!     vocabMappingTable [GPU, Int32]: Draft-token to target-token offset table, shape [draft-vocab-size].
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree. [batch, full-table-size]
//!     draftScoreFullTable [GPU, Float]: Table to store the cumulative token scores of the whole tree. [batch, full-table-size]
//!     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//!         within this table as its predecessor. [batch, full-table-size]
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void initializeDraftTreeTables(rt::Tensor const& selectedIndices, rt::Tensor const& logProbs, rt::Tensor const& rootTokens,
    rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdFullTable, rt::Tensor& draftScoreFullTable,
    rt::Tensor& draftParentFullTable, int32_t const draftTopK, cudaStream_t stream);

//! The kernel will assemble the draft tree inputs for the first round of drafting. By current design, the draft tree will be
//! padded and incrementally constructed with rounds of drafting, padded-draft-tree-size == total-draft-round x draft-topK.
//! We simply place the corresponding ids and hidden states in the input tensor.
//! Inputs:
//!     draftIdFullTable [GPU, Int32]: Full draft table that stored the token ids within the full vocab size. [batch, full-table-size]
//!     draftHiddenStatesOutput [GPU, Half]: Hidden states output from the draft model. [batch, draft-hidden-size]
//! Outputs:
//!     inputIds [GPU, Int32]: Input ids to the draft model. shape [batch, padded-draft-tree-size]
//!     draftModelHiddenStates [GPU, Half]: Hidden states input for next round of drafting. 
//!         shape [batch, padded-draft-tree-size, draft-hidden-size]
//!     draftTreeLength [GPU, Int32]: Length of the draft tree. shape [batch]
//!     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, padded-draft-tree-size, padded-draft-tree-size] 
//!         The mask will be used to mask the hidden states input.
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void assembleInitialDraftTreeInput(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftHiddenStatesOutput,
    rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput, rt::Tensor& draftTreeLength, rt::Tensor& draftTreeMask,
    int32_t const draftTopK, cudaStream_t stream);

//! The kernel will assemble the intermediate data prior to the first round of drafting.
//! Inputs:
//!     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK].
//!     stream: The CUDA stream to execute the kernel.
//! Outputs:
//!     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK].
//!     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if datatypes are invalid
void assembleInitialIntermediateData(rt::Tensor const& logProbs, rt::Tensor& intermediateParents,
    rt::Tensor& intermediateScores, int32_t const draftTopK, cudaStream_t stream);

//! The kernel will assemble the draft tree inputs After the first round of drafting, for simplicity, we will build the
//! input in a padded manner, so we will extend one more level of the inputs based on previous round of drafting.
//! Here, padded-draft-tree-size == total-draft-round x draft-topK.
//! Inputs:
//!     draftIdTable [GPU, Int32]: DraftIds from last round of drafting. shape [batch, draft-topK, draft-topK]
//!     draftHiddenOutput [GPU, Half]: Hidden states output from last round of drafting. [batch * draft-topK, draft-hidden-size]
//!     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK].
//! Outputs:
//!     inputIds [GPU, Int32]: Input ids to the draft model. shape [batch, padded-draft-tree-size]
//!     draftModelHiddenStates [GPU, Half]: Hidden states input for next round of drafting. 
//!         shape [batch, padded-draft-tree-size, draft-hidden-size]
//!     draftTreeLength [GPU, Int32]: Actual length of the draft tree. shape [batch]
//!     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, padded-draft-tree-size, padded-draft-tree-size]
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if data types or shapes are invalid
void assembleDraftTreeInput(rt::Tensor const& draftIdTable, rt::Tensor const& draftHiddenOutput,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput, rt::Tensor& draftTreeLength,
    rt::Tensor& draftTreeMask, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

//! The kernel will assemble the intermediate data for next round of drafting. In the eagle3 draft tree construction,
//! we build a table record cumulative log probabilities and parent of each "node" in the draft tree. In drafting step,
//! draftTopK nodes will be selected from draftTopK x draftTopK candidates to build next level of draft tree.
//! This function will help save the cuLogProbs and indices of the selected nodes to record meta construction info.
//! Inputs:
//!     cuLogProbs [GPU, Float]: Cumulative log probabilities of the selected tokens, shape [batch, draftTopK].
//!     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK].
//!     round: The round of drafting.
//! Outputs:
//!     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
//!     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK]. 
//!
//! @throws std::runtime_error if tensors are not located on the GPU, or if data types or shapes are invalid, or if a CUDA error occurs
void assembleIntermediateData(rt::Tensor const& cuLogProbs, rt::Tensor const& selectedIndices,
    rt::Tensor& intermediateScores, rt::Tensor& intermediateParents, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

//! The kernel will compute the cumulative scores and translate the token ids towards full vocab size.
//! Inputs:
//!     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK, draftTopK].
//!     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK, draftTopK].
//!     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
//!     vocabMappingTable [GPU, Int32]: The mapping table from draft vocab token to full vocab token, shape [draft-vocab-size].
//! Outputs:
//!     draftIdTable [GPU, Int32]: Store the translated token ids. shape [batch, draft-topK, draft-topK]
//!     draftScoreTable [GPU, Float]: Cumulative scores of the selected tokens, shape [batch, draftTopK, draftTopK].
//!
//! @throws std::runtime_error if tensors not located on GPU, or tensor datatype or shape is invalid
void computeCuScoresAndTranslateToken(rt::Tensor const& selectedIndices, rt::Tensor const& logProbs,
    rt::Tensor const& intermediateScores, rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdTable,
    rt::Tensor& draftScoreTable, int32_t const draftTopK, cudaStream_t stream);

//! The kernel will update the draft tree full tables with the new ids and scores.
//! Inputs:
//!     draftIdTable [GPU, Int32]: Store the translated token ids. shape [batch, draft-topK, draft-topK]
//!     draftScoreTable [GPU, Float]: Cumulative scores of the selected tokens, shape [batch, draftTopK, draftTopK].
//!     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK].
//! Outputs:
//!     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree.
//!     draftScoreFullTable [GPU, Float]: Table to store the cumulative token scores of the whole tree.
//!     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//!         within this table as its predecessor.
//!
//! @throws std::runtime_error if tensors not located on GPU, or tensor datatype or shape is invalid
void updateDraftTreeFullTables(rt::Tensor const& draftIdTable, rt::Tensor const& draftScoreTable, 
    rt::Tensor const& intermediateParents, rt::Tensor& draftIdFullTable, rt::Tensor& draftScoreFullTable,
    rt::Tensor& draftParentFullTable, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

//! The kernel will construct the draft tree for base model verification.
//! Inputs:
//!     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree.
//!     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//!         within this table as its predecessor.
//!     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, verify-tree-size].
//! Outputs:
//!     inputIds [GPU, Int32]: Input ids to the base model. shape [batch, verify-tree-size]
//!     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, verify-tree-size, verify-tree-size]
//!
//! @throws std::runtime_error if tensors not located on GPU, or tensor datatype or shape is invalid
void constructVerificationDraftTree(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftParentFullTable,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftTreeMask, cudaStream_t stream);

// clang-format on

} // namespace kernel
} // namespace trt_edgellm
