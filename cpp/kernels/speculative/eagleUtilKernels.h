/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

// Disable clang-format to explicitly format the documentation.
// clang-format off

// The kernel will prepare required inputs to execute the eagle prefill step.
// Inputs:
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine, shape [batch].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output, shape [batch].
void prepareEaglePrefillInputs(rt::Tensor const& sequenceContextLengths,
    rt::Tensor& selectTokenIndices,  cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle draft proposal step.
// In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
// MISC inputs to execute the draft proposal step.
// Inputs:
//     draftTreeMask [GPU, Int8]: unpacked draft tree mask denote the relationship between the draft tree nodes.
//         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//     draftTreeLength [GPU, Int32]: Real length of the draft tree.
//     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedDraftTreeMask [GPU, Int32]: Packed tree mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of draft tree nodes among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
void prepareEagleDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
    rt::Tensor const& sequenceStartIndices, rt::Tensor& packedDraftTreeMask, rt::Tensor& tensorPositionIndices,
    rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths, cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle accept decode token step.
// Since we reuse the logic of tree attention, instead of draft tree mask, we will prepare casual mask and corresponding
// position indices of each accepted token.
// Inputs:
//     sequenceStartIndices [GPU, Int32]: The start indices of the first accepted token, shape [batch].
//     acceptedTokenNums [GPU, Int32]: Number of accepted tokens for each batch, shape [batch].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedTreeMask [GPU, Int32]: Packed casual tree mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of accepted tokens among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position (always the last one) to gather the hidden states and logits.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
void prepareEagleAcceptDecodeTokenInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor const& acceptedTokenNums, rt::Tensor& packedTreeMask,
    rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths,
    cudaStream_t stream);

// The kernel will prepare required inputs to execute the eagle draft proposal step.
// In detail, the kernel will prepare packed draft tree mask, compute token positional indices, and prepare other
// MISC inputs to execute the draft proposal step.
// Inputs:
//     baseTreeDecodingMask [GPU, Int8]: unpacked base tree decoding mask denotes the relationship between the base tree decoding nodes.
//         The input is padded with shape [batch, padded-draft-tree-size, padded-draft-tree-size] to ease implementation.
//     sequenceStartIndices [GPU, Int32]: The start indices of "top level" tree nodes.
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     packedBaseTreeDecodingMask [GPU, Int32]: Packed base tree decoding mask where each flag takes 1 bit.
//     tensorPositionIndices [GPU, Int32]: Positional indices of base tree decoding nodes among the sequence.
//     selectTokenIndices [GPU, Int64]: Denote the position to gather the hidden states and logits output.
//     sequenceContextLengths [GPU, Int32]: The sequence context lengths input fed into TRT engine.
void prepareEagleBaseTreeDecodingInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
    rt::Tensor& packedBaseTreeDecodingMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream);

// The kernel will commit KVCache and assemble the hidden state inplace according to the accepted indices and accept lengths.
// The hidden state will be updated inplace from [batch, verify-tree-size, hidden-dim] to [batch, max-accept-depth, hidden-dim].
// This is safe because max-accept-depth << verify-tree-size, so output positions never overwrite unread input data.
// Inputs:
//     acceptedIndices [GPU, Int32]: The accepted indices with shape [batch, max-depth].
//     acceptLengths [GPU, Int32]: The accept lengths with shape [batch].
//     kvCacheBuffer [GPU, Half]: The KVCache buffer with shape [num-layers, max-batch-size, 2, num-heads, max-seq-len, hidden-size-per-head].
//     kvCacheLengths [GPU, Int32]: The KVCache lengths with shape [batch].
//     hiddenState [GPU, Half]: The hidden state with shape [batch, num-tokens, base-hidden-dim].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     kvCacheBuffer [GPU, Half]: The updated KVCache buffer.
//     hiddenState [GPU, Half]: The in-place updated hidden state by the selected tokens.
void eagleBaseCommitKVCacheAndAssembleHiddenState(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, rt::Tensor& kvCacheBuffer, rt::Tensor& hiddenState, cudaStream_t stream);

// The kernel will initialize the draft table for a new round of drafting. 
// Draft token ids will be translated towards full vocab size. During eagle spec-decode draft tree construction,
// we will build multiple full data tables to record the complete description of a draft tree. The full table will contain
// the root node so the full-table size is (1 + draft-topK + total-draft-round x draft-topK x draft-topK).
// Inputs:
//     selectedIndices [GPU, Int32]: Selected indices from logits, shape [batch, draftTopK].
//     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK].
//     rootTokens [GPU, Int32]: Committed tokens selected by base model to act as the root token of the draft tree [batch].
//     vocabMappingTable [GPU, Int32]: The mapping table from draft vocab token to full vocab token, shape [draft-vocab-size].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree. [batch, full-table-size]
//     draftScoreFullTable [GPU, Float]: Table to store the cumulative token scores of the whole tree. [batch, full-table-size]
//     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//         within this table as its predecessor. [batch, full-table-size]
void initializeDraftTreeTables(rt::Tensor const& selectedIndices, rt::Tensor const& logProbs, rt::Tensor const& rootTokens,
    rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdFullTable, rt::Tensor& draftScoreFullTable,
    rt::Tensor& draftParentFullTable, int32_t const draftTopK, cudaStream_t stream);

// The kernel will assemble the draft tree inputs for the first round of drafting. By current design, the draft tree will be
// padded and incrementally constructed with rounds of drafting, padded-draft-tree-size == total-draft-round x draft-topK.
// We simply place the corresponding ids and hidden states in the input tensor.
// Inputs:
//     draftIdFullTable [GPU, Int32]: Full draft table that stored the token ids within the full vocab size. [batch, full-table-size]
//     draftHiddenStatesOutput [GPU, Half]: Hidden states output from the draft model. [batch, draft-hidden-size]
// Outputs:
//     inputIds [GPU, Int32]: Input ids to the draft model. shape [batch, padded-draft-tree-size]
//     draftModelHiddenStates [GPU, Half]: Hidden states input for next round of drafting. 
//         shape [batch, padded-draft-tree-size, draft-hidden-size]
//     draftTreeLength [GPU, Int32]: Length of the draft tree. shape [batch]
//     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, padded-draft-tree-size, padded-draft-tree-size] 
//         The mask will be used to mask the hidden states input.
void assembleInitialDraftTreeInput(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftHiddenStatesOutput,
    rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput, rt::Tensor& draftTreeLength, rt::Tensor& draftTreeMask,
    int32_t const draftTopK, cudaStream_t stream);

// The kernel will assemble the intermediate data prior to the first round of drafting.
// Inputs:
//     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK].
//     stream: The CUDA stream to execute the kernel.
// Outputs:
//     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK].
//     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
void assembleInitialIntermediateData(rt::Tensor const& logProbs, rt::Tensor& intermediateParents,
    rt::Tensor& intermediateScores, int32_t const draftTopK, cudaStream_t stream);

// The kernel will assemble the draft tree inputs After the first round of drafting, for simplicity, we will build the
// input in a padded manner, so we will extend one more level of the inputs based on previous round of drafting.
// Here, padded-draft-tree-size == total-draft-round x draft-topK.
// Inputs:
//     draftIdTable [GPU, Int32]: DraftIds from last round of drafting. shape [batch, draft-topK, draft-topK]
//     draftHiddenOutput [GPU, Half]: Hidden states output from last round of drafting. [batch * draft-topK, draft-hidden-size]
//     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK].
// Outputs:
//     inputIds [GPU, Int32]: Input ids to the draft model. shape [batch, padded-draft-tree-size]
//     draftModelHiddenStates [GPU, Half]: Hidden states input for next round of drafting. 
//         shape [batch, padded-draft-tree-size, draft-hidden-size]
//     draftTreeLength [GPU, Int32]: Actual length of the draft tree. shape [batch]
//     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, padded-draft-tree-size, padded-draft-tree-size]
void assembleDraftTreeInput(rt::Tensor const& draftIdTable, rt::Tensor const& draftHiddenOutput,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput, rt::Tensor& draftTreeLength,
    rt::Tensor& draftTreeMask, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

// The kernel will assemble the intermediate data for next round of drafting. In the eagle3 draft tree construction,
// we build a table record cumulative log probabilities and parent of each "node" in the draft tree. In drafting step,
// draftTopK nodes will be selected from draftTopK x draftTopK candidates to build next level of draft tree.
// This function will help save the cuLogProbs and indices of the selected nodes to record meta construction info.
// Inputs:
//     cuLogProbs [GPU, Float]: Cumulative log probabilities of the selected tokens, shape [batch, draftTopK].
//     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK].
//     round: The round of drafting.
// Outputs:
//     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
//     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK]. 
void assembleIntermediateData(rt::Tensor const& cuLogProbs, rt::Tensor const& selectedIndices,
    rt::Tensor& intermediateScores, rt::Tensor& intermediateParents, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

// The kernel will compute the cumulative scores and translate the token ids towards full vocab size.
// Inputs:
//     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, draftTopK, draftTopK].
//     logProbs [GPU, Float]: Log probabilities of the selected tokens, shape [batch, draftTopK, draftTopK].
//     intermediateScores [GPU, Float]: Intermediate scores of the selected tokens, shape [batch, draftTopK].
//     vocabMappingTable [GPU, Int32]: The mapping table from draft vocab token to full vocab token, shape [draft-vocab-size].
// Outputs:
//     draftIdTable [GPU, Int32]: Store the translated token ids. shape [batch, draft-topK, draft-topK]
//     draftScoreTable [GPU, Float]: Cumulative scores of the selected tokens, shape [batch, draftTopK, draftTopK].
void computeCuScoresAndTranslateToken(rt::Tensor const& selectedIndices, rt::Tensor const& logProbs,
    rt::Tensor const& intermediateScores, rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdTable,
    rt::Tensor& draftScoreTable, int32_t const draftTopK, cudaStream_t stream);

// The kernel will update the draft tree full tables with the new ids and scores.
// Inputs:
//     draftIdTable [GPU, Int32]: Store the translated token ids. shape [batch, draft-topK, draft-topK]
//     draftScoreTable [GPU, Float]: Cumulative scores of the selected tokens, shape [batch, draftTopK, draftTopK].
//     intermediateParents [GPU, Int32]: Intermediate parents of the selected tokens, shape [batch, draftTopK].
// Outputs:
//     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree.
//     draftScoreFullTable [GPU, Float]: Table to store the cumulative token scores of the whole tree.
//     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//         within this table as its predecessor.
void updateDraftTreeFullTables(rt::Tensor const& draftIdTable, rt::Tensor const& draftScoreTable, 
    rt::Tensor const& intermediateParents, rt::Tensor& draftIdFullTable, rt::Tensor& draftScoreFullTable,
    rt::Tensor& draftParentFullTable, int32_t const draftTopK, int32_t const round, cudaStream_t stream);

// The kernel will construct the draft tree for base model verification.
// Inputs:
//     draftIdFullTable [GPU, Int32]: Table to store the token ids of the whole tree.
//     draftParentFullTable [GPU, Int32]: Table to store the token parents of the whole tree. Each entry will point to a location
//         within this table as its predecessor.
//     selectedIndices [GPU, Int32]: Selected indices from top logits, shape [batch, verify-tree-size].
// Outputs:
//     inputIds [GPU, Int32]: Input ids to the base model. shape [batch, verify-tree-size]
//     draftTreeMask [GPU, Int8]: Draft tree mask. shape [batch, verify-tree-size, verify-tree-size]
void constructVerificationDraftTree(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftParentFullTable,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftTreeMask, cudaStream_t stream);

// clang-format on

} // namespace kernel
} // namespace trt_edgellm