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

#include "common/specDecodeLimits.h"
#include "common/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Maximum node count supported by the DDTree builder: the shared speculative-decoding tree budget.
constexpr int32_t kDDTreeMaxVerifySize{kMAX_SPEC_VERIFY_TREE_TOKENS};

//! Maximum DFlash candidateTopK supported by the current DDTree top-k kernel.
constexpr int32_t kDDTreeMaxCandidateTopK{8};

//! Tensor inputs consumed by ddtreeBuild().
struct DDTreeBuildInputs
{
    //! Token-major [batch * draftBlockSize, vocabSize] draft logits. Internal callers with an existing
    //! rectangular compatibility view may pass [batch, draftBlockSize, vocabSize].
    rt::Tensor const& draftLogits;
    rt::Tensor const& rootTokenIds; //!< [batch] last accepted token ids.
    rt::Tensor const& baseLengths;  //!< [batch] committed base lengths before verification.

    //! Optional [vocabSize] reduced-to-full draft vocab mapping. When present, DDTree selects candidates in the
    //! reduced draft vocab space but emits full-vocab token ids for base verify and accept.
    rt::Tensor const* draftVocabMappingTable{nullptr};

    //! Optional [batch, draftBlockSize - 1] acceptance probabilities in [0, 1];
    //! depthConfidence[b][d - 1] scores depth d. Growth scores gain log(conf) along
    //! the path, steering the node budget away from low-confidence depths.
    rt::Tensor const* depthConfidence{nullptr};

    //! Survival floor in [0, 1): growth stops past depths whose confidence product
    //! falls below it. 0 (default) disables. Only meaningful with depthConfidence.
    float confidenceSurvivalThreshold{0.0F};
};

//! Tensor outputs produced by ddtreeBuild().
struct DDTreeBuildOutputs
{
    rt::Tensor& nodeTokenIds;       //!< [batch, verifyTokenCount] flattened tree token ids.
    rt::Tensor& nodeDepths;         //!< [batch, verifyTokenCount] flattened tree depths.
    rt::Tensor& parentIds;          //!< [batch, verifyTokenCount] parent node indices.
    rt::Tensor& nodeScores;         //!< [batch, verifyTokenCount] prefix log-prob scores.
    rt::Tensor& validCounts;        //!< [batch] valid node counts.
    rt::Tensor& verifyTokenIds;     //!< [batch, verifyTokenCount] base verify token ids.
    rt::Tensor& verifyPositionIds;  //!< [batch, verifyTokenCount] base verify position ids.
    rt::Tensor& packedAncestorMask; //!< [batch, verifyTokenCount, ceil(verifyTokenCount / 32)] packed mask.
    rt::Tensor& contextLengths;     //!< [batch] base context lengths for verify.
    rt::Tensor& selectTokenIndices; //!< [batch, verifyTokenCount] selected token indices.
};

//! Parameters for ddtreeBuild().
struct DDTreeBuildParams
{
    DDTreeBuildInputs inputs;           //!< Required input tensors.
    DDTreeBuildOutputs outputs;         //!< Required output tensors.
    int32_t candidateTopK;              //!< Per-depth candidateTopK from DFlash draftTopK.
    void* workspace;                    //!< Temporary workspace from getDDTreeBuildWorkspaceSize().
    size_t workspaceSize;               //!< Workspace size in bytes.
    cudaStream_t stream;                //!< CUDA stream for kernel launches.
    int32_t firstCandidateLogitsRow{1}; //!< Logits row used for depth-1 children.
};

//! Sorted child candidates per (proposal row, parent state), consumed by ddtreeAssemble().
//!
//! Row d lists the children of depth-d parents; the children land at depth d + 1. The root has state 0, and a
//! child inserted from slot k of list (b, d, m) carries state childStates[b, d, m, k] into row d + 1. Every
//! (b, d, m) list is sorted by non-increasing logProbs; a -inf logProb marks an empty slot and every later slot
//! of that list must also be -inf.
struct DDTreeCandidateTable
{
    rt::Tensor const& tokenIds;             //!< [batch, depthRows, stateCount, candidateTopK] INT32 token ids.
    rt::Tensor const& logProbs;             //!< [batch, depthRows, stateCount, candidateTopK] FP32.
    rt::Tensor const* childStates{nullptr}; //!< Same shape, INT32 in [0, stateCount); nullptr means state 0.
};

//! Parameters for ddtreeAssemble().
struct DDTreeAssembleParams
{
    DDTreeCandidateTable candidates;
    rt::Tensor const& rootTokenIds; //!< [batch] last accepted token ids.
    rt::Tensor const& baseLengths;  //!< [batch] committed base lengths before verification.
    //! Optional [batch, depthRows] acceptance probabilities; depthConfidence[b][d - 1] scores depth d.
    rt::Tensor const* depthConfidence{nullptr};
    float confidenceSurvivalThreshold{0.0F}; //!< Same semantics as DDTreeBuildInputs.
    DDTreeBuildOutputs outputs;
    cudaStream_t stream;
};

//! Parameters for ddtreeSelectLogitCandidates().
struct DDTreeLogitCandidateParams
{
    rt::Tensor const& draftLogits;                     //!< Same layouts as DDTreeBuildInputs::draftLogits.
    rt::Tensor const* draftVocabMappingTable{nullptr}; //!< Optional reduced-to-full vocab map.
    int32_t firstCandidateLogitsRow{1};                //!< Logits row used for depth-1 children.
    rt::Tensor& tokenIds;                              //!< [batch, draftBlockSize - 1, 1, candidateTopK] INT32.
    rt::Tensor& logProbs;                              //!< [batch, draftBlockSize - 1, 1, candidateTopK] FP32.
    cudaStream_t stream;
};

//! Returns temporary workspace size required by ddtreeBuild().
//!
//! The workspace stores candidate token ids and log probabilities for DFlash candidateTopK selection.
size_t getDDTreeBuildWorkspaceSize(
    int32_t batchSize, int32_t draftBlockSize, int32_t verifyTokenCount, int32_t vocabSize, int32_t candidateTopK);

//! Stage A for depth-indexed drafts: per-depth top-K over full-vocabulary log-softmax of draft logits.
//!
//! Row d of the table reads logits row firstCandidateLogitsRow + d. Log-probabilities are normalized over the
//! whole draft vocabulary; the optional vocab map is applied to the emitted token ids. stateCount is 1.
void ddtreeSelectLogitCandidates(DDTreeLogitCandidateParams const& params);

//! Stage B: best-first assembly of a prefix-closed tree from a candidate table.
//!
//! Each step appends the highest prefix score among the next unused slot of every in-tree parent, looked up in
//! list (row = parent depth, state = parent state). Ties break by parent index, slot, then token id. Outputs
//! follow the ddtreeBuild() contract; nodeScores rows are non-increasing along every root path, the root scores
//! 0, and padding nodes score -inf.
void ddtreeAssemble(DDTreeAssembleParams const& params);

//! Build a prefix-closed DDTree from one DFlash draft logits pass: ddtreeSelectLogitCandidates() followed by
//! ddtreeAssemble() with a single parent state per depth.
//!
//! The tree is flattened in score-prioritized order. Node 0 is the committed
//! root token. For each proposal depth, the builder first takes the top
//! `candidateTopK` tokens from the corresponding draft-logits row, then
//! repeatedly appends the highest-scoring available child whose parent is
//! already in the tree. This keeps the verify tree prefix-closed: every emitted
//! node can trace a valid path back to the root.
//! firstCandidateLogitsRow controls whether depth-1 children consume logits row
//! 0 (next-token block drafts) or row 1 (mask-position block drafts such as
//! DFlash/JetSpec, and MTP stacked logits where row 0 is a root placeholder).
//!
//! Example with candidateTopK = 2 and verifyTokenCount = 6:
//!
//!     node:        0     1     2     3     4     5
//!     depth:       0     1     1     2     2     3
//!     parent:     -1     0     0     1     2     3
//!
//! In this example, nodes 1 and 2 are the best depth-1 children of the root.
//! Nodes 3 and 4 extend different parents, while node 5 extends node 3. The
//! packed ancestor masks allow each verify node to attend only to the
//! tokens on its own root-to-node path.
//!
//! Inputs:
//!     draftLogits [GPU, Float]: token-major [batch * draftBlockSize, vocabSize], or an internal
//!         [batch, draftBlockSize, vocabSize] compatibility view.
//!     rootTokenIds [GPU, Int32]: last accepted token for each batch, [batch].
//!     baseLengths [GPU, Int32]: committed base length before verify, [batch].
//!     candidateTopK: DFlash DDTree candidateTopK, wired from draftTopK.
//!     firstCandidateLogitsRow: logits row used for depth-1 children. Depth D
//!         children consume row firstCandidateLogitsRow + D - 1.
//!     draftVocabMappingTable: optional reduced-to-full draft vocab mapping, [vocabSize].
//!     workspace: temporary storage with size from getDDTreeBuildWorkspaceSize().
//!
//! Outputs:
//!     nodeTokenIds [GPU, Int32]: flattened tree token ids, [batch, verifyTokenCount]. Node 0 is root.
//!     nodeDepths [GPU, Int32]: flattened tree depths, [batch, verifyTokenCount]. Root depth is 0.
//!     parentIds [GPU, Int32]: flattened parent node indices, [batch, verifyTokenCount]. Root and padding use -1.
//!     nodeScores [GPU, Float]: prefix log-prob scores, [batch, verifyTokenCount]. Padding uses -inf.
//!     validCounts [GPU, Int32]: valid node count per batch, [batch].
//!     verifyTokenIds [GPU, Int32]: base verify token ids, [batch, verifyTokenCount].
//!     verifyPositionIds [GPU, Int32]: base verify position ids, baseLengths + nodeDepths, [batch, verifyTokenCount].
//!     packedAncestorMask [GPU, Int32]: packed tree attention mask,
//!         [batch, verifyTokenCount, ceil(verifyTokenCount / 32)].
//!     contextLengths [GPU, Int32]: baseLengths + verifyTokenCount, [batch].
//!         Tree verification writes every flattened node into a contiguous KV suffix.
//!         Absolute token positions are still provided separately through positionIds.
//!     selectTokenIndices [GPU, Int64]: batch * verifyTokenCount + node, [batch, verifyTokenCount].
//!
//! The DDTree builder runs on GPU with fixed-size scratch. It does not provide a CPU runtime fallback.
void ddtreeBuild(DDTreeBuildParams const& params);

} // namespace kernel
} // namespace trt_edgellm
