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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Maximum node count supported by the DDTree builder.
constexpr int32_t kDDTreeMaxVerifySize{128};

//! Maximum DFlash candidateTopK supported by the current DDTree top-k kernel.
constexpr int32_t kDDTreeMaxCandidateTopK{8};

//! Tensor inputs consumed by ddtreeBuild().
struct DDTreeBuildInputs
{
    rt::Tensor const& draftLogits;  //!< [batch, dflashBlockSize, vocabSize] draft logits.
    rt::Tensor const& rootTokenIds; //!< [batch] last accepted token ids.
    rt::Tensor const& baseLengths;  //!< [batch] committed base lengths before verification.

    //! Optional [vocabSize] reduced-to-full draft vocab mapping. When present, DDTree selects candidates in the
    //! reduced draft vocab space but emits full-vocab token ids for base verify and accept.
    rt::Tensor const* draftVocabMappingTable{nullptr};

    //! Optional [batch, dflashBlockSize - 1] acceptance probabilities in [0, 1];
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
    rt::Tensor& nodeTokenIds;       //!< [batch, verifySize] flattened tree token ids.
    rt::Tensor& nodeDepths;         //!< [batch, verifySize] flattened tree depths.
    rt::Tensor& parentIds;          //!< [batch, verifySize] parent node indices.
    rt::Tensor& nodeScores;         //!< [batch, verifySize] prefix log-prob scores.
    rt::Tensor& validCounts;        //!< [batch] valid node counts.
    rt::Tensor& verifyTokenIds;     //!< [batch, verifySize] base verify token ids.
    rt::Tensor& verifyPositionIds;  //!< [batch, verifySize] base verify position ids.
    rt::Tensor& packedAncestorMask; //!< [batch, verifySize, ceil(verifySize / 32)] packed mask.
    rt::Tensor& ancestorMask;       //!< [batch, verifySize, verifySize] unpacked accept mask.
    rt::Tensor& contextLengths;     //!< [batch] base context lengths for verify.
    rt::Tensor& selectTokenIndices; //!< [batch, verifySize] selected token indices.
};

//! Parameters for ddtreeBuild().
struct DDTreeBuildParams
{
    DDTreeBuildInputs inputs;           //!< Required input tensors.
    DDTreeBuildOutputs outputs;         //!< Required output tensors.
    int32_t candidateTopK;              //!< Per-depth candidateTopK from DFlash draftingTopK.
    void* workspace;                    //!< Temporary workspace from getDDTreeBuildWorkspaceSize().
    size_t workspaceSize;               //!< Workspace size in bytes.
    cudaStream_t stream;                //!< CUDA stream for kernel launches.
    int32_t firstCandidateLogitsRow{1}; //!< Logits row used for depth-1 children.
};

//! Returns temporary workspace size required by ddtreeBuild().
//!
//! The workspace stores candidate token ids and log probabilities for DFlash candidateTopK selection.
size_t getDDTreeBuildWorkspaceSize(
    int32_t batchSize, int32_t dflashBlockSize, int32_t verifySize, int32_t vocabSize, int32_t candidateTopK);

//! Build a prefix-closed DDTree from one DFlash draft logits pass.
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
//! Example with candidateTopK = 2 and verifySize = 6:
//!
//!     node:        0     1     2     3     4     5
//!     depth:       0     1     1     2     2     3
//!     parent:     -1     0     0     1     2     3
//!
//! In this example, nodes 1 and 2 are the best depth-1 children of the root.
//! Nodes 3 and 4 extend different parents, while node 5 extends node 3. The
//! packed/unpacked ancestor masks allow each verify node to attend only to the
//! tokens on its own root-to-node path.
//!
//! Inputs:
//!     draftLogits [GPU, Float]: [batch, dflashBlockSize, vocabSize].
//!     rootTokenIds [GPU, Int32]: last accepted token for each batch, [batch].
//!     baseLengths [GPU, Int32]: committed base length before verify, [batch].
//!     candidateTopK: DFlash DDTree candidateTopK, wired from draftingTopK.
//!     firstCandidateLogitsRow: logits row used for depth-1 children. Depth D
//!         children consume row firstCandidateLogitsRow + D - 1.
//!     draftVocabMappingTable: optional reduced-to-full draft vocab mapping, [vocabSize].
//!     workspace: temporary storage with size from getDDTreeBuildWorkspaceSize().
//!
//! Outputs:
//!     nodeTokenIds [GPU, Int32]: flattened tree token ids, [batch, verifySize]. Node 0 is root.
//!     nodeDepths [GPU, Int32]: flattened tree depths, [batch, verifySize]. Root depth is 0.
//!     parentIds [GPU, Int32]: flattened parent node indices, [batch, verifySize]. Root and padding use -1.
//!     nodeScores [GPU, Float]: prefix log-prob scores, [batch, verifySize]. Padding uses -inf.
//!     validCounts [GPU, Int32]: valid node count per batch, [batch].
//!     verifyTokenIds [GPU, Int32]: base verify token ids, [batch, verifySize].
//!     verifyPositionIds [GPU, Int32]: base verify position ids, baseLengths + nodeDepths, [batch, verifySize].
//!     packedAncestorMask [GPU, Int32]: packed tree attention mask,
//!         [batch, verifySize, ceil(verifySize / 32)].
//!     ancestorMask [GPU, Int8]: unpacked EAGLE-style tree attention mask, [batch, verifySize, verifySize].
//!     contextLengths [GPU, Int32]: baseLengths + verifySize, [batch].
//!         Tree verification writes every flattened node into a contiguous KV suffix.
//!         Absolute token positions are still provided separately through positionIds.
//!     selectTokenIndices [GPU, Int64]: [0, 1, ..., verifySize - 1] per batch, [batch, verifySize].
//!
//! The DDTree builder runs on GPU with fixed-size scratch. It does not provide a CPU runtime fallback.
void ddtreeBuild(DDTreeBuildParams const& params);

} // namespace kernel
} // namespace trt_edgellm
