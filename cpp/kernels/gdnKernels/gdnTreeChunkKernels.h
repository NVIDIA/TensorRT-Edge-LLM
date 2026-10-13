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

// Stateless chunk-form GDN tree verify + replay commit. Verify computes all
// tree-node outputs directly from the pre-tree state h0 (read-only, no per-node
// state materialization) in "chunk/attention form"; after acceptance, a replay
// kernel advances the persistent state along the accepted path only, consuming
// per-node quantities stashed during verify.
//
// The stash is written into the head of each layer's intermediate-states
// buffer, so this path needs no new persistent allocations. Layout per (batch, node):
//   [ kNorm fp32 h*128 | g fp32 hv | beta fp32 hv | v fp16 hv*128 ]
// densely packed, gdnTreeStashNodeBytes() apart — ~128x smaller than the
// [N, hv, 128, 128] fp32 checkpoint buffer it replaces.
//
// Limits: dk = dv = 128, verify-tree size N <= 64, accepted path <= 16.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

struct MtpLayerInfo; // kernels/speculative/mtpStateScatterKernels.h

constexpr int32_t kGDN_TREE_CHUNK_MAX_NODES{64};
constexpr int32_t kGDN_TREE_CHUNK_MASK_WORDS{kGDN_TREE_CHUNK_MAX_NODES / 32};
constexpr int32_t kGDN_TREE_CHUNK_MAX_ACCEPT{16};

//! Single source of truth for "is the stateless chunk-form verify used for a
//! tree of this size?" The plugin verify dispatch and the decoder commit path
//! MUST both gate on this: if they disagreed, one side could skip replay while
//! the other expects replay stash, silently corrupting the committed recurrent
//! state. Callers add their own context guards (plugin: DDTree verify phase;
//! decoder: hybrid-state commit).
bool gdnTreeChunkVerifyEnabled(int32_t treeSize);

//! Per-(batch, hv-head) prep block exported by the verify prep kernel for the
//! apply kernel (prep/apply split): M + Attn [MAX_NODES^2 f32 each], gamma/beta/invK/
//! invQ [MAX_NODES f32 each], depth [MAX_NODES i32] + maxDepth i32, padded to
//! a 16-byte multiple. Word strides are fixed by MAX_NODES (N-independent).
//! NOTE: the prep blocks (like the KS/QS export) live in the unused tail of
//! each intermediate-states batch row, NOT in the plugin workspace: engine
//! plans serialize the workspace requirement at build time, so growing it
//! without rebuilding all engines causes out-of-bounds workspace writes.
constexpr int32_t kGDN_TREE_CHUNK_PREP_WORDS{2 * kGDN_TREE_CHUNK_MAX_NODES * kGDN_TREE_CHUNK_MAX_NODES
    + 4 * kGDN_TREE_CHUNK_MAX_NODES + kGDN_TREE_CHUNK_MAX_NODES + 1 + 3};

//! Stash sizing (bytes per (batch, node) within one layer's intermediate
//! buffer row). h = #k-heads, hv = #v-heads, dk = dv = 128.
inline size_t gdnTreeStashNodeBytes(int32_t h, int32_t hv)
{
    return static_cast<size_t>(h) * 128 * sizeof(float)   // normalized k
        + static_cast<size_t>(hv) * 2 * sizeof(float)     // g, beta
        + static_cast<size_t>(hv) * 128 * sizeof(__half); // v
}

//! Build packed inclusive ancestor masks from DDTree parent ids.
//! parentIds [GPU int32]: [batch, numNodes], root/padding = -1 (only node 0
//!   may be a root; padding nodes have parent -1 at index > 0 and mask 0).
//! masksOut [GPU uint32]: [batch, numNodes, kGDN_TREE_CHUNK_MASK_WORDS].
//! Returns the kernel launch status (cudaSuccess on success); the caller must
//! propagate a failure rather than continue with an unwritten output.
cudaError_t gdnTreeBuildAncestorMasks(int32_t const* parentIds, uint32_t* masksOut, int32_t batch, int32_t numNodes,
    int32_t maxDepth, cudaStream_t stream);

//! Chunk-form tree verify. Reads h0 (fp32, READ-ONLY), emits per-node outputs
//! o (fp16) and the replay stash. No state is written anywhere.
//! Shapes follow the GDN plugin contract:
//!   q,k [batch, N, h, 128] fp16; v [batch, N, hv, 128] fp16;
//!   a,b [batch, N, hv] fp16; A_log [hv] fp32; dt_bias [hv] fp16;
//!   h0 [batch, hv, 128, 128] fp32; o [batch, N, hv, 128] fp16;
//!   masks [batch, N, kGDN_TREE_CHUNK_MASK_WORDS] uint32 (inclusive);
//!   stash: base pointer of this layer's intermediate buffer; stride
//!     stashBatchStrideBytes between batch rows; nodes packed at
//!     gdnTreeStashNodeBytes() intervals. The unused row tail past the last
//!     stash cell also holds the verify scratch (KS/QS + prep blocks:
//!     align256(MAX_NODES * nodeBytes) + hv*2*MAX_NODES*128 f32 +
//!     hv*kGDN_TREE_CHUNK_PREP_WORDS f32); the launcher validates capacity
//!     and refuses to launch on overflow.
//! Returns the launch status: cudaSuccess, or the launch/attribute error, or
//! cudaErrorInvalidValue if the scratch does not fit the row stride. The
//! caller MUST propagate a non-success return — continuing would consume
//! garbage verify output and, via the unwritten stash, corrupt the persistent
//! recurrent state on the subsequent replay commit.
cudaError_t gdnTreeVerifyChunk(float const* h0, __half const* q, __half const* k, __half const* v, __half const* a,
    __half const* b, float const* A_log, __half const* dt_bias, uint32_t const* masks, __half* o, void* stash,
    size_t stashBatchStrideBytes, int32_t batch, int32_t numNodes, int32_t h, int32_t hv, float scale, bool useQKL2Norm,
    cudaStream_t stream);

//! Batched replay commit across ALL recurrent layers in one launch (grid =
//! [batch*hv, numLayers]). Per layer, advances the persistent state
//! (MtpLayerInfo::recurrentDst) along the accepted path using the stash the
//! chunk-form verify wrote at the head of MtpLayerInfo::recurrentSrc.
//! Identical math/op-order to a sequential scan over the accepted nodes.
//!   acceptedIndices [GPU int32]: [batch, maxAcceptLen] verify-node indices.
//!   acceptLengths [GPU int32]: [batch].
//! Returns the kernel launch status (cudaSuccess on success); the caller must
//! propagate a failure rather than continue with a partially-committed state.
cudaError_t gdnTreeReplayCommitBatched(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers,
    size_t stashBatchStrideBytes, int32_t const* acceptedIndices, int32_t const* acceptLengths,
    int32_t const* stateIndices, int32_t batch, int32_t residentPoolRows, int32_t maxAcceptLen, int32_t numNodes,
    int32_t h, int32_t hv, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
