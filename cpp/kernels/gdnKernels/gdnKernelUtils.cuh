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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/** Launch the context_lengths → cu_seqlens prefix-sum kernel. */
void launchGdnCalCuSeqLens(void const* context_lengths, // [N] int32
    void* cu_seqlens,                                   // [N+1] int32  (pre-allocated)
    int32_t batchSize, cudaStream_t stream);

/** L2-normalize Q and K in-place along the head dimension.
 *  Q, K: (N, seqLen, H, headDim) float16 — each token-head vector is divided by its L2 norm.
 *  Required preprocessing for the SM100/101/110 Blackwell GDN prefill kernel. */
void launchGdnL2NormQK(void* q, void* k, int32_t n, int32_t seqLen, int32_t h, int32_t headDim, cudaStream_t stream);

/** L2-normalize Q and K in-place in one SM12x kernel launch.
 *  Q, K: (N, seqLen, H, headDim) float16 — each token-head vector is divided by its L2 norm.
 *  One warp normalizes the same row of both buffers in a single kernel launch.
 *  When enablePdl is true, the combined grid is launched with programmatic
 *  stream serialization, waits for its PDL prerequisite before reading Q/K,
 *  and triggers its PDL-dependent consumer before the final normalized values
 *  are stored. */
cudaError_t launchGdnL2NormQKFusedSm12x(
    void* q, void* k, int32_t n, int32_t seqLen, int32_t h, int32_t headDim, bool enablePdl, cudaStream_t stream);

/** Transpose the last two dimensions of the GDN state tensor (out-of-place).
 *  The Blackwell GDN prefill MMA produces state in V-major (d_v, d_k) order,
 *  while the sequential/decode kernels use K-major (d_k, d_v).
 *  src:  (numBlocks, dim, dim) float32 — row-major 2-D blocks
 *  dst:  (numBlocks, dim, dim) float32 — each block transposed
 *  numBlocks = n * hv,  dim = head_dim (128). */
void launchGdnStateTranspose(void const* src, void* dst, int32_t numBlocks, int32_t dim, cudaStream_t stream);

/** Gather selected resident rows while transposing each state matrix from K-major to V-major. */
void launchGdnStateGatherTranspose(void const* src, void* dst, void const* stateIndices, int32_t batchSize,
    int32_t statePoolRows, int32_t numHeads, int32_t dim, cudaStream_t stream);

/** Gather selected resident state rows without changing matrix layout. */
void launchGdnStateGather(void const* src, void* dst, void const* stateIndices, int32_t batchSize, int32_t numHeads,
    int32_t statePoolRows, int32_t kDim, int32_t vDim, cudaStream_t stream);

/** Transpose selected resident state matrices in place with paired shared-memory tiles. */
void launchGdnStateIndexedTransposeInPlace(void* state, void const* stateIndices, int32_t batchSize,
    int32_t statePoolRows, int32_t numHeads, int32_t dim, cudaStream_t stream);

} // namespace trt_edgellm
