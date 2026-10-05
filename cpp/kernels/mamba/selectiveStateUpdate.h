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

/*
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Extracted selective state update kernel interface for TensorRT Edge-LLM integration
 * - Added explicit stride parameters for non-contiguous/padded memory layouts
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

/*!
 * \brief Launch the decode selective state update kernel (seq_len == 1).
 *
 * Computes:
 *   new_state = state * exp(A * dt) + B * dt * x
 *   output    = sum_i(new_state_i * C_i) + D * x
 *   if z is present: output *= silu(z)
 *
 * x:       [batch, nheads, dim]
 * A:       [nheads], FP32
 * B, C:    [batch, ngroups, dstate]
 * dt:      [batch, nheads]
 * dt_bias: [nheads] (optional)
 * D:       [nheads] (optional skip connection)
 * z:       same shape as x (optional SiLU gate)
 * state:   [batch, nheads, dim, dstate], updated in-place
 * output:  [batch, nheads, dim]
 */
void invokeSelectiveStateUpdate(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

/*!
 * \brief Launch the prefill selective state update kernel (seq_len > 1).
 *
 * Processes all seq_len tokens in a single kernel launch. x must be 4D:
 * [batch, seq_len, nheads, dim].
 *
 * When the replay stash (``replayDA``/``replayU``/``replayB``/``replayDT``) is provided (MTP spec-verify), the
 * kernel leaves the committed ``state`` read-only and instead stashes the minimal per-token replay
 * inputs — dA/dt [batch, seq_len, nheads], x [batch, seq_len, nheads, dim], and B [batch, seq_len,
 * ngroups, dstate] — so the runtime can reconstruct the accepted state after verification
 * via ``invokeMambaReplayReconstruct``.
 */
void invokeSelectiveStateUpdatePrefill(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, trt_edgellm::rt::OptionalInputTensor contextLengths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, trt_edgellm::rt::OptionalOutputTensor replayDA,
    trt_edgellm::rt::OptionalOutputTensor replayU, trt_edgellm::rt::OptionalOutputTensor replayB,
    trt_edgellm::rt::OptionalOutputTensor replayDT, cudaStream_t stream);

/*!
 * \brief Evaluate Mamba SSM states independently along every DDTree ancestor path.
 *
 * The committed state stays read-only. For each verify node, the kernel walks
 * ``treeParentIds`` from that node to the root, replays the recurrence in root-to-node order,
 * emits that node's output, and stores its dA/x/B/dt tuple for accepted-path commit.
 * Tree paths are limited to the shared speculative verify budget of 128 nodes.
 */
void invokeSelectiveStateUpdateDDTree(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dtBias, trt_edgellm::rt::OptionalInputTensor D, trt_edgellm::rt::Tensor& state,
    trt_edgellm::rt::Tensor& output, trt_edgellm::rt::Tensor const& treeParentIds,
    trt_edgellm::rt::Tensor const& treeDepths, bool dtSoftplus, trt_edgellm::rt::OptionalInputTensor stateIndices,
    trt_edgellm::rt::Tensor& replayDA, trt_edgellm::rt::Tensor& replayU, trt_edgellm::rt::Tensor& replayB,
    trt_edgellm::rt::Tensor& replayDT, cudaStream_t stream);

/*!
 * \brief Gather/scatter active Mamba state rows for kernels that require contiguous batch state.
 *
 * The resident pool and active state tensors have identical trailing dimensions. ``stateIndices``
 * maps each active execution row to its resident pool row.
 */
void invokeMambaStateGather(trt_edgellm::rt::Tensor const& residentState, trt_edgellm::rt::Tensor& activeState,
    trt_edgellm::rt::Tensor const& stateIndices, cudaStream_t stream);

void invokeMambaStateScatter(trt_edgellm::rt::Tensor const& activeState, trt_edgellm::rt::Tensor& residentState,
    trt_edgellm::rt::Tensor const& stateIndices, cudaStream_t stream);

//! Device-resident pointers for one layer's persistent recurrent and convolution states.
struct MambaStateLayerInfo
{
    void* recurrentState;
    void* convState;
};

//! Clear one resident slot, or every slot when @p slot is -1, across all Mamba layers in one launch.
void invokeMambaStateClear(MambaStateLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t residentRows,
    int32_t slot, int64_t recurrentRowBytes, int64_t convRowBytes, cudaStream_t stream);

/*!
 * \brief Reconstruct the committed recurrent state after MTP acceptance (replay).
 *
 * Re-runs the SSD recurrence over the first ``acceptedLengths[b]`` tokens of the
 * replay stash produced by the prefill kernel, in place on the read-only committed ``state``. A batch
 * whose accepted count is 0 is left untouched.
 *
 * state:           [residentRows, nheads, dim, dstate], updated in-place (half or float)
 * replayDA:        [activeBatch, seq_len, nheads] FP32
 * replayU:         [activeBatch, seq_len, nheads, dim] FP32 (stores the unscaled x input)
 * replayB:         [activeBatch, seq_len, ngroups, dstate] FP32
 * replayDT:        [activeBatch, seq_len, nheads] FP32
 * acceptedLengths: [activeBatch] INT32 — accepted draft-token count per sequence
 * stateIndices:    [activeBatch] INT32 — active row to resident state row mapping
 * activeBatchSize: number of active sequences; padded batches beyond it are left
 *                  untouched (the state pools are sized to maxBatch, so the loop
 *                  must be bounded by activeBatchSize to avoid reading past
 *                  acceptedLengths).
 */
struct MambaReplayLayerInfo
{
    void* stateDst;
    void const* replayDa;
    void const* replayU;
    void const* replayB;
    void const* replayDt;
};

void invokeMambaReplayReconstructBatched(MambaReplayLayerInfo const* deviceLayerInfos, int32_t numLayers,
    trt_edgellm::rt::Tensor const& state, trt_edgellm::rt::Tensor const& replayU,
    trt_edgellm::rt::Tensor const& replayB, trt_edgellm::rt::Tensor const& replayDT,
    trt_edgellm::rt::Tensor const& acceptedLengths, trt_edgellm::rt::Tensor const& stateIndices,
    int32_t activeBatchSize, cudaStream_t stream, trt_edgellm::rt::Tensor const* acceptedNodeIds = nullptr,
    int32_t maxAcceptLen = 0);

} // namespace mamba_ssm
