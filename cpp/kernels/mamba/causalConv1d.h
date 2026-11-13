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
 * This file contains code derived from causal-conv1d
 * (https://github.com/Dao-AILab/causal-conv1d)
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Licensed under the BSD 3-Clause License.
 *
 * Modifications by NVIDIA:
 * - Adapted causal depthwise conv1d kernel interface for TensorRT Edge-LLM integration
 * - Added stride, dilation, and padding parameters for generalized conv1d
 * - Added packed variable-length prefill and state capture interfaces
 * - Added decode-mode, state capture, and shift-insert kernel interfaces
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

// Contract shared by every entry point below. `width` comes from the weight, `dilation` from the
// plugin attribute and `state_len` from the conv state tensor; the three are independent.
//   - weight tap `k` reads the input `dilation * (width - 1 - k)` positions before the one produced
//   - `(width - 1) * dilation <= state_len <= 16` and `width <= 8`
//   - prefill needs `padding == (width - 1) * dilation` for causal output

/*!
 * \brief Prefill causal depthwise conv1d.
 *
 * x:              [batch, seq_len, dim]
 * weight:         [dim, 1, width]
 * bias:           [dim] (optional)
 * initialState:   [batch, dim, state_len] (optional continuation state)
 * contextLengths: [batch] INT32, per-batch actual token count in [0, seq_len] (optional, prefill only)
 * stateIndices:   [batch] INT32, execution-row to resident-state-pool row (optional)
 * out:            [batch, out_seq_len, dim]
 */
void invokeCausalConv1d(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out, int32_t stride, int32_t padding,
    int32_t dilation, trt_edgellm::rt::OptionalInputTensor initialState,
    trt_edgellm::rt::OptionalInputTensor contextLengths, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream);

/*!
 * \brief Causal depthwise conv1d for packed variable-length input.
 *
 * x:               [totalTokens, dim]
 * cuSeqLens:       [numSequences + 1] INT32, nondecreasing prefix sum with first=0 and last=totalTokens. The caller
 *                  must establish this device-resident metadata invariant before launch.
 * stateIndices:    [numSequences] INT32, maps each sequence to a valid row in convStates. Non-pad indices must be
 *                  unique within an invocation; violating uniqueness causes undefined concurrent state updates.
 * hasInitialState: [numSequences] INT32, nonzero when the mapped state row contains valid history
 * weight:          [dim, 1, width]
 * bias:            [dim] (optional)
 * out:             [totalTokens, dim], with storage that does not overlap x
 * convStates:      [numSlots, dim, width - 1] (updated in place except for padSlotId)
 */
void invokeCausalConv1dPacked(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& cuSeqLens,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor const& hasInitialState,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStates, int32_t padSlotId, int32_t stride, int32_t padding, int32_t dilation,
    cudaStream_t stream);

/*!
 * \brief Whether invokeCausalConv1dPacked can run: the CuTe DSL causal_conv1d group is built, the SM is one of
 * SM80/86/87/89/90/100/101/103/110/120/121, dilation is 1, and 2 <= width <= 8.
 */
bool canImplementPackedVarlen(int32_t smVersion, int32_t width, int32_t dilation);

/*!
 * \brief Gather the trailing history columns of selected resident conv states for a packed prefill.
 *
 * residentStates:  [rows, dim, state_len] FP16
 * stateIndices:    [N] INT32, resident row of each sequence
 * historyStates:   [N, dim, history_len] FP16 output, history_len <= state_len
 * historyIndices:  [N] INT32 output, filled with 0..N-1
 * hasInitialState: [N] INT32 output, 1 for in-range state indices and 0 for out-of-range ones
 */
void invokeGatherConvStateHistory(trt_edgellm::rt::Tensor const& residentStates,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor& historyStates,
    trt_edgellm::rt::Tensor& historyIndices, trt_edgellm::rt::Tensor& hasInitialState, cudaStream_t stream);

/*!
 * \brief Capture the conv state of packed sequences into their resident rows, matching invokeCaptureConvState.
 *
 * x:              [T, dim] FP16 packed sequence rows
 * cuSeqLens:      [N + 1] INT32 exclusive prefix sums of the sequence lengths
 * stateIndices:   [N] INT32, unique resident rows; rows outside [0, rows) are skipped
 * residentStates: [rows, dim, state_len] FP16; each selected row is updated in place from its prior contents
 */
void invokeCapturePackedConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& cuSeqLens,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor& residentStates, cudaStream_t stream);

/*!
 * \brief Capture conv state from prefill input.
 *
 * x:              [batch, seqLen, dim]
 * contextLengths: [batch] INT32, per-batch actual token count in [0, seqLen] (optional)
 * initialState:   [batch, dim, state_len]  (optional continuation state)
 * convState:      [batch, dim, state_len]  (output; may alias initialState)
 * stateIndices:   [batch] INT32, execution-row to resident-state-pool row (optional)
 */
void invokeCaptureConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::OptionalInputTensor initialState,
    trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::OptionalInputTensor contextLengths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

/*!
 * \brief Decode-mode conv1d: shift conv_state, insert new column, and compute dot product.
 *
 * convState: [batch, dim, state_len]  (in-place update)
 * newCol:    [batch, 1, dim]  (new single-token input)
 * weight:    [dim, 1, width]
 * bias:      [dim] (optional)
 * out:       [batch, 1, dim]
 */
void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    int32_t dilation, trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

/*!
 * \brief MTP (multi-token) decode: process T draft tokens with per-step state checkpointing.
 *
 * For each draft token t in [0, T):
 *   1. Compute output from the dilated taps of conv_state plus newCols[:, t, :]
 *   2. Shift conv_state left by 1 and insert newCols[:, t, :]
 *   3. Save intermediate conv_state to intermediateConvStates[:, t, :, :]
 *
 * convState:               [state_rows, dim, state_len]  FP16  (persistent committed state, read-only)
 * newCols:                 [batch, T, dim]               FP16  (T draft token inputs)
 * weight:                  [dim, 1, width]               FP16
 * bias:                    [dim]                         FP16  (optional)
 * out:                     [batch, T, dim]               FP16  (T outputs)
 * intermediateConvStates:  [batch, T, dim, state_len]    FP16  (per-step state cache for rollback)
 * T:                       number of draft tokens
 */
void invokeCausalConv1dDecodeMTP(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, int32_t T, int32_t dilation,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

//! Whether the measured SM100 B1/Q8 specialization applies to a linear verification shape.
bool canUseCausalConv1dDecodeChainParallel(int32_t smVersion, int32_t batch, int32_t numTokens, int32_t width,
    int32_t dilation, int32_t stateLen, int32_t stride) noexcept;

//! Node-parallel implementation of linear width-four verification.
void invokeCausalConv1dDecodeChainParallel(trt_edgellm::rt::Tensor const& convState,
    trt_edgellm::rt::Tensor const& newCols, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream);

/*!
 * \brief DDTree decode: compute each tree node from the persistent conv state plus its root-to-node path.
 *
 * Node 0 is the root verify token. Its output/state are computed by shifting x[:, 0, :] into the persistent
 * convState. Non-root nodes append the full root-to-node path and write one checkpoint per node.
 *
 * convState:               [batch, dim, state_len]       FP16  (persistent committed state, read-only)
 * newCols:                 [batch, verifySeq, dim]       FP16  (tree-node conv inputs)
 * weight:                  [dim, 1, width]               FP16
 * bias:                    [dim]                         FP16  (optional)
 * out:                     [batch, verifySeq, dim]       FP16  (tree-node conv outputs)
 * convStateOut:            aliases convState; persistent committed state remains unchanged
 * intermediateConvStates:  [batch, verifySeq, dim, state_len] FP16 (per-node states for accepted-node scatter)
 * treeParentIds:           [batch, verifySeq]            INT32 (root/padding parent is -1)
 * treeDepths:              [batch, verifySeq]            INT32 (root/padding depth is 0)
 */
void invokeCausalConv1dDecodeDDTree(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStateOut, trt_edgellm::rt::Tensor& intermediateConvStates,
    trt_edgellm::rt::Tensor const& treeParentIds, trt_edgellm::rt::Tensor const& treeDepths, int32_t dilation,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

} // namespace mamba_ssm
