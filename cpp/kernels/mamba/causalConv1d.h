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
 * - Added decode-mode, state capture, and shift-insert kernel interfaces
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

/*!
 * \brief Prefill causal depthwise conv1d.
 *
 * x:              [batch, seq_len, dim]
 * weight:         [dim, 1, width]
 * bias:           [dim] (optional)
 * initialState:   [batch, dim, width] (optional continuation state)
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
 * \brief Capture conv state from prefill input.
 *
 * x:              [batch, seqLen, dim]
 * contextLengths: [batch] INT32, per-batch actual token count in [0, seqLen] (optional)
 * initialState:   [batch, dim, width]  (optional continuation state)
 * convState:      [batch, dim, width]  (output; may alias initialState)
 * stateIndices:   [batch] INT32, execution-row to resident-state-pool row (optional)
 */
void invokeCaptureConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::OptionalInputTensor initialState,
    trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::OptionalInputTensor contextLengths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

/*!
 * \brief Decode-mode conv1d: shift conv_state, insert new column, and compute dot product.
 *
 * convState: [batch, dim, width]  (in-place update)
 * newCol:    [batch, 1, dim]  (new single-token input)
 * weight:    [dim, 1, width]
 * bias:      [dim] (optional)
 * out:       [batch, 1, dim]
 */
void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

/*!
 * \brief MTP (multi-token) decode: process T draft tokens with per-step state checkpointing.
 *
 * For each draft token t in [0, T):
 *   1. Shift conv_state left by 1, insert newCols[:, t, :]
 *   2. Compute output = dot(conv_state, weight) + bias
 *   3. Save intermediate conv_state to intermediateConvStates[:, t, :, :]
 *
 * convState:               [state_rows, dim, width]      FP16  (persistent committed state, read-only)
 * newCols:                 [batch, T, dim]               FP16  (T draft token inputs)
 * weight:                  [dim, 1, width]               FP16
 * bias:                    [dim]                         FP16  (optional)
 * out:                     [batch, T, dim]               FP16  (T outputs)
 * intermediateConvStates:  [batch, T, dim, width]        FP16  (per-step state cache for rollback)
 * T:                       number of draft tokens
 */
void invokeCausalConv1dDecodeMTP(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, int32_t T, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream);

/*!
 * \brief DDTree decode: compute each tree node from the persistent conv state plus its root-to-node path.
 *
 * Node 0 is the root verify token. Its output/state are computed by shifting x[:, 0, :] into the persistent
 * convState. Non-root nodes append the full root-to-node path and write one checkpoint per node.
 *
 * convState:               [batch, dim, width]           FP16  (persistent committed state, read-only)
 * newCols:                 [batch, verifySeq, dim]       FP16  (tree-node conv inputs)
 * weight:                  [dim, 1, width]               FP16
 * bias:                    [dim]                         FP16  (optional)
 * out:                     [batch, verifySeq, dim]       FP16  (tree-node conv outputs)
 * convStateOut:            aliases convState; persistent committed state remains unchanged
 * intermediateConvStates:  [batch, verifySeq, dim, width] FP16 (per-node states for accepted-node scatter)
 * treeParentIds:           [batch, verifySeq]            INT32 (root/padding parent is -1)
 * treeDepths:              [batch, verifySeq]            INT32 (root/padding depth is 0)
 */
void invokeCausalConv1dDecodeDDTree(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStateOut, trt_edgellm::rt::Tensor& intermediateConvStates,
    trt_edgellm::rt::Tensor const& treeParentIds, trt_edgellm::rt::Tensor const& treeDepths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream);

} // namespace mamba_ssm
