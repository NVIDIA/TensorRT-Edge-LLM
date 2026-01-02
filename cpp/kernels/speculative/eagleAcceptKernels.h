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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

// Forward declaration for internal workspace structure
struct EagleAcceptWorkspace;

/**
 * @brief Calculate workspace size required for Eagle accept algorithm
 *
 * @param batchSize Number of batches to process
 * @param numTokens Number of tokens per batch
 * @return Required workspace size in bytes
 */
size_t getEagleAcceptWorkspaceSize(int32_t batchSize, int32_t numTokens);

/**
 * @brief Eagle accept kernel for speculative decoding tree verification
 *
 * This kernel implements the eagle accept algorithm that:
 * 1. Takes logits of shape [batch_size, num_tokens, vocab_size]
 * 2. Takes a draft tree represented as token_ids [batch_size, num_tokens] and attention mask [batch_size, num_tokens,
 * num_tokens]
 * 3. Verifies the tree by selecting top-1 tokens and checking attention relationships with depth awareness
 * 4. Returns accepted token IDs and their corresponding tree indices as 2D tensors [batch_size, max_depth]
 *
 * Algorithm:
 * - Token 0 is always selected (depth 0)
 * - For each subsequent token, pick top-1 from logits
 * - If vocab mapping table is provided, map selected tokens from reduced vocab to full vocab
 * - Check if the selected token exists at the correct depth in the tree and attends to the previous token
 * - Tree depth is computed from attention mask - tokens at depth d attend to d other tokens
 * - Continue until no valid attention or max depth reached
 * - Batches are processed concurrently using parallel GPU blocks
 *
 * Optimizations:
 * - Two-stage approach: precompute top-1 tokens separately to reduce shared memory usage
 * - Concurrent batch processing: each batch runs in its own GPU block
 * - Parallel argmax reduction using CUB for finding top-1 tokens (in stage 1)
 * - Parallel token search across threads within each block
 * - Depth-aware token selection to respect tree structure layers
 * - Minimal shared memory allocation (only token depths, not full vocab logits)
 * - Uses provided workspace to avoid dynamic allocation
 *
 * @param logits Input logits tensor with shape [batch_size, num_tokens, vocab_size] (FP32, GPU)
 * @param tokenIds Draft tree token IDs with shape [batch_size, num_tokens] (INT32, GPU)
 * @param attentionMask Tree attention mask with shape [batch_size, num_tokens, num_tokens] (INT8, boolean, GPU)
 * @param acceptedTokenIds Output accepted token IDs with shape [batch_size, max_depth] (INT32, GPU)
 * @param acceptedLogitsIndices Output corresponding logits indices with shape [batch_size, max_depth] (INT32, GPU)
 * @param acceptLength Output tensor with accept lengths for each batch with shape [batch_size] (INT32, GPU)
 * @param vocabMappingTable Optional vocab mapping table for reduced vocabulary (INT32, GPU, 1D). Use std::nullopt if
 * not needed.
 * @param workspace Workspace buffer for temporary allocations
 * @param workspaceSize Size of workspace buffer in bytes
 * @param stream CUDA stream for execution
 *
 * @note All tensor parameters must be allocated on GPU device
 * @note Workspace must be at least getEagleAcceptWorkspaceSize(batchSize, numTokens) bytes
 * @note Shared memory usage: Stage 1: CUB temp storage (~1KB), Stage 2: numTokens * sizeof(int32_t) + small overhead
 * @note vocabMappingTable should be provided when base model uses reduced vocabulary
 */
void eagleAccept(rt::Tensor const& logits, rt::Tensor const& tokenIds, rt::Tensor const& attentionMask,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedLogitsIndices, rt::Tensor& acceptLength,
    rt::OptionalInputTensor const& vocabMappingTable, void* workspace, size_t workspaceSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
