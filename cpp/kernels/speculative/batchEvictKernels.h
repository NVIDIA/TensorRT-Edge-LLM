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

#include <common/tensor.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/**
 * @brief Compact KV Cache by removing evicted batches
 *
 * This kernel moves KV Cache data for active batches to dense consecutive positions.
 *
 * @param batchMapping      [oldActiveBatch] GPU tensor (const input), mapping[i] = newBatchIdx or -1 (evict)
 * @param kvCacheBuffer     [numLayers, maxBatch, 2 (K/V), numHeads, maxSeq, headDim] (input/output)
 * @param kvCacheLengths    [maxBatch] (input/output), compacted in-place
 * @param oldActiveBatch    Number of batches before eviction
 * @param newActiveBatch    Number of batches after eviction
 * @param stream            CUDA stream
 *
 * @note This function updates kvCacheBuffer and kvCacheLengths in-place with compacted values
 */
void compactKVCache(rt::Tensor const& batchMapping, rt::Tensor& kvCacheBuffer, rt::Tensor& kvCacheLengths,
    int32_t oldActiveBatch, int32_t newActiveBatch, cudaStream_t stream);

/**
 * @brief Generic tensor compaction along batch dimension
 *
 * This kernel compacts a tensor by removing evicted batches.
 *
 * @param src               Source tensor (const input)
 * @param batchMapping      [oldActiveBatch] GPU tensor (const input), mapping[i] = newBatchIdx or -1
 * @param dst               Destination tensor (output, can be same as src for in-place operation)
 * @param oldActiveBatch    Number of batches before eviction
 * @param newActiveBatch    Number of batches after eviction
 * @param stream            CUDA stream
 *
 * @note Assumes batch dimension is the first dimension (dim 0)
 * @note For in-place operation, pass the same tensor as both src and dst
 */
void compactTensorBatch(rt::Tensor const& src, rt::Tensor const& batchMapping, rt::Tensor& dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
