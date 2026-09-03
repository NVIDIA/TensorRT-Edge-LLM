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

#include <common/tensor.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Per-layer KV cache metadata for batched kernel operations.
struct KVLayerInfo
{
    void* data;         //!< Base of this layer's physical K-then-V page-pool allocation
    int32_t numKVHeads; //!< Number of KV heads for this layer
    int32_t maxSeqLen;  //!< Active-slot row capacity (capPadded)
};

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
 * @throws std::runtime_error if tensors are not located on the GPU, or tensor shapes are invalid
 */
void compactTensorBatch(rt::Tensor const& src, rt::Tensor const& batchMapping, rt::Tensor& dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream);

/**
 * @brief Batched in-place KV pool compaction across a headDim group, moving only live prefixes.
 *
 * One grouped launch covers every layer in `layerInfos` (all sharing `headDim`), K and V halves of
 * the active-slot K/V views included. For each moved row only the contiguous live prefix
 * (`liveLengths[oldBatchIdx] * numKVHeads * headDim` elements) is copied with vectorized
 * loads/stores; padding beyond the live length is left untouched. Scheduled CTAs are proportional
 * to the number of layers (not the allocated capacity), and identity (`oldActiveBatch ==
 * newActiveBatch`) or all-evicted (`newActiveBatch == 0`) calls return without launching.
 *
 * @param layerInfos        [numLayers] GPU array of KVLayerInfo for one headDim group
 * @param batchMapping      [oldActiveBatch] GPU tensor (const input), mapping[i] = newBatchIdx or -1
 * @param liveLengths       [oldActiveBatch] GPU INT32 tensor (const input), live token length per old batch slot
 * @param numLayers         Number of layers in this group
 * @param headDim           Head dimension shared by all layers in this group
 * @param kvPoolPages       Physical K-page count; determines the V-half offset
 * @param kvCacheType       KV pool dtype (kHALF or kFP8); selects the copy element type
 * @param oldActiveBatch    Number of batches before eviction
 * @param newActiveBatch    Number of batches after eviction
 * @param stream            CUDA stream
 *
 * @throws std::invalid_argument for unsupported kvCacheType
 */
void compactKVCacheBatched(KVLayerInfo const* layerInfos, rt::Tensor const& batchMapping, rt::Tensor const& liveLengths,
    int32_t numLayers, int32_t headDim, int32_t kvPoolPages, nvinfer1::DataType kvCacheType, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
