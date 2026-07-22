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

#include "common/tensor.h"
#include "kernels/speculative/batchEvictKernels.h" // KVLayerInfo

namespace trt_edgellm
{
namespace kernel
{

//! \brief Increment the lengthTensor by a scalar increment for each entry
//!
//! This overload increments all elements by a constant value.
//!
//! \param[in,out] lengthTensor The tensor to be incremented
//! \param[in] increment The scalar increment value
//! \param[in] stream The CUDA stream to be used
//! \note LengthTensor shall reside on GPU and have data type of int32_t.
//! \throws std::runtime_error if tensor has wrong location or data type
void incrementLengthTensor(rt::Tensor& lengthTensor, int32_t increment, cudaStream_t stream);

//! \brief Increment the lengthTensor by element-wise values from another tensor
//!
//! This overload increments each element by the corresponding value in newIncrementTensor.
//!
//! \param[in,out] lengthTensor The tensor to be incremented
//! \param[in] newIncrementTensor The tensor containing per-element increment values
//! \param[in] stream The CUDA stream to be used
//! \note LengthTensor and newIncrementTensor shall reside on GPU, have equal length, and have data type of int32_t.
//! \throws std::runtime_error if tensor has wrong location, shape or data type
void incrementLengthTensor(rt::Tensor& lengthTensor, rt::Tensor const& newIncrementTensor, cudaStream_t stream);

//! \brief Single-layer variant: instantiate KV cache for one layer from a saved tensor.
//!
//! \param[in,out] dstKVCacheLayer  [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] srcKVCacheTensor     [2, numKVHeads, sequenceLength, headDim]
//! \param[in] batchIdx Target batch index in the destination buffer
//! \param[in] stream CUDA stream
void instantiateKVCacheLayerFromTensor(
    rt::Tensor& dstKVCacheLayer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream);

//! \brief Single-layer variant: save KV cache for one layer into a tensor.
//!
//! \param[out] dstKVCacheTensor    [2, numKVHeads, sequenceLength, headDim]
//! \param[in] srcKVCacheLayer      [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] batchIdx Source batch index in the buffer
//! \param[in] stream CUDA stream
void saveKVCacheLayerIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheLayer, int32_t batchIdx, cudaStream_t stream);

//! \brief Monolithic variant: instantiate the whole KV cache buffer from a saved tensor.
//!
//! Used by the VLA runtime (LLMEngineRunner) whose KV cache is a single contiguous buffer.
//! \param[in,out] dstKVCacheBuffer [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] srcKVCacheTensor     [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
//! \param[in] batchIdx Target batch index in the destination buffer
//! \param[in] stream CUDA stream
void instantiateKVCacheFromTensor(
    rt::Tensor& dstKVCacheBuffer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream);

//! \brief Monolithic variant: save the whole KV cache buffer into a tensor.
//!
//! \param[out] dstKVCacheTensor    [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
//! \param[in] srcKVCacheBuffer     [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] batchIdx Source batch index in the buffer
//! \param[in] stream CUDA stream
void saveKVCacheIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheBuffer, int32_t batchIdx, cudaStream_t stream);

/// @brief Batched save: copy multiple layers' KV cache into per-layer tensors in a single launch.
/// All layers must share the same headDim. dstLayerInfos[i].data points to a [2, numKVHeads_i, seqLen, headDim] tensor.
/// @param srcLayerInfos  [numLayers] GPU array — source cache buffers
/// @param dstLayerInfos  [numLayers] GPU array — destination saved tensors
/// @param numLayers      Number of layers in this batch
/// @param headDim        Head dimension (same for all layers)
/// @param maxKVHeads     Maximum numKVHeads across all layers (for grid sizing)
/// @param maxBatchSize   Max batch size of the source cache
/// @param batchIdx       Batch index to save from
/// @param sequenceLength Number of tokens to copy
/// @param stream         CUDA stream
void saveKVCacheBatched(KVLayerInfo const* srcLayerInfos, KVLayerInfo const* dstLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t maxKVHeads, int32_t maxBatchSize, int32_t batchIdx, int32_t sequenceLength,
    cudaStream_t stream);

/// @brief Batched restore: load multiple layers' KV cache from per-layer tensors in a single launch.
/// All layers must share the same headDim. srcLayerInfos[i].data points to a [2, numKVHeads_i, seqLen, headDim] tensor.
/// @param dstLayerInfos  [numLayers] GPU array — destination cache buffers
/// @param srcLayerInfos  [numLayers] GPU array — source saved tensors
/// @param numLayers      Number of layers in this batch
/// @param headDim        Head dimension (same for all layers)
/// @param maxKVHeads     Maximum numKVHeads across all layers (for grid sizing)
/// @param maxBatchSize   Max batch size of the destination cache
/// @param batchIdx       Batch index to restore into
/// @param sequenceLength Number of tokens to copy
/// @param stream         CUDA stream
void instantiateKVCacheBatched(KVLayerInfo const* dstLayerInfos, KVLayerInfo const* srcLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t maxKVHeads, int32_t maxBatchSize, int32_t batchIdx, int32_t sequenceLength,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
