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

/// @brief Batched save: copy multiple layers' KV cache into per-layer tensors in a single launch.
/// All layers must share the same headDim. srcLayerInfos[i].data points to a two-pool NHD
/// [2, maxBatch, capPadded, numKVHeads_i, headDim] pool; dstLayerInfos[i].data points to a
/// [2, seqLen, numKVHeads_i, headDim] saved tensor (K plane then V plane).
/// @param srcLayerInfos  [numLayers] GPU array — source cache pools
/// @param dstLayerInfos  [numLayers] GPU array — destination saved tensors
/// @param numLayers      Number of layers in this batch
/// @param headDim        Head dimension (same for all layers)
/// @param kvPoolPages    Physical K-page count of the source cache (V-half offset = kvPoolPages*128*H*D)
/// @param batchIdx       Batch index to save from
/// @param sequenceLength Number of tokens to copy
/// @param stream         CUDA stream
void saveKVCacheBatched(KVLayerInfo const* srcLayerInfos, KVLayerInfo const* dstLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t kvPoolPages, int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream);

/// @brief Batched restore: load multiple layers' KV cache from per-layer tensors in a single launch.
/// All layers must share the same headDim. srcLayerInfos[i].data points to a [2, seqLen, numKVHeads_i,
/// headDim] saved tensor (K plane then V plane); dstLayerInfos[i].data points to a two-pool NHD
/// [2, maxBatch, capPadded, numKVHeads_i, headDim] pool.
/// @param dstLayerInfos  [numLayers] GPU array — destination cache pools
/// @param srcLayerInfos  [numLayers] GPU array — source saved tensors
/// @param numLayers      Number of layers in this batch
/// @param headDim        Head dimension (same for all layers)
/// @param kvPoolPages    Physical K-page count of the destination cache (V-half offset = kvPoolPages*128*H*D)
/// @param batchIdx       Batch index to restore into
/// @param sequenceLength Number of tokens to copy
/// @param stream         CUDA stream
void instantiateKVCacheBatched(KVLayerInfo const* dstLayerInfos, KVLayerInfo const* srcLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t kvPoolPages, int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream);

//! \brief Gathers logical pages 0..ceil(seqLen/128) of every slot from a paged K/V page pool into dense
//! split K/V workspaces, for FMHA_v2-style / FFPA consumers that require a contiguous [B, seqLen, H, D]
//! FP16 view. The destination is ALWAYS FP16 (half): an FP8 pool is dequantized with the K/V scales so
//! the downstream `dataPointer<half>()` consumers never reinterpret FP8 bytes as half.
//!
//! `pool` is a single flat page array (the Task-1 [2, maxBatch, capPadded, H, D] allocation
//! reinterpreted as pages); per KVPageTable's convention, V page ids are always K page id + numPages,
//! so both halves index directly into the same `pool` base -- there is no separate V-half pointer.
//!
//! Bad-page semantics: any page-table entry of -1 (unmapped) zero-fills its destination span, whether
//! it lies beyond the slot's live range (the legal padding tail) or inside it (an upstream table bug --
//! the runtime guarantees mapped coverage for live positions, so an in-range -1 cannot occur by
//! construction; if it ever does, the gather fails soft with zeros instead of reading a wild address).
//!
//! \param[in]  pool           Page pool, logically [2*numPages, 128, numKVHeads, headDim] (K pages
//!                            first, V pages at page id +numPages per KVPageTable's kernel view).
//! \param[out] kDst           Destination K tensor (FP16), [batchSize, seqLen, numKVHeads, headDim].
//! \param[out] vDst           Destination V tensor (FP16), [batchSize, seqLen, numKVHeads, headDim].
//! \param[in]  pageTable      Device page table, [batchSize, 2, maxPagesPerSeq]; row 0 = K page ids,
//!                            row 1 = V page ids. A negative id means "unallocated".
//! \param[in]  kvSeqLens      Device [batchSize] per-slot live KV length; distinguishes an in-range
//!                            unmapped page (violation) from the legal padding tail (zero-fill).
//! \param[in]  maxPagesPerSeq Number of logical pages per slot in pageTable's row stride.
//! \param[in]  batchSize      Number of slots.
//! \param[in]  seqLen         Destination token extent per slot (padded capacity spanned by the dst).
//! \param[in]  numKVHeads     Number of KV heads.
//! \param[in]  headDim        Head dimension.
//! \param[in]  elemSize       Size in bytes of one POOL element (2 for FP16, 1 for FP8) -- ignored when
//!                            dequantFp8 is true (pool is FP8, dst is FP16).
//! \param[in]  dequantFp8     If true, treat the pool as FP8 e4m3 and dequantize to FP16 with kScale/vScale.
//! \param[in]  kScale         K dequant scale (dequant value = fp8 * kScale). Unused when dequantFp8 false.
//! \param[in]  vScale         V dequant scale. Unused when dequantFp8 false.
//! \param[in]  stream         CUDA stream to launch the kernel on.
//! \throws std::runtime_error if ceil(seqLen/128) exceeds maxPagesPerSeq.
void gatherPagedKVToSplit(void const* pool, void* kDst, void* vDst, int32_t const* pageTable, int32_t const* kvSeqLens,
    int32_t maxPagesPerSeq, int32_t batchSize, int32_t seqLen, int32_t numKVHeads, int32_t headDim, size_t elemSize,
    bool dequantFp8, float kScale, float vScale, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
