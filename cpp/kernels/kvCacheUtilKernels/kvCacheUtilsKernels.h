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
void incrementLengthTensor(rt::Tensor& lengthTensor, int32_t increment, cudaStream_t stream);

//! \brief Increment the lengthTensor by element-wise values from another tensor
//!
//! This overload increments each element by the corresponding value in newIncrementTensor.
//!
//! \param[in,out] lengthTensor The tensor to be incremented
//! \param[in] newIncrementTensor The tensor containing per-element increment values
//! \param[in] stream The CUDA stream to be used
//! \note LengthTensor and newIncrementTensor shall reside on GPU, have equal length, and have data type of int32_t.
void incrementLengthTensor(rt::Tensor& lengthTensor, rt::Tensor const& newIncrementTensor, cudaStream_t stream);

//! \brief Instantiate the KVCache from a pre-computed KVCache tensor
//!
//! Helper function to instantiate the KVCache from a pre-computed KVCache tensor.
//! Used to support KVCache reuse across multiple inference requests to speedup prefill step.
//!
//! \param[in,out] dstKVCacheBuffer The KVCache buffer to be instantiated.
//!                                  Layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] srcKVCacheTensor The pre-computed KVCache tensor.
//!                              Layout: [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
//! \param[in] batchIdx The batch index of the KVCache to be instantiated
//! \param[in] stream The CUDA stream to be used
void instantiateKVCacheFromTensor(
    rt::Tensor& dstKVCacheBuffer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream);

//! \brief Save the KVCache into a tensor
//!
//! Helper function to save the KVCache into a tensor. Used to support KVCache reuse across multiple
//! inference requests to speedup prefill step. SequenceLength of dstKVCacheTensor must be saved from
//! the srcKVCacheBuffer.
//!
//! \param[out] dstKVCacheTensor The KVCache tensor to be saved.
//!                               Layout: [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
//! \param[in] srcKVCacheBuffer The KVCache buffer to be saved.
//!                              Layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! \param[in] batchIdx The batch index of the KVCache to be saved
//! \param[in] stream The CUDA stream to be used
void saveKVCacheIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheBuffer, int32_t batchIdx, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm