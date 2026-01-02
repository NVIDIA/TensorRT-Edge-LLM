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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! @brief Launch kernel to handle case where KVCache is empty. We will instantiate the KVCache and overwrite the QKV
//! tensor directly.
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in,out] qkv FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq + Hk + Hv, headDim], the tensor
//! will perform inplace update.
//! @param[out] kvCache FP16 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim], write KVCache
//! from the start positions.
//! @param[in] stream CUDA stream to launch the kernel
void launchApplyRopeWriteKVPackedQKV(
    rt::Tensor const& cosSinCache, rt::Tensor& qkv, rt::Tensor& kvCache, cudaStream_t stream);

//! @brief Launch the kernel to handle case where KVCache is not empty. We will write to a dedicated Q tensor and
//! KVCache.
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens INT32 type tensor with layout of [batchSize], the end position of KVCache after writing.
//! @param[in] qkv FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq + Hk + Hv, headDim]
//! @param[out] kvCache FP16 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim], write KVCache
//! from the end position.
//! @param[out] qOut FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim], the output Q tensor.
//! @param[in] stream CUDA stream to launch the kernel
//! @note We won't overwrite QKV tensor in this case but we use Tensor& signature to reduce duplicate code.
void launchApplyRopeWriteKVContinuousQAndKVCache(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor& qkv, rt::Tensor& kvCache, rt::Tensor& qOut, cudaStream_t stream);

//! @brief Launch the kernel when we are performing tree attention for speculative decoding.
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens INT32 type tensor with layout of [batchSize], the end position of KVCache after writing.
//! @param[in] tokenPosIds INT32 type tensor with layout of [batchSize, runtimeSeqLen], the position of token within
//! sequence.
//! @param[in] qkv FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq + Hk + Hv, headDim]
//! @param[out] kvCache FP16 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim], write KVCache
//! from the end position.
//! @param[out] qOut FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim], the output Q tensor.
//! @param[in] stream CUDA stream to launch the kernel
//! @note We won't overwrite QKV tensor in this case but we use Tensor& signature to reduce duplicate code.
void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& qkv, rt::Tensor& kvCache, rt::Tensor& qOut, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm