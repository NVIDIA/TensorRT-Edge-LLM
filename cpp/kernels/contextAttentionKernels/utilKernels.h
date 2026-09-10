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

#include "common/cudaMacros.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

//! Expand [B, S] vision-block IDs into per-position [blockBegin, blockEnd]
//! interval tensors for the vision-block overlay prefill kernels.
//!
//! Each contiguous run of an identical non-negative ID inside the per-batch
//! valid prefix (contextLengths[b], clamped to seqLen) yields
//! blockBegin = run start and blockEnd = run end for every position in the
//! run.  Text/audio positions (ID < 0) and padding positions receive the
//! -1/-1 sentinel (empty interval).  All tensors are [B, S] int32 device
//! buffers.
void launchBuildVisionBlockRanges(int32_t const* visionBlockIds, int32_t const* contextLengths, int32_t* blockBegin,
    int32_t* blockEnd, int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! \brief Host-side wrapper that launches a lightweight CUDA kernel to compute prefix-sum of sequence lengths
//! and KV cache end indices.
//!
//! \param[in]  inputSeqLen       int32_t tensor with shape [B].  Actual token length of each request.
//! \param[in]  kvCacheStartIndices int32_t tensor with shape [B].  Start index of KV cache for each request.
//!                                (optional, pass in empty tensor to indicate zero start indices)
//! \param[out] cuQSeqLens        int32_t tensor with shape [B+1]. Exclusive prefix-sum of inputSeqLen.
//! \param[out] cuKVSeqLens       int32_t tensor with shape [B+1]. Exclusive prefix-sum of (kvCacheStartIndices[i] +
//!                                inputSeqLen[i]). If kvCacheStartIndices is empty, this will be exclusive prefix-sum
//!                                of inputSeqLen.
//! \param[out] kvCacheEndIdxs    int32_t tensor with shape [B].  Each element equals
//!                                kvCacheStartIndices[i] + runtimeSeqLen (Here we use padding to ease later kernel
//!                                launch).
//! \param[out] paddedCuKVSeqLens (optional) int32_t tensor with shape [B+1]. Exclusive prefix-sum of kvCacheEndIdxs
//!                                (= kvCacheStartIdx + runtimeSeqLen per batch). Pass std::nullopt to skip.
//!                                Background: CuTe DSL FMHA kernel uses bottom_right_align with offset = s_k - s_q.
//!                                Q is padded to runtimeSeqLen for all batches, so we must use padded KV lengths
//!                                (s_k = kvCacheEndIdx per batch) to keep offset non-negative. Using actual s_k
//!                                (< runtimeSeqLen for shorter batches) would produce a negative offset that masks
//!                                out valid KV positions, breaking attention.
//! \param[in]  runtimeSeqLen     Runtime sequence length (equals to the maximum of inputSeqLen).
//! \param[in]  stream            CUDA stream used to launch the kernel.
//! \note kvCacheStartIndices is optional. If it is not provided, kvStartIndices will be assumed to be 0.
//! \throws std::runtime_error if tensor shapes are invalid
void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
