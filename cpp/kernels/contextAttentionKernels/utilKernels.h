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
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

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
//! \param[in]  runtimeSeqLen     Runtime sequence length (equals to the maximum of inputSeqLen).
//! \param[in]  stream            CUDA stream used to launch the kernel.
//! \note kvCacheStartIndices is optional. If it is not provided, kvStartIndices will be assumed to be 0.
void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs, int32_t const runtimeSeqLen,
    cudaStream_t stream);

//! \brief Converts KV cache layout from BHSD layout to BSHD layout for attention computation.
//!
//! Converts an input tensor in [B, 2, H, S, D] into [B, S, 2, H, D].
//!
//! \tparam T Element type (e.g. float, half, bfloat16, etc.).
//!
//! \param[in] src    Source tensor with shape [B, 2, H, S, D].
//! \param[out] dst   Destination tensor with shape [B, S, 2, H, D].
//! \param[in] stream CUDA stream to launch the kernel on
void cvtKVLayoutBHSDToBSHD(rt::Tensor const& src, rt::Tensor& dst, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
