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

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief INT4 group-wise quantized GEMV (matrix-vector multiplication)
 *
 * Optimized for batch size 1~4 (M=1~4). Performs: out = in @ W_dequantized
 * where W is INT4 quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K] (Primarily optimized for M ~ [1, 4])
 * @param kernel INT4 quantized weight matrix [N/2, K] in int8 (packed int4 format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void gemv_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);

/*!
 * @brief INT4 group-wise quantized GEMM (matrix-matrix multiplication)
 *
 * Optimized for batch size > 1. Performs: out = in @ W_dequantized
 * where W is INT4 quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K]
 * @param kernel INT4 quantized weight matrix [N/2, K] in int8 (packed int4 format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void gemm_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);
} // namespace kernel
} // namespace trt_edgellm