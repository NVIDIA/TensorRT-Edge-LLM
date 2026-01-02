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

/*!
 * @brief GEMM/GEMV reference code (matrix-matrix multiplication)
 *
 *
 * @param in_feats Input features [M, K]
 * @param weight weight matrix [K, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param stream CUDA stream
 */
void naive_gemm_forward(half* in_feats, half* weight, half* out_feats, int m, int n, int k, cudaStream_t stream);