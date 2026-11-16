/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Largest token count served by the small-M IQ2_XXS GEMM.
constexpr int32_t kW2A16_SMALL_M_MAX_TOKENS = 8;

//! Y[M,N] = X[M,K] @ dequant(W[N,K]).T for M <= kW2A16_SMALL_M_MAX_TOKENS on the packed IQ2_XXS layout of
//! CuteDslW2A16GemmRunner (payload [N/P,K/256,P,64], P = 128 when N % 128 == 0 else 8; FP16 bases [K/256,N];
//! BF16 magnitude table [256,8]). Each 32-weight sub-block is contracted on unscaled signed magnitudes with BF16
//! tensor cores and its FP32 partial is scaled by the sub-block scale, so no BF16 weight is materialized.
//! Requires N % 8 == 0, K % 256 == 0, 16-byte aligned pointers and an SM80+ device. No workspace.
cudaError_t launchW2A16SmallMGemm(void const* input, void const* values, void const* bases, void const* lookup,
    void* output, int32_t m, int32_t n, int32_t k, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm
