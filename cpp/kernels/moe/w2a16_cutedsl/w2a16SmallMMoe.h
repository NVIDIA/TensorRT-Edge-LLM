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
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Largest token count served by the small-M IQ2_XXS MoE kernels.
constexpr int32_t kW2A16_SMALL_M_MOE_MAX_TOKENS = 16;

struct W2A16SmallMMoeParams
{
    int32_t numTokens{};
    int32_t numExperts{};
    int32_t moeInterSize{};
    void const* hiddenStates{};  //!< BF16 [M,1024].
    void const* fc1Weights{};    //!< Packed IQ2_XXS payload int32 [E,I/64,1024/128,8,4,8,2].
    void const* fc2Weights{};    //!< Packed IQ2_XXS payload int32 [E,1024/64,I/128,8,4,8,2].
    void const* fc1Bases{};      //!< FP16 [E,1024/256,I/16,8,2].
    void const* fc2Bases{};      //!< FP16 [E,I/256,1024/16,8,2].
    float const* expertScales{}; //!< FP32 [E].
    void const* selectorTable{}; //!< 2048-byte IQ2_XXS pair-selector table of the packed layout.
    int32_t const* topkIds{};    //!< int32 [M,9].
    float const* topkWeights{};  //!< FP32 [M,9].
    void* output{};              //!< BF16 [M,1024].
};

//! Workspace bytes: the BF16 activations and per-route BF16 contributions of up to M*9 routes.
size_t w2a16SmallMMoeWorkspaceSize(int32_t numTokens, int32_t moeInterSize) noexcept;

//! Hidden 1024, top-9, ReLU2 MoE for 1 <= M <= kW2A16_SMALL_M_MOE_MAX_TOKENS on the packed layout of the fused runner.
//! Each expert's weights are read once for all tokens routed to it; each 32-weight sub-block is contracted on
//! unscaled signed magnitudes with BF16 tensor cores and scaled afterwards. Rounding points match the reference:
//! FC1 output, ReLU2 output and each weighted FC2 contribution round to BF16; the top-9 sum is FP32 in slot order.
//! Requires 256-byte aligned workspace of w2a16SmallMMoeWorkspaceSize bytes and an SM80+ device.
cudaError_t launchW2A16SmallMMoe(W2A16SmallMMoeParams const& params, void* workspace, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm
