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

#include "fmha/safetyFmhaLaunch.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace safety
{

class SafetyAttentionModules;

//! Fixed launch dimensions validated against the plugin descriptors before execution.
struct AttentionLaunch
{
    int32_t tokens{};
    int32_t qHeads{};
    int32_t kvHeads{};
    int32_t headSize{};
    int32_t capacity{};
    int32_t ropeTokens{};
    int32_t rotaryDim{};
    int32_t sequences{};
    float scale{};
};

//! Initialization only: prepare statically compiled helpers after explicit FMHA module preparation.
cudaError_t prepareAttentionKernels() noexcept;

//! Workspace includes validation/cumulative lengths followed by aligned FP16 rotated Q storage.
uint64_t attentionWorkspaceBytes(AttentionLaunch const& launch) noexcept;

//! Execute batch-one FP16 causal attention with contiguous KV, without allocating resources.
//! errorMask is the GPU-accessible bitMask of LWE's mapped pinned error storage.
//! Host inspection uses its host address after GPU completion.
cudaError_t launchSafetyLlmAttention(SafetyAttentionModules& modules, fmha::PreparedBindings& bindings,
    AttentionLaunch const& launch, half const* qkv, half const* past, int32_t const* lengths, float const* rope,
    int32_t const* starts, half* output, half* present, void* workspace, uint64_t* errorMask,
    cudaStream_t stream) noexcept;

//! Execute FP16 bidirectional ragged attention with no KV cache.
cudaError_t launchSafetyVisualAttention(SafetyAttentionModules& modules, fmha::PreparedBindings& bindings,
    AttentionLaunch const& launch, half const* q, half const* k, half const* v, int32_t const* sequenceEnds,
    half* output, void* workspace, uint64_t* errorMask, cudaStream_t stream) noexcept;

} // namespace safety
} // namespace trt_edgellm
