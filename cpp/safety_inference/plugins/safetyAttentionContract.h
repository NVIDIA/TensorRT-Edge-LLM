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

#include "../kernels/safetyAttentionKernels.h"

#include <NvInferSafePlugin.h>
#include <NvInferSafeRuntime.h>
#include <NvInferVersion.h>
#include <array>

static_assert(NV_TENSORRT_MAJOR > 11 || (NV_TENSORRT_MAJOR == 11 && NV_TENSORRT_MINOR >= 3),
    "Safety attention requires TensorRT 11.3 or later");

namespace trt_edgellm
{
namespace safety
{

enum class AttentionRole : int32_t
{
    kLLM,
    kVISUAL
};

//! Serialized identity is shared with the current exporter; only vanilla FP16 is admitted.
struct AttentionParameters
{
    AttentionRole role{AttentionRole::kLLM};
    int32_t qHeads{};
    int32_t kvHeads{};
    int32_t headSize{};
    float scale{};
};

constexpr int32_t kMAX_TOKENS{8192};
constexpr int32_t kMAX_HEADS{64};
constexpr int32_t kMAX_FIELDS{32};

inline constexpr char const* kDISABLED_INTEGER_FIELDS[]{"enable_tree_attention", "enable_fp8_kv_cache",
    "enable_context_mask_selector", "enable_vision_block_attention", "enable_qk_norm", "qk_norm_post_rope",
    "enable_kv_shared", "enable_attention_sink", "enable_contiguous_query_swa", "supports_bounded_kv_cache",
    "skip_softmax_prefill_input", "skip_softmax_decode_input"};

//! Parse the current export fields, rejecting unknown or non-vanilla features.
bool parseAttentionFields(
    AttentionRole role, nvinfer1::PluginFieldCollection const* fields, AttentionParameters& parameters) noexcept;

//! Validate a complete fixed-profile IO descriptor set, including strides and byte extents.
bool validateAttentionDescriptors(AttentionParameters const& parameters, nvinfer2::safe::TensorDescriptor const* inputs,
    int32_t nbInputs, nvinfer2::safe::TensorDescriptor const* outputs, int32_t nbOutputs,
    AttentionLaunch& launch) noexcept;

//! Validate a builder format proposal without requiring unresolved physical storage metadata.
bool supportsAttentionFormat(
    AttentionParameters const& parameters, int32_t index, nvinfer2::safe::TensorDescriptor const& descriptor) noexcept;

//! Build callbacks only: derive bounded linear storage from fixed shapes before checking the shared IO contract.
//! Runtime enqueue must use validateAttentionDescriptors on the actual descriptors instead.
bool validateAttentionBuildDescriptors(AttentionParameters const& parameters,
    nvinfer2::safe::TensorDescriptor const* inputs, int32_t nbInputs, nvinfer2::safe::TensorDescriptor const* outputs,
    int32_t nbOutputs, AttentionLaunch& launch) noexcept;

//! Validate the actual linear storage of one runtime descriptor.
bool supportsAttentionTensor(
    AttentionParameters const& parameters, int32_t index, nvinfer2::safe::TensorDescriptor const& descriptor) noexcept;

} // namespace safety
} // namespace trt_edgellm
