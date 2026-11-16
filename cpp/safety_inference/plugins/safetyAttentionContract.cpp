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

#include "safetyAttentionContract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>

namespace trt_edgellm
{
namespace safety
{
namespace
{
using nvinfer1::DataType;
using nvinfer1::PluginFieldType;
using nvinfer2::safe::TensorDescriptor;

bool sameShape(TensorDescriptor const& a, TensorDescriptor const& b) noexcept
{
    if (a.shape.nbDims != b.shape.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < a.shape.nbDims; ++i)
    {
        if (a.shape.d[i] != b.shape.d[i])
        {
            return false;
        }
    }
    return true;
}

bool isLinear(TensorDescriptor const& desc, DataType type, bool allowEmpty) noexcept
{
    // The five-axis contiguous KV cache is the highest-rank layout supported by these plugins.
    if (desc.dataType != type || desc.shape.nbDims < 1 || desc.shape.nbDims > 5
        || desc.stride.nbDims != desc.shape.nbDims || desc.componentsPerVector != 1 || desc.vectorizedDim != -1)
    {
        return false;
    }
    uint64_t size = 1;
    for (int32_t i = desc.shape.nbDims - 1; i >= 0; --i)
    {
        auto const dim = desc.shape.d[i];
        // size is the contiguous stride from the inner dimensions; singleton strides do not affect addressing.
        if (dim < 0 || dim > std::numeric_limits<int32_t>::max() || (dim == 0 && !allowEmpty)
            || (dim > 1 && desc.stride.d[i] != static_cast<int64_t>(size)))
        {
            return false;
        }
        // Bound the element count to the supported indexing range before multiplying.
        if (dim > 0 && size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / dim)
        {
            return false;
        }
        size *= dim;
    }
    uint64_t const bytes = type == DataType::kHALF ? 2 : 4;
    // Declared storage may exceed the tensor size; division avoids overflowing a byte-count multiplication.
    return desc.bytesPerComponent == bytes && size <= desc.sizeInBytes / bytes;
}
} // namespace

bool parseAttentionFields(
    AttentionRole role, nvinfer1::PluginFieldCollection const* fields, AttentionParameters& parameters) noexcept
{
    if (fields == nullptr || fields->fields == nullptr || fields->nbFields < 3 || fields->nbFields > kMAX_FIELDS)
    {
        return false;
    }
    AttentionParameters result{};
    result.role = role;
    bool hasQ = false;
    bool hasKv = false;
    bool hasDim = false;
    bool hasScale = false;
    for (int32_t i = 0; i < fields->nbFields; ++i)
    {
        auto const& field = fields->fields[i];
        if (field.name == nullptr)
        {
            return false;
        }
        for (int32_t j = 0; j < i; ++j)
        {
            if (std::strcmp(field.name, fields->fields[j].name) == 0)
            {
                return false;
            }
        }
        if (role == AttentionRole::kLLM && std::strcmp(field.name, "xqa_jit_kernels") == 0)
        {
            // Standard serializes an empty blob with null data; Safety only executes admitted AOT kernels.
            if (field.type != PluginFieldType::kCHAR || field.length != 0)
            {
                return false;
            }
            continue;
        }
        if (field.data == nullptr)
        {
            return false;
        }
        bool const integer = field.type == PluginFieldType::kINT32 && field.length == 1;
        bool const real = field.type == PluginFieldType::kFLOAT32 && field.length == 1;
        auto named = [&](char const* name) { return std::strcmp(field.name, name) == 0; };
        if (named(role == AttentionRole::kLLM ? "num_q_heads" : "num_heads") && integer)
        {
            result.qHeads = *static_cast<int32_t const*>(field.data);
            hasQ = true;
        }
        else if (role == AttentionRole::kLLM && named("num_kv_heads") && integer)
        {
            result.kvHeads = *static_cast<int32_t const*>(field.data);
            hasKv = true;
        }
        else if (named("head_size") && integer)
        {
            result.headSize = *static_cast<int32_t const*>(field.data);
            hasDim = true;
        }
        else if (named("attention_scale") && real)
        {
            result.scale = *static_cast<float const*>(field.data);
            hasScale = true;
        }
        else if (named("qkv_scales") && field.type == PluginFieldType::kFLOAT32 && field.length == 3)
        {
            auto const* scales = static_cast<float const*>(field.data);
            if (scales[0] != 1.F || scales[1] != 1.F || scales[2] != 1.F)
            {
                return false;
            }
        }
        else if (role == AttentionRole::kLLM && integer && named("sliding_window_size"))
        {
            if (*static_cast<int32_t const*>(field.data) != -1)
            {
                return false;
            }
        }
        else if (role == AttentionRole::kLLM && integer
            && std::any_of(std::begin(kDISABLED_INTEGER_FIELDS), std::end(kDISABLED_INTEGER_FIELDS), named))
        {
            if (*static_cast<int32_t const*>(field.data) != 0)
            {
                return false;
            }
        }
        else if (role == AttentionRole::kLLM && real && named("rms_norm_eps"))
        {
            if (!std::isfinite(*static_cast<float const*>(field.data)) || *static_cast<float const*>(field.data) <= 0.F)
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    if (role == AttentionRole::kVISUAL)
    {
        result.kvHeads = result.qHeads;
        hasKv = true;
    }
    if (!hasQ || !hasKv || !hasDim || result.qHeads <= 0 || result.qHeads > kMAX_HEADS || result.kvHeads <= 0
        || result.qHeads % result.kvHeads != 0 || (result.headSize != 64 && result.headSize != 128))
    {
        return false;
    }
    if (!hasScale)
    {
        result.scale = 1.F / std::sqrt(static_cast<float>(result.headSize));
    }
    if (!std::isfinite(result.scale) || result.scale <= 0.F)
    {
        return false;
    }
    parameters = result;
    return true;
}

bool supportsAttentionFormat(
    AttentionParameters const& parameters, int32_t index, TensorDescriptor const& descriptor) noexcept
{
    bool const visual = parameters.role == AttentionRole::kVISUAL;
    int32_t const total = visual ? 6 : 7;
    if (index < 0 || index >= total)
    {
        return false;
    }
    DataType type = DataType::kHALF;
    if (visual ? (index == 3 || index == 4) : (index == 2 || index == 4))
    {
        type = DataType::kINT32;
    }
    else if (!visual && index == 3)
    {
        type = DataType::kFLOAT;
    }
    return descriptor.dataType == type && descriptor.componentsPerVector == 1 && descriptor.vectorizedDim == -1;
}

bool supportsAttentionTensor(
    AttentionParameters const& parameters, int32_t index, TensorDescriptor const& descriptor) noexcept
{
    if (!supportsAttentionFormat(parameters, index, descriptor))
    {
        return false;
    }
    bool const visual = parameters.role == AttentionRole::kVISUAL;
    DataType const type = descriptor.dataType;
    bool const shapeOnly = visual && index == 4;
    bool const emptyStart = !visual && index == 4 && descriptor.shape.nbDims == 1 && descriptor.shape.d[0] == 0;
    // Neither the visual length carrier nor an omitted LLM start has data that the kernels read.
    if (shapeOnly || emptyStart)
    {
        return descriptor.dataType == type && descriptor.shape.nbDims == 1 && descriptor.shape.d[0] >= 0
            && descriptor.shape.d[0] <= kMAX_TOKENS && (!shapeOnly || descriptor.shape.d[0] > 0);
    }
    // TRT11.0.2 may leave placement unspecified; the graph owner must supply actual GPU allocations.
    using nvinfer2::safe::MemoryPlacement;
    return (descriptor.memPlacement == MemoryPlacement::kGPU || descriptor.memPlacement == MemoryPlacement::kNONE)
        && isLinear(descriptor, type, false);
}

bool validateAttentionDescriptors(AttentionParameters const& parameters, TensorDescriptor const* inputs,
    int32_t nbInputs, TensorDescriptor const* outputs, int32_t nbOutputs, AttentionLaunch& launch) noexcept
{
    // Parameters have passed field validation. Enqueue checks pointers/aliasing; device kernels check metadata values.
    bool const visual = parameters.role == AttentionRole::kVISUAL;
    if (inputs == nullptr || outputs == nullptr || nbInputs != 5 || nbOutputs != (visual ? 1 : 2))
    {
        return false;
    }
    for (int32_t i = 0; i < nbInputs + nbOutputs; ++i)
    {
        auto const& desc = i < nbInputs ? inputs[i] : outputs[i - nbInputs];
        if (!supportsAttentionTensor(parameters, i, desc))
        {
            return false;
        }
    }
    AttentionLaunch result{};
    result.qHeads = parameters.qHeads;
    result.kvHeads = parameters.kvHeads;
    result.headSize = parameters.headSize;
    result.scale = parameters.scale;
    auto const& q = inputs[0].shape;
    if (q.nbDims != 3)
    {
        return false;
    }
    if (visual)
    {
        // Q/K/V and output share [total tokens, heads, head size].
        if (q.d[0] > kMAX_TOKENS || q.d[1] != parameters.qHeads || q.d[2] != parameters.headSize
            || !sameShape(inputs[0], inputs[1]) || !sameShape(inputs[0], inputs[2])
            || !sameShape(inputs[0], outputs[0])
            // N nonempty sequences require N+1 boundaries and at least N tokens.
            || inputs[3].shape.nbDims != 1 || inputs[3].shape.d[0] < 2
            || inputs[3].shape.d[0] > q.d[0] + 1
            // The carrier's extent bounds each sequence length; its contents are unused.
            || inputs[4].shape.nbDims != 1 || inputs[4].shape.d[0] > q.d[0])
        {
            return false;
        }
        result.tokens = q.d[0];
        result.sequences = inputs[3].shape.d[0] - 1;
        // Visual attention reuses this launch field for maximum sequence length, not RoPE.
        result.ropeTokens = inputs[4].shape.d[0];
    }
    else
    {
        auto const& pool = inputs[1].shape;
        auto const& rope = inputs[3].shape;
        auto const& out = outputs[0].shape;
        // Packed QKV is [1, tokens, (query heads + 2 * KV heads) * head size].
        if (q.d[0] != 1 || q.d[1] > kMAX_TOKENS
            || q.d[2] != (parameters.qHeads + 2 * parameters.kvHeads) * parameters.headSize
            // Past/present caches share [1, K/V, KV heads, capacity, head size].
            || pool.nbDims != 5 || pool.d[0] != 1 || pool.d[1] != 2 || pool.d[2] != parameters.kvHeads
            || pool.d[3] > kMAX_TOKENS || pool.d[4] != parameters.headSize
            || !sameShape(inputs[1], outputs[1])
            // Batch one supplies one length value.
            || inputs[2].shape.nbDims != 1
            || inputs[2].shape.d[0] != 1
            // RoPE [1, positions, rotary dimension] may rotate only part of a head, but always in pairs.
            || rope.nbDims != 3 || rope.d[0] != 1 || rope.d[1] > kMAX_TOKENS || rope.d[2] > parameters.headSize
            || rope.d[2] % 2 != 0
            // An empty start tensor means position zero; otherwise there is one start value.
            || inputs[4].shape.nbDims != 1
            || inputs[4].shape.d[0] > 1
            // Attention output restores separate token/head axes: [1, tokens, query heads, head size].
            || out.nbDims != 4 || out.d[0] != 1 || out.d[1] != q.d[1] || out.d[2] != parameters.qHeads
            || out.d[3] != parameters.headSize)
        {
            return false;
        }
        result.tokens = q.d[1];
        result.capacity = pool.d[3];
        result.ropeTokens = rope.d[1];
        result.rotaryDim = rope.d[2];
    }
    launch = result;
    return true;
}

bool validateAttentionBuildDescriptors(AttentionParameters const& parameters, TensorDescriptor const* inputs,
    int32_t nbInputs, TensorDescriptor const* outputs, int32_t nbOutputs, AttentionLaunch& launch) noexcept
{
    bool const visual = parameters.role == AttentionRole::kVISUAL;
    if (inputs == nullptr || outputs == nullptr || nbInputs != 5 || nbOutputs != (visual ? 1 : 2))
    {
        return false;
    }
    // Builder proposals may lack storage metadata. Synthesize it here only; enqueue validates actual storage.
    std::array<TensorDescriptor, 8> normalized{};
    for (int32_t index = 0; index < nbInputs + nbOutputs; ++index)
    {
        auto& descriptor = normalized[index];
        descriptor = index < nbInputs ? inputs[index] : outputs[index - nbInputs];
        if (!supportsAttentionFormat(parameters, index, descriptor) || descriptor.shape.nbDims < 1
            || descriptor.shape.nbDims > 5)
        {
            return false;
        }
        uint64_t elements = 1;
        descriptor.stride.nbDims = descriptor.shape.nbDims;
        for (int32_t d = descriptor.shape.nbDims - 1; d >= 0; --d)
        {
            int64_t const dimension = descriptor.shape.d[d];
            if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max()
                || (dimension > 0 && elements > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / dimension))
            {
                return false;
            }
            descriptor.stride.d[d] = elements;
            elements *= dimension;
        }
        descriptor.bytesPerComponent = descriptor.dataType == DataType::kHALF ? 2 : 4;
        descriptor.sizeInBytes = elements * descriptor.bytesPerComponent;
        descriptor.memPlacement = nvinfer2::safe::MemoryPlacement::kGPU;
    }
    return validateAttentionDescriptors(
        parameters, normalized.data(), nbInputs, normalized.data() + nbInputs, nbOutputs, launch);
}

} // namespace safety
} // namespace trt_edgellm
