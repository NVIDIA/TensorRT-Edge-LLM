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

#include "builder/safety/safetyLLMShapeUtils.h"

#include "common/bindingNames.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace trt_edgellm::safety::builder
{
namespace
{
nvinfer1::Dims shape(std::initializer_list<int64_t> values)
{
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}
} // namespace

nvinfer1::Dims getSafetyLLMInputShape(std::string_view name, nvinfer1::Dims const& exportedShape,
    SafetyLLMStaticShapeProfile const& profile, int64_t hiddenSize, int64_t numKVHeads, int64_t headSize,
    int64_t rotaryDim)
{
    for (auto const value :
        {profile.batchSize, profile.inputSeqLen, profile.kvCacheCapacity, hiddenSize, numKVHeads, headSize, rotaryDim})
    {
        if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        {
            throw std::invalid_argument("Safety LLM dimensions must be positive and fit int32.");
        }
    }
    if (profile.inputSeqLen > profile.kvCacheCapacity || rotaryDim > headSize)
    {
        throw std::invalid_argument("Safety LLM sequence exceeds KV capacity or rotary dimension exceeds head size.");
    }

    nvinfer1::Dims expected{};
    if (name == binding_names::kContextLengths || name == binding_names::kKVCacheStartIndex)
    {
        expected = shape({profile.batchSize});
    }
    else if (name == binding_names::kRopeCosSin)
    {
        expected = shape({profile.batchSize, profile.kvCacheCapacity, rotaryDim});
    }
    else if (name.rfind(binding_names::kPastKeyValuesTemplate, 0) == 0)
    {
        expected = shape({profile.batchSize, 2, numKVHeads, profile.kvCacheCapacity, headSize});
    }
    else if (name == binding_names::kInputsEmbeds)
    {
        expected = shape({profile.batchSize, profile.inputSeqLen, hiddenSize});
    }
    else if (name == binding_names::kLastTokenIds || name == binding_names::kAttentionPosId)
    {
        expected = shape({profile.batchSize, 1});
    }
    else if (name == binding_names::kAttentionMask)
    {
        expected = shape({profile.batchSize, 1, 1});
    }
    else if (name == "input_ids")
    {
        expected = shape({profile.batchSize, profile.inputSeqLen});
    }
    else if (name == "image_embeds")
    {
        if (profile.imageTokens <= 0 || profile.imageTokens > std::numeric_limits<int32_t>::max())
        {
            throw std::invalid_argument("Safety LLM imageTokens must be positive and fit int32.");
        }
        expected = shape({profile.imageTokens, hiddenSize});
    }
    else
    {
        throw std::invalid_argument("Unsupported Safety LLM input '" + std::string(name)
            + "'; export with --safety (batched inputs and contiguous KV, without a page table).");
    }

    if (exportedShape.nbDims != expected.nbDims)
    {
        throw std::invalid_argument("Safety LLM input '" + std::string(name)
            + "' has an incompatible rank; use --safety export with batched inputs and contiguous KV.");
    }
    int64_t volume{1};
    for (int32_t index = 0; index < expected.nbDims; ++index)
    {
        if (exportedShape.d[index] != -1 && exportedShape.d[index] != expected.d[index])
        {
            throw std::invalid_argument("Safety LLM input '" + std::string(name)
                + "' has a fixed dimension incompatible with the contiguous-KV MSS profile.");
        }
        if (volume > std::numeric_limits<int64_t>::max() / expected.d[index])
        {
            throw std::invalid_argument("Safety LLM input volume exceeds int64.");
        }
        volume *= expected.d[index];
    }
    return expected;
}
} // namespace trt_edgellm::safety::builder
