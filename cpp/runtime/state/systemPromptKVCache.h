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

#include "common/checkMacros.h"
#include "common/mathUtils.h"
#include "common/tensor.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace trt_edgellm
{

using SystemPromptCacheKey = std::tuple<std::string, std::string>;

/*! \brief Structure to hold cached system prompt and its KV cache.
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    std::vector<rt::Tensor> kvCacheLayers;        //!< Per-layer KV cache tensors for the system prompt
    std::vector<rt::Tensor>
        recurrentStateContents;                //!< Cached recurrent states for hybrid layers (empty if not applicable)
    std::vector<rt::Tensor> convStateContents; //!< Cached conv states for hybrid layers (empty if not applicable)
};

inline SystemPromptCacheKey keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

/*! \brief Result of reconciling a request's input tokens against a previously-captured system-prompt KV cache.
 */
struct SystemPromptReuseResult
{
    int32_t reuseKVCacheLength;     //!< Number of cached tokens to reuse (recorded for KV cache reuse)
    std::vector<int32_t> tokenIds;  //!< Remaining input tokens still to be prefilled
    int32_t effectivePrefillLength; //!< Number of tokens left to prefill (== tokenIds.size())
};

/*! \brief Compute how many tokens of `cache` can be reused for a request with input tokens `inputIds`.
 *
 * The cached token length is taken from `cache.tokenizedPrompt`, i.e. the tokenized prompt that was
 * actually captured, rather than from any KV-cache tensor's physical shape. This keeps the result
 * correct regardless of a per-layer tensor's dimension order, and it works for pure-recurrent models
 * whose `kvCacheLayers` is empty (no attention layers to capture KV for).
 *
 * Reuses N-1 of the N cached tokens so the Nth token is treated as real input in prefill; this keeps
 * the draft prefill boundary aligned with the true next-token position.
 *
 * @param cache Previously-captured system-prompt KV cache.
 * @param inputIds Full token-id sequence for the current request.
 * @throws std::runtime_error if the cached length is not strictly between 0 and inputIds.size().
 */
inline SystemPromptReuseResult computeSystemPromptReuse(
    SystemPromptKVCache const& cache, std::vector<int32_t> const& inputIds)
{
    size_t const cachedLength = cache.tokenizedPrompt.size();
    check::check(cachedLength > 0 && cachedLength < inputIds.size(),
        "The reuse length shall be larger than 0 and not exceed the input length.");
    size_t const effectiveReuseLength = cachedLength - 1;

    SystemPromptReuseResult result;
    result.reuseKVCacheLength = math::cast<int32_t>(effectiveReuseLength);
    result.tokenIds.assign(inputIds.begin() + effectiveReuseLength, inputIds.end());
    result.effectivePrefillLength = math::cast<int32_t>(inputIds.size() - effectiveReuseLength);
    return result;
}

} // namespace trt_edgellm
