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
#include "benchLogger.h"
#include <tensorrt_edgellm/common/bindingNames.h>
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/cudaUtils.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/tensor.h>
#include <tensorrt_edgellm/kernels/speculative/ddtreeKernels.h>
#include <tensorrt_edgellm/kernels/speculative/dflashRuntimeKernels.h>
#include <tensorrt_edgellm/runtime/config/deploymentConfig.h>
#include <tensorrt_edgellm/runtime/exec/tensorMap.h>
#include <tensorrt_edgellm/runtime/hybridCacheManager.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/state/pipelineIO.h>
#include <tensorrt_edgellm/runtime/state/sharedResources.h>

#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>

namespace trt_edgellm
{
namespace exampleUtils
{
//! Output capacity for prefill plus one complete round of the resolved deployment.
inline int64_t fixedRoundOutputBudget(rt::DeploymentConfig const& deployment, int32_t forcedAcceptanceLength)
{
    int64_t const perRound
        = forcedAcceptanceLength > 0 ? forcedAcceptanceLength : deployment.maxAcceptedTokensPerRound();
    return (deployment.base.isDiffusionBackbone ? 0 : 1) + perRound;
}

//! Exact synthetic token workload; seed controls both input generation and sampling.
inline rt::LLMGenerationRequest makeSyntheticBenchRequest(
    int32_t batchSize, int32_t inputLength, int32_t vocabSize, uint64_t seed)
{
    if (batchSize <= 0 || inputLength <= 0 || vocabSize <= 0)
    {
        throw std::invalid_argument("Synthetic workload dimensions must be positive");
    }
    rt::LLMGenerationRequest request{};
    request.temperature = 0.0F;
    request.topP = 1.0F;
    request.topK = 1;
    request.applyChatTemplate = false;
    request.requests.resize(batchSize);
    request.preTokenizedInputIds.resize(batchSize);
    std::mt19937_64 generator(seed);
    for (auto& tokens : request.preTokenizedInputIds)
    {
        tokens.resize(inputLength);
        for (auto& token : tokens)
        {
            token = static_cast<int32_t>(generator() % static_cast<uint64_t>(vocabSize));
        }
    }
    request.samplingSeed = seed;
    return request;
}

//! Reject cache/stop policies that would invalidate the declared repeated workload.
inline void prepareBenchRequest(rt::LLMGenerationRequest& request, int64_t outputBudget, bool ignoreEos,
    bool fixedRound, std::optional<uint64_t> seed)
{
    for (auto& sequence : request.requests)
    {
        if (fixedRound && !sequence.stopStrings.empty())
        {
            throw std::invalid_argument("fixed_round does not support stop strings");
        }
        sequence.contextCacheLookupPolicy = rt::ContextCacheLookupPolicy::kBypass;
        if (seed)
        {
            sequence.samplingSeed = *seed;
        }
    }
    request.maxGenerateLength = outputBudget;
    request.ignoreEos = ignoreEos;
    request.contextCacheLookupPolicy = rt::ContextCacheLookupPolicy::kBypass;
    if (seed)
    {
        request.samplingSeed = *seed;
    }
}
} // namespace exampleUtils
} // namespace trt_edgellm
