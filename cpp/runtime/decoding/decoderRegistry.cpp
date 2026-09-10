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

#include "runtime/decoding/decoderRegistry.h"

#include "common/logger.h"
#include "runtime/decoding/blockDiffusionDecoder.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/dflashDecoder.h"
#include "runtime/decoding/dsparkDecoder.h"
#include "runtime/decoding/eagleDecoder.h"
#include "runtime/decoding/gemma4MTPDecoder.h"
#include "runtime/decoding/mtpDecoder.h"
#include "runtime/decoding/vanillaDecoder.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

bool shouldSelectDefaultDecoder(
    DecodingStrategyKind speculativeDecoderKind, LLMGenerationRequest const& request) noexcept
{
    // EAGLE verification currently assumes greedy sampling; vanilla preserves non-greedy request semantics.
    bool const nonGreedyEagleFallback = speculativeDecoderKind == DecodingStrategyKind::kEAGLE
        && shouldUseNonGreedySampling(request.temperature, request.topK, request.topP);
    return request.disableSpecDecode || nonGreedyEagleFallback;
}

DecoderRegistry::DecoderRegistry(DecodingRuntimeContext& runtime, DecoderRegistryInit init)
    : mDefaultDecoder([&runtime, &init]() -> std::unique_ptr<DecodingStrategy> {
        if (runtime.deployment.base.isDiffusionBackbone)
        {
            return std::make_unique<BlockDiffusionDecoder>(runtime, init.engineDir, init.stream);
        }
        return std::make_unique<VanillaDecoder>(runtime);
    }())
{
    if (runtime.deployment.base.isDiffusionBackbone)
    {
        LOG_INFO("Selected block_diffusion decoding strategy.");
    }
    if (init.draftingConfig.has_value())
    {
        switch (runtime.deployment.specDecodeMode())
        {
        case SpecDecodeMode::kMTP:
            mSpeculativeDecoder = std::make_unique<MTPDecoder>(runtime, *init.draftingConfig,
                std::move(init.draftExecutor), std::move(init.draftWeights), init.stream);
            break;
        case SpecDecodeMode::kEAGLE:
            mSpeculativeDecoder = std::make_unique<EagleDecoder>(runtime, init.engineDir, *init.draftingConfig,
                std::move(init.draftExecutor), std::move(init.draftWeights), init.stream);
            break;
        case SpecDecodeMode::kDFlash:
        case SpecDecodeMode::kJetSpec:
        {
            auto blockDraftConfig = dflash_utils::makeCachedBlockDraftRuntimeConfig(runtime.deployment);
            mSpeculativeDecoder = std::make_unique<DFlashDecoder>(runtime, init.engineDir, std::move(blockDraftConfig),
                std::move(init.draftExecutor), std::move(init.draftWeights), init.stream);
            break;
        }
        case SpecDecodeMode::kGemma4MTP:
            mSpeculativeDecoder = std::make_unique<Gemma4MTPDecoder>(runtime, *init.draftingConfig,
                std::move(init.draftExecutor), std::move(init.draftWeights), init.stream);
            break;
        case SpecDecodeMode::kDSpark:
            mSpeculativeDecoder = std::make_unique<DSparkDecoder>(runtime, init.engineDir, *init.draftingConfig,
                std::move(init.draftExecutor), std::move(init.draftWeights), init.stream);
            break;
        case SpecDecodeMode::kNONE:
            throw std::runtime_error("SpecDecode drafting config was set but no mode is active.");
        }
        LOG_INFO("Selected %s decoding strategy.", mSpeculativeDecoder->name());
    }
}

DecodingStrategy& DecoderRegistry::select(LLMGenerationRequest const& request) const noexcept
{
    if (!mSpeculativeDecoder || shouldSelectDefaultDecoder(mSpeculativeDecoder->kind(), request))
    {
        return *mDefaultDecoder;
    }

    return *mSpeculativeDecoder;
}

DecodingStrategy& DecoderRegistry::cachePrimingStrategy() const noexcept
{
    return mSpeculativeDecoder ? *mSpeculativeDecoder : *mDefaultDecoder;
}

bool DecoderRegistry::captureCudaGraphs(cudaStream_t stream) const
{
    bool const skipDefaultCapture
        = mSpeculativeDecoder && mSpeculativeDecoder->capabilities().ownsBaseVerificationCudaGraphs;
    if (skipDefaultCapture)
    {
        LOG_INFO("Skipping vanilla CUDA graph capture for %s speculative runtime.", mSpeculativeDecoder->name());
    }
    bool const defaultCaptureStatus
        = (!skipDefaultCapture && mDefaultDecoder) ? mDefaultDecoder->captureCudaGraphs(stream) : true;
    bool const speculativeCaptureStatus = mSpeculativeDecoder ? mSpeculativeDecoder->captureCudaGraphs(stream) : true;
    bool const captureStatus = defaultCaptureStatus && speculativeCaptureStatus;
    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for active decoding strategies.");
    }
    else
    {
        LOG_WARNING(
            "Failed to capture decoding CUDA graphs for some stages. The inference can proceed without "
            "CUDA graph capture, but at cost of performance degradation.");
    }
    return captureStatus;
}

int64_t DecoderRegistry::getRequiredContextMemorySize() const noexcept
{
    int64_t const defaultContextMemory = mDefaultDecoder ? mDefaultDecoder->getRequiredContextMemorySize() : 0;
    int64_t const speculativeContextMemory
        = mSpeculativeDecoder ? mSpeculativeDecoder->getRequiredContextMemorySize() : 0;
    return std::max(defaultContextMemory, speculativeContextMemory);
}

void DecoderRegistry::setContextMemory(Tensor& memory) const
{
    if (mDefaultDecoder)
    {
        mDefaultDecoder->setContextMemory(memory);
    }
    if (mSpeculativeDecoder)
    {
        mSpeculativeDecoder->setContextMemory(memory);
    }
}

} // namespace rt
} // namespace trt_edgellm
