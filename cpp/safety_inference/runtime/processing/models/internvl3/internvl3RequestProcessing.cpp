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

#include "internvl3RequestProcessing.h"

#include "safety_inference/image_preprocess/models/internvl3/internvlImagePlan.h"
#include "safety_inference/runtime/safetyRuntime.h"

#include <cmath>
#include <limits>
#include <new>
#include <nlohmann/json.hpp>

namespace trt_edgellm::safety
{
namespace
{
SafetyStatus invalid(char const* message) noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, message);
}

int32_t squareDimension(nlohmann::json const& config, char const* name)
{
    auto const& value = config.at(name);
    int32_t const first = value.is_array() ? value.at(0).get<int32_t>() : value.get<int32_t>();
    int32_t const second = value.is_array() ? value.at(1).get<int32_t>() : first;
    return first == second ? first : -1;
}

int32_t optionalTokenId(nlohmann::json const& config, char const* name, int32_t fallback)
{
    auto const found = config.find(name);
    if (found == config.end())
    {
        return fallback;
    }
    if (!found->is_number_integer())
    {
        throw std::invalid_argument("Tokenizer ID must be a scalar integer");
    }
    return found->get<int32_t>();
}

SafetyStatus readStopTokens(nlohmann::json const& llm, TokenizerArtifactSpec& output) noexcept
{
    try
    {
        auto const& value = llm.at("eos_token_id");
        std::size_t const count = value.is_array() ? value.size() : 1;
        if (count == 0 || count > output.stopTokenIds.size())
        {
            return invalid("Unsupported number of InternVL stop tokens");
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            auto const& token = value.is_array() ? value.at(index) : value;
            output.stopTokenIds[index] = token.get<int32_t>();
            if (output.stopTokenIds[index] < 0)
            {
                return invalid("InternVL stop token is out of range");
            }
        }
        output.stopTokenCount = count;
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return invalid("Cannot parse InternVL stop tokens");
    }
}

SafetyStatus configureTokens(
    nlohmann::json const& llm, nlohmann::json const& visual, TokenizerArtifactSpec& output) noexcept
{
    auto status = readStopTokens(llm, output);
    if (!status.okStatus())
    {
        return status;
    }
    try
    {
        output.padTokenId = optionalTokenId(llm, "pad_token_id", output.stopTokenIds[0]);
        output.imageContextId = optionalTokenId(llm, "image_token_id", optionalTokenId(visual, "image_token_id", -1));
        output.imageStartId = optionalTokenId(visual, "img_start_token_id", -1);
        output.imageEndId = optionalTokenId(visual, "img_end_token_id", -1);
        return output.padTokenId >= 0 && output.imageContextId >= 0 ? SafetyStatus::ok()
                                                                    : invalid("InternVL tokenizer IDs are incomplete");
    }
    catch (...)
    {
        return invalid("Cannot parse InternVL tokenizer IDs");
    }
}

SafetyStatus configureTextLimits(nlohmann::json const& llm, ProcessingLimits& output) noexcept
{
    try
    {
        auto const& profile = llm.at("builder_config").at("profiles").at(0);
        output.maxPromptTokens = profile.at("input_seq_len").get<std::size_t>();
        output.maxContextTokens = profile.at("kv_cache_capacity").get<std::size_t>();
        output.maxGeneratedTokens = output.maxContextTokens;
        output.vocabularySize = llm.at("vocab_size").get<int32_t>();
        output.hiddenSize = llm.at("hidden_size").get<int32_t>();
        output.ropeTheta = llm.value("rope_theta", 1000000.0F);
        if (output.maxPromptTokens == 0 || output.maxContextTokens == 0 || output.vocabularySize <= 0
            || output.hiddenSize <= 0 || output.maxPromptTokens > std::numeric_limits<std::size_t>::max() / 4
            || output.maxContextTokens > std::numeric_limits<std::size_t>::max() / 4)
        {
            return invalid("InternVL text limits are out of range");
        }
        output.maxPromptBytes = output.maxPromptTokens * 4;
        output.maxResponseBytes = output.maxContextTokens * 4;
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return invalid("Cannot derive InternVL text limits");
    }
}

SafetyStatus configureImageLimits(nlohmann::json const& visual, ImageProcessingLimits& output) noexcept
{
    try
    {
        auto const& profile = visual.at("builder_config").at("profiles").at(0);
        auto const& vision = visual.at("vision_config");
        int32_t const tile = squareDimension(vision, "image_size");
        int32_t const patch = squareDimension(vision, "patch_size");
        double const ratio = visual.at("downsample_ratio").get<double>();
        int32_t const patches = tile > 0 && patch > 0 && tile % patch == 0 ? tile / patch : -1;
        double const rawTokens
            = patches > 0 ? static_cast<double>(patches) * static_cast<double>(patches) * ratio * ratio : -1.0;
        if (!std::isfinite(rawTokens) || rawTokens > std::numeric_limits<int32_t>::max())
        {
            return invalid("Unsupported InternVL visual configuration");
        }
        int32_t const tokensPerTile = static_cast<int32_t>(std::llround(rawTokens));
        int32_t const imageTokens = profile.at("image_tokens").get<int32_t>();
        if (tile != kInternVLTileSize || ratio <= 0.0 || std::abs(rawTokens - tokensPerTile) > 1e-6
            || tokensPerTile <= 0 || imageTokens <= 0 || imageTokens % tokensPerTile != 0)
        {
            return invalid("Unsupported InternVL visual configuration");
        }
        int32_t const profileTiles = imageTokens / tokensPerTile;
        int32_t const mainTiles = profileTiles > 2 ? profileTiles - 1 : profileTiles;
        if (profileTiles <= 0 || profileTiles > kMaxInternVLTiles
            || mainTiles > std::numeric_limits<int32_t>::max() / tile)
        {
            return invalid("InternVL tile profile is out of range");
        }
        output = {mainTiles * tile, mainTiles * tile, mainTiles, mainTiles, profileTiles, tokensPerTile,
            static_cast<std::size_t>(imageTokens), profileTiles > 2, 1.0F / 255.0F, {0.485F, 0.456F, 0.406F},
            {0.229F, 0.224F, 0.225F}};
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return invalid("Cannot derive InternVL image limits");
    }
}

SafetyStatus configureLimits(
    nlohmann::json const& llm, nlohmann::json const& visual, ProcessingConfiguration& output) noexcept
{
    auto status = configureTextLimits(llm, output.limits);
    if (status.okStatus())
    {
        status = configureImageLimits(visual, output.image);
    }
    return status;
}
} // namespace

InternVL3RequestProcessing::InternVL3RequestProcessing(std::unique_ptr<TokenizerCapability> tokenizer,
    std::unique_ptr<MultimodalPreprocessorCapability> preprocessor, std::unique_ptr<EmbeddingCapability> embedding,
    std::unique_ptr<SamplerCapability> sampler) noexcept
    : mTokenizer(std::move(tokenizer))
    , mPreprocessor(std::move(preprocessor))
    , mEmbedding(std::move(embedding))
    , mSampler(std::move(sampler))
{
}

SafetyStatus InternVL3RequestProcessing::configure(SafetyRuntimeArtifactBundle const& artifacts,
    nlohmann::json const& llmConfig, nlohmann::json const& visualConfig) noexcept
{
    if (mState != CapabilityPreparationState::kEMPTY || mTokenizer == nullptr || mPreprocessor == nullptr
        || mEmbedding == nullptr || mSampler == nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "InternVL processing is already configured");
    }
    try
    {
        ProcessingConfiguration configuration{};
        configuration.tokenizer.path = artifacts.engineDir / "tokenizer.json";
        configuration.embedding.path = artifacts.engineDir / "embedding.safetensors";
        auto status = configureLimits(llmConfig, visualConfig, configuration);
        if (status.okStatus())
        {
            status = configureTokens(llmConfig, visualConfig, configuration.tokenizer);
        }
        if (!status.okStatus())
        {
            return status;
        }
        mConfiguration = std::move(configuration);
        mState = CapabilityPreparationState::kDECLARED;
        return SafetyStatus::ok();
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot configure InternVL artifacts");
    }
    catch (...)
    {
        return invalid("Cannot configure InternVL artifact paths");
    }
}

ProcessingConfiguration const& InternVL3RequestProcessing::configuration() const noexcept
{
    return mConfiguration;
}

ProcessingCapabilitySet InternVL3RequestProcessing::capabilities() noexcept
{
    return {*mTokenizer, *mPreprocessor, *mEmbedding, *mSampler};
}

SafetyStatus InternVL3RequestProcessing::validateRequest(LLMGenerationRequest const& request) const noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED || request.requests.size() != 1
        || request.maxGenerateLength <= 0
        || static_cast<std::size_t>(request.maxGenerateLength) > mConfiguration.limits.maxGeneratedTokens)
    {
        return invalid("Invalid InternVL request");
    }
    return SafetyStatus::ok();
}

SafetyStatus InternVL3RequestProcessing::extractImage(
    LLMGenerationRequest::Request const& input, vlm::ImageInputView& image) const noexcept
{
    if (input.imageBuffers.size() != 1)
    {
        return invalid("InternVL request requires one image");
    }
    auto const& source = input.imageBuffers.front();
    if (source.data == nullptr || source.width <= 0 || source.height <= 0
        || source.width > std::numeric_limits<int32_t>::max() / kInternVLChannels)
    {
        return invalid("InternVL request requires a valid RGB image");
    }
    image = {source.data, source.sizeInBytes, source.width, source.height,
        static_cast<std::size_t>(source.width) * kInternVLChannels};
    return SafetyStatus::ok();
}

SafetyStatus InternVL3RequestProcessing::expandPrompt(
    LLMGenerationRequest::Request const& input, std::size_t imageTokenCount, ExpandedTokenResult& expanded) noexcept
{
    auto const tokens = mTokenizer->encodeMessages(input);
    if (!tokens.status.okStatus())
    {
        return tokens.status;
    }
    expanded = mTokenizer->expandImagePlaceholder(tokens.tokens, imageTokenCount);
    return expanded.status;
}

SafetyStatus InternVL3RequestProcessing::buildRequest(LLMGenerationRequest::Request const& input,
    vlm::ImageInputView image, int64_t maxGeneratedTokens, ProcessingRequest& output) noexcept
{
    auto const plan = mPreprocessor->planImage(image);
    if (!plan.status.okStatus())
    {
        return plan.status;
    }
    ExpandedTokenResult expanded{};
    auto const status = expandPrompt(input, plan.imageTokenCount, expanded);
    auto const generated = static_cast<std::size_t>(maxGeneratedTokens);
    if (!status.okStatus())
    {
        return status;
    }
    if (expanded.tokens.size > mConfiguration.limits.maxContextTokens
        || generated > mConfiguration.limits.maxContextTokens - expanded.tokens.size)
    {
        return invalid("InternVL prompt and response exceed the context capacity");
    }
    output = {expanded.tokens, image, expanded.imageOffset, plan.imageTokenCount, maxGeneratedTokens,
        mTokenizer->stopTokens()};
    return SafetyStatus::ok();
}

SafetyStatus InternVL3RequestProcessing::prepareRequest(
    LLMGenerationRequest const& request, ProcessingRequest& prepared) noexcept
{
    auto status = validateRequest(request);
    vlm::ImageInputView image{};
    if (status.okStatus())
    {
        status = extractImage(request.requests.front(), image);
    }
    ProcessingRequest candidate{};
    if (status.okStatus())
    {
        status = buildRequest(request.requests.front(), image, request.maxGenerateLength, candidate);
    }
    if (status.okStatus())
    {
        prepared = candidate;
    }
    return status;
}

SafetyStatus InternVL3RequestProcessing::validateResponse(
    LLMGenerationRequest const&, LLMGenerationResponse const& response) const noexcept
{
    return response.outputTexts.size() == 1
            && response.outputTexts.front().capacity() > mConfiguration.limits.maxResponseBytes
        ? SafetyStatus::ok()
        : invalid("InternVL text response storage is too small");
}

SafetyStatus InternVL3RequestProcessing::formatResponse(
    ModelInferenceResult const& result, LLMGenerationResponse& response) noexcept
{
    return mTextResponse.formatResponse(*mTokenizer, mConfiguration.limits.maxResponseBytes, result, response);
}

} // namespace trt_edgellm::safety
