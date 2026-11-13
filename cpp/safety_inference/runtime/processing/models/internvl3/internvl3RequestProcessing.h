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

#include "safety_inference/runtime/processing/singleRequestTextResponse.h"
#include "safety_inference/runtime/safetyRequestProcessing.h"

#include <memory>

namespace trt_edgellm::safety
{

class InternVL3RequestProcessing final : public SafetyRequestProcessing
{
public:
    InternVL3RequestProcessing(std::unique_ptr<TokenizerCapability> tokenizer,
        std::unique_ptr<MultimodalPreprocessorCapability> preprocessor, std::unique_ptr<EmbeddingCapability> embedding,
        std::unique_ptr<SamplerCapability> sampler) noexcept;
    SafetyStatus configure(SafetyRuntimeArtifactBundle const& artifacts, nlohmann::json const& llmConfig,
        nlohmann::json const& visualConfig) noexcept override;
    ProcessingConfiguration const& configuration() const noexcept override;
    ProcessingCapabilitySet capabilities() noexcept override;
    SafetyStatus prepareRequest(LLMGenerationRequest const& request, ProcessingRequest& prepared) noexcept override;
    SafetyStatus validateResponse(
        LLMGenerationRequest const& request, LLMGenerationResponse const& response) const noexcept override;
    SafetyStatus formatResponse(ModelInferenceResult const& result, LLMGenerationResponse& response) noexcept override;

private:
    SafetyStatus validateRequest(LLMGenerationRequest const& request) const noexcept;
    SafetyStatus extractImage(LLMGenerationRequest::Request const& input, vlm::ImageInputView& image) const noexcept;
    SafetyStatus expandPrompt(LLMGenerationRequest::Request const& input, std::size_t imageTokenCount,
        ExpandedTokenResult& expanded) noexcept;
    SafetyStatus buildRequest(LLMGenerationRequest::Request const& input, vlm::ImageInputView image,
        int64_t maxGeneratedTokens, ProcessingRequest& output) noexcept;

    std::unique_ptr<TokenizerCapability> mTokenizer;
    std::unique_ptr<MultimodalPreprocessorCapability> mPreprocessor;
    std::unique_ptr<EmbeddingCapability> mEmbedding;
    std::unique_ptr<SamplerCapability> mSampler;
    ProcessingConfiguration mConfiguration{};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
    SingleRequestTextResponse mTextResponse{};
};

} // namespace trt_edgellm::safety
