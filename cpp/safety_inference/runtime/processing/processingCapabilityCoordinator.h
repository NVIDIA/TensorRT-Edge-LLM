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

#include "safety_inference/embedding/embeddingCapability.h"
#include "safety_inference/image_preprocess/multimodalPreprocessorCapability.h"
#include "safety_inference/runtime/lifecycleState.h"
#include "safety_inference/runtime/model_inference/llm/llmRequestContext.h"
#include "safety_inference/runtime/model_inference/multimodal/vlmRequestContext.h"
#include "safety_inference/sampler/samplerCapability.h"
#include "safety_inference/tokenizer/tokenizerCapability.h"

namespace trt_edgellm::safety
{

struct ModelInferenceDefinition;

struct ProcessingCapabilitySet
{
    TokenizerCapability& tokenizer;
    MultimodalPreprocessorCapability& preprocessor;
    EmbeddingCapability& embedding;
    SamplerCapability& sampler;
};

class ProcessingCapabilityCoordinator final : public llm::RequestOperations, public vlm::RequestOperations
{
public:
    ProcessingCapabilityCoordinator(LifecycleState const& lifecycle, ProcessingConfiguration const& configuration,
        ProcessingCapabilitySet capabilities) noexcept;
    SafetyStatus configureStages(
        ModelInferenceDefinition const& definition, std::vector<LMStageContext> const& contexts) noexcept;
    SafetyStatus declareResources(ResourceDeclaration& declaration) noexcept;
    SafetyStatus initializeResources(PreparedResources const& resources, llm::ModelLocalData& data) noexcept;
    SafetyStatus initializeModelData(llm::ModelLocalData& data) noexcept;
    SafetyStatus prepareWarmup(LMStageContext const& context, int64_t profile) noexcept;
    SafetyStatus deinitialize() noexcept;

    SafetyStatus prepareVisualStageInputs(LMStageContext const& context, vlm::ImageInputView input) noexcept override;
    SafetyStatus prepareLlmPrefillInputs(LMStageContext const& context, LMStageOutputs const& outputs,
        std::size_t imageOffset, std::size_t imageTokenCount) noexcept override;
    SafetyStatus preparePrefillInputs(LMStageContext const& context, TokenIdView prompt) noexcept override;
    SafetyStatus prepareDecodeInputs(LMStageContext const& context, int32_t token, int32_t kvLength) noexcept override;
    SafetyStatus selectGreedyTokenFromLogits(LMStageOutputs const& outputs, int32_t& token) noexcept override;
    SafetyStatus commitKvCacheFromLMStage(LMStageOutputs const& outputs) noexcept override;

private:
    SafetyStatus validateStages() const noexcept;
    SafetyStatus initializeComponents(PreparedResources const& resources, llm::ModelLocalData& data) noexcept;
    SafetyStatus rollbackInitialization(SafetyStatus failure) noexcept;
    bool requestOperationsReady() const noexcept;

    LifecycleState const& mLifecycle;
    ProcessingConfiguration const& mConfiguration;
    ProcessingCapabilitySet mCapabilities;
    LMStageContext const* mVisual{nullptr};
    LMStageContext const* mPrefill{nullptr};
    LMStageContext const* mDecode{nullptr};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
    bool mModelDataInitialized{false};
};

} // namespace trt_edgellm::safety
