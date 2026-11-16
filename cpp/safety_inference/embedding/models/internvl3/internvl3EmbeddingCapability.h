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

#include "safety_inference/embedding/embeddingCapability.h"
#include "safety_inference/runtime/model_inference/llm/llmModelLocalData.h"

#include <cuda_fp16.h>

namespace trt_edgellm::safety
{

class InternVL3EmbeddingCapability final : public EmbeddingCapability
{
public:
    SafetyStatus declareResources(EmbeddingArtifactSpec const& artifact, ProcessingLimits const& limits,
        ResourceDeclaration& declaration) noexcept override;
    SafetyStatus initialize(EmbeddingArtifactSpec const& artifact, PreparedResources const& prepared,
        LMStageContext const& visual, LMStageContext const& prefill, LMStageContext const& decode) noexcept override;
    SafetyStatus embedPrompt(LMStageContext const& context, TokenIdView prompt) noexcept override;
    SafetyStatus embedDecodeToken(LMStageContext const& context, int32_t token, int32_t kvLength) noexcept override;
    SafetyStatus overlayVisualEmbeddings(LMStageContext const& context, LMStageOutputs const& outputs,
        std::size_t imageOffset, std::size_t imageTokenCount) noexcept override;
    SafetyStatus initializeModelData(llm::ModelLocalData& data) noexcept override;
    SafetyStatus prepareWarmup(LMStageContext const& context, int64_t profile) noexcept override;
    SafetyStatus deinitialize() noexcept override;

private:
    SafetyStatus embed(LMStageContext const& context, TokenIdView tokens, int32_t kvLength) noexcept;
    SafetyStatus writeControls(llm::StageResources const& stage, std::size_t tokenCount, int32_t kvLength) noexcept;
    EmbeddingArtifactSpec mArtifact{};
    ProcessingLimits mLimits{};
    RuntimeMemoryView mUpload{};
    RuntimeTensorView const* mVisualOutput{nullptr};
    llm::ModelLocalData* mData{nullptr};
    LMStageContext const* mVisualContext{nullptr};
    LMStageContext const* mPrefillContext{nullptr};
    LMStageContext const* mDecodeContext{nullptr};
    cudaStream_t mStream{nullptr};
    cudaEvent_t mInitializationCompletion{nullptr};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
};

} // namespace trt_edgellm::safety
