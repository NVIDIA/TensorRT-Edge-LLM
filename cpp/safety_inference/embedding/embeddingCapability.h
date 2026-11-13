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

#include "safety_inference/runtime/processing/processingCapabilities.h"

namespace trt_edgellm::safety
{

class EmbeddingCapability
{
public:
    virtual ~EmbeddingCapability() = default;
    virtual SafetyStatus declareResources(EmbeddingArtifactSpec const& artifact, ProcessingLimits const& limits,
        ResourceDeclaration& declaration) noexcept = 0;
    virtual SafetyStatus initialize(EmbeddingArtifactSpec const& artifact, PreparedResources const& prepared,
        LMStageContext const& visual, LMStageContext const& prefill, LMStageContext const& decode) noexcept = 0;
    virtual SafetyStatus embedPrompt(LMStageContext const& context, TokenIdView prompt) noexcept = 0;
    virtual SafetyStatus embedDecodeToken(LMStageContext const& context, int32_t token, int32_t kvLength) noexcept = 0;
    virtual SafetyStatus overlayVisualEmbeddings(LMStageContext const& context, LMStageOutputs const& outputs,
        std::size_t imageOffset, std::size_t imageTokenCount) noexcept = 0;
    virtual SafetyStatus initializeModelData(llm::ModelLocalData& data) noexcept = 0;
    virtual SafetyStatus prepareWarmup(LMStageContext const& context, int64_t profile) noexcept = 0;
    virtual SafetyStatus deinitialize() noexcept = 0;
};

} // namespace trt_edgellm::safety
