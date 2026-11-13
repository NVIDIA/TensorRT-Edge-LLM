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

#include "model_inference/modelInference.h"
#include "processing/processingCapabilities.h"
#include "processing/processingCapabilityCoordinator.h"
#include "safetyGenerationTypes.h"

#include <nlohmann/json_fwd.hpp>

namespace trt_edgellm::safety
{
struct SafetyRuntimeArtifactBundle;
struct ModelInferenceDefinition;

//! Family-specific artifact parsing, chat formatting, and request shaping selected by the model recipe.
class SafetyRequestProcessing
{
public:
    virtual ~SafetyRequestProcessing() = default;
    //! Load tokenizer/template/embedding configuration and validate artifact compatibility before allocation.
    virtual SafetyStatus configure(SafetyRuntimeArtifactBundle const& artifacts, nlohmann::json const& llmConfig,
        nlohmann::json const& visualConfig) noexcept = 0;
    virtual ProcessingConfiguration const& configuration() const noexcept = 0;
    virtual ProcessingCapabilitySet capabilities() noexcept = 0;
    //! Validate all requested features before SDK work; write into prepared, bounded input storage.
    //! The returned views remain valid until the next request.
    virtual SafetyStatus prepareRequest(LLMGenerationRequest const& request, ProcessingRequest& prepared) noexcept = 0;
    //! Validate caller-owned output capacities before any request execution.
    virtual SafetyStatus validateResponse(
        LLMGenerationRequest const& request, LLMGenerationResponse const& response) const noexcept = 0;
    //! Decode text into caller-prepared storage without allocation. Called only after successful generation.
    virtual SafetyStatus formatResponse(ModelInferenceResult const& result, LLMGenerationResponse& response) noexcept
        = 0;
};

//! Internal Initialization factory for the family adapter and recipe-selected capabilities.
//! Returns nullptr when the selected implementations are unavailable; allocation failures may throw.
std::unique_ptr<SafetyRequestProcessing> createSafetyRequestProcessing(ModelInferenceDefinition const& definition);

} // namespace trt_edgellm::safety
