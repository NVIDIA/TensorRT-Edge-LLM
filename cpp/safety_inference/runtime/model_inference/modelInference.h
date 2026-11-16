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

#include "modelInferenceTypes.h"

#include <memory>
#include <vector>

namespace trt_edgellm::safety
{

struct ModelInferenceDefinition;
struct ProcessingRequest;

namespace llm
{
struct ModelLocalData;
class RequestOperations;
} // namespace llm

namespace vlm
{
class RequestOperations;
} // namespace vlm

struct ModelInferenceResult
{
    TokenIdView generatedTokens{};
    FinishReason finishReason{FinishReason::kNotFinished};
    int32_t contextLength{0};
};

//! Model inference uses this narrow Runner interface to select and execute configured stages.
class ModelInferenceRunner
{
public:
    virtual ~ModelInferenceRunner() = default;
    virtual SafetyStatus runLMStage(LMStageId stage, std::size_t slot, LMStageOutputs& outputs,
        LMStagePrepareCallback prepare, void* userData) noexcept = 0;
};

//! Recipe-selected request orchestration owned by SafetyRuntime.
class ModelInference
{
public:
    virtual ~ModelInference() = default;
    virtual SafetyStatus run(ModelInferenceRunner& runner, ProcessingRequest const& request, TokenIdBuffer output,
        ModelInferenceResult& result) const noexcept = 0;
};

//! Construct request orchestration for one admitted model recipe.
SafetyStatus createModelInference(ModelInferenceDefinition const& definition, llm::ModelLocalData const& data,
    std::vector<LMStageContext> const& contexts, llm::RequestOperations& llmOperations,
    vlm::RequestOperations& visualOperations, std::unique_ptr<ModelInference>& output) noexcept;

} // namespace trt_edgellm::safety
