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

#include "../llm/llmModelLocalData.h"
#include "../llm/llmRequestContext.h"
#include "../modelInference.h"

namespace trt_edgellm::safety
{

//! Request orchestration using Runner-selected contexts and its synchronous preparation callback.
//! SafetyRuntime checks its fatal admission gate before entry and reports any returned failure.
class InternVL3VanillaInference final : public ModelInference
{
public:
    //! Initialization-only binding to validated Runner contexts and model-local storage.
    //! Contexts and operations outlive this object; processing setup and warmup precede request admission.
    static SafetyStatus create(ModelInferenceDefinition const& definition, llm::ModelLocalData const& data,
        std::vector<LMStageContext> const& contexts, llm::RequestOperations& llmOperations,
        vlm::RequestOperations& visualOperations, std::unique_ptr<ModelInference>& output) noexcept;

    InternVL3VanillaInference(ModelInferenceDefinition const& definition, llm::RuntimeData const& llmData,
        vlm::RequestOperations& visualOperations, LMStageContext const& visualContext) noexcept
        : mDefinition(definition)
        , mLLMData(llmData)
        , mVisualOperations(visualOperations)
        , mVisualContext(visualContext)
    {
    }

    SafetyStatus run(ModelInferenceRunner& runner, ProcessingRequest const& request, TokenIdBuffer output,
        ModelInferenceResult& result) const noexcept override;

private:
    ModelInferenceDefinition const& mDefinition;
    llm::RuntimeData const mLLMData;
    vlm::RequestOperations& mVisualOperations;
    LMStageContext const& mVisualContext;
};

} // namespace trt_edgellm::safety
