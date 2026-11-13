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

#include "safety_inference/runtime/model_inference/llm/llmRequestContext.h"
#include "safety_inference/runtime/model_inference/modelInferenceCatalog.h"
#include "safety_inference/runtime/model_inference/multimodal/vlmRequestContext.h"

namespace trt_edgellm::safety
{

//! Composes shared request operations; all referenced runtime data remains externally owned.
class InternVL3RequestContext
{
public:
    InternVL3RequestContext(llm::RequestContext& llm, vlm::RequestContext& visual) noexcept
        : mLLM(llm)
        , mVisual(visual)
    {
    }

    SafetyStatus prepareLMStageInputs(LMStageDefinition const& stage, LMStageContext const& context) noexcept
    {
        SafetyStatus status = SafetyStatus::ok();
        if (stage.kind == LMStageKind::kVisualEncoder)
        {
            status = mVisual.prepareVisualStageInputs(context);
        }
        else
        {
            status = mLLM.prepareLMStageInputs(stage.id, context);
            if (status.okStatus() && stage.kind == LMStageKind::kLlmPrefill)
            {
                status = mVisual.prepareLlmPrefillInputs(context);
            }
        }
        return status;
    }

private:
    llm::RequestContext& mLLM;
    vlm::RequestContext& mVisual;
};

} // namespace trt_edgellm::safety
