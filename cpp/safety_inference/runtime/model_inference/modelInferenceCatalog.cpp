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

#include "modelInferenceCatalog.h"

namespace trt_edgellm::safety
{
namespace
{
constexpr std::string_view kLLM_PLUGINS[]{"Attention"};
constexpr ExecutorDefinition kEXECUTORS[]{
    {"internvl3_visual_executor", EngineRole::kVisual, nullptr, 0},
    {"llm_executor", EngineRole::kLlm, kLLM_PLUGINS, 1},
};
constexpr LMStageDefinition kSTAGES[]{
    {"visual_encode", LMStageKind::kVisualEncoder, 0, 0, 0, 0},
    {"prefill", LMStageKind::kLlmPrefill, 1, 0, 0, 0},
    {"decode", LMStageKind::kLlmDecode, 1, 1, 0, 0},
};
constexpr ModelInferenceDefinition kINTERNVL3_VANILLA{ModelInferenceId::kInternVL3_1B_Vlm_Vanilla,
    SafetyModelFamily::kInternVL3_1B_Vlm, SafetyDecodingStrategyKind::kVanillaAutoregressive, "internvl3_tokenizer",
    "internvl3_multimodal_preprocessor", "fp16_vlm_embedding", "greedy_sampler", kEXECUTORS, 2, kSTAGES, 3, 1, 1};
} // namespace

LMStageDefinition const* ModelInferenceDefinition::findStage(LMStageKind kind) const noexcept
{
    LMStageDefinition const* result = nullptr;
    for (std::size_t i = 0; i < stageCount; ++i)
    {
        if (stages[i].kind == kind)
        {
            result = &stages[i];
            break;
        }
    }
    return result;
}

ModelInferenceDefinition const* findModelInferenceDefinition(ModelInferenceId id) noexcept
{
    return id == ModelInferenceId::kInternVL3_1B_Vlm_Vanilla ? &kINTERNVL3_VANILLA : nullptr;
}

} // namespace trt_edgellm::safety
