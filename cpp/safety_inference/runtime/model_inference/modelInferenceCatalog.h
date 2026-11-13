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

namespace trt_edgellm::safety
{

enum class ModelInferenceId
{
    kInternVL3_1B_Vlm_Vanilla
};
enum class SafetyModelFamily
{
    kInternVL3_1B_Vlm
};
enum class SafetyDecodingStrategyKind
{
    kVanillaAutoregressive
};
enum class LMStageKind
{
    kVisualEncoder,
    kLlmPrefill,
    kLlmDecode
};
enum class EngineRole
{
    kVisual,
    kLlm
};

struct LMStageDefinition
{
    LMStageId id;
    LMStageKind kind;
    int32_t executorId;
    int32_t profileId;
    int32_t mainStreamIndex;
    std::size_t resourceSlot;
};

struct ExecutorDefinition
{
    std::string_view name;
    EngineRole engineRole;
    std::string_view const* requiredPlugins;
    std::size_t requiredPluginCount;
};

//! Immutable logical recipe consumed during Runner and capability initialization.
struct ModelInferenceDefinition
{
    ModelInferenceId id;
    SafetyModelFamily family;
    SafetyDecodingStrategyKind strategy;
    std::string_view tokenizer;
    std::string_view multimodalPreprocessor;
    std::string_view embedding;
    std::string_view sampler;
    ExecutorDefinition const* executors;
    std::size_t executorCount;
    LMStageDefinition const* stages;
    std::size_t stageCount;
    std::size_t streamCount;
    std::size_t resourceSlotCount;

    LMStageDefinition const* findStage(LMStageKind kind) const noexcept;
};

//! Returns static definition storage, or nullptr for an unsupported composition.
ModelInferenceDefinition const* findModelInferenceDefinition(ModelInferenceId id) noexcept;

} // namespace trt_edgellm::safety
