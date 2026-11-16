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

#include "internvl3VanillaInference.h"

#include "internvl3RequestContext.h"
#include "safety_inference/runtime/model_inference/decoding/vanillaAutoregressiveStrategy.h"
#include "safety_inference/runtime/model_inference/modelInferenceCatalog.h"
#include "safety_inference/runtime/processing/processingCapabilities.h"
#include "safety_inference/trt_runner/lmStageContext.h"

#include <array>
#include <new>

namespace trt_edgellm::safety
{
namespace
{
SafetyStatus invalid() noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "InternVL request resources do not match Runner");
}

bool matches(llm::StageResources const& resources, LMStageContext const& context) noexcept
{
    if (resources.stage != context.plan.id || resources.profileId != context.plan.profiles.front()
        || resources.stream != context.mainStream || resources.bindings.empty()
        || resources.bindings.size() != context.profiles.front().tensors.size())
    {
        return false;
    }
    for (auto const* binding : resources.bindings)
    {
        if (binding == nullptr || binding->desc == nullptr || binding->memory.data == nullptr)
        {
            return false;
        }
        SafetyTensorView const* tensor{nullptr};
        auto const status = context.getTensorView(binding->desc->name, tensor);
        if (!status.okStatus() || tensor == nullptr || tensor->data != binding->memory.data)
        {
            return false;
        }
    }
    return true;
}

using InternVLStrategy = VanillaAutoregressiveStrategy<llm::RequestContext>;

struct RequiredStages
{
    LMStageDefinition const* visual{nullptr};
    LMStageDefinition const* prefill{nullptr};
    LMStageDefinition const* decode{nullptr};
};

SafetyStatus resolveRequiredStages(
    ModelInferenceDefinition const& definition, llm::RuntimeData const& data, RequiredStages& stages) noexcept
{
    stages = {definition.findStage(LMStageKind::kVisualEncoder), definition.findStage(LMStageKind::kLlmPrefill),
        definition.findStage(LMStageKind::kLlmDecode)};
    if (stages.visual == nullptr || stages.prefill == nullptr || stages.decode == nullptr)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Model definition is missing required stages");
    }
    return data.prefillStage == stages.prefill->id && data.decodeStage == stages.decode->id
        ? SafetyStatus::ok()
        : SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "LLM data does not match model definition");
}

struct StagePreparation
{
    InternVL3RequestContext& requestContext;
    LMStageDefinition const& stage;
    LMStageContext const& expectedContext;
    bool invoked{false};
    SafetyStatus status{SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Missing input preparation")};
};

SafetyStatus prepareStage(LMStageContext const& selected, void* userData) noexcept
{
    auto& state = *static_cast<StagePreparation*>(userData);
    bool const alreadyInvoked = state.invoked;
    state.invoked = true;
    if (alreadyInvoked || &selected != &state.expectedContext)
    {
        state.status = SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected stage context");
    }
    else
    {
        state.status = state.requestContext.prepareLMStageInputs(state.stage, selected);
    }
    return state.status;
}

SafetyStatus validateStageOutput(LMStageOutputs const& outputs, LMStageContext const& expected) noexcept
{
    return outputs.context == &expected
        ? SafetyStatus::ok()
        : SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected stage output context");
}

SafetyStatus executeStage(ModelInferenceRunner& runner, InternVL3RequestContext& context,
    LMStageDefinition const& stage, LMStageContext const& expected, LMStageOutputs& outputs) noexcept
{
    StagePreparation preparation{context, stage, expected};
    auto status = runner.runLMStage(stage.id, stage.resourceSlot, outputs, prepareStage, &preparation);
    if (status.okStatus())
    {
        status = preparation.status;
    }
    if (status.okStatus())
    {
        status = validateStageOutput(outputs, expected);
    }
    return status;
}

SafetyStatus beginRequest(ProcessingRequest const& request, llm::RequestContext& llm, vlm::RequestContext& visual,
    InternVLStrategy& strategy) noexcept
{
    auto status = llm.beginRequest(request.prompt);
    if (status.okStatus())
    {
        status = visual.beginRequest(
            request.image, request.imageTokenOffset, request.imageTokenCount, request.prompt.size);
    }
    if (status.okStatus())
    {
        status
            = strategy.beginRequest(request.maxGeneratedTokens, request.stopTokenIds.data, request.stopTokenIds.size);
    }
    return status;
}

SafetyStatus runVisualStage(ModelInferenceRunner& runner, InternVL3RequestContext& context, vlm::RequestContext& visual,
    LMStageDefinition const& stage, LMStageContext const& expected) noexcept
{
    if (!visual.hasVisualInput())
    {
        return SafetyStatus::ok();
    }
    LMStageOutputs outputs{};
    auto status = executeStage(runner, context, stage, expected, outputs);
    if (status.okStatus())
    {
        status = visual.acceptVisualStageOutputs(outputs);
    }
    return status;
}

SafetyStatus runLanguageStages(ModelInferenceRunner& runner, InternVL3RequestContext& context,
    InternVLStrategy& strategy, LMStageDefinition const& prefill, LMStageDefinition const& decode,
    llm::RuntimeData const& data) noexcept
{
    SafetyStatus status = SafetyStatus::ok();
    while (status.okStatus() && strategy.hasNextLMStage())
    {
        auto const& stage = strategy.nextLMStage() == prefill.id ? prefill : decode;
        auto const& expected = stage.kind == LMStageKind::kLlmPrefill ? data.prefillContext : data.decodeContext;
        LMStageOutputs outputs{};
        status = executeStage(runner, context, stage, expected, outputs);
        if (status.okStatus())
        {
            status = strategy.acceptLMStageOutputs(outputs);
        }
    }
    return status;
}

SafetyStatus executeRequest(ModelInferenceRunner& runner, ProcessingRequest const& request,
    InternVL3RequestContext& context, llm::RequestContext& llm, vlm::RequestContext& visual, InternVLStrategy& strategy,
    RequiredStages const& stages, LMStageContext const& visualContext, llm::RuntimeData const& data) noexcept
{
    auto status = beginRequest(request, llm, visual, strategy);
    if (status.okStatus())
    {
        status = runVisualStage(runner, context, visual, *stages.visual, visualContext);
    }
    if (status.okStatus())
    {
        status = runLanguageStages(runner, context, strategy, *stages.prefill, *stages.decode, data);
    }
    return status;
}

SafetyStatus finishRequest(SafetyStatus status, InternVLStrategy& strategy, llm::RequestContext const& llm,
    ModelInferenceResult& result) noexcept
{
    if (!status.okStatus())
    {
        status = strategy.abortRequest(status);
    }
    result = {strategy.generatedTokenIds(), strategy.finishReason(), llm.contextLength()};
    return status;
}
} // namespace

SafetyStatus InternVL3VanillaInference::create(ModelInferenceDefinition const& definition,
    llm::ModelLocalData const& data, std::vector<LMStageContext> const& contexts, llm::RequestOperations& llmOperations,
    vlm::RequestOperations& visualOperations, std::unique_ptr<ModelInference>& output) noexcept
{
    if (output != nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "InternVL inference already exists");
    }
    constexpr std::size_t kSTAGE_COUNT{3};
    std::array<LMStageDefinition const*, kSTAGE_COUNT> const stages{definition.findStage(LMStageKind::kVisualEncoder),
        definition.findStage(LMStageKind::kLlmPrefill), definition.findStage(LMStageKind::kLlmDecode)};
    std::array<LMStageContext const*, kSTAGE_COUNT> selected{};
    if (definition.id != ModelInferenceId::kInternVL3_1B_Vlm_Vanilla
        || definition.family != SafetyModelFamily::kInternVL3_1B_Vlm
        || definition.strategy != SafetyDecodingStrategyKind::kVanillaAutoregressive || contexts.size() != kSTAGE_COUNT
        || definition.stageCount != kSTAGE_COUNT || stages[0] == nullptr || stages[1] == nullptr || stages[2] == nullptr
        || data.spec.vocabSize <= 0 || data.maxPrefillTokens <= 0 || data.maxContextLength < data.maxPrefillTokens)
    {
        return invalid();
    }
    for (std::size_t index = 0; index < kSTAGE_COUNT; ++index)
    {
        auto const& stage = *stages[index];
        for (auto const& context : contexts)
        {
            if (context.plan.id == stage.id && context.plan.slot == stage.resourceSlot)
            {
                if (selected[index] != nullptr || !context.complete || context.mainStream == nullptr
                    || context.plan.executorIndex != static_cast<std::size_t>(stage.executorId)
                    || context.plan.profiles.size() != 1 || context.plan.profiles.front() != stage.profileId
                    || context.profiles.size() != 1)
                {
                    return invalid();
                }
                selected[index] = &context;
            }
        }
        if (selected[index] == nullptr || selected[index]->mainStream != selected[0]->mainStream)
        {
            return invalid();
        }
    }
    // The catalog fixes visual/prefill/decode order and a shared main stream.
    auto const& visual = *selected[0];
    auto const& prefill = *selected[1];
    auto const& decode = *selected[2];
    if (!matches(data.prefill, prefill) || !matches(data.decode, decode))
    {
        return invalid();
    }
    try
    {
        llm::RuntimeData runtimeData{llmOperations, stages[1]->id, stages[2]->id, prefill, decode,
            data.maxPrefillTokens, data.maxContextLength, data.spec.vocabSize};
        output = std::make_unique<InternVL3VanillaInference>(definition, runtimeData, visualOperations, visual);
        return SafetyStatus::ok();
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "InternVL inference allocation failed");
    }
    catch (...)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Unexpected InternVL inference setup failure");
    }
}

SafetyStatus InternVL3VanillaInference::run(ModelInferenceRunner& runner, ProcessingRequest const& request,
    TokenIdBuffer output, ModelInferenceResult& result) const noexcept
{
    result = {};
    RequiredStages stages{};
    auto status = resolveRequiredStages(mDefinition, mLLMData, stages);
    if (!status.okStatus())
    {
        return status;
    }
    llm::RequestContext llm{mLLMData};
    vlm::RequestContext visual{mVisualOperations};
    InternVL3RequestContext context{llm, visual};
    InternVLStrategy strategy{llm, stages.prefill->id, stages.decode->id, output};
    status = executeRequest(runner, request, context, llm, visual, strategy, stages, mVisualContext, mLLMData);
    return finishRequest(status, strategy, llm, result);
}

} // namespace trt_edgellm::safety
