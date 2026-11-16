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

#include "processingCapabilityCoordinator.h"

#include "safety_inference/runtime/model_inference/modelInferenceCatalog.h"
#include "safety_inference/trt_runner/lmStageContext.h"

#include <algorithm>

namespace trt_edgellm::safety
{
namespace
{
constexpr ResourceOwnerId kPROCESSING_OWNER{3};

SafetyStatus invalidState(char const* message) noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidState, message);
}

SafetyStatus retainSignificantFailure(SafetyStatus current, SafetyStatus next) noexcept
{
    if (next.okStatus() || (!current.okStatus() && (!next.fatalStatus() || current.fatalStatus())))
    {
        return current;
    }
    return next;
}

LMStageContext const* findContext(std::vector<LMStageContext> const& contexts, LMStageId id) noexcept
{
    auto const found
        = std::find_if(contexts.begin(), contexts.end(), [id](auto const& context) { return context.plan.id == id; });
    return found == contexts.end() ? nullptr : &*found;
}
} // namespace

ProcessingCapabilityCoordinator::ProcessingCapabilityCoordinator(LifecycleState const& lifecycle,
    ProcessingConfiguration const& configuration, ProcessingCapabilitySet capabilities) noexcept
    : mLifecycle(lifecycle)
    , mConfiguration(configuration)
    , mCapabilities(capabilities)
{
}

SafetyStatus ProcessingCapabilityCoordinator::configureStages(
    ModelInferenceDefinition const& definition, std::vector<LMStageContext> const& contexts) noexcept
{
    auto const* visual = definition.findStage(LMStageKind::kVisualEncoder);
    auto const* prefill = definition.findStage(LMStageKind::kLlmPrefill);
    auto const* decode = definition.findStage(LMStageKind::kLlmDecode);
    if (mState != CapabilityPreparationState::kEMPTY || visual == nullptr || prefill == nullptr || decode == nullptr)
    {
        return invalidState("Processing stages are unavailable");
    }
    mVisual = findContext(contexts, visual->id);
    mPrefill = findContext(contexts, prefill->id);
    mDecode = findContext(contexts, decode->id);
    return validateStages();
}

SafetyStatus ProcessingCapabilityCoordinator::validateStages() const noexcept
{
    if (mVisual == nullptr || mPrefill == nullptr || mDecode == nullptr || mVisual->mainStream == nullptr
        || mVisual->mainStream != mPrefill->mainStream || mPrefill->mainStream != mDecode->mainStream
        || mVisual->profiles.size() != 1 || mPrefill->profiles.size() != 1 || mDecode->profiles.size() != 1)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Processing stages must share one stream");
    }
    return SafetyStatus::ok();
}

SafetyStatus ProcessingCapabilityCoordinator::declareResources(ResourceDeclaration& declaration) noexcept
{
    if (mLifecycle != LifecycleState::kINITIALIZATION || mState != CapabilityPreparationState::kEMPTY
        || !validateStages().okStatus())
    {
        return invalidState("Processing resource declaration is not available");
    }
    declaration.owner = kPROCESSING_OWNER;
    auto status
        = mCapabilities.tokenizer.declareResources(mConfiguration.tokenizer, mConfiguration.limits, declaration);
    if (status.okStatus())
    {
        status = mCapabilities.preprocessor.declareResources(mConfiguration.image, declaration);
    }
    if (status.okStatus())
    {
        status = mCapabilities.embedding.declareResources(mConfiguration.embedding, mConfiguration.limits, declaration);
    }
    if (status.okStatus())
    {
        status = mCapabilities.sampler.declareResources(mConfiguration.limits.vocabularySize, declaration);
    }
    mState = status.okStatus() ? CapabilityPreparationState::kDECLARED : mState;
    return status;
}

SafetyStatus ProcessingCapabilityCoordinator::initializeComponents(
    PreparedResources const& resources, llm::ModelLocalData& data) noexcept
{
    auto status = mCapabilities.tokenizer.initialize(mConfiguration.tokenizer, resources);
    if (status.okStatus())
    {
        status = mCapabilities.preprocessor.initialize(resources, *mVisual);
    }
    if (status.okStatus())
    {
        status = mCapabilities.embedding.initialize(mConfiguration.embedding, resources, *mVisual, *mPrefill, *mDecode);
    }
    if (status.okStatus())
    {
        status = mCapabilities.sampler.initialize(resources, data, *mPrefill, *mDecode);
    }
    return status;
}

SafetyStatus ProcessingCapabilityCoordinator::initializeResources(
    PreparedResources const& resources, llm::ModelLocalData& data) noexcept
{
    if (mLifecycle != LifecycleState::kINITIALIZATION || mState != CapabilityPreparationState::kDECLARED)
    {
        return invalidState("Processing resources were not declared");
    }
    auto status = resources.validateInitialization(kPROCESSING_OWNER);
    if (status.okStatus())
    {
        status = initializeComponents(resources, data);
    }
    if (!status.okStatus())
    {
        return rollbackInitialization(status);
    }
    mState = CapabilityPreparationState::kREADY;
    return SafetyStatus::ok();
}

SafetyStatus ProcessingCapabilityCoordinator::initializeModelData(llm::ModelLocalData& data) noexcept
{
    if (mLifecycle != LifecycleState::kINITIALIZATION || mState != CapabilityPreparationState::kREADY)
    {
        return invalidState("Processing model data cannot be initialized");
    }
    auto const status = mCapabilities.embedding.initializeModelData(data);
    mModelDataInitialized = status.okStatus();
    return status;
}

SafetyStatus ProcessingCapabilityCoordinator::prepareWarmup(LMStageContext const& context, int64_t profile) noexcept
{
    if (mLifecycle != LifecycleState::kINITIALIZATION || !mModelDataInitialized)
    {
        return invalidState("Processing warmup is not ready");
    }
    return mCapabilities.embedding.prepareWarmup(context, profile);
}

SafetyStatus ProcessingCapabilityCoordinator::rollbackInitialization(SafetyStatus failure) noexcept
{
    if (failure.fatalStatus())
    {
        return failure;
    }
    failure = retainSignificantFailure(failure, mCapabilities.sampler.deinitialize());
    if (failure.fatalStatus())
    {
        return failure;
    }
    failure = retainSignificantFailure(failure, mCapabilities.embedding.deinitialize());
    if (failure.fatalStatus())
    {
        return failure;
    }
    failure = retainSignificantFailure(failure, mCapabilities.preprocessor.deinitialize());
    if (failure.fatalStatus())
    {
        return failure;
    }
    return retainSignificantFailure(failure, mCapabilities.tokenizer.deinitialize());
}

SafetyStatus ProcessingCapabilityCoordinator::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    auto status = rollbackInitialization(SafetyStatus::ok());
    if (!status.fatalStatus())
    {
        mModelDataInitialized = false;
        mState = CapabilityPreparationState::kCLOSED;
    }
    return status;
}

bool ProcessingCapabilityCoordinator::requestOperationsReady() const noexcept
{
    return mLifecycle == LifecycleState::kOPERATIONAL && mState == CapabilityPreparationState::kREADY
        && mModelDataInitialized;
}

SafetyStatus ProcessingCapabilityCoordinator::prepareVisualStageInputs(
    LMStageContext const& context, vlm::ImageInputView input) noexcept
{
    return requestOperationsReady() ? mCapabilities.preprocessor.prepareVisualInput(context, input)
                                    : invalidState("Visual preprocessing is not ready");
}

SafetyStatus ProcessingCapabilityCoordinator::prepareLlmPrefillInputs(LMStageContext const& context,
    LMStageOutputs const& outputs, std::size_t imageOffset, std::size_t imageTokenCount) noexcept
{
    return requestOperationsReady()
        ? mCapabilities.embedding.overlayVisualEmbeddings(context, outputs, imageOffset, imageTokenCount)
        : invalidState("Visual embedding handoff is not ready");
}

SafetyStatus ProcessingCapabilityCoordinator::preparePrefillInputs(
    LMStageContext const& context, TokenIdView prompt) noexcept
{
    return requestOperationsReady() ? mCapabilities.embedding.embedPrompt(context, prompt)
                                    : invalidState("Prefill embedding is not ready");
}

SafetyStatus ProcessingCapabilityCoordinator::prepareDecodeInputs(
    LMStageContext const& context, int32_t token, int32_t kvLength) noexcept
{
    return requestOperationsReady() ? mCapabilities.embedding.embedDecodeToken(context, token, kvLength)
                                    : invalidState("Decode embedding is not ready");
}

SafetyStatus ProcessingCapabilityCoordinator::selectGreedyTokenFromLogits(
    LMStageOutputs const& outputs, int32_t& token) noexcept
{
    return requestOperationsReady() ? mCapabilities.sampler.select(outputs, token)
                                    : invalidState("Sampling is not ready");
}

SafetyStatus ProcessingCapabilityCoordinator::commitKvCacheFromLMStage(LMStageOutputs const& outputs) noexcept
{
    if (!requestOperationsReady() || (outputs.context != mPrefill && outputs.context != mDecode))
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unknown KV-cache output context");
    }
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
