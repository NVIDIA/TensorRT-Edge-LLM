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

#include "greedySamplerCapability.h"

#include "greedyArgmaxKernel.cuh"

#include "safety_inference/runtime/model_inference/llm/llmModelLocalData.h"
#include "safety_inference/runtime/safetyStatusUtils.h"
#include "safety_inference/trt_runner/lmStageContext.h"

#include <algorithm>

namespace trt_edgellm::safety
{
namespace
{
constexpr RuntimeResourceKey kHOST_RESULT{400};
constexpr RuntimeResourceKey kDEVICE_RESULT{401};
constexpr RuntimeResourceKey kCOMPLETION_EVENT{402};

RuntimeTensorView const* findBinding(llm::StageResources const& stage, char const* name) noexcept
{
    auto const found = std::find_if(stage.bindings.begin(), stage.bindings.end(), [name](auto const* binding) {
        return binding != nullptr && binding->desc != nullptr && binding->desc->name == name;
    });
    return found == stage.bindings.end() ? nullptr : *found;
}

bool validLogits(RuntimeTensorView const* view, int32_t vocabularySize) noexcept
{
    if (view == nullptr || view->desc == nullptr || view->memory.data == nullptr
        || view->memory.kind != RuntimeMemoryKind::kDEVICE || view->desc->dataType != nvinfer1::DataType::kFLOAT
        || view->desc->ioMode != nvinfer1::TensorIOMode::kOUTPUT || view->desc->profiles.size() != 1)
    {
        return false;
    }
    auto const& shape = view->desc->profiles.front().shape;
    return shape.nbDims >= 2 && shape.d[shape.nbDims - 1] == vocabularySize
        && view->memory.sizeInBytes >= static_cast<std::size_t>(vocabularySize) * sizeof(float);
}
} // namespace

SafetyStatus GreedySamplerCapability::declareResources(
    int32_t vocabularySize, ResourceDeclaration& declaration) noexcept
{
    if (mState != CapabilityPreparationState::kEMPTY || vocabularySize <= 0)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid sampler declaration");
    }
    try
    {
        declaration.memory.push_back({kHOST_RESULT, sizeof(ArgmaxPair), RuntimeMemoryKind::kPINNED});
        declaration.memory.push_back({kDEVICE_RESULT, sizeof(ArgmaxPair), RuntimeMemoryKind::kDEVICE});
        declaration.events.push_back(kCOMPLETION_EVENT);
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot declare sampler resources");
    }
    mVocabularySize = vocabularySize;
    mState = CapabilityPreparationState::kDECLARED;
    return SafetyStatus::ok();
}

SafetyStatus GreedySamplerCapability::initialize(PreparedResources const& prepared, llm::ModelLocalData& data,
    LMStageContext const& prefill, LMStageContext const& decode) noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED || prefill.mainStream == nullptr
        || prefill.mainStream != decode.mainStream || data.spec.vocabSize != mVocabularySize)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid sampler contexts");
    }
    auto status = prepared.memory(kHOST_RESULT, mHostResult);
    if (status.okStatus())
    {
        status = prepared.memory(kDEVICE_RESULT, mDeviceResult);
    }
    if (status.okStatus())
    {
        status = prepared.event(kCOMPLETION_EVENT, mCompletion);
    }
    if (!status.okStatus())
    {
        return status;
    }
    mPrefillLogits = findBinding(data.prefill, "logits");
    mDecodeLogits = findBinding(data.decode, "logits");
    if (!validLogits(mPrefillLogits, mVocabularySize) || !validLogits(mDecodeLogits, mVocabularySize)
        || mHostResult.kind != RuntimeMemoryKind::kPINNED || mDeviceResult.kind != RuntimeMemoryKind::kDEVICE
        || mCompletion == nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid prepared sampler resources");
    }
    mPrefillContext = &prefill;
    mDecodeContext = &decode;
    mStream = prefill.mainStream;
    mState = CapabilityPreparationState::kREADY;
    return SafetyStatus::ok();
}

SafetyStatus GreedySamplerCapability::select(LMStageOutputs const& outputs, int32_t& token) noexcept
{
    token = -1;
    RuntimeTensorView const* view = outputs.context == mPrefillContext ? mPrefillLogits
        : outputs.context == mDecodeContext                            ? mDecodeLogits
                                                                       : nullptr;
    if (mState != CapabilityPreparationState::kREADY || view == nullptr)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Sampler received an unknown output context");
    }
    auto const* logits = static_cast<float const*>(view->memory.data);
    auto status = launchGreedyArgmax(logits, mVocabularySize, static_cast<ArgmaxPair*>(mDeviceResult.data), mStream);
    if (status == cudaSuccess)
    {
        status = cudaMemcpyAsync(
            mHostResult.data, mDeviceResult.data, sizeof(ArgmaxPair), cudaMemcpyDeviceToHost, mStream);
    }
    if (status == cudaSuccess)
    {
        status = cudaEventRecord(mCompletion, mStream);
    }
    if (status == cudaSuccess)
    {
        status = cudaEventSynchronize(mCompletion);
    }
    if (status != cudaSuccess)
    {
        return fromCudaError(status, "greedy sampling");
    }
    auto const result = *static_cast<ArgmaxPair const*>(mHostResult.data);
    if (result.valid == 0 || result.index < 0 || result.index >= mVocabularySize)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Sampler logits contain no ordered value");
    }
    token = result.index;
    return SafetyStatus::ok();
}

SafetyStatus GreedySamplerCapability::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    mHostResult = {};
    mDeviceResult = {};
    mPrefillLogits = nullptr;
    mDecodeLogits = nullptr;
    mPrefillContext = nullptr;
    mDecodeContext = nullptr;
    mStream = nullptr;
    mCompletion = nullptr;
    mVocabularySize = 0;
    mState = CapabilityPreparationState::kCLOSED;
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
