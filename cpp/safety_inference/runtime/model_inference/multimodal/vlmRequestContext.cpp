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

#include "vlmRequestContext.h"

namespace trt_edgellm::safety::vlm
{

SafetyStatus RequestContext::beginRequest(ImageInputView input, std::size_t imageTokenOffset,
    std::size_t imageTokenCount, std::size_t promptTokenCount) noexcept
{
    if (mState != State::kNOT_STARTED)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "VLM request already started");
    }
    bool const hasImage = input.data != nullptr && input.sizeInBytes != 0;
    if (imageTokenOffset > promptTokenCount || imageTokenCount > promptTokenCount - imageTokenOffset
        || (imageTokenCount != 0 && !hasImage)
        || (imageTokenCount == 0 && (input.data != nullptr || input.sizeInBytes != 0 || imageTokenOffset != 0)))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid visual input or prompt span");
    }
    mInput = input;
    mImageTokenOffset = imageTokenOffset;
    mImageTokenCount = imageTokenCount;
    mState = State::kSTARTED;
    return SafetyStatus::ok();
}

SafetyStatus RequestContext::prepareVisualStageInputs(LMStageContext const& context) noexcept
{
    if (!mFailure.okStatus())
    {
        return mFailure;
    }
    if (mState != State::kSTARTED || !hasVisualInput())
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected visual stage preparation");
    }
    mFailure = mOperations.prepareVisualStageInputs(context, mInput);
    if (mFailure.okStatus())
    {
        mState = State::kVISUAL_PREPARED;
        mVisualContext = &context;
    }
    return mFailure;
}

SafetyStatus RequestContext::acceptVisualStageOutputs(LMStageOutputs const& outputs) noexcept
{
    if (!mFailure.okStatus())
    {
        return mFailure;
    }
    if (mState != State::kVISUAL_PREPARED || outputs.context != mVisualContext)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected visual stage output");
    }
    mVisualOutputs = outputs;
    mState = State::kVISUAL_READY;
    return SafetyStatus::ok();
}

SafetyStatus RequestContext::prepareLlmPrefillInputs(LMStageContext const& context) noexcept
{
    if (!mFailure.okStatus())
    {
        return mFailure;
    }
    auto const expectedState = hasVisualInput() ? State::kVISUAL_READY : State::kSTARTED;
    if (mState != expectedState)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Visual output is not ready for prefill");
    }
    if (hasVisualInput())
    {
        mFailure = mOperations.prepareLlmPrefillInputs(context, mVisualOutputs, mImageTokenOffset, mImageTokenCount);
    }
    if (mFailure.okStatus())
    {
        mState = State::kHANDOFF_COMPLETE;
    }
    return mFailure;
}

} // namespace trt_edgellm::safety::vlm
