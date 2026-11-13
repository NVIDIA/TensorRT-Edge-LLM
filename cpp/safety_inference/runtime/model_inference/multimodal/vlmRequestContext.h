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

#include "safety_inference/runtime/model_inference/modelInferenceTypes.h"
#include "safety_inference/runtime/safetyStatus.h"

namespace trt_edgellm::safety::vlm
{

//! Preprocessed image storage borrowed for this request.
struct ImageInputView
{
    void const* data{nullptr};
    std::size_t sizeInBytes{0};
    int32_t width{0};
    int32_t height{0};
    std::size_t rowStride{0};
};

class RequestOperations
{
public:
    virtual ~RequestOperations() = default;
    virtual SafetyStatus prepareVisualStageInputs(LMStageContext const& context, ImageInputView input) noexcept = 0;
    //! Copies the visual output into the reserved prompt span using prepared bindings and streams.
    virtual SafetyStatus prepareLlmPrefillInputs(LMStageContext const& context, LMStageOutputs const& visualOutputs,
        std::size_t imageTokenOffset, std::size_t imageTokenCount) noexcept = 0;
};

class RequestContext
{
public:
    explicit RequestContext(RequestOperations& operations) noexcept
        : mOperations(operations)
    {
    }
    RequestContext(RequestContext const&) = delete;
    RequestContext& operator=(RequestContext const&) = delete;

    SafetyStatus beginRequest(ImageInputView input, std::size_t imageTokenOffset, std::size_t imageTokenCount,
        std::size_t promptTokenCount) noexcept;
    SafetyStatus prepareVisualStageInputs(LMStageContext const& context) noexcept;
    SafetyStatus acceptVisualStageOutputs(LMStageOutputs const& outputs) noexcept;
    SafetyStatus prepareLlmPrefillInputs(LMStageContext const& context) noexcept;
    bool hasVisualInput() const noexcept
    {
        return mImageTokenCount != 0;
    }

private:
    enum class State
    {
        kNOT_STARTED,
        kSTARTED,
        kVISUAL_PREPARED,
        kVISUAL_READY,
        kHANDOFF_COMPLETE,
    };

    RequestOperations& mOperations;
    ImageInputView mInput{};
    std::size_t mImageTokenOffset{0};
    std::size_t mImageTokenCount{0};
    LMStageOutputs mVisualOutputs{};
    LMStageContext const* mVisualContext{nullptr};
    State mState{State::kNOT_STARTED};
    SafetyStatus mFailure{SafetyStatus::ok()};
};

} // namespace trt_edgellm::safety::vlm
