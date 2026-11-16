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

namespace trt_edgellm::safety::llm
{

//! Operations over Initialization-prepared bindings, staging buffers, sampler and contiguous KV storage.
//! Implementations allocate nothing and finish fallible output transfers before returning success.
class RequestOperations
{
public:
    virtual ~RequestOperations() = default;
    virtual SafetyStatus preparePrefillInputs(LMStageContext const& context, TokenIdView prompt) noexcept = 0;
    virtual SafetyStatus prepareDecodeInputs(LMStageContext const& context, int32_t token, int32_t kvLength) noexcept
        = 0;
    virtual SafetyStatus selectGreedyTokenFromLogits(safety::LMStageOutputs const& outputs, int32_t& token) noexcept
        = 0;
    virtual SafetyStatus commitKvCacheFromLMStage(safety::LMStageOutputs const& outputs) noexcept = 0;
};

//! Initialized after resource preparation; contexts and operation-owned resource views outlive requests.
//! Capacities come from validated engine metadata. This view allocates and releases no resources.
struct RuntimeData
{
    RequestOperations& operations;
    LMStageId prefillStage;
    LMStageId decodeStage;
    LMStageContext const& prefillContext;
    LMStageContext const& decodeContext;
    int32_t maxPrefillTokens;
    int32_t maxContextLength;
    int32_t vocabSize;
};

class RequestContext
{
public:
    using LMStageOutputs = safety::LMStageOutputs;
    struct PendingStep
    {
        int32_t selectedTokenId{-1};
        int32_t kvLength{0};
    };

    explicit RequestContext(RuntimeData const& runtimeData) noexcept
        : mRuntimeData(runtimeData)
    {
    }
    RequestContext(RequestContext const&) = delete;
    RequestContext& operator=(RequestContext const&) = delete;

    //! Prompt and RuntimeData remain valid and unchanged throughout this request.
    SafetyStatus beginRequest(TokenIdView prompt) noexcept;
    SafetyStatus prepareLMStageInputs(LMStageId stage, LMStageContext const& context) noexcept;
    SafetyStatus prepareLMStageOutputs(LMStageId stage, LMStageOutputs const& outputs, PendingStep& step) noexcept;
    //! Called once with the successful pending step; publishes host progress only.
    void commitLMStageOutputs(PendingStep const& step) noexcept;
    int64_t generatedTokenCapacity() const noexcept;
    int32_t contextLength() const noexcept
    {
        return mContextLength;
    }

private:
    enum class State
    {
        kNOT_STARTED,
        kSTARTED,
        kPREFILL_COMPLETE,
    };

    SafetyStatus fail(SafetyStatus status) noexcept;
    RuntimeData const& mRuntimeData;
    TokenIdView mPrompt{};
    int32_t mContextLength{0};
    int32_t mNextDecodeToken{-1};
    State mState{State::kNOT_STARTED};
    LMStageId mPreparedStage{};
    LMStageContext const* mPreparedContext{nullptr};
    SafetyStatus mFailure{SafetyStatus::ok()};
};

} // namespace trt_edgellm::safety::llm
