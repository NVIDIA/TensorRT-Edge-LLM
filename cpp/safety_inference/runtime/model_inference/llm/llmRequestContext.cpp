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

#include "llmRequestContext.h"

namespace trt_edgellm::safety::llm
{

SafetyStatus RequestContext::beginRequest(TokenIdView prompt) noexcept
{
    if (mState != State::kNOT_STARTED)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "LLM request already started");
    }
    if (mRuntimeData.maxPrefillTokens <= 0 || mRuntimeData.maxContextLength <= 0 || mRuntimeData.vocabSize <= 0
        || mRuntimeData.prefillStage.empty() || mRuntimeData.decodeStage.empty()
        || mRuntimeData.prefillStage == mRuntimeData.decodeStage)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "LLM runtime data is not prepared");
    }
    if (prompt.data == nullptr || prompt.size == 0
        || prompt.size > static_cast<std::size_t>(mRuntimeData.maxPrefillTokens)
        || prompt.size > static_cast<std::size_t>(mRuntimeData.maxContextLength))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Prompt exceeds prepared capacity");
    }
    for (std::size_t i = 0; i < prompt.size; ++i)
    {
        if (prompt.data[i] < 0 || prompt.data[i] >= mRuntimeData.vocabSize)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Prompt token is outside vocabulary");
        }
    }
    mPrompt = prompt;
    mState = State::kSTARTED;
    return SafetyStatus::ok();
}

SafetyStatus RequestContext::fail(SafetyStatus status) noexcept
{
    if (mFailure.okStatus())
    {
        mFailure = status;
    }
    return mFailure;
}

SafetyStatus RequestContext::prepareLMStageInputs(LMStageId stage, LMStageContext const& context) noexcept
{
    if (!mFailure.okStatus())
    {
        return mFailure;
    }
    bool const prefilled = mState == State::kPREFILL_COMPLETE;
    LMStageId const expected = prefilled ? mRuntimeData.decodeStage : mRuntimeData.prefillStage;
    auto const* expectedContext = prefilled ? &mRuntimeData.decodeContext : &mRuntimeData.prefillContext;
    if (mState == State::kNOT_STARTED || !mPreparedStage.empty() || stage != expected || &context != expectedContext
        || (prefilled && mContextLength >= mRuntimeData.maxContextLength))
    {
        return fail(SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected LLM stage preparation"));
    }
    auto const status = prefilled
        ? mRuntimeData.operations.prepareDecodeInputs(context, mNextDecodeToken, mContextLength)
        : mRuntimeData.operations.preparePrefillInputs(context, mPrompt);
    if (!status.okStatus())
    {
        return fail(status);
    }
    mPreparedStage = stage;
    mPreparedContext = &context;
    return SafetyStatus::ok();
}

SafetyStatus RequestContext::prepareLMStageOutputs(
    LMStageId stage, LMStageOutputs const& outputs, PendingStep& step) noexcept
{
    if (!mFailure.okStatus())
    {
        return mFailure;
    }
    if (mState == State::kNOT_STARTED || mPreparedStage.empty() || stage != mPreparedStage
        || outputs.context != mPreparedContext)
    {
        return fail(SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Unexpected LLM stage output"));
    }
    int32_t token{-1};
    auto status = mRuntimeData.operations.selectGreedyTokenFromLogits(outputs, token);
    if (!status.okStatus())
    {
        return fail(status);
    }
    if (token < 0 || token >= mRuntimeData.vocabSize)
    {
        return fail(SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Sampled token is outside vocabulary"));
    }
    status = mRuntimeData.operations.commitKvCacheFromLMStage(outputs);
    if (!status.okStatus())
    {
        return fail(status);
    }
    // Prefill commits the prompt KV; its sampled token is consumed only by the next decode.
    step = {token, mState == State::kPREFILL_COMPLETE ? mContextLength + 1 : static_cast<int32_t>(mPrompt.size)};
    return SafetyStatus::ok();
}

void RequestContext::commitLMStageOutputs(PendingStep const& step) noexcept
{
    mContextLength = step.kvLength;
    mNextDecodeToken = step.selectedTokenId;
    mState = State::kPREFILL_COMPLETE;
    mPreparedStage = {};
    mPreparedContext = nullptr;
}

int64_t RequestContext::generatedTokenCapacity() const noexcept
{
    return mState != State::kNOT_STARTED
        ? static_cast<int64_t>(mRuntimeData.maxContextLength) - static_cast<int64_t>(mPrompt.size) + 1
        : 0;
}

} // namespace trt_edgellm::safety::llm
