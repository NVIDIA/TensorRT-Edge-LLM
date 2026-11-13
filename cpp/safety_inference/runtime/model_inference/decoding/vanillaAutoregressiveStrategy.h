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

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace trt_edgellm::safety
{

//! Request-scoped autoregressive scheduling over borrowed, Initialization-prepared storage.
//! The request context and borrowed storage outlive this object.
//! DecodeState prepares a PendingStep without committing request progress, then
//! commitLMStageOutputs publishes that validated step through a non-failing operation.
template <typename DecodeState>
class VanillaAutoregressiveStrategy
{
public:
    using LMStageOutputs = typename DecodeState::LMStageOutputs;
    using PendingStep = typename DecodeState::PendingStep;

    VanillaAutoregressiveStrategy(
        DecodeState& state, LMStageId prefillStage, LMStageId decodeStage, TokenIdBuffer generatedTokens) noexcept
        : mState(state)
        , mPrefillStage(prefillStage)
        , mDecodeStage(decodeStage)
        , mGeneratedTokens(generatedTokens)
    {
        static_assert(noexcept(state.prepareLMStageOutputs(
            prefillStage, std::declval<LMStageOutputs const&>(), std::declval<PendingStep&>())));
        static_assert(noexcept(state.commitLMStageOutputs(std::declval<PendingStep const&>())));
    }

    VanillaAutoregressiveStrategy(VanillaAutoregressiveStrategy const&) = delete;
    VanillaAutoregressiveStrategy& operator=(VanillaAutoregressiveStrategy const&) = delete;

    //! Begins this object's request after the caller has begun its request context.
    //! Stop-token storage stays valid for the whole request.
    SafetyStatus beginRequest(
        int64_t generatedTokenBudget, int32_t const* stopTokenIds, std::size_t stopTokenCount) noexcept
    {
        if (mFatalError)
        {
            return mFailure;
        }
        if (mActive || mFinishReason != FinishReason::kNotFinished)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Decoding request cannot begin");
        }
        if (mPrefillStage.empty() || mDecodeStage.empty() || mPrefillStage == mDecodeStage
            || mGeneratedTokens.data == nullptr || mGeneratedTokens.capacity == 0)
        {
            return abortRequest(
                SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Decoding resources are not prepared"));
        }
        auto const stateCapacity = mState.generatedTokenCapacity();
        if (generatedTokenBudget <= 0 || stateCapacity <= 0 || (stopTokenCount != 0 && stopTokenIds == nullptr))
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid decoding request bounds");
        }
        uint64_t const budget
            = std::min(static_cast<uint64_t>(generatedTokenBudget), static_cast<uint64_t>(stateCapacity));
        mGeneratedTokenBudget
            = static_cast<std::size_t>(std::min(budget, static_cast<uint64_t>(mGeneratedTokens.capacity)));
        mStopTokenIds = stopTokenIds;
        mStopTokenCount = stopTokenCount;
        mNextStage = mPrefillStage;
        mActive = true;
        return SafetyStatus::ok();
    }

    bool hasNextLMStage() const noexcept
    {
        return mActive && !mFatalError;
    }
    LMStageId nextLMStage() const noexcept
    {
        return mNextStage;
    }

    SafetyStatus acceptLMStageOutputs(LMStageOutputs const& outputs) noexcept
    {
        if (!hasNextLMStage())
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidState, "No pending decoding stage");
        }
        if (mGeneratedTokenCount >= mGeneratedTokenBudget)
        {
            char message[128]{};
            std::snprintf(message, sizeof(message), "Generated token budget exceeded: count=%zu, budget=%zu",
                mGeneratedTokenCount, mGeneratedTokenBudget);
            return abortRequest(SafetyStatus::fatal(SafetyStatusCode::kInvalidState, message));
        }

        PendingStep step{};
        auto const status = mState.prepareLMStageOutputs(mNextStage, outputs, step);
        if (!status.okStatus())
        {
            return abortRequest(status);
        }
        int32_t const token = step.selectedTokenId;
        if (token < 0)
        {
            char message[128]{};
            std::snprintf(message, sizeof(message), "Invalid selected token: %" PRId32, token);
            return abortRequest(SafetyStatus::fatal(SafetyStatusCode::kInvalidState, message));
        }

        bool isStopToken = false;
        for (std::size_t i = 0; i < mStopTokenCount; ++i)
        {
            if (mStopTokenIds[i] == token)
            {
                isStopToken = true;
                break;
            }
        }

        // All fallible output processing finishes before either context or output progress changes.
        mState.commitLMStageOutputs(step);
        mGeneratedTokens.data[mGeneratedTokenCount++] = token;
        if (isStopToken)
        {
            mFinishReason = FinishReason::kEndId;
            mActive = false;
        }
        else if (mGeneratedTokenCount == mGeneratedTokenBudget)
        {
            mFinishReason = FinishReason::kLength;
            mActive = false;
        }
        else
        {
            mNextStage = mDecodeStage;
        }
        return SafetyStatus::ok();
    }

    //! Ends this request; the runtime Error Handler owns fatal rejection across requests.
    SafetyStatus abortRequest(SafetyStatus status) noexcept
    {
        if (mFatalError)
        {
            return mFailure;
        }
        mFailure = status.okStatus()
            ? SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Cannot abort with a successful status")
            : status;
        mFatalError = mFailure.fatalStatus();
        mActive = false;
        mFinishReason = FinishReason::kError;
        return mFailure;
    }

    TokenIdView generatedTokenIds() const noexcept
    {
        return {mGeneratedTokens.data, mGeneratedTokenCount};
    }
    FinishReason finishReason() const noexcept
    {
        return mFinishReason;
    }

private:
    DecodeState& mState;
    LMStageId mPrefillStage;
    LMStageId mDecodeStage;
    TokenIdBuffer mGeneratedTokens;
    LMStageId mNextStage{};
    int32_t const* mStopTokenIds{nullptr};
    std::size_t mStopTokenCount{0};
    std::size_t mGeneratedTokenBudget{0};
    std::size_t mGeneratedTokenCount{0};
    FinishReason mFinishReason{FinishReason::kNotFinished};
    SafetyStatus mFailure{SafetyStatus::ok()};
    bool mActive{false};
    bool mFatalError{false};
};

} // namespace trt_edgellm::safety
