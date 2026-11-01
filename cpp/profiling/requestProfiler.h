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

#include "profiling/metrics.h"
#include <chrono>
#include <optional>

namespace trt_edgellm
{
namespace metrics
{

//! Per-sequence wall timing, independent of aggregate GPU event profiling.
//! The runtime supplies cumulative counts at host-visible phase boundaries.
template <typename Clock = std::chrono::steady_clock>
class LLMRequestProfiler
{
public:
    //! Opt in before the first prefill; repeated calls preserve the start across prompt chunks.
    void beginPrefill(bool enabled)
    {
        if (enabled && !mPrefillStarted)
        {
            mPrefillStarted = Clock::now();
        }
    }

    //! Finish after the last prefill chunk and its first-token sampling.
    void finishPrefill(int32_t generatedTokens)
    {
        if (mPrefillStarted)
        {
            mPrefillFinished = Clock::now();
            mMetrics.promptSeconds = std::chrono::duration<double>(*mPrefillFinished - *mPrefillStarted).count();
            mPrefillGeneratedTokens = generatedTokens;
            mMetrics.generatedTokens = generatedTokens;
        }
    }

    //! Record accepted output, including multiple tokens from a speculative step.
    void recordGeneration(int32_t generatedTokens)
    {
        if (mPrefillFinished && generatedTokens > mMetrics.generatedTokens)
        {
            mMetrics.generatedTokens = generatedTokens;
            mMetrics.decodedTokens = generatedTokens - mPrefillGeneratedTokens;
            mMetrics.generationSeconds = std::chrono::duration<double>(Clock::now() - *mPrefillFinished).count();
        }
    }

    //! Use the same effective prompt length as the aggregate prefill metrics.
    LLMRequestMetrics snapshot(int32_t processedPromptTokens) const
    {
        LLMRequestMetrics result = mMetrics;
        if (mPrefillFinished)
        {
            result.processedPromptTokens = processedPromptTokens;
        }
        return result;
    }

private:
    LLMRequestMetrics mMetrics;
    std::optional<typename Clock::time_point> mPrefillStarted;
    std::optional<typename Clock::time_point> mPrefillFinished;
    int32_t mPrefillGeneratedTokens{0};
};

} // namespace metrics
} // namespace trt_edgellm
