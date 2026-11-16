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

#pragma once

#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/jumpForward.h"

#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief One sampled token per step, plus grammar jump-forward.
 *
 * A slot's pending tokens (DecodingInferenceContext::pendingTokens) are the tail of its token list without KV.
 * A plain step feeds one per slot on the decode profile. After a jump appended forced tokens, the next step is a
 * catch-up step: every slot runs as a context-chunk row covering its pending tokens on the prefill profile, eager,
 * and samples from its last row. Either step ends in the same sampling, grammar and logprobs tail.
 */
class VanillaDecoder final : public DecodingStrategy
{
public:
    explicit VanillaDecoder(DecodingRuntimeContext& runtime);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kVanilla;
    }

    char const* name() const noexcept override
    {
        return "vanilla";
    }

    bool isSpeculative() const noexcept override
    {
        return false;
    }

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    int64_t getRequiredContextMemorySize() const noexcept override
    {
        return 0;
    }
    void setContextMemory(Tensor&) override {}

    void resetForNewSequences(Tensor&, cudaStream_t) override {}
    void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) override {}

private:
    //! Whether any active slot carries more than one pending token.
    static bool needsChunkStep(DecodingInferenceContext const& context) noexcept;
    //! The one-token decode forward on the decode profile; commits one KV position per slot.
    bool runDecodeForward(DecodingInferenceContext& context);
    //! The catch-up forward: one context-chunk row per slot over its pending tokens; commits that many positions.
    bool runChunkForward(DecodingInferenceContext& context);
    //! Bias, mask, sample and append one token per slot from the step's logits, advance the grammars, collect
    //! logprobs, then append grammar-forced tokens where a slot allows it.
    bool sampleAndAppend(DecodingInferenceContext& context, bool chunkStep);
    //! Jump every slot whose only token without KV is the one just sampled; a jump raises its pending count.
    void collectJumps(DecodingInferenceContext& context);
    void ensureCoordinator();

    DecodingRuntimeContext& mRuntime;
    std::optional<JumpForwardCoordinator> mCoordinator;
};

} // namespace rt
} // namespace trt_edgellm
