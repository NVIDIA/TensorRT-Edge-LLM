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

namespace trt_edgellm
{
namespace rt
{

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

    bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const override
    {
        return false;
    }
    void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t) override {}
    bool runSystemPromptPrefill(DecodingInferenceContext&) override
    {
        return true;
    }
    void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&, std::vector<tokenizer::Rank> const&,
        int32_t, cudaStream_t) override
    {
    }

    void resetForNewSequences(Tensor&, cudaStream_t) override {}
    void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) override {}

private:
    DecodingRuntimeContext& mRuntime;
};

} // namespace rt
} // namespace trt_edgellm
