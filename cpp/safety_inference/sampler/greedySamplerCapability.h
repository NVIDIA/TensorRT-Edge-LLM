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

#include "safety_inference/sampler/samplerCapability.h"

namespace trt_edgellm::safety
{

class GreedySamplerCapability final : public SamplerCapability
{
public:
    SafetyStatus declareResources(int32_t vocabularySize, ResourceDeclaration& declaration) noexcept override;
    SafetyStatus initialize(PreparedResources const& prepared, llm::ModelLocalData& data, LMStageContext const& prefill,
        LMStageContext const& decode) noexcept override;
    SafetyStatus select(LMStageOutputs const& outputs, int32_t& token) noexcept override;
    SafetyStatus deinitialize() noexcept override;

private:
    RuntimeMemoryView mHostResult{};
    RuntimeMemoryView mDeviceResult{};
    RuntimeTensorView const* mPrefillLogits{nullptr};
    RuntimeTensorView const* mDecodeLogits{nullptr};
    LMStageContext const* mPrefillContext{nullptr};
    LMStageContext const* mDecodeContext{nullptr};
    cudaStream_t mStream{nullptr};
    cudaEvent_t mCompletion{nullptr};
    int32_t mVocabularySize{0};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
};

} // namespace trt_edgellm::safety
