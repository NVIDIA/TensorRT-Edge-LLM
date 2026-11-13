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

#include "safety_inference/runtime/model_inference/modelInference.h"
#include "safety_inference/runtime/safetyGenerationTypes.h"
#include "safety_inference/tokenizer/tokenizerCapability.h"

namespace trt_edgellm::safety
{

class SingleRequestTextResponse final
{
public:
    SafetyStatus formatResponse(TokenizerCapability& tokenizer, std::size_t maxResponseBytes,
        ModelInferenceResult const& result, LLMGenerationResponse& response) noexcept;

private:
    static SafetyStatus prepareText(
        std::size_t maxResponseBytes, LLMGenerationResponse& response, std::string*& text) noexcept;
    static SafetyStatus decodeText(
        TokenizerCapability& tokenizer, ModelInferenceResult const& result, std::string& text) noexcept;
};

} // namespace trt_edgellm::safety
