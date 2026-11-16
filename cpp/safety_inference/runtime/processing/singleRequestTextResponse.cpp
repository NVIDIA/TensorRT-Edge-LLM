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

#include "singleRequestTextResponse.h"

namespace trt_edgellm::safety
{

SafetyStatus SingleRequestTextResponse::formatResponse(TokenizerCapability& tokenizer, std::size_t maxResponseBytes,
    ModelInferenceResult const& result, LLMGenerationResponse& response) noexcept
{
    std::string* text{nullptr};
    auto const status = prepareText(maxResponseBytes, response, text);
    if (!status.okStatus())
    {
        return status;
    }
    return decodeText(tokenizer, result, *text);
}

SafetyStatus SingleRequestTextResponse::prepareText(
    std::size_t maxResponseBytes, LLMGenerationResponse& response, std::string*& text) noexcept
{
    text = nullptr;
    if (response.outputTexts.size() != 1)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Expected one text response");
    }
    text = &response.outputTexts.front();
    if (text->capacity() <= maxResponseBytes)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Text response storage is too small");
    }
    text->resize(maxResponseBytes + 1U);
    return SafetyStatus::ok();
}

SafetyStatus SingleRequestTextResponse::decodeText(
    TokenizerCapability& tokenizer, ModelInferenceResult const& result, std::string& text) noexcept
{
    std::size_t written{0};
    auto const status = tokenizer.decodeResponse(result.generatedTokens, {text.data(), text.size()}, written);
    if (status.okStatus())
    {
        text.resize(written);
    }
    return status;
}

} // namespace trt_edgellm::safety
