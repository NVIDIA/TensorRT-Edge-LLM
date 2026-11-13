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

#include "safety_inference/runtime/processing/processingCapabilities.h"
#include "safety_inference/runtime/safetyGenerationTypes.h"

namespace trt_edgellm::safety
{

struct TokenizationResult
{
    SafetyStatus status{SafetyStatus::ok()};
    TokenIdView tokens{};
};

struct ExpandedTokenResult
{
    SafetyStatus status{SafetyStatus::ok()};
    TokenIdView tokens{};
    std::size_t imageOffset{0};
};

class TokenizerCapability
{
public:
    virtual ~TokenizerCapability() = default;
    virtual SafetyStatus declareResources(TokenizerArtifactSpec const& artifacts, ProcessingLimits const& limits,
        ResourceDeclaration& declaration) noexcept = 0;
    virtual SafetyStatus initialize(TokenizerArtifactSpec const& artifacts, PreparedResources const& prepared) noexcept
        = 0;
    virtual TokenizationResult encodeMessages(LLMGenerationRequest::Request const& request) noexcept = 0;
    virtual TokenizationResult encodePrompt(TextView prompt) noexcept = 0;
    virtual ExpandedTokenResult expandImagePlaceholder(TokenIdView tokens, std::size_t imageTokenCount) noexcept = 0;
    virtual SafetyStatus decodeResponse(
        TokenIdView generated, MutableTextBuffer response, std::size_t& written) noexcept = 0;
    virtual TokenIdView stopTokens() const noexcept = 0;
    virtual SafetyStatus deinitialize() noexcept = 0;
};

} // namespace trt_edgellm::safety
