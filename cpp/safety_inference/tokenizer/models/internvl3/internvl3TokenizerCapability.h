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

#include "safety_inference/tokenizer/byteLevelBpeTokenizer.h"
#include "safety_inference/tokenizer/tokenizerCapability.h"

namespace trt_edgellm::safety
{

class InternVL3TokenizerCapability final : public TokenizerCapability
{
public:
    SafetyStatus declareResources(TokenizerArtifactSpec const& artifacts, ProcessingLimits const& limits,
        ResourceDeclaration& declaration) noexcept override;
    SafetyStatus initialize(
        TokenizerArtifactSpec const& artifacts, PreparedResources const& prepared) noexcept override;
    TokenizationResult encodeMessages(LLMGenerationRequest::Request const& request) noexcept override;
    TokenizationResult encodePrompt(TextView prompt) noexcept override;
    ExpandedTokenResult expandImagePlaceholder(TokenIdView tokens, std::size_t imageTokenCount) noexcept override;
    SafetyStatus decodeResponse(
        TokenIdView generated, MutableTextBuffer response, std::size_t& written) noexcept override;
    TokenIdView stopTokens() const noexcept override;
    SafetyStatus deinitialize() noexcept override;

private:
    SafetyStatus declareAdapterResources(
        TokenizerArtifactSpec const& artifacts, ResourceDeclaration& declaration) const noexcept;
    SafetyStatus declareConfiguredResources(
        TokenizerArtifactSpec const& artifacts, ResourceDeclaration& declaration) noexcept;
    SafetyStatus acquireResources(PreparedResources const& prepared) noexcept;
    SafetyStatus initializeControls(TokenizerArtifactSpec const& artifacts) noexcept;
    SafetyStatus initializeDeclared(TokenizerArtifactSpec const& artifacts, PreparedResources const& prepared) noexcept;
    void resolveControlTokenIds(TokenizerArtifactSpec const& artifacts) noexcept;
    int32_t resolveControlToken(int32_t configured, char const* token) const noexcept;
    bool controlsValid() const noexcept;
    bool modelTokenIdsValid() const noexcept;
    SafetyStatus formatMessages(LLMGenerationRequest::Request const& request, TextView& prompt) noexcept;
    SafetyStatus appendInitialSystem(LLMGenerationRequest::Request const& request, std::size_t& first) noexcept;
    SafetyStatus appendConversation(
        LLMGenerationRequest::Request const& request, std::size_t first, std::size_t& imageCount) noexcept;
    SafetyStatus appendSystem(Message const& message) noexcept;
    SafetyStatus appendMessage(Message const& message, std::size_t& imageCount) noexcept;
    SafetyStatus appendContents(Message const& message, bool allowImage, std::size_t& imageCount) noexcept;
    SafetyStatus appendContent(
        Message::MessageContent const& content, bool allowImage, std::size_t& imageCount) noexcept;
    SafetyStatus appendMessageHeader(Message const& message) noexcept;
    SafetyStatus appendText(std::string_view text) noexcept;
    SafetyStatus finishFormatting(SafetyStatus status, std::size_t imageCount, TextView& prompt) noexcept;
    SafetyStatus beginFormatting(LLMGenerationRequest::Request const& request, TextView& prompt) noexcept;
    std::size_t placeholderCount() const noexcept;
    TokenizationResult finishPromptEncoding(SafetyStatus status) noexcept;
    ExpandedTokenResult invalidExpansion(char const* message) const noexcept;
    SafetyStatus validateExpansion(TokenIdView tokens, std::size_t imageTokenCount) const noexcept;
    std::size_t placeholderOffset() const noexcept;
    void expandAt(std::size_t offset, std::size_t imageTokenCount) noexcept;
    void resetRuntime() noexcept;
    void resetControls() noexcept;

    ByteLevelBpeTokenizer mTokenizer{};
    ProcessingLimits mLimits{};
    RuntimeMemoryView mPromptStorage{};
    RuntimeMemoryView mStopStorage{};
    RuntimeMemoryView mFormattedPromptStorage{};
    int32_t* mPromptTokens{nullptr};
    int32_t* mStopTokens{nullptr};
    char* mFormattedPrompt{nullptr};
    std::size_t mPromptSize{0};
    std::size_t mStopCount{0};
    std::size_t mFormattedPromptSize{0};
    int32_t mImagePlaceholderId{-1};
    int32_t mImageStartId{-1};
    int32_t mImageContextId{-1};
    int32_t mImageEndId{-1};
    int32_t mChatStartId{-1};
    int32_t mChatEndId{-1};
    int32_t mPadTokenId{-1};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
};

} // namespace trt_edgellm::safety
