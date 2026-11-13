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

#include "internvl3TokenizerCapability.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>

namespace trt_edgellm::safety
{
namespace
{
constexpr RuntimeResourceKey kPROMPT{100};
constexpr RuntimeResourceKey kSTOPS{107};
constexpr RuntimeResourceKey kFORMATTED_PROMPT{108};
constexpr std::string_view kIMAGE_PLACEHOLDER{"<image>"};
constexpr char kIMAGE_START[]{"<img>"};
constexpr char kIMAGE_CONTEXT[]{"<IMG_CONTEXT>"};
constexpr char kIMAGE_END[]{"</img>"};
constexpr char kCHAT_START[]{"<|im_start|>"};
constexpr char kCHAT_END[]{"<|im_end|>"};
constexpr std::string_view kDEFAULT_SYSTEM{
    u8"你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"};

SafetyStatus invalid(char const* message) noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, message);
}

bool checkedBytes(std::size_t count, std::size_t element, std::size_t& output) noexcept
{
    if (element != 0 && count > std::numeric_limits<std::size_t>::max() / element)
    {
        return false;
    }
    output = count * element;
    return output != 0;
}

SafetyStatus declareAdapterMemory(std::size_t promptBytes, std::size_t stopBytes, std::size_t formattedPromptBytes,
    ResourceDeclaration& declaration) noexcept
{
    try
    {
        using Kind = RuntimeMemoryKind;
        declaration.memory.push_back({kPROMPT, promptBytes, Kind::kPAGEABLE});
        declaration.memory.push_back({kSTOPS, stopBytes, Kind::kPAGEABLE});
        declaration.memory.push_back({kFORMATTED_PROMPT, formattedPromptBytes, Kind::kPAGEABLE});
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot declare tokenizer resources");
    }
}
} // namespace

SafetyStatus InternVL3TokenizerCapability::declareAdapterResources(
    TokenizerArtifactSpec const& artifacts, ResourceDeclaration& declaration) const noexcept
{
    std::size_t promptBytes{0};
    std::size_t stopBytes{0};
    bool const valid = checkedBytes(mLimits.maxPromptTokens, sizeof(int32_t), promptBytes)
        && checkedBytes(artifacts.stopTokenCount, sizeof(int32_t), stopBytes);
    if (!valid)
    {
        return invalid("Tokenizer adapter storage size is out of range");
    }
    return declareAdapterMemory(promptBytes, stopBytes, mLimits.maxPromptBytes, declaration);
}

SafetyStatus InternVL3TokenizerCapability::declareConfiguredResources(
    TokenizerArtifactSpec const& artifacts, ResourceDeclaration& declaration) noexcept
{
    ByteLevelBpeTokenSpec const required{kIMAGE_PLACEHOLDER, mLimits.vocabularySize, true};
    auto const status = mTokenizer.declareResources(artifacts.path, mLimits, &required, 1, declaration);
    if (!status.okStatus())
    {
        return status;
    }
    return declareAdapterResources(artifacts, declaration);
}

SafetyStatus InternVL3TokenizerCapability::declareResources(
    TokenizerArtifactSpec const& artifacts, ProcessingLimits const& limits, ResourceDeclaration& declaration) noexcept
{
    bool const valid = mState == CapabilityPreparationState::kEMPTY && !artifacts.path.empty()
        && limits.maxPromptBytes != 0 && limits.maxPromptTokens != 0 && limits.vocabularySize > 0
        && artifacts.stopTokenCount != 0 && artifacts.stopTokenCount <= artifacts.stopTokenIds.size();
    if (!valid)
    {
        return invalid("Invalid tokenizer declaration");
    }
    mLimits = limits;
    auto const status = declareConfiguredResources(artifacts, declaration);
    if (status.okStatus())
    {
        mState = CapabilityPreparationState::kDECLARED;
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::acquireResources(PreparedResources const& prepared) noexcept
{
    auto status = prepared.memory(kPROMPT, mPromptStorage);
    if (status.okStatus())
    {
        status = prepared.memory(kSTOPS, mStopStorage);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kFORMATTED_PROMPT, mFormattedPromptStorage);
    }
    if (!status.okStatus())
    {
        return status;
    }
    mPromptTokens = static_cast<int32_t*>(mPromptStorage.data);
    mStopTokens = static_cast<int32_t*>(mStopStorage.data);
    mFormattedPrompt = static_cast<char*>(mFormattedPromptStorage.data);
    return SafetyStatus::ok();
}

int32_t InternVL3TokenizerCapability::resolveControlToken(int32_t configured, char const* token) const noexcept
{
    int32_t const tokenizerId = mTokenizer.specialTokenId(token);
    return configured < 0 || configured == tokenizerId ? tokenizerId : -1;
}

void InternVL3TokenizerCapability::resolveControlTokenIds(TokenizerArtifactSpec const& artifacts) noexcept
{
    mImagePlaceholderId = resolveControlToken(artifacts.imagePlaceholderId, kIMAGE_PLACEHOLDER.data());
    mImageStartId = resolveControlToken(artifacts.imageStartId, kIMAGE_START);
    mImageContextId = resolveControlToken(artifacts.imageContextId, kIMAGE_CONTEXT);
    mImageEndId = resolveControlToken(artifacts.imageEndId, kIMAGE_END);
    mChatStartId = mTokenizer.specialTokenId(kCHAT_START);
    mChatEndId = mTokenizer.specialTokenId(kCHAT_END);
    mPadTokenId = artifacts.padTokenId;
}

bool InternVL3TokenizerCapability::controlsValid() const noexcept
{
    return mImagePlaceholderId >= 0 && mImageStartId >= 0 && mImageContextId >= 0 && mImageEndId >= 0
        && mChatStartId >= 0 && mChatEndId >= 0 && mStopCount != 0 && modelTokenIdsValid();
}

bool InternVL3TokenizerCapability::modelTokenIdsValid() const noexcept
{
    if (!mTokenizer.hasTokenId(mImageStartId) || !mTokenizer.hasTokenId(mImageContextId)
        || !mTokenizer.hasTokenId(mImageEndId) || !mTokenizer.hasTokenId(mPadTokenId))
    {
        return false;
    }
    return std::all_of(
        mStopTokens, mStopTokens + mStopCount, [this](int32_t token) { return mTokenizer.hasTokenId(token); });
}

SafetyStatus InternVL3TokenizerCapability::initializeControls(TokenizerArtifactSpec const& artifacts) noexcept
{
    resolveControlTokenIds(artifacts);
    mStopCount = artifacts.stopTokenCount;
    std::copy_n(artifacts.stopTokenIds.begin(), mStopCount, mStopTokens);
    return controlsValid() ? SafetyStatus::ok() : invalid("Tokenizer control tokens are incomplete");
}

SafetyStatus InternVL3TokenizerCapability::initializeDeclared(
    TokenizerArtifactSpec const& artifacts, PreparedResources const& prepared) noexcept
{
    auto status = acquireResources(prepared);
    if (!status.okStatus())
    {
        return status;
    }
    status = mTokenizer.initialize(prepared);
    if (!status.okStatus())
    {
        return status;
    }
    return initializeControls(artifacts);
}

SafetyStatus InternVL3TokenizerCapability::initialize(
    TokenizerArtifactSpec const& artifacts, PreparedResources const& prepared) noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Tokenizer was not declared");
    }
    auto const status = initializeDeclared(artifacts, prepared);
    if (status.okStatus())
    {
        mState = CapabilityPreparationState::kREADY;
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::appendText(std::string_view text) noexcept
{
    if (mFormattedPromptSize > mLimits.maxPromptBytes || text.size() > mLimits.maxPromptBytes - mFormattedPromptSize)
    {
        return invalid("Formatted prompt exceeds prepared capacity");
    }
    std::memcpy(mFormattedPrompt + mFormattedPromptSize, text.data(), text.size());
    mFormattedPromptSize += text.size();
    return SafetyStatus::ok();
}

SafetyStatus InternVL3TokenizerCapability::appendContent(
    Message::MessageContent const& content, bool allowImage, std::size_t& imageCount) noexcept
{
    if (content.type == "text")
    {
        return appendText(content.content);
    }
    if (content.type == "image" && allowImage)
    {
        ++imageCount;
        return appendText("<image>\n");
    }
    return invalid("Unsupported InternVL message content");
}

SafetyStatus InternVL3TokenizerCapability::appendContents(
    Message const& message, bool allowImage, std::size_t& imageCount) noexcept
{
    for (auto const& content : message.contents)
    {
        auto const status = appendContent(content, allowImage, imageCount);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus InternVL3TokenizerCapability::appendSystem(Message const& message) noexcept
{
    std::size_t imageCount{0};
    auto status = appendText("<|im_start|>system\n");
    if (status.okStatus())
    {
        status = appendContents(message, false, imageCount);
    }
    if (status.okStatus())
    {
        status = appendText("<|im_end|>\n");
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::appendMessageHeader(Message const& message) noexcept
{
    if (message.role != "user" && message.role != "assistant")
    {
        return invalid("Unsupported InternVL message role");
    }
    auto status = appendText("<|im_start|>");
    if (status.okStatus())
    {
        status = appendText(message.role);
    }
    if (status.okStatus())
    {
        status = appendText("\n");
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::appendMessage(Message const& message, std::size_t& imageCount) noexcept
{
    auto status = appendMessageHeader(message);
    if (status.okStatus())
    {
        status = appendContents(message, true, imageCount);
    }
    if (status.okStatus())
    {
        status = appendText("<|im_end|>\n");
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::appendInitialSystem(
    LLMGenerationRequest::Request const& request, std::size_t& first) noexcept
{
    first = 0;
    if (request.messages.front().role == "system")
    {
        first = 1;
        return appendSystem(request.messages.front());
    }
    auto status = appendText("<|im_start|>system\n");
    if (status.okStatus())
    {
        status = appendText(kDEFAULT_SYSTEM);
    }
    if (status.okStatus())
    {
        status = appendText("<|im_end|>\n");
    }
    return status;
}

SafetyStatus InternVL3TokenizerCapability::appendConversation(
    LLMGenerationRequest::Request const& request, std::size_t first, std::size_t& imageCount) noexcept
{
    for (std::size_t index = first; index < request.messages.size(); ++index)
    {
        auto const status = appendMessage(request.messages[index], imageCount);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus InternVL3TokenizerCapability::finishFormatting(
    SafetyStatus status, std::size_t imageCount, TextView& prompt) noexcept
{
    if (!status.okStatus())
    {
        return status;
    }
    if (imageCount != 1)
    {
        return invalid("InternVL request requires exactly one image content");
    }
    prompt = {mFormattedPrompt, mFormattedPromptSize};
    return SafetyStatus::ok();
}

SafetyStatus InternVL3TokenizerCapability::beginFormatting(
    LLMGenerationRequest::Request const& request, TextView& prompt) noexcept
{
    prompt = {};
    mFormattedPromptSize = 0;
    return mFormattedPrompt != nullptr && !request.messages.empty() ? SafetyStatus::ok()
                                                                    : invalid("InternVL request requires messages");
}

SafetyStatus InternVL3TokenizerCapability::formatMessages(
    LLMGenerationRequest::Request const& request, TextView& prompt) noexcept
{
    auto status = beginFormatting(request, prompt);
    if (!status.okStatus())
    {
        return status;
    }
    std::size_t first{0};
    status = appendInitialSystem(request, first);
    std::size_t imageCount{0};
    if (status.okStatus())
    {
        status = appendConversation(request, first, imageCount);
    }
    if (status.okStatus())
    {
        status = appendText("<|im_start|>assistant\n");
    }
    return finishFormatting(status, imageCount, prompt);
}

TokenizationResult InternVL3TokenizerCapability::encodeMessages(LLMGenerationRequest::Request const& request) noexcept
{
    TextView prompt{};
    auto const status = formatMessages(request, prompt);
    if (!status.okStatus())
    {
        return {status, {}};
    }
    return encodePrompt(prompt);
}

std::size_t InternVL3TokenizerCapability::placeholderCount() const noexcept
{
    return static_cast<std::size_t>(std::count(mPromptTokens, mPromptTokens + mPromptSize, mImagePlaceholderId));
}

TokenizationResult InternVL3TokenizerCapability::finishPromptEncoding(SafetyStatus status) noexcept
{
    if (!status.okStatus())
    {
        mPromptSize = 0;
        return {status, {}};
    }
    if (placeholderCount() != 1)
    {
        mPromptSize = 0;
        return {invalid("Prompt must contain exactly one image placeholder"), {}};
    }
    return {SafetyStatus::ok(), {mPromptTokens, mPromptSize}};
}

TokenizationResult InternVL3TokenizerCapability::encodePrompt(TextView prompt) noexcept
{
    mPromptSize = 0;
    if (mState != CapabilityPreparationState::kREADY)
    {
        return {SafetyStatus::error(SafetyStatusCode::kInvalidState, "Tokenizer is not ready"), {}};
    }
    auto const status = mTokenizer.encode(prompt, mPromptTokens, mLimits.maxPromptTokens, mPromptSize);
    return finishPromptEncoding(status);
}

ExpandedTokenResult InternVL3TokenizerCapability::invalidExpansion(char const* message) const noexcept
{
    return {invalid(message), {}, 0};
}

SafetyStatus InternVL3TokenizerCapability::validateExpansion(
    TokenIdView tokens, std::size_t imageTokenCount) const noexcept
{
    bool const valid = mState == CapabilityPreparationState::kREADY && tokens.data == mPromptTokens
        && tokens.size == mPromptSize && imageTokenCount != 0
        && imageTokenCount <= std::numeric_limits<std::size_t>::max() - 2;
    return valid ? SafetyStatus::ok() : invalid("Invalid image placeholder expansion");
}

std::size_t InternVL3TokenizerCapability::placeholderOffset() const noexcept
{
    auto const* marker = std::find(mPromptTokens, mPromptTokens + mPromptSize, mImagePlaceholderId);
    return static_cast<std::size_t>(marker - mPromptTokens);
}

void InternVL3TokenizerCapability::expandAt(std::size_t offset, std::size_t imageTokenCount) noexcept
{
    std::size_t const replacement = imageTokenCount + 2;
    std::size_t const suffix = mPromptSize - offset - 1;
    std::memmove(mPromptTokens + offset + replacement, mPromptTokens + offset + 1, suffix * sizeof(int32_t));
    mPromptTokens[offset] = mImageStartId;
    std::fill_n(mPromptTokens + offset + 1, imageTokenCount, mImageContextId);
    mPromptTokens[offset + imageTokenCount + 1] = mImageEndId;
    mPromptSize = mPromptSize - 1 + replacement;
}

ExpandedTokenResult InternVL3TokenizerCapability::expandImagePlaceholder(
    TokenIdView tokens, std::size_t imageTokenCount) noexcept
{
    auto const status = validateExpansion(tokens, imageTokenCount);
    if (!status.okStatus())
    {
        return {status, {}, 0};
    }
    auto const offset = placeholderOffset();
    auto const replacement = imageTokenCount + 2;
    if (offset == mPromptSize || replacement > mLimits.maxPromptTokens
        || mPromptSize - 1 > mLimits.maxPromptTokens - replacement)
    {
        return invalidExpansion("Expanded prompt exceeds prepared capacity");
    }
    expandAt(offset, imageTokenCount);
    return {SafetyStatus::ok(), {mPromptTokens, mPromptSize}, offset + 1};
}

SafetyStatus InternVL3TokenizerCapability::decodeResponse(
    TokenIdView generated, MutableTextBuffer response, std::size_t& written) noexcept
{
    if (mState != CapabilityPreparationState::kREADY)
    {
        written = 0;
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Tokenizer is not ready");
    }
    return mTokenizer.decode(generated, stopTokens(), response, written);
}

TokenIdView InternVL3TokenizerCapability::stopTokens() const noexcept
{
    return {mStopTokens, mStopCount};
}

void InternVL3TokenizerCapability::resetRuntime() noexcept
{
    mPromptStorage = {};
    mStopStorage = {};
    mFormattedPromptStorage = {};
    mPromptTokens = nullptr;
    mStopTokens = nullptr;
    mFormattedPrompt = nullptr;
    mPromptSize = 0;
    mStopCount = 0;
    mFormattedPromptSize = 0;
}

void InternVL3TokenizerCapability::resetControls() noexcept
{
    mImagePlaceholderId = -1;
    mImageStartId = -1;
    mImageContextId = -1;
    mImageEndId = -1;
    mChatStartId = -1;
    mChatEndId = -1;
    mPadTokenId = -1;
}

SafetyStatus InternVL3TokenizerCapability::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    auto const status = mTokenizer.deinitialize();
    resetRuntime();
    resetControls();
    mState = CapabilityPreparationState::kCLOSED;
    return status;
}

} // namespace trt_edgellm::safety
