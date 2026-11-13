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

#include "safety_inference/runtime/processing/processingCapabilities.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace trt_edgellm::safety
{

struct ByteLevelBpeTokenSpec
{
    std::string_view bytes{};
    int32_t id{-1};
    bool special{false};
};

class ByteLevelBpeTokenizer
{
public:
    SafetyStatus declareResources(std::filesystem::path const& artifact, ProcessingLimits const& limits,
        ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount,
        ResourceDeclaration& declaration) noexcept;
    SafetyStatus initialize(PreparedResources const& prepared) noexcept;
    SafetyStatus encode(TextView text, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus decode(
        TokenIdView tokens, TokenIdView stopTokens, MutableTextBuffer output, std::size_t& outputSize) const noexcept;
    int32_t specialTokenId(std::string_view token) const noexcept;
    bool hasTokenId(int32_t id) const noexcept;
    SafetyStatus deinitialize() noexcept;

private:
    struct OwnedToken
    {
        std::string bytes{};
        int32_t id{-1};
        bool special{false};
    };
    struct TokenEntry
    {
        std::uint64_t hash{0};
        std::uint32_t offset{0};
        std::uint32_t length{0};
        int32_t id{-1};
        int32_t rank{-1};
    };
    struct ReverseEntry
    {
        std::uint32_t offset{0};
        std::uint32_t length{0};
        bool occupied{false};
    };
    struct SpecialEntry
    {
        std::uint32_t offset{0};
        std::uint32_t length{0};
        int32_t id{-1};
        bool special{false};
    };
    struct BpeNode
    {
        std::uint32_t begin{0};
        std::uint32_t length{0};
        int32_t next{-1};
        int32_t id{-1};
        int32_t rank{-1};
    };
    struct Utf8Offset
    {
        std::uint32_t begin{0};
        std::uint32_t end{0};
        std::uint32_t codepoint{0};
        std::uint8_t category{0};
    };
    struct ResourceSizes
    {
        std::size_t nodes{0};
        std::size_t offsets{0};
        std::size_t entries{0};
        std::size_t reverse{0};
        std::size_t specials{0};
    };
    struct MergeCandidate
    {
        int32_t left{-1};
        TokenEntry const* token{nullptr};
    };
    struct DecodeState
    {
        std::size_t bytes{0};
        int32_t remaining{0};
        std::uint32_t value{0};
        std::uint32_t minimum{0};
    };

    SafetyStatus parseArtifact(std::filesystem::path const& artifact, ByteLevelBpeTokenSpec const* requiredTokens,
        std::size_t requiredTokenCount) noexcept;
    SafetyStatus parseArtifactDocument(
        nlohmann::json const& document, ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount);
    SafetyStatus parseArtifactTokens(nlohmann::json const& model, nlohmann::json const& document);
    SafetyStatus parseVocabulary(nlohmann::json const& model);
    SafetyStatus parseAddedTokens(nlohmann::json const& document);
    SafetyStatus ensureRequiredTokens(ByteLevelBpeTokenSpec const* tokens, std::size_t count);
    SafetyStatus appendConfiguredToken(OwnedToken token, std::vector<OwnedToken>& destination);
    SafetyStatus computeResourceSizes(ResourceSizes& sizes) const noexcept;
    SafetyStatus declareMemory(ResourceSizes const& sizes, ResourceDeclaration& declaration) const noexcept;
    SafetyStatus declareParsedResources(ResourceDeclaration& declaration) const noexcept;
    SafetyStatus completeDeclaration(SafetyStatus status) noexcept;
    bool validDeclaration(std::filesystem::path const& artifact, ProcessingLimits const& limits,
        ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount) const noexcept;
    SafetyStatus acquireResources(PreparedResources const& prepared) noexcept;
    void bindResources() noexcept;
    SafetyStatus materializeConfiguration() noexcept;
    SafetyStatus materializeVocabulary(std::size_t& cursor) noexcept;
    SafetyStatus materializeVocabularyToken(OwnedToken const& token, std::size_t& cursor) noexcept;
    void materializeSpecialTokens(std::size_t& cursor) noexcept;
    void materializeSpecialToken(OwnedToken const& token, std::size_t& cursor) noexcept;
    void sortRuntimeTokens() noexcept;
    bool tokenEntryLess(TokenEntry const& left, TokenEntry const& right) const noexcept;
    TokenEntry const* lookup(char const* data, std::size_t size) const noexcept;
    SpecialEntry const* specialAt(char const* data, std::size_t size) const noexcept;
    SpecialEntry const* specialById(int32_t id) const noexcept;
    SafetyStatus encodeSegments(TextView text, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus encodeSegment(
        TextView text, std::size_t& cursor, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus encodeSpecialSegment(
        SpecialEntry const& special, int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept;
    SafetyStatus encodeOrdinarySegment(TextView text, std::size_t begin, std::size_t end, int32_t* output,
        std::size_t capacity, std::size_t& outputSize) noexcept;
    std::size_t nextSpecialOffset(TextView text, std::size_t begin) const noexcept;
    SafetyStatus appendOutputToken(
        int32_t id, int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept;
    SafetyStatus encodeOrdinary(
        char const* data, std::size_t size, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus decodeOffsets(char const* data, std::size_t size, std::size_t& count) noexcept;
    SafetyStatus encodePretokenizedSpans(
        char const* data, std::size_t count, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus encodePretokenizedSpan(char const* data, std::size_t begin, std::size_t end, int32_t* output,
        std::size_t capacity, std::size_t& outputSize) noexcept;
    std::size_t nextPieceEnd(std::size_t begin, std::size_t count) const noexcept;
    std::size_t matchContraction(std::size_t begin, std::size_t count) const noexcept;
    std::size_t matchSuffix(std::size_t begin, std::size_t count, char const* suffix) const noexcept;
    std::size_t matchLetters(std::size_t begin, std::size_t count) const noexcept;
    std::size_t matchDigits(std::size_t begin, std::size_t count) const noexcept;
    std::size_t matchPunctuation(std::size_t begin, std::size_t count) const noexcept;
    std::size_t matchWhitespace(std::size_t begin, std::size_t count) const noexcept;
    bool asciiAt(std::size_t index, std::size_t count, char value) const noexcept;
    SafetyStatus encodeBpe(
        char const* data, std::size_t size, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept;
    SafetyStatus initializeBpeNodes(char const* data, std::size_t size) noexcept;
    void mergeBpeNodes(char const* data, std::size_t size) noexcept;
    MergeCandidate findBestMerge(char const* data, std::size_t size) const noexcept;
    void considerMerge(char const* data, int32_t left, MergeCandidate& candidate) const noexcept;
    void applyMerge(MergeCandidate const& candidate) noexcept;
    SafetyStatus appendBpeNodes(int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept;
    SafetyStatus measureDecoded(
        TokenIdView tokens, TokenIdView stops, std::size_t limit, DecodeState& state) const noexcept;
    SafetyStatus measureDecodedToken(int32_t id, std::size_t limit, DecodeState& state) const noexcept;
    SafetyStatus validateDecodedEntry(int32_t id, std::size_t limit, DecodeState const& state) const noexcept;
    SafetyStatus consumeEntryUtf8(ReverseEntry const& entry, DecodeState& state) const noexcept;
    void copyDecoded(TokenIdView tokens, TokenIdView stops, char* output) const noexcept;
    void publishDecoded(TokenIdView tokens, TokenIdView stops, DecodeState const& state, MutableTextBuffer output,
        std::size_t& outputSize) const noexcept;
    bool validDecodeInput(TokenIdView tokens, TokenIdView stops, MutableTextBuffer output) const noexcept;
    bool isStopToken(int32_t id, TokenIdView stops) const noexcept;
    void resetRuntime() noexcept;
    void releaseConfiguration() noexcept;

    ProcessingLimits mLimits{};
    std::vector<OwnedToken> mConfiguredVocabulary{};
    std::vector<OwnedToken> mConfiguredSpecials{};
    std::size_t mTokenBytesRequired{0};
    RuntimeMemoryView mNodeStorage{};
    RuntimeMemoryView mOffsetStorage{};
    RuntimeMemoryView mEntryStorage{};
    RuntimeMemoryView mReverseStorage{};
    RuntimeMemoryView mSpecialStorage{};
    RuntimeMemoryView mTokenBytesStorage{};
    TokenEntry* mEntries{nullptr};
    ReverseEntry* mReverse{nullptr};
    SpecialEntry* mSpecials{nullptr};
    char* mTokenBytes{nullptr};
    BpeNode* mNodes{nullptr};
    Utf8Offset* mOffsets{nullptr};
    std::size_t mEntryCount{0};
    std::size_t mSpecialCount{0};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
};

} // namespace trt_edgellm::safety
