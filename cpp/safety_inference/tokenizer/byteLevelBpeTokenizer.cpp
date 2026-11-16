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

#include "byteLevelBpeTokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace trt_edgellm::safety
{
namespace
{
constexpr RuntimeResourceKey kNODES{101};
constexpr RuntimeResourceKey kOFFSETS{102};
constexpr RuntimeResourceKey kENTRIES{103};
constexpr RuntimeResourceKey kREVERSE{104};
constexpr RuntimeResourceKey kSPECIALS{105};
constexpr RuntimeResourceKey kTOKEN_BYTES{106};
constexpr std::size_t kMAX_TOKENIZER_BYTES{256U * 1024U * 1024U};

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

bool safeGptByte(std::uint32_t value) noexcept
{
    return (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || (value >= 174 && value <= 255);
}

int32_t utf8Width(std::uint8_t lead) noexcept
{
    if (lead < 0x80)
    {
        return 1;
    }
    return (lead & 0xE0) == 0xC0 ? 2 : (lead & 0xF0) == 0xE0 ? 3 : (lead & 0xF8) == 0xF0 ? 4 : 0;
}

bool validCodepoint(std::uint32_t value, std::uint32_t minimum) noexcept
{
    return value >= minimum && value <= 0x10FFFF && (value < 0xD800 || value > 0xDFFF);
}

bool decodeContinuationBytes(char const* data, std::size_t cursor, int32_t width, std::uint32_t& value) noexcept
{
    for (int32_t index = 1; index < width; ++index)
    {
        auto const byte = static_cast<std::uint8_t>(data[cursor + index]);
        if ((byte & 0xC0) != 0x80)
        {
            return false;
        }
        value = (value << 6U) | (byte & 0x3FU);
    }
    return true;
}

bool decodeCodepoint(char const* data, std::size_t size, std::size_t& cursor, std::uint32_t& output) noexcept
{
    auto const lead = static_cast<std::uint8_t>(data[cursor]);
    int32_t const width = utf8Width(lead);
    if (width == 0 || cursor + width > size)
    {
        return false;
    }
    std::uint32_t value = width == 1 ? lead : lead & (0x7FU >> width);
    std::uint32_t const minimum[]{0, 0, 0x80, 0x800, 0x10000};
    if (!decodeContinuationBytes(data, cursor, width, value) || !validCodepoint(value, minimum[width]))
    {
        return false;
    }
    cursor += width;
    output = value;
    return true;
}

bool startUtf8Sequence(std::uint8_t byte, int32_t& remaining, std::uint32_t& value, std::uint32_t& minimum) noexcept
{
    if (byte < 0x80)
    {
        return true;
    }
    int32_t const width = utf8Width(byte);
    if (width < 2)
    {
        return false;
    }
    remaining = width - 1;
    value = byte & (0x7FU >> width);
    minimum = width == 2 ? 0x80 : width == 3 ? 0x800 : 0x10000;
    return true;
}

bool consumeUtf8Byte(std::uint8_t byte, int32_t& remaining, std::uint32_t& value, std::uint32_t& minimum) noexcept
{
    if (remaining == 0)
    {
        return startUtf8Sequence(byte, remaining, value, minimum);
    }
    if ((byte & 0xC0) != 0x80)
    {
        return false;
    }
    value = (value << 6U) | (byte & 0x3FU);
    --remaining;
    return remaining != 0 || validCodepoint(value, minimum);
}

bool inverseEscapedGptCodepoint(std::uint32_t codepoint, char& output) noexcept
{
    std::uint32_t target = codepoint - 256;
    for (std::uint32_t byte = 0; byte <= 255; ++byte)
    {
        if (!safeGptByte(byte) && target-- == 0)
        {
            output = static_cast<char>(byte);
            return true;
        }
    }
    return false;
}

bool inverseGptCodepoint(std::uint32_t codepoint, char& output) noexcept
{
    if (codepoint <= 255 && safeGptByte(codepoint))
    {
        output = static_cast<char>(codepoint);
        return true;
    }
    return codepoint >= 256 && inverseEscapedGptCodepoint(codepoint, output);
}

bool appendDecodedGptBytes(std::string const& encoded, std::string& output)
{
    for (std::size_t cursor = 0; cursor < encoded.size();)
    {
        std::uint32_t codepoint{0};
        char byte{0};
        if (!decodeCodepoint(encoded.data(), encoded.size(), cursor, codepoint)
            || !inverseGptCodepoint(codepoint, byte))
        {
            return false;
        }
        output.push_back(byte);
    }
    return true;
}

bool decodeGptToken(std::string const& encoded, std::string& output) noexcept
{
    try
    {
        output.clear();
        output.reserve(encoded.size());
        return appendDecodedGptBytes(encoded, output);
    }
    catch (...)
    {
        return false;
    }
}

std::uint64_t hashBytes(char const* data, std::size_t size) noexcept
{
    std::uint64_t hash{1469598103934665603ULL};
    for (std::size_t index = 0; index < size; ++index)
    {
        hash = (hash ^ static_cast<std::uint8_t>(data[index])) * 1099511628211ULL;
    }
    return hash;
}

bool unicodeWhitespace(std::uint32_t codepoint) noexcept
{
    return codepoint == 0x20 || (codepoint >= 0x09 && codepoint <= 0x0D) || codepoint == 0x85 || codepoint == 0xA0
        || codepoint == 0x1680 || (codepoint >= 0x2000 && codepoint <= 0x200A) || codepoint == 0x2028
        || codepoint == 0x2029 || codepoint == 0x202F || codepoint == 0x205F || codepoint == 0x3000;
}

bool unicodeDigit(std::uint32_t codepoint) noexcept
{
    return (codepoint >= '0' && codepoint <= '9') || (codepoint >= 0x660 && codepoint <= 0x669)
        || (codepoint >= 0x6F0 && codepoint <= 0x6F9) || (codepoint >= 0xFF10 && codepoint <= 0xFF19);
}

std::uint8_t unicodeCategory(std::uint32_t codepoint) noexcept
{
    if (unicodeWhitespace(codepoint))
    {
        return 3;
    }
    if (unicodeDigit(codepoint))
    {
        return 2;
    }
    return (codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z') || codepoint >= 0xC0 ? 1
                                                                                                                 : 4;
}

SafetyStatus loadArtifactDocument(std::filesystem::path const& artifact, nlohmann::json& document)
{
    auto const bytes = std::filesystem::file_size(artifact);
    if (bytes == 0 || bytes > kMAX_TOKENIZER_BYTES)
    {
        return invalid("Tokenizer artifact size is out of range");
    }
    std::ifstream input(artifact);
    if (!input)
    {
        return invalid("Cannot parse byte-level tokenizer artifact");
    }
    document = nlohmann::json::parse(input);
    return SafetyStatus::ok();
}

SafetyStatus validateTokenizerModel(nlohmann::json const& model)
{
    if (model.at("type") != "BPE" || model.value("byte_fallback", false))
    {
        return invalid("Tokenizer must use byte-level BPE");
    }
    return SafetyStatus::ok();
}
} // namespace

SafetyStatus ByteLevelBpeTokenizer::parseArtifact(std::filesystem::path const& artifact,
    ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount) noexcept
{
    try
    {
        releaseConfiguration();
        nlohmann::json document{};
        auto const status = loadArtifactDocument(artifact, document);
        if (!status.okStatus())
        {
            return status;
        }
        return parseArtifactDocument(document, requiredTokens, requiredTokenCount);
    }
    catch (...)
    {
        return invalid("Cannot parse byte-level tokenizer artifact");
    }
}

SafetyStatus ByteLevelBpeTokenizer::parseArtifactTokens(nlohmann::json const& model, nlohmann::json const& document)
{
    auto status = validateTokenizerModel(model);
    if (!status.okStatus())
    {
        return status;
    }
    status = parseVocabulary(model);
    if (!status.okStatus())
    {
        return status;
    }
    return parseAddedTokens(document);
}

SafetyStatus ByteLevelBpeTokenizer::parseArtifactDocument(
    nlohmann::json const& document, ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount)
{
    auto const& model = document.at("model");
    auto const status = parseArtifactTokens(model, document);
    if (!status.okStatus())
    {
        return status;
    }
    return ensureRequiredTokens(requiredTokens, requiredTokenCount);
}

SafetyStatus ByteLevelBpeTokenizer::parseVocabulary(nlohmann::json const& model)
{
    for (auto const& item : model.at("vocab").items())
    {
        OwnedToken token{};
        if (!decodeGptToken(item.key(), token.bytes))
        {
            return invalid("Tokenizer vocabulary contains an invalid byte mapping");
        }
        token.id = item.value().get<int32_t>();
        auto const status = appendConfiguredToken(std::move(token), mConfiguredVocabulary);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::parseAddedTokens(nlohmann::json const& document)
{
    for (auto const& value : document.value("added_tokens", nlohmann::json::array()))
    {
        OwnedToken token{
            value.at("content").get<std::string>(), value.at("id").get<int32_t>(), value.value("special", false)};
        auto const status = appendConfiguredToken(std::move(token), mConfiguredSpecials);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::ensureRequiredTokens(ByteLevelBpeTokenSpec const* tokens, std::size_t count)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const& required = tokens[index];
        bool const exists = std::any_of(mConfiguredSpecials.begin(), mConfiguredSpecials.end(),
            [&](OwnedToken const& token) { return token.bytes == required.bytes; });
        if (!exists)
        {
            auto status = appendConfiguredToken(
                {std::string{required.bytes}, required.id, required.special}, mConfiguredSpecials);
            if (!status.okStatus())
            {
                return status;
            }
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::appendConfiguredToken(OwnedToken token, std::vector<OwnedToken>& destination)
{
    auto const bytes = token.bytes.size();
    if (bytes > std::numeric_limits<std::uint32_t>::max() - mTokenBytesRequired)
    {
        return invalid("Tokenizer byte storage exceeds its index range");
    }
    destination.push_back(std::move(token));
    mTokenBytesRequired += bytes;
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::computeResourceSizes(ResourceSizes& sizes) const noexcept
{
    bool const valid = checkedBytes(mLimits.maxPromptBytes, sizeof(BpeNode), sizes.nodes)
        && checkedBytes(mLimits.maxPromptBytes, sizeof(Utf8Offset), sizes.offsets)
        && checkedBytes(mConfiguredVocabulary.size(), sizeof(TokenEntry), sizes.entries)
        && checkedBytes(mLimits.vocabularySize, sizeof(ReverseEntry), sizes.reverse)
        && checkedBytes(mConfiguredSpecials.size(), sizeof(SpecialEntry), sizes.specials) && mTokenBytesRequired != 0;
    return valid ? SafetyStatus::ok() : invalid("Tokenizer resource sizes are out of range");
}

SafetyStatus ByteLevelBpeTokenizer::declareMemory(
    ResourceSizes const& sizes, ResourceDeclaration& declaration) const noexcept
{
    try
    {
        using Kind = RuntimeMemoryKind;
        declaration.memory.push_back({kNODES, sizes.nodes, Kind::kPAGEABLE});
        declaration.memory.push_back({kOFFSETS, sizes.offsets, Kind::kPAGEABLE});
        declaration.memory.push_back({kENTRIES, sizes.entries, Kind::kPAGEABLE});
        declaration.memory.push_back({kREVERSE, sizes.reverse, Kind::kPAGEABLE});
        declaration.memory.push_back({kSPECIALS, sizes.specials, Kind::kPAGEABLE});
        declaration.memory.push_back({kTOKEN_BYTES, mTokenBytesRequired, Kind::kPAGEABLE});
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot declare tokenizer resources");
    }
}

SafetyStatus ByteLevelBpeTokenizer::declareParsedResources(ResourceDeclaration& declaration) const noexcept
{
    ResourceSizes sizes{};
    auto const status = computeResourceSizes(sizes);
    if (!status.okStatus())
    {
        return status;
    }
    return declareMemory(sizes, declaration);
}

bool ByteLevelBpeTokenizer::validDeclaration(std::filesystem::path const& artifact, ProcessingLimits const& limits,
    ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount) const noexcept
{
    return mState == CapabilityPreparationState::kEMPTY && !artifact.empty() && limits.maxPromptBytes != 0
        && limits.vocabularySize > 0 && (requiredTokenCount == 0 || requiredTokens != nullptr);
}

SafetyStatus ByteLevelBpeTokenizer::declareResources(std::filesystem::path const& artifact,
    ProcessingLimits const& limits, ByteLevelBpeTokenSpec const* requiredTokens, std::size_t requiredTokenCount,
    ResourceDeclaration& declaration) noexcept
{
    if (!validDeclaration(artifact, limits, requiredTokens, requiredTokenCount))
    {
        return invalid("Invalid byte-level tokenizer declaration");
    }
    mLimits = limits;
    auto status = parseArtifact(artifact, requiredTokens, requiredTokenCount);
    if (!status.okStatus())
    {
        return status;
    }
    return completeDeclaration(declareParsedResources(declaration));
}

SafetyStatus ByteLevelBpeTokenizer::completeDeclaration(SafetyStatus status) noexcept
{
    if (status.okStatus())
    {
        mState = CapabilityPreparationState::kDECLARED;
    }
    return status;
}

SafetyStatus ByteLevelBpeTokenizer::acquireResources(PreparedResources const& prepared) noexcept
{
    auto status = prepared.memory(kNODES, mNodeStorage);
    if (status.okStatus())
    {
        status = prepared.memory(kOFFSETS, mOffsetStorage);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kENTRIES, mEntryStorage);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kREVERSE, mReverseStorage);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kSPECIALS, mSpecialStorage);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kTOKEN_BYTES, mTokenBytesStorage);
    }
    return status;
}

void ByteLevelBpeTokenizer::bindResources() noexcept
{
    mNodes = static_cast<BpeNode*>(mNodeStorage.data);
    mOffsets = static_cast<Utf8Offset*>(mOffsetStorage.data);
    mEntries = static_cast<TokenEntry*>(mEntryStorage.data);
    mReverse = static_cast<ReverseEntry*>(mReverseStorage.data);
    mSpecials = static_cast<SpecialEntry*>(mSpecialStorage.data);
    mTokenBytes = static_cast<char*>(mTokenBytesStorage.data);
    mEntryCount = 0;
    mSpecialCount = 0;
    std::fill_n(mReverse, mLimits.vocabularySize, ReverseEntry{});
}

SafetyStatus ByteLevelBpeTokenizer::materializeVocabularyToken(OwnedToken const& token, std::size_t& cursor) noexcept
{
    if (token.id < 0 || token.id >= mLimits.vocabularySize || mReverse[token.id].occupied)
    {
        return invalid("Tokenizer vocabulary IDs are not unique and bounded");
    }
    auto const offset = static_cast<std::uint32_t>(cursor);
    auto const length = static_cast<std::uint32_t>(token.bytes.size());
    std::memcpy(mTokenBytes + cursor, token.bytes.data(), token.bytes.size());
    mEntries[mEntryCount++] = {hashBytes(token.bytes.data(), token.bytes.size()), offset, length, token.id, token.id};
    mReverse[token.id] = {offset, length, true};
    cursor += token.bytes.size();
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::materializeVocabulary(std::size_t& cursor) noexcept
{
    for (auto const& token : mConfiguredVocabulary)
    {
        auto const status = materializeVocabularyToken(token, cursor);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

void ByteLevelBpeTokenizer::materializeSpecialToken(OwnedToken const& token, std::size_t& cursor) noexcept
{
    auto const offset = static_cast<std::uint32_t>(cursor);
    auto const length = static_cast<std::uint32_t>(token.bytes.size());
    std::memcpy(mTokenBytes + cursor, token.bytes.data(), token.bytes.size());
    mSpecials[mSpecialCount++] = {offset, length, token.id, token.special};
    if (token.id >= 0 && token.id < mLimits.vocabularySize && !mReverse[token.id].occupied)
    {
        mReverse[token.id] = {offset, length, true};
    }
    cursor += token.bytes.size();
}

void ByteLevelBpeTokenizer::materializeSpecialTokens(std::size_t& cursor) noexcept
{
    for (auto const& token : mConfiguredSpecials)
    {
        materializeSpecialToken(token, cursor);
    }
}

bool ByteLevelBpeTokenizer::tokenEntryLess(TokenEntry const& left, TokenEntry const& right) const noexcept
{
    if (left.hash != right.hash)
    {
        return left.hash < right.hash;
    }
    auto const compared
        = std::memcmp(mTokenBytes + left.offset, mTokenBytes + right.offset, std::min(left.length, right.length));
    return compared != 0 ? compared < 0 : left.length < right.length;
}

void ByteLevelBpeTokenizer::sortRuntimeTokens() noexcept
{
    std::sort(mEntries, mEntries + mEntryCount,
        [this](TokenEntry const& left, TokenEntry const& right) { return tokenEntryLess(left, right); });
    std::sort(mSpecials, mSpecials + mSpecialCount, [](SpecialEntry const& left, SpecialEntry const& right) {
        return left.length != right.length ? left.length > right.length : left.id < right.id;
    });
}

SafetyStatus ByteLevelBpeTokenizer::materializeConfiguration() noexcept
{
    std::size_t cursor{0};
    auto const status = materializeVocabulary(cursor);
    if (!status.okStatus())
    {
        return status;
    }
    materializeSpecialTokens(cursor);
    sortRuntimeTokens();
    releaseConfiguration();
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::initialize(PreparedResources const& prepared) noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Byte-level tokenizer was not declared");
    }
    auto const status = acquireResources(prepared);
    if (!status.okStatus())
    {
        return status;
    }
    bindResources();
    auto const materialized = materializeConfiguration();
    if (materialized.okStatus())
    {
        mState = CapabilityPreparationState::kREADY;
    }
    return materialized;
}

ByteLevelBpeTokenizer::TokenEntry const* ByteLevelBpeTokenizer::lookup(
    char const* data, std::size_t size) const noexcept
{
    std::uint64_t const hash = hashBytes(data, size);
    auto const* first = std::lower_bound(mEntries, mEntries + mEntryCount, hash,
        [](TokenEntry const& entry, std::uint64_t value) { return entry.hash < value; });
    for (auto const* entry = first; entry != mEntries + mEntryCount && entry->hash == hash; ++entry)
    {
        if (entry->length == size && std::memcmp(mTokenBytes + entry->offset, data, size) == 0)
        {
            return entry;
        }
    }
    return nullptr;
}

ByteLevelBpeTokenizer::SpecialEntry const* ByteLevelBpeTokenizer::specialAt(
    char const* data, std::size_t size) const noexcept
{
    for (std::size_t index = 0; index < mSpecialCount; ++index)
    {
        auto const& special = mSpecials[index];
        if (special.length <= size && std::memcmp(data, mTokenBytes + special.offset, special.length) == 0)
        {
            return &special;
        }
    }
    return nullptr;
}

ByteLevelBpeTokenizer::SpecialEntry const* ByteLevelBpeTokenizer::specialById(int32_t id) const noexcept
{
    SpecialEntry const* begin = mSpecials;
    SpecialEntry const* end = begin + mSpecialCount;
    auto const* found = std::find_if(begin, end, [id](SpecialEntry const& entry) { return entry.id == id; });
    return found == end ? nullptr : found;
}

int32_t ByteLevelBpeTokenizer::specialTokenId(std::string_view token) const noexcept
{
    if (mState != CapabilityPreparationState::kREADY)
    {
        return -1;
    }
    auto const* special = specialAt(token.data(), token.size());
    return special != nullptr && special->length == token.size() ? special->id : -1;
}

bool ByteLevelBpeTokenizer::hasTokenId(int32_t id) const noexcept
{
    return mState == CapabilityPreparationState::kREADY && id >= 0
        && (specialById(id) != nullptr
            || (static_cast<std::size_t>(id) < static_cast<std::size_t>(mLimits.vocabularySize)
                && mReverse[id].occupied));
}

SafetyStatus ByteLevelBpeTokenizer::appendOutputToken(
    int32_t id, int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept
{
    if (outputSize == capacity)
    {
        return invalid("Encoded prompt exceeds prepared capacity");
    }
    output[outputSize++] = id;
    return SafetyStatus::ok();
}

std::size_t ByteLevelBpeTokenizer::nextSpecialOffset(TextView text, std::size_t begin) const noexcept
{
    for (std::size_t offset = begin; offset < text.size; ++offset)
    {
        if (specialAt(text.data + offset, text.size - offset) != nullptr)
        {
            return offset;
        }
    }
    return text.size;
}

SafetyStatus ByteLevelBpeTokenizer::encodeSpecialSegment(
    SpecialEntry const& special, int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept
{
    return appendOutputToken(special.id, output, capacity, outputSize);
}

SafetyStatus ByteLevelBpeTokenizer::encodeOrdinarySegment(TextView text, std::size_t begin, std::size_t end,
    int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    return encodeOrdinary(text.data + begin, end - begin, output, capacity, outputSize);
}

SafetyStatus ByteLevelBpeTokenizer::encodeSegment(
    TextView text, std::size_t& cursor, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    auto const* special = specialAt(text.data + cursor, text.size - cursor);
    auto const end = special == nullptr ? nextSpecialOffset(text, cursor + 1) : cursor + special->length;
    auto const status = special == nullptr ? encodeOrdinarySegment(text, cursor, end, output, capacity, outputSize)
                                           : encodeSpecialSegment(*special, output, capacity, outputSize);
    if (status.okStatus())
    {
        cursor = end;
    }
    return status;
}

SafetyStatus ByteLevelBpeTokenizer::encodeSegments(
    TextView text, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    for (std::size_t cursor = 0; cursor < text.size;)
    {
        auto const status = encodeSegment(text, cursor, output, capacity, outputSize);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::encode(
    TextView text, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    outputSize = 0;
    if (mState != CapabilityPreparationState::kREADY || text.data == nullptr || text.size == 0
        || text.size > mLimits.maxPromptBytes || output == nullptr || capacity == 0)
    {
        return invalid("Invalid byte-level tokenizer input");
    }
    return encodeSegments(text, output, capacity, outputSize);
}

SafetyStatus ByteLevelBpeTokenizer::decodeOffsets(char const* data, std::size_t size, std::size_t& count) noexcept
{
    count = 0;
    for (std::size_t cursor = 0; cursor < size;)
    {
        std::size_t const begin = cursor;
        std::uint32_t codepoint{0};
        if (count == mLimits.maxPromptBytes || !decodeCodepoint(data, size, cursor, codepoint))
        {
            return invalid("Prompt is not bounded valid UTF-8");
        }
        mOffsets[count++] = {static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(cursor), codepoint,
            unicodeCategory(codepoint)};
    }
    return SafetyStatus::ok();
}

bool ByteLevelBpeTokenizer::asciiAt(std::size_t index, std::size_t count, char value) const noexcept
{
    return index < count && mOffsets[index].codepoint == static_cast<std::uint32_t>(value);
}

std::size_t ByteLevelBpeTokenizer::matchSuffix(std::size_t begin, std::size_t count, char const* suffix) const noexcept
{
    std::size_t cursor = begin + 1;
    std::size_t part{0};
    while (suffix[part] != 0 && cursor < count && mOffsets[cursor].codepoint <= 0x7F
        && std::tolower(static_cast<unsigned char>(mOffsets[cursor].codepoint)) == suffix[part])
    {
        ++cursor;
        ++part;
    }
    return suffix[part] == 0 ? cursor : begin;
}

std::size_t ByteLevelBpeTokenizer::matchContraction(std::size_t begin, std::size_t count) const noexcept
{
    if (!asciiAt(begin, count, '\''))
    {
        return begin;
    }
    char const* suffixes[]{"re", "ve", "ll", "s", "t", "m", "d"};
    for (auto const* suffix : suffixes)
    {
        auto const end = matchSuffix(begin, count, suffix);
        if (end != begin)
        {
            return end;
        }
    }
    return begin;
}

std::size_t ByteLevelBpeTokenizer::matchLetters(std::size_t begin, std::size_t count) const noexcept
{
    std::size_t cursor = begin;
    if (mOffsets[cursor].category != 1 && mOffsets[cursor].category != 2 && mOffsets[cursor].codepoint != '\r'
        && mOffsets[cursor].codepoint != '\n')
    {
        ++cursor;
    }
    auto const firstLetter = cursor;
    while (cursor < count && mOffsets[cursor].category == 1)
    {
        ++cursor;
    }
    return cursor > firstLetter ? cursor : begin;
}

std::size_t ByteLevelBpeTokenizer::matchDigits(std::size_t begin, std::size_t count) const noexcept
{
    if (mOffsets[begin].category != 2)
    {
        return begin;
    }
    std::size_t end = begin;
    while (end < count && end - begin < 3 && mOffsets[end].category == 2)
    {
        ++end;
    }
    return end;
}

std::size_t ByteLevelBpeTokenizer::matchPunctuation(std::size_t begin, std::size_t count) const noexcept
{
    std::size_t cursor = begin + (asciiAt(begin, count, ' ') ? 1 : 0);
    auto const punctuation = cursor;
    while (cursor < count && mOffsets[cursor].category == 4 && mOffsets[cursor].codepoint != '\r'
        && mOffsets[cursor].codepoint != '\n')
    {
        ++cursor;
    }
    if (cursor == punctuation)
    {
        return begin;
    }
    while (cursor < count && (mOffsets[cursor].codepoint == '\r' || mOffsets[cursor].codepoint == '\n'))
    {
        ++cursor;
    }
    return cursor;
}

std::size_t ByteLevelBpeTokenizer::matchWhitespace(std::size_t begin, std::size_t count) const noexcept
{
    if (mOffsets[begin].category != 3)
    {
        return begin;
    }
    std::size_t end = begin + 1;
    while (end < count && mOffsets[end].category == 3)
    {
        ++end;
    }
    for (std::size_t cursor = end; cursor > begin; --cursor)
    {
        auto const codepoint = mOffsets[cursor - 1].codepoint;
        if (codepoint == '\r' || codepoint == '\n')
        {
            return cursor;
        }
    }
    return end < count && end - begin > 1 ? end - 1 : end;
}

std::size_t ByteLevelBpeTokenizer::nextPieceEnd(std::size_t begin, std::size_t count) const noexcept
{
    std::size_t end = matchContraction(begin, count);
    end = end == begin ? matchLetters(begin, count) : end;
    end = end == begin ? matchDigits(begin, count) : end;
    end = end == begin ? matchPunctuation(begin, count) : end;
    return end == begin ? matchWhitespace(begin, count) : end;
}

SafetyStatus ByteLevelBpeTokenizer::encodePretokenizedSpan(char const* data, std::size_t begin, std::size_t end,
    int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    if (end <= begin)
    {
        return invalid("Prompt does not match the Qwen pre-tokenizer grammar");
    }
    auto const byteBegin = mOffsets[begin].begin;
    auto const byteEnd = mOffsets[end - 1].end;
    return encodeBpe(data + byteBegin, byteEnd - byteBegin, output, capacity, outputSize);
}

SafetyStatus ByteLevelBpeTokenizer::encodePretokenizedSpans(
    char const* data, std::size_t count, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    for (std::size_t begin = 0; begin < count;)
    {
        auto const end = nextPieceEnd(begin, count);
        auto const status = encodePretokenizedSpan(data, begin, end, output, capacity, outputSize);
        if (!status.okStatus())
        {
            return status;
        }
        begin = end;
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::encodeOrdinary(
    char const* data, std::size_t size, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    std::size_t count{0};
    auto const status = decodeOffsets(data, size, count);
    if (!status.okStatus())
    {
        return status;
    }
    return encodePretokenizedSpans(data, count, output, capacity, outputSize);
}

SafetyStatus ByteLevelBpeTokenizer::initializeBpeNodes(char const* data, std::size_t size) noexcept
{
    for (std::size_t index = 0; index < size; ++index)
    {
        auto const* token = lookup(data + index, 1);
        if (token == nullptr)
        {
            return invalid("Tokenizer vocabulary is missing a byte token");
        }
        mNodes[index] = {static_cast<std::uint32_t>(index), 1, index + 1 == size ? -1 : static_cast<int32_t>(index + 1),
            token->id, token->rank};
    }
    return SafetyStatus::ok();
}

void ByteLevelBpeTokenizer::considerMerge(char const* data, int32_t left, MergeCandidate& candidate) const noexcept
{
    int32_t const right = mNodes[left].next;
    if (right < 0)
    {
        return;
    }
    auto const length = static_cast<std::size_t>(mNodes[left].length) + mNodes[right].length;
    auto const* token = lookup(data + mNodes[left].begin, length);
    bool const better = token != nullptr
        && (candidate.token == nullptr || token->rank < candidate.token->rank
            || (token->rank == candidate.token->rank && mNodes[left].begin < mNodes[candidate.left].begin));
    if (better)
    {
        candidate = {left, token};
    }
}

ByteLevelBpeTokenizer::MergeCandidate ByteLevelBpeTokenizer::findBestMerge(
    char const* data, std::size_t size) const noexcept
{
    MergeCandidate candidate{};
    for (int32_t left = 0; left >= 0 && left < static_cast<int32_t>(size); left = mNodes[left].next)
    {
        considerMerge(data, left, candidate);
    }
    return candidate;
}

void ByteLevelBpeTokenizer::applyMerge(MergeCandidate const& candidate) noexcept
{
    int32_t const right = mNodes[candidate.left].next;
    auto& left = mNodes[candidate.left];
    left.length += mNodes[right].length;
    left.id = candidate.token->id;
    left.rank = candidate.token->rank;
    left.next = mNodes[right].next;
}

void ByteLevelBpeTokenizer::mergeBpeNodes(char const* data, std::size_t size) noexcept
{
    for (auto candidate = findBestMerge(data, size); candidate.token != nullptr; candidate = findBestMerge(data, size))
    {
        applyMerge(candidate);
    }
}

SafetyStatus ByteLevelBpeTokenizer::appendBpeNodes(
    int32_t* output, std::size_t capacity, std::size_t& outputSize) const noexcept
{
    for (int32_t index = 0; index >= 0; index = mNodes[index].next)
    {
        auto const status = appendOutputToken(mNodes[index].id, output, capacity, outputSize);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::encodeBpe(
    char const* data, std::size_t size, int32_t* output, std::size_t capacity, std::size_t& outputSize) noexcept
{
    if (size == 0 || size > mLimits.maxPromptBytes)
    {
        return invalid("BPE span exceeds prepared storage");
    }
    if (auto const* token = lookup(data, size))
    {
        return appendOutputToken(token->id, output, capacity, outputSize);
    }
    auto const status = initializeBpeNodes(data, size);
    if (!status.okStatus())
    {
        return status;
    }
    mergeBpeNodes(data, size);
    return appendBpeNodes(output, capacity, outputSize);
}

bool ByteLevelBpeTokenizer::isStopToken(int32_t id, TokenIdView stops) const noexcept
{
    return std::find(stops.data, stops.data + stops.size, id) != stops.data + stops.size;
}

SafetyStatus ByteLevelBpeTokenizer::validateDecodedEntry(
    int32_t id, std::size_t limit, DecodeState const& state) const noexcept
{
    if (id < 0 || id >= mLimits.vocabularySize || !mReverse[id].occupied || state.bytes > limit
        || mReverse[id].length > limit - state.bytes)
    {
        return invalid("Decoded response exceeds its validated bounds");
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::consumeEntryUtf8(ReverseEntry const& entry, DecodeState& state) const noexcept
{
    for (std::size_t byte = 0; byte < entry.length; ++byte)
    {
        auto const value = static_cast<std::uint8_t>(mTokenBytes[entry.offset + byte]);
        if (!consumeUtf8Byte(value, state.remaining, state.value, state.minimum))
        {
            return invalid("Decoded response is not valid UTF-8");
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus ByteLevelBpeTokenizer::measureDecodedToken(
    int32_t id, std::size_t limit, DecodeState& state) const noexcept
{
    if (auto const* special = specialById(id))
    {
        return special->special ? SafetyStatus::ok() : invalid("Generated token has unsupported added-token behavior");
    }
    auto status = validateDecodedEntry(id, limit, state);
    if (!status.okStatus())
    {
        return status;
    }
    status = consumeEntryUtf8(mReverse[id], state);
    if (status.okStatus())
    {
        state.bytes += mReverse[id].length;
    }
    return status;
}

SafetyStatus ByteLevelBpeTokenizer::measureDecoded(
    TokenIdView tokens, TokenIdView stops, std::size_t limit, DecodeState& state) const noexcept
{
    for (std::size_t index = 0; index < tokens.size; ++index)
    {
        if (isStopToken(tokens.data[index], stops))
        {
            break;
        }
        auto const status = measureDecodedToken(tokens.data[index], limit, state);
        if (!status.okStatus())
        {
            return status;
        }
    }
    return state.remaining == 0 ? SafetyStatus::ok() : invalid("Decoded response is not valid UTF-8");
}

void ByteLevelBpeTokenizer::copyDecoded(TokenIdView tokens, TokenIdView stops, char* output) const noexcept
{
    std::size_t destination{0};
    for (std::size_t index = 0; index < tokens.size; ++index)
    {
        int32_t const id = tokens.data[index];
        if (isStopToken(id, stops))
        {
            break;
        }
        if (specialById(id) == nullptr)
        {
            std::memcpy(output + destination, mTokenBytes + mReverse[id].offset, mReverse[id].length);
            destination += mReverse[id].length;
        }
    }
}

void ByteLevelBpeTokenizer::publishDecoded(TokenIdView tokens, TokenIdView stops, DecodeState const& state,
    MutableTextBuffer output, std::size_t& outputSize) const noexcept
{
    copyDecoded(tokens, stops, output.data);
    output.data[state.bytes] = '\0';
    outputSize = state.bytes;
}

bool ByteLevelBpeTokenizer::validDecodeInput(
    TokenIdView tokens, TokenIdView stops, MutableTextBuffer output) const noexcept
{
    return mState == CapabilityPreparationState::kREADY && tokens.data != nullptr && stops.data != nullptr
        && output.data != nullptr && output.capacity != 0;
}

SafetyStatus ByteLevelBpeTokenizer::decode(
    TokenIdView tokens, TokenIdView stopTokens, MutableTextBuffer output, std::size_t& outputSize) const noexcept
{
    outputSize = 0;
    if (!validDecodeInput(tokens, stopTokens, output))
    {
        return invalid("Invalid tokenizer response buffers");
    }
    DecodeState state{};
    auto const limit = std::min(mLimits.maxResponseBytes, output.capacity - 1);
    auto const status = measureDecoded(tokens, stopTokens, limit, state);
    if (!status.okStatus())
    {
        return status;
    }
    publishDecoded(tokens, stopTokens, state, output, outputSize);
    return SafetyStatus::ok();
}

void ByteLevelBpeTokenizer::resetRuntime() noexcept
{
    mNodeStorage = {};
    mOffsetStorage = {};
    mEntryStorage = {};
    mReverseStorage = {};
    mSpecialStorage = {};
    mTokenBytesStorage = {};
    mEntries = nullptr;
    mReverse = nullptr;
    mSpecials = nullptr;
    mTokenBytes = nullptr;
    mNodes = nullptr;
    mOffsets = nullptr;
    mEntryCount = 0;
    mSpecialCount = 0;
}

void ByteLevelBpeTokenizer::releaseConfiguration() noexcept
{
    std::vector<OwnedToken>().swap(mConfiguredVocabulary);
    std::vector<OwnedToken>().swap(mConfiguredSpecials);
    mTokenBytesRequired = 0;
}

SafetyStatus ByteLevelBpeTokenizer::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    resetRuntime();
    releaseConfiguration();
    mState = CapabilityPreparationState::kCLOSED;
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
