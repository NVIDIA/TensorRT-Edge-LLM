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

#include "tokenizer.h"
#include "common/inputLimits.h"
#include "common/utf8.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizerUtils.h"
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace tokenizer
{

//! Convert a single hex character to its integer value (0-15), or -1 on invalid input.
static int hexCharToInt(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    return -1;
}

// Chat template role names
constexpr char kRoleSystem[] = "system";

Tokenizer::Tokenizer() noexcept
    : mNumVocab(0)
    , mBosId(-1)
    , mEosId(-1)
    , mPadId(-1)
    , mUnkId(-1)
    , mInitialized(false)
{
}

bool Tokenizer::loadFromHF(std::filesystem::path const& modelDir)
{
    if (!std::filesystem::exists(modelDir) || !std::filesystem::is_directory(modelDir))
    {
        LOG_ERROR("Model directory does not exist or is not a directory: %s", modelDir.c_str());
        return false;
    }

    // Reset state
    mInitialized = false;
    mPreTokenizer.reset();
    mTokenEncoder.reset();
    mSpecialTokensEncoder.clear();
    mSpecialTokensDecoder.clear();

    std::filesystem::path tokenizerFile = modelDir / "tokenizer.json";
    std::filesystem::path configFile = modelDir / "tokenizer_config.json";
    std::filesystem::path modelConfigFile = modelDir / "config.json";

    // Determine encoder type and load vocabulary
    TokenToRanks vocab;
    TokenToRanks specialTokens;

    if (!std::filesystem::exists(tokenizerFile))
    {
        LOG_ERROR("tokenizer.json not found in %s", modelDir.c_str());
        return false;
    }

    // Parse main tokenizer configuration
    if (!parseTokenizerConfig(tokenizerFile, vocab, specialTokens))
    {
        LOG_ERROR("Failed to parse tokenizer configuration");
        return false;
    }

    // Parse special token configuration (optional)
    if (std::filesystem::exists(configFile))
    {
        if (!parseSpecialTokenConfig(configFile, specialTokens))
        {
            LOG_WARNING("Failed to parse special token configuration, using defaults");
        }
    }
    else
    {
        LOG_WARNING("tokenizer_config.json not found, using default special token configuration");
    }
    LOG_INFO("Loaded %zu special tokens", specialTokens.size());

    if (mTokenEncoder)
    {
        mTokenEncoder->initialize(vocab, specialTokens);
    }

    // Store special tokens for fast lookup
    mSpecialTokensEncoder = specialTokens;
    mSpecialTokensDecoder = reverseEncoder(mSpecialTokensEncoder);

    mNumVocab = static_cast<int>(mTokenEncoder->getVocabSize());

    // Pre-initialize Unicode lookup tables to avoid first-call latency during encode
    unicodeCptFlags(0);

    // Load chat template (required)
    std::filesystem::path const chatTemplatePath = modelDir / "processed_chat_template.json";
    if (!loadChatTemplate(chatTemplatePath))
    {
        LOG_ERROR(
            "Please ensure processed_chat_template.json exists in the model/engine directory, and it follows the "
            "format specified in the documentation.");
        return false;
    }

    mInitialized = true;
    LOG_INFO("Successfully loaded tokenizer from %s (vocab_size=%d)", modelDir.c_str(), mNumVocab);
    return true;
}

// Processes tokenizer.json
bool Tokenizer::parseTokenizerConfig(
    std::filesystem::path const& tokenizerFile, TokenToRanks& vocab, TokenToRanks& specialTokens)
{
    // Validate file size before reading
    if (!validateFileSize(tokenizerFile, limits::tokenizer::kMaxConfigFileSizeBytes))
    {
        return false;
    }

    std::ifstream file(tokenizerFile);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open tokenizer.json: %s", tokenizerFile.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Json jsonData;
    try
    {
        jsonData = Json::parse(content);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse tokenizer.json: %s", e.what());
        return false;
    }

    // Parse normalizer (applied before pre-tokenization)
    if (jsonData.contains("normalizer") && !jsonData["normalizer"].is_null())
    {
        auto const& normConfig = jsonData["normalizer"];
        std::string const normType = normConfig.value("type", "");
        if (normType == "Replace")
        {
            std::string pattern;
            if (normConfig.contains("pattern") && normConfig["pattern"].is_object())
            {
                pattern = normConfig["pattern"].value("String", "");
            }
            std::string content = normConfig.value("content", "");
            if (!pattern.empty())
            {
                mNormalizerReplacements.emplace_back(std::move(pattern), std::move(content));
                LOG_INFO("Tokenizer normalizer: Replace '%s' -> '%s'", mNormalizerReplacements.back().first.c_str(),
                    mNormalizerReplacements.back().second.c_str());
            }
        }
        else if (normType == "Sequence")
        {
            if (normConfig.contains("normalizers") && normConfig["normalizers"].is_array())
            {
                for (auto const& step : normConfig["normalizers"])
                {
                    std::string const stepType = step.value("type", "");
                    if (stepType == "Replace" && step.contains("pattern") && step["pattern"].is_object())
                    {
                        std::string pattern = step["pattern"].value("String", "");
                        std::string content = step.value("content", "");
                        if (!pattern.empty())
                        {
                            mNormalizerReplacements.emplace_back(std::move(pattern), std::move(content));
                        }
                    }
                }
            }
        }
    }

    // Parse decoder (applied after decoding tokens back to text)
    if (jsonData.contains("decoder") && !jsonData["decoder"].is_null())
    {
        auto const& decConfig = jsonData["decoder"];
        std::string const decType = decConfig.value("type", "");
        if (decType == "Replace")
        {
            std::string pattern;
            if (decConfig.contains("pattern") && decConfig["pattern"].is_object())
            {
                pattern = decConfig["pattern"].value("String", "");
            }
            std::string content = decConfig.value("content", "");
            if (!pattern.empty())
            {
                mDecoderReplacements.emplace_back(std::move(pattern), std::move(content));
            }
        }
        else if (decType == "Sequence")
        {
            if (decConfig.contains("decoders") && decConfig["decoders"].is_array())
            {
                for (auto const& step : decConfig["decoders"])
                {
                    std::string const stepType = step.value("type", "");
                    if (stepType == "Replace" && step.contains("pattern") && step["pattern"].is_object())
                    {
                        std::string pattern = step["pattern"].value("String", "");
                        std::string content = step.value("content", "");
                        if (!pattern.empty())
                        {
                            mDecoderReplacements.emplace_back(std::move(pattern), std::move(content));
                        }
                    }
                    else if (stepType == "ByteFallback")
                    {
                        mByteFallbackDecode = true;
                    }
                }
            }
        }
        else if (decType == "ByteFallback")
        {
            mByteFallbackDecode = true;
        }
    }

    // Create pretokenizer
    if (jsonData.contains("pre_tokenizer") && !jsonData["pre_tokenizer"].is_null())
    {
        mPreTokenizer = createPreTokenizer(jsonData["pre_tokenizer"]);
        if (!mPreTokenizer)
        {
            LOG_ERROR("Failed to create pretokenizer");
            return false;
        }
    }
    else
    {
        LOG_WARNING("No pretokenizer configuration found, using default sequence");
        mPreTokenizer = std::make_unique<Sequence>();
    }

    if (!jsonData.contains("model") || !jsonData["model"].is_object())
    {
        LOG_ERROR("No model configuration found in tokenizer.json");
        return false;
    }

    TokenEncoder::Type encoderType = determineEncoderType(jsonData["model"]);
    if (!loadVocabulary(jsonData["model"], vocab))
    {
        LOG_ERROR("Failed to load vocabulary");
        return false;
    }

    // Load special tokens from tokenizer.json
    if (!loadSpecialTokens(jsonData, specialTokens))
    {
        LOG_WARNING("Failed to load special tokens from tokenizer.json");
    }

    // Create token encoder with byte_fallback support if configured
    mTokenEncoder = std::make_unique<TokenEncoder>(encoderType);
    bool byteFallback = jsonData["model"].value("byte_fallback", false);
    if (byteFallback)
    {
        mTokenEncoder->setByteFallback(true);
    }

    // Load explicit BPE merges only for SentencePiece-style tokenizers (byte_fallback=true).
    // These tokenizers (e.g. Gemma) define merge order independently of vocab rank.
    // Tiktoken-style tokenizers (e.g. Qwen) have merges in tokenizer.json but use vocab rank
    // for merge ordering — loading merge priorities would break their tokenization.
    if (byteFallback && jsonData["model"].contains("merges") && jsonData["model"]["merges"].is_array())
    {
        auto const& mergesArray = jsonData["model"]["merges"];
        if (!mergesArray.empty())
        {
            std::unordered_map<std::string, Rank> mergePriorities;
            mergePriorities.reserve(mergesArray.size());

            for (size_t i = 0; i < mergesArray.size(); ++i)
            {
                std::string left, right;
                if (mergesArray[i].is_string())
                {
                    // Format: "tokenA tokenB" — split on first space
                    std::string const& mergeStr = mergesArray[i].get_ref<std::string const&>();
                    auto spacePos = mergeStr.find(' ');
                    if (spacePos == std::string::npos)
                    {
                        LOG_WARNING("Invalid merge entry at index %zu: '%s'", i, mergeStr.c_str());
                        continue;
                    }
                    left = mergeStr.substr(0, spacePos);
                    right = mergeStr.substr(spacePos + 1);
                }
                else if (mergesArray[i].is_array() && mergesArray[i].size() == 2 && mergesArray[i][0].is_string()
                    && mergesArray[i][1].is_string())
                {
                    // Format: ["tokenA", "tokenB"]
                    left = mergesArray[i][0].get<std::string>();
                    right = mergesArray[i][1].get<std::string>();
                }
                else
                {
                    LOG_WARNING("Unrecognized merge entry format at index %zu", i);
                    continue;
                }
                // Use null-byte separator to avoid ambiguity when different (left, right) pairs
                // produce the same concatenation (e.g. "▁th"+"e" vs "▁t"+"he" both → "▁the").
                std::string key = left;
                key += '\0';
                key += right;
                mergePriorities[std::move(key)] = static_cast<Rank>(i);
            }

            LOG_INFO("Loaded %zu BPE merge priorities", mergePriorities.size());
            mTokenEncoder->setMergePriorities(std::move(mergePriorities));
        }
    }

    return true;
}

bool Tokenizer::parseSpecialTokenConfig(std::filesystem::path const& configFile, TokenToRanks& specialTokens)
{
    // Validate file size before reading
    if (!validateFileSize(configFile, limits::tokenizer::kMaxConfigFileSizeBytes))
    {
        return false;
    }

    std::ifstream file(configFile);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open tokenizer_config.json: %s", configFile.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(content);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse tokenizer_config.json: %s", e.what());
        return false;
    }

    // Load special tokens from tokenizer_config.json
    if (!loadSpecialTokens(jsonConfig, specialTokens))
    {
        LOG_WARNING("Failed to load special tokens from tokenizer_config.json");
    }

    // Parse special token IDs
    auto parseSpecialToken = [&specialTokens](Json const& config, std::string const& field) -> Rank {
        if (!config.contains(field))
        {
            return -1;
        }

        if (config[field].is_string())
        {
            std::string token = config[field].get<std::string>();
            auto it = specialTokens.find(token);
            if (it != specialTokens.end())
            {
                return it->second;
            }
            LOG_WARNING("Special token '%s' not found in vocabulary", token.c_str());
            return -1;
        }
        else if (config[field].is_object() && config[field].contains("content"))
        {
            std::string token = config[field]["content"].get<std::string>();
            auto it = specialTokens.find(token);
            if (it != specialTokens.end())
            {
                return it->second;
            }
            LOG_WARNING("Special token '%s' not found in vocabulary", token.c_str());
            return -1;
        }

        return -1;
    };

    mBosId = parseSpecialToken(jsonConfig, "bos_token");
    mEosId = parseSpecialToken(jsonConfig, "eos_token");
    mPadId = parseSpecialToken(jsonConfig, "pad_token");
    mUnkId = parseSpecialToken(jsonConfig, "unk_token");
    mImgContextId = parseSpecialToken(jsonConfig, "context_image_token");

    return true;
}

std::unique_ptr<PreTokenizer> Tokenizer::createPreTokenizer(Json const& preTokenizerConfig)
{
    // Handle sequence of pretokenizers
    if (preTokenizerConfig.contains("pretokenizers") && preTokenizerConfig["pretokenizers"].is_array())
    {
        auto sequence = std::make_unique<Sequence>();

        for (auto const& step : preTokenizerConfig["pretokenizers"])
        {
            if (step.contains("type") && step["type"].is_string())
            {
                std::string type = step["type"].get<std::string>();

                if (type == "Split" && step.contains("pattern"))
                {
                    // Handle Split with Regex pattern
                    if (step["pattern"].is_object() && step["pattern"].contains("Regex"))
                    {
                        std::string pattern = step["pattern"]["Regex"].get<std::string>();
                        std::string normalizedPattern = normalizeRegex(pattern);
                        sequence->addStep(std::make_unique<RegexSplit>(normalizedPattern));
                    }
                }
            }
        }

        return sequence;
    }

    // Handle single pretokenizer
    if (preTokenizerConfig.contains("type") && preTokenizerConfig["type"].is_string())
    {
        std::string type = preTokenizerConfig["type"].get<std::string>();

        if (type == "Split" && preTokenizerConfig.contains("pattern"))
        {
            if (preTokenizerConfig["pattern"].is_object() && preTokenizerConfig["pattern"].contains("Regex"))
            {
                std::string pattern = preTokenizerConfig["pattern"]["Regex"].get<std::string>();
                std::string normalizedPattern = normalizeRegex(pattern);
                return std::make_unique<RegexSplit>(normalizedPattern);
            }
        }
        else if (type == "Regex" && preTokenizerConfig.contains("pattern"))
        {
            std::string pattern = preTokenizerConfig["pattern"].get<std::string>();
            std::string normalizedPattern = normalizeRegex(pattern);
            return std::make_unique<RegexSplit>(normalizedPattern);
        }
    }

    LOG_WARNING("Unknown pretokenizer configuration, using default sequence");
    return std::make_unique<Sequence>();
}

TokenEncoder::Type Tokenizer::determineEncoderType(Json const& modelConfig)
{
    if (modelConfig.contains("type") && modelConfig["type"].is_string())
    {
        std::string type = modelConfig["type"].get<std::string>();

        if (type == "BPE")
        {
            return TokenEncoder::BPE;
        }
        else if (type == "Unigram")
        {
            return TokenEncoder::SENTENCEPIECE;
        }
        else if (type == "WordPiece")
        {
            return TokenEncoder::WORDPIECE;
        }
    }

    LOG_WARNING("Unknown or missing model type, defaulting to BPE");
    return TokenEncoder::BPE;
}

bool Tokenizer::loadVocabulary(Json const& modelConfig, TokenToRanks& vocab)
{
    if (!modelConfig.contains("vocab") || !modelConfig["vocab"].is_object())
    {
        LOG_ERROR("No vocabulary found in model configuration");
        return false;
    }

    // Byte-fallback tokenizers (SentencePiece-style, e.g. Gemma) store vocab keys as raw
    // characters. Do NOT apply GPT-2 byte-to-unicode decoding, which would cause collisions
    // (e.g. raw '\n' → id 107 and 'Ċ' → id 247723 both decode to '\n').
    bool const isByteFallback = modelConfig.contains("byte_fallback") && modelConfig["byte_fallback"].get<bool>();

    for (auto const& [hfToken, rank] : modelConfig["vocab"].items())
    {
        if (rank.is_number_integer())
        {
            try
            {
                std::string token = isByteFallback ? hfToken : decodeHFTokenToNormal(hfToken);
                vocab[token] = rank.get<Rank>();
            }
            catch (std::exception const& e)
            {
                LOG_WARNING("Failed to process vocabulary token '%s': %s", hfToken.c_str(), e.what());
            }
        }
    }

    if (vocab.empty())
    {
        LOG_ERROR("No valid vocabulary tokens loaded");
        return false;
    }

    LOG_INFO("Loaded %zu vocabulary tokens", vocab.size());
    return true;
}

bool Tokenizer::loadSpecialTokens(Json const& tokenizerConfig, TokenToRanks& specialTokens)
{
    // Load from tokenizer.json added_tokens
    if (tokenizerConfig.contains("added_tokens") && tokenizerConfig["added_tokens"].is_array())
    {
        for (auto const& token : tokenizerConfig["added_tokens"])
        {
            if (token.contains("id") && token.contains("content") && token["id"].is_number_integer()
                && token["content"].is_string())
            {
                try
                {
                    Rank specialId = token["id"].get<Rank>();
                    std::string content = token["content"].get<std::string>();

                    if (!content.empty())
                    {
                        specialTokens[content] = specialId;
                    }
                }
                catch (std::exception const& e)
                {
                    LOG_WARNING("Failed to parse added token: %s", e.what());
                }
            }
        }
    }
    // Also try to load additional special tokens from added_tokens_decoder
    if (tokenizerConfig.contains("added_tokens_decoder") && tokenizerConfig["added_tokens_decoder"].is_object())
    {
        for (auto const& [idStr, tokenData] : tokenizerConfig["added_tokens_decoder"].items())
        {
            if (tokenData.contains("content") && tokenData["content"].is_string())
            {
                try
                {
                    Rank specialId = std::stoi(idStr);
                    std::string content = tokenData["content"].get<std::string>();

                    if (!content.empty())
                    {
                        specialTokens[content] = specialId;
                    }
                }
                catch (std::exception const& e)
                {
                    LOG_WARNING("Failed to parse added token ID '%s': %s", idStr.c_str(), e.what());
                }
            }
        }
    }
    return !specialTokens.empty();
}

std::vector<Rank> Tokenizer::encode(std::string const& text, bool addBos, bool addEos) const
{
    if (!mInitialized || !mPreTokenizer || !mTokenEncoder)
    {
        LOG_ERROR("Tokenizer not properly initialized");
        return {};
    }

    // Validate input text size before any processing to prevent expensive operations on oversized input
    if (text.size() > limits::tokenizer::kMaxInputTextSizeBytes)
    {
        LOG_ERROR(
            "Input text too large (%zu bytes, max: %zu).", text.size(), limits::tokenizer::kMaxInputTextSizeBytes);
        return {};
    }

    std::vector<Rank> output;
    output.reserve(text.size() + (addBos ? 1 : 0) + (addEos ? 1 : 0));

    if (addBos)
    {
        appendBos(output);
    }

    if (!text.empty())
    {
        // Partition text into special tokens and raw text segments using forward_list
        std::forward_list<textPartition> partitions;
        partitions.emplace_front(text, 0, text.length());

        if (!partitionSpecialTokens(text, partitions))
        {
            LOG_ERROR("Failed to partition special tokens");
            return {};
        }

        // Process each partition
        for (auto const& part : partitions)
        {
            if (part.type == TEXT_PART_SPECIAL_TOKEN)
            {
                output.push_back(part.token);
            }
            else
            {
                // Process raw text partition
                std::string piece = part.rawText.substr(part.offset, part.length);

                // Apply normalizer replacements (e.g., space -> ▁ for SentencePiece-style tokenizers)
                for (auto const& [pattern, replacement] : mNormalizerReplacements)
                {
                    std::string::size_type pos = 0;
                    while ((pos = piece.find(pattern, pos)) != std::string::npos)
                    {
                        piece.replace(pos, pattern.size(), replacement);
                        pos += replacement.size();
                    }
                }

                // Process through pretokenizer
                std::vector<std::string> pieces;
                try
                {
                    pieces = mPreTokenizer->process(piece);
                }
                catch (std::exception const& e)
                {
                    LOG_ERROR("Pretokenizer failed: %s", e.what());
                    return {};
                }

                // Encode each piece
                for (auto const& subpiece : pieces)
                {
                    std::vector<Rank> pieceTokens;
                    if (!mTokenEncoder->encode(subpiece, pieceTokens))
                    {
                        LOG_ERROR("Failed to encode piece: %s", subpiece.c_str());
                        return {};
                    }
                    output.insert(output.end(), pieceTokens.begin(), pieceTokens.end());
                }
            }
        }
    }

    if (addEos)
    {
        appendEos(output);
    }

    return output;
}

bool Tokenizer::partitionSpecialTokens(
    std::string const& text, std::forward_list<textPartition>& partitions) const noexcept
{
    try
    {
        for (auto const& [specialToken, specialId] : mSpecialTokensEncoder)
        {
            for (auto it = partitions.begin(); it != partitions.end(); ++it)
            {
                auto& part = (*it);

                // if part not yet processed
                if (part.type == TEXT_PART_RAW_TEXT)
                {
                    auto& rawText = part.rawText;
                    auto baseOffset = part.offset;
                    auto baseLength = part.length;

                    // find occurrences of specialToken in rawText
                    while (true)
                    {
                        auto match = rawText.find(specialToken, baseOffset);
                        if ((match == std::string::npos)
                            || (static_cast<int>(match + specialToken.length()) > (baseOffset + baseLength)))
                        {
                            break;
                        }

                        auto basePos = std::distance(partitions.begin(), it);

                        // insert left part
                        if (match > static_cast<size_t>(baseOffset))
                        {
                            partitions.emplace_after(it, rawText, baseOffset, match - baseOffset);
                            ++it;
                        }

                        // insert special token
                        partitions.emplace_after(it, specialId);
                        ++it;

                        // remove original part
                        if (basePos == 0)
                        {
                            partitions.erase_after(partitions.before_begin());
                        }
                        else
                        {
                            partitions.erase_after(std::next(partitions.begin(), (basePos - 1)));
                        }

                        // insert right part and continue loop
                        if (match + specialToken.length() < static_cast<size_t>(baseOffset + baseLength))
                        {
                            int rightOffset = match + specialToken.length();
                            int rightLength = baseLength + baseOffset - (match + specialToken.length());
                            partitions.emplace_after(it, rawText, rightOffset, rightLength);
                            ++it;

                            baseOffset = rightOffset;
                            baseLength = rightLength;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
        }

        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Tokenizer::partitionSpecialTokens failed on text: %s", text.c_str());
        return false;
    }
}

std::string Tokenizer::decode(std::vector<Rank> const& tokens, bool skipSpecialTokens) const
{
    if (!mInitialized || !mTokenEncoder)
    {
        LOG_ERROR("Tokenizer not properly initialized");
        return "";
    }

    std::string raw;
    if (!mTokenEncoder->decode(tokens, raw, skipSpecialTokens))
    {
        LOG_ERROR("Failed to decode tokens");
        return "";
    }

    // Apply ByteFallback decoding: convert <0xNN> sequences to actual bytes.
    // SentencePiece byte_fallback tokenizers store individual byte tokens as "<0xNN>"
    // in the vocabulary. The ByteFallback decoder step converts them back to raw bytes.
    if (mByteFallbackDecode)
    {
        std::string decoded;
        decoded.reserve(raw.size());
        size_t i = 0;
        while (i < raw.size())
        {
            // Match pattern: <0xNN> where NN is exactly 2 hex digits
            if (i + 5 < raw.size() && raw[i] == '<' && raw[i + 1] == '0' && raw[i + 2] == 'x' && raw[i + 5] == '>')
            {
                char hi = raw[i + 3];
                char lo = raw[i + 4];
                int hiVal = hexCharToInt(hi);
                int loVal = hexCharToInt(lo);
                if (hiVal >= 0 && loVal >= 0)
                {
                    decoded.push_back(static_cast<char>((hiVal << 4) | loVal));
                    i += 6;
                    continue;
                }
            }
            decoded.push_back(raw[i]);
            ++i;
        }
        raw = std::move(decoded);
    }

    // Apply decoder replacements (reverse of normalizer, e.g., ▁ -> space)
    for (auto const& [pattern, replacement] : mDecoderReplacements)
    {
        std::string::size_type pos = 0;
        while ((pos = raw.find(pattern, pos)) != std::string::npos)
        {
            raw.replace(pos, pattern.size(), replacement);
            pos += replacement.size();
        }
    }

    // Route through the UTF-8 sanitizer to guarantee well-formed output. This
    // is a single-shot decode, so any trailing incomplete bytes must surface
    // as U+FFFD rather than be held for a later call.
    std::string pending;
    std::string output = utf8::sanitizeUtf8Streaming(raw, pending);
    output.append(utf8::sanitizeUtf8Flush(pending));
    return output;
}

std::string Tokenizer::idToPiece(Rank token, bool skipSpecialTokens) const
{
    if (!mInitialized || !mTokenEncoder)
    {
        return "";
    }
    // Special tokens: return empty when skipping, the textual content otherwise.
    if (mSpecialTokensDecoder.find(token) != mSpecialTokensDecoder.end())
    {
        if (skipSpecialTokens)
        {
            return "";
        }
        return mSpecialTokensDecoder.at(token);
    }
    std::string piece = mTokenEncoder->getRankToken(token);

    // Apply ByteFallback: convert single <0xNN> piece to raw byte.
    // A byte-fallback token is exactly 6 chars: "<0xNN>"
    if (mByteFallbackDecode && piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x'
        && piece[5] == '>')
    {
        int hiVal = hexCharToInt(piece[3]);
        int loVal = hexCharToInt(piece[4]);
        if (hiVal >= 0 && loVal >= 0)
        {
            piece = std::string(1, static_cast<char>((hiVal << 4) | loVal));
        }
    }

    // Apply decoder replacements (e.g., ▁ -> space)
    for (auto const& [pattern, replacement] : mDecoderReplacements)
    {
        std::string::size_type pos = 0;
        while ((pos = piece.find(pattern, pos)) != std::string::npos)
        {
            piece.replace(pos, pattern.size(), replacement);
            pos += replacement.size();
        }
    }
    return piece;
}

std::string emitDelta(
    rt::SlotStreamState& s, Tokenizer const& tok, std::vector<int32_t> const& allTokenIds, bool skipSpecial)
{
    // Fast path: no new tokens and no bytes held — nothing to do.
    if (allTokenIds.size() == s.sentTokenCount && s.pendingBytes.empty())
    {
        return "";
    }

    // Concatenate the piece bytes for every newly-arrived token. Raw bytes may
    // span multi-token UTF-8 codepoints or contain adversarial isolated bytes;
    // sanitizeUtf8Streaming fixes both.
    std::string raw;
    for (size_t i = s.sentTokenCount; i < allTokenIds.size(); ++i)
    {
        raw.append(tok.idToPiece(allTokenIds[i], skipSpecial));
    }
    s.sentTokenCount = allTokenIds.size();

    return utf8::sanitizeUtf8Streaming(raw, s.pendingBytes);
}

std::string emitDeltaFlush(rt::SlotStreamState& s)
{
    return utf8::sanitizeUtf8Flush(s.pendingBytes);
}

bool Tokenizer::isInitialized() const noexcept
{
    return mInitialized && mPreTokenizer && mTokenEncoder;
}

void Tokenizer::appendBos(std::vector<Rank>& tokens) const
{
    if (mBosId != -1)
    {
        tokens.push_back(mBosId);
    }
    else
    {
        LOG_DEBUG("BOS ID is not set. Not appending BOS token.");
    }
}

void Tokenizer::appendEos(std::vector<Rank>& tokens) const
{
    if (mEosId != -1)
    {
        tokens.push_back(mEosId);
    }
    else
    {
        LOG_DEBUG("EOS ID is not set. Not appending EOS token.");
    }
}

bool Tokenizer::loadChatTemplate(std::filesystem::path const& chatTemplateFile)
{
    if (!std::filesystem::exists(chatTemplateFile))
    {
        LOG_ERROR("Chat template file not found: %s", chatTemplateFile.c_str());
        return false;
    }

    // Validate file size before reading
    if (!validateFileSize(chatTemplateFile, limits::tokenizer::kChatTemplateFileSizeBytes))
    {
        return false;
    }

    std::ifstream file(chatTemplateFile);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open chat template file: %s", chatTemplateFile.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Json jsonData;
    try
    {
        jsonData = Json::parse(content);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse chat template JSON: %s", e.what());
        return false;
    }

    try
    {
        // Parse model path
        mChatTemplate.modelPath = jsonData.value("model_path", mChatTemplate.modelPath);

        // Parse roles,  which should contains [system, user, assistant]
        check::check(jsonData.contains("roles") && jsonData["roles"].is_object(),
            "Roles-field is required in chat template. And Shall be a JSON object.");
        for (auto const& [role, roleConfig] : jsonData["roles"].items())
        {
            ChatTemplateRole templateRole;
            templateRole.prefix = roleConfig.value("prefix", "");
            templateRole.suffix = roleConfig.value("suffix", "");
            mChatTemplate.roles[role] = templateRole;
        }

        // Parse non-text content types place holder format string.
        if (jsonData.contains("content_types") && jsonData["content_types"].is_object())
        {
            for (auto const& [contentType, contentConfig] : jsonData["content_types"].items())
            {
                ChatTemplateContentType templateContentType;
                templateContentType.format = contentConfig.value("format", "");
                if (templateContentType.format.empty())
                {
                    LOG_WARNING("Content type format is empty. Skip this content type: %s.", contentType.c_str());
                    continue;
                }
                mChatTemplate.contentTypes[contentType] = templateContentType;
            }
        }

        // Collect other fields from the chat template if exists.
        mChatTemplate.generationPrompt = jsonData.value("generation_prompt", mChatTemplate.generationPrompt);
        mChatTemplate.generationPromptThinking = jsonData.value("generation_prompt_thinking", "");
        mChatTemplate.defaultSystemPrompt = jsonData.value("default_system_prompt", mChatTemplate.defaultSystemPrompt);
        mChatTemplate.trimContent = jsonData.value("trim_content", false);
        mChatTemplate.prefixStrategy = jsonData.value("prefix_strategy", "");
        mChatTemplate.maxSeqLen = jsonData.value("max_seq_len", 0);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse chat template: %s", e.what());
        return false;
    }

    LOG_INFO("Successfully loaded chat template from %s (for model: %s)", chatTemplateFile.c_str(),
        mChatTemplate.modelPath.c_str());
    return true;
}

bool Tokenizer::applyChatTemplate(rt::LLMGenerationRequest::Request const& request,
    rt::LLMGenerationRequest::FormattedRequest& formattedRequest, bool applyChatTemplate, bool addGenerationPrompt,
    bool enableThinking) const
{
    if (request.messages.empty())
    {
        LOG_ERROR("Request shall contain at least one message to proceed with execution.");
        return false;
    }

    std::string formattedPrefixSystemPrompt{};
    std::string formattedCompleteRequest{};

    // Extract system prompt from first message or use default
    auto const& leadMessage = request.messages.front();
    std::string systemPrompt{};
    bool hasExplicitSystemMessage = false;

    if (leadMessage.role == kRoleSystem)
    {
        hasExplicitSystemMessage = true;
        for (auto const& content : leadMessage.contents)
        {
            if (content.type == "text")
            {
                systemPrompt += content.content;
            }
            else
            {
                LOG_WARNING("System message contents shall be all text. Find %s content type. Skip this content.",
                    content.type.c_str());
            }
        }
    }
    else if (applyChatTemplate && !mChatTemplate.defaultSystemPrompt.empty())
    {
        hasExplicitSystemMessage = true;
        systemPrompt = mChatTemplate.defaultSystemPrompt;
    }

    // Format system prompt (also format when there's an explicit system message with empty content,
    // since some models like Qwen3-ASR expect the system role block even when empty)
    if (!systemPrompt.empty() || (hasExplicitSystemMessage && applyChatTemplate))
    {
        if (applyChatTemplate)
        {
            auto roleIt = mChatTemplate.roles.find(kRoleSystem);
            if (roleIt != mChatTemplate.roles.end())
            {
                formattedPrefixSystemPrompt = roleIt->second.prefix + systemPrompt + roleIt->second.suffix;
            }
            else
            {
                LOG_WARNING("System role not found in chat template. Using raw content.");
                formattedPrefixSystemPrompt = systemPrompt;
            }
        }
        else
        {
            formattedPrefixSystemPrompt = systemPrompt;
        }
        formattedCompleteRequest = formattedPrefixSystemPrompt;
    }

    // Process messages
    for (size_t i = 0; i < request.messages.size(); ++i)
    {
        auto const& message = request.messages[i];

        if (message.role == kRoleSystem && i == 0)
        {
            continue;
        }

        auto roleIt = mChatTemplate.roles.find(message.role);
        if (roleIt == mChatTemplate.roles.end())
        {
            LOG_WARNING("Unknown role: %s", message.role.c_str());
            continue;
        }

        std::string formattedMessage;

        // Add role prefix only in chat template mode
        if (applyChatTemplate)
        {
            formattedMessage = roleIt->second.prefix;
        }

        // Process content items
        for (auto const& contentItem : message.contents)
        {
            if (contentItem.type == "text")
            {
                if (mChatTemplate.trimContent)
                {
                    auto const& s = contentItem.content;
                    auto start = s.find_first_not_of(" \t\n\r");
                    auto end = s.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos)
                    {
                        formattedMessage += s.substr(start, end - start + 1);
                    }
                }
                else
                {
                    formattedMessage += contentItem.content;
                }
            }
            else if (contentItem.type == "trajectory")
            {
                if (request.pastTrajectory)
                {
                    formattedMessage += rt::kTrajHistoryStartStr;
                    // One <|traj_history|> pad per binned axis value; Alpamayo1ActionRunner::preprocess replaces these
                    // with discrete trajectory tokens after encode().
                    size_t const numPadTokens = 3 * request.pastTrajectory->size();
                    for (size_t k = 0; k < numPadTokens; ++k)
                    {
                        formattedMessage += rt::kTrajHistoryPadStr;
                    }
                    formattedMessage += rt::kTrajHistoryEndStr;
                }
                else
                {
                    LOG_WARNING("Content type 'trajectory' has no request.pastTrajectory; skipping.");
                }
            }
            else
            {
                // Get content type format
                auto contentTypeIt = mChatTemplate.contentTypes.find(contentItem.type);
                if (contentTypeIt != mChatTemplate.contentTypes.end())
                {
                    formattedMessage += contentTypeIt->second.format;
                }
                else
                {
                    LOG_WARNING("Unknown content type: %s", contentItem.type.c_str());
                }
            }
        }

        // Add role suffix only in chat template mode
        if (applyChatTemplate)
        {
            formattedMessage += roleIt->second.suffix;
        }

        formattedCompleteRequest += formattedMessage;
    }

    // Add generation prompt (only in chat template mode)
    if (applyChatTemplate && addGenerationPrompt)
    {
        if (enableThinking && !mChatTemplate.generationPromptThinking.empty())
        {
            formattedCompleteRequest += mChatTemplate.generationPromptThinking;
        }
        else if (!mChatTemplate.generationPrompt.empty())
        {
            formattedCompleteRequest += mChatTemplate.generationPrompt;
        }
    }

    formattedRequest.formattedSystemPrompt = formattedPrefixSystemPrompt;
    formattedRequest.formattedCompleteRequest = formattedCompleteRequest;
    return true;
}

} // namespace tokenizer
} // namespace trt_edgellm
