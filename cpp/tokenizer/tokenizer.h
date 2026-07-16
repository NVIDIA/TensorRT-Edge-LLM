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
#include <cassert>
#include <filesystem>
#include <forward_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "preTokenizer.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenEncoder.h"

namespace trt_edgellm
{
namespace tokenizer
{

/*!
 * @brief Chat template role configuration
 */
struct ChatTemplateRole
{
    std::string prefix; //!< Prefix for this role
    std::string suffix; //!< Suffix for this role
};

/*!
 * @brief Chat template content type configuration
 */
struct ChatTemplateContentType
{
    std::string format; //!< Format string for this content type
};

/*!
 * @brief Chat template configuration
 */
struct ChatTemplateConfig
{
    std::string modelPath;                                   //!< Model path or identifier
    std::unordered_map<std::string, ChatTemplateRole> roles; //!< Role configurations (system, user, assistant)
    std::unordered_map<std::string, ChatTemplateContentType>
        contentTypes;                     //!< Content type configurations (text, image, video)
    std::string generationPrompt;         //!< Standard generation prompt (thinking disabled)
    std::string generationPromptThinking; //!< Generation prompt with thinking enabled (optional, model-specific)
    std::string defaultSystemPrompt;      //!< Default system prompt
    bool trimContent{false};              //!< Whether to trim whitespace from message content (matches Jinja | trim)
    std::string prefixStrategy;           //!< Optional VLA prefix padding strategy (e.g. pi05_compact_prefix)
    int32_t maxSeqLen{0};                 //!< Optional padded prefix capacity from processed_chat_template.json
};

/*!
 * @brief Type of text partition for tokenization
 */
typedef enum TEXT_PART_TYPE
{
    TEXT_PART_SPECIAL_TOKEN, //!< Partition contains a special token
    TEXT_PART_RAW_TEXT       //!< Partition contains raw text to be tokenized
} TEXT_PART_TYPE;

/*!
 * @brief Text partition representation for tokenization
 *
 * Represents either a special token or a raw text segment to be tokenized.
 */
struct textPartition
{
    /*!
     * @brief Constructor for special token partition
     * @param _token Token ID for the special token
     */
    textPartition(Rank _token) noexcept
        : type(TEXT_PART_SPECIAL_TOKEN)
        , token(_token)
        , rawText(_dummy)
        , offset(0)
        , length(0)
    {
    }

    /*!
     * @brief Constructor for raw text partition
     * @param _rawText Reference to the raw text string
     * @param _offset Offset into the raw text string
     * @param _length Length of the text partition
     */
    textPartition(std::string const& _rawText, int _offset, int _length)
        : type(TEXT_PART_RAW_TEXT)
        , token(-1)
        , rawText(_rawText)
        , offset(_offset)
        , length(_length)
    {
        assert(offset >= 0);
        assert(length >= 1);
        assert(offset + length <= static_cast<int>(rawText.length()));
    }

    const TEXT_PART_TYPE type;  //!< Type of partition (special token or raw text)
    Rank const token;           //!< Token ID (valid when type is TEXT_PART_SPECIAL_TOKEN)
    std::string const _dummy;   //!< Dummy string for special token partitions
    std::string const& rawText; //!< Reference to raw text (valid when type is TEXT_PART_RAW_TEXT)
    int const offset;           //!< Offset into rawText
    int const length;           //!< Length of the partition
};

/*!
 * @brief Tokenizer class for encoding and decoding text
 *
 * Provides tokenization functionality including pretokenization, encoding, and decoding.
 * Supports loading from HuggingFace model directories.
 */
class Tokenizer
{
public:
    Tokenizer() noexcept;
    ~Tokenizer() noexcept = default;

    // TODO: Add constructor with preTokenizer and tokenEncoder
    // Tokenizer(std::string const& patStr, BPETokenToRanks& mergeableRanks, BPETokenToRanks& specialTokens,
    //     Rank const& bosId = -1, Rank const& eosId = -1, Rank const& padId = -1, Rank const& unkId = -1);

    /**
     * @brief Encode text to token IDs
     * @param text Input text to encode
     * @param addBos Whether to add beginning-of-sequence token
     * @param addEos Whether to add end-of-sequence token
     * @return Vector of token IDs
     * @throws std::runtime_error if tokenization encounters an error
     */
    std::vector<Rank> encode(std::string const& text, bool addBos = false, bool addEos = false) const;

    /**
     * @brief Decode token IDs back to text
     * @param tokens Vector of token IDs
     * @param skipSpecialTokens Whether to skip special tokens in output
     * @return Decoded text string (well-formed UTF-8; invalid byte sequences
     *         are replaced with U+FFFD via sanitizeUtf8Streaming/Flush)
     */
    std::string decode(std::vector<Rank> const& tokens, bool skipSpecialTokens = false) const;

    /**
     * @brief Single-token piece lookup for the streaming hot path.
     *
     * Forwards to TokenEncoder::getRankToken with the skip-special policy
     * applied. Returns an empty string when the token is a special token and
     * skipSpecialTokens=true, or when rank is unknown (matches the silent-skip
     * semantics of decode()).
     *
     * @param token           Token ID (Rank).
     * @param skipSpecialTokens Skip special tokens (BOS/EOS/etc).
     * @return Raw piece bytes (possibly not independently valid UTF-8) or "".
     */
    std::string idToPiece(Rank token, bool skipSpecialTokens = true) const;

    /**
     * @brief Load tokenizer from HuggingFace model directory
     * @param modelDir Path to the model directory containing tokenizer files
     * @return true if directory exists, tokenizer.json is found and parsed successfully,
     *         pretokenizer and encoder are created successfully; false if directory doesn't exist,
     *         tokenizer.json is missing/corrupt, or initialization fails
     */
    bool loadFromHF(std::filesystem::path const& modelDir);

    /*!
     * @brief Get total vocabulary size
     * @return Number of tokens in vocabulary
     */
    int getNumVocab() const noexcept
    {
        return mNumVocab;
    }

    /*!
     * @brief Get beginning-of-sequence token ID
     * @return BOS token ID
     */
    Rank getBosId() const noexcept
    {
        return mBosId;
    }

    /*!
     * @brief Get end-of-sequence token ID
     * @return EOS token ID
     */
    Rank getEosId() const noexcept
    {
        return mEosId;
    }

    /*!
     * @brief Check if a token is an end-of-sequence token
     * @param tokenId Token ID to check
     * @return true if the token is any EOS token (primary or additional)
     */
    bool isEosToken(Rank tokenId) const noexcept
    {
        if (tokenId == mEosId)
        {
            return true;
        }
        for (Rank id : mAdditionalEosIds)
        {
            if (tokenId == id)
            {
                return true;
            }
        }
        return false;
    }

    /*!
     * @brief Set additional end-of-sequence token IDs
     * @param ids Vector of additional EOS token IDs (from config.json eos_token_id array)
     */
    void setAdditionalEosIds(std::vector<Rank> const& ids)
    {
        mAdditionalEosIds.clear();
        for (Rank id : ids)
        {
            if (id != mEosId && id >= 0)
            {
                mAdditionalEosIds.push_back(id);
            }
        }
    }

    /*!
     * @brief Get padding token ID
     * @return PAD token ID (returns EOS if PAD is not set)
     */
    Rank getPadId() const noexcept
    {
        return mPadId == -1 ? mEosId : mPadId;
    }

    /*!
     * @brief Get unknown token ID
     * @return UNK token ID
     */
    Rank getUnkId() const noexcept
    {
        return mUnkId;
    }

    /*!
     * @brief Get token ID by token string
     * @param token Token string to look up
     * @return Token ID, or -1 if not found
     */
    Rank getTokenId(std::string const& token) const noexcept
    {
        return mTokenEncoder ? mTokenEncoder->getTokenRank(token) : -1;
    }

    /**
     * @brief Check if tokenizer is properly initialized
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const noexcept;

    /**
     * @brief Load chat template configuration from JSON file
     * @param chatTemplateFile Path to the processed_chat_template.json file
     * @return true if chat template is loaded successfully; false if file doesn't exist or parsing fails
     */
    bool loadChatTemplate(std::filesystem::path const& chatTemplateFile);

    /**
     * @brief Apply chat template to a request
     * @param request Request object containing messages
     * @param formattedRequest Output formatted request object that will be populated
     * @param applyChatTemplate Whether to apply full chat template formatting (with special tokens) or raw
     * concatenation
     * @param addGenerationPrompt Whether to add generation prompt at the end (only used when applyChatTemplate is true)
     * @param enableThinking Whether to enable thinking mode for models that support it
     * @return true if chat template is applied successfully; false if encountered errors
     */
    bool applyChatTemplate(rt::LLMGenerationRequest::Request const& request,
        rt::LLMGenerationRequest::FormattedRequest& formattedRequest, bool applyChatTemplate = true,
        bool addGenerationPrompt = true, bool enableThinking = false) const;

    /*!
     * @brief Special token string → id map (e.g. Alpamayo trajectory placeholder expansion).
     */
    TokenToRanks const& getSpecialTokensEncoder() const noexcept
    {
        return mSpecialTokensEncoder;
    }

    /**
     * @brief Get default system prompt from chat template
     * @return Default system prompt string
     */
    std::string getDefaultSystemPrompt() const noexcept
    {
        return mChatTemplate.defaultSystemPrompt;
    }

    //! Prefix padding strategy from processed_chat_template.json (e.g. pi05_compact_prefix).
    std::string const& getPrefixStrategy() const noexcept
    {
        return mChatTemplate.prefixStrategy;
    }

    //! Padded prefix capacity from processed_chat_template.json (0 = use engine max only).
    int32_t getMaxSeqLen() const noexcept
    {
        return mChatTemplate.maxSeqLen;
    }

protected:
    /**
     * @brief Parse tokenizer.json to extract configuration
     * @param tokenizerFile Path to tokenizer.json file
     * @param vocab Output vocabulary mapping
     * @param specialTokens Output special tokens mapping
     * @return true if file size is valid, file opens successfully, JSON parses correctly,
     *         and pretokenizer/vocabulary load; false if file is too large, can't be opened,
     *         contains invalid JSON, or configuration is malformed
     */
    bool parseTokenizerConfig(
        std::filesystem::path const& tokenizerFile, TokenToRanks& vocab, TokenToRanks& specialTokens);

    /**
     * @brief Parse tokenizer_config.json to extract special token IDs
     * @param configFile Path to tokenizer_config.json file
     * @param specialTokens Output special tokens mapping
     * @return true if file size is valid, file opens successfully, and JSON parses correctly;
     *         false if file is too large, can't be opened, or contains invalid JSON
     */
    bool parseSpecialTokenConfig(std::filesystem::path const& configFile, TokenToRanks& specialTokens);

    /**
     * @brief Create appropriate pretokenizer based on configuration
     * @param preTokenizerConfig JSON configuration for pretokenizer
     * @return Unique pointer to created pretokenizer: RegexSplit for recognized Split/Regex types,
     *         Sequence for pretokenizer arrays, or default empty Sequence for unknown configurations
     */
    std::unique_ptr<PreTokenizer> createPreTokenizer(nlohmann::json const& preTokenizerConfig);

    /**
     * @brief Determine encoder type from configuration
     * @param modelConfig JSON configuration for the model
     * @return TokenEncoder type
     */
    TokenEncoder::Type determineEncoderType(nlohmann::json const& modelConfig);

    /**
     * @brief Load vocabulary from tokenizer.json
     * @param modelConfig JSON model configuration
     * @param vocab Output vocabulary mapping
     * @return true if model configuration contains valid vocab object and tokens are loaded;
     *         false if vocab section is missing/invalid or no valid tokens found
     */
    bool loadVocabulary(nlohmann::json const& modelConfig, TokenToRanks& vocab);

    /**
     * @brief Load special tokens from tokenizer configuration
     * @param tokenizerConfig JSON tokenizer configuration
     * @param specialTokens Output special tokens mapping
     * @return true if special tokens are extracted and processed successfully;
     *         false if extraction fails
     */
    bool loadSpecialTokens(nlohmann::json const& tokenizerConfig, TokenToRanks& specialTokens);

    /**
     * @brief Partition text into special tokens and raw text segments using forward_list
     * @param text Input text to partition
     * @param partitions Output forward_list of text partitions
     * @return true if text is partitioned successfully without exceptions;
     *         false if partitioning fails due to processing errors
     */
    bool partitionSpecialTokens(std::string const& text, std::forward_list<textPartition>& partitions) const noexcept;

    /**
     * @brief Add BOS token if configured
     * @param tokens Token vector to modify
     */
    void appendBos(std::vector<Rank>& tokens) const;

    /**
     * @brief Add EOS token if configured
     * @param tokens Token vector to modify
     */
    void appendEos(std::vector<Rank>& tokens) const;

    // Core components
    std::unique_ptr<PreTokenizer> mPreTokenizer; //!< Pretokenizer for splitting input text
    std::unique_ptr<TokenEncoder> mTokenEncoder; //!< Token encoder for encoding/decoding

    // Special token mappings for fast lookup
    TokenToRanks mSpecialTokensEncoder;                          //!< Special tokens encoder mapping
    std::unordered_map<Rank, std::string> mSpecialTokensDecoder; //!< Special tokens decoder mapping

    // Configuration
    int mNumVocab;                       //!< Total vocabulary size
    Rank mBosId;                         //!< Beginning-of-sequence token ID
    Rank mEosId;                         //!< End-of-sequence token ID
    Rank mPadId;                         //!< Padding token ID
    Rank mUnkId;                         //!< Unknown token ID
    std::vector<Rank> mAdditionalEosIds; //!< Additional EOS token IDs (from config.json)
    Rank mImgContextId;                  //!< Image context token ID

    // Chat template
    ChatTemplateConfig mChatTemplate; //!< Chat template configuration

    // Normalizer: list of (pattern, replacement) pairs applied before pre-tokenization.
    std::vector<std::pair<std::string, std::string>> mNormalizerReplacements;

    // Decoder replacements: list of (pattern, replacement) pairs applied after decoding (reverse of normalizer).
    std::vector<std::pair<std::string, std::string>> mDecoderReplacements;

    // ByteFallback: whether to convert <0xNN> pieces back to raw bytes during decode.
    // Set when tokenizer.json decoder contains a "ByteFallback" step (SentencePiece-style).
    bool mByteFallbackDecode{false};

    // State
    bool mInitialized; //!< Whether tokenizer is initialized
};

/*!
 * @brief Streaming emit: consume newly-arrived token ids and produce valid UTF-8 delta text.
 *
 * Looks up the piece bytes for each token in `allTokenIds[s.sentTokenCount..end)`, prepends
 * `s.pendingBytes`, and passes the concatenated buffer through `sanitizeUtf8Streaming`.
 * Invalid byte sequences become U+FFFD; trailing incomplete codepoints are held in
 * `s.pendingBytes` for the next call. `s.sentTokenCount` is advanced unconditionally.
 *
 * Contract:
 *   - Must be called once per iteration per slot, after the iteration has appended its tokens.
 *   - Output is always well-formed UTF-8.
 *   - Works for both vanilla (1 new token) and spec-decode (N new tokens) paths.
 *
 * @param s             Slot state — modified in place.
 * @param tok           Tokenizer used for piece lookup (Tokenizer::idToPiece).
 * @param allTokenIds   Full token id sequence for this slot.
 * @param skipSpecial   Skip special tokens (true for consumer-facing streaming).
 * @return Delta text (may be empty if all new bytes were held as incomplete).
 */
std::string emitDelta(
    rt::SlotStreamState& s, Tokenizer const& tok, std::vector<int32_t> const& allTokenIds, bool skipSpecial);

/*!
 * @brief Final-iteration flush: convert any held incomplete bytes to U+FFFD and clear.
 *
 * Called once per slot at finish time when `finishedStates[i]` flips to 1.
 * Output is well-formed UTF-8 (one U+FFFD per held byte).
 */
std::string emitDeltaFlush(rt::SlotStreamState& s);

} // namespace tokenizer
} // namespace trt_edgellm
