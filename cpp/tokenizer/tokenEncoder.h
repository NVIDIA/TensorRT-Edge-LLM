/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/logger.h"

namespace trt_edgellm
{
namespace tokenizer
{

using Rank = std::int32_t;
using TokenToRanks = std::unordered_map<std::string, Rank>;
using RanksToToken = std::unordered_map<Rank, std::string>;

/**
 * @brief TokenEncoder class for different tokenization encoding algorithms
 */
class TokenEncoder
{
public:
    /*!
     * @brief Token encoder algorithm type
     */
    enum Type
    {
        BPE,           //!< Byte Pair Encoding
        SENTENCEPIECE, //!< SentencePiece encoding (unimplemented)
        WORDPIECE      //!< WordPiece encoding (unimplemented)
    };

    /*!
     * @brief Constructor for token encoder
     * @param type Encoder algorithm type (default: BPE)
     */
    TokenEncoder(Type type = BPE);
    ~TokenEncoder() = default;

    /**
     * @brief Initialize with vocabulary
     * @param vocab Main vocabulary mapping
     * @param specialTokens Special tokens mapping
     * @return true if vocab is non-empty and initialization completes;
     *         false if vocab is empty
     */
    bool initialize(TokenToRanks const& vocab, TokenToRanks const& specialTokens = {});

    /**
     * @brief Encode a piece of text using the algorithm
     * @param piece Input text piece (already pretokenized)
     * @param output Vector to store token IDs
     * @return true if piece is within size limits and encoding completes successfully;
     *         false if piece exceeds 1MB limit or encoding algorithm fails
     */
    bool encode(std::string const& piece, std::vector<Rank>& output) const;

    /**
     * @brief Decode token IDs back to text
     * @param tokens Vector of token IDs
     * @param output String to store result
     * @param skipSpecialTokens Whether to skip special tokens
     * @return true if all tokens are found in vocabulary or skipped successfully;
     *         false if unknown tokens are encountered and skipSpecialTokens is false
     */
    bool decode(std::vector<Rank> const& tokens, std::string& output, bool skipSpecialTokens = false) const;

    /*!
     * @brief Get encoder type
     * @return Encoder algorithm type
     */
    Type getType() const noexcept
    {
        return mType;
    }

    /*!
     * @brief Get vocabulary size
     * @return Total number of tokens in vocabulary
     */
    size_t getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    /*!
     * @brief Check if token exists in vocabulary
     * @param token Token string to check
     * @return true if token exists, false otherwise
     */
    bool hasToken(std::string const& token) const;

    /*!
     * @brief Get token rank from token string
     * @param token Token string
     * @return Token rank/ID
     */
    Rank getTokenRank(std::string const& token) const;

    /*!
     * @brief Get token string from rank
     * @param rank Token rank/ID
     * @return Token string
     */
    std::string getRankToken(Rank rank) const;

private:
    /**
     * @brief Byte Pair Encoding implementation
     */
    void bytePairEncode(std::string const& piece, std::vector<Rank>& output) const;
    /**
     * @brief Get string representation of encoder type
     */
    std::string getTypeString(Type type) const;

    Type mType;
    TokenToRanks mEncoder;
    RanksToToken mDecoder;
    TokenToRanks mSpecialTokensEncoder;
    RanksToToken mSpecialTokensDecoder;
    size_t mVocabSize;
};

} // namespace tokenizer
} // namespace trt_edgellm
