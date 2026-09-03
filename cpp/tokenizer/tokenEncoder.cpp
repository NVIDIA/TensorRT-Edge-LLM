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

#include "tokenEncoder.h"
#include "common/inputLimits.h"
#include "tokenizerUtils.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace tokenizer
{

// Size limits for token encoder processing
constexpr size_t LARGE_PIECE_WARNING_BYTES = 65536; // 64KB warning threshold

TokenEncoder::TokenEncoder(Type type) noexcept
    : mType(type)
    , mVocabSize(0)
{
}

bool TokenEncoder::initialize(TokenToRanks const& vocab, TokenToRanks const& specialTokens)
{
    if (vocab.empty())
    {
        return false;
    }

    mEncoder = vocab;
    mSpecialTokensEncoder = specialTokens;

    // Build reverse mappings using utility function
    mDecoder = reverseEncoder(mEncoder);
    mSpecialTokensDecoder = reverseEncoder(mSpecialTokensEncoder);

    // Calculate vocab size as total number of tokens
    mVocabSize = mEncoder.size() + mSpecialTokensEncoder.size();
    return true;
}

bool TokenEncoder::encode(std::string const& piece, std::vector<Rank>& output) const noexcept
{
    if (piece.empty())
    {
        return true;
    }

    if (piece.size() > limits::tokenizer::kMaxTokenPieceSizeBytes)
    {
        LOG_ERROR("Input text piece too large: %zu bytes", piece.size());
        return false;
    }

    if (piece.size() > LARGE_PIECE_WARNING_BYTES) // 64KB warning per piece
    {
        LOG_WARNING("Very large piece encountered: %zu bytes", piece.size());
    }

    try
    {
        switch (mType)
        {
        case BPE: bytePairEncode(piece, output); break;
        default: LOG_ERROR("Unknown or unsupported encoder type: %s", getTypeString(mType).c_str()); return false;
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("TokenEncoder::encode failed on piece: %s", piece.c_str());
        return false;
    }
}

bool TokenEncoder::decode(std::vector<Rank> const& tokens, std::string& output, bool skipSpecialTokens) const noexcept
{
    try
    {
        output.clear();
        output.reserve(tokens.size() * 4); // Rough estimate

        for (Rank token : tokens)
        {
            if (skipSpecialTokens && mSpecialTokensDecoder.find(token) != mSpecialTokensDecoder.end())
            {
                continue;
            }
            auto it = mDecoder.find(token);
            if (it != mDecoder.end())
            {
                output += it->second;
            }
            else if (!skipSpecialTokens)
            {
                auto specialIt = mSpecialTokensDecoder.find(token);
                if (specialIt != mSpecialTokensDecoder.end())
                {
                    output += specialIt->second;
                }
                else
                {
                    LOG_ERROR("Unknown token %d during decode", token);
                    return false;
                }
            }
            // Skip unknown tokens if skipSpecialTokens is true
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("TokenEncoder::decode failed: %s", e.what());
        return false;
    }
}

bool TokenEncoder::hasToken(std::string const& token) const noexcept
{
    return mEncoder.find(token) != mEncoder.end() || mSpecialTokensEncoder.find(token) != mSpecialTokensEncoder.end();
}

Rank TokenEncoder::getTokenRank(std::string const& token) const noexcept
{
    auto it = mEncoder.find(token);
    if (it != mEncoder.end())
    {
        return it->second;
    }

    auto specialIt = mSpecialTokensEncoder.find(token);
    if (specialIt != mSpecialTokensEncoder.end())
    {
        return specialIt->second;
    }

    return -1; // Token not found
}

std::string TokenEncoder::getRankToken(Rank rank) const
{
    auto it = mDecoder.find(rank);
    if (it != mDecoder.end())
    {
        return it->second;
    }

    auto specialIt = mSpecialTokensDecoder.find(rank);
    if (specialIt != mSpecialTokensDecoder.end())
    {
        return specialIt->second;
    }

    return ""; // Rank not found
}

void TokenEncoder::bytePairEncode(std::string const& piece, std::vector<Rank>& output) const
{
    if (piece.empty())
    {
        return;
    }

    // Check if the piece is already in vocabulary
    auto it = mEncoder.find(piece);
    if (it != mEncoder.end())
    {
        output.emplace_back(it->second);
        return;
    }

    // Initialize parts vector with (start_position, rank) pairs
    // "rank" here is used as merge priority — either vocab rank (tiktoken-style)
    // or explicit merge priority from the merges list (SentencePiece-style).
    std::vector<std::pair<int, Rank>> parts;
    parts.reserve(piece.size() + 1);

    auto const MAX_INT = std::numeric_limits<int>::max();
    auto const MAX_RANK = std::numeric_limits<Rank>::max();
    std::pair<int, Rank> minRank{MAX_INT, MAX_RANK};

    // Helper: get the merge priority for a pair of adjacent substrings.
    // When mUseMergePriorities is true, look up using null-byte separated key (left + '\0' + right)
    // to avoid ambiguity when different (left, right) splits produce the same concatenation.
    // Otherwise fall back to vocabulary rank lookup on the concatenated string (tiktoken-style).
    auto getPairPriority = [&](std::string const& left, std::string const& right) -> Rank {
        if (mUseMergePriorities)
        {
            std::string key = left;
            key += '\0';
            key += right;
            auto mergeIt = mMergePriorities.find(key);
            return (mergeIt != mMergePriorities.end()) ? mergeIt->second : MAX_RANK;
        }
        else
        {
            std::string merged = left + right;
            auto vocabIt = mEncoder.find(merged);
            return (vocabIt != mEncoder.end()) ? vocabIt->second : MAX_RANK;
        }
    };

    // Build character boundary positions.
    // For tiktoken-style (byte-level BPE), each byte is a unit.
    // For SentencePiece-style (merge priorities), each UTF-8 character is a unit.
    std::vector<int> charPositions;
    if (mUseMergePriorities)
    {
        // Split at UTF-8 character boundaries
        for (size_t pos = 0; pos < piece.size();)
        {
            charPositions.push_back(static_cast<int>(pos));
            unsigned char c = static_cast<unsigned char>(piece[pos]);
            if (c < 0x80)
                pos += 1;
            else if ((c & 0xE0) == 0xC0)
                pos += 2;
            else if ((c & 0xF0) == 0xE0)
                pos += 3;
            else
                pos += 4;
        }
        charPositions.push_back(static_cast<int>(piece.size()));
    }
    else
    {
        // Byte-level: each byte is a unit
        for (size_t i = 0; i <= piece.size(); ++i)
        {
            charPositions.push_back(static_cast<int>(i));
        }
    }

    // Initialize with bigram priorities (pairs of adjacent characters/bytes)
    size_t numChars = charPositions.size() - 1; // number of initial units
    if (numChars <= 1)
    {
        // Single character or empty piece: no merges possible, push raw token lookup.
        auto it = mEncoder.find(piece);
        if (it != mEncoder.end())
        {
            output.push_back(it->second);
        }
        return;
    }
    for (size_t i = 0; i < numChars - 1; ++i)
    {
        std::string left(piece.begin() + charPositions[i], piece.begin() + charPositions[i + 1]);
        std::string right(piece.begin() + charPositions[i + 1], piece.begin() + charPositions[i + 2]);
        Rank rank = getPairPriority(left, right);

        if (rank < minRank.second)
        {
            minRank = std::make_pair(static_cast<int>(i), rank);
        }

        parts.emplace_back(charPositions[i], rank);
    }

    // Add sentinel values
    parts.emplace_back(charPositions[numChars - 1], MAX_RANK);
    parts.emplace_back(charPositions[numChars], MAX_RANK);

    // Helper function to get merged rank for position i (merging parts[i]..parts[i+2] into one,
    // then checking the priority of the new pair: (merged, parts[i+2]..parts[i+3]))
    auto getMergedRank = [&](size_t i) -> Rank {
        if (i + 3 >= parts.size())
        {
            return MAX_RANK;
        }

        std::string left(piece.begin() + parts[i].first, piece.begin() + parts[i + 2].first);
        std::string right(piece.begin() + parts[i + 2].first, piece.begin() + parts[i + 3].first);
        return getPairPriority(left, right);
    };

    // Main BPE loop
    while (minRank.second != MAX_RANK)
    {
        size_t i = static_cast<size_t>(minRank.first);

        // Update adjacent ranks
        if (i > 0)
        {
            parts[i - 1].second = getMergedRank(i - 1);
        }
        parts[i].second = getMergedRank(i);

        // Remove the merged part
        parts.erase(parts.begin() + i + 1);

        // Find new minimum rank
        minRank = std::make_pair(MAX_INT, MAX_RANK);
        for (size_t j = 0; j < parts.size() - 1; ++j)
        {
            if (parts[j].second < minRank.second)
            {
                minRank = std::make_pair(static_cast<int>(j), parts[j].second);
            }
        }
    }

    // Collect final tokens
    for (size_t i = 0; i < parts.size() - 1; ++i)
    {
        std::string token(piece.begin() + parts[i].first, piece.begin() + parts[i + 1].first);
        auto tokenIt = mEncoder.find(token);
        if (tokenIt != mEncoder.end())
        {
            output.emplace_back(tokenIt->second);
        }
        else if (mByteFallback)
        {
            // Encode each byte as <0xNN> token
            for (unsigned char byte : token)
            {
                char buf[8];
                snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                auto byteIt = mEncoder.find(buf);
                if (byteIt != mEncoder.end())
                {
                    output.emplace_back(byteIt->second);
                }
                else
                {
                    LOG_ERROR("Byte fallback token not found: '%s'", buf);
                    return;
                }
            }
        }
        else
        {
            LOG_ERROR("Token not found in encoder during bytePairEncode: '%s'", token.c_str());
            return;
        }
    }
}

std::string TokenEncoder::getTypeString(Type type) const
{
    switch (type)
    {
    case BPE: return "BPE";
    default: return "UNKNOWN";
    }
}

} // namespace tokenizer
} // namespace trt_edgellm
