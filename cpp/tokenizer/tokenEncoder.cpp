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

#include "tokenEncoder.h"
#include "tokenizerUtils.h"
#include <cassert>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace tokenizer
{

// Size limits for token encoder processing
constexpr size_t MAX_PIECE_SIZE_BYTES = 1024 * 1024; // 1MB limit
constexpr size_t LARGE_PIECE_WARNING_BYTES = 65536;  // 64KB warning threshold

TokenEncoder::TokenEncoder(Type type)
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

bool TokenEncoder::encode(std::string const& piece, std::vector<Rank>& output) const
{
    if (piece.empty())
    {
        return true;
    }

    if (piece.size() > MAX_PIECE_SIZE_BYTES) // 1MB limit
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

bool TokenEncoder::decode(std::vector<Rank> const& tokens, std::string& output, bool skipSpecialTokens) const
{
    try
    {
        output.clear();
        output.reserve(tokens.size() * 4); // Rough estimate

        for (Rank token : tokens)
        {
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

bool TokenEncoder::hasToken(std::string const& token) const
{
    return mEncoder.find(token) != mEncoder.end() || mSpecialTokensEncoder.find(token) != mSpecialTokensEncoder.end();
}

Rank TokenEncoder::getTokenRank(std::string const& token) const
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
    std::vector<std::pair<int, Rank>> parts;
    parts.reserve(piece.size() + 1);

    auto const MAX_INT = std::numeric_limits<int>::max();
    auto const MAX_RANK = std::numeric_limits<Rank>::max();
    std::pair<int, Rank> minRank{MAX_INT, MAX_RANK};

    // Initialize with bigram ranks
    for (size_t i = 0; i < piece.size() - 1; ++i)
    {
        Rank rank = MAX_RANK;
        std::string bigram(piece.begin() + i, piece.begin() + i + 2);

        auto bigramIt = mEncoder.find(bigram);
        if (bigramIt != mEncoder.end())
        {
            rank = bigramIt->second;
        }

        if (rank < minRank.second)
        {
            minRank = std::make_pair(static_cast<int>(i), rank);
        }

        parts.emplace_back(static_cast<int>(i), rank);
    }

    // Add sentinel values
    parts.emplace_back(static_cast<int>(piece.size() - 1), MAX_RANK);
    parts.emplace_back(static_cast<int>(piece.size()), MAX_RANK);

    // Helper function to get merged rank
    auto getMergedRank = [&](size_t i) -> Rank {
        if (i + 3 >= parts.size())
        {
            return MAX_RANK;
        }

        std::string merged(piece.begin() + parts[i].first, piece.begin() + parts[i + 3].first);
        auto mergedIt = mEncoder.find(merged);
        return (mergedIt != mEncoder.end()) ? mergedIt->second : MAX_RANK;
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
