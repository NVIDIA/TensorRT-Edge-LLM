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
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>

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
            if (skipSpecialTokens && isSkippableSpecial(token))
            {
                continue;
            }

            // An added token can also live in the base vocab (Nemotron), so its
            // definition has to win before the base-vocab lookup.
            auto addedIt = mSpecialTokensDecoder.find(token);
            if (addedIt != mSpecialTokensDecoder.end())
            {
                output += addedIt->second;
                continue;
            }

            auto vocabIt = mDecoder.find(token);
            if (vocabIt != mDecoder.end())
            {
                output += vocabIt->second;
                continue;
            }

            if (skipSpecialTokens)
            {
                continue;
            }

            LOG_ERROR("Unknown token %d during decode", token);
            return false;
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

void TokenEncoder::bytePairEncode(std::string const& pieceStr, std::vector<Rank>& output) const
{
    std::string_view const piece(pieceStr);
    if (piece.empty())
    {
        return;
    }

    // Check if the piece is already in vocabulary
    auto it = mEncoder.find(pieceStr);
    if (it != mEncoder.end())
    {
        output.emplace_back(it->second);
        return;
    }

    auto const MAX_RANK = std::numeric_limits<Rank>::max();

    // Helper: get the merge priority for a pair of adjacent substrings.
    // When mUseMergePriorities is true, look up using null-byte separated key (left + '\0' + right)
    // to avoid ambiguity when different (left, right) splits produce the same concatenation.
    // Otherwise fall back to vocabulary rank lookup on the concatenated string (tiktoken-style).
    std::string key;
    auto getPairPriority = [&](std::string_view left, std::string_view right) -> Rank {
        key.assign(left);
        if (mUseMergePriorities)
        {
            key += '\0';
            key += right;
            auto mergeIt = mMergePriorities.find(key);
            return (mergeIt != mMergePriorities.end()) ? mergeIt->second : MAX_RANK;
        }
        key += right;
        auto vocabIt = mEncoder.find(key);
        return (vocabIt != mEncoder.end()) ? vocabIt->second : MAX_RANK;
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
    int32_t const numChars = static_cast<int32_t>(charPositions.size()) - 1; // number of initial units
    if (numChars <= 1)
    {
        // Single character or empty piece: no merges possible, push raw token lookup.
        auto it = mEncoder.find(pieceStr);
        if (it != mEncoder.end())
        {
            output.push_back(it->second);
        }
        return;
    }

    // Merge with a min-heap of candidate pairs over a linked list of symbols: O(n log n) in the
    // piece length. SentencePiece-style tokenizers (Gemma) normalize spaces away before the
    // pre-tokenizer, so a piece can be a whole prompt segment of thousands of characters. Pairs are
    // ordered by (priority, start offset): the lowest priority merges first and, among equal
    // priorities, the leftmost, as in the reference algorithm.
    struct Symbol
    {
        int32_t start;
        int32_t end;
        int32_t prev;
        int32_t next;
    };
    std::vector<Symbol> symbols(static_cast<size_t>(numChars));
    for (int32_t i = 0; i < numChars; ++i)
    {
        symbols[i] = Symbol{charPositions[i], charPositions[i + 1], i - 1, i + 1 < numChars ? i + 1 : -1};
    }
    struct Candidate
    {
        Rank rank;
        int32_t start;
        int32_t left;
        int32_t leftEnd;
        int32_t rightEnd;
    };
    auto const later
        = [](Candidate const& a, Candidate const& b) { return a.rank != b.rank ? a.rank > b.rank : a.start > b.start; };
    std::vector<Candidate> heap;
    heap.reserve(static_cast<size_t>(numChars) * 2);
    auto const pushPair = [&](int32_t left) {
        if (left < 0 || symbols[left].next < 0)
        {
            return;
        }
        Symbol const& l = symbols[left];
        Symbol const& r = symbols[l.next];
        Rank const rank
            = getPairPriority(piece.substr(l.start, l.end - l.start), piece.substr(r.start, r.end - r.start));
        if (rank != MAX_RANK)
        {
            heap.push_back(Candidate{rank, l.start, left, l.end, r.end});
            std::push_heap(heap.begin(), heap.end(), later);
        }
    };
    for (int32_t i = 0; i + 1 < numChars; ++i)
    {
        pushPair(i);
    }
    while (!heap.empty())
    {
        std::pop_heap(heap.begin(), heap.end(), later);
        Candidate const c = heap.back();
        heap.pop_back();
        Symbol& left = symbols[c.left];
        // Stale entry: the left symbol was merged away or either side changed since it was pushed.
        if (left.start < 0 || left.next < 0 || left.end != c.leftEnd || symbols[left.next].end != c.rightEnd)
        {
            continue;
        }
        Symbol& right = symbols[left.next];
        left.end = right.end;
        left.next = right.next;
        if (left.next >= 0)
        {
            symbols[left.next].prev = c.left;
        }
        right.start = -1;
        pushPair(left.prev);
        pushPair(c.left);
    }

    // Collect final tokens
    for (int32_t i = 0; i >= 0; i = symbols[i].next)
    {
        std::string token(piece.begin() + symbols[i].start, piece.begin() + symbols[i].end);
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
