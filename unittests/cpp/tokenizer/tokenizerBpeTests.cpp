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

//! Byte-pair merge order of `TokenEncoder`. The merge always applies the lowest-priority pair first
//! and, among equal priorities, the leftmost one. Both priority sources are covered: explicit merge
//! priorities (SentencePiece-style, `byte_fallback`) and vocabulary rank (tiktoken-style). The
//! encoder is compared with a plain quadratic reference on random inputs.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.h"

using namespace trt_edgellm;

namespace
{

//! Vocabulary and merge list of a small random BPE model over the letters a-d.
struct BpeModel
{
    std::map<std::string, tokenizer::Rank> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
};

//! Merge-priority model: every merge result is in the vocabulary; the merge index is its priority.
BpeModel makeMergePriorityModel(std::mt19937& rng)
{
    BpeModel model;
    std::vector<std::string> tokens{"a", "b", "c", "d"};
    for (auto const& token : tokens)
    {
        model.vocab.emplace(token, static_cast<tokenizer::Rank>(model.vocab.size()));
    }
    for (int attempt = 0; attempt < 200 && model.merges.size() < 60; ++attempt)
    {
        std::string const& left = tokens[rng() % tokens.size()];
        std::string const& right = tokens[rng() % tokens.size()];
        auto const pair = std::make_pair(left, right);
        if (left.size() + right.size() > 8
            || std::find(model.merges.begin(), model.merges.end(), pair) != model.merges.end())
        {
            continue;
        }
        model.merges.push_back(pair);
        std::string const merged = left + right;
        if (model.vocab.emplace(merged, static_cast<tokenizer::Rank>(model.vocab.size())).second)
        {
            tokens.push_back(merged);
        }
    }
    return model;
}

//! Vocabulary-rank model: ids of the multi-letter tokens are shuffled, so rank is unrelated to length.
BpeModel makeVocabRankModel(std::mt19937& rng)
{
    BpeModel merged = makeMergePriorityModel(rng);
    std::vector<std::string> multi;
    BpeModel model;
    for (auto const& token : {"a", "b", "c", "d"})
    {
        model.vocab.emplace(token, static_cast<tokenizer::Rank>(model.vocab.size()));
    }
    for (auto const& entry : merged.vocab)
    {
        if (entry.first.size() > 1)
        {
            multi.push_back(entry.first);
        }
    }
    std::shuffle(multi.begin(), multi.end(), rng);
    for (auto const& token : multi)
    {
        model.vocab.emplace(token, static_cast<tokenizer::Rank>(model.vocab.size()));
    }
    return model;
}

//! Reference: repeatedly merge the lowest-priority adjacent pair, leftmost on ties.
std::vector<tokenizer::Rank> referenceEncode(BpeModel const& model, bool useMergePriorities, std::string const& text)
{
    std::map<std::pair<std::string, std::string>, size_t> priorities;
    for (size_t i = 0; i < model.merges.size(); ++i)
    {
        priorities.emplace(model.merges[i], i);
    }
    std::vector<std::string> parts;
    for (char c : text)
    {
        parts.emplace_back(1, c);
    }
    while (parts.size() > 1)
    {
        size_t best = std::numeric_limits<size_t>::max();
        size_t bestIdx = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i)
        {
            size_t priority = std::numeric_limits<size_t>::max();
            if (useMergePriorities)
            {
                auto it = priorities.find({parts[i], parts[i + 1]});
                priority = it != priorities.end() ? it->second : priority;
            }
            else
            {
                auto it = model.vocab.find(parts[i] + parts[i + 1]);
                priority = it != model.vocab.end() ? static_cast<size_t>(it->second) : priority;
            }
            if (priority < best)
            {
                best = priority;
                bestIdx = i;
            }
        }
        if (best == std::numeric_limits<size_t>::max())
        {
            break;
        }
        parts[bestIdx] += parts[bestIdx + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(bestIdx) + 1);
    }
    std::vector<tokenizer::Rank> ids;
    for (auto const& part : parts)
    {
        ids.push_back(model.vocab.at(part));
    }
    return ids;
}

//! Writes `tokenizer.json` for the model and loads it; `useMergePriorities` selects the SentencePiece-style layout.
void loadModel(tokenizer::Tokenizer& tok, std::string const& name, BpeModel const& model, bool useMergePriorities)
{
    auto const dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    nlohmann::json merges = nlohmann::json::array();
    if (useMergePriorities)
    {
        for (auto const& merge : model.merges)
        {
            merges.push_back(merge.first + " " + merge.second);
        }
    }
    nlohmann::json const config = {
        {"model", {{"type", "BPE"}, {"byte_fallback", useMergePriorities}, {"vocab", model.vocab}, {"merges", merges}}},
        {"added_tokens", nlohmann::json::array()}};
    std::ofstream(dir / "tokenizer.json") << config.dump();
    std::ofstream(dir / "tokenizer_config.json") << "{}";
    ASSERT_TRUE(tok.loadFromHF(dir));
}

void expectMatchesReference(bool useMergePriorities, std::string const& name)
{
    std::mt19937 rng(12345);
    for (int modelIdx = 0; modelIdx < 5; ++modelIdx)
    {
        BpeModel const model = useMergePriorities ? makeMergePriorityModel(rng) : makeVocabRankModel(rng);
        tokenizer::Tokenizer tok;
        loadModel(tok, name, model, useMergePriorities);
        for (int sample = 0; sample < 100; ++sample)
        {
            size_t const length = 1 + rng() % (sample < 90 ? 60 : 2000);
            std::string text;
            for (size_t i = 0; i < length; ++i)
            {
                text += static_cast<char>('a' + (rng() % 8 < 6 ? rng() % 2 : rng() % 4));
            }
            ASSERT_EQ(tok.encode(text, false), referenceEncode(model, useMergePriorities, text))
                << "model " << modelIdx << " text " << text.substr(0, 80);
        }
    }
}

} // namespace

//! Equal-priority merges resolve leftmost first, a lower priority wins over a leftmost pair, and a
//! long run merges to the same result as the pairwise reference.
TEST(TokenizerBpe, MergePriorityAndLeftmostTieBreakOnLongPieces)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_bpe_tiebreak";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "byte_fallback": true,
            "vocab": {"a": 0, "b": 1, "aa": 2, "aaaa": 3, "ab": 4, "<0x63>": 5},
            "merges": ["a a", "aa aa", "a b"]},
  "added_tokens": []
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << "{}";
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));
    // "a a" (priority 0) beats "a b" (2); five a's: leftmost pairs first -> aaaa + a.
    EXPECT_EQ(tok.encode("aaaaa", false), (std::vector<tokenizer::Rank>{3, 0}));
    EXPECT_EQ(tok.encode("aaab", false), (std::vector<tokenizer::Rank>{2, 4}));
    // Unknown byte falls back to <0xNN>.
    EXPECT_EQ(tok.encode("ac", false), (std::vector<tokenizer::Rank>{0, 5}));
    // Long run: 4001 a's -> 1000 x aaaa + a.
    std::string const run(4001, 'a');
    std::vector<tokenizer::Rank> expected(1000, 3);
    expected.push_back(0);
    EXPECT_EQ(tok.encode(run, false), expected);
}

TEST(TokenizerBpe, MergePrioritiesMatchReferenceOnRandomInput)
{
    expectMatchesReference(true, "edgellm_bpe_random_priority");
}

TEST(TokenizerBpe, VocabularyRankMatchesReferenceOnRandomInput)
{
    expectMatchesReference(false, "edgellm_bpe_random_rank");
}
