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

//! The `skipSpecialTokens` option drops added tokens flagged `special`, never plain added
//! tokens, and does so whether or not the id is also in the base vocab. The two
//! layouts below are the real ones: Nemotron duplicates every added token into
//! the base vocab, Qwen keeps them out of it.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "tokenizer/tokenizer.h"

using namespace trt_edgellm;

namespace
{

enum class AddedTokenSource
{
    kTokenizerJson,
    kTokenizerConfig,
};

std::filesystem::path writeTokenizer(
    std::string const& name, bool alsoInVocab, AddedTokenSource source = AddedTokenSource::kTokenizerJson)
{
    auto const dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::string vocab = R"("a": 0, "b": 1)";
    if (alsoInVocab)
    {
        vocab += R"(, "<|im_end|>": 11, "<think>": 12)";
    }

    std::ofstream tokenizerFile(dir / "tokenizer.json");
    tokenizerFile << R"JSON({
  "model": {"type": "BPE", "vocab": {)JSON"
                  << vocab << R"JSON(}, "merges": []},
)JSON";
    if (source == AddedTokenSource::kTokenizerJson)
    {
        tokenizerFile << R"JSON(  "added_tokens": [
    {"id": 11, "content": "<|im_end|>", "special": true},
    {"id": 12, "content": "<think>", "special": false}
  ],
)JSON";
    }
    tokenizerFile << R"JSON(  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";

    std::ofstream configFile(dir / "tokenizer_config.json");
    configFile << R"JSON({"eos_token": {"content": "<|im_end|>"})JSON";
    if (source == AddedTokenSource::kTokenizerConfig)
    {
        configFile << R"JSON(, "added_tokens_decoder": {
  "11": {"content": "<|im_end|>", "special": true},
  "12": {"content": "<think>", "special": false}
})JSON";
    }
    configFile << "}";
    return dir;
}

std::filesystem::path writeConfiguredSentinelTokenizer()
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tok_configured_sentinels";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {
    "a": 0, "b": 1, "<bos>": 11, "<eos>": 12,
    "<pad>": 13, "<unk>": 14, "<turn>": 15
  }, "merges": []},
  "added_tokens": [
    {"id": 11, "content": "<bos>"},
    {"id": 12, "content": "<eos>"},
    {"id": 13, "content": "<pad>"},
    {"id": 14, "content": "<unk>"},
    {"id": 15, "content": "<turn>"}
  ],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << R"JSON({
  "bos_token": {"content": "<bos>"},
  "eos_token": {"content": "<eos>"},
  "pad_token": {"content": "<pad>"},
  "unk_token": {"content": "<unk>"}
})JSON";
    return dir;
}

void checkSkipSemantics(bool alsoInVocab, AddedTokenSource source = AddedTokenSource::kTokenizerJson)
{
    std::string const name = source == AddedTokenSource::kTokenizerConfig
        ? "edgellm_tok_special_config_decoder"
        : (alsoInVocab ? "edgellm_tok_special_overlap" : "edgellm_tok_special_disjoint");
    auto const dir = writeTokenizer(name, alsoInVocab, source);
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));

    std::vector<tokenizer::Rank> const ids{0, 12, 1, 11};

    // `special: true` is dropped, `special: false` survives.
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/true), "a<think>b");
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/false), "a<think>b<|im_end|>");

    rt::SlotStreamState streamState;
    EXPECT_EQ(tokenizer::emitDelta(streamState, tok, ids, /*skipSpecial=*/true), "a<think>b");
    EXPECT_EQ(streamState.sentTokenCount, ids.size());

    EXPECT_EQ(tok.idToPiece(11, /*skipSpecialTokens=*/true), "");
    EXPECT_EQ(tok.idToPiece(11, /*skipSpecialTokens=*/false), "<|im_end|>");
    EXPECT_EQ(tok.idToPiece(12, /*skipSpecialTokens=*/true), "<think>");

    auto const encoded = tok.encode("<think>", /*addBos=*/false);
    EXPECT_EQ(encoded, (std::vector<tokenizer::Rank>{12}));

    std::filesystem::remove_all(dir);
}

} // namespace

TEST(TokenizerSpecialTokenTest, SkipsOnlyFlaggedSpecialsWhenAddedIdsAreAlsoInVocab)
{
    checkSkipSemantics(/*alsoInVocab=*/true);
}

TEST(TokenizerSpecialTokenTest, SkipsOnlyFlaggedSpecialsWhenAddedIdsAreVocabDisjoint)
{
    checkSkipSemantics(/*alsoInVocab=*/false);
}

TEST(TokenizerSpecialTokenTest, ReadsSpecialFlagsFromAddedTokensDecoder)
{
    checkSkipSemantics(/*alsoInVocab=*/false, AddedTokenSource::kTokenizerConfig);
}

TEST(TokenizerSpecialTokenTest, ConfiguredSentinelsAndAdditionalEosAreSkippable)
{
    auto const dir = writeConfiguredSentinelTokenizer();
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));

    std::vector<tokenizer::Rank> const ids{0, 11, 12, 13, 14, 15, 1};
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/true), "a<turn>b");
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/false), "a<bos><eos><pad><unk><turn>b");

    tok.setAdditionalEosIds({12, 15, 15, -1});
    EXPECT_EQ(tok.getEosIds(), (std::vector<tokenizer::Rank>{12, 15}));
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/true), "ab");

    tok.setAdditionalEosIds({});
    EXPECT_EQ(tok.getEosIds(), (std::vector<tokenizer::Rank>{12}));
    EXPECT_EQ(tok.decode(ids, /*skipSpecialTokens=*/true), "a<turn>b");

    std::filesystem::remove_all(dir);
}

TEST(TokenizerSpecialTokenTest, UnknownIdPreservesExistingSkipBehavior)
{
    auto const dir = writeTokenizer("edgellm_tok_special_unknown", /*alsoInVocab=*/false);
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));

    EXPECT_EQ(tok.decode({0, 4242, 1}, /*skipSpecialTokens=*/true), "ab");
    EXPECT_EQ(tok.decode({0, 4242, 1}, /*skipSpecialTokens=*/false), "");

    std::filesystem::remove_all(dir);
}
