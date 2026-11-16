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

#include "multimodal/qwen3_omni/talkerTokenIds.h"
#include "scratchDir.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using trt_edgellm::rt::qwen3_omni::resolveTalkerTokenIds;

namespace
{

//! What the fixture tokenizer leaves out or splits, so one case can break one lookup.
struct Vocabulary
{
    bool assistantWord{true};
    bool imEndToken{true};
    bool userSplitAsUsPlusUnknown{false};
    bool mediaPads{true};
    bool ttsEodToken{true};
    //! Only in tokenizer_config.json's added_tokens_decoder, as qwen3_omni_next checkpoints ship them.
    bool ttsTokensInTokenizerConfig{false};
};

//! A whole-string pre-tokenizer and no byte fallback, so adjacent pieces merge while the vocabulary holds their
//! concatenation and the first piece it lacks ends the string's encoding. Ids are far from Qwen's so a hardcoded
//! constant cannot pass.
std::unique_ptr<tokenizer::Tokenizer> loadTokenizer(Vocabulary const& vocabulary)
{
    auto const dir = makeScratchDir("talkerTokenIds");
    std::string vocab = R"("a": 0, "s": 1, "i": 2, "t": 3, "n": 4, "y": 5, "m": 6, "<": 7, "|": 8, "_": 9, "d": 10,
        ">": 11, "/": 12, "h": 13, "k": 14, "u": 15, "system": 42, "Ċ": 43,
        "ĊĊ": 44)";
    vocab += vocabulary.userSplitAsUsPlusUnknown ? R"(, "us": 45)" : R"(, "e": 17, "r": 18, "user": 41)";
    if (vocabulary.assistantWord)
    {
        vocab += R"(, "assistant": 40)";
    }
    std::string added = R"({"id": 50, "content": "<|im_start|>", "special": true},
        {"id": 52, "content": "<think>", "special": false},
        {"id": 53, "content": "</think>", "special": false})";
    if (vocabulary.imEndToken)
    {
        added += R"(, {"id": 51, "content": "<|im_end|>", "special": true})";
    }
    if (vocabulary.mediaPads)
    {
        added += R"(, {"id": 54, "content": "<|audio_pad|>", "special": true},
            {"id": 55, "content": "<|image_pad|>", "special": true},
            {"id": 56, "content": "<|video_pad|>", "special": true})";
    }
    // <tts_text_bos_single> sits next to <tts_text_bos> as in Qwen's vocabularies, so a wrong name still finds a token.
    std::vector<std::pair<int, std::string>> ttsTokens{
        {57, "<tts_pad>"}, {58, "<tts_text_bos>"}, {60, "<tts_text_bos_single>"}};
    if (vocabulary.ttsEodToken)
    {
        ttsTokens.emplace_back(59, "<tts_text_eod>");
    }
    std::string decoder;
    for (auto const& [id, content] : ttsTokens)
    {
        if (vocabulary.ttsTokensInTokenizerConfig)
        {
            decoder += (decoder.empty() ? "\"" : ", \"") + std::to_string(id) + R"(": {"content": ")" + content
                + R"(", "special": true})";
        }
        else
        {
            added += R"(, {"id": )" + std::to_string(id) + R"(, "content": ")" + content + R"(", "special": true})";
        }
    }

    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {)JSON"
                                          << vocab << R"JSON(}, "merges": []},
  "added_tokens": [)JSON" << added << R"JSON(],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
    if (!decoder.empty())
    {
        std::ofstream(dir / "tokenizer_config.json") << R"({"added_tokens_decoder": {)" << decoder << "}}";
    }
    auto tokenizer = std::make_unique<tokenizer::Tokenizer>();
    EXPECT_TRUE(tokenizer->loadFromHF(dir));
    std::filesystem::remove_all(dir);
    return tokenizer;
}

//! The message resolveTalkerTokenIds() throws, or an empty string when it resolves.
std::string resolveError(tokenizer::Tokenizer const& tokenizer, bool withChatTemplate, bool withMediaPads)
{
    try
    {
        resolveTalkerTokenIds(tokenizer, withChatTemplate, withMediaPads);
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return {};
}

} // namespace

TEST(TalkerTokenIdsTest, ResolvesEachMarkerToTheTokenizersOwnId)
{
    // Given a tokenizer that gives the ChatML markers, role words and newlines ids unlike Qwen's,
    // when the talker resolves its chat-template tokens, then each is the id that tokenizer gives the string.
    auto const tokenizer = loadTokenizer(Vocabulary{});

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_EQ(ids.imStart, 50);
    EXPECT_EQ(ids.imEnd, 51);
    EXPECT_EQ(ids.systemRole, 42);
    EXPECT_EQ(ids.userRole, 41);
    EXPECT_EQ(ids.assistantRole, 40);
    EXPECT_EQ(ids.thinkOpen, 52);
    EXPECT_EQ(ids.thinkClose, 53);
    EXPECT_EQ(ids.newline, 43);
    EXPECT_EQ(ids.doubleNewline, 44);
}

TEST(TalkerTokenIdsTest, RefusesARoleWordTheTokenizerSplits)
{
    // Given a tokenizer that spells "assistant" in several tokens, when the talker resolves its chat-template
    // tokens, then it refuses and names that word, since no single id would mark an assistant turn.
    Vocabulary vocabulary;
    vocabulary.assistantWord = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const error = resolveError(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_NE(error.find("'assistant'"), std::string::npos) << error;
}

TEST(TalkerTokenIdsTest, RefusesAMissingMarker)
{
    // Given a tokenizer without an <|im_end|> token, when the talker resolves its chat-template tokens, then it
    // refuses and names the marker rather than taking the first character's id.
    Vocabulary vocabulary;
    vocabulary.imEndToken = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const error = resolveError(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_NE(error.find("'<|im_end|>'"), std::string::npos) << error;
}

TEST(TalkerTokenIdsTest, RefusesALoneTokenSpellingPartOfTheWord)
{
    // Given a tokenizer that encodes "user" as one token spelling only "us", when the talker resolves its
    // chat-template tokens, then it refuses and names "user" instead of taking the partial token's id.
    Vocabulary vocabulary;
    vocabulary.userSplitAsUsPlusUnknown = true;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const error = resolveError(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_NE(error.find("'user'"), std::string::npos) << error;
}

TEST(TalkerTokenIdsTest, ResolvesTheMediaPlaceholdersWhenAsked)
{
    // Given a tokenizer that gives <|audio_pad|>, <|image_pad|> and <|video_pad|> ids unlike Qwen's, when the
    // segments layout resolves its tokens with the media placeholders, then each is the id that tokenizer gives it.
    auto const tokenizer = loadTokenizer(Vocabulary{});

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/true);

    EXPECT_EQ(ids.audioPad, 54);
    EXPECT_EQ(ids.imagePad, 55);
    EXPECT_EQ(ids.videoPad, 56);
}

TEST(TalkerTokenIdsTest, ResolvesWithoutTheMediaPlaceholdersUnlessAsked)
{
    // Given a tokenizer without media placeholders, when the chunked layout resolves its tokens without them, then
    // the chat-template tokens resolve and the placeholders stay unset, since that layout never looks for them.
    Vocabulary vocabulary;
    vocabulary.mediaPads = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_EQ(ids.imStart, 50);
    EXPECT_EQ(ids.audioPad, -1);
    EXPECT_EQ(ids.imagePad, -1);
    EXPECT_EQ(ids.videoPad, -1);
}

TEST(TalkerTokenIdsTest, RefusesAMissingMediaPlaceholderWhenAsked)
{
    // Given a tokenizer without media placeholders, when the segments layout resolves its tokens with them, then it
    // refuses and names the first placeholder rather than finding no media rows in any request.
    Vocabulary vocabulary;
    vocabulary.mediaPads = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const error = resolveError(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/true);

    EXPECT_NE(error.find("'<|audio_pad|>'"), std::string::npos) << error;
}

TEST(TalkerTokenIdsTest, ResolvesTheTtsTokensToTheTokenizersOwnIds)
{
    // Given a tokenizer that gives <tts_pad>, <tts_text_bos> and <tts_text_eod> ids unlike Qwen's, next to a
    // <tts_text_bos_single>, when the tts layout resolves its tokens, then tts_pad, tts_bos and tts_eos are the ids
    // that tokenizer gives those three strings.
    auto const tokenizer = loadTokenizer(Vocabulary{});

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/false, /*withMediaPads=*/false);

    EXPECT_EQ(ids.ttsPad, 57);
    EXPECT_EQ(ids.ttsBos, 58);
    EXPECT_EQ(ids.ttsEos, 59);
}

TEST(TalkerTokenIdsTest, ResolvesOnlyTheTtsTokensForTheTtsLayout)
{
    // Given a tokenizer with the tts tokens but without <|im_end|>, a one-token "assistant" or media placeholders, when
    // the tts layout resolves its tokens, then the tts tokens resolve and the chat-template tokens stay unset, since
    // that layout strips a fixed wrapper instead of finding one.
    Vocabulary vocabulary;
    vocabulary.assistantWord = false;
    vocabulary.imEndToken = false;
    vocabulary.mediaPads = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/false, /*withMediaPads=*/false);

    EXPECT_EQ(ids.ttsPad, 57);
    EXPECT_EQ(ids.ttsBos, 58);
    EXPECT_EQ(ids.ttsEos, 59);
    EXPECT_EQ(ids.imStart, -1);
    EXPECT_EQ(ids.imEnd, -1);
    EXPECT_EQ(ids.assistantRole, -1);
}

TEST(TalkerTokenIdsTest, RefusesAMissingTtsToken)
{
    // Given a tokenizer with <tts_pad> and <tts_text_bos> but no <tts_text_eod>, when the segments layout resolves its
    // tokens, then it refuses and names '<tts_text_eod>' rather than leaving tts_eos unset.
    Vocabulary vocabulary;
    vocabulary.ttsEodToken = false;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const error = resolveError(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/true);

    EXPECT_NE(error.find("'<tts_text_eod>'"), std::string::npos) << error;
}

TEST(TalkerTokenIdsTest, ResolvesTtsTokensListedOnlyInTokenizerConfig)
{
    // Given a tokenizer.json without the tts tokens and a tokenizer_config.json whose added_tokens_decoder lists them,
    // as qwen3_omni_next checkpoints ship them, when the chunked layout resolves its tokens, then each tts id is the
    // one tokenizer_config.json gives it.
    Vocabulary vocabulary;
    vocabulary.ttsTokensInTokenizerConfig = true;
    auto const tokenizer = loadTokenizer(vocabulary);

    auto const ids = resolveTalkerTokenIds(*tokenizer, /*withChatTemplate=*/true, /*withMediaPads=*/false);

    EXPECT_EQ(ids.ttsPad, 57);
    EXPECT_EQ(ids.ttsBos, 58);
    EXPECT_EQ(ids.ttsEos, 59);
}
