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

#include "requestVocabulary.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>

using namespace trt_edgellm::exampleUtils;
using nlohmann::json;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace
{

//! The message @p fold throws, or "" when it does not throw.
template <typename Fold>
std::string refusalOf(Fold const& fold)
{
    try
    {
        fold();
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return {};
}

} // namespace

// Given the runner table this library ships,
// When its request vocabulary is collected,
// Then it holds the sampling keys llm_inference and qwen3_tts_inference request files set, the style fields those
// files set at their top level, the MiniMax-H3 seed and video size, and the subtalker_*, speaker and
// prompt_speaker_codes spellings of qwen3_tts_inference files.
TEST(RequestVocabularyTest, TheShippedRunnersReadWhatTheExistingRequestFilesSet)
{
    auto const vocabulary = vocabularyOf(trt_edgellm::rt::multiStage::builtinRunners());

    EXPECT_THAT(vocabulary.samplingKeys,
        UnorderedElementsAre("temperature", "top_p", "top_k", "max_generate_length", "talker_temperature",
            "talker_top_k", "talker_top_p", "repetition_penalty", "cp_temperature", "cp_top_k", "cp_top_p",
            "max_audio_length", "seed"));
    EXPECT_THAT(vocabulary.styleFields,
        UnorderedElementsAre("talker_language", "language", "assistant_instruct", "instruct", "system_instruct",
            "ref_text", "height", "width", "num_frames"));
    EXPECT_THAT(vocabulary.aliases,
        UnorderedElementsAre(Pair("subtalker_temperature", "cp_temperature"), Pair("subtalker_top_k", "cp_top_k"),
            Pair("subtalker_top_p", "cp_top_p"), Pair("speaker", "voice"),
            Pair("prompt_speaker_codes", "speaker_codes")));
}

// Given a sampling key set by --sampling, at the document's top level and in the request's own sampling object,
// When the request's sampling is folded,
// Then the request's value wins, without it the document's, and without both the flag's.
TEST(RequestVocabularyTest, SamplingLayersApplyFlagThenDocumentThenRequest)
{
    RequestVocabulary vocabulary;
    vocabulary.samplingKeys = {"temperature"};
    std::string const flag = R"({"temperature": 0.1})";
    json const document{{"temperature", 0.2}};

    EXPECT_EQ(foldSampling(vocabulary, flag, document, json{{"sampling", {{"temperature", 0.3}}}}),
        (json{{"temperature", 0.3}}));
    EXPECT_EQ(foldSampling(vocabulary, flag, document, json::object()), (json{{"temperature", 0.2}}));
    EXPECT_EQ(foldSampling(vocabulary, flag, json::object(), json::object()), (json{{"temperature", 0.1}}));
}

// Given a sampling key that a stronger layer spells by an alias and a weaker layer by name, and a layer that spells
// it both ways,
// When the request's sampling is folded,
// Then the stronger layer's alias beats the weaker layer's key, and within one layer the key beats its alias.
TEST(RequestVocabularyTest, AnAliasIsReadAsItsKeyLayerByLayer)
{
    RequestVocabulary vocabulary;
    vocabulary.samplingKeys = {"cp_temperature"};
    vocabulary.aliases = {{"subtalker_temperature", "cp_temperature"}};

    EXPECT_EQ(
        foldSampling(vocabulary, R"({"cp_temperature": 0.1})", json{{"subtalker_temperature", 0.2}}, json::object()),
        (json{{"cp_temperature", 0.2}}));
    EXPECT_EQ(foldSampling(vocabulary, "", json::object(),
                  json{{"sampling", {{"cp_temperature", 0.5}, {"subtalker_temperature", 0.6}}}}),
        (json{{"cp_temperature", 0.5}}));
}

// Given a request file that sets a sampling key no runner reads, at its top level and in a request's sampling
// object, beside one a runner does read,
// When the request's sampling is folded,
// Then only the key a runner reads is kept.
TEST(RequestVocabularyTest, AKeyNoRunnerReadsIsDropped)
{
    RequestVocabulary vocabulary;
    vocabulary.samplingKeys = {"temperature"};

    EXPECT_EQ(
        foldSampling(vocabulary, "", json{{"seed", 7}}, json{{"sampling", {{"temperature", 0.3}, {"min_p", 0.05}}}}),
        (json{{"temperature", 0.3}}));
}

// Given a --sampling value and a request sampling value that are not JSON objects,
// When each request's sampling is folded,
// Then each fold throws naming where the value came from.
TEST(RequestVocabularyTest, ASamplingLayerThatIsNotAnObjectIsRefusedByName)
{
    RequestVocabulary vocabulary;
    vocabulary.samplingKeys = {"temperature"};

    EXPECT_THAT(
        refusalOf([&] { foldSampling(vocabulary, "[0.7]", json::object(), json::object()); }), HasSubstr("--sampling"));
    EXPECT_THAT(refusalOf([&] { foldSampling(vocabulary, "", json::object(), json{{"sampling", 0.7}}); }),
        HasSubstr("request sampling"));
}

// Given a field that a request sets by an alias while the document sets it by name, a request that sets it both
// ways, and a document that alone sets it, by an alias,
// When the field is looked up for each request,
// Then the request's value wins over the document's, its name over its alias, the document's alias is found, and a
// field nobody sets is absent.
TEST(RequestVocabularyTest, ARequestFieldIsReadFromTheRequestBeforeTheDocumentUnderEitherName)
{
    RequestVocabulary vocabulary;
    vocabulary.aliases = {{"speaker", "voice"}};

    auto const voiceOf = [&vocabulary](json const& document, json const& entry) -> std::string {
        auto const* value = findRequestField(vocabulary, "voice", document, entry);
        return value != nullptr ? value->get<std::string>() : "<none>";
    };
    EXPECT_EQ(voiceOf(json{{"voice", "document"}}, json{{"speaker", "request"}}), "request");
    EXPECT_EQ(voiceOf(json::object(), json{{"voice", "name"}, {"speaker", "alias"}}), "name");
    EXPECT_EQ(voiceOf(json{{"speaker", "document"}}, json::object()), "document");
    EXPECT_EQ(voiceOf(json::object(), json::object()), "<none>");
}

// Given a request whose own style object sets one style field, whose top level sets that field and another, and a
// document whose top level sets the second field and a key that is no style field,
// When the request's style is folded,
// Then the style object's own value wins, the request's top level beats the document's, the other key stays out, and
// a style that does not parse counts as empty.
TEST(RequestVocabularyTest, TheStyleObjectIsCompletedFromTopLevelStyleFields)
{
    RequestVocabulary vocabulary;
    vocabulary.styleFields = {"language", "instruct"};
    json const document{{"instruct", "document"}, {"voice", "ryan"}};

    EXPECT_EQ(foldStyle(vocabulary, document,
                  json{{"style", {{"language", "object"}}}, {"language", "request"}, {"instruct", "request"}}),
        (json{{"language", "object"}, {"instruct", "request"}}));
    EXPECT_EQ(foldStyle(vocabulary, document, json{{"style", "{not json"}, {"language", "request"}}),
        (json{{"language", "request"}, {"instruct", "document"}}));
}

// Given numeric style fields at the request's and the document's top level, as a video request sets its size,
// When the request's style is folded,
// Then they are kept as numbers, the request's beating the document's.
TEST(RequestVocabularyTest, NumericStyleFieldsAreKept)
{
    RequestVocabulary vocabulary;
    vocabulary.styleFields = {"height", "width"};

    EXPECT_EQ(foldStyle(vocabulary, json{{"height", 512}, {"width", 896}}, json{{"height", 768}}),
        (json{{"height", 768}, {"width", 896}}));
}
