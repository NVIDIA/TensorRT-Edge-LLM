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

#include "runtime/multiStage/route.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt::multiStage;
using nlohmann::json;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

namespace
{

json makePort(
    std::string const& id, PortKind kind, bool terminal = false, Transport transport = Transport::kDeviceTensor)
{
    return json{{"id", id}, {"kind", toString(kind)}, {"transport", toString(transport)}, {"terminal", terminal}};
}

json makeInput(std::string const& name, std::string const& from, OnMissing onMissing)
{
    return json{{"name", name}, {"from", from}, {"on_missing", toString(onMissing)}};
}

json makeStage(std::string const& id, StageKind kind, int32_t lane, json inputs, json emits)
{
    return json{{"id", id}, {"runner", "unittest." + id}, {"kind", toString(kind)}, {"stream_lane", lane},
        {"inputs", std::move(inputs)}, {"emits", std::move(emits)}};
}

/*!
 * @brief Five stages covering both degrees of freedom: optional inputs (audio,
 *        images) and optional outputs (text only, or text plus audio).
 *
 *   request:audio  -> audio_encoder --\
 *   request:images -> visual ----------> thinker -> talker ==> code2wav (lane 1)
 *   request:messages -----------------/
 *
 * "==>" is the only cross-lane edge, so it is the only chunk_stream port.
 */
json fiveStageTopology()
{
    return json{{"schema_version", MultiStageConfig::kSchemaVersion},
        {"stages",
            json::array({
                makeStage("audio_encoder", StageKind::kEngine, 0,
                    json::array({makeInput("wav", "request:audio", OnMissing::kPrune)}),
                    json::array({makePort("features", PortKind::kEmbeds)})),
                makeStage("visual", StageKind::kEngine, 0,
                    json::array({makeInput("pixels", "request:images", OnMissing::kPrune)}),
                    json::array({makePort("features", PortKind::kEmbeds)})),
                makeStage("thinker", StageKind::kAutoregressive, 0,
                    json::array({makeInput("prompt", "request:messages", OnMissing::kFail),
                        makeInput("audio", "audio_encoder:features", OnMissing::kSkip),
                        makeInput("visual", "visual:features", OnMissing::kSkip)}),
                    json::array({makePort("text", PortKind::kText, true), makePort("hidden", PortKind::kHidden)})),
                makeStage("talker", StageKind::kAutoregressive, 0,
                    json::array({makeInput("hidden", "thinker:hidden", OnMissing::kFail),
                        makeInput("voice", "request:audio_params", OnMissing::kFail)}),
                    json::array({makePort("codes", PortKind::kCodec, false, Transport::kChunkStream)})),
                makeStage("code2wav", StageKind::kIterative, 1,
                    json::array({makeInput("codes", "talker:codes", OnMissing::kFail)}),
                    json::array({makePort("pcm", PortKind::kAudio, true)})),
            })},
        {"terminals", json::array({"thinker:text", "code2wav:pcm"})}};
}

RequestCapabilities capabilities(std::vector<std::string> keys)
{
    RequestCapabilities available;
    available.suppliedRequestKeys = std::move(keys);
    return available;
}

//! A Route stores indices, so an id has to be resolved against the config.
bool isActive(MultiStageConfig const& config, Route const& route, std::string const& stageId)
{
    int32_t const index = config.stageIndex(stageId);
    EXPECT_GE(index, 0) << "fixture has no stage " << stageId;
    return route.isActive(index);
}

std::vector<std::string> orderedIds(MultiStageConfig const& config, Route const& route)
{
    std::vector<std::string> ids;
    ids.reserve(route.order().size());
    for (int32_t const index : route.order())
    {
        ids.push_back(config.stages[static_cast<size_t>(index)].id);
    }
    return ids;
}

std::ptrdiff_t indexOf(std::vector<std::string> const& ids, std::string const& id)
{
    return std::find(ids.begin(), ids.end(), id) - ids.begin();
}

RouteEdge const* edgeInto(
    MultiStageConfig const& config, Route const& route, std::string const& stageId, std::string const& inputName)
{
    int32_t const consumer = config.stageIndex(stageId);
    for (RouteEdge const& edge : route.edges())
    {
        if (edge.consumer == consumer && edge.inputName == inputName)
        {
            return &edge;
        }
    }
    return nullptr;
}

RequestCapabilities everything()
{
    return capabilities({"request:messages", "request:audio", "request:images", "request:audio_params"});
}

} // namespace

// Given a request that supplies "request:messages" and "request:audio",
// When it is asked whether it supplies a key,
// Then only an exact fully-qualified key matches: an absent key and the bare field name "audio" do not.
TEST(RouteTest, RequestCapabilitiesAnswerOnFullyQualifiedKeys)
{
    RequestCapabilities const available = capabilities({"request:messages", "request:audio"});
    EXPECT_TRUE(available.supplies("request:audio"));
    EXPECT_FALSE(available.supplies("request:images"));
    // The bare field name is not the key; `from` is matched verbatim.
    EXPECT_FALSE(available.supplies("audio"));
}

template <typename T, typename = void>
struct QueryableByStageId : std::false_type
{
};

template <typename T>
struct QueryableByStageId<T,
    std::void_t<decltype(std::declval<T const&>().isActive(std::declval<std::string const&>()))>> : std::true_type
{
};

// Given a resolved route,
// When it is queried by an index outside the config,
// Then the index reads as inactive, and a query by stage id does not compile.
TEST(RouteTest, IsActiveReadsAnOutOfRangeIndexAsInactive)
{
    static_assert(!QueryableByStageId<Route>::value, "a Route holds indices, not stage ids");
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    ASSERT_TRUE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), route, error))
        << error;

    EXPECT_FALSE(route.isActive(-1));
    EXPECT_FALSE(route.isActive(static_cast<int32_t>(config.stages.size())));
}

// Given a request with messages and audio but no images,
// When only the text terminal is wanted,
// Then exactly audio_encoder and thinker run, in that order; visual and the audio-output branch do not.
TEST(RouteTest, AudioInTextOutDropsTheAudioOutputBranch)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    ASSERT_TRUE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), route, error))
        << error;

    EXPECT_THAT(orderedIds(config, route), ElementsAre("audio_encoder", "thinker"));
    EXPECT_TRUE(isActive(config, route, "audio_encoder"));
    EXPECT_TRUE(isActive(config, route, "thinker"));
    EXPECT_FALSE(isActive(config, route, "visual"));
    EXPECT_FALSE(isActive(config, route, "talker"));
    EXPECT_FALSE(isActive(config, route, "code2wav"));
}

// Given a request that supplies every request key,
// When both the text and the audio terminal are wanted,
// Then all five stages run, each after its producers, and no input is skipped.
TEST(RouteTest, AudioOutKeepsTheWholeTopology)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    ASSERT_TRUE(Route::resolve(config, {"thinker:text", "code2wav:pcm"}, everything(), route, error)) << error;

    std::vector<std::string> const ids = orderedIds(config, route);
    EXPECT_THAT(ids, UnorderedElementsAre("audio_encoder", "visual", "thinker", "talker", "code2wav"));

    EXPECT_LT(indexOf(ids, "audio_encoder"), indexOf(ids, "thinker"));
    EXPECT_LT(indexOf(ids, "visual"), indexOf(ids, "thinker"));
    EXPECT_LT(indexOf(ids, "thinker"), indexOf(ids, "talker"));
    EXPECT_LT(indexOf(ids, "talker"), indexOf(ids, "code2wav"));
    EXPECT_TRUE(route.skippedInputs().empty());
}

// Given a request without images, where thinker.visual is on_missing=skip,
// When the text terminal is resolved,
// Then visual is pruned but thinker still runs; thinker.visual is reported skipped and has no edge,
// while thinker.audio, whose source survived, keeps its edge and is not reported skipped.
TEST(RouteTest, MissingImagesPrunesVisualYetThinkerStaysAndRecordsTheSkip)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    ASSERT_TRUE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), route, error))
        << error;

    EXPECT_FALSE(isActive(config, route, "visual"));
    EXPECT_TRUE(isActive(config, route, "thinker"));

    int32_t const thinker = config.stageIndex("thinker");
    EXPECT_THAT(route.skippedInputs(), Contains(std::make_pair(thinker, std::string("visual"))));
    EXPECT_EQ(edgeInto(config, route, "thinker", "visual"), nullptr);

    // The audio source survived, so that binding is present rather than skipped.
    EXPECT_THAT(route.skippedInputs(), Not(Contains(std::make_pair(thinker, std::string("audio")))));
    EXPECT_NE(edgeInto(config, route, "thinker", "audio"), nullptr);
}

// Given the same image-less request, with thinker.visual switched to on_missing=prune,
// When the text terminal is resolved,
// Then resolution fails, and the error names the wanted sink plus the upstream stage and request key
// whose absence caused it (visual, request:images).
TEST(RouteTest, TurningThatSkipIntoAPruneRejectsTheSameRequest)
{
    // Why OnMissing cannot be a bool: with kPrune on thinker.visual, a request
    // that carries no images loses the stage that produces the wanted terminal.
    json topology = fiveStageTopology();
    for (json& stage : topology.at("stages"))
    {
        if (stage.at("id") == "thinker")
        {
            for (json& input : stage.at("inputs"))
            {
                if (input.at("name") == "visual")
                {
                    input["on_missing"] = toString(OnMissing::kPrune);
                }
            }
        }
    }
    MultiStageConfig const config = MultiStageConfig::parse(topology);

    Route route;
    std::string error;
    EXPECT_FALSE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), route, error));
    ASSERT_FALSE(error.empty());
    EXPECT_THAT(error, HasSubstr("thinker:text"));
    // The message names the stage that actually lost its source, not the sink's stage.
    EXPECT_THAT(error, HasSubstr("visual"));
    EXPECT_THAT(error, HasSubstr("request:images"));
}

// Given a request that lacks the source of talker.voice, an on_missing=fail input,
// When it is resolved for the text terminal only, and then for the audio terminal as well,
// Then the text-only route resolves with talker inactive, and the audio-out route fails naming talker and voice.
TEST(RouteTest, FailInputIsJudgedInsideTheActiveSetOnly)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());
    RequestCapabilities const noVoice = capabilities({"request:messages", "request:audio"});

    // talker.voice is kFail, but talker is outside the active set of a text-only
    // request, so the missing voice must not reject it.
    Route textOnly;
    std::string error;
    ASSERT_TRUE(Route::resolve(config, {"thinker:text"}, noVoice, textOnly, error)) << error;
    EXPECT_FALSE(isActive(config, textOnly, "talker"));

    // Ask for audio out with the very same request and that input now decides it.
    Route withAudio;
    EXPECT_FALSE(Route::resolve(config, {"thinker:text", "code2wav:pcm"}, noVoice, withAudio, error));
    ASSERT_FALSE(error.empty());
    EXPECT_THAT(error, HasSubstr("talker"));
    EXPECT_THAT(error, HasSubstr("voice"));
}

// Given a valid topology and request,
// When no terminal is wanted,
// Then resolution fails with an error and the route holds no stages.
TEST(RouteTest, EmptyWantedIsRejected)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    EXPECT_FALSE(Route::resolve(config, {}, capabilities({"request:messages"}), route, error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(route.order().empty());
}

// Given a valid topology,
// When the wanted terminal names an unknown port on a known stage, or an unknown stage,
// Then resolution fails and the error names the unknown port id or stage.
TEST(RouteTest, WantedPortThatDoesNotExistIsRejected)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    EXPECT_FALSE(Route::resolve(config, {"thinker:logits"}, capabilities({"request:messages"}), route, error));
    ASSERT_FALSE(error.empty());
    EXPECT_THAT(error, HasSubstr("thinker:logits"));

    EXPECT_FALSE(Route::resolve(config, {"ghost:text"}, capabilities({"request:messages"}), route, error));
    ASSERT_FALSE(error.empty());
    EXPECT_THAT(error, HasSubstr("ghost"));
}

// Given a port that exists but is not flagged terminal (thinker:hidden),
// When it is wanted,
// Then resolution fails and the error names the port and says it is not terminal.
TEST(RouteTest, WantedPortThatIsNotTerminalIsRejected)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    EXPECT_FALSE(Route::resolve(config, {"thinker:hidden"}, capabilities({"request:messages"}), route, error));
    ASSERT_FALSE(error.empty());
    EXPECT_THAT(error, HasSubstr("thinker:hidden"));
    EXPECT_THAT(error, HasSubstr("terminal"));
}

// Given a route with audio in and audio out,
// When its edges are inspected,
// Then each stage-sourced edge carries its producer, producer port and that port's transport, only the
// edge between lanes (talker -> code2wav) is cross-lane, and a request-sourced edge has producer -1 and
// names the request field.
TEST(RouteTest, EdgeCarriesItsProducerPortAndTransport)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());

    Route route;
    std::string error;
    ASSERT_TRUE(Route::resolve(config, {"thinker:text", "code2wav:pcm"}, everything(), route, error)) << error;

    RouteEdge const* hidden = edgeInto(config, route, "talker", "hidden");
    ASSERT_NE(hidden, nullptr);
    EXPECT_EQ(hidden->producer, config.stageIndex("thinker"));
    EXPECT_EQ(hidden->port, "hidden");
    EXPECT_FALSE(hidden->crossLane);
    EXPECT_EQ(hidden->transport, Transport::kDeviceTensor);

    RouteEdge const* codes = edgeInto(config, route, "code2wav", "codes");
    ASSERT_NE(codes, nullptr);
    EXPECT_TRUE(codes->crossLane);
    EXPECT_EQ(codes->transport, Transport::kChunkStream);

    RouteEdge const* audio = edgeInto(config, route, "thinker", "audio");
    ASSERT_NE(audio, nullptr);
    EXPECT_EQ(audio->producer, config.stageIndex("audio_encoder"));
    EXPECT_FALSE(audio->crossLane);

    // A request-sourced edge names the field rather than a producer, which is how the
    // executor knows to bind it from the payload instead of from a stage.
    RouteEdge const* prompt = edgeInto(config, route, "thinker", "prompt");
    ASSERT_NE(prompt, nullptr);
    EXPECT_EQ(prompt->producer, -1);
    EXPECT_EQ(prompt->port, "messages");

    for (RouteEdge const& edge : route.edges())
    {
        StageConfig const& consumer = config.stages[static_cast<size_t>(edge.consumer)];
        InputSpec const* input = consumer.findInput(edge.inputName);
        ASSERT_NE(input, nullptr) << consumer.id << "." << edge.inputName;
        if (edge.producer < 0)
        {
            EXPECT_EQ("request:" + edge.port, input->from) << consumer.id << "." << edge.inputName;
            EXPECT_FALSE(edge.crossLane) << consumer.id << "." << edge.inputName;
            continue;
        }
        StageConfig const& producer = config.stages[static_cast<size_t>(edge.producer)];
        EXPECT_EQ(producer.id + ":" + edge.port, input->from) << consumer.id << "." << edge.inputName;
        PortSpec const* port = producer.findPort(edge.port);
        ASSERT_NE(port, nullptr) << producer.id << ":" << edge.port;
        EXPECT_EQ(edge.transport, port->transport) << producer.id << ":" << edge.port;
        EXPECT_EQ(edge.crossLane, producer.id == "talker" && consumer.id == "code2wav")
            << producer.id << " -> " << consumer.id;
    }
}

// Given two requests whose routes run different stage sets (text only, and text plus audio out),
// When their fingerprints are compared,
// Then they differ, so the two requests may not share a batch.
TEST(RouteTest, FingerprintSeparatesDifferentActiveSets)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());
    std::string error;

    Route textOnly;
    ASSERT_TRUE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), textOnly, error))
        << error;

    Route withAudioOut;
    ASSERT_TRUE(Route::resolve(config, {"thinker:text", "code2wav:pcm"},
        capabilities({"request:messages", "request:audio", "request:audio_params"}), withAudioOut, error))
        << error;

    EXPECT_NE(textOnly.fingerprint(), withAudioOut.fingerprint());
}

// Given two text-only requests, one of which also supplies a key consumed only by an inactive stage,
// When both are resolved,
// Then they run the same stages in the same order and their fingerprints are equal.
TEST(RouteTest, FingerprintMatchesWhenTheActiveSetsAgree)
{
    MultiStageConfig const config = MultiStageConfig::parse(fiveStageTopology());
    std::string error;

    Route lean;
    ASSERT_TRUE(
        Route::resolve(config, {"thinker:text"}, capabilities({"request:messages", "request:audio"}), lean, error))
        << error;

    // The extra key only feeds a stage this request does not run, so the two
    // requests stay batchable.
    Route withUnusedKey;
    ASSERT_TRUE(Route::resolve(config, {"thinker:text"},
        capabilities({"request:messages", "request:audio", "request:audio_params"}), withUnusedKey, error))
        << error;

    EXPECT_EQ(orderedIds(config, lean), orderedIds(config, withUnusedKey));
    EXPECT_EQ(lean.fingerprint(), withUnusedKey.fingerprint());
}
