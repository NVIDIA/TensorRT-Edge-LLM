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

#include "runtime/multiStage/stageConfig.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

using namespace trt_edgellm::rt::multiStage;
using nlohmann::json;
using ::testing::HasSubstr;

namespace
{

// Fixtures spell enums through toString(), so no test here pins the JSON spelling itself.
json makePort(std::string const& id, bool terminal = false, PortKind kind = PortKind::kEmbeds,
    Transport transport = Transport::kDeviceTensor)
{
    return json{{"id", id}, {"kind", toString(kind)}, {"transport", toString(transport)}, {"terminal", terminal}};
}

json makeInput(std::string const& name, std::string const& from, OnMissing onMissing = OnMissing::kFail)
{
    return json{{"name", name}, {"from", from}, {"on_missing", toString(onMissing)}};
}

json makeStage(std::string const& id, StageKind kind, json inputs, json emits)
{
    return json{{"id", id}, {"runner", "unittest." + id}, {"kind", toString(kind)}, {"inputs", std::move(inputs)},
        {"emits", std::move(emits)}};
}

//! Two stages, one edge, one terminal. The optional stage keys and the port binding are
//! left out so their defaults are exercised too.
json minimalGraph()
{
    return json{{"schema_version", MultiStageConfig::kSchemaVersion}, {"label", "minimal"},
        {"stages",
            json::array({
                makeStage("encoder", StageKind::kEngine, json::array({makeInput("wav", "request:audio")}),
                    json::array({makePort("features")})),
                makeStage("decoder", StageKind::kAutoregressive, json::array({makeInput("audio", "encoder:features")}),
                    json::array({makePort("text", true, PortKind::kText)})),
            })},
        {"terminals", json::array({"decoder:text"})}};
}

json& stageJson(json& root, std::string const& id)
{
    for (json& stage : root.at("stages"))
    {
        if (stage.at("id") == id)
        {
            return stage;
        }
    }
    throw std::runtime_error("fixture has no stage " + id);
}

json& portJson(json& root, std::string const& stageId, std::string const& portId)
{
    for (json& port : stageJson(root, stageId).at("emits"))
    {
        if (port.at("id") == portId)
        {
            return port;
        }
    }
    throw std::runtime_error("fixture has no port " + stageId + ":" + portId);
}

json& inputJson(json& root, std::string const& stageId, std::string const& name)
{
    for (json& input : stageJson(root, stageId).at("inputs"))
    {
        if (input.at("name") == name)
        {
            return input;
        }
    }
    throw std::runtime_error("fixture has no input " + stageId + "." + name);
}

//! The message the topology is rejected with, or an empty string when it is accepted.
//! parse() already validates; the second call is what carries a config root.
std::string rejectionOf(json const& root, std::string const& configRoot = {})
{
    try
    {
        MultiStageConfig const config = MultiStageConfig::parse(root);
        validate(config, configRoot);
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return {};
}

} // namespace

// Given a two-stage graph that leaves every optional key out,
// When it is parsed and validated,
// Then it is accepted, and every field reads back as written or as its documented default.
TEST(StageConfigTest, MinimalGraphParsesAndValidates)
{
    MultiStageConfig const config = MultiStageConfig::parse(minimalGraph());
    EXPECT_NO_THROW(validate(config));

    EXPECT_EQ(config.schemaVersion, MultiStageConfig::kSchemaVersion);
    EXPECT_EQ(config.label, "minimal");
    ASSERT_EQ(config.stages.size(), 2U);
    EXPECT_EQ(config.stageIndex("encoder"), 0);
    EXPECT_EQ(config.stageIndex("nobody"), -1);

    StageConfig const* decoder = config.findStage("decoder");
    ASSERT_NE(decoder, nullptr);
    EXPECT_EQ(decoder->runner, "unittest.decoder");
    EXPECT_EQ(decoder->kind, StageKind::kAutoregressive);
    EXPECT_EQ(decoder->maxBatch, 1);
    EXPECT_EQ(decoder->maxSeq, 0);
    EXPECT_EQ(decoder->streamLane, 0);
    EXPECT_TRUE(decoder->artifact.empty());

    InputSpec const* audio = decoder->findInput("audio");
    ASSERT_NE(audio, nullptr);
    EXPECT_EQ(audio->from, "encoder:features");
    EXPECT_EQ(audio->onMissing, OnMissing::kFail);
    EXPECT_EQ(decoder->findInput("nobody"), nullptr);

    PortSpec const* text = decoder->findPort("text");
    ASSERT_NE(text, nullptr);
    EXPECT_EQ(text->kind, PortKind::kText);
    EXPECT_EQ(text->transport, Transport::kDeviceTensor);
    EXPECT_TRUE(text->terminal);
    EXPECT_TRUE(text->binding.empty());
    EXPECT_EQ(decoder->findPort("nobody"), nullptr);

    ASSERT_EQ(config.terminals.size(), 1U);
    EXPECT_EQ(config.terminals[0], "decoder:text");
}

// Given every value of every config enum,
// When it is spelled with toString() and parsed back,
// Then the same value comes out.
TEST(StageConfigTest, EveryEnumRoundTripsThroughItsJsonSpelling)
{
    for (StageKind const v : {StageKind::kHost, StageKind::kEngine, StageKind::kAutoregressive, StageKind::kIterative})
    {
        EXPECT_EQ(parseStageKind(toString(v)), v) << toString(v);
    }
    for (PortKind const v : {PortKind::kText, PortKind::kEmbeds, PortKind::kHidden, PortKind::kCodec, PortKind::kAudio,
             PortKind::kImage, PortKind::kRequest})
    {
        EXPECT_EQ(parsePortKind(toString(v)), v) << toString(v);
    }
    for (Transport const v : {Transport::kDeviceTensor, Transport::kHostValue, Transport::kChunkStream})
    {
        EXPECT_EQ(parseTransport(toString(v)), v) << toString(v);
    }
    for (OnMissing const v : {OnMissing::kFail, OnMissing::kSkip, OnMissing::kPrune})
    {
        EXPECT_EQ(parseOnMissing(toString(v)), v) << toString(v);
    }
}

// Given a spelling that names no enum value,
// When it is parsed as a PortKind or an OnMissing,
// Then it is rejected with a message naming the bad spelling and listing the legal ones.
TEST(StageConfigTest, UnknownEnumValueIsRejectedAndTheMessageListsTheLegalOnes)
{
    try
    {
        parsePortKind("video");
        FAIL() << "video is not a PortKind";
    }
    catch (std::runtime_error const& error)
    {
        std::string const what = error.what();
        EXPECT_THAT(what, HasSubstr("video"));
        EXPECT_THAT(what, HasSubstr(toString(PortKind::kText)));
        EXPECT_THAT(what, HasSubstr(toString(PortKind::kAudio)));
    }

    try
    {
        parseOnMissing("optional");
        FAIL() << "optional is not an OnMissing";
    }
    catch (std::runtime_error const& error)
    {
        std::string const what = error.what();
        EXPECT_THAT(what, HasSubstr("optional"));
        EXPECT_THAT(what, HasSubstr(toString(OnMissing::kSkip)));
        EXPECT_THAT(what, HasSubstr(toString(OnMissing::kPrune)));
    }
}

// Given a stage-port reference and a request-field reference,
// When each is parsed,
// Then each fills only its own half of SourceRef, and str() spells it back unchanged.
TEST(StageConfigTest, ParseSourceRefAcceptsBothForms)
{
    SourceRef const edge = parseSourceRef("talker:codes");
    EXPECT_FALSE(edge.fromRequest);
    EXPECT_EQ(edge.stageId, "talker");
    EXPECT_EQ(edge.portId, "codes");
    EXPECT_TRUE(edge.requestField.empty());
    EXPECT_EQ(edge.str(), "talker:codes");

    SourceRef const fromRequest = parseSourceRef("request:messages");
    EXPECT_TRUE(fromRequest.fromRequest);
    EXPECT_EQ(fromRequest.requestField, "messages");
    EXPECT_TRUE(fromRequest.stageId.empty());
    EXPECT_EQ(fromRequest.str(), "request:messages");
}

// Given a source ref with no separator, an empty side, or two separators,
// When it is parsed,
// Then it is rejected.
TEST(StageConfigTest, ParseSourceRefRejectsMalformedStrings)
{
    EXPECT_THROW(parseSourceRef(""), std::runtime_error);        // no separator
    EXPECT_THROW(parseSourceRef("talker"), std::runtime_error);  // no separator
    EXPECT_THROW(parseSourceRef("talker:"), std::runtime_error); // empty right side
    EXPECT_THROW(parseSourceRef(":codes"), std::runtime_error);  // empty left side
    EXPECT_THROW(parseSourceRef("a:b:c"), std::runtime_error);   // two separators
}

// Given a config whose schema_version is not the supported one,
// When it is parsed,
// Then it is rejected with a message naming schema_version and the version found.
TEST(StageConfigTest, ForeignSchemaVersionIsRejected)
{
    json topology = minimalGraph();
    topology["schema_version"] = MultiStageConfig::kSchemaVersion + 1;

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("schema_version"));
    EXPECT_THAT(what, HasSubstr(std::to_string(MultiStageConfig::kSchemaVersion + 1)));
}

// Given a stage carrying a key the schema does not define,
// When the config is parsed,
// Then it is rejected naming the key and the stage, rather than the setting being silently ignored.
TEST(StageConfigTest, UnknownStageKeyIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "encoder")["colour"] = "red";

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("unknown key"));
    EXPECT_THAT(what, HasSubstr("colour"));
    EXPECT_THAT(what, HasSubstr("encoder"));
}

// Given a port carrying a key the schema does not define,
// When the config is parsed,
// Then it is rejected naming the key and the port.
TEST(StageConfigTest, UnknownPortKeyIsRejected)
{
    json topology = minimalGraph();
    portJson(topology, "encoder", "features")["stride"] = 2;

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("unknown key"));
    EXPECT_THAT(what, HasSubstr("stride"));
    EXPECT_THAT(what, HasSubstr("features"));
}

// Given an input carrying a key the schema does not define,
// When the config is parsed,
// Then it is rejected naming the key and the input.
TEST(StageConfigTest, UnknownInputKeyIsRejected)
{
    json topology = minimalGraph();
    inputJson(topology, "decoder", "audio")["optional"] = true;

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("unknown key"));
    EXPECT_THAT(what, HasSubstr("optional"));
    EXPECT_THAT(what, HasSubstr("audio"));
}

// Given two stages with the same id,
// When the config is validated,
// Then it is rejected naming the duplicated id.
TEST(StageConfigTest, DuplicateStageIdIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "decoder")["id"] = "encoder";

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("duplicate stage id"));
    EXPECT_THAT(what, HasSubstr("encoder"));
}

// Given a stage whose id is the reserved request prefix,
// When the config is validated,
// Then it is rejected as reserved: "request:<x>" would no longer tell a request field from that stage's port.
TEST(StageConfigTest, StageIdRequestIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "encoder")["id"] = "request";

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("request"));
    EXPECT_THAT(what, HasSubstr("reserved"));
}

// Given a stage that emits two ports with the same id,
// When the config is validated,
// Then it is rejected naming the stage and the port.
TEST(StageConfigTest, DuplicatePortIdWithinOneStageIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "encoder").at("emits").push_back(makePort("features", false, PortKind::kText));

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("duplicate port id"));
    EXPECT_THAT(what, HasSubstr("encoder"));
    EXPECT_THAT(what, HasSubstr("features"));
}

// Given an input whose `from` names an unknown stage, or a port its stage does not emit,
// When the config is validated,
// Then it is rejected naming the input's stage or name and the missing stage or port.
TEST(StageConfigTest, DanglingFromIsRejected)
{
    json unknownStage = minimalGraph();
    inputJson(unknownStage, "decoder", "audio")["from"] = "ghost:features";
    std::string what = rejectionOf(unknownStage);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("unknown stage"));
    EXPECT_THAT(what, HasSubstr("ghost"));
    EXPECT_THAT(what, HasSubstr("audio"));

    json unknownPort = minimalGraph();
    inputJson(unknownPort, "decoder", "audio")["from"] = "encoder:logits";
    what = rejectionOf(unknownPort);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("does not emit port"));
    EXPECT_THAT(what, HasSubstr("logits"));
    EXPECT_THAT(what, HasSubstr("decoder"));
}

// Given a terminals entry naming a port that is not flagged terminal,
// When the config is validated,
// Then it is rejected naming the entry.
TEST(StageConfigTest, TerminalListingANonTerminalPortIsRejected)
{
    json topology = minimalGraph();
    topology.at("terminals").push_back("encoder:features");

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("encoder:features"));
    EXPECT_THAT(what, HasSubstr("not declared terminal"));
}

// Given an edge between two stages on different lanes,
// When the config is validated,
// Then it is rejected over host_value with a message naming the edge and chunk_stream, and accepted over chunk_stream.
TEST(StageConfigTest, CrossLaneEdgeMustBeAChunkStream)
{
    json rejected = minimalGraph();
    stageJson(rejected, "decoder")["stream_lane"] = 1;
    portJson(rejected, "encoder", "features")["transport"] = toString(Transport::kHostValue);

    std::string const what = rejectionOf(rejected);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("crosses lanes"));
    EXPECT_THAT(what, HasSubstr("encoder:features"));
    EXPECT_THAT(what, HasSubstr(toString(Transport::kChunkStream)));

    json accepted = minimalGraph();
    stageJson(accepted, "decoder")["stream_lane"] = 1;
    portJson(accepted, "encoder", "features")["transport"] = toString(Transport::kChunkStream);
    EXPECT_EQ(rejectionOf(accepted), "");
}

// Given a device_tensor edge between two stages on different lanes,
// When the config is validated,
// Then it is rejected naming the transport and the consumer.
TEST(StageConfigTest, DeviceTensorEdgeAcrossLanesIsRejected)
{
    // Same rule as above, on the transport that would hand a consumer running on
    // another stream a buffer its producer is still writing.
    json topology = minimalGraph();
    stageJson(topology, "decoder")["stream_lane"] = 1;
    ASSERT_EQ(portJson(topology, "encoder", "features").at("transport"), toString(Transport::kDeviceTensor));

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("crosses lanes"));
    EXPECT_THAT(what, HasSubstr(toString(Transport::kDeviceTensor)));
    EXPECT_THAT(what, HasSubstr("decoder"));
}

// Given two stages that each read a port of the other,
// When the config is validated,
// Then it is rejected with a cycle trace naming both.
TEST(StageConfigTest, CycleIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "encoder").at("inputs").push_back(makeInput("feedback", "decoder:text"));

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("cycle"));
    EXPECT_THAT(what, HasSubstr("encoder"));
    EXPECT_THAT(what, HasSubstr("decoder"));
}

// Given a stage that reads its own port,
// When the config is validated,
// Then it is rejected as a cycle naming the stage.
TEST(StageConfigTest, StageConsumingItsOwnPortIsRejected)
{
    // A one-node cycle. Route::resolve rejects it too, but per request; validate() is where
    // the config author is told.
    json topology = minimalGraph();
    stageJson(topology, "decoder").at("inputs").push_back(makeInput("feedback", "decoder:text"));

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("cycle"));
    EXPECT_THAT(what, HasSubstr("decoder"));
}

// Given a stage artifact that climbs out of the config root with '..',
// When the config is validated against that root,
// Then it is rejected naming the stage, and a path inside the root is accepted.
TEST(StageConfigTest, ArtifactEscapingTheConfigRootIsRejected)
{
    json topology = minimalGraph();
    stageJson(topology, "decoder")["artifact"] = "../outside";

    std::string const what = rejectionOf(topology, "/tmp");
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr(".."));

    stageJson(topology, "decoder")["artifact"] = "engines/decoder";
    EXPECT_EQ(rejectionOf(topology, "/tmp"), "");
}

// Given a graph with no port flagged terminal and a terminals list that is absent or empty,
// When the config is validated,
// Then it is rejected naming 'terminals': no sink could ever attach, so every request would fail.
TEST(StageConfigTest, MissingOrEmptyTerminalsIsRejected)
{
    json missing = minimalGraph();
    portJson(missing, "decoder", "text")["terminal"] = false;
    missing.erase("terminals");
    std::string what = rejectionOf(missing);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("'terminals'"));

    json empty = minimalGraph();
    portJson(empty, "decoder", "text")["terminal"] = false;
    empty["terminals"] = json::array();
    what = rejectionOf(empty);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("'terminals'"));
}

// Given a listed terminal port of each kind other than text, audio, and image,
// When the config is validated,
// Then it is rejected naming the terminal and its kind, and the same port as audio or image is accepted.
TEST(StageConfigTest, TerminalOfANonMediaKindIsRejected)
{
    for (PortKind const kind : {PortKind::kEmbeds, PortKind::kHidden, PortKind::kCodec, PortKind::kRequest})
    {
        json topology = minimalGraph();
        portJson(topology, "encoder", "features")["terminal"] = true;
        portJson(topology, "encoder", "features")["kind"] = toString(kind);
        topology.at("terminals").push_back("encoder:features");

        std::string const what = rejectionOf(topology);
        ASSERT_FALSE(what.empty()) << toString(kind);
        EXPECT_THAT(what, HasSubstr("encoder:features"));
        EXPECT_THAT(what, HasSubstr(toString(kind)));
    }

    for (PortKind const kind : {PortKind::kAudio, PortKind::kImage})
    {
        json media = minimalGraph();
        portJson(media, "encoder", "features")["terminal"] = true;
        portJson(media, "encoder", "features")["kind"] = toString(kind);
        media.at("terminals").push_back("encoder:features");
        EXPECT_EQ(rejectionOf(media), "") << toString(kind);
    }
}

// Given a text port flagged terminal that the terminals list leaves out,
// When the config is validated,
// Then it is rejected naming the port: the flag and the list must name the same ports.
TEST(StageConfigTest, PortFlaggedTerminalButNotListedIsRejected)
{
    json topology = minimalGraph();
    portJson(topology, "encoder", "features")["terminal"] = true;
    portJson(topology, "encoder", "features")["kind"] = toString(PortKind::kText);

    std::string const what = rejectionOf(topology);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("encoder:features"));
    EXPECT_THAT(what, HasSubstr("'terminals'"));
}

// Given an integer key whose value does not fit in 32 bits but would wrap to a legal one,
// When the config is parsed,
// Then it is rejected naming the key, rather than run with the wrapped value.
TEST(StageConfigTest, IntegerOutsideTheInt32RangeIsRejected)
{
    // Narrowed, each value lands on a legal one (2^32 + 1 on 1, +-2^32 on 0, 2^32 + kSchemaVersion
    // on the supported version), so only the range check can reject these fixtures.
    int64_t const twoTo32 = int64_t{1} << 32;

    json maxBatch = minimalGraph();
    stageJson(maxBatch, "decoder")["max_batch"] = twoTo32 + 1;
    std::string what = rejectionOf(maxBatch);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("max_batch"));
    EXPECT_THAT(what, HasSubstr("range"));

    json maxSeq = minimalGraph();
    stageJson(maxSeq, "decoder")["max_seq"] = -twoTo32;
    what = rejectionOf(maxSeq);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("max_seq"));
    EXPECT_THAT(what, HasSubstr("range"));

    // Unsigned, the form a parsed non-negative JSON literal takes.
    json streamLane = minimalGraph();
    stageJson(streamLane, "decoder")["stream_lane"] = static_cast<uint64_t>(twoTo32);
    what = rejectionOf(streamLane);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("stream_lane"));
    EXPECT_THAT(what, HasSubstr("range"));

    json version = minimalGraph();
    version["schema_version"] = twoTo32 + MultiStageConfig::kSchemaVersion;
    what = rejectionOf(version);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("schema_version"));
    EXPECT_THAT(what, HasSubstr("range"));
}

// Given both stages on one lane whose index is not below the number of stages,
// When the config is validated,
// Then it is rejected naming stream_lane, and the same graph on lane 1 is accepted.
TEST(StageConfigTest, StreamLaneNotBelowTheStageCountIsRejected)
{
    for (int32_t const lane : {2, std::numeric_limits<int32_t>::max()})
    {
        json topology = minimalGraph();
        stageJson(topology, "encoder")["stream_lane"] = lane;
        stageJson(topology, "decoder")["stream_lane"] = lane;

        std::string const what = rejectionOf(topology);
        ASSERT_FALSE(what.empty()) << lane;
        EXPECT_THAT(what, HasSubstr("stream_lane"));
        EXPECT_THAT(what, HasSubstr(std::to_string(lane)));
    }

    json accepted = minimalGraph();
    stageJson(accepted, "encoder")["stream_lane"] = 1;
    stageJson(accepted, "decoder")["stream_lane"] = 1;
    EXPECT_EQ(rejectionOf(accepted), "");
}
