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

#include "runtime/multiStage/stageRegistry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt::multiStage;
using nlohmann::json;
using ::testing::Contains;
using ::testing::HasSubstr;

namespace
{

constexpr char const* kEncoder = "unittest.encoder";
constexpr char const* kDecoder = "unittest.decoder";

//! The factory of every runner here; validation never builds a stage.
std::unique_ptr<IStage> buildNothing(StageInitContext const& /*init*/)
{
    return nullptr;
}

RunnerEntry entry(RunnerDecl decl)
{
    return {std::move(decl), &buildNothing};
}

RunnerDecl makeDecl(std::string const& name, StageKind kind)
{
    RunnerDecl decl;
    decl.name = name;
    decl.kind = kind;
    return decl;
}

//! Declaration the encoder half of twoStageGraph() satisfies.
RunnerDecl encoderDecl(std::string const& name)
{
    RunnerDecl decl = makeDecl(name, StageKind::kEngine);
    decl.ports = {{"features", PortKind::kEmbeds}};
    decl.inputs = {"wav"};
    return decl;
}

//! Declaration the decoder half satisfies. Tests break one field at a time.
RunnerDecl decoderDecl(std::string const& name)
{
    RunnerDecl decl = makeDecl(name, StageKind::kAutoregressive);
    decl.ports = {{"text", PortKind::kText}};
    decl.inputs = {"audio"};
    decl.requiredConfigKeys = {"eos_id"};
    return decl;
}

json makePort(std::string const& id, PortKind kind, bool terminal)
{
    return json{{"id", id}, {"kind", toString(kind)}, {"transport", toString(Transport::kDeviceTensor)},
        {"terminal", terminal}};
}

json makeInput(std::string const& name, std::string const& from)
{
    return json{{"name", name}, {"from", from}, {"on_missing", toString(OnMissing::kFail)}};
}

//! One producer plus one consumer, bound to the runners under test.
json twoStageGraph(std::string const& encoderRunner, std::string const& decoderRunner)
{
    json encoder{{"id", "encoder"}, {"runner", encoderRunner}, {"kind", toString(StageKind::kEngine)},
        {"inputs", json::array({makeInput("wav", "request:audio")})},
        {"emits", json::array({makePort("features", PortKind::kEmbeds, false)})}};

    json decoder{{"id", "decoder"}, {"runner", decoderRunner}, {"kind", toString(StageKind::kAutoregressive)},
        {"inputs", json::array({makeInput("audio", "encoder:features")})},
        {"emits", json::array({makePort("text", PortKind::kText, true)})}, {"config", json{{"eos_id", 2}}}};

    return json{{"schema_version", MultiStageConfig::kSchemaVersion},
        {"stages", json::array({std::move(encoder), std::move(decoder)})},
        {"terminals", json::array({"decoder:text"})}};
}

//! The message validateRunners rejects the topology with, or "" when it accepts it.
std::string rejectionOf(json const& topology, StageRegistry const& runners)
{
    try
    {
        validateRunners(MultiStageConfig::parse(topology), runners);
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return {};
}

std::ptrdiff_t indexOf(std::vector<std::string> const& names, std::string const& name)
{
    return std::find(names.begin(), names.end(), name) - names.begin();
}

} // namespace

// Given a runner declaration added to the registry,
// When it is looked up by its exact name, and by a name nobody registered,
// Then the first returns the stored declaration (name, kind, inputs) and the second returns null.
TEST(StageRegistryTest, RegisteredRunnerIsFoundAndUnknownOneIsNot)
{
    StageRegistry runners;
    runners.add(entry(encoderDecl(kEncoder)));

    RunnerEntry const* found = runners.find(kEncoder);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->decl.name, kEncoder);
    EXPECT_EQ(found->decl.kind, StageKind::kEngine);
    EXPECT_EQ(found->decl.inputs, std::vector<std::string>{"wav"});
    EXPECT_EQ(runners.find(std::string{kEncoder} + ".typo"), nullptr);
}

// Given a runner already registered under a name,
// When a second declaration is added under the same name,
// Then the add throws naming the runner, and the first declaration stays registered unchanged.
TEST(StageRegistryTest, RegisteringTheSameNameTwiceIsRejected)
{
    StageRegistry runners;
    runners.add(entry(makeDecl(kEncoder, StageKind::kEngine)));

    try
    {
        runners.add(entry(makeDecl(kEncoder, StageKind::kHost)));
        FAIL() << "a second registration under " << kEncoder << " must not be accepted";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_THAT(std::string(error.what()), HasSubstr(kEncoder));
    }

    // The first registration wins, so the loser cannot silently replace it.
    ASSERT_NE(runners.find(kEncoder), nullptr);
    EXPECT_EQ(runners.find(kEncoder)->decl.kind, StageKind::kEngine);
}

// Given a runner declaration with an empty name,
// When it is added to the registry,
// Then the add throws.
TEST(StageRegistryTest, RegisteringAnUnnamedRunnerIsRejected)
{
    StageRegistry runners;
    EXPECT_THROW(runners.add(entry(makeDecl("", StageKind::kHost))), std::runtime_error);
}

// Given a runner declaration with a name but a null factory,
// When it is added to the registry,
// Then the add throws naming the runner, and the runner is not registered.
TEST(StageRegistryTest, RegisteringARunnerWithoutAFactoryIsRejected)
{
    StageRegistry runners;
    try
    {
        runners.add({encoderDecl(kEncoder), nullptr});
        FAIL() << "a runner without a factory must not be accepted";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_THAT(std::string(error.what()), HasSubstr(kEncoder));
    }
    EXPECT_EQ(runners.find(kEncoder), nullptr);
}

// Given two runners registered in reverse lexical order,
// When the registry lists its names,
// Then the list is sorted and holds both, the lexically smaller first.
TEST(StageRegistryTest, NamesComeBackSorted)
{
    // Registered out of order, so the sorting is the registry's doing.
    StageRegistry runners;
    runners.add(entry(makeDecl("unittest.zulu", StageKind::kHost)));
    runners.add(entry(makeDecl("unittest.alpha", StageKind::kHost)));

    std::vector<std::string> const names = runners.names();
    EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
    ASSERT_THAT(names, Contains("unittest.alpha"));
    ASSERT_THAT(names, Contains("unittest.zulu"));
    EXPECT_LT(indexOf(names, "unittest.alpha"), indexOf(names, "unittest.zulu"));
}

// Given a two-stage graph whose stages match their runners' kind, ports, inputs and required config keys,
// When the graph is validated against the registry,
// Then it is accepted.
TEST(StageRegistryTest, GraphMatchingItsRunnersPasses)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));
    runners.add(entry(decoderDecl(decoder)));

    EXPECT_EQ(rejectionOf(twoStageGraph(encoder, decoder), runners), "");
}

// Given a runner that declares a port and an input its stage in the graph does not use,
// When the graph is validated,
// Then it is accepted: a stage may emit and bind a subset of what its runner declares.
TEST(StageRegistryTest, PortTheRunnerDeclaresButTheConfigOmitsIsFine)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    // Not every deployment wants every output, so a superset declaration passes.
    RunnerDecl decl = decoderDecl(decoder);
    decl.ports.emplace_back("hidden", PortKind::kHidden);
    decl.inputs.emplace_back("voice");
    runners.add(entry(std::move(decl)));

    EXPECT_EQ(rejectionOf(twoStageGraph(encoder, decoder), runners), "");
}

// Given a graph whose decoder stage names a runner nobody registered,
// When the graph is validated,
// Then it is rejected with an error naming the stage and the runner as not registered.
TEST(StageRegistryTest, UnregisteredRunnerIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    runners.add(entry(encoderDecl(encoder)));

    std::string const missing = "unittest.nobody";
    std::string const what = rejectionOf(twoStageGraph(encoder, missing), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("not registered"));
    EXPECT_THAT(what, HasSubstr(missing));
    EXPECT_THAT(what, HasSubstr("decoder"));
}

// Given a stage declared autoregressive whose runner is registered as iterative,
// When the graph is validated,
// Then it is rejected with an error naming the stage and both kinds.
TEST(StageRegistryTest, KindDisagreeingWithTheRunnerIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    // The topology declares the decoder autoregressive; the runner is iterative.
    RunnerDecl decl = decoderDecl(decoder);
    decl.kind = StageKind::kIterative;
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr(toString(StageKind::kIterative)));
    EXPECT_THAT(what, HasSubstr(toString(StageKind::kAutoregressive)));
}

// Given a stage that emits 'text' while its runner declares only 'codes',
// When the graph is validated,
// Then it is rejected with an error naming the stage, the undeclared port and the ports the runner does declare.
TEST(StageRegistryTest, EmittingAPortTheRunnerDoesNotDeclareIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    RunnerDecl decl = decoderDecl(decoder);
    decl.ports = {{"codes", PortKind::kCodec}};
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("does not declare"));
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr("'text'"));
    EXPECT_THAT(what, HasSubstr("'codes'"));
}

// Given a stage whose 'text' port is text in the graph but codec in the runner declaration,
// When the graph is validated,
// Then it is rejected with an error naming the stage, the port and the runner's kind for it.
TEST(StageRegistryTest, PortKindDisagreeingWithTheRunnerIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    RunnerDecl decl = decoderDecl(decoder);
    decl.ports = {{"text", PortKind::kCodec}};
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr("'text'"));
    EXPECT_THAT(what, HasSubstr(toString(PortKind::kCodec)));
}

// Given a stage that binds input 'audio' while its runner accepts only 'voice',
// When the graph is validated,
// Then it is rejected with an error naming the stage, the unaccepted input and the inputs the runner does accept.
TEST(StageRegistryTest, BindingAnInputTheRunnerDoesNotAcceptIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    RunnerDecl decl = decoderDecl(decoder);
    decl.inputs = {"voice"};
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("does not accept"));
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr("'audio'"));
    EXPECT_THAT(what, HasSubstr("'voice'"));
}

// Given a runner that requires a config key its stage's config object lacks,
// When the graph is validated,
// Then it is rejected with an error naming the stage and the missing key.
TEST(StageRegistryTest, MissingRequiredConfigKeyIsRejected)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(encoderDecl(encoder)));

    RunnerDecl decl = decoderDecl(decoder);
    decl.requiredConfigKeys = {"eos_id", "prefill_builder"};
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("missing required key"));
    EXPECT_THAT(what, HasSubstr("decoder"));
    EXPECT_THAT(what, HasSubstr("prefill_builder"));
}

// Given a stage with no config object at all whose runner requires a config key,
// When the graph is validated,
// Then it is rejected like a config that lacks the key, with an error naming the stage and the key.
TEST(StageRegistryTest, StageWithNoConfigObjectStillFailsARequiredKey)
{
    StageRegistry runners;
    std::string const encoder = kEncoder;
    std::string const decoder = kDecoder;
    runners.add(entry(decoderDecl(decoder)));

    // twoStageGraph gives the encoder stage no "config" key at all, so the
    // required-key check runs against the default empty object.
    RunnerDecl decl = encoderDecl(encoder);
    decl.requiredConfigKeys = {"sample_rate"};
    runners.add(entry(std::move(decl)));

    std::string const what = rejectionOf(twoStageGraph(encoder, decoder), runners);
    ASSERT_FALSE(what.empty());
    EXPECT_THAT(what, HasSubstr("missing required key"));
    EXPECT_THAT(what, HasSubstr("encoder"));
    EXPECT_THAT(what, HasSubstr("sample_rate"));
}
