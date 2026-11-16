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

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

//! What a stage does per invocation. validateRunners() checks it against the runner's declaration;
//! the executor drives every kind through the same IStage calls.
enum class StageKind : uint8_t
{
    kHost,           //!< Pure CPU, no engine.
    kEngine,         //!< One enqueue per request.
    kAutoregressive, //!< Decode loop carrying KV.
    kIterative,      //!< step = one window of a growing input, finish = flush the tail.
};

//! What travels on a port. Fixes the binding discipline, not the semantics.
enum class PortKind : uint8_t
{
    kText,    //!< Token ids.
    kEmbeds,  //!< {B, S, H} embeddings.
    kHidden,  //!< Hidden states plus a coverage window.
    kCodec,   //!< Discrete codes.
    kAudio,   //!< PCM.
    kImage,   //!< Packed RGB8 pixels, {frames, height, width, 3}.
    kRequest, //!< One host LLMGenerationRequest holding a single, already parsed request.
};

//! How an edge moves its payload.
enum class Transport : uint8_t
{
    kDeviceTensor, //!< Direct binding. Producer must not overwrite while the consumer holds it.
    kHostValue,    //!< Small host-side value.
    kChunkStream,  //!< The only transport validate() accepts across lanes; load() rejects every cross-lane edge.
};

//! What the route does when this input's source is not active for THIS request.
//! Three meanings that a single `bool optional` cannot separate: run without it,
//! prune me, or fail loudly.
enum class OnMissing : uint8_t
{
    kFail,  //!< Default. The stage cannot run; admission fails if a wanted terminal needs it.
    kSkip,  //!< The binding is ABSENT from the exec context. This stage still runs.
    kPrune, //!< This stage leaves the route. Consumers then apply their own OnMissing.
};

//! One output of a stage.
struct PortSpec
{
    std::string id;
    PortKind kind{PortKind::kText};
    Transport transport{Transport::kDeviceTensor};
    std::string binding;  //!< Engine tensor name.
    bool terminal{false}; //!< A caller may attach a sink here.
};

//! One input of a stage.
struct InputSpec
{
    std::string name;
    std::string from; //!< "<stageId>:<portId>" or "request:<field>".
    OnMissing onMissing{OnMissing::kFail};
};

//! A parsed `from`. Exactly one of the two forms is valid.
struct SourceRef
{
    bool fromRequest{false}; //!< true -> requestField is set; false -> stageId/portId are set.
    std::string stageId;
    std::string portId;
    std::string requestField;

    //! Round-trips back to the original string. Used in diagnostics.
    std::string str() const;
};

//! Parses "<stageId>:<portId>" or "request:<field>". Throws on a malformed string.
SourceRef parseSourceRef(std::string const& from);

struct StageConfig
{
    std::string id;       //!< Unique.
    std::string runner;   //!< Registry key. Decides which IStage implementation is built.
    std::string artifact; //!< Engine directory, relative to the config root. May be empty.

    StageKind kind{StageKind::kHost};

    std::vector<PortSpec> emits;
    std::vector<InputSpec> inputs;

    int32_t maxBatch{1};
    int32_t maxSeq{0};
    int32_t streamLane{0};

    //! Validated by the runner's own schema, not here. An empty object rather than a null:
    //! a stage that names no options has no options, and every runner reads this with
    //! value(), which throws on a null.
    nlohmann::json config = nlohmann::json::object();

    PortSpec const* findPort(std::string const& portId) const;
    InputSpec const* findInput(std::string const& name) const;
};

/*!
 * @brief The topology, as data.
 *
 * A MultiStageConfig is the whole topology: which stages exist, what each one
 * emits, and which port feeds which input. It is parsed from
 * `multistage_config.json` and is pure data -- no callables, and no path that
 * escapes the config root.
 *
 * Nothing here knows about a model family. A family is a set of values in this
 * structure, not a branch in the code.
 */
struct MultiStageConfig
{
    static constexpr int32_t kSchemaVersion = 1;

    int32_t schemaVersion{kSchemaVersion};
    std::string label; //!< Diagnostics only. Never dispatched on.
    std::vector<StageConfig> stages;
    //! Port ids ("<stageId>:<portId>"), NOT stage ids. Same key space as Sinks and wanted.
    //! Non-empty, and exactly the ports flagged terminal; each is of kind text, audio, or image.
    std::vector<std::string> terminals;

    StageConfig const* findStage(std::string const& id) const;
    int32_t stageIndex(std::string const& id) const; //!< -1 when absent.

    //! Parses and validates. Throws std::runtime_error naming the offending id on any failure.
    static MultiStageConfig parse(nlohmann::json const& root);
    //! Reads <configRoot>/multistage_config.json.
    static MultiStageConfig load(std::string const& configRoot);
};

/*!
 * @brief Structural checks that must hold before anything is constructed.
 *
 * Every rule here is one a layering bug could otherwise turn into a runtime
 * surprise. This is the authority at load; the Python builder runs its own, separate
 * preflight before it writes a config.
 *
 * @param config    Parsed topology.
 * @param configRoot Used only to reject an `artifact` that escapes the root.
 *                   Pass an empty string to skip the filesystem check.
 * @throws std::runtime_error naming the stage, the input and the rule.
 */
void validate(MultiStageConfig const& config, std::string const& configRoot = {});

//! Enum spelling used in JSON. Round-trips with the parse* helpers below.
char const* toString(StageKind v);
char const* toString(PortKind v);
char const* toString(Transport v);
char const* toString(OnMissing v);

StageKind parseStageKind(std::string const& v);
PortKind parsePortKind(std::string const& v);
Transport parseTransport(std::string const& v);
OnMissing parseOnMissing(std::string const& v);

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
