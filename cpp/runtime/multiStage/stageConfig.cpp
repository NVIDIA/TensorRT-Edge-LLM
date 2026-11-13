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

#include "common/checkMacros.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

using Json = nlohmann::json;

namespace
{

//! Reserved left-hand side of a `from` string, so no stage may claim it as an id.
constexpr char const* kRequestPrefix = "request";
constexpr char const* kConfigFileName = "multistage_config.json";

template <typename T>
struct EnumEntry
{
    char const* name;
    T value;
};

constexpr EnumEntry<StageKind> kStageKindNames[] = {
    {"host", StageKind::kHost},
    {"engine", StageKind::kEngine},
    {"autoregressive", StageKind::kAutoregressive},
    {"iterative", StageKind::kIterative},
};

constexpr EnumEntry<PortKind> kPortKindNames[] = {
    {"text", PortKind::kText},
    {"embeds", PortKind::kEmbeds},
    {"hidden", PortKind::kHidden},
    {"codec", PortKind::kCodec},
    {"audio", PortKind::kAudio},
    {"image", PortKind::kImage},
    {"request", PortKind::kRequest},
};

constexpr EnumEntry<Transport> kTransportNames[] = {
    {"device_tensor", Transport::kDeviceTensor},
    {"host_value", Transport::kHostValue},
    {"chunk_stream", Transport::kChunkStream},
};

constexpr EnumEntry<OnMissing> kOnMissingNames[] = {
    {"fail", OnMissing::kFail},
    {"skip", OnMissing::kSkip},
    {"prune", OnMissing::kPrune},
};

template <typename T, std::size_t N>
char const* enumName(T value, EnumEntry<T> const (&table)[N])
{
    for (auto const& entry : table)
    {
        if (entry.value == value)
        {
            return entry.name;
        }
    }
    return "<invalid>";
}

template <typename T, std::size_t N>
std::string spellings(EnumEntry<T> const (&table)[N])
{
    std::string out;
    for (auto const& entry : table)
    {
        if (!out.empty())
        {
            out += ", ";
        }
        out += entry.name;
    }
    return out;
}

template <typename T, std::size_t N>
T enumValue(std::string const& text, std::string const& what, EnumEntry<T> const (&table)[N])
{
    for (auto const& entry : table)
    {
        if (text == entry.name)
        {
            return entry.value;
        }
    }
    ELLM_CHECK(false, "unknown " + what + " '" + text + "'; expected one of: " + spellings(table));
    return table[0].value;
}

std::string joinKeys(std::initializer_list<char const*> const& keys)
{
    std::string out;
    for (auto const* key : keys)
    {
        if (!out.empty())
        {
            out += ", ";
        }
        out += key;
    }
    return out;
}

//! An unknown key is a typo we can catch now instead of a silently ignored setting later.
void rejectUnknownKeys(Json const& obj, std::string const& where, std::initializer_list<char const*> allowed)
{
    for (auto const& item : obj.items())
    {
        bool const known
            = std::any_of(allowed.begin(), allowed.end(), [&item](char const* key) { return item.key() == key; });
        ELLM_CHECK(known, where + ": unknown key '" + item.key() + "'; allowed keys are " + joinKeys(allowed));
    }
}

Json const* requireField(Json const& obj, char const* key, std::string const& where)
{
    auto const it = obj.find(key);
    ELLM_CHECK(it != obj.end(), where + ": missing required key '" + std::string(key) + "'");
    return &(*it);
}

std::string requireString(Json const& obj, char const* key, std::string const& where)
{
    auto const* field = requireField(obj, key, where);
    ELLM_CHECK(field->is_string(), where + ": key '" + std::string(key) + "' must be a string");
    return field->get<std::string>();
}

std::string optionalString(Json const& obj, char const* key, std::string const& where, std::string const& fallback)
{
    auto const it = obj.find(key);
    if (it == obj.end())
    {
        return fallback;
    }
    ELLM_CHECK(it->is_string(), where + ": key '" + std::string(key) + "' must be a string");
    return it->get<std::string>();
}

bool optionalBool(Json const& obj, char const* key, std::string const& where, bool fallback)
{
    auto const it = obj.find(key);
    if (it == obj.end())
    {
        return fallback;
    }
    ELLM_CHECK(it->is_boolean(), where + ": key '" + std::string(key) + "' must be a boolean");
    return it->get<bool>();
}

//! get<int32_t>() is a plain static_cast, so a value outside int32 would wrap silently.
int32_t requireInt32(Json const& value, char const* key, std::string const& where)
{
    std::string const what = where + ": key '" + std::string(key) + "'";
    ELLM_CHECK(value.is_number_integer(), what + " must be an integer");
    bool const fits = value.is_number_unsigned()
        ? value.get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
        : value.get<int64_t>() >= std::numeric_limits<int32_t>::min()
            && value.get<int64_t>() <= std::numeric_limits<int32_t>::max();
    ELLM_CHECK(fits, what + " is out of the 32-bit integer range, got " + value.dump());
    return static_cast<int32_t>(value.get<int64_t>());
}

int32_t optionalInt(Json const& obj, char const* key, std::string const& where, int32_t fallback)
{
    auto const it = obj.find(key);
    if (it == obj.end())
    {
        return fallback;
    }
    return requireInt32(*it, key, where);
}

std::vector<std::string> optionalStringArray(Json const& obj, char const* key, std::string const& where)
{
    auto const it = obj.find(key);
    if (it == obj.end())
    {
        return {};
    }
    ELLM_CHECK(it->is_array(), where + ": key '" + std::string(key) + "' must be an array of strings");
    std::vector<std::string> out;
    out.reserve(it->size());
    for (auto const& element : *it)
    {
        ELLM_CHECK(element.is_string(), where + ": key '" + std::string(key) + "' must be an array of strings");
        out.push_back(element.get<std::string>());
    }
    return out;
}

template <typename T, std::size_t N>
T optionalEnum(Json const& obj, char const* key, std::string const& where, EnumEntry<T> const (&table)[N], T fallback)
{
    auto const it = obj.find(key);
    if (it == obj.end())
    {
        return fallback;
    }
    ELLM_CHECK(it->is_string(), where + ": key '" + std::string(key) + "' must be a string");
    return enumValue(it->get<std::string>(), where + " " + key, table);
}

//! @param where Prefixed to the message when the caller knows which stage/input holds the
//!              string; `from` on its own rarely locates the typo.
SourceRef parseRef(std::string const& from, std::string const& where)
{
    std::string const at = where.empty() ? std::string() : where + ": ";
    std::string const shape = R"(expected "<stageId>:<portId>" or "request:<field>")";

    auto const colon = from.find(':');
    ELLM_CHECK(colon != std::string::npos && colon != 0 && colon + 1 < from.size(),
        at + "malformed source ref '" + from + "'; " + shape);
    ELLM_CHECK(from.find(':', colon + 1) == std::string::npos,
        at + "malformed source ref '" + from + "'; more than one ':', " + shape);

    SourceRef ref;
    auto const lhs = from.substr(0, colon);
    auto const rhs = from.substr(colon + 1);
    if (lhs == kRequestPrefix)
    {
        ref.fromRequest = true;
        ref.requestField = rhs;
    }
    else
    {
        ref.stageId = lhs;
        ref.portId = rhs;
    }
    return ref;
}

//! JSON to struct with key, type and schema-version checks only. Graph rules are left to
//! validate(), so load() runs them once, with the config root.
MultiStageConfig buildConfig(Json const& root)
{
    ELLM_CHECK(root.is_object(), "multistage config: root must be a JSON object");
    rejectUnknownKeys(root, "multistage config", {"schema_version", "label", "stages", "terminals"});

    MultiStageConfig config;
    auto const* version = requireField(root, "schema_version", "multistage config");
    config.schemaVersion = requireInt32(*version, "schema_version", "multistage config");
    ELLM_CHECK(config.schemaVersion == MultiStageConfig::kSchemaVersion,
        "multistage config: schema_version " + std::to_string(config.schemaVersion) + " is not supported, expected "
            + std::to_string(MultiStageConfig::kSchemaVersion));

    config.label = optionalString(root, "label", "multistage config", {});
    config.terminals = optionalStringArray(root, "terminals", "multistage config");

    auto const* stagesField = requireField(root, "stages", "multistage config");
    ELLM_CHECK(stagesField->is_array(), "multistage config: 'stages' must be an array");
    config.stages.reserve(stagesField->size());

    for (auto const& stageJson : *stagesField)
    {
        std::string const at = "multistage config: stages[" + std::to_string(config.stages.size()) + "]";
        ELLM_CHECK(stageJson.is_object(), at + " must be an object");

        StageConfig stage;
        stage.id = requireString(stageJson, "id", at);
        std::string const where = "stage '" + stage.id + "'";
        rejectUnknownKeys(stageJson, where,
            {"id", "runner", "artifact", "kind", "emits", "inputs", "max_batch", "max_seq", "stream_lane", "config"});

        stage.runner = requireString(stageJson, "runner", where);
        stage.artifact = optionalString(stageJson, "artifact", where, {});
        stage.kind = optionalEnum(stageJson, "kind", where, kStageKindNames, StageKind::kHost);
        stage.maxBatch = optionalInt(stageJson, "max_batch", where, 1);
        stage.maxSeq = optionalInt(stageJson, "max_seq", where, 0);
        stage.streamLane = optionalInt(stageJson, "stream_lane", where, 0);

        auto const emits = stageJson.find("emits");
        if (emits != stageJson.end())
        {
            ELLM_CHECK(emits->is_array(), where + ": 'emits' must be an array");
            stage.emits.reserve(emits->size());
            for (auto const& portJson : *emits)
            {
                ELLM_CHECK(portJson.is_object(), where + ": every entry of 'emits' must be an object");
                PortSpec port;
                port.id = requireString(portJson, "id", where + " port");
                std::string const portWhere = where + " port '" + port.id + "'";
                rejectUnknownKeys(portJson, portWhere, {"id", "kind", "transport", "binding", "terminal"});
                port.kind = optionalEnum(portJson, "kind", portWhere, kPortKindNames, PortKind::kText);
                port.transport
                    = optionalEnum(portJson, "transport", portWhere, kTransportNames, Transport::kDeviceTensor);
                port.binding = optionalString(portJson, "binding", portWhere, {});
                port.terminal = optionalBool(portJson, "terminal", portWhere, false);
                stage.emits.push_back(std::move(port));
            }
        }

        auto const inputs = stageJson.find("inputs");
        if (inputs != stageJson.end())
        {
            ELLM_CHECK(inputs->is_array(), where + ": 'inputs' must be an array");
            stage.inputs.reserve(inputs->size());
            for (auto const& inputJson : *inputs)
            {
                ELLM_CHECK(inputJson.is_object(), where + ": every entry of 'inputs' must be an object");
                InputSpec input;
                input.name = requireString(inputJson, "name", where + " input");
                std::string const inputWhere = where + " input '" + input.name + "'";
                rejectUnknownKeys(inputJson, inputWhere, {"name", "from", "on_missing"});
                input.from = requireString(inputJson, "from", inputWhere);
                input.onMissing = optionalEnum(inputJson, "on_missing", inputWhere, kOnMissingNames, OnMissing::kFail);
                stage.inputs.push_back(std::move(input));
            }
        }

        // Contents belong to the runner's own schema, so only the shape is checked here.
        auto const runnerConfig = stageJson.find("config");
        if (runnerConfig != stageJson.end())
        {
            ELLM_CHECK(runnerConfig->is_object(), where + ": 'config' must be an object");
            stage.config = *runnerConfig;
        }

        config.stages.push_back(std::move(stage));
    }

    return config;
}

bool findCycle(std::vector<std::vector<int32_t>> const& adjacency, int32_t node, std::vector<uint8_t>& color,
    std::vector<int32_t>& path, std::vector<int32_t>& cycle)
{
    color[static_cast<std::size_t>(node)] = 1;
    path.push_back(node);
    for (auto const next : adjacency[static_cast<std::size_t>(node)])
    {
        if (color[static_cast<std::size_t>(next)] == 1)
        {
            cycle.assign(std::find(path.begin(), path.end(), next), path.end());
            cycle.push_back(next);
            return true;
        }
        if (color[static_cast<std::size_t>(next)] == 0 && findCycle(adjacency, next, color, path, cycle))
        {
            return true;
        }
    }
    path.pop_back();
    color[static_cast<std::size_t>(node)] = 2;
    return false;
}

} // namespace

char const* toString(StageKind v)
{
    return enumName(v, kStageKindNames);
}

char const* toString(PortKind v)
{
    return enumName(v, kPortKindNames);
}

char const* toString(Transport v)
{
    return enumName(v, kTransportNames);
}

char const* toString(OnMissing v)
{
    return enumName(v, kOnMissingNames);
}

StageKind parseStageKind(std::string const& v)
{
    return enumValue(v, "stage kind", kStageKindNames);
}

PortKind parsePortKind(std::string const& v)
{
    return enumValue(v, "port kind", kPortKindNames);
}

Transport parseTransport(std::string const& v)
{
    return enumValue(v, "transport", kTransportNames);
}

OnMissing parseOnMissing(std::string const& v)
{
    return enumValue(v, "on_missing", kOnMissingNames);
}

std::string SourceRef::str() const
{
    return fromRequest ? std::string(kRequestPrefix) + ":" + requestField : stageId + ":" + portId;
}

SourceRef parseSourceRef(std::string const& from)
{
    return parseRef(from, {});
}

PortSpec const* StageConfig::findPort(std::string const& portId) const
{
    for (auto const& port : emits)
    {
        if (port.id == portId)
        {
            return &port;
        }
    }
    return nullptr;
}

InputSpec const* StageConfig::findInput(std::string const& name) const
{
    for (auto const& input : inputs)
    {
        if (input.name == name)
        {
            return &input;
        }
    }
    return nullptr;
}

StageConfig const* MultiStageConfig::findStage(std::string const& id) const
{
    auto const index = stageIndex(id);
    return index < 0 ? nullptr : &stages[static_cast<std::size_t>(index)];
}

int32_t MultiStageConfig::stageIndex(std::string const& id) const
{
    for (std::size_t i = 0; i < stages.size(); ++i)
    {
        if (stages[i].id == id)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

MultiStageConfig MultiStageConfig::parse(Json const& root)
{
    auto config = buildConfig(root);
    validate(config);
    return config;
}

MultiStageConfig MultiStageConfig::load(std::string const& configRoot)
{
    auto const path = (std::filesystem::path(configRoot) / kConfigFileName).string();
    std::ifstream file(path);
    ELLM_CHECK(file.is_open(), "cannot open multistage config '" + path + "'");

    Json root;
    try
    {
        file >> root;
    }
    catch (std::exception const& error)
    {
        ELLM_CHECK(false, "cannot parse multistage config '" + path + "': " + error.what());
    }

    auto config = buildConfig(root);
    validate(config, configRoot);
    return config;
}

void validate(MultiStageConfig const& config, std::string const& configRoot)
{
    auto const stageCount = config.stages.size();

    for (std::size_t i = 0; i < stageCount; ++i)
    {
        auto const& stage = config.stages[i];
        std::string const where = "stage '" + stage.id + "'";

        ELLM_CHECK(!stage.id.empty(), "stages[" + std::to_string(i) + "]: stage id must not be empty");
        ELLM_CHECK(stage.id != kRequestPrefix,
            where + ": '" + kRequestPrefix + "' is the reserved prefix of a request source and cannot be a stage id");
        ELLM_CHECK(config.stageIndex(stage.id) == static_cast<int32_t>(i), where + ": duplicate stage id");

        // One lane per stage is the most a topology can use, and load() creates a stream for
        // every lane up to the highest one named.
        ELLM_CHECK(stage.streamLane >= 0 && static_cast<std::size_t>(stage.streamLane) < stageCount,
            where + ": stream_lane must be in [0, " + std::to_string(stageCount) + "), got "
                + std::to_string(stage.streamLane));
        ELLM_CHECK(stage.maxBatch >= 1, where + ": max_batch must be >= 1, got " + std::to_string(stage.maxBatch));
        ELLM_CHECK(stage.maxSeq >= 0, where + ": max_seq must be >= 0, got " + std::to_string(stage.maxSeq));

        for (auto const& port : stage.emits)
        {
            ELLM_CHECK(stage.findPort(port.id) == &port, where + ": duplicate port id '" + port.id + "'");
        }
        for (auto const& input : stage.inputs)
        {
            ELLM_CHECK(stage.findInput(input.name) == &input, where + ": duplicate input name '" + input.name + "'");
        }
    }

    // Edges: resolvable, endpoints exist, transport agrees with the lane crossing. The
    // resolved indices are kept for the cycle check below rather than parsed a second time.
    std::vector<std::pair<int32_t, int32_t>> edges;
    for (std::size_t i = 0; i < stageCount; ++i)
    {
        auto const& stage = config.stages[i];
        for (auto const& input : stage.inputs)
        {
            std::string const where = "stage '" + stage.id + "' input '" + input.name + "'";
            auto const ref = parseRef(input.from, where);
            if (ref.fromRequest)
            {
                continue;
            }

            auto const producerIndex = config.stageIndex(ref.stageId);
            ELLM_CHECK(producerIndex >= 0, where + ": 'from' names unknown stage '" + ref.stageId + "'");
            auto const& producer = config.stages[static_cast<std::size_t>(producerIndex)];
            auto const* port = producer.findPort(ref.portId);
            ELLM_CHECK(
                port != nullptr, where + ": stage '" + ref.stageId + "' does not emit port '" + ref.portId + "'");

            // Only chunk_stream may cross lanes, which rules out handing a device tensor to a
            // consumer on another stream.
            ELLM_CHECK(producer.streamLane == stage.streamLane || port->transport == Transport::kChunkStream,
                where + ": edge from '" + input.from + "' crosses lanes (lane " + std::to_string(producer.streamLane)
                    + " -> lane " + std::to_string(stage.streamLane) + "), which transport " + toString(port->transport)
                    + " cannot do; use " + toString(Transport::kChunkStream));

            edges.emplace_back(producerIndex, static_cast<int32_t>(i));
        }
    }

    // A request is served only through a sink on a terminal, so a topology without one can
    // load but never answer anything.
    ELLM_CHECK(!config.terminals.empty(), "multistage config: 'terminals' must list at least one port");
    for (auto const& terminal : config.terminals)
    {
        std::string const where = "terminal '" + terminal + "'";
        auto const ref = parseRef(terminal, where);
        ELLM_CHECK(!ref.fromRequest, where + R"(: a terminal must be a port id "<stageId>:<portId>")");
        auto const* producer = config.findStage(ref.stageId);
        ELLM_CHECK(producer != nullptr, where + ": unknown stage '" + ref.stageId + "'");
        auto const* port = producer->findPort(ref.portId);
        ELLM_CHECK(port != nullptr, where + ": stage '" + ref.stageId + "' does not emit port '" + ref.portId + "'");
        ELLM_CHECK(port->terminal, where + ": port is not declared terminal, so no sink may attach to it");
        ELLM_CHECK(port->kind == PortKind::kText || port->kind == PortKind::kAudio || port->kind == PortKind::kImage,
            where + ": kind " + toString(port->kind) + " cannot carry a sink; only text, audio, and image can");
    }
    // The executor hands a sink to every port flagged terminal, so the flag and the list must agree.
    for (auto const& stage : config.stages)
    {
        for (auto const& port : stage.emits)
        {
            if (!port.terminal)
            {
                continue;
            }
            std::string const id = stage.id + ":" + port.id;
            ELLM_CHECK(std::find(config.terminals.begin(), config.terminals.end(), id) != config.terminals.end(),
                "port '" + id + "' is declared terminal but missing from 'terminals'");
        }
    }

    // A stage reading its own port is a one-node cycle, so it is kept here rather than
    // filtered out: Route::resolve would otherwise be the first thing to notice, per request.
    std::vector<std::vector<int32_t>> adjacency(stageCount);
    for (auto const& edge : edges)
    {
        adjacency[static_cast<std::size_t>(edge.first)].push_back(edge.second);
    }

    std::vector<uint8_t> color(stageCount, 0);
    std::vector<int32_t> path;
    std::vector<int32_t> cycle;
    for (std::size_t i = 0; i < stageCount; ++i)
    {
        if (color[i] != 0)
        {
            continue;
        }
        if (findCycle(adjacency, static_cast<int32_t>(i), color, path, cycle))
        {
            std::string trace;
            for (auto const node : cycle)
            {
                trace += (trace.empty() ? "" : " -> ") + config.stages[static_cast<std::size_t>(node)].id;
            }
            ELLM_CHECK(false, "stage topology must be a DAG, but found a cycle: " + trace);
        }
    }

    for (auto const& stage : config.stages)
    {
        if (stage.artifact.empty())
        {
            continue;
        }
        std::string const where = "stage '" + stage.id + "' artifact '" + stage.artifact + "'";
        std::filesystem::path const artifact(stage.artifact);
        ELLM_CHECK(artifact.is_relative(), where + ": must be relative to the config root");
        for (auto const& part : artifact)
        {
            ELLM_CHECK(part != "..", where + ": must not contain '..'");
        }
        if (!configRoot.empty())
        {
            auto const root = std::filesystem::path(configRoot).lexically_normal();
            auto const inside = (root / artifact).lexically_normal().lexically_relative(root);
            ELLM_CHECK(!inside.empty() && *inside.begin() != "..",
                where + ": resolves outside the config root '" + configRoot + "'");
        }
    }
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
