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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

namespace
{

//! One input with its `from` resolved once, so the three passes never re-parse it.
struct ParsedInput
{
    SourceRef ref;
    int32_t producer{-1};          //!< -1 when the value comes from the request.
    PortSpec const* port{nullptr}; //!< The producer's port, null for a request source.
};

using ParsedInputs = std::vector<std::vector<ParsedInput>>;

//! The input that made the forward pass drop a stage, kept so the error can name the root cause.
struct Blocker
{
    std::string inputName;
    std::string from;
    int32_t producer{-1};
};

std::string const& stageName(MultiStageConfig const& config, int32_t index)
{
    return config.stages[static_cast<size_t>(index)].id;
}

bool parseInputs(MultiStageConfig const& config, ParsedInputs& out, std::string& error)
{
    out.resize(config.stages.size());
    for (size_t i = 0; i < config.stages.size(); ++i)
    {
        auto const& stage = config.stages[i];
        out[i].reserve(stage.inputs.size());
        for (auto const& input : stage.inputs)
        {
            std::string const where = "Route: stage '" + stage.id + "' input '" + input.name + "'";
            ParsedInput parsed;
            try
            {
                parsed.ref = parseSourceRef(input.from);
            }
            catch (std::exception const& parseError)
            {
                error = where + " has an unparsable from: " + parseError.what();
                return false;
            }

            if (!parsed.ref.fromRequest)
            {
                parsed.producer = config.stageIndex(parsed.ref.stageId);
                if (parsed.producer < 0)
                {
                    error = where + " names unknown stage '" + parsed.ref.stageId + "'";
                    return false;
                }
                parsed.port = config.stages[static_cast<size_t>(parsed.producer)].findPort(parsed.ref.portId);
                if (parsed.port == nullptr)
                {
                    error = where + " names unknown port '" + parsed.ref.portId + "' on stage '" + parsed.ref.stageId
                        + "'";
                    return false;
                }
            }
            out[i].push_back(std::move(parsed));
        }
    }
    return true;
}

//! Kahn, seeded and pushed in config order so the result is deterministic.
bool topoSort(
    MultiStageConfig const& config, ParsedInputs const& parsed, std::vector<int32_t>& order, std::string& error)
{
    size_t const count = config.stages.size();
    std::vector<std::vector<int32_t>> successors(count);
    std::vector<int32_t> inDegree(count, 0);

    for (size_t i = 0; i < count; ++i)
    {
        for (auto const& input : parsed[i])
        {
            if (input.producer < 0)
            {
                continue;
            }
            successors[static_cast<size_t>(input.producer)].push_back(static_cast<int32_t>(i));
            ++inDegree[i];
        }
    }

    order.clear();
    order.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        if (inDegree[i] == 0)
        {
            order.push_back(static_cast<int32_t>(i));
        }
    }
    for (size_t head = 0; head < order.size(); ++head)
    {
        for (int32_t next : successors[static_cast<size_t>(order[head])])
        {
            if (--inDegree[static_cast<size_t>(next)] == 0)
            {
                order.push_back(next);
            }
        }
    }

    if (order.size() != count)
    {
        auto const stuck = std::find_if(inDegree.begin(), inDegree.end(), [](int32_t degree) { return degree > 0; });
        error = "Route: stage '" + config.stages[static_cast<size_t>(stuck - inDegree.begin())].id
            + "' sits on a cycle; the topology must be a DAG";
        return false;
    }
    return true;
}

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void hashByte(uint64_t& hash, uint8_t byte)
{
    hash = (hash ^ byte) * kFnvPrime;
}

//! Terminated, so that "ab" + "c" and "a" + "bc" do not collide.
void hashString(uint64_t& hash, std::string const& value)
{
    for (char c : value)
    {
        hashByte(hash, static_cast<uint8_t>(c));
    }
    hashByte(hash, 0);
}

} // namespace

bool RequestCapabilities::supplies(std::string const& requestKey) const
{
    return std::find(suppliedRequestKeys.begin(), suppliedRequestKeys.end(), requestKey) != suppliedRequestKeys.end();
}

bool Route::isActive(int32_t stageIndex) const noexcept
{
    return stageIndex >= 0 && static_cast<size_t>(stageIndex) < mActive.size()
        && mActive[static_cast<size_t>(stageIndex)];
}

bool Route::resolve(MultiStageConfig const& config, std::vector<std::string> const& wanted,
    RequestCapabilities const& available, Route& out, std::string& error)
{
    out = Route{};
    error.clear();

    if (wanted.empty())
    {
        error = "Route: no sink attached; nothing is wanted, so no stage can be required";
        return false;
    }

    size_t const count = config.stages.size();
    ParsedInputs parsed;
    std::vector<int32_t> topo;
    if (!parseInputs(config, parsed, error) || !topoSort(config, parsed, topo, error))
    {
        return false;
    }

    // (1) Forward: available, one topological pass. A missing kFail source only marks the
    // stage unavailable here; it is reported in (3), once the active set is known.
    std::vector<bool> availableStage(count, true);
    std::vector<Blocker> blocker(count);
    std::vector<std::pair<int32_t, std::string>> skipped;
    for (int32_t index : topo)
    {
        auto const& stage = config.stages[static_cast<size_t>(index)];
        for (size_t k = 0; k < stage.inputs.size(); ++k)
        {
            auto const& input = stage.inputs[k];
            auto const& source = parsed[static_cast<size_t>(index)][k];
            bool const sourceOk = source.producer < 0 ? available.supplies(input.from)
                                                      : availableStage[static_cast<size_t>(source.producer)];
            if (sourceOk)
            {
                continue;
            }
            if (input.onMissing == OnMissing::kSkip)
            {
                skipped.emplace_back(index, input.name);
            }
            else if (availableStage[static_cast<size_t>(index)])
            {
                availableStage[static_cast<size_t>(index)] = false;
                blocker[static_cast<size_t>(index)] = Blocker{input.name, input.from, source.producer};
            }
        }
    }

    // (2) Backward: required, from the terminals a sink is attached to. Every wanted port is
    // validated on the way in.
    std::vector<bool> requiredStage(count, false);
    std::vector<int32_t> pending;
    std::vector<int32_t> wantedProducers;
    wantedProducers.reserve(wanted.size());
    for (auto const& want : wanted)
    {
        SourceRef ref;
        try
        {
            ref = parseSourceRef(want);
        }
        catch (std::exception const& parseError)
        {
            error = "Route: wanted '" + want + "' is not a '<stage>:<port>' id: " + parseError.what();
            return false;
        }
        if (ref.fromRequest)
        {
            error = "Route: wanted '" + want + "' names a request field, not a terminal port";
            return false;
        }
        int32_t const index = config.stageIndex(ref.stageId);
        if (index < 0)
        {
            error = "Route: wanted '" + want + "' names unknown stage '" + ref.stageId + "'";
            return false;
        }
        auto const* port = config.stages[static_cast<size_t>(index)].findPort(ref.portId);
        if (port == nullptr)
        {
            error
                = "Route: wanted '" + want + "' names unknown port '" + ref.portId + "' on stage '" + ref.stageId + "'";
            return false;
        }
        if (!port->terminal)
        {
            error = "Route: wanted '" + want + "' is not a terminal port, so no sink may attach to it";
            return false;
        }
        wantedProducers.push_back(index);
        if (!requiredStage[static_cast<size_t>(index)])
        {
            requiredStage[static_cast<size_t>(index)] = true;
            pending.push_back(index);
        }
    }

    while (!pending.empty())
    {
        int32_t const index = pending.back();
        pending.pop_back();
        if (!availableStage[static_cast<size_t>(index)])
        {
            continue; // It will not run, so it requires nothing.
        }
        for (auto const& source : parsed[static_cast<size_t>(index)])
        {
            // A source that is gone can only be a kSkip one -- a missing kFail or kPrune source
            // would have made this stage unavailable -- and the stage runs without it. A source
            // that survived is required: a request that supplied audio must not be answered
            // without the encoder that consumed it.
            if (source.producer < 0 || !availableStage[static_cast<size_t>(source.producer)]
                || requiredStage[static_cast<size_t>(source.producer)])
            {
                continue;
            }
            requiredStage[static_cast<size_t>(source.producer)] = true;
            pending.push_back(source.producer);
        }
    }

    // (3) Active = available AND required.
    out.mActive.assign(count, false);
    for (size_t i = 0; i < count; ++i)
    {
        out.mActive[i] = availableStage[i] && requiredStage[i];
    }

    for (size_t w = 0; w < wanted.size(); ++w)
    {
        int32_t const index = wantedProducers[w];
        if (out.mActive[static_cast<size_t>(index)])
        {
            continue;
        }
        // Walk to the stage that actually lost its source; blaming the sink's own producer
        // would send the reader up the topology by hand.
        int32_t root = index;
        for (size_t hop = 0; hop < count; ++hop)
        {
            int32_t const upstream = blocker[static_cast<size_t>(root)].producer;
            if (upstream < 0 || availableStage[static_cast<size_t>(upstream)])
            {
                break;
            }
            root = upstream;
        }
        auto const& cause = blocker[static_cast<size_t>(root)];
        error = "Route: sink '" + wanted[w] + "' cannot be produced; stage '" + stageName(config, root) + "' input '"
            + cause.inputName + "' has no source '" + cause.from + "'";
        return false;
    }

    // The backward pass marks every surviving source of an active stage, so this fires only if
    // one of the rules above changes. It is the rule the config author is promised, so check it.
    for (size_t i = 0; i < count; ++i)
    {
        if (!out.mActive[i])
        {
            continue;
        }
        auto const& stage = config.stages[i];
        for (size_t k = 0; k < stage.inputs.size(); ++k)
        {
            auto const& input = stage.inputs[k];
            auto const& source = parsed[i][k];
            if (input.onMissing != OnMissing::kFail)
            {
                continue;
            }
            bool const sourceOk = source.producer < 0 ? available.supplies(input.from)
                                                      : out.mActive[static_cast<size_t>(source.producer)];
            if (!sourceOk)
            {
                error = "Route: stage '" + stage.id + "' input '" + input.name + "' is on_missing=fail but its source '"
                    + input.from + "' is not active";
                return false;
            }
        }
    }

    for (auto const& entry : skipped)
    {
        if (out.mActive[static_cast<size_t>(entry.first)])
        {
            out.mSkipped.push_back(entry);
        }
    }

    for (int32_t index : topo)
    {
        if (out.mActive[static_cast<size_t>(index)])
        {
            out.mOrder.push_back(index);
        }
    }

    for (int32_t index : topo)
    {
        if (!out.mActive[static_cast<size_t>(index)])
        {
            continue;
        }
        auto const& consumer = config.stages[static_cast<size_t>(index)];
        for (size_t k = 0; k < consumer.inputs.size(); ++k)
        {
            auto const& input = consumer.inputs[k];
            auto const& source = parsed[static_cast<size_t>(index)][k];
            RouteEdge edge;
            if (source.producer < 0)
            {
                if (!available.supplies(input.from))
                {
                    continue;
                }
                edge.port = source.ref.requestField;
            }
            else
            {
                if (!out.mActive[static_cast<size_t>(source.producer)])
                {
                    continue;
                }
                auto const& producer = config.stages[static_cast<size_t>(source.producer)];
                edge.producer = source.producer;
                edge.port = source.ref.portId;
                edge.transport = source.port->transport;
                edge.crossLane = producer.streamLane != consumer.streamLane;
            }
            edge.consumer = index;
            edge.inputName = input.name;
            out.mEdges.push_back(std::move(edge));
        }
    }

    // Topology only: two requests differing just in ids or payload must land on one batch.
    uint64_t hash = kFnvOffset;
    for (int32_t index : topo)
    {
        if (out.mActive[static_cast<size_t>(index)])
        {
            hashString(hash, stageName(config, index));
        }
    }
    for (auto const& edge : out.mEdges)
    {
        hashString(hash, edge.producer < 0 ? std::string("request") : stageName(config, edge.producer));
        hashString(hash, edge.port);
        hashString(hash, stageName(config, edge.consumer));
        hashString(hash, edge.inputName);
        hashByte(hash, static_cast<uint8_t>(edge.transport));
    }
    out.mFingerprint = hash;

    return true;
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
