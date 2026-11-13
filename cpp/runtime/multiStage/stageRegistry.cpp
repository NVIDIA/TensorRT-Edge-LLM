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

#include "common/checkMacros.h"

#include <algorithm>
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

std::string quoteJoin(std::vector<std::string> const& values)
{
    if (values.empty())
    {
        return "<none>";
    }
    std::string out;
    for (auto const& value : values)
    {
        if (!out.empty())
        {
            out += ", ";
        }
        out += "'" + value + "'";
    }
    return out;
}

std::vector<std::string> portIds(RunnerDecl const& decl)
{
    std::vector<std::string> ids;
    ids.reserve(decl.ports.size());
    for (auto const& port : decl.ports)
    {
        ids.push_back(port.first);
    }
    return ids;
}

std::pair<std::string, PortKind> const* declaredPort(RunnerDecl const& decl, std::string const& portId)
{
    auto const it = std::find_if(decl.ports.begin(), decl.ports.end(),
        [&portId](std::pair<std::string, PortKind> const& port) { return port.first == portId; });
    return it == decl.ports.end() ? nullptr : &(*it);
}

std::string where(StageConfig const& stage)
{
    return "stage '" + stage.id + "' (runner '" + stage.runner + "'): ";
}

} // namespace

StageRegistry::StageRegistry(std::initializer_list<RunnerEntry> entries)
{
    for (auto const& entry : entries)
    {
        add(entry);
    }
}

void StageRegistry::add(RunnerEntry entry)
{
    ELLM_CHECK(!entry.decl.name.empty(), "runner declaration has an empty name");
    ELLM_CHECK(entry.create != nullptr, "runner '" + entry.decl.name + "' has no factory");
    // Copy the key first; entry is about to be moved from.
    std::string const name = entry.decl.name;
    bool const inserted = mRunners.emplace(name, std::move(entry)).second;
    ELLM_CHECK(inserted, "runner '" + name + "' is registered twice");
}

RunnerEntry const* StageRegistry::find(std::string const& name) const
{
    auto const it = mRunners.find(name);
    return it == mRunners.end() ? nullptr : &it->second;
}

std::vector<std::string> StageRegistry::names() const
{
    // std::map iterates in key order, which is the sorting the header promises.
    std::vector<std::string> out;
    out.reserve(mRunners.size());
    for (auto const& entry : mRunners)
    {
        out.push_back(entry.first);
    }
    return out;
}

void validateRunners(MultiStageConfig const& config, StageRegistry const& runners)
{
    for (auto const& stage : config.stages)
    {
        auto const* entry = runners.find(stage.runner);
        ELLM_CHECK(entry != nullptr,
            where(stage) + "runner is not registered; registered runners are " + quoteJoin(runners.names()));
        auto const* decl = &entry->decl;

        ELLM_CHECK(stage.kind == decl->kind,
            where(stage) + "kind is '" + toString(stage.kind) + "' but the runner declares '" + toString(decl->kind)
                + "'");

        for (auto const& port : stage.emits)
        {
            auto const* declared = declaredPort(*decl, port.id);
            ELLM_CHECK(declared != nullptr,
                where(stage) + "emits port '" + port.id + "', which the runner does not declare; declared ports are "
                    + quoteJoin(portIds(*decl)));
            ELLM_CHECK(port.kind == declared->second,
                where(stage) + "port '" + port.id + "' is '" + toString(port.kind) + "' in the config but '"
                    + toString(declared->second) + "' in the runner declaration");
        }

        for (auto const& input : stage.inputs)
        {
            bool const accepted = std::find(decl->inputs.begin(), decl->inputs.end(), input.name) != decl->inputs.end();
            ELLM_CHECK(accepted,
                where(stage) + "binds input '" + input.name
                    + "', which the runner does not accept; accepted inputs are " + quoteJoin(decl->inputs));
        }

        for (auto const& key : decl->requiredConfigKeys)
        {
            ELLM_CHECK(stage.config.contains(key),
                where(stage) + "config is missing required key '" + key + "'; the runner requires "
                    + quoteJoin(decl->requiredConfigKeys));
        }
    }
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
