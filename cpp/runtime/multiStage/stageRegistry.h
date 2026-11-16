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

#include "runtime/multiStage/stage.h"
#include "runtime/multiStage/stageConfig.h"

#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

/*!
 * @brief What a runner promises, independent of any one deployment.
 *
 * A RunnerDecl is the implementation's half of the contract; a StageConfig is
 * the deployment's half. validateRunners() checks one against the other, so a
 * typo in multistage_config.json fails at load naming the rule it broke instead
 * of surfacing as a missing binding half way through a request.
 */
struct RunnerDecl
{
    std::string name; //!< Table key, e.g. "<family>.talker". Matches StageConfig::runner.
    StageKind kind{StageKind::kHost};

    //! Ports this runner can produce. A deployment may emit a subset of them.
    std::vector<std::pair<std::string, PortKind>> ports;
    //! Input names this runner understands. A deployment may bind a subset of them.
    std::vector<std::string> inputs;
    //! Keys that must appear in StageConfig::config. Presence only -- type and range stay
    //! with the runner, the only thing that knows what they mean.
    std::vector<std::string> requiredConfigKeys;

    //! Whether this runner can hold more than one request at a time. A deployment may declare
    //! a wider max_batch than the implementation has; the executor caps the cohort at what the
    //! runner promises rather than handing it members it would step on top of each other.
    bool batches{false};

    // What a front end that reads request files needs to know; the executor ignores these.
    //! Keys this runner reads from a JSON object bound to its `sampling` input.
    std::vector<std::string> samplingKeys;
    //! Fields of its `style` input that a request file may also set at its top level or per request.
    std::vector<std::string> styleFields;
    //! Other front ends' request-file names for a sampling key or an input name, as {spelling, name}.
    std::vector<std::pair<std::string, std::string>> requestAliases;
};

//! One runner: its declaration and the factory that builds its stage.
struct RunnerEntry
{
    RunnerDecl decl;
    StageFactory create{nullptr};
};

/*!
 * @brief Runner name to declaration and factory.
 *
 * builtinRunners() is the table of the runners this library ships. A caller that adds its own
 * runner copies it, add()s a row and passes the copy to MultiStageRuntime::load().
 */
class StageRegistry
{
public:
    StageRegistry() = default;
    //! @throws std::runtime_error as add() does for any entry.
    StageRegistry(std::initializer_list<RunnerEntry> entries);

    //! @throws std::runtime_error when the name is empty or already taken, or the entry has no factory.
    void add(RunnerEntry entry);

    //! nullptr when unknown. The pointee lives as long as the table.
    RunnerEntry const* find(std::string const& name) const;

    //! Sorted, so an "unknown runner" message can list every legal value.
    std::vector<std::string> names() const;

private:
    std::map<std::string, RunnerEntry> mRunners;
};

//! The runners this library ships, one row each; defined in stageFactories.cpp.
StageRegistry const& builtinRunners();

/*!
 * @brief The half of load-time validation that needs the runners.
 *
 * Split from multiStage::validate() on purpose: validate() is pure structure with no
 * dependency on any implementation. This one answers a different question -- does this
 * config match what the runners in @p runners actually offer.
 *
 * Checks, per stage: the runner is in the table; StageKind agrees with the
 * declaration; every emitted port is declared with the same PortKind; every
 * bound input name is accepted; every required config key is present. A port
 * the runner declares but the config omits is fine -- not every deployment
 * wants every output.
 *
 * @throws std::runtime_error naming the stage id and the rule.
 */
void validateRunners(MultiStageConfig const& config, StageRegistry const& runners);

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
