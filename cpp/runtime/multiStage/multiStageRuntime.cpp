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

#include "runtime/multiStage/multiStageRuntime.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/multiStage/route.h"
#include "runtime/multiStage/stageRegistry.h"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <exception>
#include <filesystem>
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

namespace
{

std::string where(StageConfig const& stage)
{
    return "stage '" + stage.id + "': ";
}

std::string resolveArtifactDir(std::string const& configRoot, StageConfig const& stage)
{
    if (stage.artifact.empty())
    {
        return {};
    }
    auto const dir = (std::filesystem::path(configRoot) / stage.artifact).lexically_normal();
    ELLM_CHECK(
        std::filesystem::is_directory(dir), where(stage) + "artifact directory '" + dir.string() + "' does not exist");
    return dir.string();
}

//! Qualified request keys the resolved route will read. Anything a request carries outside
//! this set was carried for nothing.
std::vector<std::string> readRequestKeys(Route const& route)
{
    std::vector<std::string> keys;
    for (auto const& edge : route.edges())
    {
        if (edge.producer < 0)
        {
            keys.push_back("request:" + edge.port);
        }
    }
    return keys;
}

//! Whether any stage in the topology names @p requestKey, whether or not it is on this route.
//! This is what separates "this deployment cannot do that" from "you asked for a subset".
bool anyStageReads(MultiStageConfig const& config, std::string const& requestKey)
{
    for (auto const& stage : config.stages)
    {
        for (auto const& input : stage.inputs)
        {
            auto const ref = parseSourceRef(input.from);
            if (ref.fromRequest && "request:" + ref.requestField == requestKey)
            {
                return true;
            }
        }
    }
    return false;
}

std::string joinQuoted(std::vector<std::string> const& keys)
{
    std::string out;
    for (auto const& key : keys)
    {
        out += (out.empty() ? "'" : ", '") + key + "'";
    }
    return out;
}

//! Bind the caller's payload to the inputs that read it. Request-sourced edges carry the
//! bare field in `port`; the payload is keyed by the qualified name the config spells.
void publishRequestInputs(RequestSlot const& slot, std::vector<PortBindings>& bindings)
{
    for (auto const& edge : slot.route.edges())
    {
        if (edge.producer >= 0)
        {
            continue;
        }
        auto const* value = slot.payload.find("request:" + edge.port);
        if (value != nullptr)
        {
            bindings[static_cast<size_t>(edge.consumer)].bind(edge.inputName, *value);
        }
    }
}

/*!
 * @brief Refuse a topology whose data edges cross a lane boundary.
 *
 * Two stages on different streams need an inter-stream handshake before the consumer
 * may read the producer's buffer, and no transport implements one, so every data edge
 * must stay inside one lane.
 */
void rejectCrossLaneEdges(MultiStageConfig const& config)
{
    for (auto const& stage : config.stages)
    {
        for (auto const& input : stage.inputs)
        {
            SourceRef const ref = parseSourceRef(input.from);
            if (ref.fromRequest)
            {
                continue;
            }
            auto const* producer = config.findStage(ref.stageId);
            ELLM_CHECK(producer != nullptr, "stage '" + stage.id + "': unknown source stage '" + ref.stageId + "'");
            ELLM_CHECK(producer->streamLane == stage.streamLane,
                "stage '" + stage.id + "' input '" + input.name + "' crosses a lane boundary ("
                    + std::to_string(producer->streamLane) + " -> " + std::to_string(stage.streamLane)
                    + "); cross-lane data transport is not implemented");
        }
    }
}

//! Bind a producer's port values to the inputs that consume them.
void publishOutputs(int32_t producer, Route const& route, std::vector<PortBindings>& bindings)
{
    for (auto const& edge : route.edges())
    {
        if (edge.producer != producer)
        {
            continue;
        }
        ELLM_CHECK(!edge.crossLane, "cross-lane edge reached publishOutputs; load() should have rejected it");
        auto const* value = bindings[static_cast<size_t>(producer)].find(edge.port);
        if (value != nullptr)
        {
            bindings[static_cast<size_t>(edge.consumer)].bind(edge.inputName, *value);
        }
    }
}

//! Only built on the stall path, where one more string costs nothing.
std::string blockedStages(
    MultiStageConfig const& config, std::vector<int32_t> const& order, std::vector<StageStatus> const& status)
{
    std::string names;
    for (auto const stageIndex : order)
    {
        auto const index = static_cast<size_t>(stageIndex);
        if (status[index] == StageStatus::kDone)
        {
            continue;
        }
        names += (names.empty() ? "" : ", ") + ("'" + config.stages[index].id + "'");
    }
    return names;
}

} // namespace

namespace
{

//! The width a stage is built and admitted at: max_batch when its runner batches, 1 otherwise.
int32_t stageWidth(StageConfig const& stage, RunnerDecl const& decl) noexcept
{
    return decl.batches ? std::max(stage.maxBatch, 1) : 1;
}

} // namespace

int32_t MultiStageRuntime::cohortLimit(Route const& route) const noexcept
{
    // route.order() holds only active stages, so a text-only request is not capped by the vocoder's width.
    int32_t width = std::numeric_limits<int32_t>::max();
    for (auto const stageIndex : route.order())
    {
        width = std::min(width, mStageWidths[static_cast<size_t>(stageIndex)]);
    }
    return std::max(width, 1);
}

/*!
 * @brief One binding table per stage, handed to it as both @c in and @c out.
 *
 * A producer writes its outputs under the port id; the executor then copies them into
 * the consumer's table under the consumer's input name. Two key spaces, one table,
 * because a seat owns exactly one PortBindings per stage.
 *
 * A stage receives a sink only for a terminal port it emits itself, so it learns whether
 * it has an exit without learning anything about the topology around it.
 */
StageExecContext MultiStageRuntime::contextFor(Seat& seat, int32_t stageIndex)
{
    auto const index = static_cast<size_t>(stageIndex);
    auto const& stage = mConfig.stages[index];

    StageExecContext ctx;
    ctx.slot = seat.slot.get();
    ctx.in = &seat.bindings[index];
    ctx.out = &seat.bindings[index];
    ctx.stream = mLanes[static_cast<size_t>(stage.streamLane)].stream;

    for (auto const& port : stage.emits)
    {
        if (!port.terminal)
        {
            continue;
        }
        if (port.kind == PortKind::kAudio)
        {
            ctx.audioSink = seat.sinks.audio;
        }
        else if (port.kind == PortKind::kImage)
        {
            ctx.videoSink = seat.sinks.video;
        }
        else
        {
            ctx.textSink = seat.sinks.text;
        }
    }
    return ctx;
}

bool MultiStageRuntime::inputsReady(int32_t stageIndex, Seat const& seat) const
{
    auto const& bindings = seat.bindings[static_cast<size_t>(stageIndex)];
    for (auto const& edge : seat.slot->route.edges())
    {
        if (edge.consumer != stageIndex)
        {
            continue;
        }
        if (bindings.find(edge.inputName) == nullptr)
        {
            return false;
        }
    }
    return true;
}

void MultiStageRuntime::destroyLanes() noexcept
{
    for (auto& lane : mLanes)
    {
        if (lane.stream != nullptr)
        {
            cudaStreamDestroy(lane.stream);
        }
    }
    mLanes.clear();
}

MultiStageRuntime::MultiStageRuntime() = default;

MultiStageRuntime::~MultiStageRuntime()
{
    // Reverse of load(). Stages record work on their lane's stream, so the streams have to
    // outlive them.
    mSeats.clear();
    mStages.clear();
    destroyLanes();
}

void MultiStageRuntime::checkLoadable(MultiStageConfig const& config, StageRegistry const& runners)
{
    validateRunners(config, runners);
    rejectCrossLaneEdges(config);
}

void MultiStageRuntime::load(std::string const& configRoot, SharedResources& shared, StageRegistry const* runners)
{
    load(MultiStageConfig::load(configRoot), configRoot, shared, runners);
}

void MultiStageRuntime::load(
    MultiStageConfig config, std::string const& configRoot, SharedResources& shared, StageRegistry const* runners)
{
    ELLM_CHECK(mStages.empty(), "MultiStageRuntime::load called twice");

    mConfig = std::move(config);
    validate(mConfig, configRoot);
    auto const& table = runners != nullptr ? *runners : builtinRunners();
    checkLoadable(mConfig, table);
    mConfigRoot = configRoot;
    // validateRunners() found every stage's runner in the table.
    mStageWidths.clear();
    for (auto const& stage : mConfig.stages)
    {
        mStageWidths.push_back(stageWidth(stage, table.find(stage.runner)->decl));
    }

    try
    {
        int32_t lanes = 1;
        for (auto const& stage : mConfig.stages)
        {
            lanes = std::max(lanes, stage.streamLane + 1);
        }
        mLanes.resize(static_cast<size_t>(lanes));
        for (auto& lane : mLanes)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking));
        }

        mStages.resize(mConfig.stages.size());
        mSeats.clear();

        // Declaration order. Nothing at init may reach a sibling stage, so the topological
        // order Route computes per request buys nothing here.
        for (size_t i = 0; i < mConfig.stages.size(); ++i)
        {
            auto const& stage = mConfig.stages[i];
            try
            {
                StageInitContext init;
                init.config = &stage;
                init.artifactDir = resolveArtifactDir(configRoot, stage);
                init.shared = &shared;
                init.stream = mLanes[static_cast<size_t>(stage.streamLane)].stream;
                // Not stage.maxBatch: a runner that does not batch is never handed a second member.
                init.maxBatch = mStageWidths[i];
                init.maxSeq = stage.maxSeq;

                mStages[i] = table.find(stage.runner)->create(init);
                ELLM_CHECK(mStages[i] != nullptr, "runner '" + stage.runner + "' built no stage");
                mStages[i]->preallocate();
            }
            catch (std::exception const& error)
            {
                ELLM_CHECK(false, where(stage) + error.what());
            }
        }

        LOG_INFO("multistage topology '%s': %zu stages on %d lane(s)", mConfig.label.c_str(), mStages.size(), lanes);
    }
    catch (...)
    {
        // Nothing half-built escapes: drop the stages, then the streams.
        mStages.clear();
        mStageWidths.clear();
        mSeats.clear();
        destroyLanes();
        mConfig = MultiStageConfig{};
        mConfigRoot.clear();
        throw;
    }
}

RequestSlot* MultiStageRuntime::submit(PortBindings const& payload, Sinks const& sinks, std::string& error)
{
    error.clear();
    if (mStages.empty())
    {
        error = "MultiStageRuntime::submit before load()";
        return nullptr;
    }

    try
    {
        auto const wanted = sinks.attachedTerminals(mConfig);
        if (wanted.empty())
        {
            error = "no sink attached; a request with no reachable terminal has nothing to produce";
            return nullptr;
        }

        RequestCapabilities capabilities;
        capabilities.suppliedRequestKeys = payload.names();

        Route route;
        if (!Route::resolve(mConfig, wanted, capabilities, route, error))
        {
            return nullptr;
        }

        // A carried-but-unread input is the quietest way to get a wrong answer: the request
        // succeeds, the modality simply never happened. Refuse what this deployment cannot
        // serve; only warn when the field exists but the caller's sink choice pruned it.
        {
            auto const read = readRequestKeys(route);
            std::vector<std::string> unserved;
            std::vector<std::string> pruned;
            for (auto const& key : capabilities.suppliedRequestKeys)
            {
                if (std::find(read.begin(), read.end(), key) != read.end())
                {
                    continue;
                }
                (anyStageReads(mConfig, key) ? pruned : unserved).push_back(key);
            }
            if (!unserved.empty())
            {
                error = "no stage in topology '" + mConfig.label + "' reads " + joinQuoted(unserved)
                    + "; this deployment cannot serve that input";
                return nullptr;
            }
            if (!pruned.empty())
            {
                LOG_WARNING("request carries %s, which only stages outside the resolved route read; ignored",
                    joinQuoted(pruned).c_str());
            }
        }

        if (static_cast<int32_t>(mSeats.size()) >= cohortLimit(route))
        {
            error = "cohort is full at " + std::to_string(mSeats.size())
                + " requests; the narrowest stage on this route is that wide";
            return nullptr;
        }

        // Route::fingerprint() is the batching key: members that resolved different subgraphs
        // would each need their own stage order, and a stage would see a cohort whose members
        // do not all want the step it is about to take.
        if (!mSeats.empty() && route.fingerprint() != mSeats.front().slot->route.fingerprint())
        {
            error
                = "this request resolves a different route than the cohort already admitted; "
                  "only requests with the same active subgraph may share a batch";
            return nullptr;
        }

        Seat seat;
        seat.bindings.resize(mConfig.stages.size());
        seat.sinks = sinks;
        seat.slot = std::make_unique<RequestSlot>();
        seat.slot->requestId = mNextRequestId;
        seat.slot->batchSlot = static_cast<int32_t>(mSeats.size());
        seat.slot->route = std::move(route);
        seat.slot->payload = payload;

        // Before admission, so a stage may reject on what the request actually carries
        // rather than discovering it one prepare() too late.
        publishRequestInputs(*seat.slot, seat.bindings);

        for (auto const stageIndex : seat.slot->route.order())
        {
            auto const index = static_cast<size_t>(stageIndex);
            auto const& stage = mConfig.stages[index];
            auto ctx = contextFor(seat, stageIndex);
            std::string reason;
            if (!mStages[index]->admit(ctx, reason))
            {
                // admit() does no GPU work, and the seat dies with its bindings, so nothing
                // has to be unwound before the caller sees the refusal.
                error = where(stage) + "refused the request: " + reason;
                return nullptr;
            }
        }

        // The route is frozen from here: the step loop is its only reader.
        mSeats.push_back(std::move(seat));
        ++mNextRequestId;
        return mSeats.back().slot.get();
    }
    catch (std::exception const& admissionError)
    {
        error = admissionError.what();
        return nullptr;
    }
}

std::vector<Response> MultiStageRuntime::run()
{
    ELLM_CHECK(!mSeats.empty(), "run() with nothing admitted");

    auto const cohort = mSeats.size();
    auto const stageCount = mConfig.stages.size();
    // submit() proved every member resolved the same subgraph, so one order covers the cohort.
    auto const& order = mSeats.front().slot->route.order();

    std::vector<Response> responses(cohort);
    for (size_t member = 0; member < cohort; ++member)
    {
        responses[member].requestId = mSeats[member].slot->requestId;
    }

    // Per stage, per member. prepare() reads the stage's inputs, so it cannot run before the
    // producers of those inputs have published; deferring it to the first round in which the
    // member is ready is what lets a topology deeper than one stage run at all.
    std::vector<StageStatus> status(stageCount * cohort, StageStatus::kMore);
    std::vector<char> prepared(stageCount * cohort, 0);
    auto const at
        = [cohort](int32_t stageIndex, size_t member) { return static_cast<size_t>(stageIndex) * cohort + member; };

    std::vector<FinishReason> reasons(cohort, FinishReason::kCompleted);
    // Cleared once a member is cancelled or fails: it stops being stepped, but the others
    // keep going, and finishAll() still runs for every seat.
    std::vector<char> live(cohort, 1);

    // Reserved once so the per-round clear() and push_back() calls reuse their storage.
    std::vector<size_t> members;
    std::vector<StageExecContext> contexts;
    std::vector<StageExecContext*> handles;
    std::vector<StageStatus> results;
    members.reserve(cohort);
    contexts.reserve(cohort);
    handles.reserve(cohort);
    results.reserve(cohort);

    auto const retire = [&](size_t member, FinishReason reason) {
        live[member] = 0;
        reasons[member] = reason;
        for (auto const stageIndex : order)
        {
            status[at(stageIndex, member)] = StageStatus::kDone;
        }
    };

    try
    {
        int32_t iteration = 0;

        while (true)
        {
            bool progressed = false;

            // Topological order, so within one round every producer is stepped before the
            // consumers that read it. A consumer that still reports kBlocked has simply not
            // been given enough yet, and the next round is what advances its producer.
            for (auto const stageIndex : order)
            {
                auto const index = static_cast<size_t>(stageIndex);

                members.clear();
                contexts.clear();
                for (size_t member = 0; member < cohort; ++member)
                {
                    if (live[member] == 0)
                    {
                        continue;
                    }
                    if (mSeats[member].slot->abandon.requested())
                    {
                        retire(member, FinishReason::kCancelled);
                        continue;
                    }
                    if (status[at(stageIndex, member)] == StageStatus::kDone)
                    {
                        continue;
                    }
                    if (prepared[at(stageIndex, member)] == 0)
                    {
                        if (!inputsReady(stageIndex, mSeats[member]))
                        {
                            continue;
                        }
                        auto prepareCtx = contextFor(mSeats[member], stageIndex);
                        try
                        {
                            mStages[index]->prepare(prepareCtx);
                        }
                        catch (std::exception const& error)
                        {
                            responses[member].error = where(mConfig.stages[index]) + error.what();
                            retire(member, FinishReason::kFailed);
                            progressed = true;
                            continue;
                        }
                        catch (...)
                        {
                            responses[member].error = where(mConfig.stages[index]) + "unknown exception in prepare()";
                            retire(member, FinishReason::kFailed);
                            progressed = true;
                            continue;
                        }
                        publishOutputs(stageIndex, mSeats[member].slot->route, mSeats[member].bindings);
                        prepared[at(stageIndex, member)] = 1;
                        progressed = true;
                    }
                    members.push_back(member);
                }

                if (members.empty())
                {
                    continue;
                }

                // contexts is filled first and only then addressed, because push_back may
                // reallocate and every handle would point at freed storage.
                for (auto const member : members)
                {
                    contexts.push_back(contextFor(mSeats[member], stageIndex));
                }
                handles.clear();
                for (auto& ctx : contexts)
                {
                    handles.push_back(&ctx);
                }
                results.assign(members.size(), StageStatus::kFailed);
                try
                {
                    mStages[index]->stepCohort(handles, results);
                }
                catch (std::exception const& error)
                {
                    throw std::runtime_error(where(mConfig.stages[index]) + error.what());
                }
                catch (...)
                {
                    throw std::runtime_error(where(mConfig.stages[index]) + "unknown exception in stepCohort()");
                }

                for (size_t i = 0; i < members.size(); ++i)
                {
                    auto const member = members[i];
                    status[at(stageIndex, member)] = results[i];
                    if (results[i] == StageStatus::kFailed)
                    {
                        LOG_ERROR("%sstep failed for request %llu", where(mConfig.stages[index]).c_str(),
                            static_cast<unsigned long long>(mSeats[member].slot->requestId));
                        responses[member].error = where(mConfig.stages[index]) + "step failed";
                        retire(member, FinishReason::kFailed);
                        continue;
                    }
                    // Publish only after the whole cohort has stepped: a batched call writes
                    // every member's output, and a consumer must not read one mid-flight.
                    publishOutputs(stageIndex, mSeats[member].slot->route, mSeats[member].bindings);
                    if (results[i] != StageStatus::kBlocked)
                    {
                        progressed = true;
                    }
                    // A stage that sees the cancel inside its own step() returns normally with what
                    // it had, so without this read a cancel landing in the last stepped stage
                    // reads as kCompleted.
                    if (mSeats[member].slot->abandon.requested())
                    {
                        retire(member, FinishReason::kCancelled);
                    }
                }
            }

            bool anyLive = false;
            for (size_t member = 0; member < cohort; ++member)
            {
                if (live[member] == 0)
                {
                    continue;
                }
                bool done = true;
                for (auto const stageIndex : order)
                {
                    done = done && status[at(stageIndex, member)] == StageStatus::kDone;
                }
                // A finished member leaves the live set, so a cohort-mate's later throw or stall
                // cannot rewrite its outcome.
                if (done)
                {
                    live[member] = 0;
                    continue;
                }
                anyLive = true;
            }
            if (!anyLive)
            {
                break;
            }

            if (!progressed)
            {
                // Nothing prepared and nothing stepped, so the next round would repeat this
                // one exactly. Either a producer stopped short of what a consumer needs, or
                // an input no stage produces was left unbound.
                for (size_t member = 0; member < cohort; ++member)
                {
                    if (live[member] == 0)
                    {
                        continue;
                    }
                    std::vector<StageStatus> memberStatus(stageCount, StageStatus::kDone);
                    for (auto const stageIndex : order)
                    {
                        memberStatus[static_cast<size_t>(stageIndex)] = status[at(stageIndex, member)];
                    }
                    responses[member].error = "no stage made progress in iteration " + std::to_string(iteration)
                        + "; still blocked: " + blockedStages(mConfig, order, memberStatus);
                    retire(member, FinishReason::kFailed);
                }
                break;
            }
            ++iteration;
        }
    }
    catch (std::exception const& error)
    {
        // A throw cannot be pinned on one member: stepCohort() is one call for all of them.
        for (size_t member = 0; member < cohort; ++member)
        {
            if (live[member] != 0)
            {
                responses[member].error = error.what();
                retire(member, FinishReason::kFailed);
            }
        }
    }
    catch (...)
    {
        for (size_t member = 0; member < cohort; ++member)
        {
            if (live[member] != 0)
            {
                responses[member].error = "unknown exception on the step path";
                retire(member, FinishReason::kFailed);
            }
        }
    }

    // Reached from every exit above, including the two catches: a stage that never gets
    // finish() never emits its terminating chunk, and the caller waits forever.
    finishAll(reasons, responses);
    for (size_t member = 0; member < cohort; ++member)
    {
        responses[member].ok = reasons[member] != FinishReason::kFailed && responses[member].error.empty();
    }
    return responses;
}

IStage const* MultiStageRuntime::stage(std::string const& stageId) const noexcept
{
    for (size_t index = 0; index < mConfig.stages.size() && index < mStages.size(); ++index)
    {
        if (mConfig.stages[index].id == stageId)
        {
            return mStages[index].get();
        }
    }
    return nullptr;
}

void MultiStageRuntime::cancel(RequestSlot& slot) noexcept
{
    // One flag on the slot. The step loop reads it before and after each of this member's
    // steps, and finishAll() then calls finish(kCancelled) on every active stage for it, so
    // cancellation reaches the whole topology rather than stopping at whichever stage was running.
    slot.abandon.request();
}

void MultiStageRuntime::finishAll(std::vector<FinishReason> const& reasons, std::vector<Response>& responses)
{
    auto const cohort = mSeats.size();
    auto const& order = mSeats.front().slot->route.order();

    auto const fail = [&](size_t member, std::string const& message) {
        LOG_ERROR("multistage request %llu failed: %s", static_cast<unsigned long long>(mSeats[member].slot->requestId),
            message.c_str());
        auto& error = responses[member].error;
        error += (error.empty() ? "" : "; ") + message;
    };
    for (auto const stageIndex : order)
    {
        auto const index = static_cast<size_t>(stageIndex);
        auto const& stage = mConfig.stages[index];
        for (size_t member = 0; member < cohort; ++member)
        {
            auto& seat = mSeats[member];
            auto ctx = contextFor(seat, stageIndex);
            try
            {
                mStages[index]->finish(ctx, reasons[member]);
                // finish() is where a producer marks its stream complete and a vocoder drains
                // its tail. Publishing here, in order(), is what lets the consumer's finish()
                // see that last binding.
                publishOutputs(stageIndex, seat.slot->route, seat.bindings);
            }
            catch (std::exception const& error)
            {
                // Keep going: one stage throwing must not cost the others their terminating
                // chunk, and the failure is this member's alone.
                fail(member, where(stage) + error.what());
            }
            catch (...)
            {
                fail(member, where(stage) + "unknown exception in finish()");
            }
        }
    }

    for (size_t member = 0; member < cohort; ++member)
    {
        auto const& slot = *mSeats[member].slot;
        for (auto const stageIndex : order)
        {
            auto const index = static_cast<size_t>(stageIndex);
            try
            {
                auto report = mStages[index]->diagnostics(slot);
                if (!report.empty())
                {
                    responses[member].stageDiagnostics[mConfig.stages[index].id] = std::move(report);
                }
            }
            catch (std::exception const& error)
            {
                fail(member, where(mConfig.stages[index]) + error.what());
            }
            catch (...)
            {
                fail(member, where(mConfig.stages[index]) + "unknown exception in diagnostics()");
            }
        }

        // An error from finish() or diagnostics() fails the member even when every step succeeded.
        auto const reason = responses[member].error.empty() ? reasons[member] : FinishReason::kFailed;
        responses[member].terminalReasons.reserve(mConfig.terminals.size());
        for (auto const& terminal : mConfig.terminals)
        {
            // validate() has already proved every terminal parses and names a real port.
            auto const ref = parseSourceRef(terminal);
            auto const producer = mConfig.stageIndex(ref.stageId);
            if (producer >= 0 && slot.route.isActive(producer))
            {
                responses[member].terminalReasons.emplace_back(terminal, reason);
            }
        }
    }
}

void MultiStageRuntime::take()
{
    // reset() is idempotent, so resetting every stage is cheaper than tracking which ones
    // the last route touched, and it cannot leave an inactive stage holding stale state.
    for (auto& stage : mStages)
    {
        if (stage != nullptr)
        {
            stage->reset();
        }
    }
    // The callers' references, the port tables and the borrowed payloads all die with the seats.
    mSeats.clear();
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
