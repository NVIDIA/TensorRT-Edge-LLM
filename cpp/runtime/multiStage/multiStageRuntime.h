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
#include "runtime/streaming.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

/*!
 * @brief Where a caller collects a terminal's output.
 *
 * Attaching a channel is what makes a terminal reachable, so this is also how
 * "only text, no audio" is expressed: leave the audio pointer null and every stage
 * that exists only to produce audio drops out of the route.
 */
struct Sinks
{
    //! Shared, not borrowed: the backbone holds this channel from prepare() to finish() and
    //! its abandon wake captures it, so its lifetime cannot be tied to the caller's stack frame.
    std::shared_ptr<StreamChannel> text;
    AudioStreamChannel* audio{nullptr};
    VideoStreamChannel* video{nullptr};

    //! Terminal port ids ("<stage>:<port>") that have a channel attached. Empty when
    //! nothing is attached, which Route::resolve rejects.
    std::vector<std::string> attachedTerminals(MultiStageConfig const& config) const;
};

//! Per-terminal outcome. A terminal that was cancelled or failed says so here; a caller
//! cannot infer it from the channel alone.
struct Response
{
    uint64_t requestId{0};
    std::vector<std::pair<std::string, FinishReason>> terminalReasons;
    bool ok{false}; //!< No stage failed. A cancelled request is ok; terminalReasons says cancelled.
    std::string error;

    //! Per-stage report, keyed by stage id. Collected after finish(), before reset(), so a
    //! caller sees it whether or not the request succeeded. A stage whose diagnostics() throws
    //! has no entry, and the request fails.
    nlohmann::json stageDiagnostics = nlohmann::json::object();
};

class StageRegistry;

/*!
 * @brief L2. Drives one MultiStageConfig.
 *
 * The only C++ entry point for running a topology. It owns admission, each seat's frozen
 * Route, the lane streams and the step loop. It does not know that omni exists:
 * every model family is a set of values in the config it was handed.
 *
 * Drives one static cohort at a time: submit() up to the cohort limit, run(), then take().
 */
class MultiStageRuntime
{
public:
    MultiStageRuntime();
    ~MultiStageRuntime();

    MultiStageRuntime(MultiStageRuntime const&) = delete;
    MultiStageRuntime& operator=(MultiStageRuntime const&) = delete;

    /*!
     * @brief Parse, validate, build every stage and claim all device memory.
     *
     * Throws naming the stage on any failure; a partially constructed runtime is never
     * handed back. @p runners defaults to builtinRunners(); it is not read after load() returns.
     */
    void load(std::string const& configRoot, SharedResources& shared, StageRegistry const* runners = nullptr);

    //! load() for a config the caller parsed and may have adjusted. Paths still resolve
    //! against @p configRoot.
    void load(MultiStageConfig config, std::string const& configRoot, SharedResources& shared,
        StageRegistry const* runners = nullptr);

    //! What load() checks of a validated config before it builds a stage: each entry against its
    //! runner's declaration in @p runners, and each edge against the lanes. Touches no device.
    static void checkLoadable(MultiStageConfig const& config, StageRegistry const& runners);

    MultiStageConfig const& config() const noexcept
    {
        return mConfig;
    }

    //! The stage built for @p stageId, for a caller that knows its runner and wants more than
    //! IStage offers, such as a backbone's aggregate metrics. Null for an id the config lacks.
    IStage const* stage(std::string const& stageId) const noexcept;

    /*!
     * @brief Seat one request in the cohort: resolve the route, admit each stage.
     *
     * Refused when the cohort is full or this request resolves a different route than the
     * seated ones.
     *
     * Pure CPU. Returns nullptr and fills @p error when the request cannot be served,
     * which is the cheap place to fail -- no GPU work has happened yet.
     *
     * @param payload Keyed by fully-qualified request field ("request:messages"). What the
     *                request supplies is derived from what it carries, so a topology cannot
     *                route around a modality the caller declared but did not bring. The
     *                storage behind each value is borrowed until take().
     */
    RequestSlot* submit(PortBindings const& payload, Sinks const& sinks, std::string& error);

    /*!
     * @brief Drive every admitted request to completion, one response per seat.
     *
     * Responses come back in admission order. Members share one step loop, so a stage that
     * batches sees them together; submit() caps the cohort at one on a route through a stage
     * that does not. Each member's abandon token is polled at every step. A member that is
     * cancelled, whose prepare(), finish() or diagnostics() throws, or whose stage returns
     * kFailed does not stop the others; an exception from stepCohort() fails every live member.
     */
    std::vector<Response> run();

    //! Request cancellation. Safe from another thread; the step loop observes it at the
    //! next abandon point. Reaches EVERY active stage, not just the first one.
    void cancel(RequestSlot& slot) noexcept;

    //! Reset every stage and release every seat, whether or not run() was called. A cohort
    //! that never ran gets no finish().
    void take();

private:
    //! One CUDA stream per stream_lane.
    struct Lane
    {
        cudaStream_t stream{nullptr};
    };

    /*!
     * @brief Everything one admitted request owns.
     *
     * Port tables and exits are per seat, so one member cannot overwrite another's published
     * ports before a consumer has read them.
     */
    struct Seat
    {
        std::unique_ptr<RequestSlot> slot;
        //! Per stage, index-aligned with mConfig.stages.
        std::vector<PortBindings> bindings;
        //! This request's text, audio, and video channels.
        Sinks sinks;
    };

    //! How many requests may share a cohort on @p route. The narrowest stage the route
    //! actually activates decides; a stage the route pruned cannot constrain it.
    int32_t cohortLimit(Route const& route) const noexcept;

    void destroyLanes() noexcept;
    StageExecContext contextFor(Seat& seat, int32_t stageIndex);
    //! finish() every active stage for every seat, then collect each seat's report.
    void finishAll(std::vector<FinishReason> const& reasons, std::vector<Response>& responses);
    //! True once every input with an edge into this stage on the seat's route has a binding.
    bool inputsReady(int32_t stageIndex, Seat const& seat) const;

    MultiStageConfig mConfig;
    std::string mConfigRoot;
    std::vector<std::unique_ptr<IStage>> mStages;
    //! Per stage, the width it is built and admitted at.
    std::vector<int32_t> mStageWidths;
    std::vector<Lane> mLanes;
    //! One entry per in-flight request, in admission order. A seat's index is its batchSlot,
    //! which is what a stage keying per-request state off ctx.slot->batchSlot indexes by.
    std::vector<Seat> mSeats;
    uint64_t mNextRequestId{1};
};

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
