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

#include "common/checkMacros.h"
#include "runtime/multiStage/route.h"
#include "runtime/multiStage/stageConfig.h"
#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct SharedResources;
class StreamChannel;
class AudioStreamChannel;
class VideoStreamChannel;

namespace multiStage
{

//! Result of one step.
enum class StageStatus : uint8_t
{
    kMore,    //!< Call step() again.
    kDone,    //!< This stage is finished for this request.
    kBlocked, //!< An input port is empty. The executor must advance the producer first.
    kFailed,  //!< Unrecoverable. The executor tears the request down.
};

char const* toString(StageStatus v);

//! Why a request ended. Response::terminalReasons repeats it for each active terminal.
enum class FinishReason : uint8_t
{
    kCompleted,
    kCancelled, //!< barge-in or caller-side abort.
    kFailed,
};

char const* toString(FinishReason v);

//! A non-owning view of a value on a port, in host or device memory.
struct PortValue
{
    void* data{nullptr};
    std::vector<int64_t> shape;
    PortKind kind{PortKind::kText};
    int32_t elementBytes{0};
    //! The producer will not extend this value. Until it is set, a consumer that has
    //! drained everything bound so far must report kBlocked rather than kDone: an
    //! empty remainder means "not yet", and only the producer knows when it means "never".
    bool complete{false};
};

class PortBindings
{
public:
    void bind(std::string const& name, PortValue value);
    //! nullptr when the input was skipped. Callers must handle it; there is no default.
    PortValue const* find(std::string const& name) const;
    //! Throws naming the input. Use when the config guarantees presence (OnMissing::kFail).
    PortValue const& at(std::string const& name) const;
    void clear();

    //! Every bound name, in unspecified order.
    std::vector<std::string> names() const;

private:
    std::unordered_map<std::string, PortValue> mValues;
};

/*!
 * @brief Read a UTF-8 JSON object bound on an input.
 *
 * Returns an empty object when the input is absent, which is the common case: per-request
 * overrides are optional and a stage falls back to its deployment config. Throws naming the
 * input when something IS bound but does not parse, because silently ignoring a malformed
 * override would run the request with settings the caller did not ask for.
 */
nlohmann::json jsonInput(PortBindings const* in, std::string const& name);

//! Cancellation token. Polled at every abandon point; never checked by a raw bool.
class AbandonToken
{
public:
    void request() noexcept
    {
        mRequested.store(true, std::memory_order_release);
        std::vector<std::function<void()>> wakes;
        try
        {
            std::lock_guard<std::mutex> const guard(mWakeMutex);
            wakes = mWakes;
        }
        catch (...)
        {
            return;
        }
        for (auto const& wake : wakes)
        {
            wake();
        }
    }

    bool requested() const noexcept
    {
        return mRequested.load(std::memory_order_acquire);
    }

    void reset() noexcept
    {
        mRequested.store(false, std::memory_order_release);
        try
        {
            std::lock_guard<std::mutex> const guard(mWakeMutex);
            mWakes.clear();
        }
        catch (...)
        {
        }
    }

    /*!
     * @brief Run @p wake when this token is requested.
     *
     * A stage blocked inside its own decode loop never reaches the next abandon point, so it
     * registers the one call that unblocks it -- for the backbone, cancelling the stream
     * channel the decode loop already polls once per iteration. Registering after the fact
     * still fires, because a cancel that arrived first must not be lost.
     */
    void onRequest(std::function<void()> wake)
    {
        if (!wake)
        {
            return;
        }
        {
            std::lock_guard<std::mutex> const guard(mWakeMutex);
            mWakes.push_back(wake);
        }
        if (requested())
        {
            wake();
        }
    }

private:
    std::atomic<bool> mRequested{false};
    std::mutex mWakeMutex;
    std::vector<std::function<void()>> mWakes;
};

//! One in-flight request.
struct RequestSlot
{
    uint64_t requestId{0};
    int32_t batchSlot{-1}; //!< Batch row this request occupies within its cohort.
    Route route;
    AbandonToken abandon;
    //! What the caller supplied, keyed by the fully-qualified request field
    //! ("request:messages", "request:audio"). Borrowed: the caller's storage must
    //! outlive the request. The executor binds from here into each stage that reads a
    //! request:* input, which is the only way such an input reaches a stage.
    PortBindings payload;
};

//! What a stage is handed on every per-request call. Borrowed, except textSink, which is shared.
struct StageExecContext
{
    RequestSlot* slot{nullptr};
    PortBindings const* in{nullptr};
    PortBindings* out{nullptr};
    cudaStream_t stream{nullptr}; //!< This stage's lane stream.

    //! Where a terminal's output leaves the topology. Null unless this stage emits a
    //! terminal port of that kind AND the caller attached a channel to it. A stage still
    //! does not learn its position: it only learns whether it has an exit.
    std::shared_ptr<StreamChannel> textSink;
    AudioStreamChannel* audioSink{nullptr};
    VideoStreamChannel* videoSink{nullptr};
};

//! What a stage is handed once, at construction.
struct StageInitContext
{
    StageConfig const* config{nullptr};
    std::string artifactDir; //!< Already resolved and checked against the config root.
    SharedResources* shared{nullptr};
    cudaStream_t stream{nullptr};
    int32_t maxBatch{1};
    int32_t maxSeq{0};
};

/*!
 * @brief One computation, behind a uniform interface.
 *
 * A stage reaches other stages only through its ports. The bindings, slots and streams it
 * borrows are owned by the executor; the sinks and SharedResources belong to the caller.
 */
class IStage
{
public:
    virtual ~IStage() = default;

    //! Init-time allocation. Every device buffer this stage will ever use is claimed here,
    //! sized by maxBatch x maxSeq. Nothing on the step path may allocate.
    virtual void preallocate() = 0;

    //! Per-request admission. Reject here -- it costs nothing -- rather than failing mid-step.
    //! @return false plus a reason when this stage cannot serve the request.
    virtual bool admit(StageExecContext const& ctx, std::string& reason) = 0;

    //! Per-request setup, once the inputs this stage reads are published. A throw fails only
    //! this request, so what prepare() writes must be this member's own or scratch that the
    //! next member's prepare() rewrites.
    virtual void prepare(StageExecContext& ctx) = 0;
    virtual StageStatus step(StageExecContext& ctx) = 0;

    /*!
     * @brief One step for every cohort member the executor has ready.
     *
     * @p statuses is index-aligned with @p members. The default serves a cohort of one and
     * refuses anything wider, because a stage holding a single request's state cannot
     * interleave members -- round-robin would have member B step on A's buffers. A stage
     * that batches overrides this: @p members is the only place the cohort is visible
     * together, so it is the only place a batched call can be formed.
     */
    virtual void stepCohort(std::vector<StageExecContext*> const& members, std::vector<StageStatus>& statuses)
    {
        ELLM_CHECK(members.size() == 1, "this stage serves one request at a time");
        statuses[0] = step(*members[0]);
    }

    //! Called for every member on every active stage when run() ends: success, failure and
    //! cancellation alike. Emits the terminating chunk and reports metrics. A stage that only
    //! cleans up on success is a leak. take() without run() skips it; see reset().
    virtual void finish(StageExecContext& ctx, FinishReason reason) = 0;

    //! Idempotent. Drops per-request state; anything longer-lived survives.
    virtual void reset() = 0;

    /*!
     * @brief What this stage did on @p slot's request, for that caller's report.
     *
     * Read after finish() and before reset(). Keys are the stage's own; the runtime nests
     * them under the stage id and never interprets them. An empty object means the stage
     * has nothing to say, which is the default. Taking the slot is what keeps a cohort's
     * members from all reporting the same numbers. A throw fails only that request, which
     * keeps the other stages' reports.
     *
     * This is deliberately not gated on profiling: a caller needs to know why generation
     * stopped even when no timing was collected.
     */
    virtual nlohmann::json diagnostics(RequestSlot const& slot) const
    {
        (void) slot;
        return nlohmann::json::object();
    }
};

using StageFactory = std::unique_ptr<IStage> (*)(StageInitContext const&);

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
