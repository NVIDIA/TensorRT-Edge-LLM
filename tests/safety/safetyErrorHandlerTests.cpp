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

#include "safetyTestLifecycle.h"

#include "safety_inference/plugins/safetyPluginRegistry.h"
#include "safety_inference/runtime/safetyErrorState.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

namespace trt_edgellm::safety::test
{

bool sameSafetyStatus(SafetyStatus const& lhs, SafetyStatus const& rhs) noexcept;
bool verifySafetyStatusBounds() noexcept;
bool verifyBoundedRecorderContracts(
    SafetyErrorState& state, nvinfer2::safe::ISafeRecorder& recorder, SafetyStatus const& firstFatal) noexcept;

namespace
{

//! Scenarios are isolated one per process so terminal-state cases cannot contaminate each other.
enum class Scenario
{
    kREQUEST_SCOPED_REJECTION,
    kFATAL_INITIALIZATION,
    kFATAL_OPERATIONAL_EXECUTION,
    kREPEATED_TERMINAL_STATE_REJECTION,
    kRECORDER_PROPAGATION,
    kHEALTHY_RECORDER_TEARDOWN,
    kUNKNOWN,
};

//! Model one graph borrower that owns one DLI recorder reference.
class TestGraphBorrower
{
public:
    //! Acquire one graph-held reference.
    //! @param recorder Recorder borrowed from the runtime handler.
    //! @param releaseCount Sink for the post-release reference count.
    TestGraphBorrower(nvinfer2::safe::ISafeRecorder& recorder, nvinfer2::safe::RefCount* releaseCount) noexcept
        : mRecorder(&recorder)
        , mAcquireCount(recorder.incRefCount())
        , mReleaseCount(releaseCount)
    {
    }

    TestGraphBorrower(TestGraphBorrower const&) = delete;
    TestGraphBorrower& operator=(TestGraphBorrower const&) = delete;

    //! Balance the held reference if explicit test teardown did not already do so.
    ~TestGraphBorrower()
    {
        destroy();
    }

    //! Return the count observed immediately after acquisition.
    //! @return Incremented recorder reference count.
    nvinfer2::safe::RefCount acquireCount() const noexcept
    {
        return mAcquireCount;
    }

    //! Release the graph-held reference exactly once.
    void destroy() noexcept
    {
        if (mRecorder != nullptr)
        {
            *mReleaseCount = mRecorder->decRefCount();
            mRecorder = nullptr;
        }
    }

private:
    nvinfer2::safe::ISafeRecorder* mRecorder;
    nvinfer2::safe::RefCount mAcquireCount;
    nvinfer2::safe::RefCount* mReleaseCount;
};

//! Invoke a fake Operational participant only when the shared state admits work.
//! @param state Runtime policy to guard.
//! @param calls Participant invocation count to update.
void invokeOnlyIfHealthy(SafetyErrorState& state, int32_t& calls) noexcept
{
    if (!state.fatal())
    {
        ++calls;
    }
}

bool testRequestScopedRejection() noexcept
{
    // Given healthy runtime state, when one request is rejected, then the next valid request remains admitted.
    // Mutation mapping: R1-R3, S1-S3.
    SafetyErrorState state;
    auto const rejected = state.reject(SafetyStatusCode::kInvalidArgument, "invalid request");
    if (state.fatal())
    {
        return false;
    }
    int32_t calls{};
    invokeOnlyIfHealthy(state, calls);
    return calls == 1 && rejected.code() == SafetyStatusCode::kInvalidArgument && verifySafetyStatusBounds();
}

bool testFatalInitialization() noexcept
{
    // Given Initialization, when a resource fatal is reported, then Operational admission and retry stay blocked.
    // Mutation mapping: I1-I3.
    SafetyErrorState state;
    auto const fatal = SafetyStatus::fatal(SafetyStatusCode::kFailedAllocation, "resource init failed");
    state.setFatal(fatal);
    if (!sameSafetyStatus(state.status(), fatal))
    {
        return false;
    }
    int32_t retryCalls{};
    invokeOnlyIfHealthy(state, retryCalls);
    return retryCalls == 0;
}

bool testFatalOperationalExecution() noexcept
{
    // Given an admitted Operational participant, when CUDA fails, then it runs once and no fallback runs.
    // Mutation mapping: O1-O4.
    SafetyErrorState state;
    int32_t calls{};
    invokeOnlyIfHealthy(state, calls);
    auto const fatal = SafetyStatus::fatal(SafetyStatusCode::kCudaError, "launch failed");
    state.setFatal(fatal);
    invokeOnlyIfHealthy(state, calls);
    return calls == 1 && sameSafetyStatus(state.status(), fatal);
}

bool testRepeatedTerminalStateRejection() noexcept
{
    // Given a latched fatal, when more requests and another fatal arrive, then work stops and first cause remains.
    // Mutation mapping: T1-T3.
    SafetyErrorState state;
    auto const first = SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "first");
    auto const later = SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "later capability failure");
    state.setFatal(first);
    int32_t calls{};
    invokeOnlyIfHealthy(state, calls);
    state.setFatal(later);
    invokeOnlyIfHealthy(state, calls);
    return calls == 0 && sameSafetyStatus(state.status(), first);
}

bool testRecorderPropagation() noexcept
{
    // Given three graph seams, when DLI reports a failure, then all share one runtime recorder and latch the status.
    // Mutation mapping: P1-P11, D1.
    SafetyErrorState state;
    auto* const runtimeRecorder = &state.recorder();
    auto* const processRecorder = &processRegistrationRecorder();
    std::array<nvinfer2::safe::ISafeRecorder*, 3> const graphRecorders{
        runtimeRecorder, runtimeRecorder, runtimeRecorder};
    if (runtimeRecorder == processRecorder
        || !std::all_of(graphRecorders.begin(), graphRecorders.end(),
            [runtimeRecorder](auto* recorder) { return recorder == runtimeRecorder; }))
    {
        return false;
    }
    constexpr char const* kDLI_DESCRIPTION{"DLI execution failed"};
    if (!runtimeRecorder->reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, kDLI_DESCRIPTION))
    {
        return false;
    }
    auto const fatal = state.status();
    if (!fatal.fatalStatus() || fatal.code() != SafetyStatusCode::kFailedExecution
        || std::strcmp(fatal.message(), kDLI_DESCRIPTION) != 0 || runtimeRecorder->getNbErrors() != 1
        || runtimeRecorder->getErrorCode(0) != nvinfer1::ErrorCode::kINTERNAL_ERROR
        || std::strcmp(runtimeRecorder->getErrorDesc(0), kDLI_DESCRIPTION) != 0)
    {
        return false;
    }
    return verifyBoundedRecorderContracts(state, *runtimeRecorder, fatal);
}

bool testHealthyRecorderTeardown() noexcept
{
    // Given two graph borrowers, when healthy teardown is reversed, then references reach zero before the handler.
    // Mutation mapping: H1-H4.
    auto* const processBefore = &processRegistrationRecorder();
    {
        SafetyErrorState state;
        auto& recorder = state.recorder();
        nvinfer2::safe::RefCount firstReleaseCount{-1};
        nvinfer2::safe::RefCount secondReleaseCount{-1};
        TestGraphBorrower firstGraph(recorder, &firstReleaseCount);
        TestGraphBorrower secondGraph(recorder, &secondReleaseCount);
        if (firstGraph.acquireCount() != 1 || secondGraph.acquireCount() != 2)
        {
            return false;
        }
        secondGraph.destroy();
        firstGraph.destroy();
        if (secondReleaseCount != 1 || firstReleaseCount != 0)
        {
            return false;
        }
    }
    return processBefore == &processRegistrationRecorder();
}

//! Parse exactly one standalone scenario name.
//! @param argc Process argument count.
//! @param argv Process argument values.
//! @return Selected scenario, or kUNKNOWN for invalid input.
Scenario readScenario(int argc, char** argv) noexcept
{
    if (argc != 2)
    {
        return Scenario::kUNKNOWN;
    }
    struct Entry
    {
        char const* name;
        Scenario scenario;
    };
    constexpr std::array<Entry, 6> kSCENARIOS{{{"request_scoped_rejection", Scenario::kREQUEST_SCOPED_REJECTION},
        {"fatal_initialization", Scenario::kFATAL_INITIALIZATION},
        {"fatal_operational_execution", Scenario::kFATAL_OPERATIONAL_EXECUTION},
        {"repeated_terminal_state_rejection", Scenario::kREPEATED_TERMINAL_STATE_REJECTION},
        {"recorder_propagation", Scenario::kRECORDER_PROPAGATION},
        {"healthy_recorder_teardown", Scenario::kHEALTHY_RECORDER_TEARDOWN}}};
    for (auto const& entry : kSCENARIOS)
    {
        if (std::strcmp(argv[1], entry.name) == 0)
        {
            return entry.scenario;
        }
    }
    return Scenario::kUNKNOWN;
}

//! Dispatch one scenario without retrying it in the same process.
//! @param scenario Scenario selected from the command line.
//! @return True when the selected contract passes.
bool runStandaloneScenario(Scenario scenario) noexcept
{
    bool passed{};
    switch (scenario)
    {
    case Scenario::kREQUEST_SCOPED_REJECTION: passed = testRequestScopedRejection(); break;
    case Scenario::kFATAL_INITIALIZATION: passed = testFatalInitialization(); break;
    case Scenario::kFATAL_OPERATIONAL_EXECUTION: passed = testFatalOperationalExecution(); break;
    case Scenario::kREPEATED_TERMINAL_STATE_REJECTION: passed = testRepeatedTerminalStateRejection(); break;
    case Scenario::kRECORDER_PROPAGATION: passed = testRecorderPropagation(); break;
    case Scenario::kHEALTHY_RECORDER_TEARDOWN: passed = testHealthyRecorderTeardown(); break;
    case Scenario::kUNKNOWN: break;
    }
    return passed;
}

} // namespace
} // namespace trt_edgellm::safety::test

int main(int argc, char** argv)
{
    using namespace trt_edgellm::safety;
    auto const scenario = test::readScenario(argc, argv);
    bool const passed = test::runStandaloneScenario(scenario);
    if (!passed)
    {
        std::fprintf(stderr, "Safety error-handler scenario failed or was not recognized.\n");
    }
    return engine_test::finishTestProcess(passed ? 0 : 1);
}
