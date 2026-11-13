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

#include "safetyRuntimeTestDoubles.h"
#include "safety_inference/runtime/safetyRuntime.h"
#include "safety_inference/trt_runner/trtSafetyRunner.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>

namespace trt_edgellm::safety::runtime_test
{
namespace
{
enum class ExpectedStatus
{
    kSUCCESS,
    kINVALID_ARGUMENT,
    kINVALID_STATE,
    kFATAL_ERROR
};

bool operator==(SafetyStatus const& actual, ExpectedStatus expected) noexcept
{
    if (expected == ExpectedStatus::kFATAL_ERROR)
    {
        return !actual.okStatus() && actual.fatalStatus();
    }
    if (actual.fatalStatus())
    {
        return false;
    }
    auto code = SafetyStatusCode::kSuccess;
    if (expected == ExpectedStatus::kINVALID_ARGUMENT)
    {
        code = SafetyStatusCode::kInvalidArgument;
    }
    else if (expected == ExpectedStatus::kINVALID_STATE)
    {
        code = SafetyStatusCode::kInvalidState;
    }
    return actual.code() == code;
}

bool operator!=(SafetyStatus const& actual, ExpectedStatus expected) noexcept
{
    return !(actual == expected);
}

bool operator==(std::vector<SafetyStatus> const& actual, std::vector<ExpectedStatus> const& expected)
{
    return actual.size() == expected.size()
        && std::equal(actual.begin(), actual.end(), expected.begin(),
            [](SafetyStatus const& status, ExpectedStatus outcome) { return status == outcome; });
}

SafetyGraphConfig graphConfig()
{
    static constexpr int64_t kENGINE_BYTES{8};
    static char const kENGINE[kENGINE_BYTES]{};
    return {kENGINE, kENGINE_BYTES, "/trusted/engine.so", {0, 1}};
}
SafetyRuntimeInitSpec spec()
{
    return {{graphConfig(), graphConfig()}, {{"prefill", 0, 0, {0, 1}, {}}, {"decode", 0, 1, {0, 1}, {}}}};
}

SafetyRuntimeInitSpec fixedSpec()
{
    auto init = spec();
    for (size_t index = 0; index < init.executors.size(); ++index)
    {
        init.executors[index].declaredProfiles = {0};
        init.stages[index].profiles = {0};
    }
    return init;
}

SafetyRuntimeInitSpec capturedSpec()
{
    auto config = graphConfig();
    config.enableCudaGraphCapture = true;
    return {{config}, {{"prefill", 0, 0, {0}, {}}, {"decode", 0, 0, {1}, {}}}};
}

SafetyRuntimeInitSpec catalogSpec()
{
    auto prefill = graphConfig();
    prefill.declaredProfiles = {0};
    auto decode = graphConfig();
    decode.declaredProfiles = {1};
    return {{prefill, decode}, {{"prefill", 0, 0, {0}, {}}, {"decode", 0, 1, {1}, {}}}, {0, 0}};
}

struct StagePreparation
{
    std::array<LMStageContext const*, 4> contexts{};
    std::array<int64_t, 4> profiles{};
    size_t calls{0};
    float input{0.0F};
    SafetyStatus status{SafetyStatus::ok()};
    Recorder* recorder{nullptr};
};

SafetyStatus prepareStage(LMStageContext const& context, void* userData) noexcept
{
    auto& state = *static_cast<StagePreparation*>(userData);
    size_t const call = state.calls++;
    if (context.profiles.size() != 1 || call >= state.contexts.size())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Unexpected preparation context");
    }
    state.contexts[call] = &context;
    state.profiles[call] = context.profiles[0].metadata->profileIndex;
    gBackend.operations.push_back({"callback", -1, state.profiles[call], context.mainStream});
    context.profiles[0].resources->bindings[0].getFloat()[0] = state.input;
    if (state.recorder != nullptr)
    {
        state.recorder->reportError(ErrorCode::kFAILED_EXECUTION, "Injected preparation recorder error");
    }
    return state.status;
}

class PreparationError : public std::exception
{
public:
    explicit PreparationError(char const* detail) noexcept
        : mDetail(detail)
    {
    }
    char const* what() const noexcept override
    {
        return mDetail;
    }

private:
    char const* mDetail;
};

SafetyStatus throwWarmup(LMStageContext const&, int64_t, void* detail)
{
    throw PreparationError(static_cast<char const*>(detail));
}

using FailureDetails = std::tuple<SafetyStatusCode, std::string, bool>;

FailureDetails failureDetails(SafetyStatus const& status)
{
    return {status.code(), status.message(), status.fatalStatus()};
}

struct WarmupFailure
{
    SafetyStatus status;
    Recorder* recorder{nullptr};
};

SafetyStatus failedWarmup(LMStageContext const& context, int64_t profile, void* userData)
{
    auto& failure = *static_cast<WarmupFailure*>(userData);
    gBackend.operations.push_back({"warmupCallback", -1, profile, context.mainStream});
    context.profiles[0].resources->bindings[0].getFloat()[0] = 29.0F;
    if (failure.recorder != nullptr)
    {
        failure.recorder->reportError(ErrorCode::kINTERNAL_ERROR, "Accompanying callback recorder failure");
    }
    return failure.status;
}

void useUniformDescriptors()
{
    gBackend.useDescriptorOverrides = true;
    for (auto& profile : gBackend.descriptors)
    {
        for (size_t index = 0; index < profile.size(); ++index)
        {
            auto& descriptor = profile[index];
            descriptor.shape = {1, {8}};
            descriptor.stride = {1, {1}};
            descriptor.userShape = {1, {8}};
            descriptor.strideOrder = {1, {0}};
            descriptor.dataType = nvinfer2::safe::DataType::kFLOAT;
            descriptor.bytesPerComponent = sizeof(float);
            descriptor.componentsPerVector = 1;
            descriptor.vectorizedDim = -1;
            descriptor.sizeInBytes = 8 * sizeof(float);
            descriptor.ioMode
                = index == 0 ? nvinfer2::safe::TensorIOMode::kINPUT : nvinfer2::safe::TensorIOMode::kOUTPUT;
            descriptor.memPlacement = nvinfer2::safe::MemoryPlacement::kGPU;
        }
    }
}

struct ModelLocalBuffer
{
    char const* stage;
    std::array<float, 2> warmupInputs{7.0F, 8.0F};
};

SafetyRuntimeInitSpec modelLocalSpec()
{
    auto init = fixedSpec();
    init.executors[0].declaredProfiles = {1};
    init.stages[0].profiles = {1};
    for (auto& stage : init.stages)
    {
        stage.modelLocalResources = {{"kv-cache", sizeof(ModelLocalBuffer)}};
    }
    return init;
}

SafetyStatus fillWarmup(LMStageContext const& context, int64_t profile, void*)
{
    for (auto const& view : context.profiles)
    {
        if (view.metadata->profileIndex == profile)
        {
            view.resources->bindings[0].getFloat()[0] = static_cast<float>(7 + profile);
            return SafetyStatus::ok();
        }
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument);
}

SafetyStatus fillModelLocalWarmup(LMStageContext const& context, int64_t profile, void*)
{
    if (context.modelLocalResources.size() != 1 || context.modelLocalResources[0].data == nullptr
        || context.modelLocalResources[0].sizeInBytes < sizeof(ModelLocalBuffer))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument);
    }
    auto const& buffer = *static_cast<ModelLocalBuffer const*>(context.modelLocalResources[0].data);
    if (buffer.stage != context.plan.id || profile < 0 || static_cast<size_t>(profile) >= buffer.warmupInputs.size())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument);
    }
    for (auto const& view : context.profiles)
    {
        if (view.metadata->profileIndex == profile)
        {
            view.resources->bindings[0].getFloat()[0] = buffer.warmupInputs[static_cast<size_t>(profile)];
            return SafetyStatus::ok();
        }
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument);
}
std::vector<std::string> operationsSince(size_t begin)
{
    std::vector<std::string> names;
    for (size_t index = begin; index < gBackend.operations.size(); ++index)
    {
        names.push_back(gBackend.operations[index].name);
    }
    return names;
}

std::vector<void*> bindingSnapshot(SafetyRuntime const& runtime)
{
    std::vector<void*> addresses;
    for (auto const& context : runtime.contexts())
    {
        addresses.push_back(context.mainStream);
        for (auto const& profile : context.profiles)
        {
            for (auto const& binding : profile.resources->bindings)
            {
                addresses.push_back(binding.getData());
            }
        }
    }
    return addresses;
}
class SafetyRuntimeHost : public testing::Test
{
protected:
    void SetUp() override
    {
        gBackend.reset();
    }
    void TearDown() override
    {
        gBackend.reset();
    }
    template <typename Runtime>
    SafetyStatus ready(Runtime& runner)
    {
        auto status = runner.initialize(fixedSpec());
        if (status != ExpectedStatus::kSUCCESS)
        {
            return status;
        }
        return runner.prepare(fillWarmup);
    }
    SafetyStatus readyWithModelLocalBuffers(TRTSafetyRunner& runner)
    {
        auto status = runner.initialize(modelLocalSpec());
        if (status != ExpectedStatus::kSUCCESS)
        {
            return status;
        }
        status = runner.completeContext("prefill", 0, {{"kv-cache", &mPrefillBuffer, sizeof(mPrefillBuffer)}});
        if (status != ExpectedStatus::kSUCCESS)
        {
            return status;
        }
        status = runner.completeContext("decode", 0, {{"kv-cache", &mDecodeBuffer, sizeof(mDecodeBuffer)}});
        if (status != ExpectedStatus::kSUCCESS)
        {
            return status;
        }
        return prepareRunner(runner, fillModelLocalWarmup);
    }
    SafetyStatus ready(TRTSafetyRunner& runner)
    {
        auto status = runner.initialize(fixedSpec());
        if (status.okStatus())
        {
            status = prepareRunner(runner, fillWarmup);
        }
        return status;
    }
    SafetyStatus prepareRunner(TRTSafetyRunner& runner, LMStageWarmupCallback warmup)
    {
        auto const status = runner.prepare(warmup);
        if (status.okStatus())
        {
            mLifecycleState = LifecycleState::kOPERATIONAL;
        }
        return status;
    }
    SafetyStatus teardownRunner(TRTSafetyRunner& runner)
    {
        if (mLifecycleState != LifecycleState::kCLOSED)
        {
            auto const status = runner.deinitializeGraphs();
            if (!status.okStatus())
            {
                return status;
            }
            mLifecycleState = LifecycleState::kCLOSED;
        }
        return runner.deinitializeResources();
    }
    SafetyStatus readyCaptured(SafetyRuntime& runtime)
    {
        auto const status = runtime.initialize(capturedSpec());
        return status == ExpectedStatus::kSUCCESS ? runtime.prepare(fillWarmup) : status;
    }
    // Fixture storage survives contexts and graph handles retained until the backend's containment reset.
    ModelLocalBuffer mPrefillBuffer{"prefill"};
    ModelLocalBuffer mDecodeBuffer{"decode"};
    Recorder mRecorder;
    SafetyErrorState mErrorState{mRecorder};
    // Direct component tests provide the lifecycle normally owned by SafetyRuntime.
    LifecycleState mLifecycleState{LifecycleState::kINITIALIZATION};
};

TEST_F(SafetyRuntimeHost, LoadsEveryExecutorBeforeCudaRuntimeSetup)
{
    // Given two executors, when initialization begins, then both modules load before CUDA runtime setup.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(runner.initialize(spec()), ExpectedStatus::kSUCCESS);
    auto const firstCreate = std::find_if(gBackend.operations.begin(), gBackend.operations.end(),
        [](Operation const& operation) { return operation.name == "create"; });
    std::vector<std::string> names;
    for (auto operation = gBackend.operations.begin(); operation != firstCreate; ++operation)
    {
        names.push_back(operation->name);
    }
    EXPECT_EQ(names, (std::vector<std::string>{"load", "load", "deviceGetLimit", "prepare"}));
    EXPECT_EQ(gBackend.queriedLimits, (std::vector<cudaLimit>{cudaLimitStackSize}));
    EXPECT_EQ(gBackend.streamFlags, std::vector<unsigned int>(4, cudaStreamNonBlocking));
}

TEST_F(SafetyRuntimeHost, OwnsOneRecorderSharedByEveryGraph)
{
    // Given a runtime-owned error state, when graphs initialize, then every graph borrows that state's recorder.
    SafetyRuntime runtime;
    auto& recorder = static_cast<SafetyErrorRecorder&>(runtime.errorRecorder());
    ASSERT_EQ(runtime.initialize(fixedSpec()), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(gBackend.graphs.size(), 2U);
    EXPECT_EQ(recorder.referenceCount(), 2);
    for (auto* graph : gBackend.graphs)
    {
        nvinfer2::safe::ISafeRecorder* graphRecorder{};
        ASSERT_EQ(static_cast<Graph*>(graph)->getSafeRecorder(graphRecorder), ErrorCode::kSUCCESS);
        EXPECT_EQ(graphRecorder, &runtime.errorRecorder());
    }
    EXPECT_EQ(runtime.deinitialize(), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(recorder.referenceCount(), 0);
}

TEST_F(SafetyRuntimeHost, RetainsOwnedRuntimeAfterFatalError)
{
    // Given an owned runtime with a fatal SDK error, when execution and teardown are retried, then graphs and the
    // queryable recorder remain retained without further SDK calls.
    auto runtime = std::make_unique<SafetyRuntime>();
    auto& recorder = static_cast<SafetyErrorRecorder&>(runtime->errorRecorder());
    ASSERT_EQ(ready(*runtime), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(recorder.referenceCount(), 2);
    gBackend.emitRecorderError = true;
    LMStageOutputs outputs;
    EXPECT_EQ(runtime->runLMStage("prefill", 0, outputs), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(runtime->lifecycleState(), LifecycleState::kFAILED);
    ASSERT_EQ(recorder.getNbErrors(), 1);
    EXPECT_EQ(recorder.getErrorCode(0), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_STREQ(recorder.getErrorDesc(0), "Injected asynchronous recorder error");
    size_t const stopped = gBackend.operations.size();
    EXPECT_EQ(runtime->runLMStage("decode", 0, outputs), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(runtime->deinitialize(), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
    EXPECT_EQ(gBackend.graphs.size(), 2U);
    EXPECT_EQ(recorder.referenceCount(), 2);
    EXPECT_STREQ(recorder.getErrorDesc(0), "Injected asynchronous recorder error");

    // Reset models application termination after fatal containment, not runtime Deinitialization.
    gBackend.reset();
    EXPECT_EQ(recorder.referenceCount(), 0);
    runtime.reset();
}

class SafetyRuntimeInitializationFailure : public SafetyRuntimeHost, public testing::WithParamInterface<bool>
{
};
TEST_P(SafetyRuntimeInitializationFailure, StopsGraphSetupWithoutRetryOrCleanup)
{
    // Given a CUDA runtime initialization failure, when the runner initializes, then graph setup stops and retries or
    // teardown perform no SDK operations.
    if (GetParam())
    {
        gBackend.cudaRecorder = &mRecorder;
        gBackend.reportOnCudaOperation = "deviceGetLimit";
    }
    else
    {
        gBackend.failCudaOperation = "deviceGetLimit";
        gBackend.cudaResult = cudaErrorInitializationError;
    }
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(0), (std::vector<std::string>{"load", "load", "deviceGetLimit"}));
    EXPECT_EQ(mRecorder.getErrorCode(0), GetParam() ? ErrorCode::kFAILED_EXECUTION : ErrorCode::kFAILED_INITIALIZATION);
    mRecorder.clear();
    size_t const stopped = gBackend.operations.size();
    EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(CudaReturnAndRecorder, SafetyRuntimeInitializationFailure, testing::Values(false, true));

TEST_F(SafetyRuntimeHost, DispatchesFixedProfilesWithoutOperationalResourceChanges)
{
    // Given ready prefill/decode contexts, when stages run, then selected graphs write their stable output buffers.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(readyWithModelLocalBuffers(runner), ExpectedStatus::kSUCCESS);
    auto* prefillOutput = resources.profiles(0)[0].bindings[1].getFloat();
    auto* decodeOutput = resources.profiles(1)[0].bindings[1].getFloat();
    resources.profiles(0)[0].bindings[0].getFloat()[0] = 8.0F;
    resources.profiles(1)[0].bindings[0].getFloat()[0] = 7.0F;
    size_t const begin = gBackend.operations.size();
    LMStageOutputs outputs;
    ASSERT_EQ(runner.runLMStage("prefill", 0, outputs), ExpectedStatus::kSUCCESS);
    EXPECT_FLOAT_EQ(prefillOutput[0], 18.0F);
    ASSERT_EQ(runner.runLMStage("decode", 0, outputs), ExpectedStatus::kSUCCESS);
    EXPECT_FLOAT_EQ(decodeOutput[0], 107.0F);
    EXPECT_EQ(outputs.context->profiles[0].resources->bindings[1].getData(), decodeOutput);
    EXPECT_EQ(
        operationsSince(begin), (std::vector<std::string>{"profile", "execute", "sync", "profile", "execute", "sync"}));
}

TEST_F(SafetyRuntimeHost, RejectsUnknownStageBeforeSdkDispatch)
{
    // Given an operational runner, when an unknown stage is requested, then it rejects without touching the SDK.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(ready(runner), ExpectedStatus::kSUCCESS);
    size_t const begin = gBackend.operations.size();
    LMStageOutputs outputs;
    EXPECT_EQ(runner.runLMStage("unknown", 0, outputs), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}

TEST_F(SafetyRuntimeHost, ProfilesShareStorageSizedForLargestPhysicalLayout)
{
    // Given profiles with distinct physical sizes, when resources bind, then both profiles share maximum-capacity
    // storage.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(runner.initialize(spec()), ExpectedStatus::kSUCCESS);
    auto const& small = resources.profiles(0)[0].bindings[0];
    auto const& large = resources.profiles(0)[1].bindings[0];
    EXPECT_EQ(small.getData(), large.getData());
    EXPECT_EQ(small.getSize(), 8 * sizeof(float));
    large.getFloat()[0] = 42.0F;
    EXPECT_FLOAT_EQ(small.getFloat()[0], 42.0F);
}

TEST_F(SafetyRuntimeHost, RequiresEveryDeclaredProfileToWarmBeforeSealing)
{
    // Given an initialized graph with unprepared profiles, when sealing is requested, then no operational execution
    // begins.
    TRTSafetyGraphExecutor executor(mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(executor.seal(), ExpectedStatus::kINVALID_STATE);
    EXPECT_EQ(executor.execute(0), ExpectedStatus::kINVALID_STATE);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}

TEST_F(SafetyRuntimeHost, GraphCreationFailureIsTerminalWithoutAutomaticCleanup)
{
    // Given graph creation failure, when callers retry or destroy the runner, then no SDK cleanup or retry occurs.
    gBackend.creationResult = ErrorCode::kFAILED_INITIALIZATION;
    TRTResourceManager resources(mLifecycleState, mErrorState);
    {
        TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
        EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
        EXPECT_TRUE(runner.fatal());
        size_t const begin = gBackend.operations.size();
        EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
        EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
        EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    }
    EXPECT_EQ(gBackend.count("release") + gBackend.count("destroy"), 0);
}

class SafetyAllocationFailure : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyAllocationFailure, RetainsPartialResources)
{
    // Given a CUDA failure after earlier allocations, when initialization fails, then partial handles remain without
    // cleanup.
    gBackend.failAllocationAt = GetParam();
    TRTResourceManager resources(mLifecycleState, mErrorState);
    {
        TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
        EXPECT_EQ(runner.initialize(modelLocalSpec()), ExpectedStatus::kFATAL_ERROR);
        EXPECT_TRUE(resources.fatal());
        EXPECT_EQ(gBackend.allocations.size(), static_cast<size_t>(GetParam()));
        size_t const begin = gBackend.operations.size();
        EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
        EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    }
    EXPECT_EQ(gBackend.count("free") + gBackend.count("streamDestroy") + gBackend.count("destroy"), 0);
}
INSTANTIATE_TEST_SUITE_P(MainAuxAndIo, SafetyAllocationFailure, testing::Values(0, 1, 2, 3));

TEST_F(SafetyRuntimeHost, BindingFailureRetainsAllocationsAndGraphs)
{
    // Given a DLI binding failure after allocation, when initialization fails, then no graph or buffer is released.
    gBackend.bindingResult = ErrorCode::kFAILED_INITIALIZATION;
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(gBackend.allocations.size(), 4U);
    EXPECT_EQ(gBackend.count("free") + gBackend.count("destroy") + gBackend.count("release"), 0);
}

class SafetyDeviceIoPlacement : public SafetyRuntimeHost,
                                public testing::WithParamInterface<nvinfer2::safe::MemoryPlacement>
{
};
TEST_P(SafetyDeviceIoPlacement, AllocatesAndReleasesDeviceStorage)
{
    // Given GPU or unspecified IO placement, when resources initialize and deinitialize, then CUDA owns the storage.
    gBackend.placement = GetParam();
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(runner.initialize(spec()), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(gBackend.count("malloc"), 4);
    ASSERT_EQ(teardownRunner(runner), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(gBackend.count("free"), 4);
}
INSTANTIATE_TEST_SUITE_P(GpuAndUnspecified, SafetyDeviceIoPlacement,
    testing::Values(nvinfer2::safe::MemoryPlacement::kGPU, nvinfer2::safe::MemoryPlacement::kNONE));

TEST_F(SafetyRuntimeHost, RejectsManagedIoBeforeResourceMutation)
{
    // Given managed IO placement, when resources prepare, then preparation rejects without SDK calls and teardown
    // remains valid.
    gBackend.placement = nvinfer2::safe::MemoryPlacement::kMANAGED;
    TRTSafetyGraphExecutor executor(mErrorState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(resources.prepareResources({&executor}), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    EXPECT_EQ(executor.deinitialize(), ExpectedStatus::kSUCCESS);
    mLifecycleState = LifecycleState::kCLOSED;
    EXPECT_EQ(resources.deinitialize({&executor}), ExpectedStatus::kSUCCESS);
}

TEST_F(SafetyRuntimeHost, UsesPinnedAllocationForPinnedIoDescriptors)
{
    // Given pinned-host IO descriptors, when resources initialize and release, then their CUDA placement is preserved.
    gBackend.placement = nvinfer2::safe::MemoryPlacement::kCPU_PINNED;
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(ready(runner), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(gBackend.count("hostAlloc"), 4);
    ASSERT_EQ(teardownRunner(runner), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(gBackend.count("freeHost"), 4);
}

TEST_F(SafetyRuntimeHost, SharedExecutorSupportsMultipleStageContexts)
{
    // Given two stages sharing one executor, when both warm and run, then both contexts retain the same fixed buffers.
    auto init = spec();
    init.executors.resize(1);
    init.stages[1].executorIndex = 0;
    init.stages[0].profiles = {0};
    init.stages[1].profiles = {1};
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(runner.initialize(init), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(prepareRunner(runner, fillWarmup), ExpectedStatus::kSUCCESS);
    LMStageOutputs prefill;
    LMStageOutputs decode;
    ASSERT_EQ(runner.runLMStage("prefill", 0, prefill), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(runner.runLMStage("decode", 0, decode), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(prefill.context->profiles[0].resources->bindings[0].getData(),
        decode.context->profiles[0].resources->bindings[0].getData());
}

TEST_F(SafetyRuntimeHost, RequiresCompleteModelLocalViewsBeforeWarmup)
{
    // Given a required model-local buffer, when a context is completed, then warmup waits for its full borrowed view.
    auto init = fixedSpec();
    init.stages[0].modelLocalResources = {{"kv-cache", 32}};
    std::array<char, 32> cache{};
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(runner.initialize(init), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(
        runner.completeContext("prefill", 0, {{"kv-cache", cache.data(), 16}}), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(runner.prepare(fillWarmup), ExpectedStatus::kINVALID_STATE);
    ASSERT_EQ(
        runner.completeContext("prefill", 0, {{"kv-cache", cache.data(), cache.size()}}), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(prepareRunner(runner, fillWarmup), ExpectedStatus::kSUCCESS);
    LMStageOutputs outputs;
    ASSERT_EQ(runner.runLMStage("prefill", 0, outputs), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(outputs.context->modelLocalResources[0].data, cache.data());
}

TEST_F(SafetyRuntimeHost, RejectsResourceTeardownFromAnUnrelatedExecutor)
{
    // Given resources borrowed by one graph owner, when another owner requests teardown, then no buffers are released.
    TRTSafetyGraphExecutor owner(mErrorState);
    TRTSafetyGraphExecutor unrelated(mErrorState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(owner.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(owner.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(resources.prepareResources({&owner}), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(owner.deinitialize(), ExpectedStatus::kSUCCESS);
    mLifecycleState = LifecycleState::kCLOSED;
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(resources.deinitialize({&unrelated}), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    EXPECT_EQ(resources.deinitialize({&owner}), ExpectedStatus::kSUCCESS);
}

TEST_F(SafetyRuntimeHost, FailedExecutorMakesResourcePreparationTerminal)
{
    // Given an executor with a separate failed recorder, when it requests resources, then the resource owner becomes
    // fatal.
    Recorder failedRecorder;
    SafetyErrorState failedState(failedRecorder);
    TRTSafetyGraphExecutor failed(failedState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(failed.loadModules(), ExpectedStatus::kSUCCESS);
    gBackend.creationResult = ErrorCode::kFAILED_INITIALIZATION;
    ASSERT_EQ(failed.initialize(graphConfig()), ExpectedStatus::kFATAL_ERROR);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(resources.prepareResources({&failed}), ExpectedStatus::kFATAL_ERROR);
    EXPECT_TRUE(resources.fatal());
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}

TEST_F(SafetyRuntimeHost, UnsupportedIoTypeAllowsHealthyGraphTeardown)
{
    // Given an unsupported IO datatype, when preflight rejects resources, then initialized graphs still deinitialize
    // safely.
    gBackend.dataType = nvinfer2::safe::DataType::kBF16;
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    EXPECT_EQ(runner.initialize(spec()), ExpectedStatus::kINVALID_ARGUMENT);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(operationsSince(begin), (std::vector<std::string>{"destroy", "release", "destroy", "release"}));
}

TEST_F(SafetyRuntimeHost, RecorderFailureDuringCudaSetupStopsFurtherOperations)
{
    // Given an asynchronous recorder error during stream creation, when setup observes it, then no further SDK call
    // occurs.
    TRTSafetyGraphExecutor executor(mErrorState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    gBackend.cudaRecorder = &mRecorder;
    gBackend.reportOnCudaOperation = "streamCreate";
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(resources.prepareResources({&executor}), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(begin), (std::vector<std::string>{"streamCreate"}));
}

class SafetyInvalidBinding : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyInvalidBinding, RejectsIncompleteOrIncompatibleBindingBeforeSdkMutation)
{
    // Given incompatible buffers or auxiliary streams, when binding is requested, then it rejects before SDK mutation.
    TRTSafetyGraphExecutor executor(mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    std::array<float, 8> input{};
    std::array<float, 8> output{};
    std::array<int32_t, 8> incompatible{};
    char mainHandle{};
    char auxHandle{};
    auto main = reinterpret_cast<cudaStream_t>(&mainHandle);
    auto aux = reinterpret_cast<cudaStream_t>(&auxHandle);
    std::vector<cudaStream_t> auxiliary{aux};
    std::vector<nvinfer2::safe::TypedArray> bindings{{input.data(), sizeof(input)}, {output.data(), sizeof(output)}};
    switch (GetParam())
    {
    case 0: auxiliary.clear(); break;
    case 1: bindings[0] = nvinfer2::safe::TypedArray(incompatible.data(), sizeof(incompatible)); break;
    case 2: bindings[0] = nvinfer2::safe::TypedArray(input.data(), sizeof(float)); break;
    case 3: bindings[0] = nvinfer2::safe::TypedArray(static_cast<float*>(nullptr), sizeof(input)); break;
    }
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(executor.bindProfile(0, bindings, main, auxiliary), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(AuxTypeCapacityPointer, SafetyInvalidBinding, testing::Values(0, 1, 2, 3));

class SafetyTeardownFailure : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyTeardownFailure, StopsAtFirstFailedReleaseWithoutRetry)
{
    // Given graph or IO release failure, when deinitialization runs, then the first failed operation ends SDK cleanup.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(ready(runner), ExpectedStatus::kSUCCESS);
    if (GetParam() == 0)
    {
        gBackend.destructionResult = ErrorCode::kFAILED_EXECUTION;
    }
    else
    {
        gBackend.failCudaOperation = "free";
        gBackend.cudaResult = cudaErrorUnknown;
    }
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
    auto const expected = GetParam() == 0
        ? std::vector<std::string>{"destroy"}
        : std::vector<std::string>{"destroy", "release", "destroy", "release", "free"};
    EXPECT_EQ(operationsSince(begin), expected);
    size_t const stopped = gBackend.operations.size();
    EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(GraphAndIoRelease, SafetyTeardownFailure, testing::Values(0, 1));

class SafetyCpuDestruction : public SafetyRuntimeHost, public testing::WithParamInterface<bool>
{
};
TEST_P(SafetyCpuDestruction, DestructionCallsNoCudaOrDliApi)
{
    // Given healthy or partially failed owners, when C++ destruction runs, then no CUDA or DLI operation occurs.
    size_t stopped{0};
    {
        SafetyRuntime runtime(mRecorder);
        if (GetParam())
        {
            gBackend.failAllocationAt = 3;
            ASSERT_EQ(runtime.initialize(spec()), ExpectedStatus::kFATAL_ERROR);
        }
        else
        {
            ASSERT_EQ(ready(runtime), ExpectedStatus::kSUCCESS);
        }
        stopped = gBackend.operations.size();
    }
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(HealthyAndFailedOwners, SafetyCpuDestruction, testing::Values(false, true));

class SafetyExecutionFailure : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyExecutionFailure, LatchesFatalAndForbidsFurtherSdkCalls)
{
    // Given execute, sync, device-bitmap or recorder failure, when a stage runs, then it becomes terminal without
    // cleanup.
    TRTResourceManager resources(mLifecycleState, mErrorState);
    TRTSafetyRunner runner(mLifecycleState, mErrorState, resources);
    ASSERT_EQ(readyWithModelLocalBuffers(runner), ExpectedStatus::kSUCCESS);
    switch (GetParam())
    {
    case 0: gBackend.executionResult = ErrorCode::kFAILED_EXECUTION; break;
    case 1: gBackend.synchronizationResult = ErrorCode::kFAILED_EXECUTION; break;
    case 2: gBackend.runtimeErrorMask = 1U; break;
    case 3: gBackend.emitRecorderError = true; break;
    }
    LMStageOutputs outputs;
    EXPECT_EQ(runner.runLMStage("prefill", 0, outputs), ExpectedStatus::kFATAL_ERROR);
    EXPECT_TRUE(runner.fatal());
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(runner.runLMStage("decode", 0, outputs), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(teardownRunner(runner), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(SdkAndDeviceFailures, SafetyExecutionFailure, testing::Values(0, 1, 2, 3));

TEST_F(SafetyRuntimeHost, HealthyTeardownDestroysGraphsBeforeFreeingBorrowedBuffers)
{
    // Given a healthy prepared runner, when teardown runs, then all graph borrowers disappear before IO and streams
    // free.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(ready(runtime), ExpectedStatus::kSUCCESS);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(runtime.deinitialize(), ExpectedStatus::kSUCCESS);
    auto const names = operationsSince(begin);
    auto const firstFree = std::find(names.begin(), names.end(), "free");
    EXPECT_EQ(std::vector<std::string>(names.begin(), firstFree),
        (std::vector<std::string>{"destroy", "release", "destroy", "release"}));
    EXPECT_TRUE(gBackend.allocations.empty());
}

TEST_F(SafetyRuntimeHost, CapturedProfilesReplayChangingInputsWithFixedResources)
{
    // Given fixed IO and two captured profiles, when inputs and selected profiles change, then replay produces the
    // correct profile's outputs without Operational capture, rebinding, allocation or stream changes.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(readyCaptured(runtime), ExpectedStatus::kSUCCESS);
    std::vector<int64_t> capturedProfiles;
    for (auto const* executable : gBackend.cudaGraphExecutables)
    {
        for (auto const& invocation : executable->invocations)
        {
            capturedProfiles.push_back(invocation.profile);
        }
    }
    EXPECT_EQ(capturedProfiles, (std::vector<int64_t>{0, 1}));
    auto const bindings = bindingSnapshot(runtime);
    size_t const begin = gBackend.operations.size();
    std::array<LMStageId, 3> const stages{"decode", "prefill", "decode"};
    std::array<size_t, 3> const contexts{1, 0, 1};
    std::array<float, 3> const inputs{17.0F, 23.0F, 41.0F};
    std::array<float, 3> const expected{27.0F, 23.0F, 51.0F};
    for (size_t index = 0; index < stages.size(); ++index)
    {
        auto const& profile = runtime.contexts()[contexts[index]].profiles[0];
        profile.resources->bindings[0].getFloat()[0] = inputs[index];
        LMStageOutputs output;
        ASSERT_EQ(runtime.runLMStage(stages[index], 0, output), ExpectedStatus::kSUCCESS);
        EXPECT_FLOAT_EQ(profile.resources->bindings[1].getFloat()[0], expected[index]);
    }
    EXPECT_EQ(bindingSnapshot(runtime), bindings);
    EXPECT_EQ(operationsSince(begin),
        (std::vector<std::string>{"profile", "graphLaunch", "capturedExecute", "sync", "profile", "graphLaunch",
            "capturedExecute", "sync", "profile", "graphLaunch", "capturedExecute", "sync"}));
}

class SafetyCapturePreparationFailure : public SafetyRuntimeHost,
                                        public testing::WithParamInterface<std::tuple<std::string, bool>>
{
};
TEST_P(SafetyCapturePreparationFailure, PreventsReadinessAndFurtherSdkCalls)
{
    // Given capture-enabled initialization, when capture or readiness preparation fails, then the runtime becomes
    // terminal and performs no retry, fallback or later SDK cleanup.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(runtime.initialize(capturedSpec()), ExpectedStatus::kSUCCESS);
    auto const& operation = std::get<0>(GetParam());
    if (operation == "captureExecute")
    {
        if (std::get<1>(GetParam()))
        {
            gBackend.emitCapturedRecorderError = true;
        }
        else
        {
            gBackend.captureExecutionResult = ErrorCode::kFAILED_EXECUTION;
        }
    }
    else if (std::get<1>(GetParam()))
    {
        gBackend.cudaRecorder = &mRecorder;
        gBackend.reportOnCudaOperation = operation;
    }
    else
    {
        gBackend.failCudaOperation = operation;
        gBackend.cudaResult = cudaErrorUnknown;
    }
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(runtime.prepare(fillWarmup), ExpectedStatus::kFATAL_ERROR);
    std::vector<std::string> const preparation{"profile", "execute", "sync", "profile", "captureBegin", "execute",
        "captureEnd", "graphInstantiate", "graphUpload", "profile", "graphLaunch"};
    auto const captureBegin = std::find(preparation.begin(), preparation.end(), "captureBegin");
    auto const failed = operation == "captureExecute" ? std::find(captureBegin + 1, preparation.end(), "execute")
                                                      : std::find(preparation.begin(), preparation.end(), operation);
    std::vector<std::string> expected(preparation.begin(), failed + 1);
    if (operation == "graphLaunch" && std::get<1>(GetParam()))
    {
        expected.push_back("capturedExecute");
    }
    EXPECT_EQ(operationsSince(begin), expected);
    mRecorder.clear();
    EXPECT_TRUE(runtime.fatal());
    size_t const stopped = gBackend.operations.size();
    LMStageOutputs output;
    EXPECT_EQ((std::vector<SafetyStatus>{
                  runtime.prepare(fillWarmup), runtime.runLMStage("prefill", 0, output), runtime.deinitialize()}),
        (std::vector<ExpectedStatus>(3, ExpectedStatus::kFATAL_ERROR)));
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(CudaReturnAndRecorder, SafetyCapturePreparationFailure,
    testing::Combine(testing::Values("captureBegin", "captureExecute", "captureEnd", "graphInstantiate", "graphUpload",
                         "graphLaunch"),
        testing::Bool()));

class SafetyCapturedExecutionFailure : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyCapturedExecutionFailure, DoesNotPublishOutputsOrPermitFurtherSdkCalls)
{
    // Given a ready captured runtime, when launch, completion or error inspection fails, then execution becomes
    // terminal, publishes no output and permits no later SDK calls.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(readyCaptured(runtime), ExpectedStatus::kSUCCESS);
    switch (GetParam())
    {
    case 0:
        gBackend.failCudaOperation = "graphLaunch";
        gBackend.cudaResult = cudaErrorUnknown;
        break;
    case 1: gBackend.synchronizationResult = ErrorCode::kFAILED_EXECUTION; break;
    case 2: gBackend.runtimeErrorMask = 1U; break;
    case 3: gBackend.emitRecorderError = true; break;
    case 4:
        gBackend.cudaRecorder = &mRecorder;
        gBackend.reportOnCudaOperation = "graphLaunch";
        break;
    }
    int32_t const directExecutions = gBackend.count("execute");
    LMStageOutputs output{&runtime.contexts()[1]};
    EXPECT_EQ(runtime.runLMStage("decode", 0, output), ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(output.context, nullptr);
    EXPECT_EQ(gBackend.count("execute"), directExecutions);
    mRecorder.clear();
    EXPECT_TRUE(runtime.fatal());
    size_t const stopped = gBackend.operations.size();
    EXPECT_EQ((std::vector<SafetyStatus>{runtime.runLMStage("prefill", 0, output), runtime.deinitialize()}),
        (std::vector<ExpectedStatus>(2, ExpectedStatus::kFATAL_ERROR)));
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(
    LaunchCompletionBitmapAndRecorders, SafetyCapturedExecutionFailure, testing::Values(0, 1, 2, 3, 4));

class SafetyCapturedTeardown : public SafetyRuntimeHost,
                               public testing::WithParamInterface<std::tuple<std::string, bool>>
{
};
TEST_P(SafetyCapturedTeardown, ReleasesBorrowersFirstAndStopsAtFailure)
{
    // Given a captured runtime, when Deinitialization runs, then captured objects are released before DLI graphs and
    // IO, and the first release failure stops teardown.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(readyCaptured(runtime), ExpectedStatus::kSUCCESS);
    auto const& operation = std::get<0>(GetParam());
    if (std::get<1>(GetParam()))
    {
        gBackend.cudaRecorder = &mRecorder;
        gBackend.reportOnCudaOperation = operation;
    }
    else
    {
        gBackend.failCudaOperation = operation;
        gBackend.cudaResult = cudaErrorUnknown;
    }
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(runtime.deinitialize(), operation.empty() ? ExpectedStatus::kSUCCESS : ExpectedStatus::kFATAL_ERROR);
    auto const names = operationsSince(begin);
    if (operation.empty())
    {
        auto const firstFree = std::find(names.begin(), names.end(), "free");
        EXPECT_EQ(std::vector<std::string>(names.begin(), firstFree),
            (std::vector<std::string>{
                "graphExecDestroy", "graphDestroy", "graphExecDestroy", "graphDestroy", "destroy", "release"}));
        EXPECT_EQ((std::array<size_t, 4>{gBackend.cudaGraphExecutables.size(), gBackend.cudaGraphs.size(),
                      gBackend.graphs.size(), gBackend.allocations.size()}),
            (std::array<size_t, 4>{0, 0, 0, 0}));
    }
    else
    {
        EXPECT_EQ(names,
            operation == "graphExecDestroy" ? std::vector<std::string>{"graphExecDestroy"}
                                            : (std::vector<std::string>{"graphExecDestroy", "graphDestroy"}));
        mRecorder.clear();
        size_t const stopped = gBackend.operations.size();
        EXPECT_EQ(runtime.deinitialize(), ExpectedStatus::kFATAL_ERROR);
        EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
    }
}
INSTANTIATE_TEST_SUITE_P(HealthyCudaReturnAndRecorder, SafetyCapturedTeardown,
    testing::Values(std::make_tuple(std::string{}, false), std::make_tuple(std::string{"graphExecDestroy"}, false),
        std::make_tuple(std::string{"graphDestroy"}, false), std::make_tuple(std::string{"graphExecDestroy"}, true),
        std::make_tuple(std::string{"graphDestroy"}, true)));

class SafetyRejectedCall : public SafetyRuntimeHost, public testing::WithParamInterface<bool>
{
};
TEST_P(SafetyRejectedCall, PreservesHealthWithBoundedDiagnosticsAndSuppressedWarnings)
{
    // Given a healthy runtime, when an argument/state check rejects a call, then a bounded diagnostic is returned
    // without SDK execution or terminal failure; suppressing a warning does not change that.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(ready(runtime), ExpectedStatus::kSUCCESS);
    mRecorder.suppressWarnings(GetParam());
    SafetyErrorState validation(mRecorder);
    std::string const diagnostic = "stage rejection: " + std::string(512, 'x');
    size_t const begin = gBackend.operations.size();
    auto const rejected = validation.reject(SafetyStatusCode::kInvalidArgument, diagnostic.c_str());
    EXPECT_EQ(rejected, ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(std::string(rejected.message()), diagnostic.substr(0, 255));
    LMStageOutputs output{&runtime.contexts()[0]};
    EXPECT_EQ(runtime.runLMStage("unknown", 0, output), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(output.context, nullptr);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    EXPECT_FALSE(runtime.fatal());
    EXPECT_EQ(mRecorder.getNbErrors(), 0);
    EXPECT_EQ(mRecorder.warningAttempts(), 2);
    if (!GetParam())
    {
        EXPECT_NE(std::strstr(mRecorder.warningDescription(), "Stage"), nullptr);
    }
    runtime.contexts()[0].profiles[0].resources->bindings[0].getFloat()[0] = 19.0F;
    ASSERT_EQ(runtime.runLMStage("prefill", 0, output), ExpectedStatus::kSUCCESS);
    EXPECT_FLOAT_EQ(output.context->profiles[0].resources->bindings[1].getFloat()[0], 19.0F);
}
INSTANTIATE_TEST_SUITE_P(VisibleAndSuppressedWarnings, SafetyRejectedCall, testing::Bool());

TEST_F(SafetyRuntimeHost, ContainsExceptionDetailsWithoutChangingFailureSeverity)
{
    // Given an initialization/preparation exception, when it is contained, then the returned bounded diagnostic
    // includes its details and the existing failure severity is preserved.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(runtime.initialize(spec()), ExpectedStatus::kSUCCESS);
    std::string detail = "callback-detail: " + std::string(512, 'x');
    auto const status = runtime.prepare(throwWarmup, detail.data());
    EXPECT_EQ(status, ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(std::string(status.message()), ("Safety warmup callback threw an exception: " + detail).substr(0, 255));
    mRecorder.clear();
    size_t const stopped = gBackend.operations.size();
    EXPECT_TRUE(runtime.fatal());
    EXPECT_EQ((std::vector<SafetyStatus>{runtime.prepare(fillWarmup), runtime.deinitialize()}),
        (std::vector<ExpectedStatus>(2, ExpectedStatus::kFATAL_ERROR)));
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}

class SafetyCatalogPreparation : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyCatalogPreparation, PreparesSelectedFixedContextsAndSuppressesFailedOutputs)
{
    // Given catalog stages sharing one main stream, when prefill/decode execute through fixed stage selection, then
    // each preparation runs exactly once on the correct context/profile before execution, and preparation failure
    // publishes no output or inference result.
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(runtime.initialize(catalogSpec()), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(runtime.prepare(fillWarmup), ExpectedStatus::kSUCCESS);
    EXPECT_EQ(runtime.contexts()[0].mainStream, runtime.contexts()[1].mainStream);
    EXPECT_EQ(gBackend.count("streamCreate"), 3);
    auto const bindings = bindingSnapshot(runtime);
    StagePreparation preparation;
    LMStageOutputs outputs;
    std::array<char const*, 3> const stages{"prefill", "decode", "decode"};
    std::array<float, 3> const inputs{13.0F, 17.0F, 21.0F};
    std::array<float, 3> const expected{13.0F, 127.0F, 131.0F};
    size_t const begin = gBackend.operations.size();
    for (size_t index = 0; index < stages.size(); ++index)
    {
        preparation.input = inputs[index];
        ASSERT_EQ(runtime.runLMStage(stages[index], 0, outputs, prepareStage, &preparation), ExpectedStatus::kSUCCESS);
        EXPECT_EQ(outputs.context, &runtime.contexts()[index == 0 ? 0 : 1]);
        EXPECT_FLOAT_EQ(outputs.context->profiles[0].resources->bindings[1].getFloat()[0], expected[index]);
    }
    EXPECT_EQ(preparation.calls, 3U);
    EXPECT_EQ((std::array<LMStageContext const*, 3>{
                  preparation.contexts[0], preparation.contexts[1], preparation.contexts[2]}),
        (std::array<LMStageContext const*, 3>{&runtime.contexts()[0], &runtime.contexts()[1], &runtime.contexts()[1]}));
    EXPECT_EQ((std::array<int64_t, 3>{preparation.profiles[0], preparation.profiles[1], preparation.profiles[2]}),
        (std::array<int64_t, 3>{0, 1, 1}));
    EXPECT_EQ(operationsSince(begin),
        (std::vector<std::string>{"callback", "profile", "execute", "sync", "callback", "profile", "execute", "sync",
            "callback", "profile", "execute", "sync"}));
    EXPECT_EQ(bindingSnapshot(runtime), bindings);
    preparation.input = 31.0F;
    preparation.status = GetParam() == 0
        ? SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Rejected payload")
        : SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected preparation failure");
    if (GetParam() == 2)
    {
        preparation.status = SafetyStatus::ok();
        preparation.recorder = &mRecorder;
    }
    size_t const failureBegin = gBackend.operations.size();
    auto const failed = runtime.runLMStage("decode", 0, outputs, prepareStage, &preparation);
    EXPECT_EQ(failed, GetParam() == 0 ? ExpectedStatus::kINVALID_ARGUMENT : ExpectedStatus::kFATAL_ERROR);
    EXPECT_EQ(outputs.context, nullptr);
    EXPECT_EQ(preparation.calls, 4U);
    EXPECT_FLOAT_EQ(runtime.contexts()[1].profiles[0].resources->bindings[1].getFloat()[0], 131.0F);
    EXPECT_EQ(operationsSince(failureBegin), (std::vector<std::string>{"callback"}));
    if (GetParam() == 0)
    {
        ASSERT_EQ(runtime.deinitialize(), ExpectedStatus::kSUCCESS);
        EXPECT_EQ(gBackend.count("streamDestroy"), 3);
    }
    else
    {
        mRecorder.clear();
        size_t const stopped = gBackend.operations.size();
        EXPECT_TRUE(runtime.fatal());
        EXPECT_EQ(runtime.deinitialize(), ExpectedStatus::kFATAL_ERROR);
        EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
    }
}
INSTANTIATE_TEST_SUITE_P(RejectedFatalAndRecorderPreparation, SafetyCatalogPreparation, testing::Values(0, 1, 2));

TEST_F(SafetyRuntimeHost, RejectsAmbiguousCatalogProfilesBeforePreparation)
{
    // Given catalog stages sharing one main stream, when prefill/decode execute through fixed stage selection, then
    // each preparation runs exactly once on the correct context/profile before execution, and preparation failure
    // publishes no output or inference result.
    auto init = spec();
    init.executorMainStreamIds = {0, 0};
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(runtime.initialize(init), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(runtime.prepare(fillWarmup), ExpectedStatus::kSUCCESS);
    StagePreparation preparation;
    LMStageOutputs outputs{&runtime.contexts()[0]};
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(runtime.runLMStage("prefill", 0, outputs, prepareStage, &preparation), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(outputs.context, nullptr);
    EXPECT_EQ(preparation.calls, 0U);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
}

class SafetyCaptureAdmission : public SafetyRuntimeHost,
                               public testing::WithParamInterface<nvinfer2::safe::MemoryPlacement>
{
};
TEST_P(SafetyCaptureAdmission, RejectsHostPayloadsAndReplaysMutableDevicePayloads)
{
    // Given capture-enabled initialization with payload-bearing host IO, when admission runs, then it rejects before
    // capture; fixed GPU payloads still replay changing inputs correctly.
    gBackend.placement = GetParam();
    auto config = graphConfig();
    config.declaredProfiles = {0};
    config.enableCudaGraphCapture = true;
    TRTSafetyGraphExecutor executor(mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(config), ExpectedStatus::kSUCCESS);
    std::array<float, 8> input{7.0F};
    std::array<float, 8> output{};
    char mainHandle{};
    char auxHandle{};
    auto const main = reinterpret_cast<cudaStream_t>(&mainHandle);
    auto const auxiliary = reinterpret_cast<cudaStream_t>(&auxHandle);
    size_t const begin = gBackend.operations.size();
    auto const bound
        = executor.bindProfile(0, {{input.data(), sizeof(input)}, {output.data(), sizeof(output)}}, main, {auxiliary});
    if (GetParam() != nvinfer2::safe::MemoryPlacement::kGPU)
    {
        EXPECT_EQ(bound, ExpectedStatus::kINVALID_ARGUMENT);
        EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    }
    else
    {
        ASSERT_EQ(bound, ExpectedStatus::kSUCCESS);
        ASSERT_EQ(executor.warmup(0), ExpectedStatus::kSUCCESS);
        ASSERT_EQ(executor.seal(), ExpectedStatus::kSUCCESS);
        size_t const operationalBegin = gBackend.operations.size();
        for (float value : {23.0F, 41.0F})
        {
            input[0] = value;
            ASSERT_EQ(executor.execute(0), ExpectedStatus::kSUCCESS);
            EXPECT_FLOAT_EQ(output[0], value);
        }
        EXPECT_EQ(operationsSince(operationalBegin),
            (std::vector<std::string>{"profile", "graphLaunch", "capturedExecute", "sync", "profile", "graphLaunch",
                "capturedExecute", "sync"}));
    }
    EXPECT_EQ(executor.deinitialize(), ExpectedStatus::kSUCCESS);
}
INSTANTIATE_TEST_SUITE_P(PageablePinnedAndDevice, SafetyCaptureAdmission,
    testing::Values(nvinfer2::safe::MemoryPlacement::kCPU, nvinfer2::safe::MemoryPlacement::kCPU_PINNED,
        nvinfer2::safe::MemoryPlacement::kGPU));

TEST_F(SafetyRuntimeHost, ExactAliasGroupsShareOneOwnerAcrossProfiles)
{
    // If LWE aliasing is supported: given declared compatible alias groups, when initialization, execution and
    // teardown run, then members share correctly sized storage and numeric results, allocation is freed once, and
    // incompatible declarations reject before acquiring resources.
    useUniformDescriptors();
    TRTSafetyGraphExecutor executor(mErrorState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(resources.prepareResources({&executor}, {}, {{0, {"input", "output"}}}), ExpectedStatus::kSUCCESS);
    auto* shared = resources.profiles(0)[0].bindings[0].getFloat();
    EXPECT_EQ((std::array<void*, 3>{resources.profiles(0)[0].bindings[1].getData(),
                  resources.profiles(0)[1].bindings[0].getData(), resources.profiles(0)[1].bindings[1].getData()}),
        (std::array<void*, 3>{shared, shared, shared}));
    EXPECT_EQ(resources.profiles(0)[1].bindings[1].getSize(), 8 * sizeof(float));
    EXPECT_EQ(gBackend.count("malloc"), 1);
    for (int64_t profile : {0, 1})
    {
        shared[0] = 7.0F;
        ASSERT_EQ(executor.warmup(profile), ExpectedStatus::kSUCCESS);
    }
    ASSERT_EQ(executor.seal(), ExpectedStatus::kSUCCESS);
    shared[0] = 13.0F;
    ASSERT_EQ(executor.execute(0), ExpectedStatus::kSUCCESS);
    EXPECT_FLOAT_EQ(shared[0], 13.0F);
    shared[0] = 17.0F;
    ASSERT_EQ(executor.execute(1), ExpectedStatus::kSUCCESS);
    EXPECT_FLOAT_EQ(shared[0], 27.0F);
    ASSERT_EQ(executor.deinitialize(), ExpectedStatus::kSUCCESS);
    mLifecycleState = LifecycleState::kCLOSED;
    ASSERT_EQ(resources.deinitialize({&executor}), ExpectedStatus::kSUCCESS);
    std::vector<void*> freed;
    for (auto const& operation : gBackend.operations)
    {
        if (operation.name == "free")
        {
            freed.push_back(operation.address);
        }
    }
    EXPECT_EQ(freed, (std::vector<void*>{shared}));
}

class SafetyInvalidAlias : public SafetyRuntimeHost, public testing::WithParamInterface<int32_t>
{
};
TEST_P(SafetyInvalidAlias, RejectsIncompatibleDeclarationsBeforeAcquiringResources)
{
    // If LWE aliasing is supported: given declared compatible alias groups, when initialization, execution and
    // teardown run, then members share correctly sized storage and numeric results, allocation is freed once, and
    // incompatible declarations reject before acquiring resources.
    useUniformDescriptors();
    std::vector<SafetyIOAliasGroup> aliases{{0, {"input", "output"}}};
    for (auto& profile : gBackend.descriptors)
    {
        auto& output = profile[1];
        switch (GetParam())
        {
        case 0: aliases[0].tensors[1] = "unknown"; break;
        case 1: output.dataType = nvinfer2::safe::DataType::kINT32; break;
        case 2: output.memPlacement = nvinfer2::safe::MemoryPlacement::kCPU_PINNED; break;
        case 3: output.shape.d[0] = 7; break;
        case 4: output.stride.d[0] = 2; break;
        case 5: output.componentsPerVector = 2; break;
        case 6:
            gBackend.descriptors[1][0].sizeInBytes = 64;
            gBackend.descriptors[1][1].sizeInBytes = 64;
            break;
        case 7: aliases[0].tensors[1] = "input"; break;
        case 8: aliases.resize(2, aliases[0]); break;
        case 9: output.ioMode = nvinfer2::safe::TensorIOMode::kINPUT; break;
        case 10: aliases[0].executorIndex = 1; break;
        case 11: output.shape.nbDims = -1; break;
        case 12: output.userShape.d[0] = 7; break;
        }
    }
    TRTSafetyGraphExecutor executor(mErrorState);
    TRTResourceManager resources(mLifecycleState, mErrorState);
    ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
    ASSERT_EQ(executor.initialize(graphConfig()), ExpectedStatus::kSUCCESS);
    size_t const begin = gBackend.operations.size();
    EXPECT_EQ(resources.prepareResources({&executor}, {}, aliases), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    EXPECT_EQ(executor.deinitialize(), ExpectedStatus::kSUCCESS);
    mLifecycleState = LifecycleState::kCLOSED;
    EXPECT_EQ(resources.deinitialize({&executor}), ExpectedStatus::kSUCCESS);
}
INSTANTIATE_TEST_SUITE_P(NameTypePlacementShapeStrideVectorExtentDuplicatesModeAndOwner, SafetyInvalidAlias,
    testing::Values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

class SafetyCachedTensorViews : public SafetyRuntimeHost, public testing::WithParamInterface<bool>
{
};
TEST_P(SafetyCachedTensorViews, PreservesLayoutsAndStorageWhilePayloadsChange)
{
    // Given prebuilt tensor views, when Operational code looks them up and changes payloads, then layouts, capacities,
    // addresses and metadata stay fixed without allocation, rebinding or SDK metadata calls; zero-byte views remain
    // valid.
    useUniformDescriptors();
    for (size_t profile = 0; profile < gBackend.descriptors.size(); ++profile)
    {
        int64_t const rows = static_cast<int64_t>(2 + 2 * profile);
        auto& input = gBackend.descriptors[profile][0];
        input.userShape = {2, {rows, 3}};
        input.shape = {3, {rows, 1, 4}};
        input.stride = {3, {4, 4, 1}};
        input.strideOrder = {3, {2, 1, 0}};
        input.componentsPerVector = 4;
        input.vectorizedDim = 1;
        input.sizeInBytes = static_cast<uint64_t>(rows) * 4 * sizeof(float);
        if (GetParam())
        {
            auto& output = gBackend.descriptors[profile][1];
            output.shape = {1, {0}};
            output.userShape = {1, {0}};
            output.sizeInBytes = 0;
        }
    }
    SafetyRuntimeInitSpec const init{{graphConfig()}, {{"prefill", 0, 0, {0}, {}}, {"decode", 0, 0, {1}, {}}}};
    SafetyRuntime runtime(mRecorder);
    ASSERT_EQ(runtime.initialize(init), ExpectedStatus::kSUCCESS);
    SafetyTensorView const* initialized = nullptr;
    auto const initializationLookup = runtime.contexts()[0].getTensorView("input", initialized);
    auto const prepared = runtime.prepare(fillWarmup);
    auto const bindings = bindingSnapshot(runtime);
    size_t const begin = gBackend.operations.size();
    SafetyTensorView const* input = nullptr;
    SafetyTensorView const* large = nullptr;
    SafetyTensorView const* output = nullptr;
    ASSERT_EQ((std::vector<SafetyStatus>{initializationLookup, prepared,
                  runtime.contexts()[0].getTensorView(0, "input", input),
                  runtime.contexts()[1].getTensorView(1, "input", large),
                  runtime.contexts()[0].getTensorView("output", output)}),
        (std::vector<ExpectedStatus>(5, ExpectedStatus::kSUCCESS)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(large, nullptr);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(input, initialized);
    EXPECT_EQ(input->data, large->data);
    EXPECT_EQ(input->data, runtime.contexts()[0].profiles[0].resources->bindings[0].getData());
    static_cast<float*>(input->data)[0] = 23.0F;
    SafetyTensorView const* missing = input;
    EXPECT_EQ(runtime.contexts()[0].getTensorView("missing", missing), ExpectedStatus::kINVALID_ARGUMENT);
    EXPECT_EQ(missing, nullptr);
    EXPECT_EQ(operationsSince(begin), std::vector<std::string>{});
    LMStageOutputs selected;
    size_t const operationalBegin = gBackend.operations.size();
    ASSERT_EQ(runtime.runLMStage("prefill", 0, selected), ExpectedStatus::kSUCCESS);
    if (!GetParam())
    {
        EXPECT_FLOAT_EQ(static_cast<float const*>(output->data)[0], 23.0F);
    }
    EXPECT_EQ(std::tie(input->name, input->dataType, input->memPlacement, input->ioMode),
        std::make_tuple(std::string_view{"input"}, SafetyTensorDataType::kFLOAT, SafetyTensorMemoryPlacement::kGPU,
            SafetyTensorIOMode::kINPUT));
    EXPECT_EQ(std::tie(input->logicalRank, input->logicalShape, input->physicalRank, input->physicalShape),
        std::make_tuple(int32_t{2}, std::array<int64_t, 8>{2, 3}, int32_t{3}, std::array<int64_t, 9>{2, 1, 4}));
    EXPECT_EQ(std::tie(input->strideRank, input->strides, input->strideOrderRank, input->strideOrder),
        std::make_tuple(int32_t{3}, std::array<int64_t, 9>{4, 4, 1}, int32_t{3}, std::array<int64_t, 9>{2, 1, 0}));
    EXPECT_EQ(std::tie(input->bytesPerComponent, input->componentsPerVector, input->vectorizedDim, input->profileBytes,
                  input->capacityBytes),
        std::make_tuple(uint64_t{4}, uint64_t{4}, int64_t{1}, uint64_t{32}, uint64_t{64}));
    EXPECT_EQ(large->profileBytes, 64U);
    EXPECT_EQ(std::tie(output->data, output->profileBytes, output->capacityBytes, output->ioMode),
        std::make_tuple(GetParam() ? nullptr : runtime.contexts()[0].profiles[0].resources->bindings[1].getData(),
            GetParam() ? uint64_t{0} : uint64_t{32}, GetParam() ? uint64_t{0} : uint64_t{32},
            SafetyTensorIOMode::kOUTPUT));
    EXPECT_EQ(bindingSnapshot(runtime), bindings);
    EXPECT_EQ(operationsSince(operationalBegin), (std::vector<std::string>{"profile", "execute", "sync"}));
}
INSTANTIATE_TEST_SUITE_P(PayloadAndZeroByteMetadata, SafetyCachedTensorViews, testing::Bool());

class SafetyModulePreparationFailure : public SafetyRuntimeHost, public testing::WithParamInterface<bool>
{
};
TEST_P(SafetyModulePreparationFailure, RetainsDriverOrRecorderFailureBeforeGraphCreation)
{
    // Given loaded modules, when preparation returns a CUDA driver failure or reports a recorder error despite success,
    // initialization becomes terminal before graph creation, retains the failure category/details, and performs no
    // retry, cleanup or later SDK calls.
    size_t stopped{};
    {
        TRTSafetyGraphExecutor executor(mErrorState);
        ASSERT_EQ(executor.loadModules(), ExpectedStatus::kSUCCESS);
        if (GetParam())
        {
            gBackend.modulePrepareRecorder = &mRecorder;
        }
        else
        {
            gBackend.moduleResult = CUDA_ERROR_INVALID_VALUE;
        }
        FailureDetails const expected{GetParam() ? SafetyStatusCode::kFailedExecution : SafetyStatusCode::kCudaError,
            GetParam() ? "Safety recorder reported a failure" : "Safety attention module preparation failed", true};
        size_t const begin = gBackend.operations.size();
        auto const status = executor.initialize(graphConfig());
        EXPECT_EQ(failureDetails(status), expected);
        EXPECT_EQ(std::make_tuple(mRecorder.getErrorCode(0), std::string(mRecorder.getErrorDesc(0))),
            std::make_tuple(GetParam() ? ErrorCode::kINTERNAL_ERROR : ErrorCode::kFAILED_INITIALIZATION,
                std::string(GetParam() ? "Injected module preparation recorder failure"
                                       : "Safety attention module preparation failed")));
        EXPECT_FALSE(executor.hasGraph());
        EXPECT_EQ(operationsSince(begin), (std::vector<std::string>{"prepare"}));
        mRecorder.clear();
        gBackend.moduleResult = CUDA_SUCCESS;
        gBackend.modulePrepareRecorder = nullptr;
        stopped = gBackend.operations.size();
        EXPECT_EQ((std::array<FailureDetails, 3>{failureDetails(executor.initialize(graphConfig())),
                      failureDetails(executor.loadModules()), failureDetails(executor.deinitialize())}),
            (std::array<FailureDetails, 3>{expected, expected, expected}));
    }
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(DriverAndSuccessfulReturnWithRecorder, SafetyModulePreparationFailure, testing::Bool());

class SafetyWarmupStatusFailure : public SafetyRuntimeHost, public testing::WithParamInterface<std::tuple<bool, bool>>
{
};
TEST_P(SafetyWarmupStatusFailure, PreservesTypedCallbackFailureBeforeRecorderObservation)
{
    // Given a warmup callback returning a failed typed status, alone or while reporting a recorder error, readiness
    // returns a fatal status preserving the callback’s category/message and performs no graph warmup, sealing or later
    // SDK calls.
    size_t stopped{};
    {
        SafetyRuntime runtime(mRecorder);
        ASSERT_EQ(runtime.initialize(fixedSpec()), ExpectedStatus::kSUCCESS);
        constexpr auto kCODE = SafetyStatusCode::kFailedAllocation;
        char const* const detail = "Warmup input workspace is exhausted";
        WarmupFailure failure{
            std::get<0>(GetParam()) ? SafetyStatus::fatal(kCODE, detail) : SafetyStatus::error(kCODE, detail),
            std::get<1>(GetParam()) ? &mRecorder : nullptr};
        FailureDetails const expected{kCODE, detail, true};
        size_t const begin = gBackend.operations.size();
        EXPECT_EQ(failureDetails(runtime.prepare(failedWarmup, &failure)), expected);
        EXPECT_FLOAT_EQ(runtime.contexts()[0].profiles[0].resources->bindings[1].getFloat()[0], 0.0F);
        EXPECT_EQ(operationsSince(begin), (std::vector<std::string>{"warmupCallback"}));
        mRecorder.clear();
        stopped = gBackend.operations.size();
        EXPECT_TRUE(runtime.fatal());
        LMStageOutputs outputs{&runtime.contexts()[0]};
        EXPECT_EQ((std::array<FailureDetails, 4>{failureDetails(runtime.prepare(fillWarmup)),
                      failureDetails(runtime.runLMStage("prefill", 0, outputs)),
                      failureDetails(runtime.initialize(fixedSpec())), failureDetails(runtime.deinitialize())}),
            (std::array<FailureDetails, 4>{expected, expected, expected, expected}));
        EXPECT_EQ(outputs.context, nullptr);
    }
    EXPECT_EQ(operationsSince(stopped), std::vector<std::string>{});
}
INSTANTIATE_TEST_SUITE_P(
    NonfatalFatalAndRecorder, SafetyWarmupStatusFailure, testing::Combine(testing::Bool(), testing::Bool()));

} // namespace
} // namespace trt_edgellm::safety::runtime_test
