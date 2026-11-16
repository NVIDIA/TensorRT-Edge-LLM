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

#include "safetyEngineTestUtils.h"
#include "safetyTestLifecycle.h"
#include "safety_inference/runtime/safetyRuntime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace
{
using namespace nvinfer2::safe;
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::engine_test;

Recorder sRuntimeRecorder;
bool sSdkFailure{false};
constexpr int64_t kPROFILE{0};
constexpr size_t kSTAGES{2};
constexpr size_t kOUTPUT{5};
constexpr double kERROR_LIMIT{0.003};

struct HostFrame
{
    std::array<half, kELEMENTS> q{};
    std::array<half, kELEMENTS> k{};
    std::array<half, kELEMENTS> v{};
    std::array<int32_t, 3> sequenceEnds{0, 2, kTOKENS};
    std::array<int32_t, 3> shapeCarrier{};
    std::array<half, kELEMENTS> output{};
};

struct BindingView
{
    char const* name;
    DataType type;
    int32_t rank;
    std::array<int64_t, 3> dimensions;
    void* host;
    size_t bytes;
    void* data{nullptr};
    bool seen{false};
};

struct WarmupState
{
    HostFrame* frame;
    size_t executions{0};
    LMStageContext const* previous{nullptr};
    double maximumError{0.0};
    bool numericFailure{false};
};

bool healthy()
{
    return sRuntimeRecorder.getNbErrors() == 0 && !sRuntimeRecorder.hasOverflowed();
}

bool checkRuntime(SafetyStatus status, SafetyStatus expected, char const* operation)
{
    if (status.fatalStatus() || !healthy())
    {
        sSdkFailure = true;
    }
    bool const matches = expected.fatalStatus() ? !status.okStatus() && status.fatalStatus()
                                                : status.code() == expected.code() && !status.fatalStatus();
    if (!matches)
    {
        std::fprintf(stderr, "%s returned code=%d fatal=%d (%s), expected code=%d fatal=%d\n", operation,
            static_cast<int>(status.code()), status.fatalStatus(), status.message(), static_cast<int>(expected.code()),
            expected.fatalStatus());
        return false;
    }
    return healthy();
}

bool checkCuda(cudaError_t status, char const* operation)
{
    bool const success = check(status, operation) && healthy();
    sSdkFailure = sSdkFailure || !success;
    return success;
}

bool checkDriver(CUresult status, char const* operation)
{
    bool const success = check(status, operation) && healthy();
    sSdkFailure = sSdkFailure || !success;
    return success;
}

std::array<BindingView, 6> bindings(HostFrame& frame)
{
    size_t const attentionBytes = sizeof(frame.q);
    return {{{"q", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, frame.q.data(), attentionBytes},
        {"k", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, frame.k.data(), attentionBytes},
        {"v", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, frame.v.data(), attentionBytes},
        {"sequence_ends", DataType::kINT32, 1, {3, 0, 0}, frame.sequenceEnds.data(), sizeof(frame.sequenceEnds)},
        {"shape_carrier", DataType::kINT32, 1, {3, 0, 0}, frame.shapeCarrier.data(), sizeof(frame.shapeCarrier)},
        {"output", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, frame.output.data(), attentionBytes}}};
}

bool resolveBindings(LMStageContext const& context, HostFrame& frame, std::array<BindingView, 6>& output)
{
    if (context.mainStream == nullptr || context.profiles.size() != 1 || context.modelLocalResources.size() != 1
        || context.modelLocalResources[0].data != &frame || context.modelLocalResources[0].sizeInBytes != sizeof(frame))
    {
        return false;
    }
    auto const& profile = context.profiles[0];
    if (profile.metadata == nullptr || profile.resources == nullptr || profile.metadata->profileIndex != kPROFILE
        || profile.resources->profileIndex != kPROFILE
        || profile.metadata->tensors.size() != profile.resources->bindings.size())
    {
        return false;
    }
    output = bindings(frame);
    for (size_t index = 0; index < profile.metadata->tensors.size(); ++index)
    {
        auto const& tensor = profile.metadata->tensors[index];
        auto const& storage = profile.resources->bindings[index];
        auto const found = std::find_if(
            output.begin(), output.end(), [&tensor](auto const& binding) { return tensor.name == binding.name; });
        auto const& descriptor = tensor.descriptor;
        if (found == output.end() || found->seen || descriptor.dataType != found->type
            || storage.getType() != found->type || descriptor.shape.nbDims != found->rank)
        {
            return false;
        }
        found->seen = true;
        int64_t stride{1};
        for (int32_t dimension = found->rank - 1; dimension >= 0; --dimension)
        {
            if (descriptor.shape.d[dimension] != found->dimensions[dimension]
                || (descriptor.sizeInBytes != 0 && descriptor.stride.d[dimension] != stride))
            {
                return false;
            }
            stride *= found->dimensions[dimension];
        }
        if (tensor.name == "shape_carrier" && descriptor.sizeInBytes == 0)
        {
            found->bytes = 0;
            continue;
        }
        found->data = found->type == DataType::kHALF ? static_cast<void*>(storage.getHalf())
                                                     : static_cast<void*>(storage.getInt32());
        if (found->data == nullptr || storage.getSize() < found->bytes || descriptor.sizeInBytes != found->bytes
            || descriptor.vectorizedDim != -1
            || (descriptor.memPlacement != MemoryPlacement::kGPU && descriptor.memPlacement != MemoryPlacement::kNONE))
        {
            return false;
        }
    }
    return std::all_of(output.begin(), output.end(),
        [](auto const& binding) { return binding.seen || std::strcmp(binding.name, "shape_carrier") == 0; });
}

bool upload(LMStageContext const& context, HostFrame& frame, int32_t iteration)
{
    std::array<half*, 3> const inputBuffers{frame.q.data(), frame.k.data(), frame.v.data()};
    for (size_t input = 0; input < inputBuffers.size(); ++input)
    {
        for (int32_t index = 0; index < kELEMENTS; ++index)
        {
            inputBuffers[input][index] = __float2half(std::sin(index * (0.031F + input * 0.017F) + iteration * 0.2F));
        }
    }
    std::array<BindingView, 6> views{};
    if (!resolveBindings(context, frame, views))
    {
        return false;
    }
    for (size_t index = 0; index < kOUTPUT; ++index)
    {
        auto const& view = views[index];
        if (view.bytes != 0 && view.data != nullptr
            && !checkCuda(cudaMemcpyAsync(view.data, view.host, view.bytes, cudaMemcpyHostToDevice, context.mainStream),
                "cudaMemcpyAsync(input)"))
        {
            return false;
        }
    }
    // Poison the fixed output so a skipped graph execution cannot pass using a previous request's values.
    return checkCuda(cudaMemsetAsync(views[kOUTPUT].data, 0x7F, views[kOUTPUT].bytes, context.mainStream),
        "cudaMemsetAsync(output)");
}

bool compare(LMStageContext const& context, HostFrame& frame, double& maximumError)
{
    std::array<BindingView, 6> views{};
    if (!resolveBindings(context, frame, views)
        || !checkCuda(cudaMemcpyAsync(frame.output.data(), views[kOUTPUT].data, sizeof(frame.output),
                          cudaMemcpyDeviceToHost, context.mainStream),
            "cudaMemcpyAsync(output)")
        || !checkCuda(cudaStreamSynchronize(context.mainStream), "cudaStreamSynchronize(output)"))
    {
        return false;
    }
    for (int32_t token = 0; token < kTOKENS; ++token)
    {
        int32_t const begin = token < 2 ? 0 : 2;
        int32_t const end = token < 2 ? 2 : kTOKENS;
        for (int32_t head = 0; head < kHEADS; ++head)
        {
            std::array<double, kTOKENS> weights{};
            double weightSum{};
            for (int32_t key = begin; key < end; ++key)
            {
                double score{};
                for (int32_t dimension = 0; dimension < kHEAD_SIZE; ++dimension)
                {
                    score += __half2float(frame.q[(token * kHEADS + head) * kHEAD_SIZE + dimension])
                        * __half2float(frame.k[(key * kHEADS + head) * kHEAD_SIZE + dimension]);
                }
                weights[key] = std::exp(score * kSCALE);
                weightSum += weights[key];
            }
            for (int32_t dimension = 0; dimension < kHEAD_SIZE; ++dimension)
            {
                double sum{};
                for (int32_t key = begin; key < end; ++key)
                {
                    sum += weights[key] * __half2float(frame.v[(key * kHEADS + head) * kHEAD_SIZE + dimension]);
                }
                double const actual = __half2float(frame.output[(token * kHEADS + head) * kHEAD_SIZE + dimension]);
                double const error = std::abs(actual - sum / weightSum);
                if (!std::isfinite(actual) || error > kERROR_LIMIT)
                {
                    std::fprintf(stderr, "Runner output mismatch: stage=%s token=%d head=%d dimension=%d error=%.9g\n",
                        context.plan.id.c_str(), token, head, dimension, error);
                    return false;
                }
                maximumError = std::max(maximumError, error);
            }
        }
    }
    return true;
}

SafetyStatus warmup(LMStageContext const& context, int64_t profile, void* userData)
{
    auto& state = *static_cast<WarmupState*>(userData);
    if (state.previous != nullptr && !compare(*state.previous, *state.frame, state.maximumError))
    {
        if (sSdkFailure || !healthy())
        {
            return SafetyStatus::fatal(SafetyStatusCode::kFailedExecution);
        }
        state.numericFailure = true;
    }
    if (profile != kPROFILE || !upload(context, *state.frame, static_cast<int32_t>(state.executions)))
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedExecution);
    }
    ++state.executions;
    state.previous = &context;
    return SafetyStatus::ok();
}

//! Protect real runner dispatch, fixed resource views, warmup readiness, and healthy graph-before-resource teardown.
bool smoke(
    std::vector<char> const& engine, char const* companionPath, double& maximumError, bool enableCudaGraphCapture)
{
    TestLifecycle lifecycle;
    CUdevice device{};
    CUcontext cudaContext{};
    if (!lifecycle.initialize() || !checkDriver(cuInit(0), "cuInit")
        || !checkDriver(cuDeviceGet(&device, 0), "cuDeviceGet")
        || !checkDriver(createContext(&cuCtxCreate, cudaContext, device), "cuCtxCreate"))
    {
        return false;
    }
    SafetyRuntime runner(sRuntimeRecorder);
    SafetyRuntimeInitSpec const spec{
        {{engine.data(), static_cast<int64_t>(engine.size()), companionPath, {kPROFILE}, enableCudaGraphCapture}},
        {{"prefill", 0, 0, {kPROFILE}, {{"host_io", sizeof(HostFrame)}}},
            {"decode", 1, 0, {kPROFILE}, {{"host_io", sizeof(HostFrame)}}}}};
    // Given two declared stages sharing one real graph, when initializing, then publish incomplete fixed contexts.
    if (!checkRuntime(runner.initialize(spec), SafetyStatus::ok(), "SafetyRuntime::initialize")
        || runner.contexts().size() != kSTAGES)
    {
        return false;
    }
    void* hostStorage{};
    if (!checkCuda(cudaMallocHost(&hostStorage, sizeof(HostFrame)), "cudaMallocHost(frame)"))
    {
        return false;
    }
    auto* frame = new (hostStorage) HostFrame{};
    WarmupState warmupState{frame};
    if (!checkRuntime(runner.prepare(warmup, &warmupState), SafetyStatus::error(SafetyStatusCode::kInvalidState),
            "prepare(incomplete)"))
    {
        return false;
    }
    for (auto const& plan : spec.stages)
    {
        if (!checkRuntime(runner.completeContext(plan.id.c_str(), plan.slot, {{"host_io", frame, sizeof(*frame)}}),
                SafetyStatus::ok(), "SafetyRuntime::completeContext"))
        {
            return false;
        }
    }
    std::array<BindingView, 6> first{};
    std::array<BindingView, 6> second{};
    auto const& contexts = runner.contexts();
    cudaStream_t const initialStream = contexts[0].mainStream;
    if (!resolveBindings(contexts[0], *frame, first) || !resolveBindings(contexts[1], *frame, second)
        || contexts[0].mainStream != contexts[1].mainStream
        || contexts[0].profiles[0].resources != contexts[1].profiles[0].resources)
    {
        return false;
    }
    for (size_t index = 0; index < first.size(); ++index)
    {
        if (first[index].data != second[index].data)
        {
            return false;
        }
    }
    // Given complete fixed bindings, when preparing, then warm every stage before entering Operational.
    if (!checkRuntime(runner.prepare(warmup, &warmupState), SafetyStatus::ok(), "SafetyRuntime::prepare"))
    {
        return false;
    }
    bool passed = warmupState.executions == kSTAGES && !warmupState.numericFailure;
    if (!compare(contexts.back(), *frame, maximumError))
    {
        passed = false;
    }
    maximumError = std::max(maximumError, warmupState.maximumError);
    if (sSdkFailure || runner.fatal() || !healthy() || !prepareForOperational() || !healthy()
        || !lifecycle.operational())
    {
        return false;
    }
    LMStageOutputs outputs{&contexts[0]};
    if (passed)
    {
        passed = checkRuntime(runner.runLMStage("unknown", 0, outputs),
                     SafetyStatus::error(SafetyStatusCode::kInvalidArgument), "runLMStage(unknown)")
            && outputs.context == nullptr;
    }
    if (passed)
    {
        outputs.context = &contexts[0];
        passed = checkRuntime(runner.runLMStage("prefill", 1, outputs),
                     SafetyStatus::error(SafetyStatusCode::kInvalidArgument), "runLMStage(unassigned slot)")
            && outputs.context == nullptr;
    }
    // Given distinct requests, when dispatching both stages, then return stable contexts and matching numeric output.
    for (size_t stage = 0; passed && stage < contexts.size(); ++stage)
    {
        auto const& expected = contexts[stage];
        if (!upload(expected, *frame, static_cast<int32_t>(kSTAGES + stage))
            || !checkRuntime(runner.runLMStage(expected.plan.id, expected.plan.slot, outputs), SafetyStatus::ok(),
                "SafetyRuntime::runLMStage")
            || outputs.context != &expected || !compare(*outputs.context, *frame, maximumError))
        {
            passed = false;
            break;
        }
        std::array<BindingView, 6> current{};
        if (!resolveBindings(*outputs.context, *frame, current) || outputs.context->mainStream != initialStream
            || contexts[1].mainStream != initialStream)
        {
            passed = false;
            break;
        }
        for (size_t index = 0; index < first.size(); ++index)
        {
            if (current[index].data != first[index].data)
            {
                passed = false;
                break;
            }
        }
    }
    if (sSdkFailure || runner.fatal() || !healthy())
    {
        return false;
    }
    // CPU contract failures still permit healthy teardown; SDK failures retain resources for external recovery.
    bool const cleanedUp = lifecycle.deinitialize()
        && checkRuntime(runner.deinitialize(), SafetyStatus::ok(), "SafetyRuntime::deinitialize")
        && checkCuda(cudaFreeHost(hostStorage), "cudaFreeHost(frame)")
        && checkDriver(cuCtxDestroy(cudaContext), "cuCtxDestroy") && lifecycle.finish();
    return passed && cleanedUp;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 5)
    {
        std::fprintf(stderr,
            "Usage: %s matching-safety-engine.plan result.json [--external-lifecycle] [--cuda-graph]\n", argv[0]);
        return finishTestProcess(2);
    }
    bool externalLifecycle{false};
    bool enableCudaGraphCapture{false};
    for (int32_t index = 3; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--external-lifecycle") == 0 && !externalLifecycle)
        {
            externalLifecycle = true;
        }
        else if (std::strcmp(argv[index], "--cuda-graph") == 0 && !enableCudaGraphCapture)
        {
            enableCudaGraphCapture = true;
        }
        else
        {
            std::fprintf(stderr, "Unknown or repeated runner option: %s\n", argv[index]);
            return finishTestProcess(2);
        }
    }
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    if (!externalLifecycle)
    {
        std::fprintf(stderr, "QNX Safety requires --external-lifecycle and an exclusive VM lifecycle coordinator.\n");
        return finishTestProcess(2);
    }
#endif
    try
    {
        std::ofstream initialReport(argv[2], std::ios::trunc);
        initialReport << "{\"success\":false}\n";
        initialReport.close();
        if (!initialReport)
        {
            return finishTestProcess(1);
        }
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        auto const bytes = input.tellg();
        if (!input || bytes <= 0 || bytes > (64LL << 20))
        {
            return finishTestProcess(1);
        }
        std::vector<char> engine(static_cast<size_t>(bytes));
        input.seekg(0);
        if (!input.read(engine.data(), bytes))
        {
            return finishTestProcess(1);
        }
        double maximumError{};
        std::string const companionPath = std::filesystem::absolute(std::string(argv[1]) + ".so").string();
        if (!smoke(engine, companionPath.c_str(), maximumError, enableCudaGraphCapture))
        {
            return finishTestProcess(1);
        }
        std::ofstream report(argv[2], std::ios::trunc);
        report << "{\"success\":true,\"real_safety_graph\":true,\"runtime_classes_validated\":true,"
                  "\"plugin\":\"trt_edgellm::ViTAttentionPlugin:1\",\"cuda_graph_capture\":"
               << (enableCudaGraphCapture ? "true" : "false")
               << ",\"warmup_executions\":" << kSTAGES + static_cast<size_t>(enableCudaGraphCapture)
               << ",\"post_warmup_executions\":2,\"shared_fixed_buffers\":true,\"async_error_mask\":0,"
                  "\"max_abs_error\":"
               << maximumError << ",\"model_export_validated\":false,\"trt_header_version\":\"" << NV_TENSORRT_MAJOR
               << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
               << "\",\"qnx_lifecycle_verified\":" << (TestLifecycle::kQNX_SAFETY ? "true" : "false") << "}\n";
        report.close();
        return finishTestProcess(report ? 0 : 1);
    }
    catch (std::exception const& error)
    {
        std::fprintf(stderr, "Safety runner smoke failed: %s\n", error.what());
        return finishTestProcess(1);
    }
}
