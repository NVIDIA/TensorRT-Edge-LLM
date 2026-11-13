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
#include "safety_inference/kernels/safetyAttentionModules.h"
#include "safety_inference/trt_runner/safetyGraphInitialization.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace trt_edgellm::safety::runtime_test
{
Backend gBackend;
using Factory = ErrorCode (*)(nvinfer2::safe::ITRTGraph*&, void const*, int64_t, char const*,
    nvinfer2::safe::ISafeRecorder&, bool, nvinfer2::safe::ISafeMemAllocator*) noexcept;
static_assert(std::is_same_v<decltype(&nvinfer2::safe::createTRTGraph), Factory>,
    "Host tests require the companion-library Safety graph factory ABI");
int32_t Recorder::getNbErrors() const noexcept
{
    return mErrorCount.load();
}
ErrorCode Recorder::getErrorCode(int32_t) const noexcept
{
    return mCode.load();
}
Recorder::ErrorDesc Recorder::getErrorDesc(int32_t) const noexcept
{
    return mErrorDescription.data();
}
bool Recorder::hasOverflowed() const noexcept
{
    return false;
}
void Recorder::clear() noexcept
{
    mErrorCount.store(0);
    mCode.store(ErrorCode::kSUCCESS);
    mErrorDescription.fill('\0');
    mWarningDescription.fill('\0');
    mWarningAttempts = 0;
}
bool Recorder::reportError(ErrorCode code, ErrorDesc description) noexcept
{
    size_t const length = std::min(std::strlen(description), mErrorDescription.size() - 1);
    std::copy_n(description, length, mErrorDescription.begin());
    mErrorDescription[length] = '\0';
    mCode.store(code);
    mErrorCount.fetch_add(1);
    return true;
}
bool Recorder::reportWarn(ErrorDesc description) noexcept
{
    ++mWarningAttempts;
    if (mWarningsSuppressed)
    {
        return false;
    }
    size_t const length = std::min(std::strlen(description), mWarningDescription.size() - 1);
    std::copy_n(description, length, mWarningDescription.begin());
    mWarningDescription[length] = '\0';
    return true;
}
void Recorder::suppressWarnings(bool suppressed) noexcept
{
    mWarningsSuppressed = suppressed;
}
int32_t Recorder::warningAttempts() const noexcept
{
    return mWarningAttempts;
}
Recorder::ErrorDesc Recorder::warningDescription() const noexcept
{
    return mWarningDescription.data();
}
Recorder::RefCount Recorder::incRefCount() noexcept
{
    return mReferences.fetch_add(1) + 1;
}
Recorder::RefCount Recorder::decRefCount() noexcept
{
    return mReferences.fetch_sub(1) - 1;
}
Recorder::RefCount Recorder::referenceCount() const noexcept
{
    return mReferences.load();
}
void Backend::reset() noexcept
{
    for (auto* executable : cudaGraphExecutables)
    {
        delete executable;
    }
    for (auto* graph : cudaGraphs)
    {
        delete graph;
    }
    for (void* graph : graphs)
    {
        delete static_cast<Graph*>(graph);
    }
    for (void* address : allocations)
    {
        std::free(address);
    }
    *this = Backend{};
}
int32_t Backend::count(char const* name) const noexcept
{
    return static_cast<int32_t>(std::count_if(
        operations.begin(), operations.end(), [name](Operation const& operation) { return operation.name == name; }));
}
Graph::Graph(int64_t id, nvinfer2::safe::ISafeRecorder& recorder) noexcept
    : mId(id)
    , mRecorder(recorder)
{
    (void) mRecorder.incRefCount();
    for (auto& bindings : mProfileBindings)
    {
        bindings.resize(mId == 1 && !gBackend.llmProfiles.empty() ? gBackend.llmProfiles[0].size() : 2);
    }
}
Graph::~Graph() noexcept
{
    (void) mRecorder.decRefCount();
}
ErrorCode Graph::clone(ITRTGraph*&, nvinfer2::safe::ISafeRecorder&) noexcept
{
    return ErrorCode::kINVALID_STATE;
}
ErrorCode Graph::getScratchMemorySize(size_t& size) const noexcept
{
    size = 64;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getTRTManagedScratch(bool& managed) const noexcept
{
    managed = true;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::setScratchMemory(void*) noexcept
{
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getScratchMemory(void*& memory) noexcept
{
    memory = nullptr;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getNbIOTensors(int64_t& count) const noexcept
{
    count = static_cast<int64_t>(mProfileBindings[0].size());
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getIOTensorName(char const*& name, size_t index) const noexcept
{
    name = mId == 1 && !gBackend.llmProfiles.empty() ? gBackend.llmProfiles[0][index].tensorName
                                                     : (index == 0 ? "input" : "output");
    return ErrorCode::kSUCCESS;
}
int32_t Graph::tensorIndex(char const* name) const noexcept
{
    for (std::size_t index = 0; index < mProfileBindings[0].size(); ++index)
    {
        char const* candidate{nullptr};
        getIOTensorName(candidate, index);
        if (std::strcmp(candidate, name) == 0)
        {
            return static_cast<int32_t>(index);
        }
    }
    return -1;
}
ErrorCode Graph::getIOTensorDescriptor(nvinfer2::safe::TensorDescriptor& descriptor, char const* name) const noexcept
{
    return getIOTensorDescriptor(descriptor, tensorIndex(name));
}
ErrorCode Graph::getIOTensorDescriptor(nvinfer2::safe::TensorDescriptor& descriptor, int32_t index) const noexcept
{
    if (mId == 1 && !gBackend.llmProfiles.empty())
    {
        descriptor = gBackend.llmProfiles.at(mProfile).at(index);
        return ErrorCode::kSUCCESS;
    }
    if (gBackend.useDescriptorOverrides)
    {
        descriptor = gBackend.descriptors.at(static_cast<size_t>(mProfile)).at(static_cast<size_t>(index));
        descriptor.tensorName = index == 0 ? "input" : "output";
        return ErrorCode::kSUCCESS;
    }
    descriptor = {};
    descriptor.tensorName = index == 0 ? "input" : "output";
    descriptor.shape = {1, {4 + 4 * mProfile}};
    descriptor.stride = {1, {1}};
    descriptor.dataType = gBackend.dataType;
    descriptor.bytesPerComponent = sizeof(float);
    descriptor.sizeInBytes = static_cast<uint64_t>(4 + 4 * mProfile) * sizeof(float);
    descriptor.ioMode = index == 0 ? nvinfer2::safe::TensorIOMode::kINPUT : nvinfer2::safe::TensorIOMode::kOUTPUT;
    descriptor.memPlacement = gBackend.placement;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::setIOTensorAddress(char const* name, nvinfer2::safe::TypedArray const& tensor) noexcept
{
    return setIOTensorAddress(tensorIndex(name), tensor);
}
ErrorCode Graph::setIOTensorAddress(int32_t index, nvinfer2::safe::TypedArray const& tensor) noexcept
{
    gBackend.operations.push_back({"bind", mId, mProfile, tensor.getData()});
    if (gBackend.bindingResult == ErrorCode::kSUCCESS)
    {
        mProfileBindings.at(static_cast<size_t>(mProfile)).at(static_cast<size_t>(index)) = tensor;
    }
    return gBackend.bindingResult;
}
ErrorCode Graph::getIOTensorAddress(char const* name, nvinfer2::safe::TypedArray& tensor) noexcept
{
    return getIOTensorAddress(tensorIndex(name), tensor);
}
ErrorCode Graph::getIOTensorAddress(int32_t index, nvinfer2::safe::TypedArray& tensor) noexcept
{
    tensor = mProfileBindings.at(static_cast<size_t>(mProfile)).at(static_cast<size_t>(index));
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::setInputConsumedEvent(cudaEvent_t) noexcept
{
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getInputConsumedEvent(cudaEvent_t& event) const noexcept
{
    event = nullptr;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getErrorBuffer(nvinfer2::safe::RuntimeErrorInformation*& buffer) const noexcept
{
    buffer = &mErrorBuffer;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getSafeRecorder(nvinfer2::safe::ISafeRecorder*& recorder) const noexcept
{
    recorder = &mRecorder;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getNbIOProfiles(int64_t& count) const noexcept
{
    count = 2;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::setIOProfile(int64_t profile) noexcept
{
    gBackend.operations.push_back({"profile", mId, profile});
    mProfile = profile;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getIOProfile(int64_t& profile) const noexcept
{
    profile = mProfile;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::getNbAuxStreams(int32_t& count) const noexcept
{
    count = 1;
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::setAuxStreams(cudaStream_t*, int32_t) noexcept
{
    gBackend.operations.push_back({"aux", mId, mProfile});
    return ErrorCode::kSUCCESS;
}
ErrorCode Graph::executeAsync(cudaStream_t stream) noexcept
{
    gBackend.operations.push_back({"execute", mId, mProfile, stream});
    auto const capture = std::find_if(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
        [stream](auto const* graph) { return graph->active && graph->stream == stream; });
    if (capture != gBackend.cudaGraphs.end())
    {
        if (gBackend.captureExecutionResult == ErrorCode::kSUCCESS)
        {
            (*capture)->invocations.push_back(
                {this, mId, mProfile, stream, mProfileBindings[static_cast<size_t>(mProfile)]});
        }
        if (gBackend.emitCapturedRecorderError)
        {
            mRecorder.reportError(ErrorCode::kUNSPECIFIED_ERROR, "Injected capture recorder error");
        }
        return gBackend.captureExecutionResult;
    }
    return applyExecution(mProfile, mProfileBindings[static_cast<size_t>(mProfile)]);
}
ErrorCode Graph::replay(CapturedInvocation const& invocation, cudaStream_t stream) noexcept
{
    gBackend.operations.push_back({"capturedExecute", mId, invocation.profile, stream});
    return applyExecution(invocation.profile, invocation.bindings);
}
ErrorCode Graph::applyExecution(int64_t profile, std::vector<nvinfer2::safe::TypedArray> const& bindings) noexcept
{
    if (gBackend.executionResult == ErrorCode::kSUCCESS)
    {
        bool const llm = mId == 1 && !gBackend.llmProfiles.empty();
        auto* input = llm ? nullptr : bindings[0].getFloat();
        auto* output = bindings[llm ? tensorIndex("logits") : 1].getFloat();
        if (llm && output != nullptr)
        {
            output[0] = static_cast<float>(bindings[tensorIndex("context_lengths")].getInt32()[0]);
        }
        if (input != nullptr && output != nullptr)
        {
            output[0] = input[0] + static_cast<float>(100 * mId + 10 * profile);
        }
        mErrorBuffer.bitMask = gBackend.runtimeErrorMask;
    }
    if (gBackend.emitRecorderError)
    {
        mRecorder.reportError(ErrorCode::kUNSPECIFIED_ERROR, "Injected asynchronous recorder error");
    }
    return gBackend.executionResult;
}
ErrorCode Graph::sync() noexcept
{
    gBackend.operations.push_back({"sync", mId, mProfile});
    if (std::any_of(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(), [this](auto const* capture) {
            return capture->active
                && std::any_of(capture->invocations.begin(), capture->invocations.end(),
                    [this](auto const& invocation) { return invocation.graph == this; });
        }))
    {
        return ErrorCode::kINVALID_STATE;
    }
    return gBackend.synchronizationResult;
}
} // namespace trt_edgellm::safety::runtime_test

namespace trt_edgellm::safety
{
CUresult SafetyAttentionModules::load() noexcept
{
    runtime_test::gBackend.operations.push_back({"load"});
    return runtime_test::gBackend.moduleResult;
}
CUresult SafetyAttentionModules::prepare() noexcept
{
    runtime_test::gBackend.operations.push_back({"prepare"});
    if (runtime_test::gBackend.modulePrepareRecorder != nullptr)
    {
        runtime_test::gBackend.modulePrepareRecorder->reportError(
            nvinfer1::ErrorCode::kINTERNAL_ERROR, "Injected module preparation recorder failure");
    }
    return runtime_test::gBackend.moduleResult;
}
CUresult SafetyAttentionModules::seal() noexcept
{
    runtime_test::gBackend.operations.push_back({"seal"});
    return runtime_test::gBackend.moduleResult;
}
CUresult SafetyAttentionModules::release() noexcept
{
    runtime_test::gBackend.operations.push_back({"release"});
    return runtime_test::gBackend.moduleResult;
}
bool SafetyAttentionModules::ready() const noexcept
{
    return runtime_test::gBackend.moduleResult == CUDA_SUCCESS;
}
nvinfer1::ErrorCode createSafetyGraph(SafetyAttentionModules&, nvinfer2::safe::ITRTGraph*& graph, void const*, int64_t,
    nvinfer2::safe::ISafeRecorder& recorder, char const*) noexcept
{
    auto& backend = runtime_test::gBackend;
    backend.operations.push_back({"create"});
    if (backend.creationResult == nvinfer1::ErrorCode::kSUCCESS)
    {
        graph = new runtime_test::Graph(backend.graphCount++, recorder);
        backend.graphs.push_back(graph);
    }
    return backend.creationResult;
}
} // namespace trt_edgellm::safety

namespace nvinfer2::safe
{
extern "C" ErrorCode destroyTRTGraph(ITRTGraph*& graph) noexcept
{
    auto& backend = trt_edgellm::safety::runtime_test::gBackend;
    backend.operations.push_back({"destroy"});
    if (backend.destructionResult == ErrorCode::kSUCCESS)
    {
        auto const found = std::find(backend.graphs.begin(), backend.graphs.end(), graph);
        if (found != backend.graphs.end())
        {
            backend.graphs.erase(found);
        }
        delete graph;
        graph = nullptr;
    }
    return backend.destructionResult;
}
} // namespace nvinfer2::safe

namespace
{
using trt_edgellm::safety::runtime_test::gBackend;
cudaError_t result(char const* operation, void* address = nullptr)
{
    gBackend.operations.push_back({operation, -1, -1, address});
    if (gBackend.cudaRecorder != nullptr && gBackend.reportOnCudaOperation == operation)
    {
        gBackend.cudaRecorder->reportError(nvinfer1::ErrorCode::kFAILED_EXECUTION, "Injected CUDA-side recorder error");
    }
    return gBackend.failCudaOperation == operation ? gBackend.cudaResult : cudaSuccess;
}
cudaError_t allocate(void** address, size_t size, char const* operation)
{
    auto const status = result(operation);
    int32_t const call = gBackend.allocationCalls++;
    if (status != cudaSuccess)
    {
        return status;
    }
    if (call == gBackend.failAllocationAt)
    {
        return cudaErrorMemoryAllocation;
    }
    *address = std::calloc(1, size);
    gBackend.allocations.push_back(*address);
    return cudaSuccess;
}
cudaError_t release(void* address, char const* operation)
{
    auto const status = result(operation, address);
    if (status == cudaSuccess)
    {
        auto const found = std::find(gBackend.allocations.begin(), gBackend.allocations.end(), address);
        if (found != gBackend.allocations.end())
        {
            gBackend.allocations.erase(found);
            std::free(address);
        }
    }
    return status;
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** address, size_t size)
{
    return allocate(address, size, "malloc");
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t* value, cudaLimit limit)
{
    gBackend.queriedLimits.push_back(limit);
    auto const status = result("deviceGetLimit");
    if (status == cudaSuccess)
    {
        *value = 1024;
    }
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** address, size_t size, unsigned int)
{
    return allocate(address, size, "hostAlloc");
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* address)
{
    return release(address, "free");
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* address)
{
    return release(address, "freeHost");
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
{
    gBackend.streamFlags.push_back(flags);
    void* address = nullptr;
    auto const status = allocate(&address, 1, "streamCreate");
    *stream = reinterpret_cast<cudaStream_t>(address);
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    return release(stream, "streamDestroy");
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    auto const status = result("streamSync", stream);
    if (status == cudaSuccess
        && std::any_of(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
            [stream](auto const* capture) { return capture->active && capture->stream == stream; }))
    {
        return cudaErrorStreamCaptureUnsupported;
    }
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    gBackend.captureModes.push_back(mode);
    auto const status = result("captureBegin", stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    if (stream == nullptr
        || std::any_of(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
            [stream](auto const* graph) { return graph->active && graph->stream == stream; }))
    {
        return cudaErrorInvalidValue;
    }
    gBackend.cudaGraphs.push_back(new trt_edgellm::safety::runtime_test::CapturedGraph{stream, true, {}});
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph)
{
    auto const status = result("captureEnd", stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const capture = std::find_if(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
        [stream](auto const* candidate) { return candidate->active && candidate->stream == stream; });
    if (graph == nullptr || capture == gBackend.cudaGraphs.end())
    {
        return cudaErrorInvalidValue;
    }
    (*capture)->active = false;
    *graph = reinterpret_cast<cudaGraph_t>(*capture);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(
    cudaGraphExec_t* executable, cudaGraph_t graph, unsigned long long flags)
{
    gBackend.graphInstantiationFlags.push_back(flags);
    auto const status = result("graphInstantiate", graph);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const capture = std::find(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
        reinterpret_cast<trt_edgellm::safety::runtime_test::CapturedGraph*>(graph));
    if (executable == nullptr || capture == gBackend.cudaGraphs.end() || (*capture)->active)
    {
        return cudaErrorInvalidValue;
    }
    auto* prepared = new trt_edgellm::safety::runtime_test::CapturedExecutable{(*capture)->invocations};
    gBackend.cudaGraphExecutables.push_back(prepared);
    *executable = reinterpret_cast<cudaGraphExec_t>(prepared);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(
    cudaGraphExec_t* executable, cudaGraph_t graph, unsigned long long flags)
{
    return cudaGraphInstantiateWithFlags(executable, graph, flags);
}
extern "C" cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t executable, cudaStream_t stream)
{
    auto const status = result("graphUpload", stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const prepared = std::find(gBackend.cudaGraphExecutables.begin(), gBackend.cudaGraphExecutables.end(),
        reinterpret_cast<trt_edgellm::safety::runtime_test::CapturedExecutable*>(executable));
    if (stream == nullptr || prepared == gBackend.cudaGraphExecutables.end())
    {
        return cudaErrorInvalidValue;
    }
    (*prepared)->uploaded = true;
    (*prepared)->uploadStream = stream;
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream)
{
    auto const status = result("graphLaunch", stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const prepared = std::find(gBackend.cudaGraphExecutables.begin(), gBackend.cudaGraphExecutables.end(),
        reinterpret_cast<trt_edgellm::safety::runtime_test::CapturedExecutable*>(executable));
    if (stream == nullptr || prepared == gBackend.cudaGraphExecutables.end())
    {
        return cudaErrorInvalidValue;
    }
    for (auto const& invocation : (*prepared)->invocations)
    {
        if (std::find(gBackend.graphs.begin(), gBackend.graphs.end(), invocation.graph) == gBackend.graphs.end())
        {
            return cudaErrorInvalidResourceHandle;
        }
        if (invocation.graph->replay(invocation, stream) != nvinfer1::ErrorCode::kSUCCESS)
        {
            return cudaErrorLaunchFailure;
        }
    }
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable)
{
    auto const status = result("graphExecDestroy", executable);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const prepared = std::find(gBackend.cudaGraphExecutables.begin(), gBackend.cudaGraphExecutables.end(),
        reinterpret_cast<trt_edgellm::safety::runtime_test::CapturedExecutable*>(executable));
    if (prepared == gBackend.cudaGraphExecutables.end())
    {
        return cudaErrorInvalidValue;
    }
    delete *prepared;
    gBackend.cudaGraphExecutables.erase(prepared);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph)
{
    auto const status = result("graphDestroy", graph);
    if (status != cudaSuccess)
    {
        return status;
    }
    auto const capture = std::find(gBackend.cudaGraphs.begin(), gBackend.cudaGraphs.end(),
        reinterpret_cast<trt_edgellm::safety::runtime_test::CapturedGraph*>(graph));
    if (capture == gBackend.cudaGraphs.end() || (*capture)->active)
    {
        return cudaErrorInvalidValue;
    }
    delete *capture;
    gBackend.cudaGraphs.erase(capture);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGetLastError()
{
    return result("lastError");
}
extern "C" char const* CUDARTAPI cudaGetErrorString(cudaError_t)
{
    return "Injected host test CUDA error";
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(
    void* destination, void const* source, size_t size, cudaMemcpyKind, cudaStream_t stream)
{
    auto const status = result("memcpy", stream);
    if (status == cudaSuccess)
    {
        std::memcpy(destination, source, size);
    }
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* destination, int value, size_t size, cudaStream_t stream)
{
    auto const status = result("memset", stream);
    if (status == cudaSuccess)
    {
        std::memset(destination, value, size);
    }
    return status;
}
