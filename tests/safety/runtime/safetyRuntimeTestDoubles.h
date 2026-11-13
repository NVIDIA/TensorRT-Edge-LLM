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

#include <NvInferSafeRuntime.h>
#include <array>
#include <atomic>
#include <cuda.h>
#include <string>
#include <vector>

namespace trt_edgellm::safety::runtime_test
{

using nvinfer1::ErrorCode;

class Recorder : public nvinfer2::safe::ISafeRecorder
{
public:
    int32_t getNbErrors() const noexcept override;
    ErrorCode getErrorCode(int32_t index) const noexcept override;
    ErrorDesc getErrorDesc(int32_t index) const noexcept override;
    bool hasOverflowed() const noexcept override;
    void clear() noexcept override;
    bool reportError(ErrorCode code, ErrorDesc description) noexcept override;
    bool reportWarn(ErrorDesc description) noexcept override;
    RefCount incRefCount() noexcept override;
    RefCount decRefCount() noexcept override;
    RefCount referenceCount() const noexcept;
    void suppressWarnings(bool suppressed) noexcept;
    int32_t warningAttempts() const noexcept;
    ErrorDesc warningDescription() const noexcept;

private:
    std::atomic<int32_t> mErrorCount{0};
    std::atomic<ErrorCode> mCode{ErrorCode::kSUCCESS};
    std::atomic<int32_t> mReferences{0};
    std::array<char, 256> mErrorDescription{};
    std::array<char, 256> mWarningDescription{};
    int32_t mWarningAttempts{0};
    bool mWarningsSuppressed{false};
};

struct Operation
{
    std::string name;
    int64_t graph{-1};
    int64_t profile{-1};
    void* address{nullptr};
};

class Graph;

struct CapturedInvocation
{
    Graph* graph{nullptr};
    int64_t graphId{-1};
    int64_t profile{-1};
    cudaStream_t stream{nullptr};
    std::vector<nvinfer2::safe::TypedArray> bindings;
};

struct CapturedGraph
{
    cudaStream_t stream{nullptr};
    bool active{false};
    std::vector<CapturedInvocation> invocations;
};

struct CapturedExecutable
{
    std::vector<CapturedInvocation> invocations;
    bool uploaded{false};
    cudaStream_t uploadStream{nullptr};
};

struct Backend
{
    std::vector<Operation> operations;
    std::vector<void*> allocations;
    std::vector<void*> graphs;
    std::vector<CapturedGraph*> cudaGraphs;
    std::vector<CapturedExecutable*> cudaGraphExecutables;
    std::vector<unsigned int> streamFlags;
    std::vector<cudaStreamCaptureMode> captureModes;
    std::vector<unsigned long long> graphInstantiationFlags;
    std::vector<cudaLimit> queriedLimits;
    ErrorCode creationResult{ErrorCode::kSUCCESS};
    ErrorCode executionResult{ErrorCode::kSUCCESS};
    ErrorCode captureExecutionResult{ErrorCode::kSUCCESS};
    ErrorCode destructionResult{ErrorCode::kSUCCESS};
    ErrorCode bindingResult{ErrorCode::kSUCCESS};
    ErrorCode synchronizationResult{ErrorCode::kSUCCESS};
    cudaError_t cudaResult{cudaSuccess};
    std::string failCudaOperation;
    int32_t failAllocationAt{-1};
    int32_t allocationCalls{0};
    int32_t graphCount{0};
    uint64_t runtimeErrorMask{0};
    bool emitRecorderError{false};
    bool emitCapturedRecorderError{false};
    CUresult moduleResult{CUDA_SUCCESS};
    nvinfer2::safe::ISafeRecorder* modulePrepareRecorder{nullptr};
    nvinfer2::safe::MemoryPlacement placement{nvinfer2::safe::MemoryPlacement::kGPU};
    nvinfer2::safe::DataType dataType{nvinfer2::safe::DataType::kFLOAT};
    bool useDescriptorOverrides{false};
    std::array<std::array<nvinfer2::safe::TensorDescriptor, 2>, 2> descriptors{};
    //! Optional second-graph metadata for full model-local factory integration.
    std::vector<std::vector<nvinfer2::safe::TensorDescriptor>> llmProfiles{};
    nvinfer2::safe::ISafeRecorder* cudaRecorder{nullptr};
    std::string reportOnCudaOperation;

    void reset() noexcept;
    int32_t count(char const* name) const noexcept;
};

extern Backend gBackend;

//! Implements the genuine SDK vtable; SDK additions intentionally break host-test compilation.
class Graph : public nvinfer2::safe::ITRTGraph
{
public:
    Graph(int64_t id, nvinfer2::safe::ISafeRecorder& recorder) noexcept;
    ~Graph() noexcept override;
    ErrorCode clone(ITRTGraph*& graph, nvinfer2::safe::ISafeRecorder& recorder) noexcept override;
    ErrorCode getScratchMemorySize(size_t& size) const noexcept override;
    ErrorCode getTRTManagedScratch(bool& managed) const noexcept override;
    ErrorCode setScratchMemory(void* memory) noexcept override;
    ErrorCode getScratchMemory(void*& memory) noexcept override;
    ErrorCode getNbIOTensors(int64_t& count) const noexcept override;
    ErrorCode getIOTensorName(char const*& name, size_t index) const noexcept override;
    ErrorCode getIOTensorDescriptor(
        nvinfer2::safe::TensorDescriptor& descriptor, char const* name) const noexcept override;
    ErrorCode getIOTensorDescriptor(
        nvinfer2::safe::TensorDescriptor& descriptor, int32_t index) const noexcept override;
    ErrorCode setIOTensorAddress(char const* name, nvinfer2::safe::TypedArray const& tensor) noexcept override;
    ErrorCode setIOTensorAddress(int32_t index, nvinfer2::safe::TypedArray const& tensor) noexcept override;
    ErrorCode getIOTensorAddress(char const* name, nvinfer2::safe::TypedArray& tensor) noexcept override;
    ErrorCode getIOTensorAddress(int32_t index, nvinfer2::safe::TypedArray& tensor) noexcept override;
    ErrorCode setInputConsumedEvent(cudaEvent_t event) noexcept override;
    ErrorCode getInputConsumedEvent(cudaEvent_t& event) const noexcept override;
    ErrorCode getErrorBuffer(nvinfer2::safe::RuntimeErrorInformation*& buffer) const noexcept override;
    ErrorCode getSafeRecorder(nvinfer2::safe::ISafeRecorder*& recorder) const noexcept override;
    ErrorCode getNbIOProfiles(int64_t& count) const noexcept override;
    ErrorCode setIOProfile(int64_t profile) noexcept override;
    ErrorCode getIOProfile(int64_t& profile) const noexcept override;
    ErrorCode getNbAuxStreams(int32_t& count) const noexcept override;
    ErrorCode setAuxStreams(cudaStream_t* streams, int32_t count) noexcept override;
    ErrorCode executeAsync(cudaStream_t stream) noexcept override;
    ErrorCode sync() noexcept override;
    ErrorCode replay(CapturedInvocation const& invocation, cudaStream_t stream) noexcept;

private:
    ErrorCode applyExecution(int64_t profile, std::vector<nvinfer2::safe::TypedArray> const& bindings) noexcept;
    int32_t tensorIndex(char const* name) const noexcept;
    int64_t mId;
    int64_t mProfile{0};
    nvinfer2::safe::ISafeRecorder& mRecorder;
    mutable nvinfer2::safe::RuntimeErrorInformation mErrorBuffer{};
    std::array<std::vector<nvinfer2::safe::TypedArray>, 2> mProfileBindings;
};

} // namespace trt_edgellm::safety::runtime_test
