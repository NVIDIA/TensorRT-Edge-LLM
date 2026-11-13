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

#include "trtSafetyGraphExecutor.h"

#include "safetyGraphInitialization.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace safety
{
using nvinfer2::safe::ErrorCode;

namespace
{
void* tensorAddress(nvinfer2::safe::TypedArray const& tensor) noexcept
{
    using nvinfer2::safe::DataType;
    void* address{nullptr};
    switch (tensor.getType())
    {
    case DataType::kFLOAT: address = tensor.getFloat(); break;
    case DataType::kHALF: address = tensor.getHalf(); break;
    case DataType::kINT64: address = tensor.getInt64(); break;
    case DataType::kINT32: address = tensor.getInt32(); break;
    case DataType::kINT8: address = tensor.getInt8(); break;
    case DataType::kUINT8: address = tensor.getUint8(); break;
    case DataType::kBOOL: address = tensor.getBool(); break;
    case DataType::kFP8:
    case DataType::kBF16:
    case DataType::kINT4:
    case DataType::kFP4:
    case DataType::kE8M0: break;
    }
    return address;
}
} // namespace

TRTSafetyGraphExecutor::TRTSafetyGraphExecutor(SafetyErrorState& errorState) noexcept
    : mErrorState(errorState)
{
}

bool TRTSafetyGraphExecutor::fatal() const noexcept
{
    return mErrorState.fatal();
}

bool TRTSafetyGraphExecutor::hasGraph() const noexcept
{
    return mGraph != nullptr;
}

bool TRTSafetyGraphExecutor::observeErrors() noexcept
{
    return mErrorState.observe();
}

SafetyStatus TRTSafetyGraphExecutor::fail(ErrorCode code, char const* message) noexcept
{
    return mErrorState.fail(code, message);
}

bool TRTSafetyGraphExecutor::updateStatus(ErrorCode status, char const* message) noexcept
{
    return mErrorState.updateStatus(fromTrtError(status, message), status);
}

bool TRTSafetyGraphExecutor::updateStatus(cudaError_t status, char const* message) noexcept
{
    return mErrorState.updateStatus(fromCudaError(status, message), ErrorCode::kFAILED_EXECUTION);
}

bool TRTSafetyGraphExecutor::updateStatus(CUresult status, char const* message) noexcept
{
    return mErrorState.updateStatus(fromCudaDriverError(status, message), ErrorCode::kFAILED_INITIALIZATION);
}

std::vector<SafetyProfileMetadata> const& TRTSafetyGraphExecutor::profiles() const noexcept
{
    return mProfiles;
}

int32_t TRTSafetyGraphExecutor::requiredAuxStreams() const noexcept
{
    return mRequiredAuxStreams;
}

size_t TRTSafetyGraphExecutor::findProfile(int64_t profile) const noexcept
{
    auto const found = std::find_if(mProfiles.begin(), mProfiles.end(),
        [profile](SafetyProfileMetadata const& metadata) { return metadata.profileIndex == profile; });
    return static_cast<size_t>(std::distance(mProfiles.begin(), found));
}

SafetyStatus TRTSafetyGraphExecutor::loadModules() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kEMPTY)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Modules can be loaded only once");
    }
    if (!updateStatus(mModules.load(), "Safety attention module loading failed"))
    {
        return mErrorState.status();
    }
    mState = State::kMODULES_LOADED;
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyGraphExecutor::initialize(SafetyGraphConfig const& config) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kMODULES_LOADED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Graph initialization requires loaded modules");
    }
    if (config.engine == nullptr || config.engineBytes <= 0 || config.companionSoPath == nullptr
        || config.companionSoPath[0] != '/' || config.declaredProfiles.empty())
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Invalid graph engine, companion library or profiles");
    }
    for (size_t i = 0; i < config.declaredProfiles.size(); ++i)
    {
        int64_t const profile = config.declaredProfiles[i];
        if (profile < 0
            || std::find(config.declaredProfiles.begin(), config.declaredProfiles.begin() + i, profile)
                != config.declaredProfiles.begin() + i)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Graph profiles must be unique and nonnegative");
        }
    }
    try
    {
        mProfiles.reserve(config.declaredProfiles.size());
        mProfileStates.resize(config.declaredProfiles.size());
        mEnableCudaGraphCapture = config.enableCudaGraphCapture;
        if (!(updateStatus(mModules.prepare(), "Safety attention module preparation failed")
                && updateStatus(createSafetyGraph(mModules, mGraph, config.engine, config.engineBytes,
                                    mErrorState.recorder(), config.companionSoPath),
                    "Safety graph creation failed")))
        {
            return mErrorState.status();
        }
        if (mGraph == nullptr)
        {
            return fail(ErrorCode::kFAILED_INITIALIZATION, "Safety graph creation returned no graph");
        }
        int64_t count{0};
        if (!updateStatus(mGraph->getNbIOProfiles(count), "Safety graph profile query failed"))
        {
            return mErrorState.status();
        }
        for (int64_t const profile : config.declaredProfiles)
        {
            if (profile >= count)
            {
                return fail(ErrorCode::kINVALID_CONFIG, "Declared Safety IO profile is absent from the engine");
            }
        }
        if (!updateStatus(mGraph->getNbAuxStreams(mRequiredAuxStreams), "Safety graph auxiliary stream query failed"))
        {
            return mErrorState.status();
        }
        if (mRequiredAuxStreams < 0)
        {
            return fail(ErrorCode::kINVALID_CONFIG, "Safety graph returned a negative auxiliary stream count");
        }
        if (!updateStatus(mGraph->getErrorBuffer(mErrorBuffer), "Safety graph device error buffer query failed"))
        {
            return mErrorState.status();
        }
        if (mErrorBuffer == nullptr)
        {
            return fail(ErrorCode::kINVALID_CONFIG, "Safety graph returned no device error buffer");
        }
        for (int64_t const profile : config.declaredProfiles)
        {
            int64_t tensorCount{0};
            if (!(updateStatus(mGraph->setIOProfile(profile), "Safety graph metadata profile selection failed")
                    && updateStatus(mGraph->getNbIOTensors(tensorCount), "Safety graph tensor count query failed")))
            {
                return mErrorState.status();
            }
            if (tensorCount <= 0 || tensorCount > std::numeric_limits<int32_t>::max())
            {
                return fail(ErrorCode::kINVALID_CONFIG, "Safety graph returned an invalid IO tensor count");
            }
            SafetyProfileMetadata metadata;
            metadata.profileIndex = profile;
            metadata.tensors.reserve(static_cast<size_t>(tensorCount));
            for (int32_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
            {
                nvinfer2::safe::TensorDescriptor descriptor;
                if (!updateStatus(mGraph->getIOTensorDescriptor(descriptor, tensorIndex),
                        "Safety graph tensor descriptor query failed"))
                {
                    return mErrorState.status();
                }
                if (descriptor.tensorName == nullptr || descriptor.tensorName[0] == '\0')
                {
                    return fail(ErrorCode::kINVALID_CONFIG, "Safety graph returned an unnamed IO tensor");
                }
                metadata.tensors.push_back({descriptor.tensorName, descriptor});
            }
            mProfiles.push_back(std::move(metadata));
        }
        mState = State::kINITIALIZED;
        return SafetyStatus::ok();
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            ErrorCode::kFAILED_ALLOCATION, "Safety graph initialization host allocation failed", error.what());
    }
    catch (...)
    {
        return fail(ErrorCode::kFAILED_ALLOCATION, "Safety graph initialization host allocation failed");
    }
}

SafetyStatus TRTSafetyGraphExecutor::bindProfile(int64_t profile,
    std::vector<nvinfer2::safe::TypedArray> const& bindings, cudaStream_t main,
    std::vector<cudaStream_t> const& aux) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kINITIALIZED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Binding requires an initialized graph");
    }
    size_t const index = findProfile(profile);
    if (index == mProfiles.size() || main == nullptr || aux.size() != static_cast<size_t>(mRequiredAuxStreams))
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Invalid profile, main stream or auxiliary stream count");
    }
    if (mProfileStates[index].bound)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Profile IO has already been bound");
    }
    if (mMainStream != nullptr && (mMainStream != main || mAuxStreams != aux))
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Executor streams must remain fixed across profiles");
    }
    for (size_t i = 0; i < aux.size(); ++i)
    {
        if (aux[i] == nullptr || aux[i] == main || std::find(aux.begin(), aux.begin() + i, aux[i]) != aux.begin() + i)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Auxiliary streams must be distinct and nondefault");
        }
    }
    auto const& tensors = mProfiles[index].tensors;
    if (bindings.size() != tensors.size())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Binding count must match tensor metadata");
    }
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        auto const& descriptor = tensors[i].descriptor;
        if (mEnableCudaGraphCapture && descriptor.sizeInBytes != 0U
            && descriptor.memPlacement != nvinfer2::safe::MemoryPlacement::kGPU
            && descriptor.memPlacement != nvinfer2::safe::MemoryPlacement::kNONE)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "CUDA graph capture requires device-resident IO payloads");
        }
        if ((tensors[i].descriptor.sizeInBytes != 0U && tensorAddress(bindings[i]) == nullptr)
            || bindings[i].getType() != tensors[i].descriptor.dataType
            || bindings[i].getSize() < tensors[i].descriptor.sizeInBytes)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Binding pointer, dtype or capacity is incompatible with metadata");
        }
    }
    try
    {
        if (mMainStream == nullptr)
        {
            mAuxStreams = aux;
            if (!updateStatus(mGraph->setAuxStreams(mAuxStreams.empty() ? nullptr : mAuxStreams.data(),
                                  static_cast<int32_t>(mAuxStreams.size())),
                    "Safety graph auxiliary stream binding failed"))
            {
                return mErrorState.status();
            }
            mMainStream = main;
        }
        if (!updateStatus(mGraph->setIOProfile(profile), "Safety graph binding profile selection failed"))
        {
            return mErrorState.status();
        }
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            if (tensors[i].descriptor.sizeInBytes == 0U)
            {
                continue;
            }
            if (!updateStatus(mGraph->setIOTensorAddress(static_cast<int32_t>(i), bindings[i]),
                    "Safety graph IO tensor binding failed"))
            {
                return mErrorState.status();
            }
        }
        mProfileStates[index].bound = true;
        return SafetyStatus::ok();
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            ErrorCode::kFAILED_ALLOCATION, "Safety graph binding host allocation failed", error.what());
    }
    catch (...)
    {
        return fail(ErrorCode::kFAILED_ALLOCATION, "Safety graph binding host allocation failed");
    }
}

SafetyStatus TRTSafetyGraphExecutor::capture(size_t profile) noexcept
{
    auto& prepared = mProfileStates[profile];
    if (!(updateStatus(mGraph->setIOProfile(mProfiles[profile].profileIndex), "Safety graph capture profile failed")
            && updateStatus(
                cudaStreamBeginCapture(mMainStream, cudaStreamCaptureModeGlobal), "Safety CUDA capture failed")
            && updateStatus(mGraph->executeAsync(mMainStream), "Safety graph capture execution failed")
            && updateStatus(
                cudaStreamEndCapture(mMainStream, &prepared.capturedGraph), "Safety CUDA capture end failed")))
    {
        return mErrorState.status();
    }
    if (prepared.capturedGraph == nullptr)
    {
        return fail(ErrorCode::kFAILED_INITIALIZATION, "Safety CUDA capture returned no graph");
    }
    // Safe CUDA retains the five-argument instantiation API.
#if defined(EDGELLM_USE_SAFE_CUDA)
    auto const instantiateStatus
        = cudaGraphInstantiate(&prepared.executableGraph, prepared.capturedGraph, nullptr, nullptr, 0U);
#else
    auto const instantiateStatus = cudaGraphInstantiate(&prepared.executableGraph, prepared.capturedGraph, 0);
#endif
    if (!updateStatus(instantiateStatus, "Safety CUDA graph instantiation failed"))
    {
        return mErrorState.status();
    }
    if (prepared.executableGraph == nullptr)
    {
        return fail(ErrorCode::kFAILED_INITIALIZATION, "Safety CUDA graph instantiation returned no executable");
    }
    if (!updateStatus(cudaGraphUpload(prepared.executableGraph, mMainStream), "Safety CUDA graph upload failed"))
    {
        return mErrorState.status();
    }
    return run(profile, true);
}

SafetyStatus TRTSafetyGraphExecutor::run(size_t profile, bool captured) noexcept
{
    if (mErrorBuffer->bitMask != 0U)
    {
        return fail(ErrorCode::kFAILED_EXECUTION, "Safety graph device error was reported before execution");
    }
    if (!updateStatus(mGraph->setIOProfile(mProfiles[profile].profileIndex), "Safety graph execution profile failed"))
    {
        return mErrorState.status();
    }
    if (captured)
    {
        auto const& prepared = mProfileStates[profile];
        if (prepared.capturedGraph == nullptr || prepared.executableGraph == nullptr)
        {
            return fail(ErrorCode::kINVALID_STATE, "Safety CUDA graph is not prepared");
        }
    }
    bool const submitted = captured
        ? updateStatus(
              cudaGraphLaunch(mProfileStates[profile].executableGraph, mMainStream), "Safety CUDA graph launch failed")
        : updateStatus(mGraph->executeAsync(mMainStream), "Safety graph asynchronous execution failed");
    if (!(submitted && updateStatus(mGraph->sync(), "Safety graph synchronization failed")))
    {
        return mErrorState.status();
    }
    // LWE owns a pinned host error buffer; GPU completion establishes when the host may inspect it.
    if (mErrorBuffer->bitMask != 0U)
    {
        return fail(ErrorCode::kFAILED_EXECUTION, "Safety graph asynchronous device error was reported");
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyGraphExecutor::warmup(int64_t profile) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kINITIALIZED)
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidState, "Warmup requires initialized graph and bound profile");
    }
    size_t const index = findProfile(profile);
    if (index == mProfiles.size())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Warmup profile is not admitted or bound");
    }
    if (!mProfileStates[index].bound)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Warmup profile IO has not been bound");
    }
    bool const captured = mEnableCudaGraphCapture && mProfileStates[index].executableGraph != nullptr;
    auto status = run(index, captured);
    if (status.okStatus() && mEnableCudaGraphCapture && !captured)
    {
        // Speculative decoding requires revisiting this fixed-LMStage boundary for draft and verification stages.
        status = capture(index);
    }
    if (status.okStatus())
    {
        mProfileStates[index].warmed = true;
    }
    return status;
}

SafetyStatus TRTSafetyGraphExecutor::seal() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kINITIALIZED
        || std::any_of(mProfileStates.begin(), mProfileStates.end(), [this](ProfileState const& profile) {
               return !profile.bound || !profile.warmed
                   || (mEnableCudaGraphCapture
                       && (profile.capturedGraph == nullptr || profile.executableGraph == nullptr));
           }))
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState,
            "Readiness requires an initialized graph and every admitted profile warmed");
    }
    if (!updateStatus(mModules.seal(), "Safety attention module readiness failed"))
    {
        return mErrorState.status();
    }
    mState = State::kSEALED;
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyGraphExecutor::execute(int64_t profile) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kSEALED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Execution requires a sealed graph");
    }
    size_t const index = findProfile(profile);
    if (index == mProfiles.size())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Execution profile is not admitted or warmed");
    }
    return run(index, mEnableCudaGraphCapture);
}

SafetyStatus TRTSafetyGraphExecutor::deinitialize() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mState != State::kMODULES_LOADED && mState != State::kINITIALIZED && mState != State::kSEALED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Graph is already closed");
    }
    for (auto& profile : mProfileStates)
    {
        if (profile.executableGraph != nullptr)
        {
            if (!updateStatus(
                    cudaGraphExecDestroy(profile.executableGraph), "Safety CUDA executable graph release failed"))
            {
                return mErrorState.status();
            }
            profile.executableGraph = nullptr;
        }
        if (profile.capturedGraph != nullptr)
        {
            if (!updateStatus(cudaGraphDestroy(profile.capturedGraph), "Safety CUDA graph release failed"))
            {
                return mErrorState.status();
            }
            profile.capturedGraph = nullptr;
        }
    }
    if (mGraph != nullptr)
    {
        if (!updateStatus(nvinfer2::safe::destroyTRTGraph(mGraph), "Safety graph destruction failed"))
        {
            return mErrorState.status();
        }
        if (mGraph != nullptr)
        {
            return fail(ErrorCode::kINVALID_STATE, "Safety graph destruction retained the graph");
        }
    }
    if (!updateStatus(mModules.release(), "Safety attention module release failed"))
    {
        return mErrorState.status();
    }
    mErrorBuffer = nullptr;
    mMainStream = nullptr;
    mAuxStreams.clear();
    // Stage contexts retain profile metadata while caller-owned resources finish teardown.
    mProfileStates.clear();
    mRequiredAuxStreams = 0;
    mEnableCudaGraphCapture = false;
    mState = State::kCLOSED;
    return SafetyStatus::ok();
}

} // namespace safety
} // namespace trt_edgellm
