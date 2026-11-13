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

#include "trtResourceManager.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <numeric>

namespace trt_edgellm
{
namespace safety
{
namespace
{
using namespace nvinfer2::safe;

bool makeTypedArray(DataType type, void* data, uint64_t bytes, TypedArray& array) noexcept
{
    switch (type)
    {
    case DataType::kFLOAT: array = TypedArray(static_cast<float*>(data), bytes); break;
    case DataType::kHALF: array = TypedArray(static_cast<half_t*>(data), bytes); break;
    case DataType::kINT64: array = TypedArray(static_cast<int64_t*>(data), bytes); break;
    case DataType::kINT32: array = TypedArray(static_cast<int32_t*>(data), bytes); break;
    case DataType::kINT8: array = TypedArray(static_cast<int8_t*>(data), bytes); break;
    case DataType::kUINT8: array = TypedArray(static_cast<uint8_t*>(data), bytes); break;
    case DataType::kBOOL: array = TypedArray(static_cast<bool*>(data), bytes); break;
    default: return false;
    }
    return true;
}

bool validPlacement(MemoryPlacement placement) noexcept
{
    return placement == MemoryPlacement::kGPU || placement == MemoryPlacement::kCPU
        || placement == MemoryPlacement::kCPU_PINNED || placement == MemoryPlacement::kNONE;
}

MemoryPlacement actualPlacement(MemoryPlacement placement) noexcept
{
    return placement == MemoryPlacement::kNONE ? MemoryPlacement::kGPU : placement;
}

size_t tensorIndex(SafetyProfileMetadata const& profile, std::string const& name) noexcept
{
    auto const found = std::find_if(
        profile.tensors.begin(), profile.tensors.end(), [&name](auto const& tensor) { return tensor.name == name; });
    return static_cast<size_t>(found - profile.tensors.begin());
}

template <typename Dims>
bool sameDimensions(Dims const& lhs, Dims const& rhs) noexcept
{
    constexpr auto kMAX_RANK = sizeof(lhs.d) / sizeof(lhs.d[0]);
    return lhs.nbDims >= -1 && lhs.nbDims <= static_cast<int32_t>(kMAX_RANK) && lhs.nbDims == rhs.nbDims
        && (lhs.nbDims < 0 || std::equal(lhs.d, lhs.d + lhs.nbDims, rhs.d));
}

bool sameLayout(TensorDescriptor const& lhs, TensorDescriptor const& rhs) noexcept
{
    return lhs.dataType == rhs.dataType && actualPlacement(lhs.memPlacement) == actualPlacement(rhs.memPlacement)
        && lhs.sizeInBytes == rhs.sizeInBytes && lhs.bytesPerComponent == rhs.bytesPerComponent
        && lhs.componentsPerVector == rhs.componentsPerVector && lhs.vectorizedDim == rhs.vectorizedDim
        && sameDimensions(lhs.shape, rhs.shape) && sameDimensions(lhs.stride, rhs.stride)
        && sameDimensions(lhs.strideOrder, rhs.strideOrder) && sameDimensions(lhs.userShape, rhs.userShape);
}

constexpr size_t kIO_ALIGNMENT{256};
} // namespace

TRTResourceManager::TRTResourceManager(LifecycleState& lifecycleState, SafetyErrorState& errorState) noexcept
    : mErrorState(errorState)
    , mLifecycleState(lifecycleState)
{
}

bool TRTResourceManager::isHealthy() const noexcept
{
    return mLifecycleState != LifecycleState::kFAILED && !mErrorState.fatal() && dependenciesHealthy();
}

bool TRTResourceManager::dependenciesHealthy() const noexcept
{
    return std::none_of(mExecutors.begin(), mExecutors.end(),
        [](auto const& owned) { return owned.executor != nullptr && owned.executor->fatal(); });
}

bool TRTResourceManager::fatal() const noexcept
{
    return !isHealthy();
}

bool TRTResourceManager::observeErrors() noexcept
{
    if (!isHealthy())
    {
        mErrorState.setFatal(
            SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety dependency reported a failure"));
        return false;
    }
    return true;
}

bool TRTResourceManager::observeDependencies() noexcept
{
    if (!dependenciesHealthy())
    {
        mErrorState.setFatal(
            SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety dependency reported a failure"));
        return false;
    }
    return true;
}

bool TRTResourceManager::updateStatus(cudaError_t status, nvinfer2::safe::ErrorCode code, char const* message) noexcept
{
    return mErrorState.updateStatus(fromCudaError(status, message), code);
}

SafetyStatus TRTResourceManager::fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept
{
    return mErrorState.fail(code, message);
}

SafetyStatus TRTResourceManager::prepareResources(std::vector<TRTSafetyGraphExecutor*> const& executors,
    std::vector<size_t> const& mainStreamIds, std::vector<SafetyIOAliasGroup> const& aliases) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || !mExecutors.empty())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Resources can be prepared only once");
    }
    auto status = validateExecutors(executors);
    if (!status.okStatus())
    {
        return status;
    }
    if (!mainStreamIds.empty())
    {
        if (mainStreamIds.size() != executors.size())
        {
            return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Each executor requires a main stream ID");
        }
        size_t const count = *std::max_element(mainStreamIds.begin(), mainStreamIds.end()) + 1;
        if (count == 0 || count > executors.size())
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Main stream IDs must be bounded and contiguous");
        }
        for (size_t id = 0; id < count; ++id)
        {
            if (std::find(mainStreamIds.begin(), mainStreamIds.end(), id) == mainStreamIds.end())
            {
                return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Main stream IDs must be contiguous");
            }
        }
    }
    status = validateAliases(executors, aliases);
    if (!status.okStatus())
    {
        return status;
    }
    status = planStorage(executors, mainStreamIds, aliases);
    if (!status.okStatus())
    {
        return status;
    }
    for (auto& owned : mExecutors)
    {
        if (!(createStreams(owned).okStatus() && allocateIO(owned).okStatus() && bindIO(owned).okStatus()))
        {
            return mErrorState.status();
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::validateExecutors(std::vector<TRTSafetyGraphExecutor*> const& executors) noexcept
{
    if (executors.empty())
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Resource preparation requires at least one executor");
    }
    for (size_t index = 0; index < executors.size(); ++index)
    {
        auto const* executor = executors[index];
        if (executor != nullptr && executor->fatal())
        {
            return fail(
                nvinfer2::safe::ErrorCode::kINVALID_STATE, "Safety IO resources requested for a failed executor");
        }
        if (executor == nullptr || !executor->hasGraph() || executor->profiles().empty()
            || executor->requiredAuxStreams() < 0)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Executor is null, uninitialized or has invalid stream metadata");
        }
        for (size_t previous = 0; previous < index; ++previous)
        {
            if (executors[previous] == executor)
            {
                return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Resource executors must be unique");
            }
        }
        auto const& first = executor->profiles().front();
        for (auto const& profile : executor->profiles())
        {
            if (profile.tensors.size() != first.tensors.size())
            {
                return mErrorState.reject(
                    SafetyStatusCode::kInvalidArgument, "Tensor counts must match across admitted profiles");
            }
            for (size_t tensorIndex = 0; tensorIndex < profile.tensors.size(); ++tensorIndex)
            {
                auto const& tensor = profile.tensors[tensorIndex];
                auto const& descriptor = tensor.descriptor;
                nvinfer2::safe::TypedArray unused;
                if (!makeTypedArray(descriptor.dataType, nullptr, 0, unused) || !validPlacement(descriptor.memPlacement)
                    || tensor.name != first.tensors[tensorIndex].name
                    || descriptor.dataType != first.tensors[tensorIndex].descriptor.dataType
                    || descriptor.memPlacement != first.tensors[tensorIndex].descriptor.memPlacement
                    || descriptor.sizeInBytes > std::numeric_limits<size_t>::max() - kIO_ALIGNMENT)
                {
                    return mErrorState.reject(SafetyStatusCode::kInvalidArgument,
                        "Tensor dtype, placement, name or capacity is incompatible");
                }
            }
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::validateAliases(
    std::vector<TRTSafetyGraphExecutor*> const& executors, std::vector<SafetyIOAliasGroup> const& aliases) noexcept
{
    for (size_t groupIndex = 0; groupIndex < aliases.size(); ++groupIndex)
    {
        auto const& group = aliases[groupIndex];
        if (group.executorIndex >= executors.size() || group.tensors.size() < 2)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "IO aliases require one executor and at least two tensors");
        }
        auto const& profiles = executors[group.executorIndex]->profiles();
        auto const& first = profiles.front();
        size_t const owner = tensorIndex(first, group.tensors.front());
        if (owner == first.tensors.size())
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "IO alias tensor is not declared by the graph");
        }
        auto const bytes = first.tensors[owner].descriptor.sizeInBytes;
        bool input{false};
        bool output{false};
        for (size_t member = 0; member < group.tensors.size(); ++member)
        {
            auto const& name = group.tensors[member];
            size_t const index = tensorIndex(first, name);
            if (index == first.tensors.size()
                || std::find(group.tensors.begin(), group.tensors.begin() + member, name)
                    != group.tensors.begin() + member)
            {
                return mErrorState.reject(
                    SafetyStatusCode::kInvalidArgument, "IO alias members must name unique declared tensors");
            }
            for (size_t previous = 0; previous < groupIndex; ++previous)
            {
                auto const& prior = aliases[previous];
                if (prior.executorIndex == group.executorIndex
                    && std::find(prior.tensors.begin(), prior.tensors.end(), name) != prior.tensors.end())
                {
                    return mErrorState.reject(
                        SafetyStatusCode::kInvalidArgument, "An IO tensor cannot belong to multiple alias groups");
                }
            }
            auto const mode = first.tensors[index].descriptor.ioMode;
            input = input || mode == TensorIOMode::kINPUT;
            output = output || mode == TensorIOMode::kOUTPUT;
            for (auto const& profile : profiles)
            {
                auto const& descriptor = profile.tensors[index].descriptor;
                // ASIL interval checking admits exact reuse, not partial or changing interval extents.
                if (bytes == 0 || descriptor.sizeInBytes != bytes || descriptor.shape.nbDims < 0
                    || descriptor.stride.nbDims < 0 || descriptor.ioMode != first.tensors[index].descriptor.ioMode
                    || actualPlacement(descriptor.memPlacement) != MemoryPlacement::kGPU
                    || !sameLayout(descriptor, profile.tensors[owner].descriptor))
                {
                    return mErrorState.reject(SafetyStatusCode::kInvalidArgument,
                        "IO aliases require identical device layouts and byte extents across admitted profiles");
                }
            }
        }
        if (!input || !output)
        {
            return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "IO aliases require an input and an output");
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::planStorage(std::vector<TRTSafetyGraphExecutor*> const& executors,
    std::vector<size_t> const& mainStreamIds, std::vector<SafetyIOAliasGroup> const& aliases) noexcept
{
    try
    {
        mExecutors.resize(executors.size());
        size_t const streamCount = mainStreamIds.empty()
            ? executors.size()
            : *std::max_element(mainStreamIds.begin(), mainStreamIds.end()) + 1;
        mMainStreams.resize(streamCount, nullptr);
        // Finish container allocation before acquiring device resources so partial handles always have an owner.
        for (size_t index = 0; index < executors.size(); ++index)
        {
            auto& owned = mExecutors[index];
            owned.executor = executors[index];
            owned.mainStreamId = mainStreamIds.empty() ? index : mainStreamIds[index];
            owned.auxiliary.resize(static_cast<size_t>(executors[index]->requiredAuxStreams()), nullptr);
            auto const& metadata = executors[index]->profiles().front();
            owned.allocations.resize(metadata.tensors.size());
            owned.tensorAllocations.resize(metadata.tensors.size());
            std::iota(owned.tensorAllocations.begin(), owned.tensorAllocations.end(), size_t{0});
            for (auto const& group : aliases)
            {
                if (group.executorIndex == index)
                {
                    size_t const owner = tensorIndex(metadata, group.tensors.front());
                    for (auto const& name : group.tensors)
                    {
                        owned.tensorAllocations[tensorIndex(metadata, name)] = owner;
                    }
                }
            }
            for (auto const& profile : executors[index]->profiles())
            {
                owned.profiles.push_back({profile.profileIndex, {}});
                owned.profiles.back().bindings.resize(profile.tensors.size());
                for (size_t tensorIndex = 0; tensorIndex < profile.tensors.size(); ++tensorIndex)
                {
                    auto const& tensor = profile.tensors[tensorIndex];
                    auto& allocation = owned.allocations[owned.tensorAllocations[tensorIndex]];
                    // LWE treats unspecified IO placement as device memory.
                    allocation.placement = tensor.descriptor.memPlacement == MemoryPlacement::kNONE
                        ? MemoryPlacement::kGPU
                        : tensor.descriptor.memPlacement;
                    allocation.bytes = std::max(allocation.bytes, tensor.descriptor.sizeInBytes);
                }
            }
        }
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety IO metadata allocation failed", error.what());
    }
    catch (...)
    {
        return fail(nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety IO metadata allocation failed");
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::createStreams(ExecutorResources& owned) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    auto& main = mMainStreams[owned.mainStreamId];
    if (main == nullptr
        && !(updateStatus(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking),
                 nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety main stream creation failed")
            && observeDependencies()))
    {
        return mErrorState.status();
    }
    owned.main = main;
    for (auto& stream : owned.auxiliary)
    {
        if (!(updateStatus(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety auxiliary stream creation failed")
                && observeDependencies()))
        {
            return mErrorState.status();
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::allocateIO(ExecutorResources& owned) noexcept
{
    for (auto& allocation : owned.allocations)
    {
        if (allocation.bytes != 0)
        {
            cudaError_t status{cudaSuccess};
            if (allocation.placement == nvinfer2::safe::MemoryPlacement::kGPU)
            {
                status = cudaMalloc(&allocation.data, static_cast<size_t>(allocation.bytes));
            }
            else if (allocation.placement == nvinfer2::safe::MemoryPlacement::kCPU_PINNED)
            {
                status = cudaHostAlloc(&allocation.data, static_cast<size_t>(allocation.bytes), cudaHostAllocDefault);
            }
            else
            {
                size_t const alignedBytes
                    = (static_cast<size_t>(allocation.bytes) + kIO_ALIGNMENT - 1) / kIO_ALIGNMENT * kIO_ALIGNMENT;
                allocation.data = std::aligned_alloc(kIO_ALIGNMENT, alignedBytes);
            }
            if (!updateStatus(status, nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety IO allocation failed"))
            {
                return mErrorState.status();
            }
            if (allocation.data == nullptr)
            {
                return fail(nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety IO buffer allocation failed");
            }
            if (!observeDependencies())
            {
                return mErrorState.status();
            }
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::bindIO(ExecutorResources& owned) noexcept
{
    for (size_t profileIndex = 0; profileIndex < owned.profiles.size(); ++profileIndex)
    {
        auto& profile = owned.profiles[profileIndex];
        auto const& metadata = owned.executor->profiles()[profileIndex];
        for (size_t tensorIndex = 0; tensorIndex < metadata.tensors.size(); ++tensorIndex)
        {
            auto const& allocation = owned.allocations[owned.tensorAllocations[tensorIndex]];
            makeTypedArray(metadata.tensors[tensorIndex].descriptor.dataType, allocation.data, allocation.bytes,
                profile.bindings[tensorIndex]);
        }
        auto const status
            = owned.executor->bindProfile(profile.profileIndex, profile.bindings, owned.main, owned.auxiliary);
        if (!status.okStatus())
        {
            return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTResourceManager::deinitialize(std::vector<TRTSafetyGraphExecutor*> const& executors) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kCLOSED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Close execution before releasing TRT resources");
    }
    if (!mExecutors.empty() && executors.size() != mExecutors.size())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Executor count differs from prepared resources");
    }
    for (size_t index = 0; index < executors.size(); ++index)
    {
        auto const* executor = executors[index];
        if (!mExecutors.empty() && executor != mExecutors[index].executor)
        {
            return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Executor does not own these resources");
        }
        if (executor == nullptr || executor->hasGraph())
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidState, "Destroy every borrowing graph before releasing resources");
        }
        if (executor->fatal())
        {
            return fail(nvinfer2::safe::ErrorCode::kINVALID_STATE, "Safety executor failed before resource teardown");
        }
    }
    for (size_t index = 0; index < mExecutors.size(); ++index)
    {
        auto& owned = mExecutors[index];
        for (auto& allocation : owned.allocations)
        {
            if (allocation.data == nullptr)
            {
                continue;
            }
            cudaError_t status{cudaSuccess};
            if (allocation.placement == nvinfer2::safe::MemoryPlacement::kGPU)
            {
                status = cudaFree(allocation.data);
            }
            else if (allocation.placement == nvinfer2::safe::MemoryPlacement::kCPU_PINNED)
            {
                status = cudaFreeHost(allocation.data);
            }
            else
            {
                std::free(allocation.data);
            }
            if (!updateStatus(status, nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety IO buffer release failed"))
            {
                return mErrorState.status();
            }
            allocation.data = nullptr;
            if (!observeDependencies())
            {
                return mErrorState.status();
            }
        }
        for (auto& stream : owned.auxiliary)
        {
            bool const hasStream = stream != nullptr;
            if (hasStream
                && !updateStatus(cudaStreamDestroy(stream), nvinfer2::safe::ErrorCode::kINVALID_STATE,
                    "Safety auxiliary stream destruction failed"))
            {
                return mErrorState.status();
            }
            stream = nullptr;
            if (!(hasStream ? observeDependencies() : observeErrors()))
            {
                return mErrorState.status();
            }
        }
        auto& main = mMainStreams[owned.mainStreamId];
        bool const lastUser = std::none_of(mExecutors.begin() + index + 1, mExecutors.end(),
            [&owned](auto const& other) { return other.mainStreamId == owned.mainStreamId; });
        bool const releaseMain = lastUser && main != nullptr;
        if (releaseMain)
        {
            if (!updateStatus(cudaStreamDestroy(main), nvinfer2::safe::ErrorCode::kINVALID_STATE,
                    "Safety main stream destruction failed"))
            {
                return mErrorState.status();
            }
            main = nullptr;
        }
        owned.main = nullptr;
        if (!(releaseMain ? observeDependencies() : observeErrors()))
        {
            return mErrorState.status();
        }
    }
    mExecutors.clear();
    mMainStreams.clear();
    return SafetyStatus::ok();
}

std::vector<SafetyProfileResources> const& TRTResourceManager::profiles(size_t executorIndex) const noexcept
{
    static std::vector<SafetyProfileResources> const kEMPTY;
    return executorIndex < mExecutors.size() ? mExecutors[executorIndex].profiles : kEMPTY;
}

cudaStream_t TRTResourceManager::mainStream(size_t executorIndex) const noexcept
{
    return executorIndex < mExecutors.size() ? mExecutors[executorIndex].main : nullptr;
}

} // namespace safety
} // namespace trt_edgellm
