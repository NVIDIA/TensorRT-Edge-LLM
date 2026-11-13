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

#include "declaredResourceManager.h"

#include "safety_inference/runtime/safetyStatusUtils.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace trt_edgellm::safety
{
namespace
{

bool validMemoryKind(RuntimeMemoryKind kind) noexcept
{
    return kind == RuntimeMemoryKind::kPAGEABLE || kind == RuntimeMemoryKind::kPINNED
        || kind == RuntimeMemoryKind::kDEVICE;
}

bool validSize(std::size_t size) noexcept
{
    return size != 0 && size <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

SafetyStatus unavailable() noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Prepared resource is unavailable");
}

class CudaResourceAllocator final : public DeclaredResourceAllocator
{
public:
    SafetyStatus allocate(RuntimeMemoryRequest const& request, void*& output) noexcept override
    {
        output = nullptr;
        SafetyStatus status = SafetyStatus::ok();
        switch (request.kind)
        {
        case RuntimeMemoryKind::kPAGEABLE:
            output = ::operator new(request.sizeInBytes, std::nothrow);
            if (output == nullptr)
            {
                status = SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot allocate host storage");
            }
            break;
        case RuntimeMemoryKind::kPINNED:
            status = fromCudaError(cudaHostAlloc(&output, request.sizeInBytes, cudaHostAllocDefault), "cudaHostAlloc");
            break;
        case RuntimeMemoryKind::kDEVICE:
            status = fromCudaError(cudaMalloc(&output, request.sizeInBytes), "cudaMalloc");
            break;
        }
        return status;
    }

    SafetyStatus release(RuntimeMemoryView memory) noexcept override
    {
        SafetyStatus status = SafetyStatus::ok();
        switch (memory.kind)
        {
        case RuntimeMemoryKind::kPAGEABLE: ::operator delete(memory.data); break;
        case RuntimeMemoryKind::kPINNED: status = fromCudaError(cudaFreeHost(memory.data), "cudaFreeHost"); break;
        case RuntimeMemoryKind::kDEVICE: status = fromCudaError(cudaFree(memory.data), "cudaFree"); break;
        }
        return status;
    }

    SafetyStatus createStream(cudaStream_t& output) noexcept override
    {
        output = nullptr;
        return fromCudaError(cudaStreamCreateWithFlags(&output, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    SafetyStatus destroyStream(cudaStream_t stream) noexcept override
    {
        return fromCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }

    SafetyStatus createEvent(cudaEvent_t& output) noexcept override
    {
        output = nullptr;
        return fromCudaError(cudaEventCreateWithFlags(&output, cudaEventBlockingSync), "cudaEventCreateWithFlags");
    }

    SafetyStatus destroyEvent(cudaEvent_t event) noexcept override
    {
        return fromCudaError(cudaEventDestroy(event), "cudaEventDestroy");
    }
};

DeclaredResourceAllocator& defaultAllocator() noexcept
{
    static CudaResourceAllocator sAllocator;
    return sAllocator;
}

SafetyStatus validate(std::vector<ResourceDeclaration> const& declarations)
{
    for (auto const& declaration : declarations)
    {
        if (declaration.owner == 0 || std::count_if(declarations.begin(), declarations.end(), [&](auto const& other) {
                return other.owner == declaration.owner;
            }) != 1)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid or duplicate resource owner");
        }
        // A local key identifies exactly one owned or borrowed resource, regardless of its kind.
        std::vector<RuntimeResourceKey> keys;
        for (auto const& request : declaration.memory)
        {
            if (!validSize(request.sizeInBytes) || !validMemoryKind(request.kind))
            {
                return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid memory declaration");
            }
            keys.push_back(request.key);
        }
        keys.insert(keys.end(), declaration.streams.begin(), declaration.streams.end());
        keys.insert(keys.end(), declaration.events.begin(), declaration.events.end());
        for (auto const& request : declaration.borrowed)
        {
            bool const stream = request.kind == RuntimeBorrowedResourceKind::kCUDA_STREAM;
            bool const binding = request.kind == RuntimeBorrowedResourceKind::kIO_BINDING
                || request.kind == RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA;
            if ((!stream && !binding) || request.stage.empty() || request.resourceSlot != 0
                || (binding && request.binding.empty()) || (stream && !request.binding.empty()))
            {
                return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid borrowed resource declaration");
            }
            keys.push_back(request.key);
        }
        std::sort(keys.begin(), keys.end());
        if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Duplicate resource key");
        }
        if (declaration.containerStorage && !validSize(declaration.containerStorage->maxCapacityBytes))
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid container storage declaration");
        }
    }
    return SafetyStatus::ok();
}

} // namespace

DeclaredResourceManager::DeclaredResourceManager(LifecycleState& lifecycleState) noexcept
    : DeclaredResourceManager(lifecycleState, defaultAllocator())
{
}

DeclaredResourceManager::DeclaredResourceManager(
    LifecycleState& lifecycleState, DeclaredResourceAllocator& allocator) noexcept
    : mAllocator(allocator)
    , mLifecycleState(lifecycleState)
{
}

DeclaredResourceManager::~DeclaredResourceManager() noexcept
{
    // Explicit healthy teardown is required. Outstanding borrowers may still reference this storage.
    for (auto& set : mSets)
    {
        (void) set.containers.release();
    }
}

void DeclaredResourceManager::copyResourceRequests(ResourceDeclaration const& declaration, ResourceSet& output)
{
    output.owner = declaration.owner;
    output.memory.reserve(declaration.memory.size());
    for (auto const& request : declaration.memory)
    {
        output.memory.push_back({request, nullptr});
    }
    output.streams.reserve(declaration.streams.size());
    for (auto key : declaration.streams)
    {
        output.streams.push_back({key, nullptr});
    }
    output.events.reserve(declaration.events.size());
    for (auto key : declaration.events)
    {
        output.events.push_back({key, nullptr});
    }
    output.borrowed.reserve(declaration.borrowed.size());
    for (auto const& request : declaration.borrowed)
    {
        output.borrowed.push_back({request, {}});
    }
    output.borrowsResolved = output.borrowed.empty();
}

SafetyStatus DeclaredResourceManager::prepareContainer(ResourceDeclaration const& declaration, ResourceSet& output)
{
    if (!declaration.containerStorage)
    {
        return SafetyStatus::ok();
    }
    auto containers = std::make_unique<SafetyContainerStorage>(mLifecycleState);
    auto const status = containers->initialize(*declaration.containerStorage);
    if (status.okStatus())
    {
        output.containers = std::move(containers);
    }
    return status;
}

SafetyStatus DeclaredResourceManager::prepareMetadata(std::vector<ResourceDeclaration> const& declarations)
{
    mSets.resize(declarations.size());
    SafetyStatus status = SafetyStatus::ok();
    for (std::size_t index = 0; status.okStatus() && index < declarations.size(); ++index)
    {
        copyResourceRequests(declarations[index], mSets[index]);
        status = prepareContainer(declarations[index], mSets[index]);
    }
    return status;
}

SafetyStatus DeclaredResourceManager::allocateMemory(Memory& memory) noexcept
{
    auto status = mAllocator.allocate(memory.request, memory.data);
    if (status.okStatus() && memory.data == nullptr)
    {
        status = SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Allocator returned null memory");
    }
    return status;
}

SafetyStatus DeclaredResourceManager::createStream(Stream& stream) noexcept
{
    auto status = mAllocator.createStream(stream.handle);
    if (status.okStatus() && stream.handle == nullptr)
    {
        status = SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Allocator returned default stream");
    }
    return status;
}

SafetyStatus DeclaredResourceManager::createEvent(Event& event) noexcept
{
    auto status = mAllocator.createEvent(event.handle);
    if (status.okStatus() && event.handle == nullptr)
    {
        status = SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Allocator returned null event");
    }
    return status;
}

SafetyStatus DeclaredResourceManager::acquireResourceSet(ResourceSet& set) noexcept
{
    SafetyStatus status = SafetyStatus::ok();
    for (auto& memory : set.memory)
    {
        if (!status.okStatus())
        {
            break;
        }
        status = allocateMemory(memory);
    }
    for (auto& stream : set.streams)
    {
        if (!status.okStatus())
        {
            break;
        }
        status = createStream(stream);
    }
    for (auto& event : set.events)
    {
        if (!status.okStatus())
        {
            break;
        }
        status = createEvent(event);
    }
    return status;
}

SafetyStatus DeclaredResourceManager::acquireResources() noexcept
{
    SafetyStatus status = SafetyStatus::ok();
    for (auto& set : mSets)
    {
        if (!status.okStatus())
        {
            break;
        }
        status = acquireResourceSet(set);
    }
    return status;
}

SafetyStatus DeclaredResourceManager::prepareResources(std::vector<ResourceDeclaration> const& declarations) noexcept
{
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mPreparationState != PreparationState::kNOT_STARTED)
    {
        auto const current = status();
        return current.okStatus() ? unavailable() : current;
    }
    try
    {
        auto status = validate(declarations);
        if (!status.okStatus())
        {
            return status;
        }
        // Finish all metadata allocation before acquiring CUDA resources.
        mPreparationState = PreparationState::kINCOMPLETE;
        status = prepareMetadata(declarations);
        if (status.okStatus())
        {
            status = acquireResources();
        }
        if (!status.okStatus())
        {
            return preparationFailed(status);
        }
    }
    catch (std::bad_alloc const&)
    {
        return preparationFailed(
            SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Resource metadata allocation failed"));
    }
    catch (std::length_error const&)
    {
        return preparationFailed(
            SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Resource metadata exceeds capacity"));
    }
    catch (...)
    {
        return preparationFailed(
            SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Unexpected resource preparation failure"));
    }
    mPreparationState = PreparationState::kCOMPLETE;
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::resolveBorrowedResources(
    ResourceOwnerId owner, BorrowedResourceResolver const& resolver) noexcept
{
    auto const current = status();
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mPreparationState != PreparationState::kCOMPLETE
        || !current.okStatus())
    {
        return current.okStatus() ? unavailable() : current;
    }
    auto const found
        = std::find_if(mSets.begin(), mSets.end(), [owner](auto const& set) { return set.owner == owner; });
    if (found == mSets.end() || found->borrowsResolved)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Owner is undeclared or already resolved");
    }
    {
        for (auto& borrowed : found->borrowed)
        {
            RuntimeBorrowedResource resource{};
            auto status = resolver.resolve(borrowed.request, resource);
            if (status.okStatus())
            {
                bool valid = false;
                switch (borrowed.request.kind)
                {
                case RuntimeBorrowedResourceKind::kIO_BINDING:
                    valid = resource.tensor != nullptr && resource.tensor->desc != nullptr
                        && resource.tensor->desc->name == borrowed.request.binding
                        && validateTensorDesc(*resource.tensor->desc).okStatus()
                        && resource.tensor->memory.data != nullptr && validSize(resource.tensor->memory.sizeInBytes)
                        && validMemoryKind(resource.tensor->memory.kind);
                    if (valid)
                    {
                        for (auto const& profile : resource.tensor->desc->profiles)
                        {
                            valid = valid && profile.sizeInBytes <= resource.tensor->memory.sizeInBytes;
                        }
                    }
                    break;
                case RuntimeBorrowedResourceKind::kCUDA_STREAM: valid = resource.stream != nullptr; break;
                case RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA: valid = resource.modelLocalData != nullptr; break;
                }
                if (!valid)
                {
                    status = SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid borrowed resource view");
                }
            }
            if (!status.okStatus())
            {
                if (status.fatalStatus())
                {
                    abandonAfterFatal(status);
                }
                return status;
            }
            borrowed.resource = resource;
        }
    }
    // Partially resolved borrows remain inaccessible until this owner's batch succeeds.
    found->borrowsResolved = true;
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::getSet(ResourceOwnerId owner, ResourceSet const*& output) const noexcept
{
    output = nullptr;
    if (!status().okStatus())
    {
        return status();
    }
    if (mPreparationState != PreparationState::kCOMPLETE
        || (mLifecycleState != LifecycleState::kINITIALIZATION && mLifecycleState != LifecycleState::kOPERATIONAL))
    {
        return unavailable();
    }
    auto const found
        = std::find_if(mSets.begin(), mSets.end(), [owner](auto const& set) { return set.owner == owner; });
    if (found == mSets.end())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Undeclared resource owner");
    }
    if (found->containers && !found->containers->status().okStatus())
    {
        return found->containers->status();
    }
    output = &*found;
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::resourcesFor(ResourceOwnerId owner, PreparedResources& output) const noexcept
{
    output = {};
    ResourceSet const* set = nullptr;
    auto status = getSet(owner, set);
    if (status.okStatus())
    {
        output = PreparedResources(*this, owner);
    }
    return status;
}

SafetyStatus DeclaredResourceManager::validatePrepared() const noexcept
{
    auto const current = status();
    if (!current.okStatus())
    {
        return current;
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mPreparationState == PreparationState::kINCOMPLETE
        || !std::all_of(mSets.begin(), mSets.end(), [](auto const& set) { return set.borrowsResolved; }))
    {
        return unavailable();
    }
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::status() const noexcept
{
    if (mFailure.fatalStatus())
    {
        return mFailure;
    }
    for (auto const& set : mSets)
    {
        if (set.containers && !set.containers->status().okStatus())
        {
            return set.containers->status();
        }
    }
    return mLifecycleState == LifecycleState::kFAILED
        ? SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety runtime is terminal")
        : SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::releaseEvents(ResourceSet& set) noexcept
{
    for (auto event = set.events.rbegin(); event != set.events.rend(); ++event)
    {
        if (event->handle != nullptr)
        {
            auto const status = mAllocator.destroyEvent(event->handle);
            if (!status.okStatus())
            {
                abandonAfterFatal(SafetyStatus::fatal(status.code(), status.message()));
                return mFailure;
            }
            event->handle = nullptr;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::releaseStreams(ResourceSet& set) noexcept
{
    for (auto stream = set.streams.rbegin(); stream != set.streams.rend(); ++stream)
    {
        if (stream->handle != nullptr)
        {
            auto const status = mAllocator.destroyStream(stream->handle);
            if (!status.okStatus())
            {
                abandonAfterFatal(SafetyStatus::fatal(status.code(), status.message()));
                return mFailure;
            }
            stream->handle = nullptr;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::releaseMemory(ResourceSet& set) noexcept
{
    for (auto memory = set.memory.rbegin(); memory != set.memory.rend(); ++memory)
    {
        if (memory->data != nullptr)
        {
            auto const status = mAllocator.release({memory->data, memory->request.sizeInBytes, memory->request.kind});
            if (!status.okStatus())
            {
                abandonAfterFatal(SafetyStatus::fatal(status.code(), status.message()));
                return mFailure;
            }
            memory->data = nullptr;
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::releaseContainer(ResourceSet& set) noexcept
{
    if (!set.containers)
    {
        return SafetyStatus::ok();
    }
    auto const status = set.containers->deinitialize();
    if (!status.okStatus())
    {
        abandonAfterFatal(SafetyStatus::fatal(status.code(), status.message()));
        return mFailure;
    }
    set.containers.reset();
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::releaseResourceSet(ResourceSet& set) noexcept
{
    auto status = releaseEvents(set);
    if (!status.okStatus())
    {
        return status;
    }
    status = releaseStreams(set);
    if (!status.okStatus())
    {
        return status;
    }
    status = releaseMemory(set);
    if (!status.okStatus())
    {
        return status;
    }
    return releaseContainer(set);
}

SafetyStatus DeclaredResourceManager::releaseResources() noexcept
{
    SafetyStatus status = SafetyStatus::ok();
    for (auto set = mSets.rbegin(); status.okStatus() && set != mSets.rend(); ++set)
    {
        status = releaseResourceSet(*set);
    }
    if (!status.okStatus())
    {
        return status;
    }
    mSets.clear();
    mPreparationState = PreparationState::kINCOMPLETE;
    return SafetyStatus::ok();
}

SafetyStatus DeclaredResourceManager::preparationFailed(SafetyStatus status) noexcept
{
    if (status.fatalStatus())
    {
        abandonAfterFatal(status);
        return mFailure;
    }
    auto const cleanup = releaseResources();
    if (!cleanup.okStatus())
    {
        return cleanup;
    }
    return status;
}

SafetyStatus DeclaredResourceManager::deinitialize() noexcept
{
    if (!status().okStatus())
    {
        abandonAfterFatal(status());
        return mFailure;
    }
    if (mLifecycleState != LifecycleState::kCLOSED || mPreparationState != PreparationState::kCOMPLETE)
    {
        return unavailable();
    }
    return releaseResources();
}

void DeclaredResourceManager::abandonAfterFatal(SafetyStatus status) noexcept
{
    if (!mFailure.fatalStatus())
    {
        mFailure = SafetyStatus::fatal(status.code(), status.message());
    }
    mLifecycleState = LifecycleState::kFAILED;
}

SafetyStatus DeclaredResourceManager::memory(
    ResourceOwnerId owner, RuntimeResourceKey key, RuntimeMemoryView& output) const noexcept
{
    output = {};
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    for (auto const& memory : set->memory)
    {
        if (memory.request.key == key)
        {
            output = {memory.data, memory.request.sizeInBytes, memory.request.kind};
            return SafetyStatus::ok();
        }
    }
    return unavailable();
}

SafetyStatus DeclaredResourceManager::validateInitialization(ResourceOwnerId owner) const noexcept
{
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    return mLifecycleState == LifecycleState::kINITIALIZATION ? SafetyStatus::ok() : unavailable();
}

SafetyStatus DeclaredResourceManager::tensor(
    ResourceOwnerId owner, RuntimeResourceKey key, RuntimeTensorView const*& output) const noexcept
{
    output = nullptr;
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    if (set->borrowsResolved)
    {
        for (auto const& borrowed : set->borrowed)
        {
            if (borrowed.request.key == key && borrowed.request.kind == RuntimeBorrowedResourceKind::kIO_BINDING)
            {
                output = borrowed.resource.tensor;
                return SafetyStatus::ok();
            }
        }
    }
    return unavailable();
}

SafetyStatus DeclaredResourceManager::stream(
    ResourceOwnerId owner, RuntimeResourceKey key, cudaStream_t& output) const noexcept
{
    output = nullptr;
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    for (auto const& stream : set->streams)
    {
        if (stream.key == key)
        {
            output = stream.handle;
            return SafetyStatus::ok();
        }
    }
    if (set->borrowsResolved)
    {
        for (auto const& borrowed : set->borrowed)
        {
            if (borrowed.request.key == key && borrowed.request.kind == RuntimeBorrowedResourceKind::kCUDA_STREAM)
            {
                output = borrowed.resource.stream;
                return SafetyStatus::ok();
            }
        }
    }
    return unavailable();
}

SafetyStatus DeclaredResourceManager::event(
    ResourceOwnerId owner, RuntimeResourceKey key, cudaEvent_t& output) const noexcept
{
    output = nullptr;
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    for (auto const& event : set->events)
    {
        if (event.key == key)
        {
            output = event.handle;
            return SafetyStatus::ok();
        }
    }
    return unavailable();
}

SafetyStatus DeclaredResourceManager::modelLocalData(
    ResourceOwnerId owner, RuntimeResourceKey key, SafetyModelLocalData*& output) const noexcept
{
    output = nullptr;
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    if (set->borrowsResolved)
    {
        for (auto const& borrowed : set->borrowed)
        {
            if (borrowed.request.key == key && borrowed.request.kind == RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA)
            {
                output = borrowed.resource.modelLocalData;
                return SafetyStatus::ok();
            }
        }
    }
    return unavailable();
}

SafetyStatus DeclaredResourceManager::containerStorage(
    ResourceOwnerId owner, SafetyContainerStorage*& output) const noexcept
{
    output = nullptr;
    ResourceSet const* set = nullptr;
    auto const status = getSet(owner, set);
    if (!status.okStatus())
    {
        return status;
    }
    output = set->containers.get();
    return output ? SafetyStatus::ok() : unavailable();
}

SafetyStatus PreparedResources::validateInitialization(ResourceOwnerId owner) const noexcept
{
    auto const status = mManager ? mManager->validateInitialization(mOwner) : unavailable();
    if (!status.okStatus())
    {
        return status;
    }
    return mOwner == owner ? SafetyStatus::ok() : unavailable();
}

SafetyStatus PreparedResources::memory(RuntimeResourceKey key, RuntimeMemoryView& output) const noexcept
{
    output = {};
    return mManager ? mManager->memory(mOwner, key, output) : unavailable();
}

SafetyStatus PreparedResources::tensor(RuntimeResourceKey key, RuntimeTensorView const*& output) const noexcept
{
    output = nullptr;
    return mManager ? mManager->tensor(mOwner, key, output) : unavailable();
}

SafetyStatus PreparedResources::stream(RuntimeResourceKey key, cudaStream_t& output) const noexcept
{
    output = nullptr;
    return mManager ? mManager->stream(mOwner, key, output) : unavailable();
}

SafetyStatus PreparedResources::event(RuntimeResourceKey key, cudaEvent_t& output) const noexcept
{
    output = nullptr;
    return mManager ? mManager->event(mOwner, key, output) : unavailable();
}

SafetyStatus PreparedResources::modelLocalData(RuntimeResourceKey key, SafetyModelLocalData*& output) const noexcept
{
    output = nullptr;
    return mManager ? mManager->modelLocalData(mOwner, key, output) : unavailable();
}

SafetyStatus PreparedResources::containerStorage(SafetyContainerStorage*& output) const noexcept
{
    output = nullptr;
    return mManager ? mManager->containerStorage(mOwner, output) : unavailable();
}

} // namespace trt_edgellm::safety
