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

#include "runtimeResource.h"

namespace trt_edgellm::safety
{

//! Allocation seam for Initialization/healthy Deinitialization and local fault injection.
//! On failure, output still identifies any acquired resource; release failure retains ownership.
//! Memory has at least fundamental alignment. The allocator outlives the resource manager.
class DeclaredResourceAllocator
{
public:
    virtual ~DeclaredResourceAllocator() = default;
    virtual SafetyStatus allocate(RuntimeMemoryRequest const& request, void*& output) noexcept = 0;
    virtual SafetyStatus release(RuntimeMemoryView memory) noexcept = 0;
    virtual SafetyStatus createStream(cudaStream_t& output) noexcept = 0;
    virtual SafetyStatus destroyStream(cudaStream_t stream) noexcept = 0;
    virtual SafetyStatus createEvent(cudaEvent_t& output) noexcept = 0;
    virtual SafetyStatus destroyEvent(cudaEvent_t event) noexcept = 0;
};

//! Owned by SafetyRuntime; access and lifecycle calls are externally serialized.
//! Destroy capabilities/model-local objects before deinitialize(). Destruction performs no SDK calls.
class DeclaredResourceManager
{
public:
    explicit DeclaredResourceManager(LifecycleState& lifecycleState) noexcept;
    DeclaredResourceManager(LifecycleState& lifecycleState, DeclaredResourceAllocator& allocator) noexcept;
    ~DeclaredResourceManager() noexcept;
    DeclaredResourceManager(DeclaredResourceManager const&) = delete;
    DeclaredResourceManager& operator=(DeclaredResourceManager const&) = delete;

    //! Validate every declaration, then allocate owned resources. Views become available only on success.
    SafetyStatus prepareResources(std::vector<ResourceDeclaration> const& declarations) noexcept;
    //! Resolve one owner's borrows atomically. Resolve factory inputs before constructing model-local objects,
    //! then resolve capabilities borrowing those objects. All owners must be resolved before Operational.
    SafetyStatus resolveBorrowedResources(ResourceOwnerId owner, BorrowedResourceResolver const& resolver) noexcept;
    SafetyStatus resourcesFor(ResourceOwnerId owner, PreparedResources& output) const noexcept;
    //! Typed lookups validate the owner, preparation and lifecycle before exposing a borrowed view.
    SafetyStatus memory(ResourceOwnerId owner, RuntimeResourceKey key, RuntimeMemoryView& output) const noexcept;
    SafetyStatus tensor(ResourceOwnerId owner, RuntimeResourceKey key, RuntimeTensorView const*& output) const noexcept;
    SafetyStatus stream(ResourceOwnerId owner, RuntimeResourceKey key, cudaStream_t& output) const noexcept;
    SafetyStatus event(ResourceOwnerId owner, RuntimeResourceKey key, cudaEvent_t& output) const noexcept;
    SafetyStatus modelLocalData(
        ResourceOwnerId owner, RuntimeResourceKey key, SafetyModelLocalData*& output) const noexcept;
    SafetyStatus containerStorage(ResourceOwnerId owner, SafetyContainerStorage*& output) const noexcept;
    //! Factory/capability setup requires prepared resources during Initialization.
    SafetyStatus validateInitialization(ResourceOwnerId owner) const noexcept;
    //! Check all borrows before SafetyRuntime publishes Operational readiness.
    SafetyStatus validatePrepared() const noexcept;
    SafetyStatus status() const noexcept;
    bool hasResources() const noexcept
    {
        return !mSets.empty();
    }
    //! SafetyRuntime closes execution before caller-owned objects and declared storage are released.
    SafetyStatus deinitialize() noexcept;
    //! Latch an external fatal failure before any attempted teardown; retain resources for containment.
    void abandonAfterFatal(SafetyStatus status) noexcept;

private:
    //! Initialization alone cannot distinguish optional, incomplete and completed preparation.
    enum class PreparationState
    {
        kNOT_STARTED,
        kINCOMPLETE,
        kCOMPLETE,
    };
    struct Memory
    {
        RuntimeMemoryRequest request{};
        void* data{nullptr};
    };
    struct Stream
    {
        RuntimeResourceKey key{0};
        cudaStream_t handle{nullptr};
    };
    struct Event
    {
        RuntimeResourceKey key{0};
        cudaEvent_t handle{nullptr};
    };
    struct Borrowed
    {
        RuntimeBorrowedResourceRequest request{};
        RuntimeBorrowedResource resource{};
    };
    struct ResourceSet
    {
        ResourceOwnerId owner{0};
        std::vector<Memory> memory{};
        std::vector<Stream> streams{};
        std::vector<Event> events{};
        std::vector<Borrowed> borrowed{};
        std::unique_ptr<SafetyContainerStorage> containers{};
        bool borrowsResolved{false};
    };
    SafetyStatus getSet(ResourceOwnerId owner, ResourceSet const*& output) const noexcept;
    void copyResourceRequests(ResourceDeclaration const& declaration, ResourceSet& output);
    SafetyStatus prepareContainer(ResourceDeclaration const& declaration, ResourceSet& output);
    SafetyStatus prepareMetadata(std::vector<ResourceDeclaration> const& declarations);
    SafetyStatus allocateMemory(Memory& memory) noexcept;
    SafetyStatus createStream(Stream& stream) noexcept;
    SafetyStatus createEvent(Event& event) noexcept;
    SafetyStatus acquireResourceSet(ResourceSet& set) noexcept;
    SafetyStatus acquireResources() noexcept;
    SafetyStatus releaseEvents(ResourceSet& set) noexcept;
    SafetyStatus releaseStreams(ResourceSet& set) noexcept;
    SafetyStatus releaseMemory(ResourceSet& set) noexcept;
    SafetyStatus releaseContainer(ResourceSet& set) noexcept;
    SafetyStatus releaseResourceSet(ResourceSet& set) noexcept;
    SafetyStatus releaseResources() noexcept;
    SafetyStatus preparationFailed(SafetyStatus status) noexcept;
    SafetyStatus mFailure{SafetyStatus::ok()};
    DeclaredResourceAllocator& mAllocator;
    std::vector<ResourceSet> mSets{};
    LifecycleState& mLifecycleState;
    PreparationState mPreparationState{PreparationState::kNOT_STARTED};
};

} // namespace trt_edgellm::safety
