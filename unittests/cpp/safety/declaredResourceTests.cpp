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

#include "safety_inference/runtime/resources/declaredResourceManager.h"
#include "safety_inference/runtime/resources/estd/estdVector.h"
#include "safety_inference/runtime/resources/safetyModelLocalData.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>

using namespace trt_edgellm;
using namespace trt_edgellm::safety;

namespace
{

class TestAllocator : public DeclaredResourceAllocator
{
public:
    SafetyStatus allocate(RuntimeMemoryRequest const&, void*& output) noexcept override
    {
        output = nullptr;
        ++calls;
        if (calls == failAt)
        {
            return fatalFailure
                ? SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected allocation failure")
                : SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Injected allocation failure");
        }
        output = blocks[calls - 1].data();
        live[calls - 1] = true;
        return SafetyStatus::ok();
    }
    SafetyStatus release(RuntimeMemoryView memory) noexcept override
    {
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            if (memory.data == blocks[i].data() && live[i])
            {
                events[eventCount++] = -static_cast<int>(i + 1);
                if (failMemoryRelease)
                {
                    return SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected release failure");
                }
                live[i] = false;
                return SafetyStatus::ok();
            }
        }
        invalidRelease = true;
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Released unowned memory");
    }
    SafetyStatus createStream(cudaStream_t& output) noexcept override
    {
        ++calls;
        output = reinterpret_cast<cudaStream_t>(&ownedStream);
        streamLive = true;
        return SafetyStatus::ok();
    }
    SafetyStatus destroyStream(cudaStream_t stream) noexcept override
    {
        invalidRelease |= stream != reinterpret_cast<cudaStream_t>(&ownedStream) || !streamLive;
        events[eventCount++] = -4;
        if (failStreamRelease)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected stream release failure");
        }
        streamLive = false;
        return SafetyStatus::ok();
    }
    SafetyStatus createEvent(cudaEvent_t& output) noexcept override
    {
        ++calls;
        output = reinterpret_cast<cudaEvent_t>(&ownedEvent);
        eventLive = true;
        return SafetyStatus::ok();
    }
    SafetyStatus destroyEvent(cudaEvent_t event) noexcept override
    {
        invalidRelease |= event != reinterpret_cast<cudaEvent_t>(&ownedEvent) || !eventLive;
        events[eventCount++] = -5;
        if (failEventRelease)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Injected event release failure");
        }
        eventLive = false;
        return SafetyStatus::ok();
    }
    std::array<std::array<std::max_align_t, 4>, 3> blocks{};
    std::array<bool, 3> live{};
    std::array<int, 8> events{};
    std::size_t eventCount{0};
    int calls{0};
    int failAt{0};
    bool fatalFailure{false};
    bool failMemoryRelease{false};
    bool failStreamRelease{false};
    bool failEventRelease{false};
    bool invalidRelease{false};
    bool streamLive{false};
    bool eventLive{false};
    int ownedStream{0};
    int ownedEvent{0};
};

struct TestModelData : SafetyModelLocalData
{
    RuntimeMemoryView buffer{};
};

ResourceDeclaration factoryDeclaration()
{
    return {1,
        {{1, 32, RuntimeMemoryKind::kPAGEABLE}, {2, 32, RuntimeMemoryKind::kPINNED},
            {3, 32, RuntimeMemoryKind::kDEVICE}},
        {4}, {}, {{5, RuntimeBorrowedResourceKind::kIO_BINDING, "prefill", 0, "input"}}, {6}};
}

class TestResolver : public BorrowedResourceResolver
{
public:
    SafetyStatus resolve(
        RuntimeBorrowedResourceRequest const& request, RuntimeBorrowedResource& output) const noexcept override
    {
        output = {};
        if (request.kind == RuntimeBorrowedResourceKind::kIO_BINDING)
        {
            output.tensor = &tensor;
        }
        else if (request.kind == RuntimeBorrowedResourceKind::kCUDA_STREAM)
        {
            output.stream = reinterpret_cast<cudaStream_t>(&borrowedStream);
        }
        else
        {
            output.modelLocalData = model;
        }
        return SafetyStatus::ok();
    }
    mutable int input{19};
    RuntimeTensorDesc desc{
        "input", nvinfer1::DataType::kINT32, nvinfer1::TensorIOMode::kINPUT, {{0, {1, {1}}, sizeof(input)}}};
    RuntimeTensorView tensor{&desc, {&input, sizeof(input), RuntimeMemoryKind::kPAGEABLE}};
    mutable int borrowedStream{0};
    SafetyModelLocalData* model{nullptr};
};

TEST(DeclaredResources, PreparedCapabilityAndFactoryLifecycle)
{
    // Given bounded owned/borrowed declarations for a factory and capability,
    // when prepared and used in Operational, then views remain stable without allocation,
    // and healthy teardown releases owned resources only, after their borrowers.
    TestAllocator allocator;
    LifecycleState state{LifecycleState::kINITIALIZATION};
    DeclaredResourceManager manager(state, allocator);
    std::vector<ResourceDeclaration> declarations{factoryDeclaration(),
        {2, {}, {}, SafetyContainerStorageRequest{64},
            {{1, RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA, "prefill", 0, "model"},
                {2, RuntimeBorrowedResourceKind::kCUDA_STREAM, "prefill", 0, {}}}}};
    ASSERT_TRUE(manager.prepareResources(declarations).okStatus());
    PreparedResources factoryResources, capabilityResources;
    ASSERT_TRUE(manager.resourcesFor(1, factoryResources).okStatus());
    ASSERT_TRUE(manager.resourcesFor(2, capabilityResources).okStatus());
    TestResolver resolver;
    ASSERT_TRUE(manager.resolveBorrowedResources(1, resolver).okStatus());
    auto model = std::make_unique<TestModelData>();
    RuntimeTensorView const* input = nullptr;
    ASSERT_TRUE(factoryResources.memory(1, model->buffer).okStatus());
    ASSERT_TRUE(factoryResources.tensor(5, input).okStatus());
    *static_cast<int*>(model->buffer.data) = *static_cast<int*>(input->memory.data);
    resolver.model = model.get();
    ASSERT_TRUE(manager.resolveBorrowedResources(2, resolver).okStatus());
    SafetyModelLocalData* borrowedModel = nullptr;
    SafetyContainerStorage* storage = nullptr;
    ASSERT_TRUE(capabilityResources.modelLocalData(1, borrowedModel).okStatus());
    ASSERT_TRUE(capabilityResources.containerStorage(storage).okStatus());
    std::optional<estd::vector<int>> values{std::in_place, estd::allocator<int>(*storage)};
    values->resize(4);
    ASSERT_TRUE(manager.validatePrepared().okStatus());
    cudaEvent_t event = nullptr;
    ASSERT_TRUE(factoryResources.event(6, event).okStatus());
    EXPECT_EQ(event, reinterpret_cast<cudaEvent_t>(&allocator.ownedEvent));
    state = LifecycleState::kOPERATIONAL;
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        RuntimeMemoryView view{};
        cudaStream_t stream = nullptr;
        ASSERT_TRUE(factoryResources.memory(1, view).okStatus());
        EXPECT_EQ(view.data, allocator.blocks[0].data());
        EXPECT_EQ(view.sizeInBytes, 32U);
        EXPECT_EQ(view.kind, RuntimeMemoryKind::kPAGEABLE);
        (*values)[iteration] = *static_cast<int*>(static_cast<TestModelData*>(borrowedModel)->buffer.data) + iteration;
        ASSERT_TRUE(capabilityResources.stream(2, stream).okStatus());
        EXPECT_EQ(stream, reinterpret_cast<cudaStream_t>(&resolver.borrowedStream));
    }
    EXPECT_EQ((*values)[2], 21);
    EXPECT_EQ(allocator.calls, 5);
    EXPECT_THROW(storage->allocate(4, alignof(int)), std::bad_alloc);
    EXPECT_FALSE(manager.prepareResources(declarations).okStatus());
    state = LifecycleState::kCLOSED;
    values.reset();
    model.reset();
    ASSERT_TRUE(manager.deinitialize().okStatus());
    EXPECT_EQ(allocator.eventCount, 5U);
    EXPECT_EQ(allocator.events, (std::array<int, 8>{-5, -4, -3, -2, -1, 0, 0, 0}));
    EXPECT_FALSE(allocator.invalidRelease);
    EXPECT_EQ(resolver.input, 19);
    RuntimeMemoryView stale{};
    EXPECT_FALSE(factoryResources.memory(1, stale).okStatus());
    EXPECT_EQ(stale.data, nullptr);
}

enum class Failure
{
    kDUPLICATE,
    kINVALID,
    kPARTIAL_ALLOCATION,
    kFATAL_ALLOCATION,
    kFATAL_MEMORY_RELEASE,
    kFATAL_STREAM_RELEASE,
    kFATAL_EVENT_RELEASE,
    kUNRESOLVED,
};

class DeclaredResourceFailure : public ::testing::TestWithParam<Failure>
{
};

TEST_P(DeclaredResourceFailure, FailedPreparationAndContainment)
{
    // Given invalid declarations or an injected preparation/teardown failure,
    // when resources are prepared, then invalid/incomplete resources stay inaccessible;
    // recoverable allocation failure rolls back, while fatal CUDA failure stops cleanup.
    TestAllocator allocator;
    std::vector<ResourceDeclaration> declarations{factoryDeclaration()};
    auto const failure = GetParam();
    if (failure == Failure::kDUPLICATE)
    {
        declarations[0].streams.push_back(1);
    }
    if (failure == Failure::kINVALID)
    {
        declarations[0].memory.back().sizeInBytes = 0;
    }
    if (failure == Failure::kPARTIAL_ALLOCATION || failure == Failure::kFATAL_ALLOCATION)
    {
        allocator.failAt = 2;
        allocator.fatalFailure = failure == Failure::kFATAL_ALLOCATION;
    }
    {
        LifecycleState state{LifecycleState::kINITIALIZATION};
        DeclaredResourceManager manager(state, allocator);
        auto status = manager.prepareResources(declarations);
        PreparedResources resources;
        if (failure == Failure::kUNRESOLVED || failure == Failure::kFATAL_MEMORY_RELEASE
            || failure == Failure::kFATAL_STREAM_RELEASE || failure == Failure::kFATAL_EVENT_RELEASE)
        {
            ASSERT_TRUE(status.okStatus());
            ASSERT_TRUE(manager.resourcesFor(1, resources).okStatus());
            RuntimeTensorView const* borrowed = nullptr;
            EXPECT_FALSE(resources.tensor(5, borrowed).okStatus());
            EXPECT_FALSE(manager.validatePrepared().okStatus());
            allocator.failMemoryRelease = failure == Failure::kFATAL_MEMORY_RELEASE;
            allocator.failStreamRelease = failure == Failure::kFATAL_STREAM_RELEASE;
            allocator.failEventRelease = failure == Failure::kFATAL_EVENT_RELEASE;
            state = LifecycleState::kCLOSED;
            status = manager.deinitialize();
        }
        bool const fatal = failure == Failure::kFATAL_ALLOCATION || failure == Failure::kFATAL_MEMORY_RELEASE
            || failure == Failure::kFATAL_STREAM_RELEASE || failure == Failure::kFATAL_EVENT_RELEASE;
        EXPECT_EQ(status.okStatus(), failure == Failure::kUNRESOLVED);
        EXPECT_EQ(status.fatalStatus(), fatal);
        EXPECT_FALSE(manager.resourcesFor(1, resources).okStatus());
        EXPECT_FALSE(manager.deinitialize().okStatus());
    }
    int const expectedCalls = failure == Failure::kDUPLICATE || failure == Failure::kINVALID ? 0
        : failure == Failure::kPARTIAL_ALLOCATION || failure == Failure::kFATAL_ALLOCATION   ? 2
                                                                                             : 5;
    EXPECT_EQ(allocator.calls, expectedCalls);
    std::array<int, 8> const expectedEvents = failure == Failure::kPARTIAL_ALLOCATION ? std::array<int, 8>{-1}
        : failure == Failure::kUNRESOLVED           ? std::array<int, 8>{-5, -4, -3, -2, -1}
        : failure == Failure::kFATAL_MEMORY_RELEASE ? std::array<int, 8>{-5, -4, -3}
        : failure == Failure::kFATAL_STREAM_RELEASE ? std::array<int, 8>{-5, -4}
        : failure == Failure::kFATAL_EVENT_RELEASE  ? std::array<int, 8>{-5}
                                                    : std::array<int, 8>{};
    EXPECT_EQ(allocator.events, expectedEvents);
    EXPECT_FALSE(allocator.invalidRelease);
}

INSTANTIATE_TEST_SUITE_P(Core, DeclaredResourceFailure,
    ::testing::Values(Failure::kDUPLICATE, Failure::kINVALID, Failure::kPARTIAL_ALLOCATION, Failure::kFATAL_ALLOCATION,
        Failure::kFATAL_MEMORY_RELEASE, Failure::kFATAL_STREAM_RELEASE, Failure::kFATAL_EVENT_RELEASE,
        Failure::kUNRESOLVED));

} // namespace
