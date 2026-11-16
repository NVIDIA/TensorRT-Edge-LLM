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

#include "safety_inference/runtime/resources/containerStorage.h"
#include "safety_inference/runtime/resources/estd/estdVector.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

namespace trt_edgellm::safety
{
namespace
{

class TrackingAllocator : public estd::ISafeMemAllocator
{
public:
    explicit TrackingAllocator(estd::ISafeMemAllocator& delegate)
        : mDelegate(delegate)
    {
    }

    void* allocate(std::size_t bytes, std::size_t alignment) override
    {
        auto* memory = mDelegate.allocate(bytes, alignment);
        ++allocations;
        lastAllocation = memory;
        return memory;
    }

    void deallocate(void* memory, std::size_t bytes, std::size_t alignment) noexcept override
    {
        ++deallocations;
        lastDeallocation = memory;
        mDelegate.deallocate(memory, bytes, alignment);
    }

    int32_t allocations{0};
    int32_t deallocations{0};
    void* lastAllocation{nullptr};
    void* lastDeallocation{nullptr};

private:
    estd::ISafeMemAllocator& mDelegate;
};

struct RestoreDefaultAllocator
{
    estd::ISafeMemAllocator* original{estd::ControlledHostAllocGetter::get()};

    ~RestoreDefaultAllocator()
    {
        estd::ControlledHostAllocGetter::setInternal(original);
    }
};

TEST(SafetyContainerStorageTest, DefaultContainerUsesInjectedTrackingAllocator)
{
    // Given an injected tracking allocator, when a default-constructed container reserves storage and is destroyed,
    // then its allocation and release pass through that allocator.
    auto& delegate = *estd::ControlledHostAllocGetter::get();
    TrackingAllocator tracking(delegate);
    RestoreDefaultAllocator restore;
    estd::ControlledHostAllocGetter::setInternal(&tracking);
    {
        estd::vector<int32_t> values;
        values.reserve(4);
        EXPECT_EQ(tracking.allocations, 1);
        EXPECT_EQ(values.data(), tracking.lastAllocation);
    }
    EXPECT_EQ(tracking.deallocations, 1);
    EXPECT_EQ(tracking.lastDeallocation, tracking.lastAllocation);
}

TEST(SafetyContainerStorageTest, AlignedRequestsUseDistinctBoundedStorage)
{
    // Given fixed-capacity storage, when valid aligned requests are prepared,
    // then each returned region is aligned, distinct and usable within that capacity.
    constexpr std::size_t kUNIT = alignof(std::max_align_t);
    LifecycleState state{LifecycleState::kINITIALIZATION};
    SafetyContainerStorage storage(state);
    ASSERT_TRUE(storage.initialize({3 * kUNIT}).okStatus());
    std::array<void*, 3> regions{};
    for (auto& region : regions)
    {
        ASSERT_NO_THROW(region = storage.allocate(1, kUNIT));
        ASSERT_NE(region, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(region) % kUNIT, 0U);
    }
    EXPECT_EQ(regions[1], static_cast<std::byte*>(regions[0]) + kUNIT);
    EXPECT_EQ(regions[2], static_cast<std::byte*>(regions[0]) + 2 * kUNIT);
    *static_cast<std::byte*>(regions[0]) = std::byte{11};
    *static_cast<std::byte*>(regions[1]) = std::byte{22};
    *static_cast<std::byte*>(regions[2]) = std::byte{33};
    EXPECT_EQ(*static_cast<std::byte*>(regions[0]), std::byte{11});
    EXPECT_EQ(*static_cast<std::byte*>(regions[1]), std::byte{22});
}

TEST(SafetyContainerStorageTest, RejectedRequestsPreserveAvailableStorage)
{
    // Given fixed-capacity storage, when size or alignment is invalid,
    // then the request fails without exposing a pointer or consuming space.
    constexpr std::size_t kUNIT = alignof(std::max_align_t);
    constexpr std::size_t kCAPACITY = 4 * kUNIT;
    LifecycleState state{LifecycleState::kINITIALIZATION};
    SafetyContainerStorage storage(state);
    ASSERT_TRUE(storage.initialize({kCAPACITY}).okStatus());
    struct Request
    {
        std::size_t bytes;
        std::size_t alignment;
    };
    std::array<Request, 5> const invalid{{
        {kCAPACITY + 1, kUNIT},
        {std::numeric_limits<std::size_t>::max(), kUNIT},
        {1, 0},
        {1, 3},
        {1, 2 * kUNIT},
    }};
    for (auto const& request : invalid)
    {
        SCOPED_TRACE(::testing::Message() << "bytes=" << request.bytes << ", alignment=" << request.alignment);
        void* output = nullptr;
        EXPECT_THROW(output = storage.allocate(request.bytes, request.alignment), std::bad_alloc);
        EXPECT_EQ(output, nullptr);
    }
    void* whole = nullptr;
    ASSERT_NO_THROW(whole = storage.allocate(kCAPACITY, kUNIT));
    ASSERT_NE(whole, nullptr);
    auto* bytes = static_cast<std::byte*>(whole);
    bytes[0] = std::byte{11};
    bytes[kCAPACITY - 1] = std::byte{22};
    void* extra = nullptr;
    EXPECT_THROW(extra = storage.allocate(1, 1), std::bad_alloc);
    EXPECT_EQ(extra, nullptr);
    EXPECT_EQ(bytes[0], std::byte{11});
    EXPECT_EQ(bytes[kCAPACITY - 1], std::byte{22});
}

} // namespace
} // namespace trt_edgellm::safety
