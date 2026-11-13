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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <utility>

namespace trt_edgellm::safety
{
namespace
{
constexpr std::size_t kALIGNMENT{alignof(std::max_align_t)};
} // namespace

SafetyStatus SafetyContainerStorage::initialize(SafetyContainerStorageRequest const& request) noexcept
{
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mStorage)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Container storage is already initialized");
    }
    std::size_t const capacity = request.maxCapacityBytes;
    std::size_t const blockCapacity = capacity / kALIGNMENT + 1;
    if (capacity == 0 || capacity > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || blockCapacity > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Block))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid container storage capacity");
    }
    try
    {
        // byte-array new provides fundamental alignment; the declared byte bound is unchanged.
        auto storage = std::make_unique<std::byte[]>(capacity);
        auto blocks = std::make_unique<Block[]>(blockCapacity);
        blocks[0] = {0, capacity, true};
        mStorage = std::move(storage);
        mBlocks = std::move(blocks);
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedAllocation, "Cannot prepare container storage");
    }
    mCapacityBytes = capacity;
    mBlockCapacity = blockCapacity;
    mBlockCount = 1;
    return SafetyStatus::ok();
}

void* SafetyContainerStorage::allocate(std::size_t sizeInBytes, std::size_t alignment)
{
    void* output = nullptr;
    if (!mFailure.okStatus() || !tryAllocate(sizeInBytes, alignment, output).okStatus())
    {
        throw std::bad_alloc{};
    }
    return output;
}

void SafetyContainerStorage::deallocate(void* input, std::size_t, std::size_t) noexcept
{
    auto const result = release(input);
    if (!result.okStatus() && mFailure.okStatus())
    {
        mFailure = SafetyStatus::fatal(result.code(), result.message());
    }
}

SafetyStatus SafetyContainerStorage::tryAllocate(std::size_t sizeInBytes, std::size_t alignment, void*& output) noexcept
{
    output = nullptr;
    if (mLifecycleState != LifecycleState::kINITIALIZATION || !mStorage)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Container allocation requires Initialization");
    }
    constexpr std::size_t kERROR_MESSAGE_CAPACITY{128};
    if (alignment == 0 || alignment > kALIGNMENT || (alignment & (alignment - 1)) != 0)
    {
        char message[kERROR_MESSAGE_CAPACITY]{};
        std::snprintf(message, sizeof(message),
            "Invalid container allocation alignment %zu; expected a power of two up to %zu", alignment, kALIGNMENT);
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, message);
    }
    if (sizeInBytes > std::numeric_limits<std::size_t>::max() - (kALIGNMENT - 1))
    {
        char message[kERROR_MESSAGE_CAPACITY]{};
        std::snprintf(
            message, sizeof(message), "Container allocation size %zu overflows alignment rounding", sizeInBytes);
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, message);
    }
    std::size_t const bytes = std::max<std::size_t>(sizeInBytes, 1);
    std::size_t const alignedSize = (bytes + kALIGNMENT - 1) & ~(kALIGNMENT - 1);
    for (std::size_t i = 0; i < mBlockCount; ++i)
    {
        auto& block = mBlocks[i];
        if (!block.available || block.sizeInBytes < alignedSize)
        {
            continue;
        }
        std::size_t const remaining = block.sizeInBytes - alignedSize;
        if (remaining != 0)
        {
            if (mBlockCount == mBlockCapacity)
            {
                return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Container block metadata is exhausted");
            }
            for (std::size_t j = mBlockCount; j > i + 1; --j)
            {
                mBlocks[j] = mBlocks[j - 1];
            }
            mBlocks[i + 1] = {block.offset + alignedSize, remaining, true};
            ++mBlockCount;
        }
        block.sizeInBytes = alignedSize;
        block.available = false;
        output = mStorage.get() + block.offset;
        return SafetyStatus::ok();
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Container storage capacity exceeded");
}

SafetyStatus SafetyContainerStorage::release(void* input) noexcept
{
    if ((mLifecycleState != LifecycleState::kINITIALIZATION && mLifecycleState != LifecycleState::kCLOSED) || !mStorage)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Container deallocation is outside its lifecycle");
    }
    if (input == nullptr)
    {
        return SafetyStatus::ok();
    }
    auto const address = reinterpret_cast<std::uintptr_t>(input);
    auto const begin = reinterpret_cast<std::uintptr_t>(mStorage.get());
    if (address < begin || address - begin >= mCapacityBytes)
    {
        return SafetyStatus::error(
            SafetyStatusCode::kInvalidArgument, "Container allocation is not owned by this storage");
    }
    auto const offset = static_cast<std::size_t>(address - begin);
    for (std::size_t i = 0; i < mBlockCount; ++i)
    {
        auto& block = mBlocks[i];
        if (block.offset != offset || block.available)
        {
            continue;
        }
        block.available = true;
        if (i + 1 < mBlockCount && mBlocks[i + 1].available)
        {
            block.sizeInBytes += mBlocks[i + 1].sizeInBytes;
            for (std::size_t j = i + 1; j + 1 < mBlockCount; ++j)
            {
                mBlocks[j] = mBlocks[j + 1];
            }
            --mBlockCount;
        }
        if (i != 0 && mBlocks[i - 1].available)
        {
            mBlocks[i - 1].sizeInBytes += block.sizeInBytes;
            for (std::size_t j = i; j + 1 < mBlockCount; ++j)
            {
                mBlocks[j] = mBlocks[j + 1];
            }
            --mBlockCount;
        }
        return SafetyStatus::ok();
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Container pointer is not an active allocation");
}

SafetyStatus SafetyContainerStorage::deinitialize() noexcept
{
    if (mLifecycleState != LifecycleState::kINITIALIZATION && mLifecycleState != LifecycleState::kCLOSED)
    {
        return SafetyStatus::error(
            SafetyStatusCode::kInvalidState, "Container storage cannot be released in this phase");
    }
    mBlocks.reset();
    mStorage.reset();
    mCapacityBytes = 0;
    mBlockCount = 0;
    mBlockCapacity = 0;
    return mFailure;
}

} // namespace trt_edgellm::safety
