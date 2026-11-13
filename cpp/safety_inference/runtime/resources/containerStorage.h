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

#include "estd/estdAllocator.h"
#include "safety_inference/runtime/lifecycleState.h"
#include "safety_inference/runtime/safetyStatus.h"

#include <cstddef>
#include <memory>

namespace trt_edgellm::safety
{

struct SafetyContainerStorageRequest
{
    std::size_t maxCapacityBytes{0};
};

//! Pageable container backing storage owned by DeclaredResourceManager.
//! Containers and their non-owning views must be destroyed before this object.
class SafetyContainerStorage : public estd::ISafeMemAllocator
{
public:
    explicit SafetyContainerStorage(LifecycleState& lifecycleState) noexcept
        : mLifecycleState(lifecycleState)
    {
    }
    SafetyContainerStorage(SafetyContainerStorage const&) = delete;
    SafetyContainerStorage& operator=(SafetyContainerStorage const&) = delete;

    SafetyStatus initialize(SafetyContainerStorageRequest const& request) noexcept;
    SafetyStatus deinitialize() noexcept;

    //! Allocation is restricted to Initialization; failure throws std::bad_alloc.
    void* allocate(std::size_t sizeInBytes, std::size_t alignment) override;
    void deallocate(void* input, std::size_t sizeInBytes, std::size_t alignment) noexcept override;

    //! Retains release failures because STL deallocate must not throw.
    SafetyStatus status() const noexcept
    {
        return mFailure;
    }

private:
    SafetyStatus tryAllocate(std::size_t sizeInBytes, std::size_t alignment, void*& output) noexcept;
    SafetyStatus release(void* input) noexcept;

    struct Block
    {
        std::size_t offset{0};
        std::size_t sizeInBytes{0};
        bool available{true};
    };

    std::unique_ptr<std::byte[]> mStorage{};
    std::unique_ptr<Block[]> mBlocks{};
    std::size_t mBlockCount{0};
    std::size_t mBlockCapacity{0};
    std::size_t mCapacityBytes{0};
    LifecycleState const& mLifecycleState;
    SafetyStatus mFailure{SafetyStatus::ok()};
};

} // namespace trt_edgellm::safety
