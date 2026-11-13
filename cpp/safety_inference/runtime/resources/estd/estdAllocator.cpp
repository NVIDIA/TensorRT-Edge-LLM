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

#include "estdAllocator.h"

#include <atomic>

namespace trt_edgellm::estd
{
namespace
{

class DefaultMemAllocator final : public ISafeMemAllocator
{
public:
    void* allocate(std::size_t sizeInBytes, std::size_t alignment) override
    {
        if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
        {
            return ::operator new(sizeInBytes, std::align_val_t{alignment});
        }
        return ::operator new(sizeInBytes);
    }

    void deallocate(void* input, std::size_t, std::size_t alignment) noexcept override
    {
        if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
        {
            ::operator delete(input, std::align_val_t{alignment});
        }
        else
        {
            ::operator delete(input);
        }
    }
};

// Atomic access protects the default pointer only; backend lifetime remains caller-managed.
std::atomic<ISafeMemAllocator*> sDefaultAllocator{nullptr};

ISafeMemAllocator& getDefaultMemAllocator() noexcept
{
    static DefaultMemAllocator sAllocator;
    return sAllocator;
}

} // namespace

ISafeMemAllocator* ControlledHostAllocGetter::get() noexcept
{
    auto* current = sDefaultAllocator.load(std::memory_order_acquire);
    return current == nullptr ? &getDefaultMemAllocator() : current;
}

void ControlledHostAllocGetter::setInternal(ISafeMemAllocator* allocator) noexcept
{
    sDefaultAllocator.store(allocator, std::memory_order_release);
}

void ControlledHostAllocGetter::reset() noexcept
{
    setInternal(nullptr);
}

} // namespace trt_edgellm::estd
