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

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace trt_edgellm::estd
{

//! Byte allocator backend, also replaceable by a tracking or fault-injection implementation.
//! Tracking backends and their associated containers are for single-threaded debug/test use.
//! The backend must outlive every allocator copy and container using it.
class ISafeMemAllocator
{
public:
    virtual ~ISafeMemAllocator() = default;
    virtual void* allocate(std::size_t sizeInBytes, std::size_t alignment) = 0;
    virtual void deallocate(void* input, std::size_t sizeInBytes, std::size_t alignment) noexcept = 0;
};

//! Process-wide default for subsequently constructed allocators.
//! Call setInternal/reset during setup or teardown with no concurrent allocator access.
class ControlledHostAllocGetter
{
public:
    //! Returns the configured allocator, initially the host heap allocator.
    static ISafeMemAllocator* get() noexcept;

    //! Configure during initialization or test setup; nullptr restores the host heap.
    //! The allocator must outlive all captured bindings. Existing bindings are unchanged.
    static void setInternal(ISafeMemAllocator* allocator) noexcept;

    static void reset() noexcept;
};

//! STL allocator that captures the default at construction, or accepts an explicit allocator.
template <typename T>
class allocator
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    template <typename U>
    struct rebind
    {
        using other = allocator<U>;
    };

    allocator() noexcept
        : mAllocator(ControlledHostAllocGetter::get())
    {
    }
    explicit allocator(ISafeMemAllocator& memAllocator) noexcept
        : mAllocator(&memAllocator)
    {
    }

    template <typename U>
    allocator(allocator<U> const& other) noexcept
        : mAllocator(&other.getMemAllocator())
    {
    }

    [[nodiscard]] T* allocate(size_type count)
    {
        if (count > max_size())
        {
            throw std::bad_array_new_length{};
        }
        auto* memory = mAllocator->allocate(count * sizeof(T), alignof(T));
        if (memory == nullptr)
        {
            throw std::bad_alloc{};
        }
        return static_cast<T*>(memory);
    }

    void deallocate(T* input, size_type count) noexcept
    {
        if (input != nullptr)
        {
            mAllocator->deallocate(input, count * sizeof(T), alignof(T));
        }
    }

    size_type max_size() const noexcept
    {
        return static_cast<size_type>(std::numeric_limits<difference_type>::max()) / sizeof(T);
    }

    //! Returns the captured backend shared by allocator copies and rebinds.
    ISafeMemAllocator& getMemAllocator() const noexcept
    {
        return *mAllocator;
    }

    template <typename U>
    bool operator==(allocator<U> const& other) const noexcept
    {
        return mAllocator == &other.getMemAllocator();
    }

    template <typename U>
    bool operator!=(allocator<U> const& other) const noexcept
    {
        return !(*this == other);
    }

private:
    ISafeMemAllocator* mAllocator;
};

} // namespace trt_edgellm::estd
