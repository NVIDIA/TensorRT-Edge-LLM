/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if (defined(__QNX__) || defined(__QNXNTO__)) && !defined(EDGELLM_TEST_QNX_STANDARD)
#error "CUDA test cleanup requires standard QNX; QNX Safety is not supported."
#endif

#include <algorithm>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace safety
{
namespace native_test
{

//! Throw std::runtime_error for a failed CUDA result.
inline void checked(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

//! Native-test stream owner; destruction requires completed test GPU work.
struct Stream
{
    cudaStream_t value{};
    //! Create a nonblocking stream; throw on CUDA failure.
    Stream()
    {
        checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    //! Release the native stream after test completion.
    ~Stream()
    {
        cudaStreamDestroy(value);
    }
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
};

//! Paired pinned-host/device storage for native tests; callers synchronize before destruction.
template <typename T>
struct Buffer
{
    T* host{};
    T* device{};
    size_t const count;
    //! Allocate size elements in each address space; throw and release partial allocation on failure.
    explicit Buffer(size_t size)
        : count(size)
    {
        checked(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(T)));
        auto const status = cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T));
        if (status != cudaSuccess)
        {
            cudaFreeHost(host);
            checked(status);
        }
        std::fill_n(host, count, T{});
    }
    //! Release both allocations after test completion.
    ~Buffer()
    {
        cudaFree(device);
        cudaFreeHost(host);
    }
    //! Enqueue the full host-to-device copy on stream; throw on CUDA failure.
    void upload(cudaStream_t stream)
    {
        checked(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    //! Enqueue the full device-to-host copy on stream; throw on CUDA failure.
    void download(cudaStream_t stream)
    {
        checked(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
};

} // namespace native_test
} // namespace safety
} // namespace trt_edgellm
