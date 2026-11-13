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

#include "common/identityHash.h"

#include "kernels/common/fnv1aHashKernel.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace rt
{
namespace
{

// Every opaque-identity byte stream starts with this tag; the block-chain streams in blockHash.cpp start with their
// own tag values, which keeps the two kinds of stream distinct from the first byte.
constexpr uint8_t kOPAQUE_IDENTITY_TAG = 18;

Hash128 hashOpaqueIdentityCpu(std::string_view bytes)
{
    size_t const totalSize = bytes.size();
    int32_t const numChunks = kernel::kHASH_NUM_CHUNKS;
    size_t const chunkSize = (totalSize + static_cast<size_t>(numChunks) - 1) / static_cast<size_t>(numChunks);

    // Phase 1: hash each chunk independently.
    FnvValue partials[kernel::kHASH_NUM_CHUNKS];
    for (int32_t c = 0; c < numChunks; ++c)
    {
        size_t const start = static_cast<size_t>(c) * chunkSize;
        size_t const end = std::min(start + chunkSize, totalSize);
        FnvValue partial = kFNV_OFFSET;
        for (size_t i = start; i < end; ++i)
        {
            fnvByte(partial, static_cast<uint8_t>(bytes[i]));
        }
        partials[c] = partial;
    }

    // Phase 2: reduce all per-chunk digests into the final hash.
    FnvValue state = kFNV_OFFSET;
    fnvByte(state, kOPAQUE_IDENTITY_TAG);
    fnvU64(state, static_cast<uint64_t>(totalSize));
    for (int32_t c = 0; c < numChunks; ++c)
    {
        fnvU64(state, partials[c].hi);
        fnvU64(state, partials[c].lo);
    }
    return finishHash(state);
}

} // namespace

Hash128 hashOpaqueIdentity(std::string_view bytes, cudaStream_t stream, bool cpuOnly)
{
    constexpr size_t kGPU_THRESHOLD = 1024 * 1024;
    if (cpuOnly || stream == nullptr || bytes.size() < kGPU_THRESHOLD)
    {
        return hashOpaqueIdentityCpu(bytes);
    }

    size_t const totalSize = bytes.size();

    // Determine whether the GPU shares physical memory with the CPU (Tegra/Thor/Orin).
    // On integrated GPUs, register the host buffer for zero-copy access to avoid the H2D memcpy.
    uint8_t const* gpuReadPtr = nullptr;
    uint8_t* deviceAlloc = nullptr;
    bool hostRegistered = false;

    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (prop.integrated)
    {
        // Unified memory: GPU can read host memory directly via mapped pointer.
        if (cudaHostRegister(
                const_cast<void*>(static_cast<void const*>(bytes.data())), totalSize, cudaHostRegisterMapped)
            == cudaSuccess)
        {
            void* devicePtr = nullptr;
            if (cudaHostGetDevicePointer(&devicePtr, const_cast<void*>(static_cast<void const*>(bytes.data())), 0)
                == cudaSuccess)
            {
                gpuReadPtr = static_cast<uint8_t const*>(devicePtr);
                hostRegistered = true;
            }
            else
            {
                cudaHostUnregister(const_cast<void*>(static_cast<void const*>(bytes.data())));
            }
        }
    }

    // Discrete GPU or registration failed: pin the source buffer then async-copy to device.
    // bytes.data() is pageable — passing it directly to cudaMemcpyAsync causes CUDA to fall back to a
    // synchronous internal copy. Pinning first enables true async DMA on the provided stream.
    uint8_t* pinnedStaging = nullptr;
    if (gpuReadPtr == nullptr)
    {
        if (cudaMallocHost(&pinnedStaging, totalSize) != cudaSuccess)
        {
            return hashOpaqueIdentityCpu(bytes);
        }
        std::memcpy(pinnedStaging, bytes.data(), totalSize);

        if (cudaMallocAsync(&deviceAlloc, totalSize, stream) != cudaSuccess)
        {
            cudaFreeHost(pinnedStaging);
            return hashOpaqueIdentityCpu(bytes);
        }
        if (cudaMemcpyAsync(deviceAlloc, pinnedStaging, totalSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            cudaFreeAsync(deviceAlloc, stream);
            cudaFreeHost(pinnedStaging);
            return hashOpaqueIdentityCpu(bytes);
        }
        gpuReadPtr = deviceAlloc;
    }

    // Allocate device buffer for kernel output and pinned host buffer for D2H readback.
    size_t const partialsBytes = static_cast<size_t>(kernel::kHASH_NUM_CHUNKS) * 2 * sizeof(uint64_t);
    uint64_t* devicePartials = nullptr;
    uint64_t* pinnedPartials = nullptr;

    if (cudaMallocAsync(&devicePartials, partialsBytes, stream) != cudaSuccess)
    {
        if (hostRegistered)
            cudaHostUnregister(const_cast<void*>(static_cast<void const*>(bytes.data())));
        if (deviceAlloc)
            cudaFreeAsync(deviceAlloc, stream);
        if (pinnedStaging)
            cudaFreeHost(pinnedStaging);
        return hashOpaqueIdentityCpu(bytes);
    }
    if (cudaMallocHost(&pinnedPartials, partialsBytes) != cudaSuccess)
    {
        cudaFreeAsync(devicePartials, stream);
        if (hostRegistered)
            cudaHostUnregister(const_cast<void*>(static_cast<void const*>(bytes.data())));
        if (deviceAlloc)
            cudaFreeAsync(deviceAlloc, stream);
        if (pinnedStaging)
            cudaFreeHost(pinnedStaging);
        return hashOpaqueIdentityCpu(bytes);
    }

    kernel::launchFnv1aHashChunksKernel(
        gpuReadPtr, totalSize, kernel::kHASH_NUM_CHUNKS, kernel::kHASH_NUM_CTAS, devicePartials, stream);
    cudaError_t launchErr = cudaGetLastError();
    cudaMemcpyAsync(pinnedPartials, devicePartials, partialsBytes, cudaMemcpyDeviceToHost, stream);
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (launchErr != cudaSuccess || syncErr != cudaSuccess)
    {
        cudaFreeHost(pinnedPartials);
        cudaFreeAsync(devicePartials, stream);
        if (hostRegistered)
        {
            cudaHostUnregister(const_cast<void*>(static_cast<void const*>(bytes.data())));
        }
        if (deviceAlloc)
        {
            cudaFreeAsync(deviceAlloc, stream);
        }
        if (pinnedStaging)
        {
            cudaFreeHost(pinnedStaging);
        }
        return hashOpaqueIdentityCpu(bytes);
    }

    // Phase 2 reduction on CPU: chain all per-chunk 128-bit digests via FNV-1a.
    FnvValue state = kFNV_OFFSET;
    fnvByte(state, kOPAQUE_IDENTITY_TAG);
    fnvU64(state, static_cast<uint64_t>(totalSize));
    for (int32_t c = 0; c < kernel::kHASH_NUM_CHUNKS; ++c)
    {
        fnvU64(state, pinnedPartials[c * 2]);
        fnvU64(state, pinnedPartials[c * 2 + 1]);
    }
    Hash128 result = finishHash(state);

    cudaFreeHost(pinnedPartials);
    cudaFreeAsync(devicePartials, stream);
    if (hostRegistered)
    {
        cudaHostUnregister(const_cast<void*>(static_cast<void const*>(bytes.data())));
    }
    if (deviceAlloc)
    {
        cudaFreeAsync(deviceAlloc, stream);
    }
    if (pinnedStaging)
    {
        cudaFreeHost(pinnedStaging);
    }
    return result;
}

} // namespace rt
} // namespace trt_edgellm
