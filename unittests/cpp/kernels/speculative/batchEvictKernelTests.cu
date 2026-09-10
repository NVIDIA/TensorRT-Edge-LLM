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

#include "common/cudaUtils.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "testUtils.h"
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Helper Functions
// ============================================================================

// Flush L2 cache by reading/writing large buffer
void flushL2Cache()
{
    // 40MB L2 cache size
    constexpr size_t L2_SIZE = 40 * 1024 * 1024;
    static void* flushBuffer = nullptr;

    if (flushBuffer == nullptr)
    {
        CUDA_CHECK(cudaMalloc(&flushBuffer, L2_SIZE));
    }

    // Read and write to flush cache
    CUDA_CHECK(cudaMemset(flushBuffer, 0, L2_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// compactTensorBatch - In-place operation for all types
// ============================================================================
TEST(BatchEvictKernels, CompactTensorBatchInPlace)
{
    cudaStream_t stream = nullptr;

    int32_t oldActiveBatch = 6;
    int32_t newActiveBatch = 4;
    int32_t dim1 = 16;
    int32_t dim2 = 32;

    // Batch mapping: evict batches 1 and 4
    std::vector<int32_t> batchMapping = {0, -1, 1, 2, -1, 3};

    // Prepare shared batchMapping on GPU (used by all sub-tests)
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(batchMappingDevice, batchMapping);

    // Test 1: half type (common for hidden states, logits)
    {
        std::vector<half> srcData(oldActiveBatch * dim1 * dim2);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                srcData[b * dim1 * dim2 + i] = __float2half(static_cast<float>(b * 100 + i) + 0.5f);
            }
        }
        std::vector<half> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kHALF);

        copyHostToDevice(tensorDevice, srcData);

        // In-place compaction
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<half>(tensorDevice);

        // Verify
        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                float expected = __half2float(srcDataBackup[oldIdx * dim1 * dim2 + i]);
                float actual = __half2float(resultData[newIdx * dim1 * dim2 + i]);
                EXPECT_NEAR(actual, expected, 1e-2f); // half precision tolerance
            }
        }
    }

    // Test 2: float type (common for RoPE cache, scores)
    {
        std::vector<float> srcData(oldActiveBatch * dim1 * dim2);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                srcData[b * dim1 * dim2 + i] = static_cast<float>(b * 1000 + i) + 0.123f;
            }
        }
        std::vector<float> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kFLOAT);

        copyHostToDevice(tensorDevice, srcData);

        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<float>(tensorDevice);

        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                EXPECT_FLOAT_EQ(resultData[newIdx * dim1 * dim2 + i], srcDataBackup[oldIdx * dim1 * dim2 + i]);
            }
        }
    }

    // Test 3: int32_t type (common for token IDs, indices)
    {
        int32_t dim1_int = 64; // Typical for token sequences
        std::vector<int32_t> srcData(oldActiveBatch * dim1_int);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1_int; ++i)
            {
                srcData[b * dim1_int + i] = b * 10000 + i;
            }
        }
        std::vector<int32_t> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1_int}, rt::DeviceType::kGPU, DataType::kINT32);

        copyHostToDevice(tensorDevice, srcData);

        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<int32_t>(tensorDevice);

        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1_int; ++i)
            {
                EXPECT_EQ(resultData[newIdx * dim1_int + i], srcDataBackup[oldIdx * dim1_int + i]);
            }
        }
    }
}

// ============================================================================
TEST(BatchEvictKernels, CompactTensorBatchSupportsInt64Indices)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t kOldBatch = 4;
    constexpr int32_t kNewBatch = 2;
    rt::Tensor mapping({kOldBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(mapping, std::vector<int32_t>{-1, 0, -1, 1});

    rt::Tensor indices({kOldBatch, 2}, rt::DeviceType::kGPU, DataType::kINT64);
    copyHostToDevice(indices, std::vector<int64_t>{10, 11, 20, 21, 30, 31, 40, 41});

    compactTensorBatch(indices, mapping, indices, kOldBatch, kNewBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int64_t> const compacted = copyDeviceToHost<int64_t>(indices);
    EXPECT_EQ(std::vector<int64_t>(compacted.begin(), compacted.begin() + 4), (std::vector<int64_t>{20, 21, 40, 41}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(BatchEvictKernels, CompactTensorBatchSupportsInt8TreeMask)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t kOldBatch = 4;
    constexpr int32_t kNewBatch = 2;
    rt::Tensor mapping({kOldBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(mapping, std::vector<int32_t>{-1, 0, -1, 1});

    rt::Tensor mask({kOldBatch, 2, 2}, rt::DeviceType::kGPU, DataType::kINT8);
    copyHostToDevice(mask, std::vector<int8_t>{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33});

    compactTensorBatch(mask, mapping, mask, kOldBatch, kNewBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int8_t> const compacted = copyDeviceToHost<int8_t>(mask);
    EXPECT_EQ(std::vector<int8_t>(compacted.begin(), compacted.begin() + 8),
        (std::vector<int8_t>{10, 11, 12, 13, 30, 31, 32, 33}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Performance Test: compactTensorBatch (In-place)
// ============================================================================
TEST(BatchEvictKernels, DISABLED_CompactTensorBatchPerformance)
{
    cudaStream_t stream = nullptr;

    // Test large tensor in-place compaction (realistic EAGLE scenario)
    int32_t oldActiveBatch = 8;
    int32_t newActiveBatch = 6;
    int32_t dim1 = 4096;
    int32_t dim2 = 4096;

    std::vector<int32_t> batchMapping = {0, -1, 1, 2, 3, -1, 4, 5};

    std::vector<half> srcData(oldActiveBatch * dim1 * dim2);
    uniformFloatInitialization(srcData, -1.0f, 1.0f);

    rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(batchMappingDevice, batchMapping);

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        copyHostToDevice(tensorDevice, srcData);
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Performance test with cold cache
    constexpr int numIterations = 100;
    std::vector<float> timings;
    timings.reserve(numIterations);

    for (int iter = 0; iter < numIterations; ++iter)
    {
        // Flush L2 cache to ensure cold data
        flushL2Cache();

        // Reset source data
        copyHostToDevice(tensorDevice, srcData);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        // In-place compaction
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Calculate statistics
    float sum = 0.0f;
    float minTime = timings[0];
    float maxTime = timings[0];
    for (float t : timings)
    {
        sum += t;
        minTime = std::min(minTime, t);
        maxTime = std::max(maxTime, t);
    }
    float avgTime = sum / numIterations;

    // Calculate actual data moved based on batchMapping
    // Count batches that need to be moved: newIdx >= 0 && oldIdx != newIdx
    int32_t numBatchesMoved = 0;
    for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
    {
        int32_t newIdx = batchMapping[oldIdx];
        if (newIdx >= 0 && oldIdx != newIdx)
        {
            numBatchesMoved++;
        }
    }
    size_t totalBytes = static_cast<size_t>(numBatchesMoved) * dim1 * dim2 * sizeof(half);
    float avgBandwidthGB = (totalBytes / (1024.0f * 1024.0f * 1024.0f)) / (avgTime / 1000.0f);

    std::cout << "\n=== compactTensorBatch In-place Performance (Cold Cache) ===" << std::endl;
    std::cout << "Configuration: " << oldActiveBatch << " -> " << newActiveBatch << " batches, [" << dim1 << ", "
              << dim2 << "] per batch" << std::endl;
    std::cout << "Batches actually moved: " << numBatchesMoved << std::endl;
    std::cout << "Average time: " << avgTime << " ms" << std::endl;
    std::cout << "Min time: " << minTime << " ms" << std::endl;
    std::cout << "Max time: " << maxTime << " ms" << std::endl;
    std::cout << "Average bandwidth: " << avgBandwidthGB << " GB/s" << std::endl;
    std::cout << "Data moved: " << (totalBytes / (1024.0f * 1024.0f)) << " MB" << std::endl;
}
