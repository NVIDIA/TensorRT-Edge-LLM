/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

void TestLongRopeCosSin(int32_t rotaryDim, int32_t kvCacheCapacity, int32_t maxPositionEmbeddings = 131072,
    int32_t originalMaxPositionEmbeddings = 4096, float rotaryBaseFrequency = 10000.0f)
{
    // Generate random extension factors
    std::vector<float> shortReference(kvCacheCapacity * rotaryDim);
    std::vector<float> longReference(kvCacheCapacity * rotaryDim);
    std::vector<float> shortFactor(rotaryDim / 2, 1.0f);
    std::vector<float> longFactor(rotaryDim / 2);
    uniformFloatInitialization(longFactor, 1.0f, float(rotaryDim / 2 - 1));

    computeLongRopeReference(shortReference, longReference, shortFactor, longFactor, rotaryBaseFrequency, rotaryDim,
        kvCacheCapacity, maxPositionEmbeddings, originalMaxPositionEmbeddings);

    // Allocate device memory
    thrust::device_vector<float> shortCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> longCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> shortFactorDevice(shortFactor);
    thrust::device_vector<float> longFactorDevice(longFactor);

    cudaStream_t stream{nullptr};

    // Launch kernel
    initializeLongRopeCosSin(thrust::raw_pointer_cast(shortCosSinCacheDevice.data()),
        thrust::raw_pointer_cast(longCosSinCacheDevice.data()), thrust::raw_pointer_cast(shortFactorDevice.data()),
        thrust::raw_pointer_cast(longFactorDevice.data()), rotaryBaseFrequency, rotaryDim, kvCacheCapacity,
        maxPositionEmbeddings, originalMaxPositionEmbeddings, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back to host
    thrust::host_vector<float> shortCosSinCacheHost(shortCosSinCacheDevice);
    thrust::host_vector<float> longCosSinCacheHost(longCosSinCacheDevice);

    // Verify short cache results
    for (int32_t i = 0; i < kvCacheCapacity * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(shortCosSinCacheHost[i], shortReference[i], 1e-3, 1e-3))
            << "Short cache mismatch at index " << i << ": got " << shortCosSinCacheHost[i] << ", expected "
            << shortReference[i];
    }

    // Verify long cache results
    for (int32_t i = 0; i < kvCacheCapacity * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(longCosSinCacheHost[i], longReference[i], 1e-3, 1e-3))
            << "Long cache mismatch at index " << i << ": got " << longCosSinCacheHost[i] << ", expected "
            << longReference[i];
    }

    std::cout << "TestLongRopeCosSin passed: rotaryDim=" << rotaryDim << ", kvCacheCapacity=" << kvCacheCapacity
              << ", maxPositionEmbeddings=" << maxPositionEmbeddings
              << ", originalMaxPositionEmbeddings=" << originalMaxPositionEmbeddings
              << ", rotaryBaseFrequency=" << rotaryBaseFrequency << std::endl;
}

void BenchmarkLongRopeCosSin(int32_t rotaryDim, int32_t kvCacheCapacity, int32_t maxPositionEmbeddings = 131072,
    int32_t originalMaxPositionEmbeddings = 4096)
{
    std::vector<float> shortFactor(rotaryDim / 2, 1.0f);
    std::vector<float> longFactor(rotaryDim / 2);
    uniformFloatInitialization(longFactor, 1.0f, float(rotaryDim / 2 - 1));

    thrust::device_vector<float> shortCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> longCosSinCacheDevice(kvCacheCapacity * rotaryDim);
    thrust::device_vector<float> shortFactorDevice(shortFactor);
    thrust::device_vector<float> longFactorDevice(longFactor);

    cudaStream_t stream{nullptr};

    auto launch = [&]() {
        initializeLongRopeCosSin(thrust::raw_pointer_cast(shortCosSinCacheDevice.data()),
            thrust::raw_pointer_cast(longCosSinCacheDevice.data()), thrust::raw_pointer_cast(shortFactorDevice.data()),
            thrust::raw_pointer_cast(longFactorDevice.data()), 10000.0f, rotaryDim, kvCacheCapacity,
            maxPositionEmbeddings, originalMaxPositionEmbeddings, stream);
    };

    // Warmup
    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "LongRopeCosSin Benchmark: rotaryDim=" << rotaryDim << ", kvCacheCapacity=" << kvCacheCapacity
              << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(InitializeLongRopeCosSin, Accuracy)
{
    TestLongRopeCosSin(96, 8192);
    TestLongRopeCosSin(128, 4096);
}

TEST(InitializeLongRopeCosSin, Benchmark)
{
    BenchmarkLongRopeCosSin(96, 8192);
    BenchmarkLongRopeCosSin(128, 4096);
}

void TestMRopeCosSin(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    float rotaryBaseFrequency = 10000.0f, bool interleaved = false)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    std::vector<float> reference(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    computeMRopeReference(reference, mropePositionIds, rotaryBaseFrequency, rotaryDim, rotaryEmbeddingMaxPositions,
        batchSize, interleaved);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    // Launch kernel
    initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
        thrust::raw_pointer_cast(mropePositionIdsDevice.data()), rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, interleaved, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back to host
    thrust::host_vector<float> cosSinCacheHost(cosSinCacheDevice);

    // Verify results
    for (int32_t i = 0; i < batchSize * rotaryEmbeddingMaxPositions * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(cosSinCacheHost[i], reference[i], 1e-3, 1e-3))
            << "MRope cache mismatch at index " << i << ": got " << cosSinCacheHost[i] << ", expected " << reference[i];
    }

    std::cout << "TestMRopeCosSin passed: rotaryDim=" << rotaryDim
              << ", rotaryEmbeddingMaxPositions=" << rotaryEmbeddingMaxPositions << ", batchSize=" << batchSize
              << ", rotaryBaseFrequency=" << rotaryBaseFrequency << ", interleaved=" << interleaved << std::endl;
}

void BenchmarkMRopeCosSin(
    int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize, bool interleaved = false)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    auto launch = [&]() {
        initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
            thrust::raw_pointer_cast(mropePositionIdsDevice.data()), 10000.0f, rotaryDim, rotaryEmbeddingMaxPositions,
            batchSize, interleaved, stream);
    };

    // Warmup
    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "MRopeCosSin Benchmark: rotaryDim=" << rotaryDim
              << ", rotaryEmbeddingMaxPositions=" << rotaryEmbeddingMaxPositions << ", batchSize=" << batchSize
              << ", interleaved=" << interleaved << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(InitializeMRopeCosSin, Accuracy)
{
    TestMRopeCosSin(128, 4096, 2);
    TestMRopeCosSin(128, 8192, 1);
    TestMRopeCosSin(128, 4096, 2, 5000000.0f, true);
    TestMRopeCosSin(128, 500, 1, 5000000.0f, true);
}

TEST(InitializeMRopeCosSin, Benchmark)
{
    BenchmarkMRopeCosSin(128, 4096, 2);
    BenchmarkMRopeCosSin(128, 8192, 1);
    BenchmarkMRopeCosSin(128, 4096, 2, true);
    BenchmarkMRopeCosSin(128, 8192, 1, true);
}
