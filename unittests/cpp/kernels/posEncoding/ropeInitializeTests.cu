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

#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

void computeNormalRopeReference(std::vector<float>& cosSinCache, float rotaryBaseFrequency, float rotaryScale,
    float partialRotaryFactor, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions)
{
    float const clampedPartialRotaryFactor
        = partialRotaryFactor < 0.0F ? 0.0F : (partialRotaryFactor > 1.0F ? 1.0F : partialRotaryFactor);
    int32_t const rotatedAngles = static_cast<int32_t>(clampedPartialRotaryFactor * static_cast<float>(rotaryDim / 2));

    for (int32_t posIdx = 0; posIdx < rotaryEmbeddingMaxPositions; ++posIdx)
    {
        int32_t const cosSinOffset = posIdx * rotaryDim;
        for (int32_t zid = 0; zid < rotaryDim / 2; ++zid)
        {
            if (zid >= rotatedAngles)
            {
                cosSinCache[cosSinOffset + zid] = 1.0F;
                cosSinCache[cosSinOffset + zid + rotaryDim / 2] = 0.0F;
                continue;
            }
            float const ropeConstant = std::pow(rotaryBaseFrequency, 2.0F * static_cast<float>(zid) / rotaryDim);
            float const invFreq = posIdx * rotaryScale / ropeConstant;
            cosSinCache[cosSinOffset + zid] = std::cos(invFreq);
            cosSinCache[cosSinOffset + zid + rotaryDim / 2] = std::sin(invFreq);
        }
    }
}

void TestNormalRopeCosSin(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, float rotaryBaseFrequency = 10000.0F,
    float rotaryScale = 1.0F, float partialRotaryFactor = 1.0F)
{
    std::vector<float> reference(rotaryEmbeddingMaxPositions * rotaryDim);
    computeNormalRopeReference(
        reference, rotaryBaseFrequency, rotaryScale, partialRotaryFactor, rotaryDim, rotaryEmbeddingMaxPositions);

    thrust::device_vector<float> cosSinCacheDevice(rotaryEmbeddingMaxPositions * rotaryDim);
    cudaStream_t stream{nullptr};

    initializeNormalRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()), rotaryBaseFrequency, rotaryScale,
        partialRotaryFactor, rotaryDim, rotaryEmbeddingMaxPositions, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::host_vector<float> cosSinCacheHost(cosSinCacheDevice);
    for (int32_t i = 0; i < rotaryEmbeddingMaxPositions * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(cosSinCacheHost[i], reference[i], 1e-3, 1e-3))
            << "Normal RoPE cache mismatch at index " << i << ": got " << cosSinCacheHost[i] << ", expected "
            << reference[i];
    }

    std::cout << "TestNormalRopeCosSin passed: rotaryDim=" << rotaryDim
              << ", rotaryEmbeddingMaxPositions=" << rotaryEmbeddingMaxPositions
              << ", rotaryBaseFrequency=" << rotaryBaseFrequency << ", rotaryScale=" << rotaryScale
              << ", partialRotaryFactor=" << partialRotaryFactor << std::endl;
}

TEST(InitializeNormalRopeCosSin, Accuracy)
{
    TestNormalRopeCosSin(32, 256);
    TestNormalRopeCosSin(64, 256);
    TestNormalRopeCosSin(96, 256);
    TestNormalRopeCosSin(128, 256);
    TestNormalRopeCosSin(256, 256, 10000.0F, 0.5F, 0.5F);
}

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

// YaRN: the kernel only does the position sweep on a precomputed per-dim inverse
// frequency, so the reference recomputes the same NTK-by-parts inv_freq (matching
// computeYarnInvFreq in the runtime) and fills cos(pos*invFreq)*mscale.
void TestYarnRopeCosSin(int32_t rotaryDim, int32_t maxPositions, float rotaryBaseFrequency = 10000.0F,
    float factor = 32.0F, int32_t originalMaxPositionEmbeddings = 8192, float betaFast = 32.0F, float betaSlow = 1.0F)
{
    int32_t const halfDim = rotaryDim / 2;
    float const kPi = 3.14159265358979323846F;
    auto correctionDim = [&](float numRotations) {
        return (static_cast<float>(rotaryDim)
                   * std::log(static_cast<float>(originalMaxPositionEmbeddings) / (numRotations * 2.0F * kPi)))
            / (2.0F * std::log(rotaryBaseFrequency));
    };
    float low = std::max(std::floor(correctionDim(betaFast)), 0.0F);
    float high = std::min(std::ceil(correctionDim(betaSlow)), static_cast<float>(rotaryDim - 1));
    float denom = high - low;
    if (denom == 0.0F)
    {
        denom = 0.001F;
    }
    std::vector<float> invFreq(halfDim);
    for (int32_t d = 0; d < halfDim; ++d)
    {
        float const posFreq
            = std::pow(rotaryBaseFrequency, 2.0F * static_cast<float>(d) / static_cast<float>(rotaryDim));
        float const extrap = 1.0F / posFreq;
        float const interp = 1.0F / (factor * posFreq);
        float const ramp = std::clamp((static_cast<float>(d) - low) / denom, 0.0F, 1.0F);
        float const extrapFactor = 1.0F - ramp;
        invFreq[d] = interp * (1.0F - extrapFactor) + extrap * extrapFactor;
    }
    float const mscale = factor > 1.0F ? 0.1F * std::log(factor) + 1.0F : 1.0F;

    std::vector<float> reference(maxPositions * rotaryDim);
    for (int32_t pos = 0; pos < maxPositions; ++pos)
    {
        for (int32_t d = 0; d < halfDim; ++d)
        {
            float const angle = static_cast<float>(pos) * invFreq[d];
            reference[pos * rotaryDim + d] = std::cos(angle) * mscale;
            reference[pos * rotaryDim + d + halfDim] = std::sin(angle) * mscale;
        }
    }

    thrust::device_vector<float> cosSinCacheDevice(maxPositions * rotaryDim);
    thrust::device_vector<float> invFreqDevice(invFreq);
    cudaStream_t stream{nullptr};
    initializeYarnCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
        thrust::raw_pointer_cast(invFreqDevice.data()), mscale, rotaryDim, maxPositions, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::host_vector<float> cosSinCacheHost(cosSinCacheDevice);
    for (int32_t i = 0; i < maxPositions * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(cosSinCacheHost[i], reference[i], 1e-3, 1e-3))
            << "YaRN RoPE cache mismatch at index " << i << ": got " << cosSinCacheHost[i] << ", expected "
            << reference[i];
    }
    std::cout << "TestYarnRopeCosSin passed: rotaryDim=" << rotaryDim << ", maxPositions=" << maxPositions << std::endl;
}

TEST(InitializeYarnRopeCosSin, Accuracy)
{
    TestYarnRopeCosSin(64, 256);
    TestYarnRopeCosSin(128, 4096);
    TestYarnRopeCosSin(128, 16384); // beyond originalMaxPositionEmbeddings=8192
}

void TestMRopeCosSin(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    float rotaryBaseFrequency = 10000.0f, bool interleaved = false, int32_t sectionH = 20, int32_t sectionW = 20)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    std::vector<float> reference(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    computeMRopeReference(reference, mropePositionIds, rotaryBaseFrequency, rotaryDim, rotaryEmbeddingMaxPositions,
        batchSize, interleaved, sectionH, sectionW);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    // Launch kernel
    initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
        thrust::raw_pointer_cast(mropePositionIdsDevice.data()), rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, interleaved, sectionH, sectionW, stream);

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

void BenchmarkMRopeCosSin(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    bool interleaved = false, int32_t sectionH = 20, int32_t sectionW = 20)
{
    std::vector<int64_t> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    uniformIntInitialization(mropePositionIds, 0, rotaryEmbeddingMaxPositions - 1);

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<int64_t> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};

    auto launch = [&]() {
        initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
            thrust::raw_pointer_cast(mropePositionIdsDevice.data()), 10000.0f, rotaryDim, rotaryEmbeddingMaxPositions,
            batchSize, interleaved, sectionH, sectionW, stream);
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

//! Fractional temporal positions (Qwen3-Omni fps-modulated MRoPE). The shared
//! reference indexes a precomputed integer-position table, so the expectation is
//! recomputed here with the kernel's own formula.
void TestMRopeCosSinFractional(int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    float rotaryBaseFrequency, bool interleaved, int32_t sectionH, int32_t sectionW, float temporalStep)
{
    int32_t const halfDim = rotaryDim / 2;
    int32_t const numTemporalPairs = halfDim - sectionH - sectionW;

    std::vector<float> mropePositionIds(batchSize * 3 * rotaryEmbeddingMaxPositions);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < rotaryEmbeddingMaxPositions; ++i)
        {
            int32_t const base = b * 3 * rotaryEmbeddingMaxPositions;
            mropePositionIds[base + 0 * rotaryEmbeddingMaxPositions + i] = i * temporalStep;
            mropePositionIds[base + 1 * rotaryEmbeddingMaxPositions + i] = static_cast<float>(i);
            mropePositionIds[base + 2 * rotaryEmbeddingMaxPositions + i] = static_cast<float>(i);
        }
    }

    std::vector<float> ropeConstants(halfDim);
    for (int32_t j = 0; j < halfDim; ++j)
    {
        ropeConstants[j] = std::pow(rotaryBaseFrequency, 2 * j / static_cast<float>(rotaryDim));
    }

    std::vector<float> reference(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < rotaryEmbeddingMaxPositions; ++i)
        {
            for (int32_t j = 0; j < halfDim; ++j)
            {
                int32_t sec;
                if (interleaved)
                {
                    int32_t const mod3 = j % 3;
                    sec = (mod3 == 1 && j < sectionH * 3) ? 1 : ((mod3 == 2 && j < sectionW * 3) ? 2 : 0);
                }
                else
                {
                    sec = (j < numTemporalPairs) ? 0 : ((j < numTemporalPairs + sectionH) ? 1 : 2);
                }
                float const pos
                    = mropePositionIds[b * 3 * rotaryEmbeddingMaxPositions + sec * rotaryEmbeddingMaxPositions + i];
                int32_t const dst = b * rotaryEmbeddingMaxPositions * rotaryDim + i * rotaryDim + j;
                reference[dst] = std::cos(pos / ropeConstants[j]);
                reference[dst + halfDim] = std::sin(pos / ropeConstants[j]);
            }
        }
    }

    thrust::device_vector<float> cosSinCacheDevice(batchSize * rotaryEmbeddingMaxPositions * rotaryDim);
    thrust::device_vector<float> mropePositionIdsDevice(mropePositionIds);

    cudaStream_t stream{nullptr};
    initializeMRopeCosSin(thrust::raw_pointer_cast(cosSinCacheDevice.data()),
        thrust::raw_pointer_cast(mropePositionIdsDevice.data()), rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, interleaved, sectionH, sectionW, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::host_vector<float> cosSinCacheHost(cosSinCacheDevice);
    for (int32_t i = 0; i < batchSize * rotaryEmbeddingMaxPositions * rotaryDim; ++i)
    {
        ASSERT_TRUE(isclose(cosSinCacheHost[i], reference[i], 1e-3, 1e-3))
            << "MRope fractional cache mismatch at index " << i << ": got " << cosSinCacheHost[i] << ", expected "
            << reference[i];
    }
}

TEST(InitializeMRopeCosSin, Accuracy)
{
    // Qwen2-VL: rotaryDim=128, non-interleaved, section [16,24,24]
    TestMRopeCosSin(128, 4096, 2, 10000.0f, false, 24, 24);
    TestMRopeCosSin(128, 8192, 1, 10000.0f, false, 24, 24);
    // Qwen3-VL: rotaryDim=128, interleaved, section [24,20,20]
    TestMRopeCosSin(128, 4096, 2, 5000000.0f, true, 20, 20);
    TestMRopeCosSin(128, 500, 1, 5000000.0f, true, 20, 20);
    // Qwen3.5: rotaryDim=64, interleaved, section [11,11,10]
    TestMRopeCosSin(64, 4096, 2, 10000000.0f, true, 11, 10);
    TestMRopeCosSin(64, 500, 1, 10000000.0f, true, 11, 10);
}

TEST(InitializeMRopeCosSin, FractionalTemporalPositions)
{
    // Qwen3-Omni step = (temporalPatchSize / fps) * positionIdPerSecond; at
    // temporalPatchSize=2, positionIdPerSecond=13 that is 13 (fps=2) and 6.5 (fps=4).
    TestMRopeCosSinFractional(128, 512, 2, 5000000.0f, true, 20, 20, 6.5f);
    TestMRopeCosSinFractional(128, 512, 1, 5000000.0f, true, 20, 20, 3.25f);
    TestMRopeCosSinFractional(128, 512, 1, 10000.0f, false, 24, 24, 6.5f);
}

TEST(InitializeMRopeCosSin, Benchmark)
{
    BenchmarkMRopeCosSin(128, 4096, 2, false, 24, 24);
    BenchmarkMRopeCosSin(128, 8192, 1, false, 24, 24);
    BenchmarkMRopeCosSin(128, 4096, 2, true, 20, 20);
    BenchmarkMRopeCosSin(128, 8192, 1, true, 20, 20);
    BenchmarkMRopeCosSin(64, 4096, 2, true, 11, 10);
}
