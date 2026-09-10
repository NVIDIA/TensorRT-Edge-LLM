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

#include "runtime/state/ropeCache.h"

#include "common/checkMacros.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

class RopeCacheTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    cudaStream_t mStream{nullptr};
};

TEST_F(RopeCacheTest, DeduplicationSameConfig)
{
    RopeCache cache;

    RopeConfig config;
    config.type = RopeType::kDefault;
    config.rotaryTheta = 10000.0F;
    config.rotaryScale = 1.0F;
    config.maxPositionEmbeddings = 2048;

    int32_t const rotaryDim = 64;
    int32_t const maxSeqLen = 256;

    // First call creates a new entry.
    rt::Tensor& t1 = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    EXPECT_EQ(cache.size(), 1U);

    // Second call with identical config should return the same tensor.
    rt::Tensor& t2 = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    EXPECT_EQ(cache.size(), 1U);

    // Must be the exact same object (address).
    EXPECT_EQ(&t1, &t2);
}

TEST_F(RopeCacheTest, DifferentConfigsCreateSeparateEntries)
{
    RopeCache cache;

    RopeConfig configA;
    configA.type = RopeType::kDefault;
    configA.rotaryTheta = 10000.0F;
    configA.rotaryScale = 1.0F;
    configA.maxPositionEmbeddings = 2048;

    RopeConfig configB;
    configB.type = RopeType::kDefault;
    configB.rotaryTheta = 500000.0F; // Different theta
    configB.rotaryScale = 1.0F;
    configB.maxPositionEmbeddings = 2048;

    int32_t const rotaryDim = 64;
    int32_t const maxSeqLen = 256;

    rt::Tensor& t1 = cache.getOrCreate(configA, rotaryDim, maxSeqLen, mStream);
    rt::Tensor& t2 = cache.getOrCreate(configB, rotaryDim, maxSeqLen, mStream);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_NE(&t1, &t2);
}

TEST_F(RopeCacheTest, DifferentRotaryDimCreatesNewEntry)
{
    RopeCache cache;

    RopeConfig config;
    config.type = RopeType::kDefault;
    config.rotaryTheta = 10000.0F;

    int32_t const maxSeqLen = 256;

    rt::Tensor& t1 = cache.getOrCreate(config, 64, maxSeqLen, mStream);
    rt::Tensor& t2 = cache.getOrCreate(config, 128, maxSeqLen, mStream);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_NE(&t1, &t2);

    // Shape should reflect rotary dim.
    EXPECT_EQ(t1.getShape()[2], 64);
    EXPECT_EQ(t2.getShape()[2], 128);
}

TEST_F(RopeCacheTest, DifferentMaxSeqLenCreatesNewEntry)
{
    RopeCache cache;

    RopeConfig config;
    config.type = RopeType::kDefault;
    config.rotaryTheta = 10000.0F;

    int32_t const rotaryDim = 64;

    rt::Tensor& t1 = cache.getOrCreate(config, rotaryDim, 256, mStream);
    rt::Tensor& t2 = cache.getOrCreate(config, rotaryDim, 512, mStream);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_NE(&t1, &t2);
}

TEST_F(RopeCacheTest, PointerStabilityAcrossInsertions)
{
    RopeCache cache;

    RopeConfig config;
    config.type = RopeType::kDefault;
    config.rotaryTheta = 10000.0F;

    int32_t const rotaryDim = 64;
    int32_t const maxSeqLen = 256;

    // Create first entry, record its address.
    rt::Tensor& first = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    void* firstAddr = &first;

    // Add several more entries with different configs.
    for (int32_t i = 1; i <= 10; ++i)
    {
        RopeConfig other;
        other.type = RopeType::kDefault;
        other.rotaryTheta = 10000.0F * static_cast<float>(i + 1);
        cache.getOrCreate(other, rotaryDim, maxSeqLen, mStream);
    }

    EXPECT_EQ(cache.size(), 11U);

    // The first entry's reference must still be valid (deque guarantee).
    rt::Tensor& firstAgain = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    EXPECT_EQ(static_cast<void*>(&firstAgain), firstAddr);
    // Still 11 entries (no new entry created).
    EXPECT_EQ(cache.size(), 11U);
}

TEST_F(RopeCacheTest, TensorShapeAndType)
{
    RopeCache cache;

    RopeConfig config;
    config.type = RopeType::kDefault;
    config.rotaryTheta = 10000.0F;
    config.maxPositionEmbeddings = 4096;

    int32_t const rotaryDim = 128;
    int32_t const maxSeqLen = 1024;

    rt::Tensor& t = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);

    // Expect shape [1, maxSeqLen, rotaryDim].
    EXPECT_EQ(t.getShape().getNumDims(), 3);
    EXPECT_EQ(t.getShape()[0], 1);
    EXPECT_EQ(t.getShape()[1], maxSeqLen);
    EXPECT_EQ(t.getShape()[2], rotaryDim);
    EXPECT_EQ(t.getDeviceType(), DeviceType::kGPU);
    EXPECT_EQ(t.getDataType(), nvinfer1::DataType::kFLOAT);
}

TEST_F(RopeCacheTest, NoRopeDeduplication)
{
    RopeCache cache;

    // Two NoRope configs with different thetas should still deduplicate,
    // because theta is irrelevant for NoRope.
    RopeConfig configA;
    configA.type = RopeType::kNoRope;
    configA.rotaryTheta = 10000.0F;

    RopeConfig configB;
    configB.type = RopeType::kNoRope;
    configB.rotaryTheta = 999999.0F;

    int32_t const rotaryDim = 64;
    int32_t const maxSeqLen = 256;

    rt::Tensor& t1 = cache.getOrCreate(configA, rotaryDim, maxSeqLen, mStream);
    rt::Tensor& t2 = cache.getOrCreate(configB, rotaryDim, maxSeqLen, mStream);

    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(&t1, &t2);
}

TEST_F(RopeCacheTest, DifferentRopeTypesNotDeduplicated)
{
    RopeCache cache;

    RopeConfig configDefault;
    configDefault.type = RopeType::kDefault;
    configDefault.rotaryTheta = 10000.0F;

    RopeConfig configDynamic;
    configDynamic.type = RopeType::kDynamic;
    configDynamic.rotaryTheta = 10000.0F;

    int32_t const rotaryDim = 64;
    int32_t const maxSeqLen = 256;

    rt::Tensor& t1 = cache.getOrCreate(configDefault, rotaryDim, maxSeqLen, mStream);
    rt::Tensor& t2 = cache.getOrCreate(configDynamic, rotaryDim, maxSeqLen, mStream);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_NE(&t1, &t2);
}

TEST_F(RopeCacheTest, LongRopeInitializesThroughCache)
{
    RopeCache cache;

    int32_t const rotaryDim = 96;
    int32_t const maxSeqLen = 256;

    RopeConfig config;
    config.type = RopeType::kLongRope;
    config.rotaryTheta = 10000.0F;
    config.maxPositionEmbeddings = 4096;
    LongRopeParams params;
    params.originalMaxPositionEmbeddings = 2048;
    params.shortFactor.assign(rotaryDim / 2, 1.0F);
    params.longFactor.assign(rotaryDim / 2, 2.0F);
    config.longRope = std::move(params);

    rt::Tensor& t1 = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(t1.getShape()[0], 1);
    EXPECT_EQ(t1.getShape()[1], maxSeqLen);
    EXPECT_EQ(t1.getShape()[2], rotaryDim);

    rt::Tensor& t2 = cache.getOrCreate(config, rotaryDim, maxSeqLen, mStream);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(&t1, &t2);
}
