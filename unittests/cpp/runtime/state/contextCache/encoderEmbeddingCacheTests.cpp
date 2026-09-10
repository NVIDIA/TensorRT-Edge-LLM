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

#include "runtime/state/contextCache/encoderEmbeddingCache.h"

#include <gtest/gtest.h>

#include <thread>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int64_t kHIDDEN_SIZE = 64;
constexpr int64_t kNUM_TOKENS = 16;
constexpr int64_t kENTRY_BYTES = kNUM_TOKENS * kHIDDEN_SIZE * sizeof(uint16_t); // fp16

Hash128 makeKey(uint64_t seed)
{
    return Hash128{seed, seed ^ 0xDEADBEEFCAFEBABEULL};
}

class EncoderEmbeddingCacheTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    Tensor makeGpuTensor(int64_t numTokens = kNUM_TOKENS, int64_t hiddenSize = kHIDDEN_SIZE)
    {
        return Tensor({numTokens, hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "testEmbedding");
    }

    cudaStream_t mStream{};
};

} // namespace

TEST_F(EncoderEmbeddingCacheTests, LookupMissOnEmptyCache)
{
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    EXPECT_FALSE(cache.lookup(makeKey(1)).has_value());
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.usedBytes(), 0);
}

TEST_F(EncoderEmbeddingCacheTests, StoreAndLookupHit)
{
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    Tensor embedding = makeGpuTensor();
    Hash128 const key = makeKey(42);

    cache.store(key, embedding, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES);

    auto result = cache.lookup(key);
    ASSERT_TRUE(result.has_value());
    Tensor const& cached = result->get();
    EXPECT_EQ(cached.getMemoryCapacity(), kENTRY_BYTES);
}

TEST_F(EncoderEmbeddingCacheTests, SameKeySkipsDuplicateStore)
{
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    Tensor embedding1 = makeGpuTensor();
    Tensor embedding2 = makeGpuTensor();
    Hash128 const key = makeKey(7);

    cache.store(key, embedding1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    cache.store(key, embedding2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES);
}

TEST_F(EncoderEmbeddingCacheTests, DifferentKeyDifferentEntry)
{
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    Tensor embedding1 = makeGpuTensor();
    Tensor embedding2 = makeGpuTensor();
    Hash128 const key1 = makeKey(1);
    Hash128 const key2 = makeKey(2);

    cache.store(key1, embedding1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    cache.store(key2, embedding2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_EQ(cache.usedBytes(), 2 * kENTRY_BYTES);
    EXPECT_TRUE(cache.lookup(key1).has_value());
    EXPECT_TRUE(cache.lookup(key2).has_value());
}

TEST_F(EncoderEmbeddingCacheTests, LruEvictionFreesOldest)
{
    // Budget fits exactly 2 entries.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 2);
    Tensor e1 = makeGpuTensor();
    Tensor e2 = makeGpuTensor();
    Tensor e3 = makeGpuTensor();
    Hash128 const key1 = makeKey(10);
    Hash128 const key2 = makeKey(20);
    Hash128 const key3 = makeKey(30);

    cache.store(key1, e1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cache.store(key2, e2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Storing key3 should evict key1 (oldest).
    cache.store(key3, e3, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_FALSE(cache.lookup(key1).has_value());
    EXPECT_TRUE(cache.lookup(key2).has_value());
    EXPECT_TRUE(cache.lookup(key3).has_value());
}

TEST_F(EncoderEmbeddingCacheTests, LruEvictionPreservesMostRecent)
{
    // Budget fits exactly 2 entries.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 2);
    Tensor e1 = makeGpuTensor();
    Tensor e2 = makeGpuTensor();
    Tensor e3 = makeGpuTensor();
    Hash128 const key1 = makeKey(10);
    Hash128 const key2 = makeKey(20);
    Hash128 const key3 = makeKey(30);

    cache.store(key1, e1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cache.store(key2, e2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Access key1 to make it most-recently used.
    cache.lookup(key1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Storing key3 should evict key2 (now the oldest access).
    cache.store(key3, e3, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_TRUE(cache.lookup(key1).has_value());
    EXPECT_FALSE(cache.lookup(key2).has_value());
    EXPECT_TRUE(cache.lookup(key3).has_value());
}

TEST_F(EncoderEmbeddingCacheTests, BudgetZeroDisablesCache)
{
    EncoderEmbeddingCache cache(0);
    Tensor embedding = makeGpuTensor();
    Hash128 const key = makeKey(99);

    cache.store(key, embedding, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_FALSE(cache.lookup(key).has_value());
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.usedBytes(), 0);
}

TEST_F(EncoderEmbeddingCacheTests, SingleEntryExceedsBudget)
{
    // Budget smaller than one entry.
    EncoderEmbeddingCache cache(kENTRY_BYTES - 1);
    Tensor embedding = makeGpuTensor();
    Hash128 const key = makeKey(55);

    cache.store(key, embedding, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_FALSE(cache.lookup(key).has_value());
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.usedBytes(), 0);
}

TEST_F(EncoderEmbeddingCacheTests, ClearRemovesAllEntries)
{
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    Tensor e1 = makeGpuTensor();
    Tensor e2 = makeGpuTensor();
    Hash128 const key1 = makeKey(1);
    Hash128 const key2 = makeKey(2);

    cache.store(key1, e1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    cache.store(key2, e2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(cache.size(), 2U);

    cache.clear();

    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.usedBytes(), 0);
    EXPECT_FALSE(cache.lookup(key1).has_value());
    EXPECT_FALSE(cache.lookup(key2).has_value());
}

TEST_F(EncoderEmbeddingCacheTests, LookupUpdatesAccessTime)
{
    // Budget fits exactly 2 entries.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 2);
    Tensor e1 = makeGpuTensor();
    Tensor e2 = makeGpuTensor();
    Tensor e3 = makeGpuTensor();
    Hash128 const key1 = makeKey(100);
    Hash128 const key2 = makeKey(200);
    Hash128 const key3 = makeKey(300);

    cache.store(key1, e1, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cache.store(key2, e2, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    // Touch key1 repeatedly to keep it fresh.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cache.lookup(key1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cache.lookup(key1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // key2 is now the LRU entry; inserting key3 should evict key2, not key1.
    cache.store(key3, e3, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_TRUE(cache.lookup(key1).has_value());
    EXPECT_FALSE(cache.lookup(key2).has_value());
    EXPECT_TRUE(cache.lookup(key3).has_value());
}
