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

TEST_F(EncoderEmbeddingCacheTests, RestoresAllFeaturesAfterInterveningEncoderOutput)
{
    // Given two cached images with three deepstack outputs, when another encode overwrites the runner buffers,
    // then cache hits restore every output in the new image order and resize stale feature shapes.
    constexpr int32_t kFEATURES = 3;
    EncoderEmbeddingCache cache(kENTRY_BYTES * (kFEATURES + 1) * 2);
    Tensor embedding = makeGpuTensor(kNUM_TOKENS * 2);
    std::vector<Tensor> features;
    features.reserve(kFEATURES);
    std::vector<std::reference_wrapper<Tensor>> featureRefs;
    std::vector<std::reference_wrapper<Tensor>> outputs{std::ref(embedding)};
    for (int32_t i = 0; i < kFEATURES; ++i)
    {
        features.emplace_back(makeGpuTensor(kNUM_TOKENS * 2));
        featureRefs.emplace_back(std::ref(features.back()));
        outputs.emplace_back(std::ref(features.back()));
    }
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto* ptr = static_cast<char*>(outputs[i].get().rawPointer());
        ASSERT_EQ(cudaMemsetAsync(ptr, 0x10 + i, kENTRY_BYTES, mStream), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(ptr + kENTRY_BYTES, 0x20 + i, kENTRY_BYTES, mStream), cudaSuccess);
    }
    cache.storeSlice(
        makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(), mStream, featureRefs);
    cache.storeSlice(makeKey(2), static_cast<char const*>(embedding.rawPointer()) + kENTRY_BYTES, kNUM_TOKENS,
        kHIDDEN_SIZE, embedding.getDataType(), mStream, featureRefs, kNUM_TOKENS);
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES * (kFEATURES + 1) * 2);
    for (auto const& output : outputs)
    {
        ASSERT_EQ(
            cudaMemsetAsync(output.get().rawPointer(), 0, output.get().getMemoryCapacity(), mStream), cudaSuccess);
    }
    for (auto& feature : features)
    {
        ASSERT_TRUE(feature.reshape({kNUM_TOKENS, kHIDDEN_SIZE}));
    }
    auto first = cache.lookupEntry(makeKey(2));
    auto second = cache.lookupEntry(makeKey(1));
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_EQ(first->get().features.size(), kFEATURES);
    first->get().restore(embedding, featureRefs, 0, mStream);
    second->get().restore(embedding, featureRefs, kNUM_TOKENS, mStream);
    Tensor host({kENTRY_BYTES * 2}, DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        Tensor const& output = outputs[i].get();
        EXPECT_EQ(output.getShape(), Coords({kNUM_TOKENS * 2, kHIDDEN_SIZE}));
        ASSERT_EQ(
            cudaMemcpyAsync(host.rawPointer(), output.rawPointer(), kENTRY_BYTES * 2, cudaMemcpyDeviceToHost, mStream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        for (int64_t byte = 0; byte < kENTRY_BYTES; ++byte)
        {
            ASSERT_EQ(host.dataPointer<uint8_t>()[byte], 0x20 + i);
            ASSERT_EQ(host.dataPointer<uint8_t>()[kENTRY_BYTES + byte], 0x10 + i);
        }
    }
}

TEST_F(EncoderEmbeddingCacheTests, CompleteStateIsBudgetedAndEvictedTogether)
{
    // Given a budget for one embedding plus one feature, when another entry needs space,
    // then both outputs are evicted together and an oversized complete entry does not evict existing data.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 2);
    Tensor embedding = makeGpuTensor();
    Tensor feature = makeGpuTensor();
    cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(), mStream,
        {std::ref(feature)});
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES * 2);
    cache.store(makeKey(2), embedding, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    EXPECT_FALSE(cache.lookupEntry(makeKey(1)));
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES);
    cache.storeSlice(makeKey(3), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(), mStream,
        {std::ref(feature), std::ref(feature)});
    EXPECT_FALSE(cache.lookupEntry(makeKey(3)));
    EXPECT_TRUE(cache.lookup(makeKey(2)));
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    cache.clear();
    EXPECT_EQ(cache.usedBytes(), 0);
}

TEST_F(EncoderEmbeddingCacheTests, RejectsFeatureSlicesOutsideLogicalShape)
{
    // Given an encoder feature whose allocation is larger than its logical shape, when storing an out-of-range
    // slice, then validation rejects it without inserting an incomplete entry.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 4);
    Tensor embedding = makeGpuTensor();
    Tensor feature = makeGpuTensor(kNUM_TOKENS * 2);
    ASSERT_TRUE(feature.reshape({kNUM_TOKENS, kHIDDEN_SIZE}));
    EXPECT_THROW(cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE,
                     embedding.getDataType(), mStream, {std::ref(feature)}, 1),
        std::runtime_error);
    EXPECT_THROW(cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE,
                     embedding.getDataType(), mStream, {std::ref(feature)}, -1),
        std::runtime_error);
    EXPECT_EQ(cache.usedBytes(), 0);
    EXPECT_EQ(cache.size(), 0U);
}

TEST_F(EncoderEmbeddingCacheTests, RejectsIncompatibleRestoreBuffers)
{
    // Given a complete cached output, when the destination has missing features, insufficient tokens or a wrong
    // feature type/width, then restoration fails instead of copying past buffers or leaving stale deepstack data.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 2);
    Tensor embedding = makeGpuTensor();
    Tensor feature = makeGpuTensor();
    cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(), mStream,
        {std::ref(feature)});
    auto hit = cache.lookupEntry(makeKey(1));
    ASSERT_TRUE(hit);
    EXPECT_THROW(hit->get().restore(embedding, {}, 0, mStream), std::runtime_error);
    EXPECT_THROW(hit->get().restore(embedding, {std::ref(feature)}, 1, mStream), std::runtime_error);
    Tensor wrongWidth = makeGpuTensor();
    ASSERT_TRUE(wrongWidth.reshape({kNUM_TOKENS, kHIDDEN_SIZE / 2}));
    EXPECT_THROW(hit->get().restore(embedding, {std::ref(wrongWidth)}, 0, mStream), std::runtime_error);
    Tensor wrongType({kNUM_TOKENS, kHIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    EXPECT_THROW(hit->get().restore(embedding, {std::ref(wrongType)}, 0, mStream), std::runtime_error);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

TEST_F(EncoderEmbeddingCacheTests, RestoresEmbeddingWithoutAuxiliaryFeatures)
{
    // Given an embedding-only encoder entry, when its output buffer is overwritten and restored,
    // then models without deepstack retain the existing cache behavior.
    EncoderEmbeddingCache cache(kENTRY_BYTES);
    Tensor embedding = makeGpuTensor();
    ASSERT_EQ(cudaMemsetAsync(embedding.rawPointer(), 0x37, kENTRY_BYTES, mStream), cudaSuccess);
    cache.store(makeKey(1), embedding, kNUM_TOKENS, kHIDDEN_SIZE, mStream);
    ASSERT_EQ(cudaMemsetAsync(embedding.rawPointer(), 0, kENTRY_BYTES, mStream), cudaSuccess);
    auto hit = cache.lookupEntry(makeKey(1));
    ASSERT_TRUE(hit);
    EXPECT_TRUE(hit->get().features.empty());
    hit->get().restore(embedding, {}, 0, mStream);
    Tensor host({kENTRY_BYTES}, DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    ASSERT_EQ(cudaMemcpyAsync(host.rawPointer(), embedding.rawPointer(), kENTRY_BYTES, cudaMemcpyDeviceToHost, mStream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    for (int64_t byte = 0; byte < kENTRY_BYTES; ++byte)
    {
        ASSERT_EQ(host.dataPointer<uint8_t>()[byte], 0x37);
    }
}

TEST_F(EncoderEmbeddingCacheTests, ZeroTokenSliceIsNotCached)
{
    // Given an empty audio clip, when storing its output, then it consumes no cache budget.
    EncoderEmbeddingCache cache(kENTRY_BYTES);
    Tensor embedding = makeGpuTensor();
    EXPECT_NO_THROW(cache.store(makeKey(1), embedding, 0, kHIDDEN_SIZE, mStream));
    EXPECT_NO_THROW(cache.storeSlice(makeKey(2), nullptr, 0, kHIDDEN_SIZE, embedding.getDataType(), mStream));
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.usedBytes(), 0);
    EXPECT_THROW(cache.store(makeKey(3), embedding, -1, kHIDDEN_SIZE, mStream), std::runtime_error);
}

TEST_F(EncoderEmbeddingCacheTests, LayoutMismatchInvalidatesEntryBeforeRestoringAnyOutput)
{
    // Given two cached images, when the second image's resize changes its token count,
    // then a cache miss leaves all destination buffers unchanged and allows the entry to be replaced.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 8);
    Tensor embedding = makeGpuTensor(kNUM_TOKENS * 3);
    Tensor feature = makeGpuTensor(kNUM_TOKENS * 3);
    ASSERT_EQ(cudaMemsetAsync(embedding.rawPointer(), 0x12, kENTRY_BYTES * 3, mStream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(feature.rawPointer(), 0x34, kENTRY_BYTES * 3, mStream), cudaSuccess);
    for (uint64_t key : {1U, 2U})
    {
        cache.storeSlice(makeKey(key), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(),
            mStream, {std::ref(feature)});
    }
    ASSERT_EQ(cudaMemsetAsync(embedding.rawPointer(), 0, kENTRY_BYTES * 3, mStream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(feature.rawPointer(), 0, kENTRY_BYTES * 3, mStream), cudaSuccess);
    ASSERT_TRUE(feature.reshape({kNUM_TOKENS, kHIDDEN_SIZE}));
    std::vector<std::reference_wrapper<Tensor>> outputs{std::ref(embedding), std::ref(feature)};
    EXPECT_FALSE(cache.tryRestore(
        {makeKey(1), makeKey(2)}, {kNUM_TOKENS, kNUM_TOKENS * 2}, embedding, {std::ref(feature)}, mStream));
    EXPECT_EQ(feature.getShape(), Coords({kNUM_TOKENS, kHIDDEN_SIZE}));
    EXPECT_FALSE(cache.lookup(makeKey(2)));
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES * 2);
    Tensor host({kENTRY_BYTES * 3}, DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    for (auto const& ref : outputs)
    {
        ASSERT_EQ(cudaMemcpyAsync(
                      host.rawPointer(), ref.get().rawPointer(), kENTRY_BYTES * 3, cudaMemcpyDeviceToHost, mStream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        for (int64_t i = 0; i < kENTRY_BYTES * 3; ++i)
        {
            ASSERT_EQ(host.dataPointer<uint8_t>()[i], 0);
        }
    }
    ASSERT_TRUE(feature.reshape({kNUM_TOKENS * 3, kHIDDEN_SIZE}));
    cache.storeSlice(makeKey(2), embedding.rawPointer(), kNUM_TOKENS * 2, kHIDDEN_SIZE, embedding.getDataType(),
        mStream, {std::ref(feature)});
    EXPECT_TRUE(cache.tryRestore(
        {makeKey(1), makeKey(2)}, {kNUM_TOKENS, kNUM_TOKENS * 2}, embedding, {std::ref(feature)}, mStream));
    EXPECT_EQ(cache.usedBytes(), kENTRY_BYTES * 6);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

TEST_F(EncoderEmbeddingCacheTests, InvalidBatchMetadataDoesNotRestore)
{
    // Given cached media, when preprocessing returns incomplete lengths or output buffers,
    // then the caller can re-encode without a failed request or a partially reshaped feature.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 3);
    Tensor embedding = makeGpuTensor(kNUM_TOKENS * 2);
    Tensor feature = makeGpuTensor(kNUM_TOKENS * 2);
    cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS, kHIDDEN_SIZE, embedding.getDataType(), mStream,
        {std::ref(feature)});
    ASSERT_TRUE(feature.reshape({kNUM_TOKENS, kHIDDEN_SIZE}));
    EXPECT_FALSE(cache.tryRestore({makeKey(1)}, {}, embedding, {std::ref(feature)}, mStream));
    EXPECT_FALSE(cache.tryRestore({makeKey(1)}, {kNUM_TOKENS}, embedding, {std::ref(feature)}, mStream));
    EXPECT_EQ(feature.getShape(), Coords({kNUM_TOKENS, kHIDDEN_SIZE}));
    EXPECT_FALSE(cache.tryRestore({makeKey(1)}, {kNUM_TOKENS}, embedding, {}, mStream));
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

TEST_F(EncoderEmbeddingCacheTests, InvalidLaterOutputDoesNotReshapeEarlierFeature)
{
    // Given a valid first feature and an invalid later feature, when restoring an entry,
    // then metadata validation completes before any reshape.
    EncoderEmbeddingCache cache(kENTRY_BYTES * 6);
    Tensor embedding = makeGpuTensor(kNUM_TOKENS * 2);
    Tensor first = makeGpuTensor(kNUM_TOKENS * 2);
    Tensor second = makeGpuTensor(kNUM_TOKENS * 2);
    cache.storeSlice(makeKey(1), embedding.rawPointer(), kNUM_TOKENS * 2, kHIDDEN_SIZE, embedding.getDataType(),
        mStream, {std::ref(first), std::ref(second)});
    ASSERT_TRUE(first.reshape({kNUM_TOKENS, kHIDDEN_SIZE}));
    ASSERT_TRUE(second.reshape({kNUM_TOKENS, kHIDDEN_SIZE / 2}));
    auto entry = cache.lookupEntry(makeKey(1));
    ASSERT_TRUE(entry);
    EXPECT_THROW(entry->get().restore(embedding, {std::ref(first), std::ref(second)}, 0, mStream), std::runtime_error);
    EXPECT_EQ(first.getShape(), Coords({kNUM_TOKENS, kHIDDEN_SIZE}));
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
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
