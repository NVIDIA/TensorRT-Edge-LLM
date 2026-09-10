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

#include "common/checkMacros.h"
#include "common/logger.h"

#include <algorithm>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

EncoderEmbeddingCache::EncoderEmbeddingCache(int64_t maxBudgetBytes)
    : mBudgetBytes(maxBudgetBytes)
{
}

std::optional<std::reference_wrapper<Tensor const>> EncoderEmbeddingCache::lookup(Hash128 key)
{
    if (mBudgetBytes <= 0)
    {
        return std::nullopt;
    }

    auto it = mEntries.find(key);
    if (it == mEntries.end())
    {
        return std::nullopt;
    }

    it->second.lastAccess = std::chrono::steady_clock::now();
    return std::cref(it->second.embedding);
}

void EncoderEmbeddingCache::store(
    Hash128 key, Tensor const& embedding, int64_t numTokens, int64_t hiddenSize, cudaStream_t stream)
{
    if (mBudgetBytes <= 0)
    {
        return;
    }

    // If already cached under this key, skip (embeddings are deterministic).
    auto it = mEntries.find(key);
    if (it != mEntries.end())
    {
        it->second.lastAccess = std::chrono::steady_clock::now();
        return;
    }

    // Compute entry size from actual dimensions, not the source tensor's capacity (which may be over-allocated
    // when the output embedding is a reshaped view over a max-capacity buffer).
    int64_t const entryBytes
        = numTokens * hiddenSize * static_cast<int64_t>(utils::getTypeSize(embedding.getDataType()));
    if (entryBytes > mBudgetBytes)
    {
        // Single entry exceeds entire budget — cannot cache.
        return;
    }

    evictUntilFits(entryBytes);

    // Allocate a new GPU tensor and async-copy the embedding data.
    Tensor cached({numTokens, hiddenSize}, DeviceType::kGPU, embedding.getDataType(), "encoderEmbeddingCache");
    CUDA_CHECK(
        cudaMemcpyAsync(cached.rawPointer(), embedding.rawPointer(), entryBytes, cudaMemcpyDeviceToDevice, stream));

    EncoderEmbeddingCacheEntry entry{};
    entry.embedding = std::move(cached);
    entry.numTokens = numTokens;
    entry.hiddenSize = hiddenSize;
    entry.lastAccess = std::chrono::steady_clock::now();

    mEntries.emplace(key, std::move(entry));
    mUsedBytes += entryBytes;

    LOG_DEBUG("EncoderEmbeddingCache: stored entry (%lld tokens, %lld bytes). Used: %lld / %lld bytes.",
        static_cast<long long>(numTokens), static_cast<long long>(entryBytes), static_cast<long long>(mUsedBytes),
        static_cast<long long>(mBudgetBytes));
}

void EncoderEmbeddingCache::storeSlice(Hash128 key, void const* devicePtr, int64_t numTokens, int64_t hiddenSize,
    nvinfer1::DataType dtype, cudaStream_t stream)
{
    if (mBudgetBytes <= 0)
    {
        return;
    }

    auto it = mEntries.find(key);
    if (it != mEntries.end())
    {
        it->second.lastAccess = std::chrono::steady_clock::now();
        return;
    }

    int64_t const entryBytes = numTokens * hiddenSize * static_cast<int64_t>(utils::getTypeSize(dtype));
    if (entryBytes > mBudgetBytes)
    {
        return;
    }

    evictUntilFits(entryBytes);

    Tensor cached({numTokens, hiddenSize}, DeviceType::kGPU, dtype, "encoderEmbeddingCacheSlice");
    CUDA_CHECK(cudaMemcpyAsync(cached.rawPointer(), devicePtr, entryBytes, cudaMemcpyDeviceToDevice, stream));

    EncoderEmbeddingCacheEntry entry{};
    entry.embedding = std::move(cached);
    entry.numTokens = numTokens;
    entry.hiddenSize = hiddenSize;
    entry.lastAccess = std::chrono::steady_clock::now();

    mEntries.emplace(key, std::move(entry));
    mUsedBytes += entryBytes;

    LOG_DEBUG("EncoderEmbeddingCache: stored slice (%lld tokens, %lld bytes). Used: %lld / %lld bytes.",
        static_cast<long long>(numTokens), static_cast<long long>(entryBytes), static_cast<long long>(mUsedBytes),
        static_cast<long long>(mBudgetBytes));
}

void EncoderEmbeddingCache::clear()
{
    mEntries.clear();
    mUsedBytes = 0;
}

void EncoderEmbeddingCache::evictUntilFits(int64_t requiredBytes)
{
    while (mUsedBytes + requiredBytes > mBudgetBytes && !mEntries.empty())
    {
        // Find LRU entry (oldest lastAccess).
        auto oldest = mEntries.begin();
        for (auto it = mEntries.begin(); it != mEntries.end(); ++it)
        {
            if (it->second.lastAccess < oldest->second.lastAccess)
            {
                oldest = it;
            }
        }

        int64_t const freedBytes = oldest->second.embedding.getMemoryCapacity();
        mUsedBytes -= freedBytes;
        mEntries.erase(oldest);

        LOG_DEBUG("EncoderEmbeddingCache: evicted entry (%lld bytes freed). Used: %lld / %lld bytes.",
            static_cast<long long>(freedBytes), static_cast<long long>(mUsedBytes),
            static_cast<long long>(mBudgetBytes));
    }
}

} // namespace rt
} // namespace trt_edgellm
