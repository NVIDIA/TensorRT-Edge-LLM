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

bool EncoderEmbeddingCacheEntry::canRestore(Tensor const& outputEmbedding,
    std::vector<std::reference_wrapper<Tensor>> const& outputFeatures, int64_t tokenOffset) const
{
    if (outputFeatures.size() != features.size() || outputEmbedding.getShape().getNumDims() != 2)
    {
        return false;
    }
    int64_t const totalTokens = outputEmbedding.getShape()[0];
    if (tokenOffset < 0 || tokenOffset > totalTokens || numTokens > totalTokens - tokenOffset)
    {
        return false;
    }
    for (size_t i = 0; i <= outputFeatures.size(); ++i)
    {
        Tensor const& output = i == 0 ? outputEmbedding : outputFeatures[i - 1].get();
        Tensor const& cached = i == 0 ? embedding : features[i - 1];
        int64_t const rowBytes = cached.getShape()[1] * static_cast<int64_t>(utils::getTypeSize(cached.getDataType()));
        if (output.getDeviceType() != DeviceType::kGPU || output.getDataType() != cached.getDataType()
            || output.getShape().getNumDims() != 2 || output.getShape()[1] != cached.getShape()[1]
            || totalTokens > output.getMemoryCapacity() / rowBytes)
        {
            return false;
        }
    }
    return true;
}

void EncoderEmbeddingCacheEntry::restore(Tensor& outputEmbedding,
    std::vector<std::reference_wrapper<Tensor>> const& outputFeatures, int64_t tokenOffset, cudaStream_t stream) const
{
    ELLM_CHECK(
        canRestore(outputEmbedding, outputFeatures, tokenOffset), "Cached encoder outputs do not match the runner");
    int64_t const totalTokens = outputEmbedding.getShape()[0];
    for (auto const& ref : outputFeatures)
    {
        Tensor& output = ref.get();
        bool const reshaped = output.reshape({totalTokens, output.getShape()[1]});
        ELLM_CHECK(reshaped, "Failed to reshape encoder output");
    }
    for (size_t i = 0; i <= outputFeatures.size(); ++i)
    {
        Tensor& output = i == 0 ? outputEmbedding : outputFeatures[i - 1].get();
        Tensor const& cached = i == 0 ? embedding : features[i - 1];
        int64_t const rowBytes = cached.getShape()[1] * static_cast<int64_t>(utils::getTypeSize(cached.getDataType()));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output.rawPointer()) + tokenOffset * rowBytes,
            cached.rawPointer(), numTokens * rowBytes, cudaMemcpyDeviceToDevice, stream));
    }
}

EncoderEmbeddingCache::EncoderEmbeddingCache(int64_t maxBudgetBytes)
    : mBudgetBytes(maxBudgetBytes)
{
}

std::optional<std::reference_wrapper<Tensor const>> EncoderEmbeddingCache::lookup(Hash128 key)
{
    auto entry = lookupEntry(key);
    return entry ? std::optional{std::cref(entry->get().embedding)} : std::nullopt;
}

std::optional<std::reference_wrapper<EncoderEmbeddingCacheEntry const>> EncoderEmbeddingCache::lookupEntry(Hash128 key)
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
    return std::cref(it->second);
}

bool EncoderEmbeddingCache::tryRestore(std::vector<Hash128> const& keys, std::vector<int64_t> const& tokenLengths,
    Tensor& outputEmbedding, std::vector<std::reference_wrapper<Tensor>> const& outputFeatures, cudaStream_t stream)
{
    if (keys.empty() || keys.size() != tokenLengths.size())
    {
        return false;
    }
    std::vector<std::reference_wrapper<EncoderEmbeddingCacheEntry const>> entries;
    int64_t tokenOffset = 0;
    for (size_t i = 0; i < keys.size(); ++i)
    {
        auto entry = lookupEntry(keys[i]);
        if (!entry)
        {
            return false;
        }
        if (entry->get().numTokens != tokenLengths[i]
            || !entry->get().canRestore(outputEmbedding, outputFeatures, tokenOffset))
        {
            erase(keys[i]);
            return false;
        }
        entries.emplace_back(*entry);
        tokenOffset += tokenLengths[i];
    }
    if (tokenOffset != outputEmbedding.getShape()[0])
    {
        return false;
    }
    tokenOffset = 0;
    for (auto const& entry : entries)
    {
        entry.get().restore(outputEmbedding, outputFeatures, tokenOffset, stream);
        tokenOffset += entry.get().numTokens;
    }
    return true;
}

void EncoderEmbeddingCache::store(
    Hash128 key, Tensor const& embedding, int64_t numTokens, int64_t hiddenSize, cudaStream_t stream)
{
    storeSlice(key, embedding.rawPointer(), numTokens, hiddenSize, embedding.getDataType(), stream);
}

void EncoderEmbeddingCache::storeSlice(Hash128 key, void const* devicePtr, int64_t numTokens, int64_t hiddenSize,
    nvinfer1::DataType dtype, cudaStream_t stream, std::vector<std::reference_wrapper<Tensor>> const& features,
    int64_t tokenOffset)
{
    if (mBudgetBytes <= 0 || numTokens == 0)
    {
        return;
    }

    auto it = mEntries.find(key);
    if (it != mEntries.end())
    {
        it->second.lastAccess = std::chrono::steady_clock::now();
        return;
    }

    ELLM_CHECK(
        devicePtr != nullptr && numTokens > 0 && hiddenSize > 0 && tokenOffset >= 0, "Invalid encoder cache slice");
    int64_t const embeddingBytes = numTokens * hiddenSize * static_cast<int64_t>(utils::getTypeSize(dtype));
    int64_t entryBytes = embeddingBytes;
    for (auto const& ref : features)
    {
        Tensor const& feature = ref.get();
        ELLM_CHECK(feature.getDeviceType() == DeviceType::kGPU && feature.getShape().getNumDims() == 2
                && feature.getShape()[1] > 0 && tokenOffset <= feature.getShape()[0]
                && numTokens <= feature.getShape()[0] - tokenOffset,
            "Invalid encoder cache feature slice");
        entryBytes
            += numTokens * feature.getShape()[1] * static_cast<int64_t>(utils::getTypeSize(feature.getDataType()));
    }
    if (entryBytes > mBudgetBytes)
    {
        return;
    }

    evictUntilFits(entryBytes);

    Tensor cached({numTokens, hiddenSize}, DeviceType::kGPU, dtype, "encoderEmbeddingCacheSlice");
    CUDA_CHECK(cudaMemcpyAsync(cached.rawPointer(), devicePtr, embeddingBytes, cudaMemcpyDeviceToDevice, stream));

    EncoderEmbeddingCacheEntry entry{};
    entry.embedding = std::move(cached);
    for (auto const& ref : features)
    {
        Tensor const& feature = ref.get();
        int64_t const rowBytes
            = feature.getShape()[1] * static_cast<int64_t>(utils::getTypeSize(feature.getDataType()));
        entry.features.emplace_back(Coords{numTokens, feature.getShape()[1]}, DeviceType::kGPU, feature.getDataType(),
            "encoderFeatureCacheSlice");
        CUDA_CHECK(cudaMemcpyAsync(entry.features.back().rawPointer(),
            static_cast<char const*>(feature.rawPointer()) + tokenOffset * rowBytes, numTokens * rowBytes,
            cudaMemcpyDeviceToDevice, stream));
    }
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

void EncoderEmbeddingCache::erase(Hash128 key)
{
    auto it = mEntries.find(key);
    if (it == mEntries.end())
    {
        return;
    }
    mUsedBytes -= it->second.embedding.getMemoryCapacity();
    for (auto const& feature : it->second.features)
    {
        mUsedBytes -= feature.getMemoryCapacity();
    }
    mEntries.erase(it);
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

        int64_t freedBytes = oldest->second.embedding.getMemoryCapacity();
        for (auto const& feature : oldest->second.features)
        {
            freedBytes += feature.getMemoryCapacity();
        }
        mUsedBytes -= freedBytes;
        mEntries.erase(oldest);

        LOG_DEBUG("EncoderEmbeddingCache: evicted entry (%lld bytes freed). Used: %lld / %lld bytes.",
            static_cast<long long>(freedBytes), static_cast<long long>(mUsedBytes),
            static_cast<long long>(mBudgetBytes));
    }
}

} // namespace rt
} // namespace trt_edgellm
