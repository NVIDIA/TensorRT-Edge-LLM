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

#pragma once

#include "common/tensor.h"
#include "runtime/state/contextCache/blockHash.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

//! One cached encoder output embedding keyed by media content hash.
struct EncoderEmbeddingCacheEntry
{
    Tensor embedding;    //!< [numTokens, hiddenSize], fp16 or bf16, GPU
    int64_t numTokens{}; //!< Actual token count (embedding shape[0])
    int64_t hiddenSize{};
    std::chrono::steady_clock::time_point lastAccess;
};

//! Content-addressed GPU cache for encoder (ViT / audio) output embeddings.
//!
//! Saves encoded embeddings keyed by a 128-bit hash of the raw media bytes, allowing
//! subsequent requests with identical media to skip the expensive encoder `infer()` call.
//! Eviction is LRU when the GPU memory budget is exceeded.
//!
//! Thread safety: inherits the single-writer contract from ContextCacheManager. No internal mutex.
class EncoderEmbeddingCache
{
public:
    //! Construct with a GPU memory budget in bytes. A budget of 0 disables the cache.
    explicit EncoderEmbeddingCache(int64_t maxBudgetBytes);

    //! Look up a cached embedding by content hash.
    //! Returns a const reference to the cached tensor on hit, std::nullopt on miss.
    //! Updates last-access time on hit.
    std::optional<std::reference_wrapper<Tensor const>> lookup(Hash128 key);

    //! Store an embedding under the given content hash by copying from the source tensor.
    //! Evicts LRU entries if the budget would be exceeded. The copy is enqueued on `stream`.
    void store(Hash128 key, Tensor const& embedding, int64_t numTokens, int64_t hiddenSize, cudaStream_t stream);

    //! Store a slice of an encoder output as a per-media-item cache entry.
    //! @param key Content hash of the individual media item
    //! @param devicePtr Pointer into the encoder output buffer at the item's byte offset
    //! @param numTokens Number of encoder output tokens for this item
    //! @param hiddenSize Hidden dimension (columns) of the embedding
    //! @param dtype Data type of the embedding elements
    //! @param stream CUDA stream for the device-to-device copy
    void storeSlice(Hash128 key, void const* devicePtr, int64_t numTokens, int64_t hiddenSize, nvinfer1::DataType dtype,
        cudaStream_t stream);

    //! Remove all entries and free GPU memory.
    void clear();

    //! Current GPU bytes used by cached embeddings.
    int64_t usedBytes() const noexcept
    {
        return mUsedBytes;
    }

    //! Number of cached entries.
    size_t size() const noexcept
    {
        return mEntries.size();
    }

private:
    void evictUntilFits(int64_t requiredBytes);

    int64_t mBudgetBytes;
    int64_t mUsedBytes{};
    std::unordered_map<Hash128, EncoderEmbeddingCacheEntry> mEntries;
};

} // namespace rt
} // namespace trt_edgellm
