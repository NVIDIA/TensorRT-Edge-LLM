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
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Complete per-media encoder outputs keyed by media content hash.
struct EncoderEmbeddingCacheEntry
{
    Tensor embedding;             //!< [numTokens, hiddenSize], fp16 or bf16, GPU
    std::vector<Tensor> features; //!< Additional per-token encoder outputs, including deepstack
    int64_t numTokens{};          //!< Actual token count (embedding shape[0])
    int64_t hiddenSize{};
    std::chrono::steady_clock::time_point lastAccess;

    //! Check destination metadata and capacity without changing any output.
    bool canRestore(Tensor const& outputEmbedding, std::vector<std::reference_wrapper<Tensor>> const& outputFeatures,
        int64_t tokenOffset) const;

    //! Restore this media item's outputs into a preprocessed request's encoder buffers.
    //! @param outputEmbedding Main encoder embedding buffer
    //! @param outputFeatures Auxiliary features in encoder order
    //! @param tokenOffset First destination row for this media item
    //! @param stream CUDA stream used to store and restore cache entries
    void restore(Tensor& outputEmbedding, std::vector<std::reference_wrapper<Tensor>> const& outputFeatures,
        int64_t tokenOffset, cudaStream_t stream) const;
};

//! Content-addressed GPU cache for encoder (ViT / audio) output embeddings.
//!
//! Saves encoded embeddings keyed by a 128-bit hash of the raw media bytes, allowing
//! subsequent requests with identical media to skip the expensive encoder `infer()` call.
//! Eviction is LRU when the GPU memory budget is exceeded.
//!
//! Thread safety: single writer on one CUDA stream. Stores and restores must use that stream.
class EncoderEmbeddingCache
{
public:
    //! Construct with a GPU memory budget in bytes. A budget of 0 disables the cache.
    explicit EncoderEmbeddingCache(int64_t maxBudgetBytes);

    //! Look up a cached embedding by content hash.
    //! Returns a const reference to the cached tensor on hit, std::nullopt on miss.
    //! Updates last-access time on hit.
    std::optional<std::reference_wrapper<Tensor const>> lookup(Hash128 key);

    //! Look up the complete encoder state. References remain valid until eviction or clear().
    //! Updates last-access time on hit.
    std::optional<std::reference_wrapper<EncoderEmbeddingCacheEntry const>> lookupEntry(Hash128 key);

    //! Restore a complete batch only if every entry matches the preprocessed media layout.
    //! Incompatible entries are invalidated so the caller can re-encode and cache their replacements.
    bool tryRestore(std::vector<Hash128> const& keys, std::vector<int64_t> const& tokenLengths, Tensor& outputEmbedding,
        std::vector<std::reference_wrapper<Tensor>> const& outputFeatures, cudaStream_t stream);

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
    //! @param features Additional [tokens, hidden] outputs, cached and evicted together with the embedding
    //! @param tokenOffset First source row for this media item in each additional output
    void storeSlice(Hash128 key, void const* devicePtr, int64_t numTokens, int64_t hiddenSize, nvinfer1::DataType dtype,
        cudaStream_t stream, std::vector<std::reference_wrapper<Tensor>> const& features = {}, int64_t tokenOffset = 0);

    //! Remove all entries and free GPU memory.
    void clear();

    //! Current GPU bytes used by cached embeddings and auxiliary features.
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
    void erase(Hash128 key);
    void evictUntilFits(int64_t requiredBytes);

    int64_t mBudgetBytes;
    int64_t mUsedBytes{};
    std::unordered_map<Hash128, EncoderEmbeddingCacheEntry> mEntries;
};

} // namespace rt
} // namespace trt_edgellm
