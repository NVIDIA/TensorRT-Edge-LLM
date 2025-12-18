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

#pragma once

#include <common/tensor.h>
#include <cstdint>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace rt
{

//! Static Linear KVCache that holds the KVCache for all decoder layers up to maxSequenceLength.
//! The KVCache implement the design of:
//! 1. Allocates memory for max supported batch size.
//! 2. Memory Layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
//! 3. Synchronous execution of batch requests, all the sequences in the batch will run prefill
//!    or decode at the same time.
class LinearKVCache
{
public:
    //! \cond INTERNAL
    /*!
     * @brief Configuration for KV cache
     *
     * Defines the dimensions and capacity of the KV cache.
     */
    struct CacheConfig
    {
        int64_t numDecoderLayers{};  //!< Number of decoder layers
        int64_t maxBatchSize{};      //!< Maximum batch size
        int64_t maxSequenceLength{}; //!< Maximum sequence length
        int64_t numKVHeads{};        //!< Number of key-value heads
        int64_t headDim{};           //!< Head dimension
    };
    //! \endcond

    using KVCacheType = half; //!< KV cache data type (half precision)
    static constexpr nvinfer1::DataType KVCacheTypeTRT{nvinfer1::DataType::kHALF}; //!< TensorRT data type

    //! @brief Default constructor
    LinearKVCache() = default;

    /*!
     * @brief Construct and initialize KV cache
     *
     * Allocates device memory for KV cache. Once allocated, memory won't be reallocated.
     *
     * @param config Cache configuration
     * @param stream CUDA stream for allocation
     */
    LinearKVCache(CacheConfig const& config, cudaStream_t stream);

    //! @brief Destructor
    ~LinearKVCache();

    //! @brief Deleted copy constructor to avoid large data copy
    LinearKVCache(LinearKVCache const&) = delete;

    //! @brief Deleted copy assignment to avoid large data copy
    //! @return Reference to this
    LinearKVCache& operator=(LinearKVCache const&) = delete;

    //! @brief Move constructor
    LinearKVCache(LinearKVCache&&) noexcept;

    //! @brief Move assignment operator
    //! @return Reference to this
    LinearKVCache& operator=(LinearKVCache&&) noexcept;

    //! Get the KVCache for the given decoder layer.
    //! @param decoderLayerIdx The index of the decoder layer.
    //! @return A non-owned tensor object that points to the KVCache memory with shape information.
    rt::Tensor getKVCacheForDecoderLayer(int32_t decoderLayerIdx);

    //! Get the full KVCache buffer as a non-owned tensor.
    rt::Tensor getKVCacheBuffer();

    //! Asynchronously reset the KVCache buffer state for a new setup of input context.
    //! @param hostReuseKVCacheLengths The lengths of the KVCache to be reused from precomputed KVCache content.
    //! @param stream The stream is used to perform GPU memory operations.
    void resetForNewSequences(rt::Tensor const& hostReuseKVCacheLengths, cudaStream_t stream);

    //! Asynchronously commit the KVCache buffer for a prefill request, record stored KVCache lengths.
    //! @param newContextLengths [GPU, Int32]: The context length to commit for the KVCache.
    //! @param stream The stream is used to perform GPU memory operations.
    void commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream);

    //! Commit the KVCache buffer for a decode request, increment the KVCache lengths by 1 for active sequences.
    //! @param increment The amount to increment sequence lengths (typically 1 for decode step)
    //! @param stream The stream is used to perform GPU memory operations.
    void commitSequenceLength(int32_t increment, cudaStream_t stream);

    //! @brief Get KV cache lengths for active sequences
    //! @return Reference to KV cache lengths tensor
    rt::Tensor& getKVCacheLengths();

    //! @brief Get KV cache configuration
    //! @return Cache configuration
    CacheConfig getConfig() const;

    //! @brief Get active batch size
    //! @return Number of active sequences
    int32_t getActiveBatchSize() const;

    //! @brief Get flag to indicate if KVCache for all sequences are empty.
    //! @return Flag to indicate if KVCache for all sequences are empty.
    bool getKVCacheAllEmpty() const;

    //! @brief Set active batch size (for batch eviction)
    //! @param newActiveBatchSize New active batch size after eviction
    void setActiveBatchSize(int32_t newActiveBatchSize);

private:
    CacheConfig mConfig{};                //!< Cache configuration
    int32_t mActiveBatchSize{};           //!< Active batch size
    bool mKVCacheAllEmpty{};              //!< Flag to indicate if KVCache for all sequences are empty.
    rt::Tensor mDeviceKVCacheLengths{};   //!< KV cache lengths on device
    KVCacheType* mDeviceKVCache{nullptr}; //!< KV cache memory buffer
};

} // namespace rt
} // namespace trt_edgellm