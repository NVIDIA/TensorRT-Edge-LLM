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
//! 2. Memory Layout: [numAttentionLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
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
        int64_t numAttentionLayers{};        //!< Number of attention layers needing KV cache
        int64_t maxBatchSize{};              //!< Maximum batch size
        int64_t maxSequenceLength{};         //!< Maximum sequence length
        int64_t numKVHeads{};                //!< Number of key-value heads
        int64_t headDim{};                   //!< Head dimension
        nvinfer1::DataType kvCacheTypeTRT{}; //!< Storage dtype for KV cache (kHALF or kFP8)

        // Recurrent state fields (all zero for pure-attention models; no memory is allocated)
        int32_t numLinearAttnLayers{0};                                   //!< Number of recurrent layers
        int32_t recurrentStateNumHeads{0};                                //!< Number of recurrent state heads
        int32_t recurrentStateHeadDim{0};                                 //!< Dimension of each recurrent head
        int32_t recurrentStateSize{0};                                    //!< Recurrent state dimension
        nvinfer1::DataType recurrentStateType{nvinfer1::DataType::kHALF}; //!< Recurrent state dtype
        int32_t convDim{0};                                               //!< Conv1d channel dimension
        int32_t convKernel{0};                                            //!< Conv1d kernel width
        nvinfer1::DataType convStateType{nvinfer1::DataType::kHALF};      //!< Conv state dtype
    };
    //! \endcond

    //! @brief Default constructor
    LinearKVCache() noexcept = default;

    /*!
     * @brief Construct and initialize KV cache
     *
     * Allocates device memory for KV cache. Once allocated, memory won't be reallocated.
     *
     * @param config Cache configuration
     * @param stream CUDA stream for allocation
     * @throws std::runtime_error if CUDA operations fail or data type is unsupported
     */
    LinearKVCache(CacheConfig const& config, cudaStream_t stream);

    //! @brief Destructor
    ~LinearKVCache() noexcept;

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

    //! Get the combined KVCache for the given decoder layer, for EdgeLLM Attention TRT plugin implementation.
    //! @param decoderLayerIdx The index of the decoder layer.
    //! @return A non-owned tensor object with shape [batch_size, 2, num_kv_heads, max_sequence_length, head_dim] that
    //! points to the combined KVCache memory with shape information.
    rt::Tensor getCombinedKVCacheForDecoderLayer(int32_t decoderLayerIdx) noexcept;

    //! Get the separate K and V caches for the given decoder layer, for TRT native KVCacheUpdate/Attention operations.
    //! Returns a pair of tensors, the first is the K cache and the second is the V cache.
    //! @param decoderLayerIdx The index of the decoder layer.
    //! @return A pair of tensors, the first is the K cache and the second is the V cache, with shapes [batch_size,
    //! num_kv_heads, max_sequence_length, head_dim].
    std::pair<rt::Tensor, rt::Tensor> getSeparateKVCacheForDecoderLayer(int32_t decoderLayerIdx) noexcept;

    //! Get the full KVCache buffer as a non-owned tensor.
    rt::Tensor getKVCacheBuffer() noexcept;

    //! Get recurrent state tensor for a recurrent layer (non-owned view).
    //! Shape: [maxBatchSize, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
    rt::Tensor getRecurrentStateForLayer(int32_t recurrentLayerIdx) noexcept;

    //! Get conv state tensor for a recurrent layer (non-owned view).
    //! Shape: [maxBatchSize, convDim, convKernel]
    rt::Tensor getConvStateForLayer(int32_t recurrentLayerIdx) noexcept;

    //! Zero all recurrent and conv state buffers (all layers, all batch slots).
    //! Called after warmup inference and before CUDA graph capture to ensure a clean starting state.
    void clearRecurrentStates(cudaStream_t stream);

    //! Copy one batch slot's recurrent states into freshly-allocated tensors (one per recurrent layer).
    //! Used to snapshot states when saving a system prompt cache entry.
    std::vector<rt::Tensor> captureRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    //! Copy one batch slot's conv states into freshly-allocated tensors (one per recurrent layer).
    //! Used to snapshot states when saving a system prompt cache entry.
    std::vector<rt::Tensor> captureConvStates(int32_t batchIdx, cudaStream_t stream);

    //! Asynchronously reset the KVCache buffer state for a new setup of input context.
    //! @param hostReuseKVCacheLengths The lengths of the KVCache to be reused from precomputed KVCache content.
    //! @param stream The stream is used to perform GPU memory operations.
    //! @throws std::runtime_error if tensor shape, location or data type are invalid, or if a CUDA operation fails
    void resetForNewSequences(rt::Tensor const& hostReuseKVCacheLengths, cudaStream_t stream);

    //! Asynchronously commit the KVCache buffer for a prefill request, record stored KVCache lengths.
    //! @param newContextLengths [GPU, Int32]: The context length to commit for the KVCache.
    //! @param stream The stream is used to perform GPU memory operations.
    //! @throws std::runtime_error if tensor shape, location or data type are invalid
    void commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream);

    //! Commit the KVCache buffer for a decode request, increment the KVCache lengths by 1 for active sequences.
    //! @param increment The amount to increment sequence lengths (typically 1 for decode step)
    //! @param stream The stream is used to perform GPU memory operations.
    //! @throws std::runtime_error if KV cache lengths tensor has wrong location or data type
    void commitSequenceLength(int32_t increment, cudaStream_t stream);

    //! @brief Get KV cache lengths for active sequences
    //! @return Reference to KV cache lengths tensor
    rt::Tensor& getKVCacheLengths() noexcept;

    //! @brief Get KV cache configuration
    //! @return Cache configuration
    CacheConfig getConfig() const noexcept;

    //! @brief Get active batch size
    //! @return Number of active sequences
    int32_t getActiveBatchSize() const noexcept;

    //! @brief Get flag to indicate if KVCache for all sequences are empty.
    //! @return Flag to indicate if KVCache for all sequences are empty.
    bool getKVCacheAllEmpty() const noexcept;

    //! @brief Set active batch size (for batch eviction)
    //! @param newActiveBatchSize New active batch size after eviction
    //! @throws std::runtime_error If newActiveBatchSize is out of valid range [0, maxBatchSize]
    void setActiveBatchSize(int32_t newActiveBatchSize);

private:
    CacheConfig mConfig{};              //!< Cache configuration
    int32_t mActiveBatchSize{};         //!< Active batch size
    bool mKVCacheAllEmpty{};            //!< Flag to indicate if KVCache for all sequences are empty.
    rt::Tensor mDeviceKVCacheLengths{}; //!< KV cache lengths on device
    rt::Tensor mDeviceKVCache{};        //!< KV cache memory buffer on device

    //! Recurrent state buffer: [numLinearAttnLayers, maxBatchSize, recurrentStateNumHeads, recurrentStateHeadDim,
    //! recurrentStateSize] Empty when numLinearAttnLayers == 0.
    rt::Tensor mDeviceRecurrentStates{};

    //! Conv state buffer: [numLinearAttnLayers, maxBatchSize, convDim, convKernel]
    //! Empty when numLinearAttnLayers == 0.
    rt::Tensor mDeviceConvStates{};
};

} // namespace rt
} // namespace trt_edgellm
