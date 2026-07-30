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

#include "common/pagedKvTypes.h"
#include <common/tensor.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-layer KV head configuration for heterogeneous models.
struct KVLayerConfig
{
    int32_t numKVHeads{}; //!< Number of key-value heads for this layer
    int32_t headDim{};    //!< Head dimension for this layer
};

//! Per-layer KV cache manager that supports heterogeneous head configurations across layers.
//! Each attention layer gets its own independently-sized tensor in slot-shaped NHD layout with
//! shape [2, maxBatchSize, capPadded, numKVHeads_i, headDim_i], where the K/V split (the leading
//! `2`) is OUTERMOST and capPadded = ceil(maxSequenceLength / kTOKENS_PER_PAGE) * kTOKENS_PER_PAGE.
//! This same buffer is a bind-time reinterpretation of a page pool [2, numPages, kTOKENS_PER_PAGE,
//! numKVHeads_i, headDim_i]; see kPoolPtr()/vPoolPtr().
//! This replaces the monolithic LinearKVCache allocation when layers have different numKVHeads or headDim.
class KVCacheManager
{
public:
    //! \cond INTERNAL
    /*!
     * @brief Configuration for per-layer KV cache manager
     *
     * Defines the dimensions and capacity of the KV cache for each attention layer.
     */
    struct Config
    {
        int32_t numAttentionLayers{};            //!< Number of attention layers needing KV cache
        int32_t maxBatchSize{};                  //!< Maximum batch size
        int32_t maxSequenceLength{};             //!< Maximum sequence length
        std::vector<KVLayerConfig> layerConfigs; //!< Per-layer head config (size == numAttentionLayers)
        nvinfer1::DataType kvCacheType{};        //!< Storage dtype for KV cache (kHALF or kFP8)
        //! Optional override of the pool's total page count. 0 (default) means "use the
        //! active-capacity floor" (`maxBatchSize * ceil(maxSequenceLength / kTOKENS_PER_PAGE)`,
        //! today's fixed behavior). A non-zero value must be >= that floor; the extra pages are
        //! allocated as unused retention headroom (see numPages()/getCombinedKVCache()).
        int32_t numPages{0};
    };
    //! \endcond

    //! @brief Default constructor
    KVCacheManager() noexcept = default;

    /*!
     * @brief Construct and initialize per-layer KV cache
     *
     * Allocates one device tensor per attention layer. Once allocated, memory won't be reallocated.
     * Determines whether all layers share the same numKVHeads and headDim (uniform mode).
     *
     * @param config Cache configuration with per-layer configs
     * @param stream CUDA stream for allocation
     * @throws std::runtime_error if config is invalid or data type is unsupported
     */
    KVCacheManager(Config const& config, cudaStream_t stream);

    //! @brief Destructor
    ~KVCacheManager() noexcept;

    //! @brief Deleted copy constructor to avoid large data copy
    KVCacheManager(KVCacheManager const&) = delete;

    //! @brief Deleted copy assignment to avoid large data copy
    //! @return Reference to this
    KVCacheManager& operator=(KVCacheManager const&) = delete;

    //! @brief Move constructor
    KVCacheManager(KVCacheManager&&) noexcept;

    //! @brief Move assignment operator
    //! @return Reference to this
    KVCacheManager& operator=(KVCacheManager&&) noexcept;

    //! Get the combined KVCache for the given attention layer.
    //! @note This slot-shaped view is only a valid reinterpretation of the pool when
    //!       numPages() == the active-capacity floor (i.e. Config::numPages was left at its
    //!       default 0, or explicitly set to the floor). When retention headroom is configured
    //!       (numPages() > floor), the K-half and V-half are no longer back-to-back at this
    //!       tensor's declared stride; use getSeparateKVCache()/kPoolPtr()/vPoolPtr() instead,
    //!       which are correct for any numPages().
    //! @param attnLayerIdx The index of the attention layer.
    //! @return A reference to the tensor with shape [2, maxBatchSize, capPadded, numKVHeads_i, headDim_i].
    rt::Tensor& getCombinedKVCache(int32_t attnLayerIdx) noexcept;

    //! Get the pool-shaped view of the given attention layer's combined KVCache — same device
    //! allocation as getCombinedKVCache(), reinterpreted as the AttentionPlugin's paged-pool
    //! binding contract [2, numPages(), kTOKENS_PER_PAGE, numKVHeads_i, headDim_i]. This is the
    //! tensor that must be bound to the engine's past/present_key_values_i inputs so
    //! `dims.d[1] == numPages` (see AttentionPlugin::enqueue).
    //! @param attnLayerIdx The index of the attention layer.
    //! @return A reference to the pool-view tensor.
    rt::Tensor& getCombinedKVCachePoolView(int32_t attnLayerIdx) noexcept;

    //! Get the K-half and V-half of the given attention layer's pool as separate tensor views.
    //! @param attnLayerIdx The index of the attention layer.
    //! @return {kView, vView}, each shaped [maxBatchSize, capPadded, numKVHeads_i, headDim_i].
    std::pair<rt::Tensor, rt::Tensor> getSeparateKVCache(int32_t attnLayerIdx) const noexcept;

    //! @brief Get the padded per-slot token capacity.
    //! @return capPadded = ceil(maxSequenceLength / kTOKENS_PER_PAGE) * kTOKENS_PER_PAGE.
    int32_t maxCapPadded() const noexcept;

    //! @brief Get the total number of pages spanned by the pool.
    //! @return Config::numPages if non-zero, else the active-capacity floor
    //!         (maxBatchSize * maxCapPadded() / kTOKENS_PER_PAGE).
    int32_t numPages() const noexcept;

    //! Get the K-half page-pool base pointer for the given attention layer, i.e. the base of the
    //! combined tensor reinterpreted as [numPages(), kTOKENS_PER_PAGE, numKVHeads_i, headDim_i].
    //! @param attnLayerIdx The index of the attention layer.
    //! @return Device pointer to the K pool.
    void* kPoolPtr(int32_t attnLayerIdx) const noexcept;

    //! Get the V-half page-pool base pointer for the given attention layer, i.e. kPoolPtr(attnLayerIdx)
    //! offset by numPages() * kTOKENS_PER_PAGE * numKVHeads_i * headDim_i elements.
    //! @param attnLayerIdx The index of the attention layer.
    //! @return Device pointer to the V pool.
    void* vPoolPtr(int32_t attnLayerIdx) const noexcept;

    //! Get the layer configuration for the given attention layer.
    //! @param attnLayerIdx The index of the attention layer.
    //! @return The KVLayerConfig for this layer.
    KVLayerConfig const& getLayerConfig(int32_t attnLayerIdx) const noexcept;

    //! @brief Get the number of attention layers
    //! @return Number of attention layers
    int32_t numLayers() const noexcept;

    //! @brief Check if all layers have the same numKVHeads and headDim
    //! @return True if all layers are uniform
    bool isUniform() const noexcept;

    //! @brief Get cache configuration
    //! @return Cache configuration
    Config const& getConfig() const noexcept;

private:
    Config mConfig{};                     //!< Cache configuration
    std::vector<rt::Tensor> mLayerCaches; //!< Per-layer KV cache tensors on device
    //! Per-layer pool-shaped views aliasing mLayerCaches' device memory (same pointer, different
    //! declared Coords); see getCombinedKVCachePoolView().
    std::vector<rt::Tensor> mLayerCachesPoolView;
    bool mIsUniform{true}; //!< True if all layers share the same numKVHeads and headDim
    int32_t mCapPadded{};  //!< maxSequenceLength padded up to a multiple of kTOKENS_PER_PAGE
    int32_t mNumPages{};   //!< Resolved total page count (Config::numPages, or the floor if 0)
};

} // namespace rt
} // namespace trt_edgellm
