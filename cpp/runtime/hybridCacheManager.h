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

#include "kernels/speculative/batchEvictKernels.h" // KVLayerInfo
#include "runtime/kvCacheManager.h"
#include "runtime/mambaCacheManager.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Top-level cache manager for hybrid Attention + Mamba architectures.
//!
//! Routes cache access by absolute decoder-layer index to the appropriate
//! sub-manager (`KVCacheManager` for attention layers, `MambaCacheManager`
//! for recurrent layers).  Owns the shared device KV cache lengths tensor
//! and provides unified compaction, prompt cache capture / restore, and
//! batch management APIs that span both sub-managers.
class HybridCacheManager
{
public:
    //! Type of a single decoder layer.
    enum class LayerType
    {
        kAttention,
        kMamba
    };

    //! \cond INTERNAL
    /*!
     * @brief Configuration for HybridCacheManager
     *
     * Combines per-layer type routing with sub-manager configurations.
     */
    struct Config
    {
        std::vector<LayerType> layerTypes;     //!< Absolute layer index -> type (size == total layers)
        KVCacheManager::Config kvConfig;       //!< Configuration for attention KV cache sub-manager
        MambaCacheManager::Config mambaConfig; //!< Configuration for Mamba state sub-manager
        int32_t maxBatchSize{};                //!< Maximum batch size
    };
    //! \endcond

    //! @brief Default constructor (no allocation)
    HybridCacheManager() noexcept = default;

    /*!
     * @brief Construct and initialise sub-managers and routing tables
     *
     * Builds absolute-to-local index mapping for each sub-manager, constructs
     * `KVCacheManager` and `MambaCacheManager`, and allocates the shared
     * device KV-cache-lengths tensor (zero-initialised).
     *
     * @param config Cache configuration with per-layer type routing
     * @param stream CUDA stream for allocation and memset
     */
    HybridCacheManager(Config const& config, cudaStream_t stream);

    //! @brief Destructor
    ~HybridCacheManager() noexcept;

    //! @brief Deleted copy constructor
    HybridCacheManager(HybridCacheManager const&) = delete;

    //! @brief Deleted copy assignment
    //! @return Reference to this
    HybridCacheManager& operator=(HybridCacheManager const&) = delete;

    //! @brief Move constructor
    HybridCacheManager(HybridCacheManager&&) noexcept;

    //! @brief Move assignment operator
    //! @return Reference to this
    HybridCacheManager& operator=(HybridCacheManager&&) noexcept;

    // ------------------------------------------------------------------
    // Routing by absolute layer index
    // ------------------------------------------------------------------

    //! @brief Total number of decoder layers (attention + Mamba).
    //! @return Number of layers (== size of the configured layer-type list).
    int32_t numLayers() const noexcept
    {
        return static_cast<int32_t>(mConfig.layerTypes.size());
    }

    //! @brief Type (attention or Mamba/recurrent) of a given absolute layer index.
    //! @param absLayerIdx Absolute decoder-layer index in [0, numLayers()).
    //! @return The layer's type.
    LayerType getLayerType(int32_t absLayerIdx) const;

    //! Get the combined KV cache for a given absolute layer index (must be an attention layer).
    //! @param absLayerIdx Absolute decoder-layer index.
    //! @return Reference to the per-layer NHD tensor [2, maxBatch, capPadded, numKVHeads, headDim].
    rt::Tensor& getCombinedKVCache(int32_t absLayerIdx);

    //! Get the pool-shaped view of a given attention layer's combined KV cache — the tensor to bind
    //! to the engine's past/present_key_values_i inputs; see KVCacheManager::getCombinedKVCachePoolView().
    //! @param absLayerIdx Absolute decoder-layer index.
    //! @return Reference to the pool-view tensor [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim].
    rt::Tensor& getCombinedKVCachePoolView(int32_t absLayerIdx);

    //! Get the K-half and V-half of a given attention layer's pool as separate tensor views.
    //! @param absLayerIdx Absolute decoder-layer index.
    //! @return {kView, vView}, each shaped [maxBatch, capPadded, numKVHeads, headDim].
    std::pair<rt::Tensor, rt::Tensor> getSeparateKVCache(int32_t absLayerIdx);

    //! Get the recurrent state for a given absolute layer index (must be a Mamba layer).
    //! @param absLayerIdx Absolute decoder-layer index.
    //! @return Reference to the per-layer recurrent state tensor.
    rt::Tensor& getRecurrentState(int32_t absLayerIdx);

    //! Get the conv state for a given absolute layer index (must be a Mamba layer).
    //! @param absLayerIdx Absolute decoder-layer index.
    //! @return Reference to the per-layer conv state tensor.
    rt::Tensor& getConvState(int32_t absLayerIdx);

    // ------------------------------------------------------------------
    // Sub-manager direct access
    // ------------------------------------------------------------------

    //! @brief Direct access to the KV cache sub-manager.
    //! @return Reference to the KVCacheManager.
    KVCacheManager& getKVCacheManager() noexcept;

    //! @brief Direct access to the Mamba state sub-manager.
    //! @return Reference to the MambaCacheManager.
    MambaCacheManager& getMambaCacheManager() noexcept;

    //! @brief Full immutable cache schema used to size and identify exact recurrent snapshots.
    Config const& getConfig() const noexcept;

    //! @brief Minimal read-only view of one pre-computed KV head-dim group.
    //!
    //! Exposes just what callers (e.g. SpecDecode base verification) need to launch a
    //! batched per-layer kernel: a device-resident `KVLayerInfo` array plus
    //! the dispatch parameters. Internal bookkeeping (`hostInfos`,
    //! `deviceScratchInfos`, etc.) stays private.
    struct KVHeadDimGroupView
    {
        kernel::KVLayerInfo const* deviceLayerInfos; //!< Device pointer to layer-info array (size == numLayers)
        int32_t numLayers;                           //!< Number of KV layers in this group
        int32_t headDim;                             //!< Head dimension shared by all layers in this group
        int32_t maxKVHeads;                          //!< Maximum numKVHeads across layers in this group
    };

    //! @brief Read-only views of the pre-computed KV head-dim groups.
    //!
    //! Uniform models return a single group; hybrid Gemma4-style models
    //! return one group per distinct head dim. The underlying
    //! `KVLayerInfo` arrays are owned by this manager and remain valid
    //! for its lifetime.
    //! @return Vector of group views (one per distinct head dim).
    std::vector<KVHeadDimGroupView> getKVHeadDimGroups() const;

    // ------------------------------------------------------------------
    // Shared state
    // ------------------------------------------------------------------

    //! @brief Get the shared device KV cache lengths tensor.
    //! @return Reference to the device tensor of shape [activeBatchSize].
    rt::Tensor& getKVCacheLengths() noexcept;

    //! Reset state for new sequences. Validates batch size, copies reuse lengths
    //! from host to device, and updates the "all empty" flag.
    //! @param reuseKVCacheLengths Host INT32 tensor with reuse lengths, shape [batchSize].
    //! @param stream CUDA stream.
    void resetForNewSequences(rt::Tensor const& reuseKVCacheLengths, cudaStream_t stream);

    //! Commit sequence lengths after prefill (element-wise increment from a GPU tensor).
    //! @param newContextLengths GPU INT32 tensor of context lengths, shape [activeBatchSize].
    //! @param stream CUDA stream.
    void commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream);

    //! Commit sequence lengths after decode (scalar increment for all active sequences).
    //! @param increment Scalar increment (typically 1).
    //! @param stream CUDA stream.
    void commitSequenceLength(int32_t increment, cudaStream_t stream);

    // ------------------------------------------------------------------
    // Batch management
    // ------------------------------------------------------------------

    //! @brief Get the number of active sequences.
    //! @return Active batch size.
    int32_t getActiveBatchSize() const noexcept;

    //! @brief Set active batch size (used after batch eviction).
    //! @param newActiveBatchSize New active batch size.
    //! @throws std::runtime_error if out of range [0, maxBatchSize].
    void setActiveBatchSize(int32_t newActiveBatchSize);

    //! @brief Check if KV cache for all sequences is empty.
    //! @return True if no prefill has been committed yet.
    bool getKVCacheAllEmpty() const noexcept;

    // ------------------------------------------------------------------
    // Compaction
    // ------------------------------------------------------------------

    //! Compact both KV caches and Mamba states after batch eviction.
    //! @param batchMapping GPU tensor [oldBatch], mapping[i] = newBatchIdx or -1 (evicted).
    //! @param oldBatch Batch size before eviction.
    //! @param newBatch Batch size after eviction.
    //! @param stream CUDA stream.
    void compactBatch(rt::Tensor const& batchMapping, int32_t oldBatch, int32_t newBatch, cudaStream_t stream);

    //! Compact only slot-addressed lengths and recurrent/conv state. Global paged KV remains in place and callers
    //! compact the corresponding KVPageTable rows separately.
    void compactBatchSlotState(rt::Tensor const& batchMapping, int32_t oldBatch, int32_t newBatch, cudaStream_t stream);

    // ------------------------------------------------------------------
    // System prompt cache
    // ------------------------------------------------------------------

    //! Capture KV cache for a single batch slot across all attention layers.
    //! @param batchIdx Batch slot to capture.
    //! @param sequenceLength Number of tokens to capture from the cache.
    //! @param stream CUDA stream.
    //! @return Vector of captured tensors (one per attention layer).
    std::vector<rt::Tensor> captureKVCache(int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream);

    //! Restore KV cache for a single batch slot across all attention layers.
    //! @param saved Previously captured KV cache tensors.
    //! @param batchIdx Target batch slot.
    //! @param stream CUDA stream.
    void restoreKVCache(std::vector<rt::Tensor> const& saved, int32_t batchIdx, cudaStream_t stream);

    //! Capture recurrent states for a single batch slot (delegates to MambaCacheManager).
    //! @param batchIdx Batch slot to capture.
    //! @param stream CUDA stream.
    //! @return Vector of captured tensors (one per recurrent layer).
    std::vector<rt::Tensor> captureRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    //! Capture conv states for a single batch slot (delegates to MambaCacheManager).
    //! @param batchIdx Batch slot to capture.
    //! @param stream CUDA stream.
    //! @return Vector of captured tensors (one per recurrent layer).
    std::vector<rt::Tensor> captureConvStates(int32_t batchIdx, cudaStream_t stream);

private:
    //! Pre-computed group of KV layers sharing the same headDim for batched kernel launches.
    struct HeadDimGroup
    {
        int32_t headDim;                            //!< Head dimension shared by all layers in this group
        int32_t maxKVHeads;                         //!< Maximum numKVHeads across layers in this group
        int32_t numLayers;                          //!< Number of layers in this group
        std::vector<int32_t> localKVIndices;        //!< Local KV-layer indices belonging to this group
        rt::Tensor deviceLayerInfos;                //!< Device buffer of KVLayerInfo for this group
        rt::Tensor deviceScratchInfos;              //!< Pre-allocated device scratch buffer for save/restore uploads
        std::vector<kernel::KVLayerInfo> hostInfos; //!< Host copy for building save/restore info arrays
    };

    Config mConfig{};                         //!< Full configuration
    KVCacheManager mKVCache;                  //!< Sub-manager for attention KV caches
    MambaCacheManager mMambaCache;            //!< Sub-manager for Mamba recurrent / conv states
    std::vector<int32_t> mAbsToKVIndex;       //!< Absolute layer -> local KV index (-1 if not attention)
    std::vector<int32_t> mAbsToMambaIndex;    //!< Absolute layer -> local Mamba index (-1 if not Mamba)
    rt::Tensor mDeviceKVCacheLengths{};       //!< Shared KV cache lengths on device [activeBatchSize]
    int32_t mActiveBatchSize{};               //!< Number of active sequences
    bool mKVCacheAllEmpty{true};              //!< True until the first commitSequenceLength call
    std::vector<HeadDimGroup> mHeadDimGroups; //!< Pre-computed per-headDim groups for batched kernels
};

} // namespace rt
} // namespace trt_edgellm
