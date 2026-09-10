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
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-layer Mamba (recurrent + conv) state manager.
//! Each recurrent layer owns two device tensors:
//!   - recurrent state: [maxBatchSize, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
//!   - conv state:      [maxBatchSize, convDim, convKernel]
//! When numRecurrentLayers == 0 the manager is a no-op and allocates nothing.
class MambaCacheManager
{
public:
    //! \cond INTERNAL
    /*!
     * @brief Configuration for Mamba state cache
     *
     * Defines the dimensions and capacity of recurrent and conv state buffers.
     */
    struct Config
    {
        int32_t numRecurrentLayers{0};     //!< Number of recurrent layers (0 = no-op)
        int32_t maxBatchSize{};            //!< Maximum batch size
        int32_t recurrentStateNumHeads{0}; //!< Number of recurrent state heads
        int32_t recurrentStateHeadDim{0};  //!< Dimension of each recurrent head
        int32_t recurrentStateSize{0};     //!< Recurrent state dimension
        int32_t convDim{0};                //!< Conv1d channel dimension
        int32_t convKernel{0};             //!< Conv1d kernel width
        int32_t maxIntermediateSeqLen{0};  //!< Spec-verify intermediate state seq dim (0 = disabled)
        nvinfer1::DataType recurrentStateType{nvinfer1::DataType::kHALF}; //!< Recurrent state dtype
        nvinfer1::DataType convStateType{nvinfer1::DataType::kHALF};      //!< Conv state dtype
        //! Number of B/C groups. Only consumed by the Mamba spec-verify replay stash (the replay-B
        //! extent); GDN/DDTree leaves it 0.
        int32_t recurrentStateNumGroups{0};
        //! MTP spec-verify commit mode: true → Mamba replay reconstruction (allocate the dA/u/B
        //! replay stash), false → GDN/DDTree per-token snapshot scatter.
        bool specVerifyUsesReplay{false};
    };
    //! \endcond

    //! @brief Default constructor (no allocation)
    MambaCacheManager() noexcept = default;

    /*!
     * @brief Construct and initialize per-layer Mamba state buffers
     *
     * If numRecurrentLayers == 0 the constructor returns immediately without allocating.
     * Otherwise it allocates one recurrent-state tensor and one conv-state tensor per layer,
     * zero-initialises them, and logs total GPU memory consumed.
     *
     * @param config Cache configuration
     * @param stream CUDA stream for allocation and memset
     * @throws std::runtime_error if config validation fails
     */
    MambaCacheManager(Config const& config, cudaStream_t stream);

    //! @brief Destructor
    ~MambaCacheManager() noexcept;

    //! @brief Deleted copy constructor to avoid large data copy
    MambaCacheManager(MambaCacheManager const&) = delete;

    //! @brief Deleted copy assignment to avoid large data copy
    //! @return Reference to this
    MambaCacheManager& operator=(MambaCacheManager const&) = delete;

    //! @brief Move constructor
    MambaCacheManager(MambaCacheManager&&) noexcept;

    //! @brief Move assignment operator
    //! @return Reference to this
    MambaCacheManager& operator=(MambaCacheManager&&) noexcept;

    //! Get the recurrent state tensor for a given recurrent layer (owned tensor reference).
    //! Shape: [maxBatchSize, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getRecurrentState(int32_t recurrentLayerIdx) noexcept;

    //! Get the conv state tensor for a given recurrent layer (owned tensor reference).
    //! Shape: [maxBatchSize, convDim, convKernel]
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getConvState(int32_t recurrentLayerIdx) noexcept;

    //! Zero all recurrent and conv state buffers (all layers, all batch slots).
    //! Called after warmup inference and before CUDA graph capture to ensure a clean starting state.
    //! No-op when numRecurrentLayers == 0.
    //! @param stream CUDA stream for memset operations.
    void clearStates(cudaStream_t stream);

    //! Copy one batch slot's recurrent states into freshly-allocated tensors (one per layer).
    //! Used to snapshot states when saving a system prompt cache entry.
    //! Returns an empty vector when numRecurrentLayers == 0.
    //! @param batchIdx The batch slot index to capture.
    //! @param stream CUDA stream for copy operations.
    //! @return Vector of device tensors with shape [1, numHeads, headDim, stateSize].
    std::vector<rt::Tensor> captureRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    //! Copy one batch slot's conv states into freshly-allocated tensors (one per layer).
    //! Used to snapshot states when saving a system prompt cache entry.
    //! Returns an empty vector when numRecurrentLayers == 0.
    //! @param batchIdx The batch slot index to capture.
    //! @param stream CUDA stream for copy operations.
    //! @return Vector of device tensors with shape [1, convDim, convKernel].
    std::vector<rt::Tensor> captureConvStates(int32_t batchIdx, cudaStream_t stream);

    //! Get a spec-verify replay-stash tensor for a given layer (FP32). During MTP verification the
    //! Mamba plugin stashes the minimal per-token replay inputs here (instead of full-state
    //! snapshots); scatterAcceptedLinearStates reconstructs the accepted recurrent state from them.
    //!   dA: [maxBatchSize, maxIntermediateSeqLen, recurrentStateNumHeads]
    //!   u:  [maxBatchSize, maxIntermediateSeqLen, recurrentStateNumHeads, recurrentStateHeadDim]
    //!   B:  [maxBatchSize, maxIntermediateSeqLen, recurrentStateNumGroups, recurrentStateSize]
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getReplayDaState(int32_t recurrentLayerIdx) noexcept;
    rt::Tensor& getReplayUState(int32_t recurrentLayerIdx) noexcept;
    rt::Tensor& getReplayBState(int32_t recurrentLayerIdx) noexcept;

    rt::Tensor& getIntermediateRecurrentState(int32_t recurrentLayerIdx) noexcept;

    //! True when the recurrent state commits via replay reconstruction rather than GDN snapshot scatter.
    bool recurrentUsesReplay() const noexcept;

    //! Get spec-verify intermediate conv state tensor for a given layer.
    //! Shape: [maxBatchSize, maxIntermediateSeqLen, convDim, convKernel]
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getIntermediateConvState(int32_t recurrentLayerIdx) noexcept;

    /*!
     * @brief Reshape spec-verify intermediate state tensors to actual runtime dimensions.
     *
     * TRT writes intermediate state outputs contiguously as [activeBatchSize, seqLen, ...],
     * but the buffers are allocated at [maxBatchSize, maxIntermediateSeqLen, ...].
     * Call this before any code that reads the tensor shape for stride calculations.
     * No-op when intermediate states are not allocated.
     *
     * @param activeBatchSize Current active batch size
     * @param seqLen Actual sequence length (for example, SpecDecode verifySize)
     */
    void reshapeIntermediateStates(int32_t activeBatchSize, int32_t seqLen);

    //! @brief Check if intermediate recurrent state buffers are allocated (spec-verify enabled)
    bool hasIntermediateRecurrentStates() const noexcept;

    //! @brief Check if intermediate conv state buffers are allocated (spec-verify enabled)
    bool hasIntermediateConvStates() const noexcept;

    //! @brief Get the number of recurrent layers
    //! @return Number of recurrent layers
    int32_t numLayers() const noexcept;

    //! @brief Get cache configuration
    //! @return Cache configuration
    Config const& getConfig() const noexcept;

    //! Scatter spec-verify intermediate states to main state pools after verify (one
    //! batched launch per state kind).  No-op when spec-verify intermediate states are not enabled.
    void scatterAcceptedLinearStates(rt::Tensor const& acceptLengths, cudaStream_t stream);

    //! Scatter DDTree base-verify intermediate states to persistent state pools.
    //! acceptedStateNodeIds shape: [activeBatchSize, maxAcceptLen], INT32 GPU.
    //! Entries with -1 are ignored. The ids should name the tree node whose
    //! intermediate state represents each accepted token.
    //! acceptLengths shape: [activeBatchSize], INT32 GPU.
    void scatterAcceptedTreeStates(
        rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream);

    //! Chunk impl (the default): commit DDTree recurrent states by
    //! REPLAYING the accepted path from the committed state, consuming the
    //! per-node stash the chunk-form verify wrote into the head of each
    //! layer's intermediate buffer. Conv states still commit via the
    //! checkpoint scatter (conv checkpoints stay valid in this mode).
    void replayCommitAcceptedTreeStates(
        rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream);

private:
    Config mConfig{};                                //!< Cache configuration
    std::vector<rt::Tensor> mRecurrentStates;        //!< Per-layer recurrent state tensors on device
    std::vector<rt::Tensor> mConvStates;             //!< Per-layer conv state tensors on device
    std::vector<rt::Tensor> mIntermediateConvStates; //!< Per-layer spec-verify intermediate conv states (scatter)
    //! Per-layer GDN/DDTree spec-verify recurrent snapshot. Empty in the Mamba replay path.
    std::vector<rt::Tensor> mIntermediateRecurrentStates;
    //! Per-layer Mamba spec-verify recurrent replay stash; reconstructs the accepted state after
    //! verify. Empty in the GDN snapshot path.
    std::vector<rt::Tensor> mReplayDaStates;
    std::vector<rt::Tensor> mReplayUStates;
    std::vector<rt::Tensor> mReplayBStates;

    rt::Tensor mDeviceMtpLayerInfos;
    rt::Tensor mDeviceMambaReplayLayerInfos;
};

} // namespace rt
} // namespace trt_edgellm
