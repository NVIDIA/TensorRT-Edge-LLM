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
        int32_t maxIntermediateSeqLen{0};  //!< Maximum speculative verify nodes (0 = disabled)
        nvinfer1::DataType recurrentStateType{nvinfer1::DataType::kHALF}; //!< Recurrent state dtype
        nvinfer1::DataType convStateType{nvinfer1::DataType::kHALF};      //!< Conv state dtype
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

    //! Get spec-verify intermediate recurrent state tensor for a given layer.
    //! Shape: [maxBatchSize, compactReplayBufferElements].
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getIntermediateRecurrentState(int32_t recurrentLayerIdx) noexcept;

    //! Get spec-verify intermediate conv state tensor for a given layer.
    //! Shape: [maxBatchSize, maxIntermediateSeqLen, convDim, convKernel]
    //! @param recurrentLayerIdx The recurrent layer index.
    //! @return A reference to the owned device tensor.
    rt::Tensor& getIntermediateConvState(int32_t recurrentLayerIdx) noexcept;

    /*!
     * @brief Reshape spec-verify intermediate state tensors to actual runtime dimensions.
     *
     * The recurrent replay buffer has a fixed per-batch row size, while conv
     * checkpoints retain a runtime sequence dimension. Call this before
     * speculative verification to select the active batch and verify length.
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

    //! Commit branching-tree or linear-chain recurrent states by replaying the
    //! accepted path from the committed state. Conv states still commit via
    //! checkpoint scatter.
    void replayCommitAcceptedTreeStates(
        rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream);

private:
    Config mConfig{};                                     //!< Cache configuration
    std::vector<rt::Tensor> mRecurrentStates;             //!< Per-layer recurrent state tensors on device
    std::vector<rt::Tensor> mConvStates;                  //!< Per-layer conv state tensors on device
    std::vector<rt::Tensor> mIntermediateRecurrentStates; //!< Per-layer compact recurrent replay buffers
    std::vector<rt::Tensor> mIntermediateConvStates;      //!< Per-layer spec-verify intermediate conv states
    int32_t mActiveIntermediateSeqLen{0};                 //!< Verify nodes selected by reshapeIntermediateStates

    //! Device-resident MtpLayerInfo[numRecurrentLayers] for batched replay and conv scatter.
    //! Built once at construction; empty when spec-verify buffers are unallocated.
    rt::Tensor mDeviceMtpLayerInfos;
};

} // namespace rt
} // namespace trt_edgellm
