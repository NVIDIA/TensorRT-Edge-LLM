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

#include "fmhaParams_v2.h"

#include <NvInferRuntime.h>
namespace trt_edgellm
{

/*!
 * @brief Runner for context-phase fused multi-head attention (FMHA)
 *
 * Manages and dispatches optimized FMHA kernels for the prefill/context phase
 * of LLM inference. Supports various attention patterns (causal, sliding window)
 * and memory layouts (packed QKV, separate Q/KV, paged KV cache).
 */
class ContextFMHARunner
{
public:
    /*!
     * @brief Construct context FMHA runner
     * @param dataType Data type (e.g., FP16, BF16)
     * @param batchSize Batch size
     * @param paddedSeqLen Padded sequence length
     * @param numQHeads Number of query heads
     * @param numKvHeads Number of key-value heads
     * @param headSize Attention head dimension
     * @param smVersion CUDA compute capability (e.g., 89 for SM 8.9)
     * @param inputLayout Input tensor layout
     */
    ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t paddedSeqLen, int32_t numQHeads,
        int32_t numKvHeads, int32_t headSize, int32_t smVersion, AttentionInputLayout inputLayout);

    //! @brief Deleted default constructor
    ContextFMHARunner() = delete;

    //! @brief Destructor
    ~ContextFMHARunner() = default;

    //! @brief Get required workspace size in bytes
    //! @return Workspace size
    size_t getWorkspaceSize();

    /*!
     * @brief Setup kernel parameters (excluding device pointers)
     *
     * Configures FMHA parameters. Device pointers must be set by caller.
     *
     * @param params FMHA parameter structure
     */
    void setupParams(FusedMultiheadAttentionParamsV2& params);

    /*!
     * @brief Dispatch FMHA kernel execution
     * @param params FMHA parameters with device pointers set
     * @param stream CUDA stream for kernel launch
     */
    void dispatchFMHAKernel(FusedMultiheadAttentionParamsV2& params, cudaStream_t const& stream);

    // Static methods to check kernel availability and load cubins into device.
    /*!
     * @brief Check if FMHA can be implemented for given configuration
     * @param headSize Attention head dimension
     * @param sm CUDA compute capability
     * @param dataType Data type
     * @return True if implementation is available
     */
    static bool canImplement(int32_t headSize, int32_t sm, nvinfer1::DataType dataType);

    /*!
     * @brief Load FMHA kernel cubins into device
     * @param sm CUDA compute capability
     * @param dataType Data type
     * @return True if successful
     */
    static bool loadContextFMHAKernels(int32_t sm, nvinfer1::DataType dataType);

private:
    nvinfer1::DataType mDataType; //!< Data type
    int32_t mBatchSize;           //!< Batch size
    int32_t mPaddedSequenceLen;   //!< Padded sequence length
    int32_t mNumHeads;            //!< Number of query heads
    int32_t mNumKVHeads;          //!< Number of key-value heads
    int32_t mHeadSize;            //!< Attention head dimension

    int32_t mSmVersion;         //!< CUDA compute capability
    LaunchParams mLaunchParams; //!< Kernel launch parameters
};

} // namespace trt_edgellm