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

#include "decoderXQAJitCompiler.h"

#include <NvInferRuntime.h>

#include <cstddef>

namespace trt_edgellm
{

//! \brief Launch parameters for XQA (eXtended Query Attention) kernel
struct XQALaunchParams
{
    //! \cond INTERNAL
    //! \brief KV cache structure
    struct KVCache
    {
        void* data = nullptr;                      //!< Pointer to KV cache data
        int32_t const* pageList = nullptr;         //!< Page table for paged KV cache
        int32_t const* sequence_lengths = nullptr; //!< Sequence lengths for each request
        uint32_t capacity = 0;                     //!< Cache capacity
        uint32_t tokensPerPage = 0;                //!< Tokens per page (0 = contiguous KV cache)
    };
    //! \endcond

    //! Device memory pointers to launch XQA kernel
    void* output = nullptr;          //!< Output tensor
    void const* qInputPtr = nullptr; //!< Query input pointer
    KVCache kvCache;                 //!< KV cache structure
    float kScale = 1.0f;             //!< K dequant scale (quantized -> original), host scalar
    float vScale = 1.0f;             //!< V dequant scale (quantized -> original), host scalar
    uint32_t slidingWinSize = 0;     //!< Sliding window size (0 = no sliding window)
    int32_t* semaphores = nullptr;   //!< Semaphores for synchronization
    void* scratch = nullptr;         //!< Scratch memory

    //! Unique device memory pointer for spec-decode tree attention
    void* treeAttnMask = nullptr; //!< Tree attention mask
    int32_t* qCuSeqLen = nullptr; //!< Cumulative query sequence lengths

    float const* attentionSinks = nullptr; //!< Attention sinks parameter

    //! MHA parameters to locate a kernel to launch
    int32_t numQheads = 0;  //!< Number of query heads
    int32_t numKVheads = 0; //!< Number of key-value heads
    int32_t headSize = 0;   //!< Head dimension size
    int32_t batchSize = 0;  //!< Batch size

    //! Parameters for spec-decode tree attention
    int32_t qSeqLen = 0;         //!< Query sequence length
    float attentionScale = 1.0F; //!< Absolute QK^T multiplier
    int32_t headGroupSize = 0;   //!< Head group size

    nvinfer1::DataType dataType;   //!< I/O data type of the kernel
    nvinfer1::DataType kvDataType; //!< KV cache data type
};

//! \brief Decoder XQA (eXtended Query Attention) kernel runner
class DecoderXQARunner
{
public:
    //! \brief Constructor for DecoderXQARunner
    //! \param[in] dataType Data type for computation
    //! \param[in] kvDataType KV cache data type
    //! \param[in] batchSize Batch size
    //! \param[in] numQHeads Number of query heads
    //! \param[in] numKvHeads Number of key-value heads
    //! \param[in] headSize Head dimension size
    //! \param[in] smVersion CUDA SM version
    DecoderXQARunner(nvinfer1::DataType const dataType, nvinfer1::DataType const kvDataType, int32_t batchSize,
        int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion) noexcept;

    DecoderXQARunner() noexcept = default;

    ~DecoderXQARunner() noexcept = default;

    //! \brief Dispatch XQA kernel and compute the attention result
    //! \param[in,out] params Launch parameters for XQA kernel
    //! \param[in] stream CUDA stream for kernel execution
    //! \throws std::runtime_error if device pointers are invalid or no available kernel available
    //! \throws std::runtime_error if a CUDA driver error occurs
    void dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream);

    //! \brief Dispatch spec-decode XQA kernel for tree attention
    //! \param[in,out] params Launch parameters for XQA kernel
    //! \param[in] stream CUDA stream for kernel execution
    //! \throws std::runtime_error if device pointers are invalid or no available kernel available
    //! \throws std::runtime_error if a CUDA driver error occurs
    void dispatchSpecDecodeXQAKernel(XQALaunchParams& params, cudaStream_t const& stream);

    //! \brief Initialize XQA parameters with MHA and hardware configuration
    //!
    //! The XQA parameter can be used by prepareToRun() to query kernel to dispatch.
    //! Device pointer shall be setup by caller to dispatch XQA kernel.
    //!
    //! \return Initialized XQA launch parameters
    XQALaunchParams initXQAParams() noexcept;

    //! \brief Load one decoder XQA kernel from a JIT-generated module image.
    //! \param[in] key JIT key that describes the kernel variant
    //! \param[in] cubinData Cubin or PTX bytes returned by NVRTC
    //! \param[in] cubinSize Number of bytes in cubinData
    //! \return True if the kernel loaded successfully and was inserted into the runtime cache
    //! \throws std::runtime_error if a CUDA driver error occurs
    static bool loadDecodeXQAKernelFromCubin(XQAJitKey const& key, void const* cubinData, size_t cubinSize);

private:
    nvinfer1::DataType mDataType;   //!< Data type for computation
    nvinfer1::DataType mKVDataType; //!< KV cache data type
    uint32_t mBatchSize;            //!< Batch size
    uint32_t mNumHeads;             //!< Number of query heads
    uint32_t mNumKVHeads;           //!< Number of key-value heads
    uint32_t mHeadSize;             //!< Head dimension size

    int32_t mSmVersion; //!< CUDA SM version
};

} // namespace trt_edgellm
