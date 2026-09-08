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

#include "talkerMLPKernels.h"

#ifdef CUTE_DSL_GEMM_ENABLED
#include "cuteDslGemmRunner.h"
#endif

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

//! \brief SiLU activation for FP16
//! \param x Input value
//! \return silu(x) = x / (1 + exp(-x))
__device__ __forceinline__ half silu(half x)
{
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + __expf(-fx)));
}

//! \brief Fused bias addition and SiLU activation kernel (vectorized)
//!
//! Each block processes one token, threads within block handle different dimensions.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//! Requires hiddenDim to be a multiple of 8 and pointers to be 16-byte aligned.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size (must be multiple of 8)
__global__ void biasAndSiLUKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int64_t numTokens, int64_t hiddenDim)
{
    constexpr int32_t kVEC_SIZE = 8; // sizeof(uint4) / sizeof(half)

    // Each block handles one token
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    half* tokenData = data + tokenIdx * hiddenDim;

    using vec_t = uint4;
    int64_t const numVecs = hiddenDim / kVEC_SIZE;

    for (int64_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[i];
        vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[i];

        half* dataPtr = reinterpret_cast<half*>(&dataVec);
        half const* biasPtr = reinterpret_cast<half const*>(&biasVec);

#pragma unroll
        for (int32_t j = 0; j < kVEC_SIZE; ++j)
        {
            dataPtr[j] = silu(__hadd(dataPtr[j], biasPtr[j]));
        }

        reinterpret_cast<vec_t*>(tokenData)[i] = dataVec;
    }
}

//! \brief Vectorized bias addition kernel using half2
//!
//! Uses 2D grid to avoid expensive modulo operation.
//! blockIdx.x handles dimension chunks, blockIdx.y handles tokens.
//!
//! \param[in,out] data Token data with shape [numTokens, hiddenDim] (FP16)
//! \param[in] bias Bias vector with shape [hiddenDim] (FP16)
//! \param[in] numTokens Number of tokens
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void addBiasKernelVectorized(
    half* __restrict__ data, half const* __restrict__ bias, int32_t numTokens, int32_t hiddenDim)
{
    using vec_t = uint4;

    int32_t const tokenIdx = blockIdx.y;
    int32_t const vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    if (tokenIdx >= numTokens || vecIdx >= numVecs)
    {
        return;
    }

    half* tokenData = data + static_cast<int64_t>(tokenIdx) * hiddenDim;

    vec_t dataVec = reinterpret_cast<vec_t const*>(tokenData)[vecIdx];
    vec_t biasVec = reinterpret_cast<vec_t const*>(bias)[vecIdx];

    half2* dataPtr = reinterpret_cast<half2*>(&dataVec);
    half2 const* biasPtr = reinterpret_cast<half2 const*>(&biasVec);

#pragma unroll
    for (int32_t i = 0; i < VEC_SIZE / 2; ++i)
    {
        dataPtr[i] = __hadd2(dataPtr[i], biasPtr[i]);
    }

    reinterpret_cast<vec_t*>(tokenData)[vecIdx] = dataVec;
}

//! \brief Vectorized gather kernel
//!
//! Each block handles one output row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [srcNumTokens, hiddenDim] (FP16)
//! \param[in] indices Indices to gather [numIndices] (INT32)
//! \param[out] output Output tensor [numIndices, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to gather
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void gatherKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one output row
    int32_t const outIdx = blockIdx.x;
    if (outIdx >= numIndices)
    {
        return;
    }

    // Get source row index
    int32_t const srcIdx = indices[outIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(outIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

//! \brief Vectorized scatter kernel
//!
//! Each block handles one source row, threads cooperate to copy hiddenDim elements.
//! Uses vectorized loads/stores (8 FP16 elements = 128-bit) for memory efficiency.
//!
//! \param[in] source Source tensor [numIndices, hiddenDim] (FP16)
//! \param[in] indices Indices to scatter to [numIndices] (INT32)
//! \param[out] output Output tensor [dstNumTokens, hiddenDim] (FP16)
//! \param[in] numIndices Number of rows to scatter
//! \param[in] hiddenDim Hidden dimension size
template <int32_t VEC_SIZE = 8>
__global__ void scatterKernelVectorized(half const* __restrict__ source, int32_t const* __restrict__ indices,
    half* __restrict__ output, int32_t numIndices, int32_t hiddenDim)
{
    // Each block handles one source row
    int32_t const srcIdx = blockIdx.x;
    if (srcIdx >= numIndices)
    {
        return;
    }

    // Get destination row index
    int32_t const dstIdx = indices[srcIdx];

    half const* srcRow = source + static_cast<int64_t>(srcIdx) * hiddenDim;
    half* dstRow = output + static_cast<int64_t>(dstIdx) * hiddenDim;

    // Vectorized processing (8 FP16 = 16 bytes = uint4)
    using vec_t = uint4;

    int32_t const numVecs = hiddenDim / VEC_SIZE;

    // Vectorized copy
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t dataVec = reinterpret_cast<vec_t const*>(srcRow)[i];
        reinterpret_cast<vec_t*>(dstRow)[i] = dataVec;
    }

    // Handle remainder
    int32_t const remainderStart = numVecs * VEC_SIZE;
    for (int32_t i = remainderStart + threadIdx.x; i < hiddenDim; i += blockDim.x)
    {
        dstRow[i] = srcRow[i];
    }
}

// Internal host function wrappers for kernel launches (not exposed in header)

void invokeBiasAndSiLU(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "hiddenDim must be a multiple of 8 for vectorized access");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int64_t const numTokens = data.getShape()[0];
    int64_t const hiddenDim = data.getShape()[1];

    dim3 const grid(numTokens);
    dim3 const block(256);

    biasAndSiLUKernelVectorized<<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeAddBias(rt::Tensor& data, rt::Tensor const& bias, cudaStream_t stream)
{
    check::check(data.getDataType() == nvinfer1::DataType::kHALF, "Data tensor must be FP16");
    check::check(bias.getDataType() == nvinfer1::DataType::kHALF, "Bias tensor must be FP16");
    check::check(data.getShape().getNumDims() == 2, "Data tensor must be 2D [numTokens, hiddenDim]");
    check::check(bias.getShape().getNumDims() == 1, "Bias tensor must be 1D [hiddenDim]");
    check::check(data.getShape()[1] == bias.getShape()[0], "Hidden dimension mismatch");
    check::check(data.getShape()[1] % 8 == 0, "Hidden dimension must be divisible by 8 for vectorization");
    check::check(reinterpret_cast<uintptr_t>(data.rawPointer()) % 16 == 0, "Data pointer must be 16-byte aligned");
    check::check(reinterpret_cast<uintptr_t>(bias.rawPointer()) % 16 == 0, "Bias pointer must be 16-byte aligned");

    int32_t const numTokens = static_cast<int32_t>(data.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(data.getShape()[1]);

    constexpr int32_t VEC_SIZE = 8;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    dim3 const block(256);
    dim3 const grid((numVecs + block.x - 1) / block.x, numTokens);

    addBiasKernelVectorized<VEC_SIZE><<<grid, block, 0, stream>>>(
        static_cast<half*>(data.rawPointer()), static_cast<half const*>(bias.rawPointer()), numTokens, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace

void invokeTalkerMLP(rt::Tensor const& input, rt::Tensor const& fc1Weight, rt::Tensor const& fc1Bias,
    rt::Tensor const& fc2Weight, rt::Tensor const& fc2Bias, rt::Tensor& output, rt::Tensor& workspace,
    cudaStream_t stream)
{
    auto inputShape = input.getShape();
    auto outputShape = output.getShape();
    auto workspaceShape = workspace.getShape();
    auto fc1WeightShape = fc1Weight.getShape();
    auto fc2WeightShape = fc2Weight.getShape();

    if (input.getDataType() != nvinfer1::DataType::kHALF || fc1Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc1Bias.getDataType() != nvinfer1::DataType::kHALF || fc2Weight.getDataType() != nvinfer1::DataType::kHALF
        || fc2Bias.getDataType() != nvinfer1::DataType::kHALF || output.getDataType() != nvinfer1::DataType::kHALF
        || workspace.getDataType() != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("All tensors must be FP16");
        return;
    }

    if (inputShape.getNumDims() != 2 || outputShape.getNumDims() != 2 || workspaceShape.getNumDims() != 2)
    {
        LOG_ERROR("Tensors must be 2D");
        return;
    }

    int32_t const numTokens = static_cast<int32_t>(inputShape[0]);
    int32_t const inputDim = static_cast<int32_t>(inputShape[1]);
    int32_t const hiddenDim = static_cast<int32_t>(fc1WeightShape[0]);
    int32_t const outputDim = static_cast<int32_t>(fc2WeightShape[0]);

    if (fc1WeightShape[1] != inputDim)
    {
        LOG_ERROR("FC1 weight shape mismatch: expected [*, %d], got [%ld, %ld]", inputDim, fc1WeightShape[0],
            fc1WeightShape[1]);
        return;
    }

    if (fc2WeightShape[1] != hiddenDim)
    {
        LOG_ERROR("FC2 weight shape mismatch: expected [*, %d], got [%ld, %ld]", hiddenDim, fc2WeightShape[0],
            fc2WeightShape[1]);
        return;
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    // FC1: workspace = SiLU(input @ fc1Weight^T + fc1Bias)
    if (!CuteDslGemmRunner::runBiasSiLU(input.rawPointer(), fc1Weight.rawPointer(), workspace.rawPointer(),
            fc1Bias.rawPointer(), numTokens, hiddenDim, inputDim, stream))
    {
        LOG_ERROR("FC1 GEMM+bias+SiLU failed");
        return;
    }

    // FC2: output = workspace @ fc2Weight^T + fc2Bias
    if (!CuteDslGemmRunner::runBias(workspace.rawPointer(), fc2Weight.rawPointer(), output.rawPointer(),
            fc2Bias.rawPointer(), numTokens, outputDim, hiddenDim, stream))
    {
        LOG_ERROR("FC2 GEMM+bias failed");
        return;
    }
#else
    ELLM_CHECK(false,
        "invokeTalkerMLP requires CuTe DSL GEMM. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL); "
        "without it the launcher returns without writing the output tensor and downstream "
        "sampling consumes uninitialised GPU memory, producing wrong-but-not-crashing audio.");
#endif
}

void invokeLinearLayer(
    rt::Tensor const& input, rt::Tensor const& weight, rt::Tensor const& bias, rt::Tensor& output, cudaStream_t stream)
{
    auto inputShape = input.getShape();
    auto outputShape = output.getShape();
    auto weightShape = weight.getShape();

    if (input.getDataType() != nvinfer1::DataType::kHALF || weight.getDataType() != nvinfer1::DataType::kHALF
        || bias.getDataType() != nvinfer1::DataType::kHALF || output.getDataType() != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("All tensors must be FP16");
        return;
    }

    if (inputShape.getNumDims() != 2 || outputShape.getNumDims() != 2 || weightShape.getNumDims() != 2)
    {
        LOG_ERROR("Tensors must be 2D [N, dim]");
        return;
    }

    int32_t const numTokens = static_cast<int32_t>(inputShape[0]);
    int32_t const inputDim = static_cast<int32_t>(inputShape[1]);
    int32_t const outputDim = static_cast<int32_t>(weightShape[0]);

    if (weightShape[1] != inputDim)
    {
        LOG_ERROR("Weight shape mismatch: expected [*, %d], got [%ld, %ld]", inputDim, weightShape[0], weightShape[1]);
        return;
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    // output = input @ weight^T + bias
    if (!CuteDslGemmRunner::runBias(input.rawPointer(), weight.rawPointer(), output.rawPointer(), bias.rawPointer(),
            numTokens, outputDim, inputDim, stream))
    {
        LOG_ERROR("Linear layer GEMM+bias failed");
        return;
    }
#else
    ELLM_CHECK(false,
        "invokeLinearLayer requires CuTe DSL GEMM. Rebuild with -DENABLE_CUTE_DSL=gemm (or ALL); "
        "without it the launcher returns without writing the output tensor and downstream "
        "consumers see uninitialised GPU memory.");
#endif
}

void invokeGather(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    gatherKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeScatter(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream)
{
    check::check(source.getDataType() == nvinfer1::DataType::kHALF, "Source tensor must be FP16");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT32, "Indices tensor must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output tensor must be FP16");

    auto const srcDims = source.getShape();
    int32_t const numIndices = static_cast<int32_t>(indices.getShape()[0]);
    int32_t const hiddenDim = static_cast<int32_t>(srcDims[srcDims.getNumDims() - 1]);

    if (numIndices == 0)
    {
        return;
    }

    dim3 const grid(numIndices);
    dim3 const block(256);

    scatterKernelVectorized<8><<<grid, block, 0, stream>>>(source.dataPointer<half>(), indices.dataPointer<int32_t>(),
        static_cast<half*>(output.rawPointer()), numIndices, hiddenDim);

    CUDA_CHECK(cudaPeekAtLastError());
}

//! \brief Non-streaming fused assistant preamble construction kernel
//!
//! Each block handles one output row (blockIdx.x). Total rows = P + textLen + 2, where
//! P = 8 (no language) or 9 (languageId >= 0: CustomVoice language conditioning).
//!
//! Canonical slot layout (slot == rowIdx when language is present; without language,
//! rows >= 5 map to slot rowIdx+1, skipping the language slot):
//!   slot 0-2:   copy projected[0-2]
//!   slot 3:     ttsPad + embTable[codecThinkId]   (with language) / embTable[codecNothinkId] (without)
//!   slot 4:     ttsPad + embTable[codecThinkBosId]
//!   slot 5:     ttsPad + embTable[languageId]     (only present with language)
//!   slot 6:     ttsPad + embTable[codecThinkEosId]
//!   slot 7:     ttsPad + embTable[speakerId]
//!   slot 8:     ttsBos + embTable[codecPadId]
//! Text/suffix rows:
//!   P..P+N-2:   projected[3+i] + embTable[codecPadId]  (i = rowIdx-P)
//!   P+N-1:      projected[3+N-1] + embTable[codecBosId]  (last text row = start-of-generation)
//!   P+N:        ttsEos + embTable[codecPadId]
//!   P+N+1:      ttsPad + embTable[codecBosId]
//!
//! Matches PyTorch reference modeling_qwen3_tts.py codec_prefill_list:
//!   language present -> [codec_think, think_bos, language_id, think_eos]
//!   language absent  -> [codec_nothink, think_bos, think_eos]
template <int32_t VEC_SIZE = 8>
__global__ void assistantPreambleKernel(half const* __restrict__ projected, half const* __restrict__ ttsPadEmbed,
    half const* __restrict__ ttsBosEmbed, half const* __restrict__ ttsEosEmbed, half const* __restrict__ embTable,
    int32_t codecNothinkId, int32_t codecThinkBosId, int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId,
    int32_t codecBosId, int32_t codecThinkId, int32_t languageId, int32_t textLen, int32_t hiddenDim,
    half* __restrict__ output)
{
    constexpr int32_t kFixedPrefixLen = 8; // canonical prefix rows without language
    bool const hasLanguage = (languageId >= 0);
    int32_t const prefixLen = kFixedPrefixLen + (hasLanguage ? 1 : 0);
    int32_t const rowIdx = blockIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    using vec_t = uint4;

    half const* srcA;
    half const* srcB = nullptr;
    half* const dstRow = output + static_cast<int64_t>(rowIdx) * hiddenDim;

    if (rowIdx < prefixLen)
    {
        // Map row to canonical slot; without language, rows >= 5 skip the language slot.
        int32_t const slot = (!hasLanguage && rowIdx >= 5) ? rowIdx + 1 : rowIdx;
        switch (slot)
        {
        case 0: srcA = projected; break;
        case 1: srcA = projected + hiddenDim; break;
        case 2: srcA = projected + 2 * hiddenDim; break;
        case 3:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(hasLanguage ? codecThinkId : codecNothinkId) * hiddenDim;
            break;
        case 4:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(codecThinkBosId) * hiddenDim;
            break;
        case 5: // language row — only reachable when hasLanguage
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(languageId) * hiddenDim;
            break;
        case 6:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(codecThinkEosId) * hiddenDim;
            break;
        case 7:
            srcA = ttsPadEmbed;
            srcB = embTable + static_cast<int64_t>(speakerId) * hiddenDim;
            break;
        default: // slot 8
            srcA = ttsBosEmbed;
            srcB = embTable + static_cast<int64_t>(codecPadId) * hiddenDim;
            break;
        }
    }
    else if (rowIdx < prefixLen + textLen)
    {
        // Text token rows: projected[3 + (rowIdx-prefixLen)] + embTable[codec]
        // Last text row uses codecBosId (start-of-generation marker);
        // all preceding text rows use codecPadId. Matches PyTorch reference:
        //   assistant_codec_hidden = [zeros(3), think-block, speaker, codecPad, codecBos]
        int32_t const textIdx = rowIdx - prefixLen;
        srcA = projected + static_cast<int64_t>(3 + textIdx) * hiddenDim;
        int32_t const codecId = (rowIdx == prefixLen + textLen - 1) ? codecBosId : codecPadId;
        srcB = embTable + static_cast<int64_t>(codecId) * hiddenDim;
    }
    else if (rowIdx == prefixLen + textLen)
    {
        // ttsEos + embTable[codecPadId]
        srcA = ttsEosEmbed;
        srcB = embTable + static_cast<int64_t>(codecPadId) * hiddenDim;
    }
    else
    {
        // ttsPad + embTable[codecBosId]
        srcA = ttsPadEmbed;
        srcB = embTable + static_cast<int64_t>(codecBosId) * hiddenDim;
    }

    // Vectorized copy-and-optionally-add
    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t va = reinterpret_cast<vec_t const*>(srcA)[i];
        if (srcB != nullptr)
        {
            vec_t vb = reinterpret_cast<vec_t const*>(srcB)[i];
            half2* aPtr = reinterpret_cast<half2*>(&va);
            half2 const* bPtr = reinterpret_cast<half2 const*>(&vb);
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE / 2; ++j)
            {
                aPtr[j] = __hadd2(aPtr[j], bPtr[j]);
            }
        }
        reinterpret_cast<vec_t*>(dstRow)[i] = va;
    }
}

void invokeAssistantPreamble(rt::Tensor const& projected, rt::Tensor const& ttsPadEmbed, rt::Tensor const& ttsBosEmbed,
    rt::Tensor const& ttsEosEmbed, rt::Tensor const& talkerEmbTable, int32_t codecNothinkId, int32_t codecThinkBosId,
    int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId, int32_t codecBosId, int32_t codecThinkId,
    int32_t languageId, int32_t textLen, rt::Tensor& output, cudaStream_t stream)
{
    constexpr int32_t kVecSize = 8;

    int32_t const hiddenDim = static_cast<int32_t>(projected.getShape()[1]);
    int32_t const numVecs = hiddenDim / kVecSize;
    // totalRows = prefix (8, or 9 with language row) + textLen text rows + 2 suffix rows
    int32_t const prefixLen = 8 + ((languageId >= 0) ? 1 : 0);
    int32_t const totalRows = prefixLen + textLen + 2;

    // 128 threads covers H=1024 with VEC_SIZE=8 in one pass
    dim3 const block(std::min(numVecs, 128));
    dim3 const grid(totalRows);

    half const* projPtr = projected.dataPointer<half>();
    half const* padPtr = ttsPadEmbed.dataPointer<half>();
    half const* bosPtr = ttsBosEmbed.dataPointer<half>();
    half const* eosPtr = ttsEosEmbed.dataPointer<half>();
    half const* embPtr = talkerEmbTable.dataPointer<half>();
    half* outPtr = static_cast<half*>(output.rawPointer());

    assistantPreambleKernel<kVecSize><<<grid, block, 0, stream>>>(projPtr, padPtr, bosPtr, eosPtr, embPtr,
        codecNothinkId, codecThinkBosId, codecThinkEosId, speakerId, codecPadId, codecBosId, codecThinkId, languageId,
        textLen, hiddenDim, outPtr);
    CUDA_CHECK(cudaPeekAtLastError());
}

//! Descriptor-driven prefill row assembly. Per-row math is identical to
//! assistantPreambleKernel (same uint4 loads + half2 adds), so identical row descriptions
//! produce bit-identical output.
template <int32_t VEC_SIZE = 8>
__global__ void prefillRowAssembleKernel(
    kernel::PrefillRowDesc const* __restrict__ descs, int32_t hiddenDim, half* __restrict__ output)
{
    int32_t const rowIdx = blockIdx.x;
    int32_t const numVecs = hiddenDim / VEC_SIZE;

    using vec_t = uint4;

    kernel::PrefillRowDesc const desc = descs[rowIdx];
    half* const dstRow = output + static_cast<int64_t>(rowIdx) * hiddenDim;

    for (int32_t i = threadIdx.x; i < numVecs; i += blockDim.x)
    {
        vec_t va = reinterpret_cast<vec_t const*>(desc.srcA)[i];
        if (desc.srcB != nullptr)
        {
            vec_t vb = reinterpret_cast<vec_t const*>(desc.srcB)[i];
            half2* aPtr = reinterpret_cast<half2*>(&va);
            half2 const* bPtr = reinterpret_cast<half2 const*>(&vb);
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE / 2; ++j)
            {
                aPtr[j] = __hadd2(aPtr[j], bPtr[j]);
            }
        }
        reinterpret_cast<vec_t*>(dstRow)[i] = va;
    }
}

void invokePrefillRowAssemble(
    PrefillRowDesc const* deviceDescs, int32_t numRows, int32_t hiddenDim, rt::Tensor& output, cudaStream_t stream)
{
    constexpr int32_t kVecSize = 8;
    int32_t const numVecs = hiddenDim / kVecSize;

    dim3 const block(std::min(numVecs, 128));
    dim3 const grid(numRows);

    prefillRowAssembleKernel<kVecSize>
        <<<grid, block, 0, stream>>>(deviceDescs, hiddenDim, static_cast<half*>(output.rawPointer()));
    CUDA_CHECK(cudaPeekAtLastError());
}

//! One block per reference frame; each thread accumulates its hidden elements over all
//! code groups in FP32 before the final half round.
__global__ void sumCodecEmbeddingsKernel(int64_t const* __restrict__ refCodes,
    half const* const* __restrict__ tablePtrs, int32_t numGroups, int32_t hiddenDim, half* __restrict__ output)
{
    int32_t const frame = blockIdx.x;
    int64_t const* frameCodes = refCodes + static_cast<int64_t>(frame) * numGroups;
    half* dstRow = output + static_cast<int64_t>(frame) * hiddenDim;

    for (int32_t d = threadIdx.x; d < hiddenDim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int32_t g = 0; g < numGroups; ++g)
        {
            half const* table = tablePtrs[g];
            acc += __half2float(table[frameCodes[g] * hiddenDim + d]);
        }
        dstRow[d] = __float2half(acc);
    }
}

void invokeSumCodecEmbeddings(int64_t const* refCodes, half const* const* tablePtrs, int32_t numFrames,
    int32_t numGroups, int32_t hiddenDim, rt::Tensor& output, cudaStream_t stream)
{
    dim3 const block(std::min(hiddenDim, 256));
    dim3 const grid(numFrames);
    sumCodecEmbeddingsKernel<<<grid, block, 0, stream>>>(
        refCodes, tablePtrs, numGroups, hiddenDim, static_cast<half*>(output.rawPointer()));
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void castFp32ToFp16Kernel(float const* __restrict__ input, half* __restrict__ output, int64_t numElements)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        output[idx] = __float2half(input[idx]);
    }
}

void invokeCastFp32ToFp16(float const* input, half* output, int64_t numElements, cudaStream_t stream)
{
    constexpr int32_t kBlock = 256;
    int32_t const grid = static_cast<int32_t>((numElements + kBlock - 1) / kBlock);
    castFp32ToFp16Kernel<<<grid, kBlock, 0, stream>>>(input, output, numElements);
    CUDA_CHECK(cudaPeekAtLastError());
}

//! Fused residual connection kernel
//!
//! Computes: output[j] = embed0[code0,j] + embedLast[codeLast,j] + addend[j] +
//! sum_{k=1}^{numInnerRows}(codecHiddens[k,j]) Single block of (hiddenDim/VEC_SIZE) threads; FP32 accumulators for
//! precision.
template <int32_t VEC_SIZE = 8>
__global__ void residualConnectionKernel(half const* __restrict__ codecHiddens, half const* __restrict__ embTable0,
    half const* __restrict__ embTableLast, int32_t code0, int32_t codeLast, half const* __restrict__ addend,
    int32_t hiddenDim, int32_t numInnerRows, half* __restrict__ output)
{
    using vec_t = uint4;
    int32_t const vecIdx = static_cast<int32_t>(threadIdx.x);
    int32_t const numVecs = hiddenDim / VEC_SIZE;
    if (vecIdx >= numVecs)
    {
        return;
    }

    float acc[VEC_SIZE];

    // embed(code0) from Talker embedding table
    {
        vec_t v = reinterpret_cast<vec_t const*>(embTable0 + static_cast<int64_t>(code0) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] = __half2float(p[i]);
        }
    }
    // embed(codeLast) from CodePredictor embedding table
    {
        vec_t v = reinterpret_cast<vec_t const*>(embTableLast + static_cast<int64_t>(codeLast) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }
    // addend: trailing_text_hidden[generationStep] or tts_pad_embed
    {
        vec_t v = reinterpret_cast<vec_t const*>(addend)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }
    // sum codecHiddens rows 1..numInnerRows (row 0 and last replaced by direct embedding lookups above)
    for (int32_t k = 1; k <= numInnerRows; ++k)
    {
        vec_t v = reinterpret_cast<vec_t const*>(codecHiddens + static_cast<int64_t>(k) * hiddenDim)[vecIdx];
        half const* p = reinterpret_cast<half const*>(&v);
#pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; ++i)
        {
            acc[i] += __half2float(p[i]);
        }
    }

    // store as FP16
    vec_t result;
    half* rp = reinterpret_cast<half*>(&result);
#pragma unroll
    for (int32_t i = 0; i < VEC_SIZE; ++i)
    {
        rp[i] = __float2half(acc[i]);
    }
    reinterpret_cast<vec_t*>(output)[vecIdx] = result;
}

void invokeResidualConnection(rt::Tensor const& codecHiddens, rt::Tensor const& embTable0,
    rt::Tensor const& embTableLast, int32_t code0, int32_t codeLast, half const* addend, rt::Tensor& output,
    cudaStream_t stream)
{
    int32_t const hiddenDim = static_cast<int32_t>(embTable0.getShape()[1]);
    constexpr int32_t kVecSize = 8;
    int32_t const numVecs = hiddenDim / kVecSize;

    // codecHiddens has shape [1, numCodesPerFrame, H]: skip leading batch dim
    // numInnerRows = total rows - 2 (excluding first and last which use direct embedding lookup)
    int32_t const numCodesPerFrame = static_cast<int32_t>(codecHiddens.getShape()[1]);
    int32_t const numInnerRows = numCodesPerFrame - 2;
    half const* codecPtr = codecHiddens.dataPointer<half>();

    residualConnectionKernel<kVecSize><<<1, numVecs, 0, stream>>>(codecPtr, embTable0.dataPointer<half>(),
        embTableLast.dataPointer<half>(), code0, codeLast, addend, hiddenDim, numInnerRows,
        static_cast<half*>(output.rawPointer()));
    CUDA_CHECK(cudaPeekAtLastError());
}

// Each thread handles one work item. The total work is (suppressCount + numSeenTokens).
// Threads [0, suppressCount) do suppression; threads [suppressCount, suppressCount+numSeenTokens) apply penalty.
// The two ranges never overlap: suppression covers [suppressStart, suppressEnd) which is always in the
// upper special-token region, while seenTokens are sampled from the normal codec range below suppressStart.
__global__ void talkerLogitAdjustKernel(float* logits, int32_t suppressStart, int32_t suppressCount, int32_t codecEosId,
    int32_t const* seenTokens, int32_t numSeenTokens, float repetitionPenalty)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x) * blockDim.x + static_cast<int32_t>(threadIdx.x);

    if (idx < suppressCount)
    {
        int32_t const tokenId = suppressStart + idx;
        if (tokenId != codecEosId)
        {
            logits[tokenId] = -INFINITY;
        }
    }
    else if (idx < suppressCount + numSeenTokens)
    {
        int32_t const tokenId = seenTokens[idx - suppressCount];
        float const logit = logits[tokenId];
        logits[tokenId] = (logit >= 0.0f) ? logit / repetitionPenalty : logit * repetitionPenalty;
    }
}

void invokeTalkerLogitAdjust(rt::Tensor const& seenTokens, rt::Tensor& logits, int32_t suppressStart,
    int32_t suppressEnd, int32_t codecEosId, int32_t numSeenTokens, float repetitionPenalty, cudaStream_t stream)
{
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "Logits tensor must be FP32");
    check::check(seenTokens.getDataType() == nvinfer1::DataType::kINT32, "seenTokens tensor must be INT32");

    int32_t const suppressCount = suppressEnd - suppressStart;
    int32_t const totalWork = suppressCount + numSeenTokens;
    if (totalWork <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 128;
    int32_t const gridSize = (totalWork + kBlockSize - 1) / kBlockSize;
    float* logitsPtr = static_cast<float*>(logits.rawPointer());

    talkerLogitAdjustKernel<<<gridSize, kBlockSize, 0, stream>>>(logitsPtr, suppressStart, suppressCount, codecEosId,
        seenTokens.dataPointer<int32_t>(), numSeenTokens, repetitionPenalty);
    CUDA_CHECK(cudaPeekAtLastError());
}

//! \brief Per-position sum across codec groups (Qwen3.5-Omni speaker codec embedding fold).
//!
//! Mirrors HF ``_get_codec_input_embeddings``: concatenate per-group embeddings then sum across
//! the group axis. Implemented here as a single-row reduction so the caller only invokes one
//! kernel per row of the speaker codec template.
namespace
{
__global__ void speakerCodecSumKernel(int64_t const* __restrict__ codes, __half const* const* __restrict__ embTables,
    int32_t const* __restrict__ embVocabSizes, int32_t numCodeGroups, int32_t hiddenSize, __half* __restrict__ output)
{
    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hiddenSize)
    {
        return;
    }
    float acc = 0.0f;
    for (int32_t g = 0; g < numCodeGroups; ++g)
    {
        int64_t const code = codes[g];
        if (code < 0 || code >= embVocabSizes[g])
        {
            continue; // HF: -1 entries are no-ops; OOB defensively skipped
        }
        acc += __half2float(embTables[g][code * hiddenSize + tid]);
    }
    output[tid] = __float2half(acc);
}
} // namespace

void invokeSpeakerCodecSum(rt::Tensor const& codes, rt::Tensor const& embPtrTable, rt::Tensor const& embVocabSizes,
    rt::Tensor& output, cudaStream_t stream)
{
    check::check(codes.getDataType() == nvinfer1::DataType::kINT64, "codes must be INT64");
    check::check(embVocabSizes.getDataType() == nvinfer1::DataType::kINT32, "embVocabSizes must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    int32_t const numCodeGroups = static_cast<int32_t>(codes.getShape()[0]);
    int32_t const hiddenSize = static_cast<int32_t>(output.getShape().volume());

    constexpr int32_t kBlockSize = 256;
    int32_t const blocks = (hiddenSize + kBlockSize - 1) / kBlockSize;
    speakerCodecSumKernel<<<blocks, kBlockSize, 0, stream>>>(codes.dataPointer<int64_t>(),
        static_cast<__half const* const*>(embPtrTable.rawPointer()), embVocabSizes.dataPointer<int32_t>(),
        numCodeGroups, hiddenSize, static_cast<__half*>(output.rawPointer()));
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace kernel
} // namespace trt_edgellm
