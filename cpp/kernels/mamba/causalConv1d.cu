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

/*
 * This file contains code derived from causal-conv1d
 * (https://github.com/Dao-AILab/causal-conv1d)
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Licensed under the BSD 3-Clause License.
 *
 * Modifications by NVIDIA:
 * - Adapted causal depthwise conv1d kernel for TensorRT Edge-LLM integration
 * - Added stride, dilation, and padding parameters for generalized conv1d
 * - Added decode-mode kernel (conv_state dot weight)
 * - Added conv state capture and shift-insert kernels
 */

#include "causalConv1d.h"

#include "common/checkMacros.h"
#include "conversion.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <stdexcept>

namespace mamba_ssm
{

template <typename T>
__device__ __forceinline__ float loadCausalConvInput(T const* x, T const* initialState, int32_t batchIdx,
    int32_t stateSlot, int32_t dimIdx, int32_t seqLen, int32_t dim, int32_t width, int32_t inputPos,
    int32_t effectiveSeqLen)
{
    if (inputPos >= 0 && inputPos < effectiveSeqLen)
    {
        int64_t const xIdx = (static_cast<int64_t>(batchIdx) * seqLen + inputPos) * static_cast<int64_t>(dim) + dimIdx;
        return conversion::toFloat(x[xIdx]);
    }
    if (initialState != nullptr && inputPos < 0 && inputPos >= -width)
    {
        int64_t const stateIdx = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * width + width + inputPos;
        return conversion::toFloat(initialState[stateIdx]);
    }
    return 0.0F;
}

// Prefill causal conv1d: sliding window with device-adaptive seq-parallel.
// Maintains a shift register of width input values per thread, reading 1 new value per output
// instead of width. Uses gridDim.z to distribute contiguous chunks across SMs.
//
// Two variants: template kWidth for compile-time unroll, runtime width as fallback.
template <typename T, int32_t kWidth>
__global__ void causalConv1dKernelT(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T const* __restrict__ initialState, T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim,
    int32_t padding, int32_t const* contextLengths, int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    int32_t const stateSlot = stateIndices ? stateIndices[batchIdx] : batchIdx;
    if (stateSlot < 0 || stateSlot >= stateRows)
    {
        return;
    }
    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    float w[kWidth];
#pragma unroll
    for (int32_t k = 0; k < kWidth; ++k)
    {
        w[k] = conversion::toFloat(weight[static_cast<int64_t>(dimIdx) * kWidth + k]);
    }

    int64_t const outBatchOff = static_cast<int64_t>(batchIdx) * outSeqLen * dim;

    int32_t const zBlocks = static_cast<int32_t>(gridDim.z);
    int32_t const chunkSize = (outSeqLen + zBlocks - 1) / zBlocks;
    int32_t const chunkStart = static_cast<int32_t>(blockIdx.z) * chunkSize;
    int32_t const chunkEnd = chunkStart + chunkSize < outSeqLen ? chunkStart + chunkSize : outSeqLen;
    if (chunkStart >= outSeqLen)
    {
        return;
    }

    float xBuf[kWidth];
#pragma unroll
    for (int32_t k = 0; k < kWidth - 1; ++k)
    {
        int32_t const inPos = chunkStart + k - padding;
        xBuf[k] = loadCausalConvInput(
            x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, kWidth, inPos, effectiveSeqLen);
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[kWidth - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + kWidth - 1 - padding;
            xBuf[kWidth - 1] = loadCausalConvInput(
                x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, kWidth, newInPos, effectiveSeqLen);

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < kWidth; ++k)
            {
                acc += xBuf[k] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t k = 0; k < kWidth - 1; ++k)
        {
            xBuf[k] = xBuf[k + 1];
        }
    }
}

// Runtime width fallback
template <typename T>
__global__ void causalConv1dKernel(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T const* __restrict__ initialState, T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim,
    int32_t width, int32_t padding, int32_t const* contextLengths, int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    int32_t const stateSlot = stateIndices ? stateIndices[batchIdx] : batchIdx;
    if (stateSlot < 0 || stateSlot >= stateRows)
    {
        return;
    }
    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    constexpr int32_t kMaxWidth = 8;
    float w[kMaxWidth];
    for (int32_t k = 0; k < width && k < kMaxWidth; ++k)
    {
        w[k] = conversion::toFloat(weight[static_cast<int64_t>(dimIdx) * width + k]);
    }

    int64_t const outBatchOff = static_cast<int64_t>(batchIdx) * outSeqLen * dim;

    // Contiguous chunk for this z-block
    int32_t const zBlocks = static_cast<int32_t>(gridDim.z);
    int32_t const chunkSize = (outSeqLen + zBlocks - 1) / zBlocks;
    int32_t const chunkStart = static_cast<int32_t>(blockIdx.z) * chunkSize;
    int32_t const chunkEnd = chunkStart + chunkSize < outSeqLen ? chunkStart + chunkSize : outSeqLen;
    if (chunkStart >= outSeqLen)
    {
        return;
    }

    // Pre-fill shift register
    float xBuf[kMaxWidth];
    for (int32_t k = 0; k < width - 1; ++k)
    {
        int32_t const inPos = chunkStart + k - padding;
        xBuf[k] = loadCausalConvInput(
            x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, width, inPos, effectiveSeqLen);
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[width - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + width - 1 - padding;
            xBuf[width - 1] = loadCausalConvInput(
                x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, width, newInPos, effectiveSeqLen);

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < width; ++k)
            {
                acc += xBuf[k] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t k = 0; k < width - 1; ++k)
        {
            xBuf[k] = xBuf[k + 1];
        }
    }
}

void invokeCausalConv1d(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out, int32_t stride, int32_t padding,
    int32_t dilation, trt_edgellm::rt::OptionalInputTensor initialState,
    trt_edgellm::rt::OptionalInputTensor contextLengths, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream)
{
    ELLM_CHECK(
        x.getShape().getNumDims() == 3 && weight.getShape().getNumDims() == 3 && out.getShape().getNumDims() == 3,
        "requires x/out [batch, seq, dim] and weight [dim, 1, width].");
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);
    int32_t const outSeqLen = static_cast<int32_t>(out.getShape()[1]);

    ELLM_CHECK(x.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    bool const isContiguous = (x.getStride(2) == 1 && x.getStride(1) == dim && out.getStride(2) == 1
        && out.getStride(1) == dim && weight.getStride(2) == 1);

    ELLM_CHECK(isContiguous && stride == 1 && dilation == 1 && width <= 8,
        "requires contiguous [B,S,D], stride=1, dilation=1, width<=8.");
    if (initialState.has_value())
    {
        trt_edgellm::rt::Tensor const& state = initialState->get();
        ELLM_CHECK(state.getShape().getNumDims() == 3 && state.getShape()[0] >= batch && state.getShape()[1] == dim
                && state.getShape()[2] == width && state.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && state.getDataType() == nvinfer1::DataType::kHALF && state.getStride(2) == 1
                && state.getStride(1) == width && state.getStride(0) == static_cast<int64_t>(dim) * width,
            "initialState must be a contiguous FP16 resident pool [rows, dim, width].");
    }
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(initialState.has_value() && indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == nvinfer1::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices requires initialState and must be contiguous GPU INT32 [batch].");
    }
    if (contextLengths.has_value())
    {
        trt_edgellm::rt::Tensor const& lengths = contextLengths->get();
        ELLM_CHECK(lengths.getShape() == trt_edgellm::rt::Coords{batch}
                && lengths.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && lengths.getDataType() == nvinfer1::DataType::kINT32 && lengths.getStride(0) == 1,
            "contextLengths must be contiguous GPU INT32 [batch].");
    }

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    uint32_t const dimBlocks = static_cast<uint32_t>((dim + kThreads - 1) / kThreads);

    // Adaptive seq-parallel: add z-blocks only when dim-blocks under-utilize the SMs
    int32_t smCount = 0;
    int32_t deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId);
    uint32_t seqBlocks = 1;
    if (dimBlocks < static_cast<uint32_t>(smCount) * 2)
    {
        seqBlocks = (static_cast<uint32_t>(smCount) * 4 + dimBlocks - 1) / dimBlocks;
    }
    if (seqBlocks > static_cast<uint32_t>(outSeqLen))
    {
        seqBlocks = static_cast<uint32_t>(outSeqLen);
    }
    dim3 const grid(batch, dimBlocks, seqBlocks);

    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* clPtr = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    half const* xPtr = x.dataPointer<half>();
    half const* wPtr = weight.dataPointer<half>();
    half const* statePtr = initialState.has_value() ? initialState->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    int32_t const stateRows
        = initialState.has_value() ? static_cast<int32_t>(initialState->get().getShape()[0]) : batch;
    half* outPtr = out.dataPointer<half>();

    switch (width)
    {
    case 2:
        causalConv1dKernelT<half, 2><<<grid, block, 0, stream>>>(
            xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
        break;
    case 3:
        causalConv1dKernelT<half, 3><<<grid, block, 0, stream>>>(
            xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
        break;
    case 4:
        causalConv1dKernelT<half, 4><<<grid, block, 0, stream>>>(
            xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
        break;
    default:
        causalConv1dKernel<half><<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen,
            dim, width, padding, clPtr, stateIndicesPtr, stateRows);
        break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

// Capture last `width` time-steps from x into conv_state (transposed).
template <typename T>
__global__ void captureConvStateKernel(T const* x, T const* initialState, T* convState, int32_t seqLen, int32_t dim,
    int32_t width, int32_t const* contextLengths, int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    int32_t const stateSlot = stateIndices ? stateIndices[batchIdx] : batchIdx;
    if (stateSlot < 0 || stateSlot >= stateRows)
    {
        return;
    }
    int32_t const tailLen = (effectiveSeqLen >= width) ? width : effectiveSeqLen;
    int32_t const tailStart = effectiveSeqLen - tailLen;
    int32_t const dstOffset = width - tailLen;
    int64_t const stateOffset = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * width;

    // Retain the newest values from the prior state when the continuation is
    // shorter than the convolution width. Forward iteration is safe when the
    // input and output states alias because every source index is greater than
    // or equal to its destination index.
    for (int32_t t = 0; t < dstOffset; ++t)
    {
        if (initialState == nullptr)
        {
            conversion::convertAndStore(&convState[stateOffset + t], 0.0F);
        }
        else
        {
            convState[stateOffset + t] = initialState[stateOffset + tailLen + t];
        }
    }

    for (int32_t t = 0; t < tailLen; ++t)
    {
        int64_t const srcIdx = (static_cast<int64_t>(batchIdx) * seqLen + tailStart + t) * dim + dimIdx;
        int64_t const dstIdx = stateOffset + dstOffset + t;
        convState[dstIdx] = x[srcIdx];
    }
}

void invokeCaptureConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::OptionalInputTensor initialState,
    trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::OptionalInputTensor contextLengths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    ELLM_CHECK(x.getShape().getNumDims() == 3 && convState.getShape().getNumDims() == 3,
        "requires x [batch, seq, dim] and convState [batch, dim, width].");
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(convState.getShape()[0] >= batch && convState.getShape()[1] == dim,
        "convState must be a resident pool [rows, dim, width] matching x.");
    ELLM_CHECK(x.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && convState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && x.getDataType() == nvinfer1::DataType::kHALF && convState.getDataType() == nvinfer1::DataType::kHALF,
        "x and convState must be GPU FP16 tensors.");
    ELLM_CHECK(x.getStride(2) == 1 && x.getStride(1) == dim && x.getStride(0) == static_cast<int64_t>(seqLen) * dim
            && convState.getStride(2) == 1 && convState.getStride(1) == width
            && convState.getStride(0) == static_cast<int64_t>(dim) * width,
        "x and convState must be contiguous.");
    if (initialState.has_value())
    {
        trt_edgellm::rt::Tensor const& state = initialState->get();
        ELLM_CHECK(state.getShape() == convState.getShape() && state.getDeviceType() == convState.getDeviceType()
                && state.getDataType() == convState.getDataType() && state.getStride(2) == 1
                && state.getStride(1) == width && state.getStride(0) == static_cast<int64_t>(dim) * width,
            "initialState must match the contiguous GPU convState layout and dtype.");
    }
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == nvinfer1::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices must be contiguous GPU INT32 [batch].");
    }
    if (contextLengths.has_value())
    {
        trt_edgellm::rt::Tensor const& lengths = contextLengths->get();
        ELLM_CHECK(lengths.getShape() == trt_edgellm::rt::Coords{batch}
                && lengths.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && lengths.getDataType() == nvinfer1::DataType::kINT32 && lengths.getStride(0) == 1,
            "contextLengths must be contiguous GPU INT32 [batch].");
    }

    int32_t const* clPtr = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    half const* statePtr = initialState.has_value() ? initialState->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    captureConvStateKernel<half><<<grid, block, 0, stream>>>(x.dataPointer<half>(), statePtr,
        convState.dataPointer<half>(), seqLen, dim, width, clPtr, stateIndicesPtr,
        static_cast<int32_t>(convState.getShape()[0]));
    CUDA_CHECK(cudaPeekAtLastError());
}

// Decode kernel: shift conv_state left by 1, insert new column, then dot with weight + bias
template <typename T>
__global__ void causalConv1dDecodeKernel(T* convState, T const* newCol, T const* weight, T const* bias, T* output,
    int32_t dim, int32_t width, int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const stateSlot = stateIndices ? stateIndices[batchIdx] : batchIdx;
    if (stateSlot < 0 || stateSlot >= stateRows)
    {
        return;
    }
    int64_t const rowOffset = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * width;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;
    T* row = convState + rowOffset;

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // Shift left and compute dot product
    for (int32_t k = 0; k < width - 1; ++k)
    {
        T val = row[k + 1];
        row[k] = val;
        acc += conversion::toFloat(val) * conversion::toFloat(weight[weightOffset + k]);
    }
    // Insert new column and accumulate last weight element
    T newVal = newCol[static_cast<int64_t>(batchIdx) * dim + dimIdx];
    row[width - 1] = newVal;
    acc += conversion::toFloat(newVal) * conversion::toFloat(weight[weightOffset + width - 1]);

    int64_t const outIdx = static_cast<int64_t>(batchIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
}

void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCol.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF && newCol.getDataType() == nvinfer1::DataType::kHALF
            && weight.getDataType() == nvinfer1::DataType::kHALF && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(convState.getShape()[0] >= batch && newCol.getShape()[1] == 1 && newCol.getShape()[2] == dim
            && out.getShape()[0] == batch && out.getShape()[1] == 1 && out.getShape()[2] == dim,
        "convState must be [rows,D,W] and newCol/out must be [batch,1,D].");
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == nvinfer1::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices must be contiguous GPU INT32 [batch].");
    }

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    causalConv1dDecodeKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCol.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(), dim, width,
        stateIndicesPtr, static_cast<int32_t>(convState.getShape()[0]));
    CUDA_CHECK(cudaPeekAtLastError());
}

// MTP decode kernel: process T draft tokens, shift+insert+dot per step, checkpoint state.
template <typename T>
__global__ void causalConv1dDecodeMTPKernel(T const* convState, T const* newCols, T const* weight, T const* bias,
    T* output, T* intermediateConvStates, int32_t batch, int32_t dim, int32_t width, int32_t numTokens,
    int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    // Load current conv_state row into registers (width is small, typically 4).
    int32_t const stateRow = stateIndices == nullptr ? batchIdx : stateIndices[batchIdx];
    if (stateRow < 0 || stateRow >= stateRows)
    {
        return;
    }
    int64_t const rowOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * width;
    float state[8]; // Max supported kernel width (compile-time upper bound).
    for (int32_t k = 0; k < width; ++k)
    {
        state[k] = conversion::toFloat(convState[rowOffset + k]);
    }

    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // Load weight into registers.
    int64_t const wOffset = static_cast<int64_t>(dimIdx) * width;
    float w[8];
    for (int32_t k = 0; k < width; ++k)
    {
        w[k] = conversion::toFloat(weight[wOffset + k]);
    }

    for (int32_t t = 0; t < numTokens; ++t)
    {
        // Shift state left by 1.
        for (int32_t k = 0; k < width - 1; ++k)
        {
            state[k] = state[k + 1];
        }
        // Insert new token at position width-1.
        int64_t const newColIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        state[width - 1] = conversion::toFloat(newCols[newColIdx]);

        // Dot product: output = conv_state · weight + bias.
        float acc = biasVal;
        for (int32_t k = 0; k < width; ++k)
        {
            acc += state[k] * w[k];
        }
        int64_t const outIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], acc);

        // Checkpoint: save intermediate conv_state for rollback.
        // Layout: [batch, T, dim, width]
        int64_t const intermBase = ((static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx) * width;
        for (int32_t k = 0; k < width; ++k)
        {
            conversion::convertAndStore(&intermediateConvStates[intermBase + k], state[k]);
        }
    }
}

void invokeCausalConv1dDecodeMTP(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, int32_t T, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCols.getShape()[0]);
    int32_t const stateRows = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(width <= 8, "kernel_size > 8 not supported.");
    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;

    causalConv1dDecodeMTPKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
        intermediateConvStates.dataPointer<half>(), batch, dim, width, T, stateIndicesPtr, stateRows);
    CUDA_CHECK(cudaPeekAtLastError());
}

// DDTree decode kernel: each node independently reconstructs its conv window by walking
// parent_ids back to root and appending the full root-to-node token path. This avoids
// cross-node synchronization and still executes all tree nodes in one launch.
// Width four is kept separate so that every state row can be transferred as one aligned
// 64-bit value; other widths retain the generic scalar implementation below.
__device__ __forceinline__ void unpackHalf4(uint64_t packed, float* values)
{
    constexpr uint64_t kHalfMask{0xFFFFU}; // bit mask for one packed FP16 value
    constexpr int32_t kHalfBits{16};       // bit width of one packed FP16 value
#pragma unroll
    for (int32_t k = 0; k < 4; ++k)
    {
        uint16_t const raw = static_cast<uint16_t>((packed >> (k * kHalfBits)) & kHalfMask);
        values[k] = __half2float(__ushort_as_half(raw));
    }
}

__device__ __forceinline__ uint64_t packHalf4(float const* values)
{
    constexpr int32_t kHalfBits{16}; // bit width of one packed FP16 value
    uint64_t packed{0};
#pragma unroll
    for (int32_t k = 0; k < 4; ++k)
    {
        uint64_t const raw = static_cast<uint64_t>(__half_as_ushort(__float2half(values[k])));
        packed |= raw << (k * kHalfBits);
    }
    return packed;
}

__global__ void causalConv1dDecodeDDTreeWidth4Kernel(half const* __restrict__ convState,
    half const* __restrict__ newCols, half const* __restrict__ weight, half const* __restrict__ bias,
    half* __restrict__ output, half* __restrict__ intermediateConvStates, int32_t const* __restrict__ treeParentIds,
    int32_t const* __restrict__ treeDepths, int32_t const* __restrict__ stateIndices, int32_t stateRows, int32_t dim,
    int32_t verifySeq)
{
    constexpr int32_t kWidth{4}; // convolution width handled by this specialization
    int32_t const batchIdx = blockIdx.x;
    int32_t const nodeIdx = blockIdx.y;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.z * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const treeOffset = static_cast<int64_t>(batchIdx) * verifySeq;
    int32_t const stateRow = stateIndices == nullptr ? batchIdx : stateIndices[batchIdx];
    if (stateRow < 0 || stateRow >= stateRows)
    {
        return;
    }
    int64_t const stateRowOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * kWidth;
    int32_t const parentIdx = treeParentIds[treeOffset + nodeIdx];
    int32_t const depth = treeDepths[treeOffset + nodeIdx];
    bool const isRoot = nodeIdx == 0 && parentIdx < 0 && depth == 0;
    bool const isValidChild = nodeIdx > 0 && parentIdx >= 0 && parentIdx < nodeIdx && depth > 0;
    bool const isValidNode = isRoot || isValidChild;
    // width==4 makes each [dim, width] row naturally 64-bit aligned.
    uint64_t const packedState = *reinterpret_cast<uint64_t const*>(convState + stateRowOffset);
    int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * kWidth;

    if (!isValidNode)
    {
        // Padding nodes must preserve the base window for a later accepted-path scatter.
        int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], 0.0F);
        *reinterpret_cast<uint64_t*>(intermediateConvStates + intermediateOffset) = packedState;
        return;
    }

    float state[kWidth];
    unpackHalf4(packedState, state);
    // Store the path leaf-to-root. The convolution window only needs its latest four tokens.
    int32_t pathNodes[kWidth];
    int32_t pathLen{0};
    int32_t const maxPathLen = (depth + 1 < kWidth) ? depth + 1 : kWidth;
    int32_t currentNode = nodeIdx;
    while (pathLen < maxPathLen && currentNode >= 0 && currentNode < verifySeq)
    {
        pathNodes[pathLen] = currentNode;
        ++pathLen;
        if (currentNode == 0)
        {
            break;
        }
        currentNode = treeParentIds[treeOffset + currentNode];
    }

#pragma unroll
    for (int32_t k = 0; k < kWidth; ++k)
    {
        if (k + pathLen < kWidth)
        {
            state[k] = state[k + pathLen];
        }
        else
        {
            // Reversing the leaf-to-root walk appends tokens in chronological order.
            int32_t const pathNode = pathNodes[kWidth - 1 - k];
            int64_t const newColIdx = (treeOffset + pathNode) * dim + dimIdx;
            state[k] = conversion::toFloat(newCols[newColIdx]);
        }
    }

    float weightValues[kWidth];
    unpackHalf4(*reinterpret_cast<uint64_t const*>(weight + static_cast<int64_t>(dimIdx) * kWidth), weightValues);
    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;
#pragma unroll
    for (int32_t k = 0; k < kWidth; ++k)
    {
        acc += state[k] * weightValues[k];
    }
    int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
    *reinterpret_cast<uint64_t*>(intermediateConvStates + intermediateOffset) = packHalf4(state);
}

template <typename T>
__global__ void causalConv1dDecodeDDTreeKernel(T const* __restrict__ convState, T const* __restrict__ newCols,
    T const* __restrict__ weight, T const* __restrict__ bias, T* __restrict__ output,
    T* __restrict__ intermediateConvStates, int32_t const* __restrict__ treeParentIds,
    int32_t const* __restrict__ treeDepths, int32_t const* __restrict__ stateIndices, int32_t stateRows, int32_t dim,
    int32_t width, int32_t verifySeq)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const nodeIdx = blockIdx.y;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.z * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const treeOffset = static_cast<int64_t>(batchIdx) * verifySeq;
    int32_t const stateRow = stateIndices == nullptr ? batchIdx : stateIndices[batchIdx];
    if (stateRow < 0 || stateRow >= stateRows)
    {
        return;
    }
    int64_t const stateRowOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * width;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;

    int32_t const parentIdx = treeParentIds[treeOffset + nodeIdx];
    int32_t const depth = treeDepths[treeOffset + nodeIdx];
    bool const isRoot = nodeIdx == 0 && parentIdx < 0 && depth == 0;
    bool const isValidChild = nodeIdx > 0 && parentIdx >= 0 && parentIdx < nodeIdx && depth > 0;
    bool const isValidNode = isRoot || isValidChild;

    float state[8];
    if (!isValidNode)
    {
        for (int32_t k = 0; k < width; ++k)
        {
            state[k] = conversion::toFloat(convState[stateRowOffset + k]);
        }

        int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], 0.0F);
        int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * width;
        for (int32_t k = 0; k < width; ++k)
        {
            conversion::convertAndStore(&intermediateConvStates[intermediateOffset + k], state[k]);
        }
        return;
    }

    int32_t pathNodes[8];
    int32_t pathLen{0};
    // The conv window only depends on the latest `width` tokens.  For deeper
    // tree nodes, truncate from the root side and keep the most recent path.
    int32_t const maxPathLen = (depth + 1 < width) ? depth + 1 : width;
    int32_t currentNode = nodeIdx;
    while (pathLen < maxPathLen && currentNode >= 0 && currentNode < verifySeq)
    {
        pathNodes[pathLen] = currentNode;
        ++pathLen;
        if (currentNode == 0)
        {
            break;
        }
        currentNode = treeParentIds[treeOffset + currentNode];
    }

    for (int32_t k = 0; k < width; ++k)
    {
        if (k + pathLen < width)
        {
            state[k] = conversion::toFloat(convState[stateRowOffset + k + pathLen]);
        }
        else
        {
            state[k] = 0.0F;
        }
    }
    for (int32_t pathOffset = 0; pathOffset < pathLen; ++pathOffset)
    {
        int32_t const pathNode = pathNodes[pathOffset];
        int64_t const newColIdx = (treeOffset + pathNode) * dim + dimIdx;
        state[width - 1 - pathOffset] = conversion::toFloat(newCols[newColIdx]);
    }

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;
    for (int32_t k = 0; k < width; ++k)
    {
        acc += state[k] * conversion::toFloat(weight[weightOffset + k]);
    }
    int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);

    int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * width;
    for (int32_t k = 0; k < width; ++k)
    {
        conversion::convertAndStore(&intermediateConvStates[intermediateOffset + k], state[k]);
    }
}

void invokeCausalConv1dDecodeDDTree(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStateOut, trt_edgellm::rt::Tensor& intermediateConvStates,
    trt_edgellm::rt::Tensor const& treeParentIds, trt_edgellm::rt::Tensor const& treeDepths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCols.getShape()[0]);
    int32_t const stateRows = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);
    int32_t const verifySeq = static_cast<int32_t>(newCols.getShape()[1]);

    ELLM_CHECK(width <= 8, "kernel_size > 8 not supported.");
    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF
            && newCols.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF && convStateOut.getDataType() == nvinfer1::DataType::kHALF
            && intermediateConvStates.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(treeParentIds.getDataType() == nvinfer1::DataType::kINT32
            && treeDepths.getDataType() == nvinfer1::DataType::kINT32,
        "tree_parent_ids/tree_depths must be INT32.");
    ELLM_CHECK(newCols.getShape()[0] == batch && newCols.getShape()[2] == dim, "newCols must be [B, S, D].");
    ELLM_CHECK(weight.getShape()[0] == dim && weight.getShape()[2] == width, "weight must be [D, 1, W].");
    ELLM_CHECK(out.getShape()[0] == batch && out.getShape()[1] == verifySeq && out.getShape()[2] == dim,
        "out must be [B, S, D].");
    ELLM_CHECK(
        convStateOut.getShape()[0] == batch && convStateOut.getShape()[1] == dim && convStateOut.getShape()[2] == width,
        "convStateOut must be [B, D, W].");
    ELLM_CHECK(intermediateConvStates.getShape()[0] == batch && intermediateConvStates.getShape()[1] == verifySeq
            && intermediateConvStates.getShape()[2] == dim && intermediateConvStates.getShape()[3] == width,
        "intermediateConvStates must be [B, S, D, W].");
    ELLM_CHECK(treeParentIds.getShape()[0] == batch && treeParentIds.getShape()[1] == verifySeq,
        "treeParentIds must be [B, S].");
    ELLM_CHECK(
        treeDepths.getShape()[0] == batch && treeDepths.getShape()[1] == verifySeq, "treeDepths must be [B, S].");

    int32_t constexpr kThreads = 256;
    constexpr int32_t kWidth4{4}; // width that selects the packed specialization
    dim3 const block(kThreads);
    dim3 const grid(batch, verifySeq, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;

    if (width == kWidth4)
    {
        causalConv1dDecodeDDTreeWidth4Kernel<<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
            newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
            intermediateConvStates.dataPointer<half>(), treeParentIds.dataPointer<int32_t>(),
            treeDepths.dataPointer<int32_t>(), stateIndicesPtr, stateRows, dim, verifySeq);
    }
    else
    {
        causalConv1dDecodeDDTreeKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
            newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
            intermediateConvStates.dataPointer<half>(), treeParentIds.dataPointer<int32_t>(),
            treeDepths.dataPointer<int32_t>(), stateIndicesPtr, stateRows, dim, width, verifySeq);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace mamba_ssm
