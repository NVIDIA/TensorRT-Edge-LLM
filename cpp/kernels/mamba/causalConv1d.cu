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
 * - Added packed variable-length prefill and state capture interfaces
 * - Added decode-mode kernel (conv_state dot weight)
 * - Added conv state capture and shift-insert kernels
 */

#include "causalConv1d.h"

#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
#include "cuteDslCausalConv1dRunner.h"
#endif

#include "common/checkMacros.h"
#include "conversion.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace mamba_ssm
{

namespace
{
//! Widest convolution weight the runtime-parameter kernels keep in registers.
int32_t constexpr kMaxWidth{8};
//! Longest conv state, and therefore longest dilated receptive field, the kernels accept.
int32_t constexpr kMaxStateLen{16};

//! \brief Validates the (width, dilation, state length) triple shared by every entry point.
void checkDilatedConvShape(char const* context, int32_t width, int32_t dilation, int32_t stateLen)
{
    ELLM_CHECK(width >= 1 && width <= kMaxWidth && dilation >= 1 && stateLen >= (width - 1) * dilation
            && stateLen <= kMaxStateLen && (width - 1) * dilation + 1 <= kMaxStateLen,
        std::string(context)
            + ": requires width <= 8, dilation >= 1, (width-1)*dilation <= state_len <= 16 and a dilated span "
              "(width-1)*dilation+1 <= 16, got width="
            + std::to_string(width) + ", dilation=" + std::to_string(dilation)
            + ", state_len=" + std::to_string(stateLen) + ".");
}
} // namespace

//! \brief Conv-state column read by weight tap \p k. A row holds the \p stateLen inputs preceding
//! the token being produced, so tap `width - 1` is that token and lands one past the row;
//! `stateLen >= (width - 1) * dilation` keeps tap 0 in range.
__device__ __forceinline__ int32_t dilatedStateTap(int32_t stateLen, int32_t width, int32_t dilation, int32_t k)
{
    return stateLen - dilation * (width - 1 - k);
}

template <typename T>
__device__ __forceinline__ float loadCausalConvInput(T const* x, T const* initialState, int32_t batchIdx,
    int32_t stateSlot, int32_t dimIdx, int32_t seqLen, int32_t dim, int32_t stateLen, int32_t inputPos,
    int32_t effectiveSeqLen)
{
    if (inputPos >= 0 && inputPos < effectiveSeqLen)
    {
        int64_t const xIdx = (static_cast<int64_t>(batchIdx) * seqLen + inputPos) * static_cast<int64_t>(dim) + dimIdx;
        return conversion::toFloat(x[xIdx]);
    }
    if (initialState != nullptr && inputPos < 0 && inputPos >= -stateLen)
    {
        int64_t const stateIdx = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * stateLen + stateLen + inputPos;
        return conversion::toFloat(initialState[stateIdx]);
    }
    return 0.0F;
}

// Prefill causal conv1d: sliding window with device-adaptive seq-parallel.
// Maintains a shift register spanning the (kWidth - 1) * kDilation + 1 most recent inputs per
// thread, reading 1 new value per output instead of kWidth. Uses gridDim.z to distribute
// contiguous chunks across SMs.
//
// Two variants: template kWidth/kDilation/kStateLen for compile-time unroll, runtime values as fallback.
template <typename T, int32_t kWidth, int32_t kDilation, int32_t kStateLen>
__global__ void causalConv1dKernelT(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T const* __restrict__ initialState, T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim,
    int32_t padding, int32_t const* contextLengths, int32_t const* stateIndices, int32_t stateRows)
{
    constexpr int32_t kSpan = (kWidth - 1) * kDilation + 1;
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

    float xBuf[kSpan];
#pragma unroll
    for (int32_t j = 0; j < kSpan - 1; ++j)
    {
        int32_t const inPos = chunkStart + j - padding;
        xBuf[j] = loadCausalConvInput(
            x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, kStateLen, inPos, effectiveSeqLen);
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[kSpan - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + kSpan - 1 - padding;
            xBuf[kSpan - 1] = loadCausalConvInput(
                x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, kStateLen, newInPos, effectiveSeqLen);

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < kWidth; ++k)
            {
                acc += xBuf[k * kDilation] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t j = 0; j < kSpan - 1; ++j)
        {
            xBuf[j] = xBuf[j + 1];
        }
    }
}

// Runtime width/dilation fallback
template <typename T>
__global__ void causalConv1dKernel(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T const* __restrict__ initialState, T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim,
    int32_t width, int32_t dilation, int32_t stateLen, int32_t padding, int32_t const* contextLengths,
    int32_t const* stateIndices, int32_t stateRows)
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

    float w[kMaxWidth];
    for (int32_t k = 0; k < width && k < kMaxWidth; ++k)
    {
        w[k] = conversion::toFloat(weight[static_cast<int64_t>(dimIdx) * width + k]);
    }
    int32_t const span = (width - 1) * dilation + 1;

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
    float xBuf[kMaxStateLen];
    for (int32_t j = 0; j < span - 1; ++j)
    {
        int32_t const inPos = chunkStart + j - padding;
        xBuf[j] = loadCausalConvInput(
            x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, stateLen, inPos, effectiveSeqLen);
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[span - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + span - 1 - padding;
            xBuf[span - 1] = loadCausalConvInput(
                x, initialState, batchIdx, stateSlot, dimIdx, seqLen, dim, stateLen, newInPos, effectiveSeqLen);

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < width; ++k)
            {
                acc += xBuf[k * dilation] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t j = 0; j < span - 1; ++j)
        {
            xBuf[j] = xBuf[j + 1];
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

    ELLM_CHECK(x.getDataType() == trt_edgellm::DataType::kHALF && weight.getDataType() == trt_edgellm::DataType::kHALF
            && out.getDataType() == trt_edgellm::DataType::kHALF,
        "only FP16 (half) is supported.");

    bool const isContiguous = (x.getStride(2) == 1 && x.getStride(1) == dim && out.getStride(2) == 1
        && out.getStride(1) == dim && weight.getStride(2) == 1);

    ELLM_CHECK(isContiguous && stride == 1 && dilation >= 1 && width >= 1 && width <= kMaxWidth
            && (width - 1) * dilation + 1 <= kMaxStateLen,
        "requires contiguous [B,S,D], stride=1, width<=8, and a dilated span (width-1)*dilation+1<=16.");

    // Only initialState is indexed by the state length, so a stateless call can assume the
    // undilated convention.
    int32_t const stateLen = initialState.has_value() ? static_cast<int32_t>(initialState->get().getShape()[2]) : width;
    if (initialState.has_value())
    {
        trt_edgellm::rt::Tensor const& state = initialState->get();
        ELLM_CHECK(state.getShape().getNumDims() == 3 && state.getShape()[0] >= batch && state.getShape()[1] == dim
                && stateLen >= (width - 1) * dilation && stateLen <= kMaxStateLen
                && state.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && state.getDataType() == trt_edgellm::DataType::kHALF && state.getStride(2) == 1
                && state.getStride(1) == stateLen && state.getStride(0) == static_cast<int64_t>(dim) * stateLen,
            "initialState must be a contiguous FP16 resident pool [rows, dim, state_len] with "
            "(width-1)*dilation <= state_len <= 16.");
    }
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(initialState.has_value() && indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == trt_edgellm::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices requires initialState and must be contiguous GPU INT32 [batch].");
    }
    if (contextLengths.has_value())
    {
        trt_edgellm::rt::Tensor const& lengths = contextLengths->get();
        ELLM_CHECK(lengths.getShape() == trt_edgellm::rt::Coords{batch}
                && lengths.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && lengths.getDataType() == trt_edgellm::DataType::kINT32 && lengths.getStride(0) == 1,
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

    auto launchFallback = [&]() {
        causalConv1dKernel<half><<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen,
            dim, width, dilation, stateLen, padding, clPtr, stateIndicesPtr, stateRows);
    };

    if (dilation == 1 && stateLen == width)
    {
        switch (width)
        {
        case 2:
            causalConv1dKernelT<half, 2, 1, 2><<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, statePtr, outPtr,
                seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
            break;
        case 3:
            causalConv1dKernelT<half, 3, 1, 3><<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, statePtr, outPtr,
                seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
            break;
        case 4:
            causalConv1dKernelT<half, 4, 1, 4><<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, statePtr, outPtr,
                seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
            break;
        default: launchFallback(); break;
        }
    }
    else if (width == 4 && dilation == 3 && stateLen == 9)
    {
        // Qwen3.8-Flash-Next n-gram short convolution.
        causalConv1dKernelT<half, 4, 3, 9><<<grid, block, 0, stream>>>(
            xPtr, wPtr, biasPtr, statePtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr, stateIndicesPtr, stateRows);
    }
    else
    {
        launchFallback();
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeCausalConv1dPacked(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& cuSeqLens,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor const& hasInitialState,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStates, int32_t padSlotId, int32_t stride, int32_t padding, int32_t dilation,
    cudaStream_t stream)
{
    ELLM_CHECK(x.getShape().getNumDims() == 2 && out.getShape().getNumDims() == 2,
        "requires packed x/out [totalTokens, dim].");
    ELLM_CHECK(weight.getShape().getNumDims() == 3 && cuSeqLens.getShape().getNumDims() == 1
            && stateIndices.getShape().getNumDims() == 1 && hasInitialState.getShape().getNumDims() == 1
            && convStates.getShape().getNumDims() == 3,
        "requires weight [dim, 1, width], cuSeqLens [N + 1], state metadata [N], and convStates [numSlots, dim, "
        "width - 1].");

    int32_t const totalTokens = static_cast<int32_t>(x.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[1]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);
    int32_t const historyLen = width - 1;
    int32_t const numSequences = static_cast<int32_t>(cuSeqLens.getShape()[0]) - 1;
    int32_t const numSlots = static_cast<int32_t>(convStates.getShape()[0]);
    ELLM_CHECK(numSequences > 0, "cuSeqLens must contain at least two entries.");
    ELLM_CHECK(totalTokens >= 0 && dim > 0 && width > 1 && numSlots > 0,
        "totalTokens must be nonnegative and dim, width, and numSlots must be positive.");
    ELLM_CHECK(weight.getShape()[0] == dim && weight.getShape()[1] == 1, "weight must have shape [dim, 1, width].");
    ELLM_CHECK(out.getShape() == x.getShape(), "out must match packed x shape [totalTokens, dim].");
    ELLM_CHECK(stateIndices.getShape() == trt_edgellm::rt::Coords{numSequences}
            && hasInitialState.getShape() == trt_edgellm::rt::Coords{numSequences},
        "stateIndices and hasInitialState must have shape [N].");
    trt_edgellm::rt::Coords const expectedStateShape{numSlots, dim, historyLen};
    ELLM_CHECK(convStates.getShape() == expectedStateShape, "convStates must have shape [numSlots, dim, width - 1].");
    ELLM_CHECK(x.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && weight.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && out.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && convStates.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && x.getDataType() == trt_edgellm::DataType::kHALF && weight.getDataType() == trt_edgellm::DataType::kHALF
            && out.getDataType() == trt_edgellm::DataType::kHALF
            && convStates.getDataType() == trt_edgellm::DataType::kHALF,
        "x, weight, out, and convStates must be GPU FP16 tensors.");
    ELLM_CHECK(cuSeqLens.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && stateIndices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && hasInitialState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && cuSeqLens.getDataType() == trt_edgellm::DataType::kINT32
            && stateIndices.getDataType() == trt_edgellm::DataType::kINT32
            && hasInitialState.getDataType() == trt_edgellm::DataType::kINT32 && cuSeqLens.getStride(0) == 1
            && stateIndices.getStride(0) == 1 && hasInitialState.getStride(0) == 1,
        "cuSeqLens, stateIndices, and hasInitialState must be contiguous GPU INT32 tensors.");
    ELLM_CHECK(x.getStride(1) == 1 && x.getStride(0) == dim && out.getStride(1) == 1 && out.getStride(0) == dim
            && weight.getStride(2) == 1 && weight.getStride(1) == width && weight.getStride(0) == width
            && convStates.getStride(2) == 1 && convStates.getStride(1) == historyLen
            && convStates.getStride(0) == static_cast<int64_t>(dim) * historyLen,
        "x, out, weight, and convStates must be contiguous.");
    ELLM_CHECK(stride == 1 && padding == historyLen && dilation == 1 && width <= 8,
        "requires stride=1, padding=width-1, dilation=1, and width<=8.");
    if (bias.has_value())
    {
        trt_edgellm::rt::Tensor const& biasTensor = bias->get();
        ELLM_CHECK(biasTensor.getShape() == trt_edgellm::rt::Coords{dim}
                && biasTensor.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && biasTensor.getDataType() == trt_edgellm::DataType::kHALF && biasTensor.getStride(0) == 1,
            "bias must be contiguous GPU FP16 [dim].");
    }
#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
    trt_edgellm::CuteDslCausalConv1dParams params{};
    params.x = x.dataPointer<half>();
    params.cuSeqLens = cuSeqLens.dataPointer<int32_t>();
    params.stateIndices = stateIndices.dataPointer<int32_t>();
    params.hasInitialState = hasInitialState.dataPointer<int32_t>();
    params.weight = weight.dataPointer<half>();
    params.bias = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    params.convStates = convStates.dataPointer<half>();
    params.out = out.dataPointer<half>();
    params.totalTokens = totalTokens;
    params.numSequences = numSequences;
    params.dim = dim;
    params.width = width;
    params.numSlots = numSlots;
    params.padSlotId = padSlotId;
    ELLM_CHECK(trt_edgellm::CuteDslCausalConv1dRunner::run(params, stream) == 0,
        "CuTe DSL packed causal Conv1D launch failed.");
#else
    ELLM_CHECK(false, "Packed causal Conv1D requires the CuTe DSL causal_conv1d kernel group.");
#endif
}

bool canImplementPackedVarlen(
    [[maybe_unused]] int32_t smVersion, [[maybe_unused]] int32_t width, [[maybe_unused]] int32_t dilation)
{
#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
    bool const supportedSm = smVersion == 80 || smVersion == 86 || smVersion == 87 || smVersion == 89 || smVersion == 90
        || smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110 || smVersion == 120
        || smVersion == 121;
    return supportedSm && dilation == 1 && width >= 2 && width <= kMaxWidth;
#else
    return false;
#endif
}

//! One thread per (sequence, channel) copies the trailing historyLen columns of a resident row into its dense row
//! and publishes identity row indices and hasInitialState for the packed kernel. A sequence whose state index is
//! outside [0, numRows) starts from a zero state, like the padded kernels.
template <typename T>
__global__ void gatherConvStateHistoryKernel(T const* residentStates, T* historyStates, int32_t const* stateIndices,
    int32_t* historyIndices, int32_t* hasInitialState, int32_t numRows, int32_t dim, int32_t stateLen,
    int32_t historyLen)
{
    int32_t const sequence = static_cast<int32_t>(blockIdx.x);
    int32_t const channel = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    int32_t const row = stateIndices[sequence];
    bool const validRow = row >= 0 && row < numRows;
    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
        historyIndices[sequence] = sequence;
        hasInitialState[sequence] = validRow ? 1 : 0;
    }
    if (channel >= dim || !validRow)
    {
        return;
    }
    int64_t const residentBase = (static_cast<int64_t>(row) * dim + channel) * stateLen + (stateLen - historyLen);
    int64_t const historyBase = (static_cast<int64_t>(sequence) * dim + channel) * historyLen;
    for (int32_t column = 0; column < historyLen; ++column)
    {
        historyStates[historyBase + column] = residentStates[residentBase + column];
    }
}

void invokeGatherConvStateHistory(trt_edgellm::rt::Tensor const& residentStates,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor& historyStates,
    trt_edgellm::rt::Tensor& historyIndices, trt_edgellm::rt::Tensor& hasInitialState, cudaStream_t stream)
{
    ELLM_CHECK(residentStates.getShape().getNumDims() == 3 && historyStates.getShape().getNumDims() == 3
            && stateIndices.getShape().getNumDims() == 1,
        "conv state history requires resident [rows, dim, state_len], history [N, dim, history_len], and indices [N].");
    int32_t const numSequences = static_cast<int32_t>(historyStates.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(historyStates.getShape()[1]);
    int32_t const historyLen = static_cast<int32_t>(historyStates.getShape()[2]);
    int32_t const stateLen = static_cast<int32_t>(residentStates.getShape()[2]);
    ELLM_CHECK(numSequences > 0 && residentStates.getShape()[1] == dim && historyLen > 0 && historyLen <= stateLen
            && stateIndices.getShape()[0] == numSequences,
        "conv state history shapes disagree.");
    ELLM_CHECK(residentStates.getDataType() == trt_edgellm::DataType::kHALF
            && historyStates.getDataType() == trt_edgellm::DataType::kHALF
            && stateIndices.getDataType() == trt_edgellm::DataType::kINT32,
        "conv state history requires FP16 states and INT32 indices.");
    ELLM_CHECK(historyIndices.getShape() == trt_edgellm::rt::Coords{numSequences}
            && hasInitialState.getShape() == trt_edgellm::rt::Coords{numSequences}
            && historyIndices.getDataType() == trt_edgellm::DataType::kINT32
            && hasInitialState.getDataType() == trt_edgellm::DataType::kINT32,
        "conv state history gather requires INT32 [N] index and initial-state outputs.");
    int32_t constexpr kThreads{256};
    dim3 const grid(static_cast<uint32_t>(numSequences), static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    gatherConvStateHistoryKernel<half><<<grid, kThreads, 0, stream>>>(residentStates.dataPointer<half>(),
        historyStates.dataPointer<half>(), stateIndices.dataPointer<int32_t>(), historyIndices.dataPointer<int32_t>(),
        hasInitialState.dataPointer<int32_t>(), static_cast<int32_t>(residentStates.getShape()[0]), dim, stateLen,
        historyLen);
    CUDA_CHECK(cudaPeekAtLastError());
}

//! One thread per (sequence, channel) writes the state captureConvStateKernel would write for the same sequence:
//! column c holds input position len - stateLen + c, or prior column len + c when that position precedes the
//! sequence. Ascending columns make the in-place update safe, since every source column lies right of its
//! destination.
template <typename T>
__global__ void capturePackedConvStateKernel(T const* x, int32_t const* cuSeqLens, int32_t const* stateIndices,
    T* residentStates, int32_t numRows, int32_t dim, int32_t stateLen)
{
    int32_t const sequence = static_cast<int32_t>(blockIdx.x);
    int32_t const channel = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    int32_t const row = stateIndices[sequence];
    if (channel >= dim || row < 0 || row >= numRows)
    {
        return;
    }
    int32_t const start = cuSeqLens[sequence];
    int32_t const len = cuSeqLens[sequence + 1] - start;
    int64_t const stateBase = (static_cast<int64_t>(row) * dim + channel) * stateLen;
    for (int32_t column = 0; column < stateLen; ++column)
    {
        int32_t const position = len - stateLen + column;
        residentStates[stateBase + column] = position >= 0 ? x[(static_cast<int64_t>(start) + position) * dim + channel]
                                                           : residentStates[stateBase + len + column];
    }
}

void invokeCapturePackedConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& cuSeqLens,
    trt_edgellm::rt::Tensor const& stateIndices, trt_edgellm::rt::Tensor& residentStates, cudaStream_t stream)
{
    ELLM_CHECK(x.getShape().getNumDims() == 2 && residentStates.getShape().getNumDims() == 3
            && cuSeqLens.getShape().getNumDims() == 1 && stateIndices.getShape().getNumDims() == 1,
        "packed conv state capture requires x [T, dim], cuSeqLens [N + 1], stateIndices [N], and states "
        "[rows, dim, state_len].");
    int32_t const numSequences = static_cast<int32_t>(stateIndices.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[1]);
    ELLM_CHECK(numSequences > 0 && cuSeqLens.getShape()[0] == numSequences + 1 && residentStates.getShape()[1] == dim,
        "packed conv state capture shapes disagree.");
    ELLM_CHECK(x.getDataType() == trt_edgellm::DataType::kHALF
            && residentStates.getDataType() == trt_edgellm::DataType::kHALF
            && cuSeqLens.getDataType() == trt_edgellm::DataType::kINT32
            && stateIndices.getDataType() == trt_edgellm::DataType::kINT32,
        "packed conv state capture requires FP16 x/states and INT32 metadata.");
    int32_t constexpr kThreads{256};
    dim3 const grid(static_cast<uint32_t>(numSequences), static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    capturePackedConvStateKernel<half><<<grid, kThreads, 0, stream>>>(x.dataPointer<half>(),
        cuSeqLens.dataPointer<int32_t>(), stateIndices.dataPointer<int32_t>(), residentStates.dataPointer<half>(),
        static_cast<int32_t>(residentStates.getShape()[0]), dim, static_cast<int32_t>(residentStates.getShape()[2]));
    CUDA_CHECK(cudaPeekAtLastError());
}

// Capture last `stateLen` time-steps from x into conv_state (transposed).
template <typename T>
__global__ void captureConvStateKernel(T const* x, T const* initialState, T* convState, int32_t seqLen, int32_t dim,
    int32_t stateLen, int32_t const* contextLengths, int32_t const* stateIndices, int32_t stateRows)
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
    int32_t const tailLen = (effectiveSeqLen >= stateLen) ? stateLen : effectiveSeqLen;
    int32_t const tailStart = effectiveSeqLen - tailLen;
    int32_t const dstOffset = stateLen - tailLen;
    int64_t const stateOffset = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * stateLen;

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
        "requires x [batch, seq, dim] and convState [batch, dim, state_len].");
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const stateLen = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(convState.getShape()[0] >= batch && convState.getShape()[1] == dim,
        "convState must be a resident pool [rows, dim, state_len] matching x.");
    ELLM_CHECK(x.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && convState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && x.getDataType() == trt_edgellm::DataType::kHALF
            && convState.getDataType() == trt_edgellm::DataType::kHALF,
        "x and convState must be GPU FP16 tensors.");
    ELLM_CHECK(x.getStride(2) == 1 && x.getStride(1) == dim && x.getStride(0) == static_cast<int64_t>(seqLen) * dim
            && convState.getStride(2) == 1 && convState.getStride(1) == stateLen
            && convState.getStride(0) == static_cast<int64_t>(dim) * stateLen,
        "x and convState must be contiguous.");
    if (initialState.has_value())
    {
        trt_edgellm::rt::Tensor const& state = initialState->get();
        ELLM_CHECK(state.getShape() == convState.getShape() && state.getDeviceType() == convState.getDeviceType()
                && state.getDataType() == convState.getDataType() && state.getStride(2) == 1
                && state.getStride(1) == stateLen && state.getStride(0) == static_cast<int64_t>(dim) * stateLen,
            "initialState must match the contiguous GPU convState layout and dtype.");
    }
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == trt_edgellm::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices must be contiguous GPU INT32 [batch].");
    }
    if (contextLengths.has_value())
    {
        trt_edgellm::rt::Tensor const& lengths = contextLengths->get();
        ELLM_CHECK(lengths.getShape() == trt_edgellm::rt::Coords{batch}
                && lengths.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && lengths.getDataType() == trt_edgellm::DataType::kINT32 && lengths.getStride(0) == 1,
            "contextLengths must be contiguous GPU INT32 [batch].");
    }

    int32_t const* clPtr = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    half const* statePtr = initialState.has_value() ? initialState->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    captureConvStateKernel<half><<<grid, block, 0, stream>>>(x.dataPointer<half>(), statePtr,
        convState.dataPointer<half>(), seqLen, dim, stateLen, clPtr, stateIndicesPtr,
        static_cast<int32_t>(convState.getShape()[0]));
    CUDA_CHECK(cudaPeekAtLastError());
}

// Decode kernel: shift conv_state left by 1, insert new column, then dot with weight + bias
template <typename T>
__global__ void causalConv1dDecodeKernel(T* convState, T const* newCol, T const* weight, T const* bias, T* output,
    int32_t dim, int32_t width, int32_t dilation, int32_t stateLen, int32_t const* stateIndices, int32_t stateRows)
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
    int64_t const rowOffset = (static_cast<int64_t>(stateSlot) * dim + dimIdx) * stateLen;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;
    T* row = convState + rowOffset;

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // The oldest tap is evicted by the shift below, so read it before the row is overwritten.
    int32_t const firstTap = dilatedStateTap(stateLen, width, dilation, 0);
    if (firstTap == 0)
    {
        acc += conversion::toFloat(row[0]) * conversion::toFloat(weight[weightOffset]);
    }

    // Shift left and accumulate every dilated tap on the way past it.
    int32_t nextTap = (firstTap == 0) ? dilation : firstTap;
    int32_t tapIdx = (firstTap == 0) ? 1 : 0;
    for (int32_t k = 0; k < stateLen - 1; ++k)
    {
        T val = row[k + 1];
        row[k] = val;
        if (k + 1 == nextTap)
        {
            acc += conversion::toFloat(val) * conversion::toFloat(weight[weightOffset + tapIdx]);
            nextTap += dilation;
            ++tapIdx;
        }
    }
    // Insert new column and accumulate last weight element
    T newVal = newCol[static_cast<int64_t>(batchIdx) * dim + dimIdx];
    row[stateLen - 1] = newVal;
    acc += conversion::toFloat(newVal) * conversion::toFloat(weight[weightOffset + width - 1]);

    int64_t const outIdx = static_cast<int64_t>(batchIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
}

void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    int32_t dilation, trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCol.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const stateLen = static_cast<int32_t>(convState.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);

    checkDilatedConvShape("causal_conv1d decode", width, dilation, stateLen);
    ELLM_CHECK(convState.getDataType() == trt_edgellm::DataType::kHALF
            && newCol.getDataType() == trt_edgellm::DataType::kHALF
            && weight.getDataType() == trt_edgellm::DataType::kHALF
            && out.getDataType() == trt_edgellm::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(convState.getShape()[0] >= batch && newCol.getShape()[1] == 1 && newCol.getShape()[2] == dim
            && out.getShape()[0] == batch && out.getShape()[1] == 1 && out.getShape()[2] == dim,
        "convState must be [rows,D,S] and newCol/out must be [batch,1,D].");
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == trt_edgellm::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices must be contiguous GPU INT32 [batch].");
    }

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    causalConv1dDecodeKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCol.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(), dim, width, dilation,
        stateLen, stateIndicesPtr, static_cast<int32_t>(convState.getShape()[0]));
    CUDA_CHECK(cudaPeekAtLastError());
}

// MTP decode kernel: process T draft tokens, shift+insert+dot per step, checkpoint state.
template <typename T>
__global__ void causalConv1dDecodeMTPKernel(T const* convState, T const* newCols, T const* weight, T const* bias,
    T* output, T* intermediateConvStates, int32_t batch, int32_t dim, int32_t width, int32_t dilation, int32_t stateLen,
    int32_t numTokens, int32_t const* stateIndices, int32_t stateRows)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    // Load current conv_state row into registers (stateLen is small, typically 4).
    int32_t const stateRow = stateIndices == nullptr ? batchIdx : stateIndices[batchIdx];
    if (stateRow < 0 || stateRow >= stateRows)
    {
        return;
    }
    int64_t const rowOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * stateLen;
    float state[kMaxStateLen];
    for (int32_t k = 0; k < stateLen; ++k)
    {
        state[k] = conversion::toFloat(convState[rowOffset + k]);
    }

    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // Load weight into registers.
    int64_t const wOffset = static_cast<int64_t>(dimIdx) * width;
    float w[kMaxWidth];
    for (int32_t k = 0; k < width; ++k)
    {
        w[k] = conversion::toFloat(weight[wOffset + k]);
    }

    for (int32_t t = 0; t < numTokens; ++t)
    {
        int64_t const newColIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        float const newVal = conversion::toFloat(newCols[newColIdx]);

        // Accumulate in chronological tap order, matching ordinary decode and tree replay.
        float acc = biasVal;
        for (int32_t k = 0; k < width - 1; ++k)
        {
            acc += state[dilatedStateTap(stateLen, width, dilation, k)] * w[k];
        }
        acc += newVal * w[width - 1];
        int64_t const outIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], acc);

        for (int32_t k = 0; k < stateLen - 1; ++k)
        {
            state[k] = state[k + 1];
        }
        state[stateLen - 1] = newVal;

        // Checkpoint: save intermediate conv_state for rollback.
        // Layout: [batch, T, dim, stateLen]
        int64_t const intermBase = ((static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx) * stateLen;
        for (int32_t k = 0; k < stateLen; ++k)
        {
            conversion::convertAndStore(&intermediateConvStates[intermBase + k], state[k]);
        }
    }
}

void invokeCausalConv1dDecodeMTP(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, int32_t T, int32_t dilation,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCols.getShape()[0]);
    int32_t const stateRows = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const stateLen = static_cast<int32_t>(convState.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);

    checkDilatedConvShape("causal_conv1d MTP decode", width, dilation, stateLen);
    ELLM_CHECK(convState.getDataType() == trt_edgellm::DataType::kHALF
            && weight.getDataType() == trt_edgellm::DataType::kHALF
            && out.getDataType() == trt_edgellm::DataType::kHALF,
        "only FP16 (half) is supported.");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;

    causalConv1dDecodeMTPKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
        intermediateConvStates.dataPointer<half>(), batch, dim, width, dilation, stateLen, T, stateIndicesPtr,
        stateRows);
    CUDA_CHECK(cudaPeekAtLastError());
}

// DDTree decode kernel: each node independently reconstructs its conv window by walking
// parent_ids back to root and appending the full root-to-node token path. This avoids
// cross-node synchronization and still executes all tree nodes in one launch.
// The undilated state length four is kept separate so that every state row can be transferred as
// one aligned 64-bit value; every other shape retains the generic scalar implementation below.
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

__global__ void causalConv1dDecodeChainParallelWidth4Kernel(half const* __restrict__ convState,
    half const* __restrict__ newCols, half const* __restrict__ weight, half const* __restrict__ bias,
    half* __restrict__ output, half* __restrict__ intermediateConvStates, int32_t const* __restrict__ stateIndices,
    int32_t stateRows, int32_t dim, int32_t numTokens)
{
    constexpr int32_t kWidth{4};
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = blockIdx.y;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.z * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const stateRow = stateIndices == nullptr ? batchIdx : stateIndices[batchIdx];
    if (stateRow < 0 || stateRow >= stateRows)
    {
        return;
    }
    int64_t const stateOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * kWidth;
    float state[kWidth];
    unpackHalf4(*reinterpret_cast<uint64_t const*>(convState + stateOffset), state);

    int32_t const pathLen = min(tokenIdx + 1, kWidth);
#pragma unroll
    for (int32_t k = 0; k < kWidth; ++k)
    {
        if (k + pathLen < kWidth)
        {
            state[k] = state[k + pathLen];
        }
        else
        {
            int32_t const pathToken = tokenIdx + k - (kWidth - 1);
            int64_t const newColIdx = (static_cast<int64_t>(batchIdx) * numTokens + pathToken) * dim + dimIdx;
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
    int64_t const outputOffset = (static_cast<int64_t>(batchIdx) * numTokens + tokenIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outputOffset], acc);
    *reinterpret_cast<uint64_t*>(intermediateConvStates + outputOffset * kWidth) = packHalf4(state);
}

bool canUseCausalConv1dDecodeChainParallel(int32_t smVersion, int32_t batch, int32_t numTokens, int32_t width,
    int32_t dilation, int32_t stateLen, int32_t stride) noexcept
{
    constexpr int32_t kTargetSm{100};
    constexpr int32_t kBatch{1};
    constexpr int32_t kTokens{8};
    constexpr int32_t kWidth{4};
    return smVersion == kTargetSm && batch == kBatch && numTokens == kTokens && width == kWidth && dilation == 1
        && stateLen == kWidth && stride == 1;
}

void invokeCausalConv1dDecodeChainParallel(trt_edgellm::rt::Tensor const& convState,
    trt_edgellm::rt::Tensor const& newCols, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, trt_edgellm::rt::OptionalInputTensor stateIndices,
    cudaStream_t stream)
{
    constexpr int32_t kWidth{4};
    trt_edgellm::rt::Coords const& convStateShape = convState.getShape();
    trt_edgellm::rt::Coords const& newColsShape = newCols.getShape();
    ELLM_CHECK(convStateShape.getNumDims() == 3 && newColsShape.getNumDims() == 3,
        "parallel chain state and inputs must be rank three.");
    int32_t const batch = static_cast<int32_t>(newColsShape[0]);
    int32_t const numTokens = static_cast<int32_t>(newColsShape[1]);
    int32_t const stateRows = static_cast<int32_t>(convStateShape[0]);
    int32_t const dim = static_cast<int32_t>(convStateShape[1]);
    trt_edgellm::rt::Coords const expectedWeightShape{dim, 1, kWidth};
    trt_edgellm::rt::Coords const expectedInputShape{batch, numTokens, dim};
    trt_edgellm::rt::Coords const expectedOutputShape{batch, numTokens, dim};
    trt_edgellm::rt::Coords const expectedStateShape{batch, numTokens, dim, kWidth};
    ELLM_CHECK(batch > 0 && numTokens > 0 && stateRows >= batch && dim > 0,
        "parallel chain extents must be positive and state rows must cover the batch.");
    ELLM_CHECK(convState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && newCols.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && weight.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && out.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && intermediateConvStates.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU,
        "parallel chain tensors must reside on the GPU.");
    ELLM_CHECK(convState.getDataType() == trt_edgellm::DataType::kHALF
            && newCols.getDataType() == trt_edgellm::DataType::kHALF
            && weight.getDataType() == trt_edgellm::DataType::kHALF && out.getDataType() == trt_edgellm::DataType::kHALF
            && intermediateConvStates.getDataType() == trt_edgellm::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(
        convStateShape[2] == kWidth && newColsShape == expectedInputShape && weight.getShape() == expectedWeightShape,
        "parallel chain convolution requires width and state length four.");
    ELLM_CHECK(out.getShape() == expectedOutputShape && intermediateConvStates.getShape() == expectedStateShape,
        "parallel chain outputs must match [batch, tokens, dim] and [batch, tokens, dim, 4].");
    if (bias.has_value())
    {
        ELLM_CHECK(bias->get().getShape() == trt_edgellm::rt::Coords{dim}
                && bias->get().getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && bias->get().getDataType() == trt_edgellm::DataType::kHALF,
            "bias must be FP16 [dim].");
    }
    if (stateIndices.has_value())
    {
        trt_edgellm::rt::Tensor const& indices = stateIndices->get();
        ELLM_CHECK(indices.getShape() == trt_edgellm::rt::Coords{batch}
                && indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
                && indices.getDataType() == trt_edgellm::DataType::kINT32 && indices.getStride(0) == 1,
            "stateIndices must be contiguous GPU INT32 [batch].");
    }

    int32_t constexpr kThreads{256};
    dim3 const grid(batch, numTokens, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;
    causalConv1dDecodeChainParallelWidth4Kernel<<<grid, kThreads, 0, stream>>>(convState.dataPointer<half>(),
        newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
        intermediateConvStates.dataPointer<half>(), stateIndicesPtr, stateRows, dim, numTokens);
    CUDA_CHECK(cudaPeekAtLastError());
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
    int32_t width, int32_t dilation, int32_t stateLen, int32_t verifySeq)
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
    int64_t const stateRowOffset = (static_cast<int64_t>(stateRow) * dim + dimIdx) * stateLen;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;

    int32_t const parentIdx = treeParentIds[treeOffset + nodeIdx];
    int32_t const depth = treeDepths[treeOffset + nodeIdx];
    bool const isRoot = nodeIdx == 0 && parentIdx < 0 && depth == 0;
    bool const isValidChild = nodeIdx > 0 && parentIdx >= 0 && parentIdx < nodeIdx && depth > 0;
    bool const isValidNode = isRoot || isValidChild;

    int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * stateLen;
    int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
    if (!isValidNode)
    {
        // Padding nodes must preserve the base window for a later accepted-path scatter.
        conversion::convertAndStore(&output[outIdx], 0.0F);
        for (int32_t k = 0; k < stateLen; ++k)
        {
            intermediateConvStates[intermediateOffset + k] = convState[stateRowOffset + k];
        }
        return;
    }

    // The window covers both the dilated dot product and the checkpoint, so it runs one entry
    // longer than the state row when stateLen == (width-1)*dilation.
    int32_t const span = (width - 1) * dilation + 1;
    int32_t const windowLen = (stateLen >= span) ? stateLen : span;
    float window[kMaxStateLen];
    int32_t pathNodes[kMaxStateLen];
    int32_t pathLen{0};
    // The window only depends on the latest `windowLen` tokens.  For deeper
    // tree nodes, truncate from the root side and keep the most recent path.
    int32_t const maxPathLen = (depth + 1 < windowLen) ? depth + 1 : windowLen;
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

    // window[i] is entry `stateLen + pathLen - windowLen + i` of the committed state followed by
    // the root-to-node path; the trailing `pathLen` entries are filled in chronological order below.
    for (int32_t i = 0; i < windowLen - pathLen; ++i)
    {
        window[i] = conversion::toFloat(convState[stateRowOffset + stateLen + pathLen - windowLen + i]);
    }
    for (int32_t pathOffset = 0; pathOffset < pathLen; ++pathOffset)
    {
        int32_t const pathNode = pathNodes[pathOffset];
        int64_t const newColIdx = (treeOffset + pathNode) * dim + dimIdx;
        window[windowLen - 1 - pathOffset] = conversion::toFloat(newCols[newColIdx]);
    }

    // Tap k sits `dilation * (width - 1 - k)` inputs before the node's own token, the window's last entry.
    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;
    for (int32_t k = 0; k < width; ++k)
    {
        acc += window[windowLen - 1 - dilation * (width - 1 - k)] * conversion::toFloat(weight[weightOffset + k]);
    }
    conversion::convertAndStore(&output[outIdx], acc);

    for (int32_t k = 0; k < stateLen; ++k)
    {
        conversion::convertAndStore(&intermediateConvStates[intermediateOffset + k], window[windowLen - stateLen + k]);
    }
}

void invokeCausalConv1dDecodeDDTree(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStateOut, trt_edgellm::rt::Tensor& intermediateConvStates,
    trt_edgellm::rt::Tensor const& treeParentIds, trt_edgellm::rt::Tensor const& treeDepths, int32_t dilation,
    trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(newCols.getShape()[0]);
    int32_t const stateRows = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const stateLen = static_cast<int32_t>(convState.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);
    int32_t const verifySeq = static_cast<int32_t>(newCols.getShape()[1]);

    checkDilatedConvShape("causal_conv1d DDTree decode", width, dilation, stateLen);
    ELLM_CHECK(convState.getDataType() == trt_edgellm::DataType::kHALF
            && newCols.getDataType() == trt_edgellm::DataType::kHALF
            && weight.getDataType() == trt_edgellm::DataType::kHALF && out.getDataType() == trt_edgellm::DataType::kHALF
            && convStateOut.getDataType() == trt_edgellm::DataType::kHALF
            && intermediateConvStates.getDataType() == trt_edgellm::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(treeParentIds.getDataType() == trt_edgellm::DataType::kINT32
            && treeDepths.getDataType() == trt_edgellm::DataType::kINT32,
        "tree_parent_ids/tree_depths must be INT32.");
    ELLM_CHECK(newCols.getShape()[0] == batch && newCols.getShape()[2] == dim, "newCols must be [B, S, D].");
    ELLM_CHECK(weight.getShape()[0] == dim, "weight must be [D, 1, W].");
    ELLM_CHECK(out.getShape()[0] == batch && out.getShape()[1] == verifySeq && out.getShape()[2] == dim,
        "out must be [B, S, D].");
    ELLM_CHECK(convStateOut.getShape()[0] == batch && convStateOut.getShape()[1] == dim
            && convStateOut.getShape()[2] == stateLen,
        "convStateOut must be [B, D, S].");
    ELLM_CHECK(intermediateConvStates.getShape()[0] == batch && intermediateConvStates.getShape()[1] == verifySeq
            && intermediateConvStates.getShape()[2] == dim && intermediateConvStates.getShape()[3] == stateLen,
        "intermediateConvStates must be [B, S, D, S_len].");
    ELLM_CHECK(treeParentIds.getShape()[0] == batch && treeParentIds.getShape()[1] == verifySeq,
        "treeParentIds must be [B, S].");
    ELLM_CHECK(
        treeDepths.getShape()[0] == batch && treeDepths.getShape()[1] == verifySeq, "treeDepths must be [B, S].");

    int32_t constexpr kThreads = 256;
    constexpr int32_t kWidth4{4}; // undilated width that selects the packed specialization
    dim3 const block(kThreads);
    dim3 const grid(batch, verifySeq, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* stateIndicesPtr = stateIndices.has_value() ? stateIndices->get().dataPointer<int32_t>() : nullptr;

    if (width == kWidth4 && dilation == 1 && stateLen == kWidth4)
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
            treeDepths.dataPointer<int32_t>(), stateIndicesPtr, stateRows, dim, width, dilation, stateLen, verifySeq);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace mamba_ssm
