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

#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/causalConv1d.h"
#include "testUtils.h"

using namespace trt_edgellm;

//! Reference for the virtual sequence `in = concat(conv state, valid inputs)`:
//! `out[t] = sum_k weight[k] * in[stateLen + t - dilation * (width - 1 - k)] + bias`.
void runCausalConv1dReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, int32_t dilation,
    int32_t stateLen, std::vector<half> const& x, std::vector<half> const& weight, std::vector<half> const& bias,
    std::vector<half>& outRef, std::vector<int32_t> const* contextLens = nullptr,
    std::vector<half> const* initialState = nullptr)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLens ? (*contextLens)[b] : seqLen;
        for (int32_t s = 0; s < seqLen; ++s)
        {
            for (int32_t d = 0; d < dim; ++d)
            {
                int64_t const outIdx = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(s) * dim + d;
                if (s >= cl)
                {
                    outRef[outIdx] = __float2half(0.F);
                    continue;
                }
                float acc = __half2float(bias[d]);
                for (int32_t k = 0; k < width; ++k)
                {
                    int32_t const inPos = s - dilation * (width - 1 - k);
                    int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                    if (inPos >= 0 && inPos < cl)
                    {
                        int64_t const xIdx
                            = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(inPos) * dim + d;
                        acc += __half2float(x[xIdx]) * __half2float(weight[wIdx]);
                    }
                    else if (initialState != nullptr && inPos < 0 && inPos >= -stateLen)
                    {
                        int64_t const stateIdx = (static_cast<int64_t>(b) * dim + d) * stateLen + stateLen + inPos;
                        acc += __half2float((*initialState)[stateIdx]) * __half2float(weight[wIdx]);
                    }
                }
                outRef[outIdx] = __float2half(acc);
            }
        }
    }
}

void runCausalConv1dTest(
    int32_t batch, int32_t seqLen, int32_t dim, int32_t width, std::vector<int32_t> const* contextLens = nullptr)
{
    std::vector<half> xHost(batch * seqLen * dim);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> outputRef(batch * seqLen * dim, __float2half(0.F));

    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);

    runCausalConv1dReference(batch, seqLen, dim, width, 1, width, xHost, weightHost, biasHost, outputRef, contextLens);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(xDevice, xHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    CUDA_CHECK(cudaMemset(outputDevice.rawPointer(), 0, outputDevice.getMemoryCapacity()));

    rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    rt::OptionalInputTensor clOpt = std::nullopt;
    rt::Tensor clDevice;
    if (contextLens)
    {
        clDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(
            clDevice.rawPointer(), contextLens->data(), contextLens->size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        clOpt = std::optional(std::cref(clDevice));
    }
    mamba_ssm::invokeCausalConv1d(
        xDevice, weightDevice, biasOpt, outputDevice, 1, width - 1, 1, std::nullopt, clOpt, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outputHost = copyDeviceToHost<half>(outputDevice);

    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], 1e-3F, 1e-3F))
            << "Output mismatch at index " << i << ": got " << __half2float(outputHost[i]) << ", expected "
            << __half2float(outputRef[i]);
    }
}

TEST(MambaCausalConv1d, Width2)
{
    runCausalConv1dTest(2, 16, 128, 2);
}

TEST(MambaCausalConv1d, Width3)
{
    runCausalConv1dTest(2, 23, 128, 3);
}

TEST(MambaCausalConv1d, Width4)
{
    runCausalConv1dTest(2, 31, 256, 4);
}

#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)

void runCausalConv1dPackedReference(std::vector<int32_t> const& cuSeqLens, std::vector<int32_t> const& stateIndices,
    std::vector<int32_t> const& hasInitialState, int32_t padSlotId, int32_t dim, int32_t width,
    std::vector<half> const& x, std::vector<half> const& weight, std::vector<half> const* bias,
    std::vector<half>& convStates, std::vector<half>& outRef)
{
    int32_t const numSequences = static_cast<int32_t>(stateIndices.size());
    int32_t const numSlots = static_cast<int32_t>(convStates.size()) / (dim * (width - 1));
    int32_t const historyLen = width - 1;
    std::vector<half> const initialStates = convStates;

    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        int32_t const seqStart = cuSeqLens[sequence];
        int32_t const seqLen = cuSeqLens[sequence + 1] - seqStart;
        int32_t const stateSlot = stateIndices[sequence];
        bool const useInitialState
            = hasInitialState[sequence] != 0 && stateSlot != padSlotId && stateSlot >= 0 && stateSlot < numSlots;
        for (int32_t token = 0; token < seqLen; ++token)
        {
            for (int32_t channel = 0; channel < dim; ++channel)
            {
                float acc = bias == nullptr ? 0.0F : __half2float((*bias)[channel]);
                for (int32_t k = 0; k < width; ++k)
                {
                    int32_t const inputPos = token - historyLen + k;
                    float input = 0.0F;
                    if (inputPos >= 0)
                    {
                        input = __half2float(x[static_cast<int64_t>(seqStart + inputPos) * dim + channel]);
                    }
                    else if (useInitialState)
                    {
                        int64_t const stateIdx
                            = (static_cast<int64_t>(stateSlot) * dim + channel) * historyLen + historyLen + inputPos;
                        input = __half2float(initialStates[stateIdx]);
                    }
                    acc += input * __half2float(weight[static_cast<int64_t>(channel) * width + k]);
                }
                outRef[static_cast<int64_t>(seqStart + token) * dim + channel] = __float2half(acc);
            }
        }
    }

    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        int32_t const stateSlot = stateIndices[sequence];
        if (stateSlot == padSlotId || stateSlot < 0 || stateSlot >= numSlots)
        {
            continue;
        }
        int32_t const seqStart = cuSeqLens[sequence];
        int32_t const seqLen = cuSeqLens[sequence + 1] - seqStart;
        int32_t const tailLen = std::min(seqLen, historyLen);
        int32_t const dstOffset = historyLen - tailLen;
        for (int32_t channel = 0; channel < dim; ++channel)
        {
            int64_t const stateOffset = (static_cast<int64_t>(stateSlot) * dim + channel) * historyLen;
            for (int32_t t = 0; t < dstOffset; ++t)
            {
                convStates[stateOffset + t]
                    = hasInitialState[sequence] != 0 ? initialStates[stateOffset + tailLen + t] : __float2half(0.0F);
            }
            for (int32_t t = 0; t < tailLen; ++t)
            {
                int64_t const xIdx = static_cast<int64_t>(seqStart + seqLen - tailLen + t) * dim + channel;
                convStates[stateOffset + dstOffset + t] = x[xIdx];
            }
        }
    }
}

void runCausalConv1dPackedStep(std::vector<int32_t> const& cuSeqLens, std::vector<int32_t> const& stateIndices,
    std::vector<int32_t> const& hasInitialState, int32_t padSlotId, int32_t dim, int32_t width,
    std::vector<half> const& xHost, std::vector<half> const& weightHost, std::vector<half> const* biasHost,
    std::vector<half>& convStatesHost, std::vector<half>* outputResult = nullptr)
{
    int32_t const numSequences = static_cast<int32_t>(stateIndices.size());
    int32_t const totalTokens = cuSeqLens.back();
    int32_t const historyLen = width - 1;
    int32_t const numSlots = static_cast<int32_t>(convStatesHost.size()) / (dim * historyLen);
    std::vector<half> outputRef(static_cast<size_t>(totalTokens) * dim);
    std::vector<half> stateRef = convStatesHost;
    runCausalConv1dPackedReference(cuSeqLens, stateIndices, hasInitialState, padSlotId, dim, width, xHost, weightHost,
        biasHost, stateRef, outputRef);

    auto xDevice = rt::Tensor({totalTokens, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto cuSeqLensDevice = rt::Tensor({numSequences + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    auto stateIndicesDevice = rt::Tensor({numSequences}, rt::DeviceType::kGPU, DataType::kINT32);
    auto hasInitialStateDevice = rt::Tensor({numSequences}, rt::DeviceType::kGPU, DataType::kINT32);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateDevice = rt::Tensor({numSlots, dim, historyLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({totalTokens, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(xDevice, xHost);
    copyHostToDevice(cuSeqLensDevice, cuSeqLens);
    copyHostToDevice(stateIndicesDevice, stateIndices);
    copyHostToDevice(hasInitialStateDevice, hasInitialState);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(stateDevice, convStatesHost);
    if (biasHost != nullptr)
    {
        copyHostToDevice(biasDevice, *biasHost);
    }

    rt::OptionalInputTensor const biasOpt
        = biasHost == nullptr ? std::nullopt : rt::OptionalInputTensor{std::cref(biasDevice)};
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    mamba_ssm::invokeCausalConv1dPacked(xDevice, cuSeqLensDevice, stateIndicesDevice, hasInitialStateDevice,
        weightDevice, biasOpt, outputDevice, stateDevice, padSlotId, 1, historyLen, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto const outputHost = copyDeviceToHost<half>(outputDevice);
    convStatesHost = copyDeviceToHost<half>(stateDevice);
    if (outputResult != nullptr)
    {
        *outputResult = outputHost;
    }
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], 1e-3F, 1e-3F))
            << "Packed output mismatch at index " << i << ": got " << __half2float(outputHost[i]) << ", expected "
            << __half2float(outputRef[i]);
    }
    for (size_t i = 0; i < stateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(convStatesHost[i], stateRef[i], 1e-3F, 1e-3F))
            << "Packed state mismatch at index " << i << ": got " << __half2float(convStatesHost[i]) << ", expected "
            << __half2float(stateRef[i]);
    }
}

TEST(MambaCausalConv1dPacked, HeterogeneousShortSequences)
{
    // Given heterogeneous cold sequences shorter and longer than the convolution width,
    // when they are processed as one packed tensor, then each output and saved tail stays within its sequence.
    int32_t constexpr kDim = 64;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kNumSlots = 5;
    std::vector<half> x(9 * kDim), weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    uniformFloatInitialization<half>(x, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    runCausalConv1dPackedStep({0, 1, 3, 9}, {4, 0, 2}, {0, 0, 0}, -1, kDim, kWidth, x, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, ChunkContinuation)
{
    // Given state produced by an initial packed chunk, when shorter continuation chunks reuse the same slots,
    // then convolution consumes the prior tails and preserves the newest width-minus-one values.
    int32_t constexpr kDim = 64;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kNumSlots = 4;
    std::vector<half> weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    std::vector<half> firstChunk(3 * kDim), secondChunk(5 * kDim);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    uniformFloatInitialization<half>(firstChunk, -0.5F, 0.5F);
    uniformFloatInitialization<half>(secondChunk, -0.5F, 0.5F);
    runCausalConv1dPackedStep({0, 2, 3}, {3, 1}, {0, 0}, -1, kDim, kWidth, firstChunk, weight, &bias, states);
    runCausalConv1dPackedStep({0, 3, 5}, {3, 1}, {1, 1}, -1, kDim, kWidth, secondChunk, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, SlotReorderAndReuse)
{
    // Given resident state slots whose execution rows are reordered and later reused cold,
    // when packed chunks are processed, then state follows slot identity rather than execution-row order.
    int32_t constexpr kDim = 32;
    int32_t constexpr kWidth = 5;
    int32_t constexpr kNumSlots = 5;
    std::vector<half> weight(kDim * kWidth), states(kNumSlots * kDim * (kWidth - 1));
    std::vector<half> firstStep(5 * kDim), reorderedStep(3 * kDim), reusedStep(2 * kDim);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    uniformFloatInitialization<half>(firstStep, -0.5F, 0.5F);
    uniformFloatInitialization<half>(reorderedStep, -0.5F, 0.5F);
    uniformFloatInitialization<half>(reusedStep, -0.5F, 0.5F);
    runCausalConv1dPackedStep({0, 2, 5}, {3, 1}, {0, 0}, -1, kDim, kWidth, firstStep, weight, nullptr, states);
    runCausalConv1dPackedStep({0, 1, 3}, {1, 3}, {1, 1}, -1, kDim, kWidth, reorderedStep, weight, nullptr, states);
    runCausalConv1dPackedStep({0, 2}, {3}, {0}, -1, kDim, kWidth, reusedStep, weight, nullptr, states);
}

TEST(MambaCausalConv1dPacked, MixedColdWarmAndPadRows)
{
    // Given warm, cold, and padding rows in one packed invocation, when convolution runs,
    // then warm history is consumed, cold history is zeroed, and the padding slot remains unchanged.
    int32_t constexpr kDim = 64;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kNumSlots = 4;
    int32_t constexpr kPadSlotId = 3;
    std::vector<half> x(6 * kDim), weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    uniformFloatInitialization<half>(x, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    runCausalConv1dPackedStep(
        {0, 2, 5, 6}, {2, 0, kPadSlotId}, {1, 0, 1}, kPadSlotId, kDim, kWidth, x, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, DecodeUsesPackedMetadata)
{
    // Given one packed token per decode row with reordered, cold, warm, and padding metadata,
    // when convolution runs, then outputs and state updates honor the per-row metadata.
    int32_t constexpr kDim = 64;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kNumSlots = 6;
    int32_t constexpr kPadSlotId = 5;
    std::vector<half> x(4 * kDim), weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    uniformFloatInitialization<half>(x, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    runCausalConv1dPackedStep(
        {0, 1, 2, 3, 4}, {kPadSlotId, 1, 4, 0}, {1, 0, 1, 0}, kPadSlotId, kDim, kWidth, x, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, DecodeSizedMixedLengths)
{
    // Given a decode-sized packed batch containing an empty row and a multi-token row,
    // when convolution runs, then the empty warm slot is preserved and the cold row is updated independently.
    int32_t constexpr kDim = 32;
    int32_t constexpr kWidth = 5;
    int32_t constexpr kNumSlots = 4;
    std::vector<half> x(2 * kDim), weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    uniformFloatInitialization<half>(x, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    runCausalConv1dPackedStep({0, 0, 2}, {3, 1}, {1, 0}, -1, kDim, kWidth, x, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, SpecializedWidths)
{
    // Given every exported width specialization, when the same packed cold/warm workload is dispatched,
    // then each AOT wrapper produces the reference output and recurrent state.
    int32_t constexpr kDim = 32;
    int32_t constexpr kNumSlots = 4;
    for (int32_t const width : {2, 3, 4, 5, 6, 7, 8})
    {
        std::vector<half> x(5 * kDim), weight(kDim * width), bias(kDim);
        std::vector<half> states(kNumSlots * kDim * (width - 1));
        uniformFloatInitialization<half>(x, -0.5F, 0.5F);
        uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
        uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
        uniformFloatInitialization<half>(states, -0.5F, 0.5F);
        runCausalConv1dPackedStep({0, 2, 5}, {3, 1}, {1, 0}, -1, kDim, width, x, weight, &bias, states);
    }
}

TEST(MambaCausalConv1dPacked, EmptySequences)
{
    // Given empty sequences before and after a nonempty packed sequence, when convolution runs,
    // then empty rows do not consume adjacent tokens and their cold/warm state semantics remain independent.
    int32_t constexpr kDim = 32;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kNumSlots = 4;
    std::vector<half> x(2 * kDim), weight(kDim * kWidth), bias(kDim);
    std::vector<half> states(kNumSlots * kDim * (kWidth - 1));
    uniformFloatInitialization<half>(x, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(states, -0.5F, 0.5F);
    runCausalConv1dPackedStep({0, 0, 2, 2}, {3, 1, 0}, {1, 0, 0}, -1, kDim, kWidth, x, weight, &bias, states);
}

TEST(MambaCausalConv1dPacked, MatchesRectangularImplementation)
{
    // Given equivalent heterogeneous inputs and warm states in packed and rectangular layouts,
    // when both convolution paths run, then all valid outputs and saved state tails match.
    int32_t constexpr kBatch = 3;
    int32_t constexpr kMaxSeqLen = 6;
    int32_t constexpr kDim = 32;
    int32_t constexpr kWidth = 4;
    int32_t constexpr kHistoryLen = kWidth - 1;
    std::vector<int32_t> const lengths{1, 3, 6};
    std::vector<int32_t> const cuSeqLens{0, 1, 4, 10};
    std::vector<half> packedX(cuSeqLens.back() * kDim), paddedX(kBatch * kMaxSeqLen * kDim, __float2half(0.0F));
    std::vector<half> weight(kDim * kWidth), bias(kDim);
    std::vector<half> packedStates(kBatch * kDim * kHistoryLen);
    uniformFloatInitialization<half>(packedX, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weight, -0.5F, 0.5F);
    uniformFloatInitialization<half>(bias, -0.5F, 0.5F);
    uniformFloatInitialization<half>(packedStates, -0.5F, 0.5F);
    for (int32_t batch = 0; batch < kBatch; ++batch)
    {
        for (int32_t token = 0; token < lengths[batch]; ++token)
        {
            std::copy_n(packedX.begin() + static_cast<int64_t>(cuSeqLens[batch] + token) * kDim, kDim,
                paddedX.begin() + (static_cast<int64_t>(batch) * kMaxSeqLen + token) * kDim);
        }
    }

    std::vector<half> paddedStates(kBatch * kDim * kWidth, __float2half(0.0F));
    for (int32_t batch = 0; batch < kBatch; ++batch)
    {
        for (int32_t channel = 0; channel < kDim; ++channel)
        {
            for (int32_t history = 0; history < kHistoryLen; ++history)
            {
                paddedStates[(static_cast<int64_t>(batch) * kDim + channel) * kWidth + history + 1]
                    = packedStates[(static_cast<int64_t>(batch) * kDim + channel) * kHistoryLen + history];
            }
        }
    }

    std::vector<half> packedOutput;
    runCausalConv1dPackedStep(
        cuSeqLens, {0, 1, 2}, {1, 1, 1}, -1, kDim, kWidth, packedX, weight, &bias, packedStates, &packedOutput);

    auto xDevice = rt::Tensor({kBatch, kMaxSeqLen, kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({kDim, 1, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateDevice = rt::Tensor({kBatch, kDim, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({kBatch, kMaxSeqLen, kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto lengthsDevice = rt::Tensor({kBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(xDevice, paddedX);
    copyHostToDevice(weightDevice, weight);
    copyHostToDevice(biasDevice, bias);
    copyHostToDevice(stateDevice, paddedStates);
    copyHostToDevice(lengthsDevice, lengths);
    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    rt::OptionalInputTensor const stateOpt = std::optional(std::cref(stateDevice));
    rt::OptionalInputTensor const lengthsOpt = std::optional(std::cref(lengthsDevice));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    mamba_ssm::invokeCausalConv1d(
        xDevice, weightDevice, biasOpt, outputDevice, 1, kHistoryLen, 1, stateOpt, lengthsOpt, std::nullopt, stream);
    mamba_ssm::invokeCaptureConvState(xDevice, stateOpt, stateDevice, lengthsOpt, std::nullopt, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    auto const paddedOutput = copyDeviceToHost<half>(outputDevice);
    paddedStates = copyDeviceToHost<half>(stateDevice);

    for (int32_t batch = 0; batch < kBatch; ++batch)
    {
        for (int32_t token = 0; token < lengths[batch]; ++token)
        {
            for (int32_t channel = 0; channel < kDim; ++channel)
            {
                int64_t const packedIdx = static_cast<int64_t>(cuSeqLens[batch] + token) * kDim + channel;
                int64_t const paddedIdx = (static_cast<int64_t>(batch) * kMaxSeqLen + token) * kDim + channel;
                EXPECT_TRUE(isclose(packedOutput[packedIdx], paddedOutput[paddedIdx], 1e-3F, 1e-3F));
            }
        }
        for (int32_t channel = 0; channel < kDim; ++channel)
        {
            for (int32_t history = 0; history < kHistoryLen; ++history)
            {
                int64_t const packedIdx = (static_cast<int64_t>(batch) * kDim + channel) * kHistoryLen + history;
                int64_t const paddedIdx = (static_cast<int64_t>(batch) * kDim + channel) * kWidth + history + 1;
                EXPECT_TRUE(isclose(packedStates[packedIdx], paddedStates[paddedIdx], 1e-3F, 1e-3F));
            }
        }
    }
}

#endif // defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)

// ---------------------------------------------------------------------------
// invokeCaptureConvState tests
// ---------------------------------------------------------------------------

//! New state = the `stateLen` newest entries of `concat(conv state, valid inputs)`.
void runCaptureConvStateReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t stateLen,
    std::vector<half> const& x, std::vector<half>& convStateRef, std::vector<int32_t> const* contextLens = nullptr,
    std::vector<half> const* initialState = nullptr)
{
    std::fill(convStateRef.begin(), convStateRef.end(), __float2half(0.F));
    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLens ? (*contextLens)[b] : seqLen;
        int32_t const tailLen = (cl >= stateLen) ? stateLen : cl;
        int32_t const tailStart = cl - tailLen;
        int32_t const dstOffset = stateLen - tailLen;
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t const stateOffset = (static_cast<int64_t>(b) * dim + d) * stateLen;
            for (int32_t t = 0; t < dstOffset; ++t)
            {
                if (initialState != nullptr)
                {
                    convStateRef[stateOffset + t] = (*initialState)[stateOffset + tailLen + t];
                }
            }
            for (int32_t t = 0; t < tailLen; ++t)
            {
                int64_t const srcIdx = (static_cast<int64_t>(b) * seqLen + tailStart + t) * dim + d;
                int64_t const dstIdx = stateOffset + dstOffset + t;
                convStateRef[dstIdx] = x[srcIdx];
            }
        }
    }
}

void runCaptureConvStateTest(
    int32_t batch, int32_t seqLen, int32_t dim, int32_t width, std::vector<int32_t> const* contextLens = nullptr)
{
    std::vector<half> xHost(batch * seqLen * dim);
    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);

    std::vector<half> convStateRef(batch * dim * width);
    runCaptureConvStateReference(batch, seqLen, dim, width, xHost, convStateRef, contextLens);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(xDevice, xHost);

    rt::OptionalInputTensor clOpt = std::nullopt;
    rt::Tensor clDevice;
    if (contextLens)
    {
        clDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(
            clDevice.rawPointer(), contextLens->data(), contextLens->size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        clOpt = std::optional(std::cref(clDevice));
    }
    mamba_ssm::invokeCaptureConvState(xDevice, std::nullopt, convStateDevice, clOpt, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const convStateHost = copyDeviceToHost<half>(convStateDevice);

    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(convStateHost[i], convStateRef[i], 1e-3F, 1e-3F))
            << "CaptureConvState mismatch at index " << i << ": got " << __half2float(convStateHost[i]) << ", expected "
            << __half2float(convStateRef[i]);
    }
}

TEST(MambaCaptureConvState, SeqGtWidth)
{
    runCaptureConvStateTest(2, 16, 128, 4);
}

TEST(MambaCaptureConvState, SeqEqWidth)
{
    runCaptureConvStateTest(2, 4, 64, 4);
}

TEST(MambaCaptureConvState, SeqLtWidth)
{
    runCaptureConvStateTest(2, 2, 64, 4);
}

TEST(MambaCausalConv1dPadding, MixedContextLengths)
{
    std::vector<int32_t> cl = {5, 16};
    runCausalConv1dTest(2, 16, 128, 4, &cl);
}

TEST(MambaCausalConv1dPadding, ShortContext)
{
    std::vector<int32_t> cl = {2};
    runCausalConv1dTest(1, 16, 64, 4, &cl);
}

TEST(MambaCaptureConvStatePadding, MixedContextLengths)
{
    std::vector<int32_t> cl = {5, 16};
    runCaptureConvStateTest(2, 16, 128, 4, &cl);
}

TEST(MambaCaptureConvStatePadding, ShortContext)
{
    std::vector<int32_t> cl = {2};
    runCaptureConvStateTest(1, 16, 64, 4, &cl);
}

TEST(MambaCausalConv1dPadding, ZeroContextLength)
{
    std::vector<int32_t> const contextLengths{0, 7};
    runCausalConv1dTest(2, 7, 64, 4, &contextLengths);
    runCaptureConvStateTest(2, 7, 64, 4, &contextLengths);
}

TEST(MambaCausalConv1dContinuation, NonzeroState)
{
    constexpr int32_t batch = 2;
    constexpr int32_t seqLen = 7;
    constexpr int32_t dim = 64;
    constexpr int32_t width = 4;
    std::vector<half> xHost(batch * seqLen * dim);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> initialStateHost(batch * dim * width);
    std::vector<half> outputRef(batch * seqLen * dim, __float2half(0.0F));
    std::vector<half> finalStateRef(batch * dim * width, __float2half(0.0F));
    std::vector<int32_t> const contextLengths{2, seqLen};

    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(initialStateHost, -0.5F, 0.5F);
    runCausalConv1dReference(batch, seqLen, dim, width, 1, width, xHost, weightHost, biasHost, outputRef,
        &contextLengths, &initialStateHost);
    runCaptureConvStateReference(batch, seqLen, dim, width, xHost, finalStateRef, &contextLengths, &initialStateHost);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto capturedStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto contextLengthsDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(xDevice, xHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(stateDevice, initialStateHost);
    CUDA_CHECK(cudaMemcpy(contextLengthsDevice.rawPointer(), contextLengths.data(),
        contextLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    rt::OptionalInputTensor const initialStateOpt = std::optional(std::cref(stateDevice));
    rt::OptionalInputTensor const contextLengthsOpt = std::optional(std::cref(contextLengthsDevice));
    mamba_ssm::invokeCausalConv1d(xDevice, weightDevice, biasOpt, outputDevice, 1, width - 1, 1, initialStateOpt,
        contextLengthsOpt, std::nullopt, nullptr);
    mamba_ssm::invokeCaptureConvState(
        xDevice, initialStateOpt, capturedStateDevice, contextLengthsOpt, std::nullopt, nullptr);
    mamba_ssm::invokeCaptureConvState(xDevice, initialStateOpt, stateDevice, contextLengthsOpt, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outputHost = copyDeviceToHost<half>(outputDevice);
    auto const capturedStateHost = copyDeviceToHost<half>(capturedStateDevice);
    auto const finalStateHost = copyDeviceToHost<half>(stateDevice);
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], 1e-3F, 1e-3F))
            << "Continuation output mismatch at index " << i << ": got " << __half2float(outputHost[i]) << ", expected "
            << __half2float(outputRef[i]);
    }
    for (size_t i = 0; i < finalStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(capturedStateHost[i], finalStateRef[i], 1e-3F, 1e-3F))
            << "Separate continuation state mismatch at index " << i << ": got " << __half2float(capturedStateHost[i])
            << ", expected " << __half2float(finalStateRef[i]);
        EXPECT_TRUE(isclose(finalStateHost[i], finalStateRef[i], 1e-3F, 1e-3F))
            << "Continuation state mismatch at index " << i << ": got " << __half2float(finalStateHost[i])
            << ", expected " << __half2float(finalStateRef[i]);
    }
}

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecode tests
// ---------------------------------------------------------------------------

//! Tap `k` reads the pre-shift row at `stateLen - dilation * (width - 1 - k)`; tap `width - 1` is
//! the incoming token, which the shift then appends.
void runCausalConv1dDecodeReference(int32_t batch, int32_t dim, int32_t width, int32_t dilation, int32_t stateLen,
    std::vector<half> const& convState, std::vector<half> const& newCol, std::vector<half> const& weight,
    std::vector<half> const& bias, std::vector<half>& convStateOut, std::vector<half>& outRef)
{
    convStateOut = convState;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t rowOff = (static_cast<int64_t>(b) * dim + d) * stateLen;
            half const newVal = newCol[static_cast<int64_t>(b) * dim + d];
            float acc = __half2float(bias[d]);
            for (int32_t k = 0; k < width - 1; ++k)
            {
                int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                acc += __half2float(convState[rowOff + stateLen - dilation * (width - 1 - k)])
                    * __half2float(weight[wIdx]);
            }
            acc += __half2float(newVal) * __half2float(weight[d * width + width - 1]);
            for (int32_t k = 0; k < stateLen - 1; ++k)
            {
                convStateOut[rowOff + k] = convStateOut[rowOff + k + 1];
            }
            convStateOut[rowOff + stateLen - 1] = newVal;
            int64_t const outIdx = static_cast<int64_t>(b) * dim + d;
            outRef[outIdx] = __float2half(acc);
        }
    }
}

void runCausalConv1dDecodeTest(int32_t batch, int32_t dim, int32_t width)
{
    std::vector<half> convStateHost(batch * dim * width);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColHost(batch * dim);
    uniformFloatInitialization<half>(convStateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColHost, -0.5F, 0.5F);

    std::vector<half> convStateRef(convStateHost.size());
    std::vector<half> outRef(batch * dim);
    runCausalConv1dDecodeReference(
        batch, dim, width, 1, width, convStateHost, newColHost, weightHost, biasHost, convStateRef, outRef);

    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColDevice = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(convStateDevice, convStateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColDevice, newColHost);

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecode(
        convStateDevice, newColDevice, weightDevice, biasOpt, outDevice, 1, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outHost = copyDeviceToHost<half>(outDevice);

    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "CausalConv1dDecode output mismatch at index " << i << ": got " << __half2float(outHost[i])
            << ", expected " << __half2float(outRef[i]);
    }

    auto const stateHost = copyDeviceToHost<half>(convStateDevice);

    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(stateHost[i], convStateRef[i], 1e-3F, 1e-3F))
            << "CausalConv1dDecode state mismatch at index " << i << ": got " << __half2float(stateHost[i])
            << ", expected " << __half2float(convStateRef[i]);
    }
}

TEST(MambaCausalConv1dDecode, Width2)
{
    runCausalConv1dDecodeTest(2, 128, 2);
}

TEST(MambaCausalConv1dDecode, Width4)
{
    runCausalConv1dDecodeTest(2, 256, 4);
}

TEST(MambaCausalConv1dDecode, LargeDim)
{
    runCausalConv1dDecodeTest(4, 512, 4);
}

TEST(MambaCausalConv1dResidentState, DecodeUsesNonContiguousSlots)
{
    int32_t constexpr batch = 2;
    int32_t constexpr poolRows = 4;
    int32_t constexpr dim = 8;
    int32_t constexpr width = 4;
    std::vector<int32_t> const stateIndices{2, 0};
    std::vector<half> stateHost(static_cast<size_t>(poolRows) * dim * width);
    std::vector<half> newColHost(static_cast<size_t>(batch) * dim);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width, __float2half(0.0F));
    std::vector<half> biasHost(dim, __float2half(0.0F));
    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        std::fill_n(
            stateHost.begin() + static_cast<size_t>(slot) * dim * width, dim * width, __float2half(10.0F + slot));
    }
    for (int32_t row = 0; row < batch; ++row)
    {
        std::fill_n(newColHost.begin() + static_cast<size_t>(row) * dim, dim, __float2half(100.0F + row));
    }

    auto state = rt::Tensor({poolRows, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newCol = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weight = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto bias = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto output = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto indices = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(state, stateHost);
    copyHostToDevice(newCol, newColHost);
    copyHostToDevice(weight, weightHost);
    copyHostToDevice(bias, biasHost);
    copyHostToDevice(indices, stateIndices);

    rt::OptionalInputTensor biasOpt = std::optional(std::cref(bias));
    rt::OptionalInputTensor indicesOpt = std::optional(std::cref(indices));
    mamba_ssm::invokeCausalConv1dDecode(state, newCol, weight, biasOpt, output, 1, indicesOpt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const actual = copyDeviceToHost<half>(state);

    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            for (int32_t k = 0; k < width; ++k)
            {
                float expected = 10.0F + slot;
                auto const found = std::find(stateIndices.begin(), stateIndices.end(), slot);
                if (found != stateIndices.end() && k == width - 1)
                {
                    expected = 100.0F + std::distance(stateIndices.begin(), found);
                }
                size_t const offset = (static_cast<size_t>(slot) * dim + d) * width + k;
                EXPECT_EQ(__half2float(actual[offset]), expected) << "slot=" << slot << ", d=" << d << ", k=" << k;
            }
        }
    }
}

TEST(MambaCausalConv1dResidentState, PaddedPrefillCapturesFinalValidTokenBySlot)
{
    int32_t constexpr batch = 3;
    int32_t constexpr poolRows = 5;
    int32_t constexpr seqLen = 4;
    int32_t constexpr dim = 8;
    int32_t constexpr width = 4;
    std::vector<int32_t> const stateIndices{3, 1, 4};
    std::vector<int32_t> const queryLengths{4, 2, 3};
    std::vector<half> stateHost(static_cast<size_t>(poolRows) * dim * width);
    std::vector<half> xHost(static_cast<size_t>(batch) * seqLen * dim);
    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        std::fill_n(
            stateHost.begin() + static_cast<size_t>(slot) * dim * width, dim * width, __float2half(10.0F + slot));
    }
    for (int32_t row = 0; row < batch; ++row)
    {
        for (int32_t token = 0; token < seqLen; ++token)
        {
            std::fill_n(xHost.begin() + (static_cast<size_t>(row) * seqLen + token) * dim, dim,
                __float2half(100.0F + row * 10.0F + token));
        }
    }

    auto x = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto state = rt::Tensor({poolRows, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto lengths = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    auto indices = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    auto weight = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto bias = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto output = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width, __float2half(0.0F));
    std::vector<half> biasHost(dim, __float2half(0.0F));
    for (int32_t d = 0; d < dim; ++d)
    {
        weightHost[static_cast<size_t>(d) * width] = __float2half(1.0F);
    }
    copyHostToDevice(x, xHost);
    copyHostToDevice(state, stateHost);
    copyHostToDevice(lengths, queryLengths);
    copyHostToDevice(indices, stateIndices);
    copyHostToDevice(weight, weightHost);
    copyHostToDevice(bias, biasHost);

    rt::OptionalInputTensor stateOpt = std::optional(std::cref(state));
    rt::OptionalInputTensor lengthsOpt = std::optional(std::cref(lengths));
    rt::OptionalInputTensor indicesOpt = std::optional(std::cref(indices));
    rt::OptionalInputTensor biasOpt = std::optional(std::cref(bias));
    mamba_ssm::invokeCausalConv1d(
        x, weight, biasOpt, output, 1, width - 1, 1, stateOpt, lengthsOpt, indicesOpt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const outputHost = copyDeviceToHost<half>(output);
    for (int32_t row = 0; row < batch; ++row)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            EXPECT_EQ(
                __half2float(outputHost[(static_cast<size_t>(row) * seqLen) * dim + d]), 10.0F + stateIndices[row]);
        }
    }

    mamba_ssm::invokeCaptureConvState(x, stateOpt, state, lengthsOpt, indicesOpt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const actual = copyDeviceToHost<half>(state);

    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        auto const found = std::find(stateIndices.begin(), stateIndices.end(), slot);
        int32_t const row
            = found == stateIndices.end() ? -1 : static_cast<int32_t>(std::distance(stateIndices.begin(), found));
        for (int32_t d = 0; d < dim; ++d)
        {
            for (int32_t k = 0; k < width; ++k)
            {
                float expected = 10.0F + slot;
                if (row >= 0)
                {
                    int32_t const valid = queryLengths[row];
                    expected = k < width - valid ? 10.0F + slot : 100.0F + row * 10.0F + k - (width - valid);
                }
                size_t const offset = (static_cast<size_t>(slot) * dim + d) * width + k;
                EXPECT_EQ(__half2float(actual[offset]), expected) << "slot=" << slot << ", d=" << d << ", k=" << k;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecodeMTP tests
// ---------------------------------------------------------------------------

/**
 * CPU reference for MTP decode: processes T tokens sequentially, with per-step
 * state checkpointing. Verifies output, final state, and intermediate states.
 */
void runCausalConv1dDecodeMTPReference(int32_t batch, int32_t dim, int32_t width, int32_t dilation, int32_t stateLen,
    int32_t T, std::vector<half> const& convState, std::vector<half> const& newCols, std::vector<half> const& weight,
    std::vector<half> const& bias, std::vector<half>& convStateOut, std::vector<half>& outRef,
    std::vector<half>& intermRef)
{
    convStateOut = convState;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t rowOff = (static_cast<int64_t>(b) * dim + d) * stateLen;
            for (int32_t t = 0; t < T; ++t)
            {
                int64_t const newIdx = (static_cast<int64_t>(b) * T + t) * dim + d;
                half const newVal = newCols[newIdx];
                // Dot product against the pre-shift row plus the incoming token.
                float acc = __half2float(bias[d]);
                for (int32_t k = 0; k < width - 1; ++k)
                {
                    int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                    acc += __half2float(convStateOut[rowOff + stateLen - dilation * (width - 1 - k)])
                        * __half2float(weight[wIdx]);
                }
                acc += __half2float(newVal) * __half2float(weight[d * width + width - 1]);
                int64_t const outIdx = (static_cast<int64_t>(b) * T + t) * dim + d;
                outRef[outIdx] = __float2half(acc);
                // Shift left and insert the new token.
                for (int32_t k = 0; k < stateLen - 1; ++k)
                {
                    convStateOut[rowOff + k] = convStateOut[rowOff + k + 1];
                }
                convStateOut[rowOff + stateLen - 1] = newVal;
                // Checkpoint intermediate state
                int64_t const intermBase = ((static_cast<int64_t>(b) * T + t) * dim + d) * stateLen;
                for (int32_t k = 0; k < stateLen; ++k)
                {
                    intermRef[intermBase + k] = convStateOut[rowOff + k];
                }
            }
        }
    }
}

void runCausalConv1dDecodeMTPTest(int32_t batch, int32_t dim, int32_t width, int32_t T)
{
    std::vector<half> convStateHost(batch * dim * width);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColsHost(batch * T * dim);
    uniformFloatInitialization<half>(convStateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);

    // CPU reference
    std::vector<half> convStateRef(convStateHost.size());
    std::vector<half> outRef(batch * T * dim);
    std::vector<half> intermRef(batch * T * dim * width);
    runCausalConv1dDecodeMTPReference(batch, dim, width, 1, width, T, convStateHost, newColsHost, weightHost, biasHost,
        convStateRef, outRef, intermRef);

    // GPU
    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColsDevice = rt::Tensor({batch, T, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, T, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto intermDevice = rt::Tensor({batch, T, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(convStateDevice, convStateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColsDevice, newColsHost);

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecodeMTP(
        convStateDevice, newColsDevice, weightDevice, biasOpt, outDevice, intermDevice, T, 1, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify output
    auto const outHost = copyDeviceToHost<half>(outDevice);
    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "MTP output mismatch at index " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outRef[i]);
    }

    // Target verification must not commit speculative state.
    auto const stateHost = copyDeviceToHost<half>(convStateDevice);
    for (size_t i = 0; i < convStateHost.size(); ++i)
    {
        EXPECT_TRUE(isclose(stateHost[i], convStateHost[i], 1e-3F, 1e-3F))
            << "MTP committed state changed at index " << i;
    }

    // Verify intermediate states
    auto const intermHost = copyDeviceToHost<half>(intermDevice);
    for (size_t i = 0; i < intermRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(intermHost[i], intermRef[i], 1e-3F, 1e-3F))
            << "MTP intermediate state mismatch at index " << i << ": got " << __half2float(intermHost[i])
            << ", expected " << __half2float(intermRef[i]);
    }
}

TEST(MambaCausalConv1dDecodeMTP, Width4_T4)
{
    runCausalConv1dDecodeMTPTest(/*batch=*/4, /*dim=*/256, /*width=*/4, /*T=*/4);
}

TEST(MambaCausalConv1dDecodeMTP, Width2_T8)
{
    runCausalConv1dDecodeMTPTest(/*batch=*/2, /*dim=*/128, /*width=*/2, /*T=*/8);
}

TEST(MambaCausalConv1dDecodeMTP, LargeDim)
{
    runCausalConv1dDecodeMTPTest(/*batch=*/4, /*dim=*/512, /*width=*/4, /*T=*/4);
}

TEST(MambaCausalConv1dDecodeMTP, AccumulatesTapsInChronologicalOrder)
{
    // Given a rounding-sensitive FP16 window, when chain verification evaluates one token, then its
    // accumulation order matches ordinary decode and tree replay from the oldest tap to the incoming token.
    std::vector<half> const stateHost{__float2half(0.0F), __float2half(0.059173583984375F),
        __float2half(0.169189453125F), __float2half(0.14697265625F)};
    std::vector<half> const newColsHost{__float2half(0.294677734375F)};
    std::vector<half> const weightHost{__float2half(0.00064849853515625F), __float2half(-0.0012969970703125F),
        __float2half(0.0390625F), __float2half(-0.00775146484375F)};
    std::vector<half> const biasHost{__float2half(0.0F)};
    auto state = rt::Tensor({1, 1, 4}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newCols = rt::Tensor({1, 1, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weight = rt::Tensor({1, 1, 4}, rt::DeviceType::kGPU, DataType::kHALF);
    auto bias = rt::Tensor({1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto output = rt::Tensor({1, 1, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto intermediate = rt::Tensor({1, 1, 1, 4}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(state, stateHost);
    copyHostToDevice(newCols, newColsHost);
    copyHostToDevice(weight, weightHost);
    copyHostToDevice(bias, biasHost);

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(bias));
    mamba_ssm::invokeCausalConv1dDecodeMTP(
        state, newCols, weight, biasOpt, output, intermediate, 1, 1, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outputHost = copyDeviceToHost<half>(output);
    EXPECT_EQ(__half2float(outputHost[0]), 0.003276824951171875F);
}

TEST(MambaCausalConv1dDecodeMTP, ParallelChainMatchesSequentialVerification)
{
    // Given a B1/Q8 width-four chain in a non-contiguous resident slot, when sequential and
    // node-parallel verification run, then every FP16 output/checkpoint bit and resident state match.
    constexpr int32_t kStateRows{2};
    constexpr int32_t kBatch{1};
    constexpr int32_t kTokens{8};
    constexpr int32_t kDim{257};
    constexpr int32_t kWidth{4};
    std::vector<half> stateHost(kStateRows * kDim * kWidth);
    std::vector<half> newColsHost(kBatch * kTokens * kDim);
    std::vector<half> weightHost(kDim * kWidth);
    std::vector<half> biasHost(kDim);
    uniformFloatInitialization<half>(stateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    size_t const witnessStateOffset = static_cast<size_t>(kDim) * kWidth;
    stateHost[witnessStateOffset + 0] = __float2half(0.0F);
    stateHost[witnessStateOffset + 1] = __float2half(0.059173583984375F);
    stateHost[witnessStateOffset + 2] = __float2half(0.169189453125F);
    stateHost[witnessStateOffset + 3] = __float2half(0.14697265625F);
    newColsHost[0] = __float2half(0.294677734375F);
    weightHost[0] = __float2half(0.00064849853515625F);
    weightHost[1] = __float2half(-0.0012969970703125F);
    weightHost[2] = __float2half(0.0390625F);
    weightHost[3] = __float2half(-0.00775146484375F);
    biasHost[0] = __float2half(0.0F);

    auto state = rt::Tensor({kStateRows, kDim, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newCols = rt::Tensor({kBatch, kTokens, kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weight = rt::Tensor({kDim, 1, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto bias = rt::Tensor({kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto sequentialOutput = rt::Tensor({kBatch, kTokens, kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto parallelOutput = rt::Tensor({kBatch, kTokens, kDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto sequentialStates = rt::Tensor({kBatch, kTokens, kDim, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto parallelStates = rt::Tensor({kBatch, kTokens, kDim, kWidth}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateIndices = rt::Tensor({kBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    std::vector<int32_t> const stateIndicesHost{1};
    copyHostToDevice(state, stateHost);
    copyHostToDevice(newCols, newColsHost);
    copyHostToDevice(weight, weightHost);
    copyHostToDevice(bias, biasHost);
    copyHostToDevice(stateIndices, stateIndicesHost);

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(bias));
    rt::OptionalInputTensor const stateIndicesOpt = std::optional(std::cref(stateIndices));
    mamba_ssm::invokeCausalConv1dDecodeMTP(
        state, newCols, weight, biasOpt, sequentialOutput, sequentialStates, kTokens, 1, stateIndicesOpt, nullptr);
    mamba_ssm::invokeCausalConv1dDecodeChainParallel(
        state, newCols, weight, biasOpt, parallelOutput, parallelStates, stateIndicesOpt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const parallelOutputHost = copyDeviceToHost<half>(parallelOutput);
    auto const sequentialOutputHost = copyDeviceToHost<half>(sequentialOutput);
    auto const parallelStatesHost = copyDeviceToHost<half>(parallelStates);
    auto const sequentialStatesHost = copyDeviceToHost<half>(sequentialStates);
    auto const finalStateHost = copyDeviceToHost<half>(state);
    EXPECT_EQ(
        std::memcmp(parallelOutputHost.data(), sequentialOutputHost.data(), parallelOutputHost.size() * sizeof(half)),
        0);
    EXPECT_EQ(
        std::memcmp(parallelStatesHost.data(), sequentialStatesHost.data(), parallelStatesHost.size() * sizeof(half)),
        0);
    EXPECT_EQ(std::memcmp(finalStateHost.data(), stateHost.data(), finalStateHost.size() * sizeof(half)), 0);
}

TEST(MambaCausalConv1dDecodeMTP, ParallelChainDispatchIsLimitedToProfiledShape)
{
    // Given the profiled SM100 shape and nearby variants, when dispatch is selected, then only B1/Q8/W4 uses it.
    struct DispatchCase
    {
        int32_t sm;
        int32_t batch;
        int32_t tokens;
        int32_t width;
        int32_t dilation;
        int32_t stateLen;
        int32_t stride;
        bool expected;
    };
    std::vector<DispatchCase> const cases{{100, 1, 8, 4, 1, 4, 1, true}, {110, 1, 8, 4, 1, 4, 1, false},
        {100, 2, 8, 4, 1, 4, 1, false}, {100, 1, 7, 4, 1, 4, 1, false}, {100, 1, 8, 3, 1, 3, 1, false},
        {100, 1, 8, 4, 3, 9, 1, false}, {100, 1, 8, 4, 1, 4, 2, false}};
    for (DispatchCase const& testCase : cases)
    {
        EXPECT_EQ(mamba_ssm::canUseCausalConv1dDecodeChainParallel(testCase.sm, testCase.batch, testCase.tokens,
                      testCase.width, testCase.dilation, testCase.stateLen, testCase.stride),
            testCase.expected);
    }
}

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecodeDDTree tests
// ---------------------------------------------------------------------------

//! Root-to-node token order, walking parent ids without the kernel's window truncation.
std::vector<int32_t> ancestorsToRoot(
    std::vector<int32_t> const& parentIds, int32_t batchIdx, int32_t verifySeq, int32_t node)
{
    std::vector<int32_t> path;
    for (int32_t current = node; current >= 0 && current < verifySeq;)
    {
        path.push_back(current);
        if (current == 0)
        {
            break;
        }
        current = parentIds[batchIdx * verifySeq + current];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

//! Direct transcription of `out[t] = sum_k w[k] * in[S + t - dilation * (width - 1 - k)]` over the
//! node's own virtual sequence, with no window or path truncation, so it shares no indexing
//! algorithm with the kernel.
void runCausalConv1dDecodeDDTreeReference(int32_t batch, int32_t dim, int32_t width, int32_t dilation, int32_t stateLen,
    int32_t verifySeq, std::vector<half> const& convState, std::vector<half> const& newCols,
    std::vector<half> const& weight, std::vector<half> const& bias, bool hasBias, std::vector<int32_t> const& parentIds,
    std::vector<int32_t> const& depths, std::vector<half>& convStateOut, std::vector<half>& outRef,
    std::vector<half>& intermRef)
{
    convStateOut = convState;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t node = 0; node < verifySeq; ++node)
        {
            int32_t const parent = parentIds[b * verifySeq + node];
            int32_t const depth = depths[b * verifySeq + node];
            bool const isRoot = node == 0 && parent < 0 && depth == 0;
            bool const isValidChild = node > 0 && parent >= 0 && parent < node && depth > 0;
            bool const isValidNode = isRoot || isValidChild;

            for (int32_t d = 0; d < dim; ++d)
            {
                int64_t const rowOff = (static_cast<int64_t>(b) * dim + d) * stateLen;
                int64_t const intermBase = ((static_cast<int64_t>(b) * verifySeq + node) * dim + d) * stateLen;
                if (!isValidNode)
                {
                    outRef[(static_cast<int64_t>(b) * verifySeq + node) * dim + d] = __float2half(0.0F);
                    for (int32_t k = 0; k < stateLen; ++k)
                    {
                        intermRef[intermBase + k] = convState[rowOff + k];
                    }
                    continue;
                }

                // Whole state row followed by the untruncated root-to-node path, in chronological
                // order: the virtual sequence the formula is written against.
                std::vector<float> virt;
                for (int32_t k = 0; k < stateLen; ++k)
                {
                    virt.push_back(__half2float(convState[rowOff + k]));
                }
                for (int32_t pathNode : ancestorsToRoot(parentIds, b, verifySeq, node))
                {
                    virt.push_back(__half2float(newCols[(static_cast<int64_t>(b) * verifySeq + pathNode) * dim + d]));
                }
                int32_t const virtLen = static_cast<int32_t>(virt.size());

                float acc = hasBias ? __half2float(bias[d]) : 0.0F;
                for (int32_t k = 0; k < width; ++k)
                {
                    acc += virt[virtLen - 1 - dilation * (width - 1 - k)]
                        * __half2float(weight[static_cast<int64_t>(d) * width + k]);
                }
                outRef[(static_cast<int64_t>(b) * verifySeq + node) * dim + d] = __float2half(acc);

                for (int32_t k = 0; k < stateLen; ++k)
                {
                    intermRef[intermBase + k] = __float2half(virt[virtLen - stateLen + k]);
                }
            }
        }
    }
}

using DDTreeTestParams = std::tuple<int32_t, int32_t, bool, int32_t>;

class MambaCausalConv1dDecodeDDTreeTest : public ::testing::TestWithParam<DDTreeTestParams>
{
};

TEST_P(MambaCausalConv1dDecodeDDTreeTest, RootToNodeState)
{
    constexpr int32_t batch = 2;
    auto const [verifySeq, width, hasBias, dim] = GetParam();

    std::vector<half> convStateHost(batch * dim * width);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColsHost(batch * verifySeq * dim);
    uniformFloatInitialization<half>(convStateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);

    std::vector<int32_t> parentIds(static_cast<size_t>(batch) * verifySeq, -1);
    std::vector<int32_t> depths(static_cast<size_t>(batch) * verifySeq, 0);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t node = 1; node < verifySeq - 1; ++node)
        {
            int32_t const parent = (b == 0) ? ((node == 3) ? 1 : node - 1) : ((node % 2 == 0) ? 0 : node - 1);
            size_t const idx = static_cast<size_t>(b) * verifySeq + node;
            parentIds[idx] = parent;
            depths[idx] = depths[static_cast<size_t>(b) * verifySeq + parent] + 1;
        }
    }

    std::vector<half> convStateRef(convStateHost.size());
    std::vector<half> outRef(batch * verifySeq * dim);
    std::vector<half> intermRef(batch * verifySeq * dim * width);
    runCausalConv1dDecodeDDTreeReference(batch, dim, width, 1, width, verifySeq, convStateHost, newColsHost, weightHost,
        biasHost, hasBias, parentIds, depths, convStateRef, outRef, intermRef);

    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateOutDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColsDevice = rt::Tensor({batch, verifySeq, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, verifySeq, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto intermDevice = rt::Tensor({batch, verifySeq, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto parentDevice = rt::Tensor({batch, verifySeq}, rt::DeviceType::kGPU, DataType::kINT32);
    auto depthDevice = rt::Tensor({batch, verifySeq}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(convStateDevice, convStateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColsDevice, newColsHost);
    CUDA_CHECK(cudaMemcpy(
        parentDevice.rawPointer(), parentIds.data(), parentIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(depthDevice.rawPointer(), depths.data(), depths.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::nullopt;
    if (hasBias)
    {
        biasOpt = std::optional(std::cref(biasDevice));
    }
    mamba_ssm::invokeCausalConv1dDecodeDDTree(convStateDevice, newColsDevice, weightDevice, biasOpt, outDevice,
        convStateOutDevice, intermDevice, parentDevice, depthDevice, 1, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outHost = copyDeviceToHost<half>(outDevice);
    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "DDTree output mismatch at index " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outRef[i]);
    }

    auto const stateHost = copyDeviceToHost<half>(convStateDevice);
    for (size_t i = 0; i < convStateHost.size(); ++i)
    {
        EXPECT_TRUE(isclose(stateHost[i], convStateHost[i], 1e-3F, 1e-3F))
            << "DDTree committed state changed at index " << i;
    }

    auto const intermHost = copyDeviceToHost<half>(intermDevice);
    for (size_t i = 0; i < intermRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(intermHost[i], intermRef[i], 1e-3F, 1e-3F))
            << "DDTree intermediate state mismatch at index " << i << ": got " << __half2float(intermHost[i])
            << ", expected " << __half2float(intermRef[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(Configurations, MambaCausalConv1dDecodeDDTreeTest,
    ::testing::Values(DDTreeTestParams{6, 4, true, 64}, DDTreeTestParams{7, 4, true, 64},
        DDTreeTestParams{17, 4, true, 64}, DDTreeTestParams{7, 3, true, 64}, DDTreeTestParams{7, 4, false, 64},
        DDTreeTestParams{7, 3, false, 64}, DDTreeTestParams{7, 8, true, 64}, DDTreeTestParams{7, 4, true, 257},
        DDTreeTestParams{7, 3, true, 257}, DDTreeTestParams{1, 4, true, 64}));

// ---------------------------------------------------------------------------
// Dilated shapes: (width, dilation, state length) across every entry point
// ---------------------------------------------------------------------------

namespace
{

struct DilatedShape
{
    int32_t width;
    int32_t dilation;
    int32_t stateLen;
};

std::string dilatedShapeName(::testing::TestParamInfo<DilatedShape> const& info)
{
    return "W" + std::to_string(info.param.width) + "D" + std::to_string(info.param.dilation) + "S"
        + std::to_string(info.param.stateLen);
}

//! Reports only the first mismatch: the real-size cases compare millions of elements.
void expectAllClose(char const* what, std::vector<half> const& expected, std::vector<half> const& actual)
{
    ASSERT_EQ(expected.size(), actual.size()) << what << " size mismatch";
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (!isclose(actual[i], expected[i], 1e-3F, 1e-3F))
        {
            ADD_FAILURE() << what << " mismatch at index " << i << ": got " << __half2float(actual[i]) << ", expected "
                          << __half2float(expected[i]);
            return;
        }
    }
}

//! Deep chain with one branch for row 0, shallow bushy tree for the others, trailing padding node.
void buildVerifyTree(int32_t batch, int32_t verifySeq, std::vector<int32_t>& parentIds, std::vector<int32_t>& depths)
{
    parentIds.assign(static_cast<size_t>(batch) * verifySeq, -1);
    depths.assign(static_cast<size_t>(batch) * verifySeq, 0);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t node = 1; node < verifySeq - 1; ++node)
        {
            int32_t const parent = (b == 0) ? ((node == 3) ? 1 : node - 1) : ((node % 2 == 0) ? 0 : node - 1);
            size_t const idx = static_cast<size_t>(b) * verifySeq + node;
            parentIds[idx] = parent;
            depths[idx] = depths[static_cast<size_t>(b) * verifySeq + parent] + 1;
        }
    }
}

void runDilatedPrefillTest(DilatedShape const& shape, int32_t batch, int32_t seqLen, int32_t dim,
    std::vector<int32_t> const* contextLens, bool withInitialState)
{
    int32_t const width = shape.width;
    int32_t const stateLen = shape.stateLen;

    std::vector<half> xHost(static_cast<size_t>(batch) * seqLen * dim);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width);
    std::vector<half> biasHost(dim);
    std::vector<half> stateHost(static_cast<size_t>(batch) * dim * stateLen);
    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(stateHost, -0.5F, 0.5F);

    std::vector<half> const* const refState = withInitialState ? &stateHost : nullptr;
    std::vector<half> outRef(xHost.size(), __float2half(0.0F));
    std::vector<half> stateRef(stateHost.size(), __float2half(0.0F));
    runCausalConv1dReference(batch, seqLen, dim, width, shape.dilation, stateLen, xHost, weightHost, biasHost, outRef,
        contextLens, refState);
    runCaptureConvStateReference(batch, seqLen, dim, stateLen, xHost, stateRef, contextLens, refState);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto capturedDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(xDevice, xHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(stateDevice, stateHost);

    rt::Tensor clDevice;
    rt::OptionalInputTensor clOpt = std::nullopt;
    if (contextLens != nullptr)
    {
        clDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(clDevice, *contextLens);
        clOpt = std::optional(std::cref(clDevice));
    }
    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    rt::OptionalInputTensor stateOpt = std::nullopt;
    if (withInitialState)
    {
        stateOpt = std::optional(std::cref(stateDevice));
    }

    mamba_ssm::invokeCausalConv1d(xDevice, weightDevice, biasOpt, outDevice, 1, (width - 1) * shape.dilation,
        shape.dilation, stateOpt, clOpt, std::nullopt, nullptr);
    mamba_ssm::invokeCaptureConvState(xDevice, stateOpt, capturedDevice, clOpt, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    expectAllClose("dilated prefill output", outRef, copyDeviceToHost<half>(outDevice));
    expectAllClose("dilated captured state", stateRef, copyDeviceToHost<half>(capturedDevice));
}

void runDilatedDecodeTest(DilatedShape const& shape, int32_t batch, int32_t dim)
{
    int32_t const width = shape.width;
    int32_t const stateLen = shape.stateLen;

    std::vector<half> stateHost(static_cast<size_t>(batch) * dim * stateLen);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColHost(static_cast<size_t>(batch) * dim);
    uniformFloatInitialization<half>(stateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColHost, -0.5F, 0.5F);

    std::vector<half> stateRef(stateHost.size());
    std::vector<half> outRef(newColHost.size());
    runCausalConv1dDecodeReference(
        batch, dim, width, shape.dilation, stateLen, stateHost, newColHost, weightHost, biasHost, stateRef, outRef);

    auto stateDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColDevice = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(stateDevice, stateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColDevice, newColHost);

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecode(
        stateDevice, newColDevice, weightDevice, biasOpt, outDevice, shape.dilation, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    expectAllClose("dilated decode output", outRef, copyDeviceToHost<half>(outDevice));
    expectAllClose("dilated decode state", stateRef, copyDeviceToHost<half>(stateDevice));
}

void runDilatedMTPTest(DilatedShape const& shape, int32_t batch, int32_t dim, int32_t numTokens)
{
    int32_t const width = shape.width;
    int32_t const stateLen = shape.stateLen;

    std::vector<half> stateHost(static_cast<size_t>(batch) * dim * stateLen);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColsHost(static_cast<size_t>(batch) * numTokens * dim);
    uniformFloatInitialization<half>(stateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);

    std::vector<half> stateRef(stateHost.size());
    std::vector<half> outRef(newColsHost.size());
    std::vector<half> intermRef(static_cast<size_t>(batch) * numTokens * dim * stateLen);
    runCausalConv1dDecodeMTPReference(batch, dim, width, shape.dilation, stateLen, numTokens, stateHost, newColsHost,
        weightHost, biasHost, stateRef, outRef, intermRef);

    auto stateDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColsDevice = rt::Tensor({batch, numTokens, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, numTokens, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto intermDevice = rt::Tensor({batch, numTokens, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(stateDevice, stateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColsDevice, newColsHost);

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecodeMTP(stateDevice, newColsDevice, weightDevice, biasOpt, outDevice, intermDevice,
        numTokens, shape.dilation, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    expectAllClose("dilated MTP output", outRef, copyDeviceToHost<half>(outDevice));
    expectAllClose("dilated MTP checkpoints", intermRef, copyDeviceToHost<half>(intermDevice));
    // Target verification must not commit speculative state.
    expectAllClose("dilated MTP committed state", stateHost, copyDeviceToHost<half>(stateDevice));
}

void runDilatedDDTreeTest(DilatedShape const& shape, int32_t batch, int32_t dim, int32_t verifySeq)
{
    int32_t const width = shape.width;
    int32_t const stateLen = shape.stateLen;

    std::vector<half> stateHost(static_cast<size_t>(batch) * dim * stateLen);
    std::vector<half> weightHost(static_cast<size_t>(dim) * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColsHost(static_cast<size_t>(batch) * verifySeq * dim);
    uniformFloatInitialization<half>(stateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);

    std::vector<int32_t> parentIds;
    std::vector<int32_t> depths;
    buildVerifyTree(batch, verifySeq, parentIds, depths);

    std::vector<half> stateRef(stateHost.size());
    std::vector<half> outRef(newColsHost.size());
    std::vector<half> intermRef(static_cast<size_t>(batch) * verifySeq * dim * stateLen);
    runCausalConv1dDecodeDDTreeReference(batch, dim, width, shape.dilation, stateLen, verifySeq, stateHost, newColsHost,
        weightHost, biasHost, /* hasBias = */ true, parentIds, depths, stateRef, outRef, intermRef);

    auto stateDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateOutDevice = rt::Tensor({batch, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newColsDevice = rt::Tensor({batch, verifySeq, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, verifySeq, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto intermDevice = rt::Tensor({batch, verifySeq, dim, stateLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto parentDevice = rt::Tensor({batch, verifySeq}, rt::DeviceType::kGPU, DataType::kINT32);
    auto depthDevice = rt::Tensor({batch, verifySeq}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(stateDevice, stateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    copyHostToDevice(newColsDevice, newColsHost);
    copyHostToDevice(parentDevice, parentIds);
    copyHostToDevice(depthDevice, depths);

    rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecodeDDTree(stateDevice, newColsDevice, weightDevice, biasOpt, outDevice,
        stateOutDevice, intermDevice, parentDevice, depthDevice, shape.dilation, std::nullopt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    expectAllClose("dilated DDTree output", outRef, copyDeviceToHost<half>(outDevice));
    expectAllClose("dilated DDTree checkpoints", intermRef, copyDeviceToHost<half>(intermDevice));
    expectAllClose("dilated DDTree committed state", stateHost, copyDeviceToHost<half>(stateDevice));
}

} // namespace

class MambaDilatedCausalConv1d : public ::testing::TestWithParam<DilatedShape>
{
};

TEST_P(MambaDilatedCausalConv1d, PrefillWithStateAndRaggedRows)
{
    std::vector<int32_t> const contextLengths{0, 3, 24};
    runDilatedPrefillTest(GetParam(), 3, 24, 96, &contextLengths, /* withInitialState = */ true);
}

TEST_P(MambaDilatedCausalConv1d, PrefillWithoutState)
{
    runDilatedPrefillTest(GetParam(), 2, 17, 96, nullptr, /* withInitialState = */ false);
}

TEST_P(MambaDilatedCausalConv1d, Decode)
{
    runDilatedDecodeTest(GetParam(), 3, 96);
}

TEST_P(MambaDilatedCausalConv1d, DecodeMTP)
{
    runDilatedMTPTest(GetParam(), 2, 96, /* numTokens = */ 5);
}

TEST_P(MambaDilatedCausalConv1d, DecodeDDTree)
{
    runDilatedDDTreeTest(GetParam(), 2, 96, /* verifySeq = */ 16);
}

// One shape per distinguishing path rather than the full (width, dilation) cross product:
// state_len == (width-1)*dilation is the tight HuggingFace/vLLM convention, where the oldest tap is
// the column decode evicts and the DDTree window outgrows the state row; state_len > (width-1)*dilation
// leaves that tap inside the row and must ignore the surplus oldest columns. W4D1S4 is the Mamba/GDN
// convention, W4D3S9 the only specialized dilated instance, W4D4S12 the widest receptive field.
INSTANTIATE_TEST_SUITE_P(Shapes, MambaDilatedCausalConv1d,
    ::testing::Values(DilatedShape{2, 1, 1}, DilatedShape{3, 2, 4}, DilatedShape{4, 1, 4}, DilatedShape{4, 3, 9},
        DilatedShape{4, 3, 10}, DilatedShape{4, 4, 12}),
    dilatedShapeName);

// Qwen3.8-Flash-Next n-gram short convolution at its deployed channel count.
TEST(MambaDilatedCausalConv1dRealSize, Width4Dilation3Prefill)
{
    std::vector<int32_t> const contextLengths{7, 48};
    runDilatedPrefillTest(DilatedShape{4, 3, 9}, 2, 48, 10240, &contextLengths, /* withInitialState = */ true);
}

TEST(MambaDilatedCausalConv1dRealSize, Width4Dilation3Decode)
{
    runDilatedDecodeTest(DilatedShape{4, 3, 9}, 2, 10240);
}

TEST(MambaDilatedCausalConv1dShapeContract, RejectsUnsupportedStateLengths)
{
    auto weight = rt::Tensor({8, 1, 4}, rt::DeviceType::kGPU, DataType::kHALF);
    auto newCol = rt::Tensor({1, 1, 8}, rt::DeviceType::kGPU, DataType::kHALF);
    auto out = rt::Tensor({1, 1, 8}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::OptionalInputTensor const noBias = std::nullopt;

    // state_len < (width-1)*dilation leaves the oldest tap outside the row.
    auto shortState = rt::Tensor({1, 8, 8}, rt::DeviceType::kGPU, DataType::kHALF);
    EXPECT_THROW(mamba_ssm::invokeCausalConv1dDecode(shortState, newCol, weight, noBias, out, 3, std::nullopt, nullptr),
        std::runtime_error);

    // state_len > 16 exceeds the register budget of the runtime-parameter kernels.
    auto longState = rt::Tensor({1, 8, 17}, rt::DeviceType::kGPU, DataType::kHALF);
    EXPECT_THROW(mamba_ssm::invokeCausalConv1dDecode(longState, newCol, weight, noBias, out, 1, std::nullopt, nullptr),
        std::runtime_error);

    // A receptive field wider than the register budget is rejected even when both state-length
    // bounds hold, which (width-1)*dilation == 16 does.
    auto wideWeight = rt::Tensor({8, 1, 5}, rt::DeviceType::kGPU, DataType::kHALF);
    auto maxState = rt::Tensor({1, 8, 16}, rt::DeviceType::kGPU, DataType::kHALF);
    EXPECT_THROW(
        mamba_ssm::invokeCausalConv1dDecode(maxState, newCol, wideWeight, noBias, out, 4, std::nullopt, nullptr),
        std::runtime_error);

    // The dilated receptive field must fit the shift register.
    auto x = rt::Tensor({1, 4, 8}, rt::DeviceType::kGPU, DataType::kHALF);
    auto prefillOut = rt::Tensor({1, 4, 8}, rt::DeviceType::kGPU, DataType::kHALF);
    EXPECT_THROW(mamba_ssm::invokeCausalConv1d(
                     x, weight, noBias, prefillOut, 1, 18, 6, std::nullopt, std::nullopt, std::nullopt, nullptr),
        std::runtime_error);
}
