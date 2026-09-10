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
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/causalConv1d.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace nvinfer1;

void runCausalConv1dReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, int32_t padding,
    std::vector<half> const& x, std::vector<half> const& weight, std::vector<half> const& bias,
    std::vector<half>& outRef, std::vector<int32_t> const* contextLens = nullptr,
    std::vector<half> const* initialState = nullptr)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLens ? (*contextLens)[b] : seqLen;
        for (int32_t s = 0; s < seqLen; ++s)
        {
            int32_t const inBase = s - padding;
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
                    int32_t const inPos = inBase + k;
                    if (inPos >= 0 && inPos < cl)
                    {
                        int64_t const xIdx
                            = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(inPos) * dim + d;
                        int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                        acc += __half2float(x[xIdx]) * __half2float(weight[wIdx]);
                    }
                    else if (initialState != nullptr && inPos < 0 && inPos >= -width)
                    {
                        int64_t const stateIdx = (static_cast<int64_t>(b) * dim + d) * width + width + inPos;
                        int64_t const wIdx = static_cast<int64_t>(d) * width + k;
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

    runCausalConv1dReference(batch, seqLen, dim, width, width - 1, xHost, weightHost, biasHost, outputRef, contextLens);

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
        xDevice, weightDevice, biasOpt, outputDevice, 1, width - 1, 1, std::nullopt, clOpt, nullptr);
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

// ---------------------------------------------------------------------------
// invokeCaptureConvState tests
// ---------------------------------------------------------------------------

void runCaptureConvStateReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, std::vector<half> const& x,
    std::vector<half>& convStateRef, std::vector<int32_t> const* contextLens = nullptr,
    std::vector<half> const* initialState = nullptr)
{
    std::fill(convStateRef.begin(), convStateRef.end(), __float2half(0.F));
    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLens ? (*contextLens)[b] : seqLen;
        int32_t const tailLen = (cl >= width) ? width : cl;
        int32_t const tailStart = cl - tailLen;
        int32_t const dstOffset = width - tailLen;
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t const stateOffset = (static_cast<int64_t>(b) * dim + d) * width;
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
    mamba_ssm::invokeCaptureConvState(xDevice, std::nullopt, convStateDevice, clOpt, nullptr);
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
    runCausalConv1dReference(batch, seqLen, dim, width, width - 1, xHost, weightHost, biasHost, outputRef,
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
    mamba_ssm::invokeCausalConv1d(
        xDevice, weightDevice, biasOpt, outputDevice, 1, width - 1, 1, initialStateOpt, contextLengthsOpt, nullptr);
    mamba_ssm::invokeCaptureConvState(xDevice, initialStateOpt, capturedStateDevice, contextLengthsOpt, nullptr);
    mamba_ssm::invokeCaptureConvState(xDevice, initialStateOpt, stateDevice, contextLengthsOpt, nullptr);
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

void runCausalConv1dDecodeReference(int32_t batch, int32_t dim, int32_t width, std::vector<half> const& convState,
    std::vector<half> const& newCol, std::vector<half> const& weight, std::vector<half> const& bias,
    std::vector<half>& convStateOut, std::vector<half>& outRef)
{
    convStateOut = convState;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t rowOff = (static_cast<int64_t>(b) * dim + d) * width;
            for (int32_t k = 0; k < width - 1; ++k)
            {
                convStateOut[rowOff + k] = convStateOut[rowOff + k + 1];
            }
            convStateOut[rowOff + width - 1] = newCol[static_cast<int64_t>(b) * dim + d];
            float acc = __half2float(bias[d]);
            for (int32_t k = 0; k < width; ++k)
            {
                int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                acc += __half2float(convStateOut[rowOff + k]) * __half2float(weight[wIdx]);
            }
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
        batch, dim, width, convStateHost, newColHost, weightHost, biasHost, convStateRef, outRef);

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
    mamba_ssm::invokeCausalConv1dDecode(convStateDevice, newColDevice, weightDevice, biasOpt, outDevice, nullptr);
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

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecodeMTP tests
// ---------------------------------------------------------------------------

/**
 * CPU reference for MTP decode: processes T tokens sequentially, with per-step
 * state checkpointing. Verifies output, final state, and intermediate states.
 */
void runCausalConv1dDecodeMTPReference(int32_t batch, int32_t dim, int32_t width, int32_t T,
    std::vector<half> const& convState, std::vector<half> const& newCols, std::vector<half> const& weight,
    std::vector<half> const& bias, std::vector<half>& convStateOut, std::vector<half>& outRef,
    std::vector<half>& intermRef)
{
    convStateOut = convState;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            int64_t rowOff = (static_cast<int64_t>(b) * dim + d) * width;
            for (int32_t t = 0; t < T; ++t)
            {
                // Shift left
                for (int32_t k = 0; k < width - 1; ++k)
                {
                    convStateOut[rowOff + k] = convStateOut[rowOff + k + 1];
                }
                // Insert new token
                int64_t const newIdx = (static_cast<int64_t>(b) * T + t) * dim + d;
                convStateOut[rowOff + width - 1] = newCols[newIdx];
                // Dot product
                float acc = __half2float(bias[d]);
                for (int32_t k = 0; k < width; ++k)
                {
                    int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                    acc += __half2float(convStateOut[rowOff + k]) * __half2float(weight[wIdx]);
                }
                int64_t const outIdx = (static_cast<int64_t>(b) * T + t) * dim + d;
                outRef[outIdx] = __float2half(acc);
                // Checkpoint intermediate state
                int64_t const intermBase = ((static_cast<int64_t>(b) * T + t) * dim + d) * width;
                for (int32_t k = 0; k < width; ++k)
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
    runCausalConv1dDecodeMTPReference(
        batch, dim, width, T, convStateHost, newColsHost, weightHost, biasHost, convStateRef, outRef, intermRef);

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
        convStateDevice, newColsDevice, weightDevice, biasOpt, outDevice, intermDevice, T, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify output
    auto const outHost = copyDeviceToHost<half>(outDevice);
    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "MTP output mismatch at index " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outRef[i]);
    }

    // Verify final state
    auto const stateHost = copyDeviceToHost<half>(convStateDevice);
    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(stateHost[i], convStateRef[i], 1e-3F, 1e-3F))
            << "MTP final state mismatch at index " << i << ": got " << __half2float(stateHost[i]) << ", expected "
            << __half2float(convStateRef[i]);
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

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecodeDDTree tests
// ---------------------------------------------------------------------------

void runCausalConv1dDecodeDDTreeReference(int32_t batch, int32_t dim, int32_t width, int32_t verifySeq,
    std::vector<half> const& convState, std::vector<half> const& newCols, std::vector<half> const& weight,
    std::vector<half> const& bias, std::vector<int32_t> const& parentIds, std::vector<int32_t> const& depths,
    std::vector<half>& convStateOut, std::vector<half>& outRef, std::vector<half>& intermRef)
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
                int64_t const rowOff = (static_cast<int64_t>(b) * dim + d) * width;
                float state[8];
                if (!isValidNode)
                {
                    for (int32_t k = 0; k < width; ++k)
                    {
                        state[k] = __half2float(convState[rowOff + k]);
                    }
                    outRef[(static_cast<int64_t>(b) * verifySeq + node) * dim + d] = __float2half(0.0F);
                }
                else
                {
                    int32_t pathNodes[8];
                    int32_t pathLen = 0;
                    int32_t const maxPathLen = (depth + 1 < width) ? depth + 1 : width;
                    int32_t currentNode = node;
                    while (pathLen < maxPathLen && currentNode >= 0 && currentNode < verifySeq)
                    {
                        pathNodes[pathLen++] = currentNode;
                        if (currentNode == 0)
                        {
                            break;
                        }
                        currentNode = parentIds[b * verifySeq + currentNode];
                    }

                    for (int32_t k = 0; k < width; ++k)
                    {
                        state[k] = (k + pathLen < width) ? __half2float(convState[rowOff + k + pathLen]) : 0.0F;
                    }
                    for (int32_t pathOffset = 0; pathOffset < pathLen; ++pathOffset)
                    {
                        int32_t const pathNode = pathNodes[pathOffset];
                        int64_t const newColIdx = (static_cast<int64_t>(b) * verifySeq + pathNode) * dim + d;
                        state[width - 1 - pathOffset] = __half2float(newCols[newColIdx]);
                    }

                    float acc = __half2float(bias[d]);
                    for (int32_t k = 0; k < width; ++k)
                    {
                        acc += state[k] * __half2float(weight[static_cast<int64_t>(d) * width + k]);
                    }
                    outRef[(static_cast<int64_t>(b) * verifySeq + node) * dim + d] = __float2half(acc);
                }

                int64_t const intermBase = ((static_cast<int64_t>(b) * verifySeq + node) * dim + d) * width;
                for (int32_t k = 0; k < width; ++k)
                {
                    intermRef[intermBase + k] = __float2half(state[k]);
                }
            }
        }
    }
}

TEST(MambaCausalConv1dDecodeDDTree, RootToNodeState)
{
    constexpr int32_t batch = 2;
    constexpr int32_t dim = 64;
    constexpr int32_t width = 4;
    constexpr int32_t verifySeq = 6;

    std::vector<half> convStateHost(batch * dim * width);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> newColsHost(batch * verifySeq * dim);
    uniformFloatInitialization<half>(convStateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(newColsHost, -0.5F, 0.5F);

    std::vector<int32_t> parentIds{
        -1, 0, 1, 1, 3, -1, // batch 0: root, chain 0->1->2, branch 1->3->4, padding node 5
        -1, 0, 0, 2, 3, -1  // batch 1: root, two depth-1 children, chain 2->3->4, padding node 5
    };
    std::vector<int32_t> depths{
        0,
        1,
        2,
        2,
        3,
        0,
        0,
        1,
        1,
        2,
        3,
        0,
    };

    std::vector<half> convStateRef(convStateHost.size());
    std::vector<half> outRef(batch * verifySeq * dim);
    std::vector<half> intermRef(batch * verifySeq * dim * width);
    runCausalConv1dDecodeDDTreeReference(batch, dim, width, verifySeq, convStateHost, newColsHost, weightHost, biasHost,
        parentIds, depths, convStateRef, outRef, intermRef);

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

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecodeDDTree(convStateDevice, newColsDevice, weightDevice, biasOpt, outDevice,
        convStateOutDevice, intermDevice, parentDevice, depthDevice, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outHost = copyDeviceToHost<half>(outDevice);
    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "DDTree output mismatch at index " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outRef[i]);
    }

    auto const stateOutHost = copyDeviceToHost<half>(convStateOutDevice);
    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(stateOutHost[i], convStateRef[i], 1e-3F, 1e-3F))
            << "DDTree convStateOut mismatch at index " << i << ": got " << __half2float(stateOutHost[i])
            << ", expected " << __half2float(convStateRef[i]);
    }

    auto const intermHost = copyDeviceToHost<half>(intermDevice);
    for (size_t i = 0; i < intermRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(intermHost[i], intermRef[i], 1e-3F, 1e-3F))
            << "DDTree intermediate state mismatch at index " << i << ": got " << __half2float(intermHost[i])
            << ", expected " << __half2float(intermRef[i]);
    }
}
