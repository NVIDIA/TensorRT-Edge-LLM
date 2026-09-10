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

// Smoke tests for NvFP4 MoE support kernels: FP4 quantize, MoE gather, layout builder.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "kernels/moe/fp4SupportKernels/buildLayout.h"
#include "kernels/moe/fp4SupportKernels/fp4Quantize.h"
#include "kernels/moe/fp4SupportKernels/nvfp4MoeTypes.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t _e = (call);                                                                                       \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);                                        \
    } while (0)

namespace
{

int32_t padUp(int32_t a, int32_t b)
{
    return ((a + b - 1) / b) * b;
}

int32_t atomSfBytes(int32_t M, int32_t K, int32_t sfVecSize = 16)
{
    int32_t const sfCols = K / sfVecSize;
    int32_t const paddedSfCols = padUp(sfCols, 4);
    int32_t const paddedM = padUp(M, 128);
    return paddedM * paddedSfCols;
}

bool isAllZero(void const* data, size_t bytes)
{
    auto const* p = static_cast<uint8_t const*>(data);
    for (size_t i = 0; i < bytes; ++i)
    {
        if (p[i] != 0)
            return false;
    }
    return true;
}

bool isSupportedSm()
{
    int32_t const sm = trt_edgellm::getSMVersion();
    return sm == 100 || sm == 101 || sm == 110;
}

using rt::Coords;
using rt::DeviceType;
using rt::Tensor;

MoELayout createMoELayout(int32_t maxTokens, int32_t topK, int32_t localNumExperts, int32_t tileSize)
{
    int32_t const maxExpanded = maxTokens * topK;
    int32_t const maxMPadded = padUp(maxExpanded + localNumExperts * (tileSize - 1), tileSize);
    int32_t const maxTiles = maxMPadded / tileSize;
    MoELayout layout;
    layout.tileIdxToGroupIdx = trt_edgellm::rt::Tensor(Coords{maxTiles}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layout.tileIdxToMnLimit = trt_edgellm::rt::Tensor(Coords{maxTiles}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layout.permutedIdxToExpandedIdx
        = trt_edgellm::rt::Tensor(Coords{maxMPadded}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layout.numNonExitingTiles = trt_edgellm::rt::Tensor(Coords{1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    return layout;
}

MoELayoutBuffers createMoELayoutBuffers(int32_t maxTokens, int32_t topK, int32_t localNumExperts, int32_t tileSize)
{
    int32_t const maxExpanded = maxTokens * topK;
    int32_t const maxMPadded = padUp(maxExpanded + localNumExperts * (tileSize - 1), tileSize);
    int32_t const maxTiles = maxMPadded / tileSize;
    MoELayoutBuffers buf;
    buf.tileIdxToGroupIdx = trt_edgellm::rt::Tensor(Coords{maxTiles}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    buf.tileIdxToMnLimit = trt_edgellm::rt::Tensor(Coords{maxTiles}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    buf.permutedIdxToExpandedIdx
        = trt_edgellm::rt::Tensor(Coords{maxMPadded}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    buf.numNonExitingTiles = trt_edgellm::rt::Tensor(Coords{1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    return buf;
}

int32_t narrowToInt32(std::size_t value, char const* name)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error(std::string(name) + " exceeds int32 range");
    return static_cast<int32_t>(value);
}

// CPU reference layout builder — used as ground truth for GPU layout tests.
void buildLayoutCpu(MoELayout& layout, int32_t const* tokenSelectedExperts, int32_t numTokens, int32_t topK,
    int32_t numExperts, int32_t localExpertOffset, int32_t localNumExperts, int32_t tileSize, cudaStream_t stream)
{
    if (numTokens < 0)
        throw std::invalid_argument("numTokens must be non-negative");
    if (numExperts <= 0)
        throw std::invalid_argument("numExperts must be positive");
    if (localExpertOffset < 0 || localNumExperts <= 0)
        throw std::invalid_argument("localExpertOffset and localNumExperts must define a valid range");
    if (localExpertOffset + localNumExperts > numExperts)
        throw std::invalid_argument("local expert range exceeds numExperts");
    if (tileSize <= 0)
        throw std::invalid_argument("tileSize must be positive");

    std::vector<int32_t> permHost, tileGroupHost, tileMnHost;
    int64_t runningLimit = 0;

    for (int32_t le = 0; le < localNumExperts; ++le)
    {
        int32_t const ge = localExpertOffset + le;
        std::vector<int32_t> selected;
        selected.reserve(static_cast<std::size_t>(numTokens));
        for (int32_t t = 0; t < numTokens; ++t)
            for (int32_t k = 0; k < topK; ++k)
                if (tokenSelectedExperts[t * topK + k] == ge)
                    selected.push_back(t * topK + k);

        int32_t const n = narrowToInt32(selected.size(), "numSelected");
        int32_t const padded = n > 0 ? padUp(n, tileSize) : 0;
        if (padded == 0)
            continue;

        std::vector<int32_t> block(static_cast<std::size_t>(padded), -1);
        std::copy(selected.begin(), selected.end(), block.begin());

        int32_t const nTiles = padded / tileSize;
        for (int32_t ti = 0; ti < nTiles; ++ti)
        {
            tileGroupHost.push_back(le);
            tileMnHost.push_back(static_cast<int32_t>(runningLimit + std::min((ti + 1) * tileSize, n)));
        }
        runningLimit += padded;
        permHost.insert(permHost.end(), block.begin(), block.end());
    }

    int32_t const numTiles = narrowToInt32(tileGroupHost.size(), "numTiles");
    int32_t const mPadded = narrowToInt32(permHost.size(), "mPadded");

    // Copy to pre-allocated device buffers
    if (!tileGroupHost.empty())
        cudaMemcpyAsync(layout.tileIdxToGroupIdx.dataPointer<int32_t>(), tileGroupHost.data(),
            tileGroupHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (!tileMnHost.empty())
        cudaMemcpyAsync(layout.tileIdxToMnLimit.dataPointer<int32_t>(), tileMnHost.data(),
            tileMnHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (!permHost.empty())
        cudaMemcpyAsync(layout.permutedIdxToExpandedIdx.dataPointer<int32_t>(), permHost.data(),
            permHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        layout.numNonExitingTiles.dataPointer<int32_t>(), &numTiles, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    layout.numTiles = numTiles;
    layout.mPadded = mPadded;
    layout.tileIdxToGroupIdxHost = std::move(tileGroupHost);
    layout.tileIdxToMnLimitHost = std::move(tileMnHost);
}

} // namespace

// =========================================================================
// FP4 Quantize Tests
// =========================================================================
// fp4Quantize emits FP8 E4M3 scale factors; skip when CUDA_VERSION < 11080.
#if SUPPORTS_FP8

TEST(NvFP4MoEQuantizeTest, fp4OutputNonZero)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t M = 128;
    constexpr int32_t N = 2688;
    constexpr int32_t sfVecSize = 16;

    std::vector<__nv_bfloat16> hostInput(M * N, __float2bfloat16(1.0f));

    int32_t const fp4Bytes = M * (N / 8) * static_cast<int32_t>(sizeof(uint32_t));
    int32_t const sfBytes = atomSfBytes(M, N, sfVecSize);

    Tensor dInput(Coords{M, N}, DeviceType::kGPU, nvinfer1::DataType::kBF16);
    Tensor dGsf(Coords{1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    Tensor dFP4(Coords{M * (N / 8)}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor dSF(Coords{sfBytes}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    CHECK_CUDA(
        cudaMemcpy(dInput.rawPointer(), hostInput.data(), M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dFP4.rawPointer(), 0, fp4Bytes));
    CHECK_CUDA(cudaMemset(dSF.rawPointer(), 0, sfBytes));

    // Forward-scale contract: hostGsf is max|x|/(448*6). The kernel inverts in-register.
    float const hostGsf = 1.0f / (448.0f * 6.0f);
    CHECK_CUDA(cudaMemcpy(dGsf.rawPointer(), &hostGsf, sizeof(float), cudaMemcpyHostToDevice));
    fp4Quantize(dInput, dGsf, dFP4, dSF, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify outputs are non-zero
    std::vector<uint8_t> hostFP4(fp4Bytes);
    std::vector<uint8_t> hostSF(sfBytes);
    CHECK_CUDA(cudaMemcpy(hostFP4.data(), dFP4.rawPointer(), fp4Bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hostSF.data(), dSF.rawPointer(), sfBytes, cudaMemcpyDeviceToHost));

    EXPECT_FALSE(isAllZero(hostFP4.data(), hostFP4.size())) << "FP4 output should not be all zeros";
    EXPECT_FALSE(isAllZero(hostSF.data(), hostSF.size())) << "SF output should not be all zeros";
}

TEST(NvFP4MoEQuantizeTest, fp4OutputNonZeroFp16)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t M = 128;
    constexpr int32_t N = 2688;
    constexpr int32_t sfVecSize = 16;

    std::vector<__half> hostInput(M * N, __float2half(1.0f));

    int32_t const fp4Bytes = M * (N / 8) * static_cast<int32_t>(sizeof(uint32_t));
    int32_t const sfBytes = atomSfBytes(M, N, sfVecSize);

    Tensor dInput(Coords{M, N}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor dGsf(Coords{1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    Tensor dFP4(Coords{M * (N / 8)}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor dSF(Coords{sfBytes}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    CHECK_CUDA(cudaMemcpy(dInput.rawPointer(), hostInput.data(), M * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dFP4.rawPointer(), 0, fp4Bytes));
    CHECK_CUDA(cudaMemset(dSF.rawPointer(), 0, sfBytes));

    float const hostGsf = 1.0f / (448.0f * 6.0f);
    CHECK_CUDA(cudaMemcpy(dGsf.rawPointer(), &hostGsf, sizeof(float), cudaMemcpyHostToDevice));
    fp4Quantize(dInput, dGsf, dFP4, dSF, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> hostFP4(fp4Bytes);
    std::vector<uint8_t> hostSF(sfBytes);
    CHECK_CUDA(cudaMemcpy(hostFP4.data(), dFP4.rawPointer(), fp4Bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hostSF.data(), dSF.rawPointer(), sfBytes, cudaMemcpyDeviceToHost));

    EXPECT_FALSE(isAllZero(hostFP4.data(), hostFP4.size())) << "FP16 FP4 output should not be all zeros";
    EXPECT_FALSE(isAllZero(hostSF.data(), hostSF.size())) << "FP16 SF output should not be all zeros";
}

TEST(NvFP4MoEQuantizeTest, bf16Fp16ProduceSameOutput)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t M = 128;
    constexpr int32_t N = 2688;
    constexpr int32_t sfVecSize = 16;

    // Fill both inputs with the same 1.0 value
    std::vector<__nv_bfloat16> hostBf16(M * N, __float2bfloat16(1.0f));
    std::vector<__half> hostFp16(M * N, __float2half(1.0f));

    int32_t const fp4Bytes = M * (N / 8) * static_cast<int32_t>(sizeof(uint32_t));
    int32_t const sfBytes = atomSfBytes(M, N, sfVecSize);

    // BF16 path
    Tensor dBf16Input(Coords{M, N}, DeviceType::kGPU, nvinfer1::DataType::kBF16);
    Tensor dGsf(Coords{1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    Tensor dBf16FP4(Coords{M * (N / 8)}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor dBf16SF(Coords{sfBytes}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    CHECK_CUDA(
        cudaMemcpy(dBf16Input.rawPointer(), hostBf16.data(), M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dBf16FP4.rawPointer(), 0, fp4Bytes));
    CHECK_CUDA(cudaMemset(dBf16SF.rawPointer(), 0, sfBytes));

    float const hostGsf = 1.0f / (448.0f * 6.0f);
    CHECK_CUDA(cudaMemcpy(dGsf.rawPointer(), &hostGsf, sizeof(float), cudaMemcpyHostToDevice));
    fp4Quantize(dBf16Input, dGsf, dBf16FP4, dBf16SF, nullptr);

    // FP16 path
    Tensor dFp16Input(Coords{M, N}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor dFp16FP4(Coords{M * (N / 8)}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor dFp16SF(Coords{sfBytes}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    CHECK_CUDA(cudaMemcpy(dFp16Input.rawPointer(), hostFp16.data(), M * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dFp16FP4.rawPointer(), 0, fp4Bytes));
    CHECK_CUDA(cudaMemset(dFp16SF.rawPointer(), 0, sfBytes));

    CHECK_CUDA(cudaMemcpy(dGsf.rawPointer(), &hostGsf, sizeof(float), cudaMemcpyHostToDevice));
    fp4Quantize(dFp16Input, dGsf, dFp16FP4, dFp16SF, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Compare outputs — same input value should produce identical FP4 + SF
    std::vector<uint8_t> bf16FP4(fp4Bytes), fp16FP4(fp4Bytes);
    std::vector<uint8_t> bf16SF(sfBytes), fp16SF(sfBytes);
    CHECK_CUDA(cudaMemcpy(bf16FP4.data(), dBf16FP4.rawPointer(), fp4Bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fp16FP4.data(), dFp16FP4.rawPointer(), fp4Bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bf16SF.data(), dBf16SF.rawPointer(), sfBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fp16SF.data(), dFp16SF.rawPointer(), sfBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(bf16FP4, fp16FP4) << "BF16 and FP16 should produce identical FP4 output for uniform 1.0 input";
    EXPECT_EQ(bf16SF, fp16SF) << "BF16 and FP16 should produce identical SF output for uniform 1.0 input";
}

#endif // SUPPORTS_FP8
// =========================================================================
// Layout Builder Tests
// =========================================================================

TEST(NvFP4MoELayoutTest, cpuBuildLayoutBasic)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t numTokens = 32;
    constexpr int32_t topK = 6;
    constexpr int32_t numExperts = 128;
    constexpr int32_t localExpertOffset = 0;
    constexpr int32_t localNumExperts = 16;
    constexpr int32_t tileSize = 128;

    // Uniform routing: token i selects experts i%16 through (i+5)%16
    std::vector<int32_t> hostRouting(numTokens * topK);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        for (int32_t k = 0; k < topK; ++k)
        {
            hostRouting[t * topK + k] = (t + k) % localNumExperts;
        }
    }

    MoELayout layout = createMoELayout(numTokens, topK, localNumExperts, tileSize);
    buildLayoutCpu(
        layout, hostRouting.data(), numTokens, topK, numExperts, localExpertOffset, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    EXPECT_GT(layout.numTiles, 0) << "Should have at least one tile";
    EXPECT_GT(layout.mPadded, 0) << "mPadded should be positive";
}

TEST(NvFP4MoELayoutTest, gpuBuildLayoutProducesValidOutput)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t numTokens = 256;
    constexpr int32_t topK = 6;
    constexpr int32_t localNumExperts = 16;
    constexpr int32_t tileSize = 128;

    // Random routing in [0, localNumExperts)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, localNumExperts - 1);
    std::vector<int32_t> hostRouting(numTokens * topK);
    for (auto& v : hostRouting)
    {
        v = dist(rng);
    }

    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, numTokens * topK * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), numTokens * topK * sizeof(int32_t), cudaMemcpyHostToDevice));

    MoELayoutBuffers buffers = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    ASSERT_NE(buffers.permutedIdxToExpandedIdx.rawPointer(), nullptr);
    ASSERT_GT(buffers.tileIdxToGroupIdx.getShape()[0], 0);

    buildLayoutGpu(buffers, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t hostNumTiles = 0;
    CHECK_CUDA(cudaMemcpy(
        &hostNumTiles, buffers.numNonExitingTiles.dataPointer<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost));

    EXPECT_GT(hostNumTiles, 0) << "GPU layout should produce at least one tile";
    EXPECT_LE(hostNumTiles, buffers.tileIdxToGroupIdx.getShape()[0])
        << "Tile count should not exceed worst-case capacity";

    cudaFree(dRouting);
}

TEST(NvFP4MoELayoutTest, gpuLayoutCpuLayoutTileCountMatch)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t numTokens = 64;
    constexpr int32_t topK = 6;
    constexpr int32_t numExperts = 128;
    constexpr int32_t localExpertOffset = 0;
    constexpr int32_t localNumExperts = 16;
    constexpr int32_t tileSize = 128;

    // Deterministic routing
    std::vector<int32_t> hostRouting(numTokens * topK);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        for (int32_t k = 0; k < topK; ++k)
        {
            hostRouting[t * topK + k] = (t * topK + k) % localNumExperts;
        }
    }

    // CPU layout
    MoELayout cpuLayout = createMoELayout(numTokens, topK, localNumExperts, tileSize);
    buildLayoutCpu(cpuLayout, hostRouting.data(), numTokens, topK, numExperts, localExpertOffset, localNumExperts,
        tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU layout
    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, numTokens * topK * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), numTokens * topK * sizeof(int32_t), cudaMemcpyHostToDevice));

    MoELayoutBuffers buffers = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    buildLayoutGpu(buffers, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t gpuNumTiles = 0;
    CHECK_CUDA(cudaMemcpy(
        &gpuNumTiles, buffers.numNonExitingTiles.dataPointer<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(gpuNumTiles, cpuLayout.numTiles) << "GPU and CPU layout should produce the same tile count";

    cudaFree(dRouting);
}

// Verify GPU layout tile metadata matches CPU reference exactly.
TEST(NvFP4MoELayoutTest, gpuTileMetadataMatchesCpu)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t numTokens = 64;
    constexpr int32_t topK = 6;
    constexpr int32_t numExperts = 128;
    constexpr int32_t localExpertOffset = 0;
    constexpr int32_t localNumExperts = 16;
    constexpr int32_t tileSize = 128;

    // Deterministic routing
    std::vector<int32_t> hostRouting(numTokens * topK);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        for (int32_t k = 0; k < topK; ++k)
        {
            hostRouting[t * topK + k] = (t * topK + k) % localNumExperts;
        }
    }

    // CPU reference
    MoELayout cpuLayout = createMoELayout(numTokens, topK, localNumExperts, tileSize);
    buildLayoutCpu(cpuLayout, hostRouting.data(), numTokens, topK, numExperts, localExpertOffset, localNumExperts,
        tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    ASSERT_GT(cpuLayout.numTiles, 0);

    // GPU layout
    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, numTokens * topK * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), numTokens * topK * sizeof(int32_t), cudaMemcpyHostToDevice));

    MoELayoutBuffers buffers = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    buildLayoutGpu(buffers, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t gpuNumTiles = 0;
    CHECK_CUDA(cudaMemcpy(
        &gpuNumTiles, buffers.numNonExitingTiles.dataPointer<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    ASSERT_EQ(gpuNumTiles, cpuLayout.numTiles);

    // Compare tileIdxToGroupIdx
    std::vector<int32_t> gpuTileGroup(gpuNumTiles);
    CHECK_CUDA(cudaMemcpy(gpuTileGroup.data(), buffers.tileIdxToGroupIdx.dataPointer<int32_t>(),
        gpuNumTiles * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (int32_t i = 0; i < gpuNumTiles; ++i)
    {
        EXPECT_EQ(gpuTileGroup[i], cpuLayout.tileIdxToGroupIdxHost[i]) << "tileIdxToGroupIdx mismatch at tile " << i;
    }

    // Compare tileIdxToMnLimit
    std::vector<int32_t> gpuTileMnLimit(gpuNumTiles);
    CHECK_CUDA(cudaMemcpy(gpuTileMnLimit.data(), buffers.tileIdxToMnLimit.dataPointer<int32_t>(),
        gpuNumTiles * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (int32_t i = 0; i < gpuNumTiles; ++i)
    {
        EXPECT_EQ(gpuTileMnLimit[i], cpuLayout.tileIdxToMnLimitHost[i]) << "tileIdxToMnLimit mismatch at tile " << i;
    }

    cudaFree(dRouting);
}

// Verify GPU permutedIdxToExpandedIdx covers the same set of expanded indices as CPU.
TEST(NvFP4MoELayoutTest, gpuPermutationCoversAllTokens)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t numTokens = 32;
    constexpr int32_t topK = 6;
    constexpr int32_t numExperts = 128;
    constexpr int32_t localExpertOffset = 0;
    constexpr int32_t localNumExperts = 16;
    constexpr int32_t tileSize = 128;

    // All tokens route to expert 0 (concentrated load)
    std::vector<int32_t> hostRouting(numTokens * topK, 0);

    // CPU reference
    MoELayout cpuLayout = createMoELayout(numTokens, topK, localNumExperts, tileSize);
    buildLayoutCpu(cpuLayout, hostRouting.data(), numTokens, topK, numExperts, localExpertOffset, localNumExperts,
        tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU layout
    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, numTokens * topK * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), numTokens * topK * sizeof(int32_t), cudaMemcpyHostToDevice));

    MoELayoutBuffers buffers = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    buildLayoutGpu(buffers, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read GPU permutation
    std::vector<int32_t> gpuPerm(cpuLayout.mPadded);
    CHECK_CUDA(cudaMemcpy(gpuPerm.data(), buffers.permutedIdxToExpandedIdx.dataPointer<int32_t>(),
        cpuLayout.mPadded * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Count valid (non -1) entries — should equal numTokens * topK (all route to expert 0)
    int32_t const numExpanded = numTokens * topK;
    int32_t gpuValidCount = 0;
    int32_t cpuValidCount = 0;
    std::vector<int32_t> gpuPermCopy;
    std::vector<int32_t> cpuPermCopy;

    for (int32_t i = 0; i < cpuLayout.mPadded; ++i)
    {
        if (gpuPerm[i] >= 0)
        {
            ++gpuValidCount;
            gpuPermCopy.push_back(gpuPerm[i]);
        }
    }

    // Read CPU permutation from device
    std::vector<int32_t> cpuPermDevice(cpuLayout.mPadded);
    CHECK_CUDA(cudaMemcpy(cpuPermDevice.data(), cpuLayout.permutedIdxToExpandedIdx.dataPointer<int32_t>(),
        cpuLayout.mPadded * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int32_t i = 0; i < cpuLayout.mPadded; ++i)
    {
        if (cpuPermDevice[i] >= 0)
        {
            ++cpuValidCount;
            cpuPermCopy.push_back(cpuPermDevice[i]);
        }
    }

    EXPECT_EQ(gpuValidCount, numExpanded) << "GPU should place all expanded indices";
    EXPECT_EQ(cpuValidCount, numExpanded) << "CPU should place all expanded indices";

    // Both should cover the same set of expanded indices (order may differ due to atomics)
    std::sort(gpuPermCopy.begin(), gpuPermCopy.end());
    std::sort(cpuPermCopy.begin(), cpuPermCopy.end());
    EXPECT_EQ(gpuPermCopy, cpuPermCopy) << "GPU and CPU should produce the same set of expanded indices";

    cudaFree(dRouting);
}

// =========================================================================
// Worst-case routing stress tests for the GPU layout builder
//
// These exercise the `T + L·(tileSize − 1)` worst-case bound baked into the
// plugin's layout-buffer sizing (see docs/source/developer_guide/software-design/
// nvfp4-moe-prefill.md § "Worst-case accounting for permutedM_max").
// =========================================================================

// Each of L=16 experts gets exactly one token; worst-case per-expert padding = 127 each.
// Worst-case layout buffer is at least `T + L·(tileSize−1) = 16 + 16·127 = 2048` rows.
TEST(NvFP4MoEWorstCaseRoutingTest, singletonPerExpert)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t localNumExperts = 16;
    constexpr int32_t topK = 1;
    constexpr int32_t tileSize = 128;
    constexpr int32_t numTokens = localNumExperts;

    std::vector<int32_t> hostRouting(numTokens * topK);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        hostRouting[t] = t;
    }

    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, hostRouting.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), hostRouting.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    MoELayoutBuffers buf = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    int32_t const maxMPadded = static_cast<int32_t>(buf.permutedIdxToExpandedIdx.getShape()[0]);
    CHECK_CUDA(cudaMemset(buf.permutedIdxToExpandedIdx.rawPointer(), 0x7A, maxMPadded * sizeof(int32_t)));
    buildLayoutGpu(buf, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int32_t> hostPerm(maxMPadded);
    CHECK_CUDA(cudaMemcpy(hostPerm.data(), buf.permutedIdxToExpandedIdx.rawPointer(), maxMPadded * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    int32_t valid = 0;
    for (int32_t v : hostPerm)
    {
        if (v >= 0)
            ++valid;
    }
    EXPECT_EQ(valid, numTokens * topK) << "Exactly one valid entry per token-expert slot expected";
    cudaFree(dRouting);
}

// All tokens route to expert 0 — one huge group, no per-expert padding elsewhere.
TEST(NvFP4MoEWorstCaseRoutingTest, hotExpertHotspot)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t localNumExperts = 16;
    constexpr int32_t topK = 2;
    constexpr int32_t tileSize = 128;
    constexpr int32_t numTokens = 256;

    std::vector<int32_t> hostRouting(numTokens * topK, 0);
    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, hostRouting.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), hostRouting.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    MoELayoutBuffers buf = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    int32_t const maxMPadded = static_cast<int32_t>(buf.permutedIdxToExpandedIdx.getShape()[0]);
    buildLayoutGpu(buf, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int32_t> hostPerm(maxMPadded);
    CHECK_CUDA(cudaMemcpy(hostPerm.data(), buf.permutedIdxToExpandedIdx.rawPointer(), maxMPadded * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    int32_t valid = 0;
    for (int32_t v : hostPerm)
    {
        if (v >= 0)
            ++valid;
    }
    EXPECT_EQ(valid, numTokens * topK) << "All T*topK entries route to expert 0";
    cudaFree(dRouting);
}

// Half the experts receive zero tokens: tile-table must handle empty groups without writing
// garbage tile metadata for inactive experts.
TEST(NvFP4MoEWorstCaseRoutingTest, zeroHitExperts)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t localNumExperts = 16;
    constexpr int32_t topK = 2;
    constexpr int32_t tileSize = 128;
    constexpr int32_t numTokens = 64;

    std::vector<int32_t> hostRouting(numTokens * topK);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        for (int32_t k = 0; k < topK; ++k)
        {
            // Only experts 0..7 receive tokens; experts 8..15 are empty.
            hostRouting[t * topK + k] = (t + k) % (localNumExperts / 2);
        }
    }

    int32_t* dRouting = nullptr;
    CHECK_CUDA(cudaMalloc(&dRouting, hostRouting.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dRouting, hostRouting.data(), hostRouting.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    MoELayoutBuffers buf = createMoELayoutBuffers(numTokens, topK, localNumExperts, tileSize);
    buildLayoutGpu(buf, dRouting, numTokens, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t hostNumTiles = 0;
    CHECK_CUDA(cudaMemcpy(&hostNumTiles, buf.numNonExitingTiles.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    // No active expert is starved: numNonExitingTiles > 0.
    EXPECT_GT(hostNumTiles, 0) << "numNonExitingTiles must be > 0 even when half the experts are idle";
    EXPECT_LE(hostNumTiles, buf.tileIdxToGroupIdx.getShape()[0])
        << "numNonExitingTiles must not exceed max-tile capacity";
    cudaFree(dRouting);
}

// numTokens == 0: empty input. Kernel contract is that numNonExitingTiles == 0 and the
// permuted-idx buffer is set to all −1 (the plugin's gather kernel relies on this so the
// kernel short-circuits every row to zero-fill).
TEST(NvFP4MoEWorstCaseRoutingTest, emptyInput)
{
    if (!isSupportedSm())
    {
        GTEST_SKIP() << "Requires SM100/101/110";
    }

    constexpr int32_t localNumExperts = 16;
    constexpr int32_t topK = 2;
    constexpr int32_t tileSize = 128;
    constexpr int32_t numTokensProfile = 32; // profile max; runtime will be zero

    MoELayoutBuffers buf = createMoELayoutBuffers(numTokensProfile, topK, localNumExperts, tileSize);
    int32_t const maxMPadded = static_cast<int32_t>(buf.permutedIdxToExpandedIdx.getShape()[0]);

    // Dirty the buffer so we can detect the kernel's zero-token path.
    CHECK_CUDA(cudaMemset(buf.permutedIdxToExpandedIdx.rawPointer(), 0x42, maxMPadded * sizeof(int32_t)));
    int32_t const prime = 0xDEADBEEF;
    CHECK_CUDA(cudaMemcpy(buf.numNonExitingTiles.rawPointer(), &prime, sizeof(int32_t), cudaMemcpyHostToDevice));

    buildLayoutGpu(buf, /*dRouting=*/nullptr, /*numTokens=*/0, topK, localNumExperts, tileSize, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t hostNumTiles = 0x7FFFFFFF;
    CHECK_CUDA(cudaMemcpy(&hostNumTiles, buf.numNonExitingTiles.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(hostNumTiles, 0) << "Empty input must produce numNonExitingTiles == 0";

    std::vector<int32_t> hostPerm(maxMPadded);
    CHECK_CUDA(cudaMemcpy(hostPerm.data(), buf.permutedIdxToExpandedIdx.rawPointer(), maxMPadded * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    int32_t bad = 0;
    for (int32_t v : hostPerm)
    {
        if (v != -1)
        {
            ++bad;
        }
    }
    EXPECT_EQ(bad, 0) << "Empty input must leave permutedIdxToExpandedIdx filled with −1 sentinels";
}
