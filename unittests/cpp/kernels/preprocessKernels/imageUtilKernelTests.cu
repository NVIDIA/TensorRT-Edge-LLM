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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/common/imageUtils.h"
#include "references.h"
#include "runtime/imageUtils.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace nvinfer1;

// Helper to build Phi-4MM batched inputs and golden output for postprocess kernel tests.
// hwBlocks: vector of (hBlocks, wBlocks) per image
static void BuildPhi4mmBatchedInputs(std::vector<std::pair<int32_t, int32_t>> const& hwBlocks, int32_t const hidden,
    std::vector<half>& srcEmbeds,          // out: raw ViT tokens [sum((1+hb*wb)*256), hidden]
    std::vector<half>& subGNHost,          // out: [hidden]
    std::vector<half>& glbGNHost,          // out: [hidden]
    std::vector<int32_t>& hBlocksHost,     // out: [numImages]
    std::vector<int32_t>& wBlocksHost,     // out: [numImages]
    std::vector<int64_t>& srcGlbStartHost, // out: [numImages]
    std::vector<int64_t>& srcSubStartHost, // out: [numImages]
    std::vector<int64_t>& dstOutStartHost, // out: [numImages]
    std::vector<int64_t>& subOutLenHost,   // out: [numImages]
    std::vector<half>& dstRef              // out: golden postprocessed output [totalOutTokens, hidden]
)
{
    hBlocksHost.clear();
    wBlocksHost.clear();
    srcGlbStartHost.clear();
    srcSubStartHost.clear();
    dstOutStartHost.clear();
    subOutLenHost.clear();

    // Compute total raw tokens and total output tokens
    int64_t totalRawTokens = 0;
    int64_t totalOutTokens = 0;
    for (auto const& hw : hwBlocks)
    {
        int32_t const hb = hw.first;
        int32_t const wb = hw.second;
        // raw tokens per image: 1 glb + hb*wb sub, each 256
        totalRawTokens += (1LL + static_cast<int64_t>(hb) * wb) * 256LL;
        // out tokens: sub grid (with newlines), 1 glb_GN, glb grid (with newlines)
        int64_t const subLen = kernel::kTokensPerBlockPhi4 * hb * wb + kernel::kTokensPerSidePhi4 * hb;
        int64_t const glbLen = kernel::kTokensPerSidePhi4 * (kernel::kTokensPerSidePhi4 + 1);
        totalOutTokens += subLen + 1 + glbLen;
    }

    // Prepare buffers
    srcEmbeds.resize(totalRawTokens * hidden);
    subGNHost.resize(hidden);
    glbGNHost.resize(hidden);
    dstRef.resize(totalOutTokens * hidden);

    // Deterministic content:
    // - For src tokens: token t's vector is filled with value = float(t)
    // - For subGN and glbGN: constant distinctive values
    for (int32_t d = 0; d < hidden; ++d)
    {
        subGNHost[d] = __float2half(-1.234f);
        glbGNHost[d] = __float2half(-2.345f);
    }
    // Fill src by token index
    for (int64_t t = 0; t < totalRawTokens; ++t)
    {
        half v = __float2half(static_cast<float>(t));
        int64_t base = t * hidden;
        for (int32_t d = 0; d < hidden; ++d)
        {
            srcEmbeds[base + d] = v;
        }
    }

    // Build index arrays and golden output
    int64_t inStartTok = 0;
    int64_t outStartTok = 0;
    for (auto const& hw : hwBlocks)
    {
        int32_t const hb = hw.first;
        int32_t const wb = hw.second;
        hBlocksHost.push_back(hb);
        wBlocksHost.push_back(wb);
        srcGlbStartHost.push_back(inStartTok);
        srcSubStartHost.push_back(inStartTok + 256);

        // Sub segment
        int64_t const rowsSub = kernel::kTokensPerSidePhi4 * hb;
        int64_t const colsSub = kernel::kTokensPerSidePhi4 * wb;
        int64_t const strideSub = colsSub + 1;
        int64_t const subLen = rowsSub * strideSub;
        subOutLenHost.push_back(subLen);
        dstOutStartHost.push_back(outStartTok);

        for (int64_t r = 0; r < rowsSub; ++r)
        {
            for (int64_t c = 0; c < strideSub; ++c)
            {
                int64_t const outTokIndex = outStartTok + r * strideSub + c;
                half* dstPtr = &dstRef[outTokIndex * hidden];
                if (c == colsSub)
                {
                    // newline: subGN
                    for (int32_t d = 0; d < hidden; ++d)
                        dstPtr[d] = subGNHost[d];
                }
                else
                {
                    // map to src sub token
                    int64_t const bRow = r / kernel::kTokensPerSidePhi4;
                    int64_t const pRow = r % kernel::kTokensPerSidePhi4;
                    int64_t const bCol = c / kernel::kTokensPerSidePhi4;
                    int64_t const pCol = c % kernel::kTokensPerSidePhi4;
                    int64_t const blockId = bRow * wb + bCol;
                    int64_t const patchId = pRow * kernel::kTokensPerSidePhi4 + pCol;
                    int64_t const srcTokIndex
                        = (inStartTok + kernel::kTokensPerBlockPhi4) + blockId * kernel::kTokensPerBlockPhi4 + patchId;
                    half const* srcPtr = &srcEmbeds[srcTokIndex * hidden];
                    for (int32_t d = 0; d < hidden; ++d)
                        dstPtr[d] = srcPtr[d];
                }
            }
        }
        outStartTok += subLen;

        // glb_GN single token
        {
            half* dstPtr = &dstRef[outStartTok * hidden];
            for (int32_t d = 0; d < hidden; ++d)
                dstPtr[d] = glbGNHost[d];
            outStartTok += 1;
        }

        // Global kTokensPerSidePhi4 x kTokensPerSidePhi4 grid with newline at end of each row
        int64_t const rowsGlb = kernel::kTokensPerSidePhi4;
        int64_t const colsGlb = kernel::kTokensPerSidePhi4;
        int64_t const strideGlb = colsGlb + 1;
        for (int64_t r = 0; r < rowsGlb; ++r)
        {
            for (int64_t c = 0; c < strideGlb; ++c)
            {
                int64_t const outTokIndex = outStartTok + r * strideGlb + c;
                half* dstPtr = &dstRef[outTokIndex * hidden];
                if (c == colsGlb)
                {
                    for (int32_t d = 0; d < hidden; ++d)
                        dstPtr[d] = subGNHost[d];
                }
                else
                {
                    int64_t const srcTokIndex = inStartTok + r * kernel::kTokensPerSidePhi4 + c;
                    half const* srcPtr = &srcEmbeds[srcTokIndex * hidden];
                    for (int32_t d = 0; d < hidden; ++d)
                        dstPtr[d] = srcPtr[d];
                }
            }
        }
        outStartTok += rowsGlb * strideGlb;

        // Advance raw pointer start
        inStartTok += (1LL + static_cast<int64_t>(hb) * wb) * 256LL;
    }
}

void TestTransposeToPatchQwenViT(int32_t const height, int32_t const width, int32_t const channels = 3,
    int32_t const T = 2, int32_t const temporalPatchSize = 2, int32_t const patchSize = 14, int32_t const mergeSize = 2,
    bool const temporalFirst = false, bool const channelLast = false)
{
    cudaStream_t stream{nullptr};

    // CPU reference
    std::vector<half> originalImage(T * height * width * channels);
    std::vector<half> inputPatchesRef(T * height * width * channels);
    uniformFloatInitialization<half>(originalImage, 0, 1);

    transposeToPatchQwenReference(originalImage, inputPatchesRef, 0, T, height, width, channels, temporalPatchSize,
        patchSize, mergeSize, temporalFirst, channelLast);

    // GPU tensors
    rt::Tensor originalImageDevice({T, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    int32_t const gridT = T / temporalPatchSize;
    int32_t const gridH = height / (mergeSize * patchSize);
    int32_t const gridW = width / (mergeSize * patchSize);
    int32_t const totalSeqLength = gridT * gridH * gridW * mergeSize * mergeSize;
    int32_t const inputDim = channels * temporalPatchSize * patchSize * patchSize;
    rt::Tensor inputPatchesDevice({totalSeqLength, inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    kernel::transposeToPatchQwenViT(originalImageDevice, inputPatchesDevice, 0, temporalPatchSize, patchSize, mergeSize,
        temporalFirst, channelLast, stream);

    std::vector<half> inputPatches(T * height * width * channels);
    CUDA_CHECK(cudaMemcpyAsync(inputPatches.data(), inputPatchesDevice.rawPointer(), inputPatches.size() * sizeof(half),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare data with debug output
    for (int32_t i = 0; i < inputPatches.size(); ++i)
    {
        ASSERT_TRUE(isclose(inputPatches[i], inputPatchesRef[i], 1e-5, 1e-5));
    }
    std::cout << "TransposeToPatchQwen Accuracy: " << height << "x" << width << "x" << channels << ", T=" << T
              << std::endl;
}

TEST(TransposeToPatchQwen, Accuracy)
{
    TestTransposeToPatchQwenViT(448, 448);
}

TEST(TransposeToPatchQwen, AccuracyT4)
{
    // Video path: gridT > 1 (T = 4 with temporalPatchSize = 2 gives gridT = 2).
    TestTransposeToPatchQwenViT(/*height*/ 224, /*width*/ 224, /*channels*/ 3, /*T*/ 4);
}

TEST(TransposeToPatchQwen, AccuracyTemporalFirst)
{
    TestTransposeToPatchQwenViT(
        /*height*/ 224, /*width*/ 224, /*channels*/ 3, /*T*/ 4, /*temporalPatchSize*/ 2, /*patchSize*/ 14,
        /*mergeSize*/ 1, /*temporalFirst*/ true);
}

TEST(TransposeToPatchQwen, AccuracyChannelLast)
{
    TestTransposeToPatchQwenViT(
        /*height*/ 224, /*width*/ 224, /*channels*/ 3, /*T*/ 4, /*temporalPatchSize*/ 2, /*patchSize*/ 14,
        /*mergeSize*/ 2, /*temporalFirst*/ false, /*channelLast*/ true);
}

void BenchmarkTransposeToPatchQwenViT(int32_t const height, int32_t const width, int32_t const channels = 3,
    int32_t const T = 2, int32_t const temporalPatchSize = 2, int32_t const patchSize = 14, int32_t const mergeSize = 2)
{
    cudaStream_t stream{nullptr};

    std::vector<half> originalImage(T * height * width * channels);
    uniformFloatInitialization<half>(originalImage, 0, 1);

    rt::Tensor originalImageDevice({T, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    int32_t const gridT = T / temporalPatchSize;
    int32_t const gridH = height / (mergeSize * patchSize);
    int32_t const gridW = width / (mergeSize * patchSize);
    int32_t const totalSeqLength = gridT * gridH * gridW * mergeSize * mergeSize;
    int32_t const inputDim = channels * temporalPatchSize * patchSize * patchSize;
    rt::Tensor inputPatchesDevice({totalSeqLength, inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    auto launch = [&]() {
        kernel::transposeToPatchQwenViT(
            originalImageDevice, inputPatchesDevice, 0, temporalPatchSize, patchSize, mergeSize, false, false, stream);
    };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "TransposeToPatchQwen Benchmark: " << height << "x" << width << "x" << channels << ", T=" << T
              << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(TransposeToPatchQwen, Benchmark)
{
    BenchmarkTransposeToPatchQwenViT(448, 448);
    BenchmarkTransposeToPatchQwenViT(728, 728);
}

void TestInitRotaryPosEmbQwenViT(int32_t const vitPosEmbDim = 40, int32_t const mergeSize = 2,
    float const rotaryBaseFrequency = 10000.0f, float const scale = 1.0f)
{
    cudaStream_t stream{nullptr};

    std::vector<std::vector<int64_t>> imageGridTHWs{{1, 36, 54}, {1, 8, 10}, {1, 32, 20}};
    std::vector<int32_t> cuSeqlens{0};
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        cuSeqlens.push_back(cuSeqlens.back() + imageGridTHWs[i][0] * imageGridTHWs[i][1] * imageGridTHWs[i][2]);
    }
    int32_t totalSeqLength = cuSeqlens.back();

    // CPU reference
    std::vector<float> rotaryPosEmb(totalSeqLength * vitPosEmbDim);
    initRotaryPosEmbQwenViTReference(
        rotaryPosEmb, imageGridTHWs, totalSeqLength, vitPosEmbDim, mergeSize, rotaryBaseFrequency, scale);

    // GPU kernel
    rt::Tensor rotaryPosEmbDevice({totalSeqLength, vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        kernel::initRotaryPosEmbQwenViT(
            rotaryPosEmbDevice, imageGridTHWs[i], mergeSize, cuSeqlens[i], rotaryBaseFrequency, scale, stream);
    }

    // Compare data
    std::vector<float> rotaryPosEmbHost(totalSeqLength * vitPosEmbDim);
    CUDA_CHECK(cudaMemcpyAsync(rotaryPosEmbHost.data(), rotaryPosEmbDevice.rawPointer(),
        rotaryPosEmbHost.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < totalSeqLength * vitPosEmbDim; ++i)
    {
        ASSERT_TRUE(isclose(rotaryPosEmbHost[i], rotaryPosEmb[i], 1e-5, 1e-5));
    }

    std::cout << "InitRotaryPosEmbQwen Accuracy: totalSeqLength=" << totalSeqLength << ", vitPosEmbDim=" << vitPosEmbDim
              << std::endl;
}

TEST(InitRotaryPosEmbQwen, Accuracy)
{
    TestInitRotaryPosEmbQwenViT();
}

void BenchmarkInitRotaryPosEmbQwenViT(int32_t const vitPosEmbDim = 40, int32_t const mergeSize = 2,
    float const rotaryBaseFrequency = 10000.0f, float const scale = 1.0f)
{
    cudaStream_t stream{nullptr};

    std::vector<int64_t> imageGridTHW{1, 32, 32};
    int64_t totalSeqLength = imageGridTHW[0] * imageGridTHW[1] * imageGridTHW[2];
    rt::Tensor rotaryPosEmbDevice({totalSeqLength, vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    auto launch = [&]() {
        kernel::initRotaryPosEmbQwenViT(
            rotaryPosEmbDevice, imageGridTHW, mergeSize, 0, rotaryBaseFrequency, scale, stream);
    };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "InitRotaryPosEmbQwen Benchmark: totalSeqLength=" << totalSeqLength
              << ", vitPosEmbDim=" << vitPosEmbDim << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(InitRotaryPosEmbQwen, Benchmark)
{
    BenchmarkInitRotaryPosEmbQwenViT();
}

void TestTransposeToPatchInternVL(int32_t const height, int32_t const width, int32_t const channels = 3,
    int32_t const blockSizeH = 448, int32_t const blockSizeW = 448)
{
    cudaStream_t stream{nullptr};

    // CPU
    std::vector<half> originalImage(height * width * channels);
    uniformFloatInitialization<half>(originalImage, 0, 1);

    std::vector<half> inputPatchesRef(height * width * channels);
    transposeToPatchInternVLReference(
        originalImage, inputPatchesRef, 0, height, width, channels, blockSizeH, blockSizeW);

    // GPU
    rt::Tensor originalImageDevice({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    int32_t const gridH = height / blockSizeH;
    int32_t const gridW = width / blockSizeW;
    int32_t const numBlocks = gridH * gridW;
    rt::Tensor inputPatchesDevice(
        {numBlocks, channels, blockSizeH, blockSizeW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::transposeToPatchInternVLPhi4MM(originalImageDevice, inputPatchesDevice, 0, stream);

    std::vector<half> inputPatches(height * width * channels);
    CUDA_CHECK(cudaMemcpyAsync(inputPatches.data(), inputPatchesDevice.rawPointer(), inputPatches.size() * sizeof(half),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare data
    for (int32_t i = 0; i < inputPatches.size(); ++i)
    {
        ASSERT_TRUE(isclose(inputPatches[i], inputPatchesRef[i], 1e-5, 1e-5));
    }
    std::cout << "transposeToPatchInternVLPhi4MM Accuracy: " << height << "x" << width << "x" << channels
              << ", blockSizeH=" << blockSizeH << ", blockSizeW=" << blockSizeW << std::endl;
}

TEST(transposeToPatchInternVLPhi4MM, Accuracy)
{
    TestTransposeToPatchInternVL(448, 448);
}

void BenchmarkTransposeToPatchInternVL(int32_t const height, int32_t const width, int32_t const channels = 3,
    int32_t const blockSizeH = 448, int32_t const blockSizeW = 448)
{
    cudaStream_t stream{nullptr};

    std::vector<half> originalImage(height * width * channels);
    uniformFloatInitialization<half>(originalImage, 0, 1);

    rt::Tensor originalImageDevice({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(half), cudaMemcpyHostToDevice, stream));

    int32_t const gridH = height / blockSizeH;
    int32_t const gridW = width / blockSizeW;
    int32_t const numBlocks = gridH * gridW;
    rt::Tensor inputPatchesDevice(
        {numBlocks, channels, blockSizeH, blockSizeW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    auto launch = [&]() { kernel::transposeToPatchInternVLPhi4MM(originalImageDevice, inputPatchesDevice, 0, stream); };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launch();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launch();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "transposeToPatchInternVLPhi4MM Benchmark: " << height << "x" << width << "x" << channels
              << ", blockSizeH=" << blockSizeH << ", blockSizeW=" << blockSizeW
              << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(transposeToPatchInternVLPhi4MM, Benchmark)
{
    BenchmarkTransposeToPatchInternVL(448, 448);
    BenchmarkTransposeToPatchInternVL(896, 896);
}

void TestInitFastPosEmbedQwenViT(int64_t const mergeSize = 2, int64_t const numGridPerSide = 48)
{
    cudaStream_t stream{nullptr};

    std::vector<std::vector<int64_t>> imageGridTHWs{{1, 36, 54}, {1, 8, 10}, {1, 32, 20}};
    std::vector<int64_t> cuSeqlens{0};
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        cuSeqlens.push_back(cuSeqlens.back() + imageGridTHWs[i][0] * imageGridTHWs[i][1] * imageGridTHWs[i][2]);
    }
    int64_t totalSeqLength = cuSeqlens.back();

    // CPU reference implementation (from fastPosEmbedInterpolate)
    std::vector<int64_t> fastPosEmbedIdxRef(4 * totalSeqLength);
    std::vector<half> fastPosEmbedWeightRef(4 * totalSeqLength);
    fastPosEmbedInterpolateReference(
        imageGridTHWs, cuSeqlens, fastPosEmbedIdxRef, fastPosEmbedWeightRef, mergeSize, numGridPerSide);

    // GPU tensors
    rt::Tensor fastPosEmbedIdxDevice({4, totalSeqLength}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor fastPosEmbedWeightDevice({4, totalSeqLength}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Call CUDA kernel
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        kernel::initFastPosEmbedQwenViT(fastPosEmbedIdxDevice, fastPosEmbedWeightDevice, imageGridTHWs[i], mergeSize,
            numGridPerSide, cuSeqlens[i], stream);
    }

    // Copy results back to host
    std::vector<int64_t> fastPosEmbedIdxHost(4 * totalSeqLength);
    std::vector<half> fastPosEmbedWeightHost(4 * totalSeqLength);
    CUDA_CHECK(cudaMemcpyAsync(fastPosEmbedIdxHost.data(), fastPosEmbedIdxDevice.rawPointer(),
        fastPosEmbedIdxHost.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fastPosEmbedWeightHost.data(), fastPosEmbedWeightDevice.rawPointer(),
        fastPosEmbedWeightHost.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare indices
    for (int32_t i = 0; i < 4 * totalSeqLength; ++i)
    {
        ASSERT_EQ(fastPosEmbedIdxHost[i], fastPosEmbedIdxRef[i])
            << "Mismatch at index " << i << ": got " << fastPosEmbedIdxHost[i] << ", expected "
            << fastPosEmbedIdxRef[i];
    }

    // Compare weights
    for (int32_t i = 0; i < 4 * totalSeqLength; ++i)
    {
        ASSERT_TRUE(isclose(fastPosEmbedWeightHost[i], fastPosEmbedWeightRef[i], 1e-5, 1e-5))
            << "Mismatch at weight index " << i << ": got " << __half2float(fastPosEmbedWeightHost[i]) << ", expected "
            << __half2float(fastPosEmbedWeightRef[i]);
    }

    std::cout << "InitFastPosEmbedQwenViT Accuracy: totalSeqLength=" << totalSeqLength << ", mergeSize=" << mergeSize
              << ", numGridPerSide=" << numGridPerSide << ", numGrids=" << imageGridTHWs.size() << std::endl;
}

TEST(InitFastPosEmbedQwenViT, Accuracy)
{
    TestInitFastPosEmbedQwenViT();
}

TEST(InitFastPosEmbedCosmos3ViT, AlignCornersFalseAccuracy)
{
    cudaStream_t stream{nullptr};
    int64_t constexpr mergeSize = 2;
    int64_t constexpr numGridPerSide = 16;
    std::vector<std::vector<int64_t>> const grids{{2, 80, 128}, {1, 24, 32}};
    std::vector<int64_t> offsets{0};
    for (auto const& grid : grids)
    {
        offsets.push_back(offsets.back() + grid[0] * grid[1] * grid[2]);
    }
    int64_t const totalSeqLength = offsets.back();
    std::vector<int64_t> expectedIdx(4 * totalSeqLength);
    std::vector<half> expectedWeight(4 * totalSeqLength);

    for (size_t image = 0; image < grids.size(); ++image)
    {
        auto const& grid = grids[image];
        int64_t const T = grid[0];
        int64_t const H = grid[1];
        int64_t const W = grid[2];
        int64_t const llmGridH = H / mergeSize;
        int64_t const llmGridW = W / mergeSize;
        for (int64_t t = 0; t < T; ++t)
        {
            for (int64_t h = 0; h < llmGridH; ++h)
            {
                for (int64_t w = 0; w < llmGridW; ++w)
                {
                    for (int64_t mh = 0; mh < mergeSize; ++mh)
                    {
                        for (int64_t mw = 0; mw < mergeSize; ++mw)
                        {
                            int64_t const grouped = ((h * llmGridW + w) * mergeSize + mh) * mergeSize + mw;
                            int64_t const targetIdx = offsets[image] + t * H * W + grouped;
                            float const sourceH = std::clamp(
                                (static_cast<float>(h * mergeSize + mh) + 0.5F) * numGridPerSide / H - 0.5F, 0.0F,
                                static_cast<float>(numGridPerSide - 1));
                            float const sourceW = std::clamp(
                                (static_cast<float>(w * mergeSize + mw) + 0.5F) * numGridPerSide / W - 0.5F, 0.0F,
                                static_cast<float>(numGridPerSide - 1));
                            int64_t const h0 = static_cast<int64_t>(std::floor(sourceH));
                            int64_t const w0 = static_cast<int64_t>(std::floor(sourceW));
                            int64_t const h1 = std::min(h0 + 1, numGridPerSide - 1);
                            int64_t const w1 = std::min(w0 + 1, numGridPerSide - 1);
                            float const dh = sourceH - h0;
                            float const dw = sourceW - w0;
                            std::array<int64_t, 4> const indices{h0 * numGridPerSide + w0, h0 * numGridPerSide + w1,
                                h1 * numGridPerSide + w0, h1 * numGridPerSide + w1};
                            std::array<float, 4> const weights{
                                (1.0F - dh) * (1.0F - dw), (1.0F - dh) * dw, dh * (1.0F - dw), dh * dw};
                            for (int64_t corner = 0; corner < 4; ++corner)
                            {
                                expectedIdx[corner * totalSeqLength + targetIdx] = indices[corner];
                                expectedWeight[corner * totalSeqLength + targetIdx] = __float2half(weights[corner]);
                            }
                        }
                    }
                }
            }
        }
    }

    rt::Tensor actualIdx({4, totalSeqLength}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor actualWeight({4, totalSeqLength}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    for (size_t image = 0; image < grids.size(); ++image)
    {
        kernel::initFastPosEmbedCosmos3ViT(
            actualIdx, actualWeight, grids[image], mergeSize, numGridPerSide, offsets[image], stream);
    }

    std::vector<int64_t> hostIdx(expectedIdx.size());
    std::vector<half> hostWeight(expectedWeight.size());
    CUDA_CHECK(cudaMemcpyAsync(
        hostIdx.data(), actualIdx.rawPointer(), hostIdx.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostWeight.data(), actualWeight.rawPointer(), hostWeight.size() * sizeof(half),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(hostIdx, expectedIdx);
    for (size_t i = 0; i < expectedWeight.size(); ++i)
    {
        EXPECT_EQ(__half2float(hostWeight[i]), __half2float(expectedWeight[i])) << "Mismatch at weight " << i;
    }
}
TEST(phi4mmPostprocessVisionTokens, Accuracy)
{
    cudaStream_t stream{nullptr};
    // Two images with different block grids
    std::vector<std::pair<int32_t, int32_t>> hwBlocks{{2, 3}};
    int32_t const hidden = 32;

    std::vector<half> srcEmbeds, subGNHost, glbGNHost, dstRef;
    std::vector<int32_t> hBlocksHost, wBlocksHost;
    std::vector<int64_t> srcGlbStartHost, srcSubStartHost, dstOutStartHost, subOutLenHost;
    BuildPhi4mmBatchedInputs(hwBlocks, hidden, srcEmbeds, subGNHost, glbGNHost, hBlocksHost, wBlocksHost,
        srcGlbStartHost, srcSubStartHost, dstOutStartHost, subOutLenHost, dstRef);

    int32_t const numImages = static_cast<int32_t>(hwBlocks.size());
    int64_t const totalRawTokens = static_cast<int64_t>(srcEmbeds.size()) / hidden;
    int64_t const totalOutTokens = static_cast<int64_t>(dstRef.size()) / hidden;

    // Device tensors
    rt::Tensor srcEmbedding({totalRawTokens, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(
        srcEmbedding.rawPointer(), srcEmbeds.data(), srcEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    rt::Tensor dstEmbedding({totalOutTokens, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor hBlocksDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor wBlocksDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor srcGlbStartDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor srcSubStartDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor dstOutStartDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor subOutLenDev({numImages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    CUDA_CHECK(cudaMemcpyAsync(
        hBlocksDev.rawPointer(), hBlocksHost.data(), numImages * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        wBlocksDev.rawPointer(), wBlocksHost.data(), numImages * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(srcGlbStartDev.rawPointer(), srcGlbStartHost.data(), numImages * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(srcSubStartDev.rawPointer(), srcSubStartHost.data(), numImages * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dstOutStartDev.rawPointer(), dstOutStartHost.data(), numImages * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        subOutLenDev.rawPointer(), subOutLenHost.data(), numImages * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    rt::Tensor subGNDev({hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor glbGNDev({hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(
        subGNDev.rawPointer(), subGNHost.data(), hidden * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        glbGNDev.rawPointer(), glbGNHost.data(), hidden * sizeof(half), cudaMemcpyHostToDevice, stream));

    // Launch batched kernel
    kernel::Phi4MMIndex indices{hBlocksDev.dataPointer<int32_t>(), wBlocksDev.dataPointer<int32_t>(),
        srcGlbStartDev.dataPointer<int64_t>(), srcSubStartDev.dataPointer<int64_t>(),
        dstOutStartDev.dataPointer<int64_t>(), subOutLenDev.dataPointer<int64_t>(), numImages, hidden, totalOutTokens};
    kernel::Phi4MMGN gn{subGNDev.dataPointer<half>(), glbGNDev.dataPointer<half>()};
    kernel::phi4mmPostprocessVisionTokens(srcEmbedding, dstEmbedding, indices, gn, totalOutTokens, stream);

    // Copy back and compare
    std::vector<half> dstHost(totalOutTokens * hidden);
    CUDA_CHECK(cudaMemcpyAsync(
        dstHost.data(), dstEmbedding.rawPointer(), dstHost.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < dstHost.size(); ++i)
    {
        ASSERT_TRUE(isclose(dstHost[i], dstRef[i], 1e-5f, 1e-5f)) << "Mismatch at index " << i;
    }
    std::cout << "phi4mmPostprocessVisionTokens Accuracy: numImages=" << numImages << ", hidden=" << hidden
              << ", totalOutTokens=" << totalOutTokens << std::endl;
}

// Fill patterns for the preprocessing tests; checkerboard (2x2 cells) is the highest-frequency case.
enum class ResizeFillPattern
{
    kRANDOM,
    kGRADIENT,
    kCHECKERBOARD,
};

static char const* ResizeFillPatternName(ResizeFillPattern const pattern)
{
    switch (pattern)
    {
    case ResizeFillPattern::kRANDOM: return "random";
    case ResizeFillPattern::kGRADIENT: return "gradient";
    default: return "checkerboard";
    }
}

static void FillResizeInput(std::vector<unsigned char>& data, int32_t const height, int32_t const width,
    int32_t const channels, ResizeFillPattern const pattern)
{
    if (pattern == ResizeFillPattern::kRANDOM)
    {
        uniformIntInitialization<unsigned char>(data, 0, 255);
        return;
    }
    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; ++x)
        {
            for (int32_t c = 0; c < channels; ++c)
            {
                int32_t value{0};
                if (pattern == ResizeFillPattern::kGRADIENT)
                {
                    // Per-channel gradients: c0 along x, c1 along y, c2 along the diagonal.
                    int32_t const gx = (x * 255) / std::max(width - 1, 1);
                    int32_t const gy = (y * 255) / std::max(height - 1, 1);
                    value = c == 0 ? gx : (c == 1 ? gy : (gx + gy) / 2);
                }
                else
                {
                    value = (((x / 2) + (y / 2)) % 2 == 0) ? 0 : 255;
                }
                data[(static_cast<size_t>(y) * width + x) * channels + c] = static_cast<unsigned char>(value);
            }
        }
    }
}

namespace
{

//! Identity normalisation, so a fused output byte is exactly (resized u8) / 255 and can be read back
//! in u8 code units for comparison against a u8 golden.
constexpr std::array<float, 3> kUnitMean{0.0F, 0.0F, 0.0F};
constexpr std::array<float, 3> kUnitStd{1.0F, 1.0F, 1.0F};

//! Wrap a host image as pinned packed RGB8. rt::Tensor's kCPU allocation is cudaMallocHost, which is
//! what the fused path requires of a host source.
rt::imageUtils::ImageData MakePinnedRgbImage(
    std::vector<unsigned char> const& pixels, int64_t const frames, int64_t const height, int64_t const width)
{
    rt::Tensor tensor({frames, height, width, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    std::memcpy(tensor.rawPointer(), pixels.data(), pixels.size());
    return rt::imageUtils::ImageData(std::move(tensor));
}

//! Read a fused output frame back as u8 code units, undoing the normalisation the call applied. The
//! kernel quantises the filtered value to u8 before normalising, so `value * std + mean` recovers that
//! code unit up to the fp16 store, which is at most 255 * 2^-11 = 0.125 of a unit whatever the std.
//! Elements are RGB interleaved, so element i carries channel i % 3.
std::vector<unsigned char> ReadBackAsU8(rt::Tensor const& dst, size_t const elems, size_t const offset = 0,
    std::array<float, 3> const& imageMean = kUnitMean, std::array<float, 3> const& imageStd = kUnitStd)
{
    std::vector<half> raw(elems);
    CUDA_CHECK(cudaMemcpy(
        raw.data(), static_cast<half const*>(dst.rawPointer()) + offset, elems * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<unsigned char> out(elems);
    for (size_t i = 0; i < elems; ++i)
    {
        size_t const channel = i % 3;
        float const code = (__half2float(raw[i]) * imageStd[channel] + imageMean[channel]) * 255.0F;
        out[i] = static_cast<unsigned char>(std::lround(std::min(std::max(code, 0.0F), 255.0F)));
    }
    return out;
}

//! |diff| distribution against a u8 golden: the share within one code unit, the worst single
//! deviation, and the signed mean that exposes a systematic rounding bias.
struct DiffStats
{
    double within1Lsb;
    double meanSignedDiff;
    int32_t maxDiff;
    std::array<int64_t, 256> histogram;
};

DiffStats CompareU8(std::vector<unsigned char> const& got, unsigned char const* ref)
{
    DiffStats s{0.0, 0.0, 0, {}};
    int64_t signedDiffSum = 0;
    for (size_t i = 0; i < got.size(); ++i)
    {
        int32_t const signedDiff = static_cast<int32_t>(got[i]) - static_cast<int32_t>(ref[i]);
        int32_t const diff = std::abs(signedDiff);
        ++s.histogram[diff];
        s.maxDiff = std::max(s.maxDiff, diff);
        signedDiffSum += signedDiff;
    }
    s.within1Lsb = static_cast<double>(s.histogram[0] + s.histogram[1]) / static_cast<double>(got.size());
    s.meanSignedDiff = static_cast<double>(signedDiffSum) / static_cast<double>(got.size());
    return s;
}

void PrintDiffStats(char const* label, DiffStats const& s)
{
    std::cout << label << ": within1Lsb=" << s.within1Lsb * 100.0 << "%, maxDiff=" << s.maxDiff
              << ", bias=" << s.meanSignedDiff << ", hist=[";
    for (int32_t d = 0; d <= s.maxDiff; ++d)
    {
        std::cout << (d == 0 ? "" : " ") << d << ":" << s.histogram[d];
    }
    std::cout << "]" << std::endl;
}

} // namespace

// Fused RGB8 preprocessing vs the CPU rt::imageUtils::resizeImage golden (stbir CATMULLROM +
// EDGE_CLAMP). The float summation order differs, so u8 outputs aren't bit-exact; asserted on the
// |diff| distribution.
void TestFusedResizeRgb(int32_t const inHeight, int32_t const inWidth, int32_t const outHeight, int32_t const outWidth,
    ResizeFillPattern const pattern, std::array<float, 3> const& imageMean = kUnitMean,
    std::array<float, 3> const& imageStd = kUnitStd)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t channels = 3;

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    FillResizeInput(input, inHeight, inWidth, channels, pattern);

    rt::imageUtils::ImageData inputImage = MakePinnedRgbImage(input, 1, inHeight, inWidth);
    rt::Tensor refTensor({1, outHeight, outWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    rt::imageUtils::ImageData resizedRef(std::move(refTensor));
    auto const& resizedRefView = rt::imageUtils::resizeImage(
        inputImage, resizedRef, outWidth, outHeight, rt::imageUtils::InterpolationMode::kBICUBIC);

    rt::Tensor dst({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(inputImage, 0, 1, imageMean, imageStd, dst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const elems = static_cast<size_t>(outHeight) * outWidth * channels;
    DiffStats const s = CompareU8(ReadBackAsU8(dst, elems, 0, imageMean, imageStd), resizedRefView.data());

    // Bounds carry a margin over stbir: the GPU's float summation order differs, so bytes aren't exact.
    EXPECT_GE(s.within1Lsb, 0.995);
    EXPECT_LE(s.maxDiff, 2);
    if (pattern != ResizeFillPattern::kCHECKERBOARD)
    {
        // The mean-signed-diff bound catches systematic rounding bias (e.g. truncation vs round-to-nearest).
        // Checkerboard is exempt: its exact-halfway values (127.5) tie-break differently between stbir and
        // the GPU, which the |diff| bounds above already cap at 1 LSB.
        EXPECT_LE(std::abs(s.meanSignedDiff), 0.05);
    }

    std::cout << "FusedPreprocessRgb Accuracy: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << ", pattern=" << ResizeFillPatternName(pattern) << " -- ";
    PrintDiffStats("stats", s);
}

//! Milliseconds per call of `fn`, after a warmup, measured with events on `stream`.
template <typename Fn>
float TimePerCall(cudaStream_t stream, Fn const& fn)
{
    constexpr int32_t numWarmup = 10;
    constexpr int32_t numBenchIter = 100;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        fn();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        fn();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime / numBenchIter;
}

void BenchmarkFusedResizeRgb(
    int32_t const inHeight, int32_t const inWidth, int32_t const outHeight, int32_t const outWidth)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t channels = 3;

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    uniformIntInitialization<unsigned char>(input, 0, 255);
    rt::imageUtils::ImageData inputImage = MakePinnedRgbImage(input, 1, inHeight, inWidth);
    rt::Tensor dst({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    float const elapsedTime = TimePerCall(stream, [&]() {
        rt::imageUtils::resizeAndNormalizeToRgb(
            inputImage, 0, 1, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
    });
    std::cout << "FusedPreprocessRgb Benchmark: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << ", time=" << elapsedTime << " ms" << std::endl;
}

//! One batched call against the same frames one launch at a time, which is what a source whose frame
//! stride exceeds the plane's footprint still costs.
void BenchmarkFusedResizeRgbVideo(int32_t const inHeight, int32_t const inWidth, int32_t const outHeight,
    int32_t const outWidth, int32_t const frames)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t channels = 3;

    std::vector<unsigned char> input(static_cast<size_t>(frames) * inHeight * inWidth * channels);
    uniformIntInitialization<unsigned char>(input, 0, 255);
    rt::imageUtils::ImageData video = MakePinnedRgbImage(input, frames, inHeight, inWidth);
    rt::Tensor dst({frames, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor slot({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    float const batched = TimePerCall(stream, [&]() {
        rt::imageUtils::resizeAndNormalizeToRgb(
            video, 0, frames, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
    });
    float const perFrame = TimePerCall(stream, [&]() {
        for (int32_t f = 0; f < frames; ++f)
        {
            rt::imageUtils::resizeAndNormalizeToRgb(
                video, f, 1, kUnitMean, kUnitStd, slot, outHeight, outWidth, stream);
        }
    });

    std::cout << "FusedPreprocessRgb VideoBenchmark: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << ", frames=" << frames << ", batched=" << batched << " ms, per-frame=" << perFrame
              << " ms, speedup=" << perFrame / batched << "x" << std::endl;
}

constexpr std::array<ResizeFillPattern, 3> kResizeFillPatterns{
    ResizeFillPattern::kRANDOM, ResizeFillPattern::kGRADIENT, ResizeFillPattern::kCHECKERBOARD};

TEST(FusedPreprocessRgb, AccuracyUpscale)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(320, 480, 640, 960, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyDownscale)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(1024, 1024, 512, 512, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyNonIntegerRatio)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(747, 1000, 608, 832, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyAsymmetricAxes)
{
    // Height upscales 1.5x while width downscales 2x.
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(512, 1536, 768, 768, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyExtremeAspect)
{
    // ~6:1 aspect ratio (wide-banner-shaped input).
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(294, 1790, 160, 960, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyOddSizes)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(331, 477, 123, 209, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyTinyEdge)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(1, 512, 1, 256, pattern);
        TestFusedResizeRgb(512, 1, 256, 1, pattern);
    }
}

TEST(FusedPreprocessRgb, AccuracyLargeFactorDownscale)
{
    // >= 4x downscale, the widest anti-alias filter regime.
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(2160, 3840, 512, 960, pattern);
    }
}

// Past roughly 4.3x the vertical support outgrows the shared memory stage and is consumed in more
// than one chunk, a path the geometries above stay just short of.
TEST(FusedPreprocessRgb, AccuracyMultiChunkSupport)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestFusedResizeRgb(512, 512, 96, 96, pattern);
        TestFusedResizeRgb(2160, 3840, 448, 448, pattern);
    }
}

// A source already at the output size must come through unchanged: at scale 1 the Catmull-Rom kernel
// is interpolating, so every tap but the centre carries zero weight.
TEST(FusedPreprocessRgb, IdentitySizeReproducesSource)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 271, width = 149, channels = 3;

    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels);
    FillResizeInput(input, height, width, channels, ResizeFillPattern::kRANDOM);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);

    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> const got = ReadBackAsU8(dst, input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        ASSERT_EQ(got[i], input[i]) << "identity-size preprocessing altered byte " << i;
    }
    std::cout << "FusedPreprocessRgb IdentitySize: " << height << "x" << width << " reproduced exactly." << std::endl;
}

// Each source frame must land in its own output slot and leave the others alone.
TEST(FusedPreprocessRgb, MultiFrameSlotsAreIndependent)
{
    cudaStream_t stream{nullptr};
    int32_t const inHeight = 480, inWidth = 640, outHeight = 512, outWidth = 960, channels = 3;
    int32_t const numFrames = 3;
    size_t const inFrameElems = static_cast<size_t>(inHeight) * inWidth * channels;
    size_t const outFrameElems = static_cast<size_t>(outHeight) * outWidth * channels;

    std::array<ResizeFillPattern, 3> const patterns{
        ResizeFillPattern::kGRADIENT, ResizeFillPattern::kCHECKERBOARD, ResizeFillPattern::kRANDOM};
    std::vector<unsigned char> input(inFrameElems * numFrames);
    for (int32_t f = 0; f < numFrames; ++f)
    {
        std::vector<unsigned char> frame(inFrameElems);
        FillResizeInput(frame, inHeight, inWidth, channels, patterns[f]);
        std::memcpy(input.data() + f * inFrameElems, frame.data(), inFrameElems);
    }
    rt::imageUtils::ImageData video = MakePinnedRgbImage(input, numFrames, inHeight, inWidth);

    rt::Tensor batched({numFrames, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(
        video, 0, numFrames, kUnitMean, kUnitStd, batched, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t f = 0; f < numFrames; ++f)
    {
        rt::imageUtils::ImageData single = MakePinnedRgbImage(
            std::vector<unsigned char>(input.begin() + f * inFrameElems, input.begin() + (f + 1) * inFrameElems), 1,
            inHeight, inWidth);
        rt::Tensor isolated({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::imageUtils::resizeAndNormalizeToRgb(
            single, 0, 1, kUnitMean, kUnitStd, isolated, outHeight, outWidth, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<unsigned char> const slot = ReadBackAsU8(batched, outFrameElems, f * outFrameElems);
        std::vector<unsigned char> const ref = ReadBackAsU8(isolated, outFrameElems);
        for (size_t k = 0; k < outFrameElems; ++k)
        {
            ASSERT_EQ(slot[k], ref[k]) << "frame " << f << " element " << k << " differs from the isolated call";
        }
    }
    std::cout << "FusedPreprocessRgb MultiFrame: " << numFrames << " frames, each slot identical to an isolated call."
              << std::endl;
}

// A frame range that starts past frame 0 must read that frame and write slot 0.
TEST(FusedPreprocessRgb, FrameRangeSelectsSource)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 64, width = 96, channels = 3, numFrames = 3;
    size_t const frameElems = static_cast<size_t>(height) * width * channels;

    std::vector<unsigned char> input(frameElems * numFrames);
    for (int32_t f = 0; f < numFrames; ++f)
    {
        std::fill(input.begin() + f * frameElems, input.begin() + (f + 1) * frameElems,
            static_cast<unsigned char>(40 * (f + 1)));
    }
    rt::imageUtils::ImageData video = MakePinnedRgbImage(input, numFrames, height, width);

    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(video, 2, 1, kUnitMean, kUnitStd, dst, height, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> const got = ReadBackAsU8(dst, frameElems);
    for (size_t i = 0; i < frameElems; ++i)
    {
        ASSERT_EQ(got[i], 120) << "frame selection read the wrong source frame at byte " << i;
    }

    // A multi-frame range is one batched launch, so the offset applies to its first frame only.
    rt::Tensor pair({2, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(video, 1, 2, kUnitMean, kUnitStd, pair, height, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t f = 0; f < 2; ++f)
    {
        std::vector<unsigned char> const slot = ReadBackAsU8(pair, frameElems, static_cast<size_t>(f) * frameElems);
        auto const expected = static_cast<unsigned char>(40 * (f + 2));
        for (size_t i = 0; i < frameElems; ++i)
        {
            ASSERT_EQ(slot[i], expected) << "batched range slot " << f << " read the wrong source frame at byte " << i;
        }
    }
}

// The folded normalisation is (v / 255 - mean) / std, evaluated as two divisions.
TEST(FusedPreprocessRgb, NormalisationMatchesTheSeparateStep)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 128, width = 192, channels = 3;
    std::array<float, 3> const mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> const stdDev{0.229F, 0.224F, 0.225F};

    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels);
    FillResizeInput(input, height, width, channels, ResizeFillPattern::kRANDOM);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);

    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, mean, stdDev, dst, height, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> got(input.size());
    CUDA_CHECK(cudaMemcpy(got.data(), dst.rawPointer(), got.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < input.size(); ++i)
    {
        half const expected = __float2half((input[i] / 255.0f - mean[i % channels]) / stdDev[i % channels]);
        ASSERT_TRUE(isclose(got[i], expected, 1e-5, 1e-5)) << "normalisation differs at element " << i;
    }
    std::cout << "FusedPreprocessRgb Normalisation: " << height << "x" << width << " matches (v/255 - mean)/std."
              << std::endl;
}

// Extreme mean/std push the result outside the half range on one side and to a denormal on the other;
// the cast is the only place that can round them wrongly.
TEST(FusedPreprocessRgb, Fp16RoundingAtExtremes)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 8, width = 8, channels = 3;
    std::array<float, 3> const mean{-100.0F, 0.5F, 1.0F};
    std::array<float, 3> const stdDev{1e-3F, 1.0F, 1e3F};

    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels);
    FillResizeInput(input, height, width, channels, ResizeFillPattern::kGRADIENT);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);

    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, mean, stdDev, dst, height, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> got(input.size());
    CUDA_CHECK(cudaMemcpy(got.data(), dst.rawPointer(), got.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < input.size(); ++i)
    {
        half const expected = __float2half((input[i] / 255.0f - mean[i % channels]) / stdDev[i % channels]);
        ASSERT_EQ(__half2float(got[i]), __half2float(expected)) << "half cast differs at element " << i;
    }
}

TEST(FusedPreprocessRgb, RejectsNonPositiveDims)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 32, width = 32, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, 0, width, stream),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, -4, stream),
        std::runtime_error);
}

TEST(FusedPreprocessRgb, RejectsFrameRangeOutsideTheImage)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 32, width = 32, channels = 3, numFrames = 2;
    std::vector<unsigned char> input(static_cast<size_t>(numFrames) * height * width * channels, 7);
    rt::imageUtils::ImageData video = MakePinnedRgbImage(input, numFrames, height, width);
    rt::Tensor dst({numFrames, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(video, 1, 2, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(video, -1, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(video, 0, 0, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

TEST(FusedPreprocessRgb, RejectsBadDestination)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 32, width = 32, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);

    rt::Tensor hostDst({1, height, width, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF);
    EXPECT_THROW(
        rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, hostDst, height, width, stream),
        std::runtime_error);

    rt::Tensor u8Dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    EXPECT_THROW(
        rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, u8Dst, height, width, stream),
        std::runtime_error);

    rt::Tensor smallDst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(
                     image, 0, 1, kUnitMean, kUnitStd, smallDst, height * 2, width * 2, stream),
        std::runtime_error);
}

// A zero std component would divide that channel by zero and write inf, so the entry point rejects it.
TEST(FusedPreprocessRgb, RejectsZeroStd)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 16, width = 16, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    std::array<float, 3> const zeroStd{1.0F, 0.0F, 1.0F};
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, zeroStd, dst, height, width, stream),
        std::runtime_error);
}

// A pitch below the plane's valid row width reads the wrong rows while staying inside the allocation,
// so it is refused at the entry point.
TEST(FusedPreprocessRgb, RejectsPitchBelowRowWidth)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 16, width = 16, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    image.layout.pitchBytes[0] = width * channels - 1;
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

// A block-linear frame is sampled through texture objects, so a buffer-backed one carries its pixels
// in a place the format says nothing about and cannot be read at all.
TEST(FusedPreprocessRgb, RejectsBlockLinearWithoutTexturePlanes)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 16, width = 16, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    image.layout.format = rt::imageUtils::ImageFormat::kNV12BL;
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

// `layout` and the extents are public members of an open aggregate, so a frame range that fitted when
// the image was wrapped can be widened afterwards; the entry point re-derives the reach rather than
// trust the wrap.
TEST(FusedPreprocessRgb, RejectsALayoutReachingPastTheBuffer)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 32, width = 32, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    image.height = height * 2;
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

// The caller may release its source frames from the catch, so the stream has to be idle by the time
// the exception arrives. The first call leaves a 4K resize in flight; without the drain the stream is
// still busy when the second one throws.
TEST(FusedPreprocessRgb, StreamIsDrainedBeforeAnExceptionLeaves)
{
    cudaStream_t stream{nullptr};
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t const height = 2160, width = 3840, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels, 7);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, height, width);
    rt::Tensor dst({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, 0, width, stream),
        std::runtime_error);
    EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(FusedPreprocessRgb, Benchmark)
{
    // 1080p and 4K frames, both resized to 512x960.
    BenchmarkFusedResizeRgb(1080, 1920, 512, 960);
    BenchmarkFusedResizeRgb(2160, 3840, 512, 960);
    // Video at the frame counts the ViT runners feed a token profile. The batch saves one launch per
    // frame, so the share it saves grows as the frame shrinks.
    BenchmarkFusedResizeRgbVideo(720, 1280, 512, 960, 32);
    BenchmarkFusedResizeRgbVideo(448, 448, 224, 224, 32);
}

namespace
{

//! Same cubic the kernel applies, evaluated in double so the reference carries no float rounding of
//! its own.
double CatmullRomWeightRef(double x)
{
    x = std::abs(x);
    if (x < 1.0)
    {
        return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    }
    if (x < 2.0)
    {
        return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    }
    return 0.0;
}

//! Anti-aliased Catmull-Rom over one interleaved plane, quantised half-up at the end. `shiftX` is the
//! horizontal sampling phase in source samples.
std::vector<unsigned char> ResizePlaneRef(unsigned char const* src, int64_t const srcH, int64_t const srcW,
    int64_t const srcPitch, int64_t const channels, int64_t const outH, int64_t const outW, double const shiftX)
{
    double const sx = static_cast<double>(srcW) / static_cast<double>(outW);
    double const sy = static_cast<double>(srcH) / static_cast<double>(outH);
    double const fsx = sx > 1.0 ? sx : 1.0;
    double const fsy = sy > 1.0 ? sy : 1.0;

    std::vector<unsigned char> out(static_cast<size_t>(outH * outW * channels));
    for (int64_t oy = 0; oy < outH; ++oy)
    {
        double const cy = (oy + 0.5) * sy;
        int64_t const y0 = static_cast<int64_t>(std::ceil(cy - 2.0 * fsy - 0.5));
        int64_t const y1 = static_cast<int64_t>(std::floor(cy + 2.0 * fsy - 0.5));
        for (int64_t ox = 0; ox < outW; ++ox)
        {
            double const cx = (ox + 0.5) * sx + shiftX;
            int64_t const x0 = static_cast<int64_t>(std::ceil(cx - 2.0 * fsx - 0.5));
            int64_t const x1 = static_cast<int64_t>(std::floor(cx + 2.0 * fsx - 0.5));
            for (int64_t c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                double wsum = 0.0;
                for (int64_t iy = y0; iy <= y1; ++iy)
                {
                    double const wy = CatmullRomWeightRef((iy + 0.5 - cy) / fsy);
                    int64_t const sy0 = iy < 0 ? 0 : (iy >= srcH ? srcH - 1 : iy);
                    for (int64_t ix = x0; ix <= x1; ++ix)
                    {
                        double const wx = CatmullRomWeightRef((ix + 0.5 - cx) / fsx);
                        int64_t const sx0 = ix < 0 ? 0 : (ix >= srcW ? srcW - 1 : ix);
                        acc += wy * wx * static_cast<double>(src[sy0 * srcPitch + sx0 * channels + c]);
                        wsum += wy * wx;
                    }
                }
                double v = wsum > 0.0 ? acc / wsum : 0.0;
                v = v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
                out[static_cast<size_t>((oy * outW + ox) * channels + c)] = static_cast<unsigned char>(v + 0.5);
            }
        }
    }
    return out;
}

struct YuvCoeffsRef
{
    double yScale, yOffset, crToR, cbToG, crToG, cbToB;
};

YuvCoeffsRef MakeCoeffsRef(rt::imageUtils::ColorStandard const standard, rt::imageUtils::ColorRange const range)
{
    double kr = 0.0, kb = 0.0;
    switch (standard)
    {
    case rt::imageUtils::ColorStandard::kBt601:
        kr = 0.299;
        kb = 0.114;
        break;
    case rt::imageUtils::ColorStandard::kBt709:
        kr = 0.2126;
        kb = 0.0722;
        break;
    default:
        kr = 0.2627;
        kb = 0.0593;
        break;
    }
    bool const limited = range == rt::imageUtils::ColorRange::kLimited;
    double const kg = 1.0 - kr - kb;
    double const s = limited ? 224.0 : 255.0;
    return YuvCoeffsRef{limited ? 255.0 / 219.0 : 1.0, limited ? 16.0 : 0.0, 255.0 * 2.0 * (1.0 - kr) / s,
        -255.0 * 2.0 * kb * (1.0 - kb) / (kg * s), -255.0 * 2.0 * kr * (1.0 - kr) / (kg * s),
        255.0 * 2.0 * (1.0 - kb) / s};
}

//! Resize in the NV12 domain and then convert, which is the order the fused kernel applies. Chroma is
//! resized onto the output chroma grid ((out + 1) / 2) and read back at (x / 2, y / 2), so a 2x2 luma
//! block shares one chroma sample on both sides of the resize.
std::vector<unsigned char> Nv12ChainRef(std::vector<unsigned char> const& y, std::vector<unsigned char> const& uv,
    int64_t const srcH, int64_t const srcW, int64_t const outH, int64_t const outW, YuvCoeffsRef const& coeffs)
{
    int64_t const srcChromaH = (srcH + 1) / 2;
    int64_t const srcChromaW = (srcW + 1) / 2;
    int64_t const outChromaH = (outH + 1) / 2;
    int64_t const outChromaW = (outW + 1) / 2;

    std::vector<unsigned char> const yOut = ResizePlaneRef(y.data(), srcH, srcW, srcW, 1, outH, outW, 0.0);
    std::vector<unsigned char> const uvOut
        = ResizePlaneRef(uv.data(), srcChromaH, srcChromaW, srcChromaW * 2, 2, outChromaH, outChromaW, 0.25);

    std::vector<unsigned char> rgb(static_cast<size_t>(outH * outW * 3));
    for (int64_t oy = 0; oy < outH; ++oy)
    {
        for (int64_t ox = 0; ox < outW; ++ox)
        {
            size_t const uvIdx = static_cast<size_t>((oy / 2) * outChromaW + (ox / 2)) * 2;
            double const luma = coeffs.yScale * (static_cast<double>(yOut[oy * outW + ox]) - coeffs.yOffset);
            double const cb = static_cast<double>(uvOut[uvIdx]) - 128.0;
            double const cr = static_cast<double>(uvOut[uvIdx + 1]) - 128.0;
            double const channel[3]{
                luma + coeffs.crToR * cr, luma + coeffs.cbToG * cb + coeffs.crToG * cr, luma + coeffs.cbToB * cb};
            for (int64_t c = 0; c < 3; ++c)
            {
                double const v = channel[c] < 0.0 ? 0.0 : (channel[c] > 255.0 ? 255.0 : channel[c]);
                rgb[static_cast<size_t>((oy * outW + ox) * 3 + c)] = static_cast<unsigned char>(v + 0.5);
            }
        }
    }
    return rgb;
}

//! Wrap two pitch-linear planes as a pinned NV12 frame. The pitches may exceed the valid row width, so
//! the padding is what a decoder or ISP would leave behind.
rt::imageUtils::ImageData MakePinnedNv12Image(std::vector<unsigned char> const& y, std::vector<unsigned char> const& uv,
    int64_t const height, int64_t const width, int64_t const yPitch, int64_t const uvPitch,
    rt::imageUtils::ColorStandard const standard, rt::imageUtils::ColorRange const range, int64_t const frames = 1)
{
    int64_t const chromaRows = (height + 1) / 2;
    int64_t const chromaRowBytes = 2 * ((width + 1) / 2);
    int64_t const yBytes = yPitch * height;
    int64_t const uvBytes = uvPitch * chromaRows;
    int64_t const frameBytes = yBytes + uvBytes;

    auto buffer = std::make_shared<rt::Tensor>(
        rt::Tensor({frames * frameBytes}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8));
    auto* const base = static_cast<unsigned char*>(buffer->rawPointer());
    std::memset(base, 0, static_cast<size_t>(frames * frameBytes));
    for (int64_t f = 0; f < frames; ++f)
    {
        unsigned char* const frame = base + f * frameBytes;
        for (int64_t r = 0; r < height; ++r)
        {
            std::memcpy(frame + r * yPitch, y.data() + (f * height + r) * width, static_cast<size_t>(width));
        }
        for (int64_t r = 0; r < chromaRows; ++r)
        {
            std::memcpy(frame + yBytes + r * uvPitch, uv.data() + (f * chromaRows + r) * chromaRowBytes,
                static_cast<size_t>(chromaRowBytes));
        }
    }

    rt::imageUtils::ImageLayout layout{};
    layout.format = rt::imageUtils::ImageFormat::kNV12PL;
    layout.colorStandard = standard;
    layout.colorRange = range;
    layout.planeOffsetBytes = {0, yBytes};
    layout.pitchBytes = {yPitch, uvPitch};
    layout.frameStrideBytes = {frameBytes, frameBytes};
    return rt::imageUtils::wrapImageBuffer(buffer, layout, width, height, frames);
}

//! Flat luma with chroma alternating per 2x2 block. Sampling chroma on the luma grid instead of the
//! chroma grid shifts this pattern by half a chroma sample, which a natural image hides inside one or
//! two code units and this does not.
void FillChromaCheckerboard(
    std::vector<unsigned char>& y, std::vector<unsigned char>& uv, int64_t const height, int64_t const width)
{
    std::fill(y.begin(), y.end(), static_cast<unsigned char>(128));
    int64_t const chromaH = (height + 1) / 2;
    int64_t const chromaW = (width + 1) / 2;
    for (int64_t r = 0; r < chromaH; ++r)
    {
        for (int64_t c = 0; c < chromaW; ++c)
        {
            bool const even = ((r + c) % 2) == 0;
            uv[static_cast<size_t>(r * chromaW + c) * 2] = even ? 16 : 240;
            uv[static_cast<size_t>(r * chromaW + c) * 2 + 1] = even ? 240 : 16;
        }
    }
}

} // namespace

// NV12 pitch-linear preprocessing vs a double-precision resize-then-convert reference. Only the
// arithmetic width and the summation order differ, so the bounds are the resize family's.
void TestFusedNv12(int64_t const inHeight, int64_t const inWidth, int64_t const outHeight, int64_t const outWidth,
    bool const chromaCheckerboard, rt::imageUtils::ColorStandard const standard, rt::imageUtils::ColorRange const range,
    int64_t const yPadBytes = 0, int64_t const uvPadBytes = 0, std::array<float, 3> const& imageMean = kUnitMean,
    std::array<float, 3> const& imageStd = kUnitStd)
{
    cudaStream_t stream{nullptr};
    int64_t const chromaRowBytes = 2 * ((inWidth + 1) / 2);

    std::vector<unsigned char> y(static_cast<size_t>(inHeight * inWidth));
    std::vector<unsigned char> uv(static_cast<size_t>(((inHeight + 1) / 2) * chromaRowBytes));
    if (chromaCheckerboard)
    {
        FillChromaCheckerboard(y, uv, inHeight, inWidth);
    }
    else
    {
        uniformIntInitialization<unsigned char>(y, 0, 255);
        uniformIntInitialization<unsigned char>(uv, 0, 255);
    }

    rt::imageUtils::ImageData image = MakePinnedNv12Image(
        y, uv, inHeight, inWidth, inWidth + yPadBytes, chromaRowBytes + uvPadBytes, standard, range);

    rt::Tensor dst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, imageMean, imageStd, dst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> const ref
        = Nv12ChainRef(y, uv, inHeight, inWidth, outHeight, outWidth, MakeCoeffsRef(standard, range));
    DiffStats const s = CompareU8(ReadBackAsU8(dst, ref.size(), 0, imageMean, imageStd), ref.data());

    // The chain quantises the resized Y and CbCr planes to code units before converting, so a
    // one-unit disagreement in a plane reaches the output multiplied by the matrix: up to 1.17
    // for luma and 2.15 for chroma across the standards this accepts.
    EXPECT_GE(s.within1Lsb, 0.995);
    EXPECT_LE(s.maxDiff, 4);

    std::cout << "FusedPreprocessNv12 Accuracy: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << (chromaCheckerboard ? ", chroma checkerboard" : ", random")
              << (yPadBytes || uvPadBytes ? ", padded" : "") << " -- ";
    PrintDiffStats("stats", s);
}

// The sampling position of chroma. Reading the chroma plane at (x, y) rather than (x / 2, y / 2)
// compiles, runs and shifts the colours; on this pattern that shift is a full block.
TEST(FusedPreprocessNv12, ChromaPhaseOnTheChromaGrid)
{
    TestFusedNv12(256, 256, 128, 128, /*chromaCheckerboard=*/true, rt::imageUtils::ColorStandard::kBt709,
        rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(256, 256, 256, 256, /*chromaCheckerboard=*/true, rt::imageUtils::ColorStandard::kBt709,
        rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(128, 128, 320, 320, /*chromaCheckerboard=*/true, rt::imageUtils::ColorStandard::kBt709,
        rt::imageUtils::ColorRange::kLimited);
}

// The size of that phase. Left-sited chroma sits a quarter of a chroma sample left of the block centroid
// and the kernel samples a quarter to the right to compensate; the case above takes that quarter from the
// same constant the kernel uses, so it holds whatever the kernel does. A linear ramp does not: Catmull-Rom
// reproduces a linear source exactly, so the resampled chroma is the ramp read at the sampling position,
// which drops kCbStep * kSitingPhase code units when the compensation goes and twice that when it flips.
TEST(FusedPreprocessNv12, ChromaSitingPhaseIsAQuarterSample)
{
    constexpr int64_t kExtent{32};
    constexpr double kCbBase{96.0};
    constexpr double kCbStep{6.0};
    constexpr double kSitingPhase{0.25};
    constexpr auto kStandard = rt::imageUtils::ColorStandard::kBt709;
    constexpr auto kRange = rt::imageUtils::ColorRange::kFull;

    int64_t const chromaExtent = kExtent / 2;
    std::vector<unsigned char> y(static_cast<size_t>(kExtent * kExtent), 128);
    std::vector<unsigned char> uv(static_cast<size_t>(chromaExtent * chromaExtent * 2), 128);
    for (int64_t r = 0; r < chromaExtent; ++r)
    {
        for (int64_t c = 0; c < chromaExtent; ++c)
        {
            uv[static_cast<size_t>((r * chromaExtent + c) * 2)]
                = static_cast<unsigned char>(kCbBase + kCbStep * static_cast<double>(c));
        }
    }

    rt::imageUtils::ImageData image
        = MakePinnedNv12Image(y, uv, kExtent, kExtent, kExtent, chromaExtent * 2, kStandard, kRange);
    rt::Tensor dst({1, kExtent, kExtent, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    cudaStream_t stream{nullptr};
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, kExtent, kExtent, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> const out = ReadBackAsU8(dst, static_cast<size_t>(kExtent * kExtent * 3));

    YuvCoeffsRef const coeffs = MakeCoeffsRef(kStandard, kRange);
    double const luma = coeffs.yScale * (128.0 - coeffs.yOffset);
    int64_t const row = kExtent / 2;
    // Cr is neutral, so blue carries the ramp alone. The two chroma columns at each edge take clamped
    // taps, where a Catmull-Rom gather stops reproducing a line.
    for (int64_t c = 2; c + 2 < chromaExtent; ++c)
    {
        double const cb = kCbBase + kCbStep * (static_cast<double>(c) + kSitingPhase);
        double const blue = luma + coeffs.cbToB * (cb - 128.0);
        EXPECT_NEAR(out[static_cast<size_t>((row * kExtent + 2 * c) * 3 + 2)], blue, 1.5) << "chroma column " << c;
    }
}

TEST(FusedPreprocessNv12, AccuracyAcrossGeometries)
{
    TestFusedNv12(
        480, 640, 224, 224, false, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(
        1080, 1920, 448, 448, false, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(224, 224, 448, 448, false, rt::imageUtils::ColorStandard::kBt601, rt::imageUtils::ColorRange::kFull);
}

// Odd extents make (out + 1) / 2 differ from out / 2 on both axes, and put the last chroma column and
// row half outside the luma grid.
TEST(FusedPreprocessNv12, OddExtents)
{
    TestFusedNv12(
        747, 1000, 331, 209, false, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(101, 99, 51, 49, true, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
}

// The two planes are padded independently. Addressing chroma with the luma pitch reads the wrong rows
// while staying inside the allocation.
TEST(FusedPreprocessNv12, IndependentPlanePitches)
{
    TestFusedNv12(240, 320, 224, 224, false, rt::imageUtils::ColorStandard::kBt709,
        rt::imageUtils::ColorRange::kLimited, /*yPadBytes=*/64, /*uvPadBytes=*/32);
    TestFusedNv12(240, 320, 224, 224, true, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited,
        /*yPadBytes=*/0, /*uvPadBytes=*/96);
}

// An NV12 frame stride spans both planes, so the batch dimension cannot reach frame f and a video
// takes one launch per frame. Each slot must still match that frame preprocessed on its own.
TEST(FusedPreprocessNv12, MultiFrameSlotsAreIndependent)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 120, inWidth = 160, outHeight = 64, outWidth = 96, numFrames = 3;
    int64_t const chromaRows = (inHeight + 1) / 2;
    int64_t const chromaRowBytes = 2 * ((inWidth + 1) / 2);
    size_t const yElems = static_cast<size_t>(inHeight * inWidth);
    size_t const uvElems = static_cast<size_t>(chromaRows * chromaRowBytes);
    size_t const outFrameElems = static_cast<size_t>(outHeight * outWidth * 3);
    auto const standard = rt::imageUtils::ColorStandard::kBt709;
    auto const range = rt::imageUtils::ColorRange::kLimited;

    std::vector<unsigned char> y(yElems * numFrames);
    std::vector<unsigned char> uv(uvElems * numFrames);
    uniformIntInitialization<unsigned char>(y, 0, 255);
    uniformIntInitialization<unsigned char>(uv, 0, 255);

    rt::imageUtils::ImageData video
        = MakePinnedNv12Image(y, uv, inHeight, inWidth, inWidth, chromaRowBytes, standard, range, numFrames);
    rt::Tensor batched({numFrames, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(
        video, 0, numFrames, kUnitMean, kUnitStd, batched, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t f = 0; f < numFrames; ++f)
    {
        size_t const yAt = static_cast<size_t>(f) * yElems;
        size_t const uvAt = static_cast<size_t>(f) * uvElems;
        rt::imageUtils::ImageData single
            = MakePinnedNv12Image(std::vector<unsigned char>(y.begin() + yAt, y.begin() + yAt + yElems),
                std::vector<unsigned char>(uv.begin() + uvAt, uv.begin() + uvAt + uvElems), inHeight, inWidth, inWidth,
                chromaRowBytes, standard, range);
        rt::Tensor isolated({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::imageUtils::resizeAndNormalizeToRgb(
            single, 0, 1, kUnitMean, kUnitStd, isolated, outHeight, outWidth, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<unsigned char> const slot
            = ReadBackAsU8(batched, outFrameElems, static_cast<size_t>(f) * outFrameElems);
        std::vector<unsigned char> const ref = ReadBackAsU8(isolated, outFrameElems);
        for (size_t k = 0; k < outFrameElems; ++k)
        {
            ASSERT_EQ(slot[k], ref[k]) << "frame " << f << " element " << k << " differs from the isolated call";
        }
    }
    std::cout << "FusedPreprocessNv12 MultiFrame: " << numFrames << " frames, each slot identical to an isolated call."
              << std::endl;
}

TEST(FusedPreprocessNv12, EveryColourStandardAndRange)
{
    for (auto const standard : {rt::imageUtils::ColorStandard::kBt601, rt::imageUtils::ColorStandard::kBt709,
             rt::imageUtils::ColorStandard::kBt2020})
    {
        for (auto const range : {rt::imageUtils::ColorRange::kLimited, rt::imageUtils::ColorRange::kFull})
        {
            TestFusedNv12(128, 160, 64, 80, false, standard, range);
        }
    }
}

// A YUV source with no colour metadata decodes to a plausible picture in the wrong colours, so an
// unset standard or range is refused rather than defaulted.
TEST(FusedPreprocessNv12, RejectsUnspecifiedColourMetadata)
{
    cudaStream_t stream{nullptr};
    int64_t const height = 64, width = 64;
    std::vector<unsigned char> y(static_cast<size_t>(height * width), 128);
    std::vector<unsigned char> uv(static_cast<size_t>(((height + 1) / 2) * 2 * ((width + 1) / 2)), 128);

    rt::imageUtils::ImageData image = MakePinnedNv12Image(y, uv, height, width, width, 2 * ((width + 1) / 2),
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    image.layout.colorStandard = rt::imageUtils::ColorStandard::kUnspecified;

    rt::Tensor dst({1, height, width, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

// A frame stride below one frame of its plane overlaps successive frames, so the wrap refuses it.
TEST(FusedPreprocessNv12, RejectsFrameStrideBelowOneFrame)
{
    int64_t const height = 32, width = 32, frames = 2;
    int64_t const chromaRows = (height + 1) / 2;
    int64_t const chromaRowBytes = 2 * ((width + 1) / 2);
    int64_t const yBytes = width * height;
    int64_t const frameBytes = yBytes + chromaRowBytes * chromaRows;

    auto buffer = std::make_shared<rt::Tensor>(
        rt::Tensor({frames * frameBytes}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8));
    rt::imageUtils::ImageLayout layout{};
    layout.format = rt::imageUtils::ImageFormat::kNV12PL;
    layout.colorStandard = rt::imageUtils::ColorStandard::kBt709;
    layout.colorRange = rt::imageUtils::ColorRange::kLimited;
    layout.planeOffsetBytes = {0, yBytes};
    layout.pitchBytes = {width, chromaRowBytes};
    layout.frameStrideBytes = {yBytes - 1, frameBytes};

    EXPECT_THROW(rt::imageUtils::wrapImageBuffer(buffer, layout, width, height, frames), std::runtime_error);
}

// The kernel narrows both plane pitches to int. A chroma plane one row tall reaches no further than
// its valid row width whatever its pitch, so the buffer-capacity check cannot bound it and the
// 32-bit guard is the only thing that does.
TEST(FusedPreprocessNv12, RejectsChromaPitchBeyond32Bits)
{
    cudaStream_t stream{nullptr};
    int64_t const height = 2, width = 32;
    std::vector<unsigned char> y(static_cast<size_t>(height * width), 128);
    std::vector<unsigned char> uv(static_cast<size_t>(((height + 1) / 2) * 2 * ((width + 1) / 2)), 128);

    rt::imageUtils::ImageData image = MakePinnedNv12Image(y, uv, height, width, width, 2 * ((width + 1) / 2),
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    image.layout.pitchBytes[1] = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;

    rt::Tensor dst({1, height, width, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    EXPECT_THROW(rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, height, width, stream),
        std::runtime_error);
}

// Pageable host memory is not addressable from a kernel; wrapping it has to fail at the wrap rather
// than at some later synchronisation point.
TEST(FusedPreprocessNv12, RejectsPageableHostBuffer)
{
    int64_t const height = 32, width = 32;
    int64_t const chromaRowBytes = 2 * ((width + 1) / 2);
    int64_t const yBytes = width * height;
    int64_t const uvBytes = chromaRowBytes * ((height + 1) / 2);

    std::vector<unsigned char> pageable(static_cast<size_t>(yBytes + uvBytes), 128);
    auto buffer = std::make_shared<rt::Tensor>(rt::Tensor(
        pageable.data(), {yBytes + uvBytes}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "pageableNv12"));

    rt::imageUtils::ImageLayout layout{};
    layout.format = rt::imageUtils::ImageFormat::kNV12PL;
    layout.colorStandard = rt::imageUtils::ColorStandard::kBt709;
    layout.colorRange = rt::imageUtils::ColorRange::kLimited;
    layout.planeOffsetBytes = {0, yBytes};
    layout.pitchBytes = {width, chromaRowBytes};
    layout.frameStrideBytes = {yBytes + uvBytes, yBytes + uvBytes};

    EXPECT_THROW(rt::imageUtils::wrapImageBuffer(buffer, layout, width, height, 1), std::runtime_error);
}

// A page-locked host source and a device-resident copy of the same bytes must produce the same result:
// the zero-copy path differs only in where the kernel reads from.
TEST(FusedPreprocessNv12, MappedHostMatchesDeviceResident)
{
    cudaStream_t stream{nullptr};
    int64_t const height = 240, width = 320, outHeight = 224, outWidth = 224;
    int64_t const chromaRowBytes = 2 * ((width + 1) / 2);
    int64_t const yBytes = width * height;
    int64_t const uvBytes = chromaRowBytes * ((height + 1) / 2);

    std::vector<unsigned char> y(static_cast<size_t>(yBytes));
    std::vector<unsigned char> uv(static_cast<size_t>(uvBytes));
    uniformIntInitialization<unsigned char>(y, 0, 255);
    uniformIntInitialization<unsigned char>(uv, 0, 255);

    rt::imageUtils::ImageData hostImage = MakePinnedNv12Image(y, uv, height, width, width, chromaRowBytes,
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);

    auto deviceBuffer = std::make_shared<rt::Tensor>(
        rt::Tensor({yBytes + uvBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8));
    CUDA_CHECK(cudaMemcpy(deviceBuffer->rawPointer(), hostImage.buffer->rawPointer(),
        static_cast<size_t>(yBytes + uvBytes), cudaMemcpyHostToDevice));
    rt::imageUtils::ImageData deviceImage
        = rt::imageUtils::wrapImageBuffer(deviceBuffer, hostImage.layout, width, height, 1);

    rt::Tensor hostDst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor deviceDst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(hostImage, 0, 1, kUnitMean, kUnitStd, hostDst, outHeight, outWidth, stream);
    rt::imageUtils::resizeAndNormalizeToRgb(
        deviceImage, 0, 1, kUnitMean, kUnitStd, deviceDst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const elems = static_cast<size_t>(outHeight * outWidth * 3);
    std::vector<unsigned char> const fromHost = ReadBackAsU8(hostDst, elems);
    std::vector<unsigned char> const fromDevice = ReadBackAsU8(deviceDst, elems);
    for (size_t i = 0; i < elems; ++i)
    {
        ASSERT_EQ(fromHost[i], fromDevice[i]) << "mapped host and device-resident results differ at " << i;
    }
    std::cout << "FusedPreprocessNv12 ZeroCopy: mapped host result identical to device-resident." << std::endl;
}

namespace
{

//! A CUDA array and the texture object over it, released together and in that order.
struct TexturePlane
{
    cudaArray_t array{nullptr};
    cudaTextureObject_t texture{};

    TexturePlane() = default;
    TexturePlane(TexturePlane const&) = delete;
    TexturePlane& operator=(TexturePlane const&) = delete;

    ~TexturePlane()
    {
        if (texture != 0)
        {
            (void) cudaDestroyTextureObject(texture);
        }
        if (array != nullptr)
        {
            (void) cudaFreeArray(array);
        }
    }
};

//! Sampler state a caller can get wrong, so the tests can vary one field at a time.
struct SamplerState
{
    cudaTextureFilterMode filterMode{cudaFilterModePoint};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    int normalizedCoords{0};
    int sRGB{0};
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
};

//! Upload one plane into a CUDA array and view it through a texture object. There is no NvSci here,
//! so the array is allocated rather than imported: what this reaches is the sampling path and the
//! descriptor checks, not the Tegra block order that only a real imported surface carries.
//! `chromaBits` is 0 for luma and 8 for the interleaved CbCr plane, which fixes the texel width.
cudaError_t MakeTexturePlane(TexturePlane& plane, unsigned char const* const src, int64_t const width,
    int64_t const height, int32_t const chromaBits, SamplerState const& sampler = {})
{
    int64_t const rowBytes = width * (chromaBits == 0 ? 1 : 2);
    cudaChannelFormatDesc const channel = cudaCreateChannelDesc(8, chromaBits, 0, 0, cudaChannelFormatKindUnsigned);
    CUDA_CHECK(cudaMallocArray(&plane.array, &channel, static_cast<size_t>(width), static_cast<size_t>(height)));
    CUDA_CHECK(cudaMemcpy2DToArray(plane.array, 0, 0, src, static_cast<size_t>(rowBytes), static_cast<size_t>(rowBytes),
        static_cast<size_t>(height), cudaMemcpyHostToDevice));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = plane.array;

    cudaTextureDesc description{};
    description.addressMode[0] = sampler.addressMode;
    description.addressMode[1] = sampler.addressMode;
    description.filterMode = sampler.filterMode;
    description.readMode = sampler.readMode;
    description.normalizedCoords = sampler.normalizedCoords;
    description.sRGB = sampler.sRGB;

    cudaError_t const status = cudaCreateTextureObject(&plane.texture, &resource, &description, nullptr);
    if (status != cudaSuccess)
    {
        (void) cudaGetLastError();
    }
    return status;
}

//! The raw half output, for comparisons that have to be exact rather than close.
std::vector<uint16_t> ReadBackRawHalf(rt::Tensor const& dst, size_t const elems)
{
    std::vector<uint16_t> raw(elems);
    CUDA_CHECK(cudaMemcpy(raw.data(), dst.rawPointer(), elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return raw;
}

//! The same pixels wrapped both ways, so a test can hold the two paths against each other. The
//! pitch-linear copy keeps its planes tightly packed, since padding has no counterpart in an array.
struct Nv12PathPair
{
    std::vector<unsigned char> y;
    std::vector<unsigned char> uv;
    TexturePlane luma;
    TexturePlane chroma;

    Nv12PathPair(int64_t const height, int64_t const width, bool const chromaCheckerboard)
    {
        int64_t const chromaW = (width + 1) / 2;
        int64_t const chromaH = (height + 1) / 2;
        y.resize(static_cast<size_t>(height * width));
        uv.resize(static_cast<size_t>(chromaH * chromaW * 2));
        if (chromaCheckerboard)
        {
            FillChromaCheckerboard(y, uv, height, width);
        }
        else
        {
            uniformIntInitialization<unsigned char>(y, 0, 255);
            uniformIntInitialization<unsigned char>(uv, 0, 255);
        }
    }
};

} // namespace

// Block-linear against the same double-precision chain the pitch-linear tests use, and against the
// pitch-linear result itself. The two paths differ only in how a tap reaches a sample, so almost
// every element matches bit for bit; the exceptions are the long accumulations of a heavy downscale,
// where the two template instantiations contract their weighted sums differently. That divergence is
// a few elements in a million and stays well inside the bound the chain itself carries.
void TestNv12BlockLinearMatchesPitchLinear(int64_t const inHeight, int64_t const inWidth, int64_t const outHeight,
    int64_t const outWidth, bool const chromaCheckerboard = false,
    rt::imageUtils::ColorStandard const standard = rt::imageUtils::ColorStandard::kBt709,
    rt::imageUtils::ColorRange const range = rt::imageUtils::ColorRange::kLimited,
    std::array<float, 3> const& imageMean = kUnitMean, std::array<float, 3> const& imageStd = kUnitStd)
{
    cudaStream_t stream{nullptr};
    int64_t const chromaW = (inWidth + 1) / 2;
    int64_t const chromaH = (inHeight + 1) / 2;

    Nv12PathPair frame(inHeight, inWidth, chromaCheckerboard);
    CUDA_CHECK(MakeTexturePlane(frame.luma, frame.y.data(), inWidth, inHeight, 0));
    CUDA_CHECK(MakeTexturePlane(frame.chroma, frame.uv.data(), chromaW, chromaH, 8));

    rt::imageUtils::ImageData const pitchLinear
        = MakePinnedNv12Image(frame.y, frame.uv, inHeight, inWidth, inWidth, 2 * chromaW, standard, range);
    rt::imageUtils::ImageData const blockLinear = rt::imageUtils::wrapImageTexture(
        frame.luma.texture, frame.chroma.texture, standard, range, inWidth, inHeight);

    rt::Tensor plDst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor blDst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(pitchLinear, 0, 1, imageMean, imageStd, plDst, outHeight, outWidth, stream);
    rt::imageUtils::resizeAndNormalizeToRgb(blockLinear, 0, 1, imageMean, imageStd, blDst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const elems = static_cast<size_t>(outHeight * outWidth * 3);

    // The same bound the pitch-linear accuracy family carries: the chain quantises each plane to code
    // units before converting, so a one-unit disagreement reaches the output through the matrix.
    std::vector<unsigned char> const reference
        = Nv12ChainRef(frame.y, frame.uv, inHeight, inWidth, outHeight, outWidth, MakeCoeffsRef(standard, range));
    DiffStats const againstReference
        = CompareU8(ReadBackAsU8(blDst, reference.size(), 0, imageMean, imageStd), reference.data());
    EXPECT_GE(againstReference.within1Lsb, 0.995);
    EXPECT_LE(againstReference.maxDiff, 4);

    std::vector<uint16_t> const fromPitchLinear = ReadBackRawHalf(plDst, elems);
    std::vector<uint16_t> const fromBlockLinear = ReadBackRawHalf(blDst, elems);
    size_t identical = 0;
    double worstCodeUnits = 0.0;
    for (size_t i = 0; i < elems; ++i)
    {
        if (fromPitchLinear[i] == fromBlockLinear[i])
        {
            ++identical;
            continue;
        }
        half pl{};
        half bl{};
        std::memcpy(&pl, &fromPitchLinear[i], sizeof(half));
        std::memcpy(&bl, &fromBlockLinear[i], sizeof(half));
        // Normalised units scale back to code units by the channel's std.
        double const inCodeUnits
            = std::abs(__half2float(pl) - __half2float(bl)) * static_cast<double>(imageStd[i % 3]) * 255.0;
        worstCodeUnits = std::max(worstCodeUnits, inCodeUnits);
    }
    // Both entries instantiate the same chain, so these guard a wrong block order rather than express a
    // numeric tolerance: every case measured so far is bit-identical, and the residual bound is the
    // pitch-linear family's own.
    double const identicalShare = static_cast<double>(identical) / static_cast<double>(elems);
    EXPECT_GE(identicalShare, 0.9999);
    EXPECT_LE(worstCodeUnits, 4.0);

    std::cout << "FusedPreprocessNv12Bl Equivalence: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << (chromaCheckerboard ? ", chroma checkerboard" : ", random") << " -- " << identical << "/"
              << elems << " bit-identical to pitch-linear, worst " << worstCodeUnits << " code units; ";
    PrintDiffStats("vs reference", againstReference);
}

// The smallest case that runs the kNV12BL kernel instance end to end, at identity size.
TEST(FusedPreprocessNv12Bl, AcceptsBlockLinear)
{
    TestNv12BlockLinearMatchesPitchLinear(64, 64, 64, 64);
}

TEST(FusedPreprocessNv12Bl, MatchesPitchLinearAcrossGeometries)
{
    TestNv12BlockLinearMatchesPitchLinear(480, 640, 224, 224);
    TestNv12BlockLinearMatchesPitchLinear(
        224, 224, 448, 448, false, rt::imageUtils::ColorStandard::kBt601, rt::imageUtils::ColorRange::kFull);
    // 4K down to the ViT input size, the shape the vision runners actually see.
    TestNv12BlockLinearMatchesPitchLinear(2160, 3840, 448, 448);
}

// Odd extents are where the contract's (w + 1) / 2 parts company with w / 2: the chroma array is a
// column and a row wider than halving would give.
TEST(FusedPreprocessNv12Bl, OddExtents)
{
    TestNv12BlockLinearMatchesPitchLinear(747, 1000, 331, 209);
    TestNv12BlockLinearMatchesPitchLinear(101, 99, 51, 49, true);
}

// The address mode is the one sampler field the contract leaves free, because the kernel clamps its
// tap coordinates before it fetches. Varying it must not move a single output element.
TEST(FusedPreprocessNv12Bl, AddressModeDoesNotReachTheResult)
{
    cudaStream_t stream{nullptr};
    int64_t const height = 96, width = 128, outHeight = 64, outWidth = 64;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    size_t const elems = static_cast<size_t>(outHeight * outWidth * 3);

    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/true);
    std::vector<uint16_t> reference;
    for (cudaTextureAddressMode const mode : {cudaAddressModeClamp, cudaAddressModeBorder, cudaAddressModeWrap})
    {
        SamplerState sampler{};
        sampler.addressMode = mode;
        TexturePlane luma;
        TexturePlane chroma;
        // Wrap and mirror need normalised coordinates, so the driver may refuse them here; a mode
        // that cannot be built is a mode the kernel can never see.
        if (MakeTexturePlane(luma, frame.y.data(), width, height, 0, sampler) != cudaSuccess)
        {
            continue;
        }
        CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8, sampler));

        rt::imageUtils::ImageData const image = rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture,
            rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height);
        rt::Tensor dst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<uint16_t> const got = ReadBackRawHalf(dst, elems);
        if (reference.empty())
        {
            reference = got;
            continue;
        }
        ASSERT_EQ(reference, got) << "address mode " << static_cast<int>(mode) << " changed the result";
    }
}

// Every sampler field the contract pins, given a wrong value. None of them fails at read time: each
// returns a plausible sample from the wrong place, so the wrap is the only chance to catch them.
TEST(FusedPreprocessNv12Bl, RejectsWrongSamplerState)
{
    int64_t const height = 32, width = 32;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/true);

    // Linear filtering of an integer format needs a normalised read, so the two travel together.
    SamplerState filtered{};
    filtered.filterMode = cudaFilterModeLinear;
    filtered.readMode = cudaReadModeNormalizedFloat;
    SamplerState normalizedRead{};
    normalizedRead.readMode = cudaReadModeNormalizedFloat;
    SamplerState normalizedCoords{};
    normalizedCoords.normalizedCoords = 1;
    SamplerState srgb{};
    srgb.sRGB = 1;

    for (SamplerState const& wrong : {filtered, normalizedRead, normalizedCoords, srgb})
    {
        TexturePlane luma;
        TexturePlane chroma;
        if (MakeTexturePlane(luma, frame.y.data(), width, height, 0, wrong) != cudaSuccess)
        {
            // The driver refuses to build this combination at all, so it can never reach the wrap.
            continue;
        }
        CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8, wrong));
        EXPECT_THROW(rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture,
                         rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height),
            std::runtime_error);
    }
}

// The plane geometry the caller states and the arrays it hands over have to agree, or the kernel
// samples outside the frame it was told about.
TEST(FusedPreprocessNv12Bl, RejectsPlaneGeometryMismatch)
{
    int64_t const height = 48, width = 64;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/false);

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeTexturePlane(luma, frame.y.data(), width, height, 0));
    CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8));

    auto const wrap = [&](int64_t const w, int64_t const h) {
        return rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture, rt::imageUtils::ColorStandard::kBt709,
            rt::imageUtils::ColorRange::kLimited, w, h);
    };
    EXPECT_THROW(wrap(width + 2, height), std::runtime_error);
    EXPECT_THROW(wrap(width, height + 2), std::runtime_error);
    EXPECT_THROW(wrap(0, height), std::runtime_error);

    // The planes swapped: each is the wrong extent and the wrong texel width for its role.
    EXPECT_THROW(rt::imageUtils::wrapImageTexture(chroma.texture, luma.texture, rt::imageUtils::ColorStandard::kBt709,
                     rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
}

// A single-channel chroma plane samples Cb where the kernel expects the pair, and a two-channel luma
// plane doubles its texel width; both read inside the array and neither reports an error.
TEST(FusedPreprocessNv12Bl, RejectsWrongChannelDescription)
{
    int64_t const height = 48, width = 64;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/false);

    // A two-channel luma plane is twice as wide in bytes, so it needs its own source to read from.
    std::vector<unsigned char> const wideSource(static_cast<size_t>(height * width * 2), 128);

    TexturePlane luma;
    TexturePlane wideLuma;
    TexturePlane chroma;
    TexturePlane narrowChroma;
    CUDA_CHECK(MakeTexturePlane(luma, frame.y.data(), width, height, 0));
    CUDA_CHECK(MakeTexturePlane(wideLuma, wideSource.data(), width, height, 8));
    CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8));
    CUDA_CHECK(MakeTexturePlane(narrowChroma, frame.uv.data(), chromaW, chromaH, 0));

    EXPECT_THROW(rt::imageUtils::wrapImageTexture(wideLuma.texture, chroma.texture,
                     rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::wrapImageTexture(luma.texture, narrowChroma.texture,
                     rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
}

//! Allocate a plane whose array carries `flags`, the way a hardware decoder's frames arrive, and wrap a
//! texture object around it. The runtime API forwards flags it has no name for.
cudaError_t MakeFlaggedTexturePlane(
    TexturePlane& plane, int64_t const width, int64_t const height, int32_t const chromaBits, unsigned int const flags)
{
    cudaChannelFormatDesc const channel = cudaCreateChannelDesc(8, chromaBits, 0, 0, cudaChannelFormatKindUnsigned);
    cudaError_t const allocated
        = cudaMallocArray(&plane.array, &channel, static_cast<size_t>(width), static_cast<size_t>(height), flags);
    if (allocated != cudaSuccess)
    {
        (void) cudaGetLastError();
        return allocated;
    }

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = plane.array;
    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModePoint;
    description.readMode = cudaReadModeElementType;

    cudaError_t const status = cudaCreateTextureObject(&plane.texture, &resource, &description, nullptr);
    if (status != cudaSuccess)
    {
        (void) cudaGetLastError();
    }
    return status;
}

// NvBufSurface and NVDEC allocate their frames for the video engines, so every real block-linear
// frame arrives carrying that allocation flag on both planes. cuda.h names the bit
// CUDA_ARRAY3D_VIDEO_ENCODE_DECODE; the runtime API does not.
TEST(FusedPreprocessNv12Bl, AcceptsVideoEncodeDecodeArrays)
{
    constexpr unsigned int kVideoEncodeDecode = 0x100U;
    int64_t const height = 48, width = 64;

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeFlaggedTexturePlane(luma, width, height, 0, kVideoEncodeDecode));
    CUDA_CHECK(MakeFlaggedTexturePlane(chroma, (width + 1) / 2, (height + 1) / 2, 8, kVideoEncodeDecode));

    EXPECT_NO_THROW(rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture,
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height));
}

// A surface-loadable array still backs a texture object, but it is not the plain 2D array the plane
// check admits, so the flag it carries has to stay rejected and be named in the error.
TEST(FusedPreprocessNv12Bl, RejectsSurfaceLoadStoreArrays)
{
    int64_t const height = 48, width = 64;

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeFlaggedTexturePlane(luma, width, height, 0, cudaArraySurfaceLoadStore));
    CUDA_CHECK(MakeFlaggedTexturePlane(chroma, (width + 1) / 2, (height + 1) / 2, 8, 0));

    try
    {
        rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture, rt::imageUtils::ColorStandard::kBt709,
            rt::imageUtils::ColorRange::kLimited, width, height);
        FAIL() << "a surface load/store array wrapped as a texture plane";
    }
    catch (std::runtime_error const& e)
    {
        EXPECT_NE(std::string(e.what()).find("flags 2"), std::string::npos) << e.what();
    }
}

// Colour metadata has no default that is right: kUnspecified is the value-initialised state, and
// guessing a standard silently shifts every colour in the frame.
TEST(FusedPreprocessNv12Bl, RejectsUnspecifiedColourMetadata)
{
    int64_t const height = 32, width = 32;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/false);

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeTexturePlane(luma, frame.y.data(), width, height, 0));
    CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8));

    EXPECT_THROW(rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture,
                     rt::imageUtils::ColorStandard::kUnspecified, rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture, rt::imageUtils::ColorStandard::kBt709,
                     rt::imageUtils::ColorRange::kUnspecified, width, height),
        std::runtime_error);
}

// A value-initialised texture object is zero, which is what a caller that forgot to create one, or
// destroyed it early, hands over.
TEST(FusedPreprocessNv12Bl, RejectsUnsetTexturePlane)
{
    int64_t const height = 32, width = 32;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/false);

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeTexturePlane(luma, frame.y.data(), width, height, 0));
    CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8));

    EXPECT_THROW(rt::imageUtils::wrapImageTexture(cudaTextureObject_t{}, chroma.texture,
                     rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
    EXPECT_THROW(rt::imageUtils::wrapImageTexture(luma.texture, cudaTextureObject_t{},
                     rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height),
        std::runtime_error);
}

// A block-linear wrap addresses no buffer at all, so the byte counts that describe one report zero
// rather than the pitched figure the layout's zeroed pitches would otherwise produce.
TEST(FusedPreprocessNv12Bl, AddressesNoBytes)
{
    int64_t const height = 32, width = 32;
    int64_t const chromaW = (width + 1) / 2, chromaH = (height + 1) / 2;
    Nv12PathPair frame(height, width, /*chromaCheckerboard=*/false);

    TexturePlane luma;
    TexturePlane chroma;
    CUDA_CHECK(MakeTexturePlane(luma, frame.y.data(), width, height, 0));
    CUDA_CHECK(MakeTexturePlane(chroma, frame.uv.data(), chromaW, chromaH, 8));

    rt::imageUtils::ImageData const image = rt::imageUtils::wrapImageTexture(luma.texture, chroma.texture,
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, width, height);
    EXPECT_EQ(image.buffer, nullptr);
    EXPECT_EQ(image.data(), nullptr);
    EXPECT_EQ(image.frameBytes(), 0);
    EXPECT_EQ(image.addressedBytes(), 0);
    EXPECT_EQ(image.frames, 1);
    EXPECT_EQ(image.channels, 3);
}

// ---------------------------------------------------------------------------
// The six ViT families' own resize targets and normalisation constants
// ---------------------------------------------------------------------------
namespace
{

// Qwen2/2.5-VL: patchSize 14, mergeSize 2, builder defaults minImageTokens 4, maxImageTokens 512.
std::tuple<int64_t, int64_t> QwenTarget(int64_t const height, int64_t const width)
{
    return rt::imageUtils::qwenSmartResize(height, width, 14, 2, 4, 512);
}

// Gemma-4: maxImageTokens 256, poolingKernelSize 4, patchSize 14.
std::tuple<int64_t, int64_t> Gemma4Target(int64_t const height, int64_t const width)
{
    return rt::imageUtils::gemma4ResizeTarget(height, width, 256, 4, 14);
}

// Gemma-4 Unified: maxPatchesPerImage 256, modelPatchSize 48, positionEmbeddingSize 64.
std::tuple<int64_t, int64_t> Gemma4UnifiedTarget(int64_t const height, int64_t const width)
{
    return rt::imageUtils::gemma4UnifiedResizeTarget(height, width, 256, 48, 64);
}

// InternVL and Phi-4-MM tile at 448; the token budget is the one the engines are built with.
std::tuple<int64_t, int64_t> BlockGrid448Target(int64_t const height, int64_t const width)
{
    return rt::imageUtils::computeBestBlockGridForResize(height, width, 256, 4096, 448, 448, 256);
}

// Nemotron-Omni tiles at its force_image_size of 512.
std::tuple<int64_t, int64_t> BlockGrid512Target(int64_t const height, int64_t const width)
{
    return rt::imageUtils::computeBestBlockGridForResize(height, width, 256, 4096, 512, 512, 256);
}

//! One family's preprocessing configuration: the resize target its runner computes for a source, and
//! the normalisation constants it applies. The target calls the shared resize-target function rather
//! than pinning numbers, so a change there reaches these cases instead of leaving them testing a
//! geometry the runners no longer ask for.
struct VitFamily
{
    char const* name;
    std::tuple<int64_t, int64_t> (*target)(int64_t height, int64_t width);
    std::array<float, 3> imageMean;
    std::array<float, 3> imageStd;
};

//! Both Gemma-4 runners rescale to [0, 1] without normalising; the other four carry per-channel
//! constants their engine config supplies.
constexpr std::array<VitFamily, 6> kVitFamilies{{
    {"Qwen", QwenTarget, {0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F}},
    {"Gemma-4", Gemma4Target, {0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}},
    {"Gemma-4 Unified", Gemma4UnifiedTarget, {0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}},
    {"InternVL", BlockGrid448Target, {0.485F, 0.456F, 0.406F}, {0.229F, 0.224F, 0.225F}},
    {"Phi-4-MM", BlockGrid448Target, {0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F}},
    {"Nemotron-Omni", BlockGrid512Target, {0.481F, 0.458F, 0.408F}, {0.269F, 0.261F, 0.276F}},
}};

//! Source geometries the families are driven with: three frames the datasets actually carry (one of
//! odd height, one a 6:1 strip), plus an upscale and the 4K downscale that reaches the multi-chunk
//! path on RGB8.
struct SourceGeometry
{
    int64_t height;
    int64_t width;
};

constexpr std::array<SourceGeometry, 4> kVitSourceGeometries{{
    {747, 1000},
    {294, 1790},
    {64, 64},
    {2160, 3840},
}};

} // namespace

TEST(FusedPreprocessVitGeometry, Rgb8AcrossFamilies)
{
    for (auto const& family : kVitFamilies)
    {
        for (auto const& source : kVitSourceGeometries)
        {
            auto const [outHeight, outWidth] = family.target(source.height, source.width);
            std::cout << family.name << ": " << source.height << "x" << source.width << " -> " << outHeight << "x"
                      << outWidth << std::endl;
            TestFusedResizeRgb(static_cast<int32_t>(source.height), static_cast<int32_t>(source.width),
                static_cast<int32_t>(outHeight), static_cast<int32_t>(outWidth), ResizeFillPattern::kRANDOM,
                family.imageMean, family.imageStd);
        }
    }
}

TEST(FusedPreprocessVitGeometry, Nv12PitchLinearAcrossFamilies)
{
    for (auto const& family : kVitFamilies)
    {
        for (auto const& source : kVitSourceGeometries)
        {
            auto const [outHeight, outWidth] = family.target(source.height, source.width);
            TestFusedNv12(source.height, source.width, outHeight, outWidth, /*chromaCheckerboard=*/false,
                rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, 0, 0, family.imageMean,
                family.imageStd);
        }
    }
}

TEST(FusedPreprocessVitGeometry, Nv12BlockLinearAcrossFamilies)
{
    for (auto const& family : kVitFamilies)
    {
        for (auto const& source : kVitSourceGeometries)
        {
            auto const [outHeight, outWidth] = family.target(source.height, source.width);
            TestNv12BlockLinearMatchesPitchLinear(source.height, source.width, outHeight, outWidth,
                /*chromaCheckerboard=*/false, rt::imageUtils::ColorStandard::kBt709,
                rt::imageUtils::ColorRange::kLimited, family.imageMean, family.imageStd);
        }
    }
}

// Normalising is an affine step on a value the kernel has already quantised to a code unit, so the
// code units a family's constants produce are the ones unit constants produce over the same geometry.
// A constant that reached the resample, or that landed on the wrong channel, would break this.
TEST(FusedPreprocessVitGeometry, NormalisationDoesNotReachTheResample)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 747, inWidth = 1000;

    std::vector<unsigned char> input(static_cast<size_t>(inHeight * inWidth * 3));
    FillResizeInput(
        input, static_cast<int32_t>(inHeight), static_cast<int32_t>(inWidth), 3, ResizeFillPattern::kRANDOM);
    rt::imageUtils::ImageData image = MakePinnedRgbImage(input, 1, inHeight, inWidth);

    for (auto const& family : kVitFamilies)
    {
        auto const [outHeight, outWidth] = family.target(inHeight, inWidth);
        size_t const elems = static_cast<size_t>(outHeight * outWidth * 3);

        rt::Tensor normalised({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor unit({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::imageUtils::resizeAndNormalizeToRgb(
            image, 0, 1, family.imageMean, family.imageStd, normalised, outHeight, outWidth, stream);
        rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, unit, outHeight, outWidth, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<unsigned char> const got = ReadBackAsU8(normalised, elems, 0, family.imageMean, family.imageStd);
        std::vector<unsigned char> const ref = ReadBackAsU8(unit, elems);
        for (size_t i = 0; i < elems; ++i)
        {
            ASSERT_EQ(got[i], ref[i]) << family.name << " element " << i << " differs from the unit-normalised run";
        }
        std::cout << "FusedPreprocessVitGeometry Normalisation: " << family.name << " " << outHeight << "x" << outWidth
                  << " recovered every code unit." << std::endl;
    }
}

// Past roughly 13.3x vertically the luma support outgrows the shared memory stage and is consumed in
// more than one chunk; the chroma plane reaches its own threshold past 23x. No resize target a runner
// computes is that steep, so the geometries the other NV12 cases carry stay on the single-chunk path.
TEST(FusedPreprocessNv12, MultiChunkSupport)
{
    TestFusedNv12(
        6400, 448, 448, 448, false, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
    TestFusedNv12(
        12000, 448, 448, 448, false, rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited);
}

TEST(FusedPreprocessNv12Bl, MultiChunkSupport)
{
    TestNv12BlockLinearMatchesPitchLinear(6400, 448, 448, 448);
    TestNv12BlockLinearMatchesPitchLinear(12000, 448, 448, 448);
}

// InternVL, Phi-4-MM and Nemotron-Omni preprocess one source twice: once onto the tile grid and once
// onto a single-tile thumbnail. The second call must not depend on the first having run.
TEST(FusedPreprocessNv12, SecondCallOnTheSameFrameIsIndependent)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 747, inWidth = 1000;
    int64_t const mainHeight = 896, mainWidth = 448, thumbHeight = 448, thumbWidth = 448;
    int64_t const chromaRowBytes = 2 * ((inWidth + 1) / 2);
    auto const standard = rt::imageUtils::ColorStandard::kBt709;
    auto const range = rt::imageUtils::ColorRange::kLimited;

    std::vector<unsigned char> y(static_cast<size_t>(inHeight * inWidth));
    std::vector<unsigned char> uv(static_cast<size_t>(((inHeight + 1) / 2) * chromaRowBytes));
    uniformIntInitialization<unsigned char>(y, 0, 255);
    uniformIntInitialization<unsigned char>(uv, 0, 255);
    rt::imageUtils::ImageData image
        = MakePinnedNv12Image(y, uv, inHeight, inWidth, inWidth, chromaRowBytes, standard, range);

    rt::Tensor mainGrid({1, mainHeight, mainWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor thumb({1, thumbHeight, thumbWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor thumbAlone({1, thumbHeight, thumbWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, mainGrid, mainHeight, mainWidth, stream);
    rt::imageUtils::resizeAndNormalizeToRgb(image, 0, 1, kUnitMean, kUnitStd, thumb, thumbHeight, thumbWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::imageUtils::ImageData fresh
        = MakePinnedNv12Image(y, uv, inHeight, inWidth, inWidth, chromaRowBytes, standard, range);
    rt::imageUtils::resizeAndNormalizeToRgb(
        fresh, 0, 1, kUnitMean, kUnitStd, thumbAlone, thumbHeight, thumbWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const elems = static_cast<size_t>(thumbHeight * thumbWidth * 3);
    std::vector<unsigned char> const after = ReadBackAsU8(thumb, elems);
    std::vector<unsigned char> const alone = ReadBackAsU8(thumbAlone, elems);
    for (size_t i = 0; i < elems; ++i)
    {
        ASSERT_EQ(after[i], alone[i]) << "thumbnail element " << i << " depends on the preceding main-grid call";
    }
    std::cout << "FusedPreprocessNv12 SecondCall: thumbnail identical whether or not the main grid ran first."
              << std::endl;
}

// A block-linear wrap carries one frame, so a video is one texture pair and one launch per frame into
// the same destination. Re-running an earlier frame has to reproduce its own result rather than keep
// any part of the frame that ran in between.
TEST(FusedPreprocessNv12Bl, SuccessiveFramesDoNotBleed)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 120, inWidth = 160, outHeight = 64, outWidth = 96;
    int64_t const chromaW = (inWidth + 1) / 2, chromaH = (inHeight + 1) / 2;
    size_t const elems = static_cast<size_t>(outHeight * outWidth * 3);

    Nv12PathPair first(inHeight, inWidth, /*chromaCheckerboard=*/false);
    Nv12PathPair second(inHeight, inWidth, /*chromaCheckerboard=*/true);
    CUDA_CHECK(MakeTexturePlane(first.luma, first.y.data(), inWidth, inHeight, 0));
    CUDA_CHECK(MakeTexturePlane(first.chroma, first.uv.data(), chromaW, chromaH, 8));
    CUDA_CHECK(MakeTexturePlane(second.luma, second.y.data(), inWidth, inHeight, 0));
    CUDA_CHECK(MakeTexturePlane(second.chroma, second.uv.data(), chromaW, chromaH, 8));

    rt::imageUtils::ImageData const frameA = rt::imageUtils::wrapImageTexture(first.luma.texture, first.chroma.texture,
        rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, inWidth, inHeight);
    rt::imageUtils::ImageData const frameB
        = rt::imageUtils::wrapImageTexture(second.luma.texture, second.chroma.texture,
            rt::imageUtils::ColorStandard::kBt709, rt::imageUtils::ColorRange::kLimited, inWidth, inHeight);

    rt::Tensor dst({1, outHeight, outWidth, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::imageUtils::resizeAndNormalizeToRgb(frameA, 0, 1, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> const firstPass = ReadBackAsU8(dst, elems);

    rt::imageUtils::resizeAndNormalizeToRgb(frameB, 0, 1, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> const between = ReadBackAsU8(dst, elems);

    rt::imageUtils::resizeAndNormalizeToRgb(frameA, 0, 1, kUnitMean, kUnitStd, dst, outHeight, outWidth, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> const thirdPass = ReadBackAsU8(dst, elems);

    size_t differing = 0;
    for (size_t i = 0; i < elems; ++i)
    {
        ASSERT_EQ(firstPass[i], thirdPass[i]) << "element " << i << " kept part of the frame that ran in between";
        differing += firstPass[i] != between[i] ? 1 : 0;
    }
    EXPECT_GT(differing, 0U) << "the two frames produced the same output, so the check proves nothing";
    std::cout << "FusedPreprocessNv12Bl SuccessiveFrames: " << differing << "/" << elems
              << " elements differ between the frames, and frame A reproduced exactly." << std::endl;
}

// ---------------------------------------------------------------------------
// Nemotron-Omni patch-embedder input kernels (no TRT engine needed)
// ---------------------------------------------------------------------------
namespace
{
void checkNemotronTranspose(int64_t T)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int64_t const C = 2, H = 4, W = 6, P = 2;
    int64_t const numFrames = 2 * T; // two temporal groups
    int64_t const gridH = H / P, gridW = W / P, numPatches = gridH * gridW;
    int64_t const numGroups = numFrames / T, rowWidth = T * C * P * P, rows = numGroups * numPatches;

    std::vector<half> src(static_cast<size_t>(numFrames * C * H * W));
    for (size_t i = 0; i < src.size(); ++i)
    {
        src[i] = __float2half(static_cast<float>(i));
    }
    rt::Tensor blockPixels({numFrames, C, H, W}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor inputPatches({rows, rowWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(
        blockPixels.rawPointer(), src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    kernel::transposeToPatchNemotronViT(blockPixels, inputPatches, T, P, stream);
    std::vector<half> out(static_cast<size_t>(rows * rowWidth));
    CUDA_CHECK(cudaMemcpyAsync(
        out.data(), inputPatches.rawPointer(), out.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // (t, c, py, px) C-major patch layout; row = group*numPatches + (pi*gridW + pj).
    for (int64_t group = 0; group < numGroups; ++group)
    {
        for (int64_t pi = 0; pi < gridH; ++pi)
        {
            for (int64_t pj = 0; pj < gridW; ++pj)
            {
                for (int64_t t = 0; t < T; ++t)
                {
                    for (int64_t c = 0; c < C; ++c)
                    {
                        for (int64_t py = 0; py < P; ++py)
                        {
                            for (int64_t px = 0; px < P; ++px)
                            {
                                int64_t const row = group * numPatches + (pi * gridW + pj);
                                int64_t const col = ((t * C + c) * P + py) * P + px;
                                int64_t const srcIdx = (((group * T + t) * C + c) * H + pi * P + py) * W + pj * P + px;
                                ASSERT_EQ(__half2float(out[row * rowWidth + col]), __half2float(src[srcIdx]))
                                    << "T=" << T << " mismatch at row " << row << " col " << col;
                            }
                        }
                    }
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
} // namespace

TEST(TransposeToPatchNemotron, ImageTileT1)
{
    checkNemotronTranspose(1);
}

TEST(TransposeToPatchNemotron, VideoTubeletT2)
{
    checkNemotronTranspose(2);
}

TEST(TransposeToPatchNemotron, RejectsNonPositiveDivisors)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    rt::Tensor blockPixels({2, 2, 4, 6}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor inputPatches({6, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    EXPECT_ANY_THROW(kernel::transposeToPatchNemotronViT(blockPixels, inputPatches, 0, 2, stream));
    EXPECT_ANY_THROW(kernel::transposeToPatchNemotronViT(blockPixels, inputPatches, 2, 0, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(AddPosEmbedNemotron, BroadcastOverBlocks)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int64_t const numBlocks = 3, numPatches = 4, hidden = 5;
    std::vector<half> embeds(static_cast<size_t>(numBlocks * numPatches * hidden));
    std::vector<half> pos(static_cast<size_t>(numPatches * hidden));
    for (size_t i = 0; i < embeds.size(); ++i)
    {
        embeds[i] = __float2half(static_cast<float>(i) * 0.5F);
    }
    for (size_t i = 0; i < pos.size(); ++i)
    {
        pos[i] = __float2half(static_cast<float>(i) + 1.0F);
    }
    rt::Tensor patchEmbeds({numBlocks, numPatches, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor posEmbed({numPatches, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(
        patchEmbeds.rawPointer(), embeds.data(), embeds.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(posEmbed.rawPointer(), pos.data(), pos.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    kernel::addPosEmbedNemotronViT(patchEmbeds, posEmbed, stream);
    std::vector<half> out(embeds.size());
    CUDA_CHECK(cudaMemcpyAsync(
        out.data(), patchEmbeds.rawPointer(), out.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int64_t b = 0; b < numBlocks; ++b)
    {
        for (int64_t p = 0; p < numPatches; ++p)
        {
            for (int64_t h = 0; h < hidden; ++h)
            {
                int64_t const idx = (b * numPatches + p) * hidden + h;
                float const expected = __half2float(embeds[idx]) + __half2float(pos[p * hidden + h]);
                ASSERT_NEAR(__half2float(out[idx]), expected, 1e-2F) << "pos-embed add mismatch at " << idx;
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(EvsScoresNemotron, SentinelAndCosineDissimilarity)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int64_t const numGroups = 3, tokensPerGroup = 2, hidden = 4;
    int64_t const numTokens = numGroups * tokensPerGroup;
    std::vector<half> embeds(static_cast<size_t>(numTokens * hidden));
    for (size_t i = 0; i < embeds.size(); ++i)
    {
        embeds[i] = __float2half(static_cast<float>((i * 7) % 5) + 0.25F);
    }
    rt::Tensor embedsDev({numTokens, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor scoresDev({numTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpyAsync(
        embedsDev.rawPointer(), embeds.data(), embeds.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    kernel::evsScoresNemotronViT(embedsDev, scoresDev, tokensPerGroup, stream);
    std::vector<float> scores(static_cast<size_t>(numTokens));
    CUDA_CHECK(cudaMemcpyAsync(
        scores.data(), scoresDev.rawPointer(), scores.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int64_t g = 0; g < numGroups; ++g)
    {
        for (int64_t s = 0; s < tokensPerGroup; ++s)
        {
            int64_t const token = g * tokensPerGroup + s;
            if (g == 0)
            {
                // First tubelet is always kept via the 255 sentinel.
                ASSERT_FLOAT_EQ(scores[token], 255.0F) << "group-0 sentinel missing at " << token;
                continue;
            }
            double dot = 0.0, nc = 0.0, np = 0.0;
            for (int64_t h = 0; h < hidden; ++h)
            {
                double const a = __half2float(embeds[token * hidden + h]);
                double const b = __half2float(embeds[(token - tokensPerGroup) * hidden + h]);
                dot += a * b;
                nc += a * a;
                np += b * b;
            }
            float const expected = static_cast<float>(1.0 - dot / (std::sqrt(nc) * std::sqrt(np)));
            ASSERT_NEAR(scores[token], expected, 1e-3F) << "EVS score mismatch at token " << token;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(EvsScoresNemotron, RejectsNonPositiveTokensPerGroup)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    rt::Tensor embedsDev({4, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor scoresDev({4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    EXPECT_ANY_THROW(kernel::evsScoresNemotronViT(embedsDev, scoresDev, 0, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
