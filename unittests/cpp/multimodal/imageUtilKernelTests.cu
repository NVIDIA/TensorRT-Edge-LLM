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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
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

void TestNormalizeImage(int32_t const batch, int32_t const height, int32_t const width, int32_t const channels = 3)
{
    cudaStream_t stream{nullptr};

    std::vector<unsigned char> originalImage(batch * height * width * channels);
    std::vector<half> normalizedImageRef(batch * height * width * channels);
    std::vector<float> mean(channels);
    std::vector<float> std(channels);
    uniformIntInitialization<unsigned char>(originalImage, 0, 255);
    uniformFloatInitialization<float>(mean, 0, 1);
    uniformFloatInitialization<float>(std, 0, 1);

    for (int32_t i = 0; i < batch * height * width; ++i)
    {
        for (int32_t j = 0; j < channels; ++j)
        {
            float normalized = (originalImage[i * channels + j] / 255.0f - mean[j]) / std[j];
            normalizedImageRef[i * channels + j] = __float2half(normalized);
        }
    }

    // GPU tensors
    rt::Tensor originalImageDevice({batch, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor normalizedImageDevice({batch, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor meanDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor stdDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        meanDevice.rawPointer(), mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        stdDevice.rawPointer(), std.data(), std.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    kernel::normalizeImage(originalImageDevice, meanDevice, stdDevice, normalizedImageDevice, stream);
    std::vector<half> normalizedImage(batch * height * width * channels);
    CUDA_CHECK(cudaMemcpyAsync(normalizedImage.data(), normalizedImageDevice.rawPointer(),
        normalizedImage.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare data
    for (int32_t i = 0; i < batch * height * width * channels; ++i)
    {
        EXPECT_TRUE(isclose(normalizedImage[i], normalizedImageRef[i], 1e-5, 1e-5));
    }

    std::cout << "NormalizeImage Accuracy: batch=" << batch << ", height=" << height << ", width=" << width
              << ", channels=" << channels << std::endl;
}

void BenchmarkNormalizeImage(int32_t const batch, int32_t const height, int32_t const width, int32_t const channels = 3)
{
    cudaStream_t stream{nullptr};

    std::vector<int8_t> originalImage(batch * height * width * channels);
    std::vector<float> mean(channels);
    std::vector<float> std(channels);
    uniformIntInitialization<int8_t>(originalImage, 0, 255);
    uniformFloatInitialization<float>(mean, 0, 1);
    uniformFloatInitialization<float>(std, 0, 1);

    rt::Tensor originalImageDevice({batch, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor normalizedImageDevice({batch, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor meanDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor stdDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpyAsync(originalImageDevice.rawPointer(), originalImage.data(),
        originalImage.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        meanDevice.rawPointer(), mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        stdDevice.rawPointer(), std.data(), std.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto launch
        = [&]() { kernel::normalizeImage(originalImageDevice, meanDevice, stdDevice, normalizedImageDevice, stream); };

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
    std::cout << "NormalizeImage Benchmark: batch=" << batch << ", height=" << height << ", width=" << width
              << ", channels=" << channels << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(NormalizeImage, Accuracy)
{
    TestNormalizeImage(4, 720, 1280);
}

TEST(NormalizeImage, Benchmark)
{
    BenchmarkNormalizeImage(4, 720, 1280);
}

void TestTransposeToPatchQwenViT(int32_t const height, int32_t const width, int32_t const channels = 3,
    int32_t const T = 2, int32_t const temporalPatchSize = 2, int32_t const patchSize = 14, int32_t const mergeSize = 2)
{
    cudaStream_t stream{nullptr};

    // CPU reference
    std::vector<half> originalImage(T * height * width * channels);
    std::vector<half> inputPatchesRef(T * height * width * channels);
    uniformFloatInitialization<half>(originalImage, 0, 1);

    transposeToPatchQwenReference(
        originalImage, inputPatchesRef, 0, T, height, width, channels, temporalPatchSize, patchSize, mergeSize);

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

    kernel::transposeToPatchQwenViT(
        originalImageDevice, inputPatchesDevice, 0, temporalPatchSize, patchSize, mergeSize, stream);

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
            originalImageDevice, inputPatchesDevice, 0, temporalPatchSize, patchSize, mergeSize, stream);
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

// Fill patterns for the resize tests; checkerboard (2x2 cells) is the highest-frequency case.
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

// GPU bicubic resize vs the CPU rt::imageUtils::resizeImage golden (stbir CATMULLROM + EDGE_CLAMP).
// The float summation order differs, so u8 outputs aren't bit-exact; asserted on the |diff| distribution.
void TestResizeBicubicCatmullRom(int32_t const inHeight, int32_t const inWidth, int32_t const outHeight,
    int32_t const outWidth, ResizeFillPattern const pattern, int32_t const channels = 3)
{
    cudaStream_t stream{nullptr};

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    FillResizeInput(input, inHeight, inWidth, channels, pattern);

    rt::Tensor inputTensor({1, inHeight, inWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    std::memcpy(inputTensor.rawPointer(), input.data(), input.size());
    rt::imageUtils::ImageData inputImage(std::move(inputTensor));
    rt::Tensor refTensor({1, outHeight, outWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    rt::imageUtils::ImageData resizedRef(std::move(refTensor));
    auto const& resizedRefView = rt::imageUtils::resizeImage(
        inputImage, resizedRef, outWidth, outHeight, rt::imageUtils::InterpolationMode::kBICUBIC);

    rt::Tensor rawImageDevice({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor resizeTmpDevice({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor resizedImageDevice({outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    CUDA_CHECK(
        cudaMemcpyAsync(rawImageDevice.rawPointer(), input.data(), input.size(), cudaMemcpyHostToDevice, stream));
    kernel::resizeImage(rawImageDevice, resizeTmpDevice, resizedImageDevice, outHeight, outWidth,
        kernel::InterpolationMode::kBICUBIC, stream);

    std::vector<unsigned char> output(static_cast<size_t>(outHeight) * outWidth * channels);
    CUDA_CHECK(
        cudaMemcpyAsync(output.data(), resizedImageDevice.rawPointer(), output.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::array<int64_t, 256> histogram{};
    int32_t maxDiff = 0;
    int64_t signedDiffSum = 0;
    unsigned char const* ref = resizedRefView.data();
    for (size_t i = 0; i < output.size(); ++i)
    {
        int32_t const signedDiff = static_cast<int32_t>(output[i]) - static_cast<int32_t>(ref[i]);
        int32_t const diff = std::abs(signedDiff);
        ++histogram[diff];
        maxDiff = std::max(maxDiff, diff);
        signedDiffSum += signedDiff;
    }
    double const within1Lsb = static_cast<double>(histogram[0] + histogram[1]) / static_cast<double>(output.size());
    double const meanSignedDiff = static_cast<double>(signedDiffSum) / static_cast<double>(output.size());

    // Bounds carry a margin over stbir: the GPU's float summation order differs, so bytes aren't exact.
    EXPECT_GE(within1Lsb, 0.995);
    EXPECT_LE(maxDiff, 2);
    if (pattern != ResizeFillPattern::kCHECKERBOARD)
    {
        // The mean-signed-diff bound catches systematic rounding bias (e.g. truncation vs round-to-nearest).
        // Checkerboard is exempt: its exact-halfway values (127.5) tie-break differently between stbir and
        // the GPU, which the |diff| bounds above already cap at 1 LSB.
        EXPECT_LE(std::abs(meanSignedDiff), 0.05);
    }

    std::cout << "ResizeBicubicCatmullRom Accuracy: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << ", pattern=" << ResizeFillPatternName(pattern) << ", within1Lsb=" << within1Lsb * 100.0
              << "%, maxDiff=" << maxDiff << ", bias=" << meanSignedDiff << ", hist=[";
    for (int32_t d = 0; d <= maxDiff; ++d)
    {
        std::cout << (d == 0 ? "" : " ") << d << ":" << histogram[d];
    }
    std::cout << "]" << std::endl;
}

void BenchmarkResizeBicubicCatmullRom(int32_t const inHeight, int32_t const inWidth, int32_t const outHeight,
    int32_t const outWidth, int32_t const channels = 3)
{
    cudaStream_t stream{nullptr};

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    uniformIntInitialization<unsigned char>(input, 0, 255);

    rt::Tensor rawImageDevice({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor resizeTmpDevice({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor resizedImageDevice({outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    CUDA_CHECK(
        cudaMemcpyAsync(rawImageDevice.rawPointer(), input.data(), input.size(), cudaMemcpyHostToDevice, stream));

    auto launch = [&]() {
        kernel::resizeImage(rawImageDevice, resizeTmpDevice, resizedImageDevice, outHeight, outWidth,
            kernel::InterpolationMode::kBICUBIC, stream);
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
    std::cout << "ResizeBicubicCatmullRom Benchmark: in=" << inHeight << "x" << inWidth << ", out=" << outHeight << "x"
              << outWidth << ", time=" << elapsedTime / numBenchIter << " ms" << std::endl;
}

constexpr std::array<ResizeFillPattern, 3> kResizeFillPatterns{
    ResizeFillPattern::kRANDOM, ResizeFillPattern::kGRADIENT, ResizeFillPattern::kCHECKERBOARD};

TEST(ResizeBicubicCatmullRom, AccuracyUpscale)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(320, 480, 640, 960, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyDownscale)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(1024, 1024, 512, 512, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyNonIntegerRatio)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(747, 1000, 608, 832, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyAsymmetricAxes)
{
    // Height upscales 1.5x while width downscales 2x.
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(512, 1536, 768, 768, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyExtremeAspect)
{
    // ~6:1 aspect ratio (wide-banner-shaped input).
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(294, 1790, 160, 960, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyOddSizes)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(331, 477, 123, 209, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyTinyEdge)
{
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(1, 512, 1, 256, pattern);
        TestResizeBicubicCatmullRom(512, 1, 256, 1, pattern);
    }
}

TEST(ResizeBicubicCatmullRom, AccuracyLargeFactorDownscale)
{
    // >= 4x downscale, the widest anti-alias filter regime.
    for (auto const pattern : kResizeFillPatterns)
    {
        TestResizeBicubicCatmullRom(2160, 3840, 512, 960, pattern);
    }
}

// Resize each frame directly into its slot of a multi-frame buffer through a non-owning view. The
// resize kernel is deterministic, so each slot must byte-match an isolated resize and leave others intact.
TEST(ResizeBicubicCatmullRom, SlotViewIntoLargerBuffer)
{
    cudaStream_t stream{nullptr};
    int32_t const inHeight = 480, inWidth = 640, outHeight = 512, outWidth = 960, channels = 3;
    int32_t const nFrames = 3;
    int64_t const slotElems = static_cast<int64_t>(outHeight) * outWidth * channels;

    // Stand-in for mImageDevice: nFrames contiguous slots, sentinel-filled so an unwritten region fails.
    rt::Tensor bigBuffer({nFrames * slotElems}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    CUDA_CHECK(cudaMemset(bigBuffer.rawPointer(), 0xAB, static_cast<size_t>(nFrames * slotElems)));

    // Scratch reused across frames.
    rt::Tensor rawImageDevice({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor resizeTmpDevice({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor refDevice({outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);

    std::array<ResizeFillPattern, 3> const patterns{
        ResizeFillPattern::kGRADIENT, ResizeFillPattern::kCHECKERBOARD, ResizeFillPattern::kRANDOM};
    std::vector<std::vector<unsigned char>> refHost(nFrames);

    for (int32_t f = 0; f < nFrames; ++f)
    {
        std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
        FillResizeInput(input, inHeight, inWidth, channels, patterns[f]);
        CUDA_CHECK(
            cudaMemcpyAsync(rawImageDevice.rawPointer(), input.data(), input.size(), cudaMemcpyHostToDevice, stream));

        // Resize into a non-owning view of slot f.
        rt::Tensor slot(static_cast<unsigned char*>(bigBuffer.rawPointer()) + f * slotElems,
            {outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        kernel::resizeImage(
            rawImageDevice, resizeTmpDevice, slot, outHeight, outWidth, kernel::InterpolationMode::kBICUBIC, stream);

        kernel::resizeImage(rawImageDevice, resizeTmpDevice, refDevice, outHeight, outWidth,
            kernel::InterpolationMode::kBICUBIC, stream);
        refHost[f].resize(static_cast<size_t>(slotElems));
        CUDA_CHECK(
            cudaMemcpyAsync(refHost[f].data(), refDevice.rawPointer(), slotElems, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Each slot must still equal its reference: correct content and no cross-slot clobber.
    std::vector<unsigned char> bigHost(static_cast<size_t>(nFrames * slotElems));
    CUDA_CHECK(cudaMemcpy(bigHost.data(), bigBuffer.rawPointer(), bigHost.size(), cudaMemcpyDeviceToHost));
    for (int32_t f = 0; f < nFrames; ++f)
    {
        for (int64_t k = 0; k < slotElems; ++k)
        {
            ASSERT_EQ(bigHost[f * slotElems + k], refHost[f][k])
                << "slot " << f << " element " << k << " differs from the isolated resize";
        }
    }

    std::cout << "ResizeBicubicCatmullRom SlotView: " << nFrames << " slots of " << outHeight << "x" << outWidth
              << " written via non-owning views, all byte-identical to isolated resize." << std::endl;
}

TEST(ResizeBicubicCatmullRom, Benchmark)
{
    // 1080p and 4K frames, both resized to 512x960.
    BenchmarkResizeBicubicCatmullRom(1080, 1920, 512, 960);
    BenchmarkResizeBicubicCatmullRom(2160, 3840, 512, 960);
}

// Contract tests for kernel::copyImageToDeviceAndResize: per-frame resize, identity skip, and the
// per-dimension raw cap. It reshapes the destination to [numFrames, outH, outW, C] and resizes each
// frame into it, or copies a frame verbatim when the target equals the source size.

// A single-frame call must be byte-identical to a direct kernel::resizeImage and track the CPU golden
// within the standalone-kernel margins.
void TestCopyImageToDeviceAndResize(int32_t const inHeight, int32_t const inWidth, int32_t const outHeight,
    int32_t const outWidth, ResizeFillPattern const pattern, int32_t const channels = 3)
{
    cudaStream_t stream{nullptr};

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    FillResizeInput(input, inHeight, inWidth, channels, pattern);

    rt::Tensor inputTensor({1, inHeight, inWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    std::memcpy(inputTensor.rawPointer(), input.data(), input.size());
    rt::imageUtils::ImageData inputImage(std::move(inputTensor));
    rt::Tensor refTensor({1, outHeight, outWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    rt::imageUtils::ImageData resizedRef(std::move(refTensor));
    auto const& resizedRefView = rt::imageUtils::resizeImage(
        inputImage, resizedRef, outWidth, outHeight, rt::imageUtils::InterpolationMode::kBICUBIC);

    // Direct kernel::resizeImage reference.
    rt::Tensor rawImageDevice({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor resizeTmpDevice({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor refDevice({outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    CUDA_CHECK(
        cudaMemcpyAsync(rawImageDevice.rawPointer(), input.data(), input.size(), cudaMemcpyHostToDevice, stream));
    kernel::resizeImage(
        rawImageDevice, resizeTmpDevice, refDevice, outHeight, outWidth, kernel::InterpolationMode::kBICUBIC, stream);
    std::vector<unsigned char> refOut(static_cast<size_t>(outHeight) * outWidth * channels);
    CUDA_CHECK(cudaMemcpyAsync(refOut.data(), refDevice.rawPointer(), refOut.size(), cudaMemcpyDeviceToHost, stream));

    // Helper path: a 4-D destination shaped like a runner's mImageDevice.
    rt::Tensor helperRawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor helperTmpScratch({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels, helperRawScratch,
        helperTmpScratch, dstImage, outHeight, outWidth, stream);
    std::vector<unsigned char> output(static_cast<size_t>(outHeight) * outWidth * channels);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), dstImage.rawPointer(), output.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Helper == direct kernel, byte for byte.
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_EQ(output[i], refOut[i]) << "helper output differs from direct kernel::resizeImage at index " << i;
    }

    std::array<int64_t, 256> histogram{};
    int32_t maxDiff = 0;
    int64_t signedDiffSum = 0;
    unsigned char const* ref = resizedRefView.data();
    for (size_t i = 0; i < output.size(); ++i)
    {
        int32_t const signedDiff = static_cast<int32_t>(output[i]) - static_cast<int32_t>(ref[i]);
        int32_t const diff = std::abs(signedDiff);
        ++histogram[diff];
        maxDiff = std::max(maxDiff, diff);
        signedDiffSum += signedDiff;
    }
    double const within1Lsb = static_cast<double>(histogram[0] + histogram[1]) / static_cast<double>(output.size());
    double const meanSignedDiff = static_cast<double>(signedDiffSum) / static_cast<double>(output.size());
    EXPECT_GE(within1Lsb, 0.995);
    EXPECT_LE(maxDiff, 2);
    if (pattern != ResizeFillPattern::kCHECKERBOARD)
    {
        EXPECT_LE(std::abs(meanSignedDiff), 0.05);
    }
}

namespace
{
struct ResizeGeometry
{
    int32_t inHeight, inWidth, outHeight, outWidth;
};

// Geometries covered by the ResizeBicubicCatmullRom accuracy suite.
constexpr std::array<ResizeGeometry, 9> kResizeGeometries{{
    {320, 480, 640, 960},
    {1024, 1024, 512, 512},
    {747, 1000, 608, 832},
    {512, 1536, 768, 768},
    {294, 1790, 160, 960},
    {331, 477, 123, 209},
    {1, 512, 1, 256},
    {512, 1, 256, 1},
    {2160, 3840, 512, 960},
}};
} // namespace

TEST(CopyImageToDeviceAndResize, Accuracy)
{
    for (auto const& g : kResizeGeometries)
    {
        for (auto const pattern : kResizeFillPatterns)
        {
            TestCopyImageToDeviceAndResize(g.inHeight, g.inWidth, g.outHeight, g.outWidth, pattern);
        }
    }
}

// Identity resize: when the target equals the source the helper copies the raw image verbatim.
TEST(CopyImageToDeviceAndResize, IdentitySkip)
{
    cudaStream_t stream{nullptr};
    int32_t const height = 273, width = 409, channels = 3; // odd dims, high-frequency content
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels);
    FillResizeInput(input, height, width, channels, ResizeFillPattern::kCHECKERBOARD);

    rt::Tensor rawScratch({height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmpScratch({height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    // Sentinel-fill the destination so a skipped copy would be caught.
    CUDA_CHECK(cudaMemset(dstImage.rawPointer(), 0x5A, input.size()));

    kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, height, width, channels, rawScratch, tmpScratch,
        dstImage, height, width, stream);

    std::vector<unsigned char> output(input.size());
    CUDA_CHECK(cudaMemcpyAsync(output.data(), dstImage.rawPointer(), output.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_EQ(output[i], input[i]) << "identity resize altered pixel " << i;
    }
    std::cout << "CopyImageToDeviceAndResize IdentitySkip: " << height << "x" << width << " copied verbatim."
              << std::endl;
}

// Raw-side cap: each raw dimension must be <= kGpuResizeMaxRawDim. The check is per-dimension, not on the
// pixel product, so a tall/thin image that fits the pixel budget but whose long side exceeds the cap is rejected.
TEST(CopyImageToDeviceAndResize, PerDimensionCap)
{
    cudaStream_t stream{nullptr};
    int32_t const channels = 3, outHeight = 64, outWidth = 64;

    // Small scratch and destination: the per-dimension cap fires before any of these are touched.
    rt::Tensor rawScratch({64, 64, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmpScratch({64, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);

    // Tall/thin: rawHeight exceeds the cap while the product stays far under kGpuResizeMaxRawDim^2.
    {
        int64_t const tallH = kernel::kGpuResizeMaxRawDim + 1, tallW = 32;
        std::vector<unsigned char> tall(static_cast<size_t>(tallH) * tallW * channels, 0);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(tall.data(), /*numFrames=*/1, tallH, tallW, channels,
                         rawScratch, tmpScratch, dstImage, outHeight, outWidth, stream),
            std::runtime_error);
    }
    // Wide/thin: rawWidth exceeds the cap, product equally under budget.
    {
        int64_t const wideH = 32, wideW = kernel::kGpuResizeMaxRawDim + 1;
        std::vector<unsigned char> wide(static_cast<size_t>(wideH) * wideW * channels, 0);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(wide.data(), /*numFrames=*/1, wideH, wideW, channels,
                         rawScratch, tmpScratch, dstImage, outHeight, outWidth, stream),
            std::runtime_error);
    }

    // Boundary: a raw height exactly at the cap is accepted.
    {
        int64_t const capH = kernel::kGpuResizeMaxRawDim, capW = 8, capOutH = 256, capOutW = 8;
        std::vector<unsigned char> atCap(static_cast<size_t>(capH) * capW * channels, 0);
        rt::Tensor capRaw({capH, capW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor capTmp({capH, capOutW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor capDst({1, capOutH, capOutW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        EXPECT_NO_THROW(kernel::copyImageToDeviceAndResize(
            atCap.data(), /*numFrames=*/1, capH, capW, channels, capRaw, capTmp, capDst, capOutH, capOutW, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    // Boundary: a raw width exactly at the cap is accepted.
    {
        int64_t const capH = 8, capW = kernel::kGpuResizeMaxRawDim, capOutH = 8, capOutW = 256;
        std::vector<unsigned char> atCap(static_cast<size_t>(capH) * capW * channels, 0);
        rt::Tensor capRaw({capH, capW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor capTmp({capH, capOutW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor capDst({1, capOutH, capOutW, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        EXPECT_NO_THROW(kernel::copyImageToDeviceAndResize(
            atCap.data(), /*numFrames=*/1, capH, capW, channels, capRaw, capTmp, capDst, capOutH, capOutW, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

// 3-D/4-D handoff: the helper writes through a 3-D view into a 4-D mImageDevice-shaped destination that
// the normalize step then reads in place — must run without a shape error and yield correct values.
TEST(CopyImageToDeviceAndResize, FourDimDestinationFeedsNormalize)
{
    cudaStream_t stream{nullptr};
    int32_t const inHeight = 480, inWidth = 640, outHeight = 224, outWidth = 224, channels = 3;

    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels);
    FillResizeInput(input, inHeight, inWidth, channels, ResizeFillPattern::kGRADIENT);

    rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmpScratch({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor imageDevice({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels, rawScratch,
        tmpScratch, imageDevice, outHeight, outWidth, stream);

    // Normalize reads the 4-D destination in place.
    std::vector<float> mean{0.5f, 0.5f, 0.5f};
    std::vector<float> stdv{0.5f, 0.5f, 0.5f};
    rt::Tensor meanDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor stdDevice({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor normalizedDevice({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpyAsync(
        meanDevice.rawPointer(), mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        stdDevice.rawPointer(), stdv.data(), stdv.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    ASSERT_NO_THROW(kernel::normalizeImage(imageDevice, meanDevice, stdDevice, normalizedDevice, stream));

    std::vector<unsigned char> resized(static_cast<size_t>(outHeight) * outWidth * channels);
    std::vector<half> normalized(resized.size());
    CUDA_CHECK(
        cudaMemcpyAsync(resized.data(), imageDevice.rawPointer(), resized.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(normalized.data(), normalizedDevice.rawPointer(), normalized.size() * sizeof(half),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < resized.size(); ++i)
    {
        float const expected = (static_cast<float>(resized[i]) / 255.0f - mean[i % channels]) / stdv[i % channels];
        ASSERT_TRUE(isclose(normalized[i], __float2half(expected), 1e-3f, 1e-3f))
            << "normalize of resized pixel " << i << " mismatched";
    }
    std::cout
        << "CopyImageToDeviceAndResize FourDimDestinationFeedsNormalize: resize+normalize through the 4-D buffer OK."
        << std::endl;
}

// A same-size resize takes the identity fast path, which bypasses the per-side cap: a pre-resized
// (doResize=false) image whose side exceeds kGpuResizeMaxRawDim still copies through cleanly.
TEST(CopyImageToDeviceAndResize, IdentitySkipAboveCap)
{
    cudaStream_t stream{nullptr};
    int64_t const height = kernel::kGpuResizeMaxRawDim + 1, width = 8, channels = 3; // one side over the cap
    std::vector<unsigned char> input(static_cast<size_t>(height) * width * channels);
    FillResizeInput(input, static_cast<int32_t>(height), static_cast<int32_t>(width), static_cast<int32_t>(channels),
        ResizeFillPattern::kRANDOM);

    rt::Tensor rawScratch({64, 64, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmpScratch({64, 64, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({1, height, width, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);

    EXPECT_NO_THROW(kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, height, width, channels,
        rawScratch, tmpScratch, dstImage, height, width, stream));

    std::vector<unsigned char> output(input.size());
    CUDA_CHECK(cudaMemcpyAsync(output.data(), dstImage.rawPointer(), output.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_EQ(output[i], input[i]) << "identity copy above the cap altered pixel " << i;
    }
    std::cout << "CopyImageToDeviceAndResize IdentitySkipAboveCap: " << height << "x" << width
              << " (over cap) copied verbatim via the identity path." << std::endl;
}

// Defensive contract: an undersized scratch or a wrong-typed/located destination throws std::runtime_error.
TEST(CopyImageToDeviceAndResize, RejectsBadScratchAndDst)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 512, inWidth = 512, outHeight = 256, outWidth = 256, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels, 0);
    rt::Tensor dstImage({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);

    // Undersized raw scratch: reshape to {inHeight,inWidth,C} exceeds its capacity.
    {
        rt::Tensor smallRaw({16, 16, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor tmp({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels,
                         smallRaw, tmp, dstImage, outHeight, outWidth, stream),
            std::runtime_error);
    }
    // Undersized horizontal-pass scratch: reshape to {inHeight,outWidth,C} exceeds its capacity.
    {
        rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor smallTmp({16, 16, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels,
                         rawScratch, smallTmp, dstImage, outHeight, outWidth, stream),
            std::runtime_error);
    }
    {
        rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor tmp({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor hostDst({1, outHeight, outWidth, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels,
                         rawScratch, tmp, hostDst, outHeight, outWidth, stream),
            std::runtime_error);
    }
    {
        rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        rt::Tensor tmp({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor halfDst({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        EXPECT_THROW(kernel::copyImageToDeviceAndResize(input.data(), /*numFrames=*/1, inHeight, inWidth, channels,
                         rawScratch, tmp, halfDst, outHeight, outWidth, stream),
            std::runtime_error);
    }
}

// Zero and negative raw or output dimensions are rejected before any reshape or device copy.
TEST(CopyImageToDeviceAndResize, RejectsNonPositiveDims)
{
    cudaStream_t stream{nullptr};
    int64_t const inHeight = 512, inWidth = 512, outHeight = 256, outWidth = 256, channels = 3;
    std::vector<unsigned char> input(static_cast<size_t>(inHeight) * inWidth * channels, 0);
    rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmp({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);

    auto launch = [&](int64_t rawH, int64_t rawW, int64_t outH, int64_t outW) {
        kernel::copyImageToDeviceAndResize(
            input.data(), /*numFrames=*/1, rawH, rawW, channels, rawScratch, tmp, dstImage, outH, outW, stream);
    };

    EXPECT_THROW(launch(0, inWidth, outHeight, outWidth), std::runtime_error);
    EXPECT_THROW(launch(inHeight, 0, outHeight, outWidth), std::runtime_error);
    EXPECT_THROW(launch(inHeight, inWidth, 0, outWidth), std::runtime_error);
    EXPECT_THROW(launch(inHeight, inWidth, outHeight, 0), std::runtime_error);
    EXPECT_THROW(launch(-inHeight, inWidth, outHeight, outWidth), std::runtime_error);
    EXPECT_THROW(launch(inHeight, inWidth, -outHeight, outWidth), std::runtime_error);
}

// Multi-frame (video) resize: numFrames > 1 reshapes the destination to [T, outH, outW, C] and resizes
// each frame independently. Frame t of the batched call must be byte-identical to a standalone
// single-frame resize of that frame, and an identity target must copy every frame verbatim.
TEST(CopyImageToDeviceAndResize, MultiFrameAccuracy)
{
    cudaStream_t stream{nullptr};
    int32_t const numFrames = 3, inHeight = 480, inWidth = 640, outHeight = 224, outWidth = 224, channels = 3;
    int64_t const inFrameBytes = static_cast<int64_t>(inHeight) * inWidth * channels;
    int64_t const outFrameBytes = static_cast<int64_t>(outHeight) * outWidth * channels;

    // Distinct content per frame so a frame mix-up would show.
    std::array<ResizeFillPattern, 3> const patterns{
        ResizeFillPattern::kGRADIENT, ResizeFillPattern::kRANDOM, ResizeFillPattern::kCHECKERBOARD};
    std::vector<unsigned char> input(static_cast<size_t>(numFrames) * inFrameBytes);
    for (int32_t t = 0; t < numFrames; ++t)
    {
        std::vector<unsigned char> frame(static_cast<size_t>(inFrameBytes));
        FillResizeInput(frame, inHeight, inWidth, channels, patterns[t]);
        std::memcpy(input.data() + t * inFrameBytes, frame.data(), frame.size());
    }

    rt::Tensor rawScratch({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor tmpScratch({inHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor dstImage({numFrames, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    kernel::copyImageToDeviceAndResize(input.data(), numFrames, inHeight, inWidth, channels, rawScratch, tmpScratch,
        dstImage, outHeight, outWidth, stream);
    std::vector<unsigned char> batched(static_cast<size_t>(numFrames) * outFrameBytes);
    CUDA_CHECK(cudaMemcpyAsync(batched.data(), dstImage.rawPointer(), batched.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Golden: a standalone single-frame resize of each frame; the batched output must match byte for byte.
    for (int32_t t = 0; t < numFrames; ++t)
    {
        rt::Tensor singleDst({1, outHeight, outWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
        kernel::copyImageToDeviceAndResize(input.data() + t * inFrameBytes, /*numFrames=*/1, inHeight, inWidth,
            channels, rawScratch, tmpScratch, singleDst, outHeight, outWidth, stream);
        std::vector<unsigned char> single(static_cast<size_t>(outFrameBytes));
        CUDA_CHECK(
            cudaMemcpyAsync(single.data(), singleDst.rawPointer(), single.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int64_t i = 0; i < outFrameBytes; ++i)
        {
            ASSERT_EQ(batched[t * outFrameBytes + i], single[i])
                << "multi-frame resize frame " << t << " differs from single-frame at " << i;
        }
    }

    // Identity multi-frame: out == in copies each frame verbatim.
    rt::Tensor idRaw({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    rt::Tensor idTmp({inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor idDst({numFrames, inHeight, inWidth, channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
    kernel::copyImageToDeviceAndResize(
        input.data(), numFrames, inHeight, inWidth, channels, idRaw, idTmp, idDst, inHeight, inWidth, stream);
    std::vector<unsigned char> identity(input.size());
    CUDA_CHECK(cudaMemcpyAsync(identity.data(), idDst.rawPointer(), identity.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < input.size(); ++i)
    {
        ASSERT_EQ(identity[i], input[i]) << "multi-frame identity copy altered byte " << i;
    }
    std::cout << "CopyImageToDeviceAndResize MultiFrameAccuracy: " << numFrames << " frames resized + identity OK."
              << std::endl;
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
