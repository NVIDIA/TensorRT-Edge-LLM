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

#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "references.h"
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

void TestInitAttentionMaskQwenViT(int32_t const curHW, int32_t const blockSize = 64)
{
    cudaStream_t stream{nullptr};

    std::vector<int64_t> cuSeqlens{0};
    int64_t currentPos = 0;
    while (currentPos < curHW)
    {
        currentPos = std::min(currentPos + blockSize, static_cast<int64_t>(curHW));
        cuSeqlens.push_back(currentPos);
    }

    half const disabledMaskValue{-20000.0F};
    std::vector<half> attentionMaskRef(curHW * curHW, disabledMaskValue);
    for (size_t s = 1; s < cuSeqlens.size(); ++s)
    {
        for (int64_t i = cuSeqlens[s - 1]; i < cuSeqlens[s]; ++i)
        {
            for (int64_t j = cuSeqlens[s - 1]; j < cuSeqlens[s]; ++j)
            {
                attentionMaskRef[i * curHW + j] = __float2half(0.0f);
            }
        }
    }

    int64_t const cuSeqlensSize = cuSeqlens.size();
    rt::Tensor cuSeqlensDevice({cuSeqlensSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    CUDA_CHECK(cudaMemcpyAsync(cuSeqlensDevice.rawPointer(), cuSeqlens.data(), cuSeqlensSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    rt::Tensor attentionMaskDevice({1, curHW, curHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    kernel::initAttentionMaskQwenViT(cuSeqlensDevice, attentionMaskDevice, stream);

    // Compare data
    std::vector<half> attentionMaskHost(curHW * curHW);
    CUDA_CHECK(cudaMemcpyAsync(attentionMaskHost.data(), attentionMaskDevice.rawPointer(),
        attentionMaskHost.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t i = 0; i < curHW * curHW; ++i)
    {
        ASSERT_TRUE(isclose(attentionMaskHost[i], attentionMaskRef[i], 1e-5, 1e-5));
    }

    std::cout << "InitAttentionMask Accuracy: curHW=" << curHW << std::endl;
}

TEST(InitAttentionMaskQwen, Accuracy)
{
    TestInitAttentionMaskQwenViT(1024);
    TestInitAttentionMaskQwenViT(2000);
}

void BenchmarkInitAttentionMaskQwenViT(int32_t const curHW, int32_t const blockSize = 64)
{
    cudaStream_t stream{nullptr};

    std::vector<int64_t> cuSeqlens{0};
    int64_t currentPos = 0;
    while (currentPos < curHW)
    {
        currentPos = std::min(currentPos + blockSize, static_cast<int64_t>(curHW));
        cuSeqlens.push_back(currentPos);
    }

    int64_t const cuSeqlensSize = cuSeqlens.size();
    rt::Tensor cuSeqlensDevice({cuSeqlensSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    CUDA_CHECK(cudaMemcpyAsync(cuSeqlensDevice.rawPointer(), cuSeqlens.data(), cuSeqlensSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    half const disabledMaskValue{-20000.0F};
    std::vector<half> attentionMask(curHW * curHW, disabledMaskValue);
    rt::Tensor attentionMaskDevice({1, curHW, curHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    auto launchGPU = [&]() { kernel::initAttentionMaskQwenViT(cuSeqlensDevice, attentionMaskDevice, stream); };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launchGPU();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchGPU();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "InitAttentionMask Benchmark: curHW=" << curHW << ", time=" << elapsedTime / numBenchIter << " ms"
              << std::endl;
}

TEST(InitAttentionMaskQwen, Benchmark)
{
    BenchmarkInitAttentionMaskQwenViT(1024);
    BenchmarkInitAttentionMaskQwenViT(4096);
}

void TestInitRotaryPosEmbQwenViT(int32_t const vitPosEmbDim = 40, int32_t const mergeSize = 2,
    float const rotaryBaseFrequency = 10000.0f, float const scale = 1.0f)
{
    cudaStream_t stream{nullptr};

    std::vector<std::vector<int64_t>> imageGridTHWs{{1, 36, 54}, {1, 8, 10}, {1, 32, 20}};
    std::vector<int64_t> cuSeqlens{0};
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        cuSeqlens.push_back(cuSeqlens.back() + imageGridTHWs[i][0] * imageGridTHWs[i][1] * imageGridTHWs[i][2]);
    }
    int64_t totalSeqLength = cuSeqlens.back();

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