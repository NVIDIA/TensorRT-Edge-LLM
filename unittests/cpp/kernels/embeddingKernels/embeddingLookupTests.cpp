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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#if SUPPORTS_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace trt_edgellm;

// Debug flag for detailed error reporting
static constexpr bool DEBUG_MODE = false;

namespace
{

// Helper function to compare results using direct half comparison
bool compareResults(
    std::vector<half> const& ref, std::vector<half> const& test, std::string const& testName = "Embedding Lookup")
{
    if (ref.size() != test.size())
    {
        std::cout << testName << " validation failed: size mismatch (ref=" << ref.size() << ", test=" << test.size()
                  << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (!isclose(test[i], ref[i], 1e-2, 1e-2))
        {
            std::cout << testName << " validation failed at index " << i << ": expected=" << __half2float(ref[i])
                      << ", got=" << __half2float(test[i]) << std::endl;
            return false;
        }
    }

    return true;
}

#if SUPPORTS_FP8
// Helper function to quantize FP16 embedding table to FP8 with per-group scales
// Uses max-abs scaling with FP8 E4M3 max value of 448
void quantizeToFP8PerGroup(std::vector<half> const& input, std::vector<__nv_fp8_e4m3>& outputFp8,
    std::vector<float>& scales, int64_t numRows, int64_t hiddenSize, int64_t blockSize)
{
    int64_t const nGroups = hiddenSize / blockSize;
    outputFp8.resize(numRows * hiddenSize);
    scales.resize(numRows * nGroups);

    for (int64_t row = 0; row < numRows; ++row)
    {
        for (int64_t group = 0; group < nGroups; ++group)
        {
            int64_t const base = row * hiddenSize + group * blockSize;

            // Find max absolute value in this block
            float amax = 0.0f;
            for (int64_t k = 0; k < blockSize; ++k)
            {
                amax = std::max(amax, std::fabs(__half2float(input[base + k])));
            }
            // Avoid division by zero
            amax = std::max(amax, 1.0e-4f);

            // Compute scale: scale = amax / fp8_max (448 for E4M3)
            float const scale = amax / 448.0f;
            scales[row * nGroups + group] = scale;

            // Quantize each element in the block
            float const invScale = 1.0f / scale;
            for (int64_t k = 0; k < blockSize; ++k)
            {
                float const value = __half2float(input[base + k]) * invScale;
                outputFp8[base + k] = static_cast<__nv_fp8_e4m3>(value);
            }
        }
    }
}

// CPU reference implementation for FP8 embedding lookup
std::vector<half> embeddingLookupFP8Ref(std::vector<int32_t> const& inputIds,
    std::vector<__nv_fp8_e4m3> const& embeddingTableFp8, std::vector<float> const& scales, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, int64_t blockSize)
{
    std::vector<half> result(batchSize * seqLen * hiddenSize, __float2half(0.0f));
    int64_t const nGroups = hiddenSize / blockSize;

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
        {
            int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];
            // Out-of-bounds tokens get zero embeddings
            if (tokenId < 0 || tokenId >= vocabSize)
            {
                continue;
            }

            int64_t const outBase = (batchIdx * seqLen + tokenIdx) * hiddenSize;
            int64_t const inBase = static_cast<int64_t>(tokenId) * hiddenSize;
            for (int64_t col = 0; col < hiddenSize; ++col)
            {
                int64_t const group = col / blockSize;
                float const scale = scales[static_cast<int64_t>(tokenId) * nGroups + group];
                float const value = static_cast<float>(embeddingTableFp8[inBase + col]) * scale;
                result[outBase + col] = __float2half(value);
            }
        }
    }

    return result;
}

// CPU reference implementation for FP8 embedding lookup with image/audio insertion
// CPU reference implementation for FP8 multimodal embedding lookup (Qwen3-Omni style)
std::vector<half> embeddingLookupMultimodalFP8Ref(std::vector<int32_t> const& inputIds,
    std::vector<__nv_fp8_e4m3> const& embeddingTableFp8, std::vector<float> const& scales,
    std::vector<int32_t> const& multimodalIndices, int32_t imageTokenId, std::vector<half> const& imageEmbeds,
    int64_t imageTokenLen, int32_t audioTokenId, std::vector<half> const& audioEmbeds, int64_t audioTokenLen,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, int64_t blockSize)
{
    std::vector<half> result(batchSize * seqLen * hiddenSize, __float2half(0.0f));
    int64_t const nGroups = hiddenSize / blockSize;

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
        {
            int64_t const linearIdx = batchIdx * seqLen + tokenIdx;
            int32_t const tokenId = inputIds[linearIdx];
            int64_t const outBase = linearIdx * hiddenSize;

            bool const isImageToken = (!imageEmbeds.empty() && tokenId == imageTokenId);
            bool const isAudioToken = (!audioEmbeds.empty() && tokenId == audioTokenId);

            if (isImageToken)
            {
                int32_t const imgIdx = multimodalIndices[linearIdx];
                if (imgIdx >= 0 && imgIdx < imageTokenLen)
                {
                    int64_t const imgBase = static_cast<int64_t>(imgIdx) * hiddenSize;
                    for (int64_t col = 0; col < hiddenSize; ++col)
                    {
                        result[outBase + col] = imageEmbeds[imgBase + col];
                    }
                }
            }
            else if (isAudioToken)
            {
                int32_t const audIdx = multimodalIndices[linearIdx];
                if (audIdx >= 0 && audIdx < audioTokenLen)
                {
                    int64_t const audBase = static_cast<int64_t>(audIdx) * hiddenSize;
                    for (int64_t col = 0; col < hiddenSize; ++col)
                    {
                        result[outBase + col] = audioEmbeds[audBase + col];
                    }
                }
            }
            else if (tokenId >= 0 && tokenId < vocabSize)
            {
                // Valid text token: lookup from FP8 table with dequantization
                int64_t const inBase = static_cast<int64_t>(tokenId) * hiddenSize;
                for (int64_t col = 0; col < hiddenSize; ++col)
                {
                    int64_t const group = col / blockSize;
                    float const scale = scales[static_cast<int64_t>(tokenId) * nGroups + group];
                    float const value = static_cast<float>(embeddingTableFp8[inBase + col]) * scale;
                    result[outBase + col] = __float2half(value);
                }
            }
            // Out-of-bounds: zero embedding (already initialized)
        }
    }

    return result;
}
#endif

} // namespace

class EmbeddingLookupTest : public ::testing::Test
{
protected:
    cudaStream_t stream;

    void SetUp() override
    {
        // Initialize CUDA device
        cudaSetDevice(0);

        // Create a non-default CUDA stream for testing
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        // Destroy the CUDA stream
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// Test standard embedding lookup accuracy
TEST_F(EmbeddingLookupTest, StandardEmbeddingLookupAccuracy)
{
    // Simple test cases for accuracy
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t>> testCases = {
        {1, 10, 10, 128},
        {2, 20, 50, 256},
        {4, 50, 100, 512},
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize));

        // Generate test data using testUtils
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        copyHostToDevice(inputIdsTensor, inputIds);
        copyHostToDevice(embeddingTableTensor, embeddingTable);

        // Run GPU kernel
        kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream);

        // Get result from GPU
        auto const gpuResult = copyDeviceToHost<half>(outputTensor);

        // Run CPU reference
        auto cpuResult = embeddingLookupRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Standard Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize;
    }
}

#if SUPPORTS_FP8
// Test FP8 embedding lookup accuracy with various configurations
TEST_F(EmbeddingLookupTest, FP8EmbeddingLookupAccuracy)
{
    // Test cases: {batchSize, seqLen, vocabSize, hiddenSize, blockSize}
    // All use production block size 128
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t>> testCases = {
        {1, 8, 32, 128, 128},   // Small test
        {2, 16, 64, 256, 128},  // Medium test
        {4, 32, 100, 512, 128}, // Large test
        {2, 16, 50, 256, 128},  // Another configuration
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, blockSize] : testCases)
    {
        int64_t const nGroups = hiddenSize / blockSize;

        SCOPED_TRACE("Testing FP8: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", blockSize=" + std::to_string(blockSize));

        // Generate random input IDs including some out-of-bounds values
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, -2, vocabSize + 1);

        // Generate random FP16 embedding table
        std::vector<half> embeddingTableFp16(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTableFp16, -1.0f, 1.0f);

        // Quantize to FP8 with per-group scales
        std::vector<__nv_fp8_e4m3> embeddingTableFp8;
        std::vector<float> scales;
        quantizeToFP8PerGroup(embeddingTableFp16, embeddingTableFp8, scales, vocabSize, hiddenSize, blockSize);

        // Create GPU tensors
        rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
        rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
            embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Run GPU kernel (unified interface)
        kernel::embeddingLookup(
            inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Compute CPU reference
        auto const cpuResult = embeddingLookupFP8Ref(
            inputIds, embeddingTableFp8, scales, batchSize, seqLen, vocabSize, hiddenSize, blockSize);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "FP8 Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for FP8 test case";
    }
}

// Test FP8 embedding lookup with out-of-bounds tokens (should produce zero embeddings)
TEST_F(EmbeddingLookupTest, FP8OutOfBoundsTokenHandling)
{
    int64_t const batchSize = 1;
    int64_t const seqLen = 6;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 128;
    int64_t const blockSize = 128;
    int64_t const nGroups = hiddenSize / blockSize;

    // Input IDs with out-of-bounds values: [-1, 0, 5, 9, 10, 100]
    std::vector<int32_t> inputIds = {-1, 0, 5, 9, 10, 100};

    // Generate FP16 embedding table
    std::vector<half> embeddingTableFp16(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTableFp16, -1.0f, 1.0f);

    // Quantize to FP8
    std::vector<__nv_fp8_e4m3> embeddingTableFp8;
    std::vector<float> scales;
    quantizeToFP8PerGroup(embeddingTableFp16, embeddingTableFp8, scales, vocabSize, hiddenSize, blockSize);

    // Create GPU tensors
    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
        embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Run GPU kernel (unified interface)
    kernel::embeddingLookup(
        inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream);

    // Get result from GPU
    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    // Compute CPU reference
    auto const cpuResult = embeddingLookupFP8Ref(
        inputIds, embeddingTableFp8, scales, batchSize, seqLen, vocabSize, hiddenSize, blockSize);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "FP8 Out-of-Bounds Token Handling Test"));

    // Verify specific out-of-bounds tokens produce zero embeddings
    std::vector<size_t> outOfBoundsIndices = {0, 4, 5}; // -1, 10, 100 are out-of-bounds
    for (auto const tokenIdx : outOfBoundsIndices)
    {
        for (int64_t elem = 0; elem < hiddenSize; ++elem)
        {
            size_t const idx = tokenIdx * hiddenSize + elem;
            EXPECT_TRUE(isclose(gpuResult[idx], __float2half(0.0f), 1e-6, 1e-6))
                << "Out-of-bounds token at position " << tokenIdx << " element " << elem
                << " should produce zero embedding";
        }
    }
}

// Test that FP8 kernel errors on uneven hidden size
TEST_F(EmbeddingLookupTest, FP8UnevenHiddenSizeError)
{
    int64_t const batchSize = 1;
    int64_t const seqLen = 4;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const nGroups = 1;

    std::vector<int32_t> inputIds(batchSize * seqLen, 0);
    std::vector<__nv_fp8_e4m3> embeddingTableFp8(vocabSize * hiddenSize);
    std::vector<float> scales(vocabSize * nGroups, 1.0f);

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(inputIdsTensor, inputIds);
    CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
        embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    EXPECT_THROW(
        {
            kernel::embeddingLookup(
                inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream);
        },
        std::runtime_error)
        << "FP8 kernel should error out when hiddenSize is not a multiple of 8";
}

// Test that FP8 kernel errors when block size is not aligned
TEST_F(EmbeddingLookupTest, FP8UnevenBlockSizeError)
{
    int64_t const batchSize = 1;
    int64_t const seqLen = 4;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 128;
    int64_t const nGroups = 17; // 128/17 is not integer, should fail

    std::vector<int32_t> inputIds(batchSize * seqLen, 0);
    std::vector<__nv_fp8_e4m3> embeddingTableFp8(vocabSize * hiddenSize);
    std::vector<float> scales(vocabSize * nGroups, 1.0f);

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(inputIdsTensor, inputIds);
    CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
        embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    EXPECT_THROW(
        {
            kernel::embeddingLookup(
                inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream);
        },
        std::runtime_error)
        << "FP8 kernel should error out when hiddenSize is not divisible by nGroups";
}

// Test FP8 embedding lookup with image insertion
// Test FP8 embedding lookup with image insertion handles out-of-bounds correctly
// Test FP8 multimodal embedding lookup (Qwen3-Omni style with audio + image)
TEST_F(EmbeddingLookupTest, FP8MultimodalAccuracy)
{
    // Test cases: {batchSize, seqLen, vocabSize, hiddenSize, blockSize, imageTokenLen, audioTokenLen}
    // All use production block size 128
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t, int64_t, int64_t>> testCases = {
        {1, 16, 100, 128, 128, 8, 4},  // Small test
        {2, 32, 200, 256, 128, 16, 8}, // Medium test
        {2, 24, 150, 256, 128, 12, 6}, // Large test
    };

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, blockSize, imageTokenLen, audioTokenLen] : testCases)
    {
        int64_t const nGroups = hiddenSize / blockSize;

        SCOPED_TRACE("Testing FP8 Multimodal: batchSize=" + std::to_string(batchSize)
            + ", seqLen=" + std::to_string(seqLen) + ", vocabSize=" + std::to_string(vocabSize)
            + ", hiddenSize=" + std::to_string(hiddenSize) + ", imageTokenLen=" + std::to_string(imageTokenLen)
            + ", audioTokenLen=" + std::to_string(audioTokenLen));

        // Generate input IDs with mix of text, image, and audio tokens
        std::vector<int32_t> inputIds(batchSize * seqLen);
        std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

        int32_t imageCounter = 0;
        int32_t audioCounter = 0;

        for (size_t i = 0; i < inputIds.size(); ++i)
        {
            int32_t choice = i % 6;
            if (choice == 0 && imageCounter < imageTokenLen)
            {
                inputIds[i] = imageTokenId;
                multimodalIndices[i] = imageCounter++;
            }
            else if (choice == 1 && audioCounter < audioTokenLen)
            {
                inputIds[i] = audioTokenId;
                multimodalIndices[i] = audioCounter++;
            }
            else if (choice == 2)
            {
                inputIds[i] = -1; // Out-of-bounds text
            }
            else if (choice == 3)
            {
                inputIds[i] = vocabSize + 10; // Out-of-bounds (> vocabSize)
            }
            else
            {
                inputIds[i] = (i * 7) % vocabSize; // Valid text token
            }
        }

        // Generate FP16 embedding table and quantize to FP8
        std::vector<half> embeddingTableFp16(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTableFp16, -1.0f, 1.0f);

        std::vector<__nv_fp8_e4m3> embeddingTableFp8;
        std::vector<float> scales;
        quantizeToFP8PerGroup(embeddingTableFp16, embeddingTableFp8, scales, vocabSize, hiddenSize, blockSize);

        // Generate image and audio embeddings (FP16)
        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

        std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
        uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

        // Create GPU tensors
        rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor multimodalIndicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
        rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor imageEmbedsTensor({imageTokenLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor audioEmbedsTensor({audioTokenLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        copyHostToDevice(inputIdsTensor, inputIds);
        copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
        CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
            embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
        copyHostToDevice(imageEmbedsTensor, imageEmbeds);
        copyHostToDevice(audioEmbedsTensor, audioEmbeds);

        // Run GPU kernel (unified interface)
        kernel::embeddingLookup(inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor,
            stream, multimodalIndicesTensor, imageTokenId, imageEmbedsTensor, audioTokenId, audioEmbedsTensor);

        // Get result from GPU
        auto const gpuResult = copyDeviceToHost<half>(outputTensor);

        // Compute CPU reference
        auto const cpuResult = embeddingLookupMultimodalFP8Ref(inputIds, embeddingTableFp8, scales, multimodalIndices,
            imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen, batchSize, seqLen,
            vocabSize, hiddenSize, blockSize);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "FP8 Multimodal Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for FP8 multimodal test case";
    }
}

// Test FP8 multimodal with image only (no audio)
TEST_F(EmbeddingLookupTest, FP8MultimodalImageOnly)
{
    int64_t const batchSize = 2;
    int64_t const seqLen = 16;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 128;
    int64_t const blockSize = 128;
    int64_t const nGroups = hiddenSize / blockSize;
    int64_t const imageTokenLen = 8;

    int32_t const imageTokenId = 33;

    // Generate inputs
    std::vector<int32_t> inputIds(batchSize * seqLen);
    std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

    int32_t imageCounter = 0;
    for (size_t i = 0; i < inputIds.size(); ++i)
    {
        if (i % 4 == 0 && imageCounter < imageTokenLen)
        {
            inputIds[i] = imageTokenId;
            multimodalIndices[i] = imageCounter++;
        }
        else
        {
            inputIds[i] = i % vocabSize;
        }
    }

    std::vector<half> embeddingTableFp16(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTableFp16, -1.0f, 1.0f);

    std::vector<__nv_fp8_e4m3> embeddingTableFp8;
    std::vector<float> scales;
    quantizeToFP8PerGroup(embeddingTableFp16, embeddingTableFp8, scales, vocabSize, hiddenSize, blockSize);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    // Create GPU tensors
    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor multimodalIndicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor imageEmbedsTensor({imageTokenLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
    CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
        embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    copyHostToDevice(imageEmbedsTensor, imageEmbeds);

    // Run GPU kernel with image only (no audio) - unified interface
    kernel::embeddingLookup(inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream,
        multimodalIndicesTensor, imageTokenId, imageEmbedsTensor, std::nullopt, std::nullopt);

    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    // CPU reference with empty audio
    auto const cpuResult = embeddingLookupMultimodalFP8Ref(inputIds, embeddingTableFp8, scales, multimodalIndices,
        imageTokenId, imageEmbeds, imageTokenLen, -1, {}, 0, batchSize, seqLen, vocabSize, hiddenSize, blockSize);

    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "FP8 Multimodal Image Only Test"));
}

// Test FP8 multimodal with text only (no image/audio)
TEST_F(EmbeddingLookupTest, FP8MultimodalTextOnly)
{
    int64_t const batchSize = 2;
    int64_t const seqLen = 16;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 128;
    int64_t const blockSize = 128;
    int64_t const nGroups = hiddenSize / blockSize;

    // Generate text-only inputs
    std::vector<int32_t> inputIds(batchSize * seqLen);
    for (size_t i = 0; i < inputIds.size(); ++i)
    {
        inputIds[i] = i % vocabSize;
    }

    std::vector<half> embeddingTableFp16(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTableFp16, -1.0f, 1.0f);

    std::vector<__nv_fp8_e4m3> embeddingTableFp8;
    std::vector<float> scales;
    quantizeToFP8PerGroup(embeddingTableFp16, embeddingTableFp8, scales, vocabSize, hiddenSize, blockSize);

    // Create GPU tensors
    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tableFp8Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    rt::Tensor scalesTensor({vocabSize, nGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    CUDA_CHECK(cudaMemcpy(tableFp8Tensor.rawPointer(), embeddingTableFp8.data(),
        embeddingTableFp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Run GPU kernel with no image/audio - unified interface
    kernel::embeddingLookup(inputIdsTensor, tableFp8Tensor, rt::OptionalInputTensor{scalesTensor}, outputTensor, stream,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    // CPU reference - should be same as basic FP8 lookup
    auto const cpuResult = embeddingLookupFP8Ref(
        inputIds, embeddingTableFp8, scales, batchSize, seqLen, vocabSize, hiddenSize, blockSize);

    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "FP8 Multimodal Text Only Test"));
}
#endif // SUPPORTS_FP8

// Test that kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, UnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 15; // Not a multiple of 8

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(embeddingTableTensor, embeddingTable);

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        { kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream); },
        std::runtime_error)
        << "Kernel should error out when hiddenSize is not a multiple of 8";
}

// Test that image insertion kernel properly errors out for uneven hiddenSize
// Test out-of-bounds token handling (should use zero embeddings)
TEST_F(EmbeddingLookupTest, OutOfBoundsTokenHandling)
{
    // Test case with out-of-bounds tokens
    int64_t const batchSize = 1;
    int64_t const seqLen = 4;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 16;

    // Generate test data with out-of-bounds tokens: [-1, 0, 9, 10]
    std::vector<int32_t> inputIds = {-1, 0, -1, 10}; // -1 and 10 are out-of-bounds

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(embeddingTableTensor, embeddingTable);

    // Run GPU kernel
    kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream);

    // Get result from GPU
    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    // Run CPU reference
    auto cpuResult = embeddingLookupRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Out-of-Bounds Token Handling Test"))
        << "GPU and CPU results don't match for out-of-bounds token handling";

    // Verify that out-of-bounds tokens produce zero embeddings
    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        int32_t const tokenId = inputIds[tokenIdx];
        bool const isOutOfBounds = (tokenId < 0 || tokenId >= vocabSize);

        if (isOutOfBounds)
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Out-of-bounds token " << tokenId << " should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test that deepstack kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, DeepstackUnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const numImageTokens = 10;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, vocabSize, vocabSize + numImageTokens - 1);

    std::vector<half> deepstackFeatures(numImageTokens * hiddenSize);
    uniformFloatInitialization<half>(deepstackFeatures, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords featuresShape{numImageTokens, hiddenSize};
    rt::Tensor deepstackFeaturesTensor(featuresShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(deepstackFeaturesTensor, deepstackFeatures);

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        { kernel::assembleDeepstackEmbedding(inputIdsTensor, deepstackFeaturesTensor, outputTensor, stream); },
        std::runtime_error)
        << "Deepstack kernel should error out when hiddenSize is not a multiple of 8";
}

// Test deepstack embedding with explicit imageTokenId and multimodalIndices (Qwen3-Omni path)
TEST_F(EmbeddingLookupTest, DeepstackEmbeddingExplicitImageTokenId)
{
    int64_t const batchSize = 1;
    int64_t const seqLen = 8;
    int64_t const hiddenSize = 128;
    int64_t const numImageTokens = 4;
    int32_t const imageTokenId = 42;

    std::vector<int32_t> inputIds = {10, imageTokenId, 20, imageTokenId, 30, imageTokenId, 40, imageTokenId};
    std::vector<int32_t> multimodalIndices = {0, 0, 0, 1, 0, 2, 0, 3};

    std::vector<half> deepstackFeatures(numImageTokens * hiddenSize);
    uniformFloatInitialization<half>(deepstackFeatures, -1.0f, 1.0f);

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor indicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor featuresTensor({numImageTokens, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor outputTensor({batchSize, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(indicesTensor, multimodalIndices);
    copyHostToDevice(featuresTensor, deepstackFeatures);

    kernel::assembleDeepstackEmbedding(
        inputIdsTensor, featuresTensor, outputTensor, stream, imageTokenId, std::ref(indicesTensor));

    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        bool const isImage = (inputIds[tokenIdx] == imageTokenId);
        for (int64_t elem = 0; elem < hiddenSize; ++elem)
        {
            int64_t const idx = tokenIdx * hiddenSize + elem;
            if (isImage)
            {
                int32_t const featureIdx = multimodalIndices[tokenIdx];
                half const expected = deepstackFeatures[featureIdx * hiddenSize + elem];
                EXPECT_TRUE(isclose(gpuResult[idx], expected, 1e-6, 1e-6))
                    << "Image token at position " << tokenIdx << " element " << elem << " mismatch";
            }
            else
            {
                EXPECT_TRUE(isclose(gpuResult[idx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Text token at position " << tokenIdx << " element " << elem << " should be zero";
            }
        }
    }
}

// Test Qwen3-Omni multimodal embedding lookup accuracy
TEST_F(EmbeddingLookupTest, MultimodalAccuracy)
{
    // Test cases with varying sizes
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t, int64_t>> testCases = {
        // {batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen, audioTokenLen}
        {1, 16, 100, 128, 8, 8},   // Small test
        {2, 32, 200, 128, 16, 9},  // Medium test
        {4, 64, 250, 128, 32, 10}, // Large test
    };

    // Special token IDs (similar to Qwen3-Omni)
    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen, audioTokenLen] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", imageTokenLen=" + std::to_string(imageTokenLen) + ", audioTokenLen=" + std::to_string(audioTokenLen));

        // Generate test data
        // Create inputIds with a mix of text tokens, image tokens, and audio tokens
        std::vector<int32_t> inputIds(batchSize * seqLen);
        std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

        int32_t imageCounter = 0;
        int32_t audioCounter = 0;

        for (int64_t i = 0; i < batchSize * seqLen; ++i)
        {
            int32_t choice = i % 5; // Distribute tokens across types
            if (choice == 0 && imageCounter < imageTokenLen)
            {
                // Image token
                inputIds[i] = imageTokenId;
                multimodalIndices[i] = imageCounter++;
            }
            else if (choice == 1 && audioCounter < audioTokenLen)
            {
                // Audio token
                inputIds[i] = audioTokenId;
                multimodalIndices[i] = audioCounter++;
            }
            else
            {
                // Text token (random valid token ID)
                inputIds[i] = i % vocabSize;
            }
        }

        // Generate embedding tables
        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

        std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
        uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords audioShape{audioTokenLen, hiddenSize};
        rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        copyHostToDevice(inputIdsTensor, inputIds);
        copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
        copyHostToDevice(embeddingTableTensor, embeddingTable);
        copyHostToDevice(imageEmbedsTensor, imageEmbeds);
        copyHostToDevice(audioEmbedsTensor, audioEmbeds);

        // Run GPU kernel
        kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream,
            std::optional{std::ref(multimodalIndicesTensor)}, imageTokenId, std::optional{std::ref(imageEmbedsTensor)},
            audioTokenId, std::optional{std::ref(audioEmbedsTensor)});

        // Get result from GPU
        auto const gpuResult = copyDeviceToHost<half>(outputTensor);

        // Run CPU reference
        auto cpuResult
            = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
                multimodalIndices, imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Qwen3-Omni Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize << ", imageTokenLen=" << imageTokenLen
            << ", audioTokenLen=" << audioTokenLen;
    }
}

// Test Qwen3-Omni embedding lookup with out-of-bounds handling
TEST_F(EmbeddingLookupTest, MultimodalOutOfBounds)
{
    // Test case with specific tokens to verify out-of-bounds handling
    int64_t const batchSize = 1;
    int64_t const seqLen = 10;
    int32_t const vocabSize = 1000;
    int64_t const hiddenSize = 128;
    int64_t const imageTokenLen = 8;
    int64_t const audioTokenLen = 4;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    // Create inputIds with specific patterns:
    // - Text tokens (valid and invalid)
    // - Image tokens (valid and invalid indices)
    // - Audio tokens (valid and invalid indices)
    std::vector<int32_t> inputIds = {
        100,          // Valid text token
        imageTokenId, // Valid image token (index 0)
        audioTokenId, // Valid audio token (index 0)
        imageTokenId, // Valid image token (index 1)
        audioTokenId, // Valid audio token (index 1)
        -1,           // Invalid text token (out of bounds)
        imageTokenId, // Image token with out-of-bounds index
        audioTokenId, // Audio token with out-of-bounds index
        2000,         // Invalid text token (> vocabSize)
        500,          // Valid text token
    };

    std::vector<int32_t> multimodalIndices = {
        0,  // Ignored (text token)
        0,  // Valid image index
        0,  // Valid audio index
        1,  // Valid image index
        1,  // Valid audio index
        0,  // Ignored (text token)
        10, // Invalid image index (>= imageTokenLen)
        -1, // Invalid audio index (< 0)
        0,  // Ignored (text token)
        0,  // Ignored (text token)
    };

    // Generate embedding tables
    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
    uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords audioShape{audioTokenLen, hiddenSize};
    rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
    copyHostToDevice(embeddingTableTensor, embeddingTable);
    copyHostToDevice(imageEmbedsTensor, imageEmbeds);
    copyHostToDevice(audioEmbedsTensor, audioEmbeds);

    // Run GPU kernel
    kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream,
        std::optional{std::ref(multimodalIndicesTensor)}, imageTokenId, std::optional{std::ref(imageEmbedsTensor)},
        audioTokenId, std::optional{std::ref(audioEmbedsTensor)});

    // Get result from GPU
    auto const gpuResult = copyDeviceToHost<half>(outputTensor);

    // Run CPU reference
    auto cpuResult = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
        multimodalIndices, imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Qwen3-Omni Embedding Lookup Out-of-Bounds Test"))
        << "GPU and CPU results don't match for out-of-bounds handling";

    // Verify specific token behaviors
    std::vector<bool> shouldBeZero = {
        false, // Valid text token
        false, // Valid image token
        false, // Valid audio token
        false, // Valid image token
        false, // Valid audio token
        true,  // Invalid text token (-1)
        true,  // Image token with out-of-bounds index (10)
        true,  // Audio token with out-of-bounds index (-1)
        true,  // Invalid text token (2000 > vocabSize)
        false, // Valid text token
    };

    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        if (shouldBeZero[tokenIdx])
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Token at position " << tokenIdx << " (tokenId=" << inputIds[tokenIdx]
                    << ", multimodalIndices=" << multimodalIndices[tokenIdx]
                    << ") should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test that Qwen3-Omni kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, MultimodalUnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 10;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const imageTokenLen = 4;
    int64_t const audioTokenLen = 4;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

    std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
    uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords audioShape{audioTokenLen, hiddenSize};
    rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
    copyHostToDevice(embeddingTableTensor, embeddingTable);
    copyHostToDevice(imageEmbedsTensor, imageEmbeds);
    copyHostToDevice(audioEmbedsTensor, audioEmbeds);

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        {
            kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream,
                std::optional{std::ref(multimodalIndicesTensor)}, imageTokenId,
                std::optional{std::ref(imageEmbedsTensor)}, audioTokenId, std::optional{std::ref(audioEmbedsTensor)});
        },
        std::runtime_error)
        << "Qwen3-Omni kernel should error out when hiddenSize is not a multiple of 8";
}

// Test multimodal embedding lookup with optional inputs (image only, audio only, text only)
TEST_F(EmbeddingLookupTest, MultimodalOptionalInputs)
{
    struct TestCase
    {
        bool hasImage;
        bool hasAudio;
        std::string name;
    };

    std::vector<TestCase> testCases = {
        {true, false, "TextImageOnly"},
        {false, true, "TextAudioOnly"},
        {false, false, "TextOnly"},
    };

    int64_t const batchSize = 2;
    int64_t const seqLen = 32;
    int32_t const vocabSize = 200;
    int64_t const hiddenSize = 128;
    int64_t const imageTokenLen = 16;
    int64_t const audioTokenLen = 12;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    for (auto const& tc : testCases)
    {
        SCOPED_TRACE("Testing: " + tc.name);

        // Create inputIds with appropriate token mix
        std::vector<int32_t> inputIds(batchSize * seqLen);
        std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

        int32_t imageCounter = 0;
        int32_t audioCounter = 0;
        for (int64_t i = 0; i < batchSize * seqLen; ++i)
        {
            if (tc.hasImage && i % 4 == 0 && imageCounter < imageTokenLen)
            {
                inputIds[i] = imageTokenId;
                multimodalIndices[i] = imageCounter++;
            }
            else if (tc.hasAudio && i % 5 == 0 && audioCounter < audioTokenLen)
            {
                inputIds[i] = audioTokenId;
                multimodalIndices[i] = audioCounter++;
            }
            else
            {
                inputIds[i] = i % vocabSize;
            }
        }

        // Generate embedding tables
        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
        if (tc.hasImage)
        {
            uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);
        }
        if (tc.hasAudio)
        {
            uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);
        }

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords audioShape{audioTokenLen, hiddenSize};
        rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        copyHostToDevice(inputIdsTensor, inputIds);
        copyHostToDevice(multimodalIndicesTensor, multimodalIndices);
        copyHostToDevice(embeddingTableTensor, embeddingTable);
        if (tc.hasImage)
        {
            copyHostToDevice(imageEmbedsTensor, imageEmbeds);
        }
        if (tc.hasAudio)
        {
            copyHostToDevice(audioEmbedsTensor, audioEmbeds);
        }

        // Set up optional parameters for kernel call
        bool const hasMultimodal = tc.hasImage || tc.hasAudio;
        rt::OptionalInputTensor multimodalIndicesOpt
            = hasMultimodal ? rt::OptionalInputTensor(multimodalIndicesTensor) : std::nullopt;
        std::optional<int32_t> imageTokenIdOpt = tc.hasImage ? std::optional(imageTokenId) : std::nullopt;
        rt::OptionalInputTensor imageEmbedsOpt
            = tc.hasImage ? rt::OptionalInputTensor(imageEmbedsTensor) : std::nullopt;
        std::optional<int32_t> audioTokenIdOpt = tc.hasAudio ? std::optional(audioTokenId) : std::nullopt;
        rt::OptionalInputTensor audioEmbedsOpt
            = tc.hasAudio ? rt::OptionalInputTensor(audioEmbedsTensor) : std::nullopt;

        // Run GPU kernel
        kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, std::nullopt, outputTensor, stream,
            multimodalIndicesOpt, imageTokenIdOpt, imageEmbedsOpt, audioTokenIdOpt, audioEmbedsOpt);

        // Get result from GPU
        auto const gpuResult = copyDeviceToHost<half>(outputTensor);

        // Run CPU reference
        auto cpuResult
            = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
                multimodalIndices, tc.hasImage ? imageTokenId : -1, tc.hasImage ? imageEmbeds : std::vector<half>{},
                tc.hasImage ? imageTokenLen : 0, tc.hasAudio ? audioTokenId : -1,
                tc.hasAudio ? audioEmbeds : std::vector<half>{}, tc.hasAudio ? audioTokenLen : 0);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Multimodal Embedding Lookup " + tc.name + " Test"))
            << "GPU and CPU results don't match for " << tc.name << " multimodal lookup";
    }
}

// Test generateMultimodalIndices GPU kernel without per-row offsets (legacy global accumulation)
TEST_F(EmbeddingLookupTest, GenerateMultimodalIndicesGlobalAccumulation)
{
    int32_t constexpr kImageTok = 50;
    int32_t constexpr kAudioTok = 99;
    int64_t constexpr batchSize = 2;
    int64_t constexpr seqLen = 4;

    // batch 0: [text, image, image, text]  -> image indices 0, 1
    // batch 1: [text, image, text, audio]  -> image index 2 (global), audio index 0
    std::vector<int32_t> inputIds = {10, kImageTok, kImageTok, 20, 30, kImageTok, 40, kAudioTok};

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor indicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(inputIdsTensor, inputIds);

    kernel::generateMultimodalIndices(inputIdsTensor, indicesTensor, kImageTok, kAudioTok, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result = copyDeviceToHost<int32_t>(indicesTensor);
    // Without per-row offsets, image counter accumulates globally: 0, 1, 2
    std::vector<int32_t> expected = {0, 0, 1, 0, 0, 2, 0, 0};
    EXPECT_EQ(result, expected);
}

// Test generateMultimodalIndices GPU kernel with per-row offsets.
TEST_F(EmbeddingLookupTest, GenerateMultimodalIndicesPerRowOffsets)
{
    int32_t constexpr kImageTok = 50;
    int32_t constexpr kAudioTok = 99;
    int64_t constexpr batchSize = 2;
    int64_t constexpr seqLen = 4;

    // batch 0: [text, image, image, text]  -> image indices starting at offset 3: 3, 4
    // batch 1: [text, image, text, audio]  -> image index starting at offset 7: 7; audio starting at 2: 2
    std::vector<int32_t> inputIds = {10, kImageTok, kImageTok, 20, 30, kImageTok, 40, kAudioTok};
    std::vector<int32_t> imageOffsets = {3, 7};
    std::vector<int32_t> audioOffsets = {0, 2};

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor indicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor imageOffsetTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor audioOffsetTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(imageOffsetTensor, imageOffsets);
    copyHostToDevice(audioOffsetTensor, audioOffsets);

    kernel::generateMultimodalIndices(inputIdsTensor, indicesTensor, kImageTok, kAudioTok, stream,
        imageOffsetTensor.dataPointer<int32_t>(), audioOffsetTensor.dataPointer<int32_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result = copyDeviceToHost<int32_t>(indicesTensor);
    // batch 0: text=0, image=3, image=4, text=0
    // batch 1: text=0, image=7, text=0, audio=2
    std::vector<int32_t> expected = {0, 3, 4, 0, 0, 7, 0, 2};
    EXPECT_EQ(result, expected);
}

// Test per-row offsets when context reuse skips media tokens in a prefix.
// When batch_size=2 and context cache skips a prefix containing image tokens,
// each row must start indexing at the number of image tokens already consumed by its prefix.
TEST_F(EmbeddingLookupTest, GenerateMultimodalIndicesContextReuseScenario)
{
    int32_t constexpr kImageTok = 50;
    int64_t constexpr batchSize = 2;
    int64_t constexpr seqLen = 3;

    // Simulates suffix-only tokens after context cache reuse:
    // Row 0 prefix had 10 image tokens (skipped), suffix has 2 more image tokens
    // Row 1 prefix had 5 image tokens (skipped), suffix has 1 more image token
    // Without the fix, both rows would start at 0, causing row 1 to read row 0's embeddings.
    std::vector<int32_t> inputIds = {kImageTok, kImageTok, 20, kImageTok, 30, 40};
    std::vector<int32_t> imageOffsets = {10, 5}; // cumulative tokens consumed by prefix

    rt::Tensor inputIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor indicesTensor({batchSize, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor imageOffsetTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    copyHostToDevice(inputIdsTensor, inputIds);
    copyHostToDevice(imageOffsetTensor, imageOffsets);

    kernel::generateMultimodalIndices(
        inputIdsTensor, indicesTensor, kImageTok, std::nullopt, stream, imageOffsetTensor.dataPointer<int32_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result = copyDeviceToHost<int32_t>(indicesTensor);
    // Row 0: image=10, image=11, text=0
    // Row 1: image=5, text=0, text=0
    std::vector<int32_t> expected = {10, 11, 0, 5, 0, 0};
    EXPECT_EQ(result, expected);
}
