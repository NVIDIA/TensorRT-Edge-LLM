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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cuda_fp16.h>
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

} // namespace

class EmbeddingLookupTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA device
        cudaSetDevice(0);
    }

    void TearDown() override
    {
        // Clean up any resources if needed
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
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult = embeddingLookupRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Standard Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize;
    }
}

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
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        { kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor); }, std::runtime_error)
        << "Kernel should error out when hiddenSize is not a multiple of 8";
}

// Test that image insertion kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, UnevenHiddenSizeErrorWithImageInsertion)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const imageTokenLen = 8;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize + imageTokenLen - 1);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        {
            kernel::embeddingLookupWithImageInsertion(
                inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor);
        },
        std::runtime_error)
        << "Image insertion kernel should error out when hiddenSize is not a multiple of 8";
}

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
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

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

// Test out-of-bounds token handling with image insertion
TEST_F(EmbeddingLookupTest, OutOfBoundsTokenHandlingWithImageInsertion)
{
    // Test case with out-of-bounds tokens and image tokens
    int64_t const batchSize = 1;
    int64_t const seqLen = 7;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 16;
    int64_t const imageTokenLen = 8;

    // Generate test data with mixed tokens: [-1, 0, 9, 10, 15, 20]
    // -1: out-of-bounds normal token (should be zero)
    // 0, 9: valid normal tokens
    // 10: out-of-bounds normal token (should be zero)
    // 15: valid image token (10 + 5)
    // 20: out-of-bounds image token (10 + 10, but imageTokenLen = 8)
    std::vector<int32_t> inputIds = {0, 9, 10, -1, 15, 20, -1};

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::embeddingLookupWithImageInsertion(inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Run CPU reference
    auto cpuResult = embeddingLookupRef(
        inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize, imageEmbeds, imageTokenLen);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Out-of-Bounds Token Handling with Image Insertion Test"))
        << "GPU and CPU results don't match for out-of-bounds token handling with image insertion";

    // Verify specific token behaviors
    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        int32_t const tokenId = inputIds[tokenIdx];
        bool const isImageToken = tokenId > (vocabSize - 1);
        bool isOutOfBounds = false; // Will be determined per token type

        if (isImageToken)
        {
            int32_t const visualTokenId = tokenId - vocabSize;
            isOutOfBounds = (visualTokenId < 0 || visualTokenId >= imageTokenLen);
        }
        else
        {
            isOutOfBounds = (tokenId < 0 || tokenId >= vocabSize);
        }

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

// Test embedding lookup with image insertion accuracy
TEST_F(EmbeddingLookupTest, EmbeddingLookupWithImageInsertionAccuracy)
{
    // Simple test cases for accuracy
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t>> testCases = {
        {1, 10, 10, 128, 64},  // Small test
        {2, 20, 50, 256, 128}, // Medium test
        {4, 50, 100, 128, 64}, // Large test
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", imageTokenLen=" + std::to_string(imageTokenLen));

        // Generate test data using testUtils
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, 0, vocabSize + imageTokenLen - 1);

        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half),
            cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::embeddingLookupWithImageInsertion(
            inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult = embeddingLookupRef(
            inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize, imageEmbeds, imageTokenLen);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Embedding Lookup with Image Insertion Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize << ", imageTokenLen=" << imageTokenLen;
    }
}
