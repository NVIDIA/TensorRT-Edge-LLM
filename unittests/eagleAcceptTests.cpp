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
#include "kernels/speculative/eagleAcceptKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

using namespace trt_edgellm;

class EagleAcceptTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        // Cleanup is handled by individual tests
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // Common test runner that handles GPU memory management, kernel execution, and result validation
    void runEagleAcceptTest(std::vector<int32_t> const& tokenIds, std::vector<int8_t> const& attentionMask,
        std::vector<float> const& logits, int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth,
        std::string const& testName,
        std::function<void(std::vector<int32_t> const&, std::vector<int32_t> const&, std::vector<int32_t> const&,
            EagleAcceptResult const&)>
            validator
        = nullptr,
        std::vector<int32_t> const& vocabMappingTableData = {})
    {
        // Create GPU tensors with proper shapes
        rt::Tensor logitsTensor(
            {batchSize * numTokens, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
        rt::Tensor tokenIdsTensor({batchSize, numTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
        rt::Tensor attentionMaskTensor(
            {batchSize, numTokens, numTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "attentionMask");
        rt::Tensor acceptedTokenIdsTensor(
            {batchSize, maxDepth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedTokenIds");
        rt::Tensor acceptedLogitsIndicesTensor(
            {batchSize, maxDepth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedLogitsIndices");
        rt::Tensor acceptLengthTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptLength");

        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpy(logitsTensor.rawPointer(), logits.data(),
            batchSize * numTokens * vocabSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tokenIdsTensor.rawPointer(), tokenIds.data(), batchSize * numTokens * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(attentionMaskTensor.rawPointer(), attentionMask.data(),
            batchSize * numTokens * numTokens * sizeof(int8_t), cudaMemcpyHostToDevice));

        // Allocate workspace for kernel temporary storage
        size_t workspaceSize = kernel::getEagleAcceptWorkspaceSize(batchSize, numTokens);
        void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

        // Setup vocab mapping table if provided
        rt::OptionalInputTensor vocabMappingTable = std::nullopt;
        rt::Tensor vocabMappingTableTensor;
        if (!vocabMappingTableData.empty())
        {
            vocabMappingTableTensor = rt::Tensor({static_cast<int64_t>(vocabMappingTableData.size())},
                rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "vocabMappingTable");
            CUDA_CHECK(cudaMemcpy(vocabMappingTableTensor.rawPointer(), vocabMappingTableData.data(),
                vocabMappingTableData.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            vocabMappingTable = std::ref(vocabMappingTableTensor);
        }

        // Execute kernel with timing
        auto start = std::chrono::high_resolution_clock::now();
        EXPECT_NO_THROW({
            kernel::eagleAccept(logitsTensor, tokenIdsTensor, attentionMaskTensor, acceptedTokenIdsTensor,
                acceptedLogitsIndicesTensor, acceptLengthTensor, vocabMappingTable, workspace, workspaceSize, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
        });
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Cleanup workspace
        CUDA_CHECK(cudaFree(workspace));

        // Run reference implementation for comparison
        EagleAcceptResult refResult = eagleAcceptRef(
            logits, tokenIds, attentionMask, batchSize, numTokens, vocabSize, maxDepth, vocabMappingTableData);

        // Copy results back to host for validation
        std::vector<int32_t> hostAcceptedTokenIds(batchSize * maxDepth);
        std::vector<int32_t> hostAcceptedLogitsIndices(batchSize * maxDepth);
        std::vector<int32_t> hostAcceptLengths(batchSize);

        CUDA_CHECK(cudaMemcpy(hostAcceptedTokenIds.data(), acceptedTokenIdsTensor.rawPointer(),
            batchSize * maxDepth * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostAcceptedLogitsIndices.data(), acceptedLogitsIndicesTensor.rawPointer(),
            batchSize * maxDepth * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostAcceptLengths.data(), acceptLengthTensor.rawPointer(), batchSize * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

        // Basic performance check
        EXPECT_LT(duration.count(), 1000) << testName << ": Kernel took too long, possible infinite loop";

        // Validate results against reference implementation
        for (int32_t b = 0; b < batchSize; ++b)
        {
            EXPECT_EQ(hostAcceptLengths[b], refResult.acceptLengths[b])
                << testName << ": Accept length mismatch at batch " << b;

            // Check accepted tokens and indices
            for (int32_t i = 0; i < hostAcceptLengths[b]; ++i)
            {
                int32_t idx = b * maxDepth + i;
                EXPECT_EQ(hostAcceptedTokenIds[idx], refResult.acceptedTokenIds[b * refResult.maxAcceptLength + i])
                    << testName << ": Token ID mismatch at batch " << b << " position " << i;
                EXPECT_EQ(
                    hostAcceptedLogitsIndices[idx], refResult.acceptedLogitsIndices[b * refResult.maxAcceptLength + i])
                    << testName << ": Logits index mismatch at batch " << b << " position " << i;
            }

            // Check that unused positions are properly initialized to 0
            // (Changed from -1 to 0 to avoid embedding lookup issues in draft model)
            for (int32_t i = hostAcceptLengths[b]; i < maxDepth; ++i)
            {
                int32_t idx = b * maxDepth + i;
                EXPECT_EQ(hostAcceptedTokenIds[idx], 0)
                    << testName << ": Unused token ID position should be 0 at batch " << b << " position " << i;
                EXPECT_EQ(hostAcceptedLogitsIndices[idx], -1)
                    << testName << ": Unused logits index position should be -1 at batch " << b << " position " << i;
            }
        }

        // Run custom validation if provided
        if (validator)
        {
            validator(hostAcceptedTokenIds, hostAcceptedLogitsIndices, hostAcceptLengths, refResult);
        }
        std::cout << testName << " - Duration: " << duration.count() << "ms" << std::endl;
        std::cout << "Accept lengths: ";
        for (int32_t b = 0; b < batchSize; ++b)
        {
            std::cout << hostAcceptLengths[b] << " ";
        }
        std::cout << std::endl;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            std::cout << "Batch " << b << " accepted path: ";
            for (int32_t i = 0; i < hostAcceptLengths[b]; ++i)
            {
                std::cout << hostAcceptedTokenIds[b * maxDepth + i] << " ";
            }
            std::cout << std::endl;
        }
    }
    cudaStream_t stream;
};

// Test basic multi-batch functionality with simple chains
TEST_F(EagleAcceptTest, MultiBatchSimple)
{
    constexpr int32_t numTokens = 3;
    constexpr int32_t batchSize = 2;
    constexpr int32_t vocabSize = 10;
    constexpr int32_t maxDepth = 3;

    /*
     * BATCH 0: Tree [1->2->3], logits pos0->2, pos1->3, pos2->0(default)
     * Expected: [2,3,0] indices=[0,1,2] length=3
     *
     * BATCH 1: Tree [4->5,6], logits pos0->5, pos1->0(default)
     * Expected: [5,0] indices=[0,1] length=2 (token6 isolated, no path)
     */

    // Setup token IDs: batch 0 = [1,2,3], batch 1 = [4,5,6]
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0 * numTokens + 0] = 1;
    tokenIds[0 * numTokens + 1] = 2;
    tokenIds[0 * numTokens + 2] = 3;
    tokenIds[1 * numTokens + 0] = 4;
    tokenIds[1 * numTokens + 1] = 5;
    tokenIds[1 * numTokens + 2] = 6;

    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: full chain attention pattern (1->2->3)
    attentionMask[0 * numTokens * numTokens + 0 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 1 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 1 * numTokens + 1] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 0] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 1] = 1;
    attentionMask[0 * numTokens * numTokens + 2 * numTokens + 2] = 1;

    // Batch 1: partial chain (4->5, no continuation to 6)
    attentionMask[1 * numTokens * numTokens + 0 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 1] = 1;

    // Setup logits to favor the expected path with more realistic values
    std::vector<float> logits(batchSize * numTokens * vocabSize, -5.0f);

    // Add some noise to avoid uniform values
    for (int32_t i = 0; i < batchSize * numTokens * vocabSize; ++i)
    {
        logits[i] += (i % 7) * 0.01f; // Small variation to break ties consistently
    }

    logits[0 * numTokens * vocabSize + 0 * vocabSize + 2] = 5.0f; // pos 0 -> token 2
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 3] = 5.0f; // pos 1 -> token 3
    logits[0 * numTokens * vocabSize + 2 * vocabSize + 0] = 5.0f; // pos 2 -> token 0 (default)
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 5] = 5.0f; // pos 0 -> token 5
    logits[1 * numTokens * vocabSize + 1 * vocabSize + 0] = 5.0f; // pos 1 -> token 0 (default)

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth, "PerBatchDesignTest",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Batch 0 should accept 3 tokens (2->3->something)";
            EXPECT_EQ(acceptLengths[1], 2) << "Batch 1 should accept 2 tokens (5->something)";

            EXPECT_EQ(acceptedTokenIds[0 * 3 + 0], 2) << "Batch 0 token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 1], 3) << "Batch 0 token 1";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 0], 0) << "Batch 0 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 1], 1) << "Batch 0 logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 2], 2) << "Batch 0 logits index 2";

            EXPECT_EQ(acceptedTokenIds[1 * 3 + 0], 5) << "Batch 1 token 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 0], 0) << "Batch 1 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 1], 1) << "Batch 1 logits index 1";

            EXPECT_EQ(acceptedTokenIds[1 * 3 + 2], 0) << "Batch 1 unused token position should be 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 2], -1) << "Batch 1 unused logits index position should be -1";
        });
}

// Test complex multi-batch scenario with varying tree structures and early termination
TEST_F(EagleAcceptTest, MultiBatchAsymmetricTree)
{
    constexpr int32_t numTokens = 5;
    constexpr int32_t batchSize = 6;
    constexpr int32_t vocabSize = 150;
    constexpr int32_t maxDepth = 5;

    /*
     * BATCH 0: Tree [90->91->92->93->94], logits 91,92,93,94,0 -> [91,92,93,94,0] length=5
     * BATCH 1: Tree [90->95->96], logits 95,96,149(not in tree) -> [95,96,149] length=3
     * BATCH 2: Tree [90->97->98->99->89], logits 97,98,99,89,0 -> [97,98,99,89,0] length=5
     * BATCH 3: Tree [90->88], logits 88,149(not in tree) -> [88,149] length=2
     * BATCH 4: Tree [90->87->86->85->84], logits 87,86,85,84,0 -> [87,86,85,84,0] length=5
     * BATCH 5: Tree [90], logits 149(not in tree) -> [149] length=1
     */

    std::vector<int32_t> tokenIds(batchSize * numTokens);
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: long chain [90->91->92->93->94]
    tokenIds[0 * numTokens + 0] = 90;
    tokenIds[0 * numTokens + 1] = 91;
    tokenIds[0 * numTokens + 2] = 92;
    tokenIds[0 * numTokens + 3] = 93;
    tokenIds[0 * numTokens + 4] = 94;

    // Batch 1: short chain [90->95->96] with early termination
    tokenIds[1 * numTokens + 0] = 90;
    tokenIds[1 * numTokens + 1] = 95;
    tokenIds[1 * numTokens + 2] = 96;
    tokenIds[1 * numTokens + 3] = 97;
    tokenIds[1 * numTokens + 4] = 98;

    // Batch 2: different long chain [90->97->98->99->89]
    tokenIds[2 * numTokens + 0] = 90;
    tokenIds[2 * numTokens + 1] = 97;
    tokenIds[2 * numTokens + 2] = 98;
    tokenIds[2 * numTokens + 3] = 99;
    tokenIds[2 * numTokens + 4] = 89;

    // Batch 3: minimal chain [90->88] with early termination
    tokenIds[3 * numTokens + 0] = 90;
    tokenIds[3 * numTokens + 1] = 88;
    tokenIds[3 * numTokens + 2] = 87;
    tokenIds[3 * numTokens + 3] = 86;
    tokenIds[3 * numTokens + 4] = 85;

    // Batch 4: full chain [90->87->86->85->84]
    tokenIds[4 * numTokens + 0] = 90;
    tokenIds[4 * numTokens + 1] = 87;
    tokenIds[4 * numTokens + 2] = 86;
    tokenIds[4 * numTokens + 3] = 85;
    tokenIds[4 * numTokens + 4] = 84;

    // Batch 5: no continuation [90] (logits favor non-existent tokens)
    tokenIds[5 * numTokens + 0] = 90;
    tokenIds[5 * numTokens + 1] = 83;
    tokenIds[5 * numTokens + 2] = 82;
    tokenIds[5 * numTokens + 3] = 81;
    tokenIds[5 * numTokens + 4] = 80;

    // Define valid token count per batch for attention mask setup
    int32_t validTokensPerBatch[] = {5, 3, 5, 2, 5, 5};

    // Create triangular attention masks for each batch based on valid token count
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < validTokensPerBatch[b]; ++i)
        {
            for (int32_t j = 0; j <= i; ++j)
            {
                attentionMask[b * numTokens * numTokens + i * numTokens + j] = 1;
            }
        }
    }

    // Setup logits to control which tokens are selected at each position
    std::vector<float> logits(batchSize * numTokens * vocabSize, -5.0f);

    // Add some noise to avoid uniform values and ensure consistent tie-breaking
    for (int32_t i = 0; i < batchSize * numTokens * vocabSize; ++i)
    {
        logits[i] += (i % 13) * 0.01f; // Small variation to break ties consistently
    }

    // Batch 0: favor path [90->91->92->93->94]
    logits[0 * numTokens * vocabSize + 0 * vocabSize + 91] = 10.0f;
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 92] = 10.0f;
    logits[0 * numTokens * vocabSize + 2 * vocabSize + 93] = 10.0f;
    logits[0 * numTokens * vocabSize + 3 * vocabSize + 94] = 10.0f;
    logits[0 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f; // final position -> token 0

    // Batch 1: favor path [90->95->96], then non-existent token for early termination
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 95] = 10.0f;
    logits[1 * numTokens * vocabSize + 1 * vocabSize + 96] = 10.0f;
    logits[1 * numTokens * vocabSize + 2 * vocabSize + 149] = 10.0f; // non-existent token
    logits[1 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f;   // default token
    logits[1 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f;   // default token

    // Batch 2: favor path [90->97->98->99->89]
    logits[2 * numTokens * vocabSize + 0 * vocabSize + 97] = 10.0f;
    logits[2 * numTokens * vocabSize + 1 * vocabSize + 98] = 10.0f;
    logits[2 * numTokens * vocabSize + 2 * vocabSize + 99] = 10.0f;
    logits[2 * numTokens * vocabSize + 3 * vocabSize + 89] = 10.0f;
    logits[2 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f; // final position -> token 0

    // Batch 3: favor path [90->88], then non-existent token for early termination
    logits[3 * numTokens * vocabSize + 0 * vocabSize + 88] = 10.0f;
    logits[3 * numTokens * vocabSize + 1 * vocabSize + 149] = 10.0f; // non-existent token
    logits[3 * numTokens * vocabSize + 2 * vocabSize + 0] = 10.0f;   // default token
    logits[3 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f;   // default token
    logits[3 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f;   // default token

    // Batch 4: favor path [90->87->86->85->84]
    logits[4 * numTokens * vocabSize + 0 * vocabSize + 87] = 10.0f;
    logits[4 * numTokens * vocabSize + 1 * vocabSize + 86] = 10.0f;
    logits[4 * numTokens * vocabSize + 2 * vocabSize + 85] = 10.0f;
    logits[4 * numTokens * vocabSize + 3 * vocabSize + 84] = 10.0f;
    logits[4 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f; // final position -> token 0

    // Batch 5: favor non-existent token immediately for no continuation
    logits[5 * numTokens * vocabSize + 0 * vocabSize + 149] = 15.0f; // non-existent token
    logits[5 * numTokens * vocabSize + 0 * vocabSize + 83] = 5.0f;   // valid but lower priority
    logits[5 * numTokens * vocabSize + 1 * vocabSize + 0] = 10.0f;   // default token
    logits[5 * numTokens * vocabSize + 2 * vocabSize + 0] = 10.0f;   // default token
    logits[5 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f;   // default token
    logits[5 * numTokens * vocabSize + 4 * vocabSize + 0] = 10.0f;   // default token

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "AsymmetricTreeMultiBatch",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 5) << "Batch 0: long path A";
            EXPECT_EQ(acceptLengths[1], 3) << "Batch 1: short path B";
            EXPECT_EQ(acceptLengths[2], 5) << "Batch 2: longest path C";
            EXPECT_EQ(acceptLengths[3], 2) << "Batch 3: isolated path";
            EXPECT_EQ(acceptLengths[4], 5) << "Batch 4: full chain";
            EXPECT_EQ(acceptLengths[5], 1) << "Batch 5: at least 1 token accepted";

            // Batch 0: path 90->91->92->93->94, logits predict 91,92,93,94
            EXPECT_EQ(acceptedTokenIds[0 * 5 + 0], 91) << "Batch 0 path A token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 5 + 1], 92) << "Batch 0 path A token 1";
            EXPECT_EQ(acceptedTokenIds[0 * 5 + 2], 93) << "Batch 0 path A token 2";
            EXPECT_EQ(acceptedTokenIds[0 * 5 + 3], 94) << "Batch 0 path A token 3";
            EXPECT_EQ(acceptedLogitsIndices[0 * 5 + 0], 0) << "Batch 0 path A logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[0 * 5 + 1], 1) << "Batch 0 path A logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[0 * 5 + 2], 2) << "Batch 0 path A logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[0 * 5 + 3], 3) << "Batch 0 path A logits index 3";

            // Batch 1: path 90->95->96, logits predict 95,96, then non-existent
            EXPECT_EQ(acceptedTokenIds[1 * 5 + 0], 95) << "Batch 1 path B token 0";
            EXPECT_EQ(acceptedTokenIds[1 * 5 + 1], 96) << "Batch 1 path B token 1";
            EXPECT_EQ(acceptedLogitsIndices[1 * 5 + 0], 0) << "Batch 1 path B logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 5 + 1], 1) << "Batch 1 path B logits index 1";

            // Batch 2: path 90->97->98->99->89, logits predict 97,98,99,89
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 0], 97) << "Batch 2 path C token 0";
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 1], 98) << "Batch 2 path C token 1";
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 2], 99) << "Batch 2 path C token 2";
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 3], 89) << "Batch 2 path C token 3";
            EXPECT_EQ(acceptedLogitsIndices[2 * 5 + 0], 0) << "Batch 2 path C logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[2 * 5 + 1], 1) << "Batch 2 path C logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[2 * 5 + 2], 2) << "Batch 2 path C logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[2 * 5 + 3], 3) << "Batch 2 path C logits index 3";

            // Batch 3: path 90->88, logits predict 88, then non-existent
            EXPECT_EQ(acceptedTokenIds[3 * 5 + 0], 88) << "Batch 3 isolated token 0";
            EXPECT_EQ(acceptedLogitsIndices[3 * 5 + 0], 0) << "Batch 3 isolated logits index 0";

            // Batch 4: path 90->87->86->85->84, logits predict 87,86,85,84
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 0], 87) << "Batch 4 chain token 0";
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 1], 86) << "Batch 4 chain token 1";
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 2], 85) << "Batch 4 chain token 2";
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 3], 84) << "Batch 4 chain token 3";
            EXPECT_EQ(acceptedLogitsIndices[4 * 5 + 0], 0) << "Batch 4 chain logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[4 * 5 + 1], 1) << "Batch 4 chain logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[4 * 5 + 2], 2) << "Batch 4 chain logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[4 * 5 + 3], 3) << "Batch 4 chain logits index 3";

            // Check final accepted tokens for each batch
            // Batch 0: accepts 5 tokens [91, 92, 93, 94, 0] - the last token (0) is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[0 * 5 + 4], 0) << "Batch 0 final token should be 0 (predicted but not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[0 * 5 + 4], 4) << "Batch 0 final logits position should be 4";

            // Batch 1: accepts 3 tokens [95, 96, 149] - token 149 is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[1 * 5 + 2], 149)
                << "Batch 1 final token should be 149 (predicted but not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[1 * 5 + 2], 2) << "Batch 1 final logits position should be 2";

            // Batch 2: accepts 5 tokens [97, 98, 99, 89, 0] - the last token (0) is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[2 * 5 + 4], 0) << "Batch 2 final token should be 0 (predicted but not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[2 * 5 + 4], 4) << "Batch 2 final logits position should be 4";

            // Batch 3: accepts 2 tokens [88, 149] - token 149 is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[3 * 5 + 1], 149)
                << "Batch 3 final token should be 149 (predicted but not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[3 * 5 + 1], 1) << "Batch 3 final logits position should be 1";

            // Batch 4: accepts 5 tokens [87, 86, 85, 84, 0] - the last token (0) is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[4 * 5 + 4], 0) << "Batch 4 final token should be 0 (predicted but not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[4 * 5 + 4], 4) << "Batch 4 final logits position should be 4";

            // Batch 5: accepts 1 token [149] - token 149 is predicted but not in tree
            EXPECT_EQ(acceptedTokenIds[5 * 5 + 0], 149) << "Batch 5 should accept first predicted token";
            EXPECT_EQ(acceptedLogitsIndices[5 * 5 + 0], 0) << "Batch 5 logits index 0";

            // Check that positions beyond accept length are -1
            for (int32_t b = 0; b < 6; ++b)
            {
                for (int32_t i = acceptLengths[b]; i < 5; ++i)
                {
                    EXPECT_EQ(acceptedTokenIds[b * 5 + i], 0)
                        << "Batch " << b << " unused position " << i << " should be -1";
                    EXPECT_EQ(acceptedLogitsIndices[b * 5 + i], -1)
                        << "Batch " << b << " unused logits position " << i << " should be -1";
                }
            }
        });
}

// Test multi-batch trees with different path lengths and attention patterns
TEST_F(EagleAcceptTest, ComplexMultiBranchTree)
{
    constexpr int32_t numTokens = 4;
    constexpr int32_t vocabSize = 1000;
    constexpr int32_t maxDepth = 4;
    constexpr int32_t batchSize = 3;

    /*
     * BATCH 0: Tree [100->200->300->400], logits 200,300,400,0 -> [200,300,400,0] length=4
     * BATCH 1: Tree [100->202->302->402], logits 202,302,402,0 -> [202,302,402,0] length=4
     * BATCH 2: Tree [100->201->304], logits 201,304,999(not in tree) -> [201,304,999] length=3
     */

    std::vector<int32_t> tokenIds(batchSize * numTokens);
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: full chain [100->200->300->400]
    tokenIds[0 * numTokens + 0] = 100;
    tokenIds[0 * numTokens + 1] = 200;
    tokenIds[0 * numTokens + 2] = 300;
    tokenIds[0 * numTokens + 3] = 400;

    // Create full triangular attention mask for batch 0
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[0 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Batch 1: different full chain [100->202->302->402]
    tokenIds[1 * numTokens + 0] = 100;
    tokenIds[1 * numTokens + 1] = 202;
    tokenIds[1 * numTokens + 2] = 302;
    tokenIds[1 * numTokens + 3] = 402;

    // Create full triangular attention mask for batch 1
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[1 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Batch 2: partial chain [100->201->304] with early termination
    tokenIds[2 * numTokens + 0] = 100;
    tokenIds[2 * numTokens + 1] = 201;
    tokenIds[2 * numTokens + 2] = 304;
    tokenIds[2 * numTokens + 3] = 999; // invalid token

    // Create partial triangular attention mask for batch 2 (only first 3 tokens)
    for (int32_t i = 0; i < 3; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[2 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Setup logits to guide token selection
    std::vector<float> logits(batchSize * numTokens * vocabSize, -5.0f);

    // Add some noise to avoid uniform values and ensure consistent tie-breaking
    for (int32_t i = 0; i < batchSize * numTokens * vocabSize; ++i)
    {
        logits[i] += (i % 17) * 0.01f; // Small variation to break ties consistently
    }

    // Batch 0: favor path [100->200->300->400]
    logits[0 * numTokens * vocabSize + 0 * vocabSize + 200] = 10.0f;
    logits[0 * numTokens * vocabSize + 1 * vocabSize + 300] = 10.0f;
    logits[0 * numTokens * vocabSize + 2 * vocabSize + 400] = 10.0f;
    logits[0 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f; // final position -> token 0

    // Batch 1: favor path [100->202->302->402]
    logits[1 * numTokens * vocabSize + 0 * vocabSize + 202] = 10.0f;
    logits[1 * numTokens * vocabSize + 1 * vocabSize + 302] = 10.0f;
    logits[1 * numTokens * vocabSize + 2 * vocabSize + 402] = 10.0f;
    logits[1 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f; // final position -> token 0

    // Batch 2: favor path [100->201->304], then non-existent token
    logits[2 * numTokens * vocabSize + 0 * vocabSize + 201] = 10.0f;
    logits[2 * numTokens * vocabSize + 1 * vocabSize + 304] = 10.0f;
    logits[2 * numTokens * vocabSize + 2 * vocabSize + 999] = 10.0f; // non-existent token
    logits[2 * numTokens * vocabSize + 3 * vocabSize + 0] = 10.0f;   // default token

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "ComplexMultiBranchTree",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 4) << "Batch 0 should complete full path";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 0], 200) << "Batch 0 token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 1], 300) << "Batch 0 token 1";
            EXPECT_EQ(acceptedTokenIds[0 * 4 + 2], 400) << "Batch 0 token 2";
            EXPECT_EQ(acceptedLogitsIndices[0 * 4 + 0], 0) << "Batch 0 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[0 * 4 + 1], 1) << "Batch 0 logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[0 * 4 + 2], 2) << "Batch 0 logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[0 * 4 + 3], 3) << "Batch 0 logits index 3";

            EXPECT_EQ(acceptLengths[1], 4) << "Batch 1 should complete different path";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 0], 202) << "Batch 1 token 0";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 1], 302) << "Batch 1 token 1";
            EXPECT_EQ(acceptedTokenIds[1 * 4 + 2], 402) << "Batch 1 token 2";
            EXPECT_EQ(acceptedLogitsIndices[1 * 4 + 0], 0) << "Batch 1 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 4 + 1], 1) << "Batch 1 logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[1 * 4 + 2], 2) << "Batch 1 logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[1 * 4 + 3], 3) << "Batch 1 logits index 3";

            EXPECT_EQ(acceptLengths[2], 3) << "Batch 2 should terminate after 999 not found";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 0], 201) << "Batch 2 token 0";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 1], 304) << "Batch 2 token 1";
            EXPECT_EQ(acceptedTokenIds[2 * 4 + 2], 999) << "Batch 2 token 2 (not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[2 * 4 + 0], 0) << "Batch 2 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[2 * 4 + 1], 1) << "Batch 2 logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[2 * 4 + 2], 2) << "Batch 2 logits index 2";

            EXPECT_EQ(acceptedTokenIds[2 * 4 + 3], 0) << "Batch 2 unused token position should be 0";
            EXPECT_EQ(acceptedLogitsIndices[2 * 4 + 3], -1) << "Batch 2 unused logits index position should be -1";
        });
}

// Test single batch with logit-controlled early termination
TEST_F(EagleAcceptTest, SingleBatchLogitTermination)
{
    constexpr int32_t numTokens = 5;
    constexpr int32_t batchSize = 1;
    constexpr int32_t vocabSize = 50;
    constexpr int32_t maxDepth = 5;

    /*
     * Tree [15->25->35->45->49], logits pos0->25, pos1->35, pos2->48(not in tree, higher than 45)
     * Expected: [25,35,48] indices=[0,1,2] length=3 (stops when 48 not found in tree)
     */

    // Setup a complete tree but use logits to cause early termination
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0] = 15; // root
    tokenIds[1] = 25; // depth 2
    tokenIds[2] = 35; // depth 3
    tokenIds[3] = 45; // depth 4
    tokenIds[4] = 49; // depth 5

    // Create full triangular attention mask (all tokens can attend to previous ones)
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[i * numTokens + j] = 1;
        }
    }

    // Setup logits to cause termination at position 2
    std::vector<float> logits(batchSize * numTokens * vocabSize, -5.0f);

    // Add some noise to avoid uniform values and ensure consistent tie-breaking
    for (int32_t i = 0; i < batchSize * numTokens * vocabSize; ++i)
    {
        logits[i] += (i % 19) * 0.01f; // Small variation to break ties consistently
    }

    logits[0 * vocabSize + 25] = 10.0f; // pos 0 -> token 25 (valid)
    logits[1 * vocabSize + 35] = 10.0f; // pos 1 -> token 35 (valid)
    logits[2 * vocabSize + 48] = 15.0f; // pos 2 -> token 48 (not in tree, higher priority)
    logits[2 * vocabSize + 45] = 5.0f;  // pos 2 -> token 45 (valid but lower priority)
    logits[3 * vocabSize + 0] = 10.0f;  // pos 3 -> token 0 (default)
    logits[4 * vocabSize + 0] = 10.0f;  // pos 4 -> token 0 (default)

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth,
        "SingleBatchLogitTermination",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Should accept 3 tokens: 25->35->48";
            EXPECT_EQ(acceptedTokenIds[0], 25) << "First predicted token";
            EXPECT_EQ(acceptedTokenIds[1], 35) << "Second predicted token";
            EXPECT_EQ(acceptedTokenIds[2], 48) << "Third predicted token (not in tree)";
            EXPECT_EQ(acceptedLogitsIndices[0], 0) << "First logits index";
            EXPECT_EQ(acceptedLogitsIndices[1], 1) << "Second logits index";
            EXPECT_EQ(acceptedLogitsIndices[2], 2) << "Third logits index";

            EXPECT_EQ(acceptedTokenIds[3], 0) << "Unused token position should be 0";
            EXPECT_EQ(acceptedLogitsIndices[3], -1) << "Unused logits index position should be -1";
            EXPECT_EQ(acceptedTokenIds[4], 0) << "Unused token position should be 0";
            EXPECT_EQ(acceptedLogitsIndices[4], -1) << "Unused logits index position should be -1";
        });
}

// Test case where predicted tokens are not in the tree - should still accept them
TEST_F(EagleAcceptTest, TokensNotInTree)
{
    constexpr int32_t numTokens = 3;
    constexpr int32_t batchSize = 1;
    constexpr int32_t vocabSize = 100;
    constexpr int32_t maxDepth = 3;

    /*
     * Tree [10->20->30], logits pos0->99(not in tree)
     * Expected: [99] indices=[0] length=1 (immediate termination)
     */

    // Setup token IDs: [10, 20, 30]
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0] = 10;
    tokenIds[1] = 20;
    tokenIds[2] = 30;

    // Setup triangular attention mask (all tokens attend to previous ones)
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[i * numTokens + j] = 1;
        }
    }

    // Setup logits to predict tokens NOT in the tree
    std::vector<float> logits(batchSize * numTokens * vocabSize, -5.0f);

    // Add some noise to avoid uniform values and ensure consistent tie-breaking
    for (int32_t i = 0; i < batchSize * numTokens * vocabSize; ++i)
    {
        logits[i] += (i % 23) * 0.01f; // Small variation to break ties consistently
    }

    logits[0 * vocabSize + 99] = 10.0f; // pos 0 -> token 99 (not in tree)
    logits[1 * vocabSize + 98] = 10.0f; // pos 1 -> token 98 (not in tree)
    logits[2 * vocabSize + 97] = 10.0f; // pos 2 -> token 97 (not in tree)

    runEagleAcceptTest(tokenIds, attentionMask, logits, batchSize, numTokens, vocabSize, maxDepth, "TokensNotInTree",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 1) << "Should accept at least 1 token (99) even if not in tree";

            EXPECT_EQ(acceptedTokenIds[0], 99) << "Should accept first predicted token (99)";
            EXPECT_EQ(acceptedLogitsIndices[0], 0) << "Should use logits[0]";

            EXPECT_EQ(acceptedTokenIds[1], 0) << "Should not accept more tokens since 99 not in tree";
            EXPECT_EQ(acceptedLogitsIndices[1], -1) << "Should not use more logits";
            EXPECT_EQ(acceptedTokenIds[2], 0) << "Should not accept more tokens";
            EXPECT_EQ(acceptedLogitsIndices[2], -1) << "Should not use more logits";
        });
}

// Test device validation - should reject CPU tensors
TEST_F(EagleAcceptTest, DeviceValidation)
{
    // Create one CPU tensor (logits) while others are on GPU - should cause validation failure
    rt::Tensor logitsTensor({2 * 4, 10}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "logits");
    rt::Tensor tokenIdsTensor({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
    rt::Tensor attentionMaskTensor({2, 4, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "attentionMask");
    rt::Tensor acceptedTokenIdsTensor({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedTokenIds");
    rt::Tensor acceptedLogitsIndicesTensor(
        {2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedLogitsIndices");
    rt::Tensor acceptLengthTensor({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptLength");

    // Allocate workspace (still needed for function call)
    size_t workspaceSize = kernel::getEagleAcceptWorkspaceSize(2, 4);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Kernel should throw due to CPU tensor
    rt::OptionalInputTensor vocabMappingTable = std::nullopt;
    EXPECT_THROW(
        kernel::eagleAccept(logitsTensor, tokenIdsTensor, attentionMaskTensor, acceptedTokenIdsTensor,
            acceptedLogitsIndicesTensor, acceptLengthTensor, vocabMappingTable, workspace, workspaceSize, stream),
        std::runtime_error)
        << "Should reject CPU tensor";

    CUDA_CHECK(cudaFree(workspace));
}

// Test vocabulary reduction with mapping table - single batch
TEST_F(EagleAcceptTest, VocabularyReductionSingleBatch)
{
    constexpr int32_t numTokens = 4;
    constexpr int32_t batchSize = 1;
    constexpr int32_t reducedVocabSize = 8; // Reduced vocabulary (output by model)
    constexpr int32_t maxDepth = 4;

    /*
     * Reduced vocab [0,1,2,3,4,5,6,7] maps to full vocab [50,51,52,53,54,55,56,57]
     * Tree [50->51->52->53] in full vocab space
     * Logits predict [1,2,3,0] in reduced vocab -> maps to [51,52,53,50] in full vocab
     * Expected: [51,52,53,50] indices=[0,1,2,3] length=4
     */

    // Setup vocab mapping table: reduced vocab index -> full vocab token
    std::vector<int32_t> vocabMappingTable(reducedVocabSize);
    for (int32_t i = 0; i < reducedVocabSize; ++i)
    {
        vocabMappingTable[i] = 50 + i; // Maps 0->50, 1->51, ..., 7->57
    }

    // Setup token IDs in full vocab space: [50, 51, 52, 53]
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0] = 50;
    tokenIds[1] = 51;
    tokenIds[2] = 52;
    tokenIds[3] = 53;

    // Setup triangular attention mask (full chain)
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[i * numTokens + j] = 1;
        }
    }

    // Setup logits in REDUCED vocab space
    std::vector<float> logits(batchSize * numTokens * reducedVocabSize, -5.0f);

    // Add noise
    for (int32_t i = 0; i < batchSize * numTokens * reducedVocabSize; ++i)
    {
        logits[i] += (i % 7) * 0.01f;
    }

    // Favor path in reduced vocab: [1,2,3,0] -> maps to full vocab [51,52,53,50]
    logits[0 * reducedVocabSize + 1] = 10.0f; // pos 0 -> reduced 1 -> full 51
    logits[1 * reducedVocabSize + 2] = 10.0f; // pos 1 -> reduced 2 -> full 52
    logits[2 * reducedVocabSize + 3] = 10.0f; // pos 2 -> reduced 3 -> full 53
    logits[3 * reducedVocabSize + 0] = 10.0f; // pos 3 -> reduced 0 -> full 50

    runEagleAcceptTest(
        tokenIds, attentionMask, logits, batchSize, numTokens, reducedVocabSize, maxDepth,
        "VocabularyReductionSingleBatch",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 4) << "Should accept all 4 tokens in the chain";

            // Verify accepted tokens are in FULL vocab space after mapping
            EXPECT_EQ(acceptedTokenIds[0], 51) << "Token 0: reduced 1 -> full 51";
            EXPECT_EQ(acceptedTokenIds[1], 52) << "Token 1: reduced 2 -> full 52";
            EXPECT_EQ(acceptedTokenIds[2], 53) << "Token 2: reduced 3 -> full 53";
            EXPECT_EQ(acceptedTokenIds[3], 50) << "Token 3: reduced 0 -> full 50";

            EXPECT_EQ(acceptedLogitsIndices[0], 0) << "Logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[1], 1) << "Logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[2], 2) << "Logits index 2";
            EXPECT_EQ(acceptedLogitsIndices[3], 3) << "Logits index 3";
        },
        vocabMappingTable);
}

// Test vocabulary reduction with multi-batch
TEST_F(EagleAcceptTest, VocabularyReductionMultiBatch)
{
    constexpr int32_t numTokens = 3;
    constexpr int32_t batchSize = 2;
    constexpr int32_t reducedVocabSize = 16; // Reduced vocabulary
    constexpr int32_t maxDepth = 3;

    /*
     * Reduced vocab [0-15] maps to full vocab [100-115]
     * BATCH 0: Tree [100->101->102], logits [1,2,0] reduced -> [101,102,100] full
     * BATCH 1: Tree [105->106], logits [6,7] reduced -> [106,107] full (107 not in tree)
     */

    // Setup vocab mapping table: reduced vocab index -> full vocab token
    std::vector<int32_t> vocabMappingTable(reducedVocabSize);
    for (int32_t i = 0; i < reducedVocabSize; ++i)
    {
        vocabMappingTable[i] = 100 + i; // Maps 0->100, 1->101, ..., 15->115
    }

    // Setup token IDs in full vocab space
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0 * numTokens + 0] = 100;
    tokenIds[0 * numTokens + 1] = 101;
    tokenIds[0 * numTokens + 2] = 102;
    tokenIds[1 * numTokens + 0] = 105;
    tokenIds[1 * numTokens + 1] = 106;
    tokenIds[1 * numTokens + 2] = 107; // This won't be matched

    // Setup attention masks
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);

    // Batch 0: full chain [100->101->102]
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[0 * numTokens * numTokens + i * numTokens + j] = 1;
        }
    }

    // Batch 1: partial chain [105->106] (107 not attended)
    attentionMask[1 * numTokens * numTokens + 0 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 0] = 1;
    attentionMask[1 * numTokens * numTokens + 1 * numTokens + 1] = 1;

    // Setup logits in REDUCED vocab space
    std::vector<float> logits(batchSize * numTokens * reducedVocabSize, -5.0f);

    // Add noise
    for (int32_t i = 0; i < batchSize * numTokens * reducedVocabSize; ++i)
    {
        logits[i] += (i % 11) * 0.01f;
    }

    // Batch 0: favor path [1,2,0] in reduced vocab -> [101,102,100] in full vocab
    logits[0 * numTokens * reducedVocabSize + 0 * reducedVocabSize + 1] = 10.0f;
    logits[0 * numTokens * reducedVocabSize + 1 * reducedVocabSize + 2] = 10.0f;
    logits[0 * numTokens * reducedVocabSize + 2 * reducedVocabSize + 0] = 10.0f;

    // Batch 1: favor path [6,7] in reduced vocab -> [106,107] in full vocab
    logits[1 * numTokens * reducedVocabSize + 0 * reducedVocabSize + 6] = 10.0f;
    logits[1 * numTokens * reducedVocabSize + 1 * reducedVocabSize + 7] = 10.0f;

    runEagleAcceptTest(
        tokenIds, attentionMask, logits, batchSize, numTokens, reducedVocabSize, maxDepth,
        "VocabularyReductionMultiBatch",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Batch 0 should accept all 3 tokens";
            EXPECT_EQ(acceptLengths[1], 2) << "Batch 1 should accept 2 tokens (107 not in tree)";

            // Batch 0: verify mapped tokens
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 0], 101) << "Batch 0 token 0";
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 1], 102) << "Batch 0 token 1";
            EXPECT_EQ(acceptedTokenIds[0 * 3 + 2], 100) << "Batch 0 token 2";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 0], 0) << "Batch 0 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 1], 1) << "Batch 0 logits index 1";
            EXPECT_EQ(acceptedLogitsIndices[0 * 3 + 2], 2) << "Batch 0 logits index 2";

            // Batch 1: verify mapped tokens
            EXPECT_EQ(acceptedTokenIds[1 * 3 + 0], 106) << "Batch 1 token 0";
            EXPECT_EQ(acceptedTokenIds[1 * 3 + 1], 107) << "Batch 1 token 1 (not in tree, terminates)";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 0], 0) << "Batch 1 logits index 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 1], 1) << "Batch 1 logits index 1";

            EXPECT_EQ(acceptedTokenIds[1 * 3 + 2], 0) << "Batch 1 unused position should be 0";
            EXPECT_EQ(acceptedLogitsIndices[1 * 3 + 2], -1) << "Batch 1 unused logits index should be -1";
        },
        vocabMappingTable);
}

// Test vocabulary reduction with edge case: mapping to very large token IDs
TEST_F(EagleAcceptTest, VocabularyReductionLargeTokenIds)
{
    constexpr int32_t numTokens = 3;
    constexpr int32_t batchSize = 1;
    constexpr int32_t reducedVocabSize = 10;
    constexpr int32_t maxDepth = 3;

    /*
     * Test that vocab mapping works with large token IDs (e.g., 32000+)
     * Reduced vocab [0-9] maps to full vocab [32000,32001,...,32009]
     * Tree [32000->32001->32002]
     */

    // Setup vocab mapping table with large token IDs
    std::vector<int32_t> vocabMappingTable(reducedVocabSize);
    for (int32_t i = 0; i < reducedVocabSize; ++i)
    {
        vocabMappingTable[i] = 32000 + i;
    }

    // Setup token IDs
    std::vector<int32_t> tokenIds(batchSize * numTokens);
    tokenIds[0] = 32000;
    tokenIds[1] = 32001;
    tokenIds[2] = 32002;

    // Setup triangular attention mask
    std::vector<int8_t> attentionMask(batchSize * numTokens * numTokens, 0);
    for (int32_t i = 0; i < numTokens; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            attentionMask[i * numTokens + j] = 1;
        }
    }

    // Setup logits in reduced vocab space
    std::vector<float> logits(batchSize * numTokens * reducedVocabSize, -5.0f);
    logits[0 * reducedVocabSize + 1] = 10.0f; // reduced 1 -> full 32001
    logits[1 * reducedVocabSize + 2] = 10.0f; // reduced 2 -> full 32002
    logits[2 * reducedVocabSize + 0] = 10.0f; // reduced 0 -> full 32000

    runEagleAcceptTest(
        tokenIds, attentionMask, logits, batchSize, numTokens, reducedVocabSize, maxDepth,
        "VocabularyReductionLargeTokenIds",
        [](auto const& acceptedTokenIds, auto const& acceptedLogitsIndices, auto const& acceptLengths, auto const&) {
            EXPECT_EQ(acceptLengths[0], 3) << "Should accept all 3 tokens";

            EXPECT_EQ(acceptedTokenIds[0], 32001) << "Token 0: reduced 1 -> full 32001";
            EXPECT_EQ(acceptedTokenIds[1], 32002) << "Token 1: reduced 2 -> full 32002";
            EXPECT_EQ(acceptedTokenIds[2], 32000) << "Token 2: reduced 0 -> full 32000";
        },
        vocabMappingTable);
}
