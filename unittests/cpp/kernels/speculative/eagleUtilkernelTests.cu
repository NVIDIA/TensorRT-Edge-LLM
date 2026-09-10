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

#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "runtime/state/kvPageTable.h"
#include "testUtils.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Test 1: prepareEaglePrefillInputs
// Description: Given sequenceContextLengths as input, compute selectIndices=sequenceLength-1 for each batch
// Now supports per-batch variable sequence lengths
// ============================================================================
TEST(EagleKernels, PrepareEaglePrefillInputs)
{
    cudaStream_t stream = nullptr;

    struct TestCase
    {
        int32_t sequenceContextLength;
        int64_t expectedSelectIndex;
    };

    // Define test cases with clear input-output relationship
    std::vector<TestCase> testCases
        = {{128, 127}, {64, 63}, {96, 95}, {112, 111}, {80, 79}, {100, 99}, {120, 119}, {88, 87}};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        // Extract input and expected data for current batch size
        std::vector<int32_t> inputSequenceContextLengths(batchSize);
        std::vector<int64_t> expectedSelectIndices(batchSize);
        for (int32_t b = 0; b < batchSize; b++)
        {
            inputSequenceContextLengths[b] = testCases[b].sequenceContextLength;
            expectedSelectIndices[b] = testCases[b].expectedSelectIndex;
        }

        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);

        // Directly populate sequenceContextLengths as input
        copyHostToDevice<int32_t>(sequenceContextLengthsDevice, inputSequenceContextLengths);

        prepareEaglePrefillInputs(sequenceContextLengthsDevice, selectIndicesDevice, stream);

        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);

        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualSelectIndices[b], expectedSelectIndices[b]);
        }
    }
}

// ============================================================================
// Test 2: initializeDraftTreeTables
// Description: Initialize draft tree with root + level1 draft tokens translated to target ids
// Format: [root(score=0, parent=-1), level1_tokens(score=logProb, parent=0), empty(-inf, -5)]
// ============================================================================
TEST(EagleKernels, InitializeDraftTreeTables)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t tableLength = 21; // 1 + 4 + 16

    std::vector<int32_t> inputSelectedIndices = {
        10, 20, 30, 40, // Batch 0
        15, 25, 35, 45, // Batch 1
        12, 22, 32, 42, // Batch 2
        18, 28, 38, 48, // Batch 3
        11, 21, 31, 41, // Batch 4
        13, 23, 33, 43, // Batch 5
        14, 24, 34, 44, // Batch 6
        16, 26, 36, 46  // Batch 7
    };

    std::vector<float> inputLogProbs = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    std::vector<int32_t> inputRootTokens = {5000, 6000, 7000, 8000, 5500, 6500, 7500, 8500};

    // Vocab mapping: full_id = draft_id + vocabMapping[draft_id]
    std::vector<int32_t> inputVocabMapping(1000);
    for (int i = 0; i < 1000; i++)
    {
        inputVocabMapping[i] = i * 100;
    }

    // Expected: Batch 0 full data (others follow same pattern)
    // Root: 5000, Level1: 10+1000=1010, 20+2000=2020, 30+3000=3030, 40+4000=4040
    std::vector<int32_t> expectedIdsBatch0
        = {5000, 1010, 2020, 3030, 4040, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Note: the parent of the root node is -1, the parent of the level 1 nodes is 0, the parent of the empty nodes is
    // -5. It's pre-defined in the kernel with below constants. constexpr int32_t kROOT_NODE_PREDECESSOR{-1}; constexpr
    // int32_t kEMPTY_NODE_PREDECESSOR{-5};
    std::vector<int32_t> expectedParentsBatch0
        = {-1, 0, 0, 0, 0, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto logProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto rootTokensDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto vocabMappingDevice = rt::Tensor({1000}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto draftParentFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * draftTopK * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rootTokensDevice.rawPointer(), inputRootTokens.data(), batchSize * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(vocabMappingDevice, inputVocabMapping);

        initializeDraftTreeTables(selectedIndicesDevice, logProbsDevice, rootTokensDevice, vocabMappingDevice,
            draftIdFullTableDevice, draftScoreFullTableDevice, draftParentFullTableDevice, draftTopK, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdFullTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreFullTableDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(draftParentFullTableDevice);

        // Verify batch 0 in detail
        for (int32_t i = 0; i < tableLength; i++)
        {
            EXPECT_EQ(actualIds[i], expectedIdsBatch0[i]);
            EXPECT_EQ(actualParents[i], expectedParentsBatch0[i]);
            if (i == 0)
            {
                EXPECT_FLOAT_EQ(actualScores[i], 0.0f);
            }
            else if (i <= draftTopK)
            {
                EXPECT_FLOAT_EQ(actualScores[i], inputLogProbs[i - 1]);
            }
            else
            {
                EXPECT_TRUE(std::isinf(actualScores[i]) && actualScores[i] < 0);
            }
        }

        // Verify all batches have correct structure
        for (int32_t b = 0; b < batchSize; b++)
        {
            int offset = b * tableLength;
            EXPECT_EQ(actualIds[offset], inputRootTokens[b]);
            EXPECT_EQ(actualParents[offset], -1);
            for (int i = 1; i <= draftTopK; i++)
            {
                EXPECT_EQ(actualParents[offset + i], 0);
            }
        }
    }
}

TEST(EagleKernels, InitializeDraftTreeTablesWithIdentityMapping)
{
    cudaStream_t stream = nullptr;
    int32_t const batchSize = 1;
    int32_t const draftTopK = 3;
    int32_t const tableLength = 13;

    std::vector<int32_t> selectedIndices = {101, 107, 113};
    std::vector<float> logProbs = {-0.1f, -0.2f, -0.3f};
    std::vector<int32_t> rootTokens = {5000};
    std::vector<int32_t> vocabOffsets(128, 0);

    auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto logProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto rootTokensDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto vocabOffsetsDevice
        = rt::Tensor({static_cast<int64_t>(vocabOffsets.size())}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftIdFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftScoreFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftParentFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<int32_t>(selectedIndicesDevice, selectedIndices);
    copyHostToDevice<float>(logProbsDevice, logProbs);
    copyHostToDevice<int32_t>(rootTokensDevice, rootTokens);
    copyHostToDevice<int32_t>(vocabOffsetsDevice, vocabOffsets);

    initializeDraftTreeTables(selectedIndicesDevice, logProbsDevice, rootTokensDevice, vocabOffsetsDevice,
        draftIdFullTableDevice, draftScoreFullTableDevice, draftParentFullTableDevice, draftTopK, stream);

    auto const actualIds = copyDeviceToHost<int32_t>(draftIdFullTableDevice);
    EXPECT_EQ(actualIds[0], 5000);
    EXPECT_EQ(actualIds[1], 101);
    EXPECT_EQ(actualIds[2], 107);
    EXPECT_EQ(actualIds[3], 113);
}

// ============================================================================
// Test 3: assembleInitialDraftTreeInput
// Description: Assemble first round draft tree input from full table
// This test simulates a multi-step scenario (draftingStep=6) but only tests initial round
// The full table size follows: 1 + topK + (step-1) * topK²
// ============================================================================
TEST(EagleKernels, AssembleInitialDraftTreeInput)
{
    cudaStream_t stream = nullptr;

    // Simulate production config: topK=10, step=6 (scaled down for testing)
    // Use topK=4, step=4 to keep test data manageable
    // fullTableLength = 1 + topK + (step-1) * topK² = 1 + 4 + 3*16 = 53
    int32_t draftTopK = 4;
    int32_t draftingStep = 4; // Simulate multi-step scenario
    int32_t fullTableLength = 1 + draftTopK + (draftingStep - 1) * draftTopK * draftTopK;
    // fullTableLength = 1 + 4 + 3*16 = 53

    int32_t paddedDraftTreeSize = 12;
    int32_t draftHiddenDim = 256;

    // Input: Full table with complete structure (root + all levels)
    // Each batch has fullTableLength=53 elements
    // Structure: [root, level1(4), level2(16), level3(16), level4(16)]
    //
    // CRITICAL: Data must be unique at each global index to distinguish correct vs wrong offset
    // Use pattern: globalIndex * 10000 + batch * 100 + localPos
    std::vector<int32_t> inputDraftIdFullTable;

    for (int b = 0; b < 8; b++)
    {
        for (int localPos = 0; localPos < fullTableLength; localPos++)
        {
            int32_t globalIdx = b * fullTableLength + localPos;
            // Each element encodes its global position uniquely
            // This ensures different offsets give completely different values
            int32_t value = globalIdx * 100 + b * 10 + localPos;
            inputDraftIdFullTable.push_back(value);
        }
    }

    std::vector<half> inputHiddenStates(8 * draftHiddenDim);
    for (int b = 0; b < 8; b++)
    {
        for (int d = 0; d < draftHiddenDim; d++)
        {
            inputHiddenStates[b * draftHiddenDim + d] = __float2half(1.0f + b * 0.1f);
        }
    }

    // Expected outputs for multiple batches
    // Initial round reads positions [1:4] (level 1) from each batch's fullTable
    // Each value = globalIdx * 100 + batch * 10 + localPos
    //
    // Batch 0: reads fullTable[globalIdx 1-4]
    //   fullTable[1] = 1*100 + 0*10 + 1 = 101
    //   fullTable[2] = 2*100 + 0*10 + 2 = 202
    //   fullTable[3] = 3*100 + 0*10 + 3 = 303
    //   fullTable[4] = 4*100 + 0*10 + 4 = 404
    std::vector<int32_t> expectedInputIdsBatch0 = {101, 202, 303, 404, 0, 0, 0, 0, 0, 0, 0, 0};

    // Batch 1: reads fullTable[globalIdx 54-57] (offset = 1*53 + [1:4])
    //   fullTable[54] = 54*100 + 1*10 + 1 = 5411
    //   fullTable[55] = 55*100 + 1*10 + 2 = 5512
    //   fullTable[56] = 56*100 + 1*10 + 3 = 5613
    //   fullTable[57] = 57*100 + 1*10 + 4 = 5714
    std::vector<int32_t> expectedInputIdsBatch1 = {5411, 5512, 5613, 5714, 0, 0, 0, 0, 0, 0, 0, 0};

    // Batch 2: reads fullTable[globalIdx 107-110] (offset = 2*53 + [1:4])
    //   fullTable[107] = 107*100 + 2*10 + 1 = 10721
    //   fullTable[108] = 108*100 + 2*10 + 2 = 10822
    //   fullTable[109] = 109*100 + 2*10 + 3 = 10923
    //   fullTable[110] = 110*100 + 2*10 + 4 = 11024
    std::vector<int32_t> expectedInputIdsBatch2 = {10721, 10822, 10923, 11024, 0, 0, 0, 0, 0, 0, 0, 0};

    // If kernel uses WRONG formula (21 instead of 53):
    //   Batch 1 would read fullTable[22-25]:
    //     [22] = 22*100 + 0*10 + 22 = 2222 ✗ (not 5411)
    //     [23] = 23*100 + 0*10 + 23 = 2323 ✗ (not 5512)
    //   Test would FAIL! ✅

    int32_t expectedTreeLength = 4;

    std::cout << "Testing with fullTableLength=" << fullTableLength << " (simulates draftingStep=" << draftingStep
              << ")\n";
    std::cout << "Kernel's wrong formula would give: " << (1 + draftTopK + draftTopK * draftTopK) << "\n";

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesOutputDevice
            = rt::Tensor({batchSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto inputIdsDevice = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesInputDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

        CUDA_CHECK(cudaMemcpy(draftIdFullTableDevice.rawPointer(), inputDraftIdFullTable.data(),
            batchSize * fullTableLength * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftHiddenStatesOutputDevice.rawPointer(), inputHiddenStates.data(),
            batchSize * draftHiddenDim * sizeof(half), cudaMemcpyHostToDevice));

        assembleInitialDraftTreeInput(draftIdFullTableDevice, draftHiddenStatesOutputDevice, inputIdsDevice,
            draftHiddenStatesInputDevice, draftTreeLengthDevice, draftTreeMaskDevice, draftTopK, stream);

        auto const actualInputIds = copyDeviceToHost<int32_t>(inputIdsDevice);
        auto const actualTreeLength = copyDeviceToHost<int32_t>(draftTreeLengthDevice);
        auto const actualMask = copyDeviceToHost<int8_t>(draftTreeMaskDevice);

        // ========== Verify Batch 0 ==========
        for (int i = 0; i < paddedDraftTreeSize; i++)
        {
            EXPECT_EQ(actualInputIds[i], expectedInputIdsBatch0[i]) << "Batch 0, position " << i << " mismatch";
        }
        EXPECT_EQ(actualTreeLength[0], expectedTreeLength);

        // Verify Batch 0 mask: diagonal only for first topK tokens
        for (int i = 0; i < paddedDraftTreeSize; i++)
        {
            for (int j = 0; j < paddedDraftTreeSize; j++)
            {
                int8_t expected = (i < draftTopK && i == j) ? 1 : 0;
                EXPECT_EQ(actualMask[i * paddedDraftTreeSize + j], expected)
                    << "Batch 0 mask mismatch at [" << i << "][" << j << "]";
            }
        }

        // ========== CRITICAL: Verify Batch 1 (if batchSize >= 2) ==========
        // This tests that kernel uses fullTableLength parameter for offset calculation
        // With fullTableLength=53 (step=4), Batch 1 should read from globalIdx [54:57]
        // If kernel used wrong formula (1+topK+topK²=21), it would read from globalIdx [22:25]
        if (batchSize >= 2)
        {
            // Kernel line 218: tableOffset = batchIdx * fullTableLength + i + 1
            // For Batch 1, i=0:
            //   CORRECT: tableOffset = 1*53 + 0 + 1 = 54 → fullTable[54] = 5411 ✓
            //   WRONG:   tableOffset = 1*21 + 0 + 1 = 22 → fullTable[22] = 2222 ✗
            // These values are COMPLETELY DIFFERENT, so bug will be caught!

            for (int i = 0; i < draftTopK; i++) // Only check first topK positions
            {
                int32_t batchOffset = 1 * paddedDraftTreeSize;
                int32_t actual = actualInputIds[batchOffset + i];
                int32_t expected = expectedInputIdsBatch1[i];

                // Calculate what we would get if kernel used wrong formula
                int32_t wrongOffset = 1 * 21 + i + 1;                          // Wrong formula result
                int32_t wrongValue = wrongOffset * 100 + 0 * 10 + wrongOffset; // Batch 0's data

                EXPECT_EQ(actual, expected)
                    << "Batch 1, position " << i << " CRITICAL MISMATCH" << "\n  Expected: " << expected
                    << " (from correct offset " << (1 * fullTableLength + i + 1) << ")"
                    << "\n  If kernel used WRONG formula (21), would get: " << wrongValue << " (from wrong offset "
                    << wrongOffset << ")" << "\n  Actual: " << actual
                    << "\n  This verifies kernel uses fullTableLength=" << fullTableLength << ", NOT hardcoded formula "
                    << (1 + draftTopK + draftTopK * draftTopK);
            }
            EXPECT_EQ(actualTreeLength[1], expectedTreeLength);

            // Verify Batch 1 mask
            for (int i = 0; i < paddedDraftTreeSize; i++)
            {
                for (int j = 0; j < paddedDraftTreeSize; j++)
                {
                    int8_t expected = (i < draftTopK && i == j) ? 1 : 0;
                    int32_t maskOffset = 1 * paddedDraftTreeSize * paddedDraftTreeSize + i * paddedDraftTreeSize + j;
                    EXPECT_EQ(actualMask[maskOffset], expected)
                        << "Batch 1 mask mismatch at [" << i << "][" << j << "]";
                }
            }
        }

        // ========== Verify Batch 2 (if batchSize >= 4) ==========
        if (batchSize >= 4)
        {
            for (int i = 0; i < paddedDraftTreeSize; i++)
            {
                int32_t batchOffset = 2 * paddedDraftTreeSize;
                EXPECT_EQ(actualInputIds[batchOffset + i], expectedInputIdsBatch2[i])
                    << "Batch 2, position " << i << " mismatch";
            }
            EXPECT_EQ(actualTreeLength[2], expectedTreeLength);
        }

        // Verify tree lengths for all batches
        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualTreeLength[b], expectedTreeLength) << "Batch " << b << " tree length mismatch";
        }

        // Print success message for multi-batch tests
        if (batchSize >= 2)
        {
            std::cout << "✓ Multi-batch test passed (batchSize=" << batchSize
                      << "): fullTableLength=" << fullTableLength << " (step=" << draftingStep
                      << ") correctly used, not wrong formula " << (1 + draftTopK + draftTopK * draftTopK) << "\n";
        }
    }
}

// ============================================================================
// Test 4: assembleInitialIntermediateData
// Description: Copy logProbs to scores; set parents to [1,2,3,4] (first level indices)
// ============================================================================
TEST(EagleKernels, AssembleInitialIntermediateData)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;

    std::vector<float> inputLogProbs = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    // Expected: parents = [1,2,3,4], scores = logProbs
    std::vector<int32_t> expectedParents = {1, 2, 3, 4};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto logProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);

        CUDA_CHECK(cudaMemcpy(logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * draftTopK * sizeof(float),
            cudaMemcpyHostToDevice));

        assembleInitialIntermediateData(
            logProbsDevice, intermediateParentsDevice, intermediateScoresDevice, draftTopK, stream);

        auto const actualParents = copyDeviceToHost<int32_t>(intermediateParentsDevice);
        auto const actualScores = copyDeviceToHost<float>(intermediateScoresDevice);

        // Verify batch 0
        for (int i = 0; i < draftTopK; i++)
        {
            EXPECT_EQ(actualParents[i], expectedParents[i]);
            EXPECT_FLOAT_EQ(actualScores[i], inputLogProbs[i]);
        }

        // Verify all batches have correct parents
        for (int32_t b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < draftTopK; i++)
            {
                EXPECT_EQ(actualParents[b * draftTopK + i], expectedParents[i]);
            }
        }
    }
}

// ============================================================================
// Test 5: computeCuScoresAndTranslateToken
// Description: Translate draft tokens to full vocab; compute cumulative scores
// score[i] = intermediateScores[i/topK] + logProbs[i]
// ============================================================================
TEST(EagleKernels, ComputeCuScoresAndTranslateToken)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;

    // Input: 16 candidate tokens per batch
    std::vector<int32_t> inputSelectedIndices(8 * 16);
    std::vector<float> inputLogProbs(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputSelectedIndices[b * 16 + i] = 10 + b * 10 + i;
            inputLogProbs[b * 16 + i] = -0.3f - i * 0.1f;
        }
    }

    std::vector<float> inputIntermediateScores = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    std::vector<int32_t> inputVocabMapping(1000);
    for (int i = 0; i < 1000; i++)
    {
        inputVocabMapping[i] = i * 100;
    }

    // Expected batch 0: draft_id=10 -> full_id=10+1000=1010
    // score[0] = intermediateScores[0] + logProbs[0] = -0.5 + (-0.3) = -0.8
    int32_t expectedId0 = 10 + 1000;        // 1010
    float expectedScore0 = -0.5f + (-0.3f); // -0.8

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto selectedIndicesDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto logProbsDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto vocabMappingDevice = rt::Tensor({1000}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);

        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(intermediateScoresDevice.rawPointer(), inputIntermediateScores.data(),
            batchSize * draftTopK * sizeof(float), cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(vocabMappingDevice, inputVocabMapping);

        computeCuScoresAndTranslateToken(selectedIndicesDevice, logProbsDevice, intermediateScoresDevice,
            vocabMappingDevice, draftIdTableDevice, draftScoreTableDevice, draftTopK, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreTableDevice);

        // Verify batch 0
        EXPECT_EQ(actualIds[0], expectedId0);
        EXPECT_FLOAT_EQ(actualScores[0], expectedScore0);

        // Verify score computation for all batches (first 4 entries use parent 0)
        for (int32_t b = 0; b < batchSize; b++)
        {
            float parentScore = inputIntermediateScores[b * draftTopK];
            for (int i = 0; i < 4; i++)
            {
                float expected = parentScore + inputLogProbs[b * 16 + i];
                EXPECT_FLOAT_EQ(actualScores[b * 16 + i], expected);
            }
        }
    }
}

// ============================================================================
// Test 6: prepareEagleAcceptDecodeTokenInputs
// Description: Create causal mask and position indices for accepted tokens
// mask[i] = 0b00..011..1 (i+1 ones), positions = startIndex + [0,1,2,..]
// Now supports per-batch variable accepted token numbers
// ============================================================================
TEST(EagleKernels, PrepareEagleAcceptDecodeTokenInputs)
{
    cudaStream_t stream = nullptr;
    int32_t maxAcceptedTokenNum = 8; // Max across all batches

    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};
    std::vector<int32_t> inputAcceptedTokenNums = {5, 3, 6, 4, 5, 7, 4, 8};

    // Expected for batch 0: 5 tokens
    std::vector<int32_t> expectedMaskBatch0 = {1, 3, 7, 15, 31};
    std::vector<int32_t> expectedPositionsBatch0 = {100, 101, 102, 103, 104};
    int64_t expectedSelectIndexBatch0 = 4; // Last token
    // Context length now uses maxAcceptedTokenNum (padded length) instead of actual acceptedTokenNum
    // This is required for correct XQA attention context K range calculation with padding
    int32_t expectedContextLenBatch0 = 100 + maxAcceptedTokenNum; // 108

    // Expected for batch 1: 3 tokens
    int64_t expectedSelectIndexBatch1 = 2;
    int32_t expectedContextLenBatch1 = 200 + maxAcceptedTokenNum; // 208

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto acceptedTokenNumsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedMaskDevice = rt::Tensor({batchSize, maxAcceptedTokenNum, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        auto positionsDevice = rt::Tensor({batchSize, maxAcceptedTokenNum}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto contextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(acceptedTokenNumsDevice.rawPointer(), inputAcceptedTokenNums.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleAcceptDecodeTokenInputs(sequenceStartIndicesDevice, acceptedTokenNumsDevice, packedMaskDevice,
            positionsDevice, selectIndicesDevice, contextLengthsDevice, stream);

        auto const actualMask = copyDeviceToHost<int32_t>(packedMaskDevice);
        auto const actualPositions = copyDeviceToHost<int32_t>(positionsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(contextLengthsDevice);

        // Verify batch 0
        for (int i = 0; i < inputAcceptedTokenNums[0]; i++)
        {
            EXPECT_EQ(actualMask[i], expectedMaskBatch0[i]);
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]);
        }
        EXPECT_EQ(actualSelectIndices[0], expectedSelectIndexBatch0);
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);

        // Verify batch 1 if available
        if (batchSize >= 2)
        {
            EXPECT_EQ(actualSelectIndices[1], expectedSelectIndexBatch1);
            EXPECT_EQ(actualContextLengths[1], expectedContextLenBatch1);
        }

        // Verify all batches have correct select indices and context lengths
        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualSelectIndices[b], inputAcceptedTokenNums[b] - 1);
            // Context length uses maxAcceptedTokenNum (padded length) for correct XQA attention range
            EXPECT_EQ(actualContextLengths[b], inputSequenceStartIndices[b] + maxAcceptedTokenNum);
        }
    }
}

// ============================================================================
// Test 7: constructVerificationDraftTree
// Description: Build verification tree from selected nodes in full table
// Track parent chain to build attention mask
// Large-scale test with deeper tree (4 layers) and batch size 2
// ============================================================================
TEST(EagleKernels, ConstructVerificationDraftTree)
{
    cudaStream_t stream = nullptr;
    int32_t const batchSize = 2;
    int32_t const fullTableLength = 50; // Larger table: 1 root + 4 L1 + 16 L2 + 29 L3
    int32_t const verifyTreeSize = 12;  // Larger verification tree

    // ========== Batch 0: Construct a 4-layer tree with specific structure ==========
    // Tree structure for Batch 0:
    // Layer 0 (root):     0
    // Layer 1 (4 nodes):  1, 2, 3, 4 (all children of 0)
    // Layer 2 (16 nodes): 5-20 (4 children per L1 node)
    // Layer 3 (29 nodes): 21-49 (varies, testing asymmetric tree)
    std::vector<int32_t> inputDraftIdFullTableBatch0(fullTableLength);
    std::vector<int32_t> inputDraftParentFullTableBatch0(fullTableLength);

    // Root node
    inputDraftIdFullTableBatch0[0] = 1000;
    inputDraftParentFullTableBatch0[0] = -1;

    // Layer 1: nodes 1-4, all children of root (0)
    for (int i = 1; i <= 4; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        inputDraftParentFullTableBatch0[i] = 0;
    }

    // Layer 2: nodes 5-20, 4 children per L1 node
    for (int i = 5; i <= 20; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        inputDraftParentFullTableBatch0[i] = 1 + (i - 5) / 4; // Parent in range [1,4]
    }

    // Layer 3: nodes 21-49, children of L2 nodes (asymmetric distribution)
    for (int i = 21; i < fullTableLength; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        // Distribute among L2 nodes: more children to early L2 nodes
        inputDraftParentFullTableBatch0[i] = 5 + (i - 21) % 16;
    }

    // Select a path through the tree for verification (Batch 0):
    // Path: 0 -> 2 -> 7 -> 11 -> 27, and additional branches
    // Selected: [0(root), 2(L1), 6(L2), 7(L2), 11(L2), 14(L2), 27(L3), 28(L3), 32(L3), 35(L3), 40(L3), 45(L3)]
    std::vector<int32_t> inputSelectedIndicesBatch0 = {0, 2, 6, 7, 11, 14, 27, 28, 32, 35, 40, 45};

    // Expected token IDs for Batch 0
    std::vector<int32_t> expectedIdsBatch0(verifyTreeSize);
    for (int i = 0; i < verifyTreeSize; i++)
    {
        expectedIdsBatch0[i] = inputDraftIdFullTableBatch0[inputSelectedIndicesBatch0[i]];
    }

    // Expected attention mask for Batch 0
    // Kernel logic: For each token, sequentially match its parent chain against prior tokens in verify tree
    // If direct parent is not found, all ancestors become unreachable
    //
    // Parent chains (computed):
    // node 0: [] (root)
    // node 2: [0]
    // node 6: [1, 0] - node 1 NOT in verify tree → only self
    // node 7: [1, 0] - node 1 NOT in verify tree → only self
    // node 11: [2, 0] - node 2 IS in verify tree (Token 1) → can reach root
    // node 14: [3, 0] - node 3 NOT in verify tree → only self
    // node 27: [11, 2, 0] - all in verify tree → full chain
    // node 28: [12, 2, 0] - node 12 NOT in verify tree → only self
    // node 32: [16, 3, 0] - node 16 NOT in verify tree → only self
    // node 35: [19, 4, 0] - node 19 NOT in verify tree → only self
    // node 40: [8, 1, 0] - node 8 NOT in verify tree → only self
    // node 45: [13, 3, 0] - node 13 NOT in verify tree → only self
    std::vector<std::vector<int8_t>> expectedMaskBatch0(verifyTreeSize, std::vector<int8_t>(verifyTreeSize, 0));
    // Token 0 (node 0): attends to [0]
    expectedMaskBatch0[0][0] = 1;
    // Token 1 (node 2): attends to [0, 1]
    expectedMaskBatch0[1][0] = 1;
    expectedMaskBatch0[1][1] = 1;
    // Token 2 (node 6): attends to [2]
    expectedMaskBatch0[2][2] = 1;
    // Token 3 (node 7): attends to [3]
    expectedMaskBatch0[3][3] = 1;
    // Token 4 (node 11): attends to [0, 1, 4]
    expectedMaskBatch0[4][0] = 1;
    expectedMaskBatch0[4][1] = 1;
    expectedMaskBatch0[4][4] = 1;
    // Token 5 (node 14): attends to [5]
    expectedMaskBatch0[5][5] = 1;
    // Token 6 (node 27): attends to [0, 1, 4, 6]
    expectedMaskBatch0[6][0] = 1;
    expectedMaskBatch0[6][1] = 1;
    expectedMaskBatch0[6][4] = 1;
    expectedMaskBatch0[6][6] = 1;
    // Token 7 (node 28): attends to [7]
    expectedMaskBatch0[7][7] = 1;
    // Token 8 (node 32): attends to [8]
    expectedMaskBatch0[8][8] = 1;
    // Token 9 (node 35): attends to [9]
    expectedMaskBatch0[9][9] = 1;
    // Token 10 (node 40): attends to [10]
    expectedMaskBatch0[10][10] = 1;
    // Token 11 (node 45): attends to [11]
    expectedMaskBatch0[11][11] = 1;

    // ========== Batch 1: Different tree structure with deeper paths ==========
    std::vector<int32_t> inputDraftIdFullTableBatch1(fullTableLength);
    std::vector<int32_t> inputDraftParentFullTableBatch1(fullTableLength);

    // Root node
    inputDraftIdFullTableBatch1[0] = 2000;
    inputDraftParentFullTableBatch1[0] = -1;

    // Layer 1: nodes 1-4
    for (int i = 1; i <= 4; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        inputDraftParentFullTableBatch1[i] = 0;
    }

    // Layer 2: nodes 5-20 (focus more children on node 3)
    for (int i = 5; i <= 20; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        // Bias towards node 3: nodes 5-12 are children of node 3
        if (i <= 12)
        {
            inputDraftParentFullTableBatch1[i] = 3;
        }
        else
        {
            inputDraftParentFullTableBatch1[i] = 1 + (i - 13) % 3;
        }
    }

    // Layer 3: nodes 21-49, children of L2 nodes (creating longer chains)
    for (int i = 21; i < fullTableLength; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        // Create deeper paths through nodes 5-12
        inputDraftParentFullTableBatch1[i] = 5 + (i - 21) % 8;
    }

    // Select a deep path for verification (Batch 1):
    // Path focusing on deep chains: 0 -> 3 -> 5 -> 6 -> 8 -> 21 -> 25 -> 29 -> 33 -> 37 -> 41 -> 48
    std::vector<int32_t> inputSelectedIndicesBatch1 = {0, 3, 5, 6, 8, 21, 25, 29, 33, 37, 41, 48};

    // Expected token IDs for Batch 1
    std::vector<int32_t> expectedIdsBatch1(verifyTreeSize);
    for (int i = 0; i < verifyTreeSize; i++)
    {
        expectedIdsBatch1[i] = inputDraftIdFullTableBatch1[inputSelectedIndicesBatch1[i]];
    }

    // Expected attention mask for Batch 1
    // Parent chains (computed):
    // node 0: [] (root)
    // node 3: [0]
    // node 5: [3, 0] - full chain in verify tree
    // node 6: [3, 0] - full chain in verify tree
    // node 8: [3, 0] - full chain in verify tree
    // node 21: [5, 3, 0] - full chain in verify tree
    // node 25: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 29: [5, 3, 0] - full chain in verify tree
    // node 33: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 37: [5, 3, 0] - full chain in verify tree
    // node 41: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 48: [8, 3, 0] - full chain in verify tree (node 8 is Token 4)
    std::vector<std::vector<int8_t>> expectedMaskBatch1(verifyTreeSize, std::vector<int8_t>(verifyTreeSize, 0));
    // Token 0 (node 0): attends to [0]
    expectedMaskBatch1[0][0] = 1;
    // Token 1 (node 3): attends to [0, 1]
    expectedMaskBatch1[1][0] = 1;
    expectedMaskBatch1[1][1] = 1;
    // Token 2 (node 5): attends to [0, 1, 2]
    expectedMaskBatch1[2][0] = 1;
    expectedMaskBatch1[2][1] = 1;
    expectedMaskBatch1[2][2] = 1;
    // Token 3 (node 6): attends to [0, 1, 3]
    expectedMaskBatch1[3][0] = 1;
    expectedMaskBatch1[3][1] = 1;
    expectedMaskBatch1[3][3] = 1;
    // Token 4 (node 8): attends to [0, 1, 4]
    expectedMaskBatch1[4][0] = 1;
    expectedMaskBatch1[4][1] = 1;
    expectedMaskBatch1[4][4] = 1;
    // Token 5 (node 21): attends to [0, 1, 2, 5]
    expectedMaskBatch1[5][0] = 1;
    expectedMaskBatch1[5][1] = 1;
    expectedMaskBatch1[5][2] = 1;
    expectedMaskBatch1[5][5] = 1;
    // Token 6 (node 25): attends to [6]
    expectedMaskBatch1[6][6] = 1;
    // Token 7 (node 29): attends to [0, 1, 2, 7]
    expectedMaskBatch1[7][0] = 1;
    expectedMaskBatch1[7][1] = 1;
    expectedMaskBatch1[7][2] = 1;
    expectedMaskBatch1[7][7] = 1;
    // Token 8 (node 33): attends to [8]
    expectedMaskBatch1[8][8] = 1;
    // Token 9 (node 37): attends to [0, 1, 2, 9]
    expectedMaskBatch1[9][0] = 1;
    expectedMaskBatch1[9][1] = 1;
    expectedMaskBatch1[9][2] = 1;
    expectedMaskBatch1[9][9] = 1;
    // Token 10 (node 41): attends to [10]
    expectedMaskBatch1[10][10] = 1;
    // Token 11 (node 48): attends to [0, 1, 4, 11]
    expectedMaskBatch1[11][0] = 1;
    expectedMaskBatch1[11][1] = 1;
    expectedMaskBatch1[11][4] = 1;
    expectedMaskBatch1[11][11] = 1;

    // ========== Prepare combined input for both batches ==========
    std::vector<int32_t> inputDraftIdFullTable(batchSize * fullTableLength);
    std::vector<int32_t> inputDraftParentFullTable(batchSize * fullTableLength);
    std::vector<int32_t> inputSelectedIndices(batchSize * verifyTreeSize);

    // Copy batch 0 data
    std::copy(inputDraftIdFullTableBatch0.begin(), inputDraftIdFullTableBatch0.end(), inputDraftIdFullTable.begin());
    std::copy(inputDraftParentFullTableBatch0.begin(), inputDraftParentFullTableBatch0.end(),
        inputDraftParentFullTable.begin());
    std::copy(inputSelectedIndicesBatch0.begin(), inputSelectedIndicesBatch0.end(), inputSelectedIndices.begin());

    // Copy batch 1 data
    std::copy(inputDraftIdFullTableBatch1.begin(), inputDraftIdFullTableBatch1.end(),
        inputDraftIdFullTable.begin() + fullTableLength);
    std::copy(inputDraftParentFullTableBatch1.begin(), inputDraftParentFullTableBatch1.end(),
        inputDraftParentFullTable.begin() + fullTableLength);
    std::copy(inputSelectedIndicesBatch1.begin(), inputSelectedIndicesBatch1.end(),
        inputSelectedIndices.begin() + verifyTreeSize);

    // ========== Execute kernel ==========
    auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftParentFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectedIndicesDevice = rt::Tensor({batchSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto inputIdsDevice = rt::Tensor({batchSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTreeMaskDevice
        = rt::Tensor({batchSize, verifyTreeSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

    copyHostToDevice<int32_t>(draftIdFullTableDevice, inputDraftIdFullTable);
    copyHostToDevice<int32_t>(draftParentFullTableDevice, inputDraftParentFullTable);
    copyHostToDevice<int32_t>(selectedIndicesDevice, inputSelectedIndices);

    constructVerificationDraftTree(draftIdFullTableDevice, draftParentFullTableDevice, selectedIndicesDevice,
        inputIdsDevice, draftTreeMaskDevice, stream);

    auto const actualIds = copyDeviceToHost<int32_t>(inputIdsDevice);
    auto const actualMask = copyDeviceToHost<int8_t>(draftTreeMaskDevice);

    // ========== Comprehensive verification for Batch 0 ==========
    // Verify token IDs for batch 0
    for (int i = 0; i < verifyTreeSize; i++)
    {
        EXPECT_EQ(actualIds[i], expectedIdsBatch0[i]) << "Batch 0 token ID mismatch at position " << i;
    }

    // Verify attention mask for batch 0
    for (int i = 0; i < verifyTreeSize; i++)
    {
        for (int j = 0; j < verifyTreeSize; j++)
        {
            int maskIdx = i * verifyTreeSize + j;
            EXPECT_EQ(actualMask[maskIdx], expectedMaskBatch0[i][j])
                << "Batch 0 mask mismatch at position [" << i << "][" << j << "]";
        }
    }

    // ========== Comprehensive verification for Batch 1 ==========
    // Verify token IDs for batch 1
    for (int i = 0; i < verifyTreeSize; i++)
    {
        int actualIdx = verifyTreeSize + i;
        EXPECT_EQ(actualIds[actualIdx], expectedIdsBatch1[i]) << "Batch 1 token ID mismatch at position " << i;
    }

    // Verify attention mask for batch 1
    for (int i = 0; i < verifyTreeSize; i++)
    {
        for (int j = 0; j < verifyTreeSize; j++)
        {
            int maskIdx = verifyTreeSize * verifyTreeSize + i * verifyTreeSize + j;
            EXPECT_EQ(actualMask[maskIdx], expectedMaskBatch1[i][j])
                << "Batch 1 mask mismatch at position [" << i << "][" << j << "]";
        }
    }
}

// Helper: upload a host vector of KVLayerInfo to a device-side blob (typed as INT8 since
// the kernel only ever reads it as `KVLayerInfo const*`). Mirrors how HybridCacheManager
// stores its per-group device arrays.
static rt::Tensor uploadLayerInfos(std::vector<KVLayerInfo> const& hostInfos, cudaStream_t stream)
{
    auto const infoBytes = static_cast<int64_t>(hostInfos.size() * sizeof(KVLayerInfo));
    rt::Tensor deviceInfos({infoBytes}, rt::DeviceType::kGPU, DataType::kINT8, "TestKVLayerInfos");
    CUDA_CHECK(cudaMemcpyAsync(
        deviceInfos.rawPointer(), hostInfos.data(), static_cast<size_t>(infoBytes), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return deviceInfos;
}

// ============================================================================
// Test 8: eagleBaseAssembleHiddenState (split out from the old combined kernel)
// Description: Test inplace compaction of accepted tokens from stride=draftTreeSize to stride=maxDepth
// Key test: Verify multi-batch scenario where Batch 1+ needs to move ALL tokens including position 0
// ============================================================================
TEST(EagleKernels, EagleBaseAssembleHiddenState)
{
    // Test with multiple headDim values (64, 128 for EAGLE; 256 for Qwen3.5 MTP)
    std::vector<int32_t> headDims = {128, 256};

    for (int32_t const headDim : headDims)
    {
        SCOPED_TRACE("headDim=" + std::to_string(headDim));

        cudaStream_t stream = nullptr;
        int32_t maxDepth = 6;       // After compaction, stride will be maxDepth
        int32_t draftTreeSize = 10; // Input stride is draftTreeSize
        int32_t baseHiddenDim = 512;

        // Test with 2 batches to verify compaction with stride change
        int32_t batchSize = 2;

        // Batch 0: accept positions [0, 3, 7] (length=3)
        // Batch 1: accept positions [0, 2, 5] (length=3)
        std::vector<int32_t> inputAcceptedIndices = {
            0, 3, 7, -1, -1, -1, // Batch 0
            0, 2, 5, -1, -1, -1  // Batch 1
        };
        std::vector<int32_t> inputAcceptLengths = {3, 3};

        // Setup input hidden states with unique markers for each batch and position
        // Use simple pattern: batch_id * 1000 + position_id * 100
        std::vector<half> inputHiddenState(batchSize * draftTreeSize * baseHiddenDim, __float2half(0.0f));

        for (int d = 0; d < baseHiddenDim; d++)
        {
            inputHiddenState[0 * draftTreeSize * baseHiddenDim + 0 * baseHiddenDim + d] = __float2half(0.0f);
            inputHiddenState[0 * draftTreeSize * baseHiddenDim + 3 * baseHiddenDim + d] = __float2half(300.0f);
            inputHiddenState[0 * draftTreeSize * baseHiddenDim + 7 * baseHiddenDim + d] = __float2half(700.0f);
        }

        for (int d = 0; d < baseHiddenDim; d++)
        {
            inputHiddenState[1 * draftTreeSize * baseHiddenDim + 0 * baseHiddenDim + d] = __float2half(1000.0f);
            inputHiddenState[1 * draftTreeSize * baseHiddenDim + 2 * baseHiddenDim + d] = __float2half(1200.0f);
            inputHiddenState[1 * draftTreeSize * baseHiddenDim + 5 * baseHiddenDim + d] = __float2half(1500.0f);
        }

        // Create device tensors
        auto acceptedIndicesDevice = rt::Tensor({batchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        auto acceptLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto hiddenStateDevice
            = rt::Tensor({batchSize, draftTreeSize, baseHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);

        // Copy data to device
        copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
        copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
        copyHostToDevice<half>(hiddenStateDevice, inputHiddenState);

        // Execute hidden state assembly.
        eagleBaseAssembleHiddenState(acceptedIndicesDevice, acceptLengthsDevice, hiddenStateDevice, stream);

        // Read back results
        auto actualHiddenState = copyDeviceToHost<half>(hiddenStateDevice);

        // Verify Batch 0: compacted to positions [0, 1, 2]
        std::vector<float> expectedBatch0 = {0.0f, 300.0f, 700.0f};
        for (int i = 0; i < 3; i++)
        {
            float actual = __half2float(actualHiddenState[0 * draftTreeSize * baseHiddenDim + i * baseHiddenDim]);
            EXPECT_TRUE(isclose(actual, expectedBatch0[i], 1e-3f, 1e-3f))
                << "Batch 0, position " << i << ": expected " << expectedBatch0[i] << ", got " << actual;
        }

        // Verify Batch 1: compacted from offset draftTreeSize*dim to maxDepth*dim
        std::vector<float> expectedBatch1 = {1000.0f, 1200.0f, 1500.0f};
        int32_t batch1OutputOffset = maxDepth * baseHiddenDim;

        for (int i = 0; i < 3; i++)
        {
            float actual = __half2float(actualHiddenState[batch1OutputOffset + i * baseHiddenDim]);
            EXPECT_TRUE(isclose(actual, expectedBatch1[i], 1e-3f, 1e-3f))
                << "Batch 1, position " << i << ": expected " << expectedBatch1[i] << ", got " << actual;
        }

        std::cout << "✓ headDim=" << headDim << " passed\n";
    }
}

// ============================================================================
// Test 8b: eagleBaseCommitKVCache with heterogeneous numKVHeads
// Description: Verify the per-layer batched KV-cache commit kernel both (a) writes the
//              correct accepted-position data into each layer's cache, and (b) tolerates
//              layers in the same group having different numKVHeads (via early-exit on
//              extra grid columns). Single head-dim group covers both layers in one launch.
// ============================================================================
TEST(EagleKernels, EagleBaseCommitKVCacheHeterogeneousLayers)
{
    cudaStream_t stream = nullptr;

    for (int32_t const headDim : {128, 512})
    {
        // Two attention layers in the same head-dim group, with different numKVHeads.
        int32_t const maxBatchSize = 1;
        int32_t const maxSeqLen = rt::kTOKENS_PER_PAGE;
        int32_t const numLayers = 2;
        std::vector<int32_t> const numKVHeadsPerLayer = {4, 2}; // heterogeneous; group maxKVHeads = 4
        int32_t const maxKVHeads = 4;
        int32_t const activeBatchSize = 1;
        int32_t const maxDepth = 4;
        int32_t const maxPagesPerSeq = maxSeqLen / rt::kTOKENS_PER_PAGE;
        int32_t const numPages = maxBatchSize * maxPagesPerSeq;

        // Single batch with pastKvCacheLength=4 and accepted indices [0, 3, 5] (length=3).
        // Kernel writes to positions {pastKv+i for i in 1..acceptLen-1} reading from
        // {pastKv+acceptedIdx[i]}; here that is pos 5 <- pos 7 and pos 6 <- pos 9.
        std::vector<int32_t> inputAcceptedIndices = {0, 3, 5, -1};
        std::vector<int32_t> inputAcceptLengths = {3};
        std::vector<int32_t> inputKvCacheLengths = {4};

        auto kvAt = [&](int32_t layer, int32_t kv, int32_t head, int32_t pos) {
            return static_cast<float>(layer * 512 + kv * 256 + head * 64 + pos);
        };

        std::vector<rt::Tensor> kvCachePerLayer;
        kvCachePerLayer.reserve(numLayers);
        std::vector<KVLayerInfo> hostInfos(numLayers);
        for (int32_t L = 0; L < numLayers; ++L)
        {
            int32_t const numKVHeads = numKVHeadsPerLayer[L];
            rt::Tensor layerCache(
                {2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
            std::vector<half> hostInit(static_cast<size_t>(2) * numPages * rt::kTOKENS_PER_PAGE * numKVHeads * headDim);
            for (int32_t kv = 0; kv < 2; ++kv)
            {
                for (int32_t h = 0; h < numKVHeads; ++h)
                {
                    for (int32_t pos = 0; pos < maxSeqLen; ++pos)
                    {
                        half const v = __float2half(kvAt(L, kv, h, pos));
                        int64_t const base = (((static_cast<int64_t>(kv) * numPages + pos / rt::kTOKENS_PER_PAGE)
                                                      * rt::kTOKENS_PER_PAGE
                                                  + pos % rt::kTOKENS_PER_PAGE)
                                                     * numKVHeads
                                                 + h)
                            * headDim;
                        for (int32_t d = 0; d < headDim; ++d)
                        {
                            hostInit[base + d] = v;
                        }
                    }
                }
            }
            copyHostToDevice<half>(layerCache, hostInit);
            hostInfos[L] = KVLayerInfo{layerCache.rawPointer(), numKVHeads, maxSeqLen};
            kvCachePerLayer.push_back(std::move(layerCache));
        }
        rt::Tensor deviceInfos = uploadLayerInfos(hostInfos, stream);

        auto acceptedIndicesDevice = rt::Tensor({activeBatchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
        auto acceptLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto kvCacheLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
        copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
        copyHostToDevice<int32_t>(kvCacheLengthsDevice, inputKvCacheLengths);

        rt::KVPageTable pageTable(maxBatchSize, maxPagesPerSeq, maxBatchSize * maxPagesPerSeq);
        pageTable.setIdentity();
        pageTable.upload(stream);

        eagleBaseCommitKVCache(acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice,
            static_cast<KVLayerInfo const*>(deviceInfos.rawPointer()), numLayers, headDim, maxKVHeads, activeBatchSize,
            maxDepth, DataType::kHALF, stream, pageTable.kernelView().dataPointer<int32_t>(), numPages, maxPagesPerSeq);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        int32_t const pastKv = inputKvCacheLengths[0];
        std::vector<int32_t> const acceptedSrc = {0, 3, 5};
        for (int32_t L = 0; L < numLayers; ++L)
        {
            int32_t const numKVHeads = numKVHeadsPerLayer[L];
            auto host = copyDeviceToHost<half>(kvCachePerLayer[L]);
            for (int32_t kv = 0; kv < 2; ++kv)
            {
                for (int32_t h = 0; h < numKVHeads; ++h)
                {
                    int64_t const headBase
                        = static_cast<int64_t>(kv) * numPages * rt::kTOKENS_PER_PAGE * numKVHeads * headDim
                        + static_cast<int64_t>(h) * headDim;
                    int64_t const tokenStride = static_cast<int64_t>(numKVHeads) * headDim;

                    {
                        float const expected = kvAt(L, kv, h, pastKv);
                        float const actual = __half2float(host[headBase + pastKv * tokenStride]);
                        EXPECT_TRUE(isclose(actual, expected, 1e-3f, 1e-3f))
                            << "headDim=" << headDim << " L=" << L << " kv=" << kv << " h=" << h << " root: expected "
                            << expected << " got " << actual;
                    }

                    for (int32_t i = 1; i < static_cast<int32_t>(acceptedSrc.size()); ++i)
                    {
                        int32_t const dstPos = pastKv + i;
                        int32_t const srcPos = pastKv + acceptedSrc[i];
                        float const expected = kvAt(L, kv, h, srcPos);
                        float const actual = __half2float(host[headBase + dstPos * tokenStride]);
                        EXPECT_TRUE(isclose(actual, expected, 1e-3f, 1e-3f))
                            << "headDim=" << headDim << " L=" << L << " kv=" << kv << " h=" << h << " dstPos=" << dstPos
                            << " srcPos=" << srcPos << ": expected " << expected << " got " << actual;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Test 8e: eagleBaseCommitKVCache with a non-identity (scrambled) page table
// Description: a real permutation must route each slot's accepted-KV writes/reads through the
// physical pages named by the table. This swaps the two slots' physical K/V pages and straddles
// a page boundary on slot 0.
// ============================================================================
TEST(EagleKernels, EagleBaseCommitKVCacheNonIdentityPageTableRemapsPhysicalPages)
{
    cudaStream_t stream = nullptr;

    int32_t const headDim = 64;
    int32_t const numKVHeads = 1;
    int32_t const maxBatchSize = 2;
    int32_t const maxSeqLen = 256; // capPadded; 2 pages of 128 tokens per slot.
    int32_t const numLayers = 1;
    int32_t const maxKVHeads = numKVHeads;
    int32_t const activeBatchSize = 2;
    int32_t const maxDepth = 4;
    int32_t const maxPagesPerSeq = maxSeqLen / rt::kTOKENS_PER_PAGE; // 2
    int32_t const numPages = maxBatchSize * maxPagesPerSeq;          // 4 physical K pages (0..3)

    // Slot 0: pastKv=126, accept [0, 5, 9] -> logical writes at pos 127 (page 0) and 128 (page 1),
    // straddling slot 0's own page boundary.
    // Slot 1: pastKv=10, accept [0, 3, 5] -> logical writes at pos 11, 12 (both page 0).
    std::vector<int32_t> const inputAcceptedIndices = {0, 5, 9, -1, 0, 3, 5, -1};
    std::vector<int32_t> const inputAcceptLengths = {3, 3};
    std::vector<int32_t> const inputKvCacheLengths = {126, 10};

    // Value encodes the RAW (physical, pre-remap) buffer coordinates it was seeded at.
    auto kvAt = [&](int32_t kv, int32_t b, int32_t h, int32_t pos) {
        return static_cast<float>(kv * 10000 + b * 1000 + h * 100 + pos % 97);
    };

    rt::Tensor layerCache(
        {2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> hostInit(static_cast<size_t>(2) * numPages * rt::kTOKENS_PER_PAGE * numKVHeads * headDim);
    auto rawIndex = [&](int32_t kv, int32_t b, int32_t h, int32_t pos) {
        int32_t const physicalPage = b * maxPagesPerSeq + pos / rt::kTOKENS_PER_PAGE;
        int32_t const inPage = pos % rt::kTOKENS_PER_PAGE;
        return (((static_cast<int64_t>(kv) * numPages + physicalPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHeads + h)
            * headDim;
    };
    for (int32_t kv = 0; kv < 2; ++kv)
    {
        for (int32_t b = 0; b < maxBatchSize; ++b)
        {
            for (int32_t p = 0; p < maxSeqLen; ++p)
            {
                half const v = __float2half(kvAt(kv, b, 0, p));
                int64_t const base = rawIndex(kv, b, 0, p);
                for (int32_t d = 0; d < headDim; ++d)
                {
                    hostInit[base + d] = v;
                }
            }
        }
    }
    copyHostToDevice<half>(layerCache, hostInit);

    std::vector<KVLayerInfo> hostInfos{KVLayerInfo{layerCache.rawPointer(), numKVHeads, maxSeqLen}};
    rt::Tensor deviceInfos = uploadLayerInfos(hostInfos, stream);

    auto acceptedIndicesDevice = rt::Tensor({activeBatchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto kvCacheLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
    copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
    copyHostToDevice<int32_t>(kvCacheLengthsDevice, inputKvCacheLengths);

    // Full swap: slot 0's logical pages [0,1] land on physical pages [2,3] (slot 1's identity
    // range); slot 1's logical pages [0,1] land on physical pages [0,1] (slot 0's identity range).
    // Neither slot maps to its own identity range, and no physical page is targeted twice.
    rt::KVPageTable pageTable(maxBatchSize, maxPagesPerSeq, numPages);
    std::vector<int32_t> const slot0Pages = {2, 3};
    std::vector<int32_t> const slot1Pages = {0, 1};
    pageTable.setRow(0, slot0Pages.data(), maxPagesPerSeq);
    pageTable.setRow(1, slot1Pages.data(), maxPagesPerSeq);
    pageTable.upload(stream);

    eagleBaseCommitKVCache(acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice,
        static_cast<KVLayerInfo const*>(deviceInfos.rawPointer()), numLayers, headDim, maxKVHeads, activeBatchSize,
        maxDepth, DataType::kHALF, stream, pageTable.kernelView().dataPointer<int32_t>(), numPages, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const host = copyDeviceToHost<half>(layerCache);
    auto valueAt = [&](int32_t kv, int32_t b, int32_t pos) { return __half2float(host[rawIndex(kv, b, 0, pos)]); };

    for (int32_t kv = 0; kv < 2; ++kv)
    {
        // Slot 0 (logical) -> physical batch 1: dst logical pos 127 <- src logical pos 131;
        // dst logical pos 128 <- src logical pos 135. Both land in raw batch-1 coordinates
        // since slot 0's pages [0,1] alias physical pages [2,3] == raw batch 1's own pages.
        EXPECT_TRUE(isclose(valueAt(kv, 1, 127), kvAt(kv, 1, 0, 131), 1e-3f, 1e-3f))
            << "kv=" << kv << " slot0->phys pos127 (page-boundary straddle) mismatch";
        EXPECT_TRUE(isclose(valueAt(kv, 1, 128), kvAt(kv, 1, 0, 135), 1e-3f, 1e-3f))
            << "kv=" << kv << " slot0->phys pos128 (page-boundary straddle) mismatch";

        // Slot 1 (logical) -> physical batch 0: dst logical pos 11 <- src logical pos 13;
        // dst logical pos 12 <- src logical pos 15.
        EXPECT_TRUE(isclose(valueAt(kv, 0, 11), kvAt(kv, 0, 0, 13), 1e-3f, 1e-3f))
            << "kv=" << kv << " slot1->phys pos11 mismatch";
        EXPECT_TRUE(isclose(valueAt(kv, 0, 12), kvAt(kv, 0, 0, 15), 1e-3f, 1e-3f))
            << "kv=" << kv << " slot1->phys pos12 mismatch";

        // The remapped commit must leave slot 0's identity-mapped physical location untouched.
        EXPECT_TRUE(isclose(valueAt(kv, 0, 127), kvAt(kv, 0, 0, 127), 1e-3f, 1e-3f))
            << "kv=" << kv << " raw batch0 pos127 was wrongly touched";
    }
}

TEST(EagleKernels, EagleBaseCommitKVCacheSkipsWrongPlanePageIds)
{
    cudaStream_t stream = nullptr;

    int32_t const headDim = 64;
    int32_t const numKVHeads = 1;
    int32_t const activeBatchSize = 2;
    int32_t const maxDepth = 3;
    int32_t const maxPagesPerSeq = 2;
    int32_t const numPages = 4;
    int32_t const maxSeqLen = maxPagesPerSeq * rt::kTOKENS_PER_PAGE;

    rt::Tensor layerCache(
        {2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> hostInit(
        static_cast<size_t>(2) * numPages * rt::kTOKENS_PER_PAGE * numKVHeads * headDim, __float2half(0.0F));
    auto const flatIndex = [&](int32_t pageId, int32_t inPage, int32_t dim) {
        return (static_cast<int64_t>(pageId) * rt::kTOKENS_PER_PAGE + inPage) * numKVHeads * headDim + dim;
    };
    auto const seedToken = [&](int32_t pageId, int32_t inPage, float value) {
        for (int32_t dim = 0; dim < headDim; ++dim)
        {
            hostInit[flatIndex(pageId, inPage, dim)] = __float2half(value);
        }
    };

    seedToken(/*pageId=*/1, /*inPage=*/0, 11.0F);
    seedToken(/*pageId=*/4, /*inPage=*/126, 55.0F);
    seedToken(/*pageId=*/4, /*inPage=*/127, 44.0F);
    seedToken(/*pageId=*/6, /*inPage=*/0, 22.0F);
    seedToken(/*pageId=*/0, /*inPage=*/126, 66.0F);
    seedToken(/*pageId=*/0, /*inPage=*/127, 77.0F);
    copyHostToDevice<half>(layerCache, hostInit);

    std::vector<KVLayerInfo> const hostInfos{KVLayerInfo{layerCache.rawPointer(), numKVHeads, maxSeqLen}};
    rt::Tensor deviceInfos = uploadLayerInfos(hostInfos, stream);

    rt::Tensor acceptedIndicesDevice({activeBatchSize, maxDepth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor acceptLengthsDevice({activeBatchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor kvCacheLengthsDevice({activeBatchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(acceptedIndicesDevice, {0, 2, 0, 0, 2, 0});
    copyHostToDevice<int32_t>(acceptLengthsDevice, {3, 3});
    copyHostToDevice<int32_t>(kvCacheLengthsDevice, {126, 126});

    rt::Tensor pageTable({activeBatchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(pageTable,
        {4, 1, rt::kUNUSED_PAGE_ENTRY, rt::kUNUSED_PAGE_ENTRY, rt::kUNUSED_PAGE_ENTRY, rt::kUNUSED_PAGE_ENTRY, 0, 6});

    eagleBaseCommitKVCache(acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice,
        static_cast<KVLayerInfo const*>(deviceInfos.rawPointer()), /*numLayers=*/1, headDim, numKVHeads,
        activeBatchSize, maxDepth, DataType::kHALF, stream, pageTable.dataPointer<int32_t>(), numPages, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const host = copyDeviceToHost<half>(layerCache);
    for (int32_t dim = 0; dim < headDim; ++dim)
    {
        EXPECT_FLOAT_EQ(__half2float(host[flatIndex(/*pageId=*/4, /*inPage=*/127, dim)]), 44.0F);
        EXPECT_FLOAT_EQ(__half2float(host[flatIndex(/*pageId=*/1, /*inPage=*/0, dim)]), 0.0F);
        EXPECT_FLOAT_EQ(__half2float(host[flatIndex(/*pageId=*/0, /*inPage=*/127, dim)]), 77.0F);
        EXPECT_FLOAT_EQ(__half2float(host[flatIndex(/*pageId=*/6, /*inPage=*/0, dim)]), 0.0F);
    }
}

// ============================================================================
// Test 8f: eagleBaseCommitKVCache accept/rollback with accepted and rejected tree nodes
// Description: Review item B5 required coverage for accept/rollback across a speculative tree:
// positions beyond `acceptLength` represent rejected tree nodes and must be left completely
// untouched (rollback), while positions within `acceptLength` must be compacted from their
// accepted source position. Batch 0 partially accepts (some tree nodes rejected); batch 1 rejects
// everything past the root (full rollback, `acceptLength == 1`, no writes at all).
// ============================================================================
TEST(EagleKernels, EagleBaseCommitKVCacheAcceptRollbackTreeNodes)
{
    cudaStream_t stream = nullptr;

    int32_t const headDim = 64;
    int32_t const numKVHeads = 1;
    int32_t const maxBatchSize = 2;
    int32_t const maxSeqLen = rt::kTOKENS_PER_PAGE;
    int32_t const numLayers = 1;
    int32_t const maxKVHeads = numKVHeads;
    int32_t const activeBatchSize = 2;
    int32_t const maxDepth = 5;
    int32_t const maxPagesPerSeq = maxSeqLen / rt::kTOKENS_PER_PAGE;
    int32_t const numPages = maxBatchSize * maxPagesPerSeq;

    // Batch 0: pastKv=20, tree nodes drafted at i=1..4; accept path keeps i=1,2 (accepted from
    // source offsets 2 and 5) and rejects tree nodes at i=3,4 (never written -> rollback).
    // Batch 1: pastKv=30, acceptLength=1 (root only) -> full rollback, no writes for i=1..4 at all.
    std::vector<int32_t> const inputAcceptedIndices = {0, 2, 5, -1, -1, 0, -1, -1, -1, -1};
    std::vector<int32_t> const inputAcceptLengths = {3, 1};
    std::vector<int32_t> const inputKvCacheLengths = {20, 30};

    auto kvAt = [&](int32_t kv, int32_t b, int32_t h, int32_t pos) {
        return static_cast<float>(kv * 10000 + b * 1000 + h * 100 + pos % 97);
    };

    rt::Tensor layerCache(
        {2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> hostInit(static_cast<size_t>(2) * numPages * rt::kTOKENS_PER_PAGE * numKVHeads * headDim);
    auto rawIndex = [&](int32_t kv, int32_t b, int32_t h, int32_t pos) {
        int32_t const physicalPage = b * maxPagesPerSeq + pos / rt::kTOKENS_PER_PAGE;
        int32_t const inPage = pos % rt::kTOKENS_PER_PAGE;
        return (((static_cast<int64_t>(kv) * numPages + physicalPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHeads + h)
            * headDim;
    };
    for (int32_t kv = 0; kv < 2; ++kv)
    {
        for (int32_t b = 0; b < maxBatchSize; ++b)
        {
            for (int32_t p = 0; p < maxSeqLen; ++p)
            {
                half const v = __float2half(kvAt(kv, b, 0, p));
                int64_t const base = rawIndex(kv, b, 0, p);
                for (int32_t d = 0; d < headDim; ++d)
                {
                    hostInit[base + d] = v;
                }
            }
        }
    }
    copyHostToDevice<half>(layerCache, hostInit);

    std::vector<KVLayerInfo> hostInfos{KVLayerInfo{layerCache.rawPointer(), numKVHeads, maxSeqLen}};
    rt::Tensor deviceInfos = uploadLayerInfos(hostInfos, stream);

    auto acceptedIndicesDevice = rt::Tensor({activeBatchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto kvCacheLengthsDevice = rt::Tensor({activeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
    copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
    copyHostToDevice<int32_t>(kvCacheLengthsDevice, inputKvCacheLengths);

    rt::KVPageTable pageTable(maxBatchSize, maxPagesPerSeq, maxBatchSize * maxPagesPerSeq);
    pageTable.setIdentity();
    pageTable.upload(stream);

    eagleBaseCommitKVCache(acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice,
        static_cast<KVLayerInfo const*>(deviceInfos.rawPointer()), numLayers, headDim, maxKVHeads, activeBatchSize,
        maxDepth, DataType::kHALF, stream, pageTable.kernelView().dataPointer<int32_t>(), numPages, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const host = copyDeviceToHost<half>(layerCache);
    auto valueAt = [&](int32_t kv, int32_t b, int32_t pos) { return __half2float(host[rawIndex(kv, b, 0, pos)]); };

    for (int32_t kv = 0; kv < 2; ++kv)
    {
        // Batch 0: accepted tree nodes moved into the compacted positions.
        EXPECT_TRUE(isclose(valueAt(kv, 0, 21), kvAt(kv, 0, 0, 22), 1e-3f, 1e-3f))
            << "kv=" << kv << " batch0 accepted pos21 mismatch";
        EXPECT_TRUE(isclose(valueAt(kv, 0, 22), kvAt(kv, 0, 0, 25), 1e-3f, 1e-3f))
            << "kv=" << kv << " batch0 accepted pos22 mismatch";
        // Batch 0: rejected tree nodes (i=3,4) must be untouched (rollback).
        EXPECT_TRUE(isclose(valueAt(kv, 0, 23), kvAt(kv, 0, 0, 23), 1e-3f, 1e-3f))
            << "kv=" << kv << " batch0 rejected pos23 was wrongly written";
        EXPECT_TRUE(isclose(valueAt(kv, 0, 24), kvAt(kv, 0, 0, 24), 1e-3f, 1e-3f))
            << "kv=" << kv << " batch0 rejected pos24 was wrongly written";

        // Batch 1: full rollback (acceptLength=1) -- every speculative tree position untouched.
        for (int32_t pos = 31; pos <= 34; ++pos)
        {
            EXPECT_TRUE(isclose(valueAt(kv, 1, pos), kvAt(kv, 1, 0, pos), 1e-3f, 1e-3f))
                << "kv=" << kv << " batch1 pos" << pos << " was wrongly written under full rollback";
        }
    }
}

// ============================================================================
// Test 8c: eagleBaseAssembleHiddenState
// Description: Exercise the 3D hidden-state compaction independently. This is the production
// path when the EAGLE accept step invokes the split assembler exactly once after looping the
// per-layer KV commit.
//
// Phase 1 validates single-call compaction against expected values.
//
// Phase 2 validates the non-idempotency property documented on the public API: calling the
// assembler twice produces different output than calling it once. For this to actually
// manifest, the second call's read positions (driven by acceptedIndices) must lie inside the
// write region of the first call. The kernel writes to flat positions [0, acceptLen) within
// each batch's compacted slab (output stride = maxDepth), so we pick acceptedIndices whose
// values are all < acceptLen. The specific pattern below (batch 0 = [2, 0, 1]) creates a cycle
// in the index → position mapping; one call produces one permutation, two calls produce a
// different permutation.
// ============================================================================
TEST(EagleKernels, EagleBaseAssembleHiddenStateNonIdempotent)
{
    cudaStream_t stream = nullptr;
    // maxDepth=3 and acceptLen=3 with acceptedIndices all < 3 guarantees the first call's write
    // region overlaps the second call's read region (both are positions [0, 3) of each slab),
    // which is the precondition for non-idempotency to manifest.
    int32_t maxDepth = 3;
    // numTokens=8 keeps maxDepth < numTokens (the kernel relies on this invariant to avoid
    // overwriting unread data during the single-call compaction) while staying compact.
    int32_t numTokens = 8;
    int32_t baseHiddenDim = 512;

    int32_t batchSize = 2;

    // Accept patterns that each form a non-trivial permutation of {0, 1, 2}. After the kernel
    // runs, positions 0..2 of each slab hold the permuted input values; running the kernel
    // again on those permuted values reads a different sequence and produces a different
    // permutation, demonstrating non-idempotency.
    //
    // Batch 0: acceptedIndices=[2, 0, 1] → after 1 call: [input[2], input[0], input[1]]
    //                                    → after 2 calls (reading from permuted buffer):
    //                                      [buf[2], buf[0], buf[1]] = [input[1], input[2], input[0]]
    // Batch 1: acceptedIndices=[1, 2, 0] → after 1 call: [input[1], input[2], input[0]]
    //                                    → after 2 calls: [input[0], input[1], input[2]] (input order!)
    std::vector<int32_t> inputAcceptedIndices = {
        2, 0, 1, // Batch 0
        1, 2, 0  // Batch 1
    };
    std::vector<int32_t> inputAcceptLengths = {3, 3};

    // Seed input with distinct markers at positions 0..2 (the only positions the test reads),
    // and leave positions 3..numTokens-1 as zero. Markers are deterministic so we can compute
    // expected outputs by hand.
    // Batch 0 positions [0, 1, 2] = [100, 200, 300]
    // Batch 1 positions [0, 1, 2] = [1100, 1200, 1300]
    std::vector<half> inputHiddenState(batchSize * numTokens * baseHiddenDim, __float2half(0.0f));
    auto setMarker = [&](int b, int t, float v) {
        for (int d = 0; d < baseHiddenDim; ++d)
        {
            inputHiddenState[b * numTokens * baseHiddenDim + t * baseHiddenDim + d] = __float2half(v);
        }
    };
    setMarker(0, 0, 100.0f);
    setMarker(0, 1, 200.0f);
    setMarker(0, 2, 300.0f);
    setMarker(1, 0, 1100.0f);
    setMarker(1, 1, 1200.0f);
    setMarker(1, 2, 1300.0f);

    auto acceptedIndicesDevice = rt::Tensor({batchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto hiddenStateDevice = rt::Tensor({batchSize, numTokens, baseHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
    copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
    copyHostToDevice<half>(hiddenStateDevice, inputHiddenState);

    // ---- Phase 1: single call — expect correct compaction. ----
    eagleBaseAssembleHiddenState(acceptedIndicesDevice, acceptLengthsDevice, hiddenStateDevice, stream);

    auto const afterOneCall = copyDeviceToHost<half>(hiddenStateDevice);

    // The kernel writes compacted output at stride=maxDepth (see eagleBaseAssembleHiddenStateKernel:
    // outputOffset = batchIdx * maxDepth * hiddenDim). So batch 0 lands at flat positions
    // [0, 1, 2] * baseHiddenDim and batch 1 at flat positions [maxDepth, maxDepth+1, maxDepth+2]
    // * baseHiddenDim (overlapping batch 0's allocation tail, which is fine per the kernel's
    // in-place contract — the caller reshapes afterwards).
    std::vector<float> expectedBatch0AfterOne{300.0f, 100.0f, 200.0f};    // input[2], input[0], input[1]
    std::vector<float> expectedBatch1AfterOne{1200.0f, 1300.0f, 1100.0f}; // input[1], input[2], input[0]

    int32_t const batch0Offset = 0;
    int32_t const batch1Offset = maxDepth * baseHiddenDim;

    for (int i = 0; i < 3; ++i)
    {
        float const actual = __half2float(afterOneCall[batch0Offset + i * baseHiddenDim]);
        EXPECT_TRUE(isclose(actual, expectedBatch0AfterOne[i], 1e-3f, 1e-3f))
            << "After 1 call — Batch 0, position " << i << ": expected " << expectedBatch0AfterOne[i] << ", got "
            << actual;
    }
    for (int i = 0; i < 3; ++i)
    {
        float const actual = __half2float(afterOneCall[batch1Offset + i * baseHiddenDim]);
        EXPECT_TRUE(isclose(actual, expectedBatch1AfterOne[i], 1e-3f, 1e-3f))
            << "After 1 call — Batch 1, position " << i << ": expected " << expectedBatch1AfterOne[i] << ", got "
            << actual;
    }

    // ---- Phase 2: second call on the already-compacted buffer. ----
    // The kernel always reads from `inputOffset = batchIdx * numTokens * hiddenDim`. For batch 0
    // (batchIdx=0) that is flat offset 0 — exactly where the first call's compacted output
    // landed. For batch 1 (batchIdx=1) that is flat offset `numTokens * hiddenDim = 8*512`, which
    // was *not* touched by the first call (the first call's batch-1 writes went to offset
    // `maxDepth * hiddenDim = 3*512`, inside batch 0's allocation slab). So batch 1 re-reads the
    // original input unchanged and produces the same output as phase 1 — we only assert
    // non-idempotency on batch 0, where the overlap is genuine.
    eagleBaseAssembleHiddenState(acceptedIndicesDevice, acceptLengthsDevice, hiddenStateDevice, stream);

    auto const afterTwoCalls = copyDeviceToHost<half>(hiddenStateDevice);

    // Batch 0 expected after two calls: the first call left positions [0, 1, 2] holding
    // [300, 100, 200]. The second call reads positions [2, 0, 1] from this permuted buffer, i.e.
    // [200, 300, 100], and writes those to positions [0, 1, 2].
    std::vector<float> expectedBatch0AfterTwo{200.0f, 300.0f, 100.0f};
    bool batch0Deviation = false;
    for (int i = 0; i < 3; ++i)
    {
        float const actual = __half2float(afterTwoCalls[batch0Offset + i * baseHiddenDim]);
        // Confirm the exact expected permutation.
        EXPECT_TRUE(isclose(actual, expectedBatch0AfterTwo[i], 1e-3f, 1e-3f))
            << "After 2 calls — Batch 0, position " << i << ": expected " << expectedBatch0AfterTwo[i] << ", got "
            << actual;
        if (!isclose(actual, expectedBatch0AfterOne[i], 1e-3f, 1e-3f))
        {
            batch0Deviation = true;
        }
    }
    EXPECT_TRUE(batch0Deviation)
        << "Non-idempotency regression guard failed: calling eagleBaseAssembleHiddenState twice "
        << "produced identical output to a single call on batch 0. The public API docstring "
        << "promises the hidden-state compaction is NOT idempotent; if this test starts passing "
        << "after a kernel change, either the docstring needs updating or the runtime's "
        << "split-call sequencing in runBaseModelVerification needs a re-audit.";
}

// ============================================================================
// Test 9: prepareEagleDraftProposalInputs
// Description: Pack tree mask, compute positions, prepare select indices
// selectIndices = [treeLen-selectTokenLen : treeLen]
// ============================================================================
TEST(EagleKernels, PrepareEagleDraftProposalInputs)
{
    cudaStream_t stream = nullptr;
    int32_t paddedDraftTreeSize = 12;
    int32_t selectTokenLength = 4;
    int32_t packedTreeMaskLen = divUp(paddedDraftTreeSize, 32);

    std::vector<int32_t> inputDraftTreeLength = {8, 9, 10, 11, 8, 9, 10, 11};
    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};

    // Simple causal mask
    std::vector<int8_t> inputDraftTreeMask(8 * paddedDraftTreeSize * paddedDraftTreeSize, 0);
    for (int b = 0; b < 8; b++)
    {
        int treeLen = inputDraftTreeLength[b];
        for (int i = 0; i < treeLen; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                inputDraftTreeMask[b * paddedDraftTreeSize * paddedDraftTreeSize + i * paddedDraftTreeSize + j] = 1;
            }
        }
    }

    // Expected batch 0: contextLen=112, selectIndices=[4,5,6,7]
    int32_t expectedContextLenBatch0 = 112;
    std::vector<int64_t> expectedSelectIndicesBatch0 = {4, 5, 6, 7};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedDraftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
        auto tensorPositionIndicesDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectTokenIndicesDevice = rt::Tensor({batchSize, selectTokenLength}, // 2D tensor
            rt::DeviceType::kGPU, DataType::kINT64);
        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(draftTreeMaskDevice.rawPointer(), inputDraftTreeMask.data(),
            batchSize * paddedDraftTreeSize * paddedDraftTreeSize * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftTreeLengthDevice.rawPointer(), inputDraftTreeLength.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleDraftProposalInputs(draftTreeMaskDevice, draftTreeLengthDevice, sequenceStartIndicesDevice,
            packedDraftTreeMaskDevice, tensorPositionIndicesDevice, selectTokenIndicesDevice,
            sequenceContextLengthsDevice, stream);

        auto const actualContextLengths = copyDeviceToHost<int32_t>(sequenceContextLengthsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectTokenIndicesDevice);

        // Verify batch 0
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);
        for (int i = 0; i < selectTokenLength; i++)
        {
            EXPECT_EQ(actualSelectIndices[i], expectedSelectIndicesBatch0[i]);
        }
    }
}

// ============================================================================
// Test 10: assembleDraftTreeInput (subsequent rounds)
// Description: Assemble draft tree input for round > 0
// Select tokens from topK x topK candidates, build tree mask from parent
// ============================================================================
TEST(EagleKernels, AssembleDraftTreeInput)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t paddedDraftTreeSize = 12;
    int32_t draftHiddenDim = 256;
    int32_t round = 1;

    std::vector<int32_t> inputDraftIdTable(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputDraftIdTable[b * 16 + i] = 200 + b * 100 + i;
        }
    }

    // Select: [0,5,10,15] for batch 0
    std::vector<int32_t> inputSelectedIndices = {
        0, 5, 10, 15, // Batch 0
        1, 6, 11, 14, // Batch 1
        2, 7, 9, 13,  // Batch 2
        3, 4, 8, 12,  // Batch 3
        0, 5, 10, 15, // Batch 4
        1, 6, 11, 14, // Batch 5
        2, 7, 9, 13,  // Batch 6
        3, 4, 8, 12   // Batch 7
    };

    std::vector<half> inputHiddenOutput(8 * draftTopK * draftHiddenDim, __float2half(2.0f));

    // Expected batch 0: inputIds[4:7] = [200,205,210,215]
    std::vector<int32_t> expectedInputIdsRound1Batch0 = {200, 205, 210, 215};
    int32_t expectedTreeLength = 8; // 4 + 4

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenOutputDevice
            = rt::Tensor({batchSize * draftTopK, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto inputIdsDevice = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesInputDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

        std::vector<int32_t> initialTreeLength(batchSize, draftTopK);
        copyHostToDevice<int32_t>(draftTreeLengthDevice, initialTreeLength);

        CUDA_CHECK(cudaMemcpy(draftIdTableDevice.rawPointer(), inputDraftIdTable.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftHiddenOutputDevice.rawPointer(), inputHiddenOutput.data(),
            batchSize * draftTopK * draftHiddenDim * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));

        assembleDraftTreeInput(draftIdTableDevice, draftHiddenOutputDevice, selectedIndicesDevice, inputIdsDevice,
            draftHiddenStatesInputDevice, draftTreeLengthDevice, draftTreeMaskDevice, draftTopK, round, stream);

        auto const actualInputIds = copyDeviceToHost<int32_t>(inputIdsDevice);
        auto const actualTreeLength = copyDeviceToHost<int32_t>(draftTreeLengthDevice);

        // Verify batch 0
        EXPECT_EQ(actualTreeLength[0], expectedTreeLength);
        for (int i = 0; i < draftTopK; i++)
        {
            int pos = round * draftTopK + i;
            EXPECT_EQ(actualInputIds[pos], expectedInputIdsRound1Batch0[i]);
        }
    }
}

// ============================================================================
// Test 11: assembleIntermediateData (subsequent rounds)
// Description: Copy cuLogProbs to scores; compute parent pointers in full table
// parent = startOffset + selectedIndices[i]
// ============================================================================
TEST(EagleKernels, AssembleIntermediateData)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t round = 1;

    std::vector<float> inputCuLogProbs = {
        -1.5f, -1.7f, -1.9f, -2.1f,     // Batch 0
        -1.6f, -1.8f, -2.0f, -2.2f,     // Batch 1
        -1.55f, -1.75f, -1.95f, -2.15f, // Batch 2
        -1.65f, -1.85f, -2.05f, -2.25f, // Batch 3
        -1.5f, -1.7f, -1.9f, -2.1f,     // Batch 4
        -1.6f, -1.8f, -2.0f, -2.2f,     // Batch 5
        -1.55f, -1.75f, -1.95f, -2.15f, // Batch 6
        -1.65f, -1.85f, -2.05f, -2.25f  // Batch 7
    };

    std::vector<int32_t> inputSelectedIndices = {
        0, 5, 10, 15, // Batch 0
        1, 6, 11, 14, // Batch 1
        2, 7, 9, 13,  // Batch 2
        3, 4, 8, 12,  // Batch 3
        0, 5, 10, 15, // Batch 4
        1, 6, 11, 14, // Batch 5
        2, 7, 9, 13,  // Batch 6
        3, 4, 8, 12   // Batch 7
    };

    // Expected batch 0: startOffset = 1 + 4 + 16*0 = 5
    // parents = [5+0, 5+5, 5+10, 5+15] = [5, 10, 15, 20]
    std::vector<int32_t> expectedParentsBatch0 = {5, 10, 15, 20};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto cuLogProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(cuLogProbsDevice.rawPointer(), inputCuLogProbs.data(),
            batchSize * draftTopK * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));

        assembleIntermediateData(cuLogProbsDevice, selectedIndicesDevice, intermediateScoresDevice,
            intermediateParentsDevice, draftTopK, round, stream);

        auto const actualScores = copyDeviceToHost<float>(intermediateScoresDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(intermediateParentsDevice);

        // Verify batch 0
        for (int i = 0; i < draftTopK; i++)
        {
            EXPECT_FLOAT_EQ(actualScores[i], inputCuLogProbs[i]);
            EXPECT_EQ(actualParents[i], expectedParentsBatch0[i]);
        }
    }
}

// ============================================================================
// Test 12: updateDraftTreeFullTables
// Description: Update full table at specific offset with draft results
// dstOffset = 1 + topK + topK^2 * round
// ============================================================================
TEST(EagleKernels, UpdateDraftTreeFullTables)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t round = 0;
    int32_t fullTableLength = 21;

    std::vector<int32_t> inputDraftIdTable(8 * 16);
    std::vector<float> inputDraftScoreTable(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputDraftIdTable[b * 16 + i] = 300 + b * 100 + i;
            inputDraftScoreTable[b * 16 + i] = -1.0f - b * 0.1f - i * 0.05f;
        }
    }

    std::vector<int32_t> inputIntermediateParents(8 * draftTopK);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < draftTopK; i++)
        {
            inputIntermediateParents[b * draftTopK + i] = 5 + i;
        }
    }

    // Expected batch 0: write to offset 5-20 (round 0)
    int32_t dstOffset = 5;

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreFullTableDevice
            = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto draftParentFullTableDevice
            = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);

        std::vector<int32_t> init_ids(batchSize * fullTableLength, 0);
        std::vector<float> init_scores(batchSize * fullTableLength, -INFINITY);
        std::vector<int32_t> init_parents(batchSize * fullTableLength, -5);

        CUDA_CHECK(cudaMemcpy(draftIdTableDevice.rawPointer(), inputDraftIdTable.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftScoreTableDevice.rawPointer(), inputDraftScoreTable.data(),
            batchSize * 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(intermediateParentsDevice.rawPointer(), inputIntermediateParents.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(draftIdFullTableDevice, init_ids);
        copyHostToDevice<float>(draftScoreFullTableDevice, init_scores);
        copyHostToDevice<int32_t>(draftParentFullTableDevice, init_parents);

        updateDraftTreeFullTables(draftIdTableDevice, draftScoreTableDevice, intermediateParentsDevice,
            draftIdFullTableDevice, draftScoreFullTableDevice, draftParentFullTableDevice, draftTopK, round, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdFullTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreFullTableDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(draftParentFullTableDevice);

        // Verify batch 0 first few entries
        EXPECT_EQ(actualIds[dstOffset], inputDraftIdTable[0]);
        EXPECT_FLOAT_EQ(actualScores[dstOffset], inputDraftScoreTable[0]);
        EXPECT_EQ(actualParents[dstOffset], inputIntermediateParents[0]);
    }
}

// ============================================================================
// Test 13: prepareEagleBaseTreeDecodingInputs
// Description: Prepare inputs for base model tree decoding (similar to draft proposal)
// ============================================================================
TEST(EagleKernels, PrepareEagleBaseTreeDecodingInputs)
{
    cudaStream_t stream = nullptr;
    int32_t treeSize = 12;
    int32_t packedTreeMaskLen = divUp(treeSize, 32);

    std::vector<int8_t> inputBaseTreeDecodingMask(8 * treeSize * treeSize, 0);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < treeSize; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                inputBaseTreeDecodingMask[b * treeSize * treeSize + i * treeSize + j] = 1;
            }
        }
    }

    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};

    // Expected batch 0: positions=[100,101,...,111], contextLen=112
    std::vector<int32_t> expectedPositionsBatch0(treeSize);
    for (int i = 0; i < treeSize; i++)
    {
        expectedPositionsBatch0[i] = 100 + i;
    }
    int32_t expectedContextLenBatch0 = 112;

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto baseTreeDecodingMaskDevice
            = rt::Tensor({batchSize, treeSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedBaseTreeDecodingMaskDevice
            = rt::Tensor({batchSize, treeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
        auto tensorPositionIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectTokenIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(baseTreeDecodingMaskDevice.rawPointer(), inputBaseTreeDecodingMask.data(),
            batchSize * treeSize * treeSize * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMaskDevice, sequenceStartIndicesDevice,
            packedBaseTreeDecodingMaskDevice, tensorPositionIndicesDevice, selectTokenIndicesDevice,
            sequenceContextLengthsDevice, stream);

        auto const actualPositions = copyDeviceToHost<int32_t>(tensorPositionIndicesDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectTokenIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(sequenceContextLengthsDevice);

        // Verify batch 0
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);
        for (int i = 0; i < treeSize; i++)
        {
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]);
            EXPECT_EQ(actualSelectIndices[i], i);
        }
    }
}
