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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

void TestXQATreeAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvSequenceLength, int32_t qSequenceLength)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    std::vector<int32_t> kvCacheLength(batchSize, kvSequenceLength);
    std::vector<half> qInput;
    std::vector<half> kvInput;
    std::vector<half> outReference;
    std::vector<int32_t> packedTreeMaskInput;

    for (int32_t i = 0; i < batchSize; i++)
    {
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        std::vector<half> ki(numKVHeads * headSize * kvSequenceLength);
        std::vector<half> vi(numKVHeads * headSize * kvSequenceLength);
        std::vector<int32_t> treeMaski(qSequenceLength * qSequenceLength);

        uniformFloatInitialization(qi);
        uniformFloatInitialization(ki);
        uniformFloatInitialization(vi);
        uniformIntInitialization(treeMaski, 0, 1);

        auto ref = casualAttentionRef(qi, ki, vi, qSequenceLength, kvSequenceLength, numQHeads, numKVHeads, headSize,
            std::make_optional(treeMaski));

        // Add data from batch to input Tensors
        qInput.insert(qInput.end(), qi.begin(), qi.end());

        // KVcache layout assumed to have layout of [B, 2n H, S, D]
        kvInput.insert(kvInput.end(), ki.begin(), ki.end());
        kvInput.insert(kvInput.end(), vi.begin(), vi.end());
        outReference.insert(outReference.end(), ref.begin(), ref.end());

        // Prepare packed tree mask. The layout of mask is [qSeqLen, qSeqLen]. Which represent whether two tokens
        // will attend to each other.
        int32_t const numBitsPerPackedMask = 32;
        int32_t const numPackedMasksPerToken = divUp(qSequenceLength, numBitsPerPackedMask);
        std::vector<int32_t> packedMaski(numPackedMasksPerToken * qSequenceLength, 0);
        for (int32_t i = 0; i < qSequenceLength; i++)
        {
            for (int32_t j = 0; j < numPackedMasksPerToken; j++)
            {
                int32_t mask = 0;
                for (int32_t k = 0; k < numBitsPerPackedMask; k++)
                {
                    int32_t const bitIndex = j * numBitsPerPackedMask + k;
                    int32_t maskFlag = 0;
                    if (j * numBitsPerPackedMask + k < qSequenceLength)
                    {
                        maskFlag = treeMaski[i * qSequenceLength + bitIndex];
                    }
                    mask |= maskFlag << k;
                }
                packedMaski[i * numPackedMasksPerToken + j] = mask;
            }
        }
        packedTreeMaskInput.insert(packedTreeMaskInput.end(), packedMaski.begin(), packedMaski.end());
    }
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 1.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> packedTreeMaskDevice(packedTreeMaskInput);

    trt_edgellm::DecoderXQARunner runner(DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qSeqLen = qSequenceLength;
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvSequenceLength;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.treeAttnMask = thrust::raw_pointer_cast(packedTreeMaskDevice.data());
    // Use default stream .
    cudaStream_t stream{nullptr};
    runner.dispatchSpecDecodeXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Check accuracy.
    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * qSequenceLength * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (__hisnan(outHost[i]))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * qSequenceLength * numQHeads * headSize);

    std::cout << "XQA Tree Attention Decoding test. batch_size: " << batchSize << " num_Q_heads: " << numQHeads
              << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " kvcache seq_len: " << kvSequenceLength << " q_seq_len: " << qSequenceLength
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio4HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 512, 10);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 256, 32);
    /// KVSequence 64, QSequence 128
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 320, 60);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio8HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 64);
    /// KVSequence 192, QSequence 60 KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 192, 60);
    /// KVSequence 512, QSequence 20， KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 512, 20);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio7HeadDim64)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 128, 64);
    /// KVSequence 192, QSequence 60
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 192, 60);
    /// KVSequence 512, QSequence 20
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 512, 20);
}

/// Test to reproduce padding issue: compare padded vs non-padded attention output
/// This test verifies that when we pad the Q sequence, the output for valid tokens
/// should remain the same as the non-padded case.
/// Also compares both outputs against CPU reference implementation.
void TestXQAPaddingConsistency(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvSequenceLength, int32_t actualQSeqLen, int32_t paddedQSeqLen)
{
    std::cout << "\n========== Padding Consistency Test ==========" << std::endl;
    std::cout << "batch_size: " << batchSize << ", numQHeads: " << numQHeads << ", numKVHeads: " << numKVHeads
              << ", headSize: " << headSize << std::endl;
    std::cout << "kvSeqLen: " << kvSequenceLength << ", actualQSeqLen: " << actualQSeqLen
              << ", paddedQSeqLen: " << paddedQSeqLen << std::endl;

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    // Generate random Q, K, V data for actual sequence length
    // Layout: Q is [seqLen, numQHeads, headSize], K/V is [seqLen, numKVHeads, headSize]
    std::vector<half> qActual(numQHeads * headSize * actualQSeqLen);
    std::vector<half> kInput(numKVHeads * headSize * kvSequenceLength);
    std::vector<half> vInput(numKVHeads * headSize * kvSequenceLength);

    uniformFloatInitialization(qActual);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    // === Compute CPU Reference ===
    // Create causal tree mask for reference: treeMask[i][j] = 1 if token i can attend to token j
    std::vector<int32_t> treeMaskForRef(actualQSeqLen * actualQSeqLen, 0);
    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        for (int32_t j = 0; j <= i; j++)
        {
            treeMaskForRef[i * actualQSeqLen + j] = 1; // Causal: attend to 0..i
        }
    }

    std::cout << "\n--- Computing CPU Reference ---" << std::endl;
    auto outReference = casualAttentionRef(qActual, kInput, vInput, actualQSeqLen, kvSequenceLength, numQHeads,
        numKVHeads, headSize, std::make_optional(treeMaskForRef));
    std::cout << "Reference output size: " << outReference.size() << std::endl;

    // === Run 1: No padding (actualQSeqLen) ===
    std::vector<half> qNoPadding = qActual;

    // Create packed causal mask for no-padding case
    int32_t const numBitsPerPackedMask = 32;
    int32_t const numPackedMasksPerTokenNoPad = divUp(actualQSeqLen, numBitsPerPackedMask);
    std::vector<int32_t> packedMaskNoPadding(numPackedMasksPerTokenNoPad * actualQSeqLen, 0);

    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        // Token i attends to tokens 0..i (causal mask)
        int32_t mask = 0;
        for (int32_t j = 0; j <= i && j < 32; j++)
        {
            mask |= (1 << j);
        }
        packedMaskNoPadding[i * numPackedMasksPerTokenNoPad] = mask;
    }

    // Print no-padding mask
    std::cout << "\n--- No-Padding Mask (packed, qSeqLen=" << actualQSeqLen << ") ---" << std::endl;
    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        std::cout << "Token " << i << ": packed=0x" << std::hex << packedMaskNoPadding[i * numPackedMasksPerTokenNoPad]
                  << std::dec << " -> attends to: ";
        int32_t mask = packedMaskNoPadding[i * numPackedMasksPerTokenNoPad];
        for (int32_t j = 0; j < actualQSeqLen; j++)
        {
            std::cout << ((mask >> j) & 1);
        }
        std::cout << std::endl;
    }

    // Prepare KV cache (layout: [2 * numKVHeads, S, D] for single batch)
    std::vector<half> kvInput;
    kvInput.insert(kvInput.end(), kInput.begin(), kInput.end());
    kvInput.insert(kvInput.end(), vInput.begin(), vInput.end());

    std::vector<int32_t> kvCacheLength(1, kvSequenceLength);

    // Device memory for no-padding run
    thrust::device_vector<half> qNoPaddingDevice(qNoPadding);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outNoPaddingDevice(actualQSeqLen * numQHeads * headSize, __float2half(0.0f));
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> packedMaskNoPaddingDevice(packedMaskNoPadding);

    trt_edgellm::DecoderXQARunner runnerNoPad(DataType::kHALF, 1, numQHeads, numKVHeads, headSize, smVersion);
    auto paramsNoPad = runnerNoPad.initXQAParams();
    paramsNoPad.qSeqLen = actualQSeqLen;
    paramsNoPad.qInputPtr = thrust::raw_pointer_cast(qNoPaddingDevice.data());
    paramsNoPad.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    paramsNoPad.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    paramsNoPad.kvCache.capacity = kvSequenceLength;
    paramsNoPad.output = thrust::raw_pointer_cast(outNoPaddingDevice.data());
    paramsNoPad.treeAttnMask = thrust::raw_pointer_cast(packedMaskNoPaddingDevice.data());

    cudaStream_t stream{nullptr};
    runnerNoPad.dispatchSpecDecodeXQAKernel(paramsNoPad, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outNoPaddingHost(outNoPaddingDevice);

    // === Run 2: With padding (paddedQSeqLen) ===
    // Pad Q with zeros
    std::vector<half> qPadded(numQHeads * headSize * paddedQSeqLen, __float2half(0.0f));
    for (int32_t t = 0; t < actualQSeqLen; t++)
    {
        for (int32_t h = 0; h < numQHeads; h++)
        {
            for (int32_t d = 0; d < headSize; d++)
            {
                int32_t srcIdx = t * numQHeads * headSize + h * headSize + d;
                int32_t dstIdx = t * numQHeads * headSize + h * headSize + d;
                qPadded[dstIdx] = qActual[srcIdx];
            }
        }
    }

    // Create mask for padded case:
    // - Valid tokens (0..actualQSeqLen-1): causal mask (attend to self and previous)
    // - Padding tokens (actualQSeqLen..paddedQSeqLen-1): only attend to token 0 (to avoid NaN in softmax)
    int32_t const numPackedMasksPerTokenPad = divUp(paddedQSeqLen, numBitsPerPackedMask);
    std::vector<int32_t> packedMaskPadding(numPackedMasksPerTokenPad * paddedQSeqLen, 0);

    for (int32_t i = 0; i < paddedQSeqLen; i++)
    {
        int32_t mask = 0;
        if (i < actualQSeqLen)
        {
            for (int32_t j = 0; j <= i && j < 32; j++)
            {
                mask |= (1 << j);
            }
        }
        else
        {
            mask = 1;
        }
        packedMaskPadding[i * numPackedMasksPerTokenPad] = mask;
    }

    std::cout << "\n--- Padded Mask (packed, qSeqLen=" << paddedQSeqLen << ", actualQSeqLen=" << actualQSeqLen
              << ") ---" << std::endl;
    for (int32_t i = 0; i < paddedQSeqLen; i++)
    {
        std::string tokenType = (i < actualQSeqLen) ? "valid" : "padding";
        std::cout << "Token " << i << " (" << tokenType << "): packed=0x" << std::hex
                  << packedMaskPadding[i * numPackedMasksPerTokenPad] << std::dec << " -> attends to: ";
        int32_t mask = packedMaskPadding[i * numPackedMasksPerTokenPad];
        for (int32_t j = 0; j < paddedQSeqLen; j++)
        {
            std::cout << ((mask >> j) & 1);
        }
        std::cout << std::endl;
    }

    thrust::device_vector<half> qPaddedDevice(qPadded);
    thrust::device_vector<half> outPaddedDevice(paddedQSeqLen * numQHeads * headSize, __float2half(0.0f));
    thrust::device_vector<int32_t> packedMaskPaddingDevice(packedMaskPadding);

    // For padded case, adjust context length to account for padding tokens
    // This ensures correct context K range: K[0 : paddedContextLen - paddedQSeqLen] = K[0 : kvSequenceLength -
    // actualQSeqLen] Same as what eagleUtilKernels does: sequenceContextLengths = sequenceStartIndices +
    // maxAcceptedTokenNum
    int32_t paddedContextLen = kvSequenceLength + (paddedQSeqLen - actualQSeqLen);
    std::vector<int32_t> kvCacheLengthPadded(1, paddedContextLen);
    thrust::device_vector<int32_t> kvCacheLengthPaddedDevice(kvCacheLengthPadded);

    trt_edgellm::DecoderXQARunner runnerPad(DataType::kHALF, 1, numQHeads, numKVHeads, headSize, smVersion);
    auto paramsPad = runnerPad.initXQAParams();
    paramsPad.qSeqLen = paddedQSeqLen;
    paramsPad.qInputPtr = thrust::raw_pointer_cast(qPaddedDevice.data());
    paramsPad.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    paramsPad.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthPaddedDevice.data());
    paramsPad.kvCache.capacity = kvSequenceLength;
    paramsPad.output = thrust::raw_pointer_cast(outPaddedDevice.data());
    paramsPad.treeAttnMask = thrust::raw_pointer_cast(packedMaskPaddingDevice.data());

    runnerPad.dispatchSpecDecodeXQAKernel(paramsPad, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outPaddedHost(outPaddedDevice);

    // === Compare No-padding vs Reference ===
    std::cout << "\n--- No-padding XQA vs CPU Reference ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t i = 0; i < totalElements; i++)
        {
            float valXQA = __half2float(outNoPaddingHost[i]);
            float valRef = __half2float(outReference[i]);
            float absDiff = std::abs(valXQA - valRef);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            if (!isclose(outNoPaddingHost[i], outReference[i], 1e-2, 1e-2))
            {
                numMismatches++;
            }
        }
        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;
        std::cout << "Max Abs Diff: " << maxAbsDiff << ", Match rate (1e-2): " << matchRate * 100 << "%" << std::endl;

        std::cout << "First 10 values:" << std::endl;
        std::cout << "  XQA:       ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outNoPaddingHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "  Reference: ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outReference[i]) << " ";
        std::cout << std::endl;

        EXPECT_GT(matchRate, 0.9f) << "No-padding XQA should match reference!";
    }

    // === Compare Padded vs Reference (only valid tokens) ===
    std::cout << "\n--- Padded XQA vs CPU Reference (valid tokens only) ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t t = 0; t < actualQSeqLen; t++)
        {
            for (int32_t h = 0; h < numQHeads; h++)
            {
                for (int32_t d = 0; d < headSize; d++)
                {
                    int32_t refIdx = t * numQHeads * headSize + h * headSize + d;
                    int32_t padIdx = t * numQHeads * headSize + h * headSize + d;

                    float valXQA = __half2float(outPaddedHost[padIdx]);
                    float valRef = __half2float(outReference[refIdx]);
                    float absDiff = std::abs(valXQA - valRef);
                    maxAbsDiff = std::max(maxAbsDiff, absDiff);
                    if (!isclose(outPaddedHost[padIdx], outReference[refIdx], 1e-2, 1e-2))
                    {
                        numMismatches++;
                    }
                }
            }
        }
        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;
        std::cout << "Max Abs Diff: " << maxAbsDiff << ", Match rate (1e-2): " << matchRate * 100 << "%" << std::endl;

        std::cout << "First 10 values:" << std::endl;
        std::cout << "  Padded XQA: ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outPaddedHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "  Reference:  ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outReference[i]) << " ";
        std::cout << std::endl;

        // This is the KEY test: padded output should also match reference!
        EXPECT_GT(matchRate, 0.9f) << "Padded XQA should also match reference for valid tokens!";
    }

    // === Compare No-padding vs Padded (for valid tokens) ===
    std::cout << "\n--- No-padding XQA vs Padded XQA (valid tokens) ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        int32_t maxAbsDiffIdx = 0;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t i = 0; i < totalElements; i++)
        {
            float valNoPad = __half2float(outNoPaddingHost[i]);
            float valPad = __half2float(outPaddedHost[i]);

            float absDiff = std::abs(valNoPad - valPad);
            float relDiff = absDiff / std::max(std::abs(valNoPad), 1e-6f);

            if (absDiff > maxAbsDiff)
            {
                maxAbsDiff = absDiff;
                maxAbsDiffIdx = i;
            }
            maxRelDiff = std::max(maxRelDiff, relDiff);

            if (!isclose(outNoPaddingHost[i], outPaddedHost[i], 1e-3, 1e-3))
            {
                numMismatches++;
            }
        }

        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;

        std::cout << "Total elements compared: " << totalElements << std::endl;
        std::cout << "Max Absolute Diff: " << maxAbsDiff << " at index " << maxAbsDiffIdx << std::endl;
        std::cout << "  No-pad: " << __half2float(outNoPaddingHost[maxAbsDiffIdx])
                  << ", Padded: " << __half2float(outPaddedHost[maxAbsDiffIdx]) << std::endl;
        std::cout << "Max Relative Diff: " << maxRelDiff * 100 << "%" << std::endl;
        std::cout << "Match rate (1e-3): " << matchRate * 100 << "%" << std::endl;
        std::cout << "Number of mismatches: " << numMismatches << std::endl;

        std::cout << "\nFirst 10 values comparison:" << std::endl;
        std::cout << "No-padding: ";
        for (int32_t i = 0; i < std::min(10, headSize); i++)
            std::cout << __half2float(outNoPaddingHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "Padded:     ";
        for (int32_t i = 0; i < std::min(10, headSize); i++)
            std::cout << __half2float(outPaddedHost[i]) << " ";
        std::cout << std::endl;

        // The key expectation: padded and non-padded should produce same results
        EXPECT_GT(matchRate, 0.99f) << "Padding should not affect valid token outputs!";
        EXPECT_LT(maxAbsDiff, 1e-2f) << "Max absolute diff too large between padded and non-padded!";
    }
}

// Test padding consistency for multi-batch EAGLE.
// The workaround is to pass padded context length (sequenceStartIndices + maxAcceptedTokenNum)
// to ensure correct K range calculation. Only compare actual accepted tokens.
// TODO: Fix XQA kernel to properly handle padding for padded sequence length.
// Currently the XQA tree attention kernel doesn't correctly handle padding tokens
// when actualQSeqLen != paddedQSeqLen. The workaround is to pass padded context length
// (sequenceStartIndices + maxAcceptedTokenNum) to ensure correct K range calculation.
// This test validates padding consistency - enable once the XQA kernel is fixed.
TEST(XQATreeAttentionDecodingTest, PaddingConsistencyTest)
{
    // Reproduce the issue: actualQSeqLen=2, paddedQSeqLen=7
    // This matches the real scenario in EAGLE runtime accept decode token step

    std::cout << "\n=== Test 1: Baseline - no padding (actualQSeqLen == paddedQSeqLen) ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 2, 2);

    std::cout << "\n=== Test 2: Reproduce issue - actualQSeqLen=2, paddedQSeqLen=7 ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 2, 7);

    std::cout << "\n=== Test 3: Different padding ratio ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 3, 10);

    std::cout << "\n=== Test 4: Larger actual sequence ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 5, 7);
}
