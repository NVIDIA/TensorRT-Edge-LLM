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
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

using namespace trt_edgellm;
using namespace nvinfer1;

struct SeqLensTestCase
{
    std::vector<int32_t> inputSeqLen;
    std::vector<int32_t> kvCacheStartIndices; // empty = normal prefill (all zeros)
    int32_t runtimeSeqLen;

    // Expected outputs
    std::vector<int32_t> expectedCuQSeqLens;
    std::vector<int32_t> expectedCuKVSeqLens;
    std::vector<int32_t> expectedKvCacheEndIdxs;
    std::vector<int32_t> expectedPaddedCuKVSeqLens;
};

static void verifyCalCuQCuKVSeqLensAndKVEndIdxs(SeqLensTestCase const& tc)
{
    int32_t const B = static_cast<int32_t>(tc.inputSeqLen.size());

    // Allocate GPU tensors
    rt::Tensor inputSeqLenTensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuQSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kvCacheEndIdxsTensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor paddedCuKVSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(
        cudaMemcpy(inputSeqLenTensor.rawPointer(), tc.inputSeqLen.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice));

    // kvCacheStartIndices: pass empty tensor for normal prefill
    rt::Tensor kvCacheStartIdxTensor;
    if (!tc.kvCacheStartIndices.empty())
    {
        kvCacheStartIdxTensor = rt::Tensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(kvCacheStartIdxTensor.rawPointer(), tc.kvCacheStartIndices.data(), B * sizeof(int32_t),
            cudaMemcpyHostToDevice));
    }

    cudaStream_t stream{nullptr};
    kernel::calCuQCuKVSeqLensAndKVEndIdxs(inputSeqLenTensor, kvCacheStartIdxTensor, cuQSeqLensTensor, cuKVSeqLensTensor,
        kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, tc.runtimeSeqLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Read back and verify all outputs
    auto readBack = [](rt::Tensor const& t, int32_t n) {
        std::vector<int32_t> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), t.rawPointer(), n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return v;
    };

    auto cuQ = readBack(cuQSeqLensTensor, B + 1);
    auto cuKV = readBack(cuKVSeqLensTensor, B + 1);
    auto endIdxs = readBack(kvCacheEndIdxsTensor, B);
    auto paddedCuKV = readBack(paddedCuKVSeqLensTensor, B + 1);

    for (int32_t i = 0; i <= B; ++i)
    {
        EXPECT_EQ(cuQ[i], tc.expectedCuQSeqLens[i]) << "cuQSeqLens mismatch at index " << i;
        EXPECT_EQ(cuKV[i], tc.expectedCuKVSeqLens[i]) << "cuKVSeqLens mismatch at index " << i;
        EXPECT_EQ(paddedCuKV[i], tc.expectedPaddedCuKVSeqLens[i]) << "paddedCuKVSeqLens mismatch at index " << i;
    }
    for (int32_t i = 0; i < B; ++i)
    {
        EXPECT_EQ(endIdxs[i], tc.expectedKvCacheEndIdxs[i]) << "kvCacheEndIdxs mismatch at index " << i;
    }
}

// Normal prefill, single batch
TEST(UtilKernelTest, seqLens_singleBatchNormalPrefill)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128},
        .expectedCuKVSeqLens = {0, 128},
        .expectedKvCacheEndIdxs = {128},
        .expectedPaddedCuKVSeqLens = {0, 128},
    });
}

// Normal prefill, multi-batch with different prompt lengths (the original bug scenario).
// runtimeSeqLen = max(inputSeqLen) = 128. Shorter prompt (64) is padded to 128.
TEST(UtilKernelTest, seqLens_multiBatchNormalPrefill)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {64, 128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 64, 192},
        .expectedCuKVSeqLens = {0, 64, 192},
        .expectedKvCacheEndIdxs = {128, 128},
        .expectedPaddedCuKVSeqLens = {0, 128, 256},
    });
}

// Multi-batch uniform lengths
TEST(UtilKernelTest, seqLens_multiBatchUniform)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128, 128, 128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128, 256, 384},
        .expectedCuKVSeqLens = {0, 128, 256, 384},
        .expectedKvCacheEndIdxs = {128, 128, 128},
        .expectedPaddedCuKVSeqLens = {0, 128, 256, 384},
    });
}

// Chunked prefill: kvCacheStartIndices > 0 for some batches
TEST(UtilKernelTest, seqLens_multiBatchChunkedPrefill)
{
    // batch 0: startIdx=0,  inputLen=128 → kvEnd=0+128=128,  actualKV=0+128=128
    // batch 1: startIdx=64, inputLen=128 → kvEnd=64+128=192, actualKV=64+128=192
    // batch 2: startIdx=32, inputLen=128 → kvEnd=32+128=160, actualKV=32+128=160
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128, 128, 128},
        .kvCacheStartIndices = {0, 64, 32},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128, 256, 384},
        .expectedCuKVSeqLens = {0, 128, 320, 480},
        .expectedKvCacheEndIdxs = {128, 192, 160},
        .expectedPaddedCuKVSeqLens = {0, 128, 320, 480},
    });
}

// Chunked prefill with varying input lengths
TEST(UtilKernelTest, seqLens_chunkedPrefillVaryingLengths)
{
    // batch 0: startIdx=100, inputLen=50  → kvEnd=100+50=150, actualKV=100+50=150
    // batch 1: startIdx=0,   inputLen=30  → kvEnd=0+50=50,    actualKV=0+30=30
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {50, 30},
        .kvCacheStartIndices = {100, 0},
        .runtimeSeqLen = 50,
        .expectedCuQSeqLens = {0, 50, 80},
        .expectedCuKVSeqLens = {0, 150, 180},
        .expectedKvCacheEndIdxs = {150, 50},
        .expectedPaddedCuKVSeqLens = {0, 150, 200},
    });
}

TEST(UtilKernelTest, raggedPagedSeqLensUseRealKVLengths)
{
    int32_t constexpr batchSize = 3;
    std::vector<int32_t> const inputSeqLen{3, 5, 2};
    std::vector<int32_t> const startIndices{0, 6, 4};
    rt::Tensor inputSeqLenTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor startIndicesTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(inputSeqLenTensor, inputSeqLen);
    copyHostToDevice(startIndicesTensor, startIndices);

    cudaStream_t stream{nullptr};
    kernel::calCuQCuKVSeqLens(inputSeqLenTensor, startIndicesTensor, cuQSeqLensTensor, cuKVSeqLensTensor, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(cuQSeqLensTensor), (std::vector<int32_t>{0, 3, 8, 10}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(cuKVSeqLensTensor), (std::vector<int32_t>{0, 3, 14, 20}));
}

TEST(UtilKernelTest, raggedPagedSeqLensSupportColdPrefillAndEmptyRows)
{
    int32_t constexpr batchSize = 4;
    std::vector<int32_t> const inputSeqLen{3, 0, 5, 2};
    rt::Tensor inputSeqLenTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(inputSeqLenTensor, inputSeqLen);

    cudaStream_t stream{nullptr};
    kernel::calCuQCuKVSeqLens(inputSeqLenTensor, rt::Tensor{}, cuQSeqLensTensor, cuKVSeqLensTensor, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> const expected{0, 3, 3, 8, 10};
    EXPECT_EQ(copyDeviceToHost<int32_t>(cuQSeqLensTensor), expected);
    EXPECT_EQ(copyDeviceToHost<int32_t>(cuKVSeqLensTensor), expected);
}

TEST(UtilKernelTest, swaChunkedPrefillMetadataClampsResidentWindow)
{
    int32_t constexpr batchSize = 3;
    int32_t constexpr runtimeSeqLen = 64;
    int32_t constexpr slidingWindowSize = 128;
    std::vector<int32_t> const inputSeqLen{64, 32, 16};
    std::vector<int32_t> const startIndices{0, 100, 300};

    rt::Tensor inputSeqLenTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor startIndicesTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor endIndicesTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor paddedCuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(inputSeqLenTensor, inputSeqLen);
    copyHostToDevice(startIndicesTensor, startIndices);

    cudaStream_t stream{nullptr};
    kernel::calSWAChunkedPrefillMetadata(inputSeqLenTensor, startIndicesTensor, cuQSeqLensTensor, cuKVSeqLensTensor,
        endIndicesTensor, paddedCuKVSeqLensTensor, runtimeSeqLen, slidingWindowSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(cuQSeqLensTensor), (std::vector<int32_t>{0, 64, 96, 112}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(cuKVSeqLensTensor), (std::vector<int32_t>{0, 64, 196, 340}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(endIndicesTensor), (std::vector<int32_t>{64, 164, 364}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(paddedCuKVSeqLensTensor), (std::vector<int32_t>{0, 64, 228, 420}));
}

TEST(UtilKernelTest, pagedSwaChunkedPrefillAssemblesSparseScrambledWindow)
{
    int32_t constexpr batchSize = 2;
    int32_t constexpr numKVHeads = 1;
    int32_t constexpr headDim = 2;
    int32_t constexpr slidingWindowSize = 150;
    int32_t constexpr qSeqLen = 4;
    int32_t constexpr workspaceSeqLen = slidingWindowSize + qSeqLen;
    int32_t constexpr maxPagesPerSeq = 4;
    int32_t constexpr numPages = 6;
    int32_t constexpr kPoolVBias = 512;
    int32_t constexpr kInputKBias = 1024;
    int32_t constexpr kInputVBias = 1536;
    int32_t constexpr kBatchStride = 128;
    int32_t constexpr kTokenStride = headDim;
    std::vector<int32_t> const inputSeqLen{4, 2};
    std::vector<int32_t> const startIndices{200, 20};

    std::vector<int32_t> pageTable(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq, rt::kUNUSED_PAGE_ENTRY);
    auto mapPage = [&](int32_t batch, int32_t logicalPage, int32_t physicalPage) {
        pageTable[(batch * 2) * maxPagesPerSeq + logicalPage] = physicalPage;
        pageTable[(batch * 2 + 1) * maxPagesPerSeq + logicalPage] = physicalPage + numPages;
    };
    mapPage(/*batch=*/0, /*logicalPage=*/0, /*physicalPage=*/3);
    mapPage(/*batch=*/0, /*logicalPage=*/1, /*physicalPage=*/1);
    mapPage(/*batch=*/1, /*logicalPage=*/0, /*physicalPage=*/4);

    size_t const pageElements = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * numKVHeads * headDim;
    std::vector<half> pool(static_cast<size_t>(2) * numPages * pageElements, __float2half(0.0F));
    auto fillLogicalPage = [&](int32_t logicalPage, int32_t physicalPage) {
        for (int32_t token = 0; token < rt::kTOKENS_PER_PAGE; ++token)
        {
            int32_t const logicalToken = logicalPage * rt::kTOKENS_PER_PAGE + token;
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                size_t const offset = static_cast<size_t>(token) * headDim + dim;
                pool[static_cast<size_t>(physicalPage) * pageElements + offset]
                    = __float2half(static_cast<float>(logicalToken * kTokenStride + dim));
                pool[static_cast<size_t>(physicalPage + numPages) * pageElements + offset]
                    = __float2half(static_cast<float>(kPoolVBias + logicalToken * kTokenStride + dim));
            }
        }
    };
    fillLogicalPage(/*logicalPage=*/0, /*physicalPage=*/3);
    fillLogicalPage(/*logicalPage=*/1, /*physicalPage=*/1);
    fillLogicalPage(/*logicalPage=*/0, /*physicalPage=*/4);

    std::vector<half> kInput(static_cast<size_t>(batchSize) * qSeqLen * numKVHeads * headDim);
    std::vector<half> vInput(kInput.size());
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t token = 0; token < qSeqLen; ++token)
        {
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                size_t const offset = (static_cast<size_t>(batch) * qSeqLen + token) * headDim + dim;
                kInput[offset]
                    = __float2half(static_cast<float>(kInputKBias + batch * kBatchStride + token * kTokenStride + dim));
                vInput[offset]
                    = __float2half(static_cast<float>(kInputVBias + batch * kBatchStride + token * kTokenStride + dim));
            }
        }
    }

    rt::Tensor poolTensor(
        {2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageTableTensor({batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kInputTensor({batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vInputTensor({batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor inputSeqLenTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor startIndicesTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kWorkspaceTensor(
        {batchSize, workspaceSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vWorkspaceTensor(
        {batchSize, workspaceSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(poolTensor, pool);
    copyHostToDevice(pageTableTensor, pageTable);
    copyHostToDevice(kInputTensor, kInput);
    copyHostToDevice(vInputTensor, vInput);
    copyHostToDevice(inputSeqLenTensor, inputSeqLen);
    copyHostToDevice(startIndicesTensor, startIndices);

    cudaStream_t stream{nullptr};
    kernel::assemblePagedSWAChunkedPrefillFMHAKV(poolTensor, pageTableTensor, kInputTensor, vInputTensor,
        inputSeqLenTensor, startIndicesTensor, kWorkspaceTensor, vWorkspaceTensor, slidingWindowSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const kOutput = copyDeviceToHost<half>(kWorkspaceTensor);
    auto const vOutput = copyDeviceToHost<half>(vWorkspaceTensor);

    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        int32_t const oldResidentLen = std::min(startIndices[batch], slidingWindowSize);
        for (int32_t workspaceToken = 0; workspaceToken < workspaceSeqLen; ++workspaceToken)
        {
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                size_t const outputOffset
                    = (static_cast<size_t>(batch) * workspaceSeqLen + workspaceToken) * headDim + dim;
                float expectedK = 0.0F;
                float expectedV = 0.0F;
                if (workspaceToken < oldResidentLen)
                {
                    int32_t const logicalToken = startIndices[batch] - oldResidentLen + workspaceToken;
                    expectedK = static_cast<float>(logicalToken * kTokenStride + dim);
                    expectedV = static_cast<float>(kPoolVBias + logicalToken * kTokenStride + dim);
                }
                else if (workspaceToken - oldResidentLen < inputSeqLen[batch])
                {
                    int32_t const newToken = workspaceToken - oldResidentLen;
                    expectedK = static_cast<float>(kInputKBias + batch * kBatchStride + newToken * kTokenStride + dim);
                    expectedV = static_cast<float>(kInputVBias + batch * kBatchStride + newToken * kTokenStride + dim);
                }
                EXPECT_FLOAT_EQ(__half2float(kOutput[outputOffset]), expectedK);
                EXPECT_FLOAT_EQ(__half2float(vOutput[outputOffset]), expectedV);
            }
        }
    }
}

// launchBuildVisionBlockRanges: image-run intervals expand per position; text and
// padding rows get the -1/-1 sentinel (empty interval).
TEST(UtilKernelTest, visionBlockRanges_runsSentinelsAndPadding)
{
    int32_t constexpr batchSize = 2;
    int32_t constexpr seqLen = 12;
    // Batch 0: block 0 = [1, 5], block 1 = [7, 9]; full context.
    // Batch 1: block 0 = [0, 1]; block 1 = [4, 8] but contextLength = 7 clips
    //          the run to [4, 6]; positions >= 7 are padding.
    std::vector<int32_t> const blockIds{-1, 0, 0, 0, 0, 0, -1, 1, 1, 1, -1, -1, //
        0, 0, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1};
    std::vector<int32_t> const contextLengths{seqLen, 7};
    std::vector<int32_t> const expectedBegin{-1, 1, 1, 1, 1, 1, -1, 7, 7, 7, -1, -1, //
        0, 0, -1, -1, 4, 4, 4, -1, -1, -1, -1, -1};
    std::vector<int32_t> const expectedEnd{-1, 5, 5, 5, 5, 5, -1, 9, 9, 9, -1, -1, //
        1, 1, -1, -1, 6, 6, 6, -1, -1, -1, -1, -1};

    rt::Tensor idsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor beginTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor endTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(idsTensor, blockIds);
    copyHostToDevice(lengthsTensor, contextLengths);

    cudaStream_t stream{nullptr};
    kernel::launchBuildVisionBlockRanges(idsTensor.dataPointer<int32_t>(), lengthsTensor.dataPointer<int32_t>(),
        beginTensor.dataPointer<int32_t>(), endTensor.dataPointer<int32_t>(), batchSize, seqLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const begin = copyDeviceToHost<int32_t>(beginTensor);
    auto const end = copyDeviceToHost<int32_t>(endTensor);
    for (size_t i = 0; i < expectedBegin.size(); ++i)
    {
        EXPECT_EQ(begin[i], expectedBegin[i]) << "blockBegin mismatch at flat index " << i;
        EXPECT_EQ(end[i], expectedEnd[i]) << "blockEnd mismatch at flat index " << i;
    }
}

TEST(UtilKernelTest, gatherTokenAlignedRope_usesResidentSlotsAndZerosPadding)
{
    int32_t constexpr sourceSlots = 3;
    int32_t constexpr cacheCapacity = 4;
    int32_t constexpr rotaryDim = 2;
    int32_t constexpr numSequences = 2;
    int32_t constexpr numTokens = 4;

    std::vector<float> source(static_cast<size_t>(sourceSlots * cacheCapacity * rotaryDim));
    std::iota(source.begin(), source.end(), 0.0F);
    std::vector<int32_t> const positions{1, 3, -1, 0};
    std::vector<int32_t> const queryStartOffsets{0, 1, 4};
    std::vector<int32_t> const queryLengths{1, 1};
    std::vector<int32_t> const stateIndices{2, 0};
    std::vector<float> const expected{18.0F, 19.0F, 6.0F, 7.0F, 0.0F, 0.0F, 0.0F, 0.0F};

    rt::Tensor sourceTensor({sourceSlots, cacheCapacity, rotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor positionsTensor({numTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor offsetsTensor({numSequences + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengthsTensor({numSequences}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor statesTensor({numSequences}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputTensor({numTokens, rotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice(sourceTensor, source);
    copyHostToDevice(positionsTensor, positions);
    copyHostToDevice(offsetsTensor, queryStartOffsets);
    copyHostToDevice(lengthsTensor, queryLengths);
    copyHostToDevice(statesTensor, stateIndices);
    copyHostToDevice(outputTensor, std::vector<float>(expected.size(), -1.0F));

    cudaStream_t stream{nullptr};
    kernel::launchGatherTokenAlignedRope(sourceTensor.dataPointer<float>(), outputTensor.dataPointer<float>(),
        positionsTensor.dataPointer<int32_t>(), offsetsTensor.dataPointer<int32_t>(),
        lengthsTensor.dataPointer<int32_t>(), statesTensor.dataPointer<int32_t>(), numTokens, numSequences, sourceSlots,
        cacheCapacity, rotaryDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<float>(outputTensor), expected);
}
