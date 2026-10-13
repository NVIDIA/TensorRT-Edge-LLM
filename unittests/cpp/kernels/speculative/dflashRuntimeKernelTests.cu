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
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

TEST(DFlashRuntimeKernels, TargetKVCacheUpdateRoutesNonIdentityPages)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t kBatchSize{2};
    constexpr int32_t kDeltaLen{1};
    constexpr int32_t kNumKVHeads{1};
    constexpr int32_t kHeadDim{8};
    constexpr int32_t kNumPages{4};
    constexpr int32_t kMaxPagesPerSeq{2};
    constexpr int32_t kFirstStart{3};
    constexpr int32_t kSecondStart{130};
    size_t const pageElements = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * kNumKVHeads * kHeadDim;

    rt::Tensor kDelta({kBatchSize, kDeltaLen, kNumKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDelta({kBatchSize, kDeltaLen, kNumKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvPool(
        {2, kNumPages, rt::kTOKENS_PER_PAGE, kNumKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor ropeCosSin({1, 1, 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor deltaPositions({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deltaTokenToSequence({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pageTable({kBatchSize, 2, kMaxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<half> const kHost = {__float2half(1.F), __float2half(2.F), __float2half(3.F), __float2half(4.F),
        __float2half(5.F), __float2half(6.F), __float2half(7.F), __float2half(8.F), __float2half(11.F),
        __float2half(12.F), __float2half(13.F), __float2half(14.F), __float2half(15.F), __float2half(16.F),
        __float2half(17.F), __float2half(18.F)};
    std::vector<half> const vHost = {__float2half(21.F), __float2half(22.F), __float2half(23.F), __float2half(24.F),
        __float2half(25.F), __float2half(26.F), __float2half(27.F), __float2half(28.F), __float2half(31.F),
        __float2half(32.F), __float2half(33.F), __float2half(34.F), __float2half(35.F), __float2half(36.F),
        __float2half(37.F), __float2half(38.F)};
    copyHostToDevice(kDelta, kHost);
    copyHostToDevice(vDelta, vHost);
    copyHostToDevice(kvPool, std::vector<half>(2 * kNumPages * pageElements, __float2half(-1.F)));
    copyHostToDevice<int32_t>(deltaPositions, {kFirstStart, kSecondStart});
    copyHostToDevice<int32_t>(deltaTokenToSequence, {0, 1});
    copyHostToDevice<int32_t>(pageTable, {2, 1, 6, 5, 3, 0, 7, 4});

    kernel::launchDFlashTargetKVCacheUpdate(kDelta.dataPointer<half>(), vDelta.dataPointer<half>(),
        kvPool.dataPointer<half>(), ropeCosSin.dataPointer<float>(), deltaPositions.dataPointer<int32_t>(),
        deltaTokenToSequence.dataPointer<int32_t>(), pageTable.dataPointer<int32_t>(), kBatchSize * kDeltaLen,
        kBatchSize, kNumKVHeads, kHeadDim, 0, kNumPages, kMaxPagesPerSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> const poolHost = copyDeviceToHost<half>(kvPool);
    auto expectToken = [&](int32_t flattenedPage, int32_t tokenOffset, std::vector<half> const& expected) {
        size_t const offset = static_cast<size_t>(flattenedPage) * pageElements
            + static_cast<size_t>(tokenOffset) * kNumKVHeads * kHeadDim;
        for (int32_t dim = 0; dim < kHeadDim; ++dim)
        {
            EXPECT_EQ(__half2float(poolHost[offset + dim]), __half2float(expected[dim]));
        }
    };
    expectToken(/* K page */ 2, kFirstStart, std::vector<half>(kHost.begin(), kHost.begin() + kHeadDim));
    expectToken(/* V page */ 6, kFirstStart, std::vector<half>(vHost.begin(), vHost.begin() + kHeadDim));
    expectToken(
        /* K page */ 0, kSecondStart % rt::kTOKENS_PER_PAGE, std::vector<half>(kHost.begin() + kHeadDim, kHost.end()));
    expectToken(
        /* V page */ 4, kSecondStart % rt::kTOKENS_PER_PAGE, std::vector<half>(vHost.begin() + kHeadDim, vHost.end()));
}

TEST(DFlashRuntimeKernels, TargetKVCacheUpdateRoutesIndependentKAndVPages)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDeltaLen{1};
    constexpr int32_t kNumKVHeads{1};
    constexpr int32_t kHeadDim{8};
    constexpr int32_t kNumPages{2};
    constexpr int32_t kMaxPagesPerSeq{1};
    size_t const poolElements = static_cast<size_t>(2 * kNumPages * rt::kTOKENS_PER_PAGE * kNumKVHeads * kHeadDim);

    rt::Tensor delta({kBatchSize, kDeltaLen, kNumKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvPool(
        {2, kNumPages, rt::kTOKENS_PER_PAGE, kNumKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor ropeCosSin({1, 1, 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor deltaPositions({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deltaTokenToSequence({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pageTable({kBatchSize, 2, kMaxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(delta, std::vector<half>(kHeadDim, __float2half(1.F)));
    copyHostToDevice(kvPool, std::vector<half>(poolElements, __float2half(-1.F)));
    copyHostToDevice<int32_t>(deltaPositions, {0});
    copyHostToDevice<int32_t>(deltaTokenToSequence, {0});
    copyHostToDevice<int32_t>(pageTable, {/* K page */ 1, /* independently mapped V page */ kNumPages});

    kernel::launchDFlashTargetKVCacheUpdate(delta.dataPointer<half>(), delta.dataPointer<half>(),
        kvPool.dataPointer<half>(), ropeCosSin.dataPointer<float>(), deltaPositions.dataPointer<int32_t>(),
        deltaTokenToSequence.dataPointer<int32_t>(), pageTable.dataPointer<int32_t>(), kBatchSize * kDeltaLen,
        kBatchSize, kNumKVHeads, kHeadDim, 0, kNumPages, kMaxPagesPerSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> const poolHost = copyDeviceToHost<half>(kvPool);
    auto expectToken = [&](int32_t flattenedPage) {
        size_t const offset = static_cast<size_t>(flattenedPage) * rt::kTOKENS_PER_PAGE * kNumKVHeads * kHeadDim;
        for (int32_t dim = 0; dim < kHeadDim; ++dim)
        {
            EXPECT_EQ(__half2float(poolHost[offset + dim]), 1.F);
        }
    };
    expectToken(/* K page */ 1);
    expectToken(/* V page */ kNumPages);

    for (int32_t flattenedPage : {0, 3})
    {
        size_t const offset = static_cast<size_t>(flattenedPage) * rt::kTOKENS_PER_PAGE * kNumKVHeads * kHeadDim;
        for (int32_t dim = 0; dim < kHeadDim; ++dim)
        {
            EXPECT_EQ(__half2float(poolHost[offset + dim]), -1.F);
        }
    }
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsNonPageAlignedCapacity)
{
    // maxKVCacheCapacity=4000 (not a multiple of 128) -> capPadded=4096. This must not throw.
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4000, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, DeltaRopeGatherUsesDeltaOwnerAndResidentStateRows)
{
    cudaStream_t stream{};
    constexpr int32_t kBatchSize{2};
    constexpr int32_t kSourceRows{3};
    constexpr int32_t kCapacity{4};
    constexpr int32_t kRotaryDim{2};
    constexpr int32_t kDeltaTokens{4};
    rt::Tensor source({kSourceRows, kCapacity, kRotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor output({kDeltaTokens, kRotaryDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor positions({kDeltaTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor owners({kDeltaTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor states({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    std::vector<float> sourceHost(static_cast<size_t>(kSourceRows * kCapacity * kRotaryDim));
    for (int32_t row = 0; row < kSourceRows; ++row)
    {
        for (int32_t position = 0; position < kCapacity; ++position)
        {
            for (int32_t channel = 0; channel < kRotaryDim; ++channel)
            {
                sourceHost[static_cast<size_t>((row * kCapacity + position) * kRotaryDim + channel)]
                    = static_cast<float>(row * 100 + position * 10 + channel);
            }
        }
    }
    copyHostToDevice(source, sourceHost);
    copyHostToDevice<int32_t>(positions, {1, -1, 2, 3});
    copyHostToDevice<int32_t>(owners, {0, -1, 1, 1});
    copyHostToDevice<int32_t>(states, {2, 0});

    kernel::launchDFlashGatherDeltaRope(source.dataPointer<float>(), output.dataPointer<float>(),
        positions.dataPointer<int32_t>(), owners.dataPointer<int32_t>(), states.dataPointer<int32_t>(), kDeltaTokens,
        kBatchSize, kSourceRows, kCapacity, kRotaryDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<float>(output), (std::vector<float>{210.F, 211.F, 0.F, 0.F, 20.F, 21.F, 30.F, 31.F}));
}

TEST(DFlashRuntimeKernels, PrepareSpecRaggedMetadataMarksPartialTreeTailAsPadding)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    constexpr int32_t kBatchSize{2};
    constexpr int32_t kQueryWidth{8};
    constexpr int32_t kPhysicalTokens{kBatchSize * kQueryWidth};

    auto attentionPositions = rt::Tensor({kPhysicalTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    auto committedPastLengths = rt::Tensor({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto validCounts = rt::Tensor({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto positions = rt::Tensor({kPhysicalTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    auto queryStartOffsets = rt::Tensor({kBatchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    auto queryLengths = rt::Tensor({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto pastLengths = rt::Tensor({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto attentionSequenceLengths = rt::Tensor({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto treeParentIds = rt::Tensor({kPhysicalTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    auto treeDepths = rt::Tensor({kPhysicalTokens}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<int32_t>(attentionPositions, {5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19});
    copyHostToDevice<int32_t>(committedPastLengths, {5, 12});
    copyHostToDevice<int32_t>(validCounts, {7, 3});

    kernel::launchPrepareSpecRaggedMetadata(attentionPositions.dataPointer<int32_t>(),
        committedPastLengths.dataPointer<int32_t>(), validCounts.dataPointer<int32_t>(), kQueryWidth,
        positions.dataPointer<int32_t>(), queryStartOffsets.dataPointer<int32_t>(), queryLengths.dataPointer<int32_t>(),
        pastLengths.dataPointer<int32_t>(), attentionSequenceLengths.dataPointer<int32_t>(),
        treeParentIds.dataPointer<int32_t>(), treeDepths.dataPointer<int32_t>(),
        /*synthesizeLinearTree=*/true, kBatchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(queryStartOffsets), (std::vector<int32_t>{0, 8, 16}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(queryLengths), (std::vector<int32_t>{7, 3}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(pastLengths), (std::vector<int32_t>{5, 12}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(attentionSequenceLengths), (std::vector<int32_t>{13, 20}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(positions),
        (std::vector<int32_t>{5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, -1, -1, -1, -1, -1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(treeParentIds),
        (std::vector<int32_t>{-1, 0, 1, 2, 3, 4, 5, -1, -1, 0, 1, -1, -1, -1, -1, -1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(treeDepths),
        (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, -1, 0, 1, 2, -1, -1, -1, -1, -1}));

    copyHostToDevice<int32_t>(validCounts, {1, 8});

    kernel::launchPrepareSpecRaggedMetadata(attentionPositions.dataPointer<int32_t>(),
        committedPastLengths.dataPointer<int32_t>(), validCounts.dataPointer<int32_t>(), kQueryWidth,
        positions.dataPointer<int32_t>(), queryStartOffsets.dataPointer<int32_t>(), queryLengths.dataPointer<int32_t>(),
        pastLengths.dataPointer<int32_t>(), attentionSequenceLengths.dataPointer<int32_t>(),
        treeParentIds.dataPointer<int32_t>(), treeDepths.dataPointer<int32_t>(),
        /*synthesizeLinearTree=*/true, kBatchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(queryStartOffsets), (std::vector<int32_t>{0, 8, 16}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(queryLengths), (std::vector<int32_t>{1, 8}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(pastLengths), (std::vector<int32_t>{5, 12}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(attentionSequenceLengths), (std::vector<int32_t>{13, 20}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(positions),
        (std::vector<int32_t>{5, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, 16, 17, 18, 19}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(treeParentIds),
        (std::vector<int32_t>{-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(treeDepths),
        (std::vector<int32_t>{0, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsExactlyPageAlignedCapacity)
{
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4096, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityRejectsSeqLenExceedingCap)
{
    // A rope cache sized past the KV pool's padded capacity indicates a genuine mismatch.
    EXPECT_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/5000, /*kvCapacity=*/4096), std::runtime_error);
}

TEST(DFlashRuntimeKernels, PrepareProposalInputsPreservesResidentMappingAndSupportsCausalMask)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t blockSize = 5;
    constexpr int32_t packedMaskLen = 1;

    auto oldDraftCacheLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto deltaLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto packedAttentionMask
        = rt::Tensor({batchSize, blockSize, packedMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto attentionPosId = rt::Tensor({batchSize, blockSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto contextLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto positions = rt::Tensor({batchSize, blockSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto queryStartOffsets = rt::Tensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    auto queryLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto pastLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto attentionSequenceLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto stateIndices = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<int32_t>(oldDraftCacheLengths, {10, 20});
    copyHostToDevice<int32_t>(deltaLengths, {2, 3});
    copyHostToDevice<int32_t>(stateIndices, {2, 0});

    kernel::launchDFlashPrepareProposalInputs(oldDraftCacheLengths.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), blockSize, packedAttentionMask.dataPointer<int32_t>(),
        attentionPosId.dataPointer<int32_t>(), contextLengths.dataPointer<int32_t>(), positions.dataPointer<int32_t>(),
        queryStartOffsets.dataPointer<int32_t>(), queryLengths.dataPointer<int32_t>(),
        pastLengths.dataPointer<int32_t>(), attentionSequenceLengths.dataPointer<int32_t>(),
        stateIndices.dataPointer<int32_t>(), false, batchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(
        copyDeviceToHost<int32_t>(packedAttentionMask), (std::vector<int32_t>{31, 31, 31, 31, 31, 31, 31, 31, 31, 31}));
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(attentionPosId), (std::vector<int32_t>{12, 13, 14, 15, 16, 23, 24, 25, 26, 27}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(contextLengths), (std::vector<int32_t>{17, 28}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(stateIndices), (std::vector<int32_t>{2, 0}));

    kernel::launchDFlashPrepareProposalInputs(oldDraftCacheLengths.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), blockSize, packedAttentionMask.dataPointer<int32_t>(),
        attentionPosId.dataPointer<int32_t>(), contextLengths.dataPointer<int32_t>(), positions.dataPointer<int32_t>(),
        queryStartOffsets.dataPointer<int32_t>(), queryLengths.dataPointer<int32_t>(),
        pastLengths.dataPointer<int32_t>(), attentionSequenceLengths.dataPointer<int32_t>(),
        stateIndices.dataPointer<int32_t>(), true, batchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(packedAttentionMask), (std::vector<int32_t>{1, 3, 7, 15, 31, 1, 3, 7, 15, 31}));
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(attentionPosId), (std::vector<int32_t>{12, 13, 14, 15, 16, 23, 24, 25, 26, 27}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(contextLengths), (std::vector<int32_t>{17, 28}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(stateIndices), (std::vector<int32_t>{2, 0}));
}

TEST(DFlashRuntimeKernels, BuildLinearVerifyInputsUsesDraftStrideForBatchRows)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t dflashBlockSize = 4;
    constexpr int32_t proposalLen = dflashBlockSize - 1;
    constexpr int32_t verifySize = proposalLen + 1;

    auto lastAcceptedTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, dflashBlockSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verifyTokenIds = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verifyTreeMask = rt::Tensor({batchSize, verifySize, verifySize}, rt::DeviceType::kGPU, DataType::kINT8);

    copyHostToDevice<int32_t>(lastAcceptedTokens, {10, 20});
    // DFlash draft output at position 0 predicts the current token (t_last), not the next token.
    // Real draft proposals start at position 1 — consistent with DDTree which skips depthIdx==0.
    // Layout per batch row: [<pos0: unused t_last prediction>, pos1, pos2, pos3]
    copyHostToDevice<int32_t>(draftTokenIds, {999, 101, 102, 103, 999, 201, 202, 203});

    kernel::launchDFlashBuildLinearVerifyInputs(lastAcceptedTokens.dataPointer<int32_t>(),
        draftTokenIds.dataPointer<int32_t>(), verifyTokenIds.dataPointer<int32_t>(),
        verifyTreeMask.dataPointer<int8_t>(), batchSize, proposalLen, dflashBlockSize, verifySize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(verifyTokenIds), (std::vector<int32_t>{10, 101, 102, 103, 20, 201, 202, 203}));

    std::vector<int8_t> expectedMask;
    expectedMask.reserve(static_cast<size_t>(batchSize) * verifySize * verifySize);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t rowIdx = 0; rowIdx < verifySize; ++rowIdx)
        {
            for (int32_t colIdx = 0; colIdx < verifySize; ++colIdx)
            {
                expectedMask.push_back(colIdx <= rowIdx ? int8_t{1} : int8_t{0});
            }
        }
    }
    EXPECT_EQ(copyDeviceToHost<int8_t>(verifyTreeMask), expectedMask);
}

TEST(DFlashRuntimeKernels, BuildLinearTreeMetadataUsesChainParentsAndDepths)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t verifySize = 4;
    auto parentIds = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto depths = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);

    kernel::launchDFlashBuildLinearTreeMetadata(
        parentIds.dataPointer<int32_t>(), depths.dataPointer<int32_t>(), batchSize, verifySize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> const expectedParents{-1, 0, 1, 2, -1, 0, 1, 2};
    std::vector<int32_t> const expectedDepths{0, 1, 2, 3, 0, 1, 2, 3};
    EXPECT_EQ(copyDeviceToHost<int32_t>(parentIds), expectedParents);
    EXPECT_EQ(copyDeviceToHost<int32_t>(depths), expectedDepths);
}
