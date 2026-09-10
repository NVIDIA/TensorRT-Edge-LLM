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
    rt::Tensor deltaStarts({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deltaLengths({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
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
    copyHostToDevice<int32_t>(deltaStarts, {kFirstStart, kSecondStart});
    copyHostToDevice<int32_t>(deltaLengths, {kDeltaLen, kDeltaLen});
    copyHostToDevice<int32_t>(pageTable, {2, 1, 6, 5, 3, 0, 7, 4});

    kernel::launchDFlashTargetKVCacheUpdate(kDelta.dataPointer<half>(), vDelta.dataPointer<half>(),
        kvPool.dataPointer<half>(), ropeCosSin.dataPointer<float>(), deltaStarts.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), pageTable.dataPointer<int32_t>(), kBatchSize, kDeltaLen, kNumKVHeads,
        kHeadDim, 0, 1, 1, kNumPages, kMaxPagesPerSeq, stream);
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
    rt::Tensor deltaStarts({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deltaLengths({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pageTable({kBatchSize, 2, kMaxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(delta, std::vector<half>(kHeadDim, __float2half(1.F)));
    copyHostToDevice(kvPool, std::vector<half>(poolElements, __float2half(-1.F)));
    copyHostToDevice<int32_t>(deltaStarts, {0});
    copyHostToDevice<int32_t>(deltaLengths, {kDeltaLen});
    copyHostToDevice<int32_t>(pageTable, {/* K page */ 1, /* independently mapped V page */ kNumPages});

    kernel::launchDFlashTargetKVCacheUpdate(delta.dataPointer<half>(), delta.dataPointer<half>(),
        kvPool.dataPointer<half>(), ropeCosSin.dataPointer<float>(), deltaStarts.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), pageTable.dataPointer<int32_t>(), kBatchSize, kDeltaLen, kNumKVHeads,
        kHeadDim, 0, 1, 1, kNumPages, kMaxPagesPerSeq, stream);
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

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsExactlyPageAlignedCapacity)
{
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4096, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityRejectsSeqLenExceedingCap)
{
    // A rope cache sized past the KV pool's padded capacity indicates a genuine mismatch.
    EXPECT_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/5000, /*kvCapacity=*/4096), std::runtime_error);
}

TEST(DFlashRuntimeKernels, PrepareProposalInputsSupportsCausalMask)
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

    copyHostToDevice<int32_t>(oldDraftCacheLengths, {10, 20});
    copyHostToDevice<int32_t>(deltaLengths, {2, 3});

    kernel::launchDFlashPrepareProposalInputs(oldDraftCacheLengths.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), blockSize, packedAttentionMask.dataPointer<int32_t>(),
        attentionPosId.dataPointer<int32_t>(), contextLengths.dataPointer<int32_t>(), false, batchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(
        copyDeviceToHost<int32_t>(packedAttentionMask), (std::vector<int32_t>{31, 31, 31, 31, 31, 31, 31, 31, 31, 31}));
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(attentionPosId), (std::vector<int32_t>{12, 13, 14, 15, 16, 23, 24, 25, 26, 27}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(contextLengths), (std::vector<int32_t>{17, 28}));

    kernel::launchDFlashPrepareProposalInputs(oldDraftCacheLengths.dataPointer<int32_t>(),
        deltaLengths.dataPointer<int32_t>(), blockSize, packedAttentionMask.dataPointer<int32_t>(),
        attentionPosId.dataPointer<int32_t>(), contextLengths.dataPointer<int32_t>(), true, batchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(packedAttentionMask), (std::vector<int32_t>{1, 3, 7, 15, 31, 1, 3, 7, 15, 31}));
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(attentionPosId), (std::vector<int32_t>{12, 13, 14, 15, 16, 23, 24, 25, 26, 27}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(contextLengths), (std::vector<int32_t>{17, 28}));
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
