/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "kernels/mamba/ssdVarlenMetadata.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using trt_edgellm::mamba::buildSSDPackedVarlenMetadata;
using trt_edgellm::mamba::fillValidLensFromCuSeqlens;
using trt_edgellm::mamba::ssdPackedLogicalChunkUpperBound;

constexpr int32_t kGARBAGE{0x5a5a5a5a};

//! Device copy of a host vector; buffers longer than the data are pre-filled with kGARBAGE.
class DeviceInts
{
public:
    explicit DeviceInts(std::vector<int32_t> const& values, size_t capacity = 0)
        : mSize(capacity > values.size() ? capacity : values.size())
    {
        std::vector<int32_t> host(mSize, kGARBAGE);
        std::copy(values.begin(), values.end(), host.begin());
        CUDA_CHECK(cudaMalloc(&mData, mSize * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(mData, host.data(), mSize * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    DeviceInts(DeviceInts const&) = delete;
    DeviceInts& operator=(DeviceInts const&) = delete;
    ~DeviceInts()
    {
        cudaFree(mData);
    }
    int32_t* data() const
    {
        return mData;
    }
    std::vector<int32_t> read() const
    {
        std::vector<int32_t> host(mSize);
        CUDA_CHECK(cudaMemcpy(host.data(), mData, mSize * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    int32_t* mData{nullptr};
    size_t mSize{};
};

struct PackedMetadata
{
    std::vector<int32_t> seqIdx;
    std::vector<int32_t> chunkIndices;
    std::vector<int32_t> chunkOffsets;
    std::vector<int32_t> seqChunkCumsum;
    std::vector<int32_t> validLens;
};

//! Builds the packed metadata into buffers with `guard` spare slots after each array.
PackedMetadata buildPacked(
    std::vector<int32_t> const& cuSeqlens, int32_t totalTokens, int32_t chunkSize, int32_t totalSlots, size_t guard)
{
    int32_t const numSeqs = static_cast<int32_t>(cuSeqlens.size()) - 1;
    DeviceInts const cu(cuSeqlens);
    DeviceInts const seqIdx({}, totalTokens + guard);
    DeviceInts const chunkIndices({}, totalSlots + guard);
    DeviceInts const chunkOffsets({}, totalSlots + guard);
    DeviceInts const seqChunkCumsum({}, numSeqs + 1 + guard);
    DeviceInts const validLens({}, numSeqs + guard);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    buildSSDPackedVarlenMetadata(seqIdx.data(), chunkIndices.data(), chunkOffsets.data(), seqChunkCumsum.data(),
        cu.data(), numSeqs, totalTokens, chunkSize, totalSlots, stream);
    fillValidLensFromCuSeqlens(validLens.data(), cu.data(), numSeqs, totalTokens, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return {seqIdx.read(), chunkIndices.read(), chunkOffsets.read(), seqChunkCumsum.read(), validLens.read()};
}

std::vector<int32_t> head(std::vector<int32_t> const& values, size_t count)
{
    return {values.begin(), values.begin() + static_cast<int64_t>(count)};
}

std::vector<int32_t> tail(std::vector<int32_t> const& values, size_t from)
{
    return {values.begin() + static_cast<int64_t>(from), values.end()};
}

TEST(SsdPackedVarlenMetadata, SplitsSequencesIntoChunkPiecesAndPadsWithSentinels)
{
    // Given: chunk size 4 and six packed sequences on a 10-token axis: lengths 3 and 1 sharing chunk 0, length 5
    //        spanning chunks 1 and 2, length 1 inside chunk 2, then one sequence ending past the axis and one
    //        starting past it.
    // When:  the packed metadata and valid lengths are built with the logical-chunk upper bound as slot count.
    // Then:  every token maps to its sequence, each sequence's chunk pieces list (chunk, offset) in order with the
    //        cumulative piece counts, the remaining slots hold the -1 sentinel with offset 0, the sequences past the
    //        axis have no piece and length 0, and nothing past the arrays is written.
    std::vector<int32_t> const cuSeqlens{0, 3, 4, 9, 10, 12, 14};
    int32_t const totalTokens = 10;
    int32_t const totalSlots = ssdPackedLogicalChunkUpperBound(6, totalTokens, 4);
    ASSERT_EQ(totalSlots, 9);
    size_t const guard = 4;

    PackedMetadata const m = buildPacked(cuSeqlens, totalTokens, 4, totalSlots, guard);

    EXPECT_EQ(head(m.seqIdx, totalTokens), (std::vector<int32_t>{0, 0, 0, 1, 2, 2, 2, 2, 2, 3}));
    EXPECT_EQ(head(m.chunkIndices, totalSlots), (std::vector<int32_t>{0, 0, 1, 2, 2, -1, -1, -1, -1}));
    EXPECT_EQ(head(m.chunkOffsets, totalSlots), (std::vector<int32_t>{0, 3, 0, 0, 1, 0, 0, 0, 0}));
    EXPECT_EQ(head(m.seqChunkCumsum, 7), (std::vector<int32_t>{0, 1, 2, 4, 5, 5, 5}));
    EXPECT_EQ(head(m.validLens, 6), (std::vector<int32_t>{3, 1, 5, 1, 0, 0}));
    std::vector<int32_t> const untouched(guard, kGARBAGE);
    EXPECT_EQ(tail(m.seqIdx, totalTokens), untouched);
    EXPECT_EQ(tail(m.chunkIndices, totalSlots), untouched);
    EXPECT_EQ(tail(m.chunkOffsets, totalSlots), untouched);
    EXPECT_EQ(tail(m.seqChunkCumsum, 7), untouched);
    EXPECT_EQ(tail(m.validLens, 6), untouched);
}

TEST(SsdPackedVarlenMetadata, TruncatesPiecesToTheSlotCount)
{
    // Given: the same packed sequences, but only 4 slots for their 5 chunk pieces.
    // When:  the packed metadata is built.
    // Then:  the first 4 pieces are written in order, the cumulative counts stop at 4, and no slot past the 4th is
    //        written.
    std::vector<int32_t> const cuSeqlens{0, 3, 4, 9, 10, 14};
    size_t const guard = 4;

    PackedMetadata const m = buildPacked(cuSeqlens, 10, 4, 4, guard);

    EXPECT_EQ(head(m.chunkIndices, 4), (std::vector<int32_t>{0, 0, 1, 2}));
    EXPECT_EQ(head(m.chunkOffsets, 4), (std::vector<int32_t>{0, 3, 0, 0}));
    EXPECT_EQ(head(m.seqChunkCumsum, 6), (std::vector<int32_t>{0, 1, 2, 4, 4, 4}));
    std::vector<int32_t> const untouched(guard, kGARBAGE);
    EXPECT_EQ(tail(m.chunkIndices, 4), untouched);
    EXPECT_EQ(tail(m.chunkOffsets, 4), untouched);
    EXPECT_EQ(tail(m.seqChunkCumsum, 6), untouched);
}

} // namespace
