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

//! Tests for the Qwen3-Omni audio encoder's chunk-window partition, which the dense attention mask
//! and the ragged-attention cu_seqlens descriptor must both encode identically.

#include "common/checkMacros.h"
#include "multimodal/common/audioUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace trt_edgellm;
namespace au = trt_edgellm::rt::audioUtils;

namespace
{

//! Shipped geometry: 100-frame chunks give 13 tokens after the CNN, and n_window_infer = 800
//! groups 8 chunks per window, so a window holds 104 tokens.
constexpr int32_t kNWindow = 50;
constexpr int32_t kNWindowInfer = 800;
constexpr int64_t kTokensPerChunk = 13;
constexpr int64_t kChunksPerWindow = kNWindowInfer / (kNWindow * 2);

std::vector<int64_t> clip(int64_t fullChunks, int64_t tailTokens)
{
    std::vector<int64_t> lens(static_cast<size_t>(fullChunks), kTokensPerChunk);
    if (tailTokens > 0)
    {
        lens.push_back(tailTokens);
    }
    return lens;
}

int64_t sum(std::vector<int64_t> const& values)
{
    return std::accumulate(values.begin(), values.end(), int64_t{0});
}

} // namespace

//! Pins the partition for a clip with a short tail (T = 358), a clip that fills its last window
//! exactly, and a clip shorter than one window.
TEST(ChunkwiseWindowPartitionTest, MatchesReferencePartitions)
{
    EXPECT_EQ(au::computeChunkwiseWindowLens(clip(27, 7), kNWindow, kNWindowInfer),
        (std::vector<int64_t>{104, 104, 104, 46}));
    EXPECT_EQ(au::computeChunkwiseWindowLens(clip(64, 0), kNWindow, kNWindowInfer), std::vector<int64_t>(8, 104));
    EXPECT_EQ(au::computeChunkwiseWindowLens({13, 13, 5}, kNWindow, kNWindowInfer), (std::vector<int64_t>{31}));
}

//! The window width follows the longest chunk of the clip, as in the reference model, not a fixed
//! 13 tokens: a clip made only of short chunks gets proportionally narrower windows.
TEST(ChunkwiseWindowPartitionTest, WidthFollowsTheLongestChunk)
{
    std::vector<int64_t> const shortChunks(10, 5);
    EXPECT_EQ(au::computeChunkwiseWindowLens(shortChunks, kNWindow, kNWindowInfer), (std::vector<int64_t>{40, 10}));
}

//! Degenerate inputs yield no windows instead of dividing by zero; the runtime then rejects the
//! clip because the partition does not cover the encoder tokens.
TEST(ChunkwiseWindowPartitionTest, DegenerateInputsYieldNoWindows)
{
    EXPECT_TRUE(au::computeChunkwiseWindowLens({}, kNWindow, kNWindowInfer).empty());
    EXPECT_TRUE(au::computeChunkwiseWindowLens(clip(4, 0), kNWindow, /*nWindowInfer=*/kNWindow).empty());
    EXPECT_TRUE(au::computeChunkwiseWindowLens(clip(4, 0), /*nWindow=*/0, kNWindowInfer).empty());
}

//! Every clip up to the default 60-chunk profile is covered exactly: full-width windows, then at
//! most one partial window, with no token dropped or counted twice.
TEST(ChunkwiseWindowPartitionTest, CoversEveryClipExactly)
{
    int64_t const width = kTokensPerChunk * kChunksPerWindow;
    for (int64_t fullChunks = 1; fullChunks <= 60; ++fullChunks)
    {
        for (int64_t tail = 0; tail < kTokensPerChunk; ++tail)
        {
            auto const lens = clip(fullChunks, tail);
            int64_t const total = sum(lens);
            auto const windows = au::computeChunkwiseWindowLens(lens, kNWindow, kNWindowInfer);

            ASSERT_EQ(static_cast<int64_t>(windows.size()), (total + width - 1) / width)
                << fullChunks << " chunks, tail " << tail;
            EXPECT_EQ(sum(windows), total) << fullChunks << " chunks, tail " << tail;
            for (size_t i = 0; i + 1 < windows.size(); ++i)
            {
                EXPECT_EQ(windows[i], width) << fullChunks << " chunks, tail " << tail << ", window " << i;
            }
            EXPECT_GE(windows.back(), 1);
            EXPECT_LE(windows.back(), width);
        }
    }
}

//! The descriptor carries one boundary per window, so a clip spanning several windows is never
//! collapsed into a single {0, T} segment, which a ragged kernel would run as full attention.
TEST(ChunkwiseCuSeqlensTest, WritesOneBoundaryPerWindow)
{
    std::vector<int32_t> cuSeqlens(5, -1);
    EXPECT_EQ(au::fillChunkwiseCuSeqlens({104, 104, 104, 46}, cuSeqlens.data()), 104);
    EXPECT_EQ(cuSeqlens, (std::vector<int32_t>{0, 104, 208, 312, 358}));

    std::vector<int32_t> empty(1, -1);
    EXPECT_EQ(au::fillChunkwiseCuSeqlens({}, empty.data()), 0);
    EXPECT_EQ(empty[0], 0);
}

//! The dense mask and cu_seqlens are two encodings of one partition: a query may attend to a key
//! exactly when both fall in the same cu_seqlens segment. That equivalence is what lets the
//! attention plugin replace the mask.
TEST(ChunkwiseCuSeqlensTest, AgreesWithTheDenseMask)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (auto const& lens : {clip(27, 7), clip(3, 5), clip(16, 0)})
    {
        rt::Tensor mask;
        ASSERT_TRUE(au::createChunkwiseAttentionMask(lens, kNWindow, kNWindowInfer, mask, stream));

        auto const windows = au::computeChunkwiseWindowLens(lens, kNWindow, kNWindowInfer);
        std::vector<int32_t> cuSeqlens(windows.size() + 1);
        au::fillChunkwiseCuSeqlens(windows, cuSeqlens.data());
        int64_t const total = cuSeqlens.back();
        ASSERT_EQ(total, sum(lens));

        std::vector<__half> host(static_cast<size_t>(total * total));
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), mask.rawPointer(), host.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const segmentOf = [&cuSeqlens](int64_t token) {
            return std::upper_bound(cuSeqlens.begin(), cuSeqlens.end(), token) - cuSeqlens.begin();
        };
        for (int64_t query = 0; query < total; ++query)
        {
            for (int64_t key = 0; key < total; ++key)
            {
                float const value = __half2float(host[static_cast<size_t>(query * total + key)]);
                if (segmentOf(query) == segmentOf(key))
                {
                    ASSERT_EQ(value, 0.0F) << "query " << query << " key " << key << ", T=" << total;
                }
                else
                {
                    ASSERT_LT(value, -60000.0F) << "query " << query << " key " << key << ", T=" << total;
                }
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}
