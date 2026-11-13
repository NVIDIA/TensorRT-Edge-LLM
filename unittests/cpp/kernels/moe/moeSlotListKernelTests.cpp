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
#include "common/tensor.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using namespace trt_edgellm;

class MoeSlotListTest : public ::testing::TestWithParam<std::pair<int32_t, int32_t>>
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }
    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }
    cudaStream_t mStream{};
};

TEST(MoeSlotListDispatchTest, SelectsOnlyMeasuredThorShapeRange)
{
    // Given SM100 and SM110 devices with small and large expert counts,
    // when selecting the stable-list kernel, then only measured SM110 grouped shapes use the optimized path.
    EXPECT_FALSE(kernel::shouldUseGroupedStableSlotLists(100, 512));
    EXPECT_FALSE(kernel::shouldUseGroupedStableSlotLists(110, 127));
    EXPECT_TRUE(kernel::shouldUseGroupedStableSlotLists(110, 128));
    EXPECT_TRUE(kernel::shouldUseGroupedStableSlotLists(110, 130));
}

TEST_P(MoeSlotListTest, PreservesTokenOrderWithinEachExpert)
{
    // Given interleaved expert assignments with invalid IDs and partial tiles,
    // when grouping routed slots through a captured graph, then every replay retains ascending slot order
    // so Marlin's reduction partition cannot vary with atomic arrival order.
    auto const [numTokens, numExperts] = GetParam();
    int32_t constexpr topK = 6;
    int32_t const totalSlots = numTokens * topK;
    int32_t const listSize = numExperts * totalSlots;
    rt::Tensor hostIds({totalSlots}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    rt::Tensor hostCounts({numExperts}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    rt::Tensor hostSlots({listSize}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    rt::Tensor ids({totalSlots}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor counts({numExperts}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor slots({listSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    std::vector<int32_t> expectedCounts(numExperts, 0);
    std::vector<int32_t> expectedSlots(listSize, -1);
    auto* assignments = static_cast<int32_t*>(hostIds.rawPointer());
    for (int32_t slot = 0; slot < totalSlots; ++slot)
    {
        int32_t const expert = slot % 19 == 0 ? -1
            : slot % 23 == 0                  ? numExperts
            : slot % 11 == 0                  ? 0
                                              : (slot / topK + (slot % topK) * 17) % numExperts;
        assignments[slot] = expert;
        if (expert >= 0 && expert < numExperts)
        {
            expectedSlots[expert * totalSlots + expectedCounts[expert]++] = slot;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        ids.rawPointer(), hostIds.rawPointer(), totalSlots * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemsetAsync(counts.rawPointer(), 0, numExperts * sizeof(int32_t), mStream));
    CUDA_CHECK(cudaMemsetAsync(slots.rawPointer(), 0xff, listSize * sizeof(int32_t), mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    (void) getSMVersion();
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
    kernel::launchBuildStableSlotListsKernel(static_cast<int32_t const*>(ids.rawPointer()),
        static_cast<int32_t*>(slots.rawPointer()), static_cast<int32_t*>(counts.rawPointer()), numTokens, topK,
        numExperts, mStream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    auto const* actualCounts = static_cast<int32_t const*>(hostCounts.rawPointer());
    auto const* actualSlots = static_cast<int32_t const*>(hostSlots.rawPointer());
    for (int32_t replay = 0; replay < 2; ++replay)
    {
        CUDA_CHECK(cudaMemsetAsync(counts.rawPointer(), 0xff, numExperts * sizeof(int32_t), mStream));
        CUDA_CHECK(cudaMemsetAsync(slots.rawPointer(), 0xff, listSize * sizeof(int32_t), mStream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, mStream));
        CUDA_CHECK(cudaMemcpyAsync(hostCounts.rawPointer(), counts.rawPointer(), numExperts * sizeof(int32_t),
            cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            hostSlots.rawPointer(), slots.rawPointer(), listSize * sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        EXPECT_EQ(std::vector<int32_t>(actualCounts, actualCounts + numExperts), expectedCounts);
        EXPECT_EQ(std::vector<int32_t>(actualSlots, actualSlots + listSize), expectedSlots);
    }
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

INSTANTIATE_TEST_SUITE_P(PartialAndFullTiles, MoeSlotListTest,
    ::testing::Values(std::pair{31, 8}, std::pair{383, 130}, std::pair{1366, 512}));
