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
#include "runtime/visionDataParallel/visionGather.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace trt_edgellm::rt
{
namespace
{
class VisionGatherTest : public ::testing::Test
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

struct FakeCollectives
{
    uint16_t* output;
    uint16_t const* peer;
    std::array<int32_t, 2> counts;
    int32_t rank;
    cudaStream_t stream;
    std::vector<int32_t> calls;
    int32_t failCall{-1};

    bool allGather(void const* send, void* recv, int32_t count, cudaStream_t callStream)
    {
        calls.push_back(2);
        if (send != output + (rank == 0 ? 0 : counts[0]) || recv != output || counts[0] != counts[1]
            || count != counts[0] || callStream != stream || failCall == static_cast<int32_t>(calls.size()))
        {
            return false;
        }
        int32_t const offset = rank == 0 ? counts[0] : 0;
        CUDA_CHECK(cudaMemcpyAsync(
            output + offset, peer + offset, count * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
        return true;
    }

    bool broadcast(void* data, int32_t count, int32_t root, cudaStream_t callStream)
    {
        calls.push_back(root);
        if (root < 0 || root > 1 || data != output + (root == 0 ? 0 : counts[0]) || count != counts[root]
            || counts[0] == counts[1] || callStream != stream || failCall == static_cast<int32_t>(calls.size()))
        {
            return false;
        }
        if (root != rank)
        {
            int32_t const offset = root == 0 ? 0 : counts[0];
            CUDA_CHECK(
                cudaMemcpyAsync(data, peer + offset, count * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
        }
        return true;
    }

    VisionGatherFn callback()
    {
        return [this](void* buffer, int32_t n0, int32_t n1, cudaStream_t callStream) {
            return gatherVisionShards(
                buffer, n0, n1, rank,
                [this](void const* send, void* recv, int32_t count, cudaStream_t s) {
                    return allGather(send, recv, count, s);
                },
                [this](void* data, int32_t count, int32_t root, cudaStream_t s) {
                    return broadcast(data, count, root, s);
                },
                callStream);
        };
    }
};

// Given distinct local/peer features and reusable storage on one GPU, for either logical rank,
// when consecutive equal and unequal shards are gathered, then original image order, full shapes,
// untouched tail storage, and the matching collective arguments are preserved for every output.
TEST_F(VisionGatherTest, PreservesOrderedFeaturesAcrossShardShapes)
{
    constexpr int64_t capacityTokens{8};
    for (int32_t rank : {0, 1})
    {
        for (int64_t hidden : {3, 5, 7, 9})
        {
            Tensor output({capacityTokens, hidden}, DeviceType::kGPU, DataType::kHALF);
            Tensor peer({capacityTokens, hidden}, DeviceType::kGPU, DataType::kHALF);
            Tensor expected({capacityTokens, hidden}, DeviceType::kCPU, DataType::kHALF);
            Tensor initial({capacityTokens, hidden}, DeviceType::kCPU, DataType::kHALF);
            Tensor actual({capacityTokens, hidden}, DeviceType::kCPU, DataType::kHALF);
            for (std::array<int64_t, 2> tokens : {std::array<int64_t, 2>{2, 2}, {3, 1}, {1, 3}, {1, 1}})
            {
                SCOPED_TRACE(::testing::Message()
                    << "rank=" << rank << " hidden=" << hidden << " tokens=" << tokens[0] << "+" << tokens[1]);
                int64_t const fullElements = (tokens[0] + tokens[1]) * hidden;
                int64_t const offset = rank == 0 ? 0 : tokens[0] * hidden;
                auto* values = expected.dataPointer<uint16_t>();
                auto* local = initial.dataPointer<uint16_t>();
                for (int64_t i = 0; i < capacityTokens * hidden; ++i)
                {
                    values[i] = i < fullElements ? static_cast<uint16_t>(100 + hidden * 10 + i) : 0xFFFF;
                    local[i] = i >= offset && i < offset + tokens[rank] * hidden ? values[i] : 0xFFFF;
                }
                ELLM_CHECK(output.reshape({tokens[rank], hidden}), "Test local shape must fit");
                CUDA_CHECK(cudaMemcpyAsync(output.rawPointer(), initial.rawPointer(), initial.getMemoryCapacity(),
                    cudaMemcpyHostToDevice, mStream));
                CUDA_CHECK(cudaMemcpyAsync(peer.rawPointer(), expected.rawPointer(), expected.getMemoryCapacity(),
                    cudaMemcpyHostToDevice, mStream));
                FakeCollectives fake{output.dataPointer<uint16_t>(), peer.dataPointer<uint16_t>(),
                    {static_cast<int32_t>(tokens[0] * hidden), static_cast<int32_t>(tokens[1] * hidden)}, rank, mStream,
                    {}};
                EXPECT_TRUE(gatherVisionTensor(output, tokens, rank, fake.callback(), mStream));
                EXPECT_EQ(output.getShape(), (Coords{tokens[0] + tokens[1], hidden}));
                EXPECT_EQ(fake.calls, tokens[0] == tokens[1] ? std::vector<int32_t>{2} : (std::vector<int32_t>{0, 1}));
                CUDA_CHECK(cudaMemcpyAsync(actual.rawPointer(), output.rawPointer(), actual.getMemoryCapacity(),
                    cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                EXPECT_EQ(std::vector<uint16_t>(
                              actual.dataPointer<uint16_t>(), actual.dataPointer<uint16_t>() + capacityTokens * hidden),
                    std::vector<uint16_t>(values, values + capacityTokens * hidden));
            }
        }
    }
}

// Given a local output whose shape, element count, or allocation cannot hold the requested gather,
// when gathering is attempted, then it fails before any communication or output-shape change.
TEST_F(VisionGatherTest, RejectsInvalidOutputsBeforeCommunication)
{
    Tensor output({4, 3}, DeviceType::kGPU, DataType::kHALF);
    int32_t calls = 0;
    VisionGatherFn callback = [&](void*, int32_t, int32_t, cudaStream_t) {
        ++calls;
        return true;
    };
    struct Case
    {
        Coords shape;
        std::array<int64_t, 2> tokens;
        int32_t rank;
    };
    std::vector<Case> const cases{{{2, 3}, {2, 2}, -1}, {{2, 3}, {2, 2}, 2}, {{2, 3}, {0, 2}, 1}, {{2, 3}, {2, -1}, 0},
        {{1, 2, 3}, {2, 2}, 0}, {{1, 3}, {2, 2}, 0}, {{2, 3}, {2, 3}, 0},
        {{2, 3}, {2, std::numeric_limits<int32_t>::max()}, 0}};
    for (auto const& c : cases)
    {
        ELLM_CHECK(output.reshape(c.shape), "Test shape must fit");
        EXPECT_THROW(gatherVisionTensor(output, c.tokens, c.rank, callback, mStream), std::runtime_error);
        EXPECT_EQ(output.getShape(), c.shape);
    }
    Tensor wrongType({8, 3}, DeviceType::kGPU, DataType::kFLOAT);
    ELLM_CHECK(wrongType.reshape({4, 3}), "Test float shape must fit");
    EXPECT_THROW(gatherVisionTensor(wrongType, {4, 4}, 0, callback, mStream), std::runtime_error);
    Tensor host({8, 3}, DeviceType::kCPU, DataType::kHALF);
    ELLM_CHECK(host.reshape({4, 3}), "Test host shape must fit");
    EXPECT_THROW(gatherVisionTensor(host, {4, 4}, 0, callback, mStream), std::runtime_error);
    EXPECT_EQ(calls, 0);
}

// Given a collective failure for either rank and equal/unequal shards,
// when gathering is attempted, then failure propagates, later collectives are not issued,
// and the local output shape remains unchanged.
TEST_F(VisionGatherTest, PropagatesCollectiveFailureWithoutPublishingFullShape)
{
    for (int32_t rank : {0, 1})
    {
        for (std::array<int64_t, 2> tokens : {std::array<int64_t, 2>{2, 2}, {3, 1}, {1, 3}})
        {
            for (int32_t failure : (tokens[0] == tokens[1] ? std::vector<int32_t>{1} : std::vector<int32_t>{1, 2}))
            {
                Tensor output({4, 3}, DeviceType::kGPU, DataType::kHALF);
                Tensor peer({4, 3}, DeviceType::kGPU, DataType::kHALF);
                ELLM_CHECK(output.reshape({tokens[rank], 3}), "Test local shape must fit");
                FakeCollectives fake{output.dataPointer<uint16_t>(), peer.dataPointer<uint16_t>(),
                    {static_cast<int32_t>(tokens[0] * 3), static_cast<int32_t>(tokens[1] * 3)}, rank, mStream, {},
                    failure};
                EXPECT_FALSE(gatherVisionTensor(output, tokens, rank, fake.callback(), mStream));
                EXPECT_EQ(output.getShape(), (Coords{tokens[rank], 3}));
                EXPECT_EQ(fake.calls,
                    tokens[0] == tokens[1] ? std::vector<int32_t>{2}
                                           : (failure == 1 ? std::vector<int32_t>{0} : std::vector<int32_t>{0, 1}));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
            }
        }
    }
}
} // namespace
} // namespace trt_edgellm::rt
