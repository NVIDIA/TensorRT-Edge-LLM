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

#include "kernels/contextCacheKernels/blockHashKernel.h"
#include "runtime/state/contextCache/blockHash.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <numeric>
#include <string_view>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

class BlockHashKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    void TearDown() override
    {
        cudaStreamDestroy(mStream);
    }

    cudaStream_t mStream{};
};

TEST_F(BlockHashKernelTest, SmallPayloadGpuMatchesCpu)
{
    // Below GPU threshold — both paths use CPU, results must be identical.
    std::vector<uint8_t> data(256);
    std::iota(data.begin(), data.end(), static_cast<uint8_t>(0));
    std::string_view const bytes(reinterpret_cast<char const*>(data.data()), data.size());

    Hash128 cpuHash = hashOpaqueIdentity(bytes);
    Hash128 gpuHash = hashOpaqueIdentity(bytes, mStream, /*cpuOnly=*/false);

    EXPECT_EQ(cpuHash.hi, gpuHash.hi);
    EXPECT_EQ(cpuHash.lo, gpuHash.lo);
}

TEST_F(BlockHashKernelTest, LargePayloadGpuMatchesCpu)
{
    // 1 MB payload — above GPU threshold. GPU parallel kernel must produce bit-identical result to CPU chunked path.
    constexpr size_t kSIZE = 1024 * 1024;
    std::vector<uint8_t> data(kSIZE);
    for (size_t i = 0; i < kSIZE; ++i)
    {
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }
    std::string_view const bytes(reinterpret_cast<char const*>(data.data()), data.size());

    Hash128 cpuHash = hashOpaqueIdentity(bytes);
    Hash128 gpuHash = hashOpaqueIdentity(bytes, mStream, /*cpuOnly=*/false);

    EXPECT_EQ(cpuHash.hi, gpuHash.hi);
    EXPECT_EQ(cpuHash.lo, gpuHash.lo);
}

TEST_F(BlockHashKernelTest, TenMbPayloadGpuMatchesCpu)
{
    // 10 MB — realistic VLM image size.
    constexpr size_t kSIZE = 10 * 1024 * 1024;
    std::vector<uint8_t> data(kSIZE);
    for (size_t i = 0; i < kSIZE; ++i)
    {
        data[i] = static_cast<uint8_t>((i * 31 + 97) & 0xFF);
    }
    std::string_view const bytes(reinterpret_cast<char const*>(data.data()), data.size());

    Hash128 cpuHash = hashOpaqueIdentity(bytes);
    Hash128 gpuHash = hashOpaqueIdentity(bytes, mStream, /*cpuOnly=*/false);

    EXPECT_EQ(cpuHash.hi, gpuHash.hi);
    EXPECT_EQ(cpuHash.lo, gpuHash.lo);
}

TEST_F(BlockHashKernelTest, DifferentDataProducesDifferentHash)
{
    constexpr size_t kSIZE = 512 * 1024;
    std::vector<uint8_t> dataA(kSIZE, 0xAA);
    std::vector<uint8_t> dataB(kSIZE, 0xBB);

    Hash128 hashA
        = hashOpaqueIdentity(std::string_view(reinterpret_cast<char const*>(dataA.data()), kSIZE), mStream, false);
    Hash128 hashB
        = hashOpaqueIdentity(std::string_view(reinterpret_cast<char const*>(dataB.data()), kSIZE), mStream, false);

    EXPECT_TRUE(hashA.hi != hashB.hi || hashA.lo != hashB.lo);
}

TEST_F(BlockHashKernelTest, SameContentDifferentSizeProducesDifferentHash)
{
    constexpr size_t kSIZE_A = 512 * 1024;
    constexpr size_t kSIZE_B = 1024 * 1024;
    std::vector<uint8_t> data(kSIZE_B, 0x42);

    Hash128 hashA
        = hashOpaqueIdentity(std::string_view(reinterpret_cast<char const*>(data.data()), kSIZE_A), mStream, false);
    Hash128 hashB
        = hashOpaqueIdentity(std::string_view(reinterpret_cast<char const*>(data.data()), kSIZE_B), mStream, false);

    EXPECT_TRUE(hashA.hi != hashB.hi || hashA.lo != hashB.lo);
}

TEST_F(BlockHashKernelTest, NonChunkAlignedSizeWorks)
{
    // Size not evenly divisible by kHASH_NUM_CHUNKS — last chunk is shorter.
    constexpr size_t kSIZE = 1024 * 1024 + 137;
    std::vector<uint8_t> data(kSIZE);
    for (size_t i = 0; i < kSIZE; ++i)
    {
        data[i] = static_cast<uint8_t>((i * 3 + 5) & 0xFF);
    }
    std::string_view const bytes(reinterpret_cast<char const*>(data.data()), data.size());

    Hash128 cpuHash = hashOpaqueIdentity(bytes);
    Hash128 gpuHash = hashOpaqueIdentity(bytes, mStream, /*cpuOnly=*/false);

    EXPECT_EQ(cpuHash.hi, gpuHash.hi);
    EXPECT_EQ(cpuHash.lo, gpuHash.lo);
}

TEST_F(BlockHashKernelTest, CpuOnlyFlagForcessCpuPath)
{
    // cpuOnly=true with a valid stream should still use CPU (and produce same result).
    constexpr size_t kSIZE = 512 * 1024;
    std::vector<uint8_t> data(kSIZE, 0x77);
    std::string_view const bytes(reinterpret_cast<char const*>(data.data()), data.size());

    Hash128 defaultHash = hashOpaqueIdentity(bytes);
    Hash128 cpuOnlyHash = hashOpaqueIdentity(bytes, mStream, /*cpuOnly=*/true);

    EXPECT_EQ(defaultHash.hi, cpuOnlyHash.hi);
    EXPECT_EQ(defaultHash.lo, cpuOnlyHash.lo);
}

} // namespace
