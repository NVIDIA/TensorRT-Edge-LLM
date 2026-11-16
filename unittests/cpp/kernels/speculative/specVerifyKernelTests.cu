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
#include "kernels/speculative/speculativeKernels.h"
#include "testUtils.h"
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;

TEST(SpecVerifyKernels, ChainMetadataKeepsPhysicalStrideAndPoisonsPadding)
{
    // Given unequal valid counts and a width crossing a mask-word boundary,
    // when preparing a chain, then positions/ancestry/masks retain physical rows.
    constexpr int32_t batch = 2;
    constexpr int32_t width = 35;
    rt::Tensor past({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor counts({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor mask({batch * width, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor positions({batch * width}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor indices({batch * width}, rt::DeviceType::kGPU, DataType::kINT64);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor parents({batch * width}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor depths({batch * width}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(past, {7, 19});
    copyHostToDevice<int32_t>(counts, {35, 3});
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    kernel::prepareSpecChainVerifyInputs(
        past, &counts, mask, positions, indices, lengths, &parents, &depths, batch, width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::vector<int32_t> expectedPositions, expectedParents, expectedDepths, expectedMask;
    std::vector<int64_t> expectedIndices;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t node = 0; node < width; ++node)
        {
            bool const valid = node < (b == 0 ? 35 : 3);
            expectedPositions.push_back(valid ? (b == 0 ? 7 : 19) + node : -1);
            expectedParents.push_back(valid && node > 0 ? node - 1 : -1);
            expectedDepths.push_back(valid ? node : -1);
            expectedIndices.push_back(b * width + node);
            for (int32_t word = 0; word < 2; ++word)
            {
                uint32_t bits = 0;
                for (int32_t ancestor = 0; valid && ancestor <= node; ++ancestor)
                {
                    if (ancestor / 32 == word)
                    {
                        bits |= 1u << (ancestor % 32);
                    }
                }
                expectedMask.push_back(static_cast<int32_t>(bits));
            }
        }
    }
    EXPECT_EQ(copyDeviceToHost<int32_t>(positions), expectedPositions);
    EXPECT_EQ(copyDeviceToHost<int32_t>(parents), expectedParents);
    EXPECT_EQ(copyDeviceToHost<int32_t>(depths), expectedDepths);
    EXPECT_EQ(copyDeviceToHost<int64_t>(indices), expectedIndices);
    EXPECT_EQ(copyDeviceToHost<int32_t>(mask), expectedMask);
    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{42, 54}));
}
