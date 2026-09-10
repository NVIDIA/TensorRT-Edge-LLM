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

#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{

using trt_edgellm::canCompileXQAKernel;
using trt_edgellm::deserializeXQAJitKernels;
using trt_edgellm::serializeXQAJitKernels;
using trt_edgellm::XQAJitKernel;
using trt_edgellm::XQAJitKey;

TEST(XQAJitCapabilityTest, SupportsSm90Qwen25)
{
    EXPECT_TRUE(canCompileXQAKernel(14, 2, 64, 90, nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF));
    EXPECT_TRUE(canCompileXQAKernel(14, 2, 64, 90, nvinfer1::DataType::kHALF, nvinfer1::DataType::kFP8));
}

XQAJitKey makeKey(bool specDecode)
{
    XQAJitKey key{};
    key.sm = 121;
    key.dataType = nvinfer1::DataType::kHALF;
    key.kvDataType = nvinfer1::DataType::kFP8;
    key.headSize = 512;
    key.qHeadsPerKv = specDecode ? 0 : 8;
    key.tokensPerPage = 128;
    key.slidingWindow = true;
    key.specDecode = specDecode;
    return key;
}

std::vector<uint8_t> makeCubin(uint8_t seed, size_t size)
{
    std::vector<uint8_t> cubin(size);
    for (size_t i = 0; i < size; ++i)
    {
        cubin[i] = static_cast<uint8_t>(seed + i);
    }
    return cubin;
}

TEST(XQAJitSerializationTest, RoundTripsKeyAndCubin)
{
    std::vector<XQAJitKernel> const kernels{{makeKey(/*specDecode=*/false), makeCubin(0x11, 257)}};

    std::vector<uint8_t> const blob = serializeXQAJitKernels(kernels);
    std::vector<XQAJitKernel> const restored = deserializeXQAJitKernels(blob.data(), blob.size());

    ASSERT_EQ(restored.size(), 1U);
    EXPECT_TRUE(restored[0].key == kernels[0].key);
    EXPECT_EQ(restored[0].cubin, kernels[0].cubin);
}

TEST(XQAJitSerializationTest, RoundTripsDistinctTreeAttentionVariants)
{
    // The tree-attention case: two entries whose keys differ only in specDecode
    // (and the qHeadsPerKv the spec-decode path forces to 0).
    std::vector<XQAJitKernel> const kernels{
        {makeKey(/*specDecode=*/false), makeCubin(0x20, 64)},
        {makeKey(/*specDecode=*/true), makeCubin(0x40, 96)},
    };

    std::vector<uint8_t> const blob = serializeXQAJitKernels(kernels);
    std::vector<XQAJitKernel> const restored = deserializeXQAJitKernels(blob.data(), blob.size());

    ASSERT_EQ(restored.size(), 2U);
    EXPECT_FALSE(restored[0].key == restored[1].key);
    for (size_t i = 0; i < kernels.size(); ++i)
    {
        EXPECT_TRUE(restored[i].key == kernels[i].key) << "entry " << i;
        EXPECT_EQ(restored[i].cubin, kernels[i].cubin) << "entry " << i;
    }
}

TEST(XQAJitSerializationTest, EmptyKernelListRoundTrips)
{
    std::vector<uint8_t> const blob = serializeXQAJitKernels({});
    EXPECT_TRUE(deserializeXQAJitKernels(blob.data(), blob.size()).empty());
}

TEST(XQAJitSerializationTest, SizeIsDeterministicAndCarriesNoPadding)
{
    // 8 uint32 key fields + 1 uint32 cubin length, plus the version and count
    // headers. Serializing the key field by field (rather than memcpy-ing the
    // struct) is what keeps engine bytes reproducible.
    constexpr size_t kHeaderBytes = 2 * sizeof(uint32_t);
    constexpr size_t kPerEntryBytes = 9 * sizeof(uint32_t);
    constexpr size_t kCubinBytes = 128;

    std::vector<XQAJitKernel> const kernels{{makeKey(/*specDecode=*/false), makeCubin(0x01, kCubinBytes)}};
    std::vector<uint8_t> const first = serializeXQAJitKernels(kernels);
    std::vector<uint8_t> const second = serializeXQAJitKernels(kernels);

    EXPECT_EQ(first.size(), kHeaderBytes + kPerEntryBytes + kCubinBytes);
    EXPECT_EQ(first, second);
}

TEST(XQAJitSerializationTest, RejectsTruncatedBlob)
{
    std::vector<XQAJitKernel> const kernels{{makeKey(/*specDecode=*/false), makeCubin(0x30, 512)}};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob.resize(blob.size() - 1);

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsTrailingBytes)
{
    std::vector<XQAJitKernel> const kernels{{makeKey(/*specDecode=*/false), makeCubin(0x30, 32)}};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob.push_back(0xAB);

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsUnknownFormatVersion)
{
    std::vector<XQAJitKernel> const kernels{{makeKey(/*specDecode=*/false), makeCubin(0x30, 32)}};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob[0] = static_cast<uint8_t>(blob[0] + 1U);

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

} // namespace
