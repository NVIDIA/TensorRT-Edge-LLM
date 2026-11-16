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

#include "kernels/PluginJitKernels/pluginJitDigest.h"
#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using trt_edgellm::canCompileXQAKernel;
using trt_edgellm::compileXQAKernel;
using trt_edgellm::computePluginJitDigest;
using trt_edgellm::DecoderXQARunner;
using trt_edgellm::deserializeXQAJitKernels;
using trt_edgellm::PluginJitDigest;
using trt_edgellm::serializeXQAJitKernels;
using trt_edgellm::XQAJitKernel;
using trt_edgellm::XQAJitKey;

//! Blob layout pinned by the tests below: magic word and count, then per entry the 9 uint32 key
//! fields, a uint32 cubin length, the 128-bit digest and the cubin bytes.
constexpr size_t kBLOB_HEADER_BYTES{2U * sizeof(uint32_t)};
constexpr size_t kENTRY_PREFIX_BYTES{10U * sizeof(uint32_t) + 2U * sizeof(uint64_t)};

TEST(XQAJitCapabilityTest, SupportsSm90Qwen25)
{
    EXPECT_TRUE(canCompileXQAKernel(14, 2, 64, 90, trt_edgellm::DataType::kHALF, trt_edgellm::DataType::kHALF));
    EXPECT_TRUE(canCompileXQAKernel(14, 2, 64, 90, trt_edgellm::DataType::kHALF, trt_edgellm::DataType::kFP8));
}

TEST(XQAJitCapabilityTest, SupportsSm103Qwen3)
{
    EXPECT_TRUE(canCompileXQAKernel(16, 8, 128, 103, trt_edgellm::DataType::kHALF, trt_edgellm::DataType::kHALF));
    EXPECT_TRUE(canCompileXQAKernel(16, 8, 128, 103, trt_edgellm::DataType::kHALF, trt_edgellm::DataType::kFP8));
}

XQAJitKey makeKey(bool specDecode, bool contiguousQuerySwa = false)
{
    XQAJitKey key{};
    key.sm = 121;
    key.dataType = trt_edgellm::DataType::kHALF;
    key.kvDataType = trt_edgellm::DataType::kFP8;
    key.headSize = 512;
    key.qHeadsPerKv = specDecode ? 0 : 8;
    key.tokensPerPage = 128;
    key.slidingWindow = true;
    key.specDecode = specDecode;
    key.contiguousQuerySwa = contiguousQuerySwa;
    return key;
}

//! Fake cubin: the ELF magic (the blob refuses anything else) followed by seeded bytes.
std::vector<uint8_t> makeCubin(uint8_t seed, size_t size)
{
    std::vector<uint8_t> cubin{0x7FU, 0x45U, 0x4CU, 0x46U};
    while (cubin.size() < size)
    {
        cubin.push_back(static_cast<uint8_t>(seed + cubin.size()));
    }
    return cubin;
}

XQAJitKernel makeKernel(XQAJitKey const& key, std::vector<uint8_t> cubin)
{
    XQAJitKernel kernel;
    kernel.key = key;
    kernel.cubin = std::move(cubin);
    kernel.digest = computePluginJitDigest(key, kernel.cubin.data(), kernel.cubin.size());
    return kernel;
}

TEST(XQAJitSerializationTest, RoundTripsKeyAndCubin)
{
    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x11, 257))};

    std::vector<uint8_t> const blob = serializeXQAJitKernels(kernels);
    std::vector<XQAJitKernel> const restored = deserializeXQAJitKernels(blob.data(), blob.size());

    ASSERT_EQ(restored.size(), 1U);
    EXPECT_TRUE(restored[0].key == kernels[0].key);
    EXPECT_EQ(restored[0].cubin, kernels[0].cubin);
    EXPECT_EQ(restored[0].digest, kernels[0].digest);
}

TEST(XQAJitSerializationTest, RoundTripsDistinctTreeAttentionVariants)
{
    // The tree-attention case: two entries whose keys differ only in specDecode
    // (and the qHeadsPerKv the spec-decode path forces to 0).
    std::vector<XQAJitKernel> const kernels{
        makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x20, 64)),
        makeKernel(makeKey(/*specDecode=*/true), makeCubin(0x40, 96)),
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

TEST(XQAJitSerializationTest, RoundTripsDistinctContiguousQuerySwaVariants)
{
    std::vector<XQAJitKernel> const kernels{
        makeKernel(makeKey(/*specDecode=*/true), makeCubin(0x50, 64)),
        makeKernel(makeKey(/*specDecode=*/true, /*contiguousQuerySwa=*/true), makeCubin(0x60, 96)),
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
    // Serializing the key field by field (rather than memcpy-ing the struct) is
    // what keeps engine bytes reproducible.
    constexpr size_t kCubinBytes = 128;

    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x01, kCubinBytes))};
    std::vector<uint8_t> const first = serializeXQAJitKernels(kernels);
    std::vector<uint8_t> const second = serializeXQAJitKernels(kernels);

    EXPECT_EQ(first.size(), kBLOB_HEADER_BYTES + kENTRY_PREFIX_BYTES + kCubinBytes);
    EXPECT_EQ(first, second);
}

TEST(XQAJitSerializationTest, RejectsTruncatedBlob)
{
    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x30, 512))};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob.resize(blob.size() - 1);

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsTrailingBytes)
{
    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x30, 32))};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob.push_back(0xAB);

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsBlobsWithoutTheMagicWord)
{
    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x30, 32))};
    std::vector<uint8_t> const blob = serializeXQAJitKernels(kernels);

    std::vector<uint8_t> tampered = blob;
    tampered[0] ^= 0xFFU;
    EXPECT_THROW(deserializeXQAJitKernels(tampered.data(), tampered.size()), std::runtime_error);

    // Blobs written before the shared header opened with a format version word
    // (1 or 2) and had no magic; an engine carrying one is refused at the magic
    // check rather than misread.
    for (uint8_t const oldVersion : {uint8_t{1U}, uint8_t{2U}})
    {
        tampered = blob;
        tampered[0] = oldVersion;
        tampered[1] = 0U;
        tampered[2] = 0U;
        tampered[3] = 0U;
        EXPECT_THROW(deserializeXQAJitKernels(tampered.data(), tampered.size()), std::runtime_error)
            << "version " << static_cast<int32_t>(oldVersion);
    }
}

TEST(XQAJitSerializationTest, RejectsTamperedCubin)
{
    std::vector<XQAJitKernel> const kernels{makeKernel(makeKey(/*specDecode=*/false), makeCubin(0x30, 32))};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);
    blob.back() ^= 0x01U;

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsNonElfCubin)
{
    std::vector<uint8_t> ptx(32, 0x2FU);
    XQAJitKernel const kernel = makeKernel(makeKey(/*specDecode=*/false), std::move(ptx));

    EXPECT_THROW(serializeXQAJitKernels({kernel}), std::invalid_argument);
}

TEST(XQAJitSerializationTest, RejectsInvalidContiguousQuerySwaVariant)
{
    XQAJitKey key = makeKey(/*specDecode=*/false, /*contiguousQuerySwa=*/true);
    EXPECT_THROW(serializeXQAJitKernels({makeKernel(key, makeCubin(0x30, 32))}), std::runtime_error);

    key.specDecode = true;
    key.slidingWindow = false;
    EXPECT_THROW(serializeXQAJitKernels({makeKernel(key, makeCubin(0x30, 32))}), std::runtime_error);
}

TEST(XQAJitValidationTest, RejectsInvalidContiguousQuerySwaBeforeCompileOrLoad)
{
    XQAJitKey const key = makeKey(/*specDecode=*/false, /*contiguousQuerySwa=*/true);
    uint8_t const cubinByte{0};

    EXPECT_THROW(compileXQAKernel(key), std::runtime_error);
    EXPECT_THROW(
        DecoderXQARunner::loadDecodeXQAKernelFromCubin(key, &cubinByte, sizeof(cubinByte)), std::runtime_error);
}

TEST(XQAJitSerializationTest, RejectsDeserializedInvalidContiguousQuerySwaVariant)
{
    std::vector<XQAJitKernel> const kernels{
        makeKernel(makeKey(/*specDecode=*/true, /*contiguousQuerySwa=*/true), makeCubin(0x30, 32))};
    std::vector<uint8_t> blob = serializeXQAJitKernels(kernels);

    // Clear slidingWindow (key word 6) and re-sign the entry so only the key
    // validation can reject the blob.
    constexpr size_t kSlidingWindowFieldIndex{6U};
    blob[kBLOB_HEADER_BYTES + kSlidingWindowFieldIndex * sizeof(uint32_t)] = 0U;
    XQAJitKey invalidKey = kernels[0].key;
    invalidKey.slidingWindow = false;
    PluginJitDigest const digest = computePluginJitDigest(invalidKey, kernels[0].cubin.data(), kernels[0].cubin.size());
    size_t const digestOffset = kBLOB_HEADER_BYTES + 10U * sizeof(uint32_t);
    for (size_t byte = 0; byte < sizeof(uint64_t); ++byte)
    {
        blob[digestOffset + byte] = static_cast<uint8_t>(digest.lo >> (8U * byte));
        blob[digestOffset + sizeof(uint64_t) + byte] = static_cast<uint8_t>(digest.hi >> (8U * byte));
    }

    EXPECT_THROW(deserializeXQAJitKernels(blob.data(), blob.size()), std::runtime_error);
}

} // namespace
