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

#include "nativeCudaTestUtils.h"
#include "safety_inference/kernels/safetyRopeKernel.cuh"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::native_test;

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/posEncoding/applyRopeWriteKV.cu,
// applyRope<half>. Test-only rename and host/device annotation; preserve Standard arithmetic as a differential oracle.
__host__ __device__ half standardApplyRope(
    half const& x, half const& y, float const& cos, float const& sin, bool isLeft)
{
    float const val
        = isLeft ? (__half2float(x) * cos - __half2float(y) * sin) : (__half2float(x) * cos + __half2float(y) * sin);
    return __float2half(val);
}

__global__ void referenceQueries(
    AttentionLaunch shape, half const* packed, float const* rope, int32_t start, int32_t length, half* output)
{
    int32_t const token = blockIdx.x;
    int32_t const head = blockIdx.y;
    int32_t const component = threadIdx.x;
    if (token >= length || component >= shape.headSize)
    {
        return;
    }
    int32_t const packedStride = (shape.qHeads + 2 * shape.kvHeads) * shape.headSize;
    half const* const vector = packed + token * packedStride + head * shape.headSize;
    half value = vector[component];
    if (component < shape.rotaryDim)
    {
        int32_t const halfRotary = shape.rotaryDim / 2;
        bool const left = component < halfRotary;
        int32_t const partner = left ? component + halfRotary : component - halfRotary;
        int32_t const frequency = component % halfRotary;
        float const* const coefficients = rope + (start + token) * shape.rotaryDim;
        value = standardApplyRope(
            value, vector[partner], coefficients[frequency], coefficients[halfRotary + frequency], left);
    }
    output[(token * shape.qHeads + head) * shape.headSize + component] = value;
}

struct RopeCase
{
    AttentionLaunch shape{};
    Stream stream;
    Buffer<half> packed;
    Buffer<half> queries;
    Buffer<half> cache;
    Buffer<float> rope;
    Buffer<int32_t> lengths{1}, starts{1}, valid{1};
    Buffer<uint64_t> error{1};

    RopeCase(int32_t headSize, int32_t rotaryDim)
        : packed(3 * 8 * headSize)
        , queries(3 * 4 * headSize)
        , cache(2 * 259 * 2 * headSize)
        , rope(512 * rotaryDim)
    {
        shape.tokens = 3;
        shape.qHeads = 4;
        shape.kvHeads = 2;
        shape.headSize = headSize;
        shape.capacity = 259;
        shape.ropeTokens = 512;
        shape.rotaryDim = rotaryDim;
        lengths.host[0] = 2;
        starts.host[0] = 127;
        valid.host[0] = 1;
        std::fill_n(queries.host, queries.count, __float2half(-9.F));
        std::fill_n(cache.host, cache.count, __float2half(-0.25F));
        for (size_t i = 0; i < packed.count; ++i)
        {
            packed.host[i] = __float2half(static_cast<float>(static_cast<int32_t>(i % 71) - 35) * 0.013F);
        }
        int32_t const livePacked = lengths.host[0] * 8 * headSize;
        std::fill(packed.host + livePacked, packed.host + packed.count,
            __float2half(std::numeric_limits<float>::quiet_NaN()));
        for (int32_t p = 0; p < shape.ropeTokens; ++p)
        {
            for (int32_t k = 0; k < rotaryDim / 2; ++k)
            {
                rope.host[p * rotaryDim + k] = std::cos(0.01F * (p + k));
                rope.host[p * rotaryDim + rotaryDim / 2 + k] = std::sin(0.01F * (p + k));
            }
        }
    }

    void run()
    {
        packed.upload(stream.value);
        queries.upload(stream.value);
        cache.upload(stream.value);
        rope.upload(stream.value);
        lengths.upload(stream.value);
        starts.upload(stream.value);
        valid.upload(stream.value);
        error.upload(stream.value);
        dim3 const block = safetyRopeBlock(shape);
        dim3 const grid((shape.tokens + block.y - 1) / block.y, shape.qHeads + shape.kvHeads);
        safetyApplyRopeFromPackedToSplit<<<grid, block, 0, stream.value>>>(shape, packed.device, lengths.device,
            starts.device, rope.device, queries.device, cache.device, valid.device, error.device);
        checked(cudaGetLastError());
        queries.download(stream.value);
        cache.download(stream.value);
        error.download(stream.value);
        checked(cudaStreamSynchronize(stream.value));
    }
};

TEST(SafetyGpuRope, StandardArithmeticPartialRotationAndContiguousCache)
{
    // Given partial RoPE and a cached prefix, when writing KV, then rotation and untouched storage match references.
    for (int32_t headSize : {64, 128})
    {
        for (int32_t rotary : {2, 6, 14, 16, 30, 32, 62, 64, 126, 128})
        {
            if (rotary > headSize)
            {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "headSize=" << headSize << " rotary=" << rotary);
            RopeCase data(headSize, rotary);
            auto const& shape = data.shape;
            std::vector<half> expectedQueries(data.queries.host, data.queries.host + data.queries.count);
            std::vector<half> expectedCache(data.cache.host, data.cache.host + data.cache.count);
            for (int32_t t = 0; t < data.lengths.host[0]; ++t)
            {
                int32_t const position = data.starts.host[0] + t;
                for (int32_t h = 0; h < shape.qHeads + shape.kvHeads; ++h)
                {
                    half const* const vector = data.packed.host + (t * 8 + h) * headSize;
                    for (int32_t c = 0; c < headSize; ++c)
                    {
                        float value = __half2float(vector[c]);
                        if (c < rotary)
                        {
                            int32_t const partner = c < rotary / 2 ? c + rotary / 2 : c - rotary / 2;
                            int32_t const frequency = c % (rotary / 2);
                            float const cos = data.rope.host[position * rotary + frequency];
                            float const sin = data.rope.host[position * rotary + rotary / 2 + frequency];
                            float const sign = c < rotary / 2 ? -1.F : 1.F;
                            value = value * cos + sign * __half2float(vector[partner]) * sin;
                        }
                        if (h < shape.qHeads)
                        {
                            expectedQueries[(t * shape.qHeads + h) * headSize + c] = __float2half(value);
                        }
                        else
                        {
                            int32_t const kv = h - shape.qHeads;
                            int32_t const offset = (kv * shape.capacity + position) * headSize + c;
                            expectedCache[offset] = __float2half(value);
                            expectedCache[shape.kvHeads * shape.capacity * headSize + offset]
                                = vector[shape.kvHeads * headSize + c];
                        }
                    }
                }
            }
            Buffer<half> standardQueries(data.queries.count);
            std::copy_n(data.queries.host, data.queries.count, standardQueries.host);
            data.run();
            ASSERT_EQ(data.error.host[0], 0U);
            standardQueries.upload(data.stream.value);
            referenceQueries<<<dim3(shape.tokens, shape.qHeads), headSize, 0, data.stream.value>>>(shape,
                data.packed.device, data.rope.device, data.starts.host[0], data.lengths.host[0],
                standardQueries.device);
            checked(cudaGetLastError());
            standardQueries.download(data.stream.value);
            checked(cudaStreamSynchronize(data.stream.value));
            for (size_t i = 0; i < data.queries.count; ++i)
            {
                EXPECT_EQ(__half2float(data.queries.host[i]), __half2float(standardQueries.host[i])) << "Q " << i;
                EXPECT_NEAR(__half2float(data.queries.host[i]), __half2float(expectedQueries[i]), 0.001F) << "Q " << i;
            }
            for (size_t i = 0; i < data.cache.count; ++i)
            {
                EXPECT_NEAR(__half2float(data.cache.host[i]), __half2float(expectedCache[i]), 0.001F) << "KV " << i;
            }
        }
    }
}

TEST(SafetyGpuRope, InvalidMetadataFlagGuardsNullDataPointers)
{
    RopeCase data(64, 2);
    data.valid.host[0] = 0;
    data.valid.upload(data.stream.value);
    data.error.upload(data.stream.value);
    dim3 const block = safetyRopeBlock(data.shape);
    safetyApplyRopeFromPackedToSplit<<<dim3(1, 6), block, 0, data.stream.value>>>(
        data.shape, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, data.valid.device, data.error.device);
    checked(cudaGetLastError());
    data.error.download(data.stream.value);
    checked(cudaStreamSynchronize(data.stream.value));
    EXPECT_EQ(data.error.host[0], 0U);
}

TEST(SafetyGpuRope, NonfiniteInputsAndHalfOverflowSetGraphErrorMask)
{
    for (int32_t failure = 0; failure < 4; ++failure)
    {
        SCOPED_TRACE(failure);
        RopeCase data(64, 16);
        int32_t const ropeRow = data.starts.host[0] * data.shape.rotaryDim;
        if (failure == 0)
        {
            data.packed.host[63] = __float2half(std::numeric_limits<float>::quiet_NaN());
        }
        else if (failure == 1)
        {
            data.packed.host[6 * 64] = __float2half(std::numeric_limits<float>::infinity());
        }
        else if (failure == 2)
        {
            data.rope.host[ropeRow] = std::numeric_limits<float>::infinity();
        }
        else
        {
            data.packed.host[0] = __float2half(65504.F);
            data.packed.host[8] = __float2half(-65504.F);
            data.rope.host[ropeRow] = 1.F;
            data.rope.host[ropeRow + 8] = 1.F;
        }
        data.run();
        EXPECT_EQ(data.error.host[0], 3U);
    }
}
} // namespace
