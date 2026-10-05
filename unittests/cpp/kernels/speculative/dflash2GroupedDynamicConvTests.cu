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
#include "kernels/speculative/dflash2GroupedDynamicConv.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

namespace
{
std::vector<half> asHalf(std::vector<float> const& input)
{
    std::vector<half> result(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        result[i] = __float2half(input[i]);
    }
    return result;
}
} // namespace

TEST(DFlash2GroupedDynamicConv, K2Group16HonorsEveryBlockBoundary)
{
    constexpr int32_t tokens = 16;
    constexpr int32_t hidden = 32;
    constexpr int32_t block = 8;
    constexpr int32_t taps = 2;
    constexpr int32_t group = 16;
    constexpr int32_t groups = hidden / group;

    std::vector<float> input(tokens * hidden);
    std::vector<float> base(taps * hidden);
    std::vector<float> delta(tokens * taps * groups);
    for (int32_t t = 0; t < tokens; ++t)
    {
        for (int32_t c = 0; c < hidden; ++c)
        {
            input[t * hidden + c] = 0.01F * static_cast<float>(1 + t * hidden + c);
            base[c] = 0.5F + 0.001F * c;
            base[hidden + c] = -0.25F + 0.002F * c;
        }
        for (int32_t tap = 0; tap < taps; ++tap)
        {
            for (int32_t g = 0; g < groups; ++g)
            {
                delta[(t * taps + tap) * groups + g] = 0.02F * (t + tap + g);
            }
        }
    }

    rt::Tensor x({tokens / block, block, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor d({tokens / block, block, taps, groups}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor b({taps, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor y({tokens / block, block, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice<half>(x, asHalf(input));
    copyHostToDevice<half>(d, asHalf(delta));
    copyHostToDevice<half>(b, asHalf(base));

    CUDA_CHECK(launchDFlash2GroupedDynamicConv(x, d, b, rt::OptionalInputTensor{}, y, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const actual = copyDeviceToHost<half>(y);
    for (int32_t t = 0; t < tokens; ++t)
    {
        for (int32_t c = 0; c < hidden; ++c)
        {
            int32_t const g = c / group;
            float expected = (base[c] + delta[(t * taps) * groups + g]) * input[t * hidden + c];
            if ((t % block) != 0)
            {
                expected += (base[hidden + c] + delta[(t * taps + 1) * groups + g]) * input[(t - 1) * hidden + c];
            }
            EXPECT_NEAR(__half2float(actual[t * hidden + c]), expected, 1.5e-2F) << "t=" << t << " c=" << c;
        }
    }
}

TEST(DFlash2GroupedDynamicConv, Fp16WritebackSaturatesInsteadOfProducingInfinity)
{
    constexpr int32_t tokens = 1;
    constexpr int32_t hidden = 16;
    constexpr int32_t taps = 2;
    constexpr int32_t group = 16;
    rt::Tensor x({1, tokens, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor d({1, tokens, taps, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor b({taps, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor y({1, tokens, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice<half>(x, asHalf(std::vector<float>(hidden, 60000.0F)));
    copyHostToDevice<half>(d, asHalf(std::vector<float>(taps, 0.0F)));
    std::vector<float> base(taps * hidden, 0.0F);
    std::fill(base.begin(), base.begin() + hidden, 2.0F);
    copyHostToDevice<half>(b, asHalf(base));

    CUDA_CHECK(launchDFlash2GroupedDynamicConv(x, d, b, rt::OptionalInputTensor{}, y, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (half value : copyDeviceToHost<half>(y))
    {
        EXPECT_TRUE(std::isfinite(__half2float(value)));
        EXPECT_FLOAT_EQ(__half2float(value), 65504.0F);
    }
}

TEST(DFlash2GroupedDynamicConv, PostConvFusesResidualInFp32)
{
    constexpr int32_t tokens = 1;
    constexpr int32_t hidden = 16;
    constexpr int32_t taps = 2;
    rt::Tensor x({1, tokens, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor d({1, tokens, taps, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor b({taps, hidden}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor residual({1, tokens, hidden}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor y({1, tokens, hidden}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<half>(x, asHalf(std::vector<float>(hidden, 1.0F)));
    copyHostToDevice<half>(d, asHalf(std::vector<float>(taps, 0.0F)));
    std::vector<float> base(taps * hidden, 0.0F);
    std::fill(base.begin(), base.begin() + hidden, 2.0F);
    copyHostToDevice<half>(b, asHalf(base));
    copyHostToDevice<float>(residual, std::vector<float>(hidden, 70000.0F));

    CUDA_CHECK(launchDFlash2GroupedDynamicConv(x, d, b, rt::OptionalInputTensor{residual}, y, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (float value : copyDeviceToHost<float>(y))
    {
        EXPECT_FLOAT_EQ(value, 70002.0F);
    }
}
