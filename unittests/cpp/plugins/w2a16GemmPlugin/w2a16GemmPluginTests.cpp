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

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
#include "kernels/gemm/w2a16_cutedsl/cuteDslW2A16GemmRunner.h"
#include "testPluginLoader.h"
#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;
// Plugin descriptors take TensorRT types, not trt_edgellm::DataType.
using nvinfer1::DataType;
class W2A16GemmPluginFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!test::loadPluginLibrary())
        {
            throw std::runtime_error("Cannot load Edge-LLM plugin library");
        }
        mCreator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("W2A16GemmPlugin", "1", ""));
        if (!mCreator)
        {
            throw std::runtime_error("W2A16 creator is not registered");
        }
    }

    std::unique_ptr<IPluginV3> makePlugin(int32_t n, int32_t k)
    {
        std::array<PluginField, 2> fields{PluginField{"gemm_n", &n, PluginFieldType::kINT32, 1},
            PluginField{"gemm_k", &k, PluginFieldType::kINT32, 1}};
        PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};
        std::unique_ptr<IPluginV3> result(mCreator->createPlugin("dense", &fc, TensorRTPhase::kBUILD));
        if (!result)
        {
            throw std::runtime_error("Cannot create W2A16 plugin");
        }
        return result;
    }

    IPluginCreatorV3One* mCreator{};
};

class W2A16GemmFieldsTest : public W2A16GemmPluginFixture, public ::testing::WithParamInterface<int32_t>
{
};
TEST_P(W2A16GemmFieldsTest, RejectsMalformedFields)
{
    // Given missing, duplicate, mistyped or unsupported dense attributes,
    // when a plugin is created, then creation fails instead of silently accepting a different operation.
    int32_t n = 128, k = 4096;
    std::array<PluginField, 2> fields{
        PluginField{"gemm_n", &n, PluginFieldType::kINT32, 1}, PluginField{"gemm_k", &k, PluginFieldType::kINT32, 1}};
    PluginFieldCollection fc{2, fields.data()};
    switch (GetParam())
    {
    case 0: fc.nbFields = 1; break;
    case 1: fields[1].name = "gemm_n"; break;
    case 2: fields[0].type = PluginFieldType::kFLOAT32; break;
    case 3: fields[0].data = nullptr; break;
    case 4: fields[0].length = 2; break;
    case 5: fields[0].name = "unknown"; break;
    case 6: n = 127; break;
    case 7: k = 255; break;
    case 8: n = 0; break;
    case 9: n = 4194304; break;
    }
    std::unique_ptr<IPluginV3> result(mCreator->createPlugin("invalid", &fc, TensorRTPhase::kBUILD));
    EXPECT_EQ(result, nullptr);
}
INSTANTIATE_TEST_SUITE_P(Attributes, W2A16GemmFieldsTest, ::testing::Range(0, 10));

class W2A16GemmProfileTest : public W2A16GemmPluginFixture, public ::testing::WithParamInterface<int32_t>
{
};
TEST_P(W2A16GemmProfileTest, ValidatesFormatsAndReservesIntermediateScratch)
{
    // Given a rank-2 or rank-3 dynamic profile whose largest M needs no split-K scratch,
    // when formats and profile bounds are checked, then only matching tensors are accepted
    // and workspace covers every intermediate M, including the split-K cases.
    int32_t const rank = GetParam(), n = 128, k = 4096;
    auto plugin = makePlugin(n, k);
    auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    std::array<DynamicPluginTensorDesc, 5> d{};
    d[0].desc = {rank == 2 ? Dims{2, {-1, k}} : Dims{3, {-1, -1, k}}, DataType::kBF16, TensorFormat::kLINEAR};
    d[1].desc = {Dims{4, {1, 16, 128, 64}}, DataType::kINT8, TensorFormat::kLINEAR};
    d[2].desc = {Dims{2, {16, 128}}, DataType::kHALF, TensorFormat::kLINEAR};
    d[3].desc = {Dims{2, {256, 8}}, DataType::kBF16, TensorFormat::kLINEAR};
    d[4].desc = d[0].desc;
    d[4].desc.dims.d[rank - 1] = n;
    for (int32_t i = 0; i < 5; ++i)
    {
        d[i].min = d[i].max = d[i].desc.dims;
        if (i == 0 || i == 4)
        {
            d[i].min.d[0] = 1;
            d[i].max.d[0] = rank == 2 ? 32768 : 2;
            if (rank == 3)
            {
                d[i].min.d[1] = 1;
                d[i].max.d[1] = 16384;
            }
        }
        EXPECT_TRUE(build->supportsFormatCombination(i, d.data(), 4, 1));
        auto const saved = d[i].desc.type;
        d[i].desc.type = DataType::kFLOAT;
        EXPECT_FALSE(build->supportsFormatCombination(i, d.data(), 4, 1));
        d[i].desc.type = saved;
    }
    EXPECT_EQ(build->configurePlugin(d.data(), 4, d.data() + 4, 1), 0);
    size_t expected = 0;
    for (int32_t m = 1; m <= 32768; ++m)
    {
        expected = std::max(expected, CuteDslW2A16GemmRunner::getWorkspaceSize(m, n, k));
    }
    EXPECT_EQ(build->getWorkspaceSize(d.data(), 4, d.data() + 4, 1), expected);
    d[0].max.d[0] += 1;
    EXPECT_NE(build->configurePlugin(d.data(), 4, d.data() + 4, 1), 0);
    d[0].max.d[0] -= 1;
    d[1].desc.dims.d[3] = 63;
    EXPECT_FALSE(build->supportsFormatCombination(1, d.data(), 4, 1));
    d[1].max.d[3] = 63;
    EXPECT_NE(build->configurePlugin(d.data(), 4, d.data() + 4, 1), 0);
}
INSTANTIATE_TEST_SUITE_P(Ranks, W2A16GemmProfileTest, ::testing::Values(2, 3));
} // namespace
} // namespace trt_edgellm::plugins
#endif
