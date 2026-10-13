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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "testPluginLoader.h"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <tuple>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;
class Nvfp4A16BlackwellRankTest : public ::testing::TestWithParam<std::tuple<int32_t, DataType>>
{
};

TEST_P(Nvfp4A16BlackwellRankTest, AcceptsPackedAndBatchedFormats)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    auto* creator
        = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("Nvfp4A16BlackwellGemmPlugin", "1", ""));
    ASSERT_NE(creator, nullptr);
    int32_t const n = 128;
    int32_t const k = 64;
    int32_t const maxM = 0;
    int32_t const layout = 1;
    int32_t const backend = 0;
    std::array<PluginField, 5> const fields{PluginField("gemm_n", &n, PluginFieldType::kINT32, 1),
        PluginField("gemm_k", &k, PluginFieldType::kINT32, 1), PluginField("max_m", &maxM, PluginFieldType::kINT32, 1),
        PluginField("layout", &layout, PluginFieldType::kINT32, 1),
        PluginField("backend", &backend, PluginFieldType::kINT32, 1)};
    PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
    std::unique_ptr<IPluginV3> plugin(creator->createPlugin("rank_contract", &fc, TensorRTPhase::kBUILD));
    ASSERT_NE(plugin, nullptr);
    auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    ASSERT_NE(build, nullptr);
    auto const [rank, type] = GetParam();
    std::array<DynamicPluginTensorDesc, 5> desc{};
    for (auto& tensor : desc)
    {
        tensor.desc.format = TensorFormat::kLINEAR;
    }
    desc[0].desc.type = desc[4].desc.type = type;
    desc[0].desc.dims = rank == 2 ? Dims{2, {8, k}} : Dims{3, {2, 4, k}};
    desc[4].desc.dims = rank == 2 ? Dims{2, {8, n}} : Dims{3, {2, 4, n}};
    desc[1].desc.type = desc[2].desc.type = DataType::kINT8;
    desc[1].desc.dims = Dims{4, {n / 128, k / 64, 128, 32}};
    desc[2].desc.dims = Dims{4, {n / 128, k / 64, 128, 4}};
    desc[3].desc.type = DataType::kFLOAT;
    desc[3].desc.dims = Dims{1, {1}};
    for (int32_t pos = 0; pos < static_cast<int32_t>(desc.size()); ++pos)
    {
        EXPECT_TRUE(build->supportsFormatCombination(pos, desc.data(), 4, 1)) << "position=" << pos;
    }
}

INSTANTIATE_TEST_SUITE_P(PackedAndBatched, Nvfp4A16BlackwellRankTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(DataType::kHALF, DataType::kBF16)));
} // namespace
} // namespace trt_edgellm::plugins
