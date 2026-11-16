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
#include "common/trtUtils.h"
#include "kernels/compactPhase/compactPhaseSupport.h"
#include "testPluginLoader.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>

using namespace trt_edgellm;

namespace
{

using BeginCompactBuildReportFn = uint64_t (*)();
using EndCompactBuildReportFn = bool (*)(uint64_t);

constexpr int32_t kChannels{64};
constexpr int32_t kWidth{4};
constexpr int32_t kMaxBatch{4};

class CompactBuildReportTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        void* const library = test::loadPluginLibrary();
        ASSERT_NE(library, nullptr) << test::pluginLoadError();
        mBegin = reinterpret_cast<BeginCompactBuildReportFn>(
            getDynamicLibrarySymbol(library, "edgellmBeginCompactBuildReport"));
        mEnd = reinterpret_cast<EndCompactBuildReportFn>(
            getDynamicLibrarySymbol(library, "edgellmEndCompactBuildReport"));
        ASSERT_NE(mBegin, nullptr);
        ASSERT_NE(mEnd, nullptr);
        mCreator
            = static_cast<nvinfer1::IPluginCreatorV3One*>(getPluginRegistry()->getCreator("causal_conv1d", "1", ""));
        ASSERT_NE(mCreator, nullptr);
    }

    //! A causal_conv1d layer @p name of kChannels channels and width kWidth, with spec-verify state or depthwise.
    std::unique_ptr<nvinfer1::IPluginV3> makeConv(char const* name, bool specVerify)
    {
        int32_t const verify = specVerify ? 1 : 0;
        int32_t const padding = kWidth - 1;
        std::array<nvinfer1::PluginField, 2> const fields{{{"padding", &padding, nvinfer1::PluginFieldType::kINT32, 1},
            {"use_spec_verify_state", &verify, nvinfer1::PluginFieldType::kINT32, 1}}};
        nvinfer1::PluginFieldCollection const collection{static_cast<int32_t>(fields.size()), fields.data()};
        return std::unique_ptr<nvinfer1::IPluginV3>(
            mCreator->createPlugin(name, &collection, nvinfer1::TensorRTPhase::kBUILD));
    }

    //! The build-time workspace query TensorRT makes for @p plugin while it builds an engine.
    static void queryWorkspace(nvinfer1::IPluginV3& plugin, bool specVerify)
    {
        auto* const build = static_cast<nvinfer1::IPluginV3OneBuild*>(
            plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
        std::array<nvinfer1::DynamicPluginTensorDesc, 12> inputs{};
        auto const setMax = [&inputs](int32_t index, std::initializer_list<int64_t> shape) {
            nvinfer1::Dims& dims = inputs[index].max;
            dims.nbDims = static_cast<int32_t>(shape.size());
            std::copy(shape.begin(), shape.end(), dims.d);
        };
        setMax(0, {kMaxBatch * 16, kChannels});
        setMax(1, {kChannels, 1, kWidth});
        setMax(3, {kMaxBatch, kChannels, kWidth - 1});
        setMax(4, {kMaxBatch});
        int32_t const nbInputs = specVerify ? 12 : 9;
        int32_t const nbOutputs = specVerify ? 3 : 2;
        build->getWorkspaceSize(inputs.data(), nbInputs, nullptr, nbOutputs);
    }

    BeginCompactBuildReportFn mBegin{};
    EndCompactBuildReportFn mEnd{};
    nvinfer1::IPluginCreatorV3One* mCreator{};
};

} // namespace

// Given the plugin library's compact build report and causal_conv1d layers: "a", depthwise (64 channels, width 4), "v",
//       with spec-verify state, which has no compact path on any SM, and "a2", a spec-verify layer that shares a's name
// When reports end around no workspace query; a's workspace queried twice; one query each of a and v; one query each
//      of a and a2; two reports that overlap around a's query; a report after them around a's query; and when an ended
//      report is ended again
// Then only the reports where every queried layer reserved, and at least one was queried, find the engine built for
//      compact execution: the second and the one after the overlapping reports, exactly when causal_conv1d accepts a's
//      shape on this SM; a layer counts as reserved only if every query of its name reserved; neither overlapping
//      report does; and an ended report does not
TEST_F(CompactBuildReportTest, FindsAnEngineBuiltForCompactOnlyWhenEveryLayerOfOneBuildReservedTheCompactWorkspace)
{
    bool const aReserves
        = causalConv1dCompactSupport(getSMVersion(), kChannels, kWidth, /*dilation=*/1, /*groups=*/0, false).supported;
    auto const a = makeConv("a", /*specVerify=*/false);
    auto const v = makeConv("v", /*specVerify=*/true);
    auto const a2 = makeConv("a", /*specVerify=*/true);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(v, nullptr);
    ASSERT_NE(a2, nullptr);

    EXPECT_FALSE(mEnd(mBegin()));

    uint64_t const once = mBegin();
    queryWorkspace(*a, false);
    queryWorkspace(*a, false);
    EXPECT_EQ(mEnd(once), aReserves);

    uint64_t const both = mBegin();
    queryWorkspace(*a, false);
    queryWorkspace(*v, true);
    EXPECT_FALSE(mEnd(both));

    uint64_t const sameName = mBegin();
    queryWorkspace(*a, false);
    queryWorkspace(*a2, true);
    EXPECT_FALSE(mEnd(sameName));

    uint64_t const first = mBegin();
    uint64_t const second = mBegin();
    queryWorkspace(*a, false);
    EXPECT_FALSE(mEnd(first));
    EXPECT_FALSE(mEnd(second));

    uint64_t const after = mBegin();
    queryWorkspace(*a, false);
    EXPECT_EQ(mEnd(after), aReserves);
    EXPECT_FALSE(mEnd(after));
}
