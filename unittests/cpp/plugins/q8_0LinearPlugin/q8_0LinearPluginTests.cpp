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

//! Q8_0LinearPlugin creator contract for the serialized activation workspace layout.

#include "kernels/q8_0/q8_0Linear.h"
#include "testPluginLoader.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace trt_edgellm::plugins
{
namespace
{

using namespace nvinfer1;

constexpr char const* kVersion{"1"};
constexpr char const* kLayoutField{"activation_layout"};

struct PackedLinearCase
{
    char const* pluginName;
    int32_t layoutVersion;
};

IPluginCreatorV3One* getCreator(char const* pluginName)
{
    if (test::loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    auto* creator = getPluginRegistry()->getCreator(pluginName, kVersion, "");
    return creator == nullptr ? nullptr : static_cast<IPluginCreatorV3One*>(creator);
}

class PackedLinearPluginLayoutTest : public ::testing::TestWithParam<PackedLinearCase>
{
protected:
    void SetUp() override
    {
        if (getCreator(GetParam().pluginName) == nullptr)
        {
            GTEST_SKIP() << GetParam().pluginName << " creator is not registered (plugin library not found)";
        }
    }
};

} // namespace

// Given a packed-linear plugin record deserialized from an engine built before its activation workspace layout was
// recorded (no activation_layout field) or with an older layout value,
// When the creator rebuilds the plugin for the runtime phase,
// Then creation fails instead of returning a plugin that would write past the smaller workspace the engine reserved.
TEST_P(PackedLinearPluginLayoutTest, RuntimeCreationRejectsEnginesWithoutCurrentActivationLayout)
{
    IPluginCreatorV3One* creator = getCreator(GetParam().pluginName);
    PluginFieldCollection const noFields{0, nullptr};
    EXPECT_EQ(creator->createPlugin("packed", &noFields, TensorRTPhase::kRUNTIME), nullptr);

    int32_t const olderLayout = GetParam().layoutVersion - 1;
    PluginField const olderField(kLayoutField, &olderLayout, PluginFieldType::kINT32, 1);
    PluginFieldCollection const olderFields{1, &olderField};
    EXPECT_EQ(creator->createPlugin("packed", &olderFields, TensorRTPhase::kRUNTIME), nullptr);
}

// Given a packed-linear plugin created for the build phase from an ONNX node without attributes,
// When its serialized fields are handed back to the creator for the runtime phase,
// Then they carry the current activation layout of that format and the runtime plugin is created.
TEST_P(PackedLinearPluginLayoutTest, BuildPluginSerializesActivationLayoutThatRuntimeAccepts)
{
    IPluginCreatorV3One* creator = getCreator(GetParam().pluginName);
    PluginFieldCollection const noFields{0, nullptr};
    std::unique_ptr<IPluginV3> built(creator->createPlugin("packed", &noFields, TensorRTPhase::kBUILD));
    ASSERT_NE(built, nullptr);
    auto* runtime = static_cast<IPluginV3OneRuntime*>(built->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    ASSERT_NE(runtime, nullptr);
    PluginFieldCollection const* serialized = runtime->getFieldsToSerialize();
    ASSERT_NE(serialized, nullptr);

    int32_t layout = -1;
    for (int32_t i = 0; i < serialized->nbFields; ++i)
    {
        PluginField const& field = serialized->fields[i];
        if (std::strcmp(field.name, kLayoutField) == 0 && field.type == PluginFieldType::kINT32 && field.length == 1)
        {
            layout = *static_cast<int32_t const*>(field.data);
        }
    }
    EXPECT_EQ(layout, GetParam().layoutVersion);

    std::unique_ptr<IPluginV3> restored(creator->createPlugin("packed", serialized, TensorRTPhase::kRUNTIME));
    EXPECT_NE(restored, nullptr);
}

INSTANTIATE_TEST_SUITE_P(PackedLinearFormats, PackedLinearPluginLayoutTest,
    ::testing::Values(PackedLinearCase{"Q8_0LinearPlugin", kernel::kQ8_0_ACTIVATION_LAYOUT_VERSION}),
    [](::testing::TestParamInfo<PackedLinearCase> const& info) { return std::string(info.param.pluginName); });

} // namespace trt_edgellm::plugins
