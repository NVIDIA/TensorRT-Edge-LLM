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

//! Nvfp4A16BlackwellMoePlugin creator contract: attribute set, validation and
//! serialization. Engine builds are covered by the Python plugin test on SM110.

#include "testPluginLoader.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{

using namespace nvinfer1;

constexpr char const* kName{"Nvfp4A16BlackwellMoePlugin"};
constexpr char const* kVersion{"1"};

struct Attrs
{
    int32_t numExperts{128};
    int32_t topK{6};
    int32_t hiddenSize{2688};
    int32_t moeInterSize{1856};
    int32_t activationType{4};
    int32_t nGroup{1};
    int32_t topkGroup{1};
    int32_t normTopkProb{1};
    float routedScalingFactor{2.5f};
    int32_t routingMode{1};
    int32_t maxRoutedRows{0};
    int32_t layout{1};
    int32_t backend{0};
};

std::vector<PluginField> makeFields(Attrs const& a)
{
    return {
        PluginField("num_experts", &a.numExperts, PluginFieldType::kINT32, 1),
        PluginField("top_k", &a.topK, PluginFieldType::kINT32, 1),
        PluginField("hidden_size", &a.hiddenSize, PluginFieldType::kINT32, 1),
        PluginField("moe_inter_size", &a.moeInterSize, PluginFieldType::kINT32, 1),
        PluginField("activation_type", &a.activationType, PluginFieldType::kINT32, 1),
        PluginField("n_group", &a.nGroup, PluginFieldType::kINT32, 1),
        PluginField("topk_group", &a.topkGroup, PluginFieldType::kINT32, 1),
        PluginField("norm_topk_prob", &a.normTopkProb, PluginFieldType::kINT32, 1),
        PluginField("routed_scaling_factor", &a.routedScalingFactor, PluginFieldType::kFLOAT32, 1),
        PluginField("routing_mode", &a.routingMode, PluginFieldType::kINT32, 1),
        PluginField("max_routed_rows", &a.maxRoutedRows, PluginFieldType::kINT32, 1),
        PluginField("layout", &a.layout, PluginFieldType::kINT32, 1),
        PluginField("backend", &a.backend, PluginFieldType::kINT32, 1),
    };
}

IPluginCreatorV3One* getCreator()
{
    if (test::loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    auto* creator = getPluginRegistry()->getCreator(kName, kVersion, "");
    return creator == nullptr ? nullptr : static_cast<IPluginCreatorV3One*>(creator);
}

IPluginV3* create(Attrs const& attrs)
{
    IPluginCreatorV3One* creator = getCreator();
    if (creator == nullptr)
    {
        return nullptr;
    }
    std::vector<PluginField> fields = makeFields(attrs);
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};
    return creator->createPlugin("moe", &fc, TensorRTPhase::kBUILD);
}

class Nvfp4A16BlackwellMoePluginTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (getCreator() == nullptr)
        {
            GTEST_SKIP() << "Nvfp4A16BlackwellMoePlugin creator is not registered (plugin library not found)";
        }
    }
};

TEST_F(Nvfp4A16BlackwellMoePluginTest, CreatorDeclaresThirteenAttributesAndTheJitBundle)
{
    PluginFieldCollection const* names = getCreator()->getFieldNames();
    ASSERT_NE(names, nullptr);
    // 13 ONNX attributes plus the runtime-only NVRTC bundle of the layer's
    // CUDA-core kernels (CHAR, never present on the ONNX node).
    ASSERT_EQ(names->nbFields, 14);
    std::vector<std::string> expected{"num_experts", "top_k", "hidden_size", "moe_inter_size", "activation_type",
        "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor", "routing_mode", "max_routed_rows", "layout",
        "backend", "moe_jit_bundle"};
    for (int32_t i = 0; i < names->nbFields; ++i)
    {
        EXPECT_EQ(std::string(names->fields[i].name), expected[i]) << i;
    }
    EXPECT_EQ(names->fields[13].type, PluginFieldType::kCHAR);
}

TEST_F(Nvfp4A16BlackwellMoePluginTest, CreatesNemotronContractAndSerializesAllAttributes)
{
    Attrs attrs{};
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    auto* core = static_cast<IPluginV3OneCore*>(plugin->getCapabilityInterface(PluginCapabilityType::kCORE));
    ASSERT_NE(core, nullptr);
    EXPECT_STREQ(core->getPluginName(), kName);
    EXPECT_STREQ(core->getPluginVersion(), kVersion);
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    ASSERT_NE(runtime, nullptr);
    PluginFieldCollection const* serialized = runtime->getFieldsToSerialize();
    ASSERT_NE(serialized, nullptr);
    // Before configurePlugin there is no JIT bundle yet: only the 13 attributes.
    EXPECT_EQ(serialized->nbFields, 13);
    for (int32_t i = 0; i < serialized->nbFields; ++i)
    {
        std::string const name(serialized->fields[i].name);
        EXPECT_NE(name, "moe_jit_bundle");
        if (name == "layout")
        {
            EXPECT_EQ(*static_cast<int32_t const*>(serialized->fields[i].data), 1);
        }
        if (name == "moe_inter_size")
        {
            EXPECT_EQ(*static_cast<int32_t const*>(serialized->fields[i].data), 1856);
        }
    }
    IPluginV3* cloned = plugin->clone();
    ASSERT_NE(cloned, nullptr);
    delete cloned;
    delete plugin;
}

TEST_F(Nvfp4A16BlackwellMoePluginTest, RejectsNonThorContracts)
{
    auto expectRejected = [](Attrs attrs, char const* label) {
        IPluginV3* plugin = create(attrs);
        EXPECT_EQ(plugin, nullptr) << label;
        delete plugin;
    };
    Attrs layout{};
    layout.layout = 0;
    expectRejected(layout, "marlin layout");
    Attrs backend{};
    backend.backend = 3;
    expectRejected(backend, "unknown backend");
    Attrs swiglu{};
    swiglu.activationType = 2;
    expectRejected(swiglu, "swiglu");
    Attrs softmax{};
    softmax.routingMode = 0;
    expectRejected(softmax, "softmax routing");
    Attrs badInter{};
    badInter.moeInterSize = 1800;
    expectRejected(badInter, "inter not multiple of 64");
    Attrs badHidden{};
    badHidden.hiddenSize = 2700;
    expectRejected(badHidden, "hidden not multiple of 128");
    Attrs badExperts{};
    badExperts.numExperts = 64;
    expectRejected(badExperts, "num_experts");
    Attrs badTopK{};
    badTopK.topK = 33;
    expectRejected(badTopK, "top_k");
    Attrs badScale{};
    badScale.routedScalingFactor = 0.0f;
    expectRejected(badScale, "routed_scaling_factor");
    Attrs badGroup{};
    badGroup.nGroup = 3;
    expectRejected(badGroup, "n_group");
}

TEST_F(Nvfp4A16BlackwellMoePluginTest, AcceptsForcedBackendsAndExplicitCapacity)
{
    for (int32_t backend : {0, 1, 2})
    {
        Attrs attrs{};
        attrs.backend = backend;
        attrs.maxRoutedRows = 28544;
        IPluginV3* plugin = create(attrs);
        EXPECT_NE(plugin, nullptr) << "backend=" << backend;
        delete plugin;
    }
}

} // namespace
} // namespace trt_edgellm::plugins
