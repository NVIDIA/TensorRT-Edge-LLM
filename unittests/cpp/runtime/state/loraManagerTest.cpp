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

#include "runtime/state/loraManager.h"

#include "common/checkMacros.h"
#include "testUtils.h"
#include <gtest/gtest.h>

#include <algorithm>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

class LoRAManagerTest : public ::testing::Test
{
protected:
    //! Helper to create a simple GPU tensor with the given shape and a name.
    static rt::Tensor makeGpuTensor(rt::Coords const& shape, std::string const& name)
    {
        return rt::Tensor(shape, DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
    }

    //! Build a weight map with two bindings for testing.
    static std::map<std::string, rt::Tensor> buildTestWeights(std::string const& prefix)
    {
        std::map<std::string, rt::Tensor> weights;
        weights.emplace("lora_A_layer_0", makeGpuTensor({16, 4}, prefix + "_A0"));
        weights.emplace("lora_B_layer_0", makeGpuTensor({4, 16}, prefix + "_B0"));
        return weights;
    }
};

TEST_F(LoRAManagerTest, InitialState)
{
    LoRAManager mgr;
    EXPECT_FALSE(mgr.hasActiveAdapter());
    EXPECT_TRUE(mgr.getActiveAdapterName().empty());
    EXPECT_TRUE(mgr.getBindingNames().empty());
    EXPECT_TRUE(mgr.getAdapterNames().empty());
}

TEST_F(LoRAManagerTest, AddAndSwitchWeights)
{
    LoRAManager mgr;

    mgr.addWeights("adapter1", buildTestWeights("a1"));
    EXPECT_FALSE(mgr.hasActiveAdapter());

    mgr.switchWeights("adapter1");
    EXPECT_TRUE(mgr.hasActiveAdapter());
    EXPECT_EQ(mgr.getActiveAdapterName(), "adapter1");

    // Retrieve active weight and verify it is the expected tensor.
    rt::Tensor& w = mgr.getActiveWeight("lora_A_layer_0");
    EXPECT_FALSE(w.isEmpty());
    EXPECT_EQ(w.getShape()[0], 16);
    EXPECT_EQ(w.getShape()[1], 4);
}

TEST_F(LoRAManagerTest, SwitchBetweenAdapters)
{
    LoRAManager mgr;

    mgr.addWeights("adapter1", buildTestWeights("a1"));

    // Create adapter2 with a different shape to distinguish them.
    std::map<std::string, rt::Tensor> weights2;
    weights2.emplace("lora_A_layer_0", makeGpuTensor({32, 8}, "a2_A0"));
    weights2.emplace("lora_B_layer_0", makeGpuTensor({8, 32}, "a2_B0"));
    mgr.addWeights("adapter2", std::move(weights2));

    mgr.switchWeights("adapter1");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], 16);

    mgr.switchWeights("adapter2");
    EXPECT_EQ(mgr.getActiveAdapterName(), "adapter2");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], 32);
}

TEST_F(LoRAManagerTest, ResetWeightsReturnsDummy)
{
    LoRAManager mgr;
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");
    EXPECT_TRUE(mgr.hasActiveAdapter());

    mgr.resetWeights();
    EXPECT_FALSE(mgr.hasActiveAdapter());
    EXPECT_TRUE(mgr.getActiveAdapterName().empty());

    // After reset, getActiveWeight should return a dummy tensor.
    rt::Tensor& dummy = mgr.getActiveWeight("lora_A_layer_0");
    // The dummy is a small tensor of shape [1].
    EXPECT_EQ(dummy.getShape().volume(), 1);
}

TEST_F(LoRAManagerTest, SwitchToUnknownAdapterThrows)
{
    LoRAManager mgr;
    EXPECT_THROW(mgr.switchWeights("nonexistent"), std::runtime_error);
}

TEST_F(LoRAManagerTest, GetActiveWeightMissingBindingThrows)
{
    LoRAManager mgr;
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");

    EXPECT_THROW(mgr.getActiveWeight("nonexistent_binding"), std::runtime_error);
}

TEST_F(LoRAManagerTest, GetBindingNamesCompleteness)
{
    LoRAManager mgr;

    // Adapter1 has bindings A and B.
    mgr.addWeights("adapter1", buildTestWeights("a1"));

    // Adapter2 has bindings A, B, and an extra C.
    auto weights2 = buildTestWeights("a2");
    weights2.emplace("lora_C_layer_0", makeGpuTensor({8, 8}, "a2_C0"));
    mgr.addWeights("adapter2", std::move(weights2));

    auto const names = mgr.getBindingNames();

    // Should contain the union of all binding names.
    EXPECT_EQ(names.size(), 3U);
    EXPECT_TRUE(std::find(names.begin(), names.end(), "lora_A_layer_0") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "lora_B_layer_0") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "lora_C_layer_0") != names.end());
}

TEST_F(LoRAManagerTest, GetAdapterNames)
{
    LoRAManager mgr;
    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.addWeights("adapter2", buildTestWeights("a2"));

    auto const adapterNames = mgr.getAdapterNames();
    EXPECT_EQ(adapterNames.size(), 2U);
    EXPECT_TRUE(std::find(adapterNames.begin(), adapterNames.end(), "adapter1") != adapterNames.end());
    EXPECT_TRUE(std::find(adapterNames.begin(), adapterNames.end(), "adapter2") != adapterNames.end());
}

TEST_F(LoRAManagerTest, DummyTensorReturnedWhenNoAdapterActive)
{
    LoRAManager mgr;

    // Even without any adapters loaded, getActiveWeight should return a dummy.
    rt::Tensor& dummy = mgr.getActiveWeight("any_binding");
    EXPECT_EQ(dummy.getShape().volume(), 1);
    EXPECT_EQ(dummy.getDeviceType(), DeviceType::kGPU);
}

TEST_F(LoRAManagerTest, OverwriteAdapter)
{
    LoRAManager mgr;

    mgr.addWeights("adapter1", buildTestWeights("a1"));
    mgr.switchWeights("adapter1");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], 16);

    // Overwrite with different shapes.
    std::map<std::string, rt::Tensor> newWeights;
    newWeights.emplace("lora_A_layer_0", makeGpuTensor({64, 8}, "a1_new_A0"));
    newWeights.emplace("lora_B_layer_0", makeGpuTensor({8, 64}, "a1_new_B0"));
    mgr.addWeights("adapter1", std::move(newWeights));

    // Still active, and should reflect the new weights.
    mgr.switchWeights("adapter1");
    EXPECT_EQ(mgr.getActiveWeight("lora_A_layer_0").getShape()[0], 64);
}
