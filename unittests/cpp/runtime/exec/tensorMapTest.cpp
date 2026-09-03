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

#include "runtime/exec/tensorMap.h"
#include "common/tensor.h"
#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

TEST(TensorMapTest, SetAndGet)
{
    TensorMap map;
    Tensor t({4, 128, 4096}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    map.set("inputs_embeds", t);
    ASSERT_NE(map.get("inputs_embeds"), nullptr);
    EXPECT_EQ(map.get("inputs_embeds")->rawPointer(), t.rawPointer());
}

TEST(TensorMapTest, GetMissingReturnsNull)
{
    TensorMap map;
    EXPECT_EQ(map.get("nonexistent"), nullptr);
}

TEST(TensorMapTest, ContainsWorks)
{
    TensorMap map;
    Tensor t({4}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    map.set("context_lengths", t);
    EXPECT_TRUE(map.contains("context_lengths"));
    EXPECT_FALSE(map.contains("other"));
}

TEST(TensorMapTest, SetOverwritesExisting)
{
    TensorMap map;
    Tensor t1({4}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    Tensor t2({8}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    map.set("x", t1);
    map.set("x", t2);
    EXPECT_EQ(map.get("x")->rawPointer(), t2.rawPointer());
}

TEST(TensorMapTest, AllNames)
{
    TensorMap map;
    Tensor t1({4}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    Tensor t2({4}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    map.set("a", t1);
    map.set("b", t2);
    auto names = map.allNames();
    EXPECT_EQ(names.size(), 2u);
}
