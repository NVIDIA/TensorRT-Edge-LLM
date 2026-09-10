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

#include "runtime/exec/tensorRegistry.h"
#include "common/tensor.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/exec/tensorMap.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

TEST(TensorRegistryTest, ResolveShapeAllFixed)
{
    TensorRegistry reg;
    reg.addTensor({"logits", TensorIO::kOutput, nvinfer1::DataType::kFLOAT, {fixed(4), fixed(32000)}});
    auto specs = reg.allExpandedSpecs();
    ASSERT_EQ(specs.size(), 1u);
    EXPECT_EQ(specs[0].name, "logits");

    InferenceDims const dims{/*.batch=*/0, /*.seqLen=*/0, /*.kvLen=*/0, /*.selectLen=*/0, /*.attnMaskSeqLen=*/0,
        /*.ropeBatch=*/0, /*.packedMaskLen=*/0, /*.contextMaskSelectorLen=*/0, /*.startIndexLen=*/0,
        /*.specVerifyPhaseLen=*/0};
    auto resolved = reg.resolveShape(specs[0].shape, dims);
    EXPECT_EQ(resolved.nbDims, 2);
    EXPECT_EQ(resolved.d[0], 4);
    EXPECT_EQ(resolved.d[1], 32000);
}

TEST(TensorRegistryTest, ResolveShapeWithSymbolicDims)
{
    TensorRegistry reg;
    reg.addTensor({"inputs_embeds", TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(4096)}});

    InferenceDims const dims{/*.batch=*/4, /*.seqLen=*/128, /*.kvLen=*/1, /*.selectLen=*/1, /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1, /*.packedMaskLen=*/1, /*.contextMaskSelectorLen=*/0, /*.startIndexLen=*/4,
        /*.specVerifyPhaseLen=*/0};
    auto specs = reg.allExpandedSpecs();
    auto resolved = reg.resolveShape(specs[0].shape, dims);
    EXPECT_EQ(resolved.nbDims, 3);
    EXPECT_EQ(resolved.d[0], 4);
    EXPECT_EQ(resolved.d[1], 128);
    EXPECT_EQ(resolved.d[2], 4096);
}

TEST(TensorRegistryTest, PerLayerExpansion)
{
    TensorRegistry reg;
    reg.addTensor({"past_key_values_%d", TensorIO::kInput, nvinfer1::DataType::kHALF,
        {fixed(4), fixed(2), fixed(8), fixed(2048), fixed(128)},
        /*perLayer=*/3});
    auto specs = reg.allExpandedSpecs();
    ASSERT_EQ(specs.size(), 3u);
    EXPECT_EQ(specs[0].name, "past_key_values_0");
    EXPECT_EQ(specs[1].name, "past_key_values_1");
    EXPECT_EQ(specs[2].name, "past_key_values_2");
}

TEST(TensorRegistryTest, AllTensorNames)
{
    TensorRegistry reg;
    reg.addTensor({"inputs_embeds", TensorIO::kInput, nvinfer1::DataType::kHALF, {fixed(4), fixed(128), fixed(4096)}});
    reg.addTensor({"kv_%d", TensorIO::kInput, nvinfer1::DataType::kHALF, {fixed(4)}, /*perLayer=*/2});
    auto names = reg.allTensorNames();
    ASSERT_EQ(names.size(), 3u); // inputs_embeds, kv_0, kv_1
}

TEST(TensorRegistryTest, ReferencedMembersDedupedAcrossTensors)
{
    TensorRegistry reg;
    // Two tensors that both reference batch; batch should appear once in the set.
    reg.addTensor(
        {"a", TensorIO::kInput, nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen)}});
    reg.addTensor(
        {"b", TensorIO::kInput, nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::kvLen)}});

    auto const& refs = reg.referencedMembers();
    ASSERT_EQ(refs.size(), 3u);
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::batch), refs.end());
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::seqLen), refs.end());
    EXPECT_NE(std::find(refs.begin(), refs.end(), &InferenceDims::kvLen), refs.end());
}

TEST(TensorRegistryTest, ReferencedMembersAllFixedIsEmpty)
{
    TensorRegistry reg;
    reg.addTensor({"logits", TensorIO::kOutput, nvinfer1::DataType::kFLOAT, {fixed(4), fixed(32000)}});
    EXPECT_TRUE(reg.referencedMembers().empty());
}
