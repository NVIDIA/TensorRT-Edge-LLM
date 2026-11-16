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

#include "common/cudaMacros.h"
#include "common/safetensorsUtils.h"
#include "runtime/llmRuntimeUtils.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

class EmbeddingDataTest : public ::testing::Test
{
protected:
    static constexpr int32_t kVOCAB_SIZE{4};
    static constexpr int64_t kHIDDEN_SIZE{32};
    static constexpr int64_t kGROUP_SIZE{16};

    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mPath = std::filesystem::temp_directory_path() / "edgellm_nvfp4_embedding_test.safetensors";
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::error_code error;
        std::filesystem::remove(mPath, error);
    }

    nlohmann::json descriptor() const
    {
        return {
            {"logical_shape", {kVOCAB_SIZE, kHIDDEN_SIZE}},
            {"group_size", kGROUP_SIZE},
            {"model_output_scale", 1.25F},
        };
    }

    bool writeNvfp4Embedding(nlohmann::json const& descriptorValue, float multiplierScale = 0.5F)
    {
        rt::Tensor table({kVOCAB_SIZE, kHIDDEN_SIZE / 2}, rt::DeviceType::kCPU, trt_edgellm::DataType::kUINT8,
            rt::kEmbeddingTensorName);
        rt::Tensor blockScales({kVOCAB_SIZE, kHIDDEN_SIZE / kGROUP_SIZE}, rt::DeviceType::kCPU,
            trt_edgellm::DataType::kFP8, rt::kNvfp4EmbeddingBlockScaleTensorName);
        rt::Tensor multiplierScaleTensor(
            {1}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, rt::kNvfp4EmbeddingMultiplierScaleTensorName);
        std::fill_n(table.dataPointer<uint8_t>(), table.getShape().volume(), 0x21U);
        std::fill_n(static_cast<uint8_t*>(blockScales.rawPointer()), blockScales.getShape().volume(), 0x38U);
        multiplierScaleTensor.dataPointer<float>()[0] = multiplierScale;

        std::vector<rt::Tensor> tensors;
        tensors.push_back(std::move(table));
        tensors.push_back(std::move(blockScales));
        tensors.push_back(std::move(multiplierScaleTensor));
        std::unordered_map<std::string, std::string> metadata{
            {rt::kEmbeddingDescriptorMetadataKey, descriptorValue.dump()}};
        return rt::safetensors::saveSafetensors(mPath, tensors, mStream, metadata);
    }

    cudaStream_t mStream{};
    std::filesystem::path mPath;
};

TEST_F(EmbeddingDataTest, LoadsPackedNvfp4WithoutDenseCopy)
{
    // Given a versioned NVFP4 sidecar, when the runtime loads it, then the physical tensors remain packed and the
    // logical hidden width comes only from the descriptor.
    ASSERT_TRUE(writeNvfp4Embedding(descriptor()));

    rt::EmbeddingData data = rt::loadEmbeddingTable(mPath, mStream);

    EXPECT_EQ(data.descriptor.format, kernel::EmbeddingTableFormat::kNVFP4);
    EXPECT_EQ(data.descriptor.logicalShape, (rt::Coords{kVOCAB_SIZE, kHIDDEN_SIZE}));
    EXPECT_EQ(data.table.getShape(), (rt::Coords{kVOCAB_SIZE, kHIDDEN_SIZE / 2}));
    EXPECT_EQ(data.table.getDataType(), trt_edgellm::DataType::kUINT8);
    EXPECT_EQ(data.tableScalingFactor.getShape(), (rt::Coords{kVOCAB_SIZE, kHIDDEN_SIZE / kGROUP_SIZE}));
    EXPECT_EQ(data.tableScalingFactor.getDataType(), trt_edgellm::DataType::kFP8);
    EXPECT_FLOAT_EQ(data.descriptor.multiplierGlobalScale, 0.5F);
    EXPECT_FLOAT_EQ(data.descriptor.modelOutputScale, 1.25F);
    int64_t const packedResidentBytes = data.table.getMemoryCapacity() + data.tableScalingFactor.getMemoryCapacity();
    int64_t const denseFp16Bytes = kVOCAB_SIZE * kHIDDEN_SIZE * static_cast<int64_t>(sizeof(half));
    EXPECT_EQ(packedResidentBytes, kVOCAB_SIZE * (kHIDDEN_SIZE / 2 + kHIDDEN_SIZE / kGROUP_SIZE));
    EXPECT_LT(packedResidentBytes, denseFp16Bytes);
}

} // namespace
