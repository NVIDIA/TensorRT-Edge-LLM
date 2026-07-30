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

#include "common/checkMacros.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <vector>

using namespace trt_edgellm;

namespace
{

TEST(EmbeddingPreprocessorTest, FirstTextOnlyEmbedDoesNotRequireMultimodalScratch)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    rt::EmbeddingData embedding;
    embedding.table = rt::Tensor({3, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "embedding");
    std::vector<half> table(24);
    for (size_t i = 0; i < table.size(); ++i)
    {
        table[i] = __float2half(static_cast<float>(i + 1));
    }
    copyHostToDevice<half>(embedding.table, table);

    rt::LLMEngineConfig config{};
    config.vocabSize = 3;
    config.hiddenSize = 8;
    config.audioTokenId = -1;
    config.imageTokenId = -1;

    rt::Tensor tokenIds({1, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
    copyHostToDevice<int32_t>(tokenIds, {0, 2});

    rt::PipelineIO io;
    io.inputsEmbeds = rt::Tensor({1, 2, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "inputsEmbeds");
    rt::EmbeddingPreprocessor preprocessor(embedding, config);

    EXPECT_NO_THROW(preprocessor.embed(tokenIds, std::nullopt, std::nullopt, io, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const output = copyDeviceToHost<half>(io.inputsEmbeds);
    ASSERT_EQ(output.size(), 16U);
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(__half2float(output[i]), static_cast<float>(i + 1));
        EXPECT_EQ(__half2float(output[i + 8]), static_cast<float>(i + 17));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(EmbeddingPreprocessorTest, TextOnlyEmbedClearsPriorExplicitMultimodalIndices)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    rt::EmbeddingData embedding;
    embedding.table = rt::Tensor({4, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "embedding");
    copyHostToDevice<half>(embedding.table, std::vector<half>(32, __float2half(1.0F)));

    rt::LLMEngineConfig config{};
    config.vocabSize = 4;
    config.hiddenSize = 8;
    config.audioTokenId = 3;
    config.imageTokenId = 2;

    rt::Tensor tokenIds({1, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "tokenIds");
    copyHostToDevice<int32_t>(tokenIds, {config.imageTokenId, 0});
    rt::Tensor vision({1, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "vision");
    copyHostToDevice<half>(vision, std::vector<half>(8, __float2half(2.0F)));

    rt::PipelineIO io;
    io.inputsEmbeds = rt::Tensor({1, 2, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "inputsEmbeds");
    io.deepstackEmbeds.emplace_back(rt::Coords{1, 2, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "deepstack");
    rt::EmbeddingPreprocessor preprocessor(embedding, config);

    preprocessor.embed(tokenIds, std::cref(vision), std::nullopt, io, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // A text-only call must clear the first call's explicit index map. If it leaked, deepstack assembly below would
    // incorrectly scatter the feature into token 0 even though no current multimodal index map exists.
    preprocessor.embed(tokenIds, std::nullopt, std::nullopt, io, stream);
    rt::Tensor feature({1, 8}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "feature");
    copyHostToDevice<half>(feature, std::vector<half>(8, __float2half(7.0F)));
    preprocessor.assembleDeepstack(tokenIds, {std::cref(feature)}, io, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const output = copyDeviceToHost<half>(io.deepstackEmbeds.front());
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](half value) { return __half2float(value) == 0.0F; }));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace
