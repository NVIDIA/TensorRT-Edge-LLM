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
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/state/pipelineIO.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

// Given a deepstack-taking engine whose deepstack inputs hold non-zero values in every row of their capacity, as a
//       previous step's media features leave them
// When a text-only step of T = 5 tokens, fewer than the 16 the buffers hold, prepares the deepstack inputs
// Then in every deepstack input the step's T rows read zero and the rows past T keep their values
TEST(EmbeddingPreprocessorDeepstackTest, TextOnlyStepZeroesExactlyItsRows)
{
    constexpr int64_t kMaxTokens{16};
    constexpr int64_t kHiddenSize{8};
    constexpr int64_t kStepTokens{5};
    constexpr float kStale{3.0F};
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    LLMEngineConfig config{};
    config.hiddenSize = static_cast<int32_t>(kHiddenSize);
    config.maxPhysicalTokens = static_cast<int32_t>(kMaxTokens);
    config.numDeepstackFeatures = 2;
    PipelineIO io{};
    std::vector<half> stale(static_cast<size_t>(kMaxTokens * kHiddenSize), __float2half(kStale));
    for (int32_t layer = 0; layer < config.numDeepstackFeatures; ++layer)
    {
        Tensor& buffer = io.deepstackEmbeds.emplace_back(
            Coords{kMaxTokens, kHiddenSize}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "test::deepstack");
        CUDA_CHECK(cudaMemcpyAsync(
            buffer.rawPointer(), stale.data(), stale.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    }
    EmbeddingData embedding{};
    EmbeddingPreprocessor preprocessor(embedding, config);
    Tensor tokenIds({1, kStepTokens}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "test::tokenIds");

    preprocessor.prepareDeepstack(tokenIds, OptionalInputTensors{}, io, stream);

    for (Tensor const& buffer : io.deepstackEmbeds)
    {
        std::vector<half> written(stale.size());
        CUDA_CHECK(cudaMemcpyAsync(
            written.data(), buffer.rawPointer(), written.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t index = 0; index < written.size(); ++index)
        {
            int64_t const row = static_cast<int64_t>(index) / kHiddenSize;
            EXPECT_EQ(__half2float(written[index]), row < kStepTokens ? 0.0F : kStale) << "row " << row;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace trt_edgellm::rt
