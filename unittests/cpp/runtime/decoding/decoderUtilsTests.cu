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

#include "runtime/decoding/decoderUtils.h"

#include "common/cudaUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

TEST(DecoderUtilsTests, FinishedSlotsIgnoreAcceptedTokens)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    rt::DecodingInferenceContext context;
    context.initialize(2, 8, std::nullopt, rt::OptionalInputTensors{}, "", stream);
    context.finishedStates = {1, 0};
    context.tokenIds = {{100}, {200}};
    context.currentGenerateLengths = {1, 1};
    context.shouldStopAfterAcceptedToken = [](int32_t, int32_t) { return false; };

    rt::Tensor hostAcceptLengths({2}, rt::DeviceType::kCPU, DataType::kINT32);
    rt::Tensor hostAcceptedTokenIds({2, 2}, rt::DeviceType::kCPU, DataType::kINT32);
    rt::Tensor deviceAcceptLengths({2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deviceAcceptedTokenIds({2, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(deviceAcceptLengths, {2, 1});
    copyHostToDevice<int32_t>(deviceAcceptedTokenIds, {101, 102, 201, 202});

    tokenizer::Tokenizer tokenizer;
    rt::decoder_utils::appendAcceptedTokens(context, hostAcceptLengths, hostAcceptedTokenIds, deviceAcceptLengths,
        deviceAcceptedTokenIds, 2, tokenizer, stream);

    EXPECT_EQ(context.tokenIds[0], std::vector<int32_t>({100}));
    EXPECT_EQ(context.tokenIds[1], std::vector<int32_t>({200, 201}));
    EXPECT_EQ(context.currentGenerateLengths, std::vector<int32_t>({1, 2}));
    EXPECT_EQ(hostAcceptLengths.dataPointer<int32_t>()[0], 0);
    EXPECT_EQ(hostAcceptLengths.dataPointer<int32_t>()[1], 1);

    CUDA_CHECK(cudaStreamDestroy(stream));
}
