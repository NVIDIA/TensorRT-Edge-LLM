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
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "testUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

constexpr float kRTol = 1e-3F;
constexpr float kATol = 1e-3F;
constexpr float kSentinelValue = -7.0F;

template <typename T>
struct Gemma4PleGatherTraits;

template <>
struct Gemma4PleGatherTraits<half>
{
    static constexpr DataType kDataType = DataType::kHALF;

    static half fromFloat(float value)
    {
        return __float2half(value);
    }

    static float toFloat(half value)
    {
        return __half2float(value);
    }
};

template <>
struct Gemma4PleGatherTraits<__nv_bfloat16>
{
    static constexpr DataType kDataType = DataType::kBF16;

    static __nv_bfloat16 fromFloat(float value)
    {
        return __float2bfloat16(value);
    }

    static float toFloat(__nv_bfloat16 value)
    {
        return __bfloat162float(value);
    }
};

template <typename T>
T makeScalar(float value)
{
    return Gemma4PleGatherTraits<T>::fromFloat(value);
}

template <typename T>
float scalarToFloat(T value)
{
    return Gemma4PleGatherTraits<T>::toFloat(value);
}

float tableValue(int32_t tokenId, int32_t layerIdx, int32_t hiddenIdx)
{
    return static_cast<float>(tokenId * 100 + layerIdx * 10 + hiddenIdx);
}

template <typename T>
std::vector<T> makePleTable(int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize)
{
    std::vector<T> table(static_cast<size_t>(vocabSize) * numLayers * pleHiddenSize);
    for (int32_t tokenId = 0; tokenId < vocabSize; ++tokenId)
    {
        for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
        {
            for (int32_t hiddenIdx = 0; hiddenIdx < pleHiddenSize; ++hiddenIdx)
            {
                size_t const offset = (static_cast<size_t>(tokenId) * numLayers + layerIdx) * pleHiddenSize + hiddenIdx;
                table[offset] = makeScalar<T>(tableValue(tokenId, layerIdx, hiddenIdx));
            }
        }
    }
    return table;
}

template <typename T>
std::vector<T> makeFilledOutput(int32_t numLayers, int32_t maxBatchSize, int32_t maxSeqLen, int32_t pleHiddenSize)
{
    return std::vector<T>(
        static_cast<size_t>(numLayers) * maxBatchSize * maxSeqLen * pleHiddenSize, makeScalar<T>(kSentinelValue));
}

bool shouldZeroFill(int32_t tokenId, int32_t vocabSize, int32_t imageTokenId, int32_t audioTokenId)
{
    return tokenId < 0 || tokenId >= vocabSize || (imageTokenId >= 0 && tokenId == imageTokenId)
        || (audioTokenId >= 0 && tokenId == audioTokenId);
}

template <typename T>
void verifyOutput(std::vector<T> const& output, std::vector<int32_t> const& inputIds, int32_t batchSize, int32_t seqLen,
    int32_t maxBatchSize, int32_t maxSeqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize,
    int32_t imageTokenId, int32_t audioTokenId)
{
    int64_t const layerCapacity = static_cast<int64_t>(maxBatchSize) * maxSeqLen * pleHiddenSize;
    std::vector<bool> written(output.size(), false);

    for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int32_t seqIdx = 0; seqIdx < seqLen; ++seqIdx)
            {
                size_t const tokenOffset = static_cast<size_t>(batchIdx) * seqLen + seqIdx;
                int32_t const tokenId = inputIds[tokenOffset];
                for (int32_t hiddenIdx = 0; hiddenIdx < pleHiddenSize; ++hiddenIdx)
                {
                    size_t const outputOffset
                        = static_cast<size_t>(layerIdx * layerCapacity) + (tokenOffset * pleHiddenSize) + hiddenIdx;
                    written[outputOffset] = true;
                    float const expected = shouldZeroFill(tokenId, vocabSize, imageTokenId, audioTokenId)
                        ? 0.0F
                        : scalarToFloat(makeScalar<T>(tableValue(tokenId, layerIdx, hiddenIdx)));
                    EXPECT_NEAR(scalarToFloat(output[outputOffset]), expected, kATol + kRTol * std::abs(expected))
                        << "layer=" << layerIdx << " batch=" << batchIdx << " seq=" << seqIdx
                        << " hidden=" << hiddenIdx;
                }
            }
        }
    }

    for (size_t idx = 0; idx < output.size(); ++idx)
    {
        if (!written[idx])
        {
            EXPECT_NEAR(scalarToFloat(output[idx]), kSentinelValue, kATol) << "unwritten index=" << idx;
        }
    }
}

template <typename T>
void runGatherTest(std::vector<int32_t> const& inputIds, int32_t batchSize, int32_t seqLen, int32_t maxBatchSize,
    int32_t maxSeqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize, int32_t imageTokenId,
    int32_t audioTokenId)
{
    rt::Tensor inputIdsDevice({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pleTableDevice(
        {vocabSize, numLayers * pleHiddenSize}, rt::DeviceType::kGPU, Gemma4PleGatherTraits<T>::kDataType);
    rt::Tensor outputDevice(
        {numLayers, maxBatchSize, maxSeqLen, pleHiddenSize}, rt::DeviceType::kGPU, Gemma4PleGatherTraits<T>::kDataType);

    std::vector<T> const table = makePleTable<T>(vocabSize, numLayers, pleHiddenSize);
    std::vector<T> const initialOutput = makeFilledOutput<T>(numLayers, maxBatchSize, maxSeqLen, pleHiddenSize);

    copyHostToDevice(inputIdsDevice, inputIds);
    copyHostToDevice(pleTableDevice, table);
    copyHostToDevice(outputDevice, initialOutput);

    kernel::gemma4PleGather(
        inputIdsDevice, pleTableDevice, outputDevice, numLayers, pleHiddenSize, imageTokenId, audioTokenId, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> const output = copyDeviceToHost<T>(outputDevice);
    verifyOutput(output, inputIds, batchSize, seqLen, maxBatchSize, maxSeqLen, vocabSize, numLayers, pleHiddenSize,
        imageTokenId, audioTokenId);
}

} // namespace

TEST(Gemma4PleGatherKernelTest, GathersFp16LayerOutputsWithCompactRuntimeShape)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 3;
    constexpr int32_t kMaxBatchSize = 4;
    constexpr int32_t kMaxSeqLen = 5;
    constexpr int32_t kVocabSize = 7;
    constexpr int32_t kNumLayers = 3;
    constexpr int32_t kPleHiddenSize = 16;
    std::vector<int32_t> const inputIds{0, 1, 2, 3, 4, 5};

    runGatherTest<half>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, /* imageTokenId = */ -1, /* audioTokenId = */ -1);
}

TEST(Gemma4PleGatherKernelTest, ZeroFillsInvalidAndMultimodalTokensFp16)
{
    constexpr int32_t kBatchSize = 1;
    constexpr int32_t kSeqLen = 5;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 6;
    constexpr int32_t kVocabSize = 5;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    constexpr int32_t kImageTokenId = 3;
    constexpr int32_t kAudioTokenId = 4;
    std::vector<int32_t> const inputIds{0, -1, kVocabSize, kImageTokenId, kAudioTokenId};

    runGatherTest<half>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, kImageTokenId, kAudioTokenId);
}

TEST(Gemma4PleGatherKernelTest, GathersBfloat16LayerOutputs)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 2;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 3;
    constexpr int32_t kVocabSize = 6;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    std::vector<int32_t> const inputIds{1, 2, 3, 4};

    runGatherTest<__nv_bfloat16>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, /* imageTokenId = */ -1, /* audioTokenId = */ -1);
}
