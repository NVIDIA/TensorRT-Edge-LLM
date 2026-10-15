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
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "testUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
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

int8_t int8TableValue(int32_t tokenId, int32_t layerIdx, int32_t hiddenIdx)
{
    return static_cast<int8_t>((tokenId * 31 + layerIdx * 11 + hiddenIdx) % 127 - 63);
}

void runInt8GatherTest(std::vector<int32_t> const& inputIds, int32_t batchSize, int32_t seqLen, int32_t maxBatchSize,
    int32_t maxSeqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize, int32_t imageTokenId,
    int32_t audioTokenId)
{
    rt::Tensor inputIdsDevice({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tableDevice({vocabSize, numLayers * pleHiddenSize}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor scalesDevice({vocabSize, numLayers}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor outputDevice({numLayers, maxBatchSize, maxSeqLen, pleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);

    std::vector<int8_t> table(static_cast<size_t>(vocabSize) * numLayers * pleHiddenSize);
    for (int32_t tokenId = 0; tokenId < vocabSize; ++tokenId)
    {
        for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
        {
            for (int32_t hiddenIdx = 0; hiddenIdx < pleHiddenSize; ++hiddenIdx)
            {
                size_t const offset = (static_cast<size_t>(tokenId) * numLayers + layerIdx) * pleHiddenSize + hiddenIdx;
                table[offset] = int8TableValue(tokenId, layerIdx, hiddenIdx);
            }
        }
    }
    std::vector<float> scales(static_cast<size_t>(vocabSize) * numLayers);
    for (int32_t tokenId = 0; tokenId < vocabSize; ++tokenId)
    {
        for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
        {
            scales[static_cast<size_t>(tokenId) * numLayers + layerIdx]
                = 0.01F * static_cast<float>((tokenId + 1) * (layerIdx + 1));
        }
    }
    std::vector<half> const initialOutput = makeFilledOutput<half>(numLayers, maxBatchSize, maxSeqLen, pleHiddenSize);

    copyHostToDevice(inputIdsDevice, inputIds);
    copyHostToDevice(tableDevice, table);
    copyHostToDevice(scalesDevice, scales);
    copyHostToDevice(outputDevice, initialOutput);

    kernel::gemma4PleGather(inputIdsDevice, tableDevice, outputDevice, numLayers, pleHiddenSize, imageTokenId,
        audioTokenId, nullptr, rt::OptionalInputTensor{scalesDevice});
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> const output = copyDeviceToHost<half>(outputDevice);
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
                        = static_cast<size_t>(layerIdx * layerCapacity) + tokenOffset * pleHiddenSize + hiddenIdx;
                    written[outputOffset] = true;
                    // The kernel rounds the single-precision product to FP16 once; the reference does the same,
                    // so the comparison is exact.
                    half const expected = shouldZeroFill(tokenId, vocabSize, imageTokenId, audioTokenId)
                        ? half{}
                        : __float2half(static_cast<float>(int8TableValue(tokenId, layerIdx, hiddenIdx))
                              * scales[static_cast<size_t>(tokenId) * numLayers + layerIdx]);
                    EXPECT_EQ(__half2float(output[outputOffset]), __half2float(expected))
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
            EXPECT_NEAR(__half2float(output[idx]), kSentinelValue, kATol) << "unwritten index=" << idx;
        }
    }
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

TEST(Gemma4PleGatherKernelTest, DequantizesInt8RowsAndZeroFillsSpecialTokens)
{
    constexpr int32_t kBatchSize = 1;
    constexpr int32_t kSeqLen = 6;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 7;
    constexpr int32_t kVocabSize = 6;
    constexpr int32_t kNumLayers = 3;
    constexpr int32_t kPleHiddenSize = 16;
    constexpr int32_t kImageTokenId = 4;
    constexpr int32_t kAudioTokenId = 5;
    std::vector<int32_t> const inputIds{0, 2, -1, kVocabSize, kImageTokenId, kAudioTokenId};

    runInt8GatherTest(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers, kPleHiddenSize,
        kImageTokenId, kAudioTokenId);
}

// Gemma 4 E4B geometry: 42 layer inputs of 256 elements; a 256-element slice is exactly one warp of 8-element
// vectors, so this also covers the block sizing of the INT8 launcher.
TEST(Gemma4PleGatherKernelTest, DequantizesInt8RowsAtGemma4E4BGeometry)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 9;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 10;
    constexpr int32_t kVocabSize = 24;
    constexpr int32_t kNumLayers = 42;
    constexpr int32_t kPleHiddenSize = 256;
    constexpr int32_t kImageTokenId = 22;
    constexpr int32_t kAudioTokenId = 23;
    std::vector<int32_t> const inputIds{0, 7, 7, kVocabSize - 3, -1, kImageTokenId, 13, kVocabSize, 2, 19, 19, 19,
        kAudioTokenId, 5, -4, kImageTokenId, 1, 0};

    runInt8GatherTest(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers, kPleHiddenSize,
        kImageTokenId, kAudioTokenId);
}

// 2056 elements are 257 vectors: the 256-thread block needs a second, one-thread iteration per slice.
TEST(Gemma4PleGatherKernelTest, DequantizesInt8RowsWithPartialFinalIteration)
{
    constexpr int32_t kBatchSize = 1;
    constexpr int32_t kSeqLen = 5;
    constexpr int32_t kMaxBatchSize = 1;
    constexpr int32_t kMaxSeqLen = 6;
    constexpr int32_t kVocabSize = 8;
    constexpr int32_t kNumLayers = 3;
    constexpr int32_t kPleHiddenSize = 2056;
    std::vector<int32_t> const inputIds{3, 0, kVocabSize, 3, -1};

    runInt8GatherTest(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers, kPleHiddenSize,
        /* imageTokenId = */ -1, /* audioTokenId = */ -1);
}

namespace
{

// Runs fn and returns the exception text, or an empty string when nothing was thrown.
template <typename Fn>
std::string thrownMessage(Fn&& fn)
{
    try
    {
        fn();
    }
    catch (std::runtime_error const& e)
    {
        return e.what();
    }
    return {};
}

void expectContains(std::string const& message, std::string const& expected)
{
    EXPECT_NE(message.find(expected), std::string::npos) << "expected \"" << expected << "\" in: " << message;
}

// Writes the tensors as ple_embedding.safetensors in a fresh temp dir and returns the preprocessor's exception text.
std::string pleLoadRejection(std::vector<rt::Tensor>&& tensors, int32_t numLayers, int32_t pleHiddenSize)
{
    auto const testDir = std::filesystem::temp_directory_path() / "trt_edgellm_int8_ple_loader_reject_test";
    std::filesystem::remove_all(testDir);
    std::filesystem::create_directories(testDir);
    if (!rt::safetensors::saveSafetensors(testDir / "ple_embedding.safetensors", tensors, nullptr))
    {
        return "saveSafetensors failed";
    }
    rt::LLMEngineConfig config;
    config.pleEnabled = true;
    config.numPleInputs = numLayers;
    config.pleHiddenSize = pleHiddenSize;
    rt::TensorMap tensorMap;
    auto const message = thrownMessage(
        [&]() { rt::Gemma4EmbeddingPreprocessor preprocessor(testDir, config, 1, 2, tensorMap, nullptr); });
    std::filesystem::remove_all(testDir);
    return message;
}

} // namespace

TEST(Gemma4PleGatherKernelTest, Int8PleSidecarLoaderRejectsMalformedFiles)
{
    constexpr int32_t kVocabSize = 3;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    auto const table = [&](DataType dtype) {
        return rt::Tensor({kVocabSize, kNumLayers * kPleHiddenSize}, rt::DeviceType::kCPU, dtype, "weight");
    };
    auto const scales = [&](std::vector<int64_t> const& shape, DataType dtype) {
        return rt::Tensor(rt::Coords(shape), rt::DeviceType::kCPU, dtype, "weight_scale");
    };

    std::vector<rt::Tensor> int8NoScales;
    int8NoScales.emplace_back(table(DataType::kINT8));
    expectContains(pleLoadRejection(std::move(int8NoScales), kNumLayers, kPleHiddenSize),
        "INT8 ple_embedding.safetensors must contain exactly weight and weight_scale");

    std::vector<rt::Tensor> fp16WithScales;
    fp16WithScales.emplace_back(table(DataType::kHALF));
    fp16WithScales.emplace_back(scales({kVocabSize, kNumLayers}, DataType::kFLOAT));
    expectContains(pleLoadRejection(std::move(fp16WithScales), kNumLayers, kPleHiddenSize),
        "FP16/BF16 ple_embedding.safetensors must contain exactly one tensor named weight");

    std::vector<rt::Tensor> int32Table;
    int32Table.emplace_back(table(DataType::kINT32));
    expectContains(
        pleLoadRejection(std::move(int32Table), kNumLayers, kPleHiddenSize), "PLE table must be FP16, BF16, or INT8");

    std::vector<rt::Tensor> oneDimScales;
    oneDimScales.emplace_back(table(DataType::kINT8));
    oneDimScales.emplace_back(scales({kVocabSize}, DataType::kFLOAT));
    expectContains(pleLoadRejection(std::move(oneDimScales), kNumLayers, kPleHiddenSize), "INT8 PLE scales must be 2D");

    std::vector<rt::Tensor> wrongColumns;
    wrongColumns.emplace_back(table(DataType::kINT8));
    wrongColumns.emplace_back(scales({kVocabSize, kNumLayers + 1}, DataType::kFLOAT));
    expectContains(pleLoadRejection(std::move(wrongColumns), kNumLayers, kPleHiddenSize),
        "INT8 PLE scale columns must match num_ple_inputs");

    std::vector<rt::Tensor> halfScales;
    halfScales.emplace_back(table(DataType::kINT8));
    halfScales.emplace_back(scales({kVocabSize, kNumLayers}, DataType::kHALF));
    expectContains(pleLoadRejection(std::move(halfScales), kNumLayers, kPleHiddenSize), "INT8 PLE scales must be FP32");
}

TEST(Gemma4PleGatherKernelTest, GatherRejectsMismatchedScalesAndOutputTypes)
{
    constexpr int32_t kVocabSize = 4;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    rt::Tensor inputIds({1, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor fp16Table({kVocabSize, kNumLayers * kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor int8Table({kVocabSize, kNumLayers * kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor fp16Output({kNumLayers, 1, 1, kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor bf16Output({kNumLayers, 1, 1, kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor scales({kVocabSize, kNumLayers}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor oneDimScales({kVocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor wrongColumns({kVocabSize, kNumLayers + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto const gather = [&](rt::Tensor const& table, rt::Tensor& output, rt::OptionalInputTensor s) {
        kernel::gemma4PleGather(inputIds, table, output, kNumLayers, kPleHiddenSize, -1, -1, nullptr, s);
    };

    expectContains(thrownMessage([&]() { gather(fp16Table, fp16Output, rt::OptionalInputTensor{scales}); }),
        "scales must not be provided for an FP16 or BF16 pleTable");
    expectContains(thrownMessage([&]() { gather(int8Table, fp16Output, std::nullopt); }),
        "scales must be provided for INT8 pleTable");
    expectContains(thrownMessage([&]() { gather(int8Table, fp16Output, rt::OptionalInputTensor{oneDimScales}); }),
        "INT8 PLE scales must be 2D");
    expectContains(thrownMessage([&]() { gather(int8Table, fp16Output, rt::OptionalInputTensor{wrongColumns}); }),
        "INT8 PLE scale columns must match numLayers");
    expectContains(thrownMessage([&]() { gather(int8Table, bf16Output, rt::OptionalInputTensor{scales}); }),
        "INT8 pleTable requires an FP16 outputBuffer");
}

TEST(Gemma4PleGatherKernelTest, Int8SidecarLoadsThroughPreprocessor)
{
    constexpr int32_t kVocabSize = 3;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    std::vector<int8_t> table(static_cast<size_t>(kVocabSize) * kNumLayers * kPleHiddenSize, 10);
    std::vector<float> scales(static_cast<size_t>(kVocabSize) * kNumLayers, 0.1F);
    rt::Tensor tableTensor({kVocabSize, kNumLayers * kPleHiddenSize}, rt::DeviceType::kCPU, DataType::kINT8, "weight");
    rt::Tensor scalesTensor({kVocabSize, kNumLayers}, rt::DeviceType::kCPU, DataType::kFLOAT, "weight_scale");
    std::memcpy(tableTensor.rawPointer(), table.data(), table.size() * sizeof(int8_t));
    std::memcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float));

    auto const testDir = std::filesystem::temp_directory_path() / "trt_edgellm_int8_ple_loader_test";
    std::filesystem::remove_all(testDir);
    std::filesystem::create_directories(testDir);
    std::vector<rt::Tensor> sidecar;
    sidecar.emplace_back(std::move(tableTensor));
    sidecar.emplace_back(std::move(scalesTensor));
    ASSERT_TRUE(rt::safetensors::saveSafetensors(testDir / "ple_embedding.safetensors", sidecar, nullptr));

    rt::LLMEngineConfig config;
    config.pleEnabled = true;
    config.numPleInputs = kNumLayers;
    config.pleHiddenSize = kPleHiddenSize;
    rt::TensorMap tensorMap;
    rt::Gemma4EmbeddingPreprocessor preprocessor(testDir, config, 1, 2, tensorMap, nullptr);
    rt::Tensor inputIds({1, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(inputIds, std::vector<int32_t>{0, 1});
    preprocessor.embed(inputIds, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int32_t layerIdx = 0; layerIdx < kNumLayers; ++layerIdx)
    {
        auto* output = tensorMap.get("ple_token_embeds_" + std::to_string(layerIdx));
        ASSERT_NE(output, nullptr);
        auto const values = copyDeviceToHost<half>(*output);
        for (half const value : values)
        {
            EXPECT_NEAR(__half2float(value), 1.0F, 1e-3F);
        }
    }
    std::filesystem::remove_all(testDir);
}

// A single PLE layer input gives [vocab, 1] scales; the writer keeps them 2-D and the loader accepts them.
TEST(Gemma4PleGatherKernelTest, Int8SidecarWithOneLayerLoadsThroughPreprocessor)
{
    constexpr int32_t kVocabSize = 3;
    constexpr int32_t kNumLayers = 1;
    constexpr int32_t kPleHiddenSize = 8;
    std::vector<int8_t> table(static_cast<size_t>(kVocabSize) * kNumLayers * kPleHiddenSize, 10);
    std::vector<float> scales(static_cast<size_t>(kVocabSize) * kNumLayers, 0.1F);
    rt::Tensor tableTensor({kVocabSize, kNumLayers * kPleHiddenSize}, rt::DeviceType::kCPU, DataType::kINT8, "weight");
    rt::Tensor scalesTensor({kVocabSize, kNumLayers}, rt::DeviceType::kCPU, DataType::kFLOAT, "weight_scale");
    std::memcpy(tableTensor.rawPointer(), table.data(), table.size() * sizeof(int8_t));
    std::memcpy(scalesTensor.rawPointer(), scales.data(), scales.size() * sizeof(float));

    auto const testDir = std::filesystem::temp_directory_path() / "trt_edgellm_int8_ple_one_layer_loader_test";
    std::filesystem::remove_all(testDir);
    std::filesystem::create_directories(testDir);
    std::vector<rt::Tensor> sidecar;
    sidecar.emplace_back(std::move(tableTensor));
    sidecar.emplace_back(std::move(scalesTensor));
    ASSERT_TRUE(rt::safetensors::saveSafetensors(testDir / "ple_embedding.safetensors", sidecar, nullptr));

    rt::LLMEngineConfig config;
    config.pleEnabled = true;
    config.numPleInputs = kNumLayers;
    config.pleHiddenSize = kPleHiddenSize;
    rt::TensorMap tensorMap;
    rt::Gemma4EmbeddingPreprocessor preprocessor(testDir, config, 1, 2, tensorMap, nullptr);
    rt::Tensor inputIds({1, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(inputIds, std::vector<int32_t>{0, 1});
    preprocessor.embed(inputIds, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int32_t layerIdx = 0; layerIdx < kNumLayers; ++layerIdx)
    {
        auto* output = tensorMap.get("ple_token_embeds_" + std::to_string(layerIdx));
        ASSERT_NE(output, nullptr);
        auto const values = copyDeviceToHost<half>(*output);
        for (half const value : values)
        {
            EXPECT_NEAR(__half2float(value), 1.0F, 1e-3F);
        }
    }
    std::filesystem::remove_all(testDir);
}
