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
#include "kernels/lora/groupedLoraGemm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{

template <typename T>
T* copyToDevice(std::vector<T> const& values, cudaStream_t stream)
{
    T* result{};
    CUDA_CHECK(cudaMallocAsync(&result, values.size() * sizeof(T), stream));
    CUDA_CHECK(cudaMemcpyAsync(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    return result;
}

std::vector<half> toHalf(std::vector<float> const& values)
{
    std::vector<half> result(values.size());
    for (size_t index = 0; index < values.size(); ++index)
    {
        result[index] = __float2half(values[index]);
    }
    return result;
}

int32_t findSegment(int32_t row, std::vector<int32_t> const& offsets)
{
    for (int32_t segment = 0; segment + 1 < static_cast<int32_t>(offsets.size()); ++segment)
    {
        if (row >= offsets[segment] && row < offsets[segment + 1])
        {
            return segment;
        }
    }
    return -1;
}

float quantizeToHalf(float value)
{
    return __half2float(__float2half(value));
}

std::vector<float> reference(std::vector<float> const& input, std::vector<float> const& loraA,
    std::vector<float> const& loraB, std::vector<int32_t> const& offsets, std::vector<int32_t> const& adapterIds,
    std::vector<int32_t> const& ranks, std::vector<float> const& scales, std::vector<float> output, int32_t inputSize,
    int32_t outputSize, int32_t rankStride)
{
    int32_t const totalRows = static_cast<int32_t>(input.size()) / inputSize;
    for (int32_t row = 0; row < totalRows; ++row)
    {
        int32_t const segment = findSegment(row, offsets);
        int32_t const adapter = adapterIds[segment];
        for (int32_t column = 0; column < outputSize; ++column)
        {
            float delta{0.0F};
            for (int32_t rank = 0; rank < ranks[adapter]; ++rank)
            {
                float intermediate{0.0F};
                for (int32_t inputColumn = 0; inputColumn < inputSize; ++inputColumn)
                {
                    intermediate += quantizeToHalf(input[row * inputSize + inputColumn])
                        * quantizeToHalf(loraA[(adapter * inputSize + inputColumn) * rankStride + rank]);
                }
                delta += quantizeToHalf(intermediate)
                    * quantizeToHalf(loraB[(adapter * rankStride + rank) * outputSize + column]);
            }
            int32_t const outputIndex = row * outputSize + column;
            output[outputIndex]
                = quantizeToHalf(quantizeToHalf(output[outputIndex]) + scales[adapter] * quantizeToHalf(delta));
        }
    }
    return output;
}

struct NumericalCase
{
    int32_t totalRows;
    int32_t inputSize;
    int32_t outputSize;
    int32_t maxRank;
    std::vector<int32_t> offsets;
    std::vector<int32_t> adapterIds;
    std::vector<int32_t> ranks;
    std::vector<float> scales;
    bool poisonBPadding{};
    bool expectBitExactOutput{};
};

uint16_t halfBits(half value)
{
    uint16_t result{};
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void expectResult(std::vector<half> const& actual, std::vector<float> const& expected,
    std::vector<half> const& initialOutput, bool expectBitExact)
{
    for (size_t index = 0; index < actual.size(); ++index)
    {
        if (expectBitExact)
        {
            EXPECT_EQ(halfBits(actual[index]), halfBits(initialOutput[index])) << "index=" << index;
        }
        else
        {
            EXPECT_NEAR(__half2float(actual[index]), expected[index], 5.0e-4F) << "index=" << index;
        }
    }
}

void runNumericalCase(NumericalCase const& testCase, NumericalCase const* replayCase = nullptr)
{
    int32_t const rankStride = getGroupedLoraGemmRankStride(testCase.maxRank);
    int32_t const numAdapters = static_cast<int32_t>(testCase.ranks.size());
    ASSERT_GT(rankStride, 0);
    ASSERT_EQ(testCase.offsets.front(), 0);
    ASSERT_EQ(testCase.offsets.back(), testCase.totalRows);
    ASSERT_EQ(testCase.offsets.size(), testCase.adapterIds.size() + 1);
    ASSERT_EQ(testCase.scales.size(), testCase.ranks.size());

    std::vector<float> input(testCase.totalRows * testCase.inputSize);
    std::vector<float> loraA(numAdapters * rankStride * testCase.inputSize);
    std::vector<float> loraB(numAdapters * testCase.outputSize * rankStride);
    std::vector<float> initialOutput(testCase.totalRows * testCase.outputSize);
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = 0.025F * static_cast<float>(static_cast<int32_t>(index % 11) - 5);
    }
    for (size_t index = 0; index < loraA.size(); ++index)
    {
        loraA[index] = 0.02F * static_cast<float>(static_cast<int32_t>(index % 7) - 3);
    }
    for (size_t index = 0; index < loraB.size(); ++index)
    {
        loraB[index] = 0.03F * static_cast<float>(static_cast<int32_t>(index % 9) - 4);
    }
    if (testCase.poisonBPadding)
    {
        for (int32_t adapter = 0; adapter < numAdapters; ++adapter)
        {
            for (int32_t rank = testCase.ranks[adapter]; rank < rankStride; ++rank)
            {
                for (int32_t column = 0; column < testCase.outputSize; ++column)
                {
                    loraB[(adapter * rankStride + rank) * testCase.outputSize + column]
                        = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }
    for (size_t index = 0; index < initialOutput.size(); ++index)
    {
        initialOutput[index] = 0.01F * static_cast<float>(index % 5);
    }
    std::vector<float> const expected = reference(input, loraA, loraB, testCase.offsets, testCase.adapterIds,
        testCase.ranks, testCase.scales, initialOutput, testCase.inputSize, testCase.outputSize, rankStride);
    std::vector<half> const inputHalf = toHalf(input);
    std::vector<half> const loraAHalf = toHalf(loraA);
    std::vector<half> const loraBHalf = toHalf(loraB);
    std::vector<half> const initialOutputHalf = toHalf(initialOutput);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaError_t const initializationStatus = initializeGroupedLoraGemmCuteDsl(stream);
    if (initializationStatus == cudaErrorNotSupported)
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
        GTEST_SKIP() << "CuTe DSL grouped LoRA GEMM is unavailable";
    }
    ASSERT_EQ(initializationStatus, cudaSuccess);
    half* deviceInput = copyToDevice(inputHalf, stream);
    half* deviceLoraA = copyToDevice(loraAHalf, stream);
    half* deviceLoraB = copyToDevice(loraBHalf, stream);
    int32_t* deviceOffsets = copyToDevice(testCase.offsets, stream);
    int32_t* deviceAdapterIds = copyToDevice(testCase.adapterIds, stream);
    int32_t* deviceRanks = copyToDevice(testCase.ranks, stream);
    float* deviceScales = copyToDevice(testCase.scales, stream);
    half* deviceOutput = copyToDevice(initialOutputHalf, stream);
    half* workspace{};
    size_t const workspaceSize
        = getGroupedLoraGemmWorkspaceSize(testCase.totalRows, testCase.outputSize, testCase.maxRank);
    ASSERT_GE(workspaceSize, static_cast<size_t>(testCase.totalRows * rankStride) * sizeof(half));
    CUDA_CHECK(cudaMallocAsync(&workspace, workspaceSize, stream));

    GroupedLoraGemmParams const params{deviceInput, deviceLoraA, deviceLoraB, deviceOffsets, deviceAdapterIds,
        deviceRanks, deviceScales, workspace, deviceOutput, testCase.totalRows, testCase.inputSize, testCase.outputSize,
        testCase.maxRank, static_cast<int32_t>(testCase.adapterIds.size()), numAdapters};
    cudaGraph_t graph{};
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(launchGroupedLoraGemm(params, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    cudaGraphExec_t graphExec{};
    ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), cudaSuccess);
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    std::vector<half> actual(initialOutput.size());
    CUDA_CHECK(
        cudaMemcpyAsync(actual.data(), deviceOutput, actual.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    expectResult(actual, expected, initialOutputHalf, testCase.expectBitExactOutput);

    if (replayCase != nullptr)
    {
        ASSERT_EQ(replayCase->totalRows, testCase.totalRows);
        ASSERT_EQ(replayCase->inputSize, testCase.inputSize);
        ASSERT_EQ(replayCase->outputSize, testCase.outputSize);
        ASSERT_EQ(replayCase->maxRank, testCase.maxRank);
        ASSERT_EQ(replayCase->offsets.size(), testCase.offsets.size());
        ASSERT_EQ(replayCase->adapterIds.size(), testCase.adapterIds.size());
        ASSERT_EQ(replayCase->ranks.size(), testCase.ranks.size());
        ASSERT_EQ(replayCase->scales.size(), testCase.scales.size());
        std::vector<float> const replayExpected
            = reference(input, loraA, loraB, replayCase->offsets, replayCase->adapterIds, replayCase->ranks,
                replayCase->scales, initialOutput, testCase.inputSize, testCase.outputSize, rankStride);
        CUDA_CHECK(cudaMemcpyAsync(deviceOffsets, replayCase->offsets.data(),
            replayCase->offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceAdapterIds, replayCase->adapterIds.data(),
            replayCase->adapterIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceRanks, replayCase->ranks.data(), replayCase->ranks.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceScales, replayCase->scales.data(), replayCase->scales.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceOutput, initialOutputHalf.data(), initialOutputHalf.size() * sizeof(half),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(actual.data(), deviceOutput, actual.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        expectResult(actual, replayExpected, initialOutputHalf, false);
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFreeAsync(deviceInput, stream));
    CUDA_CHECK(cudaFreeAsync(deviceLoraA, stream));
    CUDA_CHECK(cudaFreeAsync(deviceLoraB, stream));
    CUDA_CHECK(cudaFreeAsync(deviceOffsets, stream));
    CUDA_CHECK(cudaFreeAsync(deviceAdapterIds, stream));
    CUDA_CHECK(cudaFreeAsync(deviceRanks, stream));
    CUDA_CHECK(cudaFreeAsync(deviceScales, stream));
    CUDA_CHECK(cudaFreeAsync(deviceOutput, stream));
    CUDA_CHECK(cudaFreeAsync(workspace, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

TEST(GroupedLoraGemm, RoutesPackedRowsAcrossVariableRanks)
{
    // Given packed segments with heterogeneous ranks, scales, and a rank-zero adapter, when the graph-captured
    // grouped LoRA GEMM runs, then each row is routed and accumulated with the FP16 intermediate/delta contract.
    runNumericalCase({7, 128, 128, 13, {0, 2, 3, 6, 7}, {1, 0, 2, 1}, {0, 6, 13}, {0.0F, 0.5F, 1.25F}});
}

TEST(GroupedLoraGemm, SupportsDecodeSegmentsAndRankBoundaries)
{
    // Given one-token decode segments using ranks around the physical alignment boundary, when one packed invocation
    // runs, then every adapter uses only its logical rank and updates its routed row.
    runNumericalCase(
        {5, 64, 64, 13, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}, {1, 7, 8, 9, 13}, {0.25F, 0.5F, 0.75F, 1.0F, 1.25F}});
}

TEST(GroupedLoraGemm, SupportsPrefillSegments)
{
    // Given multiple multi-token prefill segments, when they share one packed invocation, then all rows use their
    // segment's adapter, rank, and scale through the common metadata ABI.
    runNumericalCase({9, 128, 192, 13, {0, 4, 9}, {0, 1}, {4, 13}, {0.625F, 1.375F}});
}

TEST(GroupedLoraGemm, SupportsMixedPrefillAndDecodeSegments)
{
    // Given interleaved one-token and multi-token segments including a rank-zero route, when the packed invocation
    // runs, then decode and prefill rows follow identical routing and accumulation semantics.
    runNumericalCase({8, 64, 128, 9, {0, 1, 4, 5, 8}, {1, 0, 2, 1}, {0, 3, 9}, {0.0F, 0.875F, 1.125F}});
}

TEST(GroupedLoraGemm, LeavesOutputUnchangedWhenAllRanksAreZero)
{
    // Given only adapters that do not target the module, when rank zero represents every segment, then the packed
    // output remains bit-equivalent to its FP16 input value.
    runNumericalCase({6, 64, 64, 8, {0, 2, 3, 6}, {0, 1, 0}, {0, 0}, {0.75F, 1.5F}, false, true});
}

TEST(GroupedLoraGemm, IgnoresNonFiniteBPadding)
{
    // Given nonfinite values outside every adapter's logical rank, when the aligned second GEMM runs, then TMA masks
    // those lanes and the packed result depends only on logical LoRA weights.
    runNumericalCase({5, 64, 64, 13, {0, 2, 5}, {0, 1}, {7, 13}, {0.75F, 1.25F}, true, false});
}

TEST(GroupedLoraGemm, ReadsUpdatedRoutingWhenGraphIsReplayed)
{
    // Given one captured invocation and stable binding addresses, when a later step replaces every routing array and
    // changes active adapters, then graph replay reads the new per-step metadata.
    NumericalCase const initial{6, 64, 64, 9, {0, 1, 4, 6}, {0, 1, 2}, {3, 9, 0}, {0.5F, 1.0F, 1.5F}};
    NumericalCase const replay{6, 64, 64, 9, {0, 2, 3, 6}, {2, 0, 1}, {0, 7, 9}, {1.25F, 0.75F, 0.25F}};
    runNumericalCase(initial, &replay);
}

TEST(GroupedLoraGemm, SupportsMaximumSegmentCount)
{
    // Given the maximum supported number of one-token segments, when all groups are scheduled in one invocation,
    // then descriptor generation and persistent tile scheduling cover every segment.
    constexpr int32_t kNUM_SEGMENTS{256};
    std::vector<int32_t> offsets(kNUM_SEGMENTS + 1);
    std::vector<int32_t> adapterIds(kNUM_SEGMENTS);
    for (int32_t index = 0; index < kNUM_SEGMENTS; ++index)
    {
        offsets[index] = index;
        adapterIds[index] = index % 2;
    }
    offsets.back() = kNUM_SEGMENTS;
    runNumericalCase({kNUM_SEGMENTS, 64, 64, 8, std::move(offsets), std::move(adapterIds), {1, 8}, {0.5F, 1.25F}});
}

TEST(GroupedLoraGemm, RejectsInvalidLaunchContract)
{
    // Given empty parameters and invalid dimensions, when the public API validates them, then it rejects each call.
    GroupedLoraGemmParams const params{};
    EXPECT_EQ(launchGroupedLoraGemm(params, nullptr), cudaErrorInvalidValue);
    EXPECT_EQ(getGroupedLoraGemmWorkspaceSize(0, 4, 8), 0);
    EXPECT_EQ(getGroupedLoraGemmWorkspaceSize(4, 0, 8), 0);
    EXPECT_EQ(getGroupedLoraGemmWorkspaceSize(4, 4, 0), 0);
    EXPECT_EQ(getGroupedLoraGemmRankStride(0), 0);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto* const halfPointer = reinterpret_cast<half*>(uintptr_t{0x1000});
    auto* const intPointer = reinterpret_cast<int32_t*>(uintptr_t{0x2000});
    auto* const floatPointer = reinterpret_cast<float*>(uintptr_t{0x3000});
    GroupedLoraGemmParams populated{halfPointer, halfPointer, halfPointer, intPointer, intPointer, intPointer,
        floatPointer, halfPointer, halfPointer, 1, 64, 64, 8, 2, 1};
    EXPECT_EQ(launchGroupedLoraGemm(populated, stream), cudaErrorInvalidValue);
    populated.totalRows = 257;
    populated.numSegments = 257;
    EXPECT_EQ(launchGroupedLoraGemm(populated, stream), cudaErrorNotSupported);
    populated.totalRows = 1;
    populated.numSegments = 1;
    populated.inputSize = 63;
    EXPECT_EQ(launchGroupedLoraGemm(populated, stream), cudaErrorNotSupported);
    populated.inputSize = 64;
    populated.outputSize = 63;
    EXPECT_EQ(launchGroupedLoraGemm(populated, stream), cudaErrorNotSupported);
    CUDA_CHECK(cudaStreamDestroy(stream));
}
