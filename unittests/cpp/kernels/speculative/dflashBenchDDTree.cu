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

#include "examples/utils/benchRunner.h"
#include "kernels/speculative/ddtreeKernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kCandidateTopK{2};
constexpr float kMinimumExpansionMargin{0.1F};

struct DDTreeCase
{
    int32_t batchSize;
    int32_t blockSize;
    int32_t verifySize;
    int32_t vocabSize;
    std::vector<float> logits;
    std::vector<int32_t> rootTokenIds;
    std::vector<int32_t> baseLengths;
};

struct TreeOutputs
{
    std::vector<int32_t> nodeTokenIds;
    std::vector<int32_t> nodeDepths;
    std::vector<int32_t> parentIds;
    std::vector<float> nodeScores;
    std::vector<int32_t> validCounts;
    std::vector<int32_t> verifyTokenIds;
    std::vector<int32_t> positionIds;
    std::vector<int32_t> packedMask;
    std::vector<int8_t> treeMask;
    std::vector<int32_t> contextLengths;
    std::vector<int64_t> selectTokenIndices;
};

struct CpuReference
{
    TreeOutputs outputs;
    std::vector<float> competingExpansionMargins;
};

struct Candidate
{
    int32_t tokenId;
    float logProbability;
};

struct Expansion
{
    int32_t parent;
    int32_t slot;
    int32_t tokenId;
    float score;
};

class CudaStream
{
public:
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

size_t logitOffset(DDTreeCase const& testCase, int32_t batch, int32_t depth, int32_t token)
{
    return (static_cast<size_t>(batch) * testCase.blockSize + depth) * testCase.vocabSize + token;
}

void populateDepth(
    DDTreeCase& testCase, int32_t batch, int32_t depth, int32_t bestToken, int32_t secondToken, float gap)
{
    for (int32_t token = 0; token < testCase.vocabSize; ++token)
    {
        testCase.logits[logitOffset(testCase, batch, depth, token)] = -30.0F - static_cast<float>(token) * 0.01F;
    }
    testCase.logits[logitOffset(testCase, batch, depth, bestToken)] = 8.0F;
    testCase.logits[logitOffset(testCase, batch, depth, secondToken)] = 8.0F - gap;
}

DDTreeCase makePrimaryCase()
{
    DDTreeCase testCase{1, 3, 8, 32, {}, {17}, {100}};
    testCase.logits.resize(static_cast<size_t>(testCase.batchSize) * testCase.blockSize * testCase.vocabSize);
    populateDepth(testCase, 0, 0, 0, 1, 0.5F);
    populateDepth(testCase, 0, 1, 3, 5, 2.0F);
    populateDepth(testCase, 0, 2, 7, 11, 0.7F);
    return testCase;
}

DDTreeCase makePackedBoundaryCase()
{
    DDTreeCase testCase{2, 6, 40, 64, {}, {51, 61}, {100, 240}};
    testCase.logits.resize(static_cast<size_t>(testCase.batchSize) * testCase.blockSize * testCase.vocabSize);
    std::vector<float> const batch0Gaps{0.5F, 6.0F, 10.0F, 0.8F, 8.0F, 12.0F};
    std::vector<float> const batch1Gaps{0.4F, 5.5F, 9.5F, 1.1F, 7.5F, 11.5F};
    for (int32_t depth = 0; depth < testCase.blockSize; ++depth)
    {
        populateDepth(testCase, 0, depth, depth * 2, depth * 2 + 1, batch0Gaps[depth]);
        populateDepth(testCase, 1, depth, 40 + depth * 2, 41 + depth * 2, batch1Gaps[depth]);
    }
    return testCase;
}

std::vector<Candidate> selectCandidates(DDTreeCase const& testCase, int32_t batch, int32_t depth)
{
    std::vector<int32_t> tokenIds(testCase.vocabSize);
    std::iota(tokenIds.begin(), tokenIds.end(), 0);
    std::sort(tokenIds.begin(), tokenIds.end(), [&testCase, batch, depth](int32_t lhs, int32_t rhs) {
        float const lhsLogit = testCase.logits[logitOffset(testCase, batch, depth, lhs)];
        float const rhsLogit = testCase.logits[logitOffset(testCase, batch, depth, rhs)];
        return lhsLogit > rhsLogit || (lhsLogit == rhsLogit && lhs < rhs);
    });

    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < testCase.vocabSize; ++token)
    {
        maxLogit = std::max(maxLogit, testCase.logits[logitOffset(testCase, batch, depth, token)]);
    }
    float expSum{0.0F};
    for (int32_t token = 0; token < testCase.vocabSize; ++token)
    {
        expSum += std::exp(testCase.logits[logitOffset(testCase, batch, depth, token)] - maxLogit);
    }
    float const logSumExp = maxLogit + std::log(expSum);

    std::vector<Candidate> candidates;
    candidates.reserve(kCandidateTopK);
    for (int32_t slot = 0; slot < kCandidateTopK; ++slot)
    {
        int32_t const tokenId = tokenIds[slot];
        candidates.push_back({tokenId, testCase.logits[logitOffset(testCase, batch, depth, tokenId)] - logSumExp});
    }
    return candidates;
}

bool expansionPrecedes(Expansion const& lhs, Expansion const& rhs)
{
    if (lhs.score != rhs.score)
    {
        return lhs.score > rhs.score;
    }
    if (lhs.parent != rhs.parent)
    {
        return lhs.parent < rhs.parent;
    }
    if (lhs.slot != rhs.slot)
    {
        return lhs.slot < rhs.slot;
    }
    return lhs.tokenId < rhs.tokenId;
}

CpuReference buildCpuReference(DDTreeCase const& testCase)
{
    int32_t const packedWords = (testCase.verifySize + 31) / 32;
    size_t const nodeCount = static_cast<size_t>(testCase.batchSize) * testCase.verifySize;
    CpuReference reference;
    TreeOutputs& expected = reference.outputs;
    expected.nodeTokenIds.assign(nodeCount, 0);
    expected.nodeDepths.assign(nodeCount, 0);
    expected.parentIds.assign(nodeCount, -1);
    expected.nodeScores.assign(nodeCount, -std::numeric_limits<float>::infinity());
    expected.validCounts.assign(testCase.batchSize, 0);
    expected.verifyTokenIds.assign(nodeCount, 0);
    expected.positionIds.assign(nodeCount, 0);
    expected.packedMask.assign(nodeCount * packedWords, 0);
    expected.treeMask.assign(nodeCount * testCase.verifySize, 0);
    expected.contextLengths.resize(testCase.batchSize);
    expected.selectTokenIndices.resize(nodeCount);

    for (int32_t batch = 0; batch < testCase.batchSize; ++batch)
    {
        std::vector<std::vector<Candidate>> depthCandidates(testCase.blockSize);
        for (int32_t depth = 1; depth < testCase.blockSize; ++depth)
        {
            depthCandidates[depth] = selectCandidates(testCase, batch, depth);
        }

        int32_t const treeOffset = batch * testCase.verifySize;
        expected.nodeTokenIds[treeOffset] = testCase.rootTokenIds[batch];
        expected.nodeScores[treeOffset] = 0.0F;
        std::vector<int32_t> nextCandidateSlot(testCase.verifySize, 0);
        int32_t validCount{1};
        for (int32_t outputNode = 1; outputNode < testCase.verifySize; ++outputNode)
        {
            std::vector<Expansion> available;
            for (int32_t parent = 0; parent < validCount; ++parent)
            {
                int32_t const parentIndex = treeOffset + parent;
                int32_t const childDepth = expected.nodeDepths[parentIndex] + 1;
                int32_t const slot = nextCandidateSlot[parent];
                if (childDepth >= testCase.blockSize || slot >= kCandidateTopK)
                {
                    continue;
                }
                Candidate const candidate = depthCandidates[childDepth][slot];
                available.push_back(
                    {parent, slot, candidate.tokenId, expected.nodeScores[parentIndex] + candidate.logProbability});
            }
            if (available.empty())
            {
                break;
            }
            std::sort(available.begin(), available.end(), expansionPrecedes);
            if (available.size() > 1)
            {
                reference.competingExpansionMargins.push_back(available[0].score - available[1].score);
            }

            Expansion const chosen = available.front();
            ++nextCandidateSlot[chosen.parent];
            int32_t const nodeIndex = treeOffset + outputNode;
            expected.nodeTokenIds[nodeIndex] = chosen.tokenId;
            expected.nodeDepths[nodeIndex] = expected.nodeDepths[treeOffset + chosen.parent] + 1;
            expected.parentIds[nodeIndex] = chosen.parent;
            expected.nodeScores[nodeIndex] = chosen.score;
            ++validCount;
        }
        expected.validCounts[batch] = validCount;
        expected.contextLengths[batch] = testCase.baseLengths[batch] + testCase.verifySize;

        for (int32_t node = 0; node < testCase.verifySize; ++node)
        {
            int32_t const nodeIndex = treeOffset + node;
            expected.selectTokenIndices[nodeIndex] = node;
            if (node >= validCount)
            {
                continue;
            }
            expected.verifyTokenIds[nodeIndex] = expected.nodeTokenIds[nodeIndex];
            expected.positionIds[nodeIndex] = testCase.baseLengths[batch] + expected.nodeDepths[nodeIndex];
            int32_t ancestor = node;
            while (ancestor >= 0)
            {
                expected.treeMask[(treeOffset + node) * testCase.verifySize + ancestor] = 1;
                expected.packedMask[(treeOffset + node) * packedWords + ancestor / 32] |= 1U << (ancestor % 32);
                ancestor = expected.parentIds[treeOffset + ancestor];
            }
        }
    }
    return reference;
}

template <typename T>
void copyHostToDeviceAsync(rt::Tensor& tensor, std::vector<T> const& host, cudaStream_t stream)
{
    ASSERT_EQ(static_cast<size_t>(tensor.getShape().volume()), host.size());
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.rawPointer(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
void copyDeviceToHostAsync(std::vector<T>& host, rt::Tensor const& tensor, cudaStream_t stream)
{
    host.resize(static_cast<size_t>(tensor.getShape().volume()));
    CUDA_CHECK(
        cudaMemcpyAsync(host.data(), tensor.rawPointer(), host.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

TreeOutputs runProductionDDTree(DDTreeCase const& testCase)
{
    DDTreeBuildScratch scratch = allocateDDTreeBuildScratch(
        testCase.batchSize, testCase.blockSize, testCase.vocabSize, testCase.verifySize, kCandidateTopK);
    EXPECT_EQ(scratch.selectTokenIndices.getDataType(), nvinfer1::DataType::kINT64);
    CudaStream stream;
    EXPECT_NE(stream.get(), nullptr);
    copyHostToDeviceAsync(scratch.draftLogits, testCase.logits, stream.get());
    copyHostToDeviceAsync(scratch.lastAcceptedTokens, testCase.rootTokenIds, stream.get());
    copyHostToDeviceAsync(scratch.baseKVCacheLengths, testCase.baseLengths, stream.get());

    kernel::DDTreeBuildParams const params{
        {scratch.draftLogits, scratch.lastAcceptedTokens, scratch.baseKVCacheLengths},
        {scratch.treeTokenIds, scratch.treeDepths, scratch.treeParentIds, scratch.treeNodeScores, scratch.validCounts,
            scratch.verifyTokenIds, scratch.specDecodePositionIds, scratch.packedAttentionMask, scratch.verifyTreeMask,
            scratch.contextLengths, scratch.selectTokenIndices},
        kCandidateTopK, scratch.workspace.rawPointer(), static_cast<size_t>(scratch.workspace.getShape().volume()),
        stream.get()};
    kernel::ddtreeBuild(params);

    TreeOutputs actual;
    copyDeviceToHostAsync(actual.nodeTokenIds, scratch.treeTokenIds, stream.get());
    copyDeviceToHostAsync(actual.nodeDepths, scratch.treeDepths, stream.get());
    copyDeviceToHostAsync(actual.parentIds, scratch.treeParentIds, stream.get());
    copyDeviceToHostAsync(actual.nodeScores, scratch.treeNodeScores, stream.get());
    copyDeviceToHostAsync(actual.validCounts, scratch.validCounts, stream.get());
    copyDeviceToHostAsync(actual.verifyTokenIds, scratch.verifyTokenIds, stream.get());
    copyDeviceToHostAsync(actual.positionIds, scratch.specDecodePositionIds, stream.get());
    copyDeviceToHostAsync(actual.packedMask, scratch.packedAttentionMask, stream.get());
    copyDeviceToHostAsync(actual.treeMask, scratch.verifyTreeMask, stream.get());
    copyDeviceToHostAsync(actual.contextLengths, scratch.contextLengths, stream.get());
    copyDeviceToHostAsync(actual.selectTokenIndices, scratch.selectTokenIndices, stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    return actual;
}

void expectScoresEqual(std::vector<float> const& actual, std::vector<float> const& expected,
    std::vector<int32_t> const& validCounts, int32_t verifySize)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
    {
        int32_t const batch = static_cast<int32_t>(index) / verifySize;
        int32_t const node = static_cast<int32_t>(index) % verifySize;
        if (node < validCounts[batch])
        {
            EXPECT_NEAR(actual[index], expected[index], 1.0e-5F) << "batch " << batch << ", node " << node;
        }
        else
        {
            EXPECT_TRUE(std::isinf(actual[index]) && actual[index] < 0.0F) << "batch " << batch << ", node " << node;
            EXPECT_TRUE(std::isinf(expected[index]) && expected[index] < 0.0F)
                << "batch " << batch << ", node " << node;
        }
    }
}

void expectOutputsEqual(TreeOutputs const& actual, CpuReference const& reference, DDTreeCase const& testCase)
{
    TreeOutputs const& expected = reference.outputs;
    EXPECT_EQ(actual.validCounts, expected.validCounts);
    EXPECT_EQ(actual.nodeTokenIds, expected.nodeTokenIds);
    EXPECT_EQ(actual.nodeDepths, expected.nodeDepths);
    EXPECT_EQ(actual.parentIds, expected.parentIds);
    expectScoresEqual(actual.nodeScores, expected.nodeScores, expected.validCounts, testCase.verifySize);
    EXPECT_EQ(actual.verifyTokenIds, expected.verifyTokenIds);
    EXPECT_EQ(actual.positionIds, expected.positionIds);
    EXPECT_EQ(actual.packedMask, expected.packedMask);
    EXPECT_EQ(actual.treeMask, expected.treeMask);
    EXPECT_EQ(actual.contextLengths, expected.contextLengths);
    EXPECT_EQ(actual.selectTokenIndices, expected.selectTokenIndices);
    ASSERT_FALSE(reference.competingExpansionMargins.empty());
    for (float const margin : reference.competingExpansionMargins)
    {
        EXPECT_GT(margin, kMinimumExpansionMargin);
    }
}

} // namespace

TEST(DFlashBenchDDTree, FiniteShallowTreeMatchesIndependentCpuReference)
{
    DDTreeCase const testCase = makePrimaryCase();
    CpuReference const reference = buildCpuReference(testCase);
    TreeOutputs const actual = runProductionDDTree(testCase);
    expectOutputsEqual(actual, reference, testCase);

    TreeOutputs const& expected = reference.outputs;
    EXPECT_EQ(expected.validCounts, (std::vector<int32_t>{7}));
    EXPECT_EQ(expected.nodeTokenIds, (std::vector<int32_t>{17, 3, 7, 11, 5, 7, 11, 0}));
    EXPECT_EQ(expected.nodeDepths, (std::vector<int32_t>{0, 1, 2, 2, 1, 2, 2, 0}));
    EXPECT_EQ(expected.parentIds, (std::vector<int32_t>{-1, 0, 1, 1, 0, 4, 4, -1}));
    EXPECT_EQ(expected.positionIds, (std::vector<int32_t>{100, 101, 102, 102, 101, 102, 102, 0}));
}

TEST(DFlashBenchDDTree, PackedAndUnpackedMasksMatchCpuAncestorReconstruction)
{
    DDTreeCase const testCase = makePrimaryCase();
    CpuReference const reference = buildCpuReference(testCase);
    TreeOutputs const actual = runProductionDDTree(testCase);
    EXPECT_EQ(actual.treeMask, reference.outputs.treeMask);
    EXPECT_EQ(actual.packedMask, reference.outputs.packedMask);
}

TEST(DFlashBenchDDTree, MultiBatchMaskCrossesPackedWordBoundary)
{
    DDTreeCase const testCase = makePackedBoundaryCase();
    CpuReference const reference = buildCpuReference(testCase);
    TreeOutputs const actual = runProductionDDTree(testCase);
    expectOutputsEqual(actual, reference, testCase);

    constexpr int32_t kPackedWords{2};
    constexpr int32_t kFirstBoundaryNode{31};
    constexpr int32_t kSecondBoundaryNode{32};
    constexpr int32_t kFinalWordUsedBits{8};
    EXPECT_EQ(reference.outputs.validCounts, (std::vector<int32_t>{40, 40}));
    for (int32_t batch = 0; batch < testCase.batchSize; ++batch)
    {
        size_t const row31 = static_cast<size_t>(batch * testCase.verifySize + kFirstBoundaryNode) * kPackedWords;
        size_t const row32 = static_cast<size_t>(batch * testCase.verifySize + kSecondBoundaryNode) * kPackedWords;
        EXPECT_EQ((static_cast<uint32_t>(actual.packedMask[row31]) >> 31) & 1U, 1U);
        EXPECT_EQ(static_cast<uint32_t>(actual.packedMask[row32 + 1]) & 1U, 1U);
        for (int32_t row = 0; row < testCase.verifySize; ++row)
        {
            size_t const finalWord = static_cast<size_t>(batch * testCase.verifySize + row) * kPackedWords + 1;
            EXPECT_EQ(static_cast<uint32_t>(actual.packedMask[finalWord]) >> kFinalWordUsedBits, 0U)
                << "batch " << batch << ", row " << row;
        }
    }
}
