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
#include "kernels/speculative/ddtreeKernels.h"
#include "testUtils.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kMaskBitsPerWord{32};

struct Candidate
{
    int32_t tokenId{0};
    float score{-std::numeric_limits<float>::infinity()};
};

struct DDTreeReference
{
    std::vector<int32_t> nodeTokenIds;
    std::vector<int32_t> nodeDepths;
    std::vector<int32_t> parentIds;
    std::vector<float> nodeScores;
    std::vector<int32_t> validCounts;
    std::vector<int32_t> verifyTokenIds;
    std::vector<int32_t> verifyPositionIds;
    std::vector<int32_t> packedAncestorMask;
    std::vector<int8_t> ancestorMask;
    std::vector<int32_t> contextLengths;
    std::vector<int64_t> selectTokenIndices;
};

bool betterCandidate(Candidate const& lhs, Candidate const& rhs)
{
    return lhs.score > rhs.score || (lhs.score == rhs.score && lhs.tokenId < rhs.tokenId);
}

void insertCandidate(std::vector<Candidate>& candidates, Candidate candidate)
{
    for (size_t slot = 0; slot < candidates.size(); ++slot)
    {
        if (betterCandidate(candidate, candidates[slot]))
        {
            for (size_t move = candidates.size() - 1; move > slot; --move)
            {
                candidates[move] = candidates[move - 1];
            }
            candidates[slot] = candidate;
            break;
        }
    }
}

std::vector<Candidate> selectDepthCandidates(std::vector<float> const& logits, int32_t batchIdx, int32_t depthIdx,
    int32_t dflashBlockSize, int32_t vocabSize, int32_t candidateTopK)
{
    int64_t const logitsOffset
        = static_cast<int64_t>(batchIdx) * dflashBlockSize * vocabSize + static_cast<int64_t>(depthIdx) * vocabSize;
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        maxLogit = std::max(maxLogit, logits[logitsOffset + vocabIdx]);
    }

    float expSum{0.0F};
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        expSum += std::exp(logits[logitsOffset + vocabIdx] - maxLogit);
    }
    float const logSumExp = maxLogit + std::log(expSum);

    std::vector<Candidate> candidates(static_cast<size_t>(candidateTopK));
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        insertCandidate(candidates, Candidate{vocabIdx, logits[logitsOffset + vocabIdx] - logSumExp});
    }
    return candidates;
}

bool betterExpansion(float lhsScore, int32_t lhsParent, int32_t lhsSlot, int32_t lhsToken, float rhsScore,
    int32_t rhsParent, int32_t rhsSlot, int32_t rhsToken)
{
    if (lhsScore != rhsScore)
    {
        return lhsScore > rhsScore;
    }
    if (lhsParent != rhsParent)
    {
        return lhsParent < rhsParent;
    }
    if (lhsSlot != rhsSlot)
    {
        return lhsSlot < rhsSlot;
    }
    return lhsToken < rhsToken;
}

DDTreeReference buildReference(std::vector<float> const& logits, std::vector<int32_t> const& rootTokenIds,
    std::vector<int32_t> const& baseLengths, int32_t batchSize, int32_t dflashBlockSize, int32_t vocabSize,
    int32_t verifySize, int32_t candidateTopK, int32_t firstCandidateLogitsRow = 1,
    std::vector<float> const& depthConfidence = {}, float survivalThreshold = 0.0F)
{
    int32_t const packedMaskLen = (verifySize + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    DDTreeReference ref;
    ref.nodeTokenIds.assign(static_cast<size_t>(batchSize) * verifySize, 0);
    ref.nodeDepths.assign(static_cast<size_t>(batchSize) * verifySize, 0);
    ref.parentIds.assign(static_cast<size_t>(batchSize) * verifySize, -1);
    ref.nodeScores.assign(static_cast<size_t>(batchSize) * verifySize, -std::numeric_limits<float>::infinity());
    ref.validCounts.assign(batchSize, 0);
    ref.verifyTokenIds.assign(static_cast<size_t>(batchSize) * verifySize, 0);
    ref.verifyPositionIds.assign(static_cast<size_t>(batchSize) * verifySize, 0);
    ref.packedAncestorMask.assign(static_cast<size_t>(batchSize) * verifySize * packedMaskLen, 0);
    ref.ancestorMask.assign(static_cast<size_t>(batchSize) * verifySize * verifySize, 0);

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        std::vector<std::vector<Candidate>> depthCandidates(static_cast<size_t>(dflashBlockSize));
        for (int32_t depthIdx = 0; depthIdx < dflashBlockSize; ++depthIdx)
        {
            depthCandidates[depthIdx]
                = selectDepthCandidates(logits, batchIdx, depthIdx, dflashBlockSize, vocabSize, candidateTopK);
        }

        int32_t const treeOffset = batchIdx * verifySize;
        ref.nodeTokenIds[treeOffset] = rootTokenIds[batchIdx];
        ref.nodeDepths[treeOffset] = 0;
        ref.parentIds[treeOffset] = -1;
        ref.nodeScores[treeOffset] = 0.0F;

        int32_t validCount{1};
        int32_t maxProposalDepth = dflashBlockSize - 1;
        if (!depthConfidence.empty() && survivalThreshold > 0.0F)
        {
            float survivalLog = 0.0F;
            maxProposalDepth = 0;
            for (int32_t depth = 1; depth < dflashBlockSize; ++depth)
            {
                survivalLog += std::log(std::max(
                    depthConfidence[static_cast<size_t>(batchIdx) * (dflashBlockSize - 1) + depth - 1], 1e-20F));
                if (survivalLog < std::log(survivalThreshold))
                {
                    break;
                }
                maxProposalDepth = depth;
            }
            // Parity with the chain scheduler's minProposalLen >= 1.
            maxProposalDepth = std::max(maxProposalDepth, 1);
        }
        std::vector<int32_t> nextCandidateSlot(static_cast<size_t>(verifySize), 0);
        for (int32_t outNodeIdx = 1; outNodeIdx < verifySize; ++outNodeIdx)
        {
            int32_t bestParent{-1};
            int32_t bestSlot{-1};
            int32_t bestToken{0};
            float bestScore{-std::numeric_limits<float>::infinity()};
            for (int32_t parentIdx = 0; parentIdx < validCount; ++parentIdx)
            {
                int32_t const parentDepth = ref.nodeDepths[treeOffset + parentIdx];
                if (parentDepth >= maxProposalDepth)
                {
                    continue;
                }

                int32_t const slot = nextCandidateSlot[static_cast<size_t>(parentIdx)];
                if (slot >= candidateTopK)
                {
                    continue;
                }

                Candidate const candidate
                    = depthCandidates[static_cast<size_t>(firstCandidateLogitsRow + parentDepth)][slot];
                if (!std::isfinite(candidate.score))
                {
                    continue;
                }

                float score = ref.nodeScores[treeOffset + parentIdx] + candidate.score;
                if (!depthConfidence.empty())
                {
                    float const conf
                        = depthConfidence[static_cast<size_t>(batchIdx) * (dflashBlockSize - 1) + parentDepth];
                    score += std::log(std::max(conf, 1e-20F));
                }
                if (bestParent < 0
                    || betterExpansion(
                        score, parentIdx, slot, candidate.tokenId, bestScore, bestParent, bestSlot, bestToken))
                {
                    bestParent = parentIdx;
                    bestSlot = slot;
                    bestToken = candidate.tokenId;
                    bestScore = score;
                }
            }

            if (bestParent < 0)
            {
                break;
            }
            nextCandidateSlot[static_cast<size_t>(bestParent)] += 1;
            ref.nodeTokenIds[treeOffset + outNodeIdx] = bestToken;
            ref.nodeDepths[treeOffset + outNodeIdx] = ref.nodeDepths[treeOffset + bestParent] + 1;
            ref.parentIds[treeOffset + outNodeIdx] = bestParent;
            ref.nodeScores[treeOffset + outNodeIdx] = bestScore;
            validCount += 1;
        }
        ref.validCounts[batchIdx] = validCount;

        int32_t const packedOffset = batchIdx * verifySize * packedMaskLen;
        int32_t const maskOffset = batchIdx * verifySize * verifySize;
        for (int32_t nodeIdx = 0; nodeIdx < validCount; ++nodeIdx)
        {
            ref.verifyTokenIds[treeOffset + nodeIdx] = ref.nodeTokenIds[treeOffset + nodeIdx];
            ref.verifyPositionIds[treeOffset + nodeIdx] = baseLengths[batchIdx] + ref.nodeDepths[treeOffset + nodeIdx];
            int32_t ancestorIdx = nodeIdx;
            while (ancestorIdx >= 0)
            {
                int32_t const wordIdx = ancestorIdx / kMaskBitsPerWord;
                int32_t const bitIdx = ancestorIdx % kMaskBitsPerWord;
                ref.packedAncestorMask[packedOffset + nodeIdx * packedMaskLen + wordIdx] |= 1U << bitIdx;
                ref.ancestorMask[maskOffset + nodeIdx * verifySize + ancestorIdx] = 1;
                ancestorIdx = ref.parentIds[treeOffset + ancestorIdx];
            }
        }
    }

    return ref;
}

struct DDTreeBuildOutputs
{
    std::vector<int32_t> nodeTokenIds;
    std::vector<int32_t> nodeDepths;
    std::vector<int32_t> parentIds;
    std::vector<float> nodeScores;
    std::vector<int32_t> validCounts;
    std::vector<int32_t> verifyTokenIds;
    std::vector<int32_t> verifyPositionIds;
    std::vector<int32_t> packedAncestorMask;
    std::vector<int8_t> ancestorMask;
    std::vector<int32_t> contextLengths;
    std::vector<int64_t> selectTokenIndices;
};

DDTreeBuildOutputs runDDTreeBuild(std::vector<float> const& logits, std::vector<int32_t> const& rootTokenIds,
    std::vector<int32_t> const& baseLengths, int32_t batchSize, int32_t dflashBlockSize, int32_t vocabSize,
    int32_t verifySize, int32_t candidateTopK, int32_t firstCandidateLogitsRow = 1,
    std::vector<float> const& depthConfidence = {}, float survivalThreshold = 0.0F)
{
    int32_t const packedMaskLen = (verifySize + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    rt::Tensor logitsTensor({batchSize, dflashBlockSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor rootTokenTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor baseLengthsTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor nodeTokenIdsTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor nodeDepthsTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor parentIdsTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor nodeScoresTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor validCountsTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor verifyTokenIdsTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor verifyPositionIdsTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor packedAncestorMaskTensor(
        {batchSize, verifySize, packedMaskLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor ancestorMaskTensor({batchSize, verifySize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor contextLengthsTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor selectTokenIndicesTensor({batchSize, verifySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);

    copyHostToDevice(logitsTensor, logits);
    copyHostToDevice(rootTokenTensor, rootTokenIds);
    copyHostToDevice(baseLengthsTensor, baseLengths);

    size_t const workspaceSize
        = kernel::getDDTreeBuildWorkspaceSize(batchSize, dflashBlockSize, verifySize, vocabSize, candidateTopK);
    EXPECT_GT(workspaceSize, 0U);
    void* workspace{nullptr};
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    Defer workspaceGuard{[&workspace]() { CUDA_CHECK(cudaFree(workspace)); }};

    rt::Tensor depthConfidenceTensor;
    if (!depthConfidence.empty())
    {
        depthConfidenceTensor
            = rt::Tensor({batchSize, dflashBlockSize - 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        copyHostToDevice(depthConfidenceTensor, depthConfidence);
    }

    kernel::DDTreeBuildParams const buildParams{
        {logitsTensor, rootTokenTensor, baseLengthsTensor, nullptr,
            depthConfidence.empty() ? nullptr : &depthConfidenceTensor, survivalThreshold},
        {nodeTokenIdsTensor, nodeDepthsTensor, parentIdsTensor, nodeScoresTensor, validCountsTensor,
            verifyTokenIdsTensor, verifyPositionIdsTensor, packedAncestorMaskTensor, ancestorMaskTensor,
            contextLengthsTensor, selectTokenIndicesTensor},
        candidateTopK, workspace, workspaceSize, nullptr, firstCandidateLogitsRow};
    kernel::ddtreeBuild(buildParams);
    CUDA_CHECK(cudaDeviceSynchronize());

    return DDTreeBuildOutputs{copyDeviceToHost<int32_t>(nodeTokenIdsTensor),
        copyDeviceToHost<int32_t>(nodeDepthsTensor), copyDeviceToHost<int32_t>(parentIdsTensor),
        copyDeviceToHost<float>(nodeScoresTensor), copyDeviceToHost<int32_t>(validCountsTensor),
        copyDeviceToHost<int32_t>(verifyTokenIdsTensor), copyDeviceToHost<int32_t>(verifyPositionIdsTensor),
        copyDeviceToHost<int32_t>(packedAncestorMaskTensor), copyDeviceToHost<int8_t>(ancestorMaskTensor),
        copyDeviceToHost<int32_t>(contextLengthsTensor), copyDeviceToHost<int64_t>(selectTokenIndicesTensor)};
}

void validateBuildAgainstReference(
    DDTreeBuildOutputs const& actual, DDTreeReference const& expected, int32_t batchSize, int32_t verifySize)
{
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        EXPECT_EQ(actual.validCounts[batchIdx], expected.validCounts[batchIdx]);
        for (int32_t nodeIdx = 0; nodeIdx < verifySize; ++nodeIdx)
        {
            int32_t const idx = batchIdx * verifySize + nodeIdx;
            EXPECT_EQ(actual.nodeTokenIds[idx], expected.nodeTokenIds[idx]) << "node token mismatch at " << idx;
            EXPECT_EQ(actual.nodeDepths[idx], expected.nodeDepths[idx]) << "node depth mismatch at " << idx;
            EXPECT_EQ(actual.parentIds[idx], expected.parentIds[idx]) << "parent mismatch at " << idx;
            EXPECT_EQ(actual.verifyTokenIds[idx], expected.verifyTokenIds[idx]) << "verify token mismatch at " << idx;
            EXPECT_EQ(actual.verifyPositionIds[idx], expected.verifyPositionIds[idx])
                << "position id mismatch at " << idx;
            if (nodeIdx < expected.validCounts[batchIdx])
            {
                EXPECT_NEAR(actual.nodeScores[idx], expected.nodeScores[idx], 1.0e-5F) << "score mismatch at " << idx;
                if (nodeIdx > 0)
                {
                    EXPECT_LT(actual.parentIds[idx], nodeIdx);
                    EXPECT_EQ(
                        actual.nodeDepths[idx], actual.nodeDepths[batchIdx * verifySize + actual.parentIds[idx]] + 1);
                }
            }
            else
            {
                EXPECT_EQ(actual.parentIds[idx], -1);
                EXPECT_TRUE(std::isinf(actual.nodeScores[idx]) && actual.nodeScores[idx] < 0.0F);
            }
        }
    }
    EXPECT_EQ(actual.packedAncestorMask, expected.packedAncestorMask);
    EXPECT_EQ(actual.ancestorMask, expected.ancestorMask);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        EXPECT_EQ(actual.contextLengths[batchIdx], expected.verifyPositionIds[batchIdx * verifySize] + verifySize);
        for (int32_t nodeIdx = 0; nodeIdx < verifySize; ++nodeIdx)
        {
            EXPECT_EQ(actual.selectTokenIndices[batchIdx * verifySize + nodeIdx], nodeIdx);
        }
    }
}

void setLogit(std::vector<float>& logits, int32_t batchIdx, int32_t depthIdx, int32_t tokenId, int32_t dflashBlockSize,
    int32_t vocabSize, float value)
{
    logits[static_cast<size_t>(batchIdx) * dflashBlockSize * vocabSize + depthIdx * vocabSize + tokenId] = value;
}

} // anonymous namespace

TEST(DDTreeKernels, BuildPrefixClosedTreeAndPackedMask)
{
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDFlashBlockSize{4};
    constexpr int32_t kVocabSize{12};
    constexpr int32_t kVerifySize{7};
    constexpr int32_t kCandidateTopK{3};
    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kDFlashBlockSize * kVocabSize, -12.0F);
    for (int32_t depthIdx = 1; depthIdx < kDFlashBlockSize; ++depthIdx)
    {
        for (int32_t tokenId = 0; tokenId < kVocabSize; ++tokenId)
        {
            setLogit(logits, 0, depthIdx, tokenId, kDFlashBlockSize, kVocabSize, -8.0F - tokenId);
        }
    }
    setLogit(logits, 0, 1, 1, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 1, 2, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 1, 3, kDFlashBlockSize, kVocabSize, 4.0F);
    setLogit(logits, 0, 2, 4, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 2, 5, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 6, kDFlashBlockSize, kVocabSize, 4.0F);
    setLogit(logits, 0, 3, 7, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 3, 8, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 3, 9, kDFlashBlockSize, kVocabSize, 4.0F);

    std::vector<int32_t> const rootTokenIds{90};
    std::vector<int32_t> const baseLengths{20};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);

    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);
    EXPECT_EQ(actual.nodeDepths[0], 0);
    EXPECT_EQ(actual.parentIds[0], -1);
    EXPECT_EQ(actual.packedAncestorMask[0], 1);
}

TEST(DDTreeKernels, CanUseRowZeroForDepthOne)
{
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDFlashBlockSize{4};
    constexpr int32_t kVocabSize{8};
    constexpr int32_t kVerifySize{4};
    constexpr int32_t kCandidateTopK{1};
    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kDFlashBlockSize * kVocabSize, -8.0F);
    setLogit(logits, 0, 0, 1, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 1, 2, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 2, 3, kDFlashBlockSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 3, 7, kDFlashBlockSize, kVocabSize, 99.0F);

    std::vector<int32_t> const rootTokenIds{90};
    std::vector<int32_t> const baseLengths{20};
    DDTreeBuildOutputs actual = runDDTreeBuild(logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize,
        kVocabSize, kVerifySize, kCandidateTopK, /*firstCandidateLogitsRow=*/0);
    DDTreeReference expected = buildReference(logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize,
        kVocabSize, kVerifySize, kCandidateTopK, /*firstCandidateLogitsRow=*/0);

    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);
    EXPECT_EQ(actual.nodeTokenIds[0], 90);
    EXPECT_EQ(actual.nodeTokenIds[1], 1);
    EXPECT_EQ(actual.nodeTokenIds[2], 2);
    EXPECT_EQ(actual.nodeTokenIds[3], 3);
}

// MTP tree drafting stacks one full logits row per chain depth into a
// [batch, draftingStep+1, vocab] buffer and feeds it to ddtreeBuild with
// "blockSize" = draftingStep+1. Row 0 is a root placeholder that the builder
// must ignore (the root token comes from rootTokenIds, not from logits).
TEST(DDTreeKernels, MtpChainStackedLogitsBuildsTree)
{
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDraftingStep{3};
    constexpr int32_t kProposalDepthSize{kDraftingStep + 1}; // MTP "blockSize"
    constexpr int32_t kVocabSize{10};
    constexpr int32_t kCandidateFanout{2};
    // Within the tree capacity (full fanout-ary tree of depth step: 1+2+4+8=15),
    // so the greedy builder fills every requested node.
    constexpr int32_t kVerifySize{7};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -12.0F);
    // Poison the root placeholder row: it must not contribute any candidate.
    for (int32_t tokenId = 0; tokenId < kVocabSize; ++tokenId)
    {
        setLogit(logits, 0, 0, tokenId, kProposalDepthSize, kVocabSize, 999.0F);
    }
    // Depth rows 1..step mimic the chain-conditioned per-round logits.
    setLogit(logits, 0, 1, 1, kProposalDepthSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 1, 2, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 4, kProposalDepthSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 2, 5, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 3, 7, kProposalDepthSize, kVocabSize, 6.0F);
    setLogit(logits, 0, 3, 8, kProposalDepthSize, kVocabSize, 5.0F);

    std::vector<int32_t> const rootTokenIds{42};
    std::vector<int32_t> const baseLengths{100};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    // verifySize is within the tree capacity, so every requested node is filled
    // and no poisoned root-row token appears.
    EXPECT_EQ(actual.validCounts[0], kVerifySize);
    EXPECT_EQ(actual.nodeTokenIds[0], 42);
    int32_t maxDepth{0};
    for (int32_t nodeIdx = 0; nodeIdx < actual.validCounts[0]; ++nodeIdx)
    {
        maxDepth = std::max(maxDepth, actual.nodeDepths[nodeIdx]);
        if (actual.nodeDepths[nodeIdx] == 1)
        {
            EXPECT_TRUE(actual.nodeTokenIds[nodeIdx] == 1 || actual.nodeTokenIds[nodeIdx] == 2)
                << "depth-1 candidates must come from logits row 1";
        }
    }
    // Proposal depth is capped at draftingStep (= "blockSize" - 1).
    EXPECT_EQ(maxDepth, kDraftingStep);
}

// With candidateFanout = 1 the greedy builder must reproduce the linear MTP
// chain exactly: parent[i] = i-1, depth[i] = i, one token per depth row.
TEST(DDTreeKernels, MtpFanoutOneDegeneratesToChain)
{
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDraftingStep{4};
    constexpr int32_t kProposalDepthSize{kDraftingStep + 1};
    constexpr int32_t kVocabSize{6};
    constexpr int32_t kCandidateFanout{1};
    constexpr int32_t kVerifySize{1 + kDraftingStep};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -2.0F);
    for (int32_t depthIdx = 1; depthIdx < kProposalDepthSize; ++depthIdx)
    {
        setLogit(logits, 0, depthIdx, depthIdx, kProposalDepthSize, kVocabSize, 5.0F);
    }

    std::vector<int32_t> const rootTokenIds{3};
    std::vector<int32_t> const baseLengths{64};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    EXPECT_EQ(actual.validCounts[0], kVerifySize);
    for (int32_t nodeIdx = 0; nodeIdx < kVerifySize; ++nodeIdx)
    {
        EXPECT_EQ(actual.parentIds[nodeIdx], nodeIdx - 1) << "chain parent mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.nodeDepths[nodeIdx], nodeIdx) << "chain depth mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.nodeTokenIds[nodeIdx], nodeIdx == 0 ? 3 : nodeIdx)
            << "chain token mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.verifyPositionIds[nodeIdx], 64 + nodeIdx);
    }
}

TEST(DDTreeKernels, DSparkFanoutOneWithLargerNodeBudgetDegeneratesToChain)
{
    // DSpark tree decouples verifySize from the chain length: block7 stacked logits
    // (row 0 = root placeholder) with a 16-node budget must yield the exact greedy
    // chain (root + 7) followed by padding nodes.
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kBlockSize{7};
    constexpr int32_t kProposalDepthSize{kBlockSize + 1};
    constexpr int32_t kVocabSize{9};
    constexpr int32_t kCandidateFanout{1};
    constexpr int32_t kVerifySize{16};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -2.0F);
    for (int32_t depthIdx = 1; depthIdx < kProposalDepthSize; ++depthIdx)
    {
        setLogit(logits, 0, depthIdx, depthIdx, kProposalDepthSize, kVocabSize, 5.0F);
    }

    std::vector<int32_t> const rootTokenIds{8};
    std::vector<int32_t> const baseLengths{64};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    constexpr int32_t kChainLength{kBlockSize + 1};
    EXPECT_EQ(actual.validCounts[0], kChainLength);
    for (int32_t nodeIdx = 0; nodeIdx < kChainLength; ++nodeIdx)
    {
        EXPECT_EQ(actual.parentIds[nodeIdx], nodeIdx - 1) << "chain parent mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.nodeDepths[nodeIdx], nodeIdx) << "chain depth mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.nodeTokenIds[nodeIdx], nodeIdx == 0 ? 8 : nodeIdx)
            << "chain token mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.verifyPositionIds[nodeIdx], 64 + nodeIdx);
    }
    for (int32_t nodeIdx = kChainLength; nodeIdx < kVerifySize; ++nodeIdx)
    {
        EXPECT_EQ(actual.parentIds[nodeIdx], -1) << "padding parent mismatch at node " << nodeIdx;
        EXPECT_EQ(actual.verifyTokenIds[nodeIdx], 0);
        EXPECT_EQ(actual.verifyPositionIds[nodeIdx], 0);
    }
}

TEST(DDTreeKernels, DepthConfidenceShiftsBudgetToShallowDepths)
{
    // Unbiased, the dominant depth-1 chain pulls the budget deep. With low
    // confidence at depth 2, the accumulated log(conf) penalty must flip the
    // last slot back to depth-1 width.
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kProposalDepthSize{3};
    constexpr int32_t kVocabSize{4};
    constexpr int32_t kCandidateFanout{2};
    constexpr int32_t kVerifySize{4};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -2.0F);
    setLogit(logits, 0, 1, 0, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 1, 1, kProposalDepthSize, kVocabSize, 2.5F);
    setLogit(logits, 0, 2, 2, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 3, kProposalDepthSize, kVocabSize, 4.0F);

    std::vector<int32_t> const rootTokenIds{7};
    std::vector<int32_t> const baseLengths{10};
    std::vector<float> const depthConfidence{0.9F, 0.2F};

    DDTreeBuildOutputs unbiased = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize, kVocabSize, kVerifySize, kCandidateFanout);
    DDTreeBuildOutputs actual = runDDTreeBuild(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence);
    DDTreeReference expected = buildReference(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    auto countDepth = [&](DDTreeBuildOutputs const& tree, int32_t depth) {
        int32_t count = 0;
        for (int32_t nodeIdx = 0; nodeIdx < tree.validCounts[0]; ++nodeIdx)
        {
            count += tree.nodeDepths[nodeIdx] == depth ? 1 : 0;
        }
        return count;
    };
    EXPECT_EQ(countDepth(unbiased, 1), 1) << "unbiased tree spends the last slot on depth-2 width";
    EXPECT_EQ(countDepth(unbiased, 2), 2);
    EXPECT_EQ(countDepth(actual, 1), 2) << "low-confidence depth 2 must push the budget to depth-1 width";
    EXPECT_EQ(countDepth(actual, 2), 1);
}

TEST(DDTreeKernels, SurvivalThresholdCapsGrowthDepth)
{
    // Survival product: depth 1 = 0.9, depth 2 = 0.9 * 0.2 = 0.18 < 0.25: depth 2
    // must stop growing entirely and the tree fills what depth 1 can hold.
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kProposalDepthSize{3};
    constexpr int32_t kVocabSize{4};
    constexpr int32_t kCandidateFanout{2};
    constexpr int32_t kVerifySize{4};
    constexpr float kSurvivalThreshold{0.25F};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -2.0F);
    setLogit(logits, 0, 1, 0, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 1, 1, kProposalDepthSize, kVocabSize, 2.5F);
    setLogit(logits, 0, 2, 2, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 3, kProposalDepthSize, kVocabSize, 4.0F);

    std::vector<int32_t> const rootTokenIds{7};
    std::vector<int32_t> const baseLengths{10};
    std::vector<float> const depthConfidence{0.9F, 0.2F};

    DDTreeBuildOutputs actual = runDDTreeBuild(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence, kSurvivalThreshold);
    DDTreeReference expected = buildReference(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence, kSurvivalThreshold);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    EXPECT_EQ(actual.validCounts[0], 3) << "root + full depth-1 fanout is all the capped tree can hold";
    for (int32_t nodeIdx = 0; nodeIdx < actual.validCounts[0]; ++nodeIdx)
    {
        EXPECT_LE(actual.nodeDepths[nodeIdx], 1) << "no node may grow past the survival cutoff depth";
    }
}

TEST(DDTreeKernels, SurvivalThresholdKeepsAtLeastOneDepth)
{
    // Depth 1 confidence 0.1 < 0.5 floor: without the min-one-proposal clamp the
    // tree would collapse to the root alone.
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kProposalDepthSize{3};
    constexpr int32_t kVocabSize{4};
    constexpr int32_t kCandidateFanout{2};
    constexpr int32_t kVerifySize{4};
    constexpr float kSurvivalThreshold{0.5F};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kProposalDepthSize * kVocabSize, -2.0F);
    setLogit(logits, 0, 1, 0, kProposalDepthSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 1, kProposalDepthSize, kVocabSize, 5.0F);

    std::vector<int32_t> const rootTokenIds{3};
    std::vector<int32_t> const baseLengths{16};
    std::vector<float> const depthConfidence{0.1F, 0.9F};

    DDTreeBuildOutputs actual = runDDTreeBuild(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence, kSurvivalThreshold);
    DDTreeReference expected = buildReference(logits, rootTokenIds, baseLengths, kBatchSize, kProposalDepthSize,
        kVocabSize, kVerifySize, kCandidateFanout, /*firstCandidateLogitsRow=*/1, depthConfidence, kSurvivalThreshold);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    EXPECT_EQ(actual.validCounts[0], 1 + kCandidateFanout);
    for (int32_t nodeIdx = 1; nodeIdx < actual.validCounts[0]; ++nodeIdx)
    {
        EXPECT_EQ(actual.nodeDepths[nodeIdx], 1) << "node " << nodeIdx << " must stay at depth 1";
    }
}

TEST(DDTreeKernels, MultiBatchPaddingUsesFixedVerifyShape)
{
    constexpr int32_t kBatchSize{2};
    constexpr int32_t kDFlashBlockSize{3};
    constexpr int32_t kVocabSize{2};
    constexpr int32_t kVerifySize{8};
    constexpr int32_t kCandidateTopK{2};
    constexpr int32_t kPackedMaskLen{1};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kDFlashBlockSize * kVocabSize, -2.0F);
    setLogit(logits, 0, 1, 0, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 1, 1, kDFlashBlockSize, kVocabSize, 3.0F);
    setLogit(logits, 0, 2, 0, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 0, 2, 1, kDFlashBlockSize, kVocabSize, 3.0F);
    setLogit(logits, 1, 1, 0, kDFlashBlockSize, kVocabSize, 3.0F);
    setLogit(logits, 1, 1, 1, kDFlashBlockSize, kVocabSize, 5.0F);
    setLogit(logits, 1, 2, 0, kDFlashBlockSize, kVocabSize, 3.0F);
    setLogit(logits, 1, 2, 1, kDFlashBlockSize, kVocabSize, 5.0F);

    std::vector<int32_t> const rootTokenIds{100, 101};
    std::vector<int32_t> const baseLengths{5, 12};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);
    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);

    EXPECT_EQ(actual.validCounts[0], 7);
    EXPECT_EQ(actual.validCounts[1], 7);
    EXPECT_NE(actual.nodeTokenIds[1], actual.nodeTokenIds[kVerifySize + 1]);
    for (int32_t batchIdx = 0; batchIdx < kBatchSize; ++batchIdx)
    {
        int32_t const paddingNodeIdx = batchIdx * kVerifySize + 7;
        EXPECT_EQ(actual.parentIds[paddingNodeIdx], -1);
        EXPECT_EQ(actual.verifyTokenIds[paddingNodeIdx], 0);
        EXPECT_EQ(actual.verifyPositionIds[paddingNodeIdx], 0);
        EXPECT_EQ(actual.packedAncestorMask[batchIdx * kVerifySize * kPackedMaskLen + 7], 0);
        for (int32_t colIdx = 0; colIdx < kVerifySize; ++colIdx)
        {
            EXPECT_EQ(actual.ancestorMask[batchIdx * kVerifySize * kVerifySize + 7 * kVerifySize + colIdx], 0);
        }
    }
}

TEST(DDTreeKernels, SupportsVerifySize128)
{
    constexpr int32_t kBatchSize{1};
    constexpr int32_t kDFlashBlockSize{16};
    constexpr int32_t kVocabSize{64};
    constexpr int32_t kVerifySize{128};
    constexpr int32_t kCandidateTopK{8};
    constexpr int32_t kPackedMaskLen{kVerifySize / kMaskBitsPerWord};

    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kDFlashBlockSize * kVocabSize, -12.0F);
    for (int32_t depthIdx = 1; depthIdx < kDFlashBlockSize; ++depthIdx)
    {
        for (int32_t tokenId = 0; tokenId < kVocabSize; ++tokenId)
        {
            setLogit(logits, 0, depthIdx, tokenId, kDFlashBlockSize, kVocabSize,
                10.0F - static_cast<float>(tokenId) * 0.01F - static_cast<float>(depthIdx) * 0.001F);
        }
    }

    std::vector<int32_t> const rootTokenIds{151669};
    std::vector<int32_t> const baseLengths{128};
    DDTreeBuildOutputs actual = runDDTreeBuild(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);

    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);
    EXPECT_EQ(actual.validCounts[0], kVerifySize);
    EXPECT_EQ(kPackedMaskLen, 4);
    EXPECT_EQ(actual.contextLengths[0], baseLengths[0] + kVerifySize);
}
