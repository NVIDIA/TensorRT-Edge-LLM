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
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "testUtils.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
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
    int32_t draftBlockSize, int32_t vocabSize, int32_t candidateTopK)
{
    int64_t const logitsOffset
        = static_cast<int64_t>(batchIdx) * draftBlockSize * vocabSize + static_cast<int64_t>(depthIdx) * vocabSize;
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
    std::vector<int32_t> const& baseLengths, int32_t batchSize, int32_t draftBlockSize, int32_t vocabSize,
    int32_t verifyTokenCount, int32_t candidateTopK, int32_t firstCandidateLogitsRow = 1,
    std::vector<float> const& depthConfidence = {}, float survivalThreshold = 0.0F)
{
    int32_t const packedMaskLen = (verifyTokenCount + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    DDTreeReference ref;
    ref.nodeTokenIds.assign(static_cast<size_t>(batchSize) * verifyTokenCount, 0);
    ref.nodeDepths.assign(static_cast<size_t>(batchSize) * verifyTokenCount, 0);
    ref.parentIds.assign(static_cast<size_t>(batchSize) * verifyTokenCount, -1);
    ref.nodeScores.assign(static_cast<size_t>(batchSize) * verifyTokenCount, -std::numeric_limits<float>::infinity());
    ref.validCounts.assign(batchSize, 0);
    ref.verifyTokenIds.assign(static_cast<size_t>(batchSize) * verifyTokenCount, 0);
    ref.verifyPositionIds.assign(static_cast<size_t>(batchSize) * verifyTokenCount, 0);
    ref.packedAncestorMask.assign(static_cast<size_t>(batchSize) * verifyTokenCount * packedMaskLen, 0);

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        std::vector<std::vector<Candidate>> depthCandidates(static_cast<size_t>(draftBlockSize));
        for (int32_t depthIdx = 0; depthIdx < draftBlockSize; ++depthIdx)
        {
            depthCandidates[depthIdx]
                = selectDepthCandidates(logits, batchIdx, depthIdx, draftBlockSize, vocabSize, candidateTopK);
        }

        int32_t const treeOffset = batchIdx * verifyTokenCount;
        ref.nodeTokenIds[treeOffset] = rootTokenIds[batchIdx];
        ref.nodeDepths[treeOffset] = 0;
        ref.parentIds[treeOffset] = -1;
        ref.nodeScores[treeOffset] = 0.0F;

        int32_t validCount{1};
        int32_t maxProposalDepth = draftBlockSize - 1;
        if (!depthConfidence.empty() && survivalThreshold > 0.0F)
        {
            float survivalLog = 0.0F;
            maxProposalDepth = 0;
            for (int32_t depth = 1; depth < draftBlockSize; ++depth)
            {
                survivalLog += std::log(std::max(
                    depthConfidence[static_cast<size_t>(batchIdx) * (draftBlockSize - 1) + depth - 1], 1e-20F));
                if (survivalLog < std::log(survivalThreshold))
                {
                    break;
                }
                maxProposalDepth = depth;
            }
            // Parity with the chain scheduler's minProposalLen >= 1.
            maxProposalDepth = std::max(maxProposalDepth, 1);
        }
        std::vector<int32_t> nextCandidateSlot(static_cast<size_t>(verifyTokenCount), 0);
        for (int32_t outNodeIdx = 1; outNodeIdx < verifyTokenCount; ++outNodeIdx)
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
                        = depthConfidence[static_cast<size_t>(batchIdx) * (draftBlockSize - 1) + parentDepth];
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

        int32_t const packedOffset = batchIdx * verifyTokenCount * packedMaskLen;
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
    std::vector<int32_t> contextLengths;
    std::vector<int64_t> selectTokenIndices;
};

DDTreeBuildOutputs runDDTreeBuild(std::vector<float> const& logits, std::vector<int32_t> const& rootTokenIds,
    std::vector<int32_t> const& baseLengths, int32_t batchSize, int32_t draftBlockSize, int32_t vocabSize,
    int32_t verifyTokenCount, int32_t candidateTopK, int32_t firstCandidateLogitsRow = 1,
    std::vector<float> const& depthConfidence = {}, float survivalThreshold = 0.0F, bool tokenMajorLogits = false)
{
    int32_t const packedMaskLen = (verifyTokenCount + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    rt::Tensor logitsTensor = tokenMajorLogits
        ? rt::Tensor({batchSize * draftBlockSize, vocabSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT)
        : rt::Tensor({batchSize, draftBlockSize, vocabSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    rt::Tensor rootTokenTensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor baseLengthsTensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor nodeTokenIdsTensor({batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor nodeDepthsTensor({batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor parentIdsTensor({batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor nodeScoresTensor({batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    rt::Tensor validCountsTensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor verifyTokenIdsTensor({batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor verifyPositionIdsTensor(
        {batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor packedAncestorMaskTensor(
        {batchSize, verifyTokenCount, packedMaskLen}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor contextLengthsTensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor selectTokenIndicesTensor(
        {batchSize, verifyTokenCount}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT64);

    copyHostToDevice(logitsTensor, logits);
    copyHostToDevice(rootTokenTensor, rootTokenIds);
    copyHostToDevice(baseLengthsTensor, baseLengths);

    size_t const workspaceSize
        = kernel::getDDTreeBuildWorkspaceSize(batchSize, draftBlockSize, verifyTokenCount, vocabSize, candidateTopK);
    EXPECT_GT(workspaceSize, 0U);
    void* workspace{nullptr};
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    Defer workspaceGuard{[&workspace]() { CUDA_CHECK(cudaFree(workspace)); }};

    rt::Tensor depthConfidenceTensor;
    if (!depthConfidence.empty())
    {
        depthConfidenceTensor
            = rt::Tensor({batchSize, draftBlockSize - 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
        copyHostToDevice(depthConfidenceTensor, depthConfidence);
    }

    kernel::DDTreeBuildParams const buildParams{
        {logitsTensor, rootTokenTensor, baseLengthsTensor, nullptr,
            depthConfidence.empty() ? nullptr : &depthConfidenceTensor, survivalThreshold},
        {nodeTokenIdsTensor, nodeDepthsTensor, parentIdsTensor, nodeScoresTensor, validCountsTensor,
            verifyTokenIdsTensor, verifyPositionIdsTensor, packedAncestorMaskTensor, contextLengthsTensor,
            selectTokenIndicesTensor},
        candidateTopK, workspace, workspaceSize, nullptr, firstCandidateLogitsRow};
    kernel::ddtreeBuild(buildParams);
    CUDA_CHECK(cudaDeviceSynchronize());

    return DDTreeBuildOutputs{copyDeviceToHost<int32_t>(nodeTokenIdsTensor),
        copyDeviceToHost<int32_t>(nodeDepthsTensor), copyDeviceToHost<int32_t>(parentIdsTensor),
        copyDeviceToHost<float>(nodeScoresTensor), copyDeviceToHost<int32_t>(validCountsTensor),
        copyDeviceToHost<int32_t>(verifyTokenIdsTensor), copyDeviceToHost<int32_t>(verifyPositionIdsTensor),
        copyDeviceToHost<int32_t>(packedAncestorMaskTensor), copyDeviceToHost<int32_t>(contextLengthsTensor),
        copyDeviceToHost<int64_t>(selectTokenIndicesTensor)};
}

void validateBuildAgainstReference(
    DDTreeBuildOutputs const& actual, DDTreeReference const& expected, int32_t batchSize, int32_t verifyTokenCount)
{
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        EXPECT_EQ(actual.validCounts[batchIdx], expected.validCounts[batchIdx]);
        for (int32_t nodeIdx = 0; nodeIdx < verifyTokenCount; ++nodeIdx)
        {
            int32_t const idx = batchIdx * verifyTokenCount + nodeIdx;
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
                    EXPECT_EQ(actual.nodeDepths[idx],
                        actual.nodeDepths[batchIdx * verifyTokenCount + actual.parentIds[idx]] + 1);
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
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        EXPECT_EQ(actual.contextLengths[batchIdx],
            expected.verifyPositionIds[batchIdx * verifyTokenCount] + verifyTokenCount);
        for (int32_t nodeIdx = 0; nodeIdx < verifyTokenCount; ++nodeIdx)
        {
            EXPECT_EQ(actual.selectTokenIndices[batchIdx * verifyTokenCount + nodeIdx],
                batchIdx * verifyTokenCount + nodeIdx);
        }
    }
}

void setLogit(std::vector<float>& logits, int32_t batchIdx, int32_t depthIdx, int32_t tokenId, int32_t draftBlockSize,
    int32_t vocabSize, float value)
{
    logits[static_cast<size_t>(batchIdx) * draftBlockSize * vocabSize + depthIdx * vocabSize + tokenId] = value;
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

TEST(DDTreeKernels, BuildsFromTokenMajorDraftLogits)
{
    constexpr int32_t kBatchSize{2};
    constexpr int32_t kDFlashBlockSize{4};
    constexpr int32_t kVocabSize{12};
    constexpr int32_t kVerifySize{7};
    constexpr int32_t kCandidateTopK{2};
    std::vector<float> logits(static_cast<size_t>(kBatchSize) * kDFlashBlockSize * kVocabSize, -12.0F);
    for (int32_t batchIdx = 0; batchIdx < kBatchSize; ++batchIdx)
    {
        for (int32_t depthIdx = 1; depthIdx < kDFlashBlockSize; ++depthIdx)
        {
            setLogit(logits, batchIdx, depthIdx, batchIdx * 4 + depthIdx, kDFlashBlockSize, kVocabSize, 6.0F);
            setLogit(logits, batchIdx, depthIdx, batchIdx * 4 + depthIdx + 1, kDFlashBlockSize, kVocabSize, 5.0F);
        }
    }

    std::vector<int32_t> const rootTokenIds{90, 91};
    std::vector<int32_t> const baseLengths{20, 24};
    DDTreeBuildOutputs actual = runDDTreeBuild(logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize,
        kVocabSize, kVerifySize, kCandidateTopK, /*firstCandidateLogitsRow=*/1, /*depthConfidence=*/{},
        /*survivalThreshold=*/0.0F, /*tokenMajorLogits=*/true);
    DDTreeReference expected = buildReference(
        logits, rootTokenIds, baseLengths, kBatchSize, kDFlashBlockSize, kVocabSize, kVerifySize, kCandidateTopK);

    validateBuildAgainstReference(actual, expected, kBatchSize, kVerifySize);
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
// [batch, draftSteps+1, vocab] buffer and feeds it to ddtreeBuild with
// "blockSize" = draftSteps+1. Row 0 is a root placeholder that the builder
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

    // verifyTokenCount is within the tree capacity, so every requested node is filled
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
    // Proposal depth is capped at draftSteps (= "blockSize" - 1).
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
    // DSpark tree decouples verifyTokenCount from the chain length: block7 stacked logits
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

namespace
{

//! DFlash2 selector inputs. Codebook and projected entries are in {-1, 0, 1}, so every selector score is exact in FP32
//! for any reduction order; random unary values keep the scores distinct.
struct SelectorInputs
{
    int32_t batch;
    int32_t rows;
    int32_t support{16};
    int32_t rank{64};
    int32_t vocab{256};
    std::vector<int32_t> supportIds; //!< [batch, rows, support], distinct within a row.
    std::vector<float> unary;        //!< [batch, rows, support]
    std::vector<half> projected;     //!< [batch, rows, rank]
    std::vector<half> predecessor;   //!< [vocab, rank]
    std::vector<half> successor;     //!< [vocab, rank]
    std::vector<int32_t> anchors;    //!< [batch] root tokens.

    //! s = unary + <P[predecessorToken] * z, Q[candidate]> for support entry `k` of proposal row `row`.
    float score(int32_t row, int32_t predecessorToken, int32_t k) const
    {
        int32_t const candidate = supportIds[row * support + k];
        float dot{0.0F};
        for (int32_t r = 0; r < rank; ++r)
        {
            dot += __half2float(predecessor[predecessorToken * rank + r]) * __half2float(projected[row * rank + r])
                * __half2float(successor[candidate * rank + r]);
        }
        return unary[row * support + k] + dot;
    }
};

SelectorInputs randomSelectorInputs(int32_t batch, int32_t rows, uint32_t seed)
{
    SelectorInputs in{batch, rows};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unary(-3.0F, 3.0F);
    std::uniform_int_distribution<int32_t> entry(-1, 1);
    auto const entries = [&](size_t count) {
        std::vector<half> values(count);
        for (half& v : values)
        {
            v = __float2half(static_cast<float>(entry(rng)));
        }
        return values;
    };
    std::vector<int32_t> vocabIds(in.vocab);
    std::iota(vocabIds.begin(), vocabIds.end(), 0);
    for (int32_t row = 0; row < batch * rows; ++row)
    {
        std::shuffle(vocabIds.begin(), vocabIds.end(), rng);
        for (int32_t k = 0; k < in.support; ++k)
        {
            in.supportIds.push_back(vocabIds[k]);
            in.unary.push_back(unary(rng));
        }
    }
    in.projected = entries(static_cast<size_t>(batch) * rows * in.rank);
    in.predecessor = entries(static_cast<size_t>(in.vocab) * in.rank);
    in.successor = entries(static_cast<size_t>(in.vocab) * in.rank);
    in.anchors.assign(vocabIds.begin() + in.support, vocabIds.begin() + in.support + batch);
    return in;
}

//! The topK children of a node with token `parentToken` at depth `row`: proposal row `row`'s listed support
//! (in-vocabulary ids at their first occurrence) scored with `parentToken` as predecessor, log-softmax over the listed
//! support, best first (ties to the smaller token).
std::vector<std::pair<int32_t, float>> referenceChildren(
    SelectorInputs const& in, int32_t b, int32_t row, int32_t parentToken, int32_t topK)
{
    int32_t const rowIdx = b * in.rows + row;
    int32_t const* ids = &in.supportIds[rowIdx * in.support];
    std::vector<int32_t> listed;
    for (int32_t k = 0; k < in.support; ++k)
    {
        if (ids[k] >= 0 && ids[k] < in.vocab && std::find(ids, ids + k, ids[k]) == ids + k)
        {
            listed.push_back(k);
        }
    }
    std::vector<float> scores(in.support);
    double maxScore = -std::numeric_limits<double>::infinity();
    for (int32_t k : listed)
    {
        scores[k] = in.score(rowIdx, parentToken, k);
        maxScore = std::max<double>(maxScore, scores[k]);
    }
    double sum = 0.0;
    for (int32_t k : listed)
    {
        sum += std::exp(scores[k] - maxScore);
    }
    std::sort(listed.begin(), listed.end(), [&](int32_t lhs, int32_t rhs) {
        return scores[lhs] != scores[rhs] ? scores[lhs] > scores[rhs] : ids[lhs] < ids[rhs];
    });
    std::vector<std::pair<int32_t, float>> children;
    for (int32_t slot = 0; slot < topK && slot < static_cast<int32_t>(listed.size()); ++slot)
    {
        children.emplace_back(ids[listed[slot]], static_cast<float>(scores[listed[slot]] - maxScore - std::log(sum)));
    }
    return children;
}

//! Best-first tree computed from the DFlash2 definition rather than from a candidate table: each step adds the
//! unused child with the highest prefix log-probability, where a node's children are referenceChildren() of its own
//! token at its depth. Fills tokens, depths, parents, scores and valid counts.
DDTreeReference parentConditionedReference(SelectorInputs const& in, int32_t topK, int32_t verifySize)
{
    size_t const nodes = static_cast<size_t>(in.batch) * verifySize;
    DDTreeReference ref;
    ref.nodeTokenIds.assign(nodes, 0);
    ref.nodeDepths.assign(nodes, 0);
    ref.parentIds.assign(nodes, -1);
    ref.nodeScores.assign(nodes, -std::numeric_limits<float>::infinity());
    ref.validCounts.assign(in.batch, 0);
    for (int32_t b = 0; b < in.batch; ++b)
    {
        int32_t const base = b * verifySize;
        std::vector<std::vector<std::pair<int32_t, float>>> children{referenceChildren(in, b, 0, in.anchors[b], topK)};
        std::vector<int32_t> nextSlot(verifySize, 0);
        ref.nodeTokenIds[base] = in.anchors[b];
        ref.nodeScores[base] = 0.0F;
        int32_t count{1};
        for (; count < verifySize; ++count)
        {
            int32_t best{-1};
            float bestScore{0.0F};
            for (int32_t parent = 0; parent < count; ++parent)
            {
                if (nextSlot[parent] < static_cast<int32_t>(children[parent].size()))
                {
                    float const score = ref.nodeScores[base + parent] + children[parent][nextSlot[parent]].second;
                    if (best < 0 || score > bestScore)
                    {
                        best = parent;
                        bestScore = score;
                    }
                }
            }
            if (best < 0)
            {
                break;
            }
            int32_t const token = children[best][nextSlot[best]++].first;
            int32_t const depth = ref.nodeDepths[base + best] + 1;
            ref.nodeTokenIds[base + count] = token;
            ref.nodeDepths[base + count] = depth;
            ref.parentIds[base + count] = best;
            ref.nodeScores[base + count] = bestScore;
            children.push_back(depth < in.rows ? referenceChildren(in, b, depth, token, topK)
                                               : std::vector<std::pair<int32_t, float>>{});
        }
        ref.validCounts[b] = count;
    }
    return ref;
}

//! Device copies of SelectorInputs.
struct DeviceSelectorInputs
{
    rt::Tensor supportIds;
    rt::Tensor unary;
    rt::Tensor projected;
    rt::Tensor anchors;
    rt::Tensor predecessor;
    rt::Tensor successor;

    explicit DeviceSelectorInputs(SelectorInputs const& in)
        : supportIds({in.batch, in.rows, in.support}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32)
        , unary({in.batch, in.rows, in.support}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT)
        , projected({in.batch, in.rows, in.rank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF)
        , anchors({in.batch}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32)
        , predecessor({in.vocab, in.rank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF)
        , successor({in.vocab, in.rank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF)
    {
        copyHostToDevice(supportIds, in.supportIds);
        copyHostToDevice(unary, in.unary);
        copyHostToDevice<half>(projected, in.projected);
        copyHostToDevice(anchors, in.anchors);
        copyHostToDevice<half>(predecessor, in.predecessor);
        copyHostToDevice<half>(successor, in.successor);
    }
};

//! Runs dflash2SelectorTreeCandidates() with `topK` children per list and ddtreeAssemble() rooted at the anchors.
DDTreeBuildOutputs runDFlash2Tree(
    SelectorInputs const& in, int32_t topK, std::vector<int32_t> const& baseLengths, int32_t verifySize)
{
    using trt_edgellm::DataType;
    auto const gpu = rt::DeviceType::kGPU;
    DeviceSelectorInputs const dev(in);
    rt::Coords const tableShape{in.batch, in.rows, in.support, topK};
    rt::Tensor tokenIds(tableShape, gpu, DataType::kINT32);
    rt::Tensor logProbs(tableShape, gpu, DataType::kFLOAT);
    rt::Tensor childStates(tableShape, gpu, DataType::kINT32);
    kernel::dflash2SelectorTreeCandidates(dev.supportIds, dev.unary, dev.projected, dev.anchors, dev.predecessor,
        dev.successor, tokenIds, logProbs, childStates, nullptr);

    int32_t const packedMaskLen = (verifySize + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    auto const nodeTensor = [&](DataType type) { return rt::Tensor({in.batch, verifySize}, gpu, type); };
    rt::Tensor baseLengthsTensor({in.batch}, gpu, DataType::kINT32);
    copyHostToDevice(baseLengthsTensor, baseLengths);
    rt::Tensor nodeTokenIds = nodeTensor(DataType::kINT32);
    rt::Tensor nodeDepths = nodeTensor(DataType::kINT32);
    rt::Tensor parentIds = nodeTensor(DataType::kINT32);
    rt::Tensor nodeScores = nodeTensor(DataType::kFLOAT);
    rt::Tensor validCounts({in.batch}, gpu, DataType::kINT32);
    rt::Tensor verifyTokenIds = nodeTensor(DataType::kINT32);
    rt::Tensor verifyPositionIds = nodeTensor(DataType::kINT32);
    rt::Tensor packedAncestorMask({in.batch, verifySize, packedMaskLen}, gpu, DataType::kINT32);
    rt::Tensor contextLengths({in.batch}, gpu, DataType::kINT32);
    rt::Tensor selectTokenIndices = nodeTensor(DataType::kINT64);
    kernel::ddtreeAssemble({{tokenIds, logProbs, &childStates}, dev.anchors, baseLengthsTensor, nullptr, 0.0F,
        {nodeTokenIds, nodeDepths, parentIds, nodeScores, validCounts, verifyTokenIds, verifyPositionIds,
            packedAncestorMask, contextLengths, selectTokenIndices},
        nullptr});
    CUDA_CHECK(cudaDeviceSynchronize());
    return DDTreeBuildOutputs{copyDeviceToHost<int32_t>(nodeTokenIds), copyDeviceToHost<int32_t>(nodeDepths),
        copyDeviceToHost<int32_t>(parentIds), copyDeviceToHost<float>(nodeScores),
        copyDeviceToHost<int32_t>(validCounts), copyDeviceToHost<int32_t>(verifyTokenIds),
        copyDeviceToHost<int32_t>(verifyPositionIds), copyDeviceToHost<int32_t>(packedAncestorMask),
        copyDeviceToHost<int32_t>(contextLengths), copyDeviceToHost<int64_t>(selectTokenIndices)};
}

} // namespace

// Given random DFlash2 selector inputs for two requests with four proposal rows, where one row also offers an
// out-of-vocabulary id, a negative id and a repeated id with the highest unary values, and another row has two support
// tokens whose scores tie exactly,
// When dflash2SelectorTreeCandidates lists 3 children per (row, parent state) and ddtreeAssemble grows 12 nodes,
// Then the tree equals a best-first reference in which every node scores its children with its own token as the
// selector predecessor over the row's in-vocabulary, first-occurring support ids, ties going to the smaller token: the
// same tokens, parents, depths and prefix log-probabilities.
TEST(DDTreeKernels, DFlash2TreeMatchesParentConditionedReference)
{
    constexpr int32_t kTopK{3};
    constexpr int32_t kVerifySize{12};
    SelectorInputs in = randomSelectorInputs(/*batch=*/2, /*rows=*/4, /*seed=*/1290);
    int32_t const edgeRow = (0 * in.rows + 1) * in.support;
    in.supportIds[edgeRow + 3] = in.vocab;
    in.supportIds[edgeRow + 5] = -1;
    in.supportIds[edgeRow + 7] = in.supportIds[edgeRow + 2];
    for (int32_t const k : {3, 5, 7})
    {
        in.unary[edgeRow + k] = 30.0F;
    }
    int32_t const tieRow = (1 * in.rows + 2) * in.support;
    in.unary[tieRow + 4] = 20.0F;
    in.unary[tieRow + 9] = 20.0F;
    std::copy_n(in.successor.begin() + static_cast<std::ptrdiff_t>(in.supportIds[tieRow + 9]) * in.rank, in.rank,
        in.successor.begin() + static_cast<std::ptrdiff_t>(in.supportIds[tieRow + 4]) * in.rank);

    DDTreeBuildOutputs const actual = runDFlash2Tree(in, kTopK, {10, 300}, kVerifySize);
    DDTreeReference const expected = parentConditionedReference(in, kTopK, kVerifySize);

    EXPECT_EQ(actual.validCounts, expected.validCounts);
    EXPECT_EQ(actual.nodeTokenIds, expected.nodeTokenIds);
    EXPECT_EQ(actual.parentIds, expected.parentIds);
    EXPECT_EQ(actual.nodeDepths, expected.nodeDepths);
    for (size_t node = 0; node < expected.nodeScores.size(); ++node)
    {
        EXPECT_NEAR(actual.nodeScores[node], expected.nodeScores[node], 1.0e-5F) << "node " << node;
    }
}

// Given DFlash2 selector outputs for three requests with proposal lengths 7 and 15,
// When the tree path keeps one child per parent (K = 1) and has room for the whole block,
// Then the assembled tree is a single chain whose tokens equal the greedy chain selector's proposal, so the
// tree path scores candidates exactly as the chain selector does.
TEST(DDTreeKernels, DFlash2TopOneTreeEqualsGreedyChainSelector)
{
    using trt_edgellm::DataType;
    constexpr int32_t kBatchSize{3};
    for (int32_t proposalRows : {7, 15})
    {
        SelectorInputs const in = randomSelectorInputs(kBatchSize, proposalRows, /*seed=*/7 + proposalRows);
        DeviceSelectorInputs const dev(in);
        rt::Tensor uniforms({kBatchSize, proposalRows}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor temperatures({kBatchSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor greedy({kBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor chainTokens({kBatchSize, proposalRows}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor chainProbabilities({kBatchSize, proposalRows, in.support}, rt::DeviceType::kGPU, DataType::kFLOAT);
        copyHostToDevice(uniforms, std::vector<float>(static_cast<size_t>(kBatchSize) * proposalRows, 0.5F));
        copyHostToDevice(temperatures, std::vector<float>(kBatchSize, 1.0F));
        copyHostToDevice(greedy, std::vector<int32_t>(kBatchSize, 1));
        kernel::launchDFlash2CandidateSelector(dev.supportIds, dev.unary, dev.projected, dev.anchors, dev.predecessor,
            dev.successor, uniforms, temperatures, greedy, chainTokens, chainProbabilities, nullptr);
        std::vector<int32_t> const chain = copyDeviceToHost<int32_t>(chainTokens);

        int32_t const verifySize = proposalRows + 4;
        DDTreeBuildOutputs const tree = runDFlash2Tree(in, /*topK=*/1, {50, 60, 70}, verifySize);

        for (int32_t b = 0; b < kBatchSize; ++b)
        {
            EXPECT_EQ(tree.validCounts[b], proposalRows + 1) << "P=" << proposalRows << " batch " << b;
            for (int32_t step = 0; step < proposalRows; ++step)
            {
                int32_t const node = b * verifySize + step + 1;
                EXPECT_EQ(tree.nodeTokenIds[node], chain[b * proposalRows + step])
                    << "P=" << proposalRows << " batch " << b << " step " << step;
                EXPECT_EQ(tree.parentIds[node], step);
            }
        }
    }
}
