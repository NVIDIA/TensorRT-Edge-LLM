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

#include "kernels/speculative/ddtreeKernels.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "speculativeKernelsUtils.h"

#include "common/dataType.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kCandidateBlockSize{256};
constexpr int32_t kBuildBlockSize{128};
constexpr int32_t kMaskBitsPerWord{32};
constexpr int32_t kRootParent{-1};
constexpr int32_t kPaddingParent{-1};

//! Bytes of one aligned [batch, depthRows, 1, candidateTopK] candidate list (INT32 and FP32 are both 4 bytes).
size_t getCandidateListWorkspaceBytes(int32_t batchSize, int32_t depthRows, int32_t candidateTopK)
{
    return alignSpeculativeWorkspaceSize(static_cast<size_t>(batchSize) * depthRows * candidateTopK * sizeof(int32_t));
}

__device__ void insertCandidate(float score, int32_t tokenId, float (&topScores)[kDDTreeMaxCandidateTopK],
    int32_t (&topTokenIds)[kDDTreeMaxCandidateTopK], int32_t candidateTopK)
{
    // Early exit for the overwhelmingly common case: the list is sorted
    // descending, so anything not better than the last retained slot cannot
    // be inserted anywhere. One comparison instead of candidateTopK.
    if (!isBetterCandidate(score, tokenId, topScores[candidateTopK - 1], topTokenIds[candidateTopK - 1]))
    {
        return;
    }
    for (int32_t slot = 0; slot < candidateTopK; ++slot)
    {
        if (isBetterCandidate(score, tokenId, topScores[slot], topTokenIds[slot]))
        {
            for (int32_t move = candidateTopK - 1; move > slot; --move)
            {
                topScores[move] = topScores[move - 1];
                topTokenIds[move] = topTokenIds[move - 1];
            }
            topScores[slot] = score;
            topTokenIds[slot] = tokenId;
            break;
        }
    }
}

//! Merge one online-logsumexp partial (om, os) into (m, s):
//! exp(m)*s + exp(om)*os == exp(m')*s' after the merge.
__device__ __forceinline__ void mergeLogSumExp(float& m, float& s, float om, float os)
{
    if (om == -INFINITY)
    {
        return; // empty partial; also avoids NaN from (-inf) - (-inf)
    }
    if (om > m)
    {
        s = s * expf(m - om) + os;
        m = om;
    }
    else
    {
        s += os * expf(om - m);
    }
}

// One CTA per (batch, proposal row): top-K token candidates + logsumexp over the
// draft vocab in a single pass (online logsumexp, float4 loads, guarded top-K
// insertion, warp-shuffle reduction, parallel tree merge of per-thread lists).
__global__ void selectDDTreeCandidatesKernel(float const* __restrict__ draftLogits,
    int32_t const* __restrict__ draftVocabMappingTable, int32_t* __restrict__ candidateTokenIds,
    float* __restrict__ candidateLogProbs, int32_t draftBlockSize, int32_t vocabSize, int32_t candidateTopK,
    int32_t firstCandidateLogitsRow)
{
    int32_t const depthRows = draftBlockSize - 1;
    int32_t const batchIdx = blockIdx.x / depthRows;
    int32_t const rowIdx = blockIdx.x % depthRows;
    int32_t const depthIdx = firstCandidateLogitsRow + rowIdx;
    int32_t const tid = threadIdx.x;

    int32_t const candidateOffset = (batchIdx * depthRows + rowIdx) * candidateTopK;

    __shared__ float sTopScores[kCandidateBlockSize * kDDTreeMaxCandidateTopK];
    __shared__ int32_t sTopTokenIds[kCandidateBlockSize * kDDTreeMaxCandidateTopK];
    __shared__ float sWarpMax[kCandidateBlockSize / 32];
    __shared__ float sWarpSum[kCandidateBlockSize / 32];
    __shared__ float sLogSumExp;

    float topScores[kDDTreeMaxCandidateTopK];
    int32_t topTokenIds[kDDTreeMaxCandidateTopK];
    for (int32_t slot = 0; slot < kDDTreeMaxCandidateTopK; ++slot)
    {
        topScores[slot] = -INFINITY;
        topTokenIds[slot] = 0;
    }

    int64_t const logitsOffset
        = static_cast<int64_t>(batchIdx) * draftBlockSize * vocabSize + static_cast<int64_t>(depthIdx) * vocabSize;
    float const* logits = draftLogits + logitsOffset;

    // ---- Single fused pass: online logsumexp + guarded top-K, float4 loads --
    // float4 path requires the per-row base to stay 16B-aligned; fall back to
    // the scalar tail loop for the whole row otherwise.
    float runMax = -INFINITY;
    float runSum = 0.0F;
    int32_t const vocab4 = (vocabSize % 4 == 0) ? vocabSize / 4 : 0;
    float4 const* logits4 = reinterpret_cast<float4 const*>(logits);
    for (int32_t i4 = tid; i4 < vocab4; i4 += blockDim.x)
    {
        float4 const v4 = logits4[i4];
        float const vals[4] = {v4.x, v4.y, v4.z, v4.w};
        int32_t const base = i4 * 4;
#pragma unroll
        for (int32_t j = 0; j < 4; ++j)
        {
            float const score = vals[j];
            if (score > runMax)
            {
                runSum = runSum * expf(runMax - score) + 1.0F;
                runMax = score;
            }
            else if (runMax > -INFINITY)
            {
                runSum += expf(score - runMax);
            }
            insertCandidate(score, base + j, topScores, topTokenIds, candidateTopK);
        }
    }
    for (int32_t vocabIdx = vocab4 * 4 + tid; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        float const score = logits[vocabIdx];
        if (score > runMax)
        {
            runSum = runSum * expf(runMax - score) + 1.0F;
            runMax = score;
        }
        else if (runMax > -INFINITY)
        {
            runSum += expf(score - runMax);
        }
        insertCandidate(score, vocabIdx, topScores, topTokenIds, candidateTopK);
    }

    // ---- logsumexp: warp shuffle reduce, then one merge over warps ---------
    int32_t const lane = tid & 31;
    int32_t const warp = tid >> 5;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        float const om = __shfl_down_sync(0xffffffffU, runMax, off);
        float const os = __shfl_down_sync(0xffffffffU, runSum, off);
        mergeLogSumExp(runMax, runSum, om, os);
    }
    if (lane == 0)
    {
        sWarpMax[warp] = runMax;
        sWarpSum[warp] = runSum;
    }

    // Stage per-thread top-K lists for the parallel merge.
    for (int32_t slot = 0; slot < candidateTopK; ++slot)
    {
        sTopScores[tid * kDDTreeMaxCandidateTopK + slot] = topScores[slot];
        sTopTokenIds[tid * kDDTreeMaxCandidateTopK + slot] = topTokenIds[slot];
    }
    __syncthreads();

    if (tid == 0)
    {
        float m = sWarpMax[0];
        float s = sWarpSum[0];
        for (int32_t w = 1; w < static_cast<int32_t>(blockDim.x) / 32; ++w)
        {
            mergeLogSumExp(m, s, sWarpMax[w], sWarpSum[w]);
        }
        sLogSumExp = m + logf(s);
    }

    // ---- Parallel tree merge of the 256 top-K lists (8 rounds) -------------
    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride >= 1; stride >>= 1)
    {
        if (tid < stride)
        {
            float mergedScores[kDDTreeMaxCandidateTopK];
            int32_t mergedIds[kDDTreeMaxCandidateTopK];
            for (int32_t slot = 0; slot < kDDTreeMaxCandidateTopK; ++slot)
            {
                mergedScores[slot] = sTopScores[tid * kDDTreeMaxCandidateTopK + slot];
                mergedIds[slot] = sTopTokenIds[tid * kDDTreeMaxCandidateTopK + slot];
            }
            int32_t const other = tid + stride;
            for (int32_t slot = 0; slot < candidateTopK; ++slot)
            {
                insertCandidate(sTopScores[other * kDDTreeMaxCandidateTopK + slot],
                    sTopTokenIds[other * kDDTreeMaxCandidateTopK + slot], mergedScores, mergedIds, candidateTopK);
            }
            for (int32_t slot = 0; slot < candidateTopK; ++slot)
            {
                sTopScores[tid * kDDTreeMaxCandidateTopK + slot] = mergedScores[slot];
                sTopTokenIds[tid * kDDTreeMaxCandidateTopK + slot] = mergedIds[slot];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        for (int32_t slot = 0; slot < candidateTopK; ++slot)
        {
            int32_t const tokenId = sTopTokenIds[slot];
            candidateTokenIds[candidateOffset + slot]
                = draftVocabMappingTable == nullptr ? tokenId : draftVocabMappingTable[tokenId];
            // A fully masked row (all logits -inf) yields score == lse == -inf;
            // keep the slot at -inf instead of the NaN that (-inf) - (-inf)
            // would produce (NaN would poison downstream score comparisons).
            float const score = sTopScores[slot];
            candidateLogProbs[candidateOffset + slot] = (score == -INFINITY) ? -INFINITY : score - sLogSumExp;
        }
    }
}

__device__ __forceinline__ bool isFiniteScore(float score)
{
    return score > -FLT_MAX * 0.5F;
}

__device__ __forceinline__ bool isBetterTreeExpansion(float lhsScore, int32_t lhsParent, int32_t lhsSlot,
    int32_t lhsToken, float rhsScore, int32_t rhsParent, int32_t rhsSlot, int32_t rhsToken)
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

__device__ __forceinline__ int32_t candidateListOffset(
    int32_t batchIdx, int32_t row, int32_t state, int32_t depthRows, int32_t stateCount, int32_t candidateTopK)
{
    return ((batchIdx * depthRows + row) * stateCount + state) * candidateTopK;
}

// One CTA per batch. Dynamic shared memory holds nextCandidateSlot[verifyTokenCount] then
// nodeState[verifyTokenCount].
__global__ void assembleDDTreeKernel(int32_t const* __restrict__ rootTokenIds, int32_t const* __restrict__ baseLengths,
    int32_t const* __restrict__ candidateTokenIds, float const* __restrict__ candidateLogProbs,
    int32_t const* __restrict__ candidateChildStates, float const* __restrict__ depthConfidence,
    float confSurvivalFloorLog, int32_t* __restrict__ nodeTokenIds, int32_t* __restrict__ nodeDepths,
    int32_t* __restrict__ parentIds, float* __restrict__ nodeScores, int32_t* __restrict__ validCounts,
    int32_t* __restrict__ verifyTokenIds, int32_t* __restrict__ verifyPositionIds,
    int32_t* __restrict__ packedAncestorMask, int32_t* __restrict__ contextLengths,
    int64_t* __restrict__ selectTokenIndices, int32_t depthRows, int32_t stateCount, int32_t verifyTokenCount,
    int32_t candidateTopK)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const packedMaskLen = (verifyTokenCount + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    int32_t const treeOffset = batchIdx * verifyTokenCount;
    int32_t const packedMaskOffset = batchIdx * verifyTokenCount * packedMaskLen;

    extern __shared__ int32_t sDynamic[];
    int32_t* nextCandidateSlot = sDynamic;
    int32_t* nodeState = sDynamic + verifyTokenCount;
    __shared__ int32_t sValidCount;
    __shared__ int32_t sStopBuild;
    __shared__ float sBestScores[kBuildBlockSize];
    __shared__ int32_t sBestParents[kBuildBlockSize];
    __shared__ int32_t sBestSlots[kBuildBlockSize];
    __shared__ int32_t sBestTokens[kBuildBlockSize];

    // Per-depth confidence means the threshold reduces to a per-batch max growth depth.
    int32_t confMaxDepth = depthRows;
    if (depthConfidence != nullptr && confSurvivalFloorLog > -INFINITY)
    {
        float survivalLog = 0.0F;
        confMaxDepth = 0;
        for (int32_t depth = 1; depth <= depthRows; ++depth)
        {
            survivalLog += logf(fmaxf(depthConfidence[batchIdx * depthRows + depth - 1], 1e-20F));
            if (survivalLog < confSurvivalFloorLog)
            {
                break;
            }
            confMaxDepth = depth;
        }
        // Parity with the chain scheduler's minProposalLen >= 1.
        confMaxDepth = max(confMaxDepth, 1);
    }

    for (int32_t nodeIdx = tid; nodeIdx < verifyTokenCount; nodeIdx += blockDim.x)
    {
        nodeTokenIds[treeOffset + nodeIdx] = 0;
        nodeDepths[treeOffset + nodeIdx] = 0;
        parentIds[treeOffset + nodeIdx] = kPaddingParent;
        nodeScores[treeOffset + nodeIdx] = -INFINITY;
        verifyTokenIds[treeOffset + nodeIdx] = 0;
        verifyPositionIds[treeOffset + nodeIdx] = 0;
        selectTokenIndices[treeOffset + nodeIdx] = treeOffset + nodeIdx;
        nextCandidateSlot[nodeIdx] = 0;
        nodeState[nodeIdx] = 0;
    }
    for (int32_t maskIdx = tid; maskIdx < verifyTokenCount * packedMaskLen; maskIdx += blockDim.x)
    {
        packedAncestorMask[packedMaskOffset + maskIdx] = 0;
    }

    if (tid == 0)
    {
        nodeTokenIds[treeOffset] = rootTokenIds[batchIdx];
        nodeDepths[treeOffset] = 0;
        parentIds[treeOffset] = kRootParent;
        nodeScores[treeOffset] = 0.0F;
        sValidCount = 1;
        sStopBuild = 0;
    }
    __syncthreads();

    int32_t const maxProposalDepth = min(depthRows, confMaxDepth);
    for (int32_t outNodeIdx = 1; outNodeIdx < verifyTokenCount; ++outNodeIdx)
    {
        if (sStopBuild != 0)
        {
            break;
        }

        int32_t bestParent{-1};
        int32_t bestSlot{-1};
        int32_t bestToken{0};
        float bestScore{-INFINITY};
        int32_t const validCount = sValidCount;

        for (int32_t parentIdx = tid; parentIdx < validCount; parentIdx += blockDim.x)
        {
            int32_t const parentDepth = nodeDepths[treeOffset + parentIdx];
            if (parentDepth >= maxProposalDepth)
            {
                continue;
            }

            int32_t const slot = nextCandidateSlot[parentIdx];
            if (slot >= candidateTopK)
            {
                continue;
            }

            int32_t const candidateOffset
                = candidateListOffset(batchIdx, parentDepth, nodeState[parentIdx], depthRows, stateCount, candidateTopK)
                + slot;
            float const candidateScore = candidateLogProbs[candidateOffset];
            if (!isFiniteScore(candidateScore))
            {
                continue;
            }

            int32_t const candidateToken = candidateTokenIds[candidateOffset];
            float prefixScore = nodeScores[treeOffset + parentIdx] + candidateScore;
            // log(conf) accumulates along the path via the parent score.
            if (depthConfidence != nullptr)
            {
                float const conf = depthConfidence[batchIdx * depthRows + parentDepth];
                prefixScore += logf(fmaxf(conf, 1e-20F));
            }
            if (bestParent < 0
                || isBetterTreeExpansion(
                    prefixScore, parentIdx, slot, candidateToken, bestScore, bestParent, bestSlot, bestToken))
            {
                bestParent = parentIdx;
                bestSlot = slot;
                bestToken = candidateToken;
                bestScore = prefixScore;
            }
        }

        sBestParents[tid] = bestParent;
        sBestSlots[tid] = bestSlot;
        sBestTokens[tid] = bestToken;
        sBestScores[tid] = bestScore;
        __syncthreads();

        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                int32_t const otherParent = sBestParents[tid + stride];
                if (otherParent >= 0
                    && (sBestParents[tid] < 0
                        || isBetterTreeExpansion(sBestScores[tid + stride], otherParent, sBestSlots[tid + stride],
                            sBestTokens[tid + stride], sBestScores[tid], sBestParents[tid], sBestSlots[tid],
                            sBestTokens[tid])))
                {
                    sBestParents[tid] = otherParent;
                    sBestSlots[tid] = sBestSlots[tid + stride];
                    sBestTokens[tid] = sBestTokens[tid + stride];
                    sBestScores[tid] = sBestScores[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            bestParent = sBestParents[0];
            if (bestParent < 0)
            {
                sStopBuild = 1;
            }
            else
            {
                bestSlot = sBestSlots[0];
                int32_t const parentDepth = nodeDepths[treeOffset + bestParent];
                int32_t const candidateOffset = candidateListOffset(batchIdx, parentDepth, nodeState[bestParent],
                                                    depthRows, stateCount, candidateTopK)
                    + bestSlot;
                nextCandidateSlot[bestParent] += 1;
                nodeState[outNodeIdx] = candidateChildStates == nullptr ? 0 : candidateChildStates[candidateOffset];
                nodeTokenIds[treeOffset + outNodeIdx] = sBestTokens[0];
                nodeDepths[treeOffset + outNodeIdx] = parentDepth + 1;
                parentIds[treeOffset + outNodeIdx] = bestParent;
                nodeScores[treeOffset + outNodeIdx] = sBestScores[0];
                sValidCount += 1;
            }
        }
        __syncthreads();
    }

    int32_t const validCount = sValidCount;
    int32_t const baseLength = baseLengths[batchIdx];
    if (tid == 0)
    {
        validCounts[batchIdx] = validCount;
        contextLengths[batchIdx] = baseLength + verifyTokenCount;
    }

    for (int32_t nodeIdx = tid; nodeIdx < validCount; nodeIdx += blockDim.x)
    {
        verifyTokenIds[treeOffset + nodeIdx] = nodeTokenIds[treeOffset + nodeIdx];
        verifyPositionIds[treeOffset + nodeIdx] = baseLength + nodeDepths[treeOffset + nodeIdx];

        int32_t ancestorIdx = nodeIdx;
        while (ancestorIdx >= 0)
        {
            int32_t const wordIdx = ancestorIdx / kMaskBitsPerWord;
            int32_t const bitIdx = ancestorIdx % kMaskBitsPerWord;
            packedAncestorMask[packedMaskOffset + nodeIdx * packedMaskLen + wordIdx]
                |= static_cast<int32_t>(1U << bitIdx);
            ancestorIdx = parentIds[treeOffset + ancestorIdx];
        }
    }
}

void validateGpuTensor(
    rt::Tensor const& tensor, char const* tensorName, trt_edgellm::DataType dataType, char const* dataTypeName)
{
    check::check(tensor.getDeviceType() == rt::DeviceType::kGPU, std::string(tensorName) + " must be on GPU.");
    check::check(tensor.getDataType() == dataType, std::string(tensorName) + " must be " + dataTypeName + ".");
}

//! Batch, block, and vocab extents of a token-major or rectangular draft-logits tensor.
struct DraftLogitsExtent
{
    int32_t batchSize;
    int32_t draftBlockSize;
    int32_t vocabSize;
};

DraftLogitsExtent draftLogitsExtent(rt::Tensor const& draftLogits, int32_t batchSize)
{
    auto const logitsShape = draftLogits.getShape();
    if (logitsShape.getNumDims() == 2)
    {
        check::check(logitsShape[0] % batchSize == 0,
            "Token-major draftLogits row count must be divisible by the active batch size.");
        return {batchSize, static_cast<int32_t>(logitsShape[0] / batchSize), static_cast<int32_t>(logitsShape[1])};
    }
    check::check(logitsShape.getNumDims() == 3 && logitsShape[0] == batchSize,
        "draftLogits must be token-major [batch * draftBlockSize, vocabSize] or an internal "
        "[batch, draftBlockSize, vocabSize] view.");
    return {batchSize, static_cast<int32_t>(logitsShape[1]), static_cast<int32_t>(logitsShape[2])};
}

} // anonymous namespace

size_t getDDTreeBuildWorkspaceSize(
    int32_t batchSize, int32_t draftBlockSize, int32_t verifyTokenCount, int32_t vocabSize, int32_t candidateTopK)
{
    if (batchSize <= 0 || draftBlockSize <= 1 || verifyTokenCount <= 0 || vocabSize <= 0 || candidateTopK <= 0
        || candidateTopK > kDDTreeMaxCandidateTopK || candidateTopK > vocabSize)
    {
        return 0;
    }
    return 2 * getCandidateListWorkspaceBytes(batchSize, draftBlockSize - 1, candidateTopK);
}

void ddtreeSelectLogitCandidates(DDTreeLogitCandidateParams const& params)
{
    rt::Tensor const& draftLogits = params.draftLogits;
    rt::Tensor& tokenIds = params.tokenIds;
    rt::Tensor& logProbs = params.logProbs;
    validateGpuTensor(draftLogits, "draftLogits", trt_edgellm::DataType::kFLOAT, "FLOAT");
    validateGpuTensor(tokenIds, "candidate tokenIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(logProbs, "candidate logProbs", trt_edgellm::DataType::kFLOAT, "FLOAT");
    auto const tableShape = tokenIds.getShape();
    check::check(tableShape.getNumDims() == 4 && tableShape[0] > 0 && tableShape[2] == 1,
        "Logit candidate table must be [batch, draftBlockSize - 1, 1, candidateTopK].");
    check::check(logProbs.getShape() == tableShape, "Candidate logProbs must match the tokenIds shape.");
    int32_t const batchSize = static_cast<int32_t>(tableShape[0]);
    int32_t const candidateTopK = static_cast<int32_t>(tableShape[3]);
    DraftLogitsExtent const extent = draftLogitsExtent(draftLogits, batchSize);
    check::check(extent.draftBlockSize > 1 && extent.vocabSize > 0, "Invalid DDTree logits shape.");
    check::check(tableShape[1] == extent.draftBlockSize - 1,
        "Logit candidate table rows must equal draftBlockSize - 1 proposal depths.");
    check::check(params.firstCandidateLogitsRow == 0 || params.firstCandidateLogitsRow == 1,
        "DDTree firstCandidateLogitsRow must be 0 or 1.");
    check::check(candidateTopK > 0 && candidateTopK <= kDDTreeMaxCandidateTopK,
        "DDTree candidateTopK must be in [1, " + std::to_string(kDDTreeMaxCandidateTopK) + "].");
    check::check(candidateTopK <= extent.vocabSize, "DDTree candidateTopK must not exceed vocabSize.");

    int32_t const* draftVocabMappingTablePtr{nullptr};
    if (params.draftVocabMappingTable != nullptr)
    {
        rt::Tensor const& map = *params.draftVocabMappingTable;
        validateGpuTensor(map, "draftVocabMappingTable", trt_edgellm::DataType::kINT32, "INT32");
        check::check(map.getShape().getNumDims() == 1, "draftVocabMappingTable must be 1D.");
        check::check(map.getShape()[0] == extent.vocabSize,
            "draftVocabMappingTable length must match draft logits vocabulary size.");
        draftVocabMappingTablePtr = map.dataPointer<int32_t>();
    }

    dim3 const candidateGrid(batchSize * (extent.draftBlockSize - 1));
    selectDDTreeCandidatesKernel<<<candidateGrid, kCandidateBlockSize, 0, params.stream>>>(
        draftLogits.dataPointer<float>(), draftVocabMappingTablePtr, tokenIds.dataPointer<int32_t>(),
        logProbs.dataPointer<float>(), extent.draftBlockSize, extent.vocabSize, candidateTopK,
        params.firstCandidateLogitsRow);
    CUDA_CHECK(cudaGetLastError());
}

void ddtreeAssemble(DDTreeAssembleParams const& params)
{
    DDTreeCandidateTable const& table = params.candidates;
    rt::Tensor const& rootTokenIds = params.rootTokenIds;
    rt::Tensor const& baseLengths = params.baseLengths;
    rt::Tensor const* depthConfidence = params.depthConfidence;
    rt::Tensor& nodeTokenIds = params.outputs.nodeTokenIds;
    rt::Tensor& nodeDepths = params.outputs.nodeDepths;
    rt::Tensor& parentIds = params.outputs.parentIds;
    rt::Tensor& nodeScores = params.outputs.nodeScores;
    rt::Tensor& validCounts = params.outputs.validCounts;
    rt::Tensor& verifyTokenIds = params.outputs.verifyTokenIds;
    rt::Tensor& verifyPositionIds = params.outputs.verifyPositionIds;
    rt::Tensor& packedAncestorMask = params.outputs.packedAncestorMask;
    rt::Tensor& contextLengths = params.outputs.contextLengths;
    rt::Tensor& selectTokenIndices = params.outputs.selectTokenIndices;

    validateGpuTensor(table.tokenIds, "candidate tokenIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(table.logProbs, "candidate logProbs", trt_edgellm::DataType::kFLOAT, "FLOAT");
    validateGpuTensor(rootTokenIds, "rootTokenIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(baseLengths, "baseLengths", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(nodeTokenIds, "nodeTokenIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(nodeDepths, "nodeDepths", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(parentIds, "parentIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(nodeScores, "nodeScores", trt_edgellm::DataType::kFLOAT, "FLOAT");
    validateGpuTensor(validCounts, "validCounts", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(verifyTokenIds, "verifyTokenIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(verifyPositionIds, "verifyPositionIds", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(packedAncestorMask, "packedAncestorMask", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(contextLengths, "contextLengths", trt_edgellm::DataType::kINT32, "INT32");
    validateGpuTensor(selectTokenIndices, "selectTokenIndices", trt_edgellm::DataType::kINT64, "INT64");

    auto const tableShape = table.tokenIds.getShape();
    auto const nodeShape = nodeTokenIds.getShape();
    auto const rootTokenShape = rootTokenIds.getShape();
    check::check(rootTokenShape.getNumDims() == 1 && rootTokenShape[0] > 0, "rootTokenIds must be [batch].");
    int32_t const batchSize = static_cast<int32_t>(rootTokenShape[0]);
    check::check(tableShape.getNumDims() == 4 && tableShape[0] == batchSize && tableShape[1] > 0 && tableShape[2] > 0
            && tableShape[3] > 0,
        "DDTree candidate table must be [batch, depthRows, stateCount, candidateTopK].");
    check::check(table.logProbs.getShape() == tableShape, "Candidate logProbs must match the tokenIds shape.");
    int32_t const depthRows = static_cast<int32_t>(tableShape[1]);
    int32_t const stateCount = static_cast<int32_t>(tableShape[2]);
    int32_t const candidateTopK = static_cast<int32_t>(tableShape[3]);
    int32_t const* childStatesPtr{nullptr};
    if (table.childStates != nullptr)
    {
        validateGpuTensor(*table.childStates, "candidate childStates", trt_edgellm::DataType::kINT32, "INT32");
        check::check(table.childStates->getShape() == tableShape, "Candidate childStates must match the table.");
        childStatesPtr = table.childStates->dataPointer<int32_t>();
    }

    check::check(
        nodeShape.getNumDims() == 2 && nodeShape[0] == batchSize, "nodeTokenIds must be [batch, verifyTokenCount].");
    int32_t const verifyTokenCount = static_cast<int32_t>(nodeShape[1]);
    int32_t const packedMaskLen = (verifyTokenCount + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    check::check(verifyTokenCount > 0 && verifyTokenCount <= kDDTreeMaxVerifySize,
        "DDTree supports verifyTokenCount <= " + std::to_string(kDDTreeMaxVerifySize) + ".");
    check::check(baseLengths.getShape().getNumDims() == 1 && baseLengths.getShape()[0] == batchSize,
        "baseLengths must be [batch].");
    check::check(nodeDepths.getShape() == nodeShape && parentIds.getShape() == nodeShape
            && nodeScores.getShape() == nodeShape && verifyTokenIds.getShape() == nodeShape
            && verifyPositionIds.getShape() == nodeShape,
        "DDTree node and verify tensors must share [batch, verifyTokenCount].");
    check::check(validCounts.getShape().getNumDims() == 1 && validCounts.getShape()[0] == batchSize,
        "validCounts must be [batch].");
    check::check(packedAncestorMask.getShape().getNumDims() == 3 && packedAncestorMask.getShape()[0] == batchSize
            && packedAncestorMask.getShape()[1] == verifyTokenCount
            && packedAncestorMask.getShape()[2] == packedMaskLen,
        "packedAncestorMask must be [batch, verifyTokenCount, ceil(verifyTokenCount / 32)].");
    check::check(contextLengths.getShape().getNumDims() == 1 && contextLengths.getShape()[0] == batchSize,
        "contextLengths must be [batch].");
    check::check(selectTokenIndices.getShape().getNumDims() == 2 && selectTokenIndices.getShape()[0] == batchSize
            && selectTokenIndices.getShape()[1] == verifyTokenCount,
        "selectTokenIndices must be [batch, verifyTokenCount].");

    float const* depthConfidencePtr{nullptr};
    if (depthConfidence != nullptr)
    {
        validateGpuTensor(*depthConfidence, "depthConfidence", trt_edgellm::DataType::kFLOAT, "FLOAT");
        check::check(depthConfidence->getShape().getNumDims() == 2 && depthConfidence->getShape()[0] == batchSize
                && depthConfidence->getShape()[1] == depthRows,
            "depthConfidence must be [batch, depthRows].");
        depthConfidencePtr = depthConfidence->dataPointer<float>();
        check::check(params.confidenceSurvivalThreshold >= 0.0F && params.confidenceSurvivalThreshold < 1.0F,
            "confidenceSurvivalThreshold must be in [0, 1).");
    }

    float const confSurvivalFloorLog
        = params.confidenceSurvivalThreshold > 0.0F ? logf(params.confidenceSurvivalThreshold) : -INFINITY;
    size_t const sharedBytes = 2 * static_cast<size_t>(verifyTokenCount) * sizeof(int32_t);
    assembleDDTreeKernel<<<batchSize, kBuildBlockSize, sharedBytes, params.stream>>>(
        rootTokenIds.dataPointer<int32_t>(), baseLengths.dataPointer<int32_t>(), table.tokenIds.dataPointer<int32_t>(),
        table.logProbs.dataPointer<float>(), childStatesPtr, depthConfidencePtr, confSurvivalFloorLog,
        nodeTokenIds.dataPointer<int32_t>(), nodeDepths.dataPointer<int32_t>(), parentIds.dataPointer<int32_t>(),
        nodeScores.dataPointer<float>(), validCounts.dataPointer<int32_t>(), verifyTokenIds.dataPointer<int32_t>(),
        verifyPositionIds.dataPointer<int32_t>(), packedAncestorMask.dataPointer<int32_t>(),
        contextLengths.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(), depthRows, stateCount,
        verifyTokenCount, candidateTopK);
    CUDA_CHECK(cudaGetLastError());
}

void ddtreeBuild(DDTreeBuildParams const& params)
{
    rt::Tensor const& rootTokenIds = params.inputs.rootTokenIds;
    check::check(
        rootTokenIds.getShape().getNumDims() == 1 && rootTokenIds.getShape()[0] > 0, "rootTokenIds must be [batch].");
    int32_t const batchSize = static_cast<int32_t>(rootTokenIds.getShape()[0]);
    DraftLogitsExtent const extent = draftLogitsExtent(params.inputs.draftLogits, batchSize);
    check::check(extent.draftBlockSize > 1 && extent.vocabSize > 0, "Invalid DDTree logits shape.");
    int32_t const depthRows = extent.draftBlockSize - 1;
    size_t const listBytes = getCandidateListWorkspaceBytes(batchSize, depthRows, params.candidateTopK);
    ELLM_CHECK(params.workspaceSize >= 2 * listBytes,
        "DDTree build workspace too small. Required: " + std::to_string(2 * listBytes)
            + ", provided: " + std::to_string(params.workspaceSize));
    ELLM_CHECK(reinterpret_cast<uintptr_t>(params.workspace) % kSpeculativeWorkspaceAlignment == 0,
        "DDTree workspace must be 256-byte aligned.");

    rt::Coords const tableShape{batchSize, depthRows, 1, params.candidateTopK};
    char* const workspace = static_cast<char*>(params.workspace);
    rt::Tensor candidateTokenIds(workspace, tableShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor candidateLogProbs(
        workspace + listBytes, tableShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);

    ddtreeSelectLogitCandidates({params.inputs.draftLogits, params.inputs.draftVocabMappingTable,
        params.firstCandidateLogitsRow, candidateTokenIds, candidateLogProbs, params.stream});
    ddtreeAssemble({{candidateTokenIds, candidateLogProbs, nullptr}, rootTokenIds, params.inputs.baseLengths,
        params.inputs.depthConfidence, params.inputs.confidenceSurvivalThreshold, params.outputs, params.stream});
}

} // namespace kernel
} // namespace trt_edgellm
