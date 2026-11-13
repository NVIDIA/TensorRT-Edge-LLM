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
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "common/specDecodeLimits.h"
#include "common/stringUtils.h"
#include "kernels/common/vectorizedTypes.cuh"
#include "kernels/speculative/speculativeKernels.h"
#include "speculativeKernelsUtils.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trt_edgellm::kernel
{
namespace
{
__global__ void prepareChainMetadata(int32_t const* committedLengths, int32_t const* validTokenCounts,
    int32_t* packedAttentionMask, int32_t* positions, int64_t* selectTokenIndices, int32_t* contextLengths,
    int32_t* parentIds, int32_t* depths, int32_t width)
{
    int32_t const request = blockIdx.x;
    int32_t const node = threadIdx.x;
    int32_t const row = request * width + node;
    int32_t const validCount = validTokenCounts == nullptr ? width : validTokenCounts[request];
    bool const valid = node < validCount;
    positions[row] = valid ? committedLengths[request] + node : -1;
    selectTokenIndices[row] = row;
    if (parentIds != nullptr)
    {
        parentIds[row] = valid && node > 0 ? node - 1 : -1;
    }
    if (depths != nullptr)
    {
        depths[row] = valid ? node : -1;
    }
    if (node == 0)
    {
        contextLengths[request] = committedLengths[request] + width;
    }
    int32_t const words = (width + 31) / 32;
    for (int32_t word = 0; word < words; ++word)
    {
        int32_t const bits = min(32, node - word * 32 + 1);
        uint32_t const mask = !valid || bits <= 0 ? 0u : (bits == 32 ? 0xffffffffu : (1u << bits) - 1u);
        packedAttentionMask[row * words + word] = static_cast<int32_t>(mask);
    }
}

void validateTensor(rt::Tensor const& tensor, int64_t elements, trt_edgellm::DataType dtype)
{
    ELLM_CHECK(tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.getDataType() == dtype
            && tensor.getShape().volume() == elements && tensor.rawPointer() != nullptr,
        "Invalid chain metadata tensor shape, dtype or device");
}
} // namespace

void prepareSpecChainVerifyInputs(rt::Tensor const& committedLengths, rt::Tensor const* validTokenCounts,
    rt::Tensor& packedAttentionMask, rt::Tensor& positions, rt::Tensor& selectTokenIndices, rt::Tensor& contextLengths,
    rt::Tensor* parentIds, rt::Tensor* depths, int32_t requestCount, int32_t tokensPerRequest, cudaStream_t stream)
{
    ELLM_CHECK(requestCount > 0 && tokensPerRequest > 0 && tokensPerRequest <= 1024,
        "Chain metadata requires positive requestCount and tokensPerRequest in [1,1024]");
    int64_t const rows = static_cast<int64_t>(requestCount) * tokensPerRequest;
    validateTensor(committedLengths, requestCount, trt_edgellm::DataType::kINT32);
    validateTensor(contextLengths, requestCount, trt_edgellm::DataType::kINT32);
    validateTensor(positions, rows, trt_edgellm::DataType::kINT32);
    validateTensor(selectTokenIndices, rows, trt_edgellm::DataType::kINT64);
    validateTensor(packedAttentionMask, rows * ((tokensPerRequest + 31) / 32), trt_edgellm::DataType::kINT32);
    if (validTokenCounts != nullptr)
    {
        validateTensor(*validTokenCounts, requestCount, trt_edgellm::DataType::kINT32);
    }
    if (parentIds != nullptr)
    {
        validateTensor(*parentIds, rows, trt_edgellm::DataType::kINT32);
    }
    if (depths != nullptr)
    {
        validateTensor(*depths, rows, trt_edgellm::DataType::kINT32);
    }
    prepareChainMetadata<<<requestCount, tokensPerRequest, 0, stream>>>(committedLengths.dataPointer<int32_t>(),
        validTokenCounts == nullptr ? nullptr : validTokenCounts->dataPointer<int32_t>(),
        packedAttentionMask.dataPointer<int32_t>(), positions.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), contextLengths.dataPointer<int32_t>(),
        parentIds == nullptr ? nullptr : parentIds->dataPointer<int32_t>(),
        depths == nullptr ? nullptr : depths->dataPointer<int32_t>(), tokensPerRequest);
    CUDA_CHECK(cudaGetLastError());
}

constexpr int32_t kROOT_NODE_PREDECESSOR{-1};
constexpr int32_t kEMPTY_NODE_PREDECESSOR{-5};

__global__ void prepareSpecPrefillInputKernel(int64_t* selectTokenIndices, int32_t const* sequenceContextLengths)
{
    int32_t const batchIdx = blockIdx.x;
    if (threadIdx.x == 0)
    {
        int32_t const sequenceLength = sequenceContextLengths[batchIdx];
        selectTokenIndices[batchIdx] = sequenceLength - 1;
    }
}

__global__ void prepareSpecDraftProposalMiscInputKernel(int32_t const* draftTreeSizes,
    int32_t const* sequenceStartIndices, int32_t* sequenceContextLengths, int64_t* selectTokenIndices,
    int32_t selectTokenLength, int32_t paddedDraftTreeSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const blockSize = blockDim.x;
    int32_t const tIdx = threadIdx.x;
    // Use draftTreeSizes if provided, otherwise use paddedDraftTreeSize for all batches
    int32_t const draftTreeSize = (draftTreeSizes != nullptr) ? draftTreeSizes[batchIdx] : paddedDraftTreeSize;

    if (tIdx == 0)
    {
        // Special handling eagle, add padded size to meet the tree-attentionkernel implementation.
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + paddedDraftTreeSize;
    }

    // Select token indices is in format of [batch, select-token-length]. With current implementation,
    // we always put the whole tree into the computation and select the "current round" logits/hidden states
    // to proceed with the next round.
    for (int32_t i = tIdx; i < selectTokenLength; i += blockSize)
    {
        selectTokenIndices[batchIdx * selectTokenLength + i]
            = static_cast<int64_t>(batchIdx) * paddedDraftTreeSize + draftTreeSize - selectTokenLength + i;
    }
}

__global__ void assembleDraftTreeDescKernel(int8_t const* draftTreeMask, int32_t const* draftTreeSizes,
    int32_t const* sequenceStartIndices, int32_t* packedDraftTreeMask, int32_t* tensorPositionIndices,
    int32_t const paddedDraftTreeSize)
{
    constexpr int32_t kNUM_MASK_PER_ENTRY{32};
    constexpr int32_t kMAX_DRAFT_PACKED_TREE_SIZE{kMAX_SPEC_VERIFY_TREE_TOKENS / kNUM_MASK_PER_ENTRY};

    // Each thread will handle one token in the draft tree to setup the mask and tensor position indices.
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    int32_t packedTreeMask[kMAX_DRAFT_PACKED_TREE_SIZE] = {0};
    int32_t const packedTreeMaskLen = (paddedDraftTreeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;
    int32_t const actualDraftTreeSize = (draftTreeSizes != nullptr) ? draftTreeSizes[batchIdx] : paddedDraftTreeSize;
    int32_t const sequenceStartIndex = sequenceStartIndices[batchIdx];

    // Unpacked tree mask formulate in the format of [batch, padded-draft-tree-size, padded-draft-tree-size].
    // Packed tree mask len is in format of [batch, padded-draft-tree-size, divup(padded-draft-tree-size, 32)].
    // Tensor position indices is in format of [batch, padded-draft-tree-size].
    int32_t const unpackedTreeMaskOffset
        = batchIdx * paddedDraftTreeSize * paddedDraftTreeSize + tokenIdx * paddedDraftTreeSize;
    int32_t const packedTreeMaskOffset
        = batchIdx * paddedDraftTreeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
    int32_t const tensorPositionOffset = batchIdx * paddedDraftTreeSize + tokenIdx;

    int32_t tensorPositionIdx{0};
    if (tokenIdx < actualDraftTreeSize)
    {
        // With causal attention, the node will only attend to nodes "prior" to itself.
        int32_t attendNodeNum{0};
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            int8_t const maskFlag = draftTreeMask[unpackedTreeMaskOffset + i];
            if (maskFlag)
            {
                attendNodeNum += 1;
                packedTreeMask[i / kNUM_MASK_PER_ENTRY] |= (1 << (i % kNUM_MASK_PER_ENTRY));
            }
        }
        // A token always attend to itself, subtract 1 to reflect its position in the sequence.
        tensorPositionIdx = sequenceStartIndex + attendNodeNum - 1;
    }

    // Write result to the output. For node outside the "real" draft tree size, we will write 0 value to keep mask
    // well-formed.
    if (tokenIdx < paddedDraftTreeSize)
    {
        tensorPositionIndices[tensorPositionOffset] = tensorPositionIdx;
        for (int32_t i = 0; i < packedTreeMaskLen; ++i)
        {
            packedDraftTreeMask[packedTreeMaskOffset + i] = packedTreeMask[i];
        }
    }
}

__global__ void assembleCasualTreeAndSelectIndicesKernel(int32_t const* sequenceStartIndices, int32_t* packedTreeMasks,
    int32_t* tensorPositionIndices, int64_t* selectTokenIndices, int32_t* sequenceContextLengths,
    int32_t const* acceptedTokenNums, int32_t const maxAcceptedTokenNum)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    int32_t const acceptedTokenNum = acceptedTokenNums[batchIdx];

    // One 32-bit word holds the causal mask of up to kSpecMaxAcceptedPathLength accepted tokens.
    uint32_t packedTreeMask{0};
    int32_t const packedTreeMaskOffset = batchIdx * maxAcceptedTokenNum + tokenIdx;

    if (tokenIdx < acceptedTokenNum)
    {
        // Valid tokens: set causal mask and valid position
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            packedTreeMask |= (1U << i);
        }

        // Packed tree mask shall have layout of [batch, max-accepted-token-num, divup(max-accepted-token-num, 32)].
        // tensor position indices have layout of [batch, max-accepted-token-num], the offset will be identical to
        // packed tree mask.
        packedTreeMasks[packedTreeMaskOffset] = static_cast<int32_t>(packedTreeMask);
        tensorPositionIndices[packedTreeMaskOffset] = sequenceStartIndices[batchIdx] + tokenIdx;
    }
    else if (tokenIdx < maxAcceptedTokenNum)
    {
        // Padding tokens: set position to -1 to indicate padding, mask to 0 to prevent attention
        // The -1 position ensures:
        // 1. RoPE kernel won't write K/V to cache for padding tokens
        // 2. Padding tokens won't contribute to attention computation
        packedTreeMasks[packedTreeMaskOffset] = 0;
        tensorPositionIndices[packedTreeMaskOffset] = -1;
    }
    if (threadIdx.x == 0)
    {
        selectTokenIndices[batchIdx] = static_cast<int64_t>(batchIdx) * maxAcceptedTokenNum + acceptedTokenNum - 1;
        // Use maxAcceptedTokenNum (padded length) instead of acceptedTokenNum (actual length).
        // This ensures the attention kernel computes the correct context K range:
        //   cacheSeqLen = sequenceStartIndices + maxAcceptedTokenNum
        //   actualQSeqLen = maxAcceptedTokenNum (qSeqLen passed to kernel)
        //   Context K range = K[0 : cacheSeqLen - actualQSeqLen] = K[0 : sequenceStartIndices]
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + maxAcceptedTokenNum;
    }
}

__global__ void initializeDraftTreeFullTablesKernel(int32_t const* selectedIndices, float const* logProbs,
    int32_t const* rootTokens, int32_t const* vocabMappingTable, int32_t* draftIdFullTable, float* draftScoreFullTable,
    int32_t* draftParentFullTable, int32_t const draftTopK, int32_t const tableLength)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    for (int32_t i = tIdx; i < tableLength; i += blockSize)
    {
        // Root position, token shall be last committed token, with score 0, parent -1.
        int32_t tableOffset = batchIdx * tableLength + i;
        if (i == 0)
        {
            draftIdFullTable[tableOffset] = rootTokens[batchIdx];
            draftScoreFullTable[tableOffset] = 0.0f;
            draftParentFullTable[tableOffset] = kROOT_NODE_PREDECESSOR;
        }
        else if (i <= draftTopK)
        {
            // First-level candidates come from draft logits produced by draft prefill
            // or draft accept. Translate them to target-vocabulary token ids for
            // base verification. Full-vocab drafts use a zero-offset identity table.
            // Score is 0 (root) + log probability, parent points to root (0)
            int32_t const selectedOffset = batchIdx * draftTopK + i - 1;
            int32_t const draftTokenId = selectedIndices[selectedOffset];
            int32_t const baseTokenId = draftTokenId + vocabMappingTable[draftTokenId];
            float const logProbVal = logProbs[selectedOffset];
            draftIdFullTable[tableOffset] = baseTokenId;
            draftScoreFullTable[tableOffset] = logProbVal;
            draftParentFullTable[tableOffset] = 0;
        }
        else
        {
            // Empty initialize the rest of the table, clear garbage data to reduce confusion.
            draftIdFullTable[tableOffset] = 0;
            // -INFINITY from cmath to indicate infinity float value.
            draftScoreFullTable[tableOffset] = -INFINITY;
            draftParentFullTable[tableOffset] = kEMPTY_NODE_PREDECESSOR;
        }
    }
}

__global__ void initializeDraftTreeInputFirstRoundKernel(int32_t const* draftIdFullTable, int32_t* inputIds,
    int8_t* draftTreeMask, int32_t* draftTreeLength, int32_t const draftTopK, int32_t const paddedDraftTreeSize,
    int32_t const fullTableLength)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    for (int32_t i = tIdx; i < paddedDraftTreeSize; i += blockSize)
    {
        int32_t const idsOffset = batchIdx * paddedDraftTreeSize + i;
        // Draft tree mask is in format of [batch, padded-draft-tree-size, padded-draft-tree-size].
        int32_t const maskOffset = batchIdx * paddedDraftTreeSize * paddedDraftTreeSize + i * paddedDraftTreeSize;
        if (i < draftTopK)
        {
            // Handle non padded part of the output tensors.
            // First entry of the table is root token, offset 1 to get the first level draft tree.
            // Use fullTableLength parameter for correct multi-batch offset calculation
            int32_t const tableOffset = batchIdx * fullTableLength + i + 1;
            inputIds[idsOffset] = draftIdFullTable[tableOffset];
            // Prepare tree mask for these tokens.
            for (int32_t j = 0; j < paddedDraftTreeSize; ++j)
            {
                // First layer of the draft token only attend to itself.
                draftTreeMask[maskOffset + j] = (j == i) ? 1 : 0;
            }
        }
        else
        {
            // Padded region, zero initialize the tensors.
            inputIds[idsOffset] = 0;
            for (int32_t j = 0; j < paddedDraftTreeSize; ++j)
            {
                draftTreeMask[maskOffset + j] = 0;
            }
        }
    }

    // Update the draft tree length.
    if (tIdx == 0)
    {
        draftTreeLength[batchIdx] = draftTopK;
    }
}

__global__ void initializeDraftTreeInputKernel(int32_t const* tokenIdsTable, int32_t const* selectedIndices,
    int32_t* inputIds, int8_t* draftTreeMask, int32_t* draftTreeLength, int32_t const draftTopK,
    int32_t const paddedDraftTreeSize, int32_t const round)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;

    if (tIdx < draftTopK)
    {
        int32_t const selectedIdx = selectedIndices[batchIdx * draftTopK + tIdx];
        // Each row of the draftTopK x draftTopK matrix will comes from one parent.
        int32_t const parentidx = selectedIdx / draftTopK;
        // Each batch will have draftTopK * draftTopK token candidates.
        int32_t const selectedtokenIds = tokenIdsTable[batchIdx * draftTopK * draftTopK + selectedIdx];
        // Input ids is padded to paddedDraftTreeSize where each round contains draftTopK tokens.
        int32_t const inputIdsOffset = batchIdx * paddedDraftTreeSize + (round * draftTopK + tIdx);
        inputIds[inputIdsOffset] = selectedtokenIds;

        // Where the parent and the token itself locate within the padded draft tree size.
        int32_t const parentOffset = (round - 1) * draftTopK + parentidx;
        int32_t const selfOffset = round * draftTopK + tIdx;
        // Prepare tree mask for these tokens. For this token, it shall attend to itself, and all positions its parent
        // attend to.
        int32_t const parentsMaskOffset
            = batchIdx * paddedDraftTreeSize * paddedDraftTreeSize + parentOffset * paddedDraftTreeSize;
        int32_t const selfMaskOffset
            = batchIdx * paddedDraftTreeSize * paddedDraftTreeSize + selfOffset * paddedDraftTreeSize;
        for (int32_t j = 0; j < paddedDraftTreeSize; ++j)
        {
            draftTreeMask[selfMaskOffset + j] = draftTreeMask[parentsMaskOffset + j];
            if (j == selfOffset)
            {
                draftTreeMask[selfMaskOffset + j] = 1;
            }
        }
    }

    // Update tree length from this round.
    if (tIdx == 0)
    {
        draftTreeLength[batchIdx] += draftTopK;
    }
}

__global__ void assembleDraftHiddenStatesKernel(half const* draftHiddenOutput, int32_t const* selectedIndices,
    half* draftHiddenInput, int32_t const hiddenDim, int32_t const draftTopK, int32_t const paddedDraftTreeSize,
    int32_t const round)
{
    // The kernel will copy draft hidden states data from last round of output to the input for next round of drafting.
    // For simplicity, each CTA will be responsible for "one" hidden states in the output.
    int32_t const batchIdx = blockIdx.x;
    int32_t const dstHiddenIdx = blockIdx.y;
    int32_t const tIdx = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    DVec<half> vecData;
    constexpr int32_t VEC_SIZE = DVec<half>::vec_size;
    if (round == 0)
    {
        // For first round of drafting, there is only one output hidden entry.
        int32_t const srcOffset = batchIdx * hiddenDim;
        int32_t const dstOffset = batchIdx * paddedDraftTreeSize * hiddenDim + dstHiddenIdx * hiddenDim;
        for (int32_t i = tIdx; i < hiddenDim / VEC_SIZE; i += blockSize)
        {
            vecData.load(draftHiddenOutput + srcOffset + i * VEC_SIZE);
            vecData.store(draftHiddenInput + dstOffset + i * VEC_SIZE);
        }
    }
    else
    {
        // For non-first round, the output hidden states have layout of [batch, draftTopK, draft-hidden-size].
        // We need to find out the corresponding input hidden states index based on selected indices.
        // Selected indices come from a matrix of [draftTopK, draftTopK]. Each row maps to one src hidden states index.
        int32_t const selectedIndexOffset = batchIdx * draftTopK + dstHiddenIdx;
        int32_t const srcHiddenIdx = selectedIndices[selectedIndexOffset] / draftTopK;
        int32_t const srcOffset = batchIdx * draftTopK * hiddenDim + srcHiddenIdx * hiddenDim;

        // Dst hidden states is in padded shape of [batch, padded-draft-tree-size, draft-hidden-size]. Where
        // padded-draft-tree-size equals to total-num-round * draftTopK.
        int32_t const dstOffset
            = batchIdx * paddedDraftTreeSize * hiddenDim + (round * draftTopK + dstHiddenIdx) * hiddenDim;
        for (int32_t i = tIdx; i < hiddenDim / VEC_SIZE; i += blockSize)
        {
            vecData.load(draftHiddenOutput + srcOffset + i * VEC_SIZE);
            vecData.store(draftHiddenInput + dstOffset + i * VEC_SIZE);
        }
    }
}

__global__ void assembleIntermediateparentsKernel(
    int32_t const* selectedIndices, int32_t* intermediateParents, int32_t const draftTopK, int32_t const round)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;

    // Parents points to a location within the large draft tree table, the table have layout of [batch, 1 + draftTopK +
    // draftTopK * draftTopK * total-num-round]. We can obtain the offset as:
    int32_t startOffset{};
    if (round == 0)
    {
        startOffset = 1;
    }
    else
    {
        startOffset = 1 + draftTopK + draftTopK * draftTopK * (round - 1);
    }
    // In case we launch more threads than needed in each CTA.
    if (tIdx < draftTopK)
    {
        // For round 0, we don't have to read from selected indices.
        if (round == 0)
        {
            intermediateParents[batchIdx * draftTopK + tIdx] = startOffset + tIdx;
        }
        else
        {
            // Here selected indices come from a matrix of [draftTopK, draftTopK].
            int32_t const selectedIdx = selectedIndices[batchIdx * draftTopK + tIdx];
            intermediateParents[batchIdx * draftTopK + tIdx] = startOffset + selectedIdx;
        }
    }
}

__global__ void computeCuScoresAndTranslateTokenKernel(int32_t const* selectedIndices, float const* logProbs,
    float const* intermediateScores, int32_t const* vocabMappingTable, int32_t* draftIdTable, float* draftScoreTable,
    int32_t const draftTopK)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    for (int32_t i = tIdx; i < draftTopK * draftTopK; i += blockSize)
    {
        int32_t const draftTokenIds = selectedIndices[batchIdx * draftTopK * draftTopK + i];
        int32_t const baseTokenIds = vocabMappingTable[draftTokenIds] + draftTokenIds;
        int32_t const tableOffset = batchIdx * draftTopK * draftTopK + i;
        draftIdTable[tableOffset] = baseTokenIds;

        int32_t const parentIdx = i / draftTopK;
        float const parentScore = intermediateScores[batchIdx * draftTopK + parentIdx];
        draftScoreTable[tableOffset] = parentScore + logProbs[tableOffset];
    }
}

__global__ void sliceCandidateTopKKernel(int32_t const* candidateIds, float const* candidateLogProbs,
    int32_t* outIndices, float* outScores, int32_t const candidateCount, int32_t const topK, int32_t const outElements)
{
    int32_t const outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outIdx >= outElements)
    {
        return;
    }
    int32_t const row = outIdx / topK;
    int64_t const srcIdx = static_cast<int64_t>(row) * candidateCount + (outIdx - row * topK);
    outIndices[outIdx] = candidateIds[srcIdx];
    outScores[outIdx] = candidateLogProbs[srcIdx];
}

constexpr int32_t kVERIFY_SELECT_BLOCK_SIZE{256};
//! Bounds the shared-memory row copy (8 KB) and the O(F^2) rank scan; EAGLE's draftSteps * draftTopK <= 64 keeps
//! full tables at or below 1 + 32 + 32^2 = 1057 entries.
constexpr int32_t kVERIFY_SELECT_MAX_TABLE_LENGTH{2048};

//! One CTA per row. A non-root entry's slot is 1 + its rank among entries 1..F-1 under (score desc, index asc); the
//! order is total, so ranks are unique and every slot below verifyCount is written exactly once.
__global__ void selectVerifyTreeNodesKernel(float const* scoreTable, int32_t* outIndices, float* outScores,
    int32_t const tableLength, int32_t const verifyCount)
{
    extern __shared__ float rowScores[];
    int32_t const batchIdx = blockIdx.x;
    float const* const rowIn = scoreTable + static_cast<int64_t>(batchIdx) * tableLength;
    for (int32_t i = threadIdx.x; i < tableLength; i += blockDim.x)
    {
        float const score = rowIn[i];
        rowScores[i] = isnan(score) ? -INFINITY : score;
    }
    __syncthreads();

    int64_t const outOffset = static_cast<int64_t>(batchIdx) * verifyCount;
    int32_t const nonRootSlots = verifyCount - 1;
    for (int32_t i = threadIdx.x; i < tableLength; i += blockDim.x)
    {
        float const score = rowScores[i];
        int32_t slot{0};
        if (i > 0)
        {
            int32_t rank{0};
            for (int32_t j = 1; j < tableLength && rank < nonRootSlots; ++j)
            {
                float const other = rowScores[j];
                rank += (other > score || (other == score && j < i)) ? 1 : 0;
            }
            if (rank >= nonRootSlots)
            {
                continue;
            }
            slot = 1 + rank;
        }
        outIndices[outOffset + slot] = i;
        if (outScores != nullptr)
        {
            outScores[outOffset + slot] = score;
        }
    }
}

__global__ void updateDraftTreeFullTablesKernel(int32_t const* draftIdTable, float const* draftScoreTable,
    int32_t const* intermediateParents, int32_t* draftIdFullTable, float* draftScoreFullTable,
    int32_t* draftParentFullTable, int32_t const draftTopK, int32_t const round, int32_t const fullTableLength)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    for (int32_t i = tIdx; i < draftTopK * draftTopK; i += blockSize)
    {
        int32_t const srcTableOffset = batchIdx * draftTopK * draftTopK + i;
        // Full table has layout of [batch, 1 + draftTopK + draftTopK * draftTopK * total-num-round].
        int32_t const dstTableOffset = batchIdx * fullTableLength + (1 + draftTopK + draftTopK * draftTopK * round + i);
        draftIdFullTable[dstTableOffset] = draftIdTable[srcTableOffset];
        draftScoreFullTable[dstTableOffset] = draftScoreTable[srcTableOffset];
        // Obtain the parent index from intermediate parents info we collected previously.
        int32_t const parentIdx = i / draftTopK;
        draftParentFullTable[dstTableOffset] = intermediateParents[batchIdx * draftTopK + parentIdx];
    }
}

__global__ void buildSpecVerificationTreeKernel(int32_t const* draftIdFullTable, int32_t const* draftParentFullTable,
    int32_t const* selectedIndices, int32_t* inputIds, int8_t* draftTreeMask, int32_t* parentIds, int32_t* depths,
    int32_t const fullTableLength, int32_t const verifyTreeSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;

    if (tIdx >= verifyTreeSize)
    {
        return;
    }

    // Sized for kMTPMaxAcceptDepthForCurrentEagleUtilityKernels + 1 (see deploymentConfig.cpp), so the
    // deepest accept path allowed by config validation always fits with one level of headroom.
    // We don't use shared memory since the data tables are small and can automatically fit into L1.
    constexpr int32_t kMAX_DEPTH{17};
    int32_t parentIndices[kMAX_DEPTH] = {-1};
    int32_t attendedIndices[kMAX_DEPTH + 1] = {-1};

    int32_t const verifyTreeCTAOffset = batchIdx * verifyTreeSize;
    int32_t const fullTableCTAOffset = batchIdx * fullTableLength;

    int32_t const selectedIdx = selectedIndices[verifyTreeCTAOffset + tIdx];
    inputIds[verifyTreeCTAOffset + tIdx] = draftIdFullTable[fullTableCTAOffset + selectedIdx];

    // Collect number of parents and parents indices.
    int32_t numParents{0};
    int32_t parentIter = draftParentFullTable[fullTableCTAOffset + selectedIdx];

    // By design, all token will finally trace back to root token which has parent index of -1.
    // Root token won't attend to any other token.
    while (parentIter != kROOT_NODE_PREDECESSOR)
    {
        parentIndices[numParents] = parentIter;
        numParents += 1;
        parentIter = draftParentFullTable[fullTableCTAOffset + parentIter];
    }

    // To establish the tree mask, we need to find out the location of each parent within the
    // verify tree. We can iterate the selected indices and match the parent indices we collected.
    // First each token will attend to itself.
    attendedIndices[0] = tIdx;
    // count from myself towrds prior locations.
    int32_t countIter{tIdx};
    // Attend iter starts from 1.
    int32_t attendIter{1};
    // Reset parent iterator to match from the first predecessor.
    parentIter = 0;
    while (countIter > 0)
    {
        countIter -= 1;
        // Compare the full table index at this verification position with the parent index
        int32_t const fullTableIdxAtCount = selectedIndices[verifyTreeCTAOffset + countIter];
        if (fullTableIdxAtCount == parentIndices[parentIter])
        {
            attendedIndices[attendIter] = countIter;
            attendIter += 1;
            parentIter += 1;
        }
    }
    // Now we start to establish the tree mask which have layout of [batch, verify-tree-size, verify-tree-size].
    int32_t const verifyTreeOffset = batchIdx * verifyTreeSize * verifyTreeSize + tIdx * verifyTreeSize;
    // First clear the row to all zeros and then fill in the attended indices.
    for (int32_t i = 0; i < verifyTreeSize; i++)
    {
        draftTreeMask[verifyTreeOffset + i] = 0;
    }
    for (int32_t i = 0; i < attendIter; i++)
    {
        draftTreeMask[verifyTreeOffset + attendedIndices[i]] = 1;
    }

    if (parentIds != nullptr)
    {
        // The walk above matches the immediate predecessor first, so attendedIndices[1] is the
        // parent's row. The root, and a node whose parent missed the selection, report -1.
        parentIds[verifyTreeCTAOffset + tIdx] = attendIter >= 2 ? attendedIndices[1] : -1;
    }
    if (depths != nullptr)
    {
        depths[verifyTreeCTAOffset + tIdx] = attendIter - 1;
    }
}

void prepareSpecPrefillInputs(
    rt::Tensor const& sequenceContextLengths, rt::Tensor& selectTokenIndices, cudaStream_t stream)
{
    check::check(selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for the input tensors.");
    check::check(selectTokenIndices.getDataType() == DataType::kINT64
            && sequenceContextLengths.getDataType() == DataType::kINT32,
        "Select-token-indices shall be INT64 and sequence-context-lengths shall be INT32.");
    uint32_t const batchSize = sequenceContextLengths.getShape()[0];

    // Assign one warp for each batch.
    dim3 const blockDim{32};
    dim3 const gridDim{batchSize};
    prepareSpecPrefillInputKernel<<<gridDim, blockDim, 0, stream>>>(
        selectTokenIndices.dataPointer<int64_t>(), sequenceContextLengths.dataPointer<int32_t>());
}

void prepareSpecDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
    rt::Tensor const& sequenceStartIndices, rt::Tensor& packedDraftTreeMask, rt::Tensor& tensorPositionIndices,
    rt::Tensor& selectTokenIndices, rt::Tensor& sequenceContextLengths, cudaStream_t stream)
{
    check::check(draftTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
            && sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedDraftTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftTreeMask.getDataType() == DataType::kINT8 && draftTreeLength.getDataType() == DataType::kINT32
            && sequenceStartIndices.getDataType() == DataType::kINT32
            && packedDraftTreeMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64
            && sequenceContextLengths.getDataType() == DataType::kINT32,
        "Data type check failed for the input tensors.");

    uint32_t const batchSize = draftTreeMask.getShape()[0];
    int32_t const paddedDraftTreeSize = draftTreeMask.getShape()[1];
    // Support both 1D [batch*length] and 2D [batch, length] tensors by using total volume
    int32_t const selectTokenLength = selectTokenIndices.getShape().volume() / batchSize;

    check::check(tensorPositionIndices.getShape()[1] == paddedDraftTreeSize,
        "Tensor position indices shall have shape [batch, padded-draft-tree-size].");

    // Round up block size to multiple of warp
    uint32_t const blocksize = divUp(paddedDraftTreeSize, 32) * 32;
    // Perform tree mask packing and tensor position indices.
    dim3 const blockDim1{blocksize};
    dim3 const gridDim1{batchSize};
    assembleDraftTreeDescKernel<<<gridDim1, blockDim1, 0, stream>>>(draftTreeMask.dataPointer<int8_t>(),
        draftTreeLength.dataPointer<int32_t>(), sequenceStartIndices.dataPointer<int32_t>(),
        packedDraftTreeMask.dataPointer<int32_t>(), tensorPositionIndices.dataPointer<int32_t>(), paddedDraftTreeSize);

    // Perform misc input setup, assign one warp for each batch since selectTokenLength is around 8 ~ 12.
    dim3 const blockDim2{32};
    dim3 const gridDim2{batchSize};
    prepareSpecDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(draftTreeLength.dataPointer<int32_t>(),
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), selectTokenLength, paddedDraftTreeSize);
}

void prepareSpecAcceptedDraftInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor const& acceptedTokenNums,
    rt::Tensor& packedTreeMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream)
{
    check::check(sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedTreeMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && acceptedTokenNums.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(sequenceStartIndices.getDataType() == DataType::kINT32
            && packedTreeMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64
            && acceptedTokenNums.getDataType() == DataType::kINT32,
        "Data type validation failed.");
    uint32_t const batchSize = sequenceStartIndices.getShape()[0];
    int32_t const maxAcceptedTokenNum = packedTreeMask.getShape()[1];
    check::check(maxAcceptedTokenNum <= kSpecMaxAcceptedPathLength,
        "Current kernel implementation support accepted token <= 32 per batch. "
        "Packed tree mask shall have shape [batch, max-accepted-token-num, 1].");
    check::check(acceptedTokenNums.getShape()[0] == batchSize,
        "acceptedTokenNums batch size should match sequenceStartIndices.");

    // Round up block size to multiple of warp size. Use max to handle all batches.
    // Note: This uses maxAcceptedTokenNum for uniform block size across all batches.
    // Threads with threadIdx.x >= acceptedTokenNum (per-batch) will early-exit.
    // Trade-off: Simplifies launch configuration but may launch idle threads for batches
    // with fewer accepted tokens. Since maxAcceptedTokenNum <= 32, overhead is minimal.
    uint32_t const blocksize = divUp(maxAcceptedTokenNum, 32) * 32;
    // Perform casual tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};

    assembleCasualTreeAndSelectIndicesKernel<<<gridDim, blockDim, 0, stream>>>(
        sequenceStartIndices.dataPointer<int32_t>(), packedTreeMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(),
        sequenceContextLengths.dataPointer<int32_t>(), acceptedTokenNums.dataPointer<int32_t>(), maxAcceptedTokenNum);
}

void prepareSpecFullTreeInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
    rt::Tensor& packedBaseTreeDecodingMask, rt::Tensor& tensorPositionIndices, rt::Tensor& selectTokenIndices,
    rt::Tensor& sequenceContextLengths, cudaStream_t stream)
{
    check::check(baseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
            && sequenceStartIndices.getDeviceType() == rt::DeviceType::kGPU
            && packedBaseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
            && tensorPositionIndices.getDeviceType() == rt::DeviceType::kGPU
            && selectTokenIndices.getDeviceType() == rt::DeviceType::kGPU
            && sequenceContextLengths.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(baseTreeDecodingMask.getDataType() == DataType::kINT8
            && sequenceStartIndices.getDataType() == DataType::kINT32
            && packedBaseTreeDecodingMask.getDataType() == DataType::kINT32
            && tensorPositionIndices.getDataType() == DataType::kINT32
            && selectTokenIndices.getDataType() == DataType::kINT64
            && sequenceContextLengths.getDataType() == DataType::kINT32,
        "Data type check failed for the input tensors.");

    uint32_t const batchSize = baseTreeDecodingMask.getShape()[0];
    int32_t const treeSize = baseTreeDecodingMask.getShape()[1];

    check::check(tensorPositionIndices.getShape()[1] == treeSize,
        "Tensor position indices shall have shape [batch, tree-size].");

    // Round up block size to multiple of warp
    uint32_t const blocksize = divUp(treeSize, 32) * 32;
    // Perform tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};
    assembleDraftTreeDescKernel<<<gridDim, blockDim, 0, stream>>>(baseTreeDecodingMask.dataPointer<int8_t>(), nullptr,
        sequenceStartIndices.dataPointer<int32_t>(), packedBaseTreeDecodingMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), treeSize);

    // Perform misc input setup, assign one warp for each batch since selectTokenLength is around 8 ~ 12.
    dim3 const blockDim2{32};
    dim3 const gridDim2{batchSize};
    prepareSpecDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(nullptr,
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), treeSize, treeSize);
}

__global__ void clampAcceptLengthsKernel(
    int32_t* acceptLengths, AcceptLengthBudgets const budgets, int32_t slotOffset, int32_t numSlots)
{
    int32_t const slot = static_cast<int32_t>(threadIdx.x);
    if (slot < numSlots)
    {
        int32_t const acceptLength = acceptLengths[slotOffset + slot];
        acceptLengths[slotOffset + slot] = max(0, min(acceptLength, budgets.remaining[slot]));
    }
}

void clampAcceptLengths(rt::Tensor& acceptLengths, AcceptLengthBudgets const& budgets, int32_t slotOffset,
    int32_t numSlots, cudaStream_t stream)
{
    check::check(acceptLengths.getDeviceType() == rt::DeviceType::kGPU, "acceptLengths shall reside on the GPU.");
    check::check(acceptLengths.getDataType() == DataType::kINT32, "acceptLengths should be INT32.");
    check::check(acceptLengths.getShape().getNumDims() == 1, "acceptLengths should be 1D tensor [batch].");
    check::check(numSlots > 0 && numSlots <= kMaxAcceptLengthBudgetsPerLaunch,
        "numSlots must be in [1, kMaxAcceptLengthBudgetsPerLaunch].");
    check::check(slotOffset >= 0 && static_cast<int64_t>(slotOffset) + numSlots <= acceptLengths.getShape()[0],
        "Clamped slot range exceeds acceptLengths.");

    clampAcceptLengthsKernel<<<1, kMaxAcceptLengthBudgetsPerLaunch, 0, stream>>>(
        acceptLengths.dataPointer<int32_t>(), budgets, slotOffset, numSlots);
    CUDA_CHECK(cudaGetLastError());
}

void initializeDraftTreeTables(rt::Tensor const& selectedIndices, rt::Tensor const& logProb,
    rt::Tensor const& rootTokens, rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdFullTable,
    rt::Tensor& draftScoreFullTable, rt::Tensor& draftParentFullTable, int32_t const draftTopK, cudaStream_t stream)
{
    check::check(selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && logProb.getDeviceType() == rt::DeviceType::kGPU && rootTokens.getDeviceType() == rt::DeviceType::kGPU
            && vocabMappingTable.getDeviceType() == rt::DeviceType::kGPU
            && draftIdFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftScoreFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftParentFullTable.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(selectedIndices.getDataType() == DataType::kINT32 && logProb.getDataType() == DataType::kFLOAT
            && rootTokens.getDataType() == DataType::kINT32 && vocabMappingTable.getDataType() == DataType::kINT32
            && draftIdFullTable.getDataType() == DataType::kINT32
            && draftScoreFullTable.getDataType() == DataType::kFLOAT
            && draftParentFullTable.getDataType() == DataType::kINT32,
        "All datatypes shall be valid.");
    auto const batchSize = static_cast<uint32_t>(selectedIndices.getShape()[0]);
    int32_t const tableLength = static_cast<int32_t>(draftIdFullTable.getShape()[1]);

    check::check(selectedIndices.getShape()[1] == draftTopK, "Check selected indices dimension.");
    check::check(logProb.getShape()[1] == draftTopK, "Check log probability dimension.");

    dim3 blockDim{256};
    dim3 gridDim{batchSize};
    initializeDraftTreeFullTablesKernel<<<gridDim, blockDim, 0, stream>>>(selectedIndices.dataPointer<int32_t>(),
        logProb.dataPointer<float>(), rootTokens.dataPointer<int32_t>(), vocabMappingTable.dataPointer<int32_t>(),
        draftIdFullTable.dataPointer<int32_t>(), draftScoreFullTable.dataPointer<float>(),
        draftParentFullTable.dataPointer<int32_t>(), draftTopK, tableLength);
}

void assembleInitialDraftTreeInput(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftHiddenStatesOutput,
    rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput, rt::Tensor& draftTreeLength, rt::Tensor& draftTreeMask,
    int32_t const draftTopK, cudaStream_t stream)
{
    check::check(draftIdFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftHiddenStatesOutput.getDeviceType() == rt::DeviceType::kGPU
            && inputIds.getDeviceType() == rt::DeviceType::kGPU
            && draftHiddenStatesInput.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeMask.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftIdFullTable.getDataType() == DataType::kINT32
            && draftHiddenStatesOutput.getDataType() == DataType::kHALF && inputIds.getDataType() == DataType::kINT32
            && draftHiddenStatesInput.getDataType() == DataType::kHALF
            && draftTreeLength.getDataType() == DataType::kINT32 && draftTreeMask.getDataType() == DataType::kINT8,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(draftIdFullTable.getShape()[0]);
    int32_t const draftHiddenDim = static_cast<int32_t>(draftHiddenStatesOutput.getShape()[1]);
    int32_t const paddedDraftTreeSize = static_cast<int32_t>(inputIds.getShape()[1]);
    check::check(
        draftHiddenStatesOutput.getShape()[0] == batchSize, "OutputHidden only contains last committed token.");
    check::check(draftHiddenStatesInput.getShape()[1] == paddedDraftTreeSize, "Use padded draft tree size for inputs.");
    check::check(draftTreeMask.getShape()[0] == batchSize && draftTreeMask.getShape()[1] == paddedDraftTreeSize
            && draftTreeMask.getShape()[2] == paddedDraftTreeSize,
        "Draft tree length shall have shape [batch, padded-draft-tree-size, padded-draft-tree-size].");

    dim3 blockDim1{128};
    dim3 gridDim1{static_cast<uint32_t>(batchSize)};
    int32_t const fullTableLength = static_cast<int32_t>(draftIdFullTable.getShape()[1]);
    initializeDraftTreeInputFirstRoundKernel<<<gridDim1, blockDim1, 0, stream>>>(
        draftIdFullTable.dataPointer<int32_t>(), inputIds.dataPointer<int32_t>(), draftTreeMask.dataPointer<int8_t>(),
        draftTreeLength.dataPointer<int32_t>(), draftTopK, paddedDraftTreeSize, fullTableLength);

    dim3 blockDim2{128};
    dim3 gridDim2{static_cast<uint32_t>(batchSize), static_cast<uint32_t>(draftTopK)};

    // For first round of hidden states assembly, selected indices is not needed.
    int32_t* selectedIndices{nullptr};
    int32_t const round{0};

    assembleDraftHiddenStatesKernel<<<gridDim2, blockDim2, 0, stream>>>(draftHiddenStatesOutput.dataPointer<half>(),
        selectedIndices, draftHiddenStatesInput.dataPointer<half>(), draftHiddenDim, draftTopK, paddedDraftTreeSize,
        round);
}

void assembleDraftTreeInput(rt::Tensor const& draftIdTable, rt::Tensor const& draftHiddenOutput,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftHiddenStatesInput,
    rt::Tensor& draftTreeLength, rt::Tensor& draftTreeMask, int32_t const draftTopK, int32_t const round,
    cudaStream_t stream)
{
    check::check(draftIdTable.getDeviceType() == rt::DeviceType::kGPU
            && draftHiddenOutput.getDeviceType() == rt::DeviceType::kGPU
            && selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && inputIds.getDeviceType() == rt::DeviceType::kGPU
            && draftHiddenStatesInput.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeMask.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftIdTable.getDataType() == DataType::kINT32 && draftHiddenOutput.getDataType() == DataType::kHALF
            && selectedIndices.getDataType() == DataType::kINT32 && inputIds.getDataType() == DataType::kINT32
            && draftHiddenStatesInput.getDataType() == DataType::kHALF
            && draftTreeLength.getDataType() == DataType::kINT32 && draftTreeMask.getDataType() == DataType::kINT8,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(draftIdTable.getShape()[0]);
    int32_t const draftHiddenDim = static_cast<int32_t>(draftHiddenOutput.getShape()[1]);
    int32_t const paddedDraftTreeSize = static_cast<int32_t>(inputIds.getShape()[1]);
    check::check(draftIdTable.getShape()[1] == draftTopK * draftTopK, "Check draft id table dimension.");
    check::check(draftHiddenOutput.getShape()[0] == batchSize * draftTopK, "Check draft hidden output dimension.");
    check::check(selectedIndices.getShape()[1] == draftTopK, "Check selected indices dimension.");
    check::check(inputIds.getShape()[0] == batchSize, "Check input ids dimension.");
    check::check(inputIds.getShape()[1] == paddedDraftTreeSize, "Check input ids dimension.");
    check::check(
        draftHiddenStatesInput.getShape()[1] == paddedDraftTreeSize, "Check draft hidden states input dimension.");

    dim3 blockDim1{32};
    dim3 gridDim1{static_cast<uint32_t>(batchSize)};
    initializeDraftTreeInputKernel<<<gridDim1, blockDim1, 0, stream>>>(draftIdTable.dataPointer<int32_t>(),
        selectedIndices.dataPointer<int32_t>(), inputIds.dataPointer<int32_t>(), draftTreeMask.dataPointer<int8_t>(),
        draftTreeLength.dataPointer<int32_t>(), draftTopK, paddedDraftTreeSize, round);

    dim3 blockDim2{128};
    dim3 gridDim2{static_cast<uint32_t>(batchSize), static_cast<uint32_t>(draftTopK)};

    assembleDraftHiddenStatesKernel<<<gridDim2, blockDim2, 0, stream>>>(draftHiddenOutput.dataPointer<half>(),
        selectedIndices.dataPointer<int32_t>(), draftHiddenStatesInput.dataPointer<half>(), draftHiddenDim, draftTopK,
        paddedDraftTreeSize, round);
}

void assembleInitialIntermediateData(rt::Tensor const& logProbs, rt::Tensor& intermediateParents,
    rt::Tensor& intermediateScores, int32_t const draftTopK, cudaStream_t stream)
{
    check::check(logProbs.getDeviceType() == rt::DeviceType::kGPU
            && intermediateParents.getDeviceType() == rt::DeviceType::kGPU
            && intermediateScores.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(logProbs.getDataType() == DataType::kFLOAT && intermediateParents.getDataType() == DataType::kINT32
            && intermediateScores.getDataType() == DataType::kFLOAT,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(logProbs.getShape()[0]);
    check::check(logProbs.getShape()[1] == draftTopK, "Check log probability dimension.");
    check::check(intermediateParents.getShape()[1] == draftTopK, "Check intermediate parent dimension.");
    check::check(intermediateScores.getShape()[1] == draftTopK, "Check intermediate score dimension.");

    // We can directly copy the log probabilities to intermediate scores.
    CUDA_CHECK(cudaMemcpyAsync(intermediateScores.dataPointer<float>(), logProbs.dataPointer<float>(),
        batchSize * draftTopK * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Assemble the interdemidate parents.
    dim3 blockDim{32};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};

    // For first round we don't have the selected indices from [draftTopK, draftTopK] candidates.
    int32_t* selectedIndices{nullptr};
    int32_t const round{0};
    assembleIntermediateparentsKernel<<<gridDim, blockDim, 0, stream>>>(
        selectedIndices, intermediateParents.dataPointer<int32_t>(), draftTopK, round);
}

void assembleIntermediateData(rt::Tensor const& cuLogProbs, rt::Tensor const& selectedIndices,
    rt::Tensor& intermediateScores, rt::Tensor& intermediateParents, int32_t const draftTopK, int32_t const round,
    cudaStream_t stream)
{
    check::check(cuLogProbs.getDeviceType() == rt::DeviceType::kGPU
            && selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && intermediateScores.getDeviceType() == rt::DeviceType::kGPU
            && intermediateParents.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(cuLogProbs.getDataType() == DataType::kFLOAT && selectedIndices.getDataType() == DataType::kINT32
            && intermediateScores.getDataType() == DataType::kFLOAT
            && intermediateParents.getDataType() == DataType::kINT32,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(cuLogProbs.getShape()[0]);
    check::check(cuLogProbs.getShape()[1] == draftTopK, "Check cu log probability dimension.");
    check::check(selectedIndices.getShape()[1] == draftTopK, "Check selected indices dimension.");
    check::check(intermediateScores.getShape()[1] == draftTopK, "Check intermediate score dimension.");
    check::check(intermediateParents.getShape()[1] == draftTopK, "Check intermediate parent dimension.");

    // We can directly copy the cu log probabilities to intermediate scores.
    CUDA_CHECK(cudaMemcpyAsync(intermediateScores.dataPointer<float>(), cuLogProbs.dataPointer<float>(),
        batchSize * draftTopK * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Assemble the interdemidate parents.
    dim3 blockDim{32};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};
    assembleIntermediateparentsKernel<<<gridDim, blockDim, 0, stream>>>(
        selectedIndices.dataPointer<int32_t>(), intermediateParents.dataPointer<int32_t>(), draftTopK, round);
}

void computeCuScoresAndTranslateToken(rt::Tensor const& selectedIndices, rt::Tensor const& logProbs,
    rt::Tensor const& intermediateScores, rt::Tensor const& vocabMappingTable, rt::Tensor& draftIdTable,
    rt::Tensor& draftScoreTable, int32_t const draftTopK, cudaStream_t stream)
{
    check::check(selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && logProbs.getDeviceType() == rt::DeviceType::kGPU
            && intermediateScores.getDeviceType() == rt::DeviceType::kGPU
            && vocabMappingTable.getDeviceType() == rt::DeviceType::kGPU
            && draftIdTable.getDeviceType() == rt::DeviceType::kGPU
            && draftScoreTable.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(selectedIndices.getDataType() == DataType::kINT32 && logProbs.getDataType() == DataType::kFLOAT
            && intermediateScores.getDataType() == DataType::kFLOAT
            && vocabMappingTable.getDataType() == DataType::kINT32 && draftIdTable.getDataType() == DataType::kINT32
            && draftScoreTable.getDataType() == DataType::kFLOAT,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(selectedIndices.getShape()[0]);
    check::check(selectedIndices.getShape()[1] == draftTopK * draftTopK, "Check selected indices dimension.");
    check::check(logProbs.getShape()[1] == draftTopK * draftTopK, "Check log probability dimension.");
    check::check(intermediateScores.getShape()[1] == draftTopK, "Check intermediate score dimension.");
    check::check(draftIdTable.getShape()[1] == draftTopK * draftTopK, "Check draft id table dimension.");
    check::check(draftScoreTable.getShape()[1] == draftTopK * draftTopK, "Check draft score table dimension.");

    dim3 blockDim{128};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};
    computeCuScoresAndTranslateTokenKernel<<<gridDim, blockDim, 0, stream>>>(selectedIndices.dataPointer<int32_t>(),
        logProbs.dataPointer<float>(), intermediateScores.dataPointer<float>(),
        vocabMappingTable.dataPointer<int32_t>(), draftIdTable.dataPointer<int32_t>(),
        draftScoreTable.dataPointer<float>(), draftTopK);
}

void sliceCandidateTopK(rt::Tensor const& candidateIds, rt::Tensor const& candidateLogProbs, int32_t topK,
    rt::Tensor& outIndices, rt::Tensor& outScores, cudaStream_t stream)
{
    ELLM_CHECK(candidateIds.getDeviceType() == rt::DeviceType::kGPU
            && candidateLogProbs.getDeviceType() == rt::DeviceType::kGPU
            && outIndices.getDeviceType() == rt::DeviceType::kGPU && outScores.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    ELLM_CHECK(candidateIds.getDataType() == DataType::kINT32 && candidateLogProbs.getDataType() == DataType::kFLOAT
            && outIndices.getDataType() == DataType::kINT32 && outScores.getDataType() == DataType::kFLOAT,
        "Data type shall all be valid.");
    auto const& candidateShape = candidateIds.getShape();
    ELLM_CHECK(candidateShape.getNumDims() == 2 && candidateLogProbs.getShape().getNumDims() == 2
            && candidateLogProbs.getShape()[0] == candidateShape[0]
            && candidateLogProbs.getShape()[1] == candidateShape[1],
        "Candidate ids and log-probabilities shall both have shape [rows, candidate-count].");
    int64_t const rows = candidateShape[0];
    int64_t const candidateCount = candidateShape[1];
    ELLM_CHECK(topK > 0 && topK <= candidateCount, "topK shall be in [1, candidate-count].");
    ELLM_CHECK(outIndices.getShape().getNumDims() == 2 && outIndices.getShape()[0] == rows
            && outIndices.getShape()[1] == topK && outScores.getShape().getNumDims() == 2
            && outScores.getShape()[0] == rows && outScores.getShape()[1] == topK,
        "Outputs shall have shape [rows, topK].");
    int64_t const outElements = rows * topK;
    ELLM_CHECK(outElements <= std::numeric_limits<int32_t>::max(), "Candidate slice is too large.");
    if (outElements == 0)
    {
        return;
    }
    constexpr int32_t kBLOCK_SIZE{128};
    dim3 const gridDim{static_cast<uint32_t>((outElements + kBLOCK_SIZE - 1) / kBLOCK_SIZE)};
    sliceCandidateTopKKernel<<<gridDim, kBLOCK_SIZE, 0, stream>>>(candidateIds.dataPointer<int32_t>(),
        candidateLogProbs.dataPointer<float>(), outIndices.dataPointer<int32_t>(), outScores.dataPointer<float>(),
        static_cast<int32_t>(candidateCount), topK, static_cast<int32_t>(outElements));
    CUDA_CHECK(cudaGetLastError());
}

void selectVerifyTreeNodes(rt::Tensor const& scoreTable, int32_t verifyCount, rt::Tensor& outIndices,
    rt::OptionalOutputTensor const& outScores, int32_t draftTopK, cudaStream_t stream)
{
    ELLM_CHECK(scoreTable.getDeviceType() == rt::DeviceType::kGPU && scoreTable.getDataType() == DataType::kFLOAT
            && scoreTable.getShape().getNumDims() == 2,
        "Score table shall be a GPU float tensor of shape [batch, table-length].");
    int64_t const batchSize = scoreTable.getShape()[0];
    int64_t const tableLength = scoreTable.getShape()[1];
    ELLM_CHECK(draftTopK > 0 && tableLength >= 1 + draftTopK
            && (tableLength - 1 - draftTopK) % (static_cast<int64_t>(draftTopK) * draftTopK) == 0,
        "Score table length shall be 1 + draftTopK + n x draftTopK^2.");
    ELLM_CHECK(tableLength <= kVERIFY_SELECT_MAX_TABLE_LENGTH,
        "Score table length exceeds " + std::to_string(kVERIFY_SELECT_MAX_TABLE_LENGTH) + " entries.");
    ELLM_CHECK(verifyCount > 0 && verifyCount <= tableLength, "verifyCount shall be in [1, table-length].");
    auto const isSelectionOutput = [&](rt::Tensor const& tensor, DataType dtype) {
        return tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.getDataType() == dtype
            && tensor.getShape().getNumDims() == 2 && tensor.getShape()[0] == batchSize
            && tensor.getShape()[1] == verifyCount;
    };
    ELLM_CHECK(isSelectionOutput(outIndices, DataType::kINT32),
        "outIndices shall be a GPU INT32 tensor of shape [batch, verifyCount].");
    ELLM_CHECK(!outScores.has_value() || isSelectionOutput(outScores->get(), DataType::kFLOAT),
        "outScores shall be a GPU FLOAT tensor of shape [batch, verifyCount].");
    if (batchSize == 0)
    {
        return;
    }
    size_t const sharedBytes = static_cast<size_t>(tableLength) * sizeof(float);
    selectVerifyTreeNodesKernel<<<static_cast<uint32_t>(batchSize), kVERIFY_SELECT_BLOCK_SIZE, sharedBytes, stream>>>(
        scoreTable.dataPointer<float>(), outIndices.dataPointer<int32_t>(),
        outScores.has_value() ? outScores->get().dataPointer<float>() : nullptr, static_cast<int32_t>(tableLength),
        verifyCount);
    CUDA_CHECK(cudaGetLastError());
}

void updateDraftTreeFullTables(rt::Tensor const& draftIdTable, rt::Tensor const& draftScoreTable,
    rt::Tensor const& intermediateParents, rt::Tensor& draftIdFullTable, rt::Tensor& draftScoreFullTable,
    rt::Tensor& draftParentFullTable, int32_t const draftTopK, int32_t const round, cudaStream_t stream)
{
    check::check(draftIdTable.getDeviceType() == rt::DeviceType::kGPU
            && draftScoreTable.getDeviceType() == rt::DeviceType::kGPU
            && intermediateParents.getDeviceType() == rt::DeviceType::kGPU
            && draftIdFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftScoreFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftParentFullTable.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftIdTable.getDataType() == DataType::kINT32 && draftScoreTable.getDataType() == DataType::kFLOAT
            && intermediateParents.getDataType() == DataType::kINT32
            && draftIdFullTable.getDataType() == DataType::kINT32
            && draftScoreFullTable.getDataType() == DataType::kFLOAT
            && draftParentFullTable.getDataType() == DataType::kINT32,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(draftIdTable.getShape()[0]);
    int32_t const fullTableLength = static_cast<int32_t>(draftIdFullTable.getShape()[1]);

    check::check(draftIdTable.getShape()[1] == draftTopK * draftTopK, "Check draft id table dimension.");
    check::check(draftScoreTable.getShape()[1] == draftTopK * draftTopK, "Check draft score table dimension.");
    check::check(intermediateParents.getShape()[1] == draftTopK, "Check intermediate parent dimension.");

    dim3 blockDim{128};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};
    updateDraftTreeFullTablesKernel<<<gridDim, blockDim, 0, stream>>>(draftIdTable.dataPointer<int32_t>(),
        draftScoreTable.dataPointer<float>(), intermediateParents.dataPointer<int32_t>(),
        draftIdFullTable.dataPointer<int32_t>(), draftScoreFullTable.dataPointer<float>(),
        draftParentFullTable.dataPointer<int32_t>(), draftTopK, round, fullTableLength);
}

void buildSpecVerificationTree(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftParentFullTable,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftTreeMask,
    rt::OptionalOutputTensor const& parentIds, rt::OptionalOutputTensor const& depths, cudaStream_t stream)
{
    check::check(draftIdFullTable.getDeviceType() == rt::DeviceType::kGPU
            && draftParentFullTable.getDeviceType() == rt::DeviceType::kGPU
            && selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && inputIds.getDeviceType() == rt::DeviceType::kGPU
            && draftTreeMask.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(draftIdFullTable.getDataType() == DataType::kINT32
            && draftParentFullTable.getDataType() == DataType::kINT32
            && selectedIndices.getDataType() == DataType::kINT32 && inputIds.getDataType() == DataType::kINT32
            && draftTreeMask.getDataType() == DataType::kINT8,
        "Data type shall all be valid.");
    int32_t const batchSize = static_cast<int32_t>(draftIdFullTable.getShape()[0]);
    int32_t const fullTableLength = static_cast<int32_t>(draftIdFullTable.getShape()[1]);
    int32_t const verifyTreeSize = static_cast<int32_t>(selectedIndices.getShape()[1]);

    check::check(verifyTreeSize <= kMAX_SPEC_VERIFY_TREE_TOKENS, "Verify tree size exceeds the kernel's node limit.");

    dim3 blockDim{static_cast<uint32_t>(kMAX_SPEC_VERIFY_TREE_TOKENS)};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};
    buildSpecVerificationTreeKernel<<<gridDim, blockDim, 0, stream>>>(draftIdFullTable.dataPointer<int32_t>(),
        draftParentFullTable.dataPointer<int32_t>(), selectedIndices.dataPointer<int32_t>(),
        inputIds.dataPointer<int32_t>(), draftTreeMask.dataPointer<int8_t>(),
        parentIds.has_value() ? parentIds->get().dataPointer<int32_t>() : nullptr,
        depths.has_value() ? depths->get().dataPointer<int32_t>() : nullptr, fullTableLength, verifyTreeSize);
}

constexpr int32_t kTop1BlocksPerRow{8};

// Internal workspace structure for memory management (similar to SamplingWorkspace)
struct SpecTargetSelectionWorkspace
{
    // Device pointer, owned by external caller - must remain valid for kernel lifetime
    void* ptr;
    // Size in bytes of the workspace buffer
    size_t size;

    // Device pointer to top-1 tokens buffer, owned by external caller
    // Array size: batchSize * numTokens elements (int32_t each)
    int32_t* top1Tokens;

    // Per-lane top-1 candidates for the two-stage argmax. The layout is
    // [batchSize * numTokens, kTop1BlocksPerRow].
    int32_t* top1TempIndices;
    float* top1TempValues;

    SpecTargetSelectionWorkspace()
        : ptr(nullptr)
        , size(0)
        , top1Tokens(nullptr)
        , top1TempIndices(nullptr)
        , top1TempValues(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, int32_t batchSize, int32_t numTokens)
    {
        // Check that workspace is aligned to 256 bytes for optimal GPU memory access
        ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kSpeculativeWorkspaceAlignment == 0,
            "Workspace must be aligned to 256 bytes");

        ptr = workspace;
        size = workspaceSize;

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        int32_t const totalRows = batchSize * numTokens;

        // Top-1 tokens buffer
        size_t top1TokensSize = alignSpeculativeWorkspaceSize(totalRows * sizeof(int32_t));
        top1Tokens = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TokensSize;

        size_t top1TempIndicesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(int32_t));
        top1TempIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TempIndicesSize;

        size_t top1TempValuesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(float));
        top1TempValues = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
        offset += top1TempValuesSize;

        // Validate workspace size
        ELLM_CHECK(offset <= workspaceSize,
            "Spec target selection workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
    }
};

// Calculate workspace size for Eagle accept algorithm
size_t getSpecTargetSelectionWorkspaceSize(int32_t batchSize, int32_t numTokens)
{
    int32_t const totalRows = batchSize * numTokens;
    size_t const top1TokensSize = alignSpeculativeWorkspaceSize(totalRows * sizeof(int32_t));
    size_t const top1TempIndicesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(int32_t));
    size_t const top1TempValuesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(float));
    return top1TokensSize + top1TempIndicesSize + top1TempValuesSize;
}

namespace
{

// Helper structure for top-1 selection
struct Top1Helper
{
    float value;
    int32_t index;

    __device__ __forceinline__ Top1Helper()
        : value(-FLT_MAX)
        , index(-1)
    {
    }

    __device__ __forceinline__ void update(float elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

// Reduction operator for top-1 selection
struct top1MaxOpFunctor
{
    __device__ __forceinline__ Top1Helper operator()(Top1Helper const& a, Top1Helper const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Stage 1: compute a per-lane top-1 for each verify row. The lane
// partitioning intentionally mirrors sampler::selectAllTopK(topK=1) so greedy
// speculative accept resolves exact-logit ties the same way as vanilla decode.
template <int32_t BLOCK_SIZE, int32_t BLOCKS_PER_ROW>
__global__ void eagleComputeTop1Stage1Kernel(float const* logits, int32_t* top1TempIndices, float* top1TempValues,
    int32_t batchSize, int32_t numTokens, int32_t vocabSize)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const rowIdx = static_cast<int32_t>(blockIdx.x / BLOCKS_PER_ROW);
    auto const blockLane = static_cast<int32_t>(blockIdx.x % BLOCKS_PER_ROW);
    int32_t const totalRows = batchSize * numTokens;

    if (rowIdx >= totalRows)
    {
        return;
    }

    float const* rowLogits = logits + static_cast<int64_t>(rowIdx) * vocabSize;
    Top1Helper partial;

    for (int32_t v = tid + blockLane * BLOCK_SIZE; v < vocabSize; v += BLOCK_SIZE * BLOCKS_PER_ROW)
    {
        partial.update(rowLogits[v], v);
    }

    Top1Helper laneMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    if (tid == 0)
    {
        int32_t const tempOffset = rowIdx * BLOCKS_PER_ROW + blockLane;
        top1TempIndices[tempOffset] = laneMax.index;
        top1TempValues[tempOffset] = laneMax.value;
    }
}

// Stage 2: reduce the lane candidates to the final top-1 token. The tie order
// matches the sampler top-k stage: the reduction compares values only.
template <int32_t BLOCK_SIZE, int32_t BLOCKS_PER_ROW>
__global__ void eagleComputeTop1Stage2Kernel(int32_t const* top1TempIndices, float const* top1TempValues,
    int32_t* top1Tokens, int32_t const* vocabMappingTable, int32_t totalRows)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const rowIdx = static_cast<int32_t>(blockIdx.x);

    if (rowIdx >= totalRows)
    {
        return;
    }

    Top1Helper partial;
    for (int32_t i = tid; i < BLOCKS_PER_ROW; i += BLOCK_SIZE)
    {
        int32_t const tempOffset = rowIdx * BLOCKS_PER_ROW + i;
        partial.update(top1TempValues[tempOffset], top1TempIndices[tempOffset]);
    }

    Top1Helper rowMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    if (tid == 0)
    {
        int32_t selectedIdx = (rowMax.index != -1) ? rowMax.index : 0;
        if (vocabMappingTable != nullptr)
        {
            selectedIdx = vocabMappingTable[selectedIdx];
        }
        top1Tokens[rowIdx] = selectedIdx;
    }
}

int32_t* launchTargetSelection(float const* logits, int32_t const* vocabMappingTable, int32_t batchSize,
    int32_t numTokens, int32_t vocabSize, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;

    // Setup workspace partitioning
    SpecTargetSelectionWorkspace ws;
    ws.setupWorkspace(workspace, workspaceSize, batchSize, numTokens);

    // Validate workspace buffer
    assert(ws.top1Tokens != nullptr);

    int32_t const totalRows = batchSize * numTokens;

    // Stage 1/2: Compute top-1 tokens for all positions. The two-stage
    // partitioning matches selectAllTopK(topK=1), which keeps greedy accept
    // deterministic against the vanilla decoder when logits tie exactly.
    dim3 const gridSizeStage1(totalRows * kTop1BlocksPerRow);
    dim3 const blockSizeStage1(blockSize);
    eagleComputeTop1Stage1Kernel<blockSize, kTop1BlocksPerRow><<<gridSizeStage1, blockSizeStage1, 0, stream>>>(
        logits, ws.top1TempIndices, ws.top1TempValues, batchSize, numTokens, vocabSize);

    dim3 const gridSizeStage2Top1(totalRows);
    dim3 const blockSizeStage2Top1(blockSize);
    eagleComputeTop1Stage2Kernel<blockSize, kTop1BlocksPerRow><<<gridSizeStage2Top1, blockSizeStage2Top1, 0, stream>>>(
        ws.top1TempIndices, ws.top1TempValues, ws.top1Tokens, vocabMappingTable, totalRows);
    CUDA_CHECK(cudaGetLastError());

    return ws.top1Tokens;
}

} // namespace

rt::Tensor selectSpecTargetTokens(rt::Tensor const& logits, rt::OptionalInputTensor const& vocabMappingTable,
    rt::Tensor& workspace, int32_t batchSize, int32_t numTokens, cudaStream_t stream)
{
    ELLM_CHECK(batchSize > 0 && numTokens > 0 && batchSize <= std::numeric_limits<int32_t>::max() / numTokens,
        "Invalid target selection dimensions");
    auto const shape = logits.getShape();
    ELLM_CHECK(shape.getNumDims() == 2 && shape[0] == batchSize * numTokens && shape[1] > 0
            && shape[1] <= std::numeric_limits<int32_t>::max(),
        "Target logits must have shape [N*Q,V]");
    ELLM_CHECK(logits.getDeviceType() == rt::DeviceType::kGPU && logits.getDataType() == trt_edgellm::DataType::kFLOAT,
        "Target selection requires GPU FP32 logits");
    ELLM_CHECK(workspace.getDeviceType() == rt::DeviceType::kGPU, "Target selection requires GPU workspace");
    int32_t const* map = nullptr;
    if (vocabMappingTable.has_value())
    {
        auto const& mapping = vocabMappingTable->get();
        ELLM_CHECK(mapping.getDeviceType() == rt::DeviceType::kGPU
                && mapping.getDataType() == trt_edgellm::DataType::kINT32 && mapping.getShape().getNumDims() == 1
                && mapping.getShape()[0] == shape[1],
            "Target vocabulary mapping must be GPU INT32 [V]");
        map = mapping.dataPointer<int32_t>();
    }
    int32_t* selected = launchTargetSelection(logits.dataPointer<float>(), map, batchSize, numTokens,
        static_cast<int32_t>(shape[1]), workspace.rawPointer(), workspace.getMemoryCapacity(), stream);
    return rt::Tensor(selected, {batchSize * numTokens}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
}

namespace
{
template <bool Tree>
__global__ void acceptGreedy(int32_t const* predictions, int32_t const* tokens, int32_t const* parents,
    int32_t const* validCounts, int8_t const* forcedFlags, int32_t* output, int32_t* counts, int32_t* rows,
    int32_t width, int32_t capacity)
{
    int32_t const request = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int64_t const queryOffset = static_cast<int64_t>(request) * width;
    int64_t const outputOffset = static_cast<int64_t>(request) * capacity;
    int32_t const valid = validCounts == nullptr ? width : min(width, validCounts[request]);
    for (int32_t i = tid; i < capacity; i += blockDim.x)
    {
        output[outputOffset + i] = 0;
        if (rows != nullptr)
        {
            rows[outputOffset + i] = -1;
        }
    }
    if (tid == 0)
    {
        counts[request] = 0;
    }
    __syncthreads();
    int32_t node = 0;
    for (int32_t emitted = 0; emitted < capacity && node < valid; ++emitted)
    {
        int32_t const token = predictions[queryOffset + node];
        int32_t next = valid;
        bool forced = false;
        if constexpr (Tree)
        {
            __shared__ int32_t child;
            __shared__ int32_t forcedChild;
            // All threads must finish reading the prior children before they are reset.
            __syncthreads();
            if (tid == 0)
            {
                child = valid;
                forcedChild = valid;
            }
            __syncthreads();
            for (int32_t candidate = 1 + tid; candidate < valid; candidate += blockDim.x)
            {
                if (parents[queryOffset + candidate] != node)
                {
                    continue;
                }
                if (forcedFlags != nullptr && forcedFlags[queryOffset + candidate] != 0)
                {
                    atomicMin(&forcedChild, candidate);
                }
                else if (tokens[queryOffset + candidate] == token)
                {
                    atomicMin(&child, candidate);
                }
            }
            __syncthreads();
            forced = forcedChild < valid;
            next = forced ? forcedChild : child;
        }
        else
        {
            next = node + 1;
            forced = next < valid && forcedFlags != nullptr && forcedFlags[queryOffset + next] != 0;
            if (next < valid && !forced && tokens[queryOffset + next] != token)
            {
                next = valid;
            }
        }
        if (tid == 0)
        {
            // A forced child replaces the prediction; the row stays the parent that produced the step.
            output[outputOffset + emitted] = forced ? tokens[queryOffset + next] : token;
            if (rows != nullptr)
            {
                rows[outputOffset + emitted] = node;
            }
            counts[request] = emitted + 1;
        }
        node = next;
    }
}

template <bool Tree>
__global__ void forceGreedy(int32_t const* predictions, int32_t const* tokens, int32_t const* parents,
    int32_t const* validCounts, int32_t* output, int32_t* counts, int32_t* rows, int32_t width, int32_t capacity,
    int32_t forcedAcceptanceLength)
{
    int32_t const request = blockIdx.x;
    int64_t const queryOffset = static_cast<int64_t>(request) * width;
    int64_t const outputOffset = static_cast<int64_t>(request) * capacity;
    int32_t const valid = validCounts == nullptr ? width : min(width, validCounts[request]);
    int32_t const drafts = forcedAcceptanceLength - 1;
    bool pathValid = forcedAcceptanceLength <= valid;

    for (int32_t i = 0; i < capacity; ++i)
    {
        output[outputOffset + i] = 0;
        if (rows != nullptr)
        {
            rows[outputOffset + i] = -1;
        }
    }

    if constexpr (Tree)
    {
        int32_t path[kSpecMaxAcceptedPathLength];
        int32_t endpoint = -1;
        for (int32_t candidate = 0; pathValid && candidate < valid; ++candidate)
        {
            int32_t node = candidate;
            int32_t depth = 0;
            while (node > 0 && node < valid && depth < kSpecMaxAcceptedPathLength)
            {
                node = parents[queryOffset + node];
                ++depth;
            }
            if (node == 0 && depth == drafts)
            {
                endpoint = candidate;
                break;
            }
        }
        pathValid = pathValid && endpoint >= 0;
        if (pathValid)
        {
            int32_t node = endpoint;
            for (int32_t depth = drafts; depth >= 0; --depth)
            {
                path[depth] = node;
                node = depth > 0 ? parents[queryOffset + node] : 0;
            }
            for (int32_t emitted = 0; emitted < drafts; ++emitted)
            {
                output[outputOffset + emitted] = tokens[queryOffset + path[emitted + 1]];
                rows[outputOffset + emitted] = path[emitted];
            }
            output[outputOffset + drafts] = predictions[queryOffset + path[drafts]];
            rows[outputOffset + drafts] = path[drafts];
        }
    }
    else if (pathValid)
    {
        for (int32_t emitted = 0; emitted < drafts; ++emitted)
        {
            output[outputOffset + emitted] = tokens[queryOffset + emitted + 1];
            if (rows != nullptr)
            {
                rows[outputOffset + emitted] = emitted;
            }
        }
        output[outputOffset + drafts] = predictions[queryOffset + drafts];
        if (rows != nullptr)
        {
            rows[outputOffset + drafts] = drafts;
        }
    }

    if (pathValid)
    {
        counts[request] = forcedAcceptanceLength;
    }
    else
    {
        output[outputOffset] = predictions[queryOffset];
        if (rows != nullptr)
        {
            rows[outputOffset] = 0;
        }
        counts[request] = 1;
        if (capacity > 1)
        {
            output[outputOffset + capacity - 1] = -1;
        }
    }
}

void validateTensor(rt::Tensor const& tensor, int64_t elements)
{
    ELLM_CHECK(tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.getDataType() == trt_edgellm::DataType::kINT32
            && tensor.getShape().volume() == elements && (elements == 0 || tensor.rawPointer() != nullptr),
        "Greedy acceptance requires GPU INT32 tensors with the specified physical extent");
}

template <bool Tree>
void launch(rt::Tensor const& target, rt::Tensor const& tokens, rt::Tensor const* parents, rt::Tensor const* valid,
    rt::Tensor const* forcedFlags, rt::Tensor& output, rt::Tensor& counts, rt::Tensor* rows, int32_t requests,
    int32_t width, int32_t capacity, cudaStream_t stream, int32_t forcedAcceptanceLength)
{
    ELLM_CHECK(requests > 0 && width > 0 && capacity >= 0, "Invalid greedy acceptance dimensions");
    ELLM_CHECK(forcedAcceptanceLength == -1
            || (forcedAcceptanceLength > 0 && forcedAcceptanceLength <= capacity
                && forcedAcceptanceLength <= kSpecMaxAcceptedPathLength),
        "Forced acceptance length must fit the accepted path capacity");
    int64_t const queryElements = static_cast<int64_t>(requests) * width;
    int64_t const outputElements = static_cast<int64_t>(requests) * capacity;
    validateTensor(target, queryElements);
    validateTensor(tokens, queryElements);
    validateTensor(output, outputElements);
    validateTensor(counts, requests);
    if (valid != nullptr)
    {
        validateTensor(*valid, requests);
    }
    if (rows != nullptr)
    {
        validateTensor(*rows, outputElements);
    }
    if (forcedFlags != nullptr)
    {
        ELLM_CHECK(forcedFlags->getDeviceType() == rt::DeviceType::kGPU
                && forcedFlags->getDataType() == trt_edgellm::DataType::kINT8
                && forcedFlags->getShape().volume() == queryElements && forcedFlags->rawPointer() != nullptr,
            "Forced flags must be a GPU INT8 tensor with the verification extent");
    }
    if constexpr (Tree)
    {
        ELLM_CHECK(parents != nullptr && rows != nullptr, "Tree acceptance requires parents and predictor rows");
        validateTensor(*parents, queryElements);
    }
    if (forcedAcceptanceLength > 0)
    {
        forceGreedy<Tree><<<requests, 1, 0, stream>>>(target.dataPointer<int32_t>(), tokens.dataPointer<int32_t>(),
            parents == nullptr ? nullptr : parents->dataPointer<int32_t>(),
            valid == nullptr ? nullptr : valid->dataPointer<int32_t>(), output.dataPointer<int32_t>(),
            counts.dataPointer<int32_t>(), rows == nullptr ? nullptr : rows->dataPointer<int32_t>(), width, capacity,
            forcedAcceptanceLength);
    }
    else
    {
        acceptGreedy<Tree><<<requests, Tree ? 256 : 1, 0, stream>>>(target.dataPointer<int32_t>(),
            tokens.dataPointer<int32_t>(), parents == nullptr ? nullptr : parents->dataPointer<int32_t>(),
            valid == nullptr ? nullptr : valid->dataPointer<int32_t>(),
            forcedFlags == nullptr ? nullptr : forcedFlags->dataPointer<int8_t>(), output.dataPointer<int32_t>(),
            counts.dataPointer<int32_t>(), rows == nullptr ? nullptr : rows->dataPointer<int32_t>(), width, capacity);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void acceptSpecGreedyChain(rt::Tensor const& target, rt::Tensor const& tokens, rt::Tensor const* valid,
    rt::Tensor const* forcedFlags, rt::Tensor& output, rt::Tensor& counts, rt::Tensor* rows, int32_t requests,
    int32_t width, int32_t capacity, cudaStream_t stream, int32_t forcedAcceptanceLength)
{
    launch<false>(target, tokens, nullptr, valid, forcedFlags, output, counts, rows, requests, width, capacity, stream,
        forcedAcceptanceLength);
}

void acceptSpecGreedyTree(rt::Tensor const& target, rt::Tensor const& tokens, rt::Tensor const& parents,
    rt::Tensor const* valid, rt::Tensor const* forcedFlags, rt::Tensor& output, rt::Tensor& counts, rt::Tensor* rows,
    int32_t requests, int32_t width, int32_t capacity, cudaStream_t stream, int32_t forcedAcceptanceLength)
{
    launch<true>(target, tokens, &parents, valid, forcedFlags, output, counts, rows, requests, width, capacity, stream,
        forcedAcceptanceLength);
}

// Resolves logical token position `pos` through the required page table to a flat element
// offset in the layer's [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] pool.
// `kvSel` is 0 for K and 1 for V. Unmapped or out-of-plane page ids return -1.
__device__ __forceinline__ int64_t resolveSpecKVOffset(int32_t kvSel, int32_t headInKv, int32_t numKVHeads,
    int32_t headDim, int32_t kvBatchIdx, int32_t pos, int32_t const* pageTable, int32_t numPages,
    int32_t maxPagesPerSeq, int32_t vecElemOffset)
{
    int64_t const tokenStride = static_cast<int64_t>(numKVHeads) * headDim;
    int32_t const pageRow = pos / rt::kTOKENS_PER_PAGE;
    int32_t const inPage = pos % rt::kTOKENS_PER_PAGE;
    int32_t const pageId = pageTable[(static_cast<int64_t>(kvBatchIdx) * 2 + kvSel) * maxPagesPerSeq + pageRow];
    int64_t const planeBegin = static_cast<int64_t>(kvSel) * numPages;
    int64_t const planeEnd = planeBegin + numPages;
    if (pageId < planeBegin || static_cast<int64_t>(pageId) >= planeEnd)
    {
        return -1;
    }
    return (static_cast<int64_t>(pageId) * rt::kTOKENS_PER_PAGE + inPage) * tokenStride
        + static_cast<int64_t>(headInKv) * headDim + vecElemOffset;
}

template <int32_t MAX_SPEC_PATH, typename KV_T>
__global__ void commitSpecTargetKVBatchedKernel(int32_t const* __restrict__ acceptedIndices,
    int32_t const* __restrict__ acceptLengths, int32_t const* __restrict__ kvCacheLengths,
    int32_t const* __restrict__ stateIndices, KVLayerInfo const* __restrict__ layerInfos, int32_t const activeBatchSize,
    int32_t const residentPoolRows, int32_t const maxDepth, int32_t const* __restrict__ pageTable,
    int32_t const numPages, int32_t const maxPagesPerSeq, int32_t const headDim)
{
    // NOT blanket-initialised: zeroing all MAX_SPEC_PATH vectors up front measurably slows this
    // us-scale kernel (Thor perf gate). Instead, every phase-1 path that skips the load writes an
    // explicit zero below, so phase 2 always stores deterministic bytes.
    DVec<KV_T> tempBuffer[MAX_SPEC_PATH];

    // Per-layer batched commit. One launch covers all layers in a head-dim group.
    //
    // Grid:  (blocksPerLayer, numLayers)
    //          blockIdx.y selects the layer; per-layer numKVHeads / maxSeqLen are read
    //          from the device-side KVLayerInfo array. blocksPerLayer is sized by the
    //          group's maxKVHeads, so layers with fewer KV heads early-exit on extra CTAs.
    // Block: (headDim / DVec<half>::vec_size, headPerBlock): one vector per thread, whole heads per CTA.
    //
    // Each layer's buffer is a paged pool addressed through `layerInfos[layerIdx].data`.

    int32_t const layerIdx = blockIdx.y;
    KVLayerInfo const info = layerInfos[layerIdx];
    int32_t const numKVHeads = info.numKVHeads;
    int32_t const maxSeqLen = info.maxSeqLen; // capPadded

    int32_t const tIdx = threadIdx.x;
    int32_t const tIdy = threadIdx.y;
    int32_t const bIdx = blockIdx.x;
    int32_t const headIdx = bIdx * blockDim.y + tIdy;

    if (headIdx >= activeBatchSize * 2 * numKVHeads)
    {
        return;
    }

    int32_t const kvBatchIdx = headIdx / (2 * numKVHeads);
    int32_t const residentBatchIdx = stateIndices[kvBatchIdx];
    if (residentBatchIdx < 0 || residentBatchIdx >= residentPoolRows)
    {
        return;
    }
    int32_t const kvHeadIdx = headIdx % (2 * numKVHeads);
    int32_t const kvSel = kvHeadIdx / numKVHeads; // 0 = K-half, 1 = V-half
    int32_t const headInKv = kvHeadIdx % numKVHeads;

    int32_t const actualAcceptLength = max(0, min(acceptLengths[kvBatchIdx], min(maxDepth, MAX_SPEC_PATH)));
    int32_t const pastKvCacheLength = kvCacheLengths[kvBatchIdx];

    KV_T* kvCacheBuffer = static_cast<KV_T*>(info.data);

    // PHASE 1: Collect all accepted data into local temp buffer
    // Start from 1 since the root position will always be accepted.
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const acceptedIdx = acceptedIndices[kvBatchIdx * maxDepth + i];
        if (acceptedIdx >= 0 && acceptedIdx + pastKvCacheLength < maxSeqLen)
        {
            int64_t const srcOffset = resolveSpecKVOffset(kvSel, headInKv, numKVHeads, headDim, residentBatchIdx,
                pastKvCacheLength + acceptedIdx, pageTable, numPages, maxPagesPerSeq, tIdx * DVec<half>::vec_size);
            if (srcOffset >= 0)
            {
                tempBuffer[i].load(kvCacheBuffer + srcOffset);
            }
            else
            {
                // An unmapped or out-of-plane source resolves to deterministic zeros.
                tempBuffer[i] = DVec<KV_T>{};
            }
        }
        else
        {
            // Out-of-range accepted index: nothing to collect; keep the phase-2 store (if its
            // destination is mapped) deterministic.
            tempBuffer[i] = DVec<KV_T>{};
        }
    }

    // PHASE 2: Write from local temp buffer to final positions
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int64_t const dstOffset = resolveSpecKVOffset(kvSel, headInKv, numKVHeads, headDim, residentBatchIdx,
            pastKvCacheLength + i, pageTable, numPages, maxPagesPerSeq, tIdx * DVec<half>::vec_size);
        if (dstOffset >= 0)
        {
            tempBuffer[i].store(kvCacheBuffer + dstOffset);
        }
        // An unmapped or out-of-plane destination skips the write.
    }
}

template <int32_t MAX_SPEC_PATH>
__global__ void gatherSpecTargetHiddenStatesKernel(int32_t const* acceptedIndices, int32_t const* acceptLengths,
    half* hiddenState, int32_t const batchSize, int32_t const maxDepth, int32_t const numTokens,
    int32_t const hiddenDim)
{
    DVec<half> tempBuffer[MAX_SPEC_PATH];

    int32_t const dimIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const startIdx = dimIdx * DVec<half>::vec_size;

    if (startIdx >= hiddenDim)
    {
        return;
    }

    // Dense in-place output for batch b overlaps the input slab of earlier batches.
    // A single CTA owns each hidden-dimension tile and visits batches in order, so it
    // always consumes an earlier slab before a later batch can overwrite that tile.
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const actualAcceptLength = max(0, min(acceptLengths[batchIdx], maxDepth));
        int32_t const inputOffset = batchIdx * numTokens * hiddenDim;
        int32_t const outputOffset = batchIdx * maxDepth * hiddenDim;

        for (int32_t i = 0; i < actualAcceptLength; ++i)
        {
            int32_t const acceptedIdx = acceptedIndices[batchIdx * maxDepth + i];
            if (acceptedIdx >= 0 && acceptedIdx < numTokens)
            {
                int32_t const srcOffset = inputOffset + acceptedIdx * hiddenDim + startIdx;
                tempBuffer[i].load(hiddenState + srcOffset);
            }
            else
            {
                tempBuffer[i] = DVec<half>{};
            }
        }
        for (int32_t i = 0; i < actualAcceptLength; ++i)
        {
            int32_t const dstOffset = outputOffset + i * hiddenDim + startIdx;
            tempBuffer[i].store(hiddenState + dstOffset);
        }
    }
}

void commitSpecTargetKV(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, rt::Tensor const& stateIndices, KVLayerInfo const* deviceLayerInfos,
    int32_t numLayers, int32_t headDim, int32_t maxKVHeads, int32_t activeBatchSize, int32_t residentPoolRows,
    int32_t maxDepth, trt_edgellm::DataType kvCacheType, cudaStream_t stream, int32_t const* pageTable,
    int32_t numPages, int32_t maxPagesPerSeq)
{
    check::check(acceptedIndices.getDeviceType() == rt::DeviceType::kGPU
            && acceptLengths.getDeviceType() == rt::DeviceType::kGPU
            && kvCacheLengths.getDeviceType() == rt::DeviceType::kGPU
            && stateIndices.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(acceptedIndices.getDataType() == DataType::kINT32 && acceptLengths.getDataType() == DataType::kINT32
            && kvCacheLengths.getDataType() == DataType::kINT32 && stateIndices.getDataType() == DataType::kINT32,
        "acceptedIndices, acceptLengths, kvCacheLengths, and stateIndices should be INT32.");
    check::check(kvCacheType == DataType::kHALF || kvCacheType == DataType::kFP8, "kvCacheType should be HALF or FP8.");
    check::check(deviceLayerInfos != nullptr, "deviceLayerInfos must not be null.");
    check::check(pageTable != nullptr, "pageTable must not be null.");
    check::check(numPages > 0, "numPages must be positive.");
    check::check(numPages <= rt::kMAX_KV_POOL_PAGES, "numPages exceeds the supported paged-KV pool limit.");
    check::check(maxPagesPerSeq > 0, "maxPagesPerSeq must be positive.");
    check::check(residentPoolRows > 0, "residentPoolRows must be positive.");

    if (numLayers == 0 || activeBatchSize == 0)
    {
        return;
    }

    auto const acceptIndicesShape = acceptedIndices.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    auto const kvCacheLengthsShape = kvCacheLengths.getShape();
    auto const stateIndicesShape = stateIndices.getShape();
    check::check(acceptIndicesShape.getNumDims() == 2, "acceptedIndices should be 2D tensor [batch, max-depth].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths should be 1D tensor [batch].");
    check::check(kvCacheLengthsShape.getNumDims() == 1, "kvCacheLengths should be 1D tensor [batch].");
    check::check(stateIndicesShape.getNumDims() == 1, "stateIndices should be 1D tensor [batch].");
    check::check(stateIndicesShape[0] == activeBatchSize, "stateIndices length must match activeBatchSize.");
    check::check(
        static_cast<int32_t>(acceptIndicesShape[1]) == maxDepth, "acceptedIndices second dim must match maxDepth.");

    check::check(maxDepth <= kSpecMaxAcceptedPathLength, "maxDepth > 32 is not supported by the kernel.");

    // Each CTA has 128 threads, each thread copies vecSize elements (DVec<half> = 8 elements;
    // DVec<__nv_fp8_e4m3> is also 8 elements wide, so the block-dim math is dtype-agnostic).
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t threadsPerBlock = 128;
    check::check(headDim > 0 && headDim % vecSize == 0 && headDim <= static_cast<int32_t>(threadsPerBlock * vecSize),
        "commitSpecTargetKV needs a head dim that is a multiple of " + std::to_string(vecSize) + " and at most "
            + std::to_string(threadsPerBlock * vecSize) + ", got " + std::to_string(headDim));

    uint32_t const bDimX = headDim / vecSize;
    uint32_t const headPerBlock = threadsPerBlock * vecSize / headDim;
    uint32_t const headsPerLayer = static_cast<uint32_t>(activeBatchSize) * 2u * static_cast<uint32_t>(maxKVHeads);
    uint32_t const blocksPerLayer = (headsPerLayer + headPerBlock - 1) / headPerBlock;

    dim3 const blockDim1(bDimX, headPerBlock);
    dim3 const gridDim1(blocksPerLayer, static_cast<uint32_t>(numLayers));

    int32_t const* acceptedIndicesPtr = acceptedIndices.dataPointer<int32_t>();
    int32_t const* acceptLengthsPtr = acceptLengths.dataPointer<int32_t>();
    int32_t const* kvCacheLengthsPtr = kvCacheLengths.dataPointer<int32_t>();
    int32_t const* stateIndicesPtr = stateIndices.dataPointer<int32_t>();

    if (kvCacheType == DataType::kHALF)
    {
        commitSpecTargetKVBatchedKernel<kSpecMaxAcceptedPathLength, half><<<gridDim1, blockDim1, 0, stream>>>(
            acceptedIndicesPtr, acceptLengthsPtr, kvCacheLengthsPtr, stateIndicesPtr, deviceLayerInfos, activeBatchSize,
            residentPoolRows, maxDepth, pageTable, numPages, maxPagesPerSeq, headDim);
    }
    else
    {
#if SUPPORTS_FP8
        commitSpecTargetKVBatchedKernel<kSpecMaxAcceptedPathLength, __nv_fp8_e4m3><<<gridDim1, blockDim1, 0, stream>>>(
            acceptedIndicesPtr, acceptLengthsPtr, kvCacheLengthsPtr, stateIndicesPtr, deviceLayerInfos, activeBatchSize,
            residentPoolRows, maxDepth, pageTable, numPages, maxPagesPerSeq, headDim);
#else
        throw std::runtime_error("FP8 KV cache requested but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
#endif
    }
    CUDA_CHECK(cudaGetLastError());
}

void gatherSpecTargetHiddenStates(
    rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths, rt::Tensor& hiddenState, cudaStream_t stream)
{
    check::check(acceptedIndices.getDeviceType() == rt::DeviceType::kGPU
            && acceptLengths.getDeviceType() == rt::DeviceType::kGPU
            && hiddenState.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(acceptedIndices.getDataType() == DataType::kINT32 && acceptLengths.getDataType() == DataType::kINT32
            && hiddenState.getDataType() == DataType::kHALF,
        "acceptedIndices and acceptLengths should be INT32; hiddenState should be HALF.");

    auto const acceptIndicesShape = acceptedIndices.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    auto const hiddenStateShape = hiddenState.getShape();
    check::check(acceptIndicesShape.getNumDims() == 2, "acceptedIndices should be 2D tensor [batch, max-depth].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths should be 1D tensor [batch].");
    check::check(
        hiddenStateShape.getNumDims() == 3, "hiddenState should be 3D tensor [batch, num-tokens, base-hidden-dim].");

    uint32_t const batchSize = acceptIndicesShape[0];
    int32_t const maxDepth = acceptIndicesShape[1];
    check::check(acceptLengthsShape[0] == batchSize, "acceptLengths should have same batch size as acceptedIndices.");
    check::check(hiddenStateShape[0] == batchSize, "hiddenState batch size should match acceptedIndices.");

    check::check(maxDepth <= kSpecMaxAcceptedPathLength, "maxDepth > 32 is not supported by the kernel.");

    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t threadsPerBlock = 128;
    int32_t const numTokens = hiddenStateShape[1];
    int32_t const hiddenDim = hiddenStateShape[2];
    check::check(maxDepth <= numTokens, "maxDepth must not exceed the number of verify tokens.");
    check::check(hiddenDim % vecSize == 0, "hiddenDim must be divisible by vecSize.");

    uint32_t const dimPerBlock = threadsPerBlock * vecSize;
    uint32_t const gridX = (hiddenDim + dimPerBlock - 1) / dimPerBlock;
    dim3 const blockDim2(threadsPerBlock);
    dim3 const gridDim2{gridX};

    gatherSpecTargetHiddenStatesKernel<kSpecMaxAcceptedPathLength><<<gridDim2, blockDim2, 0, stream>>>(
        acceptedIndices.dataPointer<int32_t>(), acceptLengths.dataPointer<int32_t>(), hiddenState.dataPointer<half>(),
        batchSize, maxDepth, numTokens, hiddenDim);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace trt_edgellm::kernel
