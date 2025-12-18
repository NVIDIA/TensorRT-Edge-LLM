/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "eagleUtilKernels.h"
#include "kernels/common/vectorizedTypes.cuh"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include <cmath>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kROOT_NODE_PREDECESSOR{-1};
constexpr int32_t kEMPTY_NODE_PREDECESSOR{-5};

__global__ void prepareEaglePrefillInputKernel(int64_t* selectTokenIndices, int32_t const* sequenceContextLengths)
{
    int32_t const batchIdx = blockIdx.x;
    if (threadIdx.x == 0)
    {
        int32_t const sequenceLength = sequenceContextLengths[batchIdx];
        selectTokenIndices[batchIdx] = sequenceLength - 1;
    }
}

__global__ void prepareEagleDraftProposalMiscInputKernel(int32_t const* draftTreeSizes,
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
        selectTokenIndices[batchIdx * selectTokenLength + i] = draftTreeSize - selectTokenLength + i;
    }
}

__global__ void assembleDraftTreeDescKernel(int8_t const* draftTreeMask, int32_t const* draftTreeSizes,
    int32_t const* sequenceStartIndices, int32_t* packedDraftTreeMask, int32_t* tensorPositionIndices,
    int32_t const paddedDraftTreeSize)
{
    // Supports up to draft tree size of 128 {4 x 32}, should be sufficient for now.
    constexpr int32_t kNUM_MASK_PER_ENTRY{32};
    constexpr int32_t kMAX_DRAFT_PACKED_TREE_SIZE{4};

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

    // 32 should be sufficient for accepted tokens from base model.
    int32_t packedTreeMask{0};
    int32_t const packedTreeMaskOffset = batchIdx * maxAcceptedTokenNum + tokenIdx;

    if (tokenIdx < acceptedTokenNum)
    {
        // Valid tokens: set causal mask and valid position
        for (int32_t i = 0; i <= tokenIdx; ++i)
        {
            packedTreeMask |= (1 << i);
        }

        // Packed tree mask shall have layout of [batch, max-accepted-token-num, divup(max-accepted-token-num, 32)].
        // Here the accepted token num should be strictly smaller than 32.
        // tensor position indices have layout of [batch, max-accepted-token-num], the offset will be identical to
        // packed tree mask.
        packedTreeMasks[packedTreeMaskOffset] = packedTreeMask;
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
        selectTokenIndices[batchIdx] = acceptedTokenNum - 1;
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
            // First level of the draft tree, token shall be translated into full vocab size.
            // Score is 0 (root) + log probability, parent points to root (0)
            int32_t const selectedOffset = batchIdx * draftTopK + i - 1;
            int32_t const draftTokenId = selectedIndices[selectedOffset];
            float const logProbVal = logProbs[selectedOffset];
            int32_t const baseTokenId = draftTokenId + vocabMappingTable[draftTokenId];
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

__global__ void constructVerificationDraftTreeKernel(int32_t const* draftIdFullTable,
    int32_t const* draftParentFullTable, int32_t const* selectedIndices, int32_t* inputIds, int8_t* draftTreeMask,
    int32_t const fullTableLength, int32_t const verifyTreeSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tIdx = threadIdx.x;

    if (tIdx >= verifyTreeSize)
    {
        return;
    }

    // 10 Should be sufficient since we don't have too many levels of drafting.
    // We don't use shared memory since the data tables are small and can automatically fit into L1.
    constexpr int32_t kMAX_DEPTH{10};
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
}

void prepareEaglePrefillInputs(
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
    prepareEaglePrefillInputKernel<<<gridDim, blockDim, 0, stream>>>(
        selectTokenIndices.dataPointer<int64_t>(), sequenceContextLengths.dataPointer<int32_t>());
}

void prepareEagleDraftProposalInputs(rt::Tensor const& draftTreeMask, rt::Tensor const& draftTreeLength,
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
    prepareEagleDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(draftTreeLength.dataPointer<int32_t>(),
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), selectTokenLength, paddedDraftTreeSize);
}

void prepareEagleAcceptDecodeTokenInputs(rt::Tensor const& sequenceStartIndices, rt::Tensor const& acceptedTokenNums,
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
    check::check(maxAcceptedTokenNum < 32,
        "Current kernel implementation support accepted token <= 32 per batch. "
        "Packed tree mask shall have shape [batch, max-accepted-token-num, 1].");
    check::check(acceptedTokenNums.getShape()[0] == batchSize,
        "acceptedTokenNums batch size should match sequenceStartIndices.");

    // Round up block size to multiple of warp size. Use max to handle all batches.
    // Note: This uses maxAcceptedTokenNum for uniform block size across all batches.
    // Threads with threadIdx.x >= acceptedTokenNum (per-batch) will early-exit.
    // Trade-off: Simplifies launch configuration but may launch idle threads for batches
    // with fewer accepted tokens. Since maxAcceptedTokenNum < 32, overhead is minimal.
    uint32_t const blocksize = divUp(maxAcceptedTokenNum, 32) * 32;
    // Perform casual tree mask packing and tensor position indices.
    dim3 const blockDim{blocksize};
    dim3 const gridDim{batchSize};

    assembleCasualTreeAndSelectIndicesKernel<<<gridDim, blockDim, 0, stream>>>(
        sequenceStartIndices.dataPointer<int32_t>(), packedTreeMask.dataPointer<int32_t>(),
        tensorPositionIndices.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(),
        sequenceContextLengths.dataPointer<int32_t>(), acceptedTokenNums.dataPointer<int32_t>(), maxAcceptedTokenNum);
}

void prepareEagleBaseTreeDecodingInputs(rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& sequenceStartIndices,
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
    prepareEagleDraftProposalMiscInputKernel<<<gridDim2, blockDim2, 0, stream>>>(nullptr,
        sequenceStartIndices.dataPointer<int32_t>(), sequenceContextLengths.dataPointer<int32_t>(),
        selectTokenIndices.dataPointer<int64_t>(), treeSize, treeSize);
}

template <int32_t HEAD_DIM, int32_t MAX_PATH>
__global__ void eagleBaseCommitKVCacheKernel(int32_t const* acceptedIndices, int32_t const* acceptLengths,
    int32_t const* kvCacheLengths, half* kvCacheBuffer, int32_t const activeBatchSize, int32_t const maxDepth,
    int32_t const numLayers, int32_t const maxBatchSize, int32_t const numHeads, int32_t const maxSeqLen)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Only HEAD_DIM = 64 or 128 are supported");
    DVec<half> tempBuffer[MAX_PATH];

    // The kernel have assumptions that:
    //     1. Each CTA will handle multiple heads.
    //     2. Each thread will copy 16 bytes of data (half[8]), each warp will copy 512 bytes (half[256]) data per
    //     iteration.
    //     3. Each CTA contains 128 threads (4 warps).
    //         blockDim.x = number of threads used to process 1 head = HEAD_DIM / DVec<half>::vec_size
    //         blockDim.y = number of heads handled by each CTA = 128 / blockDim.x
    //     4. The KVCache buffer has layout of [numLayers, maxBatchSize, 2, numHeads, maxSeqLen, HEAD_DIM]

    int32_t const tIdx = threadIdx.x;
    int32_t const tIdy = threadIdx.y;
    int32_t const bIdx = blockIdx.x;
    int32_t const headIdx = bIdx * blockDim.y + tIdy;

    int32_t const kvLayerIdx = headIdx / (activeBatchSize * 2 * numHeads);
    int32_t const kvBatchIdx = (headIdx % (activeBatchSize * 2 * numHeads)) / (2 * numHeads);
    int32_t const kvHeadIdx = headIdx % (2 * numHeads);

    int32_t const actualAcceptLength = acceptLengths[kvBatchIdx];
    int32_t const pastKvCacheLength = kvCacheLengths[kvBatchIdx];
    int32_t const kvCacheOffset = kvLayerIdx * maxBatchSize * 2 * numHeads * maxSeqLen * HEAD_DIM
        + kvBatchIdx * 2 * numHeads * maxSeqLen * HEAD_DIM + kvHeadIdx * maxSeqLen * HEAD_DIM
        + pastKvCacheLength * HEAD_DIM;

    // PHASE 1: Collect all accepted data into local temp buffer
    // Start from 1 since the root position will always be accepted.
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const acceptedIdx = acceptedIndices[kvBatchIdx * maxDepth + i];
        if (acceptedIdx >= 0 && acceptedIdx + pastKvCacheLength < maxSeqLen)
        {
            int32_t const srcOffset = kvCacheOffset + acceptedIdx * HEAD_DIM + tIdx * DVec<half>::vec_size;
            tempBuffer[i].load(kvCacheBuffer + srcOffset);
        }
    }

    // PHASE 2: Write from local temp buffer to final positions
    for (int32_t i = 1; i < actualAcceptLength; ++i)
    {
        int32_t const dstOffset = kvCacheOffset + i * HEAD_DIM + tIdx * DVec<half>::vec_size;
        tempBuffer[i].store(kvCacheBuffer + dstOffset);
    }
}

template <int32_t MAX_PATH>
__global__ void eagleBaseAssembleHiddenStateKernel(int32_t const* acceptedIndices, int32_t const* acceptLengths,
    half* hiddenState, int32_t const batchSize, int32_t const maxDepth, int32_t const numTokens,
    int32_t const hiddenDim)
{
    DVec<half> tempBuffer[MAX_PATH];

    // The kernel performs INPLACE compaction of accepted tokens within the same buffer.
    // Since maxAcceptDepth (e.g., 7) << numTokens (e.g., 60), we can safely write compacted
    // data at the beginning of each batch's region without overwriting unread data.
    //
    // Assumptions:
    //     1. Each thread copies 16 bytes of data (half[8]), each warp copies 512 bytes (half[256]) per iteration.
    //     2. Each CTA contains 128 threads (4 warps), total of 128*8=1024 elements.
    //         Since hiddenDim can be very large, each CTA handles part of a batch.
    //     3. acceptedIndices has layout [batch, max-depth]
    //     4. hiddenState buffer has layout [batch, num-tokens, hidden-dim]
    //     5. Output will be compacted inplace to [batch, actual-accept-length, hidden-dim]

    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int32_t const startIdx = dimIdx * DVec<half>::vec_size;

    if (startIdx >= hiddenDim)
    {
        return;
    }

    int32_t const actualAcceptLength = acceptLengths[batchIdx];

    // Input uses stride=numTokens (e.g., verifyTreeSize=60)
    int32_t const inputOffset = batchIdx * numTokens * hiddenDim;

    // The accepted token lengths are usually not equal. We will pad the output till
    // maxAcceptDepth instead of making it a true ragged tensor.
    int32_t const outputOffset = batchIdx * maxDepth * hiddenDim;

    // PHASE 1: Collect all accepted tokens from INPUT layout (stride=numTokens)
    // Read from positions scattered in the input space (e.g., [0, 3, 5, 12, ...])
    for (int32_t i = 0; i < actualAcceptLength; ++i)
    {
        int32_t const acceptedIdx = acceptedIndices[batchIdx * maxDepth + i];
        if (acceptedIdx >= 0 && acceptedIdx < numTokens)
        {
            int32_t const srcOffset = inputOffset + acceptedIdx * hiddenDim + startIdx;
            tempBuffer[i].load(hiddenState + srcOffset);
        }
    }

    // PHASE 2: Write to compacted OUTPUT layout (stride=maxDepth)
    // Write to consecutive positions in the compacted space (e.g., [0, 1, 2, 3, ...])
    // This is safe because outputOffset <= inputOffset (since maxDepth < numTokens)
    for (int32_t i = 0; i < actualAcceptLength; ++i)
    {
        int32_t const dstOffset = outputOffset + i * hiddenDim + startIdx;
        tempBuffer[i].store(hiddenState + dstOffset);
    }
}

void eagleBaseCommitKVCacheAndAssembleHiddenState(rt::Tensor const& acceptedIndices, rt::Tensor const& acceptLengths,
    rt::Tensor const& kvCacheLengths, rt::Tensor& kvCacheBuffer, rt::Tensor& hiddenState, cudaStream_t stream)
{
    check::check(acceptedIndices.getDeviceType() == rt::DeviceType::kGPU
            && acceptLengths.getDeviceType() == rt::DeviceType::kGPU
            && kvCacheBuffer.getDeviceType() == rt::DeviceType::kGPU
            && kvCacheLengths.getDeviceType() == rt::DeviceType::kGPU
            && hiddenState.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(acceptedIndices.getDataType() == DataType::kINT32 && acceptLengths.getDataType() == DataType::kINT32
            && kvCacheBuffer.getDataType() == DataType::kHALF && kvCacheLengths.getDataType() == DataType::kINT32
            && hiddenState.getDataType() == DataType::kHALF,
        "Data type validation failed: acceptedIndices, acceptLengths, and kvCacheLengths should be INT32; "
        "kvCacheBuffer and hiddenState should be HALF.");

    auto const acceptIndicesShape = acceptedIndices.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    auto const kvCacheBufferShape = kvCacheBuffer.getShape();
    auto const kvCacheLengthsShape = kvCacheLengths.getShape();
    auto const hiddenStateShape = hiddenState.getShape();
    check::check(acceptIndicesShape.getNumDims() == 2, "acceptedIndices should be 2D tensor [batch, max-depth].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths should be 1D tensor [batch].");
    check::check(kvCacheBufferShape.getNumDims() == 6,
        "kvCacheBuffer should be 6D tensor [num-layers, batch, 2, num-heads, max-seq-len, hidden-size-per-head].");
    check::check(kvCacheLengthsShape.getNumDims() == 1, "kvCacheLengths should be 1D tensor [batch].");
    check::check(
        hiddenStateShape.getNumDims() == 3, "hiddenState should be 3D tensor [batch, num-tokens, base-hidden-dim].");

    uint32_t const batchSize = acceptIndicesShape[0];
    int32_t const maxDepth = acceptIndicesShape[1];
    check::check(acceptLengthsShape[0] == batchSize, "acceptLengths should have same batch size as acceptedIndices.");

    constexpr int32_t MAX_PATH{8};
    check::check(maxDepth <= (MAX_PATH + 1), "maxDepth > 9 is not supported by the kernel.");

    // Each CTA has 128 threads, each thread will handle vecSize elements.
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t threadsPerBlock = 128;

    // Commit KVCache
    int32_t const numLayers = kvCacheBufferShape[0];
    int32_t const maxBatchSize = kvCacheBufferShape[1];
    int32_t const numHeads = kvCacheBufferShape[3];
    int32_t const maxSeqLen = kvCacheBufferShape[4];
    int32_t const headDim = kvCacheBufferShape[5];

    uint32_t const bDimX = headDim / vecSize;
    uint32_t const headPerBlock = threadsPerBlock * vecSize / headDim;
    uint32_t const totalNumHeads = numLayers * batchSize * 2 * numHeads;
    uint32_t const totalNumBlocks = (totalNumHeads + headPerBlock - 1) / headPerBlock;

    dim3 const blockDim1(bDimX, headPerBlock);
    dim3 const gridDim1{totalNumBlocks};

    switch (headDim)
    {
    case 64:
        eagleBaseCommitKVCacheKernel<64, MAX_PATH>
            <<<gridDim1, blockDim1, 0, stream>>>(acceptedIndices.dataPointer<int32_t>(),
                acceptLengths.dataPointer<int32_t>(), kvCacheLengths.dataPointer<int32_t>(),
                kvCacheBuffer.dataPointer<half>(), batchSize, maxDepth, numLayers, maxBatchSize, numHeads, maxSeqLen);
        break;
    case 128:
        eagleBaseCommitKVCacheKernel<128, MAX_PATH>
            <<<gridDim1, blockDim1, 0, stream>>>(acceptedIndices.dataPointer<int32_t>(),
                acceptLengths.dataPointer<int32_t>(), kvCacheLengths.dataPointer<int32_t>(),
                kvCacheBuffer.dataPointer<half>(), batchSize, maxDepth, numLayers, maxBatchSize, numHeads, maxSeqLen);
        break;
    default:
        throw std::runtime_error(
            "Only HEAD_DIM = 64 or 128 are supported by eagleBaseCommitKVCacheAndAssembleHiddenState, current HEAD_DIM "
            "= "
            + std::to_string(headDim));
    }

    // Assemble Hidden State
    check::check(hiddenStateShape[0] == batchSize, "hiddenState batch size should match acceptedIndices.");
    int32_t const numTokens = hiddenStateShape[1];
    int32_t const hiddenDim = hiddenStateShape[2];
    check::check(hiddenDim % vecSize == 0, "hiddenDim must be divisible by vecSize.");

    uint32_t const dimPerBlock = threadsPerBlock * vecSize;
    uint32_t const gridY = (hiddenDim + dimPerBlock - 1) / dimPerBlock;
    dim3 const blockDim2(threadsPerBlock);
    dim3 const gridDim2{batchSize, gridY};

    eagleBaseAssembleHiddenStateKernel<MAX_PATH><<<gridDim2, blockDim2, 0, stream>>>(
        acceptedIndices.dataPointer<int32_t>(), acceptLengths.dataPointer<int32_t>(), hiddenState.dataPointer<half>(),
        batchSize, maxDepth, numTokens, hiddenDim);
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

void constructVerificationDraftTree(rt::Tensor const& draftIdFullTable, rt::Tensor const& draftParentFullTable,
    rt::Tensor const& selectedIndices, rt::Tensor& inputIds, rt::Tensor& draftTreeMask, cudaStream_t stream)
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

    check::check(verifyTreeSize <= 128,
        "128 should be sufficient for verify tree size. We use 128 as CTA size to launch the kernel.");

    dim3 blockDim{128};
    dim3 gridDim{static_cast<uint32_t>(batchSize)};
    constructVerificationDraftTreeKernel<<<gridDim, blockDim, 0, stream>>>(draftIdFullTable.dataPointer<int32_t>(),
        draftParentFullTable.dataPointer<int32_t>(), selectedIndices.dataPointer<int32_t>(),
        inputIds.dataPointer<int32_t>(), draftTreeMask.dataPointer<int8_t>(), fullTableLength, verifyTreeSize);
}

} // namespace kernel
} // namespace trt_edgellm