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
#include "common/pagedKvTypes.h"
#include "dflashRuntimeKernels.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace kernel
{

// -----------------------------------------------------------------------
// DFlash target KV cache update kernel
// -----------------------------------------------------------------------
//
// Grid: (numTokens, numKVHeads, batchSize)  where numTokens = deltaLen
// Block: (threadsPerToken)  where threadsPerToken = headDim / vecSize
//
// Each thread handles vecSize=8 half elements.

static constexpr int32_t kVecSize = 8; // half8

__global__ void dflashTargetKVCacheUpdateKernel(half const* __restrict__ kDelta, half const* __restrict__ vDelta,
    half* __restrict__ kvCache, float const* __restrict__ cosSinCache, int32_t const* __restrict__ deltaStartPositions,
    int32_t const* __restrict__ deltaLengths, int32_t deltaLen, int32_t numKVHeads, int32_t headDim, int32_t rotaryDim,
    int32_t cosSinBatch, int32_t cosSinSeqLen, int32_t const* __restrict__ pageTable, int32_t numPages,
    int32_t maxPagesPerSeq)
{
    int32_t const t = blockIdx.x;  // token index within delta [0, deltaLen)
    int32_t const h = blockIdx.y;  // KV head index
    int32_t const b = blockIdx.z;  // batch index
    int32_t const d = threadIdx.x; // element group index (covers vecSize elements)

    if (t >= deltaLen || h >= numKVHeads)
    {
        return;
    }

    // Per-batch delta length guard: skip padded positions for multi-batch
    if (t >= deltaLengths[b])
    {
        return;
    }

    int32_t const elemOffset = d * kVecSize;
    if (elemOffset >= headDim)
    {
        return;
    }

    int32_t const deltaStart = deltaStartPositions[b];
    int32_t const pos = deltaStart + t; // absolute position in the cache

    if (pos < 0)
    {
        return;
    }
    int32_t const pageIndex = pos / rt::kTOKENS_PER_PAGE;
    if (pageIndex >= maxPagesPerSeq)
    {
        return;
    }

    // --- Read k_delta and v_delta ---
    // Layout: [B, deltaLen, numKVHeads, headDim]
    int64_t const kvDeltaOffset = static_cast<int64_t>(b) * deltaLen * numKVHeads * headDim
        + static_cast<int64_t>(t) * numKVHeads * headDim + static_cast<int64_t>(h) * headDim;

    // Load k_delta elements
    half kVals[kVecSize];
    half vVals[kVecSize];
#pragma unroll
    for (int32_t i = 0; i < kVecSize; ++i)
    {
        int32_t const idx = elemOffset + i;
        kVals[i] = (idx < headDim) ? kDelta[kvDeltaOffset + idx] : __float2half(0.0f);
        vVals[i] = (idx < headDim) ? vDelta[kvDeltaOffset + idx] : __float2half(0.0f);
    }

    // --- Apply RoPE to K (non-interleaved) ---
    // cos/sin layout: [cosSinBatch, cosSinSeqLen, rotaryDim]
    // Non-interleaved: cos = [0, rotaryDim/2), sin = [rotaryDim/2, rotaryDim)
    // OOB guard: if pos exceeds cos/sin cache, skip RoPE (pass-through)
    bool const ropeInBounds = (pos < cosSinSeqLen);
    int32_t const cosBatchIdx = (cosSinBatch == 1) ? 0 : b;
    int64_t const csBase = ropeInBounds
        ? (static_cast<int64_t>(cosBatchIdx) * cosSinSeqLen * rotaryDim + static_cast<int64_t>(pos) * rotaryDim)
        : 0;
    int32_t const halfRotary = rotaryDim / 2;

    half kRoped[kVecSize];
#pragma unroll
    for (int32_t i = 0; i < kVecSize; ++i)
    {
        int32_t const idx = elemOffset + i;
        if (idx < rotaryDim && ropeInBounds)
        {
            // Non-interleaved RoPE: first half uses cos[idx], second half uses cos[idx - halfRotary]
            float cosVal, sinVal;
            if (idx < halfRotary)
            {
                cosVal = cosSinCache[csBase + idx];
                sinVal = cosSinCache[csBase + halfRotary + idx];
                // k_roped = k * cos - k_permute * sin
                // For first half: permute partner is at idx + halfRotary
                float kOrig = __half2float(kVals[i]);
                // Need to read the partner element
                half kPartner = kDelta[kvDeltaOffset + idx + halfRotary];
                kRoped[i] = __float2half(kOrig * cosVal - __half2float(kPartner) * sinVal);
            }
            else
            {
                int32_t const partnerIdx = idx - halfRotary;
                cosVal = cosSinCache[csBase + partnerIdx];
                sinVal = cosSinCache[csBase + halfRotary + partnerIdx];
                // For second half: k_roped = k_permute * sin + k * cos
                float kOrig = __half2float(kVals[i]);
                half kPartner = kDelta[kvDeltaOffset + partnerIdx];
                kRoped[i] = __float2half(__half2float(kPartner) * sinVal + kOrig * cosVal);
            }
        }
        else
        {
            // Beyond rotary dim: pass through
            kRoped[i] = kVals[i];
        }
    }

    int32_t const inPageOffset = pos % rt::kTOKENS_PER_PAGE;
    int32_t const kPage = pageTable[(b * 2) * maxPagesPerSeq + pageIndex];
    int32_t const vPage = pageTable[(b * 2 + 1) * maxPagesPerSeq + pageIndex];
    if (kPage < 0 || kPage >= numPages || vPage < numPages || vPage >= 2 * numPages)
    {
        return;
    }
    int64_t const tokenStride = static_cast<int64_t>(numKVHeads) * headDim;
    int64_t const kBase = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPageOffset) * tokenStride
        + static_cast<int64_t>(h) * headDim;
#pragma unroll
    for (int32_t i = 0; i < kVecSize; ++i)
    {
        int32_t const idx = elemOffset + i;
        if (idx < headDim)
        {
            kvCache[kBase + idx] = kRoped[i];
        }
    }

    int64_t const vBase = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPageOffset) * tokenStride
        + static_cast<int64_t>(h) * headDim;
#pragma unroll
    for (int32_t i = 0; i < kVecSize; ++i)
    {
        int32_t const idx = elemOffset + i;
        if (idx < headDim)
        {
            kvCache[vBase + idx] = vVals[i];
        }
    }
}

void launchDFlashTargetKVCacheUpdate(half const* kDelta, half const* vDelta, half* kvCache, float const* cosSinCache,
    int32_t const* deltaStartPositions, int32_t const* deltaLengths, int32_t const* pageTable, int32_t batchSize,
    int32_t deltaLen, int32_t numKVHeads, int32_t headDim, int32_t rotaryDim, int32_t cosSinBatch, int32_t cosSinSeqLen,
    int32_t numPages, int32_t maxPagesPerSeq, cudaStream_t stream)
{
    if (deltaLen == 0 || batchSize == 0)
    {
        return;
    }

    int32_t const threadsPerToken = (headDim + kVecSize - 1) / kVecSize;
    assert(threadsPerToken <= 1024 && "DFlash KV cache update exceeds CUDA max threads per block");
    dim3 const grid(deltaLen, numKVHeads, batchSize);
    dim3 const block(threadsPerToken);

    dflashTargetKVCacheUpdateKernel<<<grid, block, 0, stream>>>(kDelta, vDelta, kvCache, cosSinCache,
        deltaStartPositions, deltaLengths, deltaLen, numKVHeads, headDim, rotaryDim, cosSinBatch, cosSinSeqLen,
        pageTable, numPages, maxPagesPerSeq);
    CUDA_CHECK(cudaGetLastError());
}

void checkDFlashRopeCapacity(int32_t cosSinSeqLen, int32_t kvCapacity)
{
    ELLM_CHECK(cosSinSeqLen <= kvCapacity,
        "checkDFlashRopeCapacity: rope_cos_sin sequence length (" + std::to_string(cosSinSeqLen)
            + ") exceeds the KV pool's padded per-slot capacity (" + std::to_string(kvCapacity)
            + "); the rope cache cannot cover a position the KV pool has no room for.");
}

// -----------------------------------------------------------------------
// DFlash proposal input preparation kernel
// -----------------------------------------------------------------------
//
// Grid: (batchSize)
// Block: (blockSize)

__global__ void dflashPrepareProposalInputsKernel(int32_t const* __restrict__ oldDraftCacheLengths,
    int32_t const* __restrict__ deltaLengths, int32_t blockSize, int32_t* __restrict__ packedAttentionMask,
    int32_t* __restrict__ attentionPosId, int32_t* __restrict__ contextLengths, bool causalProposalMask)
{
    int32_t const b = blockIdx.x;
    int32_t const i = threadIdx.x; // position within the block [0, blockSize)

    if (i >= blockSize)
    {
        return;
    }

    // target_len_after_delta = old cache length + per-batch delta length
    int32_t const targetLen = oldDraftCacheLengths[b] + deltaLengths[b];

    // Position ID: target_len + i
    attentionPosId[b * blockSize + i] = targetLen + i;

    // Context length: target_len + blockSize (one value per batch, write from thread 0)
    if (i == 0)
    {
        contextLengths[b] = targetLen + blockSize;
    }

    // Packed attention mask layout: [B, BS, divUp(BS, 32)]. DFlash uses full
    // non-causal rows; JetSpec uses causal proposal rows because the public
    // draft checkpoint was trained with causal_head=true.
    int32_t const packedMaskLen = (blockSize + 31) / 32;
    for (int32_t w = 0; w < packedMaskLen; ++w)
    {
        int32_t mask = 0;
        int32_t const bitStart = w * 32;
        int32_t const bitEnd = min(bitStart + 32, blockSize);
        for (int32_t bit = bitStart; bit < bitEnd; ++bit)
        {
            if (!causalProposalMask || bit <= i)
            {
                mask |= (1 << (bit - bitStart));
            }
        }
        packedAttentionMask[b * blockSize * packedMaskLen + i * packedMaskLen + w] = mask;
    }
}

void launchDFlashPrepareProposalInputs(int32_t const* oldDraftCacheLengths, int32_t const* deltaLengths,
    int32_t blockSize, int32_t* packedAttentionMask, int32_t* attentionPosId, int32_t* contextLengths,
    bool causalProposalMask, int32_t batchSize, cudaStream_t stream)
{
    if (batchSize == 0 || blockSize == 0)
    {
        return;
    }

    assert(blockSize <= 1024 && "DFlash proposal block size exceeds CUDA max threads per block");
    dim3 const grid(batchSize);
    dim3 const block(blockSize);

    dflashPrepareProposalInputsKernel<<<grid, block, 0, stream>>>(oldDraftCacheLengths, deltaLengths, blockSize,
        packedAttentionMask, attentionPosId, contextLengths, causalProposalMask);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------
// DFlash base verification input preparation kernel
// -----------------------------------------------------------------------
//
// Grid: (batchSize)
// Block: (verifySize)

__global__ void dflashPrepareBaseVerifyInputsKernel(int32_t const* __restrict__ baseKVCacheLengths, int32_t verifySize,
    int32_t* __restrict__ packedAttentionMask, int32_t* __restrict__ attentionPosId,
    int64_t* __restrict__ selectTokenIndices, int32_t* __restrict__ contextLengths)
{
    int32_t const b = blockIdx.x;
    int32_t const i = threadIdx.x; // position within the verified block [0, verifySize)

    if (i >= verifySize)
    {
        return;
    }

    int32_t const baseLen = baseKVCacheLengths[b];
    int32_t const packedMaskLen = (verifySize + 31) / 32;

    attentionPosId[b * verifySize + i] = baseLen + i;
    selectTokenIndices[b * verifySize + i] = i;
    if (i == 0)
    {
        contextLengths[b] = baseLen + verifySize;
    }

    // Causal packed mask for linear verification: row i has bits [0, i] set.
    for (int32_t w = 0; w < packedMaskLen; ++w)
    {
        int32_t const bitStart = w * 32;
        int32_t const bitEnd = min(bitStart + 32, verifySize);
        int32_t const validBits = bitEnd - bitStart;

        uint32_t mask = 0;
        if (i >= bitEnd - 1)
        {
            mask = (validBits == 32) ? 0xFFFFFFFFu : ((1u << validBits) - 1u);
        }
        else if (i >= bitStart)
        {
            mask = (1u << (i - bitStart + 1)) - 1u;
        }
        packedAttentionMask[b * verifySize * packedMaskLen + i * packedMaskLen + w] = static_cast<int32_t>(mask);
    }
}

void launchDFlashPrepareBaseVerifyInputs(int32_t const* baseKVCacheLengths, int32_t verifySize,
    int32_t* packedAttentionMask, int32_t* attentionPosId, int64_t* selectTokenIndices, int32_t* contextLengths,
    int32_t batchSize, cudaStream_t stream)
{
    if (batchSize == 0 || verifySize == 0)
    {
        return;
    }

    assert(verifySize <= 1024 && "DFlash verify block size exceeds CUDA max threads per block");
    dim3 const grid(batchSize);
    dim3 const block(verifySize);

    dflashPrepareBaseVerifyInputsKernel<<<grid, block, 0, stream>>>(
        baseKVCacheLengths, verifySize, packedAttentionMask, attentionPosId, selectTokenIndices, contextLengths);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------
// DFlash linear verification tree input builder
// -----------------------------------------------------------------------
//
// Grid: ceil(max(batchSize * blockSize, batchSize * blockSize * blockSize) / 256)
// Block: 256

__global__ void dflashBuildLinearVerifyInputsKernel(int32_t const* __restrict__ lastAcceptedTokens,
    int32_t const* __restrict__ draftTokenIds, int32_t* __restrict__ verifyTokenIds,
    int8_t* __restrict__ verifyTreeMask, int32_t proposalLen, int32_t draftTokenStride, int32_t verifySize,
    int32_t tokenElements, int32_t maskElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tokenElements)
    {
        int32_t const batchIdx = idx / verifySize;
        int32_t const posIdx = idx % verifySize;
        // DFlash draft output at position 0 predicts the current token (t_last),
        // not the next token. Real draft proposals start at position 1.
        // DDTree explicitly skips depthIdx==0 for the same reason (see ddtreeKernels.cu).
        verifyTokenIds[idx]
            = posIdx == 0 ? lastAcceptedTokens[batchIdx] : draftTokenIds[batchIdx * draftTokenStride + posIdx];
    }

    // Token ids and mask entries are independent output buffers, so one flat
    // launch can populate both ranges without cross-thread ordering.
    if (idx < maskElements)
    {
        int32_t const localIdx = idx % (verifySize * verifySize);
        int32_t const rowIdx = localIdx / verifySize;
        int32_t const colIdx = localIdx % verifySize;
        verifyTreeMask[idx] = colIdx <= rowIdx ? int8_t{1} : int8_t{0};
    }
}

void launchDFlashBuildLinearVerifyInputs(int32_t const* lastAcceptedTokens, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int8_t* verifyTreeMask, int32_t batchSize, int32_t proposalLen, int32_t draftTokenStride,
    int32_t verifySize, cudaStream_t stream)
{
    if (batchSize == 0 || proposalLen == 0 || verifySize == 0)
    {
        return;
    }

    assert(verifySize == proposalLen + 1 && "DFlash linear verify expects [anchor] + proposal tokens");
    assert(draftTokenStride >= proposalLen && "DFlash draft token stride must cover proposalLen tokens");
    int32_t const tokenElements = batchSize * verifySize;
    int32_t const maskElements = batchSize * verifySize * verifySize;
    constexpr int32_t kThreadsPerBlock{256};
    int32_t const blocks = (std::max(tokenElements, maskElements) + kThreadsPerBlock - 1) / kThreadsPerBlock;

    dflashBuildLinearVerifyInputsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(lastAcceptedTokens, draftTokenIds,
        verifyTokenIds, verifyTreeMask, proposalLen, draftTokenStride, verifySize, tokenElements, maskElements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
