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
#include "common/cudaUtils.h"
#include "common/mathUtils.h"
#include "kernels/speculative/dominoRolloutKernels.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kGruWarpsPerBlock = 8;
constexpr int32_t kGruThreadsPerBlock = kGruWarpsPerBlock * 32;
constexpr int32_t kCorrectWarpsPerBlock = 8;
constexpr int32_t kCorrectThreadsPerBlock = kCorrectWarpsPerBlock * 32;
//! Pool rows scored by one warp; 2048 / (8 * 4) = 64 CTAs per sequence.
constexpr int32_t kScoreRowsPerWarp = 4;

__device__ inline float dotHalfShared(
    half const* __restrict__ row, half const* __restrict__ sharedVec, int32_t length, int32_t laneId)
{
    // Row pointers are 4-byte aligned (even lengths), so accumulate in half2.
    float acc = 0.0F;
    half2 const* row2 = reinterpret_cast<half2 const*>(row);
    half2 const* vec2 = reinterpret_cast<half2 const*>(sharedVec);
    int32_t const pairs = length / 2;
    for (int32_t i = laneId; i < pairs; i += 32)
    {
        float2 const a = __half22float2(row2[i]);
        float2 const b = __half22float2(vec2[i]);
        acc += a.x * b.x + a.y * b.y;
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    return acc;
}

//! Orderable-float key with (0xFFFFFFFF - poolIdx) in the low bits so exact ties
//! resolve to the lowest pool index under max (same packing as the DSpark
//! greedy slots).
__device__ inline unsigned long long dominoPackCandidate(float value, int32_t poolIdx)
{
    uint32_t bits = __float_as_uint(value);
    bits = (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
    return (static_cast<unsigned long long>(bits) << 32)
        | static_cast<unsigned long long>(0xFFFFFFFFU - static_cast<uint32_t>(poolIdx));
}

__device__ inline int32_t dominoUnpackIndex(unsigned long long packed)
{
    return static_cast<int32_t>(0xFFFFFFFFU - static_cast<uint32_t>(packed & 0xFFFFFFFFULL));
}

// Grid: (ceil(G / warpsPerBlock), batch). One warp computes one GRU output g.
// The token's input projection is one [3G] row of the precomputed table; only the
// hidden state hIn (G half elements) is staged in shared memory.
__global__ void dominoGruUpdateKernel(half const* __restrict__ gruInputTable, int32_t const* __restrict__ tokenIds,
    int32_t tokenStride, int32_t tokenOffset, half const* __restrict__ weightHh, half const* __restrict__ hIn,
    half* __restrict__ hOut, int32_t gruHiddenDim)
{
    extern __shared__ half hShared[];
    int32_t const batchIdx = blockIdx.y;
    half const* hRow = hIn + static_cast<int64_t>(batchIdx) * gruHiddenDim;
    for (int32_t i = threadIdx.x; i < gruHiddenDim; i += blockDim.x)
    {
        hShared[i] = hRow[i];
    }
    __syncthreads();

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const g = blockIdx.x * kGruWarpsPerBlock + warpId;
    if (g >= gruHiddenDim)
    {
        return;
    }

    float const hr = dotHalfShared(weightHh + static_cast<int64_t>(g) * gruHiddenDim, hShared, gruHiddenDim, laneId);
    float const hz = dotHalfShared(
        weightHh + static_cast<int64_t>(g + gruHiddenDim) * gruHiddenDim, hShared, gruHiddenDim, laneId);
    float const hn = dotHalfShared(
        weightHh + static_cast<int64_t>(g + 2 * gruHiddenDim) * gruHiddenDim, hShared, gruHiddenDim, laneId);

    if (laneId == 0)
    {
        int32_t const token = tokenIds[batchIdx * tokenStride + tokenOffset];
        half const* inputRow = gruInputTable + static_cast<int64_t>(token) * 3 * gruHiddenDim;
        float const ir = __half2float(inputRow[g]);
        float const iz = __half2float(inputRow[gruHiddenDim + g]);
        float const in = __half2float(inputRow[2 * gruHiddenDim + g]);
        float const r = 1.0F / (1.0F + __expf(-(ir + hr)));
        float const z = 1.0F / (1.0F + __expf(-(iz + hz)));
        float const n = tanhf(in + r * hn);
        float const hPrev = __half2float(hShared[g]);
        hOut[static_cast<int64_t>(batchIdx) * gruHiddenDim + g] = __float2half((1.0F - z) * n + z * hPrev);
    }
}

// Grid: (ceil(projDim / warps), batch). One warp per correction row r:
// c[r] = SiLU(zProj[b, step, r] + projStateWeight[r] . h). Block 0 also clears the
// winner slot the scoring kernel accumulates into.
__global__ void dominoCorrectionVectorKernel(half const* __restrict__ zProj, half const* __restrict__ projStateWeight,
    half const* __restrict__ gruHidden, float* __restrict__ correction, unsigned long long* __restrict__ winnerSlots,
    int32_t zRow, int32_t zRows, int32_t projDim, int32_t gruHiddenDim)
{
    extern __shared__ half hShared[];
    int32_t const batchIdx = blockIdx.y;
    half const* hRow = gruHidden + static_cast<int64_t>(batchIdx) * gruHiddenDim;
    for (int32_t i = threadIdx.x; i < gruHiddenDim; i += blockDim.x)
    {
        hShared[i] = hRow[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        winnerSlots[batchIdx] = 0ULL;
    }
    __syncthreads();

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const r = blockIdx.x * kCorrectWarpsPerBlock + warpId;
    if (r >= projDim)
    {
        return;
    }
    float acc = dotHalfShared(projStateWeight + static_cast<int64_t>(r) * gruHiddenDim, hShared, gruHiddenDim, laneId);
    if (laneId == 0)
    {
        acc += __half2float(zProj[(static_cast<int64_t>(batchIdx) * zRows + zRow) * projDim + r]);
        correction[batchIdx * projDim + r] = acc / (1.0F + __expf(-acc)); // SiLU
    }
}

// Grid: (ceil(poolSize / (warps * rowsPerWarp)), batch). One warp scores kScoreRowsPerWarp
// pool rows with lanes striding the projDim pairs, so each 512 B codebook row is read
// coalesced. The block maximum is merged into the per-sequence slot with atomicMax; the
// packed key orders by score first and lowest pool index second.
__global__ void __launch_bounds__(kCorrectThreadsPerBlock) dominoPoolScoreKernel(
    float const* __restrict__ poolBaseLogits, half const* __restrict__ poolCodebook,
    float const* __restrict__ correction, unsigned long long* __restrict__ winnerSlots, float* __restrict__ scoresOut,
    int32_t logitsRow, int32_t poolRows, int32_t scoreRow, int32_t scoreRows, int32_t poolSize, int32_t projDim)
{
    extern __shared__ float cShared[];
    __shared__ unsigned long long warpBest[kCorrectWarpsPerBlock];
    int32_t const batchIdx = blockIdx.y;
    for (int32_t i = threadIdx.x; i < projDim; i += blockDim.x)
    {
        cShared[i] = correction[batchIdx * projDim + i];
    }
    __syncthreads();

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const pairs = projDim / 2;
    float const* stepLogits = poolBaseLogits + (static_cast<int64_t>(batchIdx) * poolRows + logitsRow) * poolSize;
    float* stepScores = scoresOut == nullptr
        ? nullptr
        : scoresOut + (static_cast<int64_t>(batchIdx) * scoreRows + scoreRow) * poolSize;
    int32_t const rowBegin = (blockIdx.x * kCorrectWarpsPerBlock + warpId) * kScoreRowsPerWarp;
    int32_t const rowEnd = min(rowBegin + kScoreRowsPerWarp, poolSize);
    unsigned long long best = 0ULL;
    for (int32_t p = rowBegin; p < rowEnd; ++p)
    {
        half2 const* row2
            = reinterpret_cast<half2 const*>(poolCodebook + (static_cast<int64_t>(batchIdx) * poolSize + p) * projDim);
        float acc = 0.0F;
        for (int32_t i = laneId; i < pairs; i += 32)
        {
            float2 const a = __half22float2(row2[i]);
            acc += a.x * cShared[2 * i] + a.y * cShared[2 * i + 1];
        }
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
        }
        float const score = stepLogits[p] + acc;
        if (stepScores != nullptr && laneId == 0)
        {
            stepScores[p] = score;
        }
        unsigned long long const candidate = dominoPackCandidate(score, p);
        best = (candidate > best) ? candidate : best;
    }
    if (laneId == 0)
    {
        warpBest[warpId] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long blockBest = warpBest[0];
        for (int32_t w = 1; w < kCorrectWarpsPerBlock; ++w)
        {
            blockBest = (warpBest[w] > blockBest) ? warpBest[w] : blockBest;
        }
        atomicMax(winnerSlots + batchIdx, blockBest);
    }
}

// Grid: batch, one thread. Resolves the packed winner to a token id.
__global__ void dominoPoolWinnerKernel(unsigned long long const* __restrict__ winnerSlots,
    int32_t const* __restrict__ poolCandidateIds, int32_t* __restrict__ draftTokenIds, int32_t poolSize,
    int32_t draftTokenStride, int32_t tokenOffset)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const winner = dominoUnpackIndex(winnerSlots[batchIdx]);
    draftTokenIds[batchIdx * draftTokenStride + tokenOffset] = poolCandidateIds[batchIdx * poolSize + winner];
}

// Grid: batch, kCorrectThreadsPerBlock. Argmax of poolBaseLogits row 0 (the uncorrected
// first proposal) with the same packed-key tie-break as the scored steps; optionally copies
// the row into the stacked scores so the DDTree builder sees exactly what the chain used.
__global__ void dominoFirstProposalKernel(float const* __restrict__ poolBaseLogits, int32_t poolRows,
    float* __restrict__ scoresOut, int32_t scoreRows, unsigned long long* __restrict__ winnerSlots, int32_t poolSize)
{
    __shared__ unsigned long long warpBest[kCorrectWarpsPerBlock];
    int32_t const batchIdx = blockIdx.x;
    float const* row = poolBaseLogits + static_cast<int64_t>(batchIdx) * poolRows * poolSize;
    float* outRow = scoresOut == nullptr ? nullptr : scoresOut + static_cast<int64_t>(batchIdx) * scoreRows * poolSize;
    unsigned long long best = 0ULL;
    for (int32_t p = threadIdx.x; p < poolSize; p += blockDim.x)
    {
        float const score = row[p];
        if (outRow != nullptr)
        {
            outRow[p] = score;
        }
        unsigned long long const candidate = dominoPackCandidate(score, p);
        best = (candidate > best) ? candidate : best;
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        unsigned long long const other = __shfl_xor_sync(0xFFFFFFFF, best, offset);
        best = (other > best) ? other : best;
    }
    int32_t const warpId = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0)
    {
        warpBest[warpId] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long blockBest = warpBest[0];
        for (int32_t w = 1; w < kCorrectWarpsPerBlock; ++w)
        {
            blockBest = (warpBest[w] > blockBest) ? warpBest[w] : blockBest;
        }
        winnerSlots[batchIdx] = blockBest;
    }
}

// Grid: batch, kCorrectThreadsPerBlock. Nodes 1..validCount-1 hold pool slots after ddtreeBuild.
__global__ void dominoRemapPoolSlotsKernel(int32_t* __restrict__ treeTokenIds, int32_t* __restrict__ verifyTokenIds,
    int32_t const* __restrict__ validCounts, int32_t const* __restrict__ poolCandidateIds, int32_t verifySize,
    int32_t poolSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const validCount = min(validCounts[batchIdx], verifySize);
    int32_t const* pool = poolCandidateIds + static_cast<int64_t>(batchIdx) * poolSize;
    int32_t const treeOffset = batchIdx * verifySize;
    for (int32_t node = 1 + threadIdx.x; node < validCount; node += blockDim.x)
    {
        int32_t const slot = treeTokenIds[treeOffset + node];
        int32_t const token = pool[min(max(slot, 0), poolSize - 1)];
        treeTokenIds[treeOffset + node] = token;
        verifyTokenIds[treeOffset + node] = token;
    }
}

} // namespace

void dominoGruUpdate(rt::Tensor const& gruInputTable, rt::Tensor const& tokenIds, int32_t tokenStride,
    int32_t tokenOffset, rt::Tensor const& gruWeightHh, rt::Tensor const& hIn, rt::Tensor& hOut, int32_t batchSize,
    int32_t gruHiddenDim, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }
    ELLM_CHECK(gruInputTable.getDataType() == trt_edgellm::DataType::kHALF
            && gruWeightHh.getDataType() == trt_edgellm::DataType::kHALF
            && hIn.getDataType() == trt_edgellm::DataType::kHALF && hOut.getDataType() == trt_edgellm::DataType::kHALF,
        "dominoGruUpdate requires an FP16 input table, GRU weights, and state");
    ELLM_CHECK(gruHiddenDim > 0 && gruHiddenDim % 2 == 0 && gruHiddenDim <= kDominoMaxGruHiddenDim,
        "dominoGruUpdate: gruHiddenDim must be a positive even value within the supported bound");
    ELLM_CHECK(hIn.rawPointer() != hOut.rawPointer(), "dominoGruUpdate: hIn and hOut must be distinct buffers");

    dim3 const grid(divUp(gruHiddenDim, kGruWarpsPerBlock), batchSize);
    size_t const sharedBytes = static_cast<size_t>(gruHiddenDim) * sizeof(half);
    dominoGruUpdateKernel<<<grid, kGruThreadsPerBlock, sharedBytes, stream>>>(
        static_cast<half const*>(gruInputTable.rawPointer()), static_cast<int32_t const*>(tokenIds.rawPointer()),
        tokenStride, tokenOffset, static_cast<half const*>(gruWeightHh.rawPointer()),
        static_cast<half const*>(hIn.rawPointer()), static_cast<half*>(hOut.rawPointer()), gruHiddenDim);
    CUDA_CHECK(cudaGetLastError());
}

void dominoCorrectStep(rt::Tensor const& zProj, rt::Tensor const& poolBaseLogits, rt::Tensor const& poolCodebook,
    rt::Tensor const& poolCandidateIds, rt::Tensor const& projStateWeight, rt::Tensor const& gruHidden,
    rt::Tensor& draftTokenIds, rt::Tensor* scoresOut, DominoCorrectScratch scratch, DominoStepLayout layout,
    int32_t draftTokenStride, int32_t tokenOffset, int32_t batchSize, int32_t poolSize, int32_t projDim,
    int32_t gruHiddenDim, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }
    ELLM_CHECK(zProj.getDataType() == trt_edgellm::DataType::kHALF
            && poolCodebook.getDataType() == trt_edgellm::DataType::kHALF
            && projStateWeight.getDataType() == trt_edgellm::DataType::kHALF
            && gruHidden.getDataType() == trt_edgellm::DataType::kHALF
            && poolBaseLogits.getDataType() == trt_edgellm::DataType::kFLOAT
            && poolCandidateIds.getDataType() == trt_edgellm::DataType::kINT32,
        "dominoCorrectStep: unexpected tensor dtypes");
    ELLM_CHECK(scratch.correction.getDataType() == trt_edgellm::DataType::kFLOAT
            && scratch.correction.getShape().volume() >= static_cast<int64_t>(batchSize) * projDim
            && scratch.winnerSlots.getDataType() == trt_edgellm::DataType::kINT64
            && scratch.winnerSlots.getShape().volume() >= batchSize,
        "dominoCorrectStep: scratch must hold [B, projDim] FP32 and [B] INT64");
    ELLM_CHECK(projDim > 0 && projDim % 2 == 0 && projDim <= kDominoMaxProjDim,
        "dominoCorrectStep: projDim must be a positive even value within the supported bound");
    ELLM_CHECK(gruHiddenDim > 0 && gruHiddenDim % 2 == 0, "dominoCorrectStep: gruHiddenDim must be positive and even");
    ELLM_CHECK(layout.zRow >= 0 && layout.zRow < layout.zRows, "dominoCorrectStep: zRow is outside zProj");
    ELLM_CHECK(layout.logitsRow >= 0 && layout.logitsRow < layout.poolRows,
        "dominoCorrectStep: logitsRow is outside poolBaseLogits");
    ELLM_CHECK(poolSize > 0, "dominoCorrectStep: poolSize must be positive");
    float* scores = nullptr;
    if (scoresOut != nullptr)
    {
        ELLM_CHECK(scoresOut->getDataType() == trt_edgellm::DataType::kFLOAT && layout.scoreRow >= 0
                && layout.scoreRow < layout.scoreRows
                && scoresOut->getShape().volume() >= static_cast<int64_t>(batchSize) * layout.scoreRows * poolSize,
            "dominoCorrectStep: scoresOut must be FP32 [B, scoreRows, poolSize] with scoreRow in range");
        scores = static_cast<float*>(scoresOut->rawPointer());
    }

    auto* correction = static_cast<float*>(scratch.correction.rawPointer());
    auto* winnerSlots = static_cast<unsigned long long*>(scratch.winnerSlots.rawPointer());
    dim3 const vectorGrid(divUp(projDim, kCorrectWarpsPerBlock), batchSize);
    dominoCorrectionVectorKernel<<<vectorGrid, kCorrectThreadsPerBlock,
        static_cast<size_t>(gruHiddenDim) * sizeof(half), stream>>>(static_cast<half const*>(zProj.rawPointer()),
        static_cast<half const*>(projStateWeight.rawPointer()), static_cast<half const*>(gruHidden.rawPointer()),
        correction, winnerSlots, layout.zRow, layout.zRows, projDim, gruHiddenDim);
    dim3 const scoreGrid(divUp(poolSize, kCorrectWarpsPerBlock * kScoreRowsPerWarp), batchSize);
    dominoPoolScoreKernel<<<scoreGrid, kCorrectThreadsPerBlock, static_cast<size_t>(projDim) * sizeof(float), stream>>>(
        static_cast<float const*>(poolBaseLogits.rawPointer()), static_cast<half const*>(poolCodebook.rawPointer()),
        correction, winnerSlots, scores, layout.logitsRow, layout.poolRows, layout.scoreRow, layout.scoreRows, poolSize,
        projDim);
    dominoPoolWinnerKernel<<<batchSize, 1, 0, stream>>>(winnerSlots,
        static_cast<int32_t const*>(poolCandidateIds.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
        poolSize, draftTokenStride, tokenOffset);
    CUDA_CHECK(cudaGetLastError());
}

void dominoSelectFirstProposal(rt::Tensor const& poolBaseLogits, int32_t poolRows, rt::Tensor const& poolCandidateIds,
    rt::Tensor& draftTokenIds, rt::Tensor* scoresOut, int32_t scoreRows, rt::Tensor& winnerSlots,
    int32_t draftTokenStride, int32_t tokenOffset, int32_t batchSize, int32_t poolSize, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }
    ELLM_CHECK(poolBaseLogits.getDataType() == trt_edgellm::DataType::kFLOAT
            && poolCandidateIds.getDataType() == trt_edgellm::DataType::kINT32
            && winnerSlots.getDataType() == trt_edgellm::DataType::kINT64
            && winnerSlots.getShape().volume() >= batchSize,
        "dominoSelectFirstProposal: unexpected tensor dtypes");
    ELLM_CHECK(poolRows > 0 && poolSize > 0, "dominoSelectFirstProposal: poolRows and poolSize must be positive");
    float* scores = nullptr;
    if (scoresOut != nullptr)
    {
        ELLM_CHECK(scoresOut->getDataType() == trt_edgellm::DataType::kFLOAT && scoreRows > 0
                && scoresOut->getShape().volume() >= static_cast<int64_t>(batchSize) * scoreRows * poolSize,
            "dominoSelectFirstProposal: scoresOut must be FP32 [B, scoreRows, poolSize]");
        scores = static_cast<float*>(scoresOut->rawPointer());
    }
    auto* slots = static_cast<unsigned long long*>(winnerSlots.rawPointer());
    dominoFirstProposalKernel<<<batchSize, kCorrectThreadsPerBlock, 0, stream>>>(
        static_cast<float const*>(poolBaseLogits.rawPointer()), poolRows, scores, scoreRows, slots, poolSize);
    dominoPoolWinnerKernel<<<batchSize, 1, 0, stream>>>(slots,
        static_cast<int32_t const*>(poolCandidateIds.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
        poolSize, draftTokenStride, tokenOffset);
    CUDA_CHECK(cudaGetLastError());
}

void dominoRemapPoolSlots(rt::Tensor& treeTokenIds, rt::Tensor& verifyTokenIds, rt::Tensor const& validCounts,
    rt::Tensor const& poolCandidateIds, int32_t batchSize, int32_t verifySize, int32_t poolSize, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }
    ELLM_CHECK(treeTokenIds.getDataType() == trt_edgellm::DataType::kINT32
            && verifyTokenIds.getDataType() == trt_edgellm::DataType::kINT32
            && validCounts.getDataType() == trt_edgellm::DataType::kINT32
            && poolCandidateIds.getDataType() == trt_edgellm::DataType::kINT32,
        "dominoRemapPoolSlots: all tensors must be INT32");
    ELLM_CHECK(treeTokenIds.getShape().volume() >= static_cast<int64_t>(batchSize) * verifySize
            && verifyTokenIds.getShape().volume() >= static_cast<int64_t>(batchSize) * verifySize
            && validCounts.getShape().volume() >= batchSize
            && poolCandidateIds.getShape().volume() >= static_cast<int64_t>(batchSize) * poolSize,
        "dominoRemapPoolSlots: tensors are smaller than [B, verifySize] / [B, poolSize]");
    dominoRemapPoolSlotsKernel<<<batchSize, kCorrectThreadsPerBlock, 0, stream>>>(
        static_cast<int32_t*>(treeTokenIds.rawPointer()), static_cast<int32_t*>(verifyTokenIds.rawPointer()),
        static_cast<int32_t const*>(validCounts.rawPointer()),
        static_cast<int32_t const*>(poolCandidateIds.rawPointer()), verifySize, poolSize);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
