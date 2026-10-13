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
#include "kernels/speculative/cpSpecKernels.h"
#include "kernels/speculative/speculativeKernelsUtils.h"
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kBlockSize = 256;

//! Only a handful of rows are ever in flight, so the filtering kernel is pure
//! critical path; a wide block trades idle SMs for a shorter one.
constexpr int32_t kSortBlockSize = 512;

//! Radix-select stops refining once this many candidates survive; the bitonic
//! sort over the survivors resolves the rest of the key exactly.
constexpr int32_t kCandidateTarget = 128;
constexpr int32_t kMaxCandidates = 512;

//! Monotonic uint32 key: ordering the keys orders the floats.
__device__ __forceinline__ uint32_t keyOfFloat(float value)
{
    uint32_t const bits = __float_as_uint(value);
    return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

__device__ __forceinline__ float floatOfKey(uint32_t key)
{
    return __uint_as_float((key & 0x80000000U) ? (key & 0x7FFFFFFFU) : ~key);
}

//! Value key in the high word, complemented index in the low word, so a plain
//! descending sort of the packed value breaks ties toward the lower index.
__device__ __forceinline__ uint64_t packCandidate(uint32_t key, int32_t index)
{
    return (static_cast<uint64_t>(key) << 32) | (0xFFFFFFFFU - static_cast<uint32_t>(index));
}

__device__ __forceinline__ int32_t indexOfCandidate(uint64_t packed)
{
    return static_cast<int32_t>(0xFFFFFFFFU - static_cast<uint32_t>(packed & 0xFFFFFFFFU));
}

//! In-place bitonic sort into descending order. Each thread strides by blockDim,
//! so sub-warp strides only ever exchange within a warp and skip the barrier.
__device__ void sortPackedDescending(uint64_t* sPacked, int32_t paddedLen)
{
    for (int32_t k = 2; k <= paddedLen; k <<= 1)
    {
        for (int32_t j = k >> 1; j > 0; j >>= 1)
        {
            for (int32_t i = threadIdx.x; i < paddedLen; i += blockDim.x)
            {
                int32_t const partner = i ^ j;
                if (partner <= i)
                {
                    continue;
                }
                uint64_t const lhs = sPacked[i];
                uint64_t const rhs = sPacked[partner];
                if (((i & k) == 0) == (lhs < rhs))
                {
                    sPacked[i] = rhs;
                    sPacked[partner] = lhs;
                }
            }
            if (j >= 32)
            {
                __syncthreads();
            }
            else
            {
                __syncwarp();
            }
        }
        __syncthreads();
    }
}

//! Inclusive Hillis-Steele scan over one value per thread.
__device__ __forceinline__ void blockInclusiveScan(float* sScan)
{
    for (int32_t offset = 1; offset < blockDim.x; offset <<= 1)
    {
        float const addend = (static_cast<int32_t>(threadIdx.x) >= offset) ? sScan[threadIdx.x - offset] : 0.0F;
        __syncthreads();
        sScan[threadIdx.x] += addend;
        __syncthreads();
    }
}

//! Inverse-CDF sample over a weight function evaluated on [0, vocabSize).
//!
//! The vocabulary is split into contiguous per-thread ranges so the parallel
//! scan visits weights in vocab order, matching the scalar reference walk.
//! Zero-weight entries are skipped and the last positive index is the fallback
//! when the target exceeds the accumulated mass.
template <typename WeightFn>
__device__ int32_t blockInverseCdfSample(
    WeightFn weightAt, int32_t vocabSize, float target, float* sScan, int32_t* sLastPositive, int32_t* sResult)
{
    int32_t const perThread = (vocabSize + blockDim.x - 1) / blockDim.x;
    int32_t const lo = min(static_cast<int32_t>(threadIdx.x) * perThread, vocabSize);
    int32_t const hi = min(lo + perThread, vocabSize);

    float localSum = 0.0F;
    int32_t localLast = -1;
    for (int32_t v = lo; v < hi; ++v)
    {
        float const weight = weightAt(v);
        if (weight > 0.0F)
        {
            localSum += weight;
            localLast = v;
        }
    }
    sScan[threadIdx.x] = localSum;
    sLastPositive[threadIdx.x] = localLast;
    if (threadIdx.x == 0)
    {
        *sResult = -1;
    }
    __syncthreads();

    blockInclusiveScan(sScan);

    float const inclusive = sScan[threadIdx.x];
    float const exclusive = inclusive - localSum;
    if (inclusive > target && exclusive <= target)
    {
        float cumulative = exclusive;
        for (int32_t v = lo; v < hi; ++v)
        {
            float const weight = weightAt(v);
            if (weight <= 0.0F)
            {
                continue;
            }
            cumulative += weight;
            if (target < cumulative)
            {
                *sResult = v;
                break;
            }
        }
    }
    __syncthreads();

    if (*sResult >= 0)
    {
        return *sResult;
    }

    // Target ran past the accumulated mass (rounding, or an all-zero row).
    for (int32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (static_cast<int32_t>(threadIdx.x) < stride)
        {
            sLastPositive[threadIdx.x] = max(sLastPositive[threadIdx.x], sLastPositive[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return max(sLastPositive[0], 0);
}

struct DenseWeight
{
    float const* row;

    __device__ __forceinline__ float operator()(int32_t v) const
    {
        return row[v];
    }
};

struct ResidualWeight
{
    float const* targetRow;
    float const* draftRow;

    __device__ __forceinline__ float operator()(int32_t v) const
    {
        return fmaxf(targetRow[v] - draftRow[v], 0.0F);
    }
};

template <typename TLogit>
__global__ void cpSpecTopKTopPProbsKernel(TLogit const* __restrict__ logits, float* __restrict__ probabilities,
    int32_t rows, int32_t vocabSize, float temperature, int32_t topK, float topP)
{
    int32_t const rowIdx = blockIdx.x;
    if (rowIdx >= rows)
    {
        return;
    }

    extern __shared__ char sRaw[];
    uint32_t* const sKey = reinterpret_cast<uint32_t*>(sRaw);
    uint64_t* const sCandidates = reinterpret_cast<uint64_t*>(sKey + vocabSize + (vocabSize & 1));
    float* const sScan = reinterpret_cast<float*>(sCandidates + kMaxCandidates);
    __shared__ uint32_t sHist[256];
    __shared__ uint32_t sThreshold;
    __shared__ int32_t sNeedRank;
    __shared__ int32_t sNextRank;
    __shared__ int32_t sCandidateEstimate;
    __shared__ int32_t sCandidateCount;
    __shared__ int32_t sSortLen;
    __shared__ float sSumExp;

    TLogit const* const rowLogits = logits + static_cast<int64_t>(rowIdx) * vocabSize;
    float* const rowProbs = probabilities + static_cast<int64_t>(rowIdx) * vocabSize;

    bool const greedy = temperature < 1e-3F;
    float const invTemp = greedy ? 1.0F : 1.0F / temperature;
    float const effectiveTopP = greedy ? 1.0F : topP;
    int32_t effectiveTopK = greedy ? 1 : topK;
    if (effectiveTopK <= 0 || effectiveTopK > vocabSize)
    {
        effectiveTopK = vocabSize;
    }
    bool const useTopP = effectiveTopP < 1.0F - 1e-6F;

    for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
    {
        sKey[v] = keyOfFloat(toFloat(rowLogits[v]) * invTemp);
        rowProbs[v] = 0.0F;
    }
    if (threadIdx.x == 0)
    {
        sThreshold = 0;
        sNeedRank = effectiveTopK;
        sCandidateCount = 0;
        sCandidateEstimate = vocabSize;
    }
    __syncthreads();

    // MSB-first radix select for the top-k cut, stopping as soon as the surviving
    // candidate set fits the sort buffer; the sort then resolves the remaining bits.
    int32_t const iters = (vocabSize + static_cast<int32_t>(blockDim.x) - 1) / static_cast<int32_t>(blockDim.x);
    for (int32_t shift = 24; shift >= 0; shift -= 8)
    {
        for (int32_t b = threadIdx.x; b < 256; b += blockDim.x)
        {
            sHist[b] = 0;
        }
        __syncthreads();

        uint32_t const mask = (shift == 24) ? 0U : (0xFFFFFFFFU << (shift + 8));
        uint32_t const prefix = sThreshold;
        for (int32_t it = 0; it < iters; ++it)
        {
            int32_t const v = it * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
            bool const inRange = v < vocabSize;
            uint32_t const key = inRange ? sKey[v] : 0U;
            bool const active = inRange && ((key & mask) == (prefix & mask));
            uint32_t const bin = (key >> shift) & 255U;
            // Aggregate within the warp first: the high radix digits of a logit row
            // cluster hard, and unaggregated atomics on one bin serialize the block.
            unsigned const activeMask = __ballot_sync(0xFFFFFFFFU, active);
            if (active)
            {
                unsigned const peers = __match_any_sync(activeMask, bin);
                if ((threadIdx.x & 31U) == static_cast<unsigned>(__ffs(peers) - 1))
                {
                    atomicAdd(&sHist[bin], static_cast<uint32_t>(__popc(peers)));
                }
            }
        }
        __syncthreads();

        // One warp walks the bins from the top with shuffle suffix sums; a
        // block-wide scan over 256 bins costs more barriers than the histogram.
        if (threadIdx.x < 32)
        {
            int32_t const lane = static_cast<int32_t>(threadIdx.x);
            uint32_t bins[8];
            uint32_t laneTotal = 0;
            for (int32_t t = 0; t < 8; ++t)
            {
                bins[t] = sHist[lane * 8 + t];
                laneTotal += bins[t];
            }
            uint32_t suffix = laneTotal;
            for (int32_t offset = 1; offset < 32; offset <<= 1)
            {
                uint32_t const other = __shfl_down_sync(0xFFFFFFFFU, suffix, offset);
                if (lane + offset < 32)
                {
                    suffix += other;
                }
            }
            int32_t const rank = sNeedRank;
            uint32_t above = suffix - laneTotal;
            for (int32_t t = 7; t >= 0; --t)
            {
                uint32_t const count = bins[t];
                if (static_cast<int32_t>(above + count) >= rank && static_cast<int32_t>(above) < rank)
                {
                    sThreshold |= static_cast<uint32_t>(lane * 8 + t) << shift;
                    sNextRank = rank - static_cast<int32_t>(above);
                    sCandidateEstimate = static_cast<int32_t>(above + count);
                }
                above += count;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            sNeedRank = sNextRank;
        }
        __syncthreads();
        if (sCandidateEstimate <= kCandidateTarget)
        {
            break;
        }
    }

    // Candidates are every entry at or above the cut. Exact ties beyond the buffer
    // are dropped: they carry identical probability, so only the token identity of
    // an equal-valued entry can differ.
    uint32_t const threshold = sThreshold;
    for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
    {
        if (sKey[v] >= threshold)
        {
            int32_t const slot = atomicAdd(&sCandidateCount, 1);
            if (slot < kMaxCandidates)
            {
                sCandidates[slot] = packCandidate(sKey[v], v);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        sCandidateCount = min(sCandidateCount, kMaxCandidates);
        int32_t padded = 1;
        while (padded < sCandidateCount)
        {
            padded <<= 1;
        }
        sSortLen = padded;
    }
    __syncthreads();
    for (int32_t i = sCandidateCount + static_cast<int32_t>(threadIdx.x); i < sSortLen; i += blockDim.x)
    {
        sCandidates[i] = 0;
    }
    __syncthreads();

    sortPackedDescending(sCandidates, sSortLen);

    int32_t const kept = min(effectiveTopK, sCandidateCount);
    float const maxLogit = floatOfKey(static_cast<uint32_t>(sCandidates[0] >> 32));
    float expValue = 0.0F;
    if (static_cast<int32_t>(threadIdx.x) < kept)
    {
        expValue = __expf(floatOfKey(static_cast<uint32_t>(sCandidates[threadIdx.x] >> 32)) - maxLogit);
    }
    sScan[threadIdx.x] = expValue;
    __syncthreads();
    blockInclusiveScan(sScan);
    if (threadIdx.x == blockDim.x - 1)
    {
        sSumExp = sScan[threadIdx.x];
    }
    __syncthreads();

    float const sumExp = sSumExp;
    if (!(sumExp > 0.0F) || !isfinite(sumExp))
    {
        float const uniform = 1.0F / static_cast<float>(vocabSize);
        for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
        {
            rowProbs[v] = uniform;
        }
        return;
    }

    // Top-p water-filling: each entry takes what is left of the topP * sumExp budget,
    // which truncates the entry that crosses the nucleus boundary.
    float const denom = useTopP ? fmaxf(effectiveTopP * sumExp, 1e-20F) : fmaxf(sumExp, 1e-20F);
    if (static_cast<int32_t>(threadIdx.x) < kept)
    {
        float const exclusive = sScan[threadIdx.x] - expValue;
        float const assigned = useTopP ? fmaxf(fminf(expValue, denom - exclusive), 0.0F) : expValue;
        if (assigned > 0.0F)
        {
            rowProbs[indexOfCandidate(sCandidates[threadIdx.x])] = assigned / denom;
        }
    }
}

__global__ void cpSpecSampleRowsKernel(float const* __restrict__ probabilities, float const* __restrict__ uniforms,
    int32_t* __restrict__ tokenIds, int32_t rows, int32_t vocabSize)
{
    int32_t const rowIdx = blockIdx.x;
    if (rowIdx >= rows)
    {
        return;
    }

    __shared__ float sScan[kBlockSize];
    __shared__ int32_t sLastPositive[kBlockSize];
    __shared__ int32_t sResult;

    DenseWeight const weight{probabilities + static_cast<int64_t>(rowIdx) * vocabSize};
    float const target = clampUniform(uniforms[rowIdx]);
    int32_t const token = blockInverseCdfSample(weight, vocabSize, target, sScan, sLastPositive, &sResult);
    if (threadIdx.x == 0)
    {
        tokenIds[rowIdx] = token;
    }
}

__global__ void cpSpecProbabilisticAcceptKernel(float const* __restrict__ targetProbabilities,
    float const* __restrict__ draftProbabilities, int32_t const* __restrict__ draftTokenIds,
    int32_t const* __restrict__ proposalLengths, float const* __restrict__ acceptUniforms,
    int32_t* __restrict__ acceptedTokenIds, int32_t* __restrict__ acceptLength, int32_t draftStride,
    int32_t verifyProposalLen, int32_t vocabSize)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const uniformStride = 2 * draftStride + 1;
    int32_t* const batchAccepted = acceptedTokenIds + batchIdx * verifyLen;

    __shared__ float sScan[kBlockSize];
    __shared__ int32_t sLastPositive[kBlockSize];
    __shared__ int32_t sResult;
    __shared__ int32_t sRejectStep;

    for (int32_t pos = threadIdx.x; pos < verifyLen; pos += blockDim.x)
    {
        batchAccepted[pos] = 0;
    }
    if (threadIdx.x == 0)
    {
        sRejectStep = -1;
    }
    __syncthreads();

    int32_t const rowProposalLen = max(1, min(verifyProposalLen, proposalLengths[batchIdx]));
    if (threadIdx.x == 0)
    {
        int32_t acceptedDraft = 0;
        for (int32_t step = 0; step < rowProposalLen; ++step)
        {
            int32_t const draftToken = draftTokenIds[batchIdx * draftStride + step];
            float const targetProb
                = targetProbabilities[(static_cast<int64_t>(batchIdx) * verifyLen + step) * vocabSize + draftToken];
            float const draftProb
                = draftProbabilities[(static_cast<int64_t>(batchIdx) * draftStride + step) * vocabSize + draftToken];
            float const acceptProb = draftProb <= 1e-20F ? 1.0F : fminf(1.0F, targetProb / draftProb);
            if (acceptUniforms[batchIdx * uniformStride + step] <= acceptProb)
            {
                batchAccepted[acceptedDraft] = draftToken;
                ++acceptedDraft;
                continue;
            }
            sRejectStep = step;
            break;
        }
        acceptLength[batchIdx] = acceptedDraft + 1;
    }
    __syncthreads();

    // The trailing slot is a residual draw at the first rejection, or the bonus
    // token sampled from the target row past the accepted prefix.
    int32_t const rejectStep = sRejectStep;
    int32_t const slot = (rejectStep >= 0) ? rejectStep : rowProposalLen;
    float const* const targetRow
        = targetProbabilities + (static_cast<int64_t>(batchIdx) * verifyLen + slot) * vocabSize;

    int32_t token;
    if (rejectStep >= 0)
    {
        float const* const draftRow
            = draftProbabilities + (static_cast<int64_t>(batchIdx) * draftStride + rejectStep) * vocabSize;
        ResidualWeight const residual{targetRow, draftRow};

        float localSum = 0.0F;
        for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
        {
            localSum += residual(v);
        }
        sScan[threadIdx.x] = localSum;
        __syncthreads();
        for (int32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if (static_cast<int32_t>(threadIdx.x) < stride)
            {
                sScan[threadIdx.x] += sScan[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float const residualSum = sScan[0];
        __syncthreads();

        float const uniform = acceptUniforms[batchIdx * uniformStride + draftStride + rejectStep];
        if (residualSum <= 1e-20F)
        {
            token = blockInverseCdfSample(
                DenseWeight{targetRow}, vocabSize, clampUniform(uniform), sScan, sLastPositive, &sResult);
        }
        else
        {
            token = blockInverseCdfSample(
                residual, vocabSize, clampUniform(uniform) * residualSum, sScan, sLastPositive, &sResult);
        }
    }
    else
    {
        float const uniform = acceptUniforms[batchIdx * uniformStride + 2 * draftStride];
        token = blockInverseCdfSample(
            DenseWeight{targetRow}, vocabSize, clampUniform(uniform), sScan, sLastPositive, &sResult);
    }

    if (threadIdx.x == 0)
    {
        batchAccepted[acceptLength[batchIdx] - 1] = token;
    }
}

} // namespace

bool cpSpecSupportsVocab(int32_t vocabSize)
{
    return vocabSize > 0 && vocabSize <= kCpSpecMaxVocab;
}

void cpSpecTopKTopPProbs(rt::Tensor const& logits, rt::Tensor& probabilities, int32_t rows, int32_t vocabSize,
    float temperature, int32_t topK, float topP, cudaStream_t stream)
{
    if (rows <= 0)
    {
        return;
    }
    check::check(cpSpecSupportsVocab(vocabSize), "cpSpec kernels require vocabSize <= kCpSpecMaxVocab");
    size_t const smem = static_cast<size_t>(vocabSize + (vocabSize & 1)) * sizeof(uint32_t)
        + static_cast<size_t>(kMaxCandidates) * sizeof(uint64_t) + static_cast<size_t>(kSortBlockSize) * sizeof(float);
    // Half logits come straight from the lm_head GEMV, so the CP path skips a widening pass.
    if (logits.getDataType() == nvinfer1::DataType::kHALF)
    {
        cpSpecTopKTopPProbsKernel<__half>
            <<<rows, kSortBlockSize, smem, stream>>>(static_cast<__half const*>(logits.rawPointer()),
                probabilities.dataPointer<float>(), rows, vocabSize, temperature, topK, topP);
        return;
    }
    cpSpecTopKTopPProbsKernel<float><<<rows, kSortBlockSize, smem, stream>>>(
        logits.dataPointer<float>(), probabilities.dataPointer<float>(), rows, vocabSize, temperature, topK, topP);
}

void cpSpecSampleRows(rt::Tensor const& probabilities, float const* uniforms, rt::Tensor& tokenIds, int32_t rows,
    int32_t vocabSize, cudaStream_t stream)
{
    if (rows <= 0)
    {
        return;
    }
    check::check(cpSpecSupportsVocab(vocabSize), "cpSpec kernels require vocabSize <= kCpSpecMaxVocab");
    cpSpecSampleRowsKernel<<<rows, kBlockSize, 0, stream>>>(
        probabilities.dataPointer<float>(), uniforms, tokenIds.dataPointer<int32_t>(), rows, vocabSize);
}

void cpSpecProbabilisticAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& draftProbabilities,
    rt::Tensor const& draftTokenIds, rt::Tensor const& proposalLengths, float const* acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride,
    int32_t verifyProposalLen, int32_t vocabSize, cudaStream_t stream)
{
    if (batchSize <= 0)
    {
        return;
    }
    check::check(cpSpecSupportsVocab(vocabSize), "cpSpec kernels require vocabSize <= kCpSpecMaxVocab");
    cpSpecProbabilisticAcceptKernel<<<batchSize, kBlockSize, 0, stream>>>(targetProbabilities.dataPointer<float>(),
        draftProbabilities.dataPointer<float>(), draftTokenIds.dataPointer<int32_t>(),
        proposalLengths.dataPointer<int32_t>(), acceptUniforms, acceptedTokenIds.dataPointer<int32_t>(),
        acceptLength.dataPointer<int32_t>(), draftStride, verifyProposalLen, vocabSize);
}

} // namespace kernel
} // namespace trt_edgellm
