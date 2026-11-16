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

// =============================================================================
// Gemma 4 audio-encoder attention body  (post-QKV, pre-output-projection).
//
// MATH (per chunk n, head h, batch b; query a in [0,C), context slot m in [0,M)):
//
//   Index map:   i = n*C + a            query position in the sequence
//                j = n*C - L + m        key/value position contributed by slot m
//
//   1. Scaling   (per-dim learned query scale + fixed key scale)
//        Q[i,r] = qRaw[i,r] * (d^-1/2 / ln2) * softplus(gamma[r])
//        K[j,r] = kRaw[j,r] * (ln(1+e) / ln2)
//
//   2. Content score (Q.Kᵀ over the head dim r):
//        AC[a,m] = sum_r Q[i,r] * K[j,r]
//
//   3. Relative-position score (Q.Pᵀ then Transformer-XL blocked shift):
//        R[a,t]  = sum_r Q[i,r] * P[t,r],          t in [0,P)        (relKey)
//        BD[a,m] = R[srcQuery, srcRelPos],  with  p = a*M + m,
//                  srcQuery = p / (M+1),  srcRelPos = p % (M+1)       (rel_shift)
//                  (= pad R to width M+1, flatten, slice C*M, reshape [C,M])
//
//   4. Soft-cap:   S[a,m] = cap * tanh( (AC[a,m] + BD[a,m]) / cap )
//
//   5. Local-causal + padding mask:
//        allowed(a,m) = (0 <= i-j < L) and (0 <= j < S) and (i < S) and valid[b,j]
//        S[a,m] = allowed ? S[a,m] : -1e9          (finite -> dead rows -> uniform)
//
//   6. Softmax over the M context slots (fp32):
//        A[a,m] = exp(S[a,m] - max_m S) / sum_m exp(S[a,m] - max_m S)
//
//   7. Value mix:  O[i,r] = sum_m A[a,m] * V[j,r]        (V unscaled)
//
// One CTA per (b,h,n) chunk: scaled Q/K and V/P_h are staged in shared memory and
// the three per-chunk GEMMs plus rel_shift / soft-cap / mask / softmax / value-mix
// are fully fused. Shared loads are 2-wide vectorized; the content dot is skipped
// for masked slots; shape constants are constexpr so rel_shift div/mod fold to
// mul-shift and the d=128 loops fully unroll.
//
// Specialized to the Gemma 4 audio config (C=12, L=12, M=24, P=13, d=128). T is
// half, __nv_bfloat16, or float. Scores and softmax accumulate in fp32; the float
// path keeps full precision throughout (matches the HF fp32 reference).
// =============================================================================

#include "gemma4AudioAttention.h"

#include "common/checkMacros.h"

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

float constexpr kLn2 = 0.6931471805599453f;
float constexpr kNegInfFill = -1e9f; // finite -> a fully-masked (dead) row softmaxes to uniform

//! \brief Maps the element type to its 2-wide vector type for vectorized shared loads.
template <typename T>
struct Vec2;
template <>
struct Vec2<half>
{
    using type = __half2;
};
template <>
struct Vec2<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};
template <>
struct Vec2<float>
{
    using type = float2;
};

//! \brief Convert a 2-wide vector element to float2 (one code path serves fp16/bf16/fp32).
__device__ __forceinline__ float2 vec2ToFloat2(__half2 x)
{
    return __half22float2(x);
}
__device__ __forceinline__ float2 vec2ToFloat2(__nv_bfloat162 x)
{
    return __bfloat1622float2(x);
}
__device__ __forceinline__ float2 vec2ToFloat2(float2 x)
{
    return x;
}

//! \brief softplus(x) = ln(1 + e^x), guarded against overflow for large x.
__host__ __device__ inline float softplus(float x)
{
    return x > 20.f ? x : log1pf(expf(x));
}

//! \brief Flat offset into a [B,S,H,D] tensor (row-major).
__device__ __forceinline__ int idxBSHD(int b, int s, int h, int dim, int S, int H, int D)
{
    return ((b * S + s) * H + h) * D + dim;
}

//! \brief Flat offset into the relKey [P,H,D] tensor (row-major).
__device__ __forceinline__ int idxPHD(int relPos, int h, int dim, int H, int D)
{
    return (relPos * H + h) * D + dim;
}

//! \brief One CTA computes the full attention output for one (batch, head, chunk).
//!
//! Shared memory layout (fp32 scratch first so the T region stays aligned):
//!   [ softplusGamma : D | relRawScores : C*P | chunkScores : C*M ]
//!   then  [ qShared : C*LDS | kShared : M*LDS | vShared : M*LDS | pShared : P*LDS ]
//! LDS = D+2 padded row stride (avoids bank conflicts; even, so each row base stays
//! vec2-aligned). chunkScores holds capped/masked logits, then probabilities.
template <typename T, typename Vec2T>
__global__ void fusedChunkKernel(T const* __restrict__ qRaw, T const* __restrict__ kRaw, T const* __restrict__ v,
    float const* __restrict__ gamma, T const* __restrict__ relKey, bool const* __restrict__ valid, T* __restrict__ out,
    int B, int S, int H, int numChunks, float qScalar, float kScale, float cap)
{
    int constexpr C = 12, M = 24, P = 13, L = 12, D = 128; // Gemma 4 audio defaults
    int constexpr LDS = D + 2;                             // padded shared row stride
    int constexpr HALF_D = D / 2;                          // head dim counted in 2-wide vectors

    // Decode this CTA's (batch, head, chunk) from the linear grid index.
    int const linearIdx = blockIdx.x;
    int const chunk = linearIdx % numChunks;
    int const head = (linearIdx / numChunks) % H;
    int const batch = linearIdx / (numChunks * H);
    int const threadId = threadIdx.x;
    int const numThreads = blockDim.x;

    extern __shared__ char smemRaw[];
    float* softplusGamma = reinterpret_cast<float*>(smemRaw); // [D]
    float* relRawScores = softplusGamma + D;                  // [C*P]
    float* chunkScores = relRawScores + C * P;                // [C*M]
    T* qShared = reinterpret_cast<T*>(chunkScores + C * M);   // [C*LDS]
    T* kShared = qShared + C * LDS;                           // [M*LDS]
    T* vShared = kShared + M * LDS;                           // [M*LDS]
    T* pShared = vShared + M * LDS;                           // [P*LDS]

    // Precompute softplus(gamma[dim]) once (reused by every query row).
    for (int dim = threadId; dim < D; dim += numThreads)
    {
        softplusGamma[dim] = softplus(gamma[dim]);
    }
    __syncthreads();

    // --- Stage 1: stage scaled Q, scaled K, V, and P_h into shared memory --------
    // Q[i,dim] = qRaw[i,dim] * (d^-1/2 / ln2) * softplus(gamma[dim]); out-of-range -> 0.
    for (int elem = threadId; elem < C * D; elem += numThreads)
    {
        int const queryInChunk = elem / D, dim = elem % D;
        int const queryPos = chunk * C + queryInChunk;
        float const scaledQ = (queryPos < S)
            ? static_cast<float>(qRaw[idxBSHD(batch, queryPos, head, dim, S, H, D)]) * qScalar * softplusGamma[dim]
            : 0.f;
        qShared[queryInChunk * LDS + dim] = static_cast<T>(scaledQ);
    }
    // K[j,dim] = kRaw[j,dim] * (ln(1+e) / ln2); V[j,dim] = vRaw[j,dim]; j = n*C - L + m.
    for (int elem = threadId; elem < M * D; elem += numThreads)
    {
        int const ctxSlot = elem / D, dim = elem % D;
        int const keyPos = chunk * C - L + ctxSlot;
        bool const keyInSeq = (keyPos >= 0 && keyPos < S);
        float const scaledK
            = keyInSeq ? static_cast<float>(kRaw[idxBSHD(batch, keyPos, head, dim, S, H, D)]) * kScale : 0.f;
        kShared[ctxSlot * LDS + dim] = static_cast<T>(scaledK);
        vShared[ctxSlot * LDS + dim] = keyInSeq ? v[idxBSHD(batch, keyPos, head, dim, S, H, D)] : static_cast<T>(0.f);
    }
    // P_h[t,dim] = relKey[t,head,dim] (relative-position keys for this head).
    for (int elem = threadId; elem < P * D; elem += numThreads)
    {
        int const relPos = elem / D, dim = elem % D;
        pShared[relPos * LDS + dim] = relKey[idxPHD(relPos, head, dim, H, D)];
    }
    __syncthreads();

    // --- Stage 3a: relative raw scores  R[a,t] = sum_dim Q[i,dim] * P_h[t,dim] ----
    // (fp32 accumulate; 2-wide vectorized). Computed in full because rel_shift below
    // gathers R[srcQuery,srcRelPos], whose srcQuery may differ from the current query.
    for (int elem = threadId; elem < C * P; elem += numThreads)
    {
        int const queryInChunk = elem / P, relPos = elem % P;
        Vec2T const* queryRow = reinterpret_cast<Vec2T const*>(qShared + queryInChunk * LDS);
        Vec2T const* relPosRow = reinterpret_cast<Vec2T const*>(pShared + relPos * LDS);
        float relDot = 0.f;
#pragma unroll
        for (int dim2 = 0; dim2 < HALF_D; ++dim2)
        {
            float2 const qVec = vec2ToFloat2(queryRow[dim2]), pVec = vec2ToFloat2(relPosRow[dim2]);
            relDot += qVec.x * pVec.x + qVec.y * pVec.y;
        }
        relRawScores[elem] = relDot;
    }
    __syncthreads();

    // --- Stages 2,3b,4,5: content score + rel_shift + soft-cap + mask -> logit ----
    // The local-causal + padding mask is cheap, so evaluate it first and skip the
    // (expensive) content dot for masked slots.
    for (int elem = threadId; elem < C * M; elem += numThreads)
    {
        int const queryInChunk = elem / M, ctxSlot = elem % M;
        int const queryPos = chunk * C + queryInChunk;
        int const keyPos = chunk * C - L + ctxSlot;
        int const queryKeyDist = queryPos - keyPos; // = queryInChunk + L - ctxSlot
        bool const inLocalWindow = (queryKeyDist >= 0) && (queryKeyDist < L);
        bool const keyInSeq = (keyPos >= 0) && (keyPos < S);
        bool const queryInSeq = (queryPos < S);
        bool const allowed = inLocalWindow && keyInSeq && queryInSeq && valid[batch * S + keyPos];

        float logit = kNegInfFill;
        if (allowed)
        {
            // Content score AC[a,m] = sum_dim Q[i,dim] * K[j,dim].
            Vec2T const* queryRow = reinterpret_cast<Vec2T const*>(qShared + queryInChunk * LDS);
            Vec2T const* keyRow = reinterpret_cast<Vec2T const*>(kShared + ctxSlot * LDS);
            float contentScore = 0.f;
#pragma unroll
            for (int dim2 = 0; dim2 < HALF_D; ++dim2)
            {
                float2 const qVec = vec2ToFloat2(queryRow[dim2]), kVec = vec2ToFloat2(keyRow[dim2]);
                contentScore += qVec.x * kVec.x + qVec.y * kVec.y;
            }
            // Relative score BD[a,m] = R[srcQuery,srcRelPos] via the Transformer-XL blocked
            // shift; an out-of-range source (the zero-pad region) contributes 0.
            int const shiftFlat = queryInChunk * M + ctxSlot;
            int const srcQuery = shiftFlat / (M + 1), srcRelPos = shiftFlat % (M + 1); // (M+1) constexpr -> mul/shift
            float const relScore = (srcQuery < C && srcRelPos < P) ? relRawScores[srcQuery * P + srcRelPos] : 0.f;
            // Soft-cap: S = cap * tanh((AC + BD) / cap).
            logit = cap * tanhf((contentScore + relScore) / cap);
        }
        chunkScores[elem] = logit;
    }
    __syncthreads();

    // --- Stage 6: softmax over the M context slots (fp32), one thread per query row -----
    for (int queryInChunk = threadId; queryInChunk < C; queryInChunk += numThreads)
    {
        float* rowScores = chunkScores + queryInChunk * M;
        float rowMax = -INFINITY;
        for (int ctxSlot = 0; ctxSlot < M; ++ctxSlot)
        {
            rowMax = fmaxf(rowMax, rowScores[ctxSlot]); // numerical stability
        }
        float rowSum = 0.f;
        for (int ctxSlot = 0; ctxSlot < M; ++ctxSlot)
        {
            float const expVal = expf(rowScores[ctxSlot] - rowMax);
            rowScores[ctxSlot] = expVal;
            rowSum += expVal;
        }
        float const invRowSum = 1.f / rowSum;
        for (int ctxSlot = 0; ctxSlot < M; ++ctxSlot)
        {
            rowScores[ctxSlot] *= invRowSum; // probabilities A[a,m]
        }
    }
    __syncthreads();

    // --- Stage 7: value mix  O[i,dim] = sum_m A[a,m] * V[j,dim] -------------------
    // Vectorized over the head dim: each thread produces two adjacent outputs (dim, dim+1).
    for (int elem = threadId; elem < C * HALF_D; elem += numThreads)
    {
        int const queryInChunk = elem / HALF_D, dim2 = elem % HALF_D;
        int const queryPos = chunk * C + queryInChunk;
        if (queryPos >= S)
        {
            continue; // skip padding query rows (cropped on output)
        }
        float const* probsRow = chunkScores + queryInChunk * M;
        float outVal0 = 0.f, outVal1 = 0.f;
        for (int ctxSlot = 0; ctxSlot < M; ++ctxSlot)
        {
            float2 const vVec = vec2ToFloat2(reinterpret_cast<Vec2T const*>(vShared + ctxSlot * LDS)[dim2]);
            outVal0 += probsRow[ctxSlot] * vVec.x;
            outVal1 += probsRow[ctxSlot] * vVec.y;
        }
        int const dim = 2 * dim2;
        out[idxBSHD(batch, queryPos, head, dim, S, H, D)] = static_cast<T>(outVal0);
        out[idxBSHD(batch, queryPos, head, dim + 1, S, H, D)] = static_cast<T>(outVal1);
    }
}

//! \brief Type-specialized launch: scaling constants, grid/shared-memory sizing, kernel launch.
//! Shapes are passed in already resolved from the tensor extents.
template <typename T>
void launchGemma4AudioAttention(T const* qRaw, T const* kRaw, T const* v, float const* gamma, T const* relKey,
    bool const* valid, T* out, int B, int S, int H, int D, int C, int M, int P, float cap, cudaStream_t stream)
{
    int const numChunks = (S + C - 1) / C;

    // qScalar = d^-1/2 / ln2 ;  kScale = ln(1+e) / ln2  (Gemma 4 audio Q/K scaling).
    float const qScalar = static_cast<float>(std::pow(static_cast<double>(D), -0.5) / static_cast<double>(kLn2));
    float const kScale = static_cast<float>(std::log1p(std::exp(1.0)) / static_cast<double>(kLn2));

    int const threads = 128;
    int const grid = B * H * numChunks; // one CTA per (b,h,n) chunk
    int const LDS = D + 2;
    size_t const floatBytes
        = (static_cast<size_t>(D) + static_cast<size_t>(C) * P + static_cast<size_t>(C) * M) * sizeof(float);
    size_t const stageBytes
        = (static_cast<size_t>(C) * LDS + 2 * static_cast<size_t>(M) * LDS + static_cast<size_t>(P) * LDS) * sizeof(T);
    size_t const shmem = floatBytes + stageBytes;

    using Vec2T = typename Vec2<T>::type;
    fusedChunkKernel<T, Vec2T><<<grid, threads, shmem, stream>>>(
        qRaw, kRaw, v, gamma, relKey, valid, out, B, S, H, numChunks, qScalar, kScale, cap);
}

} // namespace

void gemma4AudioAttentionForward(rt::Tensor const& qRaw, rt::Tensor const& kRaw, rt::Tensor const& v,
    rt::Tensor const& gamma, rt::Tensor const& relKey, rt::Tensor const& valid, rt::Tensor& out,
    Gemma4AudioAttentionParams const& params, cudaStream_t stream)
{
    using trt_edgellm::DataType;

    // Element type and shapes are carried by the tensors. q/k/v/relKey/out share one floating type.
    DataType const dtype = qRaw.getDataType();
    check::check(dtype == DataType::kHALF || dtype == DataType::kBF16 || dtype == DataType::kFLOAT,
        "gemma4AudioAttention: qRaw must be half, bfloat16, or float");
    check::check(kRaw.getDataType() == dtype && v.getDataType() == dtype && relKey.getDataType() == dtype
            && out.getDataType() == dtype,
        "gemma4AudioAttention: qRaw/kRaw/v/relKey/out must share the same element type");
    check::check(gamma.getDataType() == DataType::kFLOAT, "gemma4AudioAttention: gamma must be float");
    check::check(valid.getDataType() == DataType::kBOOL, "gemma4AudioAttention: valid must be bool");

    // q/k/v/out are [B, S, H, D]; relKey is [P, H, D]; gamma is [D]; valid is [B, S].
    auto const qShape = qRaw.getShape();
    check::check(qShape.getNumDims() == 4, "gemma4AudioAttention: qRaw must be 4D [B, S, H, D]");
    int const B = static_cast<int>(qShape[0]);
    int const S = static_cast<int>(qShape[1]);
    int const H = static_cast<int>(qShape[2]);
    int const D = static_cast<int>(qShape[3]);
    int const C = params.chunkSize, L = params.leftHorizon, M = params.contextSize;

    check::check(kRaw.getShape() == qShape && v.getShape() == qShape && out.getShape() == qShape,
        "gemma4AudioAttention: kRaw/v/out must match qRaw shape [B, S, H, D]");
    auto const relShape = relKey.getShape();
    check::check(relShape.getNumDims() == 3 && relShape[1] == H && relShape[2] == D,
        "gemma4AudioAttention: relKey must be [P, H, D] with H/D matching qRaw");
    int const P = static_cast<int>(relShape[0]);
    check::check(gamma.getShape() == rt::Coords{D}, "gemma4AudioAttention: gamma must be [D]");
    check::check(valid.getShape() == rt::Coords{B, S}, "gemma4AudioAttention: valid must be [B, S]");

    // The kernel is specialized (constexpr) to the Gemma 4 audio config; reject other shapes.
    check::check(D == 128 && P == 13 && C == 12 && L == 12 && M == 24,
        "gemma4AudioAttention: specialized to D=128, P=13, chunkSize=12, leftHorizon=12, contextSize=24");

    bool const* validPtr = valid.dataPointer<bool>();
    float const* gammaPtr = gamma.dataPointer<float>();
    float const cap = params.logitCap;

    if (dtype == DataType::kHALF)
    {
        launchGemma4AudioAttention<half>(qRaw.dataPointer<half>(), kRaw.dataPointer<half>(), v.dataPointer<half>(),
            gammaPtr, relKey.dataPointer<half>(), validPtr, out.dataPointer<half>(), B, S, H, D, C, M, P, cap, stream);
    }
    else if (dtype == DataType::kBF16)
    {
        launchGemma4AudioAttention<__nv_bfloat16>(qRaw.dataPointer<__nv_bfloat16>(), kRaw.dataPointer<__nv_bfloat16>(),
            v.dataPointer<__nv_bfloat16>(), gammaPtr, relKey.dataPointer<__nv_bfloat16>(), validPtr,
            out.dataPointer<__nv_bfloat16>(), B, S, H, D, C, M, P, cap, stream);
    }
    else // DataType::kFLOAT
    {
        launchGemma4AudioAttention<float>(qRaw.dataPointer<float>(), kRaw.dataPointer<float>(), v.dataPointer<float>(),
            gammaPtr, relKey.dataPointer<float>(), validPtr, out.dataPointer<float>(), B, S, H, D, C, M, P, cap,
            stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
