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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>
#include <stdint.h>
#include <vector>

template <typename T>
__device__ __forceinline__ float to_float(T x)
{
    return (float) x;
}

template <>
__device__ __forceinline__ float to_float<half>(half x)
{
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x)
{
    return (T) x;
}
template <>
__device__ __forceinline__ half from_float<half>(float x)
{
    return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x)
{
    return __float2bfloat16_rn(x);
}
// Optional mask:
// - causal: disallow k > q
// - optional custom mask packed as [B, Sq, Sk] (broadcast over heads)
struct AttnMaskBS
{
    bool causal;
    uint8_t const* mask_bqk; // nullable, packed: ((b*Sq + q)*Sk + k)
};

__device__ __forceinline__ bool mask_allow(int64_t b, int64_t q, int64_t k, int64_t Sq, int64_t Sk, AttnMaskBS const& m)
{
    if (m.causal && (k > q))
    {
        return false;
    }
    if (m.mask_bqk)
    {
        int64_t idx = ((int64_t) b * Sq + q) * (int64_t) Sk + k;
        return m.mask_bqk[idx] != 0;
    }
    return true;
}

// BSHD indexing helper: (((b*S + s)*H + h)*D + d)
__device__ __forceinline__ int64_t idx_bshd(int64_t b, int64_t s, int64_t h, int64_t d, int64_t S, int64_t H, int64_t D)
{
    return ((b * S + s) * H + h) * D + d;
}

// Compact layout indexing helper: ((token*H + h)*D + d)
__device__ __forceinline__ int64_t idx_compact(int64_t token, int64_t h, int64_t d, int64_t H, int64_t D)
{
    return (token * H + h) * D + d;
}

// A very simple FMHA fwd kernel that only cares about simplicity and correctness.
template <typename Tqkv, typename To>
__global__ void fmha_reference(Tqkv const* __restrict__ Q, Tqkv const* __restrict__ K, Tqkv const* __restrict__ V,
    To* __restrict__ O, int64_t B, int64_t Sq, int64_t Sk, int64_t Hq, int64_t Hkv, int64_t D, AttnMaskBS mask,
    float attentionScale, bool compactLayout, int32_t const* __restrict__ cuSeqlens)
{
    extern __shared__ float s_scores[]; // size >= Sk floats
    int64_t const LIdx = blockIdx.y;
    int64_t const batchIdx = LIdx / Hq;
    int64_t const qHeadIdx = LIdx % Hq;

    int64_t const kvHeadIdx = (Hq == Hkv) ? qHeadIdx : (qHeadIdx * Hkv / Hq);

    int64_t seqStart = 0;
    int64_t seqLen = 0;
    int64_t qLoopEnd = 0;
    if (compactLayout)
    {
        seqStart = cuSeqlens[batchIdx];
        seqLen = cuSeqlens[batchIdx + 1] - seqStart;
        if (seqLen <= 0)
        {
            return;
        }
        qLoopEnd = seqLen;
    }
    else
    {
        seqStart = 0;
        seqLen = Sk;
        qLoopEnd = Sq;
    }

    if (batchIdx >= B)
    {
        return;
    }

    // Each CTA will loop through SQ dimension (Sq) in a grid-stride manner.
    int64_t const startQ = blockIdx.x;
    int64_t const strideQ = gridDim.x;

    for (int64_t qIdx = startQ; qIdx < qLoopEnd; qIdx += strideQ)
    {
        // 1) Each thread in the CTA will loop through SK dimension. This memory access pattern is bad
        // but we intentionally do this to make reference implementation very easy to understand.
        for (int kIdx = threadIdx.x; kIdx < seqLen; kIdx += blockDim.x)
        {
            float acc = 0.0f;

            int64_t const qBase = compactLayout ? idx_compact((seqStart + qIdx), qHeadIdx, 0, Hq, D)
                                                : idx_bshd(batchIdx, qIdx, qHeadIdx, 0, Sq, Hq, D);
            int64_t const kBase = compactLayout ? idx_compact((seqStart + kIdx), kvHeadIdx, 0, Hkv, D)
                                                : idx_bshd(batchIdx, kIdx, kvHeadIdx, 0, Sk, Hkv, D);

            for (int64_t d = 0; d < D; ++d)
            {
                acc += to_float(Q[qBase + d]) * to_float(K[kBase + d]);
            }

            int64_t const maskSq = compactLayout ? seqLen : Sq;
            int64_t const maskSk = compactLayout ? seqLen : Sk;
            if (!mask_allow(batchIdx, qIdx, kIdx, maskSq, maskSk, mask))
            {
                acc = -CUDART_INF_F;
            }
            s_scores[kIdx] = acc;
        }

        __syncthreads();

        // 2) max (reference-style: every thread loops all k)
        float maxS = -CUDART_INF_F;
        for (int64_t kIdx = 0; kIdx < seqLen; ++kIdx)
        {
            maxS = fmaxf(maxS, s_scores[kIdx]);
        }

        __syncthreads();

        // 3) exp
        for (int k = threadIdx.x; k < seqLen; k += blockDim.x)
        {
            float x = s_scores[k];
            s_scores[k] = __expf(attentionScale * (x - maxS));
        }

        __syncthreads();

        // 4) In a normal kernel we shall perform a reduction sum across the whole CTA.
        // but since we are doing a simple reference, let's let each thread to derive the complete sum for simplicity.
        float sum = 0.0f;
        for (int64_t kIdx = 0; kIdx < seqLen; ++kIdx)
        {
            sum += s_scores[kIdx];
        }
        float const inv_sum = 1.0f / sum;

        // 5) output
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x)
        {
            float acc = 0.0f;
            for (int64_t kIdx = 0; kIdx < seqLen; ++kIdx)
            {
                float const pk = s_scores[kIdx];
                int64_t const vBase = compactLayout ? idx_compact((seqStart + kIdx), kvHeadIdx, 0, Hkv, D)
                                                    : idx_bshd(batchIdx, kIdx, kvHeadIdx, 0, Sk, Hkv, D);
                acc += pk * to_float(V[vBase + d]);
            }
            int64_t const oIdx = compactLayout ? idx_compact((seqStart + qIdx), qHeadIdx, d, Hq, D)
                                               : idx_bshd(batchIdx, qIdx, qHeadIdx, d, Sq, Hq, D);
            O[oIdx] = from_float<To>(acc * inv_sum);
        }

        __syncthreads();
    }
}

// --- Launcher: QKV in, attention O out (BSHD layout) ---

template <typename Tqkv, typename To>
void launchFmhaReferenceTyped(Tqkv const* Q, Tqkv const* K, Tqkv const* V, To* O, int64_t B, int64_t Sq, int64_t Sk,
    int64_t Hq, int64_t Hkv, int64_t D, bool causal, bool compactLayout, float attentionScale, int32_t const* cuSeqlens,
    cudaStream_t stream)
{
    AttnMaskBS mask;
    mask.causal = causal;
    mask.mask_bqk = nullptr;

    int64_t const numBlocksY = B * Hq;
    int64_t const numBlocksX = (Sq < 1024) ? Sq : 1024;

    // Use a fixed block size of 64 for simplicity.
    int const blockSize = D == 64 ? 64 : 128;
    size_t const smemBytes = (size_t) Sk * sizeof(float);

    // We are able to test though context length till 12K, should be enough for most cases. Let's just bound it for
    // simplicity.
    if (smemBytes > 48 * 1024)
    {
        throw std::runtime_error("The KV Length is too long which exceeds the shared memory limit of 48KB.");
    }

    dim3 grid((unsigned int) numBlocksX, (unsigned int) numBlocksY);
    dim3 block((unsigned int) blockSize);

    fmha_reference<Tqkv, To><<<grid, block, smemBytes, stream>>>(
        Q, K, V, O, B, Sq, Sk, Hq, Hkv, D, mask, attentionScale, compactLayout, cuSeqlens);
}

// --- Tensor-based launcher ---

namespace trt_edgellm
{
namespace rt
{

void launchFmhaReferenceImpl(Tensor const& Q, Tensor const& K, Tensor const& V, Tensor& O, bool causal,
    bool compactLayout, int64_t B, int64_t Sq, int64_t Sk, int64_t Hq, int64_t Hkv, int64_t D, float attentionScale,
    int32_t const* cuSeqlensPtr, cudaStream_t stream)
{
    nvinfer1::DataType const dtype = Q.getDataType();
    if (dtype != K.getDataType() || dtype != V.getDataType() || dtype != O.getDataType())
    {
        throw std::runtime_error("launchFmhaReference: Q, K, V, O must have the same data type.");
    }

    switch (dtype)
    {
    case nvinfer1::DataType::kHALF:
        launchFmhaReferenceTyped<__half, __half>(Q.dataPointer<__half>(), K.dataPointer<__half>(),
            V.dataPointer<__half>(), O.dataPointer<__half>(), B, Sq, Sk, Hq, Hkv, D, causal, compactLayout,
            attentionScale, cuSeqlensPtr, stream);
        break;
    case nvinfer1::DataType::kBF16:
        launchFmhaReferenceTyped<__nv_bfloat16, __nv_bfloat16>(Q.dataPointer<__nv_bfloat16>(),
            K.dataPointer<__nv_bfloat16>(), V.dataPointer<__nv_bfloat16>(), O.dataPointer<__nv_bfloat16>(), B, Sq, Sk,
            Hq, Hkv, D, causal, compactLayout, attentionScale, cuSeqlensPtr, stream);
        break;
    default: throw std::runtime_error("launchFmhaReference: unsupported data type (use kHALF, or kBF16).");
    }
}

void launchFmhaReferenceBshd(Tensor const& Q, Tensor const& K, Tensor const& V, Tensor& O, bool causal,
    float attentionScale, cudaStream_t stream)
{
    Coords const qShape = Q.getShape();
    Coords const kShape = K.getShape();
    Coords const vShape = V.getShape();
    Coords const oShape = O.getShape();

    if (qShape.getNumDims() != 4 || kShape.getNumDims() != 4 || vShape.getNumDims() != 4 || oShape.getNumDims() != 4)
    {
        throw std::runtime_error("launchFmhaReferenceBshd: Q, K, V, O must be 4D (BSHD).");
    }

    int64_t const B = qShape[0];
    int64_t const Sq = qShape[1];
    int64_t const Hq = qShape[2];
    int64_t const D = qShape[3];

    if (kShape[0] != B || vShape[0] != B || oShape[0] != B)
    {
        throw std::runtime_error("launchFmhaReferenceBshd: batch size mismatch.");
    }
    int64_t const Sk = kShape[1];

    if (vShape[1] != Sk)
    {
        throw std::runtime_error("launchFmhaReferenceBshd: K and V sequence length mismatch.");
    }
    int64_t const Hkv = kShape[2];
    if (kShape[3] != D || vShape[3] != D)
    {
        throw std::runtime_error("launchFmhaReferenceBshd: head dim D mismatch between Q and K/V.");
    }
    if (oShape[1] != Sq || oShape[2] != Hq || oShape[3] != D)
    {
        throw std::runtime_error("launchFmhaReferenceBshd: O shape must be [B, Sq, Hq, D].");
    }

    launchFmhaReferenceImpl(Q, K, V, O, causal, false, B, Sq, Sk, Hq, Hkv, D, attentionScale, nullptr, stream);
}

void launchFmhaReferenceCompact(Tensor const& Q, Tensor const& K, Tensor const& V, Tensor& O, Tensor const& cuSeqlens,
    int32_t maxSeqLen, bool causal, float attentionScale, cudaStream_t stream)
{
    Coords const qShape = Q.getShape();
    Coords const kShape = K.getShape();
    Coords const vShape = V.getShape();
    Coords const oShape = O.getShape();

    if (qShape.getNumDims() != 3 || kShape.getNumDims() != 3 || vShape.getNumDims() != 3 || oShape.getNumDims() != 3)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: Q/K/V/O must be 3D.");
    }

    Coords const cuShape = cuSeqlens.getShape();
    if (cuShape.getNumDims() != 1 || cuShape[0] < 2)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: cuSeqlens must be 1D [B+1].");
    }

    if (maxSeqLen <= 0)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: maxSeqLen must be positive.");
    }

    int32_t const totalTokens = static_cast<int32_t>(qShape[0]);
    int64_t const Hq = qShape[1];
    int64_t const Hkv = kShape[1];
    int64_t const D = qShape[2];
    if (kShape[0] != totalTokens || vShape[0] != totalTokens || oShape[0] != totalTokens)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: total_tokens mismatch.");
    }
    if (kShape[2] != D || vShape[2] != D)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: head dim D mismatch between Q and K/V.");
    }
    if (oShape[1] != Hq || oShape[2] != D)
    {
        throw std::runtime_error("launchFmhaReferenceCompact: O shape must be [total_tokens, Hq, D].");
    }

    int64_t const B = cuShape[0] - 1;

    launchFmhaReferenceImpl(Q, K, V, O, causal, true, B, maxSeqLen, maxSeqLen, Hq, Hkv, D, attentionScale,
        cuSeqlens.dataPointer<int32_t>(), stream);
}

} // namespace rt
} // namespace trt_edgellm
