/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vitAttentionRunner.h"

#include "common/checkMacros.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

template <typename T>
__device__ __forceinline__ float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value)
{
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value)
{
    return __float2half(value);
}

__device__ __forceinline__ float blockReduceMax(float value)
{
    extern __shared__ float shared[];
    int32_t const tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ float blockReduceSum(float value)
{
    extern __shared__ float shared[];
    int32_t const tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__device__ __forceinline__ float ropeValue(T const* tensor, T const* rope, int64_t base, int32_t pos, int32_t dim,
    int32_t headSize, bool useSin)
{
    int32_t const halfSize = headSize / 2;
    float const value = toFloat(tensor[base + dim]);
    int32_t const rotatedDim = dim < halfSize ? dim + halfSize : dim - halfSize;
    float const rotated = dim < halfSize ? -toFloat(tensor[base + rotatedDim]) : toFloat(tensor[base + rotatedDim]);
    float const ropeComponent = toFloat(rope[static_cast<int64_t>(pos) * headSize + dim]);
    return useSin ? rotated * ropeComponent : value * ropeComponent;
}

template <typename T>
__device__ __forceinline__ float applyRoPE(T const* tensor, T const* cos, T const* sin, int64_t base, int32_t pos,
    int32_t dim, int32_t headSize)
{
    return ropeValue(tensor, cos, base, pos, dim, headSize, false)
        + ropeValue(tensor, sin, base, pos, dim, headSize, true);
}

__device__ __forceinline__ int32_t getMaskRow(int32_t batchIdx, int32_t headIdx, int32_t batchSize, int32_t numHeads,
    int32_t maskRows)
{
    if (maskRows == 1)
    {
        return 0;
    }
    if (maskRows == batchSize)
    {
        return batchIdx;
    }
    if (maskRows == batchSize * numHeads)
    {
        return batchIdx * numHeads + headIdx;
    }
    return 0;
}

template <typename T>
__global__ void precomputeViTRoPEQKKernel(
    T const* qkv, T const* cos, T const* sin, float* qRope, float* kRope, int32_t totalElems, int32_t seqLen,
    int32_t numHeads, int32_t headSize)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElems)
    {
        return;
    }

    int32_t const dim = idx % headSize;
    int32_t const headIdx = (idx / headSize) % numHeads;
    int32_t const tokenIdx = (idx / (headSize * numHeads)) % seqLen;
    int32_t const batchIdx = idx / (headSize * numHeads * seqLen);

    int32_t const hiddenSize = numHeads * headSize;
    int64_t const tokenBase
        = (static_cast<int64_t>(batchIdx) * seqLen + tokenIdx) * 3 * hiddenSize + headIdx * headSize;
    int64_t const qBase = tokenBase;
    int64_t const kBase = tokenBase + hiddenSize;
    int64_t const ropeBase
        = ((static_cast<int64_t>(batchIdx) * numHeads + headIdx) * seqLen + tokenIdx) * headSize + dim;

    qRope[ropeBase] = applyRoPE(qkv, cos, sin, qBase, tokenIdx, dim, headSize);
    kRope[ropeBase] = applyRoPE(qkv, cos, sin, kBase, tokenIdx, dim, headSize);
}

template <typename T>
__global__ void computeViTAttentionFusedOutputKernel(float const* qRope, float const* kRope, T const* qkv,
    T const* attentionMask, T* output, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize,
    int32_t maskRows, float scale)
{
    int32_t const row = blockIdx.x;
    int32_t const qIdx = row % seqLen;
    int32_t const headIdx = (row / seqLen) % numHeads;
    int32_t const batchIdx = row / (seqLen * numHeads);

    extern __shared__ float shared[];
    float* scores = shared;
    float* reductions = shared + seqLen;

    int32_t const hiddenSize = numHeads * headSize;
    int64_t const qBase = ((static_cast<int64_t>(batchIdx) * numHeads + headIdx) * seqLen + qIdx) * headSize;
    int64_t const qkvBatchBase = static_cast<int64_t>(batchIdx) * seqLen * 3 * hiddenSize;
    int64_t const outputBase = (static_cast<int64_t>(batchIdx) * seqLen + qIdx) * hiddenSize + headIdx * headSize;
    int32_t const maskRow = getMaskRow(batchIdx, headIdx, batchSize, numHeads, maskRows);
    int64_t const maskBase = static_cast<int64_t>(maskRow) * seqLen * seqLen + static_cast<int64_t>(qIdx) * seqLen;

    float localMax = -INFINITY;
    for (int32_t kIdx = threadIdx.x; kIdx < seqLen; kIdx += blockDim.x)
    {
        int64_t const kBase = ((static_cast<int64_t>(batchIdx) * numHeads + headIdx) * seqLen + kIdx) * headSize;
        float dot = 0.0F;
        for (int32_t dim = 0; dim < headSize; ++dim)
        {
            dot += qRope[qBase + dim] * kRope[kBase + dim];
        }
        float score = dot * scale + toFloat(attentionMask[maskBase + kIdx]);
        scores[kIdx] = score;
        localMax = fmaxf(localMax, score);
    }

    reductions[threadIdx.x] = localMax;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float const rowMax = reductions[0];

    float localSum = 0.0F;
    for (int32_t kIdx = threadIdx.x; kIdx < seqLen; kIdx += blockDim.x)
    {
        float prob = expf(scores[kIdx] - rowMax);
        scores[kIdx] = prob;
        localSum += prob;
    }

    reductions[threadIdx.x] = localSum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const rowSum = reductions[0];
    float const invRowSum = 1.0F / rowSum;

    for (int32_t kIdx = threadIdx.x; kIdx < seqLen; kIdx += blockDim.x)
    {
        scores[kIdx] *= invRowSum;
    }
    __syncthreads();

    for (int32_t dim = threadIdx.x; dim < headSize; dim += blockDim.x)
    {
        float value = 0.0F;
        for (int32_t kIdx = 0; kIdx < seqLen; ++kIdx)
        {
            int64_t const vBase
                = qkvBatchBase + static_cast<int64_t>(kIdx) * 3 * hiddenSize + 2 * hiddenSize + headIdx * headSize;
            value += scores[kIdx] * toFloat(qkv[vBase + dim]);
        }
        output[outputBase + dim] = fromFloat<T>(value);
    }
}

template <typename T>
void launchViTAttentionTyped(T const* qkv, T const* cos, T const* sin, T const* attentionMask, T* output,
    float* softmaxWorkspace, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize, int32_t maskRows,
    cudaStream_t stream)
{
    constexpr int32_t kSoftmaxBlockSize = 256;
    constexpr int32_t kOutputBlockSize = 256;
    int32_t const rows = batchSize * numHeads * seqLen;
    int32_t const totalOutputElems = batchSize * seqLen * numHeads * headSize;
    int32_t const totalRopeElems = totalOutputElems;
    float* qRope = softmaxWorkspace;
    float* kRope = qRope + static_cast<size_t>(totalRopeElems);
    float const scale = 1.0F / std::sqrt(static_cast<float>(headSize));
    size_t const sharedBytes = (static_cast<size_t>(seqLen) + kSoftmaxBlockSize) * sizeof(float);

    int32_t const ropeGrid = (totalRopeElems + kOutputBlockSize - 1) / kOutputBlockSize;
    precomputeViTRoPEQKKernel<T><<<ropeGrid, kOutputBlockSize, 0, stream>>>(
        qkv, cos, sin, qRope, kRope, totalRopeElems, seqLen, numHeads, headSize);

    computeViTAttentionFusedOutputKernel<T><<<rows, kSoftmaxBlockSize, sharedBytes, stream>>>(
        qRope, kRope, qkv, attentionMask, output, batchSize, seqLen, numHeads, headSize, maskRows, scale);
}

} // namespace

void launchViTAttention(nvinfer1::DataType dataType, void const* qkv, void const* cos, void const* sin,
    void const* attentionMask, void* output, float* softmaxWorkspace, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headSize, int32_t maskRows, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        launchViTAttentionTyped<half>(static_cast<half const*>(qkv), static_cast<half const*>(cos),
            static_cast<half const*>(sin), static_cast<half const*>(attentionMask), static_cast<half*>(output),
            softmaxWorkspace, batchSize, seqLen, numHeads, headSize, maskRows, stream);
    }
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
        launchViTAttentionTyped<float>(static_cast<float const*>(qkv), static_cast<float const*>(cos),
            static_cast<float const*>(sin), static_cast<float const*>(attentionMask), static_cast<float*>(output),
            softmaxWorkspace, batchSize, seqLen, numHeads, headSize, maskRows, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for ViT attention kernel.");
    }
}

} // namespace kernel
} // namespace trt_edgellm
