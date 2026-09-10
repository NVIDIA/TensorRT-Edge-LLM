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
#include "dartSelectorKernels.h"
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kBlockSize = 256;

//! Warp-then-shared block reduction of two float accumulators.
__device__ void blockReduceSum2(float& a, float& b)
{
    __shared__ float smemA[kBlockSize / 32];
    __shared__ float smemB[kBlockSize / 32];
    int32_t const lane = threadIdx.x % 32;
    int32_t const warp = threadIdx.x / 32;
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2)
    {
        a += __shfl_down_sync(0xFFFFFFFFU, a, offset);
        b += __shfl_down_sync(0xFFFFFFFFU, b, offset);
    }
    if (lane == 0)
    {
        smemA[warp] = a;
        smemB[warp] = b;
    }
    __syncthreads();
    if (warp == 0)
    {
        constexpr int32_t kNumWarps = kBlockSize / 32;
        a = (lane < kNumWarps) ? smemA[lane] : 0.0F;
        b = (lane < kNumWarps) ? smemB[lane] : 0.0F;
#pragma unroll
        for (int32_t offset = 16; offset > 0; offset /= 2)
        {
            a += __shfl_down_sync(0xFFFFFFFFU, a, offset);
            b += __shfl_down_sync(0xFFFFFFFFU, b, offset);
        }
    }
}

//! One block per row: norms[row] = {L1, L2} of embeds[row, :].
__global__ void rowNormsKernel(half const* embeds, float* norms, int32_t hiddenSize)
{
    int64_t const row = blockIdx.x;
    half const* rowPtr = embeds + row * hiddenSize;
    float l1 = 0.0F;
    float sq = 0.0F;
    for (int32_t h = threadIdx.x; h < hiddenSize; h += blockDim.x)
    {
        float const v = __half2float(rowPtr[h]);
        l1 += fabsf(v);
        sq += v * v;
    }
    blockReduceSum2(l1, sq);
    if (threadIdx.x == 0)
    {
        norms[row * 2 + 0] = l1;
        norms[row * 2 + 1] = sqrtf(sq);
    }
}

//! Grid (numRows, numPivots); one block computes dot(embeds[pivot], embeds[row]).
__global__ void pivotDotsKernel(
    half const* embeds, int32_t const* pivotIndices, float* dots, int32_t hiddenSize, int32_t numRows)
{
    int64_t const row = blockIdx.x;
    int32_t const pivotSlot = blockIdx.y;
    half const* rowPtr = embeds + row * hiddenSize;
    half const* pivotPtr = embeds + static_cast<int64_t>(pivotIndices[pivotSlot]) * hiddenSize;
    float dot = 0.0F;
    float unused = 0.0F;
    for (int32_t h = threadIdx.x; h < hiddenSize; h += blockDim.x)
    {
        dot += __half2float(pivotPtr[h]) * __half2float(rowPtr[h]);
    }
    blockReduceSum2(dot, unused);
    if (threadIdx.x == 0)
    {
        dots[static_cast<int64_t>(pivotSlot) * numRows + row] = dot;
    }
}

} // namespace

void computeDartRowNorms(rt::Tensor const& embeds, rt::Tensor& norms, cudaStream_t stream)
{
    check::check(embeds.getDataType() == nvinfer1::DataType::kHALF, "DART selector expects FP16 embeddings");
    check::check(embeds.getShape().getNumDims() == 2, "embeds must be [numRows, hiddenSize]");
    check::check(norms.getDataType() == nvinfer1::DataType::kFLOAT, "norms must be FP32");
    int64_t const numRows = embeds.getShape()[0];
    int64_t const hiddenSize = embeds.getShape()[1];
    check::check(norms.getShape().volume() >= numRows * 2, "norms buffer too small");

    rowNormsKernel<<<static_cast<uint32_t>(numRows), kBlockSize, 0, stream>>>(
        embeds.dataPointer<half>(), norms.dataPointer<float>(), static_cast<int32_t>(hiddenSize));
    CUDA_CHECK(cudaGetLastError());
}

void computeDartPivotDots(
    rt::Tensor const& embeds, int32_t const* pivotIndices, int32_t numPivots, rt::Tensor& dots, cudaStream_t stream)
{
    check::check(embeds.getDataType() == nvinfer1::DataType::kHALF, "DART selector expects FP16 embeddings");
    check::check(embeds.getShape().getNumDims() == 2, "embeds must be [numRows, hiddenSize]");
    check::check(dots.getDataType() == nvinfer1::DataType::kFLOAT, "dots must be FP32");
    check::check(numPivots > 0 && numPivots <= kDartMaxPivots, "numPivots out of range");
    int64_t const numRows = embeds.getShape()[0];
    int64_t const hiddenSize = embeds.getShape()[1];
    check::check(dots.getShape().volume() >= static_cast<int64_t>(numPivots) * numRows, "dots buffer too small");

    dim3 const grid(static_cast<uint32_t>(numRows), static_cast<uint32_t>(numPivots));
    pivotDotsKernel<<<grid, kBlockSize, 0, stream>>>(embeds.dataPointer<half>(), pivotIndices,
        dots.dataPointer<float>(), static_cast<int32_t>(hiddenSize), static_cast<int32_t>(numRows));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
