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

#include "gdnKernelUtils.cuh"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"

#include <cuda_fp16.h>
#include <type_traits>

namespace trt_edgellm
{

/**
 * Single-thread prefix-sum kernel: converts context_lengths[N] to cu_seqlens[N+1].
 *
 * cu_seqlens[0] = 0
 * cu_seqlens[i+1] = cu_seqlens[i] + context_lengths[i]
 */
__global__ void gdnCalCuSeqLensKernel(int32_t const* context_lengths, // [N]
    int32_t* cu_seqlens,                                              // [N+1]  output
    int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cu_seqlens[0] = 0;
        int32_t running = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            running += context_lengths[i];
            cu_seqlens[i + 1] = running;
        }
    }
}

void launchGdnCalCuSeqLens(void const* context_lengths, void* cu_seqlens, int32_t batchSize, cudaStream_t stream)
{
    gdnCalCuSeqLensKernel<<<1, 1, 0, stream>>>(
        static_cast<int32_t const*>(context_lengths), static_cast<int32_t*>(cu_seqlens), batchSize);
}

/**
 * L2-normalize Q and K along the head dimension (last axis) in-place.
 *
 * Input layout: (N * T * H, D) where D = head_dim (e.g. 128).
 * Each row (token-head pair) is divided by its L2 norm + eps.
 * One warp per row, warp-level reduction for the norm.
 */
__global__ void gdnL2NormQKKernel(half* data, // [numRows, headDim]
    int32_t numRows, int32_t headDim)
{
    int32_t const row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int32_t const lane = threadIdx.x % 32;
    if (row >= numRows)
        return;

    half* rowPtr = data + static_cast<int64_t>(row) * headDim;

    // Pass 1: compute sum of squares
    float sumSq = 0.0f;
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float val = __half2float(rowPtr[d]);
        sumSq += val * val;
    }
    // Warp reduction
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        sumSq += __shfl_xor_sync(0xFFFFFFFF, sumSq, offset);
    }
    float invNorm = rsqrtf(sumSq + 1e-6f);

    // Pass 2: normalize in-place
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float val = __half2float(rowPtr[d]);
        rowPtr[d] = __float2half(val * invNorm);
    }
}

__device__ __forceinline__ float computeGdnL2InvNormSm12x(half const* rowPtr, int32_t lane, int32_t headDim)
{
    float sumSq = 0.0f;
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float const val = __half2float(rowPtr[d]);
        sumSq += val * val;
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        sumSq += __shfl_xor_sync(0xFFFFFFFF, sumSq, offset);
    }
    return rsqrtf(sumSq + 1e-6f);
}

__device__ __forceinline__ void applyGdnL2NormSm12x(half* rowPtr, int32_t lane, int32_t headDim, float invNorm)
{
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float const val = __half2float(rowPtr[d]);
        rowPtr[d] = __float2half(val * invNorm);
    }
}

/**
 * SM12x fused Q/K L2 normalization and PDL consumer/producer.
 *
 * Q and K have the same layout and shape. Each warp handles the same row in
 * both buffers while preserving the legacy unary path's FP32 accumulation,
 * shuffle-reduction, and store order for each tensor.
 */
template <bool EnablePdl>
__global__ void gdnL2NormQKFusedSm12xKernel(half* q, // [numRows, headDim]
    half* k,                                         // [numRows, headDim]
    int32_t numRows, int32_t headDim)
{
    int32_t const row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int32_t const lane = threadIdx.x % 32;
    if (row >= numRows)
    {
        return;
    }

    int64_t const rowOffset = static_cast<int64_t>(row) * headDim;
    half* qRowPtr = q + rowOffset;
    half* kRowPtr = k + rowOffset;

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (EnablePdl)
    {
        // Q and K are TensorRT plugin inputs and may be produced by the
        // prerequisite grid. Keep only dependency-independent index and
        // pointer setup above this wait; the memory clobber prevents either
        // input load from moving across the visibility boundary.
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }
#endif

    float const invNormQ = computeGdnL2InvNormSm12x(qRowPtr, lane, headDim);
    float const invNormK = computeGdnL2InvNormSm12x(kRowPtr, lane, headDim);

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (EnablePdl)
    {
        // Every CTA's thread 0 reaches this point after its warp computes both
        // inverse norms and before any pass-2 store in that warp. The dependent
        // consumer's griddepcontrol.wait is the grid completion and
        // memory-visibility boundary, so no CTA barrier is required here.
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
#endif

    applyGdnL2NormSm12x(qRowPtr, lane, headDim, invNormQ);
    applyGdnL2NormSm12x(kRowPtr, lane, headDim, invNormK);
}

void launchGdnL2NormQK(void* q, void* k, int32_t n, int32_t seqLen, int32_t h, int32_t headDim, cudaStream_t stream)
{
    int32_t const numRowsQ = n * seqLen * h;
    int32_t const warpsPerBlock = 8;
    int32_t const threadsPerBlock = warpsPerBlock * 32;
    int32_t const numBlocksQ = (numRowsQ + warpsPerBlock - 1) / warpsPerBlock;
    int32_t const numBlocksK = numBlocksQ; // same shape

    gdnL2NormQKKernel<<<numBlocksQ, threadsPerBlock, 0, stream>>>(static_cast<half*>(q), numRowsQ, headDim);
    gdnL2NormQKKernel<<<numBlocksK, threadsPerBlock, 0, stream>>>(static_cast<half*>(k), numRowsQ, headDim);
}

cudaError_t launchGdnL2NormQKFusedSm12x(
    void* q, void* k, int32_t n, int32_t seqLen, int32_t h, int32_t headDim, bool enablePdl, cudaStream_t stream)
{
    int32_t const numRows = n * seqLen * h;
    int32_t const warpsPerBlock = 8;
    int32_t const threadsPerBlock = warpsPerBlock * 32;
    int32_t const numBlocks = (numRows + warpsPerBlock - 1) / warpsPerBlock;

    auto const launchKernel = [&](auto pdlTag) {
        constexpr bool kEnablePdl = decltype(pdlTag)::value;
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
        cudaLaunchAttribute pdlAttribute{};
        cudaLaunchConfig_t launchConfig{};
        launchConfig.gridDim = dim3(numBlocks);
        launchConfig.blockDim = dim3(threadsPerBlock);
        launchConfig.dynamicSmemBytes = 0;
        launchConfig.stream = stream;
        launchConfig.attrs = kEnablePdl ? &pdlAttribute : nullptr;
        launchConfig.numAttrs = kEnablePdl ? 1U : 0U;

        if constexpr (kEnablePdl)
        {
            pdlAttribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            pdlAttribute.val.programmaticStreamSerializationAllowed = 1;
        }

        return cudaLaunchKernelEx(&launchConfig, gdnL2NormQKFusedSm12xKernel<kEnablePdl>, static_cast<half*>(q),
            static_cast<half*>(k), numRows, headDim);
#else
        gdnL2NormQKFusedSm12xKernel<kEnablePdl>
            <<<numBlocks, threadsPerBlock, 0, stream>>>(static_cast<half*>(q), static_cast<half*>(k), numRows, headDim);
        return cudaPeekAtLastError();
#endif // SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    };

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    if (enablePdl)
    {
        return launchKernel(std::true_type{});
    }
#else
    if (enablePdl)
    {
        LOG_DEBUG("GDN fused Q/K normalization: PDL requested but unavailable in this build; launching without PDL.");
    }
#endif
    return launchKernel(std::false_type{});
}

/**
 * Transpose the last two dims of a batch of square float32 matrices.
 *
 * Layout: src/dst are contiguous (numBlocks, dim, dim) float32.
 * One block handles one (dim, dim) matrix using 32x32 shared-memory tiles
 * with +1 padding to avoid bank conflicts.
 *
 * Grid: (numTilesPerMatrix, numBlocks)  where numTilesPerMatrix = (dim/32)^2.
 * Block: (32, 8)  — 8 rows per thread to amortize tile overhead.
 */
__global__ void gdnStateTransposeKernel(
    float const* __restrict__ src, float* __restrict__ dst, int32_t numBlocks, int32_t dim)
{
    // Shared tile with bank-conflict padding
    __shared__ float tile[32][33];

    int32_t const tilesPerRow = dim / 32;
    int32_t const tileIdx = blockIdx.x;            // which 32x32 tile within the matrix
    int32_t const matIdx = blockIdx.y;             // which matrix
    int32_t const tileRow = tileIdx / tilesPerRow; // tile row in the matrix
    int32_t const tileCol = tileIdx % tilesPerRow; // tile col in the matrix

    int64_t const matOffset = static_cast<int64_t>(matIdx) * dim * dim;

    // Read tile from src[matIdx, tileRow*32.., tileCol*32..]
    int32_t const baseRow = tileRow * 32;
    int32_t const baseCol = tileCol * 32;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const r = baseRow + threadIdx.y + j;
        int32_t const c = baseCol + threadIdx.x;
        if (r < dim && c < dim)
        {
            tile[threadIdx.y + j][threadIdx.x] = src[matOffset + static_cast<int64_t>(r) * dim + c];
        }
    }
    __syncthreads();

    // Write transposed tile to dst[matIdx, tileCol*32.., tileRow*32..]
    int32_t const dstBaseRow = baseCol; // transposed
    int32_t const dstBaseCol = baseRow;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const r = dstBaseRow + threadIdx.y + j;
        int32_t const c = dstBaseCol + threadIdx.x;
        if (r < dim && c < dim)
        {
            dst[matOffset + static_cast<int64_t>(r) * dim + c] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launchGdnStateTranspose(void const* src, void* dst, int32_t numBlocks, int32_t dim, cudaStream_t stream)
{
    int32_t const tilesPerRow = (dim + 31) / 32;
    int32_t const tilesPerMatrix = tilesPerRow * tilesPerRow;
    dim3 grid(tilesPerMatrix, numBlocks);
    dim3 block(32, 8);
    gdnStateTransposeKernel<<<grid, block, 0, stream>>>(
        static_cast<float const*>(src), static_cast<float*>(dst), numBlocks, dim);
}

__global__ void gdnStateGatherTransposeKernel(float const* __restrict__ src, float* __restrict__ dst,
    int32_t const* __restrict__ stateIndices, int32_t statePoolRows, int32_t numHeads, int32_t dim)
{
    __shared__ float tile[32][33];

    int32_t const matrix = blockIdx.y;
    int32_t const batch = matrix / numHeads;
    int32_t const head = matrix % numHeads;
    int32_t const residentRow = __ldg(stateIndices + batch);
    bool const validResidentRow = residentRow >= 0 && residentRow < statePoolRows;
    int32_t const srcMatrix = residentRow * numHeads + head;
    int32_t const tilesPerRow = (dim + 31) / 32;
    int32_t const tileRow = blockIdx.x / tilesPerRow;
    int32_t const tileCol = blockIdx.x % tilesPerRow;
    int32_t const baseRow = tileRow * 32;
    int32_t const baseCol = tileCol * 32;
    int64_t const srcOffset = static_cast<int64_t>(srcMatrix) * dim * dim;
    int64_t const dstOffset = static_cast<int64_t>(matrix) * dim * dim;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const row = baseRow + threadIdx.y + j;
        int32_t const col = baseCol + threadIdx.x;
        if (row < dim && col < dim)
        {
            tile[threadIdx.y + j][threadIdx.x]
                = validResidentRow ? src[srcOffset + static_cast<int64_t>(row) * dim + col] : 0.0F;
        }
    }
    __syncthreads();

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const row = baseCol + threadIdx.y + j;
        int32_t const col = baseRow + threadIdx.x;
        if (row < dim && col < dim)
        {
            dst[dstOffset + static_cast<int64_t>(row) * dim + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launchGdnStateGatherTranspose(void const* src, void* dst, void const* stateIndices, int32_t batchSize,
    int32_t statePoolRows, int32_t numHeads, int32_t dim, cudaStream_t stream)
{
    int32_t const tilesPerRow = (dim + 31) / 32;
    dim3 const grid(tilesPerRow * tilesPerRow, batchSize * numHeads);
    dim3 const block(32, 8);
    gdnStateGatherTransposeKernel<<<grid, block, 0, stream>>>(static_cast<float const*>(src), static_cast<float*>(dst),
        static_cast<int32_t const*>(stateIndices), statePoolRows, numHeads, dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gdnStateGatherKernel(
    float const* src, float* dst, int32_t const* stateIndices, int32_t statePoolRows, int64_t rowElements)
{
    int32_t const batch = blockIdx.y;
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < rowElements)
    {
        int32_t const residentRow = stateIndices[batch];
        dst[static_cast<int64_t>(batch) * rowElements + element] = residentRow >= 0 && residentRow < statePoolRows
            ? src[static_cast<int64_t>(residentRow) * rowElements + element]
            : 0.0F;
    }
}

void launchGdnStateGather(void const* src, void* dst, void const* stateIndices, int32_t batchSize, int32_t numHeads,
    int32_t statePoolRows, int32_t kDim, int32_t vDim, cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;
    int64_t const rowElements = static_cast<int64_t>(numHeads) * kDim * vDim;
    dim3 const grid(static_cast<uint32_t>((rowElements + blockSize - 1) / blockSize), batchSize);
    gdnStateGatherKernel<<<grid, blockSize, 0, stream>>>(static_cast<float const*>(src), static_cast<float*>(dst),
        static_cast<int32_t const*>(stateIndices), statePoolRows, rowElements);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gdnStateIndexedTransposeInPlaceKernel(float* state, int32_t const* __restrict__ stateIndices,
    int32_t statePoolRows, int32_t numHeads, int32_t dim, int32_t tilesPerRow)
{
    __shared__ float tileA[32][33];
    __shared__ float tileB[32][33];

    int32_t tileRow = 0;
    int32_t tileCol = blockIdx.x;
    while (tileCol >= tilesPerRow - tileRow)
    {
        tileCol -= tilesPerRow - tileRow;
        ++tileRow;
    }
    tileCol += tileRow;

    int32_t const matrix = blockIdx.y;
    int32_t const batch = matrix / numHeads;
    int32_t const head = matrix % numHeads;
    int32_t const residentRow = __ldg(stateIndices + batch);
    if (residentRow < 0 || residentRow >= statePoolRows)
    {
        return;
    }
    int64_t const matrixOffset = static_cast<int64_t>(residentRow * numHeads + head) * dim * dim;
    int32_t const rowBase = tileRow * 32;
    int32_t const colBase = tileCol * 32;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const row = rowBase + threadIdx.y + j;
        int32_t const col = colBase + threadIdx.x;
        if (row < dim && col < dim)
        {
            tileA[threadIdx.y + j][threadIdx.x] = state[matrixOffset + static_cast<int64_t>(row) * dim + col];
        }
        if (tileRow != tileCol)
        {
            int32_t const peerRow = colBase + threadIdx.y + j;
            int32_t const peerCol = rowBase + threadIdx.x;
            if (peerRow < dim && peerCol < dim)
            {
                tileB[threadIdx.y + j][threadIdx.x]
                    = state[matrixOffset + static_cast<int64_t>(peerRow) * dim + peerCol];
            }
        }
    }
    __syncthreads();

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const row = colBase + threadIdx.y + j;
        int32_t const col = rowBase + threadIdx.x;
        if (row < dim && col < dim)
        {
            state[matrixOffset + static_cast<int64_t>(row) * dim + col] = tileA[threadIdx.x][threadIdx.y + j];
        }
        if (tileRow != tileCol)
        {
            int32_t const peerRow = rowBase + threadIdx.y + j;
            int32_t const peerCol = colBase + threadIdx.x;
            if (peerRow < dim && peerCol < dim)
            {
                state[matrixOffset + static_cast<int64_t>(peerRow) * dim + peerCol]
                    = tileB[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

void launchGdnStateIndexedTransposeInPlace(void* state, void const* stateIndices, int32_t batchSize,
    int32_t statePoolRows, int32_t numHeads, int32_t dim, cudaStream_t stream)
{
    int32_t const tilesPerRow = (dim + 31) / 32;
    int32_t const tilePairs = tilesPerRow * (tilesPerRow + 1) / 2;
    dim3 const grid(tilePairs, batchSize * numHeads);
    dim3 const block(32, 8);
    gdnStateIndexedTransposeInPlaceKernel<<<grid, block, 0, stream>>>(static_cast<float*>(state),
        static_cast<int32_t const*>(stateIndices), statePoolRows, numHeads, dim, tilesPerRow);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm
