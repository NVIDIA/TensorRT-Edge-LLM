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

// Adapted from https://github.com/sgl-project/sglang/blob/v0.5.7/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
// Adapt from https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/moe/topk_softmax_kernels.cu
// which is originally adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "common/checkMacros.h"
#include "moeTopkSoftmaxKernels.h"
#include <cfloat>
#include <cub/cub.cuh>
// CCCL <cuda/functional> must be included at file scope. Including it inside
// trt_edgellm::kernel breaks ::cuda::__is_commutative_static_assert on CUDA 13.3.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12090
#include <cuda/functional>
#endif

namespace trt_edgellm
{
namespace kernel
{

// ====================== Constants and Macros ======================

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Warp size constant
static constexpr int32_t WARP_SIZE = 32;

// Mask value used to exclude already-selected experts in TopK iterations
// Must be sufficiently negative to never be selected, but not so extreme as to cause numerical issues
static constexpr float TOPK_MASK_VALUE = -10000.f;

// Shuffle operations for warp-level communication
#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(0xffffffff, var, lane_mask)
#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) __shfl_xor_sync(0xffffffff, var, lane_mask, width)

// CUDA 12.9+ deprecated cub::Max/Min in favor of cuda::maximum/minimum.
// CUDA_VERSION: 13020 = 13.2, 13030 = 13.3 (nvcc macro).
// 13.3 needs ::cuda:: for the type aliases; 13.2 uses cuda:: (validated on Thor).
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13030
using MaxReduceOp = ::cuda::maximum<>;
using MinReduceOp = ::cuda::minimum<>;
#elif defined(CUDA_VERSION) && CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
using MinReduceOp = cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

// ====================== Aligned Array Type ======================

/// Aligned array type for vectorized memory access
template <typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray
{
    T data[N];
};

// ====================== Type Conversion Utilities ======================

template <typename T>
__device__ __forceinline__ float convertToFloat(T x)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return __half2float(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __bfloat162float(x);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return x;
    }
    return static_cast<float>(x);
}

// ====================== Fallback Path: Separate Softmax + TopK ======================

/**
 * @brief Basic softmax kernel for non-power-of-2 expert counts
 *
 * Each block processes one row (one token).
 */
template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeSoftmaxKernel(
    T const* input, float* output, int32_t numCols, float moeSoftcapping, float const* correctionBias)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizingFactor;
    __shared__ float floatMax;

    int32_t const threadRowOffset = blockIdx.x * numCols;

    float threadData = -FLT_MAX;

    // First pass: Apply transformation, find max, and write transformed values to output
    for (int32_t ii = threadIdx.x; ii < numCols; ii += TPB)
    {
        int32_t const idx = threadRowOffset + ii;
        float val = convertToFloat<T>(input[idx]);

        // Apply tanh softcapping if enabled
        if (moeSoftcapping != 0.0f)
        {
            val = tanhf(val / moeSoftcapping) * moeSoftcapping;
        }

        // Apply correction bias if provided
        if (correctionBias != nullptr)
        {
            val = val + correctionBias[ii];
        }

        output[idx] = val; // Store transformed value
        threadData = max(val, threadData);
    }

    float const maxElem = BlockReduce(tmpStorage).Reduce(threadData, MaxReduceOp());

    if (threadIdx.x == 0)
    {
        floatMax = maxElem;
    }
    __syncthreads();

    // Second pass: Compute sum using transformed values from output
    threadData = 0;
    for (int32_t ii = threadIdx.x; ii < numCols; ii += TPB)
    {
        int32_t const idx = threadRowOffset + ii;
        threadData += exp((output[idx] - floatMax));
    }

    auto const Z = BlockReduce(tmpStorage).Sum(threadData);

    if (threadIdx.x == 0)
    {
        normalizingFactor = 1.f / Z;
    }
    __syncthreads();

    // Third pass: Compute final softmax using transformed values from output
    for (int32_t ii = threadIdx.x; ii < numCols; ii += TPB)
    {
        int32_t const idx = threadRowOffset + ii;
        float const softmaxVal = exp((output[idx] - floatMax)) * normalizingFactor;
        output[idx] = softmaxVal;
    }
}

/**
 * @brief TopK selection kernel for non-power-of-2 expert counts
 *
 * Each block processes one row (one token).
 */
template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopKKernel(
    float const* inputsAfterSoftmax, float* output, int32_t* indices, int32_t numExperts, int32_t k, bool renormalize)
{
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp threadKvp;
    cub::ArgMax argMax;

    int32_t const blockRow = blockIdx.x;
    int32_t const threadReadOffset = blockIdx.x * numExperts;
    float rowSumForRenormalize = 0;

    for (int32_t kIdx = 0; kIdx < k; ++kIdx)
    {
        threadKvp.key = 0;
        threadKvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inpKvp;
        for (int32_t expert = threadIdx.x; expert < numExperts; expert += TPB)
        {
            int32_t const idx = threadReadOffset + expert;
            inpKvp.key = expert;
            inpKvp.value = inputsAfterSoftmax[idx];

            for (int32_t priorK = 0; priorK < kIdx; ++priorK)
            {
                int32_t const priorWinningExpert = indices[k * blockRow + priorK];

                if (priorWinningExpert == expert)
                {
                    inpKvp = threadKvp;
                }
            }

            threadKvp = argMax(inpKvp, threadKvp);
        }

        cub_kvp const resultKvp = BlockReduce(tmpStorage).Reduce(threadKvp, argMax);
        if (threadIdx.x == 0)
        {
            int32_t const idx = k * blockRow + kIdx;
            output[idx] = resultKvp.value;
            indices[idx] = resultKvp.key;
            assert(indices[idx] >= 0);
            rowSumForRenormalize += resultKvp.value;
        }
        __syncthreads();
    }

    if (renormalize && threadIdx.x == 0)
    {
        float rowSumForRenormalizeInv = 1.f / rowSumForRenormalize;
        for (int32_t kIdx = 0; kIdx < k; ++kIdx)
        {
            int32_t const idx = k * blockRow + kIdx;
            output[idx] = output[idx] * rowSumForRenormalizeInv;
        }
    }
}

// ====================== Optimized Fused TopK Gating Softmax ======================

/*
 * A Top-K gating softmax written to exploit when the number of experts in the MoE layers
 * are a small power of 2. This allows us to cleanly share the rows among the threads in
 * a single warp and eliminate communication between warps (so no need to use shared mem).
 *
 * It fuses the softmax, max and argmax into a single kernel.
 *
 * Limitations:
 * 1) This implementation is intended for when the number of experts is a small power of 2.
 * 2) This implementation assumes k is small, but will work for any k.
 */

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmaxKernel(T const* input, float* output,
    int32_t numRows, int32_t* indices, int32_t k, bool renormalize, float moeSoftcapping, float const* correctionBias)
{
    // Compile time assertions and constants
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== Runtime variables ========================

    // Compute CTA and warp rows
    int32_t const ctaBaseRow = blockIdx.x * ROWS_PER_CTA;
    int32_t const warpBaseRow = ctaBaseRow + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row
    int32_t const threadRowInWarp = threadIdx.x / THREADS_PER_ROW;
    int32_t const threadRow = warpBaseRow + threadRowInWarp;

    // Early exit for out of bounds
    if (threadRow >= numRows)
    {
        return;
    }

    // Setup read pointers
    T const* threadRowPtr = input + threadRow * ELTS_PER_ROW;

    // Compute group and first element for this thread
    int32_t const threadGroupIdx = threadIdx.x % THREADS_PER_ROW;
    int32_t const firstEltReadByThread = threadGroupIdx * ELTS_PER_LDG;
    T const* threadReadPtr = threadRowPtr + firstEltReadByThread;

    // Aligned array type for vectorized loads
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;

    // Pull in data from global memory
    T rowChunkTemp[VPT];
    AccessType* rowChunkVecPtr = reinterpret_cast<AccessType*>(&rowChunkTemp);
    AccessType const* vecThreadReadPtr = reinterpret_cast<AccessType const*>(threadReadPtr);

#pragma unroll
    // Interleaved loads for better memory coalescing
    for (int32_t ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        rowChunkVecPtr[ii] = vecThreadReadPtr[ii * THREADS_PER_ROW];
    }

    float rowChunk[VPT];
#pragma unroll
    // Upcast logits to float32
    for (int32_t ii = 0; ii < VPT; ++ii)
    {
        rowChunk[ii] = convertToFloat<T>(rowChunkTemp[ii]);
    }

    // Apply tanh softcapping and correction bias
    if (moeSoftcapping != 0.0f || correctionBias != nullptr)
    {
#pragma unroll
        for (int32_t ii = 0; ii < VPT; ++ii)
        {
            float val = rowChunk[ii];

            // Apply tanh softcapping if enabled
            if (moeSoftcapping != 0.0f)
            {
                val = tanhf(val / moeSoftcapping) * moeSoftcapping;
            }

            // Apply correction bias if provided
            if (correctionBias != nullptr)
            {
                // LDG is interleaved
                int32_t const groupId = ii / ELTS_PER_LDG;
                int32_t const localId = ii % ELTS_PER_LDG;
                int32_t const expertIdx = firstEltReadByThread + groupId * THREADS_PER_ROW * ELTS_PER_LDG + localId;
                val = val + correctionBias[expertIdx];
            }

            rowChunk[ii] = val;
        }
    }

    // ===================== Softmax Begin =====================

    // First, perform max reduce within the thread
    float threadMax = rowChunk[0];
#pragma unroll
    for (int32_t ii = 1; ii < VPT; ++ii)
    {
        threadMax = max(threadMax, rowChunk[ii]);
    }

// Butterfly reduce to find max within thread group
#pragma unroll
    for (int32_t mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        threadMax = max(threadMax, SHFL_XOR_SYNC_WIDTH(threadMax, mask, THREADS_PER_ROW));
    }

    // Subtract max and compute exp, also compute thread local sum
    float rowSum = 0;
#pragma unroll
    for (int32_t ii = 0; ii < VPT; ++ii)
    {
        rowChunk[ii] = expf(rowChunk[ii] - threadMax);
        rowSum += rowChunk[ii];
    }

// Butterfly reduce for sum
#pragma unroll
    for (int32_t mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        rowSum += SHFL_XOR_SYNC_WIDTH(rowSum, mask, THREADS_PER_ROW);
    }

    // Scale for softmax
    float const reciprocalRowSum = 1.f / rowSum;

#pragma unroll
    for (int32_t ii = 0; ii < VPT; ++ii)
    {
        rowChunk[ii] = rowChunk[ii] * reciprocalRowSum;
    }

    // ===================== Softmax End =====================

    // ===================== TopK Selection =====================

    int32_t startCol = firstEltReadByThread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float rowSumForRenormalize = 0;

    for (int32_t kIdx = 0; kIdx < k; ++kIdx)
    {
        // Each thread does local argmax
        float maxVal = rowChunk[0];
        int32_t expert = startCol;

#pragma unroll
        for (int32_t ldg = 0, col = startCol; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int32_t ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = rowChunk[ldg * ELTS_PER_LDG + ii];

                if (val > maxVal)
                {
                    maxVal = val;
                    expert = col + ii;
                }
            }
        }

// Butterfly argmax reduce
#pragma unroll
        for (int32_t mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float otherMax = SHFL_XOR_SYNC_WIDTH(maxVal, mask, THREADS_PER_ROW);
            int32_t otherExpert = SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // Lower indices win in ties
            if (otherMax > maxVal || (otherMax == maxVal && otherExpert < expert))
            {
                maxVal = otherMax;
                expert = otherExpert;
            }
        }

        // Write result to global memory
        if (threadGroupIdx == 0)
        {
            int32_t const idx = k * threadRow + kIdx;
            output[idx] = maxVal;
            indices[idx] = expert;
            rowSumForRenormalize += maxVal;
        }

        // Clear the winning value for next iteration
        if (kIdx + 1 < k)
        {
            int32_t const ldgGroupForExpert = expert / COLS_PER_GROUP_LDG;
            int32_t const threadToClearInGroup = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            if (threadGroupIdx == threadToClearInGroup)
            {
                int32_t const offsetForExpert = expert % ELTS_PER_LDG;
                rowChunk[ldgGroupForExpert * ELTS_PER_LDG + offsetForExpert] = TOPK_MASK_VALUE;
            }
        }
    }

    // Fuse renormalization
    if (renormalize && threadGroupIdx == 0)
    {
        float rowSumForRenormalizeInv = 1.f / rowSumForRenormalize;
#pragma unroll
        for (int32_t kIdx = 0; kIdx < k; ++kIdx)
        {
            int32_t const idx = k * threadRow + kIdx;
            output[idx] = output[idx] * rowSumForRenormalizeInv;
        }
    }
}

// ====================== Compile-time Constants for Kernel Launch ======================

namespace detail
{

// Constructs compile-time constants needed to partition work across threads
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
        "EXPERTS must be compatible with vectorized memory access pattern");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

} // namespace detail

// ====================== Kernel Launcher Helpers ======================

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(T const* input, float* output, int32_t* indices, int32_t numRows, int32_t k,
    bool renormalize, float moeSoftcapping, float const* correctionBias, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
    using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;

    int32_t const numWarps = (numRows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int32_t const numBlocks = (numWarps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 blockDim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmaxKernel<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<numBlocks, blockDim, 0, stream>>>(
        input, output, numRows, indices, k, renormalize, moeSoftcapping, correctionBias);
}

// Macro for launching the optimized kernel
#define LAUNCH_TOPK_SOFTMAX(TYPE, NUM_EXPERTS, WARPS_PER_TB)                                                           \
    topkGatingSoftmaxLauncherHelper<TYPE, NUM_EXPERTS, WARPS_PER_TB>(                                                  \
        inputPtr, topkWeightsPtr, topkIndicesPtr, numTokens, topk, renormalize, moeSoftcapping, biasPtr, stream);

// ====================== Main Kernel Launcher ======================

template <typename T>
void topkGatingSoftmaxKernelLauncher(T const* inputPtr, float* topkWeightsPtr, int32_t* topkIndicesPtr,
    float* softmaxWorkspace, int32_t numTokens, int32_t numExperts, int32_t topk, bool renormalize,
    float moeSoftcapping, float const* biasPtr, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;

    switch (numExperts)
    {
    case 1: LAUNCH_TOPK_SOFTMAX(T, 1, WARPS_PER_TB); break;
    case 2: LAUNCH_TOPK_SOFTMAX(T, 2, WARPS_PER_TB); break;
    case 4: LAUNCH_TOPK_SOFTMAX(T, 4, WARPS_PER_TB); break;
    case 8: LAUNCH_TOPK_SOFTMAX(T, 8, WARPS_PER_TB); break;
    case 16: LAUNCH_TOPK_SOFTMAX(T, 16, WARPS_PER_TB); break;
    case 32: LAUNCH_TOPK_SOFTMAX(T, 32, WARPS_PER_TB); break;
    case 64: LAUNCH_TOPK_SOFTMAX(T, 64, WARPS_PER_TB); break;
    case 128: LAUNCH_TOPK_SOFTMAX(T, 128, WARPS_PER_TB); break;
    case 256: LAUNCH_TOPK_SOFTMAX(T, 256, WARPS_PER_TB); break;
    default:
    {
        // Fallback path for non-power-of-2 expert counts
        check::check(
            softmaxWorkspace != nullptr, "softmaxWorkspace must be provided for numExperts that are not a power of 2.");
        static constexpr int TPB = 256;
        moeSoftmaxKernel<T, TPB>
            <<<numTokens, TPB, 0, stream>>>(inputPtr, softmaxWorkspace, numExperts, moeSoftcapping, biasPtr);
        moeTopKKernel<TPB><<<numTokens, TPB, 0, stream>>>(
            softmaxWorkspace, topkWeightsPtr, topkIndicesPtr, numExperts, topk, renormalize);
    }
    }
}

#undef LAUNCH_TOPK_SOFTMAX

// ====================== Public API Implementation ======================

size_t getMoeTopkSoftmaxWorkspaceSize(int32_t numTokens, int32_t numExperts)
{
    // Check if optimized path can be used
    bool const isPow2 = (numExperts != 0) && ((numExperts & (numExperts - 1)) == 0);
    bool const needsWorkspace = !isPow2 || numExperts > 256;

    if (needsWorkspace)
    {
        // Workspace stores softmax output: [numTokens, numExperts] as float
        return static_cast<size_t>(numTokens) * static_cast<size_t>(numExperts) * sizeof(float);
    }
    return 0;
}

void moeTopkSoftmax(rt::Tensor const& gatingOutput, rt::Tensor& topkWeights, rt::Tensor& topkIndices, int32_t topk,
    void* workspace, size_t workspaceSize, cudaStream_t stream, bool renormalize, float moeSoftcapping,
    rt::OptionalInputTensor correctionBias)
{
    // Validate input shapes
    auto const gatingShape = gatingOutput.getShape();
    auto const weightsShape = topkWeights.getShape();
    auto const indicesShape = topkIndices.getShape();

    check::check(gatingShape.getNumDims() == 2, "gatingOutput must be 2D tensor [numTokens, numExperts]");
    check::check(weightsShape.getNumDims() == 2, "topkWeights must be 2D tensor [numTokens, topk]");
    check::check(indicesShape.getNumDims() == 2, "topkIndices must be 2D tensor [numTokens, topk]");

    int32_t const numTokens = static_cast<int32_t>(gatingShape[0]);
    int32_t const numExperts = static_cast<int32_t>(gatingShape[1]);

    check::check(weightsShape[0] == numTokens, "topkWeights first dimension must match numTokens");
    check::check(indicesShape[0] == numTokens, "topkIndices first dimension must match numTokens");
    check::check(weightsShape[1] == topk, "topkWeights second dimension must match topk");
    check::check(indicesShape[1] == topk, "topkIndices second dimension must match topk");
    check::check(topk <= numExperts, "topk must be less than or equal to numExperts");

    // Validate data types
    auto const dtype = gatingOutput.getDataType();
    check::check(dtype == trt_edgellm::DataType::kFLOAT || dtype == trt_edgellm::DataType::kHALF
            || dtype == trt_edgellm::DataType::kBF16,
        "gatingOutput must be float32, float16, or bfloat16");
    check::check(topkWeights.getDataType() == trt_edgellm::DataType::kFLOAT, "topkWeights must be float32");
    check::check(topkIndices.getDataType() == trt_edgellm::DataType::kINT32, "topkIndices must be int32");

    // Validate device types
    check::check(gatingOutput.getDeviceType() == rt::DeviceType::kGPU, "gatingOutput must be on GPU");
    check::check(topkWeights.getDeviceType() == rt::DeviceType::kGPU, "topkWeights must be on GPU");
    check::check(topkIndices.getDeviceType() == rt::DeviceType::kGPU, "topkIndices must be on GPU");

    // Validate correction bias if provided
    float const* biasPtr = nullptr;
    if (correctionBias.has_value())
    {
        auto const& biasTensor = correctionBias.value().get();
        auto const biasShape = biasTensor.getShape();
        check::check(biasShape.getNumDims() == 1, "correctionBias must be 1D tensor [numExperts]");
        check::check(biasShape[0] == numExperts, "correctionBias size must match numExperts");
        check::check(biasTensor.getDataType() == trt_edgellm::DataType::kFLOAT, "correctionBias must be float32");
        check::check(biasTensor.getDeviceType() == rt::DeviceType::kGPU, "correctionBias must be on GPU");
        biasPtr = biasTensor.dataPointer<float>();
    }

    // Check workspace size
    size_t const requiredWorkspace = getMoeTopkSoftmaxWorkspaceSize(numTokens, numExperts);
    check::check(workspaceSize >= requiredWorkspace,
        "Workspace size too small. Required: " + std::to_string(requiredWorkspace)
            + ", provided: " + std::to_string(workspaceSize));

    // Get device pointers
    float* topkWeightsPtr = topkWeights.dataPointer<float>();
    int32_t* topkIndicesPtr = topkIndices.dataPointer<int32_t>();
    float* softmaxWorkspace = (requiredWorkspace > 0) ? static_cast<float*>(workspace) : nullptr;

    // Dispatch based on input data type
    if (dtype == trt_edgellm::DataType::kFLOAT)
    {
        topkGatingSoftmaxKernelLauncher<float>(gatingOutput.dataPointer<float>(), topkWeightsPtr, topkIndicesPtr,
            softmaxWorkspace, numTokens, numExperts, topk, renormalize, moeSoftcapping, biasPtr, stream);
    }
    else if (dtype == trt_edgellm::DataType::kHALF)
    {
        topkGatingSoftmaxKernelLauncher<__half>(reinterpret_cast<__half const*>(gatingOutput.dataPointer<half>()),
            topkWeightsPtr, topkIndicesPtr, softmaxWorkspace, numTokens, numExperts, topk, renormalize, moeSoftcapping,
            biasPtr, stream);
    }
    else if (dtype == trt_edgellm::DataType::kBF16)
    {
        topkGatingSoftmaxKernelLauncher<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(gatingOutput.dataPointer<__nv_bfloat16>()), topkWeightsPtr,
            topkIndicesPtr, softmaxWorkspace, numTokens, numExperts, topk, renormalize, moeSoftcapping, biasPtr,
            stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
