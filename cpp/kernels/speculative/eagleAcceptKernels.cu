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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "eagleAcceptKernels.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{

// Shared alignment function for workspace buffers
inline size_t alignWorkspaceSize(size_t size)
{
    size_t const alignment = 256; // 256-byte alignment for optimal GPU memory access
    return (size + alignment - 1) & ~(alignment - 1);
}

// Internal workspace structure for memory management (similar to SamplingWorkspace)
struct EagleAcceptWorkspace
{
    // Device pointer, owned by external caller - must remain valid for kernel lifetime
    void* ptr;
    // Size in bytes of the workspace buffer
    size_t size;

    // Device pointer to top-1 tokens buffer, owned by external caller
    // Array size: batchSize * numTokens elements (int32_t each)
    int32_t* top1Tokens;

    EagleAcceptWorkspace()
        : ptr(nullptr)
        , size(0)
        , top1Tokens(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, int32_t batchSize, int32_t numTokens)
    {
        // Check that workspace is aligned to 256 bytes for optimal GPU memory access
        if (reinterpret_cast<uintptr_t>(workspace) % 256 != 0)
        {
            throw std::runtime_error("Workspace must be aligned to 256 bytes");
        }

        ptr = workspace;
        size = workspaceSize;

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        // Top-1 tokens buffer
        size_t top1TokensSize = alignWorkspaceSize(batchSize * numTokens * sizeof(int32_t));
        top1Tokens = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TokensSize;

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Eagle workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }
};

// Calculate workspace size for Eagle accept algorithm
size_t getEagleAcceptWorkspaceSize(int32_t batchSize, int32_t numTokens)
{
    // Top-1 tokens buffer
    return alignWorkspaceSize(batchSize * numTokens * sizeof(int32_t));
}

namespace
{

// Helper structure for top-1 selection
struct Top1Helper
{
    float value;
    int32_t index;

    __device__ __forceinline__ Top1Helper()
        : value(-FLT_MAX)
        , index(-1)
    {
    }

    __device__ __forceinline__ void update(float elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

// Reduction operator for top-1 selection
struct top1MaxOpFunctor
{
    __device__ __forceinline__ Top1Helper operator()(Top1Helper const& a, Top1Helper const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Inline function for shared memory alignment
__forceinline__ size_t alignSharedMem(size_t size)
{
    return ((size + 15) / 16) * 16; // Align to 16 bytes
}

// Stage 1: Compute top-1 tokens for all positions using sampling strategy
// Optionally map from reduced vocab to full vocab if mapping table is provided
template <int32_t BLOCK_SIZE>
__global__ void eagleComputeTop1Kernel(float const* logits, int32_t* top1Tokens, int32_t const* vocabMappingTable,
    int32_t batchSize, int32_t numTokens, int32_t vocabSize)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x / numTokens);
    auto const tokenIdx = static_cast<int32_t>(blockIdx.x % numTokens);

    if (batchIdx >= batchSize || tokenIdx >= numTokens)
        return;

    // Calculate logits offset for this batch and token position
    int32_t const logitsOffset = batchIdx * numTokens * vocabSize + tokenIdx * vocabSize;

    Top1Helper partial;

    // Find top-1 token using parallel reduction across vocab
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE)
    {
        float logitValue = logits[logitsOffset + v];
        partial.update(logitValue, v);
    }

    // Block-level reduction to find global max
    Top1Helper blockMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    // Store result - map from reduced vocab to full vocab if mapping table provided
    if (tid == 0)
    {
        int32_t outputIdx = batchIdx * numTokens + tokenIdx;
        int32_t selectedIdx = (blockMax.index != -1) ? blockMax.index : 0;

        // Apply vocab mapping if provided (for reduced vocabulary)
        if (vocabMappingTable != nullptr)
        {
            selectedIdx = vocabMappingTable[selectedIdx];
        }

        top1Tokens[outputIdx] = selectedIdx;
    }
}
// Helper function to compute tree depth - count total connections (sum of 1s)
__device__ int32_t computeTokenDepth(int32_t tokenIdx, int8_t const* attentionMask, int32_t numTokens)
{
    int32_t depth = 0;
    for (int32_t i = 0; i < numTokens; ++i)
    {
        if (attentionMask[tokenIdx * numTokens + i] == 1)
        {
            depth++;
        }
    }
    return depth;
}

// CUDA kernel for eagle accept algorithm - optimized for concurrent batch processing
__global__ void eagleAcceptKernel(int32_t const* top1Tokens, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedLogitsIndices, int32_t* acceptLength, int32_t batchSize,
    int32_t numTokens, int32_t maxDepth)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    if (batchIdx >= batchSize)
        return;

    // Optimized shared memory layout
    extern __shared__ char sharedMem[];
    int32_t* tokenDepths = reinterpret_cast<int32_t*>(sharedMem);

    // Batch-specific pointers for better cache locality
    int8_t const* batchAttentionMask = attentionMask + batchIdx * numTokens * numTokens;
    int32_t const* batchTokenIds = tokenIds + batchIdx * numTokens;
    int32_t const* batchTop1Tokens = top1Tokens + batchIdx * numTokens;

    // Parallel initialization of output arrays
    // Use 0 for padding token IDs instead of -1 to avoid embedding lookup issues in draft model.
    // The actual padding positions will be skipped based on acceptLength.
    for (int32_t i = tid; i < maxDepth; i += blockSize)
    {
        acceptedTokenIds[batchIdx * maxDepth + i] = 0;
        acceptedLogitsIndices[batchIdx * maxDepth + i] = -1;
    }
    if (tid == 0)
    {
        acceptLength[batchIdx] = 0;
    }

    // Parallel computation of token depths with better memory access
    for (int32_t i = tid; i < numTokens; i += blockSize)
    {
        tokenDepths[i] = computeTokenDepth(i, batchAttentionMask, numTokens);
    }
    __syncthreads();

    // Process this batch - use warp-level operations where possible
    int32_t currentDepth = 0;
    int32_t currentTokenIdx = 0;
    int32_t expectedNextDepth = tokenDepths[0] + 1;

    for (int32_t step = 0; step < maxDepth && currentTokenIdx < numTokens; ++step)
    {
        // Step 1: Get precomputed top-1 token (broadcast to all threads)
        int32_t selectedTokenId = batchTop1Tokens[currentTokenIdx];

        // Step 2: Accept the selected token (single thread writes)
        if (tid == 0)
        {
            acceptedTokenIds[batchIdx * maxDepth + currentDepth] = selectedTokenId;
            acceptedLogitsIndices[batchIdx * maxDepth + currentDepth] = currentTokenIdx;
            acceptLength[batchIdx] = currentDepth + 1;
            currentDepth++;
        }

        // Step 3: Parallel tree search with block-level reduction
        __shared__ int32_t nextTokenIdx;
        if (tid == 0)
        {
            nextTokenIdx = numTokens; // Initialize to invalid value
        }
        __syncthreads();

        // Each thread checks different tokens in parallel
        for (int32_t checkIdx = 1 + tid; checkIdx < numTokens; checkIdx += blockSize)
        {
            if (batchTokenIds[checkIdx] == selectedTokenId && tokenDepths[checkIdx] == expectedNextDepth)
            {
                // Check attention mask: does checkIdx attend to currentTokenIdx?
                int32_t maskOffset = batchIdx * numTokens * numTokens + checkIdx * numTokens + currentTokenIdx;
                if (attentionMask[maskOffset] == 1)
                {
                    // Found a valid next token - use atomic to get the minimum index for deterministic behavior
                    atomicMin(&nextTokenIdx, checkIdx);
                }
            }
        }
        __syncthreads();

        // Step 4: Update for next iteration (all threads participate)
        if (nextTokenIdx < numTokens)
        {
            // Found valid next token in tree, continue from there
            currentTokenIdx = nextTokenIdx;
            expectedNextDepth++;
        }
        else
        {
            // No valid next token found in tree, stop here
            break;
        }
    }
}

// Optimized kernel launcher function using workspace and two-stage approach
void launchEagleAcceptKernel(float const* logits, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedLogitsIndices, int32_t* acceptLength, int32_t const* vocabMappingTable,
    int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth, void* workspace, size_t workspaceSize,
    cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;

    // Setup workspace partitioning
    EagleAcceptWorkspace ws;
    ws.setupWorkspace(workspace, workspaceSize, batchSize, numTokens);

    // Validate workspace buffer
    assert(ws.top1Tokens != nullptr);

    // Stage 1: Compute top-1 tokens for all positions (with optional vocab mapping)
    dim3 const gridSizeStage1(batchSize * numTokens);
    dim3 const blockSizeStage1(blockSize);

    // Calculate shared memory for stage 1 (only CUB temp storage)
    size_t sharedMemSizeStage1 = sizeof(typename cub::BlockReduce<Top1Helper, blockSize>::TempStorage);
    sharedMemSizeStage1 = alignSharedMem(sharedMemSizeStage1);

    eagleComputeTop1Kernel<blockSize><<<gridSizeStage1, blockSizeStage1, sharedMemSizeStage1, stream>>>(
        logits, ws.top1Tokens, vocabMappingTable, batchSize, numTokens, vocabSize);

    // Stage 2: Run optimized eagle accept algorithm
    dim3 const gridSizeStage2(batchSize);
    dim3 const blockSizeStage2(blockSize);

    // Calculate shared memory for stage 2 (only token depths - much smaller!)
    size_t sharedMemSizeStage2 = numTokens * sizeof(int32_t);
    sharedMemSizeStage2 = alignSharedMem(sharedMemSizeStage2);

    eagleAcceptKernel<<<gridSizeStage2, blockSizeStage2, sharedMemSizeStage2, stream>>>(ws.top1Tokens, tokenIds,
        attentionMask, acceptedTokenIds, acceptedLogitsIndices, acceptLength, batchSize, numTokens, maxDepth);
}

} // namespace

void eagleAccept(rt::Tensor const& logits, rt::Tensor const& tokenIds, rt::Tensor const& attentionMask,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedLogitsIndices, rt::Tensor& acceptLength,
    rt::OptionalInputTensor const& vocabMappingTable, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // Validate input shapes
    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();
    auto const maskShape = attentionMask.getShape();
    auto const acceptedTokenIdsShape = acceptedTokenIds.getShape();
    auto const acceptedLogitsIndicesShape = acceptedLogitsIndices.getShape();
    auto const acceptLengthShape = acceptLength.getShape();

    check::check(logitsShape.getNumDims() == 2, "logits must be 2D tensor [batch_size * num_tokens, vocab_size]");
    check::check(tokenIdsShape.getNumDims() == 2, "tokenIds must be 2D tensor [batch_size, num_tokens]");
    check::check(maskShape.getNumDims() == 3, "attentionMask must be 3D tensor [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape.getNumDims() == 2, "acceptedTokenIds must be 2D tensor [batch_size, max_depth]");
    check::check(acceptedLogitsIndicesShape.getNumDims() == 2,
        "acceptedLogitsIndices must be 2D tensor [batch_size, max_depth]");
    check::check(acceptLengthShape.getNumDims() == 1, "acceptLength must be 1D tensor [batch_size]");

    int32_t const batchSize = tokenIdsShape[0];
    int32_t const numTokens = tokenIdsShape[1];
    int32_t const vocabSize = logitsShape[1];
    int32_t const maxDepth = acceptedTokenIdsShape[1];

    check::check(logitsShape[0] == batchSize * numTokens, "logits must be [batch_size * num_tokens, vocab_size]");
    check::check(maskShape[0] == batchSize && maskShape[1] == numTokens && maskShape[2] == numTokens,
        "attentionMask must be [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape[0] == batchSize && acceptedTokenIdsShape[1] == maxDepth,
        "acceptedTokenIds must be [batch_size, max_depth]");
    check::check(acceptedLogitsIndicesShape[0] == batchSize && acceptedLogitsIndicesShape[1] == maxDepth,
        "acceptedLogitsIndices must be [batch_size, max_depth]");
    check::check(acceptLengthShape[0] == batchSize, "acceptLength length must match batch_size");

    // Validate data types
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "logits must be FP32");
    check::check(tokenIds.getDataType() == nvinfer1::DataType::kINT32, "tokenIds must be INT32");
    check::check(attentionMask.getDataType() == nvinfer1::DataType::kINT8, "attentionMask must be INT8");
    check::check(acceptedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "acceptedTokenIds must be INT32");
    check::check(
        acceptedLogitsIndices.getDataType() == nvinfer1::DataType::kINT32, "acceptedLogitsIndices must be INT32");
    check::check(acceptLength.getDataType() == nvinfer1::DataType::kINT32, "acceptLength must be INT32");

    // Validate device types - all tensors must be on GPU
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU, "logits must be on GPU device");
    check::check(tokenIds.getDeviceType() == rt::DeviceType::kGPU, "tokenIds must be on GPU device");
    check::check(attentionMask.getDeviceType() == rt::DeviceType::kGPU, "attentionMask must be on GPU device");
    check::check(acceptedTokenIds.getDeviceType() == rt::DeviceType::kGPU, "acceptedTokenIds must be on GPU device");
    check::check(
        acceptedLogitsIndices.getDeviceType() == rt::DeviceType::kGPU, "acceptedLogitsIndices must be on GPU device");
    check::check(acceptLength.getDeviceType() == rt::DeviceType::kGPU, "acceptLength must be on GPU device");
    check::check(maxDepth > 0 && maxDepth <= numTokens, "maxDepth must be positive and <= numTokens");

    // Validate vocab mapping table if provided
    int32_t const* vocabMappingTablePtr = nullptr;
    if (vocabMappingTable.has_value())
    {
        rt::Tensor const& vocabMapTensor = vocabMappingTable.value().get();
        check::check(vocabMapTensor.getDeviceType() == rt::DeviceType::kGPU, "vocabMappingTable must be on GPU device");
        check::check(vocabMapTensor.getDataType() == nvinfer1::DataType::kINT32, "vocabMappingTable must be INT32");
        check::check(vocabMapTensor.getShape().getNumDims() == 1, "vocabMappingTable must be 1D");
        check::check(vocabMapTensor.getShape()[0] == vocabSize, "vocabMappingTable size must match vocab size");
        vocabMappingTablePtr = vocabMapTensor.dataPointer<int32_t>();
    }

    // Get device pointers
    float const* logitsPtr = logits.dataPointer<float>();
    int32_t const* tokenIdsPtr = tokenIds.dataPointer<int32_t>();
    int8_t const* attentionMaskPtr = attentionMask.dataPointer<int8_t>();
    int32_t* acceptedTokenIdsPtr = acceptedTokenIds.dataPointer<int32_t>();
    int32_t* acceptedLogitsIndicesPtr = acceptedLogitsIndices.dataPointer<int32_t>();
    int32_t* acceptLengthPtr = acceptLength.dataPointer<int32_t>();

    // Validate workspace size
    size_t requiredWorkspaceSize = getEagleAcceptWorkspaceSize(batchSize, numTokens);
    if (workspaceSize < requiredWorkspaceSize)
    {
        throw std::runtime_error("Eagle workspace size too small. Required: " + std::to_string(requiredWorkspaceSize)
            + ", provided: " + std::to_string(workspaceSize));
    }

    // Launch kernel
    launchEagleAcceptKernel(logitsPtr, tokenIdsPtr, attentionMaskPtr, acceptedTokenIdsPtr, acceptedLogitsIndicesPtr,
        acceptLengthPtr, vocabMappingTablePtr, batchSize, numTokens, vocabSize, maxDepth, workspace, workspaceSize,
        stream);
}

} // namespace kernel
} // namespace trt_edgellm