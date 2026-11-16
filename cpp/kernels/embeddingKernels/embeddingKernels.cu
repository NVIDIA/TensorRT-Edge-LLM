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
#include "common/stringUtils.h"
#include "embeddingKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#if SUPPORTS_FP8
#include <cuda_fp8.h>
#endif

namespace trt_edgellm
{
namespace kernel
{

namespace
{

//! \brief FP16 embedding loader for template-based kernel
struct Fp16EmbeddingLoader
{
    static constexpr uint32_t vecSize = DVec<half>::vec_size;
    half const* table{nullptr};

    __device__ __forceinline__ void load(int32_t tokenId, int64_t hiddenSize, uint32_t offset, DVec<half>& out) const
    {
        out.load(table + static_cast<int64_t>(tokenId) * hiddenSize + offset);
    }
};

#if SUPPORTS_FP8
//! \brief FP8 embedding loader with per-group dequantization
struct Fp8EmbeddingLoader
{
    static constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
    __nv_fp8_e4m3 const* table{nullptr};
    float const* scales{nullptr};
    int64_t blockSize{0};
    int64_t nGroups{0};
    int64_t rowBlockSize{1};

    __device__ __forceinline__ void load(int32_t tokenId, int64_t hiddenSize, uint32_t offset, DVec<half>& out) const
    {
        // Determine which scale group this offset belongs to
        int64_t const group = static_cast<int64_t>(offset) / blockSize;
        float const scale = scales[(static_cast<int64_t>(tokenId) / rowBlockSize) * nGroups + group];

        // Load FP8 values and dequantize to FP16
        DVec<__nv_fp8_e4m3> in;
        in.load(table + static_cast<int64_t>(tokenId) * hiddenSize + offset);
#pragma unroll
        for (uint32_t i = 0; i < vecSize; ++i)
        {
            out[i] = __float2half(static_cast<float>(in[i]) * scale);
        }
    }
};

__device__ __forceinline__ float decodeE2m1(uint8_t nibble)
{
    uint32_t const magnitude = nibble & 0x7U;
    float value = magnitude == 0U ? 0.0F : magnitude == 1U ? 0.5F : __uint_as_float(0x3F000000U + (magnitude << 22U));
    return (nibble & 0x8U) != 0U ? -value : value;
}

//! \brief Packed NVFP4 loader. Only the requested rows are decoded; no dense FP16 table is created.
struct Nvfp4EmbeddingLoader
{
    static constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint8_t const* table{nullptr};
    __nv_fp8_e4m3 const* scales{nullptr};
    int64_t packedRowSize{0};
    int64_t groupSize{0};
    int64_t nGroups{0};
    float scalarScale{1.0F};

    __device__ __forceinline__ void load(int32_t tokenId, int64_t, uint32_t offset, DVec<half>& out) const
    {
        int64_t const rowOffset = static_cast<int64_t>(tokenId) * packedRowSize;
        int64_t const scaleOffset = static_cast<int64_t>(tokenId) * nGroups + static_cast<int64_t>(offset) / groupSize;
        float const scale = static_cast<float>(scales[scaleOffset]) * scalarScale;

#pragma unroll
        for (uint32_t i = 0; i < vecSize / 2; ++i)
        {
            uint8_t const packed = table[rowOffset + offset / 2 + i];
            // The provider contract stores the even element in the low nibble and the odd element in the high.
            out[2 * i] = __float2half(decodeE2m1(packed & 0xFU) * scale);
            out[2 * i + 1] = __float2half(decodeE2m1(packed >> 4U) * scale);
        }
    }
};
#endif

// CUDA kernel for assembling deepstack embeddings (FP16 only)
// Extracts image-token embeddings from deepstack features. Positions whose token id == imageTokenId
// are mapped to a deepstack feature row via multimodalIndices; all other positions get zero embeddings.
__global__ void assembleDeepstackEmbeddingKernel(int32_t const* inputIds, half const* deepstackFeatures, half* output,
    int64_t batchSize, int64_t seqLen, int32_t imageTokenId, int64_t hiddenSize, int64_t numImageTokens,
    int32_t const* multimodalIndices)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Get token ID (warpId == batchIdx * seqLen + tokenIdx)
    int64_t const pos = static_cast<int64_t>(warpId);
    int32_t const tokenId = inputIds[pos];

    // Image tokens are identified by the explicit imageTokenId; their deepstack
    // feature row is selected via multimodalIndices.
    bool const isImageToken = (imageTokenId > 0 && tokenId == imageTokenId);

    int64_t const baseOutputIdx = pos * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isImageToken)
        {
            // Select the deepstack feature row for this image position.
            int32_t const deepstackIdx = multimodalIndices[pos];

            // -1 marks a placeholder-id token with no feature row, e.g. a compact-step decode row that
            // sampled the image token id; it gets a zero row.
            if (deepstackIdx >= 0 && deepstackIdx < numImageTokens)
            {
                // Load embedding data from deepstack features
                int64_t const embeddingOffset = static_cast<int64_t>(deepstackIdx) * hiddenSize + offset;
                embeddingVec.load(deepstackFeatures + embeddingOffset);
            }
            else
            {
                // Out-of-bounds image token, use zero embedding
#pragma unroll
                for (uint32_t i = 0; i < vecSize; ++i)
                {
                    embeddingVec[i] = __float2half(0.0f);
                }
            }
        }
        else
        {
            // Not an image token, use zero embedding
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        int64_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

//! \brief Template-based embedding lookup kernel (text + optional image/audio)
//! Text tokens use the template loader (FP16 or FP8), image/audio tokens use FP16 embeds
//! \tparam TLoader Embedding loader type (Fp16EmbeddingLoader or Fp8EmbeddingLoader)
template <typename TLoader>
__global__ void embeddingLookupKernelImpl(int32_t const* inputIds, TLoader loader, int32_t const* multimodalIndices,
    int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen, int32_t audioTokenId, half const* audioEmbeds,
    int64_t audioTokenLen, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize)
{
    // Each warp handles one hidden state (one token's embedding)
    constexpr uint32_t vecSize = TLoader::vecSize;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;
    uint32_t const linearIdx = batchIdx * seqLen + tokenIdx;

    // Get token ID
    int32_t const tokenId = inputIds[linearIdx];

    // Determine token type
    bool isImageToken = (imageEmbeds != nullptr && tokenId == imageTokenId);
    bool isAudioToken = (audioEmbeds != nullptr && tokenId == audioTokenId);
    bool isValidTextToken = false;
    bool isValidImageToken = false;
    bool isValidAudioToken = false;
    int32_t multimodalIdx = 0;

    if (isImageToken)
    {
        multimodalIdx = multimodalIndices[linearIdx];
        isValidImageToken = (multimodalIdx >= 0 && multimodalIdx < imageTokenLen);
    }
    else if (isAudioToken)
    {
        multimodalIdx = multimodalIndices[linearIdx];
        isValidAudioToken = (multimodalIdx >= 0 && multimodalIdx < audioTokenLen);
    }
    else
    {
        isValidTextToken = (tokenId >= 0 && tokenId < vocabSize);
    }

    // Row offsets are 64-bit: per-slot multimodal reservations can hold more than 2^32 elements.
    int64_t const baseOutputIdx = static_cast<int64_t>(warpId) * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isValidTextToken)
        {
            // Use template loader for text tokens (supports FP16 or FP8)
            loader.load(tokenId, hiddenSize, offset, embeddingVec);
        }
        else if (isValidImageToken)
        {
            // Load from FP16 imageEmbeds directly
            int64_t const embeddingOffset = static_cast<int64_t>(multimodalIdx) * hiddenSize + offset;
            embeddingVec.load(imageEmbeds + embeddingOffset);
        }
        else if (isValidAudioToken)
        {
            // Load from FP16 audioEmbeds directly
            int64_t const embeddingOffset = static_cast<int64_t>(multimodalIdx) * hiddenSize + offset;
            embeddingVec.load(audioEmbeds + embeddingOffset);
        }
        else
        {
            // Use zero embedding for out-of-bounds tokens
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        int64_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// Template helper function to launch multimodal embedding lookup kernel
template <typename TLoader>
void launchEmbeddingLookupKernel(int32_t const* inputIds, TLoader const& loader, int32_t const* multimodalIndices,
    int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen, int32_t audioTokenId, half const* audioEmbeds,
    int64_t audioTokenLen, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    cudaStream_t stream)
{
    constexpr uint32_t vecSize = TLoader::vecSize;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupKernelImpl<<<gridSize, threadsPerBlock, 0, stream>>>(inputIds, loader, multimodalIndices,
        imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen, output, batchSize, seqLen,
        vocabSize, hiddenSize);
}

} // namespace

constexpr int32_t kMetadataScanThreads{256};

template <bool PerBatchOffsets>
__global__ void generateMultimodalIndicesKernel(int32_t const* inputIds, int32_t* multimodalIndices, int64_t seqLen,
    int32_t batchSize, int32_t imageTokenId, int32_t audioTokenId, int32_t const* imageBaseOffsets,
    int32_t const* audioBaseOffsets)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t imageBase;
    __shared__ int32_t audioBase;

    int32_t const batch = PerBatchOffsets ? static_cast<int32_t>(blockIdx.x) : 0;
    int64_t const begin = PerBatchOffsets ? static_cast<int64_t>(batch) * seqLen : 0;
    int64_t const end = PerBatchOffsets ? begin + seqLen : static_cast<int64_t>(batchSize) * seqLen;
    if (threadIdx.x == 0)
    {
        imageBase = PerBatchOffsets && imageBaseOffsets != nullptr ? imageBaseOffsets[batch] : 0;
        audioBase = PerBatchOffsets && audioBaseOffsets != nullptr ? audioBaseOffsets[batch] : 0;
    }
    __syncthreads();

    for (int64_t tile = begin; tile < end; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const valid = position < end;
        int32_t const tokenId = valid ? inputIds[position] : -1;
        int32_t const isAudio = valid && audioTokenId >= 0 && tokenId == audioTokenId;
        int32_t const isImage = valid && !isAudio && imageTokenId >= 0 && tokenId == imageTokenId;
        int32_t imagePrefix{};
        int32_t imageCount{};
        BlockScan(scanStorage).ExclusiveSum(isImage, imagePrefix, imageCount);
        if (valid && isImage)
        {
            multimodalIndices[position] = imageBase + imagePrefix;
        }
        __syncthreads();
        int32_t audioPrefix{};
        int32_t audioCount{};
        BlockScan(scanStorage).ExclusiveSum(isAudio, audioPrefix, audioCount);
        if (valid)
        {
            multimodalIndices[position] = isAudio ? audioBase + audioPrefix
                : isImage                         ? multimodalIndices[position]
                                                  : -1;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            imageBase += imageCount;
            audioBase += audioCount;
        }
        __syncthreads();
    }
}

template <bool PerBatchOffsets>
__global__ void generateSingleModalityIndicesKernel(int32_t const* inputIds, int32_t* multimodalIndices, int64_t seqLen,
    int32_t batchSize, int32_t tokenId, int32_t const* baseOffsets)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t base;

    int32_t const batch = PerBatchOffsets ? static_cast<int32_t>(blockIdx.x) : 0;
    int64_t const begin = PerBatchOffsets ? static_cast<int64_t>(batch) * seqLen : 0;
    int64_t const end = PerBatchOffsets ? begin + seqLen : static_cast<int64_t>(batchSize) * seqLen;
    if (threadIdx.x == 0)
    {
        base = PerBatchOffsets ? baseOffsets[batch] : 0;
    }
    __syncthreads();

    for (int64_t tile = begin; tile < end; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const matches = position < end && inputIds[position] == tokenId;
        int32_t prefix{};
        int32_t count{};
        BlockScan(scanStorage).ExclusiveSum(static_cast<int32_t>(matches), prefix, count);
        if (matches)
        {
            multimodalIndices[position] = base + prefix;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            base += count;
        }
        __syncthreads();
    }
}

__global__ void generateVisionBlockIdsKernel(
    int32_t const* inputIds, int32_t* visionBlockIds, int64_t batchSize, int64_t seqLen, int32_t imageTokenId)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t blockBase;
    int64_t const batch = blockIdx.x;
    if (batch >= batchSize)
    {
        return;
    }

    int64_t const base = batch * seqLen;
    if (threadIdx.x == 0)
    {
        blockBase = 0;
    }
    __syncthreads();
    for (int64_t tile = 0; tile < seqLen; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const valid = position < seqLen;
        bool const isVision = valid && inputIds[base + position] == imageTokenId;
        bool const startsBlock = isVision && (position == 0 || inputIds[base + position - 1] != imageTokenId);
        int32_t precedingBlocks{};
        int32_t tileBlocks{};
        BlockScan(scanStorage).ExclusiveSum(static_cast<int32_t>(startsBlock), precedingBlocks, tileBlocks);
        if (valid)
        {
            int32_t const currentBlock = blockBase + precedingBlocks - (startsBlock ? 0 : 1);
            visionBlockIds[base + position] = isVision ? currentBlock : -1;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            blockBase += tileBlocks;
        }
        __syncthreads();
    }
}

void generateMultimodalIndices(rt::Tensor const& inputIds, rt::Tensor& multimodalIndices,
    std::optional<int32_t> imageTokenId, std::optional<int32_t> audioTokenId, cudaStream_t stream,
    int32_t const* imageBaseOffsets, int32_t const* audioBaseOffsets)
{
    auto const inputShape = inputIds.getShape();
    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(multimodalIndices.getShape().volume() == inputShape.volume(),
        "multimodalIndices must have the same element count as inputIds");
    check::check(inputIds.getDataType() == trt_edgellm::DataType::kINT32, "inputIds must be INT32");
    check::check(multimodalIndices.getDataType() == trt_edgellm::DataType::kINT32, "multimodalIndices must be INT32");

    int32_t const batchSize = static_cast<int32_t>(inputShape[0]);
    int64_t const seqLen = inputShape[1];
    bool const imageHasOffsets = imageBaseOffsets != nullptr;
    bool const audioHasOffsets = audioBaseOffsets != nullptr;
    if (imageHasOffsets == audioHasOffsets)
    {
        if (imageHasOffsets)
        {
            generateMultimodalIndicesKernel<true><<<batchSize, kMetadataScanThreads, 0, stream>>>(
                inputIds.dataPointer<int32_t>(), multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize,
                imageTokenId.value_or(-1), audioTokenId.value_or(-1), imageBaseOffsets, audioBaseOffsets);
        }
        else
        {
            generateMultimodalIndicesKernel<false><<<1, kMetadataScanThreads, 0, stream>>>(
                inputIds.dataPointer<int32_t>(), multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize,
                imageTokenId.value_or(-1), audioTokenId.value_or(-1), nullptr, nullptr);
        }
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(
            multimodalIndices.rawPointer(), 0xFF, static_cast<size_t>(inputShape.volume()) * sizeof(int32_t), stream));
        if (imageTokenId.has_value())
        {
            if (imageHasOffsets)
            {
                generateSingleModalityIndicesKernel<true>
                    <<<batchSize, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *imageTokenId, imageBaseOffsets);
            }
            else
            {
                generateSingleModalityIndicesKernel<false>
                    <<<1, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *imageTokenId, nullptr);
            }
        }
        if (audioTokenId.has_value())
        {
            if (audioHasOffsets)
            {
                generateSingleModalityIndicesKernel<true>
                    <<<batchSize, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *audioTokenId, audioBaseOffsets);
            }
            else
            {
                generateSingleModalityIndicesKernel<false>
                    <<<1, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *audioTokenId, nullptr);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void generateVisionBlockIds(
    rt::Tensor const& inputIds, rt::Tensor& visionBlockIds, int32_t imageTokenId, cudaStream_t stream)
{
    auto const shape = inputIds.getShape();
    check::check(shape.getNumDims() == 2, "inputIds must be 2D tensor");
    check::check(inputIds.getDeviceType() == rt::DeviceType::kGPU, "inputIds must be a GPU tensor");
    check::check(inputIds.getDataType() == trt_edgellm::DataType::kINT32, "inputIds must have INT32 dtype");
    check::check(visionBlockIds.getDeviceType() == rt::DeviceType::kGPU
            && visionBlockIds.getDataType() == trt_edgellm::DataType::kINT32
            && visionBlockIds.getShape().volume() == shape.volume(),
        "visionBlockIds must be an equally-sized GPU INT32 tensor");
    check::check(imageTokenId >= 0, "imageTokenId must be non-negative");

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    generateVisionBlockIdsKernel<<<batchSize, kMetadataScanThreads, 0, stream>>>(
        inputIds.dataPointer<int32_t>(), visionBlockIds.dataPointer<int32_t>(), batchSize, seqLen, imageTokenId);
    CUDA_CHECK(cudaGetLastError());
}

void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream, int32_t imageTokenId, rt::OptionalInputTensor multimodalIndices)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const featuresShape = deepstackFeatures.getShape();
    auto const outputShape = deepstackEmbeds.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(featuresShape.getNumDims() == 2, "deepstackFeatures must be 2D tensor [numImageTokens, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "deepstackEmbeds must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const numImageTokens = featuresShape[0];
    int64_t const hiddenSize = featuresShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == trt_edgellm::DataType::kINT32, "inputIds must be INT32");
    check::check(deepstackFeatures.getDataType() == trt_edgellm::DataType::kHALF, "deepstackFeatures must be FP16");
    check::check(deepstackEmbeds.getDataType() == trt_edgellm::DataType::kHALF, "deepstackEmbeds must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* deepstackFeaturesPtr = deepstackFeatures.dataPointer<half>();
    half* outputPtr = deepstackEmbeds.dataPointer<half>();

    // Launch kernel
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Image positions are mapped to deepstack feature rows via multimodalIndices.
    check::check(multimodalIndices.has_value(), "assembleDeepstackEmbedding requires multimodalIndices");
    int32_t const* multimodalIndicesPtr = multimodalIndices.value().get().dataPointer<int32_t>();

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    assembleDeepstackEmbeddingKernel<<<gridSize, threadsPerBlock, 0, stream>>>(inputIdsPtr, deepstackFeaturesPtr,
        outputPtr, batchSize, seqLen, imageTokenId, hiddenSize, numImageTokens, multimodalIndicesPtr);
}

static void embeddingLookupImpl(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor scales, rt::Tensor& output, cudaStream_t stream, rt::OptionalInputTensor multimodalIndices,
    std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId,
    rt::OptionalInputTensor audioEmbeds, EmbeddingTableDescriptor const* descriptor)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 1 || inputShape.getNumDims() == 2,
        "inputIds must have shape [physicalTokens] or [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 2 || outputShape.getNumDims() == 3,
        "output must be token-major [physicalTokens, hiddenSize] or [batchSize, seqLen, hiddenSize]");

    bool const flat = inputShape.getNumDims() == 1;
    int64_t const batchSize = flat ? 1 : inputShape[0];
    int64_t const seqLen = flat ? inputShape[0] : inputShape[1];

    if (descriptor != nullptr)
    {
        check::check(descriptor->logicalShape.getNumDims() == 2,
            "embedding descriptor logicalShape must be [vocabSize, hiddenSize]");
        check::check(descriptor->packedShape == embeddingShape,
            "embedding descriptor packedShape must match the embedding table shape");
        check::check(descriptor->packedDataType == embeddingTable.getDataType(),
            "embedding descriptor packedDataType must match the embedding table datatype");
    }

    auto const& logicalShape = descriptor == nullptr ? embeddingShape : descriptor->logicalShape;
    int32_t const vocabSize = static_cast<int32_t>(logicalShape[0]);
    int64_t const hiddenSize = logicalShape[1];
    check::check(
        logicalShape[0] == embeddingShape[0], "embedding descriptor logical and packed vocab dimensions must match");

    // Validate output shape
    if (outputShape.getNumDims() == 2)
    {
        check::check(outputShape[0] == batchSize * seqLen, "Output physical token count mismatch");
        check::check(outputShape[1] == hiddenSize, "Output hidden size mismatch");
    }
    else
    {
        check::check(outputShape[0] == batchSize, "Output batch size mismatch");
        check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
        check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");
    }

    // Validate common data types for required inputs
    check::check(inputIds.getDataType() == trt_edgellm::DataType::kINT32, "inputIds must be INT32");
    check::check(output.getDataType() == trt_edgellm::DataType::kHALF, "output must be FP16");

    // Handle optional image parameters
    bool const hasImage = imageTokenId.has_value() && imageEmbeds.has_value();
    if (hasImage)
    {
        auto const imageShape = imageEmbeds->get().getShape();
        check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
        check::check(imageShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and imageEmbeds");
        check::check(imageEmbeds->get().getDataType() == trt_edgellm::DataType::kHALF, "imageEmbeds must be FP16");
    }

    // Handle optional audio parameters
    bool const hasAudio = audioTokenId.has_value() && audioEmbeds.has_value();
    if (hasAudio)
    {
        auto const audioShape = audioEmbeds->get().getShape();
        check::check(audioShape.getNumDims() == 2, "audioEmbeds must be 2D tensor [audioTokenLen, hiddenSize]");
        check::check(audioShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and audioEmbeds");
        check::check(audioEmbeds->get().getDataType() == trt_edgellm::DataType::kHALF, "audioEmbeds must be FP16");
    }

    // Validate that imageTokenId and audioTokenId are different when both are present
    if (hasImage && hasAudio)
    {
        check::check(*imageTokenId != *audioTokenId, "imageTokenId and audioTokenId must be different");
    }

    // Validate multimodalIndices if any multimodal input is present
    bool const hasMultimodal = hasImage || hasAudio;
    if (hasMultimodal)
    {
        check::check(
            multimodalIndices.has_value(), "multimodalIndices is required when image or audio inputs are provided");
        auto const multimodalIndicesShape = multimodalIndices->get().getShape();
        check::check(
            multimodalIndicesShape.getNumDims() == 2, "multimodalIndices must be 2D tensor [batchSize, seqLen]");
        check::check(multimodalIndicesShape[0] == batchSize, "multimodalIndices batch size mismatch");
        check::check(multimodalIndicesShape[1] == seqLen, "multimodalIndices sequence length mismatch");
        check::check(
            multimodalIndices->get().getDataType() == trt_edgellm::DataType::kINT32, "multimodalIndices must be INT32");
    }

    // Extract values for kernel - use safe defaults when modalities are absent
    int32_t const imageTokenIdValue = hasImage ? *imageTokenId : -1;
    half const* imageEmbedsPtr = hasImage ? imageEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const imageTokenLen = hasImage ? imageEmbeds->get().getShape()[0] : 0;

    int32_t const audioTokenIdValue = hasAudio ? *audioTokenId : -1;
    half const* audioEmbedsPtr = hasAudio ? audioEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const audioTokenLen = hasAudio ? audioEmbeds->get().getShape()[0] : 0;

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t const* multimodalIndicesPtr = hasMultimodal ? multimodalIndices->get().dataPointer<int32_t>() : nullptr;
    half* outputPtr = output.dataPointer<half>();

    EmbeddingTableFormat const format = descriptor != nullptr         ? descriptor->format
        : embeddingTable.getDataType() == trt_edgellm::DataType::kFP8 ? EmbeddingTableFormat::kFP8
                                                                      : EmbeddingTableFormat::kFP16;

    if (format == EmbeddingTableFormat::kNVFP4)
    {
#if SUPPORTS_FP8
        check::check(descriptor != nullptr, "packed NVFP4 embedding tables require an explicit descriptor");
        check::check(embeddingTable.getDataType() == trt_edgellm::DataType::kUINT8,
            "packed NVFP4 embedding storage must use an unsigned byte tensor");
        check::check(hiddenSize > 0 && hiddenSize % 2 == 0, "NVFP4 logical hidden size must be positive and even");
        check::check(
            embeddingShape[1] == hiddenSize / 2, "NVFP4 packed table width must be half the logical hidden size");
        check::check(descriptor->groupSize > 0 && hiddenSize % descriptor->groupSize == 0,
            "NVFP4 logical hidden size must be divisible by groupSize");
        check::check(descriptor->groupSize % Nvfp4EmbeddingLoader::vecSize == 0,
            "NVFP4 groupSize must be a multiple of the vector width");
        check::check(scales.has_value(), "NVFP4 block scales are required");
        check::check(std::isfinite(descriptor->multiplierGlobalScale), "NVFP4 multiplier global scale must be finite");
        check::check(std::isfinite(descriptor->modelOutputScale), "NVFP4 model output scale must be finite");
        check::check(std::isfinite(descriptor->multiplierGlobalScale * descriptor->modelOutputScale),
            "NVFP4 combined scalar scale must be finite");

        auto const& scalesTensor = scales->get();
        int64_t const nGroups = hiddenSize / descriptor->groupSize;
        rt::Coords const expectedScaleShape{vocabSize, nGroups};
        check::check(descriptor->scaleShape == expectedScaleShape,
            "NVFP4 descriptor scaleShape must be [vocabSize, hiddenSize / groupSize]");
        check::check(
            scalesTensor.getShape() == expectedScaleShape, "NVFP4 block scale shape must match the descriptor");
        check::check(descriptor->scaleDataType == trt_edgellm::DataType::kFP8
                && scalesTensor.getDataType() == trt_edgellm::DataType::kFP8,
            "NVFP4 block scales must use FP8 E4M3");

        Nvfp4EmbeddingLoader loader{embeddingTable.dataPointer<uint8_t>(), scalesTensor.dataPointer<__nv_fp8_e4m3>(),
            embeddingShape[1], descriptor->groupSize, nGroups,
            descriptor->multiplierGlobalScale * descriptor->modelOutputScale};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue, imageEmbedsPtr,
            imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize, seqLen, vocabSize,
            hiddenSize, stream);
#else
        check::check(
            false, "NVFP4 embedding lookup is unavailable: build does not support FP8 block scales (SUPPORTS_FP8=0)");
#endif
    }
    else if (format == EmbeddingTableFormat::kFP8)
    {
#if SUPPORTS_FP8
        constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
        check::check(embeddingTable.getDataType() == trt_edgellm::DataType::kFP8,
            "FP8 embedding table descriptor requires FP8 storage");
        check::check(scales.has_value(), "scales must be provided for FP8 embedding table");
        auto const& scalesTensor = scales.value().get();
        auto const scaleShape = scalesTensor.getShape();

        check::check(scaleShape.getNumDims() == 2, "scales must be a 2D tensor");
        check::check(scaleShape[0] > 0 && vocabSize % scaleShape[0] == 0,
            "Scale rows must evenly divide embeddingTable vocab size");
        check::check(scaleShape[1] > 0, "Scale second dimension must be positive");
        check::check(hiddenSize % scaleShape[1] == 0, "hiddenSize must be divisible by number of scale groups");
        check::check(scalesTensor.getDataType() == trt_edgellm::DataType::kFLOAT, "scales must be FP32");

        int64_t const nGroups = scaleShape[1];
        int64_t const blockSize = hiddenSize / nGroups;
        int64_t const rowBlockSize = vocabSize / scaleShape[0];
        check::check(hiddenSize % vecSize == 0,
            format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));
        check::check(blockSize % vecSize == 0,
            format::fmtstr("blockSize must be a multiple of %d for efficient vectorized FP8 dequantization", vecSize));

        __nv_fp8_e4m3 const* tablePtr = embeddingTable.dataPointer<__nv_fp8_e4m3>();
        float const* scalePtr = scalesTensor.dataPointer<float>();

        Fp8EmbeddingLoader loader{tablePtr, scalePtr, blockSize, nGroups, rowBlockSize};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue, imageEmbedsPtr,
            imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize, seqLen, vocabSize,
            hiddenSize, stream);
#else
        check::check(
            false, "FP8 multimodal embedding lookup is unavailable: build does not support FP8 (SUPPORTS_FP8=0)");
#endif
    }
    else
    {
        check::check(format == EmbeddingTableFormat::kFP16, "unsupported embedding table format");
        check::check(embeddingTable.getDataType() == trt_edgellm::DataType::kHALF,
            "FP16 embedding table descriptor requires FP16 storage");
        check::check(!scales.has_value(), "FP16 embedding tables must not provide block scales");
        half const* embeddingTablePtr = embeddingTable.dataPointer<half>();

        Fp16EmbeddingLoader loader{embeddingTablePtr};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue, imageEmbedsPtr,
            imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize, seqLen, vocabSize,
            hiddenSize, stream);
    }
}

void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream, rt::OptionalInputTensor multimodalIndices,
    std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId,
    rt::OptionalInputTensor audioEmbeds)
{
    embeddingLookupImpl(inputIds, embeddingTable, scales, output, stream, multimodalIndices, imageTokenId, imageEmbeds,
        audioTokenId, audioEmbeds, nullptr);
}

void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    EmbeddingTableDescriptor const& descriptor, rt::Tensor& output, cudaStream_t stream,
    rt::OptionalInputTensor multimodalIndices, std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds,
    std::optional<int32_t> audioTokenId, rt::OptionalInputTensor audioEmbeds)
{
    if (descriptor.format == EmbeddingTableFormat::kNVFP4)
    {
        embeddingLookupImpl(inputIds, embeddingTable, scales, output, stream, multimodalIndices, imageTokenId,
            imageEmbeds, audioTokenId, audioEmbeds, &descriptor);
    }
    else
    {
        embeddingLookup(inputIds, embeddingTable, scales, output, stream, multimodalIndices, imageTokenId, imageEmbeds,
            audioTokenId, audioEmbeds);
    }
}

} // namespace kernel
} // namespace trt_edgellm
