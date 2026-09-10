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

    __device__ __forceinline__ void load(int32_t tokenId, int64_t hiddenSize, uint32_t offset, DVec<half>& out) const
    {
        // Determine which scale group this offset belongs to
        int64_t const group = static_cast<int64_t>(offset) / blockSize;
        float const scale = scales[static_cast<int64_t>(tokenId) * nGroups + group];

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

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isImageToken)
        {
            // Select the deepstack feature row for this image position.
            int32_t const deepstackIdx = multimodalIndices[pos];

            // Validate that deepstackIdx is within bounds
            if (deepstackIdx >= 0 && deepstackIdx < numImageTokens)
            {
                // Load embedding data from deepstack features
                uint32_t const embeddingOffset = deepstackIdx * hiddenSize + offset;
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
        uint32_t const outputIdx = baseOutputIdx + offset;
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

    uint32_t const baseOutputIdx = warpId * hiddenSize;

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
            uint32_t const embeddingOffset = multimodalIdx * hiddenSize + offset;
            embeddingVec.load(imageEmbeds + embeddingOffset);
        }
        else if (isValidAudioToken)
        {
            // Load from FP16 audioEmbeds directly
            uint32_t const embeddingOffset = multimodalIdx * hiddenSize + offset;
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
        uint32_t const outputIdx = baseOutputIdx + offset;
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

template <typename T>
__global__ void gemma4PleGatherKernel(int32_t const* inputIds, T const* pleTable, T* outputBuffer,
    int64_t layerOutputCapacity, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int32_t numLayers,
    int32_t pleHiddenSize, int32_t imageTokenId, int32_t audioTokenId)
{
    int64_t const seqIdx = blockIdx.x;
    int64_t const batchIdx = blockIdx.y;
    int32_t const layerIdx = blockIdx.z;

    if (batchIdx >= batchSize || seqIdx >= seqLen || layerIdx >= numLayers)
    {
        return;
    }

    int64_t const tokenOffset = batchIdx * seqLen + seqIdx;
    int32_t const tokenId = inputIds[tokenOffset];
    bool const zeroFill = tokenId < 0 || tokenId >= vocabSize || (imageTokenId >= 0 && tokenId == imageTokenId)
        || (audioTokenId >= 0 && tokenId == audioTokenId);

    int64_t const outputOffset = static_cast<int64_t>(layerIdx) * layerOutputCapacity + tokenOffset * pleHiddenSize;
    int64_t const tableOffset = (static_cast<int64_t>(tokenId) * numLayers + layerIdx) * pleHiddenSize;

    constexpr uint32_t kVecSize = DVec<T>::vec_size;
    for (int32_t hiddenIdx = threadIdx.x * kVecSize; hiddenIdx < pleHiddenSize; hiddenIdx += blockDim.x * kVecSize)
    {
        DVec<T> valueVec;
        if (zeroFill)
        {
#pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++i)
            {
                valueVec[i] = T{};
            }
        }
        else
        {
            valueVec.load(pleTable + tableOffset + hiddenIdx);
        }
        valueVec.store(outputBuffer + outputOffset + hiddenIdx);
    }
}

template <typename T>
void launchGemma4PleGather(int32_t const* inputIds, T const* pleTable, T* outputBuffer, int64_t layerOutputCapacity,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize,
    int32_t imageTokenId, int32_t audioTokenId, cudaStream_t stream)
{
    constexpr uint32_t kVecSize = DVec<T>::vec_size;
    check::check(pleHiddenSize % kVecSize == 0,
        format::fmtstr("pleHiddenSize must be a multiple of %d for vectorized access", kVecSize));

    dim3 const grid(seqLen, batchSize, numLayers);
    dim3 const block(256);
    gemma4PleGatherKernel<<<grid, block, 0, stream>>>(inputIds, pleTable, outputBuffer, layerOutputCapacity, batchSize,
        seqLen, vocabSize, numLayers, pleHiddenSize, imageTokenId, audioTokenId);
}

// Generate multimodal indices on-device: for each image/audio placeholder position, store the running
// count of that modality's placeholders seen so far (its row in imageEmbeds/audioEmbeds); other
// positions get 0. The counters are global across the whole [batchSize, seqLen] range in batch-major
// order (they are not reset per row), matching the host reference. The scan is inherently sequential, so
// a single thread performs it; the buffer is a prefill-length row of ints and this runs once per prefill,
// avoiding any device<->host copy to build the indices.
//
// When KV cache prefix reuse trims media tokens from the beginning of a sequence, the embedding tensor
// still contains rows for those prefix media items. Per-batch-item base offsets (imageBaseOffsets,
// audioBaseOffsets) shift the counters so suffix tokens index past the prefix rows. At each batch
// boundary (every seqLen positions), the counter resets to that batch item's base offset.
__global__ void generateMultimodalIndicesKernel(int32_t const* inputIds, int32_t* multimodalIndices, int64_t seqLen,
    int32_t batchSize, int32_t imageTokenId, int32_t audioTokenId, int32_t const* imageBaseOffsets,
    int32_t const* audioBaseOffsets)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int32_t imageIndex = (imageBaseOffsets != nullptr) ? imageBaseOffsets[0] : 0;
    int32_t audioIndex = (audioBaseOffsets != nullptr) ? audioBaseOffsets[0] : 0;
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        if (batch > 0)
        {
            imageIndex = (imageBaseOffsets != nullptr) ? imageBaseOffsets[batch] : imageIndex;
            audioIndex = (audioBaseOffsets != nullptr) ? audioBaseOffsets[batch] : audioIndex;
        }
        int64_t const rowStart = static_cast<int64_t>(batch) * seqLen;
        for (int64_t col = 0; col < seqLen; ++col)
        {
            int64_t const pos = rowStart + col;
            int32_t const tokenId = inputIds[pos];
            if (audioTokenId >= 0 && tokenId == audioTokenId)
            {
                multimodalIndices[pos] = audioIndex++;
            }
            else if (imageTokenId >= 0 && tokenId == imageTokenId)
            {
                multimodalIndices[pos] = imageIndex++;
            }
            else
            {
                multimodalIndices[pos] = 0;
            }
        }
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
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(multimodalIndices.getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");

    int32_t const batchSize = static_cast<int32_t>(inputShape[0]);
    int64_t const seqLen = inputShape[1];
    generateMultimodalIndicesKernel<<<1, 1, 0, stream>>>(inputIds.dataPointer<int32_t>(),
        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, imageTokenId.value_or(-1),
        audioTokenId.value_or(-1), imageBaseOffsets, audioBaseOffsets);
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
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(deepstackFeatures.getDataType() == nvinfer1::DataType::kHALF, "deepstackFeatures must be FP16");
    check::check(deepstackEmbeds.getDataType() == nvinfer1::DataType::kHALF, "deepstackEmbeds must be FP16");

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

void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream, rt::OptionalInputTensor multimodalIndices,
    std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId,
    rt::OptionalInputTensor audioEmbeds)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int32_t const vocabSize = static_cast<int32_t>(embeddingShape[0]);
    int64_t const hiddenSize = embeddingShape[1];

    // Validate output shape
    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate common data types for required inputs
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Handle optional image parameters
    bool const hasImage = imageTokenId.has_value() && imageEmbeds.has_value();
    if (hasImage)
    {
        auto const imageShape = imageEmbeds->get().getShape();
        check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
        check::check(imageShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and imageEmbeds");
        check::check(imageEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    }

    // Handle optional audio parameters
    bool const hasAudio = audioTokenId.has_value() && audioEmbeds.has_value();
    if (hasAudio)
    {
        auto const audioShape = audioEmbeds->get().getShape();
        check::check(audioShape.getNumDims() == 2, "audioEmbeds must be 2D tensor [audioTokenLen, hiddenSize]");
        check::check(audioShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and audioEmbeds");
        check::check(audioEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "audioEmbeds must be FP16");
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
            multimodalIndices->get().getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");
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

    // Dispatch based on embedding table datatype
    bool const isFP8Embedding = (embeddingTable.getDataType() == nvinfer1::DataType::kFP8);
    if (isFP8Embedding)
    {
#if SUPPORTS_FP8
        constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
        check::check(scales.has_value(), "scales must be provided for FP8 embedding table");
        auto const& scalesTensor = scales.value().get();
        auto const scaleShape = scalesTensor.getShape();

        check::check(scaleShape.getNumDims() == 2, "scales must be 2D tensor [vocabSize, hiddenSize / blockSize]");
        check::check(scaleShape[0] == vocabSize, "Scale vocab size must match embeddingTable vocab size");
        check::check(scaleShape[1] > 0, "Scale second dimension must be positive");
        check::check(hiddenSize % scaleShape[1] == 0, "hiddenSize must be divisible by number of scale groups");
        check::check(scalesTensor.getDataType() == nvinfer1::DataType::kFLOAT, "scales must be FP32");

        int64_t const nGroups = scaleShape[1];
        int64_t const blockSize = hiddenSize / nGroups;
        check::check(hiddenSize % vecSize == 0,
            format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));
        check::check(blockSize % vecSize == 0,
            format::fmtstr("blockSize must be a multiple of %d for efficient vectorized FP8 dequantization", vecSize));

        __nv_fp8_e4m3 const* tablePtr = embeddingTable.dataPointer<__nv_fp8_e4m3>();
        float const* scalePtr = scalesTensor.dataPointer<float>();

        Fp8EmbeddingLoader loader{tablePtr, scalePtr, blockSize, nGroups};
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
        check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16 or FP8");
        half const* embeddingTablePtr = embeddingTable.dataPointer<half>();

        Fp16EmbeddingLoader loader{embeddingTablePtr};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue, imageEmbedsPtr,
            imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize, seqLen, vocabSize,
            hiddenSize, stream);
    }
}

void gemma4PleGather(rt::Tensor const& inputIds, rt::Tensor const& pleTable, rt::Tensor& outputBuffer,
    int32_t numLayers, int32_t pleHiddenSize, int32_t imageTokenId, int32_t audioTokenId, cudaStream_t stream)
{
    auto const inputShape = inputIds.getShape();
    auto const tableShape = pleTable.getShape();
    auto const outputShape = outputBuffer.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(tableShape.getNumDims() == 2, "pleTable must be 2D tensor [vocabSize, numLayers * pleHiddenSize]");
    check::check(outputShape.getNumDims() == 4,
        "outputBuffer must be 4D tensor [numLayers, maxBatchSize, maxSeqLen, pleHiddenSize]");
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(outputBuffer.getDataType() == pleTable.getDataType(), "outputBuffer dtype must match pleTable dtype");
    check::check(numLayers > 0, "numLayers must be positive");
    check::check(pleHiddenSize > 0, "pleHiddenSize must be positive");
    check::check(outputShape[0] == numLayers, "outputBuffer first dimension must match numLayers");
    check::check(outputShape[3] == pleHiddenSize, "outputBuffer hidden dimension must match pleHiddenSize");
    check::check(tableShape[1] == static_cast<int64_t>(numLayers) * pleHiddenSize,
        "pleTable second dimension must match numLayers * pleHiddenSize");
    check::check(tableShape[0] <= std::numeric_limits<int32_t>::max(), "pleTable vocab dimension exceeds int32 range");
    check::check(
        pleTable.getDataType() == nvinfer1::DataType::kHALF || pleTable.getDataType() == nvinfer1::DataType::kBF16,
        "pleTable must be FP16 or BF16");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    check::check(batchSize <= outputShape[1], "inputIds batch size exceeds outputBuffer capacity");
    check::check(seqLen <= outputShape[2], "inputIds sequence length exceeds outputBuffer capacity");

    int64_t const layerOutputCapacity = outputShape[1] * outputShape[2] * pleHiddenSize;
    int32_t const vocabSize = static_cast<int32_t>(tableShape[0]);
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();

    if (pleTable.getDataType() == nvinfer1::DataType::kHALF)
    {
        launchGemma4PleGather(inputIdsPtr, pleTable.dataPointer<half>(), outputBuffer.dataPointer<half>(),
            layerOutputCapacity, batchSize, seqLen, vocabSize, numLayers, pleHiddenSize, imageTokenId, audioTokenId,
            stream);
    }
    else
    {
        launchGemma4PleGather(inputIdsPtr, pleTable.dataPointer<__nv_bfloat16>(),
            outputBuffer.dataPointer<__nv_bfloat16>(), layerOutputCapacity, batchSize, seqLen, vocabSize, numLayers,
            pleHiddenSize, imageTokenId, audioTokenId, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
