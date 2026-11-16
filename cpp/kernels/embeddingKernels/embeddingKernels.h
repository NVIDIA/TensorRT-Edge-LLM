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

#pragma once

#include "common/tensor.h"
#include <cuda_runtime.h>
#include <functional>
#include <optional>

namespace trt_edgellm
{
namespace kernel
{

enum class EmbeddingTableFormat
{
    kFP16,
    kFP8,
    kNVFP4,
};

//! \brief Describes the physical representation of an embedding table.
//!
//! Keeping the logical shape separate from the packed shape prevents the runtime from interpreting
//! the byte width of an NVFP4 tensor as its hidden size.
struct EmbeddingTableDescriptor
{
    EmbeddingTableFormat format{EmbeddingTableFormat::kFP16};
    rt::Coords logicalShape;
    rt::Coords packedShape;
    trt_edgellm::DataType packedDataType{trt_edgellm::DataType::kHALF};
    int64_t groupSize{0};
    rt::Coords scaleShape;
    trt_edgellm::DataType scaleDataType{trt_edgellm::DataType::kFLOAT};
    float multiplierGlobalScale{1.0F};
    float modelOutputScale{1.0F};
};

//! \brief Embedding lookup for all modalities (supports FP16, FP8, and packed NVFP4 tables).
//!
//! Produces input embeddings for a batch of token ids. Text tokens are looked up from the embedding
//! table. When image and/or audio embeddings are supplied (prefill), positions whose token id equals
//! imageTokenId / audioTokenId are filled from imageEmbeds / audioEmbeds, selected by multimodalIndices;
//! with no image/audio inputs (decode) it performs a pure text lookup. Automatically dispatches to the
//! FP16 or FP8 implementation based on the embedding table's datatype. Packed NVFP4 requires an
//! explicit descriptor because its physical byte width is half its logical hidden width.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Text embedding table with shape [vocabSize, hiddenSize] (FP16 or FP8)
//! \param[in] scales FP32 per-group scales with shape [vocabSize / rowBlockSize, hiddenSize / blockSize];
//!                   required when embeddingTable is FP8, std::nullopt for FP16
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//! \param[in] multimodalIndices Per-position indices into imageEmbeds/audioEmbeds [batchSize, seqLen];
//!                              required when image or audio inputs are provided, std::nullopt otherwise
//! \param[in] imageTokenId Token ID marking image positions, or std::nullopt if no image input
//! \param[in] imageEmbeds Image embeddings [totalImageTokens, hiddenSize] (FP16), or std::nullopt
//! \param[in] audioTokenId Token ID marking audio positions, or std::nullopt if no audio input
//! \param[in] audioEmbeds Audio embeddings [totalAudioTokens, hiddenSize] (FP16), or std::nullopt
//! \note imageTokenId and audioTokenId must differ when both are provided.
//! \note Element offsets are 64-bit: imageEmbeds, audioEmbeds and output may hold more than 2^32 elements.
//! \throws std::runtime_error if tensor shapes or data types are invalid, or FP8 is not supported
void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream, rt::OptionalInputTensor multimodalIndices = std::nullopt,
    std::optional<int32_t> imageTokenId = std::nullopt, rt::OptionalInputTensor imageEmbeds = std::nullopt,
    std::optional<int32_t> audioTokenId = std::nullopt, rt::OptionalInputTensor audioEmbeds = std::nullopt);

//! \brief Embedding lookup using an explicit physical table descriptor.
void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    EmbeddingTableDescriptor const& descriptor, rt::Tensor& output, cudaStream_t stream,
    rt::OptionalInputTensor multimodalIndices = std::nullopt, std::optional<int32_t> imageTokenId = std::nullopt,
    rt::OptionalInputTensor imageEmbeds = std::nullopt, std::optional<int32_t> audioTokenId = std::nullopt,
    rt::OptionalInputTensor audioEmbeds = std::nullopt);

//! \brief Assemble deepstack embeddings by extracting image token embeddings from deepstack features
//!
//! This function processes input token IDs and selectively extracts embeddings for image tokens from
//! the provided deepstack features. Image tokens are identified by the explicit imageTokenId, and
//! multimodalIndices selects the deepstack feature row for each image position.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] deepstackFeatures Deepstack image features with shape [numImageTokens, hiddenSize]
//! \param[in] imageTokenId Image token ID; positions with this id receive a deepstack feature row
//! \param[in] multimodalIndices Pre-computed indices for image embeddings [batchSize, seqLen]
//! \param[out] deepstackEmbeds Output embeddings with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//! \note Element offsets are 64-bit: deepstackFeatures and deepstackEmbeds may hold more than 2^32 elements.
//! \throws std::runtime_error if tensor shapes or data types are invalid
void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream, int32_t imageTokenId = 0,
    rt::OptionalInputTensor multimodalIndices = std::nullopt);

//! \brief Generate per-position multimodal indices on-device from GPU token IDs.
//!
//! For each position holding an image/audio placeholder token, writes the running count of that
//! modality's placeholders seen so far (the row of imageEmbeds/audioEmbeds to insert); other positions
//! get -1. Counters are global across the whole [batchSize, seqLen] range in batch-major order, matching
//! the host `generateMultimodalIndices` reference. Runs entirely on `stream` — no host round-trip, so
//! no D2H/H2D copy is needed to feed `embeddingLookup`/`assembleDeepstackEmbedding`.
//!
//! When KV cache prefix reuse is active, the suffix token IDs do not include media tokens from the
//! reused prefix, but the embedding tensor still contains rows for ALL media items (prefix + suffix).
//! The optional per-batch base offsets shift each batch item's starting counter so suffix tokens index
//! past the prefix rows. Without offsets (nullptr), counters start at 0 as before.
//!
//! \param[in]  inputIds          GPU token IDs [batchSize, seqLen] (INT32)
//! \param[out] multimodalIndices GPU indices [batchSize, seqLen] (INT32), same element count as inputIds
//! \param[in]  imageTokenId      Image placeholder token id, or std::nullopt if no image
//! \param[in]  audioTokenId      Audio placeholder token id, or std::nullopt if no audio
//! \param[in]  stream            CUDA stream for execution
//! \param[in]  imageBaseOffsets  Device pointer to per-batch image index offsets [batchSize], or nullptr
//! \param[in]  audioBaseOffsets  Device pointer to per-batch audio index offsets [batchSize], or nullptr
void generateMultimodalIndices(rt::Tensor const& inputIds, rt::Tensor& multimodalIndices,
    std::optional<int32_t> imageTokenId = std::nullopt, std::optional<int32_t> audioTokenId = std::nullopt,
    cudaStream_t stream = nullptr, int32_t const* imageBaseOffsets = nullptr,
    int32_t const* audioBaseOffsets = nullptr);

//! Generate contiguous per-request image-block IDs on the device.
//! Text and padding tokens receive `-1`.
void generateVisionBlockIds(
    rt::Tensor const& inputIds, rt::Tensor& visionBlockIds, int32_t imageTokenId, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
