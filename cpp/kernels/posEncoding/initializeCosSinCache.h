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

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Initialize normal RoPE cos/sin cache
 *
 * Precomputes cos/sin values for standard rotary position encoding.
 *
 * @param cosSinCache Output cos/sin cache
 * @param rotaryBaseFrequency Base frequency for RoPE
 * @param rotaryScale Scaling factor
 * @param partialRotaryFactor Fraction of RoPE angle slots initialized with non-identity values
 * @param rotaryDim Rotary embedding dimension
 * @param rotaryEmbeddingMaxPositions Maximum positions
 * @param stream CUDA stream
 */
void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale,
    float partialRotaryFactor, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

/*!
 * @brief Initialize long RoPE cos/sin cache with interpolation
 *
 * Precomputes cos/sin values for long-context RoPE with position interpolation.
 * Used for extending context length beyond original training range.
 *
 * @param shortCosSinCache Short-range cos/sin cache
 * @param longCosSinCache Long-range cos/sin cache
 * @param shortFactor Short interpolation factors
 * @param longFactor Long interpolation factors
 * @param rotaryBaseFrequency Base frequency
 * @param rotaryDim Rotary dimension
 * @param rotaryEmbeddingMaxPositions Maximum positions
 * @param maxPositionEmbeddings Maximum position embeddings
 * @param originalMaxPositionEmbeddings Original max positions from training
 * @param stream CUDA stream
 */
void initializeLongRopeCosSin(float* shortCosSinCache, float* longCosSinCache, float* shortFactor, float* longFactor,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t maxPositionEmbeddings,
    int32_t originalMaxPositionEmbeddings, cudaStream_t stream);

/*!
 * @brief Initialize YaRN RoPE cos/sin cache from a precomputed per-dim inverse frequency.
 *
 * Fills cos/sin as cos(pos * invFreq[d]) * mscale (and sin likewise) on device.
 * ``invFreq`` (length rotaryDim/2) is the NTK-by-parts blended inverse frequency
 * computed once on the host; this kernel only does the position sweep, mirroring
 * ``initializeNormalRopeCosSin`` for the common rotary dims.
 *
 * @param cosSinCache Output cos/sin cache, shape [1, rotaryEmbeddingMaxPositions, rotaryDim]
 * @param invFreq Device array of per-dimension inverse frequencies (length rotaryDim/2)
 * @param mscale YaRN attention magnitude scale applied to cos/sin
 * @param rotaryDim Rotary dimension
 * @param rotaryEmbeddingMaxPositions Maximum positions
 * @param stream CUDA stream
 */
void initializeYarnCosSin(float* cosSinCache, float const* invFreq, float mscale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

/*!
 * @brief Initialize multi-dimensional RoPE cos/sin cache (MRoPE)
 *
 * Precomputes cos/sin values for multi-dimensional rotary encoding (e.g., Qwen2-VL).
 * Supports separate position encodings for different dimensions (temporal, spatial).
 *
 * @param cosSinCache Output cos/sin cache
 * @param mropePositionIds Multi-dimensional position IDs
 * @param rotaryBaseFrequency Base frequency
 * @param rotaryDim Rotary dimension
 * @param rotaryEmbeddingMaxPositions Maximum positions
 * @param batchSize Batch size
 * @param interleaved Whether to use interleaved MRoPE (Qwen3-VL/Qwen3.5 style)
 * @param sectionH Number of frequency pairs for height dimension
 * @param sectionW Number of frequency pairs for width dimension
 * @param stream CUDA stream
 * @param invFreq Optional device array of rotaryDim/2 inverse frequencies (e.g. YaRN); replaces rotaryBaseFrequency
 * @param mscale Scale applied to cos/sin when invFreq is set
 */
void initializeMRopeCosSin(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, int32_t sectionH, int32_t sectionW,
    cudaStream_t stream, float const* invFreq = nullptr, float mscale = 1.0F);

/*!
 * @brief Float-position overload of initializeMRopeCosSin.
 *
 * Identical math and layouts, but takes FLOAT position IDs for models whose temporal
 * positions are fractional (e.g. fps-modulated diffusion streams such as Cosmos3-Edge GEN).
 */
void initializeMRopeCosSin(float* cosSinCache, float* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, int32_t sectionH, int32_t sectionW,
    cudaStream_t stream, float const* invFreq = nullptr, float mscale = 1.0F);

/*!
 * @brief Initialize MRoPE cos/sin cache for text-only inputs with sequential positions.
 *
 * Initializes the MRoPE cache using sequential position IDs (pos[i] = i) for all 3
 * MRoPE sections. This is the correct default for text-only and audio-only modes where
 * no spatial position information is available.
 *
 * @param cosSinCache Output cos/sin cache, shape [batchSize, maxPositions, rotaryDim]
 * @param rotaryBaseFrequency Base frequency
 * @param rotaryDim Rotary dimension
 * @param maxPositions Maximum number of positions
 * @param batchSize Number of batch slots to initialize
 * @param stream CUDA stream
 * @param invFreq Optional device array of rotaryDim/2 inverse frequencies (e.g. YaRN); replaces rotaryBaseFrequency
 * @param mscale Scale applied to cos/sin when invFreq is set
 */
void initializeTextOnlyMRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t maxPositions, int64_t batchSize, cudaStream_t stream, float const* invFreq = nullptr, float mscale = 1.0F);

} // namespace kernel
} // namespace trt_edgellm