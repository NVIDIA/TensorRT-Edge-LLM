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
 * @param rotaryDim Rotary embedding dimension
 * @param rotaryEmbeddingMaxPositions Maximum positions
 * @param stream CUDA stream
 */
void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream);

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
 * @param interleaved Whether to use interleaved MRoPE
 * @param stream CUDA stream
 */
void initializeMRopeCosSin(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm