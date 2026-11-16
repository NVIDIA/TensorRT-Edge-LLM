/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Small-vocabulary specialisations of the DSpark sampling and verification kernels.
//!
//! They differ in how the top-k set is found. DSpark selects it in `topK` passes over the row,
//! each a block-wide max-scan, and drops to a single-threaded walk when top-p is used without
//! top-k or `topK` exceeds its parallel bound; that cost scales with `topK`. A residual-VQ
//! codebook row is small enough to stage in shared memory, so these kernels instead run one
//! MSB-radix select plus a bitonic sort over the surviving candidates — a single pass whose
//! cost is independent of `topK`.
//!
//! Semantics, tensor layouts, and accept/residual/bonus behaviour are identical to the DSpark
//! entry points named in each declaration, which remain the fallback above kCpSpecMaxVocab.
namespace trt_edgellm
{
namespace kernel
{

//! Largest vocabulary these kernels accept: the bound at which a row still fits in shared memory.
constexpr int32_t kCpSpecMaxVocab = 4096;

/*!
 * @brief Whether the CodePredictor speculative path can use these kernels.
 */
bool cpSpecSupportsVocab(int32_t vocabSize);

/*!
 * @brief Temperature + top-k + top-p filtering into dense probability rows.
 *
 * Semantics match dsparkLogitsToProbabilities: the top-k logits are softmaxed,
 * then truncated at the top-p mass and renormalized by that mass. One CTA sorts
 * one row, so cost is independent of top-k.
 */
void cpSpecTopKTopPProbs(rt::Tensor const& logits, rt::Tensor& probabilities, int32_t rows, int32_t vocabSize,
    float temperature, int32_t topK, float topP, cudaStream_t stream);

/*!
 * @brief Sample one token per row: tokenIds[r] ~ probabilities[r], driven by uniforms[r].
 */
void cpSpecSampleRows(rt::Tensor const& probabilities, float const* uniforms, rt::Tensor& tokenIds, int32_t rows,
    int32_t vocabSize, cudaStream_t stream);

/*!
 * @brief Speculative-sampling verifier over dense target/draft rows.
 *
 * Drop-in for dsparkProbabilisticAccept with identical tensor layouts and
 * accept/residual/bonus semantics.
 */
void cpSpecProbabilisticAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& draftProbabilities,
    rt::Tensor const& draftTokenIds, rt::Tensor const& proposalLengths, float const* acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride,
    int32_t verifyProposalLen, int32_t vocabSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
