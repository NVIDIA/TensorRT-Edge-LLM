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

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Number of vocab blocks the Markov kernels partition the vocabulary into.
 */
int32_t dsparkMarkovPartialCount(int32_t vocabSize);

/*!
 * @brief Build DSpark base-verify input IDs.
 *
 * verify[b, 0] = lastAcceptedTokens[b]
 * verify[b, j + 1] = draftTokenIds[b, j], j in [0, verifyProposalLen)
 */
void dsparkBuildVerifyTokens(rt::Tensor const& lastAcceptedTokens, rt::Tensor const& draftTokenIds,
    rt::Tensor& verifyTokenIds, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, cudaStream_t stream);

/*!
 * @brief Greedy DSpark accept for a linear proposal block.
 *
 * This deterministic path accepts a draft token while it matches the base
 * argmax at the previous verify position, then appends the base bonus token.
 * It is the top-1 analogue of DeepSpec's rejection-sampling verifier.
 */
void dsparkGreedyAccept(rt::Tensor const& baseLogits, rt::Tensor const& draftTokenIds,
    rt::Tensor const& proposalLengths, rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength,
    rt::Tensor& argmaxScratch, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, int32_t vocabSize,
    cudaStream_t stream);

/*!
 * @brief Fill logical proposal lengths with a fixed full-block value.
 */
void dsparkFillProposalLengths(
    rt::Tensor& proposalLengths, int32_t batchSize, int32_t proposalLen, cudaStream_t stream);

/*!
 * @brief Compute DSpark confidence scores and threshold-scheduled proposal lengths.
 */
void dsparkComputeConfidenceAndProposalLengths(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, rt::Tensor& proposalLengths, int32_t batchSize,
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, float threshold,
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream);

/*!
 * @brief Compute DSpark confidence scores and SPS-scheduled proposal lengths.
 *
 * SPS uses draft-side confidence only. It chooses the logical prefix length
 * that maximizes expected output tokens per estimated B100 verify-bucket cost.
 */
void dsparkComputeConfidenceAndSPSProposalLengths(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, rt::Tensor& proposalLengths, int32_t batchSize,
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, float survivalFloor,
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream);

/*!
 * @brief Compute DSpark per-step acceptance confidence scores only (no scheduling).
 *
 * Used by DDTree drafting to bias tree growth; confidenceScores is [batch, proposalLen].
 */
void dsparkComputeConfidenceScores(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, int32_t batchSize, int32_t proposalLen,
    int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, cudaStream_t stream);

/*!
 * @brief Convert logits to the sampling probability distribution used by DSpark.
 *
 * This materializes a dense probability row after temperature, top-k, and top-p
 * filtering. It is a correctness-first Track A helper for DSpark probabilistic
 * verification; optimized residual paths can avoid keeping all rows later.
 */
void dsparkLogitsToProbabilities(rt::Tensor const& logits, rt::Tensor& probabilities, int32_t rows, int32_t vocabSize,
    float temperature, int32_t topK, float topP, cudaStream_t stream);

/*!
 * @brief Fill a GPU tensor with deterministic uniform random values in [0, 1).
 */
void dsparkFillUniforms(
    rt::Tensor& uniforms, int32_t totalElements, uint64_t philoxSeed, uint64_t philoxOffset, cudaStream_t stream);

/*!
 * @brief Stochastic DSpark vanilla Markov proposal.
 *
 * For each proposal step this builds the corrected Markov distribution,
 * materializes it in draftProbabilities [B, proposalLen, vocabSize], and samples
 * one draft token with proposalUniforms [B, proposalLen].
 */
void dsparkVanillaMarkovSample(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1, rt::Tensor const& markovW2,
    rt::Tensor const& firstPrevTokens, rt::Tensor const& proposalUniforms, rt::Tensor& draftTokenIds,
    rt::Tensor& draftProbabilities, rt::Tensor& correctedLogitsScratch, rt::Tensor& probabilityScratch,
    int32_t batchSize, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, float temperature, int32_t topK,
    float topP, cudaStream_t stream);

/*!
 * @brief Build corrected DSpark Markov logits for one proposal step.
 */
void dsparkBuildMarkovLogits(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1, rt::Tensor const& markovW2,
    rt::Tensor const& firstPrevTokens, rt::Tensor const& draftTokenIds, rt::Tensor& correctedLogitsScratch,
    int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, cudaStream_t stream);

/*!
 * @brief Sample one token per row from probabilityScratch [B, vocabSize].
 */
void dsparkSampleProbabilityRows(rt::Tensor const& probabilityScratch, rt::Tensor const& proposalUniforms,
    rt::Tensor& draftTokenIds, int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize,
    cudaStream_t stream);

/*!
 * @brief Store one proposal step probability row into draftProbabilities [B, P, V].
 */
void dsparkStoreDraftStepProbabilities(rt::Tensor const& probabilityScratch, rt::Tensor& draftProbabilities,
    int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize, cudaStream_t stream);

/*!
 * @brief Normalize selected top-k logits into sparse probabilities.
 */
void dsparkNormalizeTopKRows(rt::Tensor const& topKValues, rt::Tensor& topKProbabilities, int32_t rows, int32_t topK,
    float temperature, cudaStream_t stream);

/*!
 * @brief Store one selected top-1 token per batch row into draftTokenIds [B, P].
 */
void dsparkStoreDraftStepTop1(rt::Tensor const& top1Indices, rt::Tensor& draftTokenIds, int32_t batchSize, int32_t step,
    int32_t proposalLen, cudaStream_t stream);

/*!
 * @brief Sample one DSpark draft token from selected top-k logits and store sparse draft probabilities/indices.
 */
void dsparkSampleTopKRowsAndStore(rt::Tensor const& topKValues, rt::Tensor const& topKIndices,
    rt::Tensor const& proposalUniforms, rt::Tensor& draftTokenIds, rt::Tensor& draftTopKProbabilities,
    rt::Tensor& draftTopKIndices, int32_t batchSize, int32_t step, int32_t proposalLen, int32_t topK, float temperature,
    cudaStream_t stream);

/*!
 * @brief DSpark probabilistic verifier over sparse top-k target/draft supports.
 */
void dsparkSparseTopKAccept(rt::Tensor const& targetTopKProbabilities, rt::Tensor const& targetTopKIndices,
    rt::Tensor const& draftTopKProbabilities, rt::Tensor const& draftTopKIndices, rt::Tensor const& draftTokenIds,
    rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms, rt::Tensor& acceptedTokenIds,
    rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, int32_t targetTopK,
    int32_t draftTopK, cudaStream_t stream);

/*!
 * @brief DSpark probabilistic verifier with residual sampling.
 *
 * targetProbabilities is [B, proposalLen + 1, vocabSize]. draftProbabilities is
 * [B, proposalLen, vocabSize]. acceptUniforms is [B, 2 * proposalLen + 1]:
 * first proposalLen values drive accept/reject, the next proposalLen values
 * drive residual sampling for a rejection at the matching step, and the final
 * value drives the bonus-token sample when the full proposal is accepted.
 */
void dsparkProbabilisticAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& draftProbabilities,
    rt::Tensor const& draftTokenIds, rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride,
    int32_t verifyProposalLen, int32_t vocabSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
