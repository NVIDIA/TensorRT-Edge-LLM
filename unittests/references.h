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

#include <cstdint>
#include <cuda_fp16.h>
#include <optional>
#include <set>
#include <vector>

std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k, std::vector<half> const& v,
    int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    std::optional<std::vector<int32_t>> const& treeAttnMask = std::nullopt);

std::vector<half> ropeRef(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, int32_t const seqIdx, float const ropeScale, float const ropeTheta, bool const permute);

std::vector<half> ropeRefCosSin(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, std::vector<float> const& cos, std::vector<float> const& sin, bool const permute);

// Sampling reference functions
std::vector<float> softmaxRef(std::vector<float> const& logits, float temperature = 1.0f);

std::set<int32_t> getTopKAllowedTokensRef(std::vector<float> const& logits, int32_t topK);

std::set<int32_t> getTopPAllowedTokensRef(std::vector<float> const& logits, float topP, float temperature);

std::set<int32_t> getCombinedAllowedTokensRef(
    std::vector<float> const& logits, int32_t topK, float topP, float temperature);

std::vector<std::pair<float, int32_t>> getTopKElementsRef(std::vector<float> const& logits, int32_t topK);

// Returns top-K indices and raw values only
std::vector<std::pair<float, int32_t>> returnAllTopKReference(std::vector<float> const& input, int32_t topK);

void computeLongRopeReference(std::vector<float>& shortCosSinCache, std::vector<float>& longCosSinCache,
    std::vector<float> const& shortFactor, std::vector<float> const& longFactor, float rotaryBaseFrequency,
    int32_t rotaryDim, int32_t kvCacheCapacity, int32_t rotaryEmbeddingMaxPositions,
    int32_t originalMaxPositionEmbeddings);

void computeMRopeReference(std::vector<float>& mropeRotaryCosSin, std::vector<int64_t> const& mropePositionIds,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    bool interleaved);

// Embedding lookup reference functions
std::vector<half> embeddingLookupRef(std::vector<int32_t> const& inputIds, std::vector<half> const& embeddingTable,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    std::optional<std::vector<half>> const& imageEmbeds = std::nullopt, int64_t imageTokenLen = 0);

// Eagle reference functions
void assembleDraftTreeDescReference(std::vector<int8_t> const& draftTreeMask,
    std::vector<int32_t> const& draftTreeLength, std::vector<int32_t> const& sequenceStartIndex,
    std::vector<int32_t>& packedDraftTreeMask, std::vector<int32_t>& tensorPositionIndices,
    int32_t paddedDraftTreeSize);

void prepareEagleDraftProposalMiscInputReference(std::vector<int32_t> const& draftTreeLength,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t selectTokenLength, int32_t paddedDraftTreeSize);

void prepareEaglePrefillInputReference(
    std::vector<int32_t>& sequenceContextLengths, std::vector<int64_t>& selectTokenIndices, int32_t sequenceLength);

void prepareEagleAcceptDecodeTokenInputReference(std::vector<int32_t> const& sequenceStartIndices,
    std::vector<int32_t>& packedTreeMask, std::vector<int32_t>& tensorPositionIndices,
    std::vector<int64_t>& selectTokenIndices, std::vector<int32_t>& sequenceContextLengths, int32_t acceptedTokenNum);

// Eagle base reference functions
void prepareEagleBaseTreeDecodingInputReference(std::vector<int8_t> const& baseTreeDecodingMask,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& packedBaseTreeDecodingMask,
    std::vector<int32_t>& tensorPositionIndices, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t treeSize);

void eagleBaseCommitKVCacheAndAssembleHiddenStateReference(std::vector<int32_t> const& acceptedIndices,
    std::vector<int32_t> const& acceptLengths, std::vector<half> const& kvCacheBuffer,
    std::vector<int32_t> const& kvCacheLengths, std::vector<half> const& hiddenState,
    std::vector<half>& kvCacheBufferOut, std::vector<half>& hiddenStateOut, int32_t const numLayers,
    int32_t const maxBatchSize, int32_t const numHeads, int32_t const maxSeqLen, int32_t const headDim,
    int32_t const maxDepth, int32_t const draftTreeSize, int32_t const baseHiddenDim);

// Eagle accept reference function
struct EagleAcceptResult
{
    std::vector<int32_t> acceptedTokenIds;
    std::vector<int32_t> acceptedLogitsIndices;
    std::vector<int32_t> acceptLengths;
    int32_t maxAcceptLength;
};

EagleAcceptResult eagleAcceptRef(std::vector<float> const& logits, std::vector<int32_t> const& tokenIds,
    std::vector<int8_t> const& attentionMask, int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth,
    std::vector<int32_t> const& vocabMappingTable = {});

// Image utility reference functions
void transposeToPatchQwenReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const T, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const temporalPatchSize, int32_t const patchSize, int32_t const mergeSize);

void transposeToPatchInternVLReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const blockSizeH, int32_t const blockSizeW);

void fastPosEmbedInterpolateReference(std::vector<std::vector<int64_t>> const& imageGridTHWs,
    std::vector<int64_t> const& cuSeqlens, std::vector<int64_t>& fastPosEmbedIdx, std::vector<half>& fastPosEmbedWeight,
    int64_t const mergeSize, int64_t const numGridPerSide);

void initRotaryPosEmbQwenViTReference(std::vector<float>& rotaryPosEmb,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, int64_t const totalSeqLength, int64_t const vitPosEmbDim,
    int64_t const mergeSize, float const rotaryBaseFrequency, float const scale);

// GEMM weight packing and weight scale reference function for int4-WOQ kernel
void awqPackReference(int16_t const* kernel_KxN, int N_in, int K_in, int16_t* out_Ndiv4xK);

void scaledWeightsReference(
    int16_t const* kernel_KxN, half const* scales_KdivGxN, int K, int N, int group_size, std::vector<half>& out_KxN);
