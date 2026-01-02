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

#include "references.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

std::vector<half> casualAttentionRef(std::vector<half> const& q, std::vector<half> const& k, std::vector<half> const& v,
    int32_t const qlen, int32_t kvlen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    std::optional<std::vector<int32_t>> const& treeAttnMask)
{
    assert(qlen <= kvlen);
    int32_t const numQheadPerKV = numQHeads / numKVHeads;
    float qkScale = 1.0f / std::sqrt(static_cast<float>(headSize));

    auto qoIndexer = [numQHeads, headSize](int32_t tokenIdx, int32_t qHeadIdx, int32_t valIdx) {
        // Q and Out Tensor has layout of [QToken, Qhead, featureVal]
        return tokenIdx * numQHeads * headSize + qHeadIdx * headSize + valIdx;
    };
    auto kvIndexer = [kvlen, headSize](int32_t kvSequenceIdx, int32_t kvHeadIdx, int32_t valIdx) {
        // KV Tensor has layout of [KVhead, kv_sequence, featureVal]
        return kvHeadIdx * kvlen * headSize + kvSequenceIdx * headSize + valIdx;
    };

    std::vector<half> result(numQHeads * headSize * qlen);
    for (int32_t tokenIdx = 0; tokenIdx < qlen; ++tokenIdx)
    {
        for (int32_t qHeadIdx = 0; qHeadIdx < numQHeads; ++qHeadIdx)
        {
            std::vector<float> attnScores(kvlen, 0.0F);
            float maxVal = -std::numeric_limits<half>::infinity();
            int32_t kvHeadIdx = qHeadIdx / numQheadPerKV;
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                for (int32_t valIdx = 0; valIdx < headSize; ++valIdx)
                {
                    float qVal = __half2float(q[qoIndexer(tokenIdx, qHeadIdx, valIdx)]);
                    float kvVal = __half2float(k[kvIndexer(kvIdx, kvHeadIdx, valIdx)]);
                    attnScores[kvIdx] += qVal * kvVal * qkScale;
                }

                maxVal = std::max(maxVal, attnScores[kvIdx]);
            }
            // Apply Mask for casual and tree mask
            if (qlen > 1 && treeAttnMask.has_value())
            {
                int32_t const kvStartIdxForQ = kvlen - qlen;
                auto const treeMasks = treeAttnMask.value();
                for (int32_t maskQIdx = 0; maskQIdx < qlen; ++maskQIdx)
                {
                    int32_t const mask = treeMasks[tokenIdx * qlen + maskQIdx];
                    if (mask == 0)
                    {
                        // Set to -1e5 to make softmax result close to 0
                        attnScores[kvStartIdxForQ + maskQIdx] = -5e5;
                    }
                }
            }
            // Compute softmax using attnScores - maxVal
            float sumExp = 0.0F;
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                attnScores[kvIdx] = std::exp(attnScores[kvIdx] - maxVal);
                sumExp += attnScores[kvIdx];
            }
            for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
            {
                attnScores[kvIdx] /= sumExp;
            }

            // Compute BMM2 Attn_score @ V
            for (int32_t valIdx = 0; valIdx < headSize; ++valIdx)
            {
                float outVal = 0.0F;
                for (int32_t kvIdx = 0; kvIdx < kvlen; ++kvIdx)
                {
                    outVal += attnScores[kvIdx] * __half2float(v[kvIndexer(kvIdx, kvHeadIdx, valIdx)]);
                }
                result[qoIndexer(tokenIdx, qHeadIdx, valIdx)] = __float2half(outVal);
            }
        }
    }

    return result;
}

std::vector<half> ropeRef(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, int32_t const seqIdx, float const ropeScale, float const ropeTheta, bool const permute)
{
    std::vector<half> result;
    for (int32_t i = 0; i < numHeads; i++)
    {
        std::vector<half> x(input.begin() + headSize * i, input.begin() + headSize * (i + 1));
        std::vector<half> y(headSize);
        for (int32_t j = 0; j < rotaryDim / 2; j++)
        {
            int32_t leftIndex, rightIndex;
            // Determine whether to apply gpt-neox style rope to permute.
            if (permute)
            {
                leftIndex = j;
                rightIndex = rotaryDim / 2 + j;
            }
            else
            {
                leftIndex = j * 2;
                rightIndex = j * 2 + 1;
            }
            float invFreq = (seqIdx * ropeScale) / std::pow(ropeTheta, 2 * j / float(rotaryDim));
            float cos = std::cos(invFreq);
            float sin = std::sin(invFreq);
            y[leftIndex] = __half2float(x[leftIndex]) * cos - __half2float(x[rightIndex]) * sin;
            y[rightIndex] = __half2float(x[leftIndex]) * sin + __half2float(x[rightIndex]) * cos;
        }
        // Insert RoPE part
        result.insert(result.end(), y.begin(), y.begin() + rotaryDim);
        // Copy the remaining part of the input vector
        result.insert(result.end(), x.begin() + rotaryDim, x.end());
    }
    return result;
}

std::vector<half> ropeRefCosSin(std::vector<half> const& input, int32_t const numHeads, int32_t const headSize,
    int32_t const rotaryDim, std::vector<float> const& cosCache, std::vector<float> const& sinCache, bool const permute)
{
    std::vector<half> result;
    for (int32_t i = 0; i < numHeads; i++)
    {
        std::vector<half> x(input.begin() + headSize * i, input.begin() + headSize * (i + 1));
        std::vector<half> y(headSize);
        for (int32_t j = 0; j < rotaryDim / 2; j++)
        {
            int32_t leftIndex, rightIndex;
            // Determine whether to apply gpt-neox style rope to permute.
            if (permute)
            {
                leftIndex = j;
                rightIndex = rotaryDim / 2 + j;
            }
            else
            {
                leftIndex = j * 2;
                rightIndex = j * 2 + 1;
            }
            float cos = cosCache[j];
            float sin = sinCache[j];
            y[leftIndex] = __half2float(x[leftIndex]) * cos - __half2float(x[rightIndex]) * sin;
            y[rightIndex] = __half2float(x[leftIndex]) * sin + __half2float(x[rightIndex]) * cos;
        }
        // Insert RoPE part
        result.insert(result.end(), y.begin(), y.begin() + rotaryDim);
        // Copy the remaining part of the input vector
        result.insert(result.end(), x.begin() + rotaryDim, x.end());
    }
    return result;
}

std::vector<float> softmaxRef(std::vector<float> const& logits, float temperature)
{
    std::vector<float> scaledLogits(logits.size());
    float invTemp = (temperature < 1e-3f) ? 1000.0f : 1.0f / temperature;

    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaledLogits[i] = logits[i] * invTemp;
    }

    // Find max for numerical stability
    float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());

    // Compute softmax
    std::vector<float> probs(logits.size());
    float sumExp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] = std::exp(scaledLogits[i] - maxLogit);
        sumExp += probs[i];
    }

    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] /= sumExp;
    }

    return probs;
}

std::set<int32_t> getTopKAllowedTokensRef(std::vector<float> const& logits, int32_t topK)
{
    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + kLimit, logitPairs.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i < kLimit; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::set<int32_t> getTopPAllowedTokensRef(std::vector<float> const& logits, float topP, float temperature)
{
    // When temperature = 0.0f, we should always pick the highest probability token
    // This matches the behavior in SamplingParams constructor
    if (temperature < 1e-3f)
    {
        int32_t idxMax = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        // Return only the highest probability token
        return std::set<int32_t>{idxMax};
    }

    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    std::sort(logitPairs.begin(), logitPairs.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

    // Extract all logits and compute probabilities
    std::vector<float> allLogits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        allLogits[i] = logitPairs[i].first;
    }
    auto allProbs = softmaxRef(allLogits, temperature);

    // Handle edge case: topP = 0.0 means only the highest probability token
    if (topP <= 0.0f)
    {
        std::set<int32_t> allowedTokens;
        allowedTokens.insert(logitPairs[0].second);
        return allowedTokens;
    }

    // Find the cutoff point for top-p
    float cumsum = 0.0f;
    int32_t cutoff = 0;
    for (size_t i = 0; i < allProbs.size(); ++i)
    {
        cumsum += allProbs[i];
        cutoff = i + 1;
        if (cumsum >= topP)
        {
            break;
        }
    }

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i < cutoff; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::set<int32_t> getCombinedAllowedTokensRef(
    std::vector<float> const& logits, int32_t topK, float topP, float temperature)
{
    // When temperature = 0.0f, we should always pick the highest probability token
    // This matches the behavior in SamplingParams constructor
    if (temperature < 1e-3f)
    {
        int32_t idxMax = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        // Return only the highest probability token
        return std::set<int32_t>{idxMax};
    }

    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    std::sort(logitPairs.begin(), logitPairs.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

    // Apply top-k constraint first
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));

    // Extract top-k logits and compute probabilities
    std::vector<float> topKLogits(kLimit);
    for (int32_t i = 0; i < kLimit; ++i)
    {
        topKLogits[i] = logitPairs[i].first;
    }
    auto topKProbs = softmaxRef(topKLogits, temperature);

    // Apply top-p constraint to the top-k elements
    float cumsum = 0.0f;
    int32_t cutoff = kLimit - 1;

    for (int32_t i = 0; i < kLimit; ++i)
    {
        cumsum += topKProbs[i];
        if (cumsum >= topP)
        {
            cutoff = i;
            break;
        }
    }

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i <= cutoff; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}

std::vector<std::pair<float, int32_t>> getTopKElementsRef(std::vector<float> const& logits, int32_t topK)
{
    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + kLimit, logitPairs.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    logitPairs.resize(kLimit);
    return logitPairs;
}

// Returns top-K indices and raw values
std::vector<std::pair<float, int32_t>> returnAllTopKReference(std::vector<float> const& input, int32_t topK)
{
    // Get the top-k elements from the entire vocabulary
    auto topKElements = getTopKElementsRef(input, topK);

    // Return raw values and indices
    std::vector<std::pair<float, int32_t>> result;
    for (auto const& element : topKElements)
    {
        int32_t idx = element.second;
        float value = input[idx];
        result.emplace_back(value, idx);
    }

    return result;
}

void computeLongRopeReference(std::vector<float>& shortCosSinCache, std::vector<float>& longCosSinCache,
    std::vector<float> const& shortFactor, std::vector<float> const& longFactor, float rotaryBaseFrequency,
    int32_t rotaryDim, int32_t kvCacheCapacity, int32_t rotaryEmbeddingMaxPositions,
    int32_t originalMaxPositionEmbeddings)
{
    float scalingFactor = 1.0f;
    float scale = static_cast<float>(rotaryEmbeddingMaxPositions) / static_cast<float>(originalMaxPositionEmbeddings);
    if (scale > 1.0f)
    {
        scalingFactor = std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(originalMaxPositionEmbeddings)));
    }

    auto initCosSin = [&](std::vector<float> const& extFactors, std::vector<float>& cosSin, int32_t maxPositions) {
        for (int32_t pos = 0; pos < maxPositions; ++pos)
        {
            for (int32_t i = 0; i < rotaryDim / 2; ++i)
            {
                float invFreq = pos / (extFactors[i] * std::pow(rotaryBaseFrequency, 2 * i / float(rotaryDim)));
                float cos = std::cos(invFreq) * scalingFactor;
                float sin = std::sin(invFreq) * scalingFactor;
                cosSin[pos * rotaryDim + i] = cos;
                cosSin[pos * rotaryDim + i + rotaryDim / 2] = sin;
            }
        }
    };

    // LongCosSinCache for context length > originalMaxPositionEmbeddings
    // For all positions, use longFactor to compute cosSinCache
    initCosSin(longFactor, longCosSinCache, kvCacheCapacity);

    // ShortCosSinCache for context length <= originalMaxPositionEmbeddings
    // For positions <= originalMaxPositionEmbeddings, use shortFactor to compute cosSinCache
    // For positions > originalMaxPositionEmbeddings, use longFactor to compute cosSinCache. Copy from longCosSinCache.
    int32_t shortMaxPositions = std::min(originalMaxPositionEmbeddings, kvCacheCapacity);
    initCosSin(shortFactor, shortCosSinCache, shortMaxPositions);
    if (shortMaxPositions < kvCacheCapacity)
    {
        std::copy(longCosSinCache.begin() + shortMaxPositions * rotaryDim, longCosSinCache.end(),
            shortCosSinCache.begin() + shortMaxPositions * rotaryDim);
    }

    return;
}

void computeMRopeReference(std::vector<float>& mropeRotaryCosSin, std::vector<int64_t> const& mropePositionIds,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t batchSize,
    bool interleaved)
{
    // mropePositionIds: (bs, 3, maxPositionEmbeddings)
    // mropeRotaryCosSin: (bs, maxPositionEmbeddings, rotaryDim)

    std::vector<float> invFreq;
    for (int i = 0; i < rotaryDim / 2; ++i)
    {
        float value = pow(rotaryBaseFrequency, 2 * i / (float) rotaryDim);
        invFreq.emplace_back(value);
    }

    std::vector<std::vector<float>> cosOri(rotaryEmbeddingMaxPositions, std::vector<float>(rotaryDim / 2));
    std::vector<std::vector<float>> sinOri(rotaryEmbeddingMaxPositions, std::vector<float>(rotaryDim / 2));
    for (int i = 0; i < rotaryEmbeddingMaxPositions; ++i)
    {
        for (int j = 0; j < (rotaryDim / 2); ++j)
        {
            cosOri[i][j] = std::cos(i / invFreq[j]);
            sinOri[i][j] = std::sin(i / invFreq[j]);
        }
    }

    int32_t sinOffset = rotaryDim / 2;
    for (int b = 0; b < batchSize; ++b)
    {
        for (int i = 0; i < rotaryEmbeddingMaxPositions; ++i)
        {
            if (interleaved)
            {
                // Interleaved format: [THWTHWHTHW...TTTT] Qwen3-VL
                //     mrope section is [24, 20, 20]
                //     [THWTHWHTHW...TTTT] has 20 groups of THW and 4 additional T
                for (int j = 0; j < (rotaryDim / 2); ++j)
                {
                    int sec = (j >= 60) ? 0 : (j % 3);
                    int pos
                        = mropePositionIds[b * 3 * rotaryEmbeddingMaxPositions + sec * rotaryEmbeddingMaxPositions + i];
                    int cosDstIdx = b * rotaryEmbeddingMaxPositions * rotaryDim + i * rotaryDim + j;
                    mropeRotaryCosSin[cosDstIdx] = cosOri[pos][j];
                    mropeRotaryCosSin[cosDstIdx + sinOffset] = sinOri[pos][j];
                }
            }
            else
            {
                std::vector<int> mRopeSections{0, 16, 40, 64}; // cumsum of {16, 24, 24}
                for (int sec = 0; sec < 3; ++sec)
                {
                    int pos
                        = mropePositionIds[b * 3 * rotaryEmbeddingMaxPositions + sec * rotaryEmbeddingMaxPositions + i];
                    for (int j = mRopeSections[sec]; j < mRopeSections[sec + 1]; ++j)
                    {
                        int cosDstIdx = b * rotaryEmbeddingMaxPositions * rotaryDim + i * rotaryDim + j;
                        mropeRotaryCosSin[cosDstIdx] = cosOri[pos][j];
                        mropeRotaryCosSin[cosDstIdx + sinOffset] = sinOri[pos][j];
                    }
                }
            }
        }
    }
}

std::vector<half> embeddingLookupRef(std::vector<int32_t> const& inputIds, std::vector<half> const& embeddingTable,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    std::optional<std::vector<half>> const& imageEmbeds, int64_t imageTokenLen)
{
    std::vector<half> result(batchSize * seqLen * hiddenSize, __float2half(0.0f));

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
        {
            int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];
            bool const isImageToken = imageEmbeds.has_value() && tokenId > (vocabSize - 1);

            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = batchIdx * seqLen * hiddenSize + tokenIdx * hiddenSize + elementIdx;

                half embeddingValue;
                if (isImageToken)
                {
                    int32_t const visualTokenId = tokenId - vocabSize;
                    if (visualTokenId >= 0 && visualTokenId < imageTokenLen)
                    {
                        int64_t const imageEmbedIdx = visualTokenId * hiddenSize + elementIdx;
                        embeddingValue = imageEmbeds.value()[imageEmbedIdx];
                    }
                    else
                    {
                        embeddingValue = __float2half(0.0f);
                    }
                }
                else
                {
                    // For normal tokens, check bounds and use zero embedding for out-of-bounds
                    if (tokenId >= 0 && tokenId < vocabSize)
                    {
                        int64_t const embeddingIdx = tokenId * hiddenSize + elementIdx;
                        embeddingValue = embeddingTable[embeddingIdx];
                    }
                    else
                    {
                        embeddingValue = __float2half(0.0f);
                    }
                }

                result[resultIdx] = embeddingValue;
            }
        }
    }

    return result;
}

void assembleDraftTreeDescReference(std::vector<int8_t> const& draftTreeMask,
    std::vector<int32_t> const& draftTreeLength, std::vector<int32_t> const& sequenceStartIndex,
    std::vector<int32_t>& packedDraftTreeMask, std::vector<int32_t>& tensorPositionIndices, int32_t paddedDraftTreeSize)
{
    int32_t const kNUM_MASK_PER_ENTRY{32};
    size_t batchSize = draftTreeLength.size();
    int32_t const packedTreeMaskLen = (paddedDraftTreeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const actualDraftTreeSize = draftTreeLength[batchIdx];
        int32_t const sequenceStartIdx = sequenceStartIndex[batchIdx];

        for (size_t tokenIdx = 0; tokenIdx < actualDraftTreeSize; ++tokenIdx)
        {
            int32_t attendNodeNum = 0;
            int32_t const packedTreeMaskOffset
                = batchIdx * paddedDraftTreeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
            for (size_t i = 0; i <= tokenIdx; ++i)
            {
                int8_t const maskFlag = draftTreeMask[batchIdx * paddedDraftTreeSize * paddedDraftTreeSize
                    + tokenIdx * paddedDraftTreeSize + i];
                if (maskFlag)
                {
                    attendNodeNum += 1;
                    packedDraftTreeMask[packedTreeMaskOffset + i / kNUM_MASK_PER_ENTRY]
                        |= (1 << (i % kNUM_MASK_PER_ENTRY));
                }
            }
            // A token always attend to itself, subtract 1 to reflect its position in the sequence
            int32_t tensorPositionIdx = sequenceStartIdx + attendNodeNum - 1;
            tensorPositionIndices[batchIdx * paddedDraftTreeSize + tokenIdx] = tensorPositionIdx;
        }
    }
}

void prepareEagleDraftProposalMiscInputReference(std::vector<int32_t> const& draftTreeLength,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t selectTokenLength, int32_t paddedDraftTreeSize)
{
    size_t batchSize = draftTreeLength.size();

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const draftTreeSize = draftTreeLength[batchIdx];
        sequenceContextLengths[batchIdx] = sequenceStartIndex[batchIdx] + paddedDraftTreeSize;

        for (size_t i = 0; i < selectTokenLength; ++i)
        {
            selectTokenIndices[batchIdx * selectTokenLength + i] = draftTreeSize - selectTokenLength + i;
        }
    }
}

void prepareEaglePrefillInputReference(
    std::vector<int32_t>& sequenceContextLengths, std::vector<int64_t>& selectTokenIndices, int32_t sequenceLength)
{
    size_t const batchSize = sequenceContextLengths.size();
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        sequenceContextLengths[batchIdx] = sequenceLength;
        selectTokenIndices[batchIdx] = sequenceLength - 1;
    }
}

void prepareEagleAcceptDecodeTokenInputReference(std::vector<int32_t> const& sequenceStartIndices,
    std::vector<int32_t>& packedTreeMask, std::vector<int32_t>& tensorPositionIndices,
    std::vector<int64_t>& selectTokenIndices, std::vector<int32_t>& sequenceContextLengths, int32_t acceptedTokenNum)
{
    size_t const batchSize = sequenceStartIndices.size();
    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        // Generate packed tree mask and tensor position indices for each accepted token
        for (int32_t tokenIdx = 0; tokenIdx < acceptedTokenNum; ++tokenIdx)
        {
            int32_t packedTreeMaskValue = 0;
            // Create casual attention mask: each token attends to all previous tokens including itself
            for (int32_t i = 0; i <= tokenIdx; ++i)
            {
                packedTreeMaskValue |= (1 << i);
            }

            int32_t const packedTreeMaskOffset = batchIdx * acceptedTokenNum + tokenIdx;
            packedTreeMask[packedTreeMaskOffset] = packedTreeMaskValue;
            tensorPositionIndices[packedTreeMaskOffset] = sequenceStartIndices[batchIdx] + tokenIdx;
        }

        // Set select token index to the last accepted token
        selectTokenIndices[batchIdx] = acceptedTokenNum - 1;
        sequenceContextLengths[batchIdx] = sequenceStartIndices[batchIdx] + acceptedTokenNum;
    }
}

void prepareEagleBaseTreeDecodingInputReference(std::vector<int8_t> const& baseTreeDecodingMask,
    std::vector<int32_t> const& sequenceStartIndex, std::vector<int32_t>& packedBaseTreeDecodingMask,
    std::vector<int32_t>& tensorPositionIndices, std::vector<int32_t>& sequenceContextLengths,
    std::vector<int64_t>& selectTokenIndices, int32_t treeSize)
{
    // baseTreeDecodingMask: (bs, tree-size, tree-size)
    // sequenceStartIndex: (bs)
    // packedBaseTreeDecodingMask: (bs, tree-size, divup(tree-size, 32))
    // tensorPositionIndices: (bs, tree-size)
    // sequenceContextLengths: (bs)
    // selectTokenIndices: (bs, tree-size)

    int32_t const kNUM_MASK_PER_ENTRY{32};
    size_t batchSize = sequenceStartIndex.size();
    int32_t const packedTreeMaskLen = (treeSize + kNUM_MASK_PER_ENTRY - 1) / kNUM_MASK_PER_ENTRY;

    for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const sequenceStartIdx = sequenceStartIndex[batchIdx];
        sequenceContextLengths[batchIdx] = sequenceStartIdx + treeSize;

        for (size_t tokenIdx = 0; tokenIdx < treeSize; ++tokenIdx)
        {
            int32_t attendNodeNum = 0;
            int32_t const packedTreeMaskOffset = batchIdx * treeSize * packedTreeMaskLen + tokenIdx * packedTreeMaskLen;
            for (size_t i = 0; i <= tokenIdx; ++i)
            {
                int8_t const maskFlag = baseTreeDecodingMask[batchIdx * treeSize * treeSize + tokenIdx * treeSize + i];
                if (maskFlag)
                {
                    attendNodeNum += 1;
                    packedBaseTreeDecodingMask[packedTreeMaskOffset + i / kNUM_MASK_PER_ENTRY]
                        |= (1 << (i % kNUM_MASK_PER_ENTRY));
                }
            }
            // A token always attend to itself, subtract 1 to reflect its position in the sequence
            int32_t tensorPositionIdx = sequenceStartIdx + attendNodeNum - 1;
            tensorPositionIndices[batchIdx * treeSize + tokenIdx] = tensorPositionIdx;
            selectTokenIndices[batchIdx * treeSize + tokenIdx] = tokenIdx;
        }
    }
}

void eagleBaseCommitKVCacheAndAssembleHiddenStateReference(std::vector<int32_t> const& acceptedIndices,
    std::vector<int32_t> const& acceptLengths, std::vector<half> const& kvCacheBuffer,
    std::vector<int32_t> const& kvCacheLengths, std::vector<half> const& hiddenState,
    std::vector<half>& kvCacheBufferOut, std::vector<half>& hiddenStateOut, int32_t const numLayers,
    int32_t const maxBatchSize, int32_t const numHeads, int32_t const maxSeqLen, int32_t const headDim,
    int32_t const maxDepth, int32_t const draftTreeSize, int32_t const baseHiddenDim)
{
    size_t const activeBatchSize = acceptLengths.size();

    for (int b = 0; b < activeBatchSize; b++)
    {
        int32_t const kvCacheLength = kvCacheLengths[b];
        int32_t const acceptLength = acceptLengths[b];

        // Start from 1 since the root position will always be accepted.
        for (int i = 1; i < acceptLength; i++)
        {
            int32_t const acceptedIdx = acceptedIndices[b * maxDepth + i];
            assert(acceptedIdx >= 0 && acceptedIdx + kvCacheLength < maxSeqLen && acceptedIdx < draftTreeSize
                && "Accepted index out of bounds");

            // kvCacheBuffer: [num-layers, max-batch-size, 2, num-heads, max-seq-len, hidden-size-per-head].
            for (int l = 0; l < numLayers; l++)
            {
                for (int k = 0; k < 2; k++)
                {
                    for (int h = 0; h < numHeads; h++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            int32_t const srcOffset = l * maxBatchSize * 2 * numHeads * maxSeqLen * headDim
                                + b * 2 * numHeads * maxSeqLen * headDim + k * numHeads * maxSeqLen * headDim
                                + h * maxSeqLen * headDim + (kvCacheLength + acceptedIdx) * headDim + d;

                            int32_t const dstOffset = l * maxBatchSize * 2 * numHeads * maxSeqLen * headDim
                                + b * 2 * numHeads * maxSeqLen * headDim + k * numHeads * maxSeqLen * headDim
                                + h * maxSeqLen * headDim + (kvCacheLength + i) * headDim + d;
                            kvCacheBufferOut[dstOffset] = kvCacheBuffer[srcOffset];
                        }
                    }
                }
            }

            // hiddenState: [batch, num-tokens, hidden-dim].
            for (int d = 0; d < baseHiddenDim; d++)
            {
                int32_t const srcOffset = b * draftTreeSize * baseHiddenDim + acceptedIdx * baseHiddenDim + d;
                int32_t const dstOffset = b * draftTreeSize * baseHiddenDim + i * baseHiddenDim + d;
                hiddenStateOut[dstOffset] = hiddenState[srcOffset];
            }
        }
    }
}

// Helper function to compute token depth - count total connections (sum of 1s)
int32_t computeTokenDepthRef(int32_t tokenIdx, std::vector<int8_t> const& attentionMask, int32_t numTokens)
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

EagleAcceptResult eagleAcceptRef(std::vector<float> const& logits, std::vector<int32_t> const& tokenIds,
    std::vector<int8_t> const& attentionMask, int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth,
    std::vector<int32_t> const& vocabMappingTable)
{
    EagleAcceptResult result;
    result.acceptedTokenIds.resize(batchSize * maxDepth, -1);
    result.acceptedLogitsIndices.resize(batchSize * maxDepth, -1);
    result.acceptLengths.resize(batchSize, 0);

    int32_t maxAcceptLength = 0;

    // Process each batch
    for (int32_t b = 0; b < batchSize; ++b)
    {
        // Precompute token depths for this batch
        std::vector<int32_t> tokenDepths(numTokens);
        int32_t const batchMaskOffset = b * numTokens * numTokens;
        int32_t const batchTokenOffset = b * numTokens;
        for (int32_t i = 0; i < numTokens; ++i)
        {
            int32_t depth = 0;
            for (int32_t j = 0; j < numTokens; ++j)
            {
                if (attentionMask[batchMaskOffset + i * numTokens + j] == 1)
                {
                    depth++;
                }
            }
            tokenDepths[i] = depth;
        }

        int32_t currentDepth = 0;
        int32_t currentTokenIdx = 0;                    // Start with token[0], use logits[0] to predict next
        int32_t expectedNextDepth = tokenDepths[0] + 1; // Next depth should be current token's depth + 1

        // Start with accept length 0, will be set to at least 1 in the loop
        result.acceptLengths[b] = 0;

        // Process tokens - always accept at least one
        for (int32_t step = 0; step < maxDepth && currentTokenIdx < numTokens; ++step)
        {
            // Step 1: Find top-1 token from logits[b][currentTokenIdx]
            int32_t const logitsOffset = b * numTokens * vocabSize + currentTokenIdx * vocabSize;

            // Find argmax with consistent tie-breaking (prefer lower indices)
            float maxLogit = -std::numeric_limits<float>::infinity();
            int32_t selectedTokenId = 0; // Default to token 0 for consistent behavior

            for (int32_t v = 0; v < vocabSize; ++v)
            {
                if (logits[logitsOffset + v] > maxLogit)
                {
                    maxLogit = logits[logitsOffset + v];
                    selectedTokenId = v;
                }
            }

            // Apply vocab mapping if provided (for reduced vocabulary)
            if (!vocabMappingTable.empty() && selectedTokenId < static_cast<int32_t>(vocabMappingTable.size()))
            {
                selectedTokenId = vocabMappingTable[selectedTokenId];
            }

            // Step 2: Always accept the selected token
            result.acceptedTokenIds[b * maxDepth + currentDepth] = selectedTokenId;
            result.acceptedLogitsIndices[b * maxDepth + currentDepth] = currentTokenIdx;
            result.acceptLengths[b] = currentDepth + 1;
            currentDepth++;

            // Step 3: Check if the selected token exists in the tree to continue
            int32_t nextTokenIdx = -1;

            for (int32_t checkIdx = 1; checkIdx < numTokens; ++checkIdx)
            {
                if (tokenIds[batchTokenOffset + checkIdx] == selectedTokenId
                    && tokenDepths[checkIdx] == expectedNextDepth)
                {
                    // Check attention mask: does checkIdx attend to currentTokenIdx?
                    int32_t maskOffset = batchMaskOffset + checkIdx * numTokens + currentTokenIdx;
                    if (attentionMask[maskOffset] == 1)
                    {
                        // Found a valid next token in tree
                        nextTokenIdx = checkIdx;
                        break; // Take the first match
                    }
                }
            }

            // Step 4: Update for next iteration
            if (nextTokenIdx != -1)
            {
                // Found valid next token in tree, continue from there
                currentTokenIdx = nextTokenIdx;
                expectedNextDepth++;
            }
            else
            {
                // No valid next token found in tree, stop
                break;
            }
        }

        // Ensure at least 1 token is always accepted
        if (result.acceptLengths[b] == 0)
        {
            // This should never happen, but as a safety net, accept the first predicted token
            int32_t const logitsOffset = b * numTokens * vocabSize + 0 * vocabSize;
            float maxLogit = -std::numeric_limits<float>::infinity();
            int32_t selectedTokenId = 0; // Default to token 0 for consistent behavior
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                if (logits[logitsOffset + v] > maxLogit)
                {
                    maxLogit = logits[logitsOffset + v];
                    selectedTokenId = v;
                }
            }
            result.acceptedTokenIds[b * maxDepth + 0] = selectedTokenId;
            result.acceptedLogitsIndices[b * maxDepth + 0] = 0;
            result.acceptLengths[b] = 1;
        }

        // Update max accept length
        maxAcceptLength = std::max(maxAcceptLength, result.acceptLengths[b]);
    }

    // Ensure maxAcceptLength is at least 1
    maxAcceptLength = std::max(maxAcceptLength, 1);
    result.maxAcceptLength = maxAcceptLength;

    // Reshape the result vectors to [batchSize, maxAcceptLength]
    std::vector<int32_t> reshapedTokenIds(batchSize * maxAcceptLength, -1);
    std::vector<int32_t> reshapedLogitsIndices(batchSize * maxAcceptLength, -1);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < result.acceptLengths[b] && i < maxAcceptLength; ++i)
        {
            reshapedTokenIds[b * maxAcceptLength + i] = result.acceptedTokenIds[b * maxDepth + i];
            reshapedLogitsIndices[b * maxAcceptLength + i] = result.acceptedLogitsIndices[b * maxDepth + i];
        }
    }

    result.acceptedTokenIds = std::move(reshapedTokenIds);
    result.acceptedLogitsIndices = std::move(reshapedLogitsIndices);

    return result;
}

void transposeToPatchQwenReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const T, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const temporalPatchSize, int32_t const patchSize, int32_t const mergeSize)
{
    assert(originalImage.size() == T * height * width * channels);
    assert(patch.size() == T * height * width * channels);

    int const gridT = T / temporalPatchSize;
    int const gridH = height / (mergeSize * patchSize);
    int const gridW = width / (mergeSize * patchSize);

    for (int gt = 0; gt < gridT; ++gt)
    {
        for (int gh = 0; gh < gridH; ++gh)
        {
            for (int gw = 0; gw < gridW; ++gw)
            {
                for (int mergeH = 0; mergeH < mergeSize; ++mergeH)
                {
                    for (int mergeW = 0; mergeW < mergeSize; ++mergeW)
                    {
                        for (int c = 0; c < channels; ++c)
                        {
                            for (int t = 0; t < temporalPatchSize; ++t)
                            {
                                for (int patchH = 0; patchH < patchSize; ++patchH)
                                {
                                    for (int patchW = 0; patchW < patchSize; ++patchW)
                                    {
                                        // src dimensions: (T, H, W, C) => (gridT, temporalPatchSize, gridH, mergeSize,
                                        // patchSize, gridW, mergeSize, patchSize, C)
                                        int originalT = gt * temporalPatchSize + t;
                                        int originalH = gh * mergeSize * patchSize + mergeH * patchSize + patchH;
                                        int originalW = gw * mergeSize * patchSize + mergeW * patchSize + patchW;
                                        half value = originalImage[originalT * height * width * channels
                                            + originalH * width * channels + originalW * channels + c];

                                        // dst dimensions: (gridT, gridH, gridW, mergeSize, mergeSize) x (channels,
                                        // temporalPatchSize, patchSize, patchSize)
                                        int dstHW = gt * gridH * gridW * mergeSize * mergeSize
                                            + gh * gridW * mergeSize * mergeSize + gw * mergeSize * mergeSize
                                            + mergeH * mergeSize + mergeW;
                                        int dstDim = c * temporalPatchSize * patchSize * patchSize
                                            + t * patchSize * patchSize + patchH * patchSize + patchW;
                                        patch[dstHW * channels * temporalPatchSize * patchSize * patchSize + dstDim]
                                            = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void transposeToPatchInternVLReference(std::vector<half> const& originalImage, std::vector<half>& patch,
    int32_t const inputOffset, int32_t const height, int32_t const width, int32_t const channels,
    int32_t const blockSizeH, int32_t const blockSizeW)
{
    assert(originalImage.size() == height * width * channels);
    assert(patch.size() == height * width * channels);

    for (int gridH = 0; gridH < height / blockSizeH; ++gridH)
    {
        for (int gridW = 0; gridW < width / blockSizeW; ++gridW)
        {
            for (int blockH = 0; blockH < blockSizeH; ++blockH)
            {
                for (int blockW = 0; blockW < blockSizeW; ++blockW)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        // src dimensions: (H, W, C) => (gridH, blockSizeH, gridW, blockSizeW, C)
                        int originalH = gridH * blockSizeH + blockH;
                        int originalW = gridW * blockSizeW + blockW;
                        half value = originalImage[originalH * width * channels + originalW * channels + c];

                        // dst dimensions: (gridH*gridW, C, blockSizeH, blockSizeW)
                        int dstNumBlocks = gridH * (width / blockSizeW) + gridW;
                        patch[dstNumBlocks * channels * blockSizeH * blockSizeW + c * blockSizeH * blockSizeW
                            + blockH * blockSizeW + blockW]
                            = value;
                    }
                }
            }
        }
    }
}

// Helper functions for GEMM and GEMV unit tests
static inline size_t idx4(int i, int j, int k, int t, int A, int B, int C, int Tlast)
{
    // shape: [A, B, C, Tlast]
    return ((((size_t) i * B + j) * C + k) * Tlast) + t;
}
static inline size_t idx3(int i, int j, int k, int A, int B, int C)
{
    // shape: [A, B, C]
    return ((size_t) i * B + j) * C + k;
}

static inline size_t idx2(int i0, int i1, int dim1)
{
    return static_cast<size_t>(i0) * dim1 + i1;
}

// C++ implementation for https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/qmodule.py#L26
void awqPackReference(int16_t const* kernel_KxN, // [K_in, N_in], row-major
    int N_in, int K_in,
    int16_t* out_Ndiv4xK // [N_in/4, K_in], row-major
)
{
    // Python constants
    int const interleave = 4;
    int const kstride = 64;

    // After Python's: unpacked_kernel = (kernel + 8).T
    // So our working dims become:
    // N = K_in, K = N_in
    int const N = K_in;
    int const K = N_in;

    // A = (kernel + 8).T  -> uint8 [K, N]
    std::vector<int16_t> A((size_t) K * N);
    for (int k_in = 0; k_in < K_in; ++k_in)
    {
        for (int n_in = 0; n_in < N_in; ++n_in)
        {
            int16_t v = kernel_KxN[idx2(k_in, n_in, N_in)] + 8; // shift to nibble
            // transpose: (N_in, K_in) -> (K_in, N_in) == (N, K)
            A[idx2(n_in, k_in, N)] = v;
        }
    }
    // Step 1: reshape(N, K//32, 32) -> reshape(...,4,4,2) -> transpose(..., 1,0,2) inside that block
    // Mapping within each 32-lane chunk:
    //   index_in  t  = ((i0*4 + i1) * 2 + i2)
    //   index_out t' = ((i1*4 + i0) * 2 + i2)
    std::vector<int16_t> B((size_t) K * N);
    int const K32 = K / 32;
    for (int n = 0; n < N; ++n)
    {
        size_t const in_row = (size_t) n * K;
        size_t const out_row = (size_t) n * K;

        for (int b = 0; b < K32; ++b)
        {
            size_t const in_blk = in_row + (size_t) b * 32;
            size_t const out_blk = out_row + (size_t) b * 32;

            // Inside each 32-lane block, map offset o -> o' as:
            //   decompose o in (4,4,2): a=o//8, b4=(o//2)%4, c=o%2
            //   o' = ((b4*4)+a)*2 + c
            for (int o = 0; o < 32; ++o)
            {
                int const a = o >> 3;          // 0..3
                int const b4 = (o >> 1) & 0x3; // 0..3
                int const c = o & 1;           // 0..1
                int const o2 = (((b4 << 2) | a) << 1) | c;

                B[out_blk + o2] = A[in_blk + o];
            }
        }
    }
    // Step 2: reorder each 8 within each block of 32:
    // Python: reshape(...,4,8) -> reshape(...,4,4,2).transpose(0,1,2,4,3)
    // This is exactly: within each group of 8, [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]
    static int const REORDER8[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<int16_t> C((size_t) K * N);
    for (int n = 0; n < N; ++n)
    {
        for (int q = 0; q < K32; ++q)
        {
            int base = q * 32;
            for (int g = 0; g < 4; ++g)
            { // 4 groups of 8 within 32
                int gbase = base + g * 8;
                for (int j = 0; j < 8; ++j)
                {
                    int in_idx = gbase + j;
                    int out_idx = gbase + REORDER8[j];
                    C[idx2(n, out_idx, K)] = B[idx2(n, in_idx, K)];
                }
            }
        }
    }
    // Step 3: interleave every 4 rows (first dim), pack along that dimension
    // Python: x.reshape(N//4, 4, K//64, 64)
    //         x = x.transpose(0,2,1,3) -> (N//4, K//64, 4, 64)
    //         pack last-4 into a uint16: [lane0|lane1<<4|lane2<<8|lane3<<12]
    int const Ng = K / 4;  // N4 = N_in / 4
    int const Kg = N / 64; // K64 = K_in / 64

    // Step 3.1: transpose (0,2,1,3) → B[Ng, Kg, interleave, kstride] ---
    std::vector<int16_t> D(static_cast<size_t>(Ng) * Kg * interleave * kstride);
    for (int ng = 0; ng < Ng; ++ng)
    {
        for (int ks = 0; ks < Kg; ++ks)
        {
            for (int j = 0; j < interleave; ++j)
            {
                for (int i = 0; i < kstride; ++i)
                {
                    D[idx4(ng, ks, j, i, 0, Kg, interleave, kstride)]
                        = C[idx4(ng, j, ks, i, 0, interleave, Kg, kstride)];
                }
            }
        }
    }
    // Step 3.2: view as [Ng, Kg, kstride, interleave] and pack last-4 nibbles ---
    // No data move for the view; we just compute using the view's (ip, jp) mapping.
    //
    // For fixed (ng, ks, ip), the 4 lanes correspond to jp = 0..3, where
    //   L = ip*interleave + jp         (linear within the (kstride, interleave) view)
    //   j = L / kstride,  i = L % kstride    (indices in B's (..., interleave, kstride))
    //
    // Then final flatten (ks, ip) → out column k_out = ks*kstride + ip.
    for (int i = 0; i < Ng; ++i)
    {
        for (int j = 0; j < Kg; ++j)
        {
            for (int k = 0; k < kstride; ++k)
            {
                size_t const base4 = idx4(i, j, k, 0, Ng, Kg, kstride, /*Tlast=*/4);

                // Do shifts in uint16_t (well-defined), then cast back to int16_t.
                uint16_t const a0 = static_cast<uint16_t>(D[base4 + 0]);
                uint16_t const a1 = static_cast<uint16_t>(D[base4 + 1]);
                uint16_t const a2 = static_cast<uint16_t>(D[base4 + 2]);
                uint16_t const a3 = static_cast<uint16_t>(D[base4 + 3]);

                uint16_t const packed = static_cast<uint16_t>((a0) | (static_cast<uint16_t>(a1) << 4)
                    | (static_cast<uint16_t>(a2) << 8) | (static_cast<uint16_t>(a3) << 12));

                out_Ndiv4xK[idx3(i, j, k, Ng, Kg, kstride)] = static_cast<int16_t>(packed);
            }
        }
    }
}

void scaledWeightsReference(
    int16_t const* kernel_KxN, half const* scales_KdivGxN, int K, int N, int group_size, std::vector<half>& out_KxN)
{
    out_KxN.resize(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
    {
        int srow = k / group_size; // (K//G, N)
        for (int n = 0; n < N; ++n)
        {
            float w = static_cast<float>(kernel_KxN[idx2(k, n, N)]);
            float sf = __half2float(scales_KdivGxN[idx2(srow, n, N)]);
            out_KxN[idx2(k, n, N)] = __float2half(w * sf);
        }
    }
}

void fastPosEmbedInterpolateReference(std::vector<std::vector<int64_t>> const& imageGridTHWs,
    std::vector<int64_t> const& cuSeqlens, std::vector<int64_t>& fastPosEmbedIdx, std::vector<half>& fastPosEmbedWeight,
    int64_t const mergeSize, int64_t const numGridPerSide)
{
    int64_t totalSeqLength = cuSeqlens.back();

    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        int64_t startIdx = cuSeqlens[i];
        auto grid = imageGridTHWs[i];
        int64_t T = grid[0], H = grid[1], W = grid[2];
        int64_t llmGridH = H / mergeSize;
        int64_t llmGridW = W / mergeSize;
        float lineSpaceH = static_cast<float>(numGridPerSide - 1) / (H - 1);
        float lineSpaceW = static_cast<float>(numGridPerSide - 1) / (W - 1);

        for (int64_t h = 0; h < llmGridH; ++h)
        {
            for (int64_t w = 0; w < llmGridW; ++w)
            {
                for (int64_t m = 0; m < mergeSize; ++m)
                {
                    for (int64_t n = 0; n < mergeSize; ++n)
                    {
                        float hIdx = lineSpaceH * (h * mergeSize + m);
                        float wIdx = lineSpaceW * (w * mergeSize + n);

                        int64_t hIdxFloor = static_cast<int64_t>(hIdx);
                        int64_t wIdxFloor = static_cast<int64_t>(wIdx);
                        int64_t hIdxCeil = std::min(hIdxFloor + 1, (numGridPerSide - 1));
                        int64_t wIdxCeil = std::min(wIdxFloor + 1, (numGridPerSide - 1));

                        float dh = hIdx - hIdxFloor;
                        float dw = wIdx - wIdxFloor;

                        int64_t baseH = hIdxFloor * numGridPerSide;
                        int64_t baseHCeil = hIdxCeil * numGridPerSide;

                        int64_t targetIdx = startIdx + h * llmGridW * mergeSize * mergeSize + w * mergeSize * mergeSize
                            + m * mergeSize + n;

                        // Compute indices and weights in standard order
                        fastPosEmbedIdx[0 * totalSeqLength + targetIdx] = baseH + wIdxFloor;
                        fastPosEmbedIdx[1 * totalSeqLength + targetIdx] = baseH + wIdxCeil;
                        fastPosEmbedIdx[2 * totalSeqLength + targetIdx] = baseHCeil + wIdxFloor;
                        fastPosEmbedIdx[3 * totalSeqLength + targetIdx] = baseHCeil + wIdxCeil;

                        fastPosEmbedWeight[0 * totalSeqLength + targetIdx] = __float2half((1 - dh) * (1 - dw));
                        fastPosEmbedWeight[1 * totalSeqLength + targetIdx] = __float2half((1 - dh) * dw);
                        fastPosEmbedWeight[2 * totalSeqLength + targetIdx] = __float2half(dh * (1 - dw));
                        fastPosEmbedWeight[3 * totalSeqLength + targetIdx] = __float2half(dh * dw);
                    }
                }
            }
        }
    }
}

void initRotaryPosEmbQwenViTReference(std::vector<float>& rotaryPosEmb,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, int64_t const totalSeqLength, int64_t const vitPosEmbDim,
    int64_t const mergeSize, float const rotaryBaseFrequency, float const scale)
{
    // Get position ids
    std::vector<int64_t> posIds(totalSeqLength * 2);
    int64_t posIdsOffset = 0;

    for (auto const& grid : imageGridTHWs)
    {
        int64_t T = grid[0];
        int64_t H = grid[1];
        int64_t W = grid[2];

        for (int64_t i = 0; i < H; ++i)
        {
            for (int64_t j = 0; j < W; ++j)
            {
                // (H, W) => (H / mergeSize, mergeSize, W / mergeSize, mergeSize)
                // => (H / mergeSize, W / mergeSize, mergeSize, mergeSize)
                int64_t dstHW = (i / mergeSize) * W * mergeSize + (j / mergeSize) * mergeSize * mergeSize
                    + (i % mergeSize) * mergeSize + (j % mergeSize);

                // duplicate for T
                for (int64_t t = 0; t < T; ++t)
                {
                    int64_t baseIdx = t * H * W * 2 + dstHW * 2;
                    posIds[posIdsOffset + baseIdx] = i;
                    posIds[posIdsOffset + baseIdx + 1] = j;
                }
            }
        }

        posIdsOffset += T * H * W * 2;
    }

    int64_t maxGridSize = posIds.empty() ? 0 : *std::max_element(posIds.begin(), posIds.end());
    std::vector<std::vector<float>> rotaryPosEmbFull(maxGridSize + 1, std::vector<float>(vitPosEmbDim / 2));
    for (int64_t i = 0; i <= maxGridSize; ++i)
    {
        for (int j = 0; j < (vitPosEmbDim / 2); ++j)
        {
            float value = i * scale / pow(rotaryBaseFrequency, (static_cast<float>(j * 2) / vitPosEmbDim));
            rotaryPosEmbFull[i][j] = value;
        }
    }

    rotaryPosEmb.resize(totalSeqLength * vitPosEmbDim);
    for (size_t i = 0; i < posIds.size(); ++i)
    {
        auto const& emb = rotaryPosEmbFull[posIds[i]];
        std::copy(emb.begin(), emb.end(), rotaryPosEmb.begin() + i * (vitPosEmbDim / 2));
    }
}
