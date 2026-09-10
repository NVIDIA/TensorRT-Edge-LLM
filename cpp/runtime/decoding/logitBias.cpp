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

#include "runtime/decoding/logitBias.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

void applySamplerLogitBiasRepeatedRows(
    LogitBias const& logitBias, Tensor& logits, int32_t rowsPerSlot, cudaStream_t stream)
{
    ::trt_edgellm::applyLogitBiasRepeatedRows(
        logits, logitBias.tokenIds, logitBias.values, logitBias.offsets, rowsPerSlot, stream);
}

void uploadLogitBias(LogitBias& logitBias, DecodingInferenceContext& context, cudaStream_t stream)
{
    if (!context.hasLogitBias || !context.logitBiasGpuDirty)
    {
        return;
    }

    check::check(
        context.activeBatchSize >= 0 && static_cast<size_t>(context.activeBatchSize) <= context.logitBiasPerSlot.size(),
        "Logit bias slot state does not match the active batch size");

    logitBias.hostOffsets.assign(static_cast<size_t>(context.activeBatchSize) + 1U, 0);
    size_t totalEntries = 0;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        logitBias.hostOffsets[static_cast<size_t>(i)] = static_cast<int32_t>(totalEntries);
        totalEntries += context.logitBiasPerSlot[static_cast<size_t>(i)].size();
    }
    logitBias.hostOffsets[static_cast<size_t>(context.activeBatchSize)] = static_cast<int32_t>(totalEntries);

    logitBias.hostTokenIds.clear();
    logitBias.hostValues.clear();
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        for (auto const& [tokenId, bias] : context.logitBiasPerSlot[static_cast<size_t>(i)])
        {
            logitBias.hostTokenIds.push_back(tokenId);
            logitBias.hostValues.push_back(bias);
        }
    }

    check::check(logitBias.tokenIds.reshape({static_cast<int64_t>(totalEntries)}), "Tensor reshape failed");
    check::check(logitBias.values.reshape({static_cast<int64_t>(totalEntries)}), "Tensor reshape failed");
    check::check(
        logitBias.offsets.reshape({static_cast<int64_t>(logitBias.hostOffsets.size())}), "Tensor reshape failed");

    if (totalEntries > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(logitBias.tokenIds.rawPointer(), logitBias.hostTokenIds.data(),
            totalEntries * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(logitBias.values.rawPointer(), logitBias.hostValues.data(),
            totalEntries * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(logitBias.offsets.rawPointer(), logitBias.hostOffsets.data(),
        logitBias.hostOffsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    context.logitBiasGpuDirty = false;
}

} // namespace

void allocateLogitBias(LogitBias& logitBias, int32_t maxBatchSize)
{
    check::check(maxBatchSize > 0, "Maximum batch size for logit bias must be positive");

    size_t const maxEntriesHost = static_cast<size_t>(maxBatchSize) * limits::security::kMaxLogitBiasTokens;
    int64_t const maxEntries = static_cast<int64_t>(maxEntriesHost);
    logitBias.tokenIds = Tensor({maxEntries}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "LogitBias::tokenIds");
    logitBias.values = Tensor({maxEntries}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "LogitBias::values");
    logitBias.offsets = Tensor(
        {static_cast<int64_t>(maxBatchSize) + 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "LogitBias::offsets");
    logitBias.hostOffsets.reserve(static_cast<size_t>(maxBatchSize) + 1U);
    logitBias.hostTokenIds.reserve(maxEntriesHost);
    logitBias.hostValues.reserve(maxEntriesHost);
}

void setLogitBiasVocabMap(LogitBias& logitBias, Tensor const& reducedToFullVocabMap, int32_t fullVocabSize,
    int32_t reducedVocabSize, cudaStream_t stream)
{
    check::check(fullVocabSize > 0, "Full vocabulary size must be positive");
    check::check(reducedVocabSize > 0, "Reduced vocabulary size must be positive");
    check::check(reducedToFullVocabMap.getDeviceType() == DeviceType::kGPU, "Vocabulary map must be on GPU");
    check::check(
        reducedToFullVocabMap.getDataType() == nvinfer1::DataType::kINT32, "Vocabulary map must use INT32 values");

    auto const mapShape = reducedToFullVocabMap.getShape();
    check::check(mapShape.getNumDims() == 1, "Vocabulary map must be one-dimensional");
    check::check(mapShape[0] == reducedVocabSize, "Vocabulary map length must match the reduced vocabulary size");

    std::vector<int32_t> hostVocabMap(static_cast<size_t>(reducedVocabSize));
    CUDA_CHECK(cudaMemcpyAsync(hostVocabMap.data(), reducedToFullVocabMap.rawPointer(),
        hostVocabMap.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    logitBias.fullToReducedVocabMap.assign(static_cast<size_t>(fullVocabSize), -1);
    for (int32_t reducedTokenId = 0; reducedTokenId < reducedVocabSize; ++reducedTokenId)
    {
        int32_t const fullTokenId = hostVocabMap[static_cast<size_t>(reducedTokenId)];
        check::check(
            fullTokenId >= 0 && fullTokenId < fullVocabSize, "Vocabulary map token ID is outside the full vocabulary");
        logitBias.fullToReducedVocabMap[static_cast<size_t>(fullTokenId)] = reducedTokenId;
    }
}

bool hasLogitBias(LLMGenerationRequest const& request) noexcept
{
    return std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& slot) { return !slot.logitBias.empty(); });
}

void prepareLogitBias(
    LogitBias const& logitBias, LLMGenerationRequest const& request, DecodingInferenceContext& context)
{
    context.logitBiasPerSlot.clear();
    context.logitBiasPerSlot.resize(static_cast<size_t>(context.activeBatchSize));
    context.hasLogitBias = false;
    context.logitBiasGpuDirty = false;

    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto& destination = context.logitBiasPerSlot[static_cast<size_t>(i)];
        for (auto const& [fullTokenId, bias] : request.requests[static_cast<size_t>(i)].logitBias)
        {
            int32_t outputTokenId = fullTokenId;
            if (!logitBias.fullToReducedVocabMap.empty())
            {
                check::check(
                    fullTokenId >= 0 && static_cast<size_t>(fullTokenId) < logitBias.fullToReducedVocabMap.size(),
                    "Logit bias token ID is outside the full vocabulary map");
                int32_t const reducedTokenId = logitBias.fullToReducedVocabMap[static_cast<size_t>(fullTokenId)];
                if (reducedTokenId < 0)
                {
                    if (bias > 0.0F)
                    {
                        LOG_WARNING(
                            "Ignoring positive logit_bias %.6f for token ID %d in request %d because the token is "
                            "not present in the reduced vocabulary.",
                            bias, fullTokenId, i);
                    }
                    continue;
                }
                outputTokenId = reducedTokenId;
            }
            destination[outputTokenId] = bias;
            context.hasLogitBias = true;
        }
    }

    context.logitBiasGpuDirty = context.hasLogitBias;
}

void applyLogitBias(LogitBias& logitBias, Tensor& logits, DecodingInferenceContext& context, cudaStream_t stream)
{
    applyLogitBiasRepeatedRows(logitBias, logits, context, 1, stream);
}

void applyLogitBiasRepeatedRows(
    LogitBias& logitBias, Tensor& logits, DecodingInferenceContext& context, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (!context.hasLogitBias)
    {
        return;
    }
    check::check(rowsPerSlot > 0, "Logit bias rowsPerSlot must be positive");
    uploadLogitBias(logitBias, context, stream);

    auto const originalShape = logits.getShape();
    if (originalShape.getNumDims() == 3)
    {
        check::check(originalShape[0] == context.activeBatchSize, "Spec logit bias batch size mismatch");
        check::check(originalShape[1] == rowsPerSlot, "Spec logit bias rowsPerSlot shape mismatch");
        check::check(logits.reshape({originalShape[0] * originalShape[1], originalShape[2]}), "Tensor reshape failed");
        applySamplerLogitBiasRepeatedRows(logitBias, logits, rowsPerSlot, stream);
        check::check(logits.reshape(originalShape), "Tensor reshape failed");
        return;
    }

    applySamplerLogitBiasRepeatedRows(logitBias, logits, rowsPerSlot, stream);
}

} // namespace rt
} // namespace trt_edgellm
