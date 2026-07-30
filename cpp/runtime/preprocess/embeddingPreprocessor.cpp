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

#include "runtime/preprocess/embeddingPreprocessor.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstring>

namespace trt_edgellm
{
namespace rt
{

EmbeddingPreprocessor::EmbeddingPreprocessor(EmbeddingData const& embedding, LLMEngineConfig const& config)
    : mEmbedding(embedding)
    , mConfig(config)
{
}

void EmbeddingPreprocessor::embed(Tensor const& tokenIds, OptionalInputTensor visionEmbeds,
    OptionalInputTensor audioEmbeds, PipelineIO& io, cudaStream_t stream,
    std::vector<int32_t> const& visionIndexOffsets, std::vector<int32_t> const& audioIndexOffsets)
{
    // Deepstack assembly consults this tensor after embed(). Clear the previous request before deciding whether the
    // current path needs explicit multimodal indices.
    if (!mMultimodalIndices.isEmpty())
    {
        check::check(mMultimodalIndices.reshape({0}), "Multimodal index reset failed");
    }

    // Use the explicit-token-id multimodal path when audio is present, or for
    // image-only inference on audio-capable model families (Nemotron-Omni /
    // Qwen3-Omni) — these keep <image> in-stream and identify themselves via
    // audioTokenId >= 0. Legacy vision-only families (Qwen2.5-VL, InternVL)
    // leave audioTokenId unset (-1) and fall through to the remap path below.
    bool const useExplicitId = audioEmbeds.has_value() || (visionEmbeds.has_value() && mConfig.audioTokenId >= 0);
    if (useExplicitId)
    {
        auto const inputShape = tokenIds.getShape();
        size_t const inputSizeBytes = inputShape.volume() * sizeof(int32_t);
        Tensor inputIdsCPU(inputShape, DeviceType::kCPU, tokenIds.getDataType());
        CUDA_CHECK(cudaMemcpyAsync(
            inputIdsCPU.rawPointer(), tokenIds.rawPointer(), inputSizeBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::optional<int32_t> audioTokenOpt
            = (mConfig.audioTokenId >= 0) ? std::optional{mConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenOpt
            = (mConfig.imageTokenId >= 0) ? std::optional{mConfig.imageTokenId} : std::nullopt;
        Tensor multimodalIndicesCPU = generateMultimodalIndices(
            inputIdsCPU, audioTokenOpt, imageTokenOpt, mConfig.vocabSize, audioIndexOffsets, visionIndexOffsets);

        auto const indicesShape = multimodalIndicesCPU.getShape();
        size_t const indicesSizeBytes = indicesShape.volume() * sizeof(int32_t);
        mMultimodalIndices = Tensor(indicesShape, DeviceType::kGPU, multimodalIndicesCPU.getDataType());
        CUDA_CHECK(cudaMemcpy(mMultimodalIndices.rawPointer(), multimodalIndicesCPU.rawPointer(), indicesSizeBytes,
            cudaMemcpyHostToDevice));

        kernel::embeddingLookupMultimodal(tokenIds, mEmbedding.table, mEmbedding.scalesAsOptional(),
            std::optional{std::cref(mMultimodalIndices)}, imageTokenOpt, visionEmbeds, audioTokenOpt, audioEmbeds,
            io.inputsEmbeds, stream);
    }
    else if (visionEmbeds.has_value())
    {
        // Legacy vision path (Qwen2.5-VL, InternVL: imageTokenId >= vocabSize or not set)
        Tensor const& imageEmbedsTensor = visionEmbeds.value().get();
        kernel::embeddingLookupWithImageInsertion(
            tokenIds, mEmbedding.table, mEmbedding.scalesAsOptional(), imageEmbedsTensor, io.inputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(tokenIds, mEmbedding.table, mEmbedding.scalesAsOptional(), io.inputsEmbeds, stream);
    }
}

OptionalInputTensors EmbeddingPreprocessor::assembleDeepstack(
    Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream)
{
    OptionalInputTensors deepstackEmbeds{};

    if (features.empty())
    {
        return deepstackEmbeds;
    }

    auto const inputShape = tokenIds.getShape();
    int64_t const activeBatchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];

    // Prepare multimodal indices for deepstack assembly (needed when imageTokenId < vocabSize)
    OptionalInputTensor deepstackMultimodalIndices{std::nullopt};
    if (mMultimodalIndices.getShape().volume() > 0)
    {
        deepstackMultimodalIndices = std::cref(mMultimodalIndices);
    }

    for (int32_t idx = 0; idx < static_cast<int32_t>(features.size()); ++idx)
    {
        Tensor const& featureTensor = features[idx].get();

        // Reshape the output slot to match the current batch/sequence dimensions
        check::check(
            io.deepstackEmbeds[idx].reshape({activeBatchSize, seqLen, mConfig.hiddenSize}), "Tensor reshape failed");
        kernel::assembleDeepstackEmbedding(tokenIds, featureTensor, mConfig.vocabSize, io.deepstackEmbeds[idx], stream,
            mConfig.imageTokenId, deepstackMultimodalIndices);

        deepstackEmbeds.push_back(std::cref(io.deepstackEmbeds[idx]));
    }

    return deepstackEmbeds;
}

void EmbeddingPreprocessor::prepareDeepstack(
    Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream)
{
    if (mConfig.numDeepstackFeatures == 0)
    {
        return;
    }

    if (!features.empty())
    {
        assembleDeepstack(tokenIds, features, io, stream);
        return;
    }

    // Text-only request on a VLM engine: zero the deepstack slots so the engine reads known-zero bytes.
    LOG_DEBUG("Deepstack features configured but not available for this text-only request, using zero tensors.");
    auto const inputShape = tokenIds.getShape();
    int64_t const activeBatchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
    {
        check::check(
            io.deepstackEmbeds[idx].reshape({activeBatchSize, seqLen, mConfig.hiddenSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemsetAsync(
            io.deepstackEmbeds[idx].rawPointer(), 0, io.deepstackEmbeds[idx].getMemoryCapacity(), stream));
    }
}

} // namespace rt
} // namespace trt_edgellm
