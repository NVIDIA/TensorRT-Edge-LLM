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

#include "llmInferenceRuntime.h"

#include <algorithm>

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/alpamayoExpertKernels/alpamayoExpertKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{
std::tuple<std::string, std::string> keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

//! \brief Generate multimodal indices for embeddingLookupMultimodal kernel
//!
//! This function scans the input IDs and generates indices for audio/image embeddings.
//! For each multimodal token position, it assigns a sequential index starting from 0.
//!
//! \param inputIds Input token IDs on CPU [batchSize, seqLen]
//! \param audioTokenId Special token ID for audio (<|audio_pad|>), or std::nullopt if no audio
//! \param imageTokenId Special token ID for image (usually >= vocabSize), or std::nullopt if no image
//! \param vocabSize Vocabulary size (for detecting image tokens when imageTokenId is not explicitly set)
//! \return multimodalIndices tensor on CPU [batchSize, seqLen]
rt::Tensor generateMultimodalIndices(rt::Tensor const& inputIds, std::optional<int32_t> audioTokenId,
    std::optional<int32_t> imageTokenId, int32_t vocabSize)
{
    auto const shape = inputIds.getShape();
    check::check(shape.getNumDims() == 2, "inputIds must be 2D tensor");
    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];

    // Create output tensor on CPU
    rt::Tensor multimodalIndices({batchSize, seqLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t* indicesPtr = multimodalIndices.dataPointer<int32_t>();

    // Global indexing across the entire batch (not per-sample)
    int32_t audioIndex = 0;
    int32_t imageIndex = 0;

    // Process each sample in the batch
    for (int64_t b = 0; b < batchSize; ++b)
    {
        for (int64_t s = 0; s < seqLen; ++s)
        {
            int64_t const pos = b * seqLen + s;
            int32_t const tokenId = inputIdsPtr[pos];

            // Check if this is an audio token
            if (audioTokenId.has_value() && tokenId == *audioTokenId)
            {
                indicesPtr[pos] = audioIndex++;
            }
            // Check if this is an image token (either explicit imageTokenId or >= vocabSize)
            else if ((imageTokenId.has_value() && tokenId == *imageTokenId) || tokenId >= vocabSize)
            {
                indicesPtr[pos] = imageIndex++;
            }
            else
            {
                // Normal text token, index not used
                indicesPtr[pos] = 0;
            }
        }
    }

    return multimodalIndices;
}

} // namespace
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    // Find the first .engine file in engineDir
    // For Qwen3-Omni: export ensures only thinker.engine exists in this directory
    std::filesystem::path enginePath;
    for (auto const& entry : std::filesystem::directory_iterator(engineDir))
    {
        if (entry.path().extension() == ".engine")
        {
            enginePath = entry.path();
            break;
        }
    }
    if (enginePath.empty())
    {
        throw std::runtime_error("No .engine file found in directory: " + engineDir);
    }
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    // Load embedding table from embedding.safetensors
    std::filesystem::path const embeddingPath = std::filesystem::path(engineDir) / "embedding.safetensors";
    LOG_INFO("Loading embedding table from: %s", embeddingPath.string().c_str());
    std::vector<rt::Tensor> embeddingTensors;
    if (!safetensors::loadSafetensors(embeddingPath, embeddingTensors, stream))
    {
        LOG_ERROR("Failed to load embedding table from: %s", embeddingPath.string().c_str());
        throw std::runtime_error("Failed to load embedding table from: " + embeddingPath.string());
    }
    check::check(embeddingTensors.size() == 1, "embedding.safetensors should contain exactly one tensor");
    check::check(
        embeddingTensors[0].getShape().getNumDims() == 2, "embedding tensor should be 2D [vocabSize, hiddenSize]");
    mEmbeddingTable = std::move(embeddingTensors[0]);
    LOG_INFO("Embedding table loaded successfully with shape [%d, %d]", mEmbeddingTable.getShape()[0],
        mEmbeddingTable.getShape()[1]);

    try
    {
        mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize LLMEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("LLMEngineRunner successfully loaded and initialized llm engine.");

    mEngineConfig = mLLMEngineRunner->getEngineConfig();

    // Use TopP sampling parameter to reserve max possible workspace size for sampling.
    int32_t const defaultTopK{0};
    float const defaultTopP{0.9F};
    trt_edgellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(trt_edgellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, samplingParams));

    // Allocate workspace and activation tensors for LLM engine.
    try
    {
        // Use Int8 to indicate byte for workspace.
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mSamplingWorkspace");
        mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mInputIds");
        mInputsEmbeds = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
            rt::DeviceType::kGPU, DataType::kHALF, "LLMInferenceRuntime::mInputsEmbeds");
        // Allocate deepstack embeddings if needed (one tensor per feature)
        if (mEngineConfig.numDeepstackFeatures > 0)
        {
            mDeepstackEmbeds.resize(mEngineConfig.numDeepstackFeatures);
            for (int32_t i = 0; i < mEngineConfig.numDeepstackFeatures; ++i)
            {
                mDeepstackEmbeds[i] = rt::Tensor({mEngineConfig.maxSupportedBatchSize,
                                                     mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
                    rt::DeviceType::kGPU, DataType::kHALF,
                    format::fmtstr("LLMInferenceRuntime::mDeepstackEmbeds[%d]", i));
            }
            LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]",
                mEngineConfig.numDeepstackFeatures, mEngineConfig.maxSupportedBatchSize,
                mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize);
        }
        mHostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kCPU, DataType::kINT32, "LLMInferenceRuntime::mHostPackedInputIds");
        mOutputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMInferenceRuntime::mOutputLogits");
        mSelectedIndices = rt::Tensor({mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mSelectedIndices");
        mHostSelectedTokenIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostContextLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostContextLengths");
        mHostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostReuseKVCacheLengths");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate workspace and activation tensors for LLM Inference Runtime: %s", e.what());
        throw std::runtime_error(
            "Failed to allocate workspace and activation tensors for LLM Inference Runtime: " + std::string(e.what()));
    }

    // Setup tokenizer
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }

    // Optional: Load vocabulary mapping table if reduced vocabulary is used
    if (mEngineConfig.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for reduced vocab size: %d -> %d", mEngineConfig.reducedVocabSize,
            mEngineConfig.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        if (!safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream))
        {
            LOG_ERROR(
                "Failed to load %s from model directory: %s", binding_names::kVocabMapFileName, engineDir.c_str());
            throw std::runtime_error("Failed to load " + std::string(binding_names::kVocabMapFileName)
                + " from model directory: " + engineDir);
        }

        // Check we have exactly one tensor and use it
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mEngineConfig.reducedVocabSize,
            "vocab_map tensor length should match reduced vocab size");
        mVocabMappingTable = std::move(vocabMapTensors[0]);
        LOG_INFO("Vocabulary mapping table successfully loaded.");
    }

    // Optional: Setup multimodal engine runners
    if (!multimodalEngineDir.empty())
    {
        // Multimodal engine directory structure:
        //   multimodalEngineDir/audio/  - Audio encoder (audio_encoder.engine, config.json)
        //   multimodalEngineDir/visual/ - Visual encoder (visual.engine, config.json)
        //
        // Note: audio_build and visual_build automatically append /audio and /visual subdirectories.
        // Both builders should use the same base --engineDir path.

        // Helper lambda to try loading a runner from a directory
        auto tryLoadRunner = [&](std::string const& dir, std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            try
            {
                LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
                auto runner = MultimodalRunner::create(
                    dir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
                LOG_INFO("%s runner successfully initialized", name.c_str());
                return runner;
            }
            catch (std::exception const& e)
            {
                LOG_DEBUG("Failed to load %s runner from %s: %s", name.c_str(), dir.c_str(), e.what());
                return nullptr;
            }
        };

        // Try to load audio runner from multimodalEngineDir/audio
        mAudioRunner = tryLoadRunner(multimodalEngineDir + "/audio", "Audio");

        // Try to load visual runner from multimodalEngineDir/visual (with fallback to root for pure visual models)
        mVisionRunner = tryLoadRunner(multimodalEngineDir + "/visual", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = tryLoadRunner(multimodalEngineDir, "Vision");
        }

        // At least one runner must be available
        if (!mAudioRunner && !mVisionRunner)
        {
            throw std::runtime_error("No valid multimodal engine found in " + multimodalEngineDir);
        }
    }
}

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request) noexcept
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("LLMInferenceRuntime(): The request is empty with no requests supplied.");
        return false;
    }

    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("LLMInferenceRuntime(): The batched request size (%d) exceeds the max supported batch size (%d).",
            activeBatchSize, mEngineConfig.maxSupportedBatchSize);
        return false;
    }

    for (auto const& request : request.requests)
    {
        if (request.messages.empty())
        {
            LOG_ERROR(
                "There is an empty request in the batch. 'messages' must be provided. "
                "Skip this batch of requests. Please check the input data contents.");
            return false;
        }
    }

    return true;
}

bool LLMInferenceRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
    std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    // Record the length of the reused KVCache for each sequence using pre-allocated tensor
    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();

    // Search if the system prompt has been cached. If there are cached system prompts, insert
    // the pre-computed KVCache and remove the contents from inputIds.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptKey];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            auto reuseLength = math::cast<size_t>(kvCacheContent.getShape()[3]);
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(math::cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = math::cast<int32_t>(reuseLength);
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            bool const matchIds = std::equal(precachedKVCache.tokenizedPrompt.begin(),
                precachedKVCache.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "LLMInferenceRuntime(): Though system prompt strings are matched, token_ids are not perfectly "
                    "aligned. "
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    // Pack inputIds, instantiate input data for prefill step, and reset the KVCache state.
    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The max input length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    // Reshape and fill the pre-allocated pinned host tensor with pad tokens
    int32_t const packedInputLength = maxInputLength;
    check::check(mHostPackedInputIds.reshape({activeBatchSize, packedInputLength}), "Tensor reshape failed");
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Pad each sequence to the max length of this batch.
        // TODO: Implement remove input padding for better efficiency until multi-batch.
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // For each Mamba layer and each batch element, either restore the cached
    // SSM/conv state if the cache hit, or zero the state.
    if (mEngineConfig.numMambaLayers > 0)
    {
        rt::LinearKVCache& kvCache = mLLMEngineRunner->getLinearKVCache();
        rt::LinearKVCache::CacheConfig const& cacheConfig = kvCache.getConfig();
        size_t const ssmElemSize = rt::utils::getTypeSize(cacheConfig.ssmStateType);
        size_t const convElemSize = rt::utils::getTypeSize(cacheConfig.convStateType);
        size_t const ssmBatchBytes
            = static_cast<size_t>(cacheConfig.mambaNumHeads * cacheConfig.mambaHeadDim * cacheConfig.ssmStateSize)
            * ssmElemSize;
        size_t const convBatchBytes = static_cast<size_t>(cacheConfig.convDim * cacheConfig.convKernel) * convElemSize;

        for (int32_t layer = 0; layer < mEngineConfig.numMambaLayers; ++layer)
        {
            rt::Tensor ssmLayer = kvCache.getSSMStateForLayer(layer);
            rt::Tensor convLayer = kvCache.getConvStateForLayer(layer);

            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                auto* ssmDst = static_cast<std::byte*>(ssmLayer.rawPointer()) + i * ssmBatchBytes;
                auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + i * convBatchBytes;

                auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
                auto it = mSystemPromptKVCache.find(promptKey);
                bool const hasCache = (it != mSystemPromptKVCache.end());

                if (hasCache && layer < static_cast<int32_t>(it->second.ssmStateContents.size()))
                {
                    CUDA_CHECK(cudaMemcpyAsync(ssmDst, it->second.ssmStateContents[layer].rawPointer(), ssmBatchBytes,
                        cudaMemcpyDeviceToDevice, stream));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(ssmDst, 0, ssmBatchBytes, stream));
                }

                if (hasCache && layer < static_cast<int32_t>(it->second.convStateContents.size()))
                {
                    CUDA_CHECK(cudaMemcpyAsync(convDst, it->second.convStateContents[layer].rawPointer(),
                        convBatchBytes, cudaMemcpyDeviceToDevice, stream));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
                }
            }
        }
    }

    check::check(mInputIds.reshape({activeBatchSize, packedInputLength}), "Tensor reshape failed");
    check::check(mHostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize}), "Tensor reshape failed");

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", loraWeightsName.c_str());
        return false;
    }

    return true;
}

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;
    std::string loraWeightsName = request.loraWeightsName;

    if (!examineRequest(request))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request examination failed. This request cannot be handled.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    // Apply chat template, extract system prompts, and optionally save KVCache
    request.formattedRequests.resize(activeBatchSize);
    batchSystemPrompts.reserve(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.enableThinking);

        // Extract system prompt
        batchSystemPrompts.emplace_back(request.formattedRequests[i].formattedSystemPrompt);

        // Save KVCache if requested
        if (request.saveSystemPromptKVCache)
        {
            if (mVisionRunner)
            {
                mVisionRunner->preprocessSystemPrompt(
                    batchSystemPrompts[i], mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }
            else if (mAudioRunner)
            {
                mAudioRunner->preprocessSystemPrompt(
                    batchSystemPrompts[i], mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(batchSystemPrompts[i], loraWeightsName, stream);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache. Continue to handle the request without saving the system "
                    "prompt KVCache.");
            }
        }
    }

    // Preprocess user prompts and encode them.
    // Check if request has audio or vision inputs
    bool hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool hasPreparedVision = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return !req.preparedVisualInputPath.empty(); });

    if ((hasAudio && mAudioRunner) || ((hasVision || hasPreparedVision) && mVisionRunner))
    {
        // Mark multimodal preprocessing and inference for NVTX profiling
        NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);

        // Process audio inputs (if present)
        if (hasAudio && mAudioRunner)
        {
            LOG_INFO("Processing audio inputs");
            if (!mAudioRunner->preprocess(
                    request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Audio preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mAudioRunner->infer(stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Audio inference failed. This request cannot be handled.");
                return false;
            }
        }

        // Process vision inputs (if present)
        if ((hasVision || hasPreparedVision) && mVisionRunner)
        {
            LOG_INFO("Processing vision inputs");
            bool visionStatus = false;
            if (hasPreparedVision)
            {
                check::check(activeBatchSize == 1, "prepared_visual_input currently supports batch_size == 1 only");
                check::check(!hasVision, "prepared_visual_input cannot be mixed with raw image inputs in the same request");
                std::vector<rt::Tensor> preparedTensors;
                if (!safetensors::loadSafetensors(request.requests[0].preparedVisualInputPath, preparedTensors, stream))
                {
                    LOG_ERROR("Failed to load prepared visual input from: %s",
                        request.requests[0].preparedVisualInputPath.c_str());
                    return false;
                }

                rt::Tensor const* pixelValues = nullptr;
                rt::Tensor const* imageGridTHW = nullptr;
                for (auto const& tensor : preparedTensors)
                {
                    if (tensor.getName() == "pixel_values")
                    {
                        pixelValues = &tensor;
                    }
                    else if (tensor.getName() == "image_grid_thw")
                    {
                        imageGridTHW = &tensor;
                    }
                }
                check::check(pixelValues != nullptr, "prepared_visual_input is missing tensor 'pixel_values'");
                check::check(imageGridTHW != nullptr, "prepared_visual_input is missing tensor 'image_grid_thw'");

                visionStatus = mVisionRunner->preprocessPreparedVisual(request, batchedInputIds, mTokenizer.get(),
                    mLLMEngineRunner->getRopeCosSinCacheTensor(), *pixelValues, *imageGridTHW, stream);
            }
            else
            {
                visionStatus = mVisionRunner->preprocess(
                    request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }

            if (!visionStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Vision preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mVisionRunner->infer(stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Vision inference failed. This request cannot be handled.");
                return false;
            }
        }
    }
    else
    {
        // Pure text mode: directly tokenize
        batchedInputIds.reserve(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, true));
            if (batchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to encode input text for request %d in batch", i);
                return false;
            }
        }
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(batchedInputIds, batchSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR("LLMInferenceRuntime(): Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // Record context information for performance tracking
    auto tokenCount = calculateTokenCounts(batchedInputIds, batchSystemPrompts, loraWeightsName);

    int32_t const maxInputIdsLength = mInputIds.getShape()[1];
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (maxInputIdsLength + maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
    {
        maxGenerationLength = mEngineConfig.maxKVCacheCapacity - maxInputIdsLength;
        LOG_WARNING(
            "The requested input length (%d) + max generation length (%d) = %d exceeds the max KV "
            "cache capacity (%d). Reduce the generation length to %d to avoid the truncation of the generated tokens.",
            maxInputIdsLength, maxGenerationLength, maxInputIdsLength + maxGenerationLength,
            mEngineConfig.maxKVCacheCapacity, maxGenerationLength);
    }

    // Set up data structures to store the generated results during decoding.
    // Also set up sampling parameters and sampling lambda function.
    int32_t unFinishedBatchNum = activeBatchSize;
    int32_t generationIter{0};
    std::vector<std::vector<int32_t>> outputIds(activeBatchSize);
    std::vector<bool> finishedStates(activeBatchSize, false);
    check::check(mSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();

    SamplingParams params(
        activeBatchSize, mEngineConfig.outputVocabSize, request.temperature, request.topK, request.topP);
    auto sampleTokens = [&]() {
        trt_edgellm::topKtopPSamplingFromLogits(mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
        // Apply vocabulary mapping if reduced vocabulary is used
        if (mEngineConfig.reducedVocabSize > 0)
        {
            trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!finishedStates[i])
            {
                outputIds[i].push_back(hostSelectedTokenIdsData[i]);
                finishedStates[i] = hostSelectedTokenIdsData[i] == mTokenizer->getEosId();
                // Early stop AFTER <traj_future_start> when Expert runner is active.
                // We need traj_future_start to be processed by the decoder (one more iteration)
                // so its KV cache entry exists. Stop on the token AFTER it.
                if (mExpertRunner && outputIds[i].size() >= 2)
                {
                    int32_t prevToken = outputIds[i][outputIds[i].size() - 2];
                    if (prevToken == mExpertRunner->getConfig().trajFutureStartTokenId)
                    {
                        finishedStates[i] = true;
                    }
                }
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
    };

    // Perform embedding lookup for prefill
    int32_t const prefillSequenceLength = mInputIds.getShape()[1];
    check::check(mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
        "Tensor reshape failed");

    // Get embeddings from independent runners
    rt::OptionalInputTensor visionEmbeddings
        = mVisionRunner ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = mAudioRunner ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;

    if (audioEmbeddings.has_value())
    {
        // Audio present: use embeddingLookupMultimodal (handles audio and/or vision)
        auto const inputShape = mInputIds.getShape();
        size_t const inputSizeBytes = inputShape.volume() * sizeof(int32_t);
        rt::Tensor inputIdsCPU(inputShape, rt::DeviceType::kCPU, mInputIds.getDataType());
        CUDA_CHECK(
            cudaMemcpy(inputIdsCPU.rawPointer(), mInputIds.rawPointer(), inputSizeBytes, cudaMemcpyDeviceToHost));

        std::optional<int32_t> audioTokenId
            = (mEngineConfig.audioTokenId != 0) ? std::optional{mEngineConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenId
            = (mEngineConfig.imageTokenId != 0) ? std::optional{mEngineConfig.imageTokenId} : std::nullopt;
        rt::Tensor multimodalIndicesCPU
            = generateMultimodalIndices(inputIdsCPU, audioTokenId, imageTokenId, mEngineConfig.vocabSize);

        auto const indicesShape = multimodalIndicesCPU.getShape();
        size_t const indicesSizeBytes = indicesShape.volume() * sizeof(int32_t);
        mMultimodalIndices = rt::Tensor(indicesShape, rt::DeviceType::kGPU, multimodalIndicesCPU.getDataType());
        CUDA_CHECK(cudaMemcpy(mMultimodalIndices.rawPointer(), multimodalIndicesCPU.rawPointer(), indicesSizeBytes,
            cudaMemcpyHostToDevice));

        kernel::embeddingLookupMultimodal(mInputIds, mEmbeddingTable, std::optional{std::ref(mMultimodalIndices)},
            imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings, mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Legacy vision path (Qwen2.5-VL, InternVL: imageTokenId >= vocabSize or not set)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding assembly if vision runner is available
    // Note: Deepstack features are only provided by VisionRunner, not Qwen3OmniAudioRunner
    rt::OptionalInputTensors deepstackEmbeds{};
    if (mEngineConfig.numDeepstackFeatures > 0 && mVisionRunner)
    {
        rt::OptionalInputTensors deepstackFeatures = mVisionRunner->getDeepstackFeatures();

        // Prepare multimodal indices for deepstack assembly (needed when imageTokenId < vocabSize)
        rt::OptionalInputTensor deepstackMultimodalIndices{std::nullopt};
        if (mMultimodalIndices.getShape().volume() > 0)
        {
            deepstackMultimodalIndices = std::ref(mMultimodalIndices);
        }

        for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
        {
            rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

            // Reshape and perform embedding assembly for this feature
            check::check(
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
                "Tensor reshape failed");
            kernel::assembleDeepstackEmbedding(mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx],
                stream, mEngineConfig.imageTokenId, deepstackMultimodalIndices);

            // Add to output vector (engine will bind by index)
            deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
        }
    }

    // Profile all sampling operations as one stage
    // Prefill profiling session
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);
        // Enhanced NVTX range with detailed information
        NVTX_SCOPED_RANGE(nvtx_prefill,
            ("LLM_PREFILL[BS=" + std::to_string(activeBatchSize)
                + ",Reused=" + std::to_string(tokenCount.totalReusedTokens)
                + ",Computed=" + std::to_string(tokenCount.totalComputedTokens) + "]")
                .c_str(),
            nvtx_colors::BLUE);

        bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputsEmbeds, mHostContextLengths, deepstackEmbeds,
            mOutputLogits, rt::OptionalOutputTensor{std::nullopt}, stream);
        if (!prefillStatus)
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Failed to execute prefill step. Cannot generate the KVCache for this prompt.");
            return false;
        }

        sampleTokens();
    }

    // Record prefill metrics
    mPrefillMetrics.recordRun(tokenCount.totalReusedTokens, tokenCount.totalComputedTokens);

    bool const hasPrefillDumpRequest = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return !req.dumpPrefillKVCachePath.empty(); });
    bool const hasPrefillOnlyRequest
        = std::any_of(request.requests.begin(), request.requests.end(), [](auto const& req) { return req.prefillOnly; });

    if (hasPrefillDumpRequest || hasPrefillOnlyRequest)
    {
        check::check(activeBatchSize == 1,
            "dump_prefill_kv_cache / prefill_only currently supports batch_size == 1 only");

        if (!request.requests[0].dumpPrefillKVCachePath.empty())
        {
            int32_t const sequenceLength = mHostContextLengths.dataPointer<int32_t>()[0];
            if (!dumpCurrentPrefillKVCache(request.requests[0].dumpPrefillKVCachePath, sequenceLength, stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to dump request prefill KV cache to %s",
                    request.requests[0].dumpPrefillKVCachePath.c_str());
                return false;
            }
        }

        if (request.requests[0].prefillOnly)
        {
            response.outputIds.assign(activeBatchSize, {});
            response.outputTexts.assign(activeBatchSize, "");
            return true;
        }
    }

    // Reshape for decoding step
    check::check(mInputsEmbeds.reshape({activeBatchSize, 1, mEngineConfig.hiddenSize}), "Tensor reshape failed");

    // Profile entire generation phase like benchmark profiler
    {
        TIME_STAGE(metrics::StageNames::kLLM_GENERATION, stream);
        // Enhanced NVTX range with batch size
        NVTX_SCOPED_RANGE(nvtx_generation,
            ("LLM_GENERATION[BS=" + std::to_string(activeBatchSize) + ",MaxLen=" + std::to_string(maxGenerationLength)
                + "]")
                .c_str(),
            nvtx_colors::GREEN);

        while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
        {
            // Mark each decoding iteration with detailed info
            NVTX_SCOPED_RANGE(iter_range,
                ("Decode_Iter[" + std::to_string(generationIter) + "/" + std::to_string(maxGenerationLength)
                    + ",Active=" + std::to_string(unFinishedBatchNum) + "]")
                    .c_str(),
                nvtx_colors::LIGHT_GREEN);

            // Perform embedding lookup for the selected token indices (decode only has text, no images)
            kernel::embeddingLookup(mSelectedIndices, mEmbeddingTable, mInputsEmbeds, stream);

            // Use the embedded tokens as input for the decoding step.
            // No hidden states output needed for standard LLM decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(
                mInputsEmbeds, mOutputLogits, outputHiddenStates, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            sampleTokens();
        }
    }

    // Record generation and sampling metrics
    int32_t totalGeneratedTokens = 0;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        totalGeneratedTokens += static_cast<int32_t>(outputIds[i].size() - 1);
    }

    if (totalGeneratedTokens > 0)
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // ===== KV cache dump after decode (Path A) =====
    if (!mKVCacheDumpPath.empty())
    {
        auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
        auto cacheConfig = linearKVCache.getConfig();
        auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();

        int32_t totalGenLen = static_cast<int32_t>(outputIds[0].size());
        int32_t totalSeqLen = prefillSequenceLength + totalGenLen - 1;

        rt::Coords dumpShape{cacheConfig.numAttentionLayers, 2, cacheConfig.numKVHeads,
            static_cast<int64_t>(totalSeqLen), cacheConfig.headDim};
        rt::Tensor dumpTensor(dumpShape, rt::DeviceType::kGPU,
            cacheConfig.kvCacheTypeTRT, "LLMInferenceRuntime::dumpKVCache");
        kernel::saveKVCacheIntoTensor(dumpTensor, kvCacheBuffer, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int64_t numBytes = dumpTensor.getMemoryCapacity();
        std::vector<uint8_t> hostBuf(numBytes);
        CUDA_CHECK(cudaMemcpy(hostBuf.data(), dumpTensor.rawPointer(), numBytes, cudaMemcpyDeviceToHost));

        std::string binPath = mKVCacheDumpPath + ".req" + std::to_string(mKVCacheDumpReqIdx) + ".bin";
        std::string metaPath = mKVCacheDumpPath + ".req" + std::to_string(mKVCacheDumpReqIdx) + ".shape.json";
        {
            std::ofstream binOut(binPath, std::ios::binary);
            binOut.write(reinterpret_cast<char const*>(hostBuf.data()), numBytes);
        }
        {
            std::ofstream metaOut(metaPath);
            metaOut << "{\n";
            metaOut << "  \"shape\": [" << cacheConfig.numAttentionLayers << ", 2, "
                    << cacheConfig.numKVHeads << ", " << totalSeqLen << ", "
                    << cacheConfig.headDim << "],\n";
            metaOut << "  \"dtype\": " << static_cast<int>(cacheConfig.kvCacheTypeTRT) << ",\n";
            metaOut << "  \"prefill_seq_len\": " << prefillSequenceLength << ",\n";
            metaOut << "  \"total_seq_len\": " << totalSeqLen << ",\n";
            metaOut << "  \"num_generated\": " << (totalGenLen - 1) << ",\n";
            metaOut << "  \"num_bytes\": " << numBytes << ",\n";
            metaOut << "  \"req_idx\": " << mKVCacheDumpReqIdx << ",\n";
            std::vector<int32_t> hostInputIds(prefillSequenceLength);
            CUDA_CHECK(cudaMemcpy(hostInputIds.data(), mInputIds.rawPointer(),
                prefillSequenceLength * sizeof(int32_t), cudaMemcpyDeviceToHost));
            metaOut << "  \"input_ids\": [";
            for (int32_t i = 0; i < prefillSequenceLength; ++i) {
                if (i > 0) metaOut << ", ";
                metaOut << hostInputIds[i];
            }
            metaOut << "],\n";
            metaOut << "  \"output_ids\": [";
            for (size_t i = 0; i < outputIds[0].size(); ++i) {
                if (i > 0) metaOut << ", ";
                metaOut << outputIds[0][i];
            }
            metaOut << "]\n";
            metaOut << "}\n";
        }
        LOG_INFO("Dumped KV cache (post-decode): %s (%ld bytes, total_seq_len=%d, generated=%d)",
            binPath.c_str(), numBytes, totalSeqLen, totalGenLen - 1);
        mKVCacheDumpReqIdx++;
    }
    // ===== end KV cache dump =====
    // ===== Run Expert diffusion with multi-sequence VLM decode =====
    if (mExpertRunner)
    {
        auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
        auto cacheConfig = linearKVCache.getConfig();
        auto const& expertCfg = mExpertRunner->getConfig();
        int32_t const numCandidates = expertCfg.numCandidates;
        int32_t const L = cacheConfig.numAttentionLayers;
        int32_t const H = cacheConfig.numKVHeads;
        int32_t const D = cacheConfig.headDim;

        int32_t truncSeqLen = mExpertRunner->findTrajFutureStart(outputIds[0], prefillSequenceLength);
        if (truncSeqLen < 0)
        {
            LOG_WARNING("ExpertRunner: <traj_future_start> not found, skipping expert");
        }
        else if (!expertCfg.multiSeq)
        {
            // ── Single-sequence mode ──
            LOG_INFO("ExpertRunner: single-seq, truncKV=%d", truncSeqLen);
            auto kvBuf = linearKVCache.getKVCacheBuffer();
            rt::Tensor vlmKV({L, 2, H, static_cast<int64_t>(truncSeqLen), D},
                rt::DeviceType::kGPU, cacheConfig.kvCacheTypeTRT, "vlmKV");
            kernel::saveKVCacheIntoTensor(vlmKV, kvBuf, 0, stream);

            rt::Tensor actions({expertCfg.numCandidates, expertCfg.numDiffusionTokens, expertCfg.actionDim},
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "actions");
            mExpertRunner->runDiffusion(vlmKV, truncSeqLen, expertCfg.ropeDelta, actions, stream);

            std::vector<float> hostAct(expertCfg.numCandidates * expertCfg.numDiffusionTokens * expertCfg.actionDim);
            CUDA_CHECK(cudaMemcpy(hostAct.data(), actions.rawPointer(),
                hostAct.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float minV = *std::min_element(hostAct.begin(), hostAct.end());
            float maxV = *std::max_element(hostAct.begin(), hostAct.end());
            LOG_INFO("ExpertRunner: actions [%.4f, %.4f]", minV, maxV);
            if (!mKVCacheDumpPath.empty())
            {
                std::string p = mKVCacheDumpPath + ".req" + std::to_string(mKVCacheDumpReqIdx - 1) + ".actions.bin";
                std::ofstream o(p, std::ios::binary);
                o.write(reinterpret_cast<char const*>(hostAct.data()), hostAct.size() * sizeof(float));
                LOG_INFO("ExpertRunner: dumped %s", p.c_str());
            }
        }
        else
        {
            // ── Multi-sequence mode: decode N times from prefill checkpoint ──
            LOG_INFO("ExpertRunner: multi-seq mode, %d candidates", numCandidates);

            // Candidate 0: already decoded, extract its KV
            auto kvBuf0 = linearKVCache.getKVCacheBuffer();
            rt::Tensor cand0KV({L, 2, H, static_cast<int64_t>(truncSeqLen), D},
                rt::DeviceType::kGPU, cacheConfig.kvCacheTypeTRT, "cand0KV");
            kernel::saveKVCacheIntoTensor(cand0KV, kvBuf0, 0, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            LOG_INFO("ExpertRunner: cand 0 done (truncLen=%d)", truncSeqLen);

            // Save the full KV buffer state (for restoration)
            size_t const kvBufBytes = kvBuf0.getMemoryCapacity();
            rt::Tensor savedFullKV({static_cast<int64_t>(kvBufBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "savedFullKV");
            CUDA_CHECK(cudaMemcpyAsync(savedFullKV.rawPointer(), kvBuf0.rawPointer(),
                kvBufBytes, cudaMemcpyDeviceToDevice, stream));

            // Also save KV lengths
            auto& kvLengths = linearKVCache.getKVCacheLengths();
            rt::Tensor savedKVLens({kvLengths.getShape()[0]}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kINT32, "savedKVLens");
            CUDA_CHECK(cudaMemcpyAsync(savedKVLens.rawPointer(), kvLengths.rawPointer(),
                kvLengths.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));

            // Create prefill-only checkpoint: reset KV lengths to prefillSequenceLength
            int32_t prefillLenHost = prefillSequenceLength;
            rt::Tensor savedPrefillKV({static_cast<int64_t>(kvBufBytes)}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "savedPrefillKV");
            CUDA_CHECK(cudaMemcpyAsync(savedPrefillKV.rawPointer(), kvBuf0.rawPointer(),
                kvBufBytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Collect all candidate KVs and truncation lengths
            std::vector<rt::Tensor> candKVs;
            candKVs.push_back(std::move(cand0KV));
            std::vector<int32_t> candTruncLens;
            candTruncLens.push_back(truncSeqLen);

            // Decode candidates 1..N-1
            for (int32_t ci = 1; ci < numCandidates; ++ci)
            {
                // Restore prefill KV state
                CUDA_CHECK(cudaMemcpyAsync(linearKVCache.getKVCacheBuffer().rawPointer(),
                    savedPrefillKV.rawPointer(), kvBufBytes, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(kvLengths.rawPointer(), &prefillLenHost,
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Decode loop
                std::vector<int32_t> candIds;
                candIds.push_back(outputIds[0][0]);  // first token same for all (from prefill)
                bool candDone = false;
                int32_t candIter = 1;

                check::check(mInputsEmbeds.reshape({1, 1, mEngineConfig.hiddenSize}), "reshape failed");

                while (!candDone && candIter < maxGenerationLength)
                {
                    int32_t lastTok = candIds.back();
                    CUDA_CHECK(cudaMemcpyAsync(mSelectedIndices.rawPointer(), &lastTok,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    kernel::embeddingLookup(mSelectedIndices, mEmbeddingTable, mInputsEmbeds, stream);

                    bool ok = mLLMEngineRunner->executeVanillaDecodingStep(
                        mInputsEmbeds, mOutputLogits, rt::OptionalOutputTensor{std::nullopt}, stream);
                    if (!ok) break;

                    SamplingParams params(1, mEngineConfig.outputVocabSize,
                        request.temperature, request.topK, request.topP);
                    trt_edgellm::topKtopPSamplingFromLogits(
                        mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
                    if (mEngineConfig.reducedVocabSize > 0)
                        trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);

                    int32_t sampled;
                    CUDA_CHECK(cudaMemcpyAsync(&sampled, mSelectedIndices.rawPointer(),
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    candIds.push_back(sampled);

                    if (candIds.size() >= 2)
                    {
                        int32_t prev = candIds[candIds.size() - 2];
                        if (prev == expertCfg.trajFutureStartTokenId)
                            candDone = true;
                    }
                    if (sampled == mTokenizer->getEosId())
                        candDone = true;

                    ++candIter;
                }

                int32_t candTrunc = mExpertRunner->findTrajFutureStart(candIds, prefillSequenceLength);
                if (candTrunc < 0) candTrunc = truncSeqLen;

                auto candBuf = linearKVCache.getKVCacheBuffer();
                rt::Tensor candKV({L, 2, H, static_cast<int64_t>(candTrunc), D},
                    rt::DeviceType::kGPU, cacheConfig.kvCacheTypeTRT,
                    ("cand" + std::to_string(ci) + "KV").c_str());
                kernel::saveKVCacheIntoTensor(candKV, candBuf, 0, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                candKVs.push_back(std::move(candKV));
                candTruncLens.push_back(candTrunc);
                LOG_INFO("ExpertRunner: cand %d done (truncLen=%d, decoded %d)", ci, candTrunc, candIter);
            }

            // Use minimum truncation length across candidates
            int32_t minTrunc = *std::min_element(candTruncLens.begin(), candTruncLens.end());

            // Stack KV caches: per-candidate (L,2,H,S,D) fp16 → combined (L*2,N,H,S,D) fp32
            int64_t const L2 = L * 2;
            int32_t const T = expertCfg.numDiffusionTokens;
            int32_t const A = expertCfg.actionDim;

            rt::Tensor stackedKV({L2, static_cast<int64_t>(numCandidates), H,
                static_cast<int64_t>(minTrunc), D},
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "stackedKV");

            int64_t const sliceElems = static_cast<int64_t>(H) * minTrunc * D;
            for (int32_t c = 0; c < numCandidates; ++c)
            {
                // Reshape single candidate to (L2, 1, H, minTrunc, D) fp32
                rt::Tensor tmp({L2, 1, H, static_cast<int64_t>(minTrunc), D},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "tmp");
                kernel::kvCacheReshapeRepeat(tmp.dataPointer<float>(),
                    reinterpret_cast<half const*>(candKVs[c].rawPointer()),
                    L, H, minTrunc, D, 1, stream);

                // Copy into stacked[:, c, :, :, :]
                for (int64_t l = 0; l < L2; ++l)
                {
                    float* dst = stackedKV.dataPointer<float>() + l * numCandidates * sliceElems + c * sliceElems;
                    float const* src2 = tmp.dataPointer<float>() + l * sliceElems;
                    CUDA_CHECK(cudaMemcpyAsync(dst, src2, sliceElems * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Build Expert inputs and run diffusion
            int32_t const totalMaskLen = minTrunc + T;
            int64_t basePos = static_cast<int64_t>(minTrunc) + expertCfg.ropeDelta;

            rt::Tensor posIds({3, static_cast<int64_t>(numCandidates), static_cast<int64_t>(T)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "posIds");
            kernel::buildPositionIds(posIds.dataPointer<int64_t>(), numCandidates, T, basePos, stream);

            rt::Tensor mask({numCandidates, 1, T, static_cast<int64_t>(totalMaskLen)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mask");
            CUDA_CHECK(cudaMemsetAsync(mask.rawPointer(), 0,
                static_cast<size_t>(numCandidates) * T * totalMaskLen * sizeof(float), stream));

            rt::Tensor noisyAct({numCandidates, T, A},
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "noisyAct");
            curandGenerator_t rng;
            curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(rng, expertCfg.seed);
            curandSetStream(rng, stream);
            check::check((numCandidates * T * A) % 2 == 0, "curandGenerateNormal requires even element count");
            curandGenerateNormal(rng, noisyAct.dataPointer<float>(), numCandidates * T * A, 0.0f, 1.0f);

            rt::Tensor predVel({numCandidates, T, A},
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "predVel");
            rt::Tensor ts({numCandidates, 1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ts");

            auto* ctx = mExpertRunner->getContext();
            nvinfer1::Dims dims;
            dims.nbDims = 3; dims.d[0] = numCandidates; dims.d[1] = T; dims.d[2] = A;
            ctx->setInputShape("noisy_action", dims);
            dims.d[0] = numCandidates; dims.d[1] = 1; dims.d[2] = 1;
            ctx->setInputShape("timestep", dims);
            dims.d[0] = 3; dims.d[1] = numCandidates; dims.d[2] = T;
            ctx->setInputShape("position_ids", dims);
            dims.nbDims = 4; dims.d[0] = numCandidates; dims.d[1] = 1; dims.d[2] = T; dims.d[3] = totalMaskLen;
            ctx->setInputShape("attention_mask", dims);
            dims.nbDims = 5; dims.d[0] = L2; dims.d[1] = numCandidates; dims.d[2] = H;
            dims.d[3] = minTrunc; dims.d[4] = D;
            ctx->setInputShape("kv_cache", dims);

            ctx->setTensorAddress("noisy_action", noisyAct.rawPointer());
            ctx->setTensorAddress("timestep", ts.rawPointer());
            ctx->setTensorAddress("position_ids", posIds.rawPointer());
            ctx->setTensorAddress("attention_mask", mask.rawPointer());
            ctx->setTensorAddress("kv_cache", stackedKV.rawPointer());
            ctx->setTensorAddress("pred_velocity", predVel.rawPointer());

            int32_t const nSteps = expertCfg.numDiffusionSteps;
            float const dt = 1.0f / static_cast<float>(nSteps);
            int32_t const actElems = numCandidates * T * A;
            for (int32_t step = 0; step < nSteps; ++step)
            {
                kernel::fillTimestep(ts.dataPointer<float>(), numCandidates, step * dt, stream);
                auto ok = ctx->enqueueV3(stream);
                check::check(ok, "Expert enqueueV3 failed in multi-seq diffusion step");
                kernel::eulerUpdate(noisyAct.dataPointer<float>(), predVel.dataPointer<float>(), dt, actElems, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            curandDestroyGenerator(rng);

            std::vector<float> hostAct(actElems);
            CUDA_CHECK(cudaMemcpy(hostAct.data(), noisyAct.rawPointer(), actElems * sizeof(float), cudaMemcpyDeviceToHost));
            float minV = *std::min_element(hostAct.begin(), hostAct.end());
            float maxV = *std::max_element(hostAct.begin(), hostAct.end());
            LOG_INFO("ExpertRunner: multi-seq done, %d cands, actions [%.4f, %.4f]", numCandidates, minV, maxV);

            if (!mKVCacheDumpPath.empty())
            {
                std::string p = mKVCacheDumpPath + ".req" + std::to_string(mKVCacheDumpReqIdx - 1) + ".actions.bin";
                std::ofstream o(p, std::ios::binary);
                o.write(reinterpret_cast<char const*>(hostAct.data()), hostAct.size() * sizeof(float));
                LOG_INFO("ExpertRunner: dumped %s", p.c_str());
            }

            // Restore original KV state
            CUDA_CHECK(cudaMemcpyAsync(linearKVCache.getKVCacheBuffer().rawPointer(),
                savedFullKV.rawPointer(), kvBufBytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(kvLengths.rawPointer(), savedKVLens.rawPointer(),
                kvLengths.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
        }
    }
    // ===== end Expert diffusion =====

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
    }

    return true;
}

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    int32_t const maxSupportedBatchSize = mEngineConfig.maxSupportedBatchSize;
    int32_t const minSupportedBatchSize = 1;

    bool captureStatus{true};
    // Capture the CUDA graph for all available batch sizes.
    for (int32_t batchSize = minSupportedBatchSize; batchSize <= maxSupportedBatchSize; ++batchSize)
    {
        check::check(mSelectedIndices.reshape({batchSize, 1}), "Tensor reshape failed");
        check::check(mInputsEmbeds.reshape({batchSize, 1, mEngineConfig.hiddenSize}), "Tensor reshape failed");
        check::check(mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize}), "Tensor reshape failed");

        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
            mInputsEmbeds, mOutputLogits, mEmptyLoraWeightsName, stream);
        if (mEngineConfig.maxSupportedLoraRank > 0)
        {
            for (auto const& loraWeightsName : mLLMEngineRunner->getAvailableLoraWeights())
            {
                captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
                    mInputsEmbeds, mOutputLogits, loraWeightsName, stream);
            }
        }
    }

    if (captureStatus)
    {
        LOG_INFO(
            "LLMInferenceRuntime(): Successfully captured the decoding CUDA graph for all execution batch sizes and "
            "LoRA weights.");
    }
    else
    {
        LOG_WARNING(
            "LLMInferenceRuntime(): Failed to capture the decoding CUDA graph for some of execution batch sizes and "
            "LoRA weights.");
    }
    return captureStatus;
}

LLMInferenceRuntime::TokenCountInfo LLMInferenceRuntime::calculateTokenCounts(
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<std::string> const& systemPrompts,
    std::string const& loraWeightsName) const noexcept
{
    TokenCountInfo tokenCount;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t contextLength = static_cast<int32_t>(batchedInputIds[i].size());
        // Calculate reused length from system prompt cache
        auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
        {
            int32_t reusedLength = static_cast<int32_t>(mSystemPromptKVCache.at(promptKey).tokenizedPrompt.size());
            tokenCount.totalReusedTokens += reusedLength;
            tokenCount.totalComputedTokens += (contextLength - reusedLength);
        }
        else
        {
            tokenCount.totalComputedTokens += contextLength;
        }
    }

    return tokenCount;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("LLMInferenceRuntime(): The prompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    // hash the prompt if check if the prompt cache already exists.
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
    {
        LOG_DEBUG(
            "LLMInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());
    int32_t const activeBatchSize = 1;

    if (promptIdsLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The prompt length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            promptIdsLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(activeBatchSize, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(activeBatchSize, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): Prefill execution setup failed. Cannot generate the KVCache for this prompt.");
        return false;
    }

    // Execute prefill step to initialize the KVCache data.
    // Perform embedding lookup
    int32_t const prefillSequenceLength = mInputIds.getShape()[1];
    check::check(mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
        "Tensor reshape failed");

    // Get embeddings from independent runners
    rt::OptionalInputTensor visionEmbeddings
        = mVisionRunner ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = mAudioRunner ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;

    if (audioEmbeddings.has_value())
    {
        // Audio present: use embeddingLookupMultimodal (handles audio and/or vision)
        auto const inputShape = mInputIds.getShape();
        size_t const inputSizeBytes = inputShape.volume() * sizeof(int32_t);
        rt::Tensor inputIdsCPU(inputShape, rt::DeviceType::kCPU, mInputIds.getDataType());
        CUDA_CHECK(
            cudaMemcpy(inputIdsCPU.rawPointer(), mInputIds.rawPointer(), inputSizeBytes, cudaMemcpyDeviceToHost));

        std::optional<int32_t> audioTokenId
            = (mEngineConfig.audioTokenId != 0) ? std::optional{mEngineConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenId
            = (mEngineConfig.imageTokenId != 0) ? std::optional{mEngineConfig.imageTokenId} : std::nullopt;
        rt::Tensor multimodalIndicesCPU
            = generateMultimodalIndices(inputIdsCPU, audioTokenId, imageTokenId, mEngineConfig.vocabSize);

        auto const indicesShape = multimodalIndicesCPU.getShape();
        size_t const indicesSizeBytes = indicesShape.volume() * sizeof(int32_t);
        mMultimodalIndices = rt::Tensor(indicesShape, rt::DeviceType::kGPU, multimodalIndicesCPU.getDataType());
        CUDA_CHECK(cudaMemcpy(mMultimodalIndices.rawPointer(), multimodalIndicesCPU.rawPointer(), indicesSizeBytes,
            cudaMemcpyHostToDevice));

        kernel::embeddingLookupMultimodal(mInputIds, mEmbeddingTable, std::optional{std::ref(mMultimodalIndices)},
            imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings, mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Vision-only (Qwen2-VL, InternVL, etc.)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding lookup if vision runner is available
    rt::OptionalInputTensors deepstackEmbeds{};
    if (mEngineConfig.numDeepstackFeatures > 0 && mVisionRunner)
    {
        rt::OptionalInputTensors deepstackFeatures = mVisionRunner->getDeepstackFeatures();

        rt::OptionalInputTensor deepstackMultimodalIndices{std::nullopt};
        if (mMultimodalIndices.getShape().volume() > 0)
        {
            deepstackMultimodalIndices = std::ref(mMultimodalIndices);
        }

        for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
        {
            rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

            check::check(
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
                "Tensor reshape failed");
            kernel::assembleDeepstackEmbedding(mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx],
                stream, mEngineConfig.imageTokenId, deepstackMultimodalIndices);

            deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
        }
    }

    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    bool prefillStatus = mLLMEngineRunner->executePrefillStep(
        mInputsEmbeds, mHostContextLengths, deepstackEmbeds, mOutputLogits, outputHiddenStates, stream);
    if (!prefillStatus)
    {
        LOG_ERROR("LLMInferenceRuntime(): Failed to execute prefill step.");
        return false;
    }

    // Copy out the KVCache content from the prefill step.
    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numAttentionLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU,
        linearKVCache.getConfig().kvCacheTypeTRT, "LLMInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);

    // Save SSM and conv states for Mamba layers
    if (mEngineConfig.numMambaLayers > 0)
    {
        savedKVCache.ssmStateContents = mLLMEngineRunner->getLinearKVCache().captureSSMStates(CACHE_BATCH_IDX, stream);
        savedKVCache.convStateContents
            = mLLMEngineRunner->getLinearKVCache().captureConvStates(CACHE_BATCH_IDX, stream);
    }

    mSystemPromptKVCache.insert({promptKey, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

bool LLMInferenceRuntime::dumpCurrentPrefillKVCache(
    std::string const& outputPath, int32_t sequenceLength, cudaStream_t stream)
{
    check::check(sequenceLength > 0, "sequenceLength must be positive for prefill KV cache dump");

    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto const cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    rt::Tensor kvCacheContent({cacheConfig.numAttentionLayers, 2, cacheConfig.numKVHeads, sequenceLength,
                                  cacheConfig.headDim},
        rt::DeviceType::kGPU, cacheConfig.kvCacheTypeTRT, "kv_cache");
    kernel::saveKVCacheIntoTensor(kvCacheContent, kvCacheBuffer, /*batchIdx=*/0, stream);

    rt::Tensor sequenceLengthTensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "sequence_length");
    sequenceLengthTensor.dataPointer<int32_t>()[0] = sequenceLength;

    rt::Tensor inputIdsTensor({sequenceLength}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "input_ids");
    CUDA_CHECK(cudaMemcpyAsync(inputIdsTensor.rawPointer(), mInputIds.rawPointer(), sequenceLength * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));

    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(kvCacheContent));
    tensors.push_back(std::move(sequenceLengthTensor));
    tensors.push_back(std::move(inputIdsTensor));

    if (!safetensors::saveSafetensors(outputPath, tensors, stream))
    {
        LOG_ERROR("LLMInferenceRuntime(): Failed to save safetensors to %s", outputPath.c_str());
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("LLMInferenceRuntime(): Prefill KV cache dumped to %s", outputPath.c_str());
    return true;
}

} // namespace rt
} // namespace trt_edgellm
