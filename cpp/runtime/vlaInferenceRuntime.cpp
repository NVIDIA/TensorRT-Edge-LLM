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

#include "vlaInferenceRuntime.h"

#include <algorithm>

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <chrono>
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

// generateMultimodalIndices is provided by runtime/llmRuntimeUtils.{h,cpp} (rt namespace).

} // namespace
namespace rt
{
VlaInferenceRuntime::VlaInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
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
    mEmbedding = loadEmbeddingTable(embeddingPath, stream);

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
            "VlaInferenceRuntime::mSamplingWorkspace");
        mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32, "VlaInferenceRuntime::mInputIds");
        mInputsEmbeds = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
            rt::DeviceType::kGPU, DataType::kHALF, "VlaInferenceRuntime::mInputsEmbeds");
        // Allocate deepstack embeddings if needed (one tensor per feature)
        if (mEngineConfig.numDeepstackFeatures > 0)
        {
            mDeepstackEmbeds.resize(mEngineConfig.numDeepstackFeatures);
            for (int32_t i = 0; i < mEngineConfig.numDeepstackFeatures; ++i)
            {
                mDeepstackEmbeds[i] = rt::Tensor({mEngineConfig.maxSupportedBatchSize,
                                                     mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
                    rt::DeviceType::kGPU, DataType::kHALF,
                    format::fmtstr("VlaInferenceRuntime::mDeepstackEmbeds[%d]", i));
            }
            LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]",
                mEngineConfig.numDeepstackFeatures, mEngineConfig.maxSupportedBatchSize,
                mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize);
        }
        mHostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kCPU, DataType::kINT32, "VlaInferenceRuntime::mHostPackedInputIds");
        mOutputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize},
            rt::DeviceType::kGPU, DataType::kFLOAT, "VlaInferenceRuntime::mOutputLogits");
        if (mEngineConfig.enableContextEmb || mEngineConfig.enableLmHiddenStates)
        {
            int32_t const seqOutputDim
                = mEngineConfig.enableContextEmb ? mEngineConfig.contextEmbDim : mEngineConfig.hiddenSize;
            mOutputContextEmbeds = rt::Tensor({mEngineConfig.maxSupportedBatchSize,
                                                  mEngineConfig.maxSupportedInputLength, seqOutputDim},
                rt::DeviceType::kGPU, DataType::kHALF, "VlaInferenceRuntime::mOutputContextEmbeds");
        }
        if (mEngineConfig.enablePrefixKVOutputs)
        {
            check::check(!mEngineConfig.prefixKVOutputShape.empty(), "prefixKVOutputShape must be populated");
            mOutputPrefixK = rt::Tensor(rt::Coords(mEngineConfig.prefixKVOutputShape), rt::DeviceType::kGPU,
                DataType::kHALF, "VlaInferenceRuntime::mOutputPrefixK");
            mOutputPrefixV = rt::Tensor(rt::Coords(mEngineConfig.prefixKVOutputShape), rt::DeviceType::kGPU,
                DataType::kHALF, "VlaInferenceRuntime::mOutputPrefixV");
        }
        mSelectedIndices = rt::Tensor({mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32,
            "VlaInferenceRuntime::mSelectedIndices");
        mHostSelectedTokenIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "VlaInferenceRuntime::mHostSelectedTokenIds");
        mHostContextLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "VlaInferenceRuntime::mHostContextLengths");
        mHostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "VlaInferenceRuntime::mHostReuseKVCacheLengths");
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
    mUseCompactPrefixPadding = mTokenizer->getPrefixStrategy() == "pi05_compact_prefix"
        || mTokenizer->getPrefixStrategy() == "smolvla_compact_prefix";

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
        //   multimodalEngineDir/audio/   - Audio encoder (audio_encoder.engine, config.json)
        //   multimodalEngineDir/visual/  - Visual encoder (visual.engine, config.json)
        //   multimodalEngineDir/action/  - Action expert (action.engine)
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

        try
        {
            std::string const actionCtxDir = multimodalEngineDir + "/action_context";
            LOG_INFO("Attempting to load action context runner from %s", actionCtxDir.c_str());
            mActionContextRunner = std::make_unique<ActionContextRunner>(actionCtxDir, stream);
            LOG_INFO("Action context runner loaded from %s", actionCtxDir.c_str());
        }
        catch (std::exception const& e)
        {
            LOG_INFO("Failed to load action context runner from %s: %s",
                (multimodalEngineDir + "/action_context").c_str(), e.what());
        }

        // Try to load action expert from multimodalEngineDir/action
        try
        {
            std::string const actionDir = multimodalEngineDir + "/action";
            LOG_INFO("Attempting to load Action runner from %s", actionDir.c_str());
            mActionRunner = std::make_unique<ActionRunner>(
                actionDir, stream, mLLMEngineRunner->getLinearKVCache().getConfig());
            LOG_INFO("Action runner loaded (handoff=%s, rollout=%s).",
                mActionRunner->getContextHandoff() == ActionContextHandoff::PREFIX_KV ? "prefix_kv" : "context_tensor",
                mActionRunner->getRolloutMode() == ActionRolloutMode::FLOW_MATCHING ? "flow_matching" : "velocity");
        }
        catch (std::exception const& e)
        {
            LOG_INFO("Failed to load Action runner from %s: %s", (multimodalEngineDir + "/action").c_str(), e.what());
        }

        // Prefix-KV action engines must match the LM KV cache capacity.
        if (mActionRunner && mActionRunner->getContextHandoff() == ActionContextHandoff::PREFIX_KV)
        {
            int32_t const actionMaxKVCacheCapacity = mActionRunner->getMaxKVCacheCapacity();
            int32_t const llmMaxKVCacheCapacity = mEngineConfig.maxKVCacheCapacity;
            if (actionMaxKVCacheCapacity != llmMaxKVCacheCapacity)
            {
                throw std::runtime_error(format::fmtstr(
                    "Action engine max_kv_cache_capacity (%d) does not match LLM engine max_kv_cache_capacity (%d). "
                    "Re-export and rebuild the action engine with --max_kv_cache_capacity=%d to match the LLM engine.",
                    actionMaxKVCacheCapacity, llmMaxKVCacheCapacity, llmMaxKVCacheCapacity));
            }
        }
    }

    // Setup shared execution context memory for LLM and multimodal engines.
    // All engines execute serially (not concurrently), so they can share a single buffer
    // sized to the maximum requirement among all engines.
    int64_t const llmContextMemorySize = mLLMEngineRunner->getRequiredContextMemorySize();
    int64_t const visionContextMemorySize = mVisionRunner ? mVisionRunner->getRequiredContextMemorySize() : 0;
    int64_t const audioContextMemorySize = mAudioRunner ? mAudioRunner->getRequiredContextMemorySize() : 0;
    int64_t const actionContextRunnerMemorySize
        = mActionContextRunner ? mActionContextRunner->getRequiredContextMemorySize() : 0;
    int64_t const actionContextMemorySize = mActionRunner ? mActionRunner->getRequiredContextMemorySize() : 0;
    int64_t const sharedContextMemorySize = std::max(
        {llmContextMemorySize, visionContextMemorySize, audioContextMemorySize, actionContextMemorySize});
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "VlaInferenceRuntime::mSharedExecContextMemory");
    mLLMEngineRunner->setContextMemory(mSharedExecContextMemory);
    if (mVisionRunner)
    {
        mVisionRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mAudioRunner)
    {
        mAudioRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mActionContextRunner)
    {
        mActionContextRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mActionRunner)
    {
        mActionRunner->setContextMemory(mSharedExecContextMemory);
    }
    LOG_INFO(
        "Setup shared execution context memory: %zu bytes (llm requires: %zu, vision requires: %zu, audio "
        "requires: %zu, action_context requires: %zu, action requires: %zu)",
        static_cast<size_t>(sharedContextMemorySize), static_cast<size_t>(llmContextMemorySize),
        static_cast<size_t>(visionContextMemorySize), static_cast<size_t>(audioContextMemorySize),
        static_cast<size_t>(actionContextRunnerMemorySize), static_cast<size_t>(actionContextMemorySize));
}

void VlaInferenceRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

bool VlaInferenceRuntime::examineRequest(LLMGenerationRequest const& request) noexcept
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("VlaInferenceRuntime(): The request is empty with no requests supplied.");
        return false;
    }

    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("VlaInferenceRuntime(): The batched request size (%d) exceeds the max supported batch size (%d).",
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

bool VlaInferenceRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
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
                    "VlaInferenceRuntime(): Though system prompt strings are matched, token_ids are not perfectly "
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
            "VlaInferenceRuntime(): The max input length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    // Pad each batch to engine max length for static-shape VLA engines (pi05_compact_prefix),
    // otherwise pad only to the batch max valid length.
    int32_t const packedInputLength = mUseCompactPrefixPadding ? mEngineConfig.maxSupportedInputLength : maxInputLength;
    if (maxInputLength > packedInputLength)
    {
        LOG_ERROR(
            "VlaInferenceRuntime(): The max input length (%d) exceeds the packed input length (%d) of the LLM "
            "Engine.",
            maxInputLength, packedInputLength);
        return false;
    }
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

    // For each recurrent layer and each batch element, either restore the cached
    // recurrent/conv state if the cache hit, or zero the state.
    if (mEngineConfig.numLinearAttnLayers > 0)
    {
        rt::LinearKVCache& kvCache = mLLMEngineRunner->getLinearKVCache();
        rt::LinearKVCache::CacheConfig const& cacheConfig = kvCache.getConfig();
        size_t const recurrentElemSize = rt::utils::getTypeSize(cacheConfig.recurrentStateType);
        size_t const convElemSize = rt::utils::getTypeSize(cacheConfig.convStateType);
        size_t const recurrentBatchBytes = static_cast<size_t>(cacheConfig.recurrentStateNumHeads
                                               * cacheConfig.recurrentStateHeadDim * cacheConfig.recurrentStateSize)
            * recurrentElemSize;
        size_t const convBatchBytes = static_cast<size_t>(cacheConfig.convDim * cacheConfig.convKernel) * convElemSize;

        for (int32_t layer = 0; layer < mEngineConfig.numLinearAttnLayers; ++layer)
        {
            rt::Tensor recurrentLayer = kvCache.getRecurrentStateForLayer(layer);
            rt::Tensor convLayer = kvCache.getConvStateForLayer(layer);

            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + i * recurrentBatchBytes;
                auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + i * convBatchBytes;

                auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
                auto it = mSystemPromptKVCache.find(promptKey);
                bool const hasCache = (it != mSystemPromptKVCache.end());

                if (hasCache && layer < static_cast<int32_t>(it->second.recurrentStateContents.size()))
                {
                    CUDA_CHECK(cudaMemcpyAsync(recurrentDst, it->second.recurrentStateContents[layer].rawPointer(),
                        recurrentBatchBytes, cudaMemcpyDeviceToDevice, stream));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
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
    if (mEngineConfig.hasLogitsOutput)
    {
        check::check(mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize}), "Tensor reshape failed");
    }
    if (mEngineConfig.enableContextEmb || mEngineConfig.enableLmHiddenStates)
    {
        int32_t const seqOutputDim
            = mEngineConfig.enableContextEmb ? mEngineConfig.contextEmbDim : mEngineConfig.hiddenSize;
        check::check(
            mOutputContextEmbeds.reshape({activeBatchSize, packedInputLength, seqOutputDim}), "Tensor reshape failed");
    }
    if (mEngineConfig.enablePrefixKVOutputs)
    {
        rt::Coords prefixShape(mEngineConfig.prefixKVOutputShape);
        if (prefixShape.getNumDims() > 0)
        {
            prefixShape[1] = activeBatchSize;
        }
        check::check(mOutputPrefixK.reshape(prefixShape), "Tensor reshape failed");
        check::check(mOutputPrefixV.reshape(prefixShape), "Tensor reshape failed");
    }

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

bool VlaInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    auto const e2eStart = std::chrono::steady_clock::now();

    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;
    std::string loraWeightsName = request.loraWeightsName;

    if (!examineRequest(request))
    {
        LOG_ERROR("VlaInferenceRuntime(): Input request examination failed. This request cannot be handled.");
        return false;
    }

    int32_t activeBatchSize = static_cast<int32_t>(request.requests.size());

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
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    int32_t const actionBatchSize
        = (request.actionBatchSize > 0) ? request.actionBatchSize : activeBatchSize;

    bool const runPrefixKvAction = mActionRunner != nullptr
        && mActionRunner->getContextHandoff() == ActionContextHandoff::PREFIX_KV && hasTrajectoryHistory;
    bool const runPi05VelocityAction = mActionRunner != nullptr
        && mActionRunner->getContextHandoff() == ActionContextHandoff::CONTEXT_TENSOR && hasVision
        && mEngineConfig.enablePrefixKVOutputs && mActionContextRunner == nullptr;
    bool const runContextTensorAction = mActionRunner != nullptr
        && mActionRunner->getContextHandoff() == ActionContextHandoff::CONTEXT_TENSOR && hasVision && !runPi05VelocityAction
        && ((mEngineConfig.enableContextEmb && mActionContextRunner == nullptr)
            || (mActionContextRunner != nullptr && mEngineConfig.enableLmHiddenStates));

    if ((hasAudio && mAudioRunner) || (hasVision && mVisionRunner))
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
                LOG_ERROR("VlaInferenceRuntime(): Audio preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mAudioRunner->infer(stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): Audio inference failed. This request cannot be handled.");
                return false;
            }
        }

        // Process vision inputs (if present)
        if (hasVision && mVisionRunner)
        {
            LOG_INFO("Processing vision inputs");
            auto const vitStart = std::chrono::steady_clock::now();
            if (!mVisionRunner->preprocess(
                    request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): Vision preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mVisionRunner->infer(stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): Vision inference failed. This request cannot be handled.");
                return false;
            }
            auto const vitEnd = std::chrono::steady_clock::now();
            double const vitMs = std::chrono::duration<double, std::milli>(vitEnd - vitStart).count();
            LOG_INFO("Stage timings - ViT: %.3f ms", vitMs);
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

    if (runPrefixKvAction || runContextTensorAction || runPi05VelocityAction)
    {
        LOG_INFO("Preprocessing action inputs (LLM batch=%d, action batch=%d, handoff=%s)", activeBatchSize,
            actionBatchSize,
            runPrefixKvAction ? "prefix_kv" : (runPi05VelocityAction ? "pi05_prefix_kv" : "context_tensor"));
        if (!mActionRunner->preprocess(request, batchedInputIds, mTokenizer.get()))
        {
            LOG_ERROR("VlaInferenceRuntime(): Action preprocessing failed. This request cannot be handled.");
            return false;
        }
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(batchedInputIds, batchSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR("VlaInferenceRuntime(): Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // Record context information for performance tracking
    auto tokenCount = calculateTokenCounts(batchedInputIds, batchSystemPrompts, loraWeightsName);

    int32_t const* contextLengthsData = mHostContextLengths.dataPointer<int32_t>();
    int32_t actualContextLength{0};
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        actualContextLength = std::max(actualContextLength, contextLengthsData[i]);
    }
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (actualContextLength + maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
    {
        maxGenerationLength = mEngineConfig.maxKVCacheCapacity - actualContextLength;
        LOG_WARNING(
            "The requested input length (%d) + max generation length (%d) = %d exceeds the max KV "
            "cache capacity (%d). Reduce the generation length to %d to avoid the truncation of the generated tokens.",
            actualContextLength, request.maxGenerateLength, actualContextLength + request.maxGenerateLength,
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

    // Used for prefix-KV trajectory models (e.g. Alpamayo): stop decode after traj marker token.
    int32_t trajFutureStartId = 0;
    if (runPrefixKvAction)
    {
        trajFutureStartId = static_cast<int32_t>(mTokenizer->getTokenId("<|traj_future_start|>"));
    }
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

                // Prefix-KV trajectory models stop one token after traj_future_start so KV includes the marker.
                if (runPrefixKvAction)
                {
                    if (outputIds[i].size() > 1 && trajFutureStartId >= 0)
                    {
                        finishedStates[i] = outputIds[i][outputIds[i].size() - 2] == trajFutureStartId;
                    }
                }
                else
                {
                    finishedStates[i] = hostSelectedTokenIdsData[i] == mTokenizer->getEosId();
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

        kernel::embeddingLookupMultimodal(mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(),
            std::optional{std::ref(mMultimodalIndices)}, imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings,
            mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Legacy vision path (Qwen2.5-VL, InternVL: imageTokenId >= vocabSize or not set)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(
            mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(), imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(), mInputsEmbeds, stream);
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
    auto const llmPrefillStart = std::chrono::steady_clock::now();
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);
        // Enhanced NVTX range with detailed information
        NVTX_SCOPED_RANGE(nvtx_prefill,
            ("LLM_PREFILL[BS=" + std::to_string(activeBatchSize)
                + ",Reused=" + std::to_string(tokenCount.totalReusedTokens)
                + ",Computed=" + std::to_string(tokenCount.totalComputedTokens) + "]")
                .c_str(),
            nvtx_colors::BLUE);

        rt::OptionalOutputTensor outputContextEmbeds{std::nullopt};
        rt::OptionalOutputTensor outputPrefixK{std::nullopt};
        rt::OptionalOutputTensor outputPrefixV{std::nullopt};
        if (mEngineConfig.enablePrefixKVOutputs)
        {
            outputPrefixK = std::ref(mOutputPrefixK);
            outputPrefixV = std::ref(mOutputPrefixV);
        }
        if (mEngineConfig.enableContextEmb || mEngineConfig.enableLmHiddenStates)
        {
            outputContextEmbeds = std::ref(mOutputContextEmbeds);
        }
        bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputsEmbeds, mHostContextLengths, deepstackEmbeds,
            mOutputLogits, rt::OptionalOutputTensor{std::nullopt}, stream, outputContextEmbeds, outputPrefixK,
            outputPrefixV);
        if (!prefillStatus)
        {
            LOG_ERROR(
                "VlaInferenceRuntime(): Failed to execute prefill step. Cannot generate the KVCache for this prompt.");
            return false;
        }

        if (mEngineConfig.hasLogitsOutput)
        {
            sampleTokens();
        }
    }
    auto const llmPrefillEnd = std::chrono::steady_clock::now();
    double const llmPrefillMs = std::chrono::duration<double, std::milli>(llmPrefillEnd - llmPrefillStart).count();
    LOG_INFO("Stage timings - LLM Prefill: %.3f ms", llmPrefillMs);

    // Record prefill metrics
    mPrefillMetrics.recordRun(tokenCount.totalReusedTokens, tokenCount.totalComputedTokens);

    // Reshape for decoding step
    check::check(mInputsEmbeds.reshape({activeBatchSize, 1, mEngineConfig.hiddenSize}), "Tensor reshape failed");

    // Profile entire generation phase like benchmark profiler
    auto const llmGenStart = std::chrono::steady_clock::now();
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
            kernel::embeddingLookup(
                mSelectedIndices, mEmbedding.table, mEmbedding.scalesAsOptional(), mInputsEmbeds, stream);

            // Use the embedded tokens as input for the decoding step.
            // No hidden states output needed for standard LLM decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(
                mInputsEmbeds, mOutputLogits, outputHiddenStates, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("VlaInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            sampleTokens();
        }
    }
    auto const llmGenEnd = std::chrono::steady_clock::now();
    double const llmGenMs = std::chrono::duration<double, std::milli>(llmGenEnd - llmGenStart).count();
    LOG_INFO("Stage timings - LLM Generation: %.3f ms", llmGenMs);

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

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();
    response.outputTrajectories.clear();
    response.outputActions.clear();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
    }
    if (actionBatchSize > activeBatchSize && !response.outputIds.empty())
    {
        std::vector<int32_t> const llmOutputIds = response.outputIds[0];
        std::string const llmOutputText = response.outputTexts[0];
        response.outputIds.assign(actionBatchSize, llmOutputIds);
        response.outputTexts.assign(actionBatchSize, llmOutputText);
        activeBatchSize = actionBatchSize;
    }

    if (runPrefixKvAction)
    {
        if (!mVisionRunner)
        {
            LOG_ERROR("Prefix-KV action runner requires a vision runner for MRoPE rope deltas.");
            return false;
        }

        multimodal::ModelType const visionType = mVisionRunner->getModelType();
        bool const isQwen3ViT = visionType == multimodal::ModelType::QWEN3_VL;
        if (!isQwen3ViT)
        {
            LOG_ERROR("Prefix-KV action runner requires a Qwen3-VL vision runner but a different vision runner is loaded.");
            return false;
        }

        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltasFromVit = qwenVision->getMropeRopeDeltasPerBatch();
        std::vector<int64_t> ropeDeltasBroadcast;
        std::vector<int64_t> const* ropeDeltasPtr = &ropeDeltasFromVit;
        if (static_cast<int32_t>(ropeDeltasFromVit.size()) < activeBatchSize)
        {
            int64_t const ropeDelta0 = ropeDeltasFromVit.empty() ? int64_t{0} : ropeDeltasFromVit[0];
            ropeDeltasBroadcast.assign(activeBatchSize, ropeDelta0);
            ropeDeltasPtr = &ropeDeltasBroadcast;
        }
        std::vector<int64_t> const& ropeDeltas = *ropeDeltasPtr;
        response.outputTrajectories.resize(activeBatchSize);
        rt::LinearKVCache& kvcache = mLLMEngineRunner->getLinearKVCache();
        auto const diffusorStart = std::chrono::steady_clock::now();
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories
            = mActionRunner->sampleTrajectory(stream, activeBatchSize, kvcache, ropeDeltas);
        auto const diffusorEnd = std::chrono::steady_clock::now();
        double const diffusorMs = std::chrono::duration<double, std::milli>(diffusorEnd - diffusorStart).count();
        LOG_INFO("Stage timings - Diffusor: %.3f ms", diffusorMs);
        if (trajectories.size() != static_cast<size_t>(activeBatchSize))
        {
            LOG_ERROR("VlaInferenceRuntime(): prefix-KV action sampling failed.");
            return false;
        }
        for (size_t i = 0; i < trajectories.size() && i < static_cast<size_t>(activeBatchSize); ++i)
        {
            if (!trajectories[i].empty())
            {
                response.outputTrajectories[i] = std::move(trajectories[i]);
            }
        }
    }
    else if (runPi05VelocityAction)
    {
        auto const diffusorStart = std::chrono::steady_clock::now();
        if (!mActionRunner->wireStaticInputs(request, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to wire static action inputs.");
            return false;
        }

        int32_t prefixValidLen{0};
        int32_t const* contextLengthsData = mHostContextLengths.dataPointer<int32_t>();
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            prefixValidLen = std::max(prefixValidLen, contextLengthsData[i]);
        }
        if (!mActionRunner->preparePi05SuffixInputs(activeBatchSize, prefixValidLen, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to prepare PI0.5 suffix action inputs.");
            return false;
        }

        std::vector<rt::Tensor const*> languageOutputs(3, nullptr);
        languageOutputs[0] = &mOutputContextEmbeds;
        languageOutputs[1] = &mOutputPrefixK;
        languageOutputs[2] = &mOutputPrefixV;
        if (!mActionRunner->wireLanguageOutputs(languageOutputs, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to wire PI0.5 prefix_k/prefix_v into action runner.");
            return false;
        }
        if (!mActionRunner->sampleActions(stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): PI0.5 action sampling failed.");
            return false;
        }
        if (!mActionRunner->copyActionsToHost(response.outputActions, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to copy denoised actions to host.");
            return false;
        }
        auto const diffusorEnd = std::chrono::steady_clock::now();
        double const diffusorMs = std::chrono::duration<double, std::milli>(diffusorEnd - diffusorStart).count();
        LOG_INFO("Stage timings - Diffusor: %.3f ms", diffusorMs);
    }
    else if (runContextTensorAction)
    {
        auto const diffusorStart = std::chrono::steady_clock::now();
        if (!mActionRunner->wireStaticInputs(request, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to wire static action inputs (state/embodiment).");
            return false;
        }
        if (mActionContextRunner)
        {
            int32_t const actionContextSeqLen = mActionContextRunner->getMaxSeqLen();
            if (!mActionContextRunner->reshapeForContext(activeBatchSize, actionContextSeqLen))
            {
                LOG_ERROR("VlaInferenceRuntime(): failed to reshape action context runner.");
                return false;
            }
            if (!mActionContextRunner->copyLmHiddenFrom(mOutputContextEmbeds, stream, actualContextLength))
            {
                LOG_ERROR("VlaInferenceRuntime(): failed to copy lm_hidden_states into action context runner.");
                return false;
            }
            if (!mActionContextRunner->resetExecutionContext(stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): failed to reset action context execution context.");
                return false;
            }
            if (!mActionContextRunner->infer(stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): action context inference failed.");
                return false;
            }
            if (!mActionRunner->copyInputFrom("context_embs", mActionContextRunner->getVlEmbs(), stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): failed to wire vl_embs into action runner.");
                return false;
            }
        }
        else
        {
            std::vector<rt::Tensor const*> languageOutputs;
            languageOutputs.push_back(&mOutputLogits);
            languageOutputs.push_back(&mOutputContextEmbeds);
            if (!mActionRunner->wireLanguageOutputs(languageOutputs, stream))
            {
                LOG_ERROR("VlaInferenceRuntime(): failed to wire language context_embs into action runner.");
                return false;
            }
        }
        if (!mActionRunner->resetExecutionContext(stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to reset action runner execution context.");
            return false;
        }
        if (!mActionRunner->sampleActions(stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): context-tensor action sampling failed.");
            return false;
        }
        if (!mActionRunner->copyActionsToHost(response.outputActions, stream))
        {
            LOG_ERROR("VlaInferenceRuntime(): failed to copy denoised actions to host.");
            return false;
        }
        auto const diffusorEnd = std::chrono::steady_clock::now();
        double const diffusorMs = std::chrono::duration<double, std::milli>(diffusorEnd - diffusorStart).count();
        LOG_INFO("Stage timings - Diffusor: %.3f ms", diffusorMs);
    }

    auto const e2eEnd = std::chrono::steady_clock::now();
    double const e2eMs = std::chrono::duration<double, std::milli>(e2eEnd - e2eStart).count();
    LOG_INFO("Stage timings - E2E: %.3f ms", e2eMs);

    return true;
}

bool VlaInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
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
            "VlaInferenceRuntime(): Successfully captured the decoding CUDA graph for all execution batch sizes and "
            "LoRA weights.");
    }
    else
    {
        LOG_WARNING(
            "VlaInferenceRuntime(): Failed to capture the decoding CUDA graph for some of execution batch sizes and "
            "LoRA weights.");
    }
    return captureStatus;
}

VlaInferenceRuntime::TokenCountInfo VlaInferenceRuntime::calculateTokenCounts(
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

bool VlaInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("VlaInferenceRuntime(): The prompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    // hash the prompt if check if the prompt cache already exists.
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
    {
        LOG_DEBUG(
            "VlaInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
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
            "VlaInferenceRuntime(): The prompt length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            promptIdsLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(activeBatchSize, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(activeBatchSize, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR(
            "VlaInferenceRuntime(): Prefill execution setup failed. Cannot generate the KVCache for this prompt.");
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

        kernel::embeddingLookupMultimodal(mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(),
            std::optional{std::ref(mMultimodalIndices)}, imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings,
            mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Vision-only (Qwen2-VL, InternVL, etc.)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(
            mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(), imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbedding.table, mEmbedding.scalesAsOptional(), mInputsEmbeds, stream);
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
    rt::OptionalOutputTensor outputContextEmbeds{std::nullopt};
    if (mEngineConfig.enableContextEmb || mEngineConfig.enableLmHiddenStates)
    {
        outputContextEmbeds = std::ref(mOutputContextEmbeds);
    }
    bool prefillStatus = mLLMEngineRunner->executePrefillStep(
        mInputsEmbeds, mHostContextLengths, deepstackEmbeds, mOutputLogits, outputHiddenStates, stream,
        outputContextEmbeds);
    if (!prefillStatus)
    {
        LOG_ERROR("VlaInferenceRuntime(): Failed to execute prefill step.");
        return false;
    }

    // Copy out the KVCache content from the prefill step.
    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numAttentionLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    VlaSystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU,
        linearKVCache.getConfig().kvCacheTypeTRT, "VlaInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);

    // Save recurrent and conv states for hybrid layers
    if (mEngineConfig.numLinearAttnLayers > 0)
    {
        savedKVCache.recurrentStateContents
            = mLLMEngineRunner->getLinearKVCache().captureRecurrentStates(CACHE_BATCH_IDX, stream);
        savedKVCache.convStateContents
            = mLLMEngineRunner->getLinearKVCache().captureConvStates(CACHE_BATCH_IDX, stream);
    }

    mSystemPromptKVCache.insert({promptKey, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("VlaInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

} // namespace rt
} // namespace trt_edgellm
