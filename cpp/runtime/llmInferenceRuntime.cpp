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

#include "llmInferenceRuntime.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <fstream>
#include <functional>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{

// Left a utility function here in case we want to move to a better hashing method.
size_t hashSystemPromptWithLoraWeights(std::string const& systemPrompt, std::string const& loraWeightsName)
{
    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, systemPrompt);
    hash_utils::hashCombine(hashValue, loraWeightsName);
    return hashValue;
}

} // namespace
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

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

    // Optional: Setup multimodal engine runner
    if (!multimodalEngineDir.empty())
    {
        try
        {
            mMultimodalRunner = MultimodalRunner::create(
                multimodalEngineDir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize MultimodalRunner: %s", e.what());
            throw std::runtime_error("Failed to initialize MultimodalRunner: " + std::string(e.what()));
        }
        LOG_INFO("MultimodalRunner successfully loaded and initialized multimodal engine.");
    }
}

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request)
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
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    // Record the length of the reused KVCache for each sequence using pre-allocated tensor
    mHostReuseKVCacheLengths.reshape({activeBatchSize});
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();

    // Search if the system prompt has been cached. If there are cached system prompts, insert
    // the pre-computed KVCache and remove the contents from inputIds.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptHash];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
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

    // The LLM Engine could also have minSupportedInputLength constraint.
    int32_t const packedInputLength = std::max(maxInputLength, mEngineConfig.minSupportedInputLength);

    // Reshape and fill the pre-allocated pinned host tensor with pad tokens
    mHostPackedInputIds.reshape({activeBatchSize, packedInputLength});
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Pad each sequence to the max length of this batch.
        // TODO: Implement remove input padding for better efficiency until multi-batch.
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    mInputIds.reshape({activeBatchSize, packedInputLength});
    mHostContextLengths.reshape({activeBatchSize});
    mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize});

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName, stream))
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
            if (mMultimodalRunner)
            {
                mMultimodalRunner->preprocessSystemPrompt(
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
    if (!mMultimodalRunner)
    {
        batchedInputIds.reserve(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, true));
        }
    }
    else
    {
        if (!mMultimodalRunner->preprocess(
                request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("LLMInferenceRuntime(): Multimodal inference failed. This request cannot be handled.");
            return false;
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
    mSelectedIndices.reshape({activeBatchSize, 1});
    mHostSelectedTokenIds.reshape({activeBatchSize});
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
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
    };

    // Use empty tensor for when no multimodal runner is available.
    // All other data input used by prefill step is already set up in setUpForPrefillExecution().
    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};

    // Profile all sampling operations as one stage
    // Prefill profiling session
    // For non-spec decode, we don't need to output hidden states.
    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);

        bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings,
            extraVisualFeatures, mOutputLogits, outputHiddenStates, stream);
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

    // Reshape inputIds for decoding step
    mInputIds.reshape({activeBatchSize, 1});

    // Profile entire generation phase like benchmark profiler
    {
        TIME_STAGE(metrics::StageNames::kLLM_GENERATION, stream);

        while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
        {
            // Use the selected token indices as the input token indices for the decoding step.
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(mSelectedIndices, mOutputLogits, stream);
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
        mSelectedIndices.reshape({batchSize, 1});
        mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize});
        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
            mSelectedIndices, mOutputLogits, mEmptyLoraWeightsName, stream);
        if (mEngineConfig.maxSupportedLoraRank > 0)
        {
            for (auto const& loraWeightsName : mLLMEngineRunner->getAvailableLoraWeights())
            {
                captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
                    mSelectedIndices, mOutputLogits, loraWeightsName, stream);
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
    std::string const& loraWeightsName) const
{
    TokenCountInfo tokenCount;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t contextLength = static_cast<int32_t>(batchedInputIds[i].size());
        // Calculate reused length from system prompt cache
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            int32_t reusedLength = static_cast<int32_t>(mSystemPromptKVCache.at(promptHash).tokenizedPrompt.size());
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
    size_t const promptHash = hashSystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
    {
        LOG_DEBUG(
            "LLMInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
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
    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};

    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings,
        extraVisualFeatures, mOutputLogits, outputHiddenStates, stream);
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
        cacheConfig.numDecoderLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT,
        "LLMInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);
    mSystemPromptKVCache.insert({promptHash, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

} // namespace rt
} // namespace trt_edgellm