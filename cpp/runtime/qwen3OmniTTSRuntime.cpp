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

#include "qwen3OmniTTSRuntime.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "llmInferenceSpecDecodeRuntime.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;
using namespace talker_constants;

namespace
{
// Helper: Extract MLP weights from safetensors (eliminates code duplication)
bool extractMLPWeightsFromTensors(std::vector<rt::Tensor>& tensors, rt::Tensor& fc1Weight, rt::Tensor& fc1Bias,
    rt::Tensor& fc2Weight, rt::Tensor& fc2Bias, std::string const& projectionName)
{
    constexpr int32_t kExpectedTensorCount = 4;
    check::check(
        tensors.size() == kExpectedTensorCount, projectionName + ".safetensors should contain exactly 4 tensors");

    bool foundFC1Weight = false, foundFC1Bias = false, foundFC2Weight = false, foundFC2Bias = false;

    for (auto& tensor : tensors)
    {
        std::string const& name = tensor.getName();
        if (name.find("fc1.weight") != std::string::npos)
        {
            fc1Weight = std::move(tensor);
            foundFC1Weight = true;
        }
        else if (name.find("fc1.bias") != std::string::npos)
        {
            fc1Bias = std::move(tensor);
            foundFC1Bias = true;
        }
        else if (name.find("fc2.weight") != std::string::npos)
        {
            fc2Weight = std::move(tensor);
            foundFC2Weight = true;
        }
        else if (name.find("fc2.bias") != std::string::npos)
        {
            fc2Bias = std::move(tensor);
            foundFC2Bias = true;
        }
    }

    if (!foundFC1Weight || !foundFC1Bias || !foundFC2Weight || !foundFC2Bias)
    {
        LOG_ERROR("Failed to find all required tensors in %s.safetensors", projectionName.c_str());
        return false;
    }

    return true;
}
} // anonymous namespace

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream)
    : mStream(stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::init", nvtx_colors::YELLOW);
    LOG_INFO("Initializing Qwen3-Omni Talker runner");
    LOG_INFO("  Talker: %s", talkerEngineDir.c_str());
    LOG_INFO("  CodePredictor: %s", codePredictorEngineDir.c_str());

    // Load tokenizer
    std::filesystem::path const tokenizerPath = tokenizerDir.empty()
        ? std::filesystem::path(talkerEngineDir).parent_path()
        : std::filesystem::path(tokenizerDir);
    LOG_INFO("  Tokenizer: %s", tokenizerPath.string().c_str());
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    bool const tokenizerLoaded = mTokenizer->loadFromHF(tokenizerPath);
    ELLM_CHECK(tokenizerLoaded, "Failed to load tokenizer from: " + tokenizerPath.string());

    bool const configValid = validateAndFillConfig(talkerEngineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    bool const runnersInitialized = initializeEngineRunners(talkerEngineDir, codePredictorEngineDir);
    ELLM_CHECK(runnersInitialized, "Failed to initialize engine runners");

    // Setup shared execution context memory for Talker and CodePredictor engines.
    // LLMEngineRunner uses kUSER_MANAGED allocation and requires setContextMemory() before execution.
    {
        int64_t const talkerCtxSize = mTalkerLLMRunner->getRequiredContextMemorySize();
        int64_t const cpCtxSize = mCodePredictorRunner ? mCodePredictorRunner->getRequiredContextMemorySize() : 0;
        int64_t const sharedCtxSize = std::max(talkerCtxSize, cpCtxSize);
        LOG_INFO("Setup shared execution context memory: %zu bytes (talker: %zu, code_predictor: %zu)",
            static_cast<size_t>(sharedCtxSize), static_cast<size_t>(talkerCtxSize), static_cast<size_t>(cpCtxSize));
        mSharedExecContextMemory = rt::Tensor({sharedCtxSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
            "Qwen3OmniTTSRuntime::mSharedExecContextMemory");
        bool const talkerCtxSet = mTalkerLLMRunner->setContextMemory(mSharedExecContextMemory);
        ELLM_CHECK(talkerCtxSet, "Failed to set context memory for Talker LLM engine");
        ELLM_CHECK(!mCodePredictorRunner || mCodePredictorRunner->setContextMemory(mSharedExecContextMemory),
            "Failed to set context memory for CodePredictor engine");
    }

    // Determine max batch size from engine configs (use the minimum of Talker and CodePredictor)
    mMaxBatchSize = std::min(mTalkerLLMConfig.maxSupportedBatchSize, mCodePredictorConfig.maxSupportedBatchSize);
    check::check(mMaxBatchSize >= 1, "maxBatchSize must be >= 1");
    LOG_INFO("Max batch size: %d (Talker=%d, CodePredictor=%d)", mMaxBatchSize, mTalkerLLMConfig.maxSupportedBatchSize,
        mCodePredictorConfig.maxSupportedBatchSize);

    bool const codePredictorWeightsLoaded = loadCodePredictorWeights(codePredictorEngineDir);
    ELLM_CHECK(codePredictorWeightsLoaded, "Failed to load CodePredictor weights");

#ifdef CUTE_DSL_GEMM_ENABLED
    bool const cuteDslLoaded = CuteDslGemmRunner::loadKernelModule();
    ELLM_CHECK(cuteDslLoaded, "Failed to load CuTe DSL GEMM kernel module");
#endif

    bool const bufferAllocated = allocateBuffer();
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    bool const talkerWeightsLoaded = loadTalkerWeights(talkerEngineDir, stream);
    ELLM_CHECK(talkerWeightsLoaded, "Failed to load Talker weights");

    // Load text embedding table (thinker vocab).
    // Used for standalone TTS and for projecting TTS special tokens.
    // TTS: text_embedding.safetensors in talkerEngineDir (copied by builder).
    // Omni: use thinker's embedding.safetensors from tokenizerPath instead.
    {
        std::filesystem::path const textEmbedPath = mIsOmni
            ? tokenizerPath / "embedding.safetensors"
            : std::filesystem::path(talkerEngineDir) / "text_embedding.safetensors";
        LOG_INFO("Loading text embedding from: %s", textEmbedPath.string().c_str());
        std::vector<rt::Tensor> textEmbedTensors;
        bool const textEmbedLoaded = safetensors::loadSafetensors(textEmbedPath, textEmbedTensors, stream);
        ELLM_CHECK(textEmbedLoaded, "Failed to load text embedding from: " + textEmbedPath.string());
        check::check(!textEmbedTensors.empty(), "text embedding file is empty");
        check::check(textEmbedTensors[0].getShape().getNumDims() == 2,
            "text embedding tensor should be 2D [vocabSize, hiddenSize]");
        mTextEmbeddingTable = std::move(textEmbedTensors[0]);
        LOG_INFO("Text embedding table loaded: [%lld, %lld]", mTextEmbeddingTable.getShape()[0],
            mTextEmbeddingTable.getShape()[1]);
    }

    // Note: mTalkerEmbeddingTable is loaded by loadTalkerWeights() above.

    // Load CodePredictor embedding tables from codec_embeddings.safetensors
    // mNumRvqLayers is already set from config.json num_code_groups in validateAndFillConfig()
    {
        std::filesystem::path const embedPath
            = std::filesystem::path(codePredictorEngineDir) / "codec_embeddings.safetensors";
        std::vector<rt::Tensor> allEmbedTensors;
        bool const codecEmbedLoaded = safetensors::loadSafetensors(embedPath, allEmbedTensors, stream);
        ELLM_CHECK(codecEmbedLoaded, "Failed to load codec_embeddings.safetensors from: " + embedPath.string());
        check::check(static_cast<int32_t>(allEmbedTensors.size()) == mNumRvqLayers,
            "codec_embeddings.safetensors has " + std::to_string(allEmbedTensors.size()) + " tensors, expected "
                + std::to_string(mNumRvqLayers) + " (num_code_groups - 1)");
        mCodePredictorEmbeddingTables.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const key = "embedding_" + std::to_string(i);
            auto it = std::find_if(allEmbedTensors.begin(), allEmbedTensors.end(),
                [&key](rt::Tensor const& t) { return t.getName() == key; });
            check::check(it != allEmbedTensors.end(), "Missing key '" + key + "' in codec_embeddings.safetensors");
            check::check(it->getShape().getNumDims() == 2, key + " should be 2D [codebookSize, hiddenSize]");
            mCodePredictorEmbeddingTables[i] = std::move(*it);
        }
    }
    LOG_INFO("Loaded %d CodePredictor embedding tables (from config num_code_groups=%d)", mNumRvqLayers,
        mTalkerConfig.numCodeGroups);

    initializeTTSEmbeddings(stream);

    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaStart, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaEnd, cudaEventDefault));

    LOG_INFO("Qwen3-Omni TTS runtime initialized successfully");
}

Qwen3OmniTTSRuntime::~Qwen3OmniTTSRuntime()
{
#ifdef CUTE_DSL_GEMM_ENABLED
    CuteDslGemmRunner::unloadKernelModule();
#endif
    if (mTtfaStart)
    {
        cudaEventDestroy(mTtfaStart);
    }
    if (mTtfaEnd)
    {
        cudaEventDestroy(mTtfaEnd);
    }
}

bool Qwen3OmniTTSRuntime::initializeEngineRunners(
    std::string const& talkerEngineDir, std::string const& codePredictorEngineDir)
{
    // Load Talker LLM engine
    std::filesystem::path talkerEnginePath = std::filesystem::path(talkerEngineDir) / "llm.engine";
    std::filesystem::path talkerConfigPath = std::filesystem::path(talkerEngineDir) / "config.json";

    LOG_INFO("Loading Talker LLM engine from: %s", talkerEnginePath.string().c_str());

    try
    {
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mTalkerLLMRunner = std::make_unique<LLMEngineRunner>(talkerEnginePath, talkerConfigPath, emptyLoraMap, mStream);
        mTalkerLLMConfig = mTalkerLLMRunner->getEngineConfig();

        LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
            mTalkerLLMConfig.hiddenSize);
        auto talkerKVType = mTalkerLLMRunner->getCacheManager().getKVCacheManager().getConfig().kvCacheType;
        LOG_INFO("Talker KV cache dtype: %s",
            talkerKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                      : (talkerKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", e.what());
        return false;
    }

    // Load CodePredictor engine from separate directory
    std::filesystem::path codePredictorEnginePath = std::filesystem::path(codePredictorEngineDir) / "llm.engine";
    std::filesystem::path codePredictorConfigPath = std::filesystem::path(codePredictorEngineDir) / "config.json";

    LOG_INFO("Loading CodePredictor engine from: %s", codePredictorEnginePath.string().c_str());

    try
    {
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mCodePredictorRunner = std::make_unique<LLMEngineRunner>(
            codePredictorEnginePath, codePredictorConfigPath, emptyLoraMap, mStream);

        // NOTE: CodePredictor ONNX now outputs FP32 logits directly (lm_head + cast in ONNX),
        // so standard logits shape validation applies.

        mCodePredictorConfig = mCodePredictorRunner->getEngineConfig();

        // Now read CodePredictor dimensions from loaded config
        mTalkerConfig.codePredictorHiddenSize = mCodePredictorConfig.hiddenSize;
        // NOTE: config.vocab_size == hidden_size (for engine compatibility since output is last_hidden)
        //       Real codebook_size is inferred from lm_head weight shape later

        LOG_INFO("CodePredictor engine loaded: vocabSize=%d, hiddenSize=%d, numLayers=%d",
            mCodePredictorConfig.vocabSize, mCodePredictorConfig.hiddenSize, mCodePredictorConfig.numDecoderLayers);
        auto cpKVType = mCodePredictorRunner->getCacheManager().getKVCacheManager().getConfig().kvCacheType;
        LOG_INFO("CodePredictor KV cache dtype: %s",
            cpKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                  : (cpKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load CodePredictor engine: %s", e.what());
        return false;
    }

    return true;
}

bool Qwen3OmniTTSRuntime::validateAndFillConfig(std::string const& talkerEngineDir)
{
    // Load config.json from talker directory
    std::filesystem::path configPath = std::filesystem::path(talkerEngineDir) / "config.json";
    LOG_INFO("Loading Talker config from: %s", configPath.string().c_str());

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        return false;
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config: %s", e.what());
        return false;
    }

    // Model dimensions
    mTalkerConfig.thinkerHiddenSize = configJson.value("thinker_hidden_size", 2048);
    mTalkerConfig.talkerHiddenSize = configJson["hidden_size"].get<int32_t>();
    mTalkerConfig.talkerVocabSize = configJson["vocab_size"].get<int32_t>();

    // CodePredictor RVQ configuration: Omni uses 16 code groups, TTS uses 32
    mTalkerConfig.numCodeGroups = configJson["num_code_groups"].get<int32_t>();
    check::check(mTalkerConfig.numCodeGroups >= 2,
        "num_code_groups must be >= 2, got: " + std::to_string(mTalkerConfig.numCodeGroups));
    mNumRvqLayers = mTalkerConfig.numCodeGroups - 1;
    mNumCodesPerFrame = mTalkerConfig.numCodeGroups;
    LOG_INFO("Config num_code_groups=%d -> RVQ layers=%d, codes per frame=%d", mTalkerConfig.numCodeGroups,
        mNumRvqLayers, mNumCodesPerFrame);

    // Runtime parameters
    mTalkerConfig.maxSeqLen = configJson.value("max_position_embeddings", 8192);

    // Validate dimensions with reasonable limits
    constexpr int32_t kMaxReasonableVocabSize = 200000;
    constexpr int32_t kMaxReasonableHiddenSize = 16384;
    constexpr int32_t kMaxReasonableSeqLen = 131072;

    check::check(mTalkerConfig.talkerVocabSize > 0 && mTalkerConfig.talkerVocabSize < kMaxReasonableVocabSize,
        "Invalid talker vocab size: " + std::to_string(mTalkerConfig.talkerVocabSize));
    check::check(mTalkerConfig.thinkerHiddenSize > 0 && mTalkerConfig.thinkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid thinker hidden size: " + std::to_string(mTalkerConfig.thinkerHiddenSize));
    check::check(mTalkerConfig.talkerHiddenSize > 0 && mTalkerConfig.talkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid talker hidden size: " + std::to_string(mTalkerConfig.talkerHiddenSize));
    check::check(mTalkerConfig.maxSeqLen > 0 && mTalkerConfig.maxSeqLen < kMaxReasonableSeqLen,
        "Invalid max sequence length: " + std::to_string(mTalkerConfig.maxSeqLen));

    // TTS special tokens (from thinker vocab)
    mTalkerConfig.ttsPadTokenId = configJson.value("tts_pad_token_id", 151671);
    mTalkerConfig.ttsBosTokenId = configJson.value("tts_bos_token_id", 151672);
    mTalkerConfig.ttsEosTokenId = configJson.value("tts_eos_token_id", 151673);

    // Codec special tokens (from talker vocab)
    mTalkerConfig.codecNothinkId = configJson["codec_nothink_id"].get<int32_t>();
    mTalkerConfig.codecThinkBosId = configJson["codec_think_bos_id"].get<int32_t>();
    mTalkerConfig.codecThinkEosId = configJson["codec_think_eos_id"].get<int32_t>();
    mTalkerConfig.codecPadId = configJson["codec_pad_id"].get<int32_t>();
    mTalkerConfig.codecBosId = configJson["codec_bos_id"].get<int32_t>();

    // CustomVoice: language-conditioned prefill path needs codec_think_id + codec_language_id map.
    // Both fields are optional; if absent the runtime falls back to the legacy no-language path.
    if (configJson.contains("codec_think_id"))
    {
        mTalkerConfig.codecThinkId = configJson["codec_think_id"].get<int32_t>();
    }
    if (configJson.contains("codec_language_id") && configJson["codec_language_id"].is_object())
    {
        for (auto const& [k, v] : configJson["codec_language_id"].items())
        {
            mTalkerConfig.codecLanguageId[k] = v.get<int32_t>();
        }
    }
    LOG_INFO("CustomVoice language config: codecThinkId=%d, codecLanguageId entries=%zu", mTalkerConfig.codecThinkId,
        mTalkerConfig.codecLanguageId.size());
    // Support both codec_eos_token_id (original) and codec_eos_id (legacy) for backward compatibility
    if (configJson.contains("codec_eos_token_id"))
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_token_id"].get<int32_t>();
    }
    else
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_id"].get<int32_t>();
    }

    // Speaker ID configuration
    mTalkerConfig.defaultSpeakerId = configJson.value("default_speaker_id", 2301);

    // Load speaker ID mapping if available
    if (configJson.contains("speaker_id") && configJson["speaker_id"].is_object())
    {
        for (auto const& [speaker_name, speaker_id] : configJson["speaker_id"].items())
        {
            mSpeakerIdMap[speaker_name] = speaker_id.get<int32_t>();
        }
        LOG_INFO("Loaded %zu speaker IDs from config", mSpeakerIdMap.size());

        // Log available speakers
        if (!mSpeakerIdMap.empty())
        {
            std::string speakerList;
            for (auto const& [name, id] : mSpeakerIdMap)
            {
                if (!speakerList.empty())
                {
                    speakerList += ", ";
                }
                speakerList += name + ":" + std::to_string(id);
            }
            LOG_DEBUG("Available speakers: %s", speakerList.c_str());
        }
    }

    LOG_INFO("Talker config: vocabSize=%d, hiddenSize=%d, thinkerHiddenSize=%d, defaultSpeaker=%d",
        mTalkerConfig.talkerVocabSize, mTalkerConfig.talkerHiddenSize, mTalkerConfig.thinkerHiddenSize,
        mTalkerConfig.defaultSpeakerId);
    LOG_DEBUG("TTS tokens: pad=%d, bos=%d, eos=%d", mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId,
        mTalkerConfig.ttsEosTokenId);
    LOG_DEBUG("Codec tokens: skipThink=%d, thinkBos=%d, thinkEos=%d, pad=%d, bos=%d, eos=%d",
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, mTalkerConfig.codecEosId);

    return true;
}

bool Qwen3OmniTTSRuntime::loadCodePredictorWeights(std::string const& codePredictorEngineDir)
{
    LOG_INFO("Loading %d CodePredictor lm_head weights", mNumRvqLayers);
    mCodePredictorLmHeadWeights.resize(mNumRvqLayers);
    {
        std::filesystem::path const lmHeadPath = std::filesystem::path(codePredictorEngineDir) / "lm_heads.safetensors";
        std::vector<rt::Tensor> allLmHeadTensors;
        if (!safetensors::loadSafetensors(lmHeadPath, allLmHeadTensors, mStream))
        {
            LOG_ERROR("Failed to load lm_heads.safetensors from: %s", lmHeadPath.string().c_str());
            return false;
        }
        if (static_cast<int32_t>(allLmHeadTensors.size()) != mNumRvqLayers)
        {
            LOG_ERROR("lm_heads.safetensors has %zu entries, expected %d (matching codec_embeddings count)",
                allLmHeadTensors.size(), mNumRvqLayers);
            return false;
        }
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const weightKey = "lm_head_" + std::to_string(i) + ".weight";
            auto it = std::find_if(allLmHeadTensors.begin(), allLmHeadTensors.end(),
                [&weightKey](rt::Tensor const& t) { return t.getName() == weightKey; });
            if (it == allLmHeadTensors.end())
            {
                LOG_ERROR("Missing key '%s' in lm_heads.safetensors", weightKey.c_str());
                return false;
            }
            if (it->getShape().getNumDims() != 2)
            {
                LOG_ERROR("%s should be 2D [vocabSize, hiddenSize]", weightKey.c_str());
                return false;
            }
            LOG_DEBUG("Loaded %s [%d, %d]", weightKey.c_str(), it->getShape()[0], it->getShape()[1]);
            mCodePredictorLmHeadWeights[i] = std::move(*it);
        }
    }

    mTalkerConfig.codebookSize = static_cast<int32_t>(mCodePredictorLmHeadWeights[0].getShape()[0]);
    LOG_INFO("Loaded %d CodePredictor lm_head weights, codebookSize=%d", mNumRvqLayers, mTalkerConfig.codebookSize);

    // Load small_to_mtp_projection: projects Talker hidden (2048) → CodePredictor input (1024)
    {
        std::filesystem::path const projPath
            = std::filesystem::path(codePredictorEngineDir) / "small_to_mtp_projection.safetensors";
        if (std::filesystem::exists(projPath))
        {
            std::vector<rt::Tensor> projTensors;
            if (!safetensors::loadSafetensors(projPath, projTensors, mStream))
            {
                LOG_ERROR("Failed to load small_to_mtp_projection from: %s", projPath.string().c_str());
                return false;
            }
            bool foundWeight = false, foundBias = false;
            for (auto& t : projTensors)
            {
                if (t.getName() == "weight")
                {
                    mSmallToMtpWeight = std::move(t);
                    foundWeight = true;
                }
                else if (t.getName() == "bias")
                {
                    mSmallToMtpBias = std::move(t);
                    foundBias = true;
                }
            }
            if (!foundWeight || !foundBias)
            {
                LOG_ERROR("Missing 'weight' or 'bias' in small_to_mtp_projection.safetensors");
                return false;
            }
            mUseSmallToMtpProjection = true;
            LOG_INFO("Loaded small_to_mtp_projection: weight=%ldx%ld, bias=%ld", mSmallToMtpWeight.getShape()[0],
                mSmallToMtpWeight.getShape()[1], mSmallToMtpBias.getShape()[0]);
        }
        else if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
        {
            mUseSmallToMtpProjection = false;
            LOG_INFO("No small_to_mtp_projection needed (talkerHiddenSize == codePredictorHiddenSize = %d)",
                mTalkerConfig.talkerHiddenSize);
        }
        else
        {
            LOG_ERROR(
                "small_to_mtp_projection.safetensors required when talkerHiddenSize (%d) != "
                "codePredictorHiddenSize (%d)",
                mTalkerConfig.talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
            return false;
        }
    }

    // CodePredictor embedding tables are already loaded in the constructor (with auto-detection
    // of mNumRvqLayers from codec_embeddings.safetensors).

    return true;
}

bool Qwen3OmniTTSRuntime::allocateBuffer()
{
    LOG_INFO("Allocating Qwen3-Omni TTS Runtime inference workspace buffers (maxBatchSize=%d)...", mMaxBatchSize);

    int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const maxBS = mMaxBatchSize;

    try
    {
        // Per-batch prefill workspace (reused sequentially per batch in buildTalkerPrefillFromSegments)
        mThinkerEmbedBuffer = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mThinkerEmbedBuffer");
        mGpuTokenIdsBuffer
            = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGpuTokenIdsBuffer");
        mMLPWorkspace = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mMLPWorkspace");
        mProjectedBuffer = rt::Tensor(
            {maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mProjectedBuffer");
        // Talker input embeds: [maxBS, maxSeqLen, H] for batched prefill
        mTalkerInputEmbeds = rt::Tensor({maxBS * maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerInputEmbeds");

        // Talker LLM workspace — batched
        mTalkerLogits = rt::Tensor(
            {maxBS, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mTalkerLogits");
        mTalkerSelectedIndices
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerSelectedIndices");
        mHostSelectedTokenIds
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedTokenIds");
        mHostTalkerContextLength
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostTalkerContextLength");

        // CodePredictor workspace — batch=1 (per-batch for-loop, each frame resets KV cache)
        mCodePredictorLogits = rt::Tensor(
            {1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mCodePredictorLogits");

        mCodePredictorLogitsPerHead.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            mCodePredictorLogitsPerHead[i] = rt::Tensor({1, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kFLOAT, "mCodePredictorLogitsPerHead_" + std::to_string(i));
        }

        mCodePredictorPrefillInput = rt::Tensor({1, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorPrefillInput");
        mCodePredictorCodecIds
            = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorCodecIds");
        mCodePredictorCodecEmbed = rt::Tensor({1, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorCodecEmbed");
        mRawCodecEmbed = rt::Tensor(
            {1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mRawCodecEmbed");
        mSmallToMtpProjectedHidden = rt::Tensor({1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mSmallToMtpProjectedHidden");
        mCodePredictorSelectedIndices
            = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorSelectedIndices");
        mHostSelectedCodeIds
            = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedCodeIds");
        mHostCodePredictorContextLength
            = rt::Tensor({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostCodePredictorContextLength");

        // Residual + decode buffers — batched for Talker engine execution
        mResidualEmbedBuffer = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mResidualEmbedBuffer");
        mTalkerDecodingIds
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerDecodingIds");
        mTalkerDecodingEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerDecodingEmbed");

        // KVCache reset — batched
        mHostReuseKVCacheLengths
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostReuseKVCacheLengths");

        // Sampling workspace — allocated at maxBS for batched topKtopP
        int32_t const defaultTopK{0};
        float const defaultTopP{0.9F};
        trt_edgellm::SamplingParams samplingParams(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, 1.0f, defaultTopK, defaultTopP);
        int64_t const samplingWorkspaceSize = trt_edgellm::getTopKtopPSamplingWorkspaceSize(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, samplingParams);
        mSamplingWorkspace = rt::Tensor(
            {samplingWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mSamplingWorkspace");
        // Per-batch seen codec tokens for repetition penalty
        mSeenCodecTokensBuf = rt::Tensor({maxBS, mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "mSeenCodecTokensBuf");

        // Generation loop workspace — Talker batched, CodePredictor batch=1
        mTalkerHiddenStatesBuffer = rt::Tensor({maxBS, maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerHiddenStatesBuffer");
        mCodePredictorHiddenStatesBuffer = rt::Tensor({1, mNumCodesPerFrame, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");
        mTalkerLastHidden = rt::Tensor(
            {maxBS, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mTalkerLastHidden");
        mCodecHiddensBuffer = rt::Tensor({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodecHiddensBuffer");

        // Trailing text hidden buffer: [maxBS * (maxSeqLen+1), H] — per-batch regions for Omni multi-batch
        mStreamingTrailingHidden = rt::Tensor({maxBS * (maxSeqLen + 1), talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mStreamingTrailingHidden");

        // Gather/scatter index buffer for multimodal token projection
        mGatherIndicesBuffer
            = rt::Tensor({maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGatherIndicesBuffer");

        // Streaming: single-token workspace for appendTrailingToken
        mStreamingTokenId = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mStreamingTokenId");
        mStreamingTokenEmbed = rt::Tensor(
            {1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingTokenEmbed");
        mStreamingProjOut
            = rt::Tensor({1, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingProjOut");
        mStreamingMlpWork
            = rt::Tensor({1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingMlpWork");

        LOG_INFO("Talker buffers allocated (maxBS=%d, maxSeqLen=%ld, talkerH=%ld, cpH=%d)", mMaxBatchSize, maxSeqLen,
            talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate Talker buffers: %s", e.what());
        return false;
    }
}

bool Qwen3OmniTTSRuntime::loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::loadTalkerWeights", nvtx_colors::YELLOW);

    // Load text_projection weights
    std::filesystem::path const textProjPath = std::filesystem::path(weightsDir) / "text_projection.safetensors";
    std::vector<rt::Tensor> textTensors;
    if (!safetensors::loadSafetensors(textProjPath, textTensors, stream))
    {
        LOG_ERROR("Failed to load text_projection from: %s", textProjPath.string().c_str());
        return false;
    }
    if (!extractMLPWeightsFromTensors(
            textTensors, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, "text_projection"))
    {
        return false;
    }

    // Load hidden_projection weights (same architecture as text_projection, different weights)
    std::filesystem::path const hiddenProjPath = std::filesystem::path(weightsDir) / "hidden_projection.safetensors";
    if (std::filesystem::exists(hiddenProjPath))
    {
        mIsOmni = true;
        std::vector<rt::Tensor> hiddenTensors;
        if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
        {
            LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
            return false;
        }
        if (!extractMLPWeightsFromTensors(
                hiddenTensors, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight, mHiddenFC2Bias, "hidden_projection"))
        {
            return false;
        }
        LOG_INFO("hidden_projection weights loaded from: %s", hiddenProjPath.string().c_str());
    }
    else
    {
        mIsOmni = false;
        LOG_INFO("hidden_projection.safetensors not found at %s (multimodal token projection unavailable)",
            hiddenProjPath.string().c_str());
    }

    // Note: mTextEmbeddingTable is loaded separately in the constructor with mIsOmni-aware path selection
    // (TTS: text_embedding.safetensors from talkerEngineDir, Omni: embedding.safetensors from tokenizerDir)

    // Load Talker embedding table
    std::filesystem::path const talkerEmbedPath = std::filesystem::path(weightsDir) / "embedding.safetensors";
    std::vector<rt::Tensor> talkerEmbedTensors;
    if (!safetensors::loadSafetensors(talkerEmbedPath, talkerEmbedTensors, stream))
    {
        LOG_ERROR("Failed to load Talker embedding from: %s", talkerEmbedPath.string().c_str());
        return false;
    }
    check::check(talkerEmbedTensors.size() == 1, "Talker embedding.safetensors should contain exactly one tensor");
    check::check(talkerEmbedTensors[0].getShape().getNumDims() == 2,
        "Talker embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTalkerEmbeddingTable = std::move(talkerEmbedTensors[0]);
    LOG_INFO("Talker embedding table loaded: [%lld, %lld]", mTalkerEmbeddingTable.getShape()[0],
        mTalkerEmbeddingTable.getShape()[1]);

    LOG_INFO("Talker weights loaded successfully");
    return true;
}

void Qwen3OmniTTSRuntime::initializeTTSEmbeddings(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::initializeTTSEmbeddings", nvtx_colors::YELLOW);

    auto const shape = mTextEmbeddingTable.getShape();
    ELLM_CHECK(
        shape.getNumDims() == 2, "Text embedding table must be 2D, got " + std::to_string(shape.getNumDims()) + "D");

    int64_t const vocabSize = shape[0];
    int64_t const thinkerHiddenSize = shape[1];

    ELLM_CHECK(mTalkerConfig.ttsPadTokenId < vocabSize && mTalkerConfig.ttsBosTokenId < vocabSize
            && mTalkerConfig.ttsEosTokenId < vocabSize,
        "TTS token IDs out of vocab range: pad=" + std::to_string(mTalkerConfig.ttsPadTokenId)
            + ", bos=" + std::to_string(mTalkerConfig.ttsBosTokenId)
            + ", eos=" + std::to_string(mTalkerConfig.ttsEosTokenId) + ", vocabSize=" + std::to_string(vocabSize));

    constexpr int32_t kNumTtsTokens = 3;
    std::vector<int32_t> const hostTtsIds
        = {mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId, mTalkerConfig.ttsEosTokenId};

    rt::Tensor ttsIds({1, kNumTtsTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor ttsRaw({1, kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor ttsProjected(
        {kNumTtsTokens, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor workspace({kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(
        ttsIds.rawPointer(), hostTtsIds.data(), kNumTtsTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::embeddingLookup(ttsIds, mTextEmbeddingTable, std::nullopt, ttsRaw, stream);
    // Reshape from [1, 3, hidden] to [3, hidden] for MLP (expects 2D input)
    check::check(ttsRaw.reshape({kNumTtsTokens, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(
        ttsRaw, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, ttsProjected, workspace, stream);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    mTtsPadEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsBosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsEosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    __half* const projectedPtr = static_cast<__half*>(ttsProjected.rawPointer());
    size_t const embedSize = hiddenSize * sizeof(__half);

    CUDA_CHECK(cudaMemcpyAsync(
        mTtsPadEmbed.rawPointer(), projectedPtr + 0 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsBosEmbed.rawPointer(), projectedPtr + 1 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsEosEmbed.rawPointer(), projectedPtr + 2 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));

    LOG_INFO("TTS embeddings initialized");
}

bool Qwen3OmniTTSRuntime::projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t speakerId, int32_t langId,
    rt::Tensor& output, int64_t& outputSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;

    // N = text tokens after stripping 3-token role prefix and 5-token suffix
    int64_t const N = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
    // Non-streaming prefill: kFixedPrefixLen rows + N text rows + 2 suffix rows.
    // langId >= 0 uses the 9-row CustomVoice language prefix; otherwise 8-row legacy prefix.
    int64_t const kFixedPrefixLen = (langId >= 0) ? 9 : kNonStreamingPrefixRows;
    outputSeqLen = kFixedPrefixLen + N + 2;
    LOG_INFO(
        "projectToTalkerInput: seqLen=%ld, N=%ld (stripped prefix=%d suffix=%d), outputSeqLen=%ld, speakerId=%d, "
        "langId=%d, prefixRows=%ld",
        seqLen, N, kAssistantPrefixLen, kAssistantTrailingSuffix, outputSeqLen, speakerId, langId, kFixedPrefixLen);

    // Project all tokens via text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, mProjectedBuffer,
        mMLPWorkspace, stream);

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkId, mTalkerConfig.codecThinkBosId,
        mTalkerConfig.codecThinkEosId, speakerId, mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, langId,
        static_cast<int32_t>(N), output, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream, std::vector<int64_t> const& perBatchContextLengths)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerPrefillStep", nvtx_colors::PURPLE);

    auto inputShape = inputEmbeds.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("executeTalkerPrefillStep: Input must be 3D [batchSize, seqLen, hiddenSize], got %dD",
            inputShape.getNumDims());
        return false;
    }

    int64_t const batchSize = inputEmbeds.getTRTDims().d[0];
    int64_t const seqLen = inputEmbeds.getTRTDims().d[1];

    if (batchSize > mMaxBatchSize)
    {
        LOG_ERROR("executeTalkerPrefillStep: batchSize %ld exceeds maxBatchSize %d", batchSize, mMaxBatchSize);
        return false;
    }

    // Reset Talker KV cache — shape must match batchSize so commitSequenceLength sees consistent activeBatchSize
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mHostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    mTalkerLLMRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Set per-batch context lengths (determines which token's logits the engine extracts)
    check::check(mHostTalkerContextLength.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostContextLength = mHostTalkerContextLength.dataPointer<int32_t>();
    if (!perBatchContextLengths.empty())
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(perBatchContextLengths[i]);
        }
    }
    else
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(seqLen);
        }
    }

    // Reshape logits to match the prefill batch size (CUDA graph capture may leave it at maxBS)
    check::check(outputLogits.reshape({batchSize, outputLogits.getShape()[outputLogits.getShape().getNumDims() - 1]}),
        "Tensor reshape failed");

    // Execute prefill
    rt::OptionalInputTensors emptyDeepstack{};
    return mTalkerLLMRunner->executePrefillStep(inputEmbeds, mHostTalkerContextLength, emptyDeepstack, outputLogits,
        rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream);
}

bool Qwen3OmniTTSRuntime::executeCodePredictorPrefillStep(rt::Tensor const& codecTokenEmbeds, int32_t generationStep,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorPrefillStep", nvtx_colors::ORANGE);

    // Reset CodePredictor KV cache for new frame (always per-batch, BS=1)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    mHostReuseKVCacheLengths.dataPointer<int32_t>()[0] = 0;
    auto& cpCacheManager = mCodePredictorRunner->getCacheManager();
    cpCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    auto& cpKVManager = cpCacheManager.getKVCacheManager();
    for (int32_t i = 0; i < cpKVManager.numLayers(); ++i)
    {
        rt::Tensor& layerKV = cpKVManager.getCombinedKVCache(i);
        CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
    }

    int32_t* const hostContextLength = mHostCodePredictorContextLength.dataPointer<int32_t>();
    hostContextLength[0] = kCodePredictorPrefillSeqLen;

    int32_t const lmHeadIdx = std::min(generationStep, mNumRvqLayers - 1);
    if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[lmHeadIdx]))
    {
        LOG_ERROR("Failed to bind lm_head_weight[%d]", lmHeadIdx);
        return false;
    }

    // Execute prefill - engine outputs logits directly (with lm_head applied)
    rt::OptionalInputTensors emptyDeepstack{};
    if (!mCodePredictorRunner->executePrefillStep(codecTokenEmbeds, mHostCodePredictorContextLength, emptyDeepstack,
            outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
    {
        LOG_ERROR("CodePredictor prefill step failed");
        return false;
    }

    return true;
}

bool Qwen3OmniTTSRuntime::executeCodePredictorDecodingStep(int32_t tokenId, int32_t embeddingTableIndex,
    int32_t generationStep, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorDecodingStep", nvtx_colors::ORANGE);

    CUDA_CHECK(cudaMemcpyAsync(
        mCodePredictorCodecIds.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const embedIdx = std::min(embeddingTableIndex, mNumRvqLayers - 1);
    // Lookup into mRawCodecEmbed (talkerHiddenSize=2048) — codec embedding tables are in Talker's space
    check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(
        mCodePredictorCodecIds, mCodePredictorEmbeddingTables[embedIdx], std::nullopt, mRawCodecEmbed, stream);

    // Save raw (2048-dim) embedding to mCodecHiddensBuffer for residual connection
    // Position mapping: generationStep 1->pos 1, 2->pos 2, ..., (mNumRvqLayers-1)->pos (mNumRvqLayers-1)
    if (generationStep >= 1 && generationStep <= mNumRvqLayers - 1)
    {
        int64_t const H = mTalkerConfig.talkerHiddenSize;
        __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer()) + generationStep * H;
        CUDA_CHECK(
            cudaMemcpyAsync(dst, mRawCodecEmbed.rawPointer(), H * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    // Project mRawCodecEmbed → mCodePredictorCodecEmbed if needed.
    check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    if (!mUseSmallToMtpProjection)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecEmbed.rawPointer(), mRawCodecEmbed.rawPointer(),
            mTalkerConfig.codePredictorHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeLinearLayer(mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
    }

    int32_t const lmHeadIdx = std::min(generationStep, mNumRvqLayers - 1);

    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");

    if (mCodePredictorGraphsCaptured)
    {
        // Graph path: lm_head_weight addresses were bound during capture and remain unchanged,
        // so setLmHeadWeight is unnecessary. Each graph is keyed by its per-head output buffer.
        if (!mCodePredictorRunner->executeVanillaDecodingStep(mCodePredictorCodecEmbed,
                mCodePredictorLogitsPerHead[lmHeadIdx], rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
        {
            LOG_ERROR("CodePredictor decoding step failed (graph path)");
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(outputLogits.rawPointer(), mCodePredictorLogitsPerHead[lmHeadIdx].rawPointer(),
            outputLogits.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        // Non-graph path: must bind lm_head_weight before each enqueueV3
        if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[lmHeadIdx]))
        {
            LOG_ERROR("Failed to bind lm_head_weight[%d]", lmHeadIdx);
            return false;
        }
        if (!mCodePredictorRunner->executeVanillaDecodingStep(
                mCodePredictorCodecEmbed, outputLogits, rt::OptionalOutputTensor{std::ref(outputHiddenStates)}, stream))
        {
            LOG_ERROR("CodePredictor decoding step failed");
            return false;
        }
    }

    return true;
}

// ========== CUDA Graph Capture ==========

bool Qwen3OmniTTSRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    std::string const emptyLoraWeightsName = "";

    // Talker: capture for all supported batch sizes (1..maxBatchSize)
    bool captureStatus{true};
    for (int32_t bs = 1; bs <= mMaxBatchSize; ++bs)
    {
        check::check(mResidualEmbedBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mTalkerHiddenStatesBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(mTalkerLogits.reshape({bs, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
        captureStatus &= mTalkerLLMRunner->captureVanillaDecodingCudaGraph(mResidualEmbedBuffer, mTalkerLogits,
            emptyLoraWeightsName, stream, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)});
    }

    // CodePredictor: mNumRvqLayers graphs, one per lm_head_weight.
    // Each graph uses a distinct output logits buffer so that the decodingKey (which includes
    // the output address) naturally produces a unique key per graph.
    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");
    rt::OptionalOutputTensor cpHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

    for (int32_t i = 0; i < mNumRvqLayers; ++i)
    {
        if (!mCodePredictorRunner->setLmHeadWeight(mCodePredictorLmHeadWeights[i]))
        {
            LOG_ERROR("Failed to bind lm_head_weight[%d] for CUDA graph capture", i);
            captureStatus = false;
            continue;
        }
        captureStatus &= mCodePredictorRunner->captureVanillaDecodingCudaGraph(
            mCodePredictorCodecEmbed, mCodePredictorLogitsPerHead[i], emptyLoraWeightsName, stream, cpHiddenOpt);
    }

    mCodePredictorGraphsCaptured = captureStatus;

    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for Talker and all CodePredictor lm_heads.");
    }
    else
    {
        LOG_WARNING("Failed to capture some decoding CUDA graphs. Will use fallback engine execution.");
    }

    return captureStatus;
}

// ========== Audio Generation API ==========

bool Qwen3OmniTTSRuntime::prepareTalkerInput(std::vector<int32_t> const& textTokenIds,
    TalkerGenerationRequest const& request, int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    if (seqLen == 0)
    {
        LOG_ERROR("prepareTalkerInput: empty token ID list");
        return false;
    }
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check::check(mThinkerEmbedBuffer.reshape({1, seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, mThinkerEmbedBuffer, stream);
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    // Determine speaker ID
    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (request.speakerId >= 0)
    {
        speakerId = request.speakerId;
    }
    else if (!request.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(request.speakerName);
    }

    // CustomVoice language conditioning: resolve language string -> codec token ID.
    // Lower-case the request string; lookup in codecLanguageId map. -1 means use legacy no-language path.
    int32_t langId = -1;
    if (!request.language.empty())
    {
        std::string langLc = request.language;
        std::transform(langLc.begin(), langLc.end(), langLc.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        auto it = mTalkerConfig.codecLanguageId.find(langLc);
        if (it != mTalkerConfig.codecLanguageId.end())
        {
            langId = it->second;
            LOG_INFO(
                "CustomVoice language conditioning enabled: language=\"%s\" -> codec_id=%d", langLc.c_str(), langId);
        }
        else
        {
            LOG_WARNING(
                "Requested language=\"%s\" not found in codec_language_id map (size=%zu); "
                "falling back to no-language prefill path",
                langLc.c_str(), mTalkerConfig.codecLanguageId.size());
        }
    }

    // MLP projection: thinker embed → talker input embeds (non-streaming, outputSeqLen = seqLen + 2 or +3 w/ lang)
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!projectToTalkerInput(mThinkerEmbedBuffer, speakerId, langId, mTalkerInputEmbeds, outSeqLen, stream))
    {
        LOG_ERROR("MLP projection failed");
        return false;
    }

    // Reshape buffers to 3D [1, seqLen, H] for Talker LLM input
    check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGeneration(
    std::vector<TalkerGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGeneration", nvtx_colors::PURPLE);
    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceSpecDecodeRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams singleSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);

    // Per-batch sequential: tokenize → prepareTalkerInput → prefill → sample first codec token
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<int64_t> perBatchSeqLens(activeBatchSize);

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        LLMGenerationRequest::Request llmReq;
        llmReq.messages = requests[b].messages;
        LLMGenerationRequest::FormattedRequest formatted;
        if (!mTokenizer->applyChatTemplate(llmReq, formatted, requests[b].applyChatTemplate,
                requests[b].addGenerationPrompt, requests[b].enableThinking))
        {
            LOG_ERROR("Chat template failed for batch %d", b);
            return false;
        }
        std::vector<int32_t> const textTokenIds = mTokenizer->encode(formatted.formattedCompleteRequest);

        int64_t seqLen = 0;
        if (!prepareTalkerInput(textTokenIds, requests[b], seqLen, stream))
        {
            LOG_ERROR("Input preparation failed for batch %d", b);
            return false;
        }
        perBatchSeqLens[b] = seqLen;

        {
            TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
            if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
            {
                LOG_ERROR("Talker prefill failed for batch %d", b);
                return false;
            }
        }

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
        trt_edgellm::topKtopPSamplingFromLogits(
            mTalkerLogits, mTalkerSelectedIndices, singleSamplingParams, mSamplingWorkspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        states[b].codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
        states[b].rvqCodes.reserve(requests[b].maxAudioLength);
        LOG_INFO("Batch %d: first codec token: %d", b, states[b].codecToken);
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const safe = std::max(1, talkerKVCapacity - static_cast<int32_t>(perBatchSeqLens[b]));
        effectiveMaxFrames = std::max(effectiveMaxFrames, std::min(requests[b].maxAudioLength, safe));
    }

    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);
    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, stream))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    for (auto const& s : states)
        totalFrames += s.talkerFrames;
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    response.success = true;
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGenerationFromThinker(
    std::vector<OmniGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGenerationFromThinker", nvtx_colors::PURPLE);

    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched Omni audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceSpecDecodeRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<rt::Tensor> perBatchTrailingViews(activeBatchSize);
    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Phase 1: Build per-batch Talker prefill embeddings.
    // buildTalkerPrefillFromSegments writes to mTalkerInputEmbeds starting at row 0 (the "scratch"
    // region). After building each batch we relocate the result to a safe per-batch slot at
    // row ((b+1) * maxInputSeqLen) so that the next build's scratch write doesn't clobber it.
    // Batch 0 is special: it's built last (after all others are safely stashed), so its data
    // can stay in place at row 0.
    int64_t const maxInputSeqLen = mTalkerLLMConfig.maxSupportedInputLength;
    std::vector<int64_t> perBatchSeqLen(activeBatchSize);
    int64_t maxOutSeqLen = 0;

    // Process batches 1..N-1 first (stash each), then batch 0 last (stays in scratch = row 0)
    auto buildOneBatch = [&](int32_t b) -> bool {
        auto const& req = requests[b];
        if (req.fullText.empty() && req.textTokenIds.empty())
        {
            LOG_ERROR("Omni request batch %d has empty fullText and no textTokenIds", b);
            return false;
        }

        std::vector<int32_t> const textTokenIds
            = req.textTokenIds.empty() ? mTokenizer->encode(req.fullText) : req.textTokenIds;

        int32_t speakerId = mTalkerConfig.defaultSpeakerId;
        if (req.speakerId >= 0)
            speakerId = req.speakerId;
        else if (!req.speakerName.empty())
            speakerId = getSpeakerIdByName(req.speakerName);

        __half* const batchTrailingPtr
            = static_cast<__half*>(mStreamingTrailingHidden.rawPointer()) + b * trailingStride * hiddenSize;
        rt::Tensor batchTrailingBuf(
            batchTrailingPtr, rt::Coords{trailingStride, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        CUDA_CHECK(cudaMemsetAsync(batchTrailingPtr, 0, trailingStride * hiddenSize * sizeof(__half), stream));

        int32_t trailingCount = 0;
        int64_t outSeqLen = 0;
        if (!buildTalkerPrefillFromSegments(textTokenIds, req.thinkerPrefillEmbeds, req.thinkerHiddenStates,
                req.prefillLength, mTextEmbeddingTable, speakerId, batchTrailingBuf, trailingCount, outSeqLen, stream))
        {
            return false;
        }

        finalizeTrailing(batchTrailingBuf, trailingCount, stream);
        trailingCount++;

        perBatchSeqLen[b] = outSeqLen;
        maxOutSeqLen = std::max(maxOutSeqLen, outSeqLen);

        perBatchTrailingViews[b]
            = rt::Tensor(batchTrailingPtr, rt::Coords{static_cast<int64_t>(trailingCount), hiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        trailingPtrs[b] = &perBatchTrailingViews[b];
        states[b].rvqCodes.reserve(req.maxAudioLength);
        return true;
    };

    // Build batches 1..N-1 first and stash each to slot (b * maxInputSeqLen)
    for (int32_t b = activeBatchSize - 1; b >= 1; --b)
    {
        if (!buildOneBatch(b))
            return false;

        __half* const slotPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + b * maxInputSeqLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(slotPtr, mTalkerInputEmbeds.rawPointer(),
            perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    // Build batch 0 last — result stays at row 0 (no copy needed)
    if (!buildOneBatch(0))
        return false;

    // Phase 2: Assemble contiguous padded [BS, maxOutSeqLen, H] from per-batch slots.
    // Slot layout: batch 0 at row 0, batch b>=1 at row (b * maxInputSeqLen).
    // Target layout: batch b at row (b * maxOutSeqLen).
    // Process in reverse so high-address batches are moved first (avoids overlap).
    __half* const base = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    for (int32_t b = activeBatchSize - 1; b >= 0; --b)
    {
        __half* const src = base + (b == 0 ? 0 : b * maxInputSeqLen * hiddenSize);
        __half* const dst = base + b * maxOutSeqLen * hiddenSize;
        if (src != dst)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                dst, src, perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        if (perBatchSeqLen[b] < maxOutSeqLen)
        {
            CUDA_CHECK(cudaMemsetAsync(dst + perBatchSeqLen[b] * hiddenSize, 0,
                (maxOutSeqLen - perBatchSeqLen[b]) * hiddenSize * sizeof(__half), stream));
        }
    }

    // Single batched Talker prefill with per-batch context lengths
    check::check(mTalkerInputEmbeds.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, maxOutSeqLen, mTalkerConfig.talkerHiddenSize}),
        "Tensor reshape failed");

    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream, perBatchSeqLen))
        {
            LOG_ERROR("Batched Talker prefill failed");
            return false;
        }
    }

    // Phase 3: Per-batch logit adjustment and sampling from batched logits [BS, vocabSize]
    check::check(mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        rt::Tensor logitsSlice(static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
            rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
    }

    trt_edgellm::topKtopPSamplingFromLogits(
        mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].codecToken = hostTokens[b];
        LOG_INFO("Omni batch %d: first codec token: %d, trailing=%d, seqLen=%ld", b, states[b].codecToken,
            static_cast<int32_t>(perBatchTrailingViews[b].getShape()[0]), perBatchSeqLen[b]);
    }

    if (getProfilingEnabled())
    {
        CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const eff = std::min(requests[b].maxAudioLength,
            std::max(1, talkerKVCapacity - static_cast<int32_t>(requests[b].textTokenIds.size())));
        effectiveMaxFrames = std::max(effectiveMaxFrames, eff);
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, stream, perBatchSeqLen))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    bool anyHitEos = false;
    for (auto const& s : states)
    {
        totalFrames += s.talkerFrames;
        if (s.codecToken == mTalkerConfig.codecEosId)
        {
            anyHitEos = true;
        }
    }
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;
        int32_t prefillSeqLen = perBatchSeqLen.empty() ? 0 : static_cast<int32_t>(perBatchSeqLen[0]);

        mOmniTalkerMetrics.recordRun(totalFrames, totalFrames * codesPerFrame, prefillMs, prefillSeqLen,
            anyHitEos ? "eos" : "max_length", false);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS
            = static_cast<float>(totalFrames * kAudioSamplesPerFrame) / static_cast<float>(kAudioSampleRate);
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(totalFrames) * kAudioSamplesPerFrame;
        mOmniLatencyMetrics.sampleRate = kAudioSampleRate;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    response.success = true;
    return true;
}

// ========== Shared Generation Loop ==========

bool Qwen3OmniTTSRuntime::runTalkerGenerationLoop(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize,
    int32_t maxFrames, SamplingParams const& talkerSamplingParams, SamplingParams const& predictorSamplingParams,
    float repetitionPenalty, std::vector<rt::Tensor const*> const& trailingTextHiddens, cudaStream_t stream,
    std::vector<int64_t> const& prefillSeqLens)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runTalkerGenerationLoop", nvtx_colors::PURPLE);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t unfinished = activeBatchSize;

    // Initialize per-batch seen token tracking. numSeenTokens starts at 0 so the repetition
    // penalty kernel reads no entries on the first decode frame, avoiding a read of the
    // unseeded GPU buffer. The in-loop code below records tokens into both the host set and
    // GPU buffer starting from the first iteration.
    // NOTE: the host set / GPU buffer are still off-by-one (host set tracks the token entering
    // each iteration while the GPU buffer receives the newly sampled token). That functional
    // drift is pre-existing and tracked as a follow-up together with the standalone-TTS
    // Chinese-prompt prefill-EOS anomaly surfaced during MR 770 review.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].numSeenTokens = 0;
    }

    int32_t globalFrame = 0;
    {
        TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

        while (unfinished > 0 && globalFrame < maxFrames)
        {
            // Per-batch: CodePredictor + residual (batch=1 engine calls)
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Reset CodePredictor KV cache for this batch (batch=1)
                int32_t* cpReuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
                cpReuseData[0] = 0;
                check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
                mCodePredictorRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

                // Extract this batch's last hidden from mTalkerHiddenStatesBuffer.
                // After batched prefill: shape [BS, maxOutSeqLen, H] with padding; use the
                // per-batch actual length for correct last-token extraction.
                // After decode: shape [BS, 1, H], seqLen=1 for all batches.
                auto const& fullShape = mTalkerHiddenStatesBuffer.getShape();
                int64_t const paddedSeqDim = fullShape[1];
                int64_t const hDim = fullShape[2];
                int64_t const actualSeqDim
                    = (!prefillSeqLens.empty() && paddedSeqDim > 1) ? prefillSeqLens[b] : paddedSeqDim;
                rt::Tensor batchHiddenView(
                    static_cast<__half*>(mTalkerHiddenStatesBuffer.rawPointer()) + b * paddedSeqDim * hDim,
                    rt::Coords{1, actualSeqDim, hDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                check::check(mTalkerLastHidden.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
                if (!extractTalkerLastHidden(batchHiddenView, mTalkerLastHidden, stream))
                {
                    LOG_ERROR("extractTalkerLastHidden failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }

                std::vector<int32_t> frameCodes;
                if (!runCodePredictorGenerationForFrame(
                        states[b].codecToken, mTalkerLastHidden, predictorSamplingParams, frameCodes, stream))
                {
                    LOG_ERROR("CodePredictor failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }

                states[b].rvqCodes.push_back(std::move(frameCodes));

                // Compute residual for this batch, writing to batch b's slot in mResidualEmbedBuffer
                rt::Tensor residualSlice(
                    static_cast<__half*>(mResidualEmbedBuffer.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                if (!computeResidualConnection(
                        states[b].rvqCodes.back(), trailingTextHiddens[b], globalFrame, residualSlice, stream))
                {
                    LOG_ERROR("Residual connection failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }
            }

            // Batched Talker decode step: mResidualEmbedBuffer [BS, 1, H] → mTalkerLogits [BS, vocabSize]
            check::check(mResidualEmbedBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            if (!mTalkerLLMRunner->executeVanillaDecodingStep(mResidualEmbedBuffer, mTalkerLogits,
                    rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)}, stream))
            {
                LOG_ERROR("Batched Talker decoding step failed at frame %d", globalFrame);
                return false;
            }

            // Per-batch logit adjustment (different seenTokens per batch)
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Create views into batch b's logits row
                rt::Tensor logitsSlice(
                    static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
                    rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

                // Per-batch seenTokens buffer
                rt::Tensor seenBufSlice(
                    mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity,
                    rt::Coords{mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

                kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
                    mTalkerConfig.talkerVocabSize, codecEosId, states[b].numSeenTokens, repetitionPenalty, stream);
            }

            // Batched sampling
            check::check(
                mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
            check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
                mSamplingWorkspace, stream, 42, static_cast<uint64_t>(globalFrame + 1));
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Per-batch: update state, check EOS
            int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Track seen token for repetition penalty
                if (states[b].seenTokenSet.insert(states[b].codecToken).second)
                {
                    int32_t* seenBuf
                        = mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity;
                    CUDA_CHECK(cudaMemcpyAsync(seenBuf + states[b].numSeenTokens,
                        mTalkerSelectedIndices.dataPointer<int32_t>() + b, sizeof(int32_t), cudaMemcpyDeviceToDevice,
                        stream));
                    states[b].numSeenTokens++;
                }

                states[b].codecToken = hostTokens[b];
                states[b].talkerFrames++;

                if (states[b].codecToken == codecEosId || states[b].talkerFrames >= maxFrames)
                {
                    states[b].finished = true;
                    unfinished--;
                }
            }
            globalFrame++;
        }
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        bool const hitEos = (states[b].codecToken == codecEosId);
        LOG_INFO("Batch %d: %d audio frames (exit: %s)", b, states[b].talkerFrames, hitEos ? "EOS" : "maxFrames");
    }

    return true;
}

bool Qwen3OmniTTSRuntime::runSingleTalkerDecodeFrame(int32_t& codecToken, SamplingParams const& talkerSamplingParams,
    SamplingParams const& predictorSamplingParams, rt::Tensor const* trailingPtr, int32_t frameIdx,
    std::unordered_set<int32_t>& seenTokenSet, int32_t& numSeenTokens, float repetitionPenalty,
    std::vector<std::vector<int32_t>>& rvqCodes, cudaStream_t stream)
{
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // Ensure batch=1 shape for CodePredictor KV cache reset (streaming path is per-batch)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    mHostReuseKVCacheLengths.dataPointer<int32_t>()[0] = 0;
    mCodePredictorRunner->getCacheManager().resetForNewSequences(mHostReuseKVCacheLengths, stream);

    if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mTalkerLastHidden, stream))
    {
        LOG_ERROR("extractTalkerLastHidden failed at frame %d", frameIdx);
        return false;
    }

    std::vector<int32_t> frameCodes;
    if (!runCodePredictorGenerationForFrame(codecToken, mTalkerLastHidden, predictorSamplingParams, frameCodes, stream))
    {
        LOG_ERROR("CodePredictor generation failed at frame %d", frameIdx);
        return false;
    }

    rvqCodes.push_back(std::move(frameCodes));

    if (!computeResidualConnection(rvqCodes.back(), trailingPtr, frameIdx, mResidualEmbedBuffer, stream))
    {
        LOG_ERROR("Residual connection failed at frame %d", frameIdx);
        return false;
    }

    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

    if (!mTalkerLLMRunner->executeVanillaDecodingStep(
            mResidualEmbedBuffer, mTalkerLogits, rt::OptionalOutputTensor{std::ref(mTalkerHiddenStatesBuffer)}, stream))
    {
        LOG_ERROR("Talker decoding step failed at frame %d", frameIdx);
        return false;
    }

    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
        mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream, 42, static_cast<uint64_t>(frameIdx + 1));
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (seenTokenSet.insert(codecToken).second)
    {
        CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
            mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        ++numSeenTokens;
    }

    codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
    SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    // Original model logic:
    // - codes: code_0 (from Talker) + code_1..code_N (from CodePredictor) = mNumCodesPerFrame codes
    // - hidden_states: written directly to mCodecHiddensBuffer

    outputCodes.clear();
    outputCodes.reserve(mNumCodesPerFrame);

    // code_0 comes from Talker
    outputCodes.push_back(codecToken);

    int64_t const hiddenSize = mTalkerConfig.codePredictorHiddenSize;

    // ========== Prefill: generate code_1 ==========
    // Input: concat([proj(talker_hidden), proj(embed(code_0))]) -> [1, 2, codePredictorHiddenSize]
    // NOTE: code_0 embedding uses TALKER's codec_embedding (2048-dim),
    // projected to 1024 via small_to_mtp_projection
    // PyTorch: last_id_hidden = self.small_to_mtp_projection(self.get_input_embeddings()(input_ids))
    int32_t code = 0;
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_PREFILL, stream);

        // Step 1: Lookup code_0 from Talker's embedding table into mRawCodecEmbed (2048-dim)
        CUDA_CHECK(cudaMemcpyAsync(
            mCodePredictorCodecIds.rawPointer(), &codecToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        check::check(mRawCodecEmbed.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        kernel::embeddingLookup(mCodePredictorCodecIds, mTalkerEmbeddingTable, std::nullopt, mRawCodecEmbed, stream);

        // Step 2: Project talkerHiddenState → slot 0 of prefill input
        // Step 3: Project mRawCodecEmbed → slot 1 of prefill input
        // Step 4: Concat into mCodePredictorPrefillInput [1, 2, codePredictorHiddenSize]
        check::check(mRawCodecEmbed.reshape({1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mCodePredictorCodecEmbed.reshape({1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
        if (!mUseSmallToMtpProjection)
        {
            CUDA_CHECK(cudaMemcpyAsync(mCodePredictorPrefillInput.rawPointer(), talkerHiddenState.rawPointer(),
                hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + hiddenSize,
                mRawCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            kernel::invokeLinearLayer(
                talkerHiddenState, mSmallToMtpWeight, mSmallToMtpBias, mSmallToMtpProjectedHidden, stream);
            kernel::invokeLinearLayer(
                mRawCodecEmbed, mSmallToMtpWeight, mSmallToMtpBias, mCodePredictorCodecEmbed, stream);
            CUDA_CHECK(cudaMemcpyAsync(mCodePredictorPrefillInput.rawPointer(), mSmallToMtpProjectedHidden.rawPointer(),
                hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + hiddenSize,
                mCodePredictorCodecEmbed.rawPointer(), hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }

        check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 2, mTalkerConfig.codePredictorHiddenSize}),
            "Tensor reshape failed");

        // NOTE: CodePredictor ONNX outputs FP32 logits directly (lm_head + cast in ONNX)
        // generationStep=0 corresponds to code_1 (using lm_head_0)
        if (!executeCodePredictorPrefillStep(
                mCodePredictorPrefillInput, 0, mCodePredictorLogits, mCodePredictorHiddenStatesBuffer, stream))
        {
            return false;
        }

        // Sample code_1
        trt_edgellm::topKtopPSamplingFromLogits(
            mCodePredictorLogits, mCodePredictorSelectedIndices, samplingParams, mSamplingWorkspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        code = mHostSelectedCodeIds.dataPointer<int32_t>()[0];
        outputCodes.push_back(code); // code_1
    } // end kCODEPREDICTOR_PREFILL scope

    // ========== Write embedding lookups to mCodecHiddensBuffer for residual connection ==========
    // mCodecHiddensBuffer layout: [1, mNumCodesPerFrame, H]
    // Position 0:  embed(code_0)  using Talker's embedding   - filled in computeResidualConnection
    // Position 1..(mNumRvqLayers-1): codec_embedding[step-1](code_step)
    //   — filled here (input embeddings, NOT engine hidden states)
    // Position mNumRvqLayers: embed(code_last) using CodePredictor embed[-1] - filled in computeResidualConnection
    //
    // PyTorch original (modeling_qwen3_omni.py:3419):
    //   mid_residual_hiddens = [hid[0] for hid in predictor_result.hidden_states[1:]]
    //   hid[0] = inputs_embeds for each decode step = codec_embedding[step-1](code_step)
    //   NOT the transformer output (last layer hidden states)

    // mCodecHiddensBuffer stores raw (2048-dim) codec embeddings for the residual connection
    check::check(
        mCodecHiddensBuffer.reshape({1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

    // ========== Decoding loop: generate code_2 to code_(mNumRvqLayers) ==========
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        for (int step = 2; step <= mNumRvqLayers; ++step)
        {
            check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
                "Tensor reshape failed");

            rt::OptionalInputTensor prevHiddenOpt{std::ref(mCodePredictorHiddenStatesBuffer)};

            // PyTorch generation_steps logic:
            //   - generation_steps=1: embed(code_1) with codec_embedding[0], output with lm_head[1] -> code_2
            //   - generation_steps=N: embed(code_N) with codec_embedding[N-1], output with lm_head[N] -> code_(N+1)
            int32_t const embeddingIdx = step - 2; // step=2->embed[0], step=3->embed[1], ...
            int32_t const lmHeadIdx = step - 1;    // step=2->lm_head[1], step=3->lm_head[2], ...

            if (!executeCodePredictorDecodingStep(
                    code, embeddingIdx, lmHeadIdx, mCodePredictorLogits, mCodePredictorHiddenStatesBuffer, stream))
            {
                return false;
            }

            // Embedding is now saved inside executeCodePredictorDecodingStep before engine execution

            trt_edgellm::topKtopPSamplingFromLogits(
                mCodePredictorLogits, mCodePredictorSelectedIndices, samplingParams, mSamplingWorkspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            code = mHostSelectedCodeIds.dataPointer<int32_t>()[0];
            outputCodes.push_back(code); // code_2 to code_15
        }
    } // end kCODEPREDICTOR_GENERATION scope

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(std::vector<int32_t> const& codes,
    rt::Tensor const* trailingTextHidden, int32_t generationStep, rt::Tensor& outputResidual, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // PyTorch (modeling_qwen3_omni.py:3429-3434):
    //   if generation_step < trailing_text_hidden.shape[1]:
    //       inputs_embeds += trailing_text_hidden[:, generation_step]
    //   else:
    //       inputs_embeds += tts_pad_embed

    check::check(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
        "Expected " + std::to_string(mNumCodesPerFrame) + " codes, got " + std::to_string(codes.size()));

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    // reshape only when the tensor owns its memory (skip for non-owning views
    // which are already constructed with the correct shape by the caller)
    if (!outputResidual.reshape({1, 1, hiddenSize}))
    {
        check::check(
            outputResidual.getShape() == rt::Coords{1, 1, hiddenSize}, "Non-owning residual tensor has wrong shape");
    }

    __half const* addend = mTtsPadEmbed.dataPointer<__half>();
    if (trailingTextHidden != nullptr)
    {
        int64_t const trailingLen = trailingTextHidden->getShape()[0];
        if (generationStep < trailingLen)
        {
            addend = static_cast<__half const*>(trailingTextHidden->rawPointer()) + generationStep * hiddenSize;
        }
    }

    kernel::invokeResidualConnection(mCodecHiddensBuffer, mTalkerEmbeddingTable,
        mCodePredictorEmbeddingTables[mNumRvqLayers - 1], codes[0], codes[mNumRvqLayers], addend, outputResidual,
        stream);

    return true;
}

bool Qwen3OmniTTSRuntime::extractTalkerLastHidden(
    rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream)
{
    auto const& shape = talkerHiddenStates.getShape();
    int32_t const numDims = shape.getNumDims();

    if (numDims != 3)
    {
        LOG_ERROR("extractTalkerLastHidden: Expected 3D tensor [batchSize, seqLen, hiddenSize], got %dD", numDims);
        return false;
    }

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    int64_t const hiddenSize = shape[2];

    check::check(outputLastHidden.reshape({batchSize, hiddenSize}), "Tensor reshape failed");

    // Extract last token for each batch: output[b] = input[b, seqLen-1, :]
    size_t const copySize = hiddenSize * sizeof(__half);
    for (int64_t b = 0; b < batchSize; ++b)
    {
        size_t const srcOffset = (b * seqLen + seqLen - 1) * hiddenSize * sizeof(__half);
        size_t const dstOffset = b * hiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(outputLastHidden.rawPointer()) + dstOffset,
            static_cast<char const*>(talkerHiddenStates.rawPointer()) + srcOffset, copySize, cudaMemcpyDeviceToDevice,
            stream));
    }

    return true;
}

int32_t Qwen3OmniTTSRuntime::getSpeakerIdByName(std::string const& speakerName) const
{
    auto it = mSpeakerIdMap.find(speakerName);
    if (it != mSpeakerIdMap.end())
    {
        return it->second;
    }

    LOG_WARNING(
        "Speaker '%s' not found, using default speaker ID %d", speakerName.c_str(), mTalkerConfig.defaultSpeakerId);
    return mTalkerConfig.defaultSpeakerId;
}

// ═══════════════════════════════════════════════════════════════════════════
//        Shared Decode Frame + Prefill Construction
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::buildTalkerPrefillFromSegments(std::vector<int32_t> const& textTokenIds,
    rt::Tensor const* prefillEmbedPtr, rt::Tensor const* prefillHiddenPtr, int32_t prefillLen,
    rt::Tensor const& thinkerEmbedTable, int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount,
    int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    // Step 1: Build layer-0 embeddings in mThinkerEmbedBuffer
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    if (prefillEmbedPtr != nullptr)
    {
        int32_t const copyLen = std::min(prefillLen, static_cast<int32_t>(seqLen));
        size_t const prefillBytes = copyLen * thinkerHiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(mThinkerEmbedBuffer.rawPointer(), prefillEmbedPtr->rawPointer(), prefillBytes,
            cudaMemcpyDeviceToDevice, stream));

        int32_t const genLen = static_cast<int32_t>(seqLen) - copyLen;
        if (genLen > 0)
        {
            check::check(mGpuTokenIdsBuffer.reshape({1, genLen}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data() + copyLen,
                genLen * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            __half* genDst = static_cast<__half*>(mThinkerEmbedBuffer.rawPointer()) + copyLen * thinkerHiddenSize;
            rt::Tensor genEmbedView(genDst, rt::Coords{1, static_cast<int64_t>(genLen), thinkerHiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, genEmbedView, stream);
        }
    }
    else
    {
        check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        rt::Tensor embedView(mThinkerEmbedBuffer.rawPointer(), rt::Coords{1, seqLen, thinkerHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, embedView, stream);
    }

    // Step 2: Project ALL tokens through text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mProjectedBuffer, mMLPWorkspace, stream);

    // Step 3: Parse segments by <|im_start|> positions
    std::vector<SegmentInfo> segments;
    for (int64_t i = 0; i < seqLen; ++i)
    {
        if (textTokenIds[i] == kImStartTokenId)
        {
            int32_t const roleId = (i + 1 < seqLen) ? textTokenIds[i + 1] : -1;
            segments.push_back({i, seqLen, roleId});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].endPos = segments[s + 1].startPos;
    }

    std::vector<size_t> userSegmentIndices;
    int64_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == kSystemRoleId)
            continue;
        else if (segments[s].roleId == kUserRoleId)
            userSegmentIndices.push_back(s);
        else if (segments[s].roleId == kAssistantRoleId)
            assistantSegIdx = static_cast<int64_t>(s);
    }

    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildTalkerPrefillFromSegments: could not find assistant segment");
        return false;
    }

    // Step 4: Project multimodal tokens via hidden_projection using gather/scatter kernels
    bool const hasHiddenProjection = (mHiddenFC1Weight.getShape().volume() > 0);
    if (hasHiddenProjection && prefillHiddenPtr != nullptr)
    {
        std::vector<int32_t> mmPositions;
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos && i < prefillLen; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                    mmPositions.push_back(static_cast<int32_t>(i));
            }
        }

        if (!mmPositions.empty())
        {
            int64_t const numMM = static_cast<int64_t>(mmPositions.size());
            // Safe to repurpose mThinkerEmbedBuffer here: step 1 filled it with full-sequence
            // layer-0 embeddings, but those have already been projected into mTalkerInputEmbeds
            // by step 2's invokeTalkerMLP call. The reshape below overwrites that data, which is
            // intentional — we only need it as scratch for the gather/MLP/scatter chain.
            check::check(mThinkerEmbedBuffer.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mMLPWorkspace.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mTalkerInputEmbeds.reshape({numMM, hiddenSize}), "Tensor reshape failed");

            // Upload indices and use vectorized gather kernel instead of per-row cudaMemcpy
            check::check(mGatherIndicesBuffer.reshape({numMM}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGatherIndicesBuffer.rawPointer(), mmPositions.data(), numMM * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            rt::Tensor hiddenSource(const_cast<void*>(prefillHiddenPtr->rawPointer()),
                rt::Coords{static_cast<int64_t>(prefillLen), thinkerHiddenSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF);
            kernel::invokeGather(hiddenSource, mGatherIndicesBuffer, mThinkerEmbedBuffer, stream);

            kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                mHiddenFC2Bias, mTalkerInputEmbeds, mMLPWorkspace, stream);

            // Scatter projected multimodal embeddings back into mProjectedBuffer
            kernel::invokeScatter(mTalkerInputEmbeds, mGatherIndicesBuffer, mProjectedBuffer, stream);
        }
    }
    else if (!hasHiddenProjection || prefillHiddenPtr == nullptr)
    {
        // Zero out multimodal rows as fallback
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                {
                    __half* rowPtr = static_cast<__half*>(mProjectedBuffer.rawPointer()) + i * hiddenSize;
                    CUDA_CHECK(cudaMemsetAsync(rowPtr, 0, hiddenSize * sizeof(__half), stream));
                }
            }
        }
    }

    // Step 5: Build Talker prefill input — user segments + restructured assistant preamble
    auto const& assistantSeg = segments[assistantSegIdx];
    constexpr int32_t kAssistantRestructuredLen = kNonStreamingPrefixRows + 1;
    constexpr int32_t kAssistantTrailingOffset = kAssistantPrefixLen + 1;

    int64_t userTotalLen = 0;
    for (size_t sIdx : userSegmentIndices)
        userTotalLen += (segments[sIdx].endPos - segments[sIdx].startPos);

    outSeqLen = userTotalLen + kAssistantRestructuredLen;
    check::check(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(mTalkerInputEmbeds.rawPointer(), 0, outSeqLen * hiddenSize * sizeof(__half), stream));

    int64_t outOffset = 0;
    for (size_t sIdx : userSegmentIndices)
    {
        auto const& seg = segments[sIdx];
        int64_t const segLen = seg.endPos - seg.startPos;
        __half const* src = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + seg.startPos * hiddenSize;
        __half* dst = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outOffset * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, segLen * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        outOffset += segLen;
    }

    // Build restructured assistant preamble via invokeAssistantPreamble
    {
        __half const* const assistantProjPtr
            = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + assistantSeg.startPos * hiddenSize;
        constexpr int64_t kPreambleFullLen = kNonStreamingPrefixRows + 1 + 2;
        int64_t const requiredCapacity = (outSeqLen + kPreambleFullLen) * hiddenSize * sizeof(__half);
        check::check(static_cast<int64_t>(mTalkerInputEmbeds.getMemoryCapacity()) >= requiredCapacity,
            "mTalkerInputEmbeds too small for preamble scratch");
        __half* const scratchPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outSeqLen * hiddenSize;
        rt::Tensor preambleScratch(
            scratchPtr, rt::Coords{kPreambleFullLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        int64_t const assistantInputLen = kAssistantPrefixLen + 1;
        rt::Tensor assistantSlice(const_cast<__half*>(assistantProjPtr), rt::Coords{assistantInputLen, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Omni segment path: language conditioning is not (yet) propagated through Omni; pass langId=-1
        // to keep the legacy 8-row prefix.
        kernel::invokeAssistantPreamble(assistantSlice, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
            mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkId, mTalkerConfig.codecThinkBosId,
            mTalkerConfig.codecThinkEosId, speakerId, mTalkerConfig.codecPadId, mTalkerConfig.codecBosId,
            /*langId=*/-1, /*textLen=*/1, preambleScratch, stream);

        __half* const aOut = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + userTotalLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(aOut, scratchPtr, kAssistantRestructuredLen * hiddenSize * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Step 6: Fill trailing text hidden states
    int64_t const assistantSegLen = assistantSeg.endPos - assistantSeg.startPos;
    trailingCount = std::min(static_cast<int32_t>(assistantSegLen - kAssistantTrailingOffset),
        static_cast<int32_t>(trailingTextHidden.getShape()[0]) - 1);
    if (trailingCount > 0)
    {
        __half const* trailSrc = static_cast<__half const*>(mProjectedBuffer.rawPointer())
            + (assistantSeg.startPos + kAssistantTrailingOffset) * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(trailingTextHidden.rawPointer(), trailSrc,
            trailingCount * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    LOG_INFO("buildTalkerPrefillFromSegments: outSeqLen=%ld (user=%ld + assistant=%d), trailing=%d", outSeqLen,
        userTotalLen, kAssistantRestructuredLen, trailingCount);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//        Incremental Trailing Hidden Helpers (for streaming)
// ═══════════════════════════════════════════════════════════════════════════

void Qwen3OmniTTSRuntime::appendTrailingToken(int32_t tokenId, rt::Tensor const& thinkerEmbedTable,
    rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;

    // Upload token ID (reuse pre-allocated GPU buffer)
    CUDA_CHECK(
        cudaMemcpyAsync(mStreamingTokenId.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // embed_tokens(tokenId) → mStreamingTokenEmbed [1, thinkerH]
    // embeddingLookup expects [1, 1, H] output; mStreamingTokenEmbed is [1, H] — same memory, just reshape for kernel
    rt::Tensor embedView(mStreamingTokenEmbed.rawPointer(), rt::Coords{1, 1, mTalkerConfig.thinkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::embeddingLookup(mStreamingTokenId, thinkerEmbedTable, std::nullopt, embedView, stream);

    // text_projection: mStreamingTokenEmbed [1, thinkerH] → mStreamingProjOut [1, talkerH]
    kernel::invokeTalkerMLP(mStreamingTokenEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mStreamingProjOut, mStreamingMlpWork, stream);

    // Write to trailingTextHidden[trailingIdx]
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mStreamingProjOut.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

void Qwen3OmniTTSRuntime::finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mTtsEosEmbed.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

// ═══════════════════════════════════════════════════════════════════════════
//        Thinker-Talker Streaming Pipeline (single CUDA stream)
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::handleStreamingGeneration(LLMInferenceSpecDecodeRuntime& thinkerRuntime,
    LLMGenerationRequest& thinkerRequest, LLMGenerationResponse& thinkerResponse,
    ThinkerTalkerStreamingConfig const& streamingConfig, OmniGenerationRequest const& omniBaseRequest,
    TalkerGenerationResponse& talkerResponse, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TTSRuntime::handleStreamingGeneration", nvtx_colors::PURPLE);

    LOG_INFO(
        "Starting Thinker-Talker streaming pipeline (prefillThreshold=%d)", streamingConfig.talkerPrefillThreshold);

    talkerResponse.batchRvqCodes.clear();
    talkerResponse.numFramesPerSample.clear();
    talkerResponse.success = false;

    float const talkerTemperature = (omniBaseRequest.talkerTemperature > 0) ? omniBaseRequest.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (omniBaseRequest.talkerTopK > 0) ? omniBaseRequest.talkerTopK : 50;
    float const talkerTopP = (omniBaseRequest.talkerTopP > 0) ? omniBaseRequest.talkerTopP : 1.0f;
    float const repetitionPenalty = omniBaseRequest.repetitionPenalty;

    SamplingParams talkerSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize, talkerTemperature, talkerTopK, talkerTopP);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t numSeenTokens = 0;
    std::unordered_set<int32_t> seenTokenSet;

    struct StreamingState
    {
        std::vector<int32_t> assistantTokens;
        bool thinkerFinished{false};
        bool talkerPrefillDone{false};
        bool talkerError{false};
        int32_t talkerFrames{0};
        int32_t codecToken{-1};
        int32_t trailingIdx{0};
        int32_t lastChunkEnd{0};
        std::vector<std::vector<int32_t>> rvqCodes;
        std::vector<int32_t> inputIds;
    };
    StreamingState state;
    state.rvqCodes.reserve(omniBaseRequest.maxAudioLength);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    int32_t const maxTrailingLen = static_cast<int32_t>(trailingStride);

    // Streaming uses batch slot 0 of the trailing buffer
    size_t const slot0Bytes = trailingStride * hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(mStreamingTrailingHidden.rawPointer(), 0, slot0Bytes, stream));

    rt::Tensor const& thinkerEmbedTable = thinkerRuntime.getEmbeddingTable();

    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (omniBaseRequest.speakerId >= 0)
    {
        speakerId = omniBaseRequest.speakerId;
    }
    else if (!omniBaseRequest.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(omniBaseRequest.speakerName);
    }

    int32_t const prefillThreshold = streamingConfig.talkerPrefillThreshold;
    int32_t const maxAudioLength = omniBaseRequest.maxAudioLength;

    // Reset reuse lengths to batch=1 for streaming (per-batch independent Talker)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    reuseData[0] = 0;

    auto makeTrailingView = [&]() -> rt::Tensor {
        return rt::Tensor(mStreamingTrailingHidden.rawPointer(),
            rt::Coords{static_cast<int64_t>(state.trailingIdx), hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF);
    };

    // ===== Per-token callback for the Thinker decode loop =====
    // SAFETY: This lambda captures stack-local state by reference. It is ONLY safe because
    // thinkerRuntime.handleRequest() invokes the callback synchronously on the same thread
    // (inside cudaStreamSynchronize in the decode loop). Never call this callback asynchronously.
    auto userCallback = thinkerRequest.onTokenGenerated;

    thinkerRequest.onTokenGenerated = [&, userCallback](TokenCallbackInfo const& info) {
        if (userCallback.has_value())
        {
            userCallback.value()(info);
        }
        if (info.batchIdx != 0 || state.talkerError)
        {
            return;
        }

        state.assistantTokens.push_back(info.tokenId);
        state.thinkerFinished = info.isFinished;

        int32_t const numAssistantTokens = static_cast<int32_t>(state.assistantTokens.size());

        if (!state.talkerPrefillDone && numAssistantTokens >= prefillThreshold)
        {
            LOG_INFO("Thinker produced %d assistant tokens, triggering Talker prefill", numAssistantTokens);

            // Reset KV caches
            auto& talkerCacheManager = mTalkerLLMRunner->getCacheManager();
            auto& cpCacheManager = mCodePredictorRunner->getCacheManager();
            talkerCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            cpCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            {
                auto& talkerKVManager = talkerCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < talkerKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = talkerKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
                auto& cpKVManager = cpCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < cpKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = cpKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
            }

            // Build combined token IDs — fetch Thinker input IDs from the runtime portal.
            auto const& thinkerInputs = thinkerRuntime.getBaseModelInputTokenIds();
            auto& textTokenIds = state.inputIds;
            if (textTokenIds.empty() && !thinkerInputs.empty())
            {
                textTokenIds = thinkerInputs[0];
            }
            textTokenIds.insert(textTokenIds.end(), state.assistantTokens.begin(), state.assistantTokens.end());

            // Use buildTalkerPrefillFromSegments for segment parsing, MLP projection, and prefill assembly
            auto prefillEmbedPtr = thinkerRuntime.getBaseModelHiddenStates(0);
            auto prefillHiddenPtr = thinkerRuntime.getBaseModelHiddenStates(thinkerRequest.acceptHiddenLayer);
            int32_t const prefillLen = thinkerRuntime.getBaseModelPrefillLength();

            int64_t outSeqLen = 0;
            if (!buildTalkerPrefillFromSegments(textTokenIds, prefillEmbedPtr, prefillHiddenPtr, prefillLen,
                    thinkerEmbedTable, speakerId, mStreamingTrailingHidden, state.trailingIdx, outSeqLen, stream))
            {
                state.talkerError = true;
                return;
            }

            // Talker prefill
            check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            {
                TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
                if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
                {
                    LOG_ERROR("Talker prefill failed during streaming pipeline");
                    state.talkerError = true;
                    return;
                }
            }

            kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
                mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
            trt_edgellm::topKtopPSamplingFromLogits(
                mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            state.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
            if (seenTokenSet.insert(state.codecToken).second)
            {
                CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
                    mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
                ++numSeenTokens;
            }
            state.talkerPrefillDone = true;
            if (getProfilingEnabled())
            {
                CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
            }
            LOG_INFO("Talker prefill done, first codec token: %d", state.codecToken);

            // Generate frame 0 immediately after prefill
            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;
            }
        }
        else if (state.talkerPrefillDone && numAssistantTokens > prefillThreshold)
        {
            // Incremental: append new trailing token + run Talker decode step
            if (state.trailingIdx >= maxTrailingLen - 1)
            {
                LOG_WARNING("Trailing buffer full (%d/%d), skipping token append", state.trailingIdx, maxTrailingLen);
            }
            else
            {
                appendTrailingToken(
                    info.tokenId, thinkerEmbedTable, mStreamingTrailingHidden, state.trailingIdx, stream);
                state.trailingIdx++;
            }

            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;

                if (streamingConfig.codecChunkFrames > 0 && streamingConfig.onAudioChunkReady
                    && (state.talkerFrames - state.lastChunkEnd) >= streamingConfig.codecChunkFrames)
                {
                    std::vector<std::vector<int32_t>> chunk(
                        state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
                    streamingConfig.onAudioChunkReady(chunk);
                    state.lastChunkEnd = state.talkerFrames;
                }
            }
        }
    };

    // ===== Run Thinker with the callback installed =====
    auto hiddenLayers = getThinkerHiddenLayerIndices();
    thinkerRequest.generateAudio = true;
    thinkerRequest.acceptHiddenLayer = hiddenLayers[1];
    bool thinkerSuccess = thinkerRuntime.handleRequest(thinkerRequest, thinkerResponse, stream, true);

    if (!thinkerSuccess)
    {
        LOG_ERROR("Thinker handleRequest failed in streaming pipeline");
        return false;
    }

    // ===== After Thinker finishes: finalize trailing and flush remaining Talker frames =====
    if (state.talkerPrefillDone && !state.talkerError)
    {
        if (state.trailingIdx < maxTrailingLen)
        {
            finalizeTrailing(mStreamingTrailingHidden, state.trailingIdx, stream);
            state.trailingIdx++;
        }
        else
        {
            LOG_WARNING("Trailing buffer full, cannot append tts_eos");
        }

        LOG_INFO("Thinker done. Flushing remaining Talker frames (current: %d, codec=%d, trailingIdx=%d)",
            state.talkerFrames, state.codecToken, state.trailingIdx);

        rt::Tensor flushTrailingView = makeTrailingView();
        rt::Tensor const* flushTrailingPtr = (state.trailingIdx > 0) ? &flushTrailingView : nullptr;

        int32_t const chunkSize = streamingConfig.codecChunkFrames;

        while (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
        {
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        flushTrailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty,
                        state.rvqCodes, stream))
                {
                    break;
                }
            }
            state.talkerFrames++;

            if (chunkSize > 0 && streamingConfig.onAudioChunkReady
                && (state.talkerFrames - state.lastChunkEnd) >= chunkSize)
            {
                std::vector<std::vector<int32_t>> chunk(
                    state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
                streamingConfig.onAudioChunkReady(chunk);
                state.lastChunkEnd = state.talkerFrames;
            }
        }

        if (streamingConfig.onAudioChunkReady && state.lastChunkEnd < state.talkerFrames)
        {
            std::vector<std::vector<int32_t>> remainder(
                state.rvqCodes.begin() + state.lastChunkEnd, state.rvqCodes.begin() + state.talkerFrames);
            streamingConfig.onAudioChunkReady(remainder);
        }
    }
    else
    {
        LOG_WARNING("Thinker finished but Talker prefill was never triggered (only %zu assistant tokens)",
            state.assistantTokens.size());
    }

    bool const hitEos = (state.codecToken == codecEosId);
    LOG_INFO("Streaming pipeline: %d audio frames (exit: %s, codec=%d)", state.talkerFrames,
        hitEos ? "EOS" : "maxAudioLength", state.codecToken);

    talkerResponse.batchRvqCodes.push_back(std::move(state.rvqCodes));
    talkerResponse.numFramesPerSample.push_back(state.talkerFrames);
    talkerResponse.success = state.talkerPrefillDone && !state.talkerError;

    mMultimodalMetrics.recordRun(0, 0, 1, state.talkerFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;

        mOmniTalkerMetrics.recordRun(state.talkerFrames, state.talkerFrames * codesPerFrame, prefillMs,
            state.talkerPrefillDone ? static_cast<int32_t>(state.assistantTokens.size() + 30) : 0,
            hitEos ? "eos" : "max_length", true);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS = static_cast<float>(state.talkerFrames * 1920) / 24000.0f;
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(state.talkerFrames) * 1920;
        mOmniLatencyMetrics.sampleRate = 24000;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
