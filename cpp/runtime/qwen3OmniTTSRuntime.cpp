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

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "multimodal/qwen3_omni/cloneEncoderRunner.h"
#include "runtime/audioLoader.h"

#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "llmInferenceRuntime.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cctype>
#include <chrono>
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

// Shared chunk accumulator + emitter for streaming RVQ codes.
// Used by both runTalkerGenerationLoop (per-batch in TTS) and handleStreamingGeneration (bs=1 Omni)
// so chunk semantics (threshold-triggered non-final emit + single final flush) live in one place.
struct ChunkEmitter
{
    int32_t chunkFrames{0};
    std::function<void(std::vector<std::vector<int32_t>> const& chunk, bool isFinal)> onChunk;
    std::vector<std::vector<int32_t>> buffer;

    bool active() const
    {
        return chunkFrames > 0 && static_cast<bool>(onChunk);
    }

    // Append a frame; emit non-final chunk when buffer hits the threshold.
    void append(std::vector<int32_t> const& frame)
    {
        if (!active())
            return;
        buffer.push_back(frame);
        if (static_cast<int32_t>(buffer.size()) >= chunkFrames)
        {
            onChunk(buffer, /*isFinal=*/false);
            buffer.clear();
        }
    }

    // Final flush — always invoked once per active emitter at end-of-stream, even if buffer empty
    // (callers rely on this as the end-of-stream signal).
    void flushFinal()
    {
        if (!active())
            return;
        onChunk(buffer, /*isFinal=*/true);
        buffer.clear();
    }
};
} // anonymous namespace

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, std::string const& cloneEncoderDir, cudaStream_t stream,
    std::string const& checkpointDir)
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

    bool const runnersInitialized = initializeEngineRunners(talkerEngineDir, codePredictorEngineDir, checkpointDir);
    ELLM_CHECK(runnersInitialized, "Failed to initialize engine runners");

    // Setup shared execution context memory for Talker and CodePredictor engines.
    // Both use kUSER_MANAGED allocation and require setContextMemory() before execution.
    {
        int64_t const talkerCtxSize = mTalkerExec->getRequiredContextMemorySize();
        int64_t const cpCtxSize = mCodePredictorExec ? mCodePredictorExec->getRequiredContextMemorySize() : 0;
        int64_t const sharedCtxSize = std::max(talkerCtxSize, cpCtxSize);
        LOG_INFO("Setup shared execution context memory: %zu bytes (talker: %zu, code_predictor: %zu)",
            static_cast<size_t>(sharedCtxSize), static_cast<size_t>(talkerCtxSize), static_cast<size_t>(cpCtxSize));
        mSharedExecContextMemory = rt::Tensor({sharedCtxSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
            "Qwen3OmniTTSRuntime::mSharedExecContextMemory");
        bool const talkerCtxSet = mTalkerExec->setContextMemory(mSharedExecContextMemory);
        ELLM_CHECK(talkerCtxSet, "Failed to set context memory for Talker LLM engine");
        ELLM_CHECK(!mCodePredictorExec || mCodePredictorExec->setContextMemory(mSharedExecContextMemory),
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

    // Prefill row builder for non-fused layouts (instruction segments, VoiceDesign
    // no-speaker prefixes). Configured after embeddings/tables are resident.
    {
        int64_t const maxRows = mTalkerLLMConfig.maxSupportedInputLength;
        int64_t const stagingBytes = maxRows * static_cast<int64_t>(sizeof(kernel::PrefillRowDesc));
        mPrefillRows.reserve(maxRows);
        mPrefillDescsHost
            = rt::Tensor({stagingBytes}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT8, "prefillDescsHost");
        mPrefillDescsDevice
            = rt::Tensor({stagingBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "prefillDescsDevice");
    }

    // Voice clone workspace (Base checkpoints): x-vector slot, reference codes/frame sums for
    // the ICL prefill, and the per-group embedding-table pointer array for the sum kernel.
    {
        constexpr int64_t kMaxRefFrames = 512; // ~40s reference audio at 12.5 Hz
        int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
        mVoiceCloneXVector = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "cloneXVector");
        mIclFrameSumBuffer
            = rt::Tensor({kMaxRefFrames, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "iclFrameSums");

        std::vector<void const*> tablePtrs(mTalkerConfig.numCodeGroups);
        tablePtrs[0] = mTalkerEmbeddingTable.rawPointer();
        for (int32_t g = 1; g < mTalkerConfig.numCodeGroups; ++g)
        {
            tablePtrs[g] = mCodePredictorEmbeddingTables[g - 1].rawPointer();
        }
        mIclTablePtrsGpu = rt::Tensor({static_cast<int64_t>(tablePtrs.size() * sizeof(void*))}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "iclTablePtrs");
        CUDA_CHECK(cudaMemcpyAsync(mIclTablePtrsGpu.rawPointer(), tablePtrs.data(), tablePtrs.size() * sizeof(void*),
            cudaMemcpyHostToDevice, stream));
    }

    if (!cloneEncoderDir.empty())
    {
        mCloneEncoders = std::make_unique<CloneEncoderRunner>(cloneEncoderDir, stream);
    }

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
    std::string const& talkerEngineDir, std::string const& codePredictorEngineDir, std::string const& checkpointDir)
{
    // Load Talker LLM engine via EngineExecutor (migrated from LLMEngineRunner)
    std::filesystem::path talkerEnginePath = std::filesystem::path(talkerEngineDir) / "llm.engine";
    std::filesystem::path talkerConfigPath = std::filesystem::path(talkerEngineDir) / "config.json";

    LOG_INFO("Loading Talker LLM engine from: %s", talkerEnginePath.string().c_str());

    try
    {
        mTalkerLLMConfig = rt::parseEngineConfig(talkerConfigPath);
        mTalkerExec = rt::EngineExecutor::createForLLM(talkerEnginePath, mTalkerLLMConfig);
        rt::validateAgainstEngine(mTalkerLLMConfig, *mTalkerExec, "qwen3_omni_talker");
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mTalkerSharedRes = rt::SharedResources::createForLLM(mTalkerLLMConfig, emptyLoraMap, mStream);
        mTalkerPipelineIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mTalkerLLMConfig, mStream));
        mTalkerStepPreparer = std::make_unique<rt::StepPreparer>(mTalkerLLMConfig);
        rt::buildTensorMap(
            mTalkerTensorMap, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, /*kvCacheIndex=*/0);
        mTalkerSharedRes->externalWeightManager->load(talkerEngineDir, talkerConfigPath, mStream, checkpointDir);
        mTalkerSharedRes->externalWeightManager->validateAgainstEngine(*mTalkerExec, "talker");
        mTalkerSharedRes->externalWeightManager->registerTensorMapEntries(mTalkerTensorMap);

        LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
            mTalkerLLMConfig.hiddenSize);
        auto talkerKVType = mTalkerLLMConfig.kvCacheDtype;
        LOG_INFO("Talker KV cache dtype: %s",
            talkerKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                      : (talkerKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", e.what());
        return false;
    }

    // Load CodePredictor engine via EngineExecutor (migrated from LLMEngineRunner)
    std::filesystem::path codePredictorEnginePath = std::filesystem::path(codePredictorEngineDir) / "llm.engine";
    std::filesystem::path codePredictorConfigPath = std::filesystem::path(codePredictorEngineDir) / "config.json";

    LOG_INFO("Loading CodePredictor engine from: %s", codePredictorEnginePath.string().c_str());

    try
    {
        mCodePredictorConfig = rt::parseEngineConfig(codePredictorConfigPath);
        mCodePredictorExec = rt::EngineExecutor::createForLLM(codePredictorEnginePath, mCodePredictorConfig);
        rt::validateAgainstEngine(mCodePredictorConfig, *mCodePredictorExec, "qwen3_omni_code_predictor");
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mCodePredictorSharedRes = rt::SharedResources::createForLLM(mCodePredictorConfig, emptyLoraMap, mStream);
        mCodePredictorPipelineIO
            = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mCodePredictorConfig, mStream));
        mCodePredictorStepPreparer = std::make_unique<rt::StepPreparer>(mCodePredictorConfig);
        rt::buildTensorMap(mCodePredictorTensorMap, *mCodePredictorPipelineIO, *mCodePredictorSharedRes,
            mCodePredictorConfig, /*kvCacheIndex=*/0);
        mCodePredictorSharedRes->externalWeightManager->load(
            codePredictorEngineDir, codePredictorConfigPath, mStream, checkpointDir);
        mCodePredictorSharedRes->externalWeightManager->validateAgainstEngine(*mCodePredictorExec, "code predictor");
        mCodePredictorSharedRes->externalWeightManager->registerTensorMapEntries(mCodePredictorTensorMap);

        // CodePredictor ONNX outputs FP32 logits directly (the active lm_head is gathered
        // in-engine from the stacked lm_heads input by the device lm_head_idx).

        // Read CodePredictor dimensions from loaded config (vocab_size==hidden_size since
        // engine output is last_hidden; real codebook_size is inferred from lm_head shape later)
        mTalkerConfig.codePredictorHiddenSize = mCodePredictorConfig.hiddenSize;

        LOG_INFO("CodePredictor engine loaded: vocabSize=%d, hiddenSize=%d, numLayers=%d",
            mCodePredictorConfig.vocabSize, mCodePredictorConfig.hiddenSize, mCodePredictorConfig.numDecoderLayers);
        auto cpKVType = mCodePredictorConfig.kvCacheDtype;
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
    mTalkerConfig.codecThinkId = configJson.value("codec_think_id", -1);
    mTalkerConfig.codecThinkBosId = configJson["codec_think_bos_id"].get<int32_t>();
    mTalkerConfig.codecThinkEosId = configJson["codec_think_eos_id"].get<int32_t>();

    // Optional style/emotion instruction and language conditioning tables (OmniNext).
    if (configJson.contains("talker_assistant_prompt_id_mapping")
        && configJson["talker_assistant_prompt_id_mapping"].is_object())
    {
        for (auto const& [name, ids] : configJson["talker_assistant_prompt_id_mapping"].items())
        {
            if (!ids.is_array())
            {
                continue;
            }
            std::vector<int32_t> tokens;
            tokens.reserve(ids.size());
            for (auto const& tid : ids)
            {
                tokens.push_back(tid.get<int32_t>());
            }
            mAssistantPromptIds[name] = std::move(tokens);
        }
        LOG_INFO("Loaded %zu talker_assistant_prompt_id_mapping entries", mAssistantPromptIds.size());
    }
    if (configJson.contains("talker_language_id") && configJson["talker_language_id"].is_object())
    {
        for (auto const& [lang, id] : configJson["talker_language_id"].items())
        {
            if (id.is_number_integer())
            {
                mLanguageIds[lang] = id.get<int32_t>();
            }
        }
        LOG_INFO("Loaded %zu talker_language_id entries", mLanguageIds.size());
    }
    mTalkerConfig.codecPadId = configJson["codec_pad_id"].get<int32_t>();
    mTalkerConfig.codecBosId = configJson["codec_bos_id"].get<int32_t>();
    // Support both codec_eos_token_id (original) and codec_eos_id (legacy) for backward compatibility
    if (configJson.contains("codec_eos_token_id"))
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_token_id"].get<int32_t>();
    }
    else
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_id"].get<int32_t>();
    }

    // Checkpoint family: custom_voice / voice_design / base; "" for Qwen3-Omni or legacy configs.
    mTalkerConfig.ttsModelType = configJson.value("tts_model_type", "");

    // CustomVoice language conditioning (optional; absent for Qwen3-Omni checkpoints and
    // engines exported before language support — resolveLanguageId then always returns -1).
    mTalkerConfig.codecThinkId = configJson.value("codec_think_id", -1);
    if (configJson.contains("codec_language_id") && configJson["codec_language_id"].is_object())
    {
        for (auto const& [languageName, codecId] : configJson["codec_language_id"].items())
        {
            std::string key = languageName;
            std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
            mTalkerConfig.codecLanguageIdMap[key] = codecId.get<int32_t>();
        }
        LOG_INFO("Loaded %zu language IDs from config (codec_think_id=%d)", mTalkerConfig.codecLanguageIdMap.size(),
            mTalkerConfig.codecThinkId);
    }
    if (configJson.contains("spk_is_dialect") && configJson["spk_is_dialect"].is_object())
    {
        // Values are heterogeneous: JSON false for non-dialect speakers, a dialect-name string
        // (e.g. "sichuan_dialect") for dialect speakers. Only the strings are kept.
        for (auto const& [speakerName, dialect] : configJson["spk_is_dialect"].items())
        {
            if (dialect.is_string())
            {
                std::string key = speakerName;
                std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
                mTalkerConfig.spkDialectMap[key] = dialect.get<std::string>();
            }
        }
        if (!mTalkerConfig.spkDialectMap.empty())
        {
            LOG_INFO("Loaded %zu dialect speakers from config", mTalkerConfig.spkDialectMap.size());
        }
    }

    // Speaker ID configuration
    mTalkerConfig.defaultSpeakerId = configJson.value("default_speaker_id", 2301);

    // Decoder layer index whose pre-norm hidden_states the Talker consumes from
    // the Thinker. Must match the layer the Thinker engine was exported to
    // emit on its ``hidden_states`` output (Qwen3-Omni dense default 14,
    // Qwen3-Next Omni reads it from talker_config.accept_hidden_layer). Sentinel
    // -1 (the struct default) means "unconfigured"; only the standalone-TTS
    // path that never invokes the streaming hidden-states portal is allowed
    // to leave it that way.
    mTalkerConfig.acceptHiddenLayer = configJson.value("accept_hidden_layer", mTalkerConfig.acceptHiddenLayer);

    // Chat-template / placeholder token IDs. Read from config.json so the
    // talker pipeline stays variant-agnostic — every engine carries its own
    // tokenizer's IDs at export time. Defaults are the Qwen3-Omni values
    // baked into the TalkerConfig struct so legacy engines keep working.
    mTalkerConfig.imStartTokenId = configJson.value("im_start_token_id", mTalkerConfig.imStartTokenId);
    mTalkerConfig.assistantRoleId = configJson.value("assistant_role_id", mTalkerConfig.assistantRoleId);
    mTalkerConfig.userRoleId = configJson.value("user_role_id", mTalkerConfig.userRoleId);
    mTalkerConfig.systemRoleId = configJson.value("system_role_id", mTalkerConfig.systemRoleId);
    mTalkerConfig.audioTokenId = configJson.value("audio_token_id", mTalkerConfig.audioTokenId);
    mTalkerConfig.imageTokenId = configJson.value("image_token_id", mTalkerConfig.imageTokenId);
    mTalkerConfig.videoTokenId = configJson.value("video_token_id", mTalkerConfig.videoTokenId);
    mTalkerConfig.thinkOpenTokenId = configJson.value("think_open_token_id", mTalkerConfig.thinkOpenTokenId);
    mTalkerConfig.thinkCloseTokenId = configJson.value("think_close_token_id", mTalkerConfig.thinkCloseTokenId);
    mTalkerConfig.maxThinkerToTalkerMmTokens
        = configJson.value("max_thinker_to_talker_mm_tokens", mTalkerConfig.maxThinkerToTalkerMmTokens);

    // Load speaker ID mapping if available
    if (configJson.contains("speaker_id") && configJson["speaker_id"].is_object())
    {
        for (auto const& [speaker_name, speaker_id] : configJson["speaker_id"].items())
        {
            mSpeakerIdMap[speaker_name] = speaker_id.get<int32_t>();
        }
        LOG_INFO("Loaded %zu speaker IDs from config", mSpeakerIdMap.size());

        // Qwen3-Next Omni: load speaker_system_prompt_id (speaker_name → token list) and index by
        // speaker_id so the prefill builder can look up rows for a given numeric speakerId.
        if (configJson.contains("speaker_system_prompt_id") && configJson["speaker_system_prompt_id"].is_object())
        {
            for (auto const& [speaker_name, ids] : configJson["speaker_system_prompt_id"].items())
            {
                auto it = mSpeakerIdMap.find(speaker_name);
                if (it == mSpeakerIdMap.end())
                {
                    continue;
                }
                std::vector<int32_t> tokens;
                tokens.reserve(ids.size());
                for (auto const& tid : ids)
                {
                    tokens.push_back(tid.get<int32_t>());
                }
                mSpeakerSystemPromptIds[it->second] = std::move(tokens);
            }
            LOG_INFO("Loaded %zu speaker_system_prompt_id entries", mSpeakerSystemPromptIds.size());
        }

        // Friendly speaker aliases: voice_map.json maps display names (e.g. "Ryan")
        // to internal speaker names (e.g. "m36"). Optional — internal names keep working.
        std::filesystem::path const voiceMapPath = std::filesystem::path(talkerEngineDir) / "voice_map.json";
        if (std::filesystem::exists(voiceMapPath))
        {
            std::ifstream vmStream(voiceMapPath);
            nlohmann::json vm;
            try
            {
                vmStream >> vm;
                for (auto const& [friendlyName, internalName] : vm.items())
                {
                    if (!internalName.is_string())
                    {
                        continue;
                    }
                    std::string key = friendlyName;
                    std::transform(
                        key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
                    mVoiceAliasMap[std::move(key)] = internalName.get<std::string>();
                }
                LOG_INFO("Loaded %zu voice_map.json aliases", mVoiceAliasMap.size());
            }
            catch (std::exception const& e)
            {
                LOG_WARNING("Failed to parse %s: %s", voiceMapPath.string().c_str(), e.what());
            }
        }

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
        // Stack the heads into one [numHeads, vocab, hidden] tensor; the engine
        // gathers the active head by the device lm_head_idx.
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
            if (i == 0)
            {
                int64_t const vocab = it->getShape()[0];
                int64_t const hidden = it->getShape()[1];
                mTalkerConfig.codebookSize = static_cast<int32_t>(vocab);
                mCodePredictorLmHeads = rt::Tensor(
                    {mNumRvqLayers, vocab, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "lm_heads");
            }
            check::check(it->getShape()[0] == mCodePredictorLmHeads.getShape()[1]
                    && it->getShape()[1] == mCodePredictorLmHeads.getShape()[2],
                "lm_head shape mismatch across heads");
            int64_t const headBytes = it->getMemoryCapacity();
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mCodePredictorLmHeads.rawPointer()) + i * headBytes,
                it->rawPointer(), headBytes, cudaMemcpyDeviceToDevice, mStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }
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
        mSeenSeedHostScratch
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mSeenSeedHostScratch");

        // CodePredictor workspace — sized to maxBS so any batch in [1, maxBS] just reshapes per-call.
        // Same pattern as Talker: framework primitives (EngineExecutor / StepPreparer) are
        // batch-size-agnostic; the per-call reshape({activeBatchSize, ...}) makes them work.
        mCodePredictorLogits = rt::Tensor({maxBS, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "mCodePredictorLogits");

        // Device-selected lm_head index; contents advance per step, bindings stay put.
        mCpLmHeadIdx = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCpLmHeadIdx");

        mCodePredictorPrefillInput = rt::Tensor({maxBS, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorPrefillInput");
        mCodePredictorCodecIds
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorCodecIds");
        mCodePredictorCodecEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorCodecEmbed");
        mRawCodecEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mRawCodecEmbed");
        mSmallToMtpProjectedHidden = rt::Tensor({maxBS, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mSmallToMtpProjectedHidden");
        mCodePredictorSelectedIndices
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorSelectedIndices");
        mHostSelectedCodeIds
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedCodeIds");
        // Pinned (cudaMallocHost) buffer that accumulates the deferred CP gen samples
        // across all (mNumRvqLayers - 1) steps for up to maxBS active batches, so we
        // can do a single cudaStreamSynchronize per frame instead of one per step.
        // Layout: [step_idx, batch_idx] — step k writes to row k.
        mHostGenCodeBuf = rt::Tensor({static_cast<int64_t>(mNumRvqLayers - 1), maxBS}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "mHostGenCodeBuf");
        mHostCodePredictorContextLength
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostCodePredictorContextLength");

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
        mCodePredictorHiddenStatesBuffer = rt::Tensor({maxBS, mNumCodesPerFrame, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");
        mTalkerLastHidden = rt::Tensor(
            {maxBS, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mTalkerLastHidden");
        mCodecHiddensBuffer = rt::Tensor({maxBS, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodecHiddensBuffer");

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

        // OmniNext persistent buffers; values filled by initializeTTSEmbeddings.
        mQwen3OmniNextZeroResidualAddend = rt::Tensor(
            {talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mQwen3OmniNextZeroResidualAddend");
        mNegInfConst = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mNegInfConst");
        int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
        mCodecEmbPtrTable = rt::Tensor({numCodeGroups, static_cast<int64_t>(sizeof(__half const*))},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mCodecEmbPtrTable");
        mCodecEmbVocabSizes
            = rt::Tensor({numCodeGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodecEmbVocabSizes");
        mCodecRowCodes
            = rt::Tensor({numCodeGroups}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "mCodecRowCodes");

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

    std::filesystem::path const textProjPath = std::filesystem::path(weightsDir) / "text_projection.safetensors";
    std::filesystem::path const hiddenProjPath = std::filesystem::path(weightsDir) / "hidden_projection.safetensors";

    // Detect Qwen3-Next Omni Talker: text_projection.safetensors absent + hidden_projection.safetensors
    // contains exactly 2 tensors (weight, bias for a single Linear) rather than the 4-tensor 2-layer
    // MLP layout used by Qwen3-Omni / Qwen3-TTS. In that mode text tokens are looked up directly in
    // Talker's own embed_tokens (already at talker hidden dim) — no text projection MLP — and
    // multimodal tokens go through a single Linear.
    mTalkerVariant = TalkerVariant::Omni;
    if (!std::filesystem::exists(textProjPath) && std::filesystem::exists(hiddenProjPath))
    {
        std::vector<rt::Tensor> probeTensors;
        if (safetensors::loadSafetensors(hiddenProjPath, probeTensors, stream) && probeTensors.size() == 2)
        {
            mTalkerVariant = TalkerVariant::OmniNext;
        }
    }

    // HF talker_supppressed_tokens range = [vocab_size - K, vocab_size). K is
    // 1024 for Qwen3-Omni and 3072 for Qwen3-Omni-Next; see
    // ``modeling_qwen3_omni{,_next}.py::generate``. Without this the Talker
    // can sample reserved tokens that have no codec_embedding entry, producing
    // garbage codec ids that fold into mispronunciations / inserted syllables.
    int32_t const kSuppressTailSize = isOmniNext() ? 3072 : 1024;
    mTalkerConfig.talkerSuppressStart = std::max(0, mTalkerConfig.talkerVocabSize - kSuppressTailSize);

    if (isOmniNext())
    {
        // Qwen3-Next Omni: single Linear hidden_projection, no text_projection.
        std::vector<rt::Tensor> hiddenTensors;
        if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
        {
            LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
            return false;
        }
        // Locate weight + bias by shape (weight is 2D, bias is 1D) — the safetensors loader does not
        // preserve original key names, so match by rank instead.
        for (auto& t : hiddenTensors)
        {
            int32_t const ndims = t.getShape().getNumDims();
            if (ndims == 2)
            {
                mHiddenProjLinearWeight = std::move(t);
            }
            else if (ndims == 1)
            {
                mHiddenProjLinearBias = std::move(t);
            }
        }
        check::check(mHiddenProjLinearWeight.rawPointer() != nullptr && mHiddenProjLinearBias.rawPointer() != nullptr,
            "hidden_projection.safetensors must contain a 2D weight and a 1D bias");
        mIsOmni = false; // text embedding comes from talkerEngineDir/text_embedding.safetensors
        // Qwen3-Next Omni uses a different chat-template token set (vocab 248320).
        // Engines exported with up-to-date ``_export_omni_next_talker`` carry the
        // IDs in talker config.json already (the ``configJson.value(...)`` reads
        // above pick them up); this block patches legacy engines whose
        // config.json predates the export-side update so they keep working.
        if (mTalkerConfig.imStartTokenId == talker_constants::kImStartTokenId)
        {
            mTalkerConfig.imStartTokenId = talker_constants::kImStartTokenIdNext;
            mTalkerConfig.assistantRoleId = talker_constants::kAssistantRoleIdNext;
            mTalkerConfig.userRoleId = talker_constants::kUserRoleIdNext;
            mTalkerConfig.systemRoleId = talker_constants::kSystemRoleIdNext;
            mTalkerConfig.audioTokenId = talker_constants::kAudioTokenIdNext;
            mTalkerConfig.imageTokenId = talker_constants::kImageTokenIdNext;
            mTalkerConfig.videoTokenId = talker_constants::kVideoTokenIdNext;
        }
        if (mTalkerConfig.thinkOpenTokenId != talker_constants::kThinkOpenTokenIdNext)
        {
            mTalkerConfig.thinkOpenTokenId = talker_constants::kThinkOpenTokenIdNext;
            mTalkerConfig.thinkCloseTokenId = talker_constants::kThinkCloseTokenIdNext;
        }
        LOG_INFO(
            "Qwen3-Next Omni Talker detected: single-Linear hidden_projection loaded from %s "
            "(weight [%lld, %lld], bias [%lld])",
            hiddenProjPath.string().c_str(), mHiddenProjLinearWeight.getShape()[0],
            mHiddenProjLinearWeight.getShape()[1], mHiddenProjLinearBias.getShape()[0]);

        // Load Talker codec_embedding (distinct from text embed_tokens — see header comment).
        std::filesystem::path const codecEmbedPath = std::filesystem::path(weightsDir) / "codec_embedding.safetensors";
        std::vector<rt::Tensor> codecEmbedTensors;
        if (!safetensors::loadSafetensors(codecEmbedPath, codecEmbedTensors, stream) || codecEmbedTensors.empty())
        {
            LOG_ERROR(
                "Qwen3-Next Omni Talker requires codec_embedding.safetensors at %s", codecEmbedPath.string().c_str());
            return false;
        }
        mTalkerCodecEmbedTable = std::move(codecEmbedTensors[0]);
        LOG_INFO("Qwen3-Next Omni Talker codec_embedding loaded: [%lld, %lld]", mTalkerCodecEmbedTable.getShape()[0],
            mTalkerCodecEmbedTable.getShape()[1]);

        // Load speaker_codec_embeddings (int64 templates per speaker).
        std::filesystem::path const spkPath
            = std::filesystem::path(weightsDir) / "speaker_codec_embeddings.safetensors";
        std::vector<rt::Tensor> spkTensors;
        if (!safetensors::loadSafetensors(spkPath, spkTensors, stream) || spkTensors.empty())
        {
            LOG_ERROR(
                "Qwen3-Next Omni Talker requires speaker_codec_embeddings.safetensors at %s", spkPath.string().c_str());
            return false;
        }
        mSpeakerCodecEmbeddings = std::move(spkTensors[0]);
        auto const& spkShape = mSpeakerCodecEmbeddings.getShape();
        LOG_INFO("Qwen3-Next Omni Talker speaker_codec_embeddings loaded: [%lld, %lld, %lld]", spkShape[0], spkShape[1],
            spkShape[2]);
    }
    else
    {
        // Qwen3-Omni / Qwen3-TTS: 2-layer MLP text_projection + optional 2-layer MLP hidden_projection.
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

        if (std::filesystem::exists(hiddenProjPath))
        {
            mIsOmni = true;
            std::vector<rt::Tensor> hiddenTensors;
            if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
            {
                LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
                return false;
            }
            if (!extractMLPWeightsFromTensors(hiddenTensors, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                    mHiddenFC2Bias, "hidden_projection"))
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
    if (isOmniNext())
    {
        // Qwen3-Next Omni: Talker's own embed_tokens already produces talker-hidden vectors —
        // no projection needed. mTextEmbeddingTable.shape[1] equals talkerHiddenSize here.
        CUDA_CHECK(cudaMemcpyAsync(ttsProjected.rawPointer(), ttsRaw.rawPointer(),
            static_cast<size_t>(kNumTtsTokens) * thinkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeTalkerMLP(
            ttsRaw, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, ttsProjected, workspace, stream);
    }

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

    // OmniNext decode addend is zero (text is fed via re-prefill, not per step).
    CUDA_CHECK(cudaMemsetAsync(mQwen3OmniNextZeroResidualAddend.rawPointer(), 0, embedSize, stream));

    float const ninf = -std::numeric_limits<float>::infinity();
    CUDA_CHECK(cudaMemcpyAsync(mNegInfConst.rawPointer(), &ninf, sizeof(float), cudaMemcpyHostToDevice, stream));

    if (isOmniNext() && mTalkerCodecEmbedTable.rawPointer() != nullptr && !mCodePredictorEmbeddingTables.empty())
    {
        check::check(buildCodecEmbedPointerTable(stream), "buildCodecEmbedPointerTable failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    LOG_INFO("TTS embeddings initialized");
}

bool Qwen3OmniTTSRuntime::projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t speakerId, int32_t languageId,
    rt::Tensor& output, int64_t& outputSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;

    // N = text tokens after stripping 3-token role prefix and 5-token suffix
    int64_t const N = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
    // Non-streaming prefill: prefix rows (8, or 9 with language) + N text rows + 2 suffix rows
    int32_t const prefixRows = (languageId >= 0) ? kPrefixRowsWithLanguage : kNonStreamingPrefixRows;
    outputSeqLen = prefixRows + N + 2; // = seqLen + 2 (+1 with language)
    LOG_INFO(
        "projectToTalkerInput: seqLen=%ld, N=%ld (stripped prefix=%d suffix=%d), outputSeqLen=%ld, speakerId=%d, "
        "languageId=%d, prefixRows=%d",
        seqLen, N, kAssistantPrefixLen, kAssistantTrailingSuffix, outputSeqLen, speakerId, languageId, prefixRows);

    // Project all tokens via text_projection MLP (Qwen3-Omni) or copy directly (Qwen3-Next Omni
    // — Talker's own embed_tokens is already at talker hidden dim).
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    if (isOmniNext())
    {
        CUDA_CHECK(cudaMemcpyAsync(mProjectedBuffer.rawPointer(), thinkerEmbed.rawPointer(),
            static_cast<size_t>(seqLen) * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
        kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
            mProjectedBuffer, mMLPWorkspace, stream);
    }

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, mTalkerConfig.codecThinkId, languageId,
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
    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mHostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    talkerCacheMgr.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Zero Mamba SSM buffers — engine inputs, not covered by resetForNewSequences.
    talkerCacheMgr.getMambaCacheManager().clearStates(stream);

    // Stage per-batch context lengths into PipelineIO's host buffer.
    // StepPreparer consumes hostContextLengths to fill GPU contextLengths + selectTokenIndices.
    check::check(mTalkerPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostContextLength = mTalkerPipelineIO->hostContextLengths.dataPointer<int32_t>();
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
    // Reshape logits to match the prefill batch size (CUDA graph capture may leave it at maxBS).
    check::check(outputLogits.reshape({batchSize, outputLogits.getShape()[outputLogits.getShape().getNumDims() - 1]}),
        "Tensor reshape failed");

    // Bind step-specific tensors into the Talker TensorMap. Engine I/O bindings whose
    // address/shape change per step (inputs_embeds, logits, hidden_states) get rewired here;
    // KV cache and rope cache were registered statically by buildTensorMap.
    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    // Prepare per-step metadata (selectTokenIndices, contextLengths, kvcache_start_index sentinel).
    mTalkerStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), talkerCacheMgr, *mTalkerPipelineIO, stream);

    bool const kvAllEmpty = talkerCacheMgr.getKVCacheAllEmpty();
    auto const prefillDims = mTalkerLLMConfig.prefillDims(batchSize, seqLen, kvAllEmpty);
    if (!mTalkerExec->prepare(/*prefillProfile=*/0, prefillDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker prefill prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker prefill execute failed");
        return false;
    }
    talkerCacheMgr.commitSequenceLength(mTalkerPipelineIO->contextLengths, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerDecodingStep(
    rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerDecodingStep", nvtx_colors::PURPLE);

    auto inputShape = inputEmbeds.getShape();
    check::check(inputShape.getNumDims() == 3, "executeTalkerDecodingStep: inputEmbeds must be 3D [batch, 1, hidden]");
    int64_t const batchSize = inputShape[0];
    check::check(batchSize <= mMaxBatchSize, "executeTalkerDecodingStep: batchSize exceeds maxBatchSize");

    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];

    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    mTalkerStepPreparer->prepare(
        rt::InferencePhase::kDecode, static_cast<int32_t>(batchSize), talkerCacheMgr, *mTalkerPipelineIO, stream);

    auto const decodeDims = mTalkerLLMConfig.decodeDims(batchSize);
    if (!mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker decode prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker decode execute failed");
        return false;
    }
    // Vanilla decode advances the KV cache by exactly 1 token per call (matches kVANILLA_DECODE_INCREMENT).
    talkerCacheMgr.commitSequenceLength(/*increment=*/1, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::executeCodePredictorPrefillStep(rt::Tensor const& inputsEmbeds, int32_t lmHeadIdx,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorPrefillStep", nvtx_colors::ORANGE);

    // Batch dim is implicit in input shape — same pattern as executeTalkerPrefillStep / spec decode.
    auto inputShape = inputsEmbeds.getShape();
    check::check(inputShape.getNumDims() == 3,
        "executeCodePredictorPrefillStep: inputsEmbeds must be 3D [batch, seqLen, cpHidden]");
    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    check::check(batchSize <= mMaxBatchSize, "executeCodePredictorPrefillStep: batchSize exceeds maxBatchSize");

    auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];

    // Reset CP KV cache for this batch (CP resets every frame — no cross-frame state).
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill_n(reuseData, batchSize, 0);
    cpCacheMgr.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Stage per-batch host context lengths (all = kCodePredictorPrefillSeqLen).
    check::check(mCodePredictorPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostCtxLen = mCodePredictorPipelineIO->hostContextLengths.dataPointer<int32_t>();
    std::fill_n(hostCtxLen, batchSize, static_cast<int32_t>(seqLen));

    check::check(lmHeadIdx == 0, "CP prefill always uses lm_head 0");
    CUDA_CHECK(cudaMemsetAsync(mCpLmHeadIdx.rawPointer(), 0, sizeof(int32_t), stream));

    mCodePredictorTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputsEmbeds));
    mCodePredictorTensorMap.set(binding_names::kLogits, outputLogits);
    mCodePredictorTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);
    mCodePredictorTensorMap.set(binding_names::kLmHeads, mCodePredictorLmHeads);
    mCodePredictorTensorMap.set(binding_names::kLmHeadIdx, mCpLmHeadIdx);

    mCodePredictorStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), cpCacheMgr, *mCodePredictorPipelineIO, stream);

    bool const kvAllEmpty = cpCacheMgr.getKVCacheAllEmpty();
    auto const prefillDims = mCodePredictorConfig.prefillDims(batchSize, seqLen, kvAllEmpty);
    if (!mCodePredictorExec->prepare(/*prefillProfile=*/0, prefillDims, mCodePredictorTensorMap, stream))
    {
        LOG_ERROR("CodePredictor prefill prepare failed");
        return false;
    }
    if (!mCodePredictorExec->execute(stream))
    {
        LOG_ERROR("CodePredictor prefill execute failed");
        return false;
    }
    cpCacheMgr.commitSequenceLength(mCodePredictorPipelineIO->contextLengths, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::prepareCpDecodeBindings(int32_t activeBatchSize, cudaStream_t stream)
{
    int64_t const cpH = mTalkerConfig.codePredictorHiddenSize;

    // Step-invariant decode bindings: the stacked lm_heads plus the device head
    // index. One prepare covers the whole per-frame loop; per-step execute() then
    // hash-hits the graph captured for this batch size.
    check::check(mCodePredictorCodecEmbed.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
    check::check(mCodePredictorHiddenStatesBuffer.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");

    // Shape the step metadata (kvcache_start_index et al) for decode BEFORE binding —
    // they are prefill-shaped coming out of the frame's prefill, and a mismatch here
    // changes the binding hash so execute() would never hit the captured graph.
    mCodePredictorStepPreparer->prepare(rt::InferencePhase::kDecode, activeBatchSize,
        *mCodePredictorSharedRes->cacheManagers[0], *mCodePredictorPipelineIO, stream);

    mCodePredictorTensorMap.set(binding_names::kInputsEmbeds, mCodePredictorCodecEmbed);
    mCodePredictorTensorMap.set(binding_names::kLogits, mCodePredictorLogits);
    mCodePredictorTensorMap.set(binding_names::kOutputHiddenStates, mCodePredictorHiddenStatesBuffer);
    mCodePredictorTensorMap.set(binding_names::kLmHeads, mCodePredictorLmHeads);
    mCodePredictorTensorMap.set(binding_names::kLmHeadIdx, mCpLmHeadIdx);

    auto const decodeDims = mCodePredictorConfig.decodeDims(activeBatchSize);
    if (!mCodePredictorExec->prepare(/*decodeProfile=*/1, decodeDims, mCodePredictorTensorMap, stream))
    {
        LOG_ERROR("CP decode prepare failed (bs=%d)", activeBatchSize);
        return false;
    }
    return true;
}

// ========== CUDA Graph Capture ==========

bool Qwen3OmniTTSRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    std::string const emptyLoraWeightsName = "";

    // Talker: capture for all supported batch sizes (1..maxBatchSize).
    // EngineExecutor::captureGraph() hashes the current binding state, so each bs
    // produces its own graph slot automatically.
    bool captureStatus{true};
    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];
    for (int32_t bs = 1; bs <= mMaxBatchSize; ++bs)
    {
        // Simulate a mid-sequence decode state for capture (matches `simulateCacheLength=128`),
        // so the captured graph reflects realistic plugin shapes / KV-length math.
        constexpr int32_t kSimulateCacheLength{128};
        std::vector<int32_t> reuseLens(bs, kSimulateCacheLength);
        rt::Tensor simulatedReuse(reuseLens.data(), rt::Coords{bs}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        talkerCacheMgr.resetForNewSequences(simulatedReuse, stream);

        check::check(mResidualEmbedBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mTalkerHiddenStatesBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(mTalkerLogits.reshape({bs, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");

        mTalkerTensorMap.set(binding_names::kInputsEmbeds, mResidualEmbedBuffer);
        mTalkerTensorMap.set(binding_names::kLogits, mTalkerLogits);
        mTalkerTensorMap.set(binding_names::kOutputHiddenStates, mTalkerHiddenStatesBuffer);

        mTalkerStepPreparer->prepare(rt::InferencePhase::kDecode, bs, talkerCacheMgr, *mTalkerPipelineIO, stream);

        auto const decodeDims = mTalkerLLMConfig.decodeDims(bs);
        captureStatus &= mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream);
        captureStatus &= mTalkerExec->captureGraph(stream);
    }
    // Restore Talker KV cache to "empty" state for the first real prefill.
    // The simulated-cache-length init above leaves both the per-batch lengths AND
    // the engine-written stale KV contents in mid-sequence state; resetForNewSequences
    // with zero lengths is what every real prefill expects.
    {
        std::vector<int32_t> zeroLens(mMaxBatchSize, 0);
        rt::Tensor zeroReuse(
            zeroLens.data(), rt::Coords{mMaxBatchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        talkerCacheMgr.resetForNewSequences(zeroReuse, stream);
    }

    // CodePredictor: the head is gathered in-engine by the device lm_head_idx, so
    // decode bindings are step-invariant — one graph per batch size covers all steps.
    check::check(mCodePredictorExec->hasIOTensor(binding_names::kLmHeads),
        "CP engine has no lm_heads input; re-export the CodePredictor with the current exporter");
    auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];
    for (int32_t bs = 1; bs <= mMaxBatchSize; ++bs)
    {
        // Simulate a mid-sequence CP decode state for capture, clamped to engine capacity.
        int32_t const simLen = std::min(128, mCodePredictorConfig.maxKVCacheCapacity - 1);
        std::vector<int32_t> simLens(bs, simLen);
        rt::Tensor simReuse(simLens.data(), rt::Coords{bs}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cpCacheMgr.resetForNewSequences(simReuse, stream);

        CUDA_CHECK(cudaMemsetAsync(mCpLmHeadIdx.rawPointer(), 0, sizeof(int32_t), stream));
        captureStatus &= prepareCpDecodeBindings(bs, stream);
        captureStatus &= mCodePredictorExec->captureGraph(stream);
    }
    // Restore CP KV cache to empty so the first real prefill starts from a clean state.
    {
        std::vector<int32_t> zeroLens(mMaxBatchSize, 0);
        rt::Tensor zeroReuse(
            zeroLens.data(), rt::Coords{mMaxBatchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cpCacheMgr.resetForNewSequences(zeroReuse, stream);
    }

    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for Talker and CodePredictor.");
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

    bool const isVoiceDesign = (mTalkerConfig.ttsModelType == "voice_design");

    // Voice clone prompt (Base checkpoints): load per request; iclFrames > 0 selects ICL mode.
    bool const hasClonePrompt = !request.refAudioPath.empty();
    int32_t iclFrames = 0;
    if (hasClonePrompt)
    {
        if (!encodeVoiceCloneReference(request.refAudioPath, request.refText, iclFrames, stream))
        {
            return false;
        }
    }
    bool const iclMode = (iclFrames > 0);
    // PyTorch strips the assistant wrap from the reference transcript: ref_ids[:, 3:-2].
    // The stripped range is consumed as iterators below — no intermediate copy.
    if (iclMode)
    {
        check::check(static_cast<int64_t>(mIclRefTextIds.size()) > 5, "voice clone ref_text_ids too short");
    }
    int64_t const refLen = iclMode ? static_cast<int64_t>(mIclRefTextIds.size()) - 5 : 0;

    // Instruction control: wrap as a user turn (matches PyTorch _build_instruct_text) and
    // prepend its token IDs so the whole sequence shares one embed+projection pass.
    std::vector<int32_t> instructIds;
    if (!request.instructText.empty())
    {
        std::string const wrapped = "<|im_start|>user\n" + request.instructText + "<|im_end|>\n";
        instructIds = mTokenizer->encode(wrapped);
    }
    int64_t const instructLen = static_cast<int64_t>(instructIds.size());

    // Projection layout: [instruct K][main text seqLen][ref transcript refLen].
    std::vector<int32_t> combinedIds;
    combinedIds.reserve(instructLen + seqLen + refLen);
    combinedIds.insert(combinedIds.end(), instructIds.begin(), instructIds.end());
    combinedIds.insert(combinedIds.end(), textTokenIds.begin(), textTokenIds.end());
    if (refLen > 0)
    {
        combinedIds.insert(combinedIds.end(), mIclRefTextIds.begin() + 3, mIclRefTextIds.end() - 2);
    }
    int64_t const combinedLen = static_cast<int64_t>(combinedIds.size());

    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    check::check(mGpuTokenIdsBuffer.reshape({1, combinedLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), combinedIds.data(), combinedLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check::check(mThinkerEmbedBuffer.reshape({1, combinedLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, mThinkerEmbedBuffer, stream);
    check::check(mThinkerEmbedBuffer.reshape({combinedLen, thinkerHiddenSize}), "Tensor reshape failed");

    // Determine speaker ID (unused for VoiceDesign — its prefix has no speaker row).
    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (request.speakerId >= 0)
    {
        speakerId = request.speakerId;
    }
    else if (!request.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(request.speakerName);
    }

    // CustomVoice language conditioning: -1 keeps the historical no-language prefill.
    int32_t const languageId = resolveLanguageId(request.languageName, request.speakerName);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (instructLen == 0 && !isVoiceDesign && !hasClonePrompt)
    {
        // Fast path: fused fixed-layout kernel (byte-identical historical behavior).
        if (!projectToTalkerInput(mThinkerEmbedBuffer, speakerId, languageId, mTalkerInputEmbeds, outSeqLen, stream))
        {
            LOG_ERROR("MLP projection failed");
            return false;
        }
    }
    else
    {
        // Builder path: instruction segment / VoiceDesign no-speaker prefix / voice clone.
        check::check(mProjectedBuffer.reshape({combinedLen, hiddenSize}), "Tensor reshape failed");
        check::check(mMLPWorkspace.reshape({combinedLen, thinkerHiddenSize}), "Tensor reshape failed");
        kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
            mProjectedBuffer, mMLPWorkspace, stream);

        // N = text tokens after stripping the 3-token role prefix and 5-token suffix.
        int64_t const N = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
        check::check(N > 0, "prepareTalkerInput: text too short after prefix/suffix strip");

        half const* projBase = static_cast<half const*>(mProjectedBuffer.rawPointer());
        half const* ttsPad = static_cast<half const*>(mTtsPadEmbed.rawPointer());
        half const* ttsBos = static_cast<half const*>(mTtsBosEmbed.rawPointer());
        half const* ttsEos = static_cast<half const*>(mTtsEosEmbed.rawPointer());

        mPrefillRows.clear();
        // Instruct segment: pure text-projected rows (no codec addend).
        for (int64_t i = 0; i < instructLen; ++i)
        {
            pushPrefillRow(projBase + i * hiddenSize, nullptr);
        }
        // Role prefix rows: projected[instructLen .. instructLen+3) copied as-is.
        for (int64_t i = 0; i < kAssistantPrefixLen; ++i)
        {
            pushPrefillRow(projBase + (instructLen + i) * hiddenSize, nullptr);
        }
        // Think block: 3 rows without language, 4 with (language row before think-eos).
        bool const hasLanguage = (languageId >= 0);
        check::check(!hasLanguage || mTalkerConfig.codecThinkId >= 0,
            "prepareTalkerInput: language requested but codec_think_id unavailable");
        pushPrefillRow(ttsPad, talkerEmbRow(hasLanguage ? mTalkerConfig.codecThinkId : mTalkerConfig.codecNothinkId));
        pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecThinkBosId));
        if (hasLanguage)
        {
            pushPrefillRow(ttsPad, talkerEmbRow(languageId));
        }
        pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecThinkEosId));
        // Speaker row: continuous x-vector for voice clone, codec token for CustomVoice,
        // omitted entirely for VoiceDesign (speaker_embed is None in the PyTorch reference).
        if (hasClonePrompt)
        {
            pushPrefillRow(ttsPad, static_cast<half const*>(mVoiceCloneXVector.rawPointer()));
        }
        else if (!isVoiceDesign)
        {
            pushPrefillRow(ttsPad, talkerEmbRow(speakerId));
        }
        pushPrefillRow(ttsBos, talkerEmbRow(mTalkerConfig.codecPadId));

        if (iclMode)
        {
            // ICL segment replaces the standard text/suffix rows. Uses the overlapped-add
            // layout from PyTorch generate_icl_prompt's default (streaming) branch — the
            // sequential non-streaming variant drifts badly in the reference implementation
            // itself (hundreds of frames for a one-sentence target). Row i pairs the text-side
            // row (ref transcript, main text, tts_eos, then tts_pad padding) with the
            // codec-side row (codec_bos, then per-frame summed reference codec embeddings).
            int64_t const textLens = refLen + N + 1;
            int64_t const codecLens = 1 + iclFrames;
            check::check(textLens <= codecLens,
                "ICL target text (" + std::to_string(textLens) + " rows) exceeds reference codes ("
                    + std::to_string(codecLens)
                    + " rows); streaming text feed-in is not supported yet — use a longer reference or "
                      "x-vector-only mode");
            half const* frameSums = static_cast<half const*>(mIclFrameSumBuffer.rawPointer());
            for (int64_t i = 0; i < codecLens; ++i)
            {
                half const* textRow;
                if (i < refLen)
                {
                    textRow = projBase + (instructLen + seqLen + i) * hiddenSize;
                }
                else if (i < refLen + N)
                {
                    textRow = projBase + (instructLen + kAssistantPrefixLen + (i - refLen)) * hiddenSize;
                }
                else if (i == textLens - 1)
                {
                    textRow = ttsEos;
                }
                else
                {
                    textRow = ttsPad;
                }
                half const* codecRow
                    = (i == 0) ? talkerEmbRow(mTalkerConfig.codecBosId) : frameSums + (i - 1) * hiddenSize;
                pushPrefillRow(textRow, codecRow);
            }
        }
        else
        {
            // Text rows: projected text + codec_pad, last row pairs codec_bos; then suffix.
            for (int64_t i = 0; i < N; ++i)
            {
                int32_t const codecId = (i == N - 1) ? mTalkerConfig.codecBosId : mTalkerConfig.codecPadId;
                pushPrefillRow(projBase + (instructLen + kAssistantPrefixLen + i) * hiddenSize, talkerEmbRow(codecId));
            }
            pushPrefillRow(ttsEos, talkerEmbRow(mTalkerConfig.codecPadId));
            pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecBosId));
        }

        outSeqLen = static_cast<int64_t>(mPrefillRows.size());
        check::check(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
        flushPrefillRows(mTalkerInputEmbeds, stream);
        LOG_INFO(
            "prepareTalkerInput (builder): instruct=%ld, N=%ld, languageId=%d, voiceDesign=%d, clone=%d, "
            "iclFrames=%d, outputSeqLen=%ld",
            instructLen, N, languageId, isVoiceDesign ? 1 : 0, hasClonePrompt ? 1 : 0, iclFrames, outSeqLen);
    }

    // Guard the workspace bound (mTalkerInputEmbeds slots are strided by maxSupportedInputLength
    // for batched prefill; instruct/language rows extend outputSeqLen beyond seqLen + 2).
    check::check(outSeqLen <= mTalkerLLMConfig.maxSupportedInputLength,
        "Talker prefill length " + std::to_string(outSeqLen) + " exceeds engine maxInputLen "
            + std::to_string(mTalkerLLMConfig.maxSupportedInputLength));

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

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    // Sub-talker == CodePredictor (CP) sampling: per-request override wins; otherwise HF's
    // per-arch ``code_predictor.generate`` defaults (kCPSampling*; Next uses the *Next variants).
    // These do NOT inherit the talker* request values.
    float const subtalkerTemperature = (req0.subtalkerTemperature > 0)
        ? req0.subtalkerTemperature
        : (isOmniNext() ? kCPSamplingTemperatureNext : kCPSamplingTemperature);
    int32_t const subtalkerTopK = (req0.subtalkerTopK > 0) ? req0.subtalkerTopK : kCPSamplingTopK;
    float const subtalkerTopP
        = (req0.subtalkerTopP > 0) ? req0.subtalkerTopP : (isOmniNext() ? kCPSamplingTopPNext : kCPSamplingTopP);
    SamplingParams predictorSamplingParams(
        1, mTalkerConfig.codebookSize, subtalkerTemperature, subtalkerTopK, subtalkerTopP);
    SamplingParams singleSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);

    // Build per-batch Talker prefill embeddings into mTalkerInputEmbeds, then run a single
    // batched prefill at bs=activeBatchSize (same pattern as handleAudioGenerationFromThinker).
    // Per-batch sequential prefill is fundamentally wrong for bs>1: it overwrites mTalkerHiddenStatesBuffer
    // slot 0 + Talker KV cache slot 0 each iteration, losing earlier batches' state.
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<int64_t> perBatchSeqLens(activeBatchSize);
    int64_t const maxInputSeqLen = mTalkerLLMConfig.maxSupportedInputLength;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t maxOutSeqLen = 0;

    auto buildOneBatchTts = [&](int32_t b) -> bool {
        if (isOmniNext())
        {
            // Qwen3-Next Omni standalone TTS: this arch has no text_projection MLP —
            // the Talker is trained on the OmniNext chunked text-feeding scheme
            // (buildQwen3OmniNextTalkerPrefill + re-prefill every framesPerCall
            // frames). The legacy prepareTalkerInput path builds an unconditioned
            // prompt, so the Talker speaks unrelated/degenerate content. Build the
            // ChatML stream the OmniNext builder expects and point prefillLen at the
            // assistant content. User messages are dropped: their content rows would
            // need thinker hidden states (hidden_projection), absent in standalone.
            LLMGenerationRequest::Request llmReq;
            for (auto const& msg : requests[b].messages)
            {
                if (msg.role == "user")
                {
                    LOG_WARNING(
                        "OmniNext standalone TTS batch %d: dropping user message (requires thinker hidden states)", b);
                    continue;
                }
                llmReq.messages.push_back(msg);
            }
            LLMGenerationRequest::FormattedRequest formatted;
            if (!mTokenizer->applyChatTemplate(llmReq, formatted, /*applyChatTemplate=*/true,
                    /*addGenerationPrompt=*/false, /*enableThinking=*/false))
            {
                LOG_ERROR("Chat template failed for batch %d", b);
                return false;
            }
            std::vector<int32_t> const textTokenIds = mTokenizer->encode(formatted.formattedCompleteRequest);

            // Locate the first content token of the (last) assistant segment:
            // <|im_start|>assistant\n<content>…  → prefillLen = index of <content>.
            int32_t contentStart = -1;
            for (size_t i = 0; i + 1 < textTokenIds.size(); ++i)
            {
                if (textTokenIds[i] == mTalkerConfig.imStartTokenId
                    && textTokenIds[i + 1] == mTalkerConfig.assistantRoleId)
                {
                    int32_t pos = static_cast<int32_t>(i) + 2;
                    if (pos < static_cast<int32_t>(textTokenIds.size()) && textTokenIds[pos] == kNlTokenIdNext)
                    {
                        ++pos;
                    }
                    contentStart = pos;
                }
            }
            if (contentStart < 0 || contentStart >= static_cast<int32_t>(textTokenIds.size()))
            {
                LOG_ERROR("OmniNext standalone TTS batch %d: no assistant content in templated request", b);
                return false;
            }

            // The C++ chat template renders an empty think block
            // (<think>\n\n</think>\n\n) inside assistant messages; HF's python
            // template does not. Its tokens are not speakable text — skip past
            // </think> and one following newline token.
            if (textTokenIds[contentStart] == mTalkerConfig.thinkOpenTokenId)
            {
                for (int32_t j = contentStart + 1; j < static_cast<int32_t>(textTokenIds.size()); ++j)
                {
                    if (textTokenIds[j] == mTalkerConfig.thinkCloseTokenId)
                    {
                        contentStart = j + 1;
                        if (contentStart < static_cast<int32_t>(textTokenIds.size())
                            && (textTokenIds[contentStart] == kNlTokenIdNext
                                || textTokenIds[contentStart] == kDoubleNlTokenIdNext))
                        {
                            ++contentStart;
                        }
                        break;
                    }
                }
            }

            int32_t speakerId = mTalkerConfig.defaultSpeakerId;
            if (requests[b].speakerId >= 0)
            {
                speakerId = requests[b].speakerId;
            }
            else if (!requests[b].speakerName.empty())
            {
                speakerId = getSpeakerIdByName(requests[b].speakerName);
            }

            int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
            __half* const trailingPtr
                = static_cast<__half*>(mStreamingTrailingHidden.rawPointer()) + b * trailingStride * hiddenSize;
            rt::Tensor trailingBuf(
                trailingPtr, rt::Coords{trailingStride, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            CUDA_CHECK(cudaMemsetAsync(trailingPtr, 0, trailingStride * hiddenSize * sizeof(__half), stream));

            int32_t trailingCount = 0;
            int64_t seqLen = 0;
            auto const* promptCodes
                = requests[b].promptSpeakerCodes.empty() ? nullptr : &requests[b].promptSpeakerCodes;
            std::vector<int32_t> systemInstructIds;
            if (!requests[b].systemInstruct.empty())
            {
                systemInstructIds = mTokenizer->encode(requests[b].systemInstruct);
            }
            if (!buildQwen3OmniNextTalkerPrefill(textTokenIds, /*prefillHiddenPtr=*/nullptr,
                    /*prefillLen=*/contentStart, speakerId, trailingBuf, trailingCount, seqLen, stream, promptCodes,
                    requests[b].assistantInstruct, requests[b].talkerLanguage,
                    systemInstructIds.empty() ? nullptr : &systemInstructIds))
            {
                LOG_ERROR("OmniNext standalone Talker prefill build failed for batch %d", b);
                return false;
            }
            perBatchSeqLens[b] = seqLen;
            maxOutSeqLen = std::max(maxOutSeqLen, seqLen);
            return true;
        }

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
        maxOutSeqLen = std::max(maxOutSeqLen, seqLen);
        return true;
    };

    // The OmniNext chunk stream is single-slot (chunk state lives at batch index 0);
    // warn if a multi-batch standalone request would leave later batches unchunked.
    if (isOmniNext() && activeBatchSize > 1)
    {
        LOG_WARNING(
            "OmniNext standalone TTS with batch size %d: chunked text feeding only supports batch 0; "
            "other batches may produce degraded audio",
            activeBatchSize);
    }

    // Build batches N-1..1 first, stash each into slot (b * maxInputSeqLen).
    for (int32_t b = activeBatchSize - 1; b >= 1; --b)
    {
        if (!buildOneBatchTts(b))
            return false;
        __half* const slot = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + b * maxInputSeqLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(slot, mTalkerInputEmbeds.rawPointer(),
            perBatchSeqLens[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    // Build batch 0 last — stays at slot 0.
    if (!buildOneBatchTts(0))
        return false;

    // Re-pack from slots [b * maxInputSeqLen] → contiguous [BS, maxOutSeqLen, H]; pad with zeros.
    __half* const inputBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    for (int32_t b = activeBatchSize - 1; b >= 0; --b)
    {
        __half* const src = inputBase + (b == 0 ? 0 : b * maxInputSeqLen * hiddenSize);
        __half* const dst = inputBase + b * maxOutSeqLen * hiddenSize;
        if (src != dst)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                dst, src, perBatchSeqLens[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        if (perBatchSeqLens[b] < maxOutSeqLen)
        {
            CUDA_CHECK(cudaMemsetAsync(dst + perBatchSeqLens[b] * hiddenSize, 0,
                (maxOutSeqLen - perBatchSeqLens[b]) * hiddenSize * sizeof(__half), stream));
        }
    }

    // Single batched Talker prefill at bs=activeBatchSize.
    check::check(mTalkerInputEmbeds.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        mTalkerHiddenStatesBuffer.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream, perBatchSeqLens))
        {
            LOG_ERROR("Batched Talker prefill failed");
            return false;
        }
    }

    // Per-batch logit adjust on [BS, vocab] slices.
    check::check(mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        rt::Tensor logitsSlice(static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
            rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor seenBufSlice(mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity,
            rt::Coords{mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerSuppressStart,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);

        // Qwen3-Next Omni: if this batch has more text chunks to feed, suppress EOS so the model
        // generates the full ``framesPerCall`` frames for this call (mirrors PT's
        // min_new_tokens=chunk_m+1 enforcement via MinLengthLogitsProcessor).
        if (b < static_cast<int32_t>(mQwen3OmniNextChunkStates.size()) && mQwen3OmniNextChunkStates[b].active)
        {
            suppressTalkerEosLogit(b, mTalkerConfig.talkerVocabSize, stream);
        }
    }

    // Batched sampling at bs=activeBatchSize. PhiloxOffset=0 for the FIRST frame after prefill so
    // it aligns with the runTalkerGenerationLoop chain that uses ``globalFrame+1`` for subsequent
    // frames (frame 0 sampled here, frame 1 uses offset 1, etc.). Without an explicit offset the
    // sampler default also produces 0 here, so the chain happens to align, but we pass it
    // explicitly to make the intent clear and avoid silent breakage if defaults change.
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream,
        /*philoxSeed=*/42, /*philoxOffset=*/0);
    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].codecToken = hostTokens[b];
        states[b].rvqCodes.reserve(requests[b].maxAudioLength);
        LOG_INFO("Batch %d: first codec token: %d", b, states[b].codecToken);
    }
    (void) singleSamplingParams; // bs=1-only param kept for streaming path; unused here after batching.

    // First codec token sampled — record TTFA-end so streaming CLIs can compute time-to-first-code.
    // Same point as handleAudioGenerationFromThinker / handleStreamingGeneration; unconditional record
    // is cheap (~µs) and lets non-profiling callers still consume the event for streaming latency.
    if (mTtfaEnd)
    {
        CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const safe = std::max(1, talkerKVCapacity - static_cast<int32_t>(perBatchSeqLens[b]));
        effectiveMaxFrames = std::max(effectiveMaxFrames, std::min(requests[b].maxAudioLength, safe));
    }

    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Wire optional per-request streaming callbacks. Empty vector when no request has streaming on
    // (zero overhead for the non-streaming path).
    std::vector<PerBatchStreamingHandler> streamingHandlers;
    for (auto const& r : requests)
    {
        if (r.streamingChunkFrames > 0 && r.onChunkReady)
        {
            streamingHandlers.resize(activeBatchSize);
            break;
        }
    }
    if (!streamingHandlers.empty())
    {
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            streamingHandlers[b].chunkFrames = requests[b].streamingChunkFrames;
            streamingHandlers[b].onChunk = requests[b].onChunkReady;
        }
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, stream, perBatchSeqLens, streamingHandlers))
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

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize,
        isOmniNext() ? kCPSamplingTemperatureNext : kCPSamplingTemperature, kCPSamplingTopK,
        isOmniNext() ? kCPSamplingTopPNext : kCPSamplingTopP);

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
        bool const buildOk = isOmniNext()
            ? buildQwen3OmniNextTalkerPrefill(textTokenIds, req.thinkerHiddenStates, req.prefillLength, speakerId,
                  batchTrailingBuf, trailingCount, outSeqLen, stream)
            : buildTalkerPrefillFromSegments(textTokenIds, req.thinkerPrefillEmbeds, req.thinkerHiddenStates,
                  req.prefillLength, mTextEmbeddingTable, speakerId, batchTrailingBuf, trailingCount, outSeqLen,
                  stream);
        if (!buildOk)
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

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, logitsSlice, mTalkerConfig.talkerSuppressStart,
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

    // Always record (unconditional, matches handleAudioGeneration / handleStreamingGeneration):
    // streaming callers consume the event for TTFC measurement even without profiling enabled.
    // Cost is negligible (~µs per request).
    if (mTtfaEnd)
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
    std::vector<int64_t>& prefillSeqLens, std::vector<PerBatchStreamingHandler> const& streamingHandlers)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runTalkerGenerationLoop", nvtx_colors::PURPLE);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t unfinished = activeBatchSize;

    // Per-batch streaming chunk emitters (no-op for batches that have streaming disabled).
    std::vector<ChunkEmitter> emitters(activeBatchSize);
    if (!streamingHandlers.empty())
    {
        check::check(static_cast<int32_t>(streamingHandlers.size()) == activeBatchSize,
            "streamingHandlers size mismatch with activeBatchSize");
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            emitters[b].chunkFrames = streamingHandlers[b].chunkFrames;
            emitters[b].onChunk = streamingHandlers[b].onChunk;
        }
    }

    // Reset the repetition-penalty window, then seed it with the first sampled token.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].numSeenTokens = 0;
        states[b].seenTokenSet.clear();
        if (!states[b].finished)
        {
            trackSeenToken(states[b].seenTokenSet, states[b].numSeenTokens, b, states[b].codecToken,
                /*tokenDev=*/nullptr, stream);
        }
    }

    int32_t globalFrame = 0;
    {
        TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

        while (unfinished > 0 && globalFrame < maxFrames)
        {
            // ---- Phase A: extract per-batch Talker last hidden into a stacked [activeBS, H] buffer ----
            auto const& fullShape = mTalkerHiddenStatesBuffer.getShape();
            int64_t const paddedSeqDim = fullShape[1];
            int64_t const hDim = fullShape[2];
            check::check(
                mTalkerLastHidden.reshape({activeBatchSize, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
            std::vector<int32_t> activeBatchCodes(activeBatchSize, 0);
            std::vector<bool> activeMask(activeBatchSize, true);
            int32_t activeBatchPresentCount = 0;
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                {
                    activeMask[b] = false;
                    continue;
                }
                ++activeBatchPresentCount;
                activeBatchCodes[b] = states[b].codecToken;

                int64_t const actualSeqDim
                    = (!prefillSeqLens.empty() && paddedSeqDim > 1) ? prefillSeqLens[b] : paddedSeqDim;
                rt::Tensor batchHiddenView(
                    static_cast<__half*>(mTalkerHiddenStatesBuffer.rawPointer()) + b * paddedSeqDim * hDim,
                    rt::Coords{1, actualSeqDim, hDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                rt::Tensor outSlice(
                    static_cast<__half*>(mTalkerLastHidden.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                if (!extractTalkerLastHidden(batchHiddenView, outSlice, stream))
                {
                    LOG_ERROR("extractTalkerLastHidden failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    activeMask[b] = false;
                    --activeBatchPresentCount;
                    continue;
                }
            }

            // ---- Phase B: CodePredictor — single batched call for all active batches.
            // The unified runCodePredictorGenerationForFrame handles activeBatchSize=1..maxBS
            // uniformly (same engine call, just different batch dim). Finished batches still
            // occupy a slot (dummy work) until a future evict pass is added.
            std::vector<std::vector<int32_t>> framesPerBatch;
            if (!runCodePredictorGenerationForFrame(activeBatchSize, activeBatchCodes, mTalkerLastHidden,
                    predictorSamplingParams, framesPerBatch, stream))
            {
                LOG_ERROR("CodePredictor failed at frame %d", globalFrame);
                return false;
            }

            // ---- Phase C: per-batch residual (kernel is per-batch; runtime loops over slots) ----
            int64_t const codecHiddensRowStride = mNumCodesPerFrame * mTalkerConfig.talkerHiddenSize;
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished || !activeMask[b])
                    continue;

                states[b].rvqCodes.push_back(std::move(framesPerBatch[b]));

                // Streaming chunk accumulate (no-op for non-streaming batches). Final flush happens
                // post-loop for every active emitter so isFinal=true is emitted exactly once per
                // streaming batch regardless of which exit path was taken.
                emitters[b].append(states[b].rvqCodes.back());

                rt::Tensor residualSlice(
                    static_cast<__half*>(mResidualEmbedBuffer.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                // Per-batch view into mCodecHiddensBuffer[b] for this batch's residual computation.
                rt::Tensor codecHiddensView(
                    static_cast<__half*>(mCodecHiddensBuffer.rawPointer()) + b * codecHiddensRowStride,
                    rt::Coords{1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
                    nvinfer1::DataType::kHALF);

                if (!computeResidualConnection(codecHiddensView, states[b].rvqCodes.back(), trailingTextHiddens[b],
                        globalFrame, residualSlice, stream))
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

            if (!executeTalkerDecodingStep(mResidualEmbedBuffer, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
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

                kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerSuppressStart,
                    mTalkerConfig.talkerVocabSize, codecEosId, states[b].numSeenTokens, repetitionPenalty, stream);

                // Suppress EOS through the chunk-boundary iter so re-prefill always runs before
                // the loop's finished check. Last-chunk (cs.active==false) is exempted.
                if (b < static_cast<int32_t>(mQwen3OmniNextChunkStates.size()))
                {
                    auto const& cs = mQwen3OmniNextChunkStates[b];
                    if (cs.active)
                    {
                        int32_t const framesThisCall
                            = static_cast<int32_t>(states[b].rvqCodes.size()) - cs.firstFrameOfCallIdx;
                        if (framesThisCall <= cs.framesPerCall)
                        {
                            suppressTalkerEosLogit(b, mTalkerConfig.talkerVocabSize, stream);
                        }
                    }
                }
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

                states[b].codecToken = hostTokens[b];

                trackSeenToken(states[b].seenTokenSet, states[b].numSeenTokens, b, states[b].codecToken,
                    mTalkerSelectedIndices.dataPointer<int32_t>() + b, stream);

                states[b].talkerFrames++;

                if (states[b].codecToken == codecEosId || states[b].talkerFrames >= maxFrames)
                {
                    states[b].finished = true;
                    unfinished--;
                }
            }
            globalFrame++;

            if (!driveOmniNextChunkReprefills(
                    states, activeBatchSize, globalFrame, talkerSamplingParams, repetitionPenalty, unfinished, stream))
            {
                return false;
            }
            // driveOmniNextChunkReprefills may have grown mQwen3OmniNextChunkStates[b].cumulativeSeqLen;
            // mirror that into prefillSeqLens so the next iteration extracts the correct last row.
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (b < static_cast<int32_t>(mQwen3OmniNextChunkStates.size())
                    && b < static_cast<int32_t>(prefillSeqLens.size())
                    && mQwen3OmniNextChunkStates[b].cumulativeSeqLen != 0)
                {
                    prefillSeqLens[b] = mQwen3OmniNextChunkStates[b].cumulativeSeqLen;
                }
            }
        }
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        bool const hitEos = (states[b].codecToken == codecEosId);
        LOG_INFO("Batch %d: %d audio frames (exit: %s)", b, states[b].talkerFrames, hitEos ? "EOS" : "maxFrames");
    }

    // Final per-batch flush: every active emitter gets exactly one isFinal=true callback (with
    // remaining buffered codes, or empty buffer as an end-of-stream signal).
    for (auto& emitter : emitters)
    {
        emitter.flushFinal();
    }

    return true;
}

bool Qwen3OmniTTSRuntime::runSingleTalkerDecodeFrame(int32_t& codecToken, SamplingParams const& talkerSamplingParams,
    SamplingParams const& predictorSamplingParams, rt::Tensor const* trailingPtr, int32_t frameIdx,
    std::unordered_set<int32_t>& seenTokenSet, int32_t& numSeenTokens, float repetitionPenalty,
    std::vector<std::vector<int32_t>>& rvqCodes, cudaStream_t stream)
{
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // Streaming TTFA path: always bs=1. Reset CP KV is now done inside runCodePredictorGenerationForFrame.
    if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mTalkerLastHidden, stream))
    {
        LOG_ERROR("extractTalkerLastHidden failed at frame %d", frameIdx);
        return false;
    }

    // Wrap mTalkerLastHidden as a [1, talkerH] batched view for the unified CP call.
    rt::Tensor talkerLastBatched(mTalkerLastHidden.rawPointer(), rt::Coords{1, mTalkerConfig.talkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    std::vector<std::vector<int32_t>> framesPerBatch;
    if (!runCodePredictorGenerationForFrame(/*activeBatchSize=*/1, std::vector<int32_t>{codecToken}, talkerLastBatched,
            predictorSamplingParams, framesPerBatch, stream))
    {
        LOG_ERROR("CodePredictor generation failed at frame %d", frameIdx);
        return false;
    }

    rvqCodes.push_back(std::move(framesPerBatch[0]));

    // Per-batch view (batch 0) into mCodecHiddensBuffer for residual.
    rt::Tensor codecHiddensView(mCodecHiddensBuffer.rawPointer(),
        rt::Coords{1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    if (!computeResidualConnection(
            codecHiddensView, rvqCodes.back(), trailingPtr, frameIdx, mResidualEmbedBuffer, stream))
    {
        LOG_ERROR("Residual connection failed at frame %d", frameIdx);
        return false;
    }

    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

    if (!executeTalkerDecodingStep(mResidualEmbedBuffer, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
    {
        LOG_ERROR("Talker decoding step failed at frame %d", frameIdx);
        return false;
    }

    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerSuppressStart,
        mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream, 42, static_cast<uint64_t>(frameIdx + 1));
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];

    trackSeenToken(
        seenTokenSet, numSeenTokens, /*batchIdx=*/0, codecToken, mTalkerSelectedIndices.dataPointer<int32_t>(), stream);
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t activeBatchSize,
    std::vector<int32_t> const& codecTokensPerBatch, rt::Tensor const& talkerLastHiddenBatched,
    SamplingParams const& samplingParams, std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    int64_t const talkerH = mTalkerConfig.talkerHiddenSize;
    int64_t const cpH = mTalkerConfig.codePredictorHiddenSize;
    int32_t const codebookSize = mTalkerConfig.codebookSize;

    check::check(activeBatchSize >= 1 && activeBatchSize <= mMaxBatchSize,
        "runCodePredictorGenerationForFrame: activeBatchSize out of range");
    check::check(static_cast<int32_t>(codecTokensPerBatch.size()) == activeBatchSize,
        "codecTokensPerBatch.size() must equal activeBatchSize");
    if (!mUseSmallToMtpProjection)
    {
        check::check(talkerH == cpH, "no-projection CP path requires talkerH == cpH");
    }

    // Helper: returns a [activeBatchSize, cpH] tensor view containing srcTalkerSpace2D projected
    // into CP hidden space. When mUseSmallToMtpProjection is false (talkerH == cpH), the source
    // view IS already the cp view — returned directly. Otherwise, invokeLinearLayer writes the
    // projected output into projectScratch (which must be sized [activeBatchSize, cpH]).
    auto projectToCpView = [&](rt::Tensor const& srcTalkerSpace2D, rt::Tensor& projectScratch) -> rt::Tensor const* {
        if (!mUseSmallToMtpProjection)
        {
            return &srcTalkerSpace2D;
        }
        check::check(projectScratch.reshape({activeBatchSize, cpH}), "Tensor reshape failed");
        kernel::invokeLinearLayer(srcTalkerSpace2D, mSmallToMtpWeight, mSmallToMtpBias, projectScratch, stream);
        return &projectScratch;
    };

    // Output containers: code_0 (from Talker) + 1..mNumRvqLayers (from CP).
    outputCodesPerBatch.assign(activeBatchSize, std::vector<int32_t>{});
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        outputCodesPerBatch[b].reserve(mNumCodesPerFrame);
        outputCodesPerBatch[b].push_back(codecTokensPerBatch[b]);
    }

    // ---- Step 1: Batched lookup of code_0 embeddings (Talker codec_embedding, talkerH-dim) ----
    // For Qwen3-Next Omni Talker we MUST use mTalkerCodecEmbedTable (codec vocab 5120). The legacy
    // Qwen3-Omni Talker uses mTalkerEmbeddingTable because there embed_tokens IS the codec table.
    rt::Tensor const& codeZeroEmbedTable = isOmniNext() ? mTalkerCodecEmbedTable : mTalkerEmbeddingTable;
    check::check(mCodePredictorCodecIds.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecIds.rawPointer(), codecTokensPerBatch.data(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(mRawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
    kernel::embeddingLookup(mCodePredictorCodecIds, codeZeroEmbedTable, std::nullopt, mRawCodecEmbed, stream);

    // ---- Step 2: Build [activeBS, 2, cpH] prefill input — slot 0 = proj(talker_h), slot 1 = proj(code_0_embed)
    check::check(mCodePredictorPrefillInput.reshape({activeBatchSize, 2, cpH}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_PREFILL, stream);

        // 2D views into the [activeBS, 1, talkerH] sources to feed projectToCpView.
        rt::Tensor talkerInput2D(const_cast<__half*>(static_cast<__half const*>(talkerLastHiddenBatched.rawPointer())),
            rt::Coords{activeBatchSize, talkerH}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor rawCodec2D(mRawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF);

        rt::Tensor const* talkerCpView = projectToCpView(talkerInput2D, mSmallToMtpProjectedHidden);
        rt::Tensor const* codecCpView = projectToCpView(rawCodec2D, mCodePredictorCodecEmbed);

        // Interleave the two per-batch [bs, cpH] tensors into [bs, 2, cpH] prefill input.
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            __half* dst = static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + b * 2 * cpH;
            __half const* talkerSrc = static_cast<__half const*>(talkerCpView->rawPointer()) + b * cpH;
            __half const* codecSrc = static_cast<__half const*>(codecCpView->rawPointer()) + b * cpH;
            CUDA_CHECK(cudaMemcpyAsync(dst, talkerSrc, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst + cpH, codecSrc, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }

        // ---- Step 3: CP prefill engine call (produces logits for code_1) ----
        check::check(mCodePredictorHiddenStatesBuffer.reshape({activeBatchSize, 2, cpH}), "Tensor reshape failed");
        rt::Tensor& logitsHead0 = mCodePredictorLogits;
        if (!executeCodePredictorPrefillStep(
                mCodePredictorPrefillInput, /*lmHeadIdx=*/0, logitsHead0, mCodePredictorHiddenStatesBuffer, stream))
        {
            return false;
        }

        // ---- Step 4: Sample code_1 (batched) ----
        check::check(logitsHead0.reshape({activeBatchSize, codebookSize}), "Tensor reshape failed");
        check::check(mCodePredictorSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        SamplingParams const perCallParams(
            activeBatchSize, codebookSize, samplingParams.temperature, samplingParams.topK, samplingParams.topP);
        trt_edgellm::topKtopPSamplingFromLogits(
            logitsHead0, mCodePredictorSelectedIndices, perCallParams, mSamplingWorkspace, stream);
        check::check(mHostSelectedCodeIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int32_t* hostCodes = mHostSelectedCodeIds.dataPointer<int32_t>();
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            outputCodesPerBatch[b].push_back(hostCodes[b]); // code_1
        }
    } // end kCODEPREDICTOR_PREFILL scope

    // ---- Step 5: Mid-RVQ codec hiddens buffer for residual.
    // Layout: mCodecHiddensBuffer[maxBS, mNumCodesPerFrame, talkerH].
    // Positions 0 and mNumRvqLayers are filled by computeResidualConnection (Talker embed lookup).
    // Positions 1..mNumRvqLayers-1 are filled here from the per-step raw codec embedding.
    check::check(mCodecHiddensBuffer.reshape({activeBatchSize, mNumCodesPerFrame, talkerH}), "Tensor reshape failed");

    // ---- Step 6: Decode loop for codes 2 .. mNumRvqLayers (batched) ----
    //
    // Tokens flow GPU-only across the loop: each step's embeddingLookup reads
    // the PREVIOUS step's sample directly from ``mCodePredictorSelectedIndices``
    // (device tensor) instead of round-tripping through host memory.  Per-step
    // sample is async-D2H'd into row ``step - 2`` of ``mHostGenCodeBuf`` on
    // pinned host memory; one ``cudaStreamSynchronize`` at the end of the loop
    // drains all 14 steps × activeBatchSize codes.
    //
    // CP has no EOS / no per-token streaming callback, so the host doesn't need
    // any sample value during the inner loop — making this loop a strictly
    // safer place to defer sync than the Thinker/Talker loops.  AR ordering is
    // preserved by stream order: step k's topK kernel happens-before step k+1's
    // embeddingLookup on the same CUDA stream.
    //
    // Entry condition: ``mCodePredictorSelectedIndices`` already holds code_1
    // from the prefill block above (its topK wrote there), so step 2 reads it
    // correctly.  See the prefill block for the post-sample state.
    int32_t* const hostGenBuf = mHostGenCodeBuf.dataPointer<int32_t>();
    int64_t const hostGenBufStride = mMaxBatchSize; // row stride of mHostGenCodeBuf
    int32_t const numGenSteps = mNumRvqLayers - 1;
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        // Bindings are step-invariant (stacked lm_heads + device head index), so one
        // prepare covers the loop and each step's execute() hash-hits the CUDA graph
        // captured for this batch size. The head index advances on-device.
        if (!prepareCpDecodeBindings(activeBatchSize, stream))
        {
            return false;
        }
        auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];
        for (int32_t step = 2; step <= mNumRvqLayers; ++step)
        {
            int32_t const embedIdx = step - 2; // step=2 -> codec_embedding[0]
            int32_t const savePos = step - 1;  // 1..mNumRvqLayers-1

            // lm_head_idx: 0 after prefill -> 1..mNumRvqLayers-1 across the loop.
            kernel::incrementLengthTensor(mCpLmHeadIdx, 1, stream);

            // Embed code_(step-1) from the PREVIOUS step's sampled-token device tensor.
            check::check(mCodePredictorSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            check::check(mRawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
            kernel::embeddingLookup(mCodePredictorSelectedIndices, mCodePredictorEmbeddingTables[embedIdx],
                std::nullopt, mRawCodecEmbed, stream);

            // Save raw embedding to mCodecHiddensBuffer[b][savePos] for the residual.
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer())
                    + (b * mNumCodesPerFrame + savePos) * talkerH;
                __half const* src = static_cast<__half const*>(mRawCodecEmbed.rawPointer()) + b * talkerH;
                CUDA_CHECK(cudaMemcpyAsync(dst, src, talkerH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            }

            // Project raw codec embed -> mCodePredictorCodecEmbed [activeBS, 1, cpH].
            rt::Tensor rawCodec2D(mRawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            rt::Tensor const* projectedView = projectToCpView(rawCodec2D, mSmallToMtpProjectedHidden);
            check::check(mCodePredictorCodecEmbed.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                __half* dst = static_cast<__half*>(mCodePredictorCodecEmbed.rawPointer()) + b * cpH;
                __half const* src = static_cast<__half const*>(projectedView->rawPointer()) + b * cpH;
                CUDA_CHECK(cudaMemcpyAsync(dst, src, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            }

            // Engine forward: graph replay via the binding-hash cache (or enqueueV3 fallback).
            mCodePredictorStepPreparer->prepare(
                rt::InferencePhase::kDecode, activeBatchSize, cpCacheMgr, *mCodePredictorPipelineIO, stream);
            if (!mCodePredictorExec->execute(stream))
            {
                LOG_ERROR("CP decode execute failed (step=%d)", step);
                return false;
            }
            cpCacheMgr.commitSequenceLength(/*increment=*/1, stream);

            // Sample code_step.
            check::check(mCodePredictorLogits.reshape({activeBatchSize, codebookSize}), "Tensor reshape failed");
            SamplingParams const perCallParams(
                activeBatchSize, codebookSize, samplingParams.temperature, samplingParams.topK, samplingParams.topP);
            trt_edgellm::topKtopPSamplingFromLogits(
                mCodePredictorLogits, mCodePredictorSelectedIndices, perCallParams, mSamplingWorkspace, stream);

            // Async D2H into row (step-2) of the pinned host buffer; drained after the loop.
            CUDA_CHECK(
                cudaMemcpyAsync(hostGenBuf + (step - 2) * hostGenBufStride, mCodePredictorSelectedIndices.rawPointer(),
                    activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        }

        // Single sync per frame: blocks until all numGenSteps × activeBatchSize
        // D2H copies (and the sampling kernels they depend on) have completed.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int32_t row = 0; row < numGenSteps; ++row)
        {
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                outputCodesPerBatch[b].push_back(hostGenBuf[row * hostGenBufStride + b]); // code_(row+2)
            }
        }
    }

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(rt::Tensor const& codecHiddensThisBatch,
    std::vector<int32_t> const& codes, rt::Tensor const* trailingTextHidden, int32_t generationStep,
    rt::Tensor& outputResidual, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // codecHiddensThisBatch: [1, mNumCodesPerFrame, talkerH] per-batch slot of mCodecHiddensBuffer.
    // Kernel reads middle rows filled by CP; positions 0/last are looked up from codes[0]/codes[last].

    check::check(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
        "Expected " + std::to_string(mNumCodesPerFrame) + " codes, got " + std::to_string(codes.size()));

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!outputResidual.reshape({1, 1, hiddenSize}))
    {
        check::check(
            outputResidual.getShape() == rt::Coords{1, 1, hiddenSize}, "Non-owning residual tensor has wrong shape");
    }

    // Residual addend: Qwen3-Omni/TTS injects per-frame trailing text (falls back to pad);
    // OmniNext supplies text via chunked re-prefill, so its per-frame addend is zero.
    __half const* addend = isOmniNext() ? static_cast<__half const*>(mQwen3OmniNextZeroResidualAddend.rawPointer())
                                        : mTtsPadEmbed.dataPointer<__half>();
    if (!isOmniNext() && trailingTextHidden != nullptr)
    {
        int64_t const trailingLen = trailingTextHidden->getShape()[0];
        if (generationStep < trailingLen)
        {
            addend = static_cast<__half const*>(trailingTextHidden->rawPointer()) + generationStep * hiddenSize;
        }
    }

    // Position-0 codec token lookup: OmniNext keeps text and codec embeds in separate tables;
    // Qwen3-Omni/TTS store only the codec table in mTalkerEmbeddingTable.
    rt::Tensor const& position0Table = isOmniNext() ? mTalkerCodecEmbedTable : mTalkerEmbeddingTable;
    kernel::invokeResidualConnection(codecHiddensThisBatch, position0Table,
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

    // Owning tensors get reshaped; non-owning views must already have the right shape.
    if (!outputLastHidden.reshape({batchSize, hiddenSize}))
    {
        check::check(outputLastHidden.getShape() == rt::Coords{batchSize, hiddenSize},
            "extractTalkerLastHidden: non-owning output has wrong shape");
    }

    size_t const copySize = hiddenSize * sizeof(__half);
    for (int64_t b = 0; b < batchSize; ++b)
    {
        size_t const srcOffset = ((b + 1) * seqLen - 1) * hiddenSize * sizeof(__half);
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

    // Fall back to voice_map.json friendly aliases (case-insensitive): "Ryan" → "m36".
    std::string lowered = speakerName;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    if (auto alias = mVoiceAliasMap.find(lowered); alias != mVoiceAliasMap.end())
    {
        if (auto mapped = mSpeakerIdMap.find(alias->second); mapped != mSpeakerIdMap.end())
        {
            return mapped->second;
        }
    }
    if (auto direct = mSpeakerIdMap.find(lowered); direct != mSpeakerIdMap.end())
    {
        return direct->second;
    }

    LOG_WARNING(
        "Speaker '%s' not found, using default speaker ID %d", speakerName.c_str(), mTalkerConfig.defaultSpeakerId);
    return mTalkerConfig.defaultSpeakerId;
}

half const* Qwen3OmniTTSRuntime::talkerEmbRow(int32_t tokenId) const
{
    return static_cast<half const*>(mTalkerEmbeddingTable.rawPointer())
        + static_cast<int64_t>(tokenId) * mTalkerConfig.talkerHiddenSize;
}

void Qwen3OmniTTSRuntime::pushPrefillRow(half const* srcA, half const* srcB)
{
    check::check(srcA != nullptr, "pushPrefillRow: null srcA");
    mPrefillRows.push_back({srcA, srcB});
}

int64_t Qwen3OmniTTSRuntime::flushPrefillRows(rt::Tensor& output, cudaStream_t stream)
{
    int64_t const numRows = static_cast<int64_t>(mPrefillRows.size());
    check::check(numRows > 0, "flushPrefillRows: no rows queued");
    int64_t const bytes = numRows * static_cast<int64_t>(sizeof(kernel::PrefillRowDesc));
    check::check(bytes <= static_cast<int64_t>(mPrefillDescsHost.getMemoryCapacity()),
        "flushPrefillRows: " + std::to_string(numRows) + " rows exceed staging capacity");
    std::memcpy(mPrefillDescsHost.rawPointer(), mPrefillRows.data(), bytes);
    CUDA_CHECK(cudaMemcpyAsync(
        mPrefillDescsDevice.rawPointer(), mPrefillDescsHost.rawPointer(), bytes, cudaMemcpyHostToDevice, stream));
    kernel::invokePrefillRowAssemble(reinterpret_cast<kernel::PrefillRowDesc const*>(mPrefillDescsDevice.rawPointer()),
        static_cast<int32_t>(numRows), static_cast<int32_t>(mTalkerConfig.talkerHiddenSize), output, stream);
    return numRows;
}

bool Qwen3OmniTTSRuntime::encodeVoiceCloneReference(
    std::string const& refAudioPath, std::string const& refText, int32_t& iclFrames, cudaStream_t stream)
{
    iclFrames = 0;
    if (mCloneEncoders == nullptr)
    {
        LOG_ERROR("Voice clone requested but no clone encoder engines loaded (pass --cloneEncoderDir)");
        return false;
    }

    audio::AudioPCM pcm;
    if (!audio::loadAudioFile(refAudioPath, /*targetSampleRate=*/24000, pcm))
    {
        LOG_ERROR("Failed to load reference audio: %s", refAudioPath.c_str());
        return false;
    }

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    check::check(mCloneEncoders->speakerEmbeddingDim() == hiddenSize, "speaker encoder dim != talker hidden");
    if (!mCloneEncoders->extractSpeakerEmbedding(pcm.samples, mVoiceCloneXVector, stream))
    {
        return false;
    }

    mIclRefTextIds.clear();
    if (refText.empty())
    {
        return true; // x-vector-only mode
    }
    check::check(
        mCloneEncoders->hasTokenizerEncoder() && mCloneEncoders->numQuantizers() == mTalkerConfig.numCodeGroups,
        "ICL cloning needs speech_tokenizer_encoder.engine with matching code groups");

    int32_t numFrames = 0;
    if (!mCloneEncoders->encodeReferenceCodes(pcm.samples, numFrames, stream))
    {
        return false;
    }
    // Capacity check must not depend on the current shape — a previous request may have
    // reshaped the buffer to fewer rows.
    int64_t const maxRefFrames
        = static_cast<int64_t>(mIclFrameSumBuffer.getMemoryCapacity()) / (hiddenSize * sizeof(__half));
    check::check(numFrames <= maxRefFrames,
        "reference frames " + std::to_string(numFrames) + " exceed ICL workspace capacity "
            + std::to_string(maxRefFrames));

    // Per-frame sum across all code groups (group 0 = talker table, 1.. = CodePredictor tables).
    check::check(mIclFrameSumBuffer.reshape({numFrames, hiddenSize}), "Tensor reshape failed");
    // Sum kernel consumes the codec-encoder engine output in place (INT64, still on device).
    kernel::invokeSumCodecEmbeddings(mCloneEncoders->refCodesDevice(),
        reinterpret_cast<half const* const*>(mIclTablePtrsGpu.rawPointer()), numFrames, mTalkerConfig.numCodeGroups,
        static_cast<int32_t>(hiddenSize), mIclFrameSumBuffer, stream);

    // Reference transcript: assistant-wrapped then stripped [3:-2] at recipe time, matching
    // the PyTorch reference tokenization exactly.
    mIclRefTextIds = mTokenizer->encode("<|im_start|>assistant\n" + refText + "<|im_end|>\n");

    iclFrames = numFrames;
    return true;
}

int32_t Qwen3OmniTTSRuntime::resolveLanguageId(std::string const& languageName, std::string const& speakerName) const
{
    // Engines without language support (no codec_think_id or empty map) always use the
    // no-language path, regardless of what the request asks for.
    if (mTalkerConfig.codecThinkId < 0 || mTalkerConfig.codecLanguageIdMap.empty())
    {
        if (!languageName.empty() && languageName != "auto")
        {
            LOG_WARNING(
                "Request language '%s' ignored: engine config has no codec_language_id map "
                "(re-export with a CustomVoice checkpoint to enable language conditioning)",
                languageName.c_str());
        }
        return -1;
    }

    auto toLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    };
    std::string const lang = toLower(languageName);

    int32_t languageId = -1;
    if (!lang.empty() && lang != "auto")
    {
        auto it = mTalkerConfig.codecLanguageIdMap.find(lang);
        if (it != mTalkerConfig.codecLanguageIdMap.end())
        {
            languageId = it->second;
        }
        else
        {
            LOG_WARNING("Language '%s' not found in codec_language_id map, falling back to auto (no-language)",
                languageName.c_str());
        }
    }

    // Dialect override (matches PyTorch modeling_qwen3_tts.py): when the language is auto or
    // chinese and the speaker is a dialect speaker, the dialect's language ID wins.
    if ((lang.empty() || lang == "auto" || lang == "chinese") && !speakerName.empty())
    {
        auto dialectIt = mTalkerConfig.spkDialectMap.find(toLower(speakerName));
        if (dialectIt != mTalkerConfig.spkDialectMap.end())
        {
            auto langIt = mTalkerConfig.codecLanguageIdMap.find(dialectIt->second);
            if (langIt != mTalkerConfig.codecLanguageIdMap.end())
            {
                LOG_INFO("Dialect speaker '%s' → language '%s' (codec id %d)", speakerName.c_str(),
                    dialectIt->second.c_str(), langIt->second);
                languageId = langIt->second;
            }
        }
    }

    return languageId;
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

    // Step 2: Project ALL tokens through text_projection (Qwen3-Omni: 2-layer MLP) or
    // copy them directly (Qwen3-Next Omni: Talker's own embed_tokens is already at talker hidden dim).
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    if (isOmniNext())
    {
        CUDA_CHECK(cudaMemcpyAsync(mProjectedBuffer.rawPointer(), mThinkerEmbedBuffer.rawPointer(),
            static_cast<size_t>(seqLen) * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
        kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
            mProjectedBuffer, mMLPWorkspace, stream);
    }

    // Step 3: Parse segments by <|im_start|> positions
    std::vector<SegmentInfo> segments;
    for (int64_t i = 0; i < seqLen; ++i)
    {
        if (textTokenIds[i] == mTalkerConfig.imStartTokenId)
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
        if (segments[s].roleId == mTalkerConfig.systemRoleId)
            continue;
        else if (segments[s].roleId == mTalkerConfig.userRoleId)
            userSegmentIndices.push_back(s);
        else if (segments[s].roleId == mTalkerConfig.assistantRoleId)
            assistantSegIdx = static_cast<int64_t>(s);
    }

    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildTalkerPrefillFromSegments: could not find assistant segment");
        return false;
    }

    // Step 4: Project multimodal tokens via hidden_projection using gather/scatter kernels
    bool const hasHiddenProjection
        = isOmniNext() ? (mHiddenProjLinearWeight.getShape().volume() > 0) : (mHiddenFC1Weight.getShape().volume() > 0);
    if (hasHiddenProjection && prefillHiddenPtr != nullptr)
    {
        std::vector<int32_t> mmPositions;
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos && i < prefillLen; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == mTalkerConfig.audioTokenId || tid == mTalkerConfig.imageTokenId
                    || tid == mTalkerConfig.videoTokenId)
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

            if (isOmniNext())
            {
                // Qwen3-Next Omni: single Linear (weight [talkerHidden, thinkerHidden] + bias [talkerHidden]).
                kernel::invokeLinearLayer(
                    mThinkerEmbedBuffer, mHiddenProjLinearWeight, mHiddenProjLinearBias, mTalkerInputEmbeds, stream);
            }
            else
            {
                kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                    mHiddenFC2Bias, mTalkerInputEmbeds, mMLPWorkspace, stream);
            }

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
                if (tid == mTalkerConfig.audioTokenId || tid == mTalkerConfig.imageTokenId
                    || tid == mTalkerConfig.videoTokenId)
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

        // Omni path stays language-free: Omni checkpoints have no codec_think_id and build
        // their prefill without language conditioning.
        kernel::invokeAssistantPreamble(assistantSlice, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
            mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
            mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, /*codecThinkId=*/-1, /*languageId=*/-1, 1,
            preambleScratch, stream);

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

// ---------------------------------------------------------------------------
// OmniNext Talker prefill: row-assembly primitives shared by build + reprefill.
// See buildQwen3OmniNextTalkerPrefill for the row sequence.
// ---------------------------------------------------------------------------

bool Qwen3OmniTTSRuntime::copyEmbedRow(
    rt::Tensor const& table, int32_t tokenId, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const vocab = table.getShape()[0];
    int64_t const hiddenSize = table.getShape()[1];
    if (tokenId < 0 || tokenId >= vocab)
    {
        LOG_ERROR("Embed lookup OOB: tokenId=%d vocab=%ld", tokenId, vocab);
        return false;
    }
    __half const* src = static_cast<__half const*>(table.rawPointer()) + static_cast<int64_t>(tokenId) * hiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(dstBase + dstRow * hiddenSize, src, static_cast<size_t>(hiddenSize) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream));
    return true;
}

void Qwen3OmniTTSRuntime::copyRawRow(rt::Tensor const& src, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(dstBase + dstRow * hiddenSize, src.rawPointer(),
        static_cast<size_t>(hiddenSize) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

bool Qwen3OmniTTSRuntime::buildCodecEmbedPointerTable(cudaStream_t stream)
{
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    if (static_cast<int32_t>(mCodePredictorEmbeddingTables.size()) < numCodeGroups - 1)
    {
        LOG_ERROR("CodePredictor codec embed tables (%zu) < numCodeGroups-1 (%d)", mCodePredictorEmbeddingTables.size(),
            numCodeGroups - 1);
        return false;
    }
    std::vector<__half const*> hostEmbPtrs(numCodeGroups);
    std::vector<int32_t> hostVocabSizes(numCodeGroups);
    hostEmbPtrs[0] = static_cast<__half const*>(mTalkerCodecEmbedTable.rawPointer());
    hostVocabSizes[0] = static_cast<int32_t>(mTalkerCodecEmbedTable.getShape()[0]);
    for (int32_t g = 1; g < numCodeGroups; ++g)
    {
        rt::Tensor const& t = mCodePredictorEmbeddingTables[g - 1];
        hostEmbPtrs[g] = static_cast<__half const*>(t.rawPointer());
        hostVocabSizes[g] = static_cast<int32_t>(t.getShape()[0]);
    }
    CUDA_CHECK(cudaMemcpyAsync(mCodecEmbPtrTable.rawPointer(), hostEmbPtrs.data(),
        numCodeGroups * sizeof(__half const*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mCodecEmbVocabSizes.rawPointer(), hostVocabSizes.data(), numCodeGroups * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    return true;
}

void Qwen3OmniTTSRuntime::sumSpeakerCodecRow(
    int64_t const* hostCodes, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    CUDA_CHECK(cudaMemcpyAsync(
        mCodecRowCodes.rawPointer(), hostCodes, numCodeGroups * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    rt::Tensor rowOut(
        dstBase + dstRow * hiddenSize, rt::Coords{hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::invokeSpeakerCodecSum(mCodecRowCodes, mCodecEmbPtrTable, mCodecEmbVocabSizes, rowOut, stream);
}

void Qwen3OmniTTSRuntime::projectHiddenRow(
    rt::Tensor const& prefillHidden, int64_t srcRow, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half const* hSrc = static_cast<__half const*>(prefillHidden.rawPointer()) + srcRow * thinkerHiddenSize;
    rt::Tensor hSlice(
        const_cast<__half*>(hSrc), rt::Coords{1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor outSlice(dstBase + dstRow * talkerHiddenSize, rt::Coords{1, talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    kernel::invokeLinearLayer(hSlice, mHiddenProjLinearWeight, mHiddenProjLinearBias, outSlice, stream);
}

// ---------------------------------------------------------------------------
// OmniNext Talker prefill (system + user + assistant), mirrors HF
// _get_talker_{system,user,assistant}_parts. Row sequence:
//   sys:  [im_start, system, nl] + speaker_system_prompt +
//         codec_bos + speaker_codec_sum[..] + codec_eos + [im_end, nl]
//   usr:  role tokens via text embed, content tokens via hidden_projection(prefillHidden)
//   asst: [im_start, assistant, nl] + [codec_nothink, codec_think_bos, codec_think_eos]
//         + tts_bos + codec_bos + first text chunk (up to kTextInChunkN rows).
// Remaining text chunks + tts_eos go into mQwen3OmniNextChunkStates[0] for reprefill.
// ---------------------------------------------------------------------------

bool Qwen3OmniTTSRuntime::buildQwen3OmniNextTalkerPrefill(std::vector<int32_t> const& textTokenIds,
    rt::Tensor const* prefillHiddenPtr, int32_t prefillLen, int32_t speakerId, rt::Tensor& trailingTextHidden,
    int32_t& trailingCount, int64_t& outSeqLen, cudaStream_t stream,
    std::vector<std::vector<int32_t>> const* promptSpeakerCodes, std::string const& assistantInstruct,
    std::string const& talkerLanguage, std::vector<int32_t> const* systemInstructIds)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::buildQwen3OmniNextTalkerPrefill", nvtx_colors::PALE_GREEN);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    int32_t const codecBosId = mTalkerConfig.codecBosId;
    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t const codecNothinkId = mTalkerConfig.codecNothinkId;
    int32_t const codecThinkBosId = mTalkerConfig.codecThinkBosId;
    int32_t const codecThinkEosId = mTalkerConfig.codecThinkEosId;

    bool const hasCustomVoice = promptSpeakerCodes != nullptr && !promptSpeakerCodes->empty();

    if (mTalkerCodecEmbedTable.rawPointer() == nullptr
        || (!hasCustomVoice && mSpeakerCodecEmbeddings.rawPointer() == nullptr))
    {
        LOG_ERROR("OmniNext prefill requires codec_embedding and speaker_codec_embeddings sidecars");
        return false;
    }

    int32_t speakerEmbedLen = 0;
    if (!hasCustomVoice)
    {
        auto const& spkShape = mSpeakerCodecEmbeddings.getShape();
        int32_t const maxSpeakerNum = static_cast<int32_t>(spkShape[0]);
        int32_t const numGroupsSpk = static_cast<int32_t>(spkShape[1]);
        speakerEmbedLen = static_cast<int32_t>(spkShape[2]);
        if (speakerId < 0 || speakerId >= maxSpeakerNum)
        {
            LOG_ERROR("speakerId %d out of range [0, %d)", speakerId, maxSpeakerNum);
            return false;
        }
        if (numGroupsSpk != numCodeGroups)
        {
            LOG_ERROR("speaker_codec_embeddings groups (%d) != numCodeGroups (%d)", numGroupsSpk, numCodeGroups);
            return false;
        }
    }

    if (!buildCodecEmbedPointerTable(stream))
    {
        return false;
    }

    // Speaker codec rows: either the caller-provided reference-voice codes (custom
    // voice, HF ``prompt_speaker_codes``) or the built-in speaker LUT row. Both are
    // staged into ``speakerRow`` with the LUT's [group][pos] layout.
    std::vector<int64_t> speakerRow;
    if (hasCustomVoice)
    {
        speakerEmbedLen = static_cast<int32_t>(promptSpeakerCodes->size());
        speakerRow.assign(static_cast<size_t>(numCodeGroups) * speakerEmbedLen, -1);
        for (int32_t p = 0; p < speakerEmbedLen; ++p)
        {
            auto const& frame = (*promptSpeakerCodes)[p];
            if (static_cast<int32_t>(frame.size()) != numCodeGroups)
            {
                LOG_ERROR("prompt_speaker_codes frame %d has %zu codes, expected %d", p, frame.size(), numCodeGroups);
                return false;
            }
            for (int32_t g = 0; g < numCodeGroups; ++g)
            {
                speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p] = frame[g];
            }
        }
        LOG_INFO("OmniNext custom voice: using %d prompt_speaker_codes frames (built-in speaker rows skipped)",
            speakerEmbedLen);
    }
    else
    {
        // Pull the speaker's codec-code matrix to host once and trim its -1 padding
        // (HF: speaker_code[..., :valid_len]).
        int64_t const speakerRowBytes = static_cast<int64_t>(numCodeGroups) * speakerEmbedLen * sizeof(int64_t);
        speakerRow.resize(static_cast<size_t>(numCodeGroups) * speakerEmbedLen);
        CUDA_CHECK(cudaMemcpyAsync(speakerRow.data(),
            static_cast<int64_t const*>(mSpeakerCodecEmbeddings.rawPointer())
                + static_cast<int64_t>(speakerId) * numCodeGroups * speakerEmbedLen,
            speakerRowBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int32_t validSpeakerLen = 0;
    for (int32_t p = speakerEmbedLen - 1; p >= 0; --p)
    {
        bool any = false;
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            if (speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p] != -1)
            {
                any = true;
                break;
            }
        }
        if (any)
        {
            validSpeakerLen = p + 1;
            break;
        }
    }
    if (validSpeakerLen == 0)
    {
        validSpeakerLen = speakerEmbedLen;
    }

    // Row plan (mirrors HF _get_talker_system_parts / _user_parts / _assistant_parts).
    constexpr int32_t kSysRolePrefix = 3;  // [im_start, system, nl]
    constexpr int32_t kSysSuffix = 2;      // [im_end, nl]
    constexpr int32_t kAsstRolePrefix = 3; // [im_start, assistant, nl]

    // Optional style/emotion instruction rows (text-embed ids) inserted right after the
    // assistant role trio (HF: assistant_instruct_ids).
    std::vector<int32_t> const* instructIdsPtr = nullptr;
    if (hasCustomVoice && !assistantInstruct.empty())
    {
        // HF only applies assistant_instruct when prompt_speaker_codes is None.
        LOG_WARNING("assistant_instruct '%s' ignored with prompt_speaker_codes (built-in speakers only)",
            assistantInstruct.c_str());
    }
    else if (!assistantInstruct.empty())
    {
        if (auto it = mAssistantPromptIds.find(assistantInstruct); it != mAssistantPromptIds.end())
        {
            instructIdsPtr = &it->second;
        }
        else
        {
            LOG_WARNING("assistant_instruct '%s' not in talker_assistant_prompt_id_mapping; ignored",
                assistantInstruct.c_str());
        }
    }
    int32_t const instructRows = instructIdsPtr ? static_cast<int32_t>(instructIdsPtr->size()) : 0;

    // Language conditioning: a valid language swaps the codec-special trio for the
    // 4-row think block [codec_think, think_bos, LANGUAGE_ID, think_eos] (HF semantics).
    int32_t languageCodecId = -1;
    if (!talkerLanguage.empty())
    {
        std::string lang = talkerLanguage;
        std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) { return std::tolower(c); });
        if (lang != "auto")
        {
            if (auto it = mLanguageIds.find(lang); it != mLanguageIds.end() && mTalkerConfig.codecThinkId >= 0)
            {
                languageCodecId = it->second;
            }
            else
            {
                LOG_WARNING(
                    "talker_language '%s' not in talker_language_id mapping; using auto", talkerLanguage.c_str());
            }
        }
    }
    int32_t const kCodecSpecial
        = languageCodecId >= 0 ? 4 : 3; // [no_think, think_bos, think_eos] or [think, think_bos, LANG, think_eos]

    // HF inserts a per-speaker text prompt between [im_start,system,nl] and codec_bos —
    // but only for built-in speakers; custom-voice requests skip it (HF only appends
    // ``speaker_system_prompt_id`` when ``prompt_speaker_codes is None``).
    std::vector<int32_t> const* speakerSysPromptPtr = nullptr;
    if (!hasCustomVoice)
    {
        if (auto it = mSpeakerSystemPromptIds.find(speakerId); it != mSpeakerSystemPromptIds.end())
        {
            speakerSysPromptPtr = &it->second;
        }
    }
    int32_t const sysPromptLen = speakerSysPromptPtr ? static_cast<int32_t>(speakerSysPromptPtr->size()) : 0;
    // Optional free-text system instruction rows: HF appends them right after the
    // system role trio, before the per-speaker prompt.
    int32_t const sysInstructRows = systemInstructIds ? static_cast<int32_t>(systemInstructIds->size()) : 0;
    int32_t const systemRows = kSysRolePrefix + sysInstructRows + sysPromptLen + 1 /*codec_bos*/ + validSpeakerLen
        + 1 /*codec_eos*/ + kSysSuffix;

    // Split the ChatML token stream into (start, end, roleId) segments so we can
    // replay user segments and locate the assistant text.
    struct ChatSegment
    {
        int32_t start;
        int32_t end;
        int32_t roleId;
    };
    std::vector<ChatSegment> segments;
    for (size_t i = 0; i < textTokenIds.size(); ++i)
    {
        if (textTokenIds[i] == mTalkerConfig.imStartTokenId && i + 1 < textTokenIds.size())
        {
            segments.push_back(
                {static_cast<int32_t>(i), static_cast<int32_t>(textTokenIds.size()), textTokenIds[i + 1]});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].end = segments[s + 1].start;
    }
    int32_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == mTalkerConfig.assistantRoleId)
        {
            assistantSegIdx = static_cast<int32_t>(s);
        }
    }
    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildQwen3OmniNextTalkerPrefill: assistant role marker not found in textTokenIds");
        return false;
    }
    // The chat template appends <|im_start|>assistant\n<think>\n</think>\n via
    // add_generation_prompt, so prefillLen already sits at the first generated content token.
    auto const& asstSeg = segments[assistantSegIdx];
    int32_t assistantTextStart = prefillLen;
    int32_t assistantTextEnd = asstSeg.end;
    for (int32_t i = assistantTextStart; i < asstSeg.end; ++i)
    {
        if (textTokenIds[i] == kImEndTokenIdNext)
        {
            assistantTextEnd = i;
            break;
        }
    }
    int32_t const assistantTextLen = std::max(0, assistantTextEnd - assistantTextStart);
    int32_t const totalTextLen = assistantTextLen + 1; // +1 for appended tts_eos
    int32_t const firstChunkLen = std::min(totalTextLen, kTextInChunkN);

    int32_t const assistantRows
        = kAsstRolePrefix + instructRows + kCodecSpecial + 1 /*tts_bos*/ + 1 /*codec_bos*/ + firstChunkLen;

    // User part: role tokens use Talker text embed; content tokens go through the
    // single-Linear hidden_projection. mmPos is subsampled per segment via
    // torch.linspace(0,N-1,M).long() to at most maxThinkerToTalkerMmTokens.
    auto isUserRoleToken = [&](int32_t t) {
        return t == mTalkerConfig.imStartTokenId || t == kImEndTokenIdNext || t == mTalkerConfig.userRoleId
            || t == kNlTokenIdNext;
    };
    std::vector<std::pair<int32_t /*tokenId*/, int32_t /*absPos*/>> userPositions;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (static_cast<int32_t>(s) == assistantSegIdx)
        {
            break;
        }
        if (segments[s].roleId == mTalkerConfig.systemRoleId)
        {
            continue; // system role is handled by the speaker_part above
        }
        std::vector<int32_t> rolePos;
        std::vector<int32_t> mmPos;
        for (int32_t i = segments[s].start; i < segments[s].end; ++i)
        {
            (isUserRoleToken(textTokenIds[i]) ? rolePos : mmPos).push_back(i);
        }
        if (mTalkerConfig.maxThinkerToTalkerMmTokens > 0
            && static_cast<int32_t>(mmPos.size()) > mTalkerConfig.maxThinkerToTalkerMmTokens)
        {
            int32_t const N = static_cast<int32_t>(mmPos.size());
            int32_t const M = mTalkerConfig.maxThinkerToTalkerMmTokens;
            std::vector<int32_t> kept;
            kept.reserve(M);
            if (M == 1)
            {
                // torch.linspace(0, N-1, 1) yields tensor([0.]).long() = [0].
                kept.push_back(mmPos.front());
            }
            else
            {
                for (int32_t i = 0; i < M; ++i)
                {
                    // double avoids FP drift vs torch.linspace's ladder.
                    double const idxF
                        = static_cast<double>(i) * static_cast<double>(N - 1) / static_cast<double>(M - 1);
                    kept.push_back(mmPos[static_cast<int32_t>(idxF)]);
                }
            }
            mmPos.swap(kept);
        }
        std::vector<int32_t> merged;
        merged.reserve(rolePos.size() + mmPos.size());
        merged.insert(merged.end(), rolePos.begin(), rolePos.end());
        merged.insert(merged.end(), mmPos.begin(), mmPos.end());
        std::sort(merged.begin(), merged.end());
        for (int32_t pos : merged)
        {
            userPositions.emplace_back(textTokenIds[pos], pos);
        }
    }
    int32_t const userPartRows = static_cast<int32_t>(userPositions.size());

    int64_t const totalSeqLen = static_cast<int64_t>(systemRows) + userPartRows + assistantRows;
    if (totalSeqLen > mTalkerLLMConfig.maxSupportedInputLength)
    {
        LOG_ERROR("buildQwen3OmniNextTalkerPrefill: assembled length %ld > maxInputLen %d", totalSeqLen,
            mTalkerLLMConfig.maxSupportedInputLength);
        return false;
    }
    outSeqLen = totalSeqLen;
    check::check(mTalkerInputEmbeds.reshape({1, totalSeqLen, hiddenSize}), "Tensor reshape failed");
    __half* const dstBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());

    // System part.
    int64_t row = 0;
    if (!copyEmbedRow(mTextEmbeddingTable, mTalkerConfig.imStartTokenId, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, mTalkerConfig.systemRoleId, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, kNlTokenIdNext, dstBase, row++, stream))
        return false;
    if (systemInstructIds)
    {
        for (int32_t tid : *systemInstructIds)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
                return false;
        }
    }
    if (speakerSysPromptPtr)
    {
        for (int32_t tid : *speakerSysPromptPtr)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
                return false;
        }
    }
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecBosId, dstBase, row++, stream))
        return false;
    for (int32_t p = 0; p < validSpeakerLen; ++p)
    {
        int64_t hostCodesAtPos[64]; // numCodeGroups <= 64
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            hostCodesAtPos[g] = speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p];
        }
        sumSpeakerCodecRow(hostCodesAtPos, dstBase, row++, stream);
    }
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecEosId, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, kImEndTokenIdNext, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, kNlTokenIdNext, dstBase, row++, stream))
        return false;
    ELLM_CHECK(row == systemRows, "system row count mismatch");

    // User part: role → text embed, content → hidden_projection(thinker_hidden[absPos]).
    auto isRoleToken = [&](int32_t t) {
        return t == mTalkerConfig.imStartTokenId || t == kImEndTokenIdNext || t == mTalkerConfig.userRoleId
            || t == kNlTokenIdNext;
    };
    for (auto const& [tid, absPos] : userPositions)
    {
        if (isRoleToken(tid))
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
                return false;
        }
        else
        {
            if (prefillHiddenPtr == nullptr || absPos >= prefillLen)
            {
                LOG_ERROR(
                    "OmniNext user content pos %d out of range (prefillLen=%d) or hidden ptr null", absPos, prefillLen);
                return false;
            }
            projectHiddenRow(*prefillHiddenPtr, absPos, dstBase, row++, stream);
        }
    }
    ELLM_CHECK(row == systemRows + userPartRows, "user row count mismatch");

    // Assistant part.
    if (!copyEmbedRow(mTextEmbeddingTable, mTalkerConfig.imStartTokenId, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, mTalkerConfig.assistantRoleId, dstBase, row++, stream))
        return false;
    if (!copyEmbedRow(mTextEmbeddingTable, kNlTokenIdNext, dstBase, row++, stream))
        return false;
    if (instructIdsPtr)
    {
        for (int32_t tid : *instructIdsPtr)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
                return false;
        }
    }
    if (languageCodecId >= 0)
    {
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkId, dstBase, row++, stream))
            return false;
        if (!copyEmbedRow(mTalkerCodecEmbedTable, codecThinkBosId, dstBase, row++, stream))
            return false;
        if (!copyEmbedRow(mTalkerCodecEmbedTable, languageCodecId, dstBase, row++, stream))
            return false;
        if (!copyEmbedRow(mTalkerCodecEmbedTable, codecThinkEosId, dstBase, row++, stream))
            return false;
    }
    else
    {
        if (!copyEmbedRow(mTalkerCodecEmbedTable, codecNothinkId, dstBase, row++, stream))
            return false;
        if (!copyEmbedRow(mTalkerCodecEmbedTable, codecThinkBosId, dstBase, row++, stream))
            return false;
        if (!copyEmbedRow(mTalkerCodecEmbedTable, codecThinkEosId, dstBase, row++, stream))
            return false;
    }
    copyRawRow(mTtsBosEmbed, dstBase, row++, stream);
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecBosId, dstBase, row++, stream))
        return false;

    // First text chunk: up to kTextInChunkN rows; tts_eos occupies the last slot when the
    // chunk covers the end of the text.
    int32_t const chunkTextRows = std::min(firstChunkLen, assistantTextLen);
    for (int32_t i = 0; i < chunkTextRows; ++i)
    {
        if (!copyEmbedRow(mTextEmbeddingTable, textTokenIds[assistantTextStart + i], dstBase, row++, stream))
            return false;
    }
    if (firstChunkLen > assistantTextLen)
    {
        copyRawRow(mTtsEosEmbed, dstBase, row++, stream);
    }
    ELLM_CHECK(row == totalSeqLen,
        "row count mismatch (got " + std::to_string(row) + ", expected " + std::to_string(totalSeqLen) + ")");

    // Trailing rows (unused by the OmniNext decode step but kept for interface parity).
    int32_t const remainingTextStart = chunkTextRows;
    int32_t const remainingTextLen = assistantTextLen - remainingTextStart;
    int32_t const remainingHasEos = (firstChunkLen <= assistantTextLen) ? 1 : 0;
    int32_t const remainingTotal = remainingTextLen + remainingHasEos;
    int32_t const trailingCapacity = static_cast<int32_t>(trailingTextHidden.getShape()[0]);
    trailingCount = std::min(remainingTotal, trailingCapacity - 1);
    __half* const trailDst = static_cast<__half*>(trailingTextHidden.rawPointer());
    for (int32_t i = 0; i < trailingCount; ++i)
    {
        if (i < remainingTextLen)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, textTokenIds[assistantTextStart + remainingTextStart + i], trailDst,
                    i, stream))
                return false;
        }
        else
        {
            copyRawRow(mTtsEosEmbed, trailDst, i, stream);
        }
    }

    // Arm chunked streaming: remaining text tokens (chunks 1..) feed the re-prefill loop.
    if (static_cast<int32_t>(mQwen3OmniNextChunkStates.size()) <= 0)
    {
        mQwen3OmniNextChunkStates.resize(std::max(1, mMaxBatchSize));
    }
    {
        auto& cs = mQwen3OmniNextChunkStates[0];
        cs.remainingTextTokens.clear();
        cs.remainingTextTokens.reserve(static_cast<size_t>(std::max(0, remainingTextLen)));
        for (int32_t i = 0; i < remainingTextLen; ++i)
        {
            cs.remainingTextTokens.push_back(textTokenIds[assistantTextStart + remainingTextStart + i]);
        }
        cs.hasTrailingTtsEos = (remainingHasEos != 0);
        cs.active = !cs.remainingTextTokens.empty() || cs.hasTrailingTtsEos;
        cs.chunkTokensPerCall = kTextInChunkN;
        cs.framesPerCall = 4;
        cs.cumulativeSeqLen = outSeqLen;
        cs.cursorToken = 0;
        cs.framesSinceLastPrefill = 0;
        cs.firstFrameOfCallIdx = 0;
        if (cs.active)
        {
            LOG_INFO("OmniNext chunk stream armed: remainingText=%zu hasTtsEos=%d cumSeq=%ld",
                cs.remainingTextTokens.size(), static_cast<int>(cs.hasTrailingTtsEos), cs.cumulativeSeqLen);
        }
    }

    LOG_INFO("buildQwen3OmniNextTalkerPrefill: outSeqLen=%ld (sys=%d[+%d], usr=%d, asst=%d), trailing=%d", outSeqLen,
        systemRows, sysPromptLen, userPartRows, assistantRows, trailingCount);
    return true;
}

// ---------------------------------------------------------------------------
// OmniNext chunked re-prefill. HF generate_talker reissues prefill with
// [cumulative | codec_input_embeds(prev frames) | next text chunk] every
// framesPerCall frames until the assistant text is exhausted.
// ---------------------------------------------------------------------------

int32_t Qwen3OmniTTSRuntime::maybeReprefillOmniNextChunkForBatch(int32_t batchIdx,
    std::vector<std::vector<int32_t>> const& rvqCodes, int32_t& codecTokenInOut,
    SamplingParams const& talkerSamplingParams, float repetitionPenalty, int32_t& numSeenTokens,
    std::unordered_set<int32_t>& seenTokenSet, cudaStream_t stream)
{
    if (batchIdx < 0 || batchIdx >= static_cast<int32_t>(mQwen3OmniNextChunkStates.size()))
    {
        return 0;
    }
    auto& cs = mQwen3OmniNextChunkStates[batchIdx];
    if (!cs.active)
    {
        return 0;
    }
    int32_t const framesThisCall = static_cast<int32_t>(rvqCodes.size()) - cs.firstFrameOfCallIdx;
    if (framesThisCall < cs.framesPerCall)
    {
        return 0;
    }
    int32_t newCodecToken = 0;
    if (!reprefillQwen3OmniNextChunk(batchIdx, rvqCodes, newCodecToken, talkerSamplingParams, repetitionPenalty,
            numSeenTokens, seenTokenSet, stream))
    {
        return -1;
    }
    codecTokenInOut = newCodecToken;
    return 1;
}

bool Qwen3OmniTTSRuntime::driveOmniNextChunkReprefills(std::vector<PerBatchTalkerState>& states,
    int32_t activeBatchSize, int32_t globalFrame, SamplingParams const& talkerSamplingParams, float repetitionPenalty,
    int32_t& unfinished, cudaStream_t stream)
{
    int32_t const codecEosId = mTalkerConfig.codecEosId;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        if (states[b].finished)
        {
            continue;
        }
        int32_t newCodecToken = states[b].codecToken;
        int32_t const outcome = maybeReprefillOmniNextChunkForBatch(b, states[b].rvqCodes, newCodecToken,
            talkerSamplingParams, repetitionPenalty, states[b].numSeenTokens, states[b].seenTokenSet, stream);
        if (outcome == -1)
        {
            LOG_ERROR("reprefillQwen3OmniNextChunk failed at frame %d batch %d", globalFrame, b);
            states[b].finished = true;
            states[b].talkerError = true;
            unfinished--;
            continue;
        }
        if (outcome == 1)
        {
            // Freshly-sampled first frame of the new call replaces the stale unconstrained sample.
            states[b].codecToken = newCodecToken;
            if (newCodecToken == codecEosId)
            {
                states[b].finished = true;
                unfinished--;
            }
        }
    }
    return true;
}

void Qwen3OmniTTSRuntime::appendOmniNextChunkStreamToken(int32_t batchIdx, int32_t tokenId)
{
    if (batchIdx < 0 || batchIdx >= static_cast<int32_t>(mQwen3OmniNextChunkStates.size()))
    {
        return;
    }
    auto& cs = mQwen3OmniNextChunkStates[batchIdx];
    cs.remainingTextTokens.push_back(tokenId);
    cs.active = true;
}

void Qwen3OmniTTSRuntime::finalizeOmniNextChunkStream(int32_t batchIdx)
{
    if (batchIdx < 0 || batchIdx >= static_cast<int32_t>(mQwen3OmniNextChunkStates.size()))
    {
        return;
    }
    auto& cs = mQwen3OmniNextChunkStates[batchIdx];
    cs.hasTrailingTtsEos = true;
    cs.active = true;
}

void Qwen3OmniTTSRuntime::suppressTalkerEosLogit(int32_t batchIdx, int32_t batchVocabSize, cudaStream_t stream)
{
    // D2D copy from the persistent -INF constant — async-safe (unlike a stack source).
    float* const dst = static_cast<float*>(mTalkerLogits.rawPointer()) + static_cast<int64_t>(batchIdx) * batchVocabSize
        + mTalkerConfig.codecEosId;
    CUDA_CHECK(cudaMemcpyAsync(dst, mNegInfConst.rawPointer(), sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

void Qwen3OmniTTSRuntime::trackSeenToken(std::unordered_set<int32_t>& seenSet, int32_t& numSeen, int32_t batchIdx,
    int32_t token, int32_t const* tokenDev, cudaStream_t stream)
{
    // Freshly-sampled token enters the repetition-penalty window once; the host
    // set gates the append so it stays coherent with the GPU buffer (HF
    // penalizes the ids generated within the current call).
    if (!seenSet.insert(token).second)
    {
        return;
    }
    int32_t* const dst = mSeenCodecTokensBuf.dataPointer<int32_t>()
        + static_cast<int64_t>(batchIdx) * mTalkerLLMConfig.maxKVCacheCapacity + numSeen;
    if (tokenDev != nullptr)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst, tokenDev, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        // Async H2D without racing: the stack local `token` can't be the source
        // (it would outlive the copy), so stage it into a persistent pinned
        // host slot keyed by batchIdx. Each batch owns its slot, and this path
        // is only hit once per batch at seed time, so no in-flight copy is
        // overwritten before it completes.
        int32_t* const seedHost = mSeenSeedHostScratch.dataPointer<int32_t>() + batchIdx;
        *seedHost = token;
        CUDA_CHECK(cudaMemcpyAsync(dst, seedHost, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    ++numSeen;
}

bool Qwen3OmniTTSRuntime::reprefillQwen3OmniNextChunk(int32_t batchIdx,
    std::vector<std::vector<int32_t>> const& rvqCodes, int32_t& outFirstCodecTok,
    SamplingParams const& talkerSamplingParams, float repetitionPenalty, int32_t& numSeenTokens,
    std::unordered_set<int32_t>& seenTokenSet, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::reprefillQwen3OmniNextChunk", nvtx_colors::PALE_GREEN);

    if (batchIdx < 0 || batchIdx >= static_cast<int32_t>(mQwen3OmniNextChunkStates.size()))
    {
        LOG_ERROR("reprefillQwen3OmniNextChunk: batchIdx %d out of range (size=%zu)", batchIdx,
            mQwen3OmniNextChunkStates.size());
        return false;
    }
    auto& cs = mQwen3OmniNextChunkStates[batchIdx];
    if (!cs.active)
    {
        return true;
    }

    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const framesAvail = static_cast<int32_t>(rvqCodes.size()) - cs.firstFrameOfCallIdx;
    if (framesAvail < cs.codecEmbedFrames)
    {
        LOG_ERROR("reprefillQwen3OmniNextChunk: only %d frames since last prefill, need at least %d", framesAvail,
            cs.codecEmbedFrames);
        return false;
    }

    // (1) Codec-embed rows for the FIRST codecEmbedFrames frames of this call. PT drops the
    // last "lookahead" frame from the codec_input handoff — feeding all framesPerCall breaks
    // cumulative seqLen alignment with PT (108 vs 107) and degrades long-utterance audio.
    __half* const dstBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    int64_t writeRow = cs.cumulativeSeqLen;
    for (int32_t f = 0; f < cs.codecEmbedFrames; ++f)
    {
        auto const& frameCodes = rvqCodes[cs.firstFrameOfCallIdx + f];
        if (static_cast<int32_t>(frameCodes.size()) != numCodeGroups)
        {
            LOG_ERROR("reprefillQwen3OmniNextChunk: rvqCodes[%d] size %zu != numCodeGroups %d",
                cs.firstFrameOfCallIdx + f, frameCodes.size(), numCodeGroups);
            return false;
        }
        int64_t hostCodes[64]; // numCodeGroups <= 64
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            hostCodes[g] = static_cast<int64_t>(frameCodes[g]);
        }
        sumSpeakerCodecRow(hostCodes, dstBase, writeRow++, stream);
    }

    // (2) Next text chunk (up to chunkTokensPerCall rows) + optional trailing tts_eos.
    int32_t const textRemain = static_cast<int32_t>(cs.remainingTextTokens.size()) - cs.cursorToken;
    int32_t const textRows = std::min(cs.chunkTokensPerCall, textRemain);
    for (int32_t i = 0; i < textRows; ++i)
    {
        if (!copyEmbedRow(mTextEmbeddingTable, cs.remainingTextTokens[cs.cursorToken + i], dstBase, writeRow++, stream))
        {
            return false;
        }
    }
    cs.cursorToken += textRows;

    bool appendedEos = false;
    bool const textExhausted = cs.cursorToken >= static_cast<int32_t>(cs.remainingTextTokens.size());
    if (textExhausted && cs.hasTrailingTtsEos && textRows < cs.chunkTokensPerCall)
    {
        copyRawRow(mTtsEosEmbed, dstBase, writeRow++, stream);
        cs.hasTrailingTtsEos = false;
        appendedEos = true;
    }
    cs.cumulativeSeqLen = writeRow;

    bool const exhaustedAfter = textExhausted && !cs.hasTrailingTtsEos;
    int64_t const maxInputLen = mTalkerLLMConfig.maxSupportedInputLength;
    if (cs.cumulativeSeqLen > maxInputLen)
    {
        LOG_ERROR("reprefillQwen3OmniNextChunk: cumulative seq %ld exceeds maxInputLen %ld", cs.cumulativeSeqLen,
            maxInputLen);
        return false;
    }

    LOG_INFO("OmniNext re-prefill: cumSeq=%ld (added %d codec + %d text%s); more chunks=%s", cs.cumulativeSeqLen,
        cs.codecEmbedFrames, textRows, appendedEos ? " + tts_eos" : "", exhaustedAfter ? "no" : "yes");

    // (3) Re-prefill — executeTalkerPrefillStep resets KV cache implicitly.
    check::check(mTalkerInputEmbeds.reshape({1, cs.cumulativeSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, cs.cumulativeSeqLen, hiddenSize}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
        {
            LOG_ERROR("OmniNext chunked re-prefill failed at cumSeq=%ld", cs.cumulativeSeqLen);
            return false;
        }
    }

    // (4) Sample first codec token of the new call. Suppress EOS while more chunks are still due
    // (PT enforces min_new_tokens=chunk_m+1 on every non-last call). Vary the philox offset per
    // call so every talker sample sees a distinct RNG state.
    // HF reissues a fresh generate() per chunk call, so the repetition-penalty window
    // resets at every re-prefill; only the post-text final stretch accumulates.
    seenTokenSet.clear();
    numSeenTokens = 0;
    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerSuppressStart,
        mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, numSeenTokens, repetitionPenalty, stream);
    if (!exhaustedAfter)
    {
        suppressTalkerEosLogit(batchIdx, mTalkerConfig.talkerVocabSize, stream);
    }
    uint64_t const reprefillOffset = static_cast<uint64_t>(mQwen3OmniNextChunkStates[batchIdx].firstFrameOfCallIdx) + 1;
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream, /*philoxSeed=*/42, /*philoxOffset=*/reprefillOffset);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    outFirstCodecTok = mHostSelectedTokenIds.dataPointer<int32_t>()[0];

    trackSeenToken(
        seenTokenSet, numSeenTokens, batchIdx, outFirstCodecTok, mTalkerSelectedIndices.dataPointer<int32_t>(), stream);

    cs.framesSinceLastPrefill = 0;
    cs.firstFrameOfCallIdx = static_cast<int32_t>(rvqCodes.size());
    if (exhaustedAfter)
    {
        cs.active = false;
    }
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

    // text_projection: mStreamingTokenEmbed [1, thinkerH] → mStreamingProjOut [1, talkerH].
    // Qwen3-Next Omni Talker has its own embed_tokens at talker hidden dim, so just copy.
    if (isOmniNext())
    {
        CUDA_CHECK(cudaMemcpyAsync(mStreamingProjOut.rawPointer(), mStreamingTokenEmbed.rawPointer(),
            talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeTalkerMLP(mStreamingTokenEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
            mStreamingProjOut, mStreamingMlpWork, stream);
    }

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

bool Qwen3OmniTTSRuntime::handleStreamingGeneration(LLMInferenceRuntime& thinkerRuntime,
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
    SamplingParams predictorSamplingParams(1, mTalkerConfig.codebookSize,
        isOmniNext() ? kCPSamplingTemperatureNext : kCPSamplingTemperature, kCPSamplingTopK,
        isOmniNext() ? kCPSamplingTopPNext : kCPSamplingTopP);

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
        std::vector<std::vector<int32_t>> rvqCodes;
        std::vector<int32_t> inputIds;
    };
    StreamingState state;
    state.rvqCodes.reserve(omniBaseRequest.maxAudioLength);

    // Shared chunk emitter — same accumulator used by runTalkerGenerationLoop's TTS streaming path.
    ChunkEmitter emitter{streamingConfig.codecChunkFrames, streamingConfig.onAudioChunkReady, {}};

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
            auto& talkerCacheManager = *mTalkerSharedRes->cacheManagers[0];
            auto& cpCacheManager = *mCodePredictorSharedRes->cacheManagers[0];
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

            // Use buildTalkerPrefillFromSegments for segment parsing, MLP projection, and prefill assembly.
            // For Qwen3-Next Omni (isOmniNext()), dispatch to buildQwen3OmniNextTalkerPrefill — mirrors the
            // sequential batched dispatch at the buildOneBatch lambda. The OmniNext builder uses Talker's
            // own embed_tokens internally and does NOT consume prefillEmbedPtr or thinkerEmbedTable.
            auto prefillEmbedPtr = thinkerRuntime.getBaseModelHiddenStates(0);
            auto prefillHiddenPtr = thinkerRuntime.getBaseModelHiddenStates(thinkerRequest.acceptHiddenLayer);
            int32_t const prefillLen = thinkerRuntime.getBaseModelPrefillLength();

            int64_t outSeqLen = 0;
            bool const buildOk = isOmniNext()
                ? buildQwen3OmniNextTalkerPrefill(textTokenIds, prefillHiddenPtr, prefillLen, speakerId,
                      mStreamingTrailingHidden, state.trailingIdx, outSeqLen, stream)
                : buildTalkerPrefillFromSegments(textTokenIds, prefillEmbedPtr, prefillHiddenPtr, prefillLen,
                      thinkerEmbedTable, speakerId, mStreamingTrailingHidden, state.trailingIdx, outSeqLen, stream);
            if (!buildOk)
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

            kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerSuppressStart,
                mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
            // Streaming Talker runs at batch=1, so selectedIndices must match the sampler's {1, 1} input shape.
            check::check(mTalkerSelectedIndices.reshape({1, 1}), "Tensor reshape failed");
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
            // Always record (matches handleAudioGeneration / handleAudioGenerationFromThinker):
            // streaming callers consume the event for TTFC measurement even without profiling.
            if (mTtfaEnd)
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
                if (isOmniNext())
                {
                    int32_t const outcome = maybeReprefillOmniNextChunkForBatch(0, state.rvqCodes, state.codecToken,
                        talkerSamplingParams, repetitionPenalty, numSeenTokens, seenTokenSet, stream);
                    if (outcome == -1)
                    {
                        state.talkerError = true;
                        return;
                    }
                }
            }
        }
        else if (state.talkerPrefillDone && numAssistantTokens > prefillThreshold)
        {
            if (isOmniNext())
            {
                // OmniNext streaming: feed new Thinker tokens into the chunk stream — the model
                // is trained for chunk-based re-prefills, not per-decode-step text injection.
                appendOmniNextChunkStreamToken(0, info.tokenId);
                if (state.thinkerFinished)
                {
                    finalizeOmniNextChunkStream(0);
                }
            }
            else if (state.trailingIdx >= maxTrailingLen - 1)
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
                rt::Tensor const* trailingPtr = (!isOmniNext() && state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;
                emitter.append(state.rvqCodes.back());
                if (isOmniNext())
                {
                    int32_t const outcome = maybeReprefillOmniNextChunkForBatch(0, state.rvqCodes, state.codecToken,
                        talkerSamplingParams, repetitionPenalty, numSeenTokens, seenTokenSet, stream);
                    if (outcome == -1)
                    {
                        state.talkerError = true;
                        return;
                    }
                    if (state.codecToken == codecEosId)
                    {
                        return;
                    }
                }
            }
        }
    };

    // ===== Run Thinker with the callback installed =====
    auto hiddenLayers = getThinkerHiddenLayerIndices();
    if (hiddenLayers[1] < 0)
    {
        LOG_ERROR(
            "Talker config is missing 'accept_hidden_layer'; Thinker->Talker streaming requires it to "
            "match the layer index the Thinker engine emits on its hidden_states output.");
        return false;
    }
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
        if (isOmniNext())
        {
            // OmniNext consumes tts_eos through the chunk stream, not the trailing buffer.
            finalizeOmniNextChunkStream(0);
        }
        else if (state.trailingIdx < maxTrailingLen)
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
        rt::Tensor const* flushTrailingPtr = (!isOmniNext() && state.trailingIdx > 0) ? &flushTrailingView : nullptr;

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
            emitter.append(state.rvqCodes.back());
            if (isOmniNext())
            {
                int32_t const outcome = maybeReprefillOmniNextChunkForBatch(0, state.rvqCodes, state.codecToken,
                    talkerSamplingParams, repetitionPenalty, numSeenTokens, seenTokenSet, stream);
                if (outcome == -1)
                {
                    break;
                }
            }
        }

        emitter.flushFinal();
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
