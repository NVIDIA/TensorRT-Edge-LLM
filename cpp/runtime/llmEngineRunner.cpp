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

#include "runtime/llmEngineRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "common/version.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{
//! Dummy dimension for LoRA weights when no LoRA is active (use 1 instead of 0 to avoid zero-shape issues)
constexpr int32_t kEMPTY_LORA_RANK = 1;

std::string formatEngineConfig(trt_edgellm::rt::LLMEngineRunnerConfig const& config)
{
    std::stringstream ss;

    ss << std::boolalpha;
    ss << "LLMEngineRunnerConfig:"
       << "  enableEagleSpecDecode: " << config.enableEagleSpecDecode << "  isVlm: " << config.isVlm
       << "  numDecoderLayers: " << config.numDecoderLayers << "  numKVHeads: " << config.numKVHeads
       << "  headDim: " << config.headDim << "  rotaryDim: " << config.rotaryDim
       << "  hiddenSize: " << config.hiddenSize << "  maxSupportedBatchSize: " << config.maxSupportedBatchSize
       << "  minSupportedInputLength: " << config.minSupportedInputLength
       << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  maxKVCacheCapacity: " << config.maxKVCacheCapacity
       << "  maxSupportedLoraRank: " << config.maxSupportedLoraRank;
    if (config.enableEagleSpecDecode)
    {
        ss << "  outputHiddenDim (For Eagle SpecDecode): " << config.outputHiddenDim;
        ss << "  maxVerifyTreeSize (For Eagle SpecDecode): " << config.maxVerifyTreeSize;
    }
    return ss.str();
}

// Compute a unique hash value that can distinguish the various decoding steps.
// Extend this function when we need to capture more information.
size_t hashDecodingInput(rt::Tensor const& inputIds, rt::Tensor const& outputLogits, std::string const& loraWeightsName)
{
    // For vanilla decoding step, the shape can be distingusihed by active batch size.
    // Also capture the pointer address to ensure we are read/write correct locations.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    uintptr_t const inputIdsAddr = reinterpret_cast<uintptr_t>(inputIds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());

    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, activeBatchSize);
    hash_utils::hashCombine(hashValue, inputIdsAddr);
    hash_utils::hashCombine(hashValue, outputLogitsAddr);
    hash_utils::hashCombine(hashValue, loraWeightsName);
    return hashValue;
}

size_t hashBaseTreeDecodingInput(
    rt::Tensor const& baseTreeDecodingInputIds, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates)
{
    int64_t const activeBatchSize = baseTreeDecodingInputIds.getShape()[0];
    uintptr_t const inputIdsAddr = reinterpret_cast<uintptr_t>(baseTreeDecodingInputIds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());
    uintptr_t const outputHiddenStatesAddr = reinterpret_cast<uintptr_t>(outputHiddenStates.rawPointer());

    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, activeBatchSize);
    hash_utils::hashCombine(hashValue, inputIdsAddr);
    hash_utils::hashCombine(hashValue, outputLogitsAddr);
    hash_utils::hashCombine(hashValue, outputHiddenStatesAddr);
    return hashValue;
}

} // namespace

namespace trt_edgellm
{
namespace rt
{

//! Current implementation limits to two optimization profiles per LLM engine.
static constexpr int32_t kPREFILL_PROFILE_INDEX{0};
static constexpr int32_t kGENERATION_PROFILE_INDEX{1};

LLMEngineRunner::LLMEngineRunner(std::filesystem::path const& enginePath, std::filesystem::path const& configPath,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    LOG_INFO("Loading config file %s", configPath.string().c_str());

    // Parse configuration from JSON file
    Json configJson;
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file with error: %s", e.what());
        throw std::runtime_error("Failed to parse config file: " + configPath.string());
    }

    if (!this->initializeConfigFromJson(configJson))
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner from config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to initialize LLMEngineRunner from config file: " + configPath.string());
    }

    // Load the engine after config loading succeeds
    LOG_INFO("Loading engine file: %s", enginePath.string().c_str());
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("LLMEngineRunner(): Failed to use MMap to read engine from file path: %s", enginePath.string());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    int64_t const execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
    // Allocate device memory for the execution contexts. UINT8 is used to represent raw bytes.
    mExecContextMemory = rt::Tensor({execContextMemoryInBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "LLMEngineRunner::mExecContextMemory");

    mPrefillExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));

    // The prefill and generation contexts of the LLM engine execute serially, can therefore share a single device
    // memory block.
    mPrefillExecutionContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(), execContextMemoryInBytes);
    mGenerationExecutionContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(), execContextMemoryInBytes);
    LOG_INFO("Allocated a shared device memory of %zu bytes for the prefill and generation contexts.",
        execContextMemoryInBytes);

    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mPrefillExecutionContext->setOptimizationProfileAsync(kPREFILL_PROFILE_INDEX, stream);
    setOptimizationProfileStatus
        &= mGenerationExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    if (!this->validateConfigFromEngine())
    {
        LOG_ERROR("Failed to match config file %s with engine file: %s", configPath.string().c_str(),
            enginePath.string().c_str());
        throw std::runtime_error(
            "Failed to match config file " + configPath.string() + " with engine file: " + enginePath.string());
    }

    RopeConfig const& ropeConfig = mConfig.ropeConfig;
    switch (ropeConfig.type)
    {
    case RopeType::kLongRope:
    {
        LOG_DEBUG("Initialize long Rope CosSinCache.");
        check::check(ropeConfig.longRope.has_value() && ropeConfig.longRope.value().originalMaxPositionEmbeddings != -1,
            "longRope is not set correctly");

        rt::Tensor shortCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim},
            rt::DeviceType::kGPU, DataType::kFLOAT, "LLMEngineRunner::shortCosSinCache");
        rt::Tensor longCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim},
            rt::DeviceType::kGPU, DataType::kFLOAT, "LLMEngineRunner::longCosSinCache");
        bool const initRopeStatus
            = initializeLongRopeCosSinCache(shortCosSinCache, longCosSinCache, ropeConfig, configJson, stream);
        if (!initRopeStatus)
        {
            LOG_ERROR("Failed to initialize long Rope CosSinCache.");
            throw std::runtime_error("Failed to initialize long Rope CosSinCache.");
        }
        if (mConfig.maxKVCacheCapacity <= ropeConfig.longRope.value().originalMaxPositionEmbeddings)
        {
            mPosEncCosSinCache = std::move(shortCosSinCache);
        }
        else
        {
            mPosEncCosSinCache = std::move(longCosSinCache);
        }
        break;
    }
    case RopeType::kMRope:
    {
        this->mPosEncCosSinCache
            = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim},
                rt::DeviceType::kGPU, DataType::kFLOAT, "LLMEngineRunner::mPosEncCosSinCache");
        CUDA_CHECK(cudaMemsetAsync(mPosEncCosSinCache.rawPointer(), 0, mPosEncCosSinCache.getMemoryCapacity(), stream));
        break;
    }
    default:
    {
        LOG_DEBUG("Initialize persistent Rope CosSinCache.");
        this->mPosEncCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMEngineRunner::mPosEncCosSinCache");
        bool const initRopeStatus = initializeRopeCosSinCache(mPosEncCosSinCache, ropeConfig, configJson, stream);
        if (!initRopeStatus)
        {
            LOG_ERROR("Failed to initialize persistent Rope CosSinCache.");
            throw std::runtime_error("Failed to initialize persistent Rope CosSinCache.");
        }
        break;
    }
    }
    // Bind RopeCosSin cache
    bool setRopeCosSinCacheStatus{true};
    setRopeCosSinCacheStatus
        &= mPrefillExecutionContext->setTensorAddress(binding_names::kRopeCosSin, mPosEncCosSinCache.rawPointer());
    setRopeCosSinCacheStatus
        &= mGenerationExecutionContext->setTensorAddress(binding_names::kRopeCosSin, mPosEncCosSinCache.rawPointer());
    if (!setRopeCosSinCacheStatus)
    {
        LOG_ERROR("Failed to set rope cos sin cache to the engine");
        throw std::runtime_error("Failed to set rope cos sin cache to the engine");
    }

    // Instantiate the KVCache instance of the EngineRunner.
    this->mKVCache
        = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers, mConfig.maxSupportedBatchSize,
                                mConfig.maxKVCacheCapacity, mConfig.numKVHeads, mConfig.headDim},
            stream);

    // Instantiate other GPU memory input that needed by the Engine execution.
    this->mSequenceContextLengths = rt::Tensor({mConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
        "LLMEngineRunner::mSequenceContextLengths");
    CUDA_CHECK(
        cudaMemsetAsync(mSequenceContextLengths.rawPointer(), 0, mSequenceContextLengths.getMemoryCapacity(), stream));

    if (mConfig.enableEagleSpecDecode)
    {
        // For EAGLE: last_token_ids is 2D [batch_size, num_selected_tokens] to support multi-batch
        this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxVerifyTreeSize},
            rt::DeviceType::kGPU, DataType::kINT64, "LLMEngineRunner::mSelectTokenIndices");
        CUDA_CHECK(
            cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, mSelectTokenIndices.getMemoryCapacity(), stream));
        this->mHostSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxVerifyTreeSize},
            rt::DeviceType::kCPU, DataType::kINT64, "LLMEngineRunner::mHostSelectTokenIndices");
        this->mEagleBasePositionIds = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxVerifyTreeSize},
            rt::DeviceType::kGPU, DataType::kINT32, "LLMEngineRunner::mEagleBasePositionIds");
        CUDA_CHECK(
            cudaMemsetAsync(mEagleBasePositionIds.rawPointer(), 0, mEagleBasePositionIds.getMemoryCapacity(), stream));
        int32_t const packedMaskSize = divUp(mConfig.maxVerifyTreeSize, 32);
        this->mEagleBasePackedMask
            = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxVerifyTreeSize, packedMaskSize},
                rt::DeviceType::kGPU, DataType::kINT32, "LLMEngineRunner::mEagleBasePackedMask");
        CUDA_CHECK(
            cudaMemsetAsync(mEagleBasePackedMask.rawPointer(), 0, mEagleBasePackedMask.getMemoryCapacity(), stream));
    }
    else
    {
        this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU,
            DataType::kINT64, "LLMEngineRunner::mSelectTokenIndices");
        CUDA_CHECK(
            cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, mSelectTokenIndices.getMemoryCapacity(), stream));
        this->mHostSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kCPU,
            DataType::kINT64, "LLMEngineRunner::mHostSelectTokenIndices");
    }

    // Add the LoRA weights to the engine.
    if (isLoraWeightsSupported())
    {
        for (auto const& [loraWeightsName, loraWeightsPath] : loraWeightsMap)
        {
            if (loraWeightsPath.empty())
            {
                continue;
            }
            if (!this->addLoraWeights(loraWeightsName, loraWeightsPath, stream))
            {
                LOG_ERROR("Failed to add LoRA weights: %s", loraWeightsName.c_str());
                throw std::runtime_error("Failed to add LoRA weights: " + loraWeightsName);
            }
        }
    }

    // Initialize the dummy tensor as TensorRT does not support nullptr for binding
    // Calculate maximum memory requirements across all use cases:
    // 1. Multimodal embeddings: {1, hiddenSize}
    // 2. Attention mask: {maxSupportedBatchSize, 1, 1}
    // 3. Attention position IDs: {maxSupportedBatchSize, 1}
    // 4. LoRA weights: max dimension across all adapters
    // 5. KV cache start index: {maxSupportedBatchSize}
    int64_t maxDummyElements = std::max({
        static_cast<int64_t>(mConfig.hiddenSize),            // multimodal embeddings
        static_cast<int64_t>(mConfig.maxSupportedBatchSize), // attention mask/pos IDs/KV cache start index
        static_cast<int64_t>(getMaxLoraWeightsDimension() * kEMPTY_LORA_RANK), // LoRA weights
    });
    mDummyTensor = rt::Tensor(
        {maxDummyElements}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "LLMEngineRunner::mDummyTensor");
    // Initialize dummy tensor memory to zero
    CUDA_CHECK(cudaMemsetAsync(mDummyTensor.rawPointer(), 0, mDummyTensor.getMemoryCapacity(), stream));

    // Set multimodal embeddings to dummy tensor for generation contexts if VLM is enabled
    if (mConfig.isVlm)
    {
        bool setMultimodalStatus{true};
        setMultimodalStatus
            &= mGenerationExecutionContext->setTensorAddress(binding_names::kImageEmbeds, mDummyTensor.rawPointer());
        setMultimodalStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kImageEmbeds, rt::Coords{1, mConfig.hiddenSize}.getTRTDims());

        // Set deepstack features if exists.
        for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
        {
            std::string deepstackFeatureName = binding_names::formatDeepstackFeaturesName(idx);
            setMultimodalStatus &= mGenerationExecutionContext->setTensorAddress(
                deepstackFeatureName.c_str(), mDummyTensor.rawPointer());
            setMultimodalStatus &= mGenerationExecutionContext->setInputShape(
                deepstackFeatureName.c_str(), rt::Coords{1, mConfig.hiddenSize}.getTRTDims());
        }

        if (!setMultimodalStatus)
        {
            LOG_ERROR("Failed to set multimodal embeddings dummy tensor for generation context");
            throw std::runtime_error("Failed to set multimodal embeddings dummy tensor for generation context");
        }
    }

    // Reset the LoRA weights to zero tensors.
    if (!this->resetLoraWeights(stream))
    {
        LOG_ERROR("Failed to initialize LoRA weights to zero tensors");
        throw std::runtime_error("Failed to initialize LoRA weights to zero tensors");
    }

    // Synchronize the stream to ensure all the operations have completed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool LLMEngineRunner::initializeConfigFromJson(Json const& configJson)
{
    try
    {
        // Check model version
        std::string modelVersion = configJson.value(binding_names::kEdgellmVersion, "");
        version::checkVersion(modelVersion);

        // Define required fields for main config
        std::vector<std::string> const requiredConfigFields
            = {"num_hidden_layers", "num_key_value_heads", "head_dim", "vocab_size", "builder_config"};

        // Validate required fields exist in main config
        for (auto const& field : requiredConfigFields)
        {
            if (!configJson.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in config", field.c_str());
                return false;
            }
        }

        auto const& builderConfig = configJson["builder_config"];

        // Define required fields for builder_config
        std::vector<std::string> const requiredBuilderConfigFields
            = {"max_batch_size", "max_input_len", "max_kv_cache_capacity", "max_lora_rank", "eagle_base", "is_vlm"};

        // Validate required fields exist in builder_config
        for (auto const& field : requiredBuilderConfigFields)
        {
            if (!builderConfig.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in builder_config", field.c_str());
                return false;
            }
        }

        // Extract values with proper type checking
        mConfig.numDecoderLayers = configJson["num_hidden_layers"].get<int32_t>();
        mConfig.numKVHeads = configJson["num_key_value_heads"].get<int32_t>();
        mConfig.headDim = configJson["head_dim"].get<int32_t>();
        mConfig.rotaryDim = static_cast<int32_t>(mConfig.headDim * configJson.value("partial_rotary_factor", 1.0f));
        mConfig.hiddenSize = configJson["hidden_size"].get<int32_t>();
        mConfig.vocabSize = configJson["vocab_size"].get<int32_t>();
        // Optional: reduced vocabulary size (0 if not present)
        mConfig.reducedVocabSize = configJson.value(binding_names::kReducedVocabSizeKey, 0);
        // Set actual output vocab size: use reduced size if enabled, otherwise full size
        mConfig.outputVocabSize = (mConfig.reducedVocabSize > 0) ? mConfig.reducedVocabSize : mConfig.vocabSize;

        // Extract builder_config values
        mConfig.isVlm = builderConfig["is_vlm"].get<bool>();
        mConfig.maxSupportedBatchSize = builderConfig["max_batch_size"].get<int32_t>();
        mConfig.minSupportedInputLength = 1; // TODO: Change this to min input length
        mConfig.maxSupportedInputLength = builderConfig["max_input_len"].get<int32_t>();
        mConfig.maxKVCacheCapacity = builderConfig["max_kv_cache_capacity"].get<int32_t>();
        mConfig.maxSupportedLoraRank = builderConfig["max_lora_rank"].get<int32_t>();
        mConfig.enableEagleSpecDecode = builderConfig["eagle_base"].get<bool>();

        // Collect RoPE configuration
        mConfig.ropeConfig = collectRopeConfig(configJson);

        // Validate configuration values - all must be positive except max_lora_rank
        std::vector<std::pair<std::string, int32_t>> positiveFields = {{"num_decoder_layers", mConfig.numDecoderLayers},
            {"num_key_value_heads", mConfig.numKVHeads}, {"head_dim", mConfig.headDim},
            {"rotary_dim", mConfig.rotaryDim}, {"hidden_size", mConfig.hiddenSize}, {"vocab_size", mConfig.vocabSize},
            {"max_batch_size", mConfig.maxSupportedBatchSize}, {"max_input_len", mConfig.maxSupportedInputLength},
            {"max_kv_cache_capacity", mConfig.maxKVCacheCapacity}};

        for (auto const& [fieldName, value] : positiveFields)
        {
            if (value <= 0)
            {
                LOG_ERROR("initializeConfigFromJson(): Invalid %s: %d (must be positive)", fieldName.c_str(), value);
                return false;
            }
        }

        // FIXME: Not a proper way to determine the output hidden dim.
        // Hardcode output hidden_dim to 3 x model hidden_size which is default in eagle3.
        if (mConfig.enableEagleSpecDecode)
        {
            mConfig.outputHiddenDim = configJson["hidden_size"].get<int32_t>() * 3;

            // maxVerifyTreeSize is only required when eagle_base is true
            if (!builderConfig.contains("max_verify_tree_size"))
            {
                LOG_ERROR(
                    "initializeConfigFromJson(): Missing required field 'max_verify_tree_size' in builder_config for "
                    "Eagle base model");
                return false;
            }
            mConfig.maxVerifyTreeSize = builderConfig["max_verify_tree_size"].get<int32_t>();

            // Validate maxVerifyTreeSize (must be positive)
            if (mConfig.maxVerifyTreeSize <= 0)
            {
                LOG_ERROR("initializeConfigFromJson(): Invalid max_verify_tree_size: %d (must be positive)",
                    mConfig.maxVerifyTreeSize);
                return false;
            }
        }

        // Validate max_lora_rank separately (must be non-negative)
        if (mConfig.maxSupportedLoraRank < 0)
        {
            LOG_ERROR("initializeConfigFromJson(): Invalid max_lora_rank: %d (must be non-negative)",
                mConfig.maxSupportedLoraRank);
            return false;
        }
        if (mConfig.maxSupportedInputLength > mConfig.maxKVCacheCapacity)
        {
            LOG_ERROR(
                "initializeConfigFromJson(): Invalid configuration: max_input_len (%d) cannot be greater than "
                "max_kv_cache_capacity (%d)",
                mConfig.maxSupportedInputLength, mConfig.maxKVCacheCapacity);
            return false;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("initializeConfigFromJson(): Unexpected error while parsing config: %s", e.what());
        return false;
    }

    LOG_INFO("initializeConfigFromJson(): Loaded LLMEngineRunner with config: %s", formatEngineConfig(mConfig).c_str());
    return true;
}

bool LLMEngineRunner::validateConfigFromEngine()
{
    auto identifyKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 5 && bindingName.find(binding_names::kPastKeyValuesTemplate) != std::string::npos;
    };

    // If the engine comes with multimodal embeddings binding, it means the engine supports VLM.
    auto identifyMultimodalEmbeddingsBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 2 && bindingName == binding_names::kImageEmbeds;
    };

    // If the engine comes with deepstack features binding, it means the engine is Qwen3-VL.
    auto identifyDeepstackFeaturesBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 2
            && bindingName.find(binding_names::kDeepstackFeaturesTemplate) != std::string::npos;
    };

    int32_t nbKVCacheInputs{0};
    bool foundMultimodalEmbeddingsInput{false};
    int32_t numIOBindings = mEngine->getNbIOTensors();

    // Lambda to validate KV cache dimensions against profile shape
    auto validateKVCacheProfile = [&](Dims const& maxKVCacheShape, std::string const& profileName) -> bool {
        if (mConfig.numKVHeads != maxKVCacheShape.d[2])
        {
            LOG_ERROR("numKVHeads is not consistent. From engine %s profile: %d, from config: %d", profileName.c_str(),
                maxKVCacheShape.d[2], mConfig.numKVHeads);
            return false;
        }
        if (mConfig.maxKVCacheCapacity != maxKVCacheShape.d[3])
        {
            LOG_ERROR("maxKVCacheCapacity is not consistent. From engine %s profile max: %d, from config: %d",
                profileName.c_str(), maxKVCacheShape.d[3], mConfig.maxKVCacheCapacity);
            return false;
        }
        if (mConfig.headDim != maxKVCacheShape.d[4])
        {
            LOG_ERROR("headDim is not consistent. From engine %s profile: %d, from config: %d", profileName.c_str(),
                maxKVCacheShape.d[4], mConfig.headDim);
            return false;
        }
        return true;
    };

    for (int32_t i = 0; i < numIOBindings; ++i)
    {
        std::string const bindingName = mEngine->getIOTensorName(i);
        Dims const tensorDim = mEngine->getTensorShape(bindingName.c_str());

        if (identifyKVCacheBinding(bindingName, tensorDim))
        {
            // Get max profile shapes for both prefill and generation profiles
            Dims const maxKVCacheShapePrefill
                = mEngine->getProfileShape(bindingName.c_str(), kPREFILL_PROFILE_INDEX, OptProfileSelector::kMAX);
            Dims const maxKVCacheShapeGen
                = mEngine->getProfileShape(bindingName.c_str(), kGENERATION_PROFILE_INDEX, OptProfileSelector::kMAX);

            // Validate both profiles
            if (!validateKVCacheProfile(maxKVCacheShapePrefill, "prefill"))
            {
                return false;
            }
            if (!validateKVCacheProfile(maxKVCacheShapeGen, "generation"))
            {
                return false;
            }
            ++nbKVCacheInputs;
        }
        if (identifyMultimodalEmbeddingsBinding(bindingName, tensorDim))
        {
            foundMultimodalEmbeddingsInput = true;
            if (mConfig.hiddenSize != tensorDim.d[1])
            {
                LOG_ERROR("hiddenSize is not consistent. From engine multimodal embeddings: %d, from config: %d",
                    tensorDim.d[1], mConfig.hiddenSize);
                return false;
            }
            if (!mConfig.isVlm)
            {
                LOG_ERROR("VLM is not enabled but multimodal embeddings input image_embeds found in engine");
                return false;
            }
        }
        if (identifyDeepstackFeaturesBinding(bindingName, tensorDim))
        {
            if (mConfig.hiddenSize != tensorDim.d[1])
            {
                LOG_ERROR("hiddenSize is not consistent. From engine multimodal embeddings: %d, from config: %d",
                    tensorDim.d[1], mConfig.hiddenSize);
                return false;
            }
            if (!mConfig.isVlm)
            {
                LOG_ERROR("VLM is not enabled but deepstack features input found in engine");
                return false;
            }
            LOG_DEBUG("validateConfigFromEngine(): Found deepstack features binding: %s", bindingName.c_str());
            ++mConfig.numDeepstackFeatures;
        }
    }
    // Validate hiddenSize from multimodal embeddings if VLM is enabled
    if (mConfig.isVlm && !foundMultimodalEmbeddingsInput)
    {
        LOG_ERROR(
            "VLM is enabled but multimodal embeddings input (%s) not found in engine", binding_names::kImageEmbeds);
        return false;
    }
    if (nbKVCacheInputs != mConfig.numDecoderLayers)
    {
        LOG_ERROR("numDecoderLayers is not consistent. From engine: %d, from config: %d", nbKVCacheInputs,
            mConfig.numDecoderLayers);
        return false;
    }
    Dims const minInputPrefillShape
        = mEngine->getProfileShape(binding_names::kInputIds, kPREFILL_PROFILE_INDEX, OptProfileSelector::kMIN);
    Dims const maxInputPrefillShape
        = mEngine->getProfileShape(binding_names::kInputIds, kPREFILL_PROFILE_INDEX, OptProfileSelector::kMAX);
    if (mConfig.minSupportedInputLength != minInputPrefillShape.d[1])
    {
        LOG_ERROR("minSupportedInputLength is not consistent. From engine: %d, from config: %d",
            minInputPrefillShape.d[1], mConfig.minSupportedInputLength);
        return false;
    }
    if (mConfig.maxSupportedInputLength != maxInputPrefillShape.d[1])
    {
        LOG_ERROR("maxSupportedInputLength is not consistent. From engine: %d, from config: %d",
            maxInputPrefillShape.d[1], mConfig.maxSupportedInputLength);
        return false;
    }

    // Validate and potentially override maxSupportedBatchSize from engine's actual max profile
    int32_t const engineMaxBatchSize = maxInputPrefillShape.d[0];
    if (mConfig.maxSupportedBatchSize != engineMaxBatchSize)
    {
        LOG_ERROR("maxSupportedBatchSize mismatch! Config is %d, engine's max optimization profile is %d.",
            mConfig.maxSupportedBatchSize, engineMaxBatchSize);
        return false;
    }

    // Obtain vocab size from the engine.
    // Logits shape is [batch_size, num_tokens/num_selected_tokens, vocab_size] for both EAGLE and vanilla models
    Dims const logitsDim = mEngine->getTensorShape(binding_names::kLogits);
    if (mConfig.outputVocabSize != logitsDim.d[2])
    {
        LOG_ERROR("vocabSize is not consistent. From engine: %d, expected output vocab size: %d", logitsDim.d[2],
            mConfig.outputVocabSize);
        return false;
    }

    // Obtain rotary dim from the engine.
    Dims const ropeCosSinCacheDim = mEngine->getTensorShape(binding_names::kRopeCosSin);
    if (mConfig.rotaryDim != ropeCosSinCacheDim.d[2])
    {
        LOG_ERROR("rotaryDim is not consistent. From engine: %d, from config: %d", ropeCosSinCacheDim.d[2],
            mConfig.rotaryDim);
        return false;
    }

    return true;
}

LLMEngineRunner::~LLMEngineRunner()
{
    for (auto& [hashValue, graphPair] : mCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
    for (auto& [hashValue, graphPair] : mBaseTreeDecodingCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
}

bool LLMEngineRunner::bindKVCacheToEngine(int32_t activeBatchSize)
{
    // Prepare special input binding shape for prefill stage KVCache input.
    Dims const kvCacheDims = {5, {activeBatchSize, 2, mConfig.numKVHeads, mConfig.maxKVCacheCapacity, mConfig.headDim}};
    bool status{true};
    // Bind KV cache tensors to execution contexts
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string const pastKeyValuesName = binding_names::formatKVCacheName(i, true);
        std::string const presentKeyValuesName = binding_names::formatKVCacheName(i, false);

        rt::Tensor kvCacheBlock = mKVCache.getKVCacheForDecoderLayer(i);
        status &= mPrefillExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mPrefillExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mGenerationExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status
            &= mGenerationExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());

        status &= mPrefillExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDims);
        status &= mGenerationExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDims);
    }
    return status;
}

rt::Tensor& LLMEngineRunner::getRopeCosSinCacheTensor()
{
    return mPosEncCosSinCache;
}

LLMEngineRunnerConfig LLMEngineRunner::getEngineConfig() const
{
    return mConfig;
}

rt::LinearKVCache& LLMEngineRunner::getLinearKVCache()
{
    return mKVCache;
}

bool LLMEngineRunner::prefillStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& contextLengths,
    rt::Tensor const& outputLogits, OptionalOutputTensor outputHiddenStates,
    rt::OptionalInputTensor multimodalEmbeddings, rt::OptionalInputTensors extraInputTensors)
{
    int32_t activeBatchSize = inputIds.getShape()[0];
    int32_t prefillSequenceLength = inputIds.getShape()[1];

    bool const checkInputsGPUTensor = inputIds.getDeviceType() == rt::DeviceType::kGPU
        && contextLengths.getDeviceType() == rt::DeviceType::kCPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "Invalid device type of I/O tensors. ContextLengths input should reside on CPU and "
            "the rest should reside on GPU.");
        return false;
    }
    bool const isBatchValid = activeBatchSize <= mConfig.maxSupportedBatchSize
        && contextLengths.getShape()[0] == activeBatchSize && outputLogits.getShape()[0] == activeBatchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batchSize of the input tensors. Either batchSize is larger than "
            "maxSupportedBatchSize or batchSize is not consistent among the input tensors. "
            "Current inputIds shape: %s, contextLengths shape: %s, logits shape: %s",
            inputIds.getShape().formatString().c_str(), contextLengths.getShape().formatString().c_str(),
            outputLogits.getShape().formatString().c_str());
        return false;
    }
    if (prefillSequenceLength > mConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "Invalid sequence length of the input tensors. Input sequence length (%d) is larger "
            "than maxSupportedInputLength (%d). Current inputIds shape: %s.",
            prefillSequenceLength, mConfig.maxSupportedInputLength, inputIds.getShape().formatString().c_str());
        return false;
    }

    // Validate multimodal embeddings based on is_vlm flag
    bool const isMultimodalEmbeddingsValid
        = (mConfig.isVlm && multimodalEmbeddings.has_value()
              && multimodalEmbeddings.value().get().getShape().getNumDims() == 2
              && multimodalEmbeddings.value().get().getShape()[1] == mConfig.hiddenSize)
        || (!mConfig.isVlm && !multimodalEmbeddings.has_value());
    if (!isMultimodalEmbeddingsValid)
    {
        LOG_ERROR("Invalid multimodal embeddings. VLM=%s, provided=%s, expected shape=[*, %d]. Current shape: %s",
            mConfig.isVlm ? "true" : "false", multimodalEmbeddings.has_value() ? "true" : "false", mConfig.hiddenSize,
            multimodalEmbeddings.has_value() ? multimodalEmbeddings.value().get().getShape().formatString().c_str()
                                             : "None");
        return false;
    }

    // Validate extra input tensors, e.g. deepstack features for Qwen3-VL
    int32_t deepstackFeaturesCount{0};
    for (auto const& tensorRef : extraInputTensors)
    {
        rt::Tensor const& tensor = tensorRef.get();
        std::string const tensorName = tensor.getName();

        // Deepstack features
        if (tensorName.find(binding_names::kDeepstackFeaturesTemplate) != std::string::npos)
        {
            bool const isTensorValid = tensor.getDeviceType() == rt::DeviceType::kGPU
                && tensor.getShape().getNumDims() == 2 && tensor.getShape()[1] == mConfig.hiddenSize;
            if (!isTensorValid)
            {
                LOG_ERROR(
                    "Invalid deepstack feature '%s'. Expected device type: GPU, shape: [*, %d]. Current shape: %s",
                    tensorName.c_str(), mConfig.hiddenSize, tensor.getShape().formatString().c_str());
                return false;
            }
            ++deepstackFeaturesCount;
        }
    }
    if (deepstackFeaturesCount != mConfig.numDeepstackFeatures)
    {
        LOG_ERROR("Invalid deepstack features count. Expected %d, got %d", mConfig.numDeepstackFeatures,
            deepstackFeaturesCount);
        return false;
    }

    bool const isLogitsShapeValid
        = outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.outputVocabSize;
    if (!isLogitsShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output logits tensor. The output logits tensor should have shape "
            "[activeBatchSize, outputVocabSize]. Current logits shape is %s.",
            outputLogits.getShape().formatString().c_str());
        return false;
    }
    if (mConfig.enableEagleSpecDecode)
    {
        bool const isHiddenStatesShapeValid = outputHiddenStates.has_value()
            && outputHiddenStates.value().get().getShape().getNumDims() == 3
            && outputHiddenStates.value().get().getShape()[0] == activeBatchSize
            && outputHiddenStates.value().get().getShape()[1] == prefillSequenceLength
            && outputHiddenStates.value().get().getShape()[2] == mConfig.outputHiddenDim;
        if (!isHiddenStatesShapeValid)
        {
            LOG_ERROR(
                "With SpecDecode enabled, the output hidden states tensor shall be valid and has shape "
                "[activeBatchSize, %d, %d]. Current hidden states shape is %s.",
                prefillSequenceLength, mConfig.outputHiddenDim,
                outputHiddenStates.value().get().getShape().formatString().c_str());
            return false;
        }
    }

    return true;
}

bool LLMEngineRunner::executePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& hostContextLengths,
    rt::OptionalInputTensor multimodalEmbeddings, rt::OptionalInputTensors extraInputTensors, rt::Tensor& outputLogits,
    rt::OptionalOutputTensor outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->prefillStepInputValidation(
        inputIds, hostContextLengths, outputLogits, outputHiddenStates, multimodalEmbeddings, extraInputTensors);
    if (!validateInputStatus)
    {
        LOG_ERROR("executePrefill(): Prefill request not performed due to invalid input tensors.");
        return false;
    }

    // Verirify input tensorShape is valid.
    int32_t activeBatchSize = inputIds.getShape()[0];

    bool reshapeStatus{true};
    // conduct preparation work for the engine execution. Provide correct shapes for MISC input tensors.
    // All models (EAGLE and vanilla) now use 2D shape [batch_size, num_tokens] for last_token_ids
    reshapeStatus &= mSelectTokenIndices.reshape({activeBatchSize, 1});
    reshapeStatus &= mSequenceContextLengths.reshape({activeBatchSize});
    if (!reshapeStatus)
    {
        LOG_ERROR("Failed to reshape select token indices and sequence context lengths for prefill step.");
        return false;
    }

    mHostSelectTokenIndices.reshape({activeBatchSize, 1});
    int64_t* selectTokenIndicesData = mHostSelectTokenIndices.dataPointer<int64_t>();
    int32_t const* contextLengthsData = hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectTokenIndicesData[i] = static_cast<int64_t>(contextLengthsData[i] - 1);
    }
    CUDA_CHECK(cudaMemcpyAsync(mSelectTokenIndices.rawPointer(), mHostSelectTokenIndices.rawPointer(),
        activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), hostContextLengths.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    bool setEngineIOStatus{true};
    // Engine input tensors.
    setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
        binding_names::kInputIds, const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mPrefillExecutionContext->setInputShape(binding_names::kInputIds, inputIds.getShape().getTRTDims());
    setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
        binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mPrefillExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());

    // Setup the KVCache start index tensor. If all KVCache are empty then we can supply zero tensor to the engine.
    // Otherwise, we shall supply the KVCache lengths tensor to the engine.
    if (mKVCache.getKVCacheAllEmpty())
    {
        setEngineIOStatus
            &= mPrefillExecutionContext->setTensorAddress(binding_names::kKVCacheStartIndex, mDummyTensor.rawPointer());
        setEngineIOStatus
            &= mPrefillExecutionContext->setInputShape(binding_names::kKVCacheStartIndex, rt::Coords{0}.getTRTDims());
    }
    else
    {
        setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());
    }

    // RopeCosSin tensor address is set during object construction. We only set shape here to accommodate ND-Rope.
    // For MRope, the cache is initialized with maxBatchSize and does not need reshaping during prefill.
    // For non-MRope, the cache is fixed at {1, maxSeqLen, rotaryDim} and shared across all batches.
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }
    setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
        binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());
    if (multimodalEmbeddings.has_value())
    {
        rt::Tensor const& multimodalEmbeddingsTensor = multimodalEmbeddings.value().get();
        setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
            binding_names::kImageEmbeds, const_cast<void*>(multimodalEmbeddingsTensor.rawPointer()));
        setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
            binding_names::kImageEmbeds, multimodalEmbeddingsTensor.getShape().getTRTDims());
    }
    if (!extraInputTensors.empty())
    {
        for (auto const& tensorRef : extraInputTensors)
        {
            // Bind the extra input tensor to the engine according to its name.
            rt::Tensor const& tensor = tensorRef.get();
            setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
                tensor.getName().c_str(), const_cast<void*>(tensor.rawPointer()));
            setEngineIOStatus
                &= mPrefillExecutionContext->setInputShape(tensor.getName().c_str(), tensor.getShape().getTRTDims());
        }
    }

    if (mConfig.enableEagleSpecDecode)
    {
        setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.value().get().rawPointer());
        // Mask input and optional token pos-ids are not used, set to dummy data.
        setEngineIOStatus
            &= mPrefillExecutionContext->setTensorAddress(binding_names::kAttentionMask, mDummyTensor.rawPointer());
        setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
            binding_names::kAttentionMask, Coords{activeBatchSize, 1, 1}.getTRTDims());
        setEngineIOStatus
            &= mPrefillExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mDummyTensor.rawPointer());
        setEngineIOStatus &= mPrefillExecutionContext->setInputShape(
            binding_names::kAttentionPosId, Coords{activeBatchSize, 1}.getTRTDims());
    }

    // Engine output tensors.
    setEngineIOStatus &= mPrefillExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    // Bind the KVCache IO to the engine.
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executePrefill(): Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mPrefillExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("executePrefill(): Failed on TensorRT prefill stage enqueueV3() call.");
        return false;
    }
    // Prefill operation has completed, commit the new contents with KVCache.
    mKVCache.commitSequenceLength(mSequenceContextLengths, stream);

    LOG_DEBUG("executePrefill(): Prefill stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool LLMEngineRunner::vanillaDecodingStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& outputLogits)
{
    int32_t activeBatchSize = inputIds.getShape()[0];
    bool const checkInputsGPUTensor
        = inputIds.getDeviceType() == rt::DeviceType::kGPU && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid device type of the input tensors. inputIds and outputLogits should reside on "
            "GPU.");
        return false;
    }
    bool const isBatchValid = activeBatchSize == mKVCache.getActiveBatchSize();
    if (!isBatchValid)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid batchSize of the input tensors. batchSize shall be equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }
    bool checkInputShapeValid = inputIds.getShape().getNumDims() == 2 && inputIds.getShape()[1] == 1
        && outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.outputVocabSize;
    if (!checkInputShapeValid)
    {
        LOG_ERROR(
            "executeGeneration(): Invalid shape of the input tensors. The input tensor should have shape "
            "[activeBatchSize, 1] and the output tensor should have shape [activeBatchSize, outputVocabSize].");
        return false;
    }

    return true;
}

bool LLMEngineRunner::executeVanillaDecodingStep(
    rt::Tensor const& inputIds, rt::Tensor& outputLogits, cudaStream_t stream)
{
    bool const validateInputStatus = this->vanillaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("executeGeneration(): Generation request not performed due to invalid input tensors.");
        return false;
    }

    int32_t activeBatchSize = inputIds.getShape()[0];
    // For vanllia decode stage, the selected token indices are always 0.
    // Also setup the sequence length of each sequence for this run based on committed KVCache length.
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), mKVCache.getKVCacheLengths().rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    // Increment the sequence length due to the implementation constraint of AttentionPlugin.
    constexpr int32_t kDECODE_INCREMENT{1};
    kernel::incrementLengthTensor(mSequenceContextLengths, kDECODE_INCREMENT, stream);

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    size_t const graphHash = hashDecodingInput(inputIds, outputLogits, mActiveLoraWeightsName);
    if (mCudaGraphs.find(graphHash) != mCudaGraphs.end())
    {
        LOG_DEBUG("executeVanillaDecodingStep(): Use pre-captured CUDA graph for vanilla decoding step.");
        cudaGraphExec_t graphExec = mCudaGraphs[graphHash].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setEngineIOStatus{true};
        // Engine input tensors.
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kInputIds, const_cast<void*>(inputIds.rawPointer()));
        setEngineIOStatus
            &= mGenerationExecutionContext->setInputShape(binding_names::kInputIds, inputIds.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());

        // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
        if (mConfig.ropeConfig.type == RopeType::kMRope)
        {
            mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
        }

        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

        // Update KV cache shapes to match activeBatchSize
        setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

        // Engine output tensors.
        setEngineIOStatus
            &= mGenerationExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("executeVanillaDecodingStep(): Failed to set engine input tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("executeVanillaDecodingStep(): Failed on TensorRT decode stage enqueueV3() call.");
            return false;
        }
    }

    // Completed decoding step, commit the KVCache length of this run.
    constexpr int32_t kVANILLA_DECODE_INCREMENT{1};
    mKVCache.commitSequenceLength(kVANILLA_DECODE_INCREMENT, stream);
    LOG_DEBUG("executeVanillaDecodingStep(): Decoding stage execution completed for request with batch size %d.",
        activeBatchSize);
    return true;
}

bool LLMEngineRunner::eagleBaseTreeDecodingStepInputValidation(rt::Tensor const& baseTreeDecodingInputIds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = baseTreeDecodingInputIds.getDeviceType() == rt::DeviceType::kGPU
        && baseTreeDecodingMask.getDeviceType() == rt::DeviceType::kGPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid device type of I/O tensors. All inputs and outputs "
            "shall "
            "reside on GPU.");
        return false;
    }
    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = baseTreeDecodingInputIds.getDataType() == DataType::kINT32
        && baseTreeDecodingMask.getDataType() == DataType::kINT8 && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Input token ids shall be INT32, hidden states I/O shall be "
            "FLOAT16, "
            "base tree decoding mask shall be INT8, output logits shall be FLOAT32.");
        return false;
    }
    // Validate shapes of the input tensors.
    bool const isBatchValid = baseTreeDecodingInputIds.getShape()[0] == mKVCache.getActiveBatchSize()
        && baseTreeDecodingMask.getShape()[0] == mKVCache.getActiveBatchSize();
    if (!isBatchValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid batchSize of the input tensors. batchSize shall be "
            "equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }

    int64_t const baseTreeDecodingSize = baseTreeDecodingInputIds.getShape()[1];
    bool const isBaseTreeDecodingSizeValid = baseTreeDecodingMask.getShape()[1] == baseTreeDecodingSize
        && baseTreeDecodingMask.getShape()[2] == baseTreeDecodingSize;
    if (!isBaseTreeDecodingSizeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid base tree decoding size of the input tensors. "
            "Base tree decoding size %d, current base tree decoding mask shape: %s",
            baseTreeDecodingSize, baseTreeDecodingMask.getShape().formatString().c_str());
        return false;
    }

    bool const isOutputShapeValid = outputLogits.getShape()[0] == outputHiddenStates.getShape()[0]
        && outputLogits.getShape()[1] == mConfig.outputVocabSize
        && outputHiddenStates.getShape()[1] == mConfig.outputHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid shape of the output tensors. Logits shape shall be "
            "[select-token-size, %d], hidden states shape shall be [select-token-size, %d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            mConfig.outputVocabSize, mConfig.outputHiddenDim, outputLogits.getShape().formatString().c_str(),
            outputHiddenStates.getShape().formatString().c_str());
        return false;
    }

    return true;
}

bool LLMEngineRunner::executeEagleBaseTreeDecodingStep(rt::Tensor const& baseTreeDecodingInputIds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
    cudaStream_t stream)
{
    bool const validateInputStatus = this->eagleBaseTreeDecodingStepInputValidation(
        baseTreeDecodingInputIds, baseTreeDecodingMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR(
            "executeEagleBaseTreeDecodingStep(): Eagle base tree decoding request not performed due to invalid input "
            "tensors.");
        return false;
    }

    int32_t const activeBatchSize = baseTreeDecodingInputIds.getShape()[0];
    int32_t const baseTreeDecodingSize = static_cast<int32_t>(baseTreeDecodingInputIds.getShape()[1]);
    int32_t const packedBaseTreeDecodingMaskLen = static_cast<int32_t>(divUp(baseTreeDecodingSize, 32));

    // Prepare extra input for engine execution. Assemble packed base tree decoding mask, position indices, select token
    // indices, sequence context lengths.
    mSelectTokenIndices.reshape({activeBatchSize, baseTreeDecodingSize}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mEagleBasePositionIds.reshape({activeBatchSize, baseTreeDecodingSize});
    mEagleBasePackedMask.reshape({activeBatchSize, baseTreeDecodingSize, packedBaseTreeDecodingMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndices = mKVCache.getKVCacheLengths();
    kernel::prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMask, sequenceStartIndices, mEagleBasePackedMask,
        mEagleBasePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    size_t const graphHash = hashBaseTreeDecodingInput(baseTreeDecodingInputIds, outputLogits, outputHiddenStates);
    if (mBaseTreeDecodingCudaGraphs.find(graphHash) != mBaseTreeDecodingCudaGraphs.end())
    {
        LOG_DEBUG("executeEagleBaseTreeDecodingStep(): Use pre-captured CUDA graph for eagle base tree decoding step.");
        cudaGraphExec_t graphExec = mBaseTreeDecodingCudaGraphs[graphHash].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
        // initialization.
        bool setEngineIOStatus{true};
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kInputIds, const_cast<void*>(baseTreeDecodingInputIds.rawPointer()));
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kInputIds, baseTreeDecodingInputIds.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());

        // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
        if (mConfig.ropeConfig.type == RopeType::kMRope)
        {
            mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
        }

        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

        // Update KV cache shapes to match activeBatchSize
        setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kAttentionMask, mEagleBasePackedMask.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kAttentionMask, mEagleBasePackedMask.getShape().getTRTDims());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kAttentionPosId, mEagleBasePositionIds.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
            binding_names::kAttentionPosId, mEagleBasePositionIds.getShape().getTRTDims());
        // Bind the output tensor into the engine.
        setEngineIOStatus
            &= mGenerationExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
        setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("executeEagleBaseTreeDecodingStep(): Failed to bind engine input and output tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR(
                "executeEagleBaseTreeDecodingStep(): Failed on TensorRT eagle base tree decoding stage enqueueV3() "
                "call.");
            return false;
        }
    }

    // Note in the base tree decoding step we explicitly don't commit the KVCache since we process the "whole tree" in
    // these steps.
    LOG_DEBUG(
        "executeEagleBaseTreeDecodingStep(): Eagle base tree decoding stage execution completed for request with batch "
        "size %d.",
        activeBatchSize);
    return true;
}

bool LLMEngineRunner::captureVanillaDecodingCudaGraph(
    rt::Tensor const& inputIds, rt::Tensor& outputLogits, std::string const& loraWeightsPath, cudaStream_t stream)
{
    size_t const hashValue = hashDecodingInput(inputIds, outputLogits, loraWeightsPath);
    if (mCudaGraphs.find(hashValue) != mCudaGraphs.end())
    {
        LOG_INFO(
            "captureVanillaDecodingCudaGraph(): CUDA graph already captured for the input tensors with LoRA weights "
            "%s.",
            loraWeightsPath.c_str());
        return true;
    }

    if (isLoraWeightsSupported() && !this->switchLoraWeights(loraWeightsPath, stream))
    {
        LOG_ERROR(
            "captureVanillaDecodingCudaGraph(): Failed to switch LoRA weights to '%s', unable to capture CUDA graph.",
            loraWeightsPath.c_str());
        return false;
    }

    // To avoid CUDA graph error from TensorRT engine, we need to enqueueV3() once prior to graph capture.
    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int64_t const activeBatchSize = inputIds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(
        reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);

    mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    // Validate the condition here after the simulate prefill step.
    bool const validateInputStatus = this->vanillaDecodingStepInputValidation(inputIds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("captureVanillaDecodingCudaGraph(): Generation request is invalid, unable to capture CUDA graph.");
        return false;
    }

    // Set shape of mSelectTokenIndices and set value to all zero.
    // Set sequence context length input for decoding step.
    mSelectTokenIndices.reshape({activeBatchSize, 1});
    mSequenceContextLengths.reshape({activeBatchSize});
    // Need to reshape the mPosEncCosSinCache for MROPE.
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }
    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), mKVCache.getKVCacheLengths().rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));

    // Set engine I/O using the same logic as executeVanillaDecodingStep().
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kInputIds, const_cast<void*>(inputIds.rawPointer()));
    setEngineIOStatus
        &= mGenerationExecutionContext->setInputShape(binding_names::kInputIds, inputIds.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());

    // Engine output tensors.
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());

    // Bind the KVCache since we haven't executed the real prefill step.
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);
    if (!setEngineIOStatus)
    {
        LOG_ERROR(
            "captureVanillaDecodingCudaGraph(): Failed to set engine input tensors, unable to capture CUDA graph.");
        return false;
    }

    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("captureVanillaDecodingCudaGraph(): Failed on TensorRT engine enqueueV3() call.");
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    mCudaGraphs[hashValue] = std::make_pair(graph, graphExec);

    if (!executeStatus)
    {
        LOG_WARNING(
            "captureVanillaDecodingCudaGraph(): Failed on TensorRT engine enqueueV3() call during CUDA graph capture.");
        return false;
    }
    else
    {
        LOG_DEBUG(
            "captureVanillaDecodingCudaGraph(): CUDA graph captured successfully for input shape %s with LoRA weights "
            "'%s' (Empty string if no LoRA weights).",
            inputIds.getShape().formatString().c_str(), loraWeightsPath.c_str());
    }

    return true;
}

bool LLMEngineRunner::captureEagleBaseTreeDecodingCudaGraph(rt::Tensor const& baseTreeDecodingInputIds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
    cudaStream_t stream)
{
    size_t const hashValue = hashBaseTreeDecodingInput(baseTreeDecodingInputIds, outputLogits, outputHiddenStates);
    if (mBaseTreeDecodingCudaGraphs.find(hashValue) != mBaseTreeDecodingCudaGraphs.end())
    {
        LOG_INFO("captureEagleBaseTreeDecodingCudaGraph(): CUDA graph already captured for the input tensors.");
        return true;
    }

    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int32_t const activeBatchSize = baseTreeDecodingInputIds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(
        reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);

    mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    bool const validateInputStatus = this->eagleBaseTreeDecodingStepInputValidation(
        baseTreeDecodingInputIds, baseTreeDecodingMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR(
            "captureEagleBaseTreeDecodingCudaGraph(): Eagle base tree decoding request not performed due to invalid "
            "input "
            "tensors.");
        return false;
    }

    // Prepare extra input for engine execution. Assemble packed base tree decoding mask, position indices, select token
    // indices, sequence context lengths.
    int32_t const baseTreeDecodingSize = static_cast<int32_t>(baseTreeDecodingInputIds.getShape()[1]);
    int32_t const packedBaseTreeDecodingMaskLen = static_cast<int32_t>(divUp(baseTreeDecodingSize, 32));
    mSelectTokenIndices.reshape({activeBatchSize, baseTreeDecodingSize}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mEagleBasePositionIds.reshape({activeBatchSize, baseTreeDecodingSize});
    mEagleBasePackedMask.reshape({activeBatchSize, baseTreeDecodingSize, packedBaseTreeDecodingMaskLen});

    rt::Tensor const& sequenceStartIndices = mKVCache.getKVCacheLengths();

    kernel::prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMask, sequenceStartIndices, mEagleBasePackedMask,
        mEagleBasePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};

    // Update KV cache shapes to match activeBatchSize for CUDA graph capture
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());

    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kInputIds, const_cast<void*>(baseTreeDecodingInputIds.rawPointer()));
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kInputIds, baseTreeDecodingInputIds.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());

    // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }

    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kAttentionMask, mEagleBasePackedMask.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kAttentionMask, mEagleBasePackedMask.getShape().getTRTDims());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kAttentionPosId, mEagleBasePositionIds.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kAttentionPosId, mEagleBasePositionIds.getShape().getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus
        &= mGenerationExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    if (!setEngineIOStatus)
    {
        LOG_ERROR("captureEagleBaseTreeDecodingCudaGraph(): Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution. This will trigger the shape machine of TensorRT engine to avoid cudaGraph capture.
    // error.
    bool executeStatus{true};
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);

    if (!executeStatus)
    {
        LOG_ERROR(
            "captureEagleBaseTreeDecodingCudaGraph(): Failed on TensorRT eagle base tree decoding stage enqueueV3() "
            "call.");
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    mBaseTreeDecodingCudaGraphs[hashValue] = std::make_pair(graph, graphExec);

    if (!executeStatus)
    {
        LOG_WARNING(
            "captureEagleBaseTreeDecodingCudaGraph(): Failed on TensorRT engine enqueueV3() call during CUDA graph "
            "capture.");
        return false;
    }
    else
    {
        LOG_DEBUG("captureEagleBaseTreeDecodingCudaGraph(): CUDA graph captured successfully for input shape %s.",
            baseTreeDecodingInputIds.getShape().formatString().c_str());
    }

    return true;
}

bool LLMEngineRunner::resetLoraWeights(cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        return true;
    }
    mActiveLoraWeightsName = "";
    bool resetStatus{true};
    for (auto const& loraWeightsTensorName : getLoraWeightsTensorNames())
    {
        nvinfer1::Dims emptyLoraShape
            = mEngine->getProfileShape(loraWeightsTensorName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        // Use dummy tensor as zero tensor for LoRA weights
        resetStatus
            &= mPrefillExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), mDummyTensor.rawPointer());
        resetStatus
            &= mGenerationExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), mDummyTensor.rawPointer());

        // Set shape to kEMPTY_LORA_RANK and assign zero value tensor to disable LoRA
        if (loraWeightsTensorName.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            // LoRA A has shape [k, rank], set rank to kEMPTY_LORA_RANK
            emptyLoraShape.d[1] = kEMPTY_LORA_RANK;
        }
        else if (loraWeightsTensorName.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            // LoRA B has shape [rank, n], set rank to kEMPTY_LORA_RANK
            emptyLoraShape.d[0] = kEMPTY_LORA_RANK;
        }
        resetStatus &= mPrefillExecutionContext->setInputShape(loraWeightsTensorName.c_str(), emptyLoraShape);
        resetStatus &= mGenerationExecutionContext->setInputShape(loraWeightsTensorName.c_str(), emptyLoraShape);
        if (!resetStatus)
        {
            LOG_ERROR("Failed to reset LoRA weights: %s", loraWeightsTensorName.c_str());
            return false;
        }
    }
    return resetStatus;
}

bool LLMEngineRunner::addLoraWeights(
    std::string const& loraWeightsName, std::string const& loraWeightsPath, cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        LOG_ERROR("addLoraWeights(): Engine does not support LoRA weights.");
    }

    if (mLoraWeights.find(loraWeightsName) != mLoraWeights.end())
    {
        LOG_ERROR("addLoraWeights(): LoRA weights %s already added", loraWeightsName.c_str());
        return false;
    }

    // Load tensors using the new unified interface
    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(loraWeightsPath, tensors, stream))
    {
        LOG_ERROR("addLoraWeights(): Failed to load LoRA weights %s from: %s", loraWeightsName.c_str(),
            loraWeightsPath.c_str());
        return false;
    }

    // Validate the LoRA weights do not exceed the max LoRA rank
    for (auto const& tensor : tensors)
    {
        if (tensor.getName().find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            if (tensor.getShape()[1] > mConfig.maxSupportedLoraRank)
            {
                LOG_ERROR("addLoraWeights(): LoRA A (%s) tensor's rank (%d) exceeds the max LoRA rank (%d)",
                    tensor.getName().c_str(), tensor.getShape()[1], mConfig.maxSupportedLoraRank);
                return false;
            }
        }
        else if (tensor.getName().find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            if (tensor.getShape()[0] > mConfig.maxSupportedLoraRank)
            {
                LOG_ERROR("addLoraWeights(): LoRA B (%s) tensor's rank (%d) exceeds the max LoRA rank (%d)",
                    tensor.getName().c_str(), tensor.getShape()[0], mConfig.maxSupportedLoraRank);
                return false;
            }
        }
    }

    // Store the tensors in our map
    mLoraWeights[loraWeightsName] = std::move(tensors);
    LOG_INFO("addLoraWeights(): Added LoRA weights %s from: %s", loraWeightsName.c_str(), loraWeightsPath.c_str());
    return true;
}

std::vector<std::string> LLMEngineRunner::getLoraWeightsTensorNames() const
{
    std::vector<std::string> loraWeightsTensorNames;
    // Get the number of bindings in the engine
    int32_t numBindings = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numBindings; ++i)
    {
        char const* bindingName = mEngine->getIOTensorName(i);
        std::string bindingNameStr(bindingName);
        if (binding_names::isLoraBinding(bindingNameStr))
        {
            loraWeightsTensorNames.push_back(bindingNameStr);
        }
    }
    return loraWeightsTensorNames;
}

bool LLMEngineRunner::switchLoraWeights(std::string const& loraWeightsName, cudaStream_t stream)
{
    if (!isLoraWeightsSupported())
    {
        LOG_ERROR("switchLoraWeights(): API call is invalid. LLM engine does not support LoRA weights.");
        return false;
    }
    if (loraWeightsName.empty())
    {
        this->resetLoraWeights(stream);
        LOG_DEBUG("switchLoraWeights(): Switched to no LoRA weights.");
        return true;
    }

    // Check if the requested LoRA exists
    auto it = mLoraWeights.find(loraWeightsName);
    if (it == mLoraWeights.end())
    {
        LOG_ERROR("switchLoraWeights(): LoRA weights with name '%s' not found", loraWeightsName.c_str());
        return false;
    }

    auto& loraTensors = it->second;

    // Iterate through all LoRA weights bindings
    for (auto const& loraWeightsTensorName : this->getLoraWeightsTensorNames())
    {
        // Try to find the tensor in the LoRA weights
        auto loraTensorIt = std::find_if(loraTensors.begin(), loraTensors.end(),
            [loraWeightsTensorName](rt::Tensor const& tensor) { return tensor.getName() == loraWeightsTensorName; });

        bool setLoraWeightsStatus{true};

        if (loraTensorIt != loraTensors.end())
        {
            // Found matching tensor, use its data
            setLoraWeightsStatus
                &= mPrefillExecutionContext->setInputShape(loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus &= mGenerationExecutionContext->setInputShape(
                loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus &= mPrefillExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
            setLoraWeightsStatus &= mGenerationExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
            LOG_DEBUG("switchLoraWeights(): LoRA weights tensor with name '%s' found. Set shape to %s.",
                loraWeightsTensorName.c_str(), loraTensorIt->getShape().formatString().c_str());
        }
        else
        {
            // Tensor not found in this LoRA adapter, use dummy tensor as zero tensor with shape kEMPTY_LORA_RANK
            nvinfer1::Dims shape
                = mEngine->getProfileShape(loraWeightsTensorName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            if (loraWeightsTensorName.find(binding_names::kLoraAPrefix) != std::string::npos)
            {
                // LoRA A has shape [k, rank], set rank to kEMPTY_LORA_RANK
                shape.d[1] = kEMPTY_LORA_RANK;
            }
            else if (loraWeightsTensorName.find(binding_names::kLoraBPrefix) != std::string::npos)
            {
                // LoRA B has shape [rank, n], set rank to kEMPTY_LORA_RANK
                shape.d[0] = kEMPTY_LORA_RANK;
            }
            setLoraWeightsStatus &= mPrefillExecutionContext->setInputShape(loraWeightsTensorName.c_str(), shape);
            setLoraWeightsStatus &= mGenerationExecutionContext->setInputShape(loraWeightsTensorName.c_str(), shape);
            setLoraWeightsStatus
                &= mPrefillExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), mDummyTensor.rawPointer());
            setLoraWeightsStatus &= mGenerationExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), mDummyTensor.rawPointer());
            LOG_DEBUG(
                "LoRA weights tensor with name '%s' not found. Set shape to rank %d with zero "
                "tensor.",
                loraWeightsTensorName.c_str(), kEMPTY_LORA_RANK);
        }
        if (!setLoraWeightsStatus)
        {
            LOG_ERROR("Failed to set LoRA weights: %s", loraWeightsTensorName.c_str());
            return false;
        }
    }
    // Set the active LoRA weights name
    mActiveLoraWeightsName = loraWeightsName;
    LOG_DEBUG("switchLoraWeights(): Switched to LoRA weights with name '%s'.", loraWeightsName.c_str());
    return true;
}

std::string LLMEngineRunner::getActiveLoraWeightsName() const
{
    return mActiveLoraWeightsName;
}

std::vector<std::string> LLMEngineRunner::getAvailableLoraWeights() const
{
    std::vector<std::string> loraWeightsNames;
    for (auto const& [loraWeightsName, _] : mLoraWeights)
    {
        loraWeightsNames.push_back(loraWeightsName);
    }
    return loraWeightsNames;
}

bool LLMEngineRunner::isLoraWeightsSupported() const
{
    return mConfig.maxSupportedLoraRank > 0;
}

int32_t LLMEngineRunner::getMaxLoraWeightsDimension() const
{
    if (!isLoraWeightsSupported())
    {
        return 0;
    }

    int32_t maxDim = 0;

    // Query engine profile shapes for all LoRA weight tensors
    for (auto const& loraWeightsTensorName : getLoraWeightsTensorNames())
    {
        nvinfer1::Dims maxShape
            = mEngine->getProfileShape(loraWeightsTensorName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        if (loraWeightsTensorName.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            // LoRA A has shape [k, rank], we want max k
            maxDim = std::max(maxDim, static_cast<int32_t>(maxShape.d[0]));
        }
        else if (loraWeightsTensorName.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            // LoRA B has shape [rank, n], we want max n
            maxDim = std::max(maxDim, static_cast<int32_t>(maxShape.d[1]));
        }
    }

    return maxDim;
}

} // namespace rt
} // namespace trt_edgellm
