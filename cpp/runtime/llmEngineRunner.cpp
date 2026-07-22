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

#include "runtime/llmEngineRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "profiling/layerProfiler.h"
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
       << "  enableEagleSpecDecode: " << config.enableEagleSpecDecode
       << "  numDecoderLayers: " << config.numDecoderLayers << "  numKVHeads: " << config.numKVHeads
       << "  headDim: " << config.headDim << "  rotaryDim: " << config.rotaryDim
       << "  hiddenSize: " << config.hiddenSize << "  maxSupportedBatchSize: " << config.maxSupportedBatchSize
       << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  maxKVCacheCapacity: " << config.maxKVCacheCapacity
       << "  maxSupportedLoraRank: " << config.maxSupportedLoraRank
       << "  numDeepstackFeatures: " << config.numDeepstackFeatures;
    if (config.enableEagleSpecDecode)
    {
        ss << "  outputHiddenDim (For Eagle SpecDecode): " << config.outputHiddenDim;
        ss << "  maxVerifyTreeSize (For Eagle SpecDecode): " << config.maxVerifyTreeSize;
    }
    if (config.enableContextEmb)
    {
        ss << "  contextEmbDim (For context_emb): " << config.contextEmbDim;
    }
    if (config.enableLmHiddenStates)
    {
        ss << "  enableLmHiddenStates: true";
    }
    return ss.str();
}

// Compute a unique key value that can distinguish the various decoding steps.
// Extend this function when we need to capture more information.
trt_edgellm::rt::LLMEngineRunner::DecodingGraphKey decodingKey(
    rt::Tensor const& inputsEmbeds, rt::Tensor const& outputLogits, std::string const& loraWeightsName) noexcept
{
    // For vanilla decoding step, the shape can be distingusihed by active batch size.
    // Also capture the pointer address to ensure we are read/write correct locations.
    int64_t const activeBatchSize = inputsEmbeds.getShape()[0];
    uintptr_t const inputsEmbedsAddr = reinterpret_cast<uintptr_t>(inputsEmbeds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());
    return std::make_tuple(activeBatchSize, inputsEmbedsAddr, outputLogitsAddr, loraWeightsName);
}

trt_edgellm::rt::LLMEngineRunner::BaseGraphKey baseKey(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates, std::string const& loraWeightsName) noexcept
{
    int64_t const activeBatchSize = baseTreeDecodingInputsEmbeds.getShape()[0];
    uintptr_t const inputsEmbedsAddr = reinterpret_cast<uintptr_t>(baseTreeDecodingInputsEmbeds.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());
    uintptr_t const outputHiddenStatesAddr = reinterpret_cast<uintptr_t>(outputHiddenStates.rawPointer());
    return std::make_tuple(
        activeBatchSize, inputsEmbedsAddr, outputLogitsAddr, outputHiddenStatesAddr, loraWeightsName);
}

} // namespace

namespace trt_edgellm
{
namespace rt
{

//! Current implementation limits to two optimization profiles per LLM engine.
static constexpr int32_t kPREFILL_PROFILE_INDEX{0};
static constexpr int32_t kGENERATION_PROFILE_INDEX{1};

bool engineHasIOTensor(nvinfer1::ICudaEngine const* engine, std::string const& name)
{
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const* tensorName = engine->getIOTensorName(i);
        if (tensorName != nullptr && name == tensorName)
        {
            return true;
        }
    }
    return false;
}

//! Read output vocab size from logits binding; dual-profile exports may report invalid profile dims.
static int32_t getOutputVocabSizeFromEngine(nvinfer1::ICudaEngine const* engine)
{
    if (!engineHasIOTensor(engine, binding_names::kLogits))
    {
        return 0;
    }

    auto extractVocabSize = [](nvinfer1::Dims const& dims) -> int32_t {
        if (dims.nbDims == 3 && dims.d[2] > 0)
        {
            return dims.d[2];
        }
        if (dims.nbDims == 2 && dims.d[1] > 0)
        {
            return dims.d[1];
        }
        return 0;
    };

    for (int32_t profileIndex : {kPREFILL_PROFILE_INDEX, kGENERATION_PROFILE_INDEX})
    {
        int32_t const vocabSize = extractVocabSize(
            engine->getProfileShape(binding_names::kLogits, profileIndex, OptProfileSelector::kMAX));
        if (vocabSize > 0)
        {
            return vocabSize;
        }
    }
    return extractVocabSize(engine->getTensorShape(binding_names::kLogits));
}

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
        LOG_ERROR("Failed to use MMap to read engine from file path: %s", enginePath.string().c_str());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    // Use single executionContext for both prefill and generation.
    // Context memory is user-managed to enable sharing with other engines.
    // The caller must provide shared context memory via setContextMemory() before execution.
    mTRTExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));

    if (trt_edgellm::layerProfiler::LayerProfiler::getInstance().isEnabled())
    {
        mTRTExecutionContext->setProfiler(&trt_edgellm::layerProfiler::LayerProfiler::getInstance());
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
            = initializeLongRopeCosSinCache(shortCosSinCache, longCosSinCache, ropeConfig, stream);
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

        // Initialize MRoPE cache for all batch slots using text-only sequential positions.
        kernel::initializeTextOnlyMRopeCosSin(mPosEncCosSinCache.dataPointer<float>(), ropeConfig.rotaryTheta,
            mConfig.rotaryDim, mConfig.maxKVCacheCapacity, mConfig.maxSupportedBatchSize, stream);
        break;
    }
    case RopeType::kNoRope:
    {
        LOG_DEBUG("No RoPE: initializing identity CosSinCache (cos=1, sin=0).");
        this->mPosEncCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMEngineRunner::mPosEncCosSinCache");
        bool const initStatus = initializeNopeCosSinCache(mPosEncCosSinCache, stream);
        if (!initStatus)
        {
            LOG_ERROR("Failed to initialize identity CosSinCache.");
            throw std::runtime_error("Failed to initialize identity CosSinCache.");
        }
        break;
    }
    default:
    {
        LOG_DEBUG("Initialize persistent Rope CosSinCache.");
        this->mPosEncCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMEngineRunner::mPosEncCosSinCache");
        bool const initRopeStatus = initializeRopeCosSinCache(mPosEncCosSinCache, ropeConfig, stream);
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
        &= mTRTExecutionContext->setTensorAddress(binding_names::kRopeCosSin, mPosEncCosSinCache.rawPointer());
    if (!setRopeCosSinCacheStatus)
    {
        LOG_ERROR("Failed to set rope cos sin cache to the engine");
        throw std::runtime_error("Failed to set rope cos sin cache to the engine");
    }

    if (!validateKVCacheType())
    {
        LOG_ERROR("Failed to validate KV cache type");
        throw std::runtime_error("Failed to validate KV cache type");
    }

    // Detect KV cache storage dtype from engine bindings.
    nvinfer1::DataType kvCacheType = getKVCacheType();

    int32_t const kvCacheLayers
        = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
    DataType const recurrentStateType = (mConfig.numLinearAttnLayers > 0) ? getRecurrentStateType() : DataType::kHALF;
    DataType const convStateType = (mConfig.numLinearAttnLayers > 0) ? getConvStateType() : DataType::kHALF;
    this->mKVCache = rt::LinearKVCache(
        rt::LinearKVCache::CacheConfig{kvCacheLayers, mConfig.maxSupportedBatchSize, mConfig.maxKVCacheCapacity,
            mConfig.numKVHeads, mConfig.headDim, kvCacheType,
            // Recurrent state config (zero for pure-attention models)
            mConfig.numLinearAttnLayers, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim,
            mConfig.recurrentStateSize, recurrentStateType, mConfig.convDim, mConfig.convKernel, convStateType},
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
        // Allocate position IDs to support both prefill and tree decoding
        int32_t const maxSeqLen = std::max(mConfig.maxSupportedInputLength, mConfig.maxVerifyTreeSize);
        this->mEagleBasePositionIds = rt::Tensor({mConfig.maxSupportedBatchSize, maxSeqLen}, rt::DeviceType::kGPU,
            DataType::kINT32, "LLMEngineRunner::mEagleBasePositionIds");
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
    // 1. Attention mask: {maxSupportedBatchSize, 1, 1}
    // 2. Attention position IDs: {maxSupportedBatchSize, 1}
    // 3. LoRA weights: max dimension across all adapters
    // 4. KV cache start index: {maxSupportedBatchSize}
    // 5. Deepstack embeds for generation: {maxSupportedBatchSize, maxVerifyTreeSize or 1, hiddenSize}
    std::vector<int64_t> dummyInputSizes = {
        static_cast<int64_t>(
            mConfig.maxSupportedBatchSize), // attention mask/attention position IDs/KV cache start index
        static_cast<int64_t>(getMaxLoraWeightsDimension() * kEMPTY_LORA_RANK), // LoRA weights
    };

    // Add deepstack_embeds size for generation profile
    // Use maxVerifyTreeSize for eagle or 1 for vanilla decoding
    if (mConfig.numDeepstackFeatures > 0)
    {
        int64_t const deepstackSeqLen = mConfig.enableEagleSpecDecode ? mConfig.maxVerifyTreeSize : 1;
        int64_t const deepstackSize
            = static_cast<int64_t>(mConfig.maxSupportedBatchSize) * deepstackSeqLen * mConfig.hiddenSize;
        dummyInputSizes.push_back(deepstackSize);
    }

    int64_t maxDummyElements = *std::max_element(dummyInputSizes.begin(), dummyInputSizes.end());
    mDummyInputTensor = rt::Tensor(
        {maxDummyElements}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "LLMEngineRunner::mDummyInputTensor");
    // Initialize dummy tensor memory to zero
    CUDA_CHECK(cudaMemsetAsync(mDummyInputTensor.rawPointer(), 0, mDummyInputTensor.getMemoryCapacity(), stream));

    // Allocate dummy output tensor for hidden_states for Eagle speculative decoding.
    // TRT engine under this mode will produce output hidden states. we reserve this buffer to hold the data when
    // conduct vanilla decoding. This will make runtime design cleaner.
    if (mConfig.enableEagleSpecDecode)
    {
        int64_t const dummyOutputSize = static_cast<int64_t>(mConfig.maxSupportedBatchSize) * mConfig.outputHiddenDim;
        mDummyOutputTensor = rt::Tensor(
            {dummyOutputSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "LLMEngineRunner::mDummyOutputTensor");
    }
    else if (mConfig.enablePrefixKVOutputs)
    {
        int64_t dummyOutputSize{1};
        for (int64_t const dim : mConfig.prefixKVOutputShape)
        {
            check::check(dim > 0, "Invalid prefix_k output dimension in config");
            dummyOutputSize *= dim;
        }
        mDummyOutputTensor = rt::Tensor({dummyOutputSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
            "LLMEngineRunner::mDummyOutputTensor");
    }

    if (mConfig.enableContextEmb || mConfig.enableLmHiddenStates)
    {
        int32_t const auxDim = mConfig.enableContextEmb ? mConfig.contextEmbDim : mConfig.hiddenSize;
        int64_t const dummyContextEmbSize = static_cast<int64_t>(mConfig.maxSupportedBatchSize)
            * mConfig.maxSupportedInputLength * auxDim;
        mDummyContextEmbTensor = rt::Tensor({dummyContextEmbSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
            "LLMEngineRunner::mDummyContextEmbTensor");
    }

    // Initialize kKVCacheStartIndex to dummy tensor for both profiles to avoid "address not set" error
    // when switching optimization profiles. The actual address will be set during runtime execution.
    {
        bool setKVCacheStartIndexStatus{true};
        Dims const kvStartIndexEngineDim = mEngine->getTensorShape(binding_names::kKVCacheStartIndex);
        setKVCacheStartIndexStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mDummyInputTensor.rawPointer());
        setKVCacheStartIndexStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, kvStartIndexEngineDim);
        if (!setKVCacheStartIndexStatus)
        {
            LOG_ERROR("Failed to set kKVCacheStartIndex dummy tensor for initialization");
            throw std::runtime_error("Failed to set kKVCacheStartIndex dummy tensor for initialization");
        }
    }
    // Reset the LoRA weights to zero tensors.
    if (!this->resetLoraWeights())
    {
        LOG_ERROR("Failed to initialize LoRA weights to zero tensors");
        throw std::runtime_error("Failed to initialize LoRA weights to zero tensors");
    }

    // Synchronize the stream to ensure all the operations have completed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int64_t LLMEngineRunner::getRequiredContextMemorySize() const
{
    return mEngine->getDeviceMemorySizeV2();
}

bool LLMEngineRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("Shared context memory (%zu bytes) is smaller than required (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }
    mTRTExecutionContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

nvinfer1::DataType LLMEngineRunner::getKVCacheType() const
{
    if (mConfig.useTrtNativeOps)
    {
        std::string const trtNativeKVBindingName0 = binding_names::formatKCacheName(/*layerIdx=*/0, /*isPast=*/true);
        return mEngine->getTensorDataType(trtNativeKVBindingName0.c_str());
    }
    else
    {
        std::string const pluginKVBindingName0 = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        return mEngine->getTensorDataType(pluginKVBindingName0.c_str());
    }
}

nvinfer1::DataType LLMEngineRunner::getRecurrentStateType() const
{
    std::string const name = binding_names::formatRecurrentStateName(/*recurrentLayerIdx=*/0, /*isPast=*/true);
    return mEngine->getTensorDataType(name.c_str());
}

nvinfer1::DataType LLMEngineRunner::getConvStateType() const
{
    std::string const name = binding_names::formatConvStateName(/*recurrentLayerIdx=*/0, /*isPast=*/true);
    return mEngine->getTensorDataType(name.c_str());
}

bool LLMEngineRunner::validateKVCacheType() const
{
    // Sanity check: ensure KV-cache precision (dtype) is consistent across all layers (and both past/present).
    // We rely on a single dtype when allocating/owning the KV cache buffers.
    if (mConfig.useTrtNativeOps)
    {
        auto kBindingName0 = binding_names::formatKCacheName(/*layerIdx=*/0, /*isPast=*/true);
        DataType const kCacheType0 = mEngine->getTensorDataType(kBindingName0.c_str());
        auto vBindingName0 = binding_names::formatVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        auto const checkKVCacheDType = [&](int32_t layerIdx, bool isPast) {
            std::string const kBindingName = binding_names::formatKCacheName(layerIdx, isPast);
            DataType const kCacheType = mEngine->getTensorDataType(kBindingName.c_str());
            std::string const vBindingName = binding_names::formatVCacheName(layerIdx, isPast);
            DataType const vCacheType = mEngine->getTensorDataType(vBindingName.c_str());
            if (kCacheType != kCacheType0 || vCacheType != kCacheType0)
            {
                LOG_ERROR(
                    "KV cache dtype mismatch detected. Expected all layers to use the same dtype as '%s' (dtype=%d), "
                    "but "
                    "binding '%s' has dtype=%d and '%s' has dtype=%d.",
                    kBindingName0.c_str(), static_cast<int32_t>(kCacheType0), kBindingName.c_str(),
                    static_cast<int32_t>(kCacheType), vBindingName.c_str(), static_cast<int32_t>(vCacheType));
                throw std::runtime_error("KV cache dtype mismatch across layers");
            }
        };
        int32_t const kvLayers
            = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
        for (int32_t layerIdx = 0; layerIdx < kvLayers; ++layerIdx)
        {
            checkKVCacheDType(layerIdx, /*isPast=*/true);
            std::string const presentKBindingName = binding_names::formatKCacheName(layerIdx, /*isPast=*/false);
            if (engineHasIOTensor(mEngine.get(), presentKBindingName))
            {
                checkKVCacheDType(layerIdx, /*isPast=*/false);
            }
        }
    }
    else
    {
        auto kvBindingName0 = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        DataType const kvCacheType = mEngine->getTensorDataType(kvBindingName0.c_str());
        auto const checkKVCacheDType = [&](int32_t layerIdx, bool isPast) {
            std::string const kvBindingName = binding_names::formatKVCacheName(layerIdx, isPast);
            DataType const dt = mEngine->getTensorDataType(kvBindingName.c_str());
            if (dt != kvCacheType)
            {
                LOG_ERROR(
                    "KV cache dtype mismatch detected. Expected all layers to use the same dtype as '%s' (dtype=%d), "
                    "but "
                    "binding '%s' has dtype=%d.",
                    kvBindingName0.c_str(), static_cast<int32_t>(kvCacheType), kvBindingName.c_str(),
                    static_cast<int32_t>(dt));
                throw std::runtime_error("KV cache dtype mismatch across layers");
            }
        };
        int32_t const kvLayers
            = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
        for (int32_t layerIdx = 0; layerIdx < kvLayers; ++layerIdx)
        {
            checkKVCacheDType(layerIdx, /*isPast=*/true);
            std::string const presentBindingName = binding_names::formatKVCacheName(layerIdx, /*isPast=*/false);
            if (engineHasIOTensor(mEngine.get(), presentBindingName))
            {
                checkKVCacheDType(layerIdx, /*isPast=*/false);
            }
        }
    }

    return true;
}

bool LLMEngineRunner::initializeConfigFromJson(Json const& configJson) noexcept
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
            = {"max_batch_size", "max_input_len", "max_kv_cache_capacity", "max_lora_rank", "eagle_base"};

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
        // Read num_deepstack_features if present (Qwen3-VL and Qwen3-Omni models)
        mConfig.numDeepstackFeatures = configJson.value("num_deepstack_features", 0);

        // Read audio and image token IDs for Qwen3-Omni (used by embeddingLookupQwen3Omni kernel)
        mConfig.audioTokenId = configJson.value("audio_token_id", 0);
        mConfig.imageTokenId = configJson.value("image_token_id", 0);

        // Hybrid linear attention configuration (Mamba, GDN, or other linear attention)
        mConfig.numLinearAttnLayers = configJson.value("num_linear_attn_layers", 0);
        mConfig.numAttentionLayers = configJson.value("num_attention_layers", mConfig.numDecoderLayers);
        mConfig.recurrentStateNumHeads = configJson.value("recurrent_state_num_heads", 0);
        mConfig.recurrentStateHeadDim = configJson.value("recurrent_state_head_dim", 0);
        mConfig.recurrentStateSize = configJson.value("recurrent_state_size", 0);
        mConfig.convDim = configJson.value("conv_dim", 0);
        mConfig.convKernel = configJson.value("conv_kernel", 0);

        // Extract builder_config values
        mConfig.maxSupportedBatchSize = builderConfig["max_batch_size"].get<int32_t>();
        mConfig.maxSupportedInputLength = builderConfig["max_input_len"].get<int32_t>();
        mConfig.maxKVCacheCapacity = builderConfig["max_kv_cache_capacity"].get<int32_t>();
        mConfig.maxSupportedLoraRank = builderConfig["max_lora_rank"].get<int32_t>();
        mConfig.enableEagleSpecDecode = builderConfig["eagle_base"].get<bool>();
        mConfig.enableContextEmb = builderConfig.value("context_emb", false);

        // Collect RoPE configuration
        mConfig.ropeConfig = collectRopeConfig(configJson);

        // Initialize useTrtNativeOps from builder_config
        if (builderConfig.contains("trt_native_ops"))
        {
            mConfig.useTrtNativeOps = builderConfig["trt_native_ops"].get<bool>();
        }

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

        if (mConfig.enableContextEmb)
        {
            if (!configJson.contains("context_hidden_size"))
            {
                LOG_ERROR(
                    "initializeConfigFromJson(): Missing required field 'context_hidden_size' in config for "
                    "context_emb model");
                return false;
            }
            mConfig.contextEmbDim = configJson["context_hidden_size"].get<int32_t>();
            if (mConfig.contextEmbDim <= 0)
            {
                LOG_ERROR("initializeConfigFromJson(): Invalid context_hidden_size: %d (must be positive)",
                    mConfig.contextEmbDim);
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

        if (configJson.contains("output_names") && configJson.contains("outputs"))
        {
            auto const& outputNamesJson = configJson.at("output_names");
            auto const& outputsJson = configJson.at("outputs");
            if (outputNamesJson.is_array() && outputsJson.is_array())
            {
                for (size_t outputIdx = 0; outputIdx < outputNamesJson.size() && outputIdx < outputsJson.size();
                    ++outputIdx)
                {
                    if (outputNamesJson.at(outputIdx).get<std::string>() != binding_names::kOutputPrefixK)
                    {
                        continue;
                    }
                    mConfig.enablePrefixKVOutputs = true;
                    mConfig.prefixKVOutputShape.clear();
                    for (auto const& dimJson : outputsJson.at(outputIdx).at("shape"))
                    {
                        mConfig.prefixKVOutputShape.push_back(dimJson.get<int64_t>());
                    }
                    break;
                }
            }
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
    // Plugin path: combined KV cache [batch, 2, num_kv_heads, seq_len, head_dim]
    auto identifyKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 5 && bindingName.find(binding_names::kPastKeyValuesTemplate) != std::string::npos;
    };

    // TRT native: separate K cache [batch, num_kv_heads, seq_len, head_dim]
    auto identifyTRTNativeKCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 4 && bindingName.find(binding_names::kPresentKCacheTemplate) != std::string::npos;
    };

    // TRT native: separate V cache [batch, num_kv_heads, seq_len, head_dim]
    auto identifyTRTNativeVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 4 && bindingName.find(binding_names::kPresentVCacheTemplate) != std::string::npos;
    };

    // If the engine comes with deepstack embeds binding, it means the engine is Qwen3-VL.
    auto identifyDeepstackEmbedsBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 3 && bindingName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos;
    };

    auto validate_eq_engine_with_config
        = [&](int32_t const& configValue, int32_t const& engineValue, std::string const& name) -> bool {
        if (configValue != engineValue)
        {
            LOG_ERROR("%s is not consistent. From engine: %d, from config: %d", name.c_str(), engineValue, configValue);
            return false;
        }
        return true;
    };

    LOG_DEBUG("Prefill profile info: %s", printEngineInfo(mEngine.get(), kPREFILL_PROFILE_INDEX).c_str());
    LOG_DEBUG("Generation profile info: %s", printEngineInfo(mEngine.get(), kGENERATION_PROFILE_INDEX).c_str());

    int32_t nbKVCacheInputs{0};
    int32_t nbTRTNativeKCacheInputs{0};
    int32_t nbTRTNativeVCacheInputs{0};
    int32_t nbDeepstackEmbedsInputs{0};
    int32_t numIOBindings = mEngine->getNbIOTensors();

    // Lambda to validate KV cache dimensions against profile shape
    auto validateKVCacheProfile = [&](Dims const& maxKVCacheShape, std::string const& profileName) -> bool {
        bool status{true};
        status
            &= validate_eq_engine_with_config(mConfig.numKVHeads, maxKVCacheShape.d[2], profileName + ": numKVHeads");
        status &= validate_eq_engine_with_config(
            mConfig.maxKVCacheCapacity, maxKVCacheShape.d[3], profileName + ": maxKVCacheCapacity");
        status &= validate_eq_engine_with_config(mConfig.headDim, maxKVCacheShape.d[4], profileName + ": headDim");
        return status;
    };

    bool isOk{true};
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
            isOk &= validateKVCacheProfile(maxKVCacheShapePrefill, "prefill");
            isOk &= validateKVCacheProfile(maxKVCacheShapeGen, "generation");
            ++nbKVCacheInputs;
        }
        if (identifyDeepstackEmbedsBinding(bindingName, tensorDim))
        {
            isOk &= validate_eq_engine_with_config(mConfig.hiddenSize, tensorDim.d[2], "hiddenSize");
            LOG_DEBUG("validateConfigFromEngine(): Found deepstack embeds binding: %s", bindingName.c_str());
            ++nbDeepstackEmbedsInputs;
        }

        bool const isTRTNativeKCacheBinding = identifyTRTNativeKCacheBinding(bindingName, tensorDim);
        bool const isTRTNativeVCacheBinding = identifyTRTNativeVCacheBinding(bindingName, tensorDim);
        if (isTRTNativeKCacheBinding || isTRTNativeVCacheBinding)
        {
            if (mConfig.numKVHeads != tensorDim.d[1])
            {
                LOG_ERROR("numKVHeads is not consistent (TRT native K or V cache). From engine: %d, from config: %d",
                    tensorDim.d[1], mConfig.numKVHeads);
                return false;
            }
            if (mConfig.maxKVCacheCapacity != tensorDim.d[2])
            {
                LOG_ERROR(
                    "maxSequenceLength is not consistent (TRT native K or V cache). From engine: %d, from config: %d",
                    tensorDim.d[2], mConfig.maxKVCacheCapacity);
                return false;
            }
            if (mConfig.headDim != tensorDim.d[3])
            {
                LOG_ERROR("headDim is not consistent (TRT native K or V cache). From engine: %d, from config: %d",
                    tensorDim.d[3], mConfig.headDim);
                return false;
            }

            if (isTRTNativeKCacheBinding)
            {
                ++nbTRTNativeKCacheInputs;
            }
            if (isTRTNativeVCacheBinding)
            {
                ++nbTRTNativeVCacheInputs;
            }
        }
    }

    // Validate KV cache counts based on attention mode
    int32_t const expectedKVLayers
        = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
    if (mConfig.useTrtNativeOps)
    {
        // TRT native mode: expect separate K and V caches
        if (nbTRTNativeKCacheInputs != expectedKVLayers)
        {
            LOG_ERROR("KV cache layer count mismatch (TRT native K cache). From engine: %d, expected: %d",
                nbTRTNativeKCacheInputs, expectedKVLayers);
            return false;
        }
        if (nbTRTNativeVCacheInputs != expectedKVLayers)
        {
            LOG_ERROR("KV cache layer count mismatch (TRT native V cache). From engine: %d, expected: %d",
                nbTRTNativeVCacheInputs, expectedKVLayers);
            return false;
        }
        if (nbKVCacheInputs > 0)
        {
            LOG_ERROR("Found plugin-style KV cache bindings but config specifies TRT native mode");
            return false;
        }
    }
    else
    {
        // Plugin mode: expect combined KV caches.
        if (nbKVCacheInputs != expectedKVLayers)
        {
            LOG_ERROR(
                "KV cache layer count mismatch. From engine: %d, expected: %d", nbKVCacheInputs, expectedKVLayers);
            return false;
        }
        if (nbTRTNativeKCacheInputs > 0 || nbTRTNativeVCacheInputs > 0)
        {
            LOG_ERROR("Found TRT native-style K/V cache bindings but config specifies plugin attention mode");
            return false;
        }
    }
    isOk &= validate_eq_engine_with_config(
        mConfig.numDeepstackFeatures, nbDeepstackEmbedsInputs, "numDeepstackFeatures");

    Dims const maxInputPrefillShape
        = mEngine->getProfileShape(binding_names::kInputsEmbeds, kPREFILL_PROFILE_INDEX, OptProfileSelector::kMAX);

    // inputs_embeds is 3D: [batch_size, seq_len, hidden_size]
    isOk &= validate_eq_engine_with_config(
        mConfig.maxSupportedInputLength, maxInputPrefillShape.d[1], "maxSupportedInputLength");
    isOk &= validate_eq_engine_with_config(mConfig.hiddenSize, maxInputPrefillShape.d[2], "hiddenSize");

    // Validate and potentially override maxSupportedBatchSize from engine's actual max profile
    int32_t const engineMaxBatchSize = maxInputPrefillShape.d[0];
    isOk &= validate_eq_engine_with_config(mConfig.maxSupportedBatchSize, engineMaxBatchSize, "maxSupportedBatchSize");

    // Obtain vocab size from logits binding; dual-profile exports may report invalid profile dims.
    int32_t const engineOutputVocabSize = getOutputVocabSizeFromEngine(mEngine.get());
    mConfig.hasLogitsOutput = engineHasIOTensor(mEngine.get(), binding_names::kLogits);
    if (engineOutputVocabSize <= 0)
    {
        if (mConfig.hasLogitsOutput)
        {
            Dims const logitsDim = mEngine->getTensorShape(binding_names::kLogits);
            LOG_ERROR("Unexpected logits tensor rank: %d (expected 2 or 3)", logitsDim.nbDims);
            isOk = false;
        }
        else
        {
            mConfig.outputVocabSize = 1;
        }
    }
    else
    {
        isOk &= validate_eq_engine_with_config(mConfig.outputVocabSize, engineOutputVocabSize, "outputVocabSize");
    }

    mConfig.hasLastTokenIdsInput = engineHasIOTensor(mEngine.get(), binding_names::kLastTokenIds);

    // Obtain rotary dim from the engine.
    Dims const ropeCosSinCacheDim = mEngine->getTensorShape(binding_names::kRopeCosSin);
    isOk &= validate_eq_engine_with_config(mConfig.rotaryDim, ropeCosSinCacheDim.d[2], "rotaryDim");

    if (mConfig.enableContextEmb)
    {
        Dims const contextEmbedsDim = mEngine->getTensorShape(binding_names::kOutputContextEmbeds);
        isOk &= validate_eq_engine_with_config(mConfig.contextEmbDim, contextEmbedsDim.d[2], "contextEmbDim");
        mPrefillAuxOutputName = binding_names::kOutputContextEmbeds;
    }
    else if (engineHasIOTensor(mEngine.get(), binding_names::kOutputLmHiddenStates))
    {
        Dims const lmHiddenDim = mEngine->getTensorShape(binding_names::kOutputLmHiddenStates);
        isOk &= validate_eq_engine_with_config(mConfig.hiddenSize, lmHiddenDim.d[2], "lmHiddenDim");
        mConfig.enableLmHiddenStates = true;
        mPrefillAuxOutputName = binding_names::kOutputLmHiddenStates;
    }

    if (engineHasIOTensor(mEngine.get(), binding_names::kOutputPrefixK)
        && engineHasIOTensor(mEngine.get(), binding_names::kOutputPrefixV))
    {
        if (!mConfig.enablePrefixKVOutputs || mConfig.prefixKVOutputShape.empty())
        {
            LOG_ERROR("Engine exports prefix_k/prefix_v but config.json is missing prefix_k output metadata");
            isOk = false;
        }
    }

    if (!isOk)
    {
        LOG_ERROR("Validation failed. Please check the engine configuration.");
    }
    return isOk;
}

LLMEngineRunner::~LLMEngineRunner() noexcept
{
    for (auto& [key, graphPair] : mCudaGraphs)
    {
        cudaGraphDestroy(graphPair.first);
        cudaGraphExecDestroy(graphPair.second);
    }
    for (auto& [key, graphPair] : mBaseTreeDecodingCudaGraphs)
    {
        cudaGraphDestroy(graphPair.first);
        cudaGraphExecDestroy(graphPair.second);
    }
}

bool LLMEngineRunner::bindPluginKVCacheToEngine(int32_t activeBatchSize)
{
    // Prepare special input binding shape for prefill stage KVCache input.
    Dims const kvCacheDims = {5, {activeBatchSize, 2, mConfig.numKVHeads, mConfig.maxKVCacheCapacity, mConfig.headDim}};
    int32_t const kvCacheLayers
        = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
    bool status{true};
    // Bind KV cache tensors to execution contexts
    for (int32_t i = 0; i < kvCacheLayers; ++i)
    {
        std::string const pastKeyValuesName = binding_names::formatKVCacheName(i, true);
        std::string const presentKeyValuesName = binding_names::formatKVCacheName(i, false);

        rt::Tensor kvCacheBlock = mKVCache.getCombinedKVCacheForDecoderLayer(i);
        status &= mTRTExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        if (engineHasIOTensor(mEngine.get(), presentKeyValuesName))
        {
            status &= mTRTExecutionContext->setTensorAddress(
                presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        }
        status &= mTRTExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDims);
    }
    return status;
}

bool LLMEngineRunner::bindTRTNativeKVCacheToEngine(int32_t activeBatchSize)
{
    // TRT native path: separate K and V caches without the "2" dimension
    // Shape: [batch, num_kv_heads, seq_len, head_dim]
    Dims const kCacheDimIn = {4, {activeBatchSize, mConfig.numKVHeads, mConfig.maxKVCacheCapacity, mConfig.headDim}};
    Dims const vCacheDimIn = {4, {activeBatchSize, mConfig.numKVHeads, mConfig.maxKVCacheCapacity, mConfig.headDim}};

    int32_t const kvCacheLayers
        = (mConfig.numAttentionLayers > 0) ? mConfig.numAttentionLayers : mConfig.numDecoderLayers;
    bool status{true};
    // Bind separate K and V cache tensors to execution contexts
    for (int32_t i = 0; i < kvCacheLayers; ++i)
    {
        std::string const pastKCacheName = binding_names::formatKCacheName(i, true);
        std::string const presentKCacheName = binding_names::formatKCacheName(i, false);
        std::string const pastVCacheName = binding_names::formatVCacheName(i, true);
        std::string const presentVCacheName = binding_names::formatVCacheName(i, false);

        std::pair<rt::Tensor, rt::Tensor> kvCacheBlocks = mKVCache.getSeparateKVCacheForDecoderLayer(i);
        rt::Tensor& kCacheBlock = kvCacheBlocks.first;
        rt::Tensor& vCacheBlock = kvCacheBlocks.second;

        // Bind K cache
        status &= mTRTExecutionContext->setTensorAddress(pastKCacheName.c_str(), kCacheBlock.rawPointer());
        status &= mTRTExecutionContext->setTensorAddress(presentKCacheName.c_str(), kCacheBlock.rawPointer());

        // Bind V cache
        status &= mTRTExecutionContext->setTensorAddress(pastVCacheName.c_str(), vCacheBlock.rawPointer());
        status &= mTRTExecutionContext->setTensorAddress(presentVCacheName.c_str(), vCacheBlock.rawPointer());

        // Set shapes for K/V cache
        status &= mTRTExecutionContext->setInputShape(pastKCacheName.c_str(), kCacheDimIn);
        status &= mTRTExecutionContext->setInputShape(pastVCacheName.c_str(), vCacheDimIn);
    }
    return status;
}

bool LLMEngineRunner::bindKVCacheToEngine(int32_t activeBatchSize)
{
    if (mConfig.useTrtNativeOps)
    {
        return bindTRTNativeKVCacheToEngine(activeBatchSize);
    }
    else
    {
        return bindPluginKVCacheToEngine(activeBatchSize);
    }
}

bool LLMEngineRunner::bindRecurrentStateToEngine(int32_t activeBatchSize)
{
    if (mConfig.numLinearAttnLayers == 0)
    {
        return true;
    }

    Dims const recurrentStateDims = {4,
        {activeBatchSize, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numLinearAttnLayers; ++i)
    {
        rt::Tensor recurrentState = mKVCache.getRecurrentStateForLayer(i);
        std::string const pastName = binding_names::formatRecurrentStateName(i, /*isPast=*/true);
        std::string const presentName = binding_names::formatRecurrentStateName(i, /*isPast=*/false);

        status &= mTRTExecutionContext->setTensorAddress(pastName.c_str(), recurrentState.rawPointer());
        status &= mTRTExecutionContext->setTensorAddress(presentName.c_str(), recurrentState.rawPointer());
        status &= mTRTExecutionContext->setInputShape(pastName.c_str(), recurrentStateDims);
    }
    return status;
}

bool LLMEngineRunner::bindConvStateToEngine(int32_t activeBatchSize)
{
    if (mConfig.numLinearAttnLayers == 0)
    {
        return true;
    }

    Dims const convStateDims = {3, {activeBatchSize, mConfig.convDim, mConfig.convKernel}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numLinearAttnLayers; ++i)
    {
        rt::Tensor convState = mKVCache.getConvStateForLayer(i);
        std::string const pastName = binding_names::formatConvStateName(i, /*isPast=*/true);
        std::string const presentName = binding_names::formatConvStateName(i, /*isPast=*/false);

        status &= mTRTExecutionContext->setTensorAddress(pastName.c_str(), convState.rawPointer());
        status &= mTRTExecutionContext->setTensorAddress(presentName.c_str(), convState.rawPointer());
        status &= mTRTExecutionContext->setInputShape(pastName.c_str(), convStateDims);
    }
    return status;
}

rt::Tensor& LLMEngineRunner::getRopeCosSinCacheTensor() noexcept
{
    return mPosEncCosSinCache;
}

LLMEngineRunnerConfig LLMEngineRunner::getEngineConfig() const noexcept
{
    return mConfig;
}

rt::LinearKVCache& LLMEngineRunner::getLinearKVCache() noexcept
{
    return mKVCache;
}

bool LLMEngineRunner::setLMHeadWeights(std::string const& name, rt::Tensor const& tensor)
{
    bool status = mTRTExecutionContext->setTensorAddress(name.c_str(), const_cast<void*>(tensor.rawPointer()));
    if (!status)
    {
        LOG_ERROR("setTensorAddress failed for '%s'", name.c_str());
        return false;
    }

    bool shapeStatus = mTRTExecutionContext->setInputShape(name.c_str(), tensor.getShape().getTRTDims());
    if (!shapeStatus)
    {
        LOG_ERROR(
            "setInputShape failed for '%s' with shape %s", name.c_str(), tensor.getShape().formatString().c_str());
        return false;
    }

    return true;
}

bool LLMEngineRunner::prefillStepInputValidation(rt::Tensor const& inputsEmbeds, rt::Tensor const& contextLengths,
    rt::Tensor const& outputLogits, OptionalOutputTensor outputHiddenStates, OptionalOutputTensor outputContextEmbeds,
    rt::OptionalInputTensors deepstackEmbeds) noexcept
{
    int32_t activeBatchSize = inputsEmbeds.getShape()[0];
    int32_t prefillSequenceLength = inputsEmbeds.getShape()[1];

    // Validate inputsEmbeds
    bool const checkInputsGPUTensor = inputsEmbeds.getDeviceType() == rt::DeviceType::kGPU
        && inputsEmbeds.getDataType() == nvinfer1::DataType::kHALF && inputsEmbeds.getShape().getNumDims() == 3
        && inputsEmbeds.getShape()[2] == mConfig.hiddenSize && contextLengths.getDeviceType() == rt::DeviceType::kCPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "Invalid device type or shape of I/O tensors. InputsEmbeds should be 3D FLOAT16 on GPU with shape "
            "[batchSize, seqLen, %d], "
            "ContextLengths input should reside on CPU and the rest should reside on GPU.",
            mConfig.hiddenSize);
        return false;
    }
    bool const isBatchValid = activeBatchSize <= mConfig.maxSupportedBatchSize
        && contextLengths.getShape()[0] == activeBatchSize
        && (!mConfig.hasLogitsOutput || outputLogits.getShape()[0] == activeBatchSize);
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batchSize of the input tensors. Either batchSize is larger than "
            "maxSupportedBatchSize or batchSize is not consistent among the input tensors. "
            "Current inputsEmbeds shape: %s, contextLengths shape: %s, logits shape: %s",
            inputsEmbeds.getShape().formatString().c_str(), contextLengths.getShape().formatString().c_str(),
            outputLogits.getShape().formatString().c_str());
        return false;
    }
    if (prefillSequenceLength > mConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "Invalid sequence length of the input tensors. Input sequence length (%d) is larger "
            "than maxSupportedInputLength (%d). Current inputsEmbeds shape: %s.",
            prefillSequenceLength, mConfig.maxSupportedInputLength, inputsEmbeds.getShape().formatString().c_str());
        return false;
    }

    // Validate deepstack embeds for Qwen3-VL (these are already embedded)
    int32_t deepstackEmbedsCount = static_cast<int32_t>(deepstackEmbeds.size());
    if ((deepstackEmbedsCount != mConfig.numDeepstackFeatures) && (deepstackEmbedsCount != 0))
    {
        LOG_ERROR("Invalid deepstack embeds count. Expected either %d or 0, got %d", mConfig.numDeepstackFeatures,
            deepstackEmbedsCount);
        return false;
    }

    // Validate each deepstack embed tensor
    for (int32_t i = 0; i < deepstackEmbedsCount; ++i)
    {
        rt::Tensor const& tensor = deepstackEmbeds[i].get();
        bool const isTensorValid = tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.getShape().getNumDims() == 3
            && tensor.getShape()[0] == activeBatchSize && tensor.getShape()[1] == prefillSequenceLength
            && tensor.getShape()[2] == mConfig.hiddenSize;
        if (!isTensorValid)
        {
            LOG_ERROR(
                "Invalid deepstack embed at index %d. Expected device type: GPU, shape: [%d, %d, %d]. Current shape: "
                "%s",
                i, activeBatchSize, prefillSequenceLength, mConfig.hiddenSize,
                tensor.getShape().formatString().c_str());
            return false;
        }
    }

    bool const isLogitsShapeValid = !mConfig.hasLogitsOutput
        || (outputLogits.getShape().getNumDims() == 2 && outputLogits.getShape()[1] == mConfig.outputVocabSize);
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
    if (mConfig.enableContextEmb || mConfig.enableLmHiddenStates)
    {
        int32_t const auxDim = mConfig.enableContextEmb ? mConfig.contextEmbDim : mConfig.hiddenSize;
        char const* auxName = mConfig.enableContextEmb ? binding_names::kOutputContextEmbeds
                                                       : binding_names::kOutputLmHiddenStates;
        bool const isContextEmbedsShapeValid = outputContextEmbeds.has_value()
            && outputContextEmbeds.value().get().getShape().getNumDims() == 3
            && outputContextEmbeds.value().get().getShape()[0] == activeBatchSize
            && outputContextEmbeds.value().get().getShape()[1] == prefillSequenceLength
            && outputContextEmbeds.value().get().getShape()[2] == auxDim;
        if (!isContextEmbedsShapeValid)
        {
            LOG_ERROR(
                "With %s enabled, the output %s tensor shall be valid and has shape "
                "[activeBatchSize, %d, %d]. Current tensor shape is %s.",
                mConfig.enableContextEmb ? "context_emb" : "lm_hidden_states", auxName, prefillSequenceLength, auxDim,
                outputContextEmbeds.has_value() ? outputContextEmbeds.value().get().getShape().formatString().c_str()
                                                : "nullopt");
            return false;
        }
    }

    return true;
}

bool LLMEngineRunner::executePrefillStep(rt::Tensor const& inputsEmbeds, rt::Tensor const& hostContextLengths,
    rt::OptionalInputTensors deepstackEmbeds, rt::Tensor& outputLogits, rt::OptionalOutputTensor outputHiddenStates,
    cudaStream_t stream, rt::OptionalOutputTensor outputContextEmbeds, rt::OptionalOutputTensor outputPrefixK,
    rt::OptionalOutputTensor outputPrefixV)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus &= mTRTExecutionContext->setOptimizationProfileAsync(kPREFILL_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    bool const validateInputStatus = this->prefillStepInputValidation(
        inputsEmbeds, hostContextLengths, outputLogits, outputHiddenStates, outputContextEmbeds, deepstackEmbeds);
    if (!validateInputStatus)
    {
        LOG_ERROR("executePrefill(): Prefill request not performed due to invalid input tensors.");
        return false;
    }

    // Verify input tensorShape is valid.
    int32_t activeBatchSize = inputsEmbeds.getShape()[0];

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

    check::check(mHostSelectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
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
    // Engine input tensors - bind inputs_embeds directly
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(inputsEmbeds.rawPointer()));
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kInputsEmbeds, inputsEmbeds.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    if (mConfig.hasLastTokenIdsInput)
    {
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    }

    // Setup the KVCache start index tensor. If all KVCache are empty then we can supply zero tensor to the engine.
    // Otherwise, we shall supply the KVCache lengths tensor to the engine.
    if (!mConfig.useTrtNativeOps && mKVCache.getKVCacheAllEmpty())
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mDummyInputTensor.rawPointer());
        setEngineIOStatus
            &= mTRTExecutionContext->setInputShape(binding_names::kKVCacheStartIndex, rt::Coords{0}.getTRTDims());
    }
    else
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());
    }

    // RopeCosSin tensor address is set during object construction. We only set shape here to accommodate ND-Rope.
    // For ND-RoPE like MRope, reshape the RopeCosSinCache to match the activeBatchSize
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        check::check(mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}),
            "Tensor reshape failed");
    }

    // For persistent rope, the cache is fixed at {1, maxSeqLen, rotaryDim} and shared across all batches.
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Process deepstack embeds: bind already-embedded tensors to engine
    // Runtime must provide these tensors (zero tensors for non-multimodal use cases)
    if (mConfig.numDeepstackFeatures > 0)
    {
        // Bind deepstack embeds to engine by index
        for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
        {
            rt::Tensor const& embedTensor = deepstackEmbeds[idx].get();
            std::string embedName = binding_names::formatDeepstackEmbedsName(idx);

            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                embedName.c_str(), const_cast<void*>(embedTensor.rawPointer()));
            setEngineIOStatus
                &= mTRTExecutionContext->setInputShape(embedName.c_str(), embedTensor.getShape().getTRTDims());
        }
    }
    // Bind hidden states output if requested
    if (outputHiddenStates.has_value())
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.value().get().rawPointer());
    }

    if (mConfig.enablePrefixKVOutputs)
    {
        if (outputPrefixK.has_value())
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixK, outputPrefixK.value().get().rawPointer());
        }
        else if (mDummyOutputTensor.getMemoryCapacity() > 0)
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixK, mDummyOutputTensor.rawPointer());
        }
        if (outputPrefixV.has_value())
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixV, outputPrefixV.value().get().rawPointer());
        }
        else if (mDummyOutputTensor.getMemoryCapacity() > 0)
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixV, mDummyOutputTensor.rawPointer());
        }
    }

    if (!mPrefillAuxOutputName.empty())
    {
        if (outputContextEmbeds.has_value())
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                mPrefillAuxOutputName.c_str(), outputContextEmbeds.value().get().rawPointer());
        }
        else if (mDummyContextEmbTensor.getMemoryCapacity() > 0)
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                mPrefillAuxOutputName.c_str(), mDummyContextEmbTensor.rawPointer());
        }
    }

    if (mConfig.enableEagleSpecDecode)
    {
        // Mask input and optional token pos-ids are not used, set to dummy data.
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mDummyInputTensor.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionMask, Coords{activeBatchSize, 1, 1}.getTRTDims());
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mDummyInputTensor.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionPosId, Coords{activeBatchSize, 1}.getTRTDims());
    }

    // Engine output tensors.
    if (mConfig.hasLogitsOutput)
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    }

    // Bind the KVCache IO to the engine
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    // Bind recurrent state for hybrid layers
    setEngineIOStatus &= this->bindConvStateToEngine(activeBatchSize);
    setEngineIOStatus &= this->bindRecurrentStateToEngine(activeBatchSize);

    if (!setEngineIOStatus)
    {
        LOG_ERROR("executePrefill(): Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
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

bool LLMEngineRunner::vanillaDecodingStepInputValidation(
    rt::Tensor const& inputsEmbeds, rt::Tensor const& outputLogits) noexcept
{
    int32_t activeBatchSize = inputsEmbeds.getShape()[0];
    bool const checkInputsGPUTensor = inputsEmbeds.getDeviceType() == rt::DeviceType::kGPU
        && inputsEmbeds.getDataType() == nvinfer1::DataType::kHALF
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR(
            "Invalid device type of the input tensors. inputsEmbeds (FLOAT16) and outputLogits "
            "should reside on GPU.");
        return false;
    }
    int32_t activeKVCacheBatchSize = mKVCache.getActiveBatchSize();
    bool const isBatchValid = activeBatchSize == activeKVCacheBatchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batchSize of the input tensors. batchSize shall be equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }
    bool checkInputShapeValid = inputsEmbeds.getShape().getNumDims() == 3 && inputsEmbeds.getShape()[1] == 1
        && inputsEmbeds.getShape()[2] == mConfig.hiddenSize && outputLogits.getShape().getNumDims() == 2
        && outputLogits.getShape()[1] == mConfig.outputVocabSize;
    if (!checkInputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the input tensors. The input tensor should have shape "
            "[activeBatchSize, 1, hiddenSize] and the output tensor should have shape [activeBatchSize, "
            "outputVocabSize].");
        return false;
    }

    return true;
}

bool LLMEngineRunner::vanillaDecodingStepPrepareInputs(int32_t activeBatchSize, cudaStream_t stream)
{
    // For vanilla decode stage, the selected token indices are always 0.
    // Also setup the sequence length of each sequence for this run based on committed KVCache length.
    check::check(mSelectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mSequenceContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        check::check(mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}),
            "Tensor reshape failed");
    }

    CUDA_CHECK(cudaMemsetAsync(mSelectTokenIndices.rawPointer(), 0, activeBatchSize * sizeof(int64_t), stream));

    // For TRT native path, the sequence length input always refer to the length of Q.
    // For plugin path, the sequence length refer to the length of K and V.
    if (mConfig.useTrtNativeOps)
    {
        CUDA_CHECK(cudaMemsetAsync(mSequenceContextLengths.rawPointer(), 0, activeBatchSize * sizeof(int32_t), stream));
    }
    else
    {
        // Get KV cache lengths
        rt::Tensor& kvCacheLengths = mKVCache.getKVCacheLengths();
        CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), kvCacheLengths.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    }

    // Increment the sequence length due to the implementation constraint of AttentionPlugin.
    constexpr int32_t kDECODE_INCREMENT{1};
    kernel::incrementLengthTensor(mSequenceContextLengths, kDECODE_INCREMENT, stream);

    return true;
}

bool LLMEngineRunner::vanillaDecodingStepBindTensors(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
    rt::OptionalOutputTensor outputHiddenStates, rt::OptionalOutputTensor outputContextEmbeds, int32_t activeBatchSize)
{
    bool setEngineIOStatus{true};
    // Engine input tensors - bind inputs_embeds directly
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(inputsEmbeds.rawPointer()));
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kInputsEmbeds, inputsEmbeds.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());

    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Update KV cache shapes to match activeBatchSize
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);
    setEngineIOStatus &= this->bindConvStateToEngine(activeBatchSize);
    setEngineIOStatus &= this->bindRecurrentStateToEngine(activeBatchSize);

    // Bind deepstack_embeds to dummy tensors for Qwen3VL models during decoding
    if (mConfig.numDeepstackFeatures > 0)
    {
        for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
        {
            std::string deepstackEmbedName = binding_names::formatDeepstackEmbedsName(idx);
            setEngineIOStatus
                &= mTRTExecutionContext->setTensorAddress(deepstackEmbedName.c_str(), mDummyInputTensor.rawPointer());
            setEngineIOStatus &= mTRTExecutionContext->setInputShape(
                deepstackEmbedName.c_str(), rt::Coords{activeBatchSize, 1, mConfig.hiddenSize}.getTRTDims());
        }
    }

    // Engine output tensors.
    if (mConfig.hasLogitsOutput)
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    }

    if (outputHiddenStates.has_value())
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.value().get().rawPointer());
    }
    else if (mDummyOutputTensor.getMemoryCapacity() > 0
        && engineHasIOTensor(mEngine.get(), binding_names::kOutputHiddenStates))
    {
        // Engine has hidden_states output but user doesn't need it
        // Bind to dummy buffer (EAGLE or Qwen3-Omni text-only mode)
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, mDummyOutputTensor.rawPointer());
    }

    if (mConfig.enablePrefixKVOutputs && mDummyOutputTensor.getMemoryCapacity() > 0)
    {
        if (engineHasIOTensor(mEngine.get(), binding_names::kOutputPrefixK))
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixK, mDummyOutputTensor.rawPointer());
        }
        if (engineHasIOTensor(mEngine.get(), binding_names::kOutputPrefixV))
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                binding_names::kOutputPrefixV, mDummyOutputTensor.rawPointer());
        }
    }

    if (!mPrefillAuxOutputName.empty())
    {
        if (outputContextEmbeds.has_value())
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                mPrefillAuxOutputName.c_str(), outputContextEmbeds.value().get().rawPointer());
        }
        else if (mDummyContextEmbTensor.getMemoryCapacity() > 0)
        {
            setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
                mPrefillAuxOutputName.c_str(), mDummyContextEmbTensor.rawPointer());
        }
    }

    return setEngineIOStatus;
}

bool LLMEngineRunner::executeVanillaDecodingStep(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
    rt::OptionalOutputTensor outputHiddenStates, cudaStream_t stream, rt::OptionalOutputTensor outputContextEmbeds)
{
    bool const validateInputStatus = this->vanillaDecodingStepInputValidation(inputsEmbeds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("executeGeneration(): Generation request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const activeBatchSize = inputsEmbeds.getShape()[0];
    if (!vanillaDecodingStepPrepareInputs(activeBatchSize, stream))
    {
        LOG_ERROR("Failed to prepare inputs.");
        return false;
    }

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    auto const graphHash = decodingKey(inputsEmbeds, outputLogits, mActiveLoraWeightsName);
    if (mCudaGraphs.find(graphHash) != mCudaGraphs.end())
    {
        LOG_DEBUG("Use pre-captured CUDA graph for vanilla decoding step.");
        cudaGraphExec_t graphExec = mCudaGraphs[graphHash].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setOptimizationProfileStatus{true};
        setOptimizationProfileStatus
            &= mTRTExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
        if (!setOptimizationProfileStatus)
        {
            LOG_ERROR("Failed to set optimization profile to the engine");
            throw std::runtime_error("Failed to set optimization profile to the engine");
        }

        LOG_INFO("Vanilla decoding step CUDA graph not captured.");
        if (!vanillaDecodingStepBindTensors(
                inputsEmbeds, outputLogits, outputHiddenStates, outputContextEmbeds, activeBatchSize))
        {
            LOG_ERROR("Failed to bind tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mTRTExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("Failed on TensorRT decode stage enqueueV3() call.");
            return false;
        }
    }

    // Completed decoding step, commit the KVCache length of this run.
    constexpr int32_t kVANILLA_DECODE_INCREMENT{1};
    mKVCache.commitSequenceLength(kVANILLA_DECODE_INCREMENT, stream);
    LOG_DEBUG("Decoding stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool LLMEngineRunner::eagleBaseTreeDecodingStepInputValidation(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& outputLogits,
    rt::Tensor const& outputHiddenStates) noexcept
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = baseTreeDecodingInputsEmbeds.getDeviceType() == rt::DeviceType::kGPU
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
    bool const isInputTypeValid = baseTreeDecodingInputsEmbeds.getDataType() == DataType::kHALF
        && baseTreeDecodingMask.getDataType() == DataType::kINT8 && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Input embeds shall be FLOAT16, hidden states I/O shall be "
            "FLOAT16, "
            "base tree decoding mask shall be INT8, output logits shall be FLOAT32.");
        return false;
    }
    // Validate shapes of the input tensors.
    bool const isBatchValid = baseTreeDecodingInputsEmbeds.getShape()[0] == mKVCache.getActiveBatchSize()
        && baseTreeDecodingMask.getShape()[0] == mKVCache.getActiveBatchSize();
    if (!isBatchValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid batchSize of the input tensors. batchSize shall be "
            "equal to the active batch "
            "size set by the previous prefill stage.");
        return false;
    }

    int64_t const baseTreeDecodingSize = baseTreeDecodingInputsEmbeds.getShape()[1];
    bool const isBaseTreeDecodingSizeValid = baseTreeDecodingMask.getShape()[1] == baseTreeDecodingSize
        && baseTreeDecodingMask.getShape()[2] == baseTreeDecodingSize
        && baseTreeDecodingInputsEmbeds.getShape()[2] == mConfig.hiddenSize;
    if (!isBaseTreeDecodingSizeValid)
    {
        LOG_ERROR(
            "eagleBaseTreeDecodingStepInputValidation(): Invalid base tree decoding size of the input tensors. "
            "Base tree decoding size %d, expected hiddenSize %d, current base tree decoding mask shape: %s, "
            "inputsEmbeds shape: %s",
            baseTreeDecodingSize, mConfig.hiddenSize, baseTreeDecodingMask.getShape().formatString().c_str(),
            baseTreeDecodingInputsEmbeds.getShape().formatString().c_str());
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

bool LLMEngineRunner::eagleBaseTreeDecodingStepPrepareInputs(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor const& baseTreeDecodingMask, int32_t activeBatchSize, cudaStream_t stream)
{
    int32_t const baseTreeDecodingSize = static_cast<int32_t>(baseTreeDecodingInputsEmbeds.getShape()[1]);
    int32_t const packedBaseTreeDecodingMaskLen = static_cast<int32_t>(divUp(baseTreeDecodingSize, 32));

    // Prepare extra input for engine execution. Assemble mask, position indices, select token
    // indices, sequence context lengths.
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndices = mKVCache.getKVCacheLengths();

    // Prepare inputs for plugin-based attention
    check::check(mSelectTokenIndices.reshape({activeBatchSize, baseTreeDecodingSize}),
        "Tensor reshape failed"); // 2D tensor [batch, num_tokens]
    check::check(mSequenceContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mEagleBasePositionIds.reshape({activeBatchSize, baseTreeDecodingSize}), "Tensor reshape failed");
    check::check(mEagleBasePackedMask.reshape({activeBatchSize, baseTreeDecodingSize, packedBaseTreeDecodingMaskLen}),
        "Tensor reshape failed");

    kernel::prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMask, sequenceStartIndices, mEagleBasePackedMask,
        mEagleBasePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);
    return true;
}

bool LLMEngineRunner::eagleBaseTreeDecodingStepBindTensors(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, int32_t activeBatchSize)
{
    int32_t const baseTreeDecodingSize = static_cast<int32_t>(baseTreeDecodingInputsEmbeds.getShape()[1]);
    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(baseTreeDecodingInputsEmbeds.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kInputsEmbeds, baseTreeDecodingInputsEmbeds.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mKVCache.getKVCacheLengths().getShape().getTRTDims());

    // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        check::check(mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}),
            "Tensor reshape failed");
    }

    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Update KV cache shapes to match activeBatchSize
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);
    setEngineIOStatus &= this->bindConvStateToEngine(activeBatchSize);
    setEngineIOStatus &= this->bindRecurrentStateToEngine(activeBatchSize);

    // Bind packed attention mask for plugin-based attention
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mEagleBasePackedMask.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kAttentionMask, mEagleBasePackedMask.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mEagleBasePositionIds.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kAttentionPosId, mEagleBasePositionIds.getShape().getTRTDims());

    // Bind deepstack_embeds to dummy tensors for Qwen3VL models during Eagle base tree decoding
    if (mConfig.numDeepstackFeatures > 0)
    {
        for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
        {
            std::string deepstackEmbedName = binding_names::formatDeepstackEmbedsName(idx);
            setEngineIOStatus
                &= mTRTExecutionContext->setTensorAddress(deepstackEmbedName.c_str(), mDummyInputTensor.rawPointer());
            setEngineIOStatus &= mTRTExecutionContext->setInputShape(deepstackEmbedName.c_str(),
                rt::Coords{activeBatchSize, baseTreeDecodingSize, mConfig.hiddenSize}.getTRTDims());
        }
    }

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    return true;
}

bool LLMEngineRunner::executeEagleBaseTreeDecodingStep(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
    cudaStream_t stream)
{
    bool const validateInputStatus = this->eagleBaseTreeDecodingStepInputValidation(
        baseTreeDecodingInputsEmbeds, baseTreeDecodingMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Eagle base tree decoding request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const activeBatchSize = baseTreeDecodingInputsEmbeds.getShape()[0];

    if (!eagleBaseTreeDecodingStepPrepareInputs(
            baseTreeDecodingInputsEmbeds, baseTreeDecodingMask, activeBatchSize, stream))
    {
        LOG_ERROR("Failed to prepare inputs.");
        return false;
    }

    // Launch cuda graph if available for this request, otherwise proceed with normal TensorRT engine execution step.
    auto const graphHash
        = baseKey(baseTreeDecodingInputsEmbeds, outputLogits, outputHiddenStates, mActiveLoraWeightsName);
    if (mBaseTreeDecodingCudaGraphs.find(graphHash) != mBaseTreeDecodingCudaGraphs.end())
    {
        LOG_DEBUG("Use pre-captured CUDA graph for eagle base tree decoding step.");
        cudaGraphExec_t graphExec = mBaseTreeDecodingCudaGraphs[graphHash].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setOptimizationProfileStatus{true};
        setOptimizationProfileStatus
            &= mTRTExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
        if (!setOptimizationProfileStatus)
        {
            LOG_ERROR("Failed to set optimization profile to the engine");
            throw std::runtime_error("Failed to set optimization profile to the engine");
        }

        // Prepare and bind tensors using shared helper function
        if (!eagleBaseTreeDecodingStepBindTensors(
                baseTreeDecodingInputsEmbeds, outputLogits, outputHiddenStates, activeBatchSize))
        {
            LOG_ERROR("Failed to bind tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mTRTExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("Failed on TensorRT eagle base tree decoding stage enqueueV3() call.");
            return false;
        }
    }

    // Note in the base tree decoding step we explicitly don't commit the KVCache since we process the "whole tree" in
    // these steps.
    LOG_DEBUG("Eagle base tree decoding stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool LLMEngineRunner::captureVanillaDecodingCudaGraph(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
    std::string const& loraWeightsPath, cudaStream_t stream, rt::OptionalOutputTensor outputHiddenStates,
    rt::OptionalOutputTensor outputContextEmbeds)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mTRTExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    auto const key = decodingKey(inputsEmbeds, outputLogits, loraWeightsPath);
    if (mCudaGraphs.find(key) != mCudaGraphs.end())
    {
        LOG_INFO("CUDA graph already captured for the input tensors with LoRA weights %s.", loraWeightsPath.c_str());
        return true;
    }

    if (isLoraWeightsSupported() && !this->switchLoraWeights(loraWeightsPath))
    {
        LOG_ERROR("Failed to switch LoRA weights to '%s', unable to capture CUDA graph.", loraWeightsPath.c_str());
        return false;
    }

    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int32_t const activeBatchSize = inputsEmbeds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU,
        DataType::kINT32, "vanilla_reuse_kv_cache_lengths");

    mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    // Validate the input tensors.
    bool const validateInputStatus = this->vanillaDecodingStepInputValidation(inputsEmbeds, outputLogits);
    if (!validateInputStatus)
    {
        LOG_ERROR("Generation request is invalid, unable to capture CUDA graph.");
        return false;
    }
    if (!vanillaDecodingStepPrepareInputs(activeBatchSize, stream))
    {
        LOG_ERROR("Failed to prepare inputs.");
        return false;
    }

    if (!vanillaDecodingStepBindTensors(
            inputsEmbeds, outputLogits, outputHiddenStates, outputContextEmbeds, activeBatchSize))
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution. This will trigger the shape machine of TensorRT engine to avoid cudaGraph capture
    // error.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT engine enqueueV3() call.");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto graphPair = captureTRTCudaGraph(mTRTExecutionContext.get(), stream);
    if (!graphPair)
    {
        LOG_WARNING("Failed to capture CUDA graph.");
        return false;
    }
    else
    {
        LOG_DEBUG(
            "CUDA graph captured successfully for input shape %s with LoRA weights '%s' (Empty string if no LoRA "
            "weights).",
            inputsEmbeds.getShape().formatString().c_str(), loraWeightsPath.c_str());
        mCudaGraphs[key] = graphPair.value();
        return true;
    }
}

bool LLMEngineRunner::captureEagleBaseTreeDecodingCudaGraph(rt::Tensor const& baseTreeDecodingInputsEmbeds,
    rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
    std::string const& loraWeightsName, cudaStream_t stream)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mTRTExecutionContext->setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    auto const key = baseKey(baseTreeDecodingInputsEmbeds, outputLogits, outputHiddenStates, loraWeightsName);
    if (mBaseTreeDecodingCudaGraphs.find(key) != mBaseTreeDecodingCudaGraphs.end())
    {
        LOG_INFO("CUDA graph already captured for the input tensors with LoRA weights %s.", loraWeightsName.c_str());
        return true;
    }

    if (isLoraWeightsSupported() && !this->switchLoraWeights(loraWeightsName))
    {
        LOG_ERROR("Failed to switch LoRA weights to '%s', unable to capture CUDA graph.", loraWeightsName.c_str());
        return false;
    }

    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int32_t const activeBatchSize = baseTreeDecodingInputsEmbeds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(
        reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);

    mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    bool const validateInputStatus = this->eagleBaseTreeDecodingStepInputValidation(
        baseTreeDecodingInputsEmbeds, baseTreeDecodingMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Eagle base tree decoding request not performed due to invalid input tensors.");
        return false;
    }

    // Prepare and bind tensors using shared helper function
    if (!eagleBaseTreeDecodingStepPrepareInputs(
            baseTreeDecodingInputsEmbeds, baseTreeDecodingMask, activeBatchSize, stream))
    {
        LOG_ERROR("Failed to prepare inputs.");
        return false;
    }

    if (!eagleBaseTreeDecodingStepBindTensors(
            baseTreeDecodingInputsEmbeds, outputLogits, outputHiddenStates, activeBatchSize))
    {
        LOG_ERROR("Failed to bind tensors.");
        return false;
    }

    // launch the engine execution. This will trigger the shape machine of TensorRT engine to avoid cudaGraph capture.
    // error.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);

    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT eagle base tree decoding stage enqueueV3() call.");
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto graphPair = captureTRTCudaGraph(mTRTExecutionContext.get(), stream);
    if (!graphPair)
    {
        LOG_WARNING("Failed to capture CUDA graph.");
        return false;
    }
    else
    {
        LOG_DEBUG(
            "CUDA graph captured successfully for input shape %s with LoRA weights '%s' (Empty string if no LoRA "
            "weights).",
            baseTreeDecodingInputsEmbeds.getShape().formatString().c_str(), loraWeightsName.c_str());
        mBaseTreeDecodingCudaGraphs[key] = graphPair.value();
        return true;
    }
}

bool LLMEngineRunner::resetLoraWeights()
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
            &= mTRTExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), mDummyInputTensor.rawPointer());

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
        resetStatus &= mTRTExecutionContext->setInputShape(loraWeightsTensorName.c_str(), emptyLoraShape);
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

bool LLMEngineRunner::switchLoraWeights(std::string const& loraWeightsName)
{
    if (!isLoraWeightsSupported())
    {
        LOG_ERROR("switchLoraWeights(): API call is invalid. LLM engine does not support LoRA weights.");
        return false;
    }
    if (loraWeightsName.empty())
    {
        this->resetLoraWeights();
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
                &= mTRTExecutionContext->setInputShape(loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus
                &= mTRTExecutionContext->setInputShape(loraWeightsTensorName.c_str(), loraTensorIt->getTRTDims());
            setLoraWeightsStatus
                &= mTRTExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
            setLoraWeightsStatus
                &= mTRTExecutionContext->setTensorAddress(loraWeightsTensorName.c_str(), loraTensorIt->rawPointer());
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
            setLoraWeightsStatus &= mTRTExecutionContext->setInputShape(loraWeightsTensorName.c_str(), shape);
            setLoraWeightsStatus &= mTRTExecutionContext->setTensorAddress(
                loraWeightsTensorName.c_str(), mDummyInputTensor.rawPointer());
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

bool LLMEngineRunner::isLoraWeightsSupported() const noexcept
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
