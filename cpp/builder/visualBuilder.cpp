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

#include "visualBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "multimodal/common/imageUtils.h"

#include <cmath>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

VisualBuilder::VisualBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, VisualBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool VisualBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    // Without these workarounds, fusing the separate Q/K/V projections that
    // feed ViTAttentionPlugin yields an engine that faults at execution
    // context creation.
    std::string const lunowudFlags = applyCompileWorkarounds(/*maxBatchSize=*/1);
    if (!lunowudFlags.empty())
    {
        LOG_INFO("Using __LUNOWUD=%s", lunowudFlags.c_str());
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Parse ONNX model
    std::string const onnxPath = (mOnnxDir / "model.onnx").string();
    auto parser = parseOnnxModel(network.get(), onnxPath);
    if (!parser)
    {
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "Visual").c_str());

    // Create builder config
    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        return false;
    }

    if (mBuilderConfig.profilingDetailed)
    {
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        LOG_INFO("Profiling verbosity set to DETAILED");
    }

    // Setup optimization profile
    if (!setupVisualOptimizationProfile(*builder.get(), *config.get(), *network.get()))
    {
        return false;
    }

    // Create engine directory
    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving Visual engine.", mEngineDir.string().c_str());
    }

    // Copy the model's required weight file (GN projection / patch embedder) next to the engine so
    // the runtime finds it under the engine dir in the standard onnx_dir -> engine_dir flow.
    auto copyRequiredWeightFile = [this](char const* filename) -> bool {
        std::string const dst = (mEngineDir / filename).string();
        if (!file_io::copyFile((mOnnxDir / filename).string(), dst))
        {
            LOG_ERROR("Failed to copy required weight file %s to %s", filename, dst.c_str());
            return false;
        }
        LOG_INFO("Copied required weight file to %s", dst.c_str());
        return true;
    };
    if (mModelType == multimodal::ModelType::PHI4MM && !copyRequiredWeightFile("phi4mm_gn_proj.safetensors"))
    {
        return false;
    }
    if (mModelType == multimodal::ModelType::NEMOTRON_OMNI_VISION_ENCODER
        && !copyRequiredWeightFile("nemotron_omni_embedder.safetensors"))
    {
        return false;
    }

    // Build and save engine
    std::string engineFilePath = (mEngineDir / "visual.engine").string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        return false;
    }

    // Copy config
    if (!copyConfig())
    {
        return false;
    }

    return true;
}

bool VisualBuilder::parseConfig()
{
    std::string const configPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(configPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // Read model type from vision_config.model_type or top-level model_type
    std::string modelTypeStr;
    if (mModelConfig.contains(kVisionConfigKey) && mModelConfig[kVisionConfigKey].contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kVisionConfigKey][kModelTypeKey].get<std::string>();
    }
    else if (mModelConfig.contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kModelTypeKey].get<std::string>();
    }
    else
    {
        LOG_ERROR(
            "model_type not found in config.json (expected either vision_config.model_type or top-level model_type)");
        return false;
    }

    mModelType = multimodal::stringToModelType(modelTypeStr);

    switch (mModelType)
    {
    case multimodal::ModelType::INTERNVL:
        mNumChannels = mModelConfig[kVisionConfigKey]["num_channels"].get<int64_t>();
        mImageSizeH = mModelConfig[kVisionConfigKey]["image_size"][0].get<int64_t>();
        mImageSizeW = mModelConfig[kVisionConfigKey]["image_size"][1].get<int64_t>();
        break;

    case multimodal::ModelType::PHI4MM:
        // Default Phi-4MM vision input
        mNumChannels = 3;
        // Prefer HF config's crop_size if available
        if (mModelConfig.contains(kEmbdLayerKey) && mModelConfig[kEmbdLayerKey].contains(kImageEmbdLayerKey)
            && mModelConfig[kEmbdLayerKey][kImageEmbdLayerKey].contains("crop_size"))
        {
            int64_t const crop = mModelConfig[kEmbdLayerKey][kImageEmbdLayerKey]["crop_size"].get<int64_t>();
            mImageSizeH = crop;
            mImageSizeW = crop;
        }
        else
        {
            LOG_INFO("Phi-4MM crop_size not found in config.json; defaulting to 448x448.");
            mImageSizeH = 448;
            mImageSizeW = 448;
        }
        break;

    case multimodal::ModelType::NEMOTRON_OMNI_VISION_ENCODER:
    {
        mNumChannels = 3;
        if (mModelConfig.contains("force_image_size"))
        {
            int64_t const imgSize = mModelConfig["force_image_size"].get<int64_t>();
            mImageSizeH = imgSize;
            mImageSizeW = imgSize;
        }
        else
        {
            LOG_ERROR("Nemotron-Omni: force_image_size not found in config.json");
            return false;
        }
        LOG_INFO("Nemotron-Omni RADIO: channels=%ld, image_size=%ldx%ld", mNumChannels, mImageSizeH, mImageSizeW);
        break;
    }

    case multimodal::ModelType::GEMMA4_UNIFIED_VISION:
    {
        // The exporter records the checkpoint's fixed per-image soft-token
        // budget in builder_config. The total request budget remains a
        // builder/CLI choice so a profile can support multiple images.
        if (mModelConfig.contains("builder_config"))
        {
            auto const& exportedBuilderConfig = mModelConfig["builder_config"];
            int64_t const exportedMaxPerImage
                = exportedBuilderConfig.value("max_image_tokens_per_image", mBuilderConfig.maxImageTokensPerImage);
            if (exportedMaxPerImage < 1 || mBuilderConfig.minImageTokens < 1
                || mBuilderConfig.maxImageTokens < mBuilderConfig.minImageTokens)
            {
                LOG_ERROR(
                    "Invalid Gemma4 Unified visual token bounds: requested min=%ld, requested max=%ld, "
                    "exported max_per_image=%ld",
                    mBuilderConfig.minImageTokens, mBuilderConfig.maxImageTokens, exportedMaxPerImage);
                return false;
            }
            mBuilderConfig.maxImageTokensPerImage
                = std::min({mBuilderConfig.maxImageTokensPerImage, exportedMaxPerImage, mBuilderConfig.maxImageTokens});
            LOG_INFO(
                "Gemma4 Unified visual profile: min=%ld, configurable total max=%ld, "
                "sidecar-capped max_per_image=%ld",
                mBuilderConfig.minImageTokens, mBuilderConfig.maxImageTokens, mBuilderConfig.maxImageTokensPerImage);
        }
        break;
    }

    case multimodal::ModelType::UNKNOWN: LOG_ERROR("Unsupported model type: %s", modelTypeStr.c_str()); return false;

    default: break;
    }

    if (mModelConfig.value("use_trt_native_vit_attn", false))
    {
        logTrtNativeAttentionPath("ViTAttention");
    }

    return true;
}

bool VisualBuilder::setupVisualOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* visualProfile = builder.createOptimizationProfile();
    bool result = true;

    switch (mModelType)
    {
    case multimodal::ModelType::QWEN2_VL:
    case multimodal::ModelType::QWEN2_5_VL:
    case multimodal::ModelType::QWEN3_VL:
    case multimodal::ModelType::QWEN3_5:
    case multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER:
    case multimodal::ModelType::COSMOS3_EDGE: result = setupQwenViTProfile(*visualProfile, network); break;

    case multimodal::ModelType::INTERNVL:
    case multimodal::ModelType::PHI4MM: result = setupInternPhi4ViTProfile(*visualProfile); break;

    case multimodal::ModelType::NEMOTRON_OMNI_VISION_ENCODER:
        result = setupNemotronOmniViTProfile(*visualProfile);
        break;

    case multimodal::ModelType::GEMMA4_VISION: result = setupGemma4ViTProfile(*visualProfile, network); break;

    case multimodal::ModelType::GEMMA4_UNIFIED_VISION:
        result = setupGemma4UnifiedVisionProfile(*visualProfile, network);
        break;

    default: LOG_ERROR("Unsupported model type for visual encoder: %d", static_cast<int>(mModelType)); return false;
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(visualProfile, "visual_profile", &network).c_str());

    config.addOptimizationProfile(visualProfile);
    return true;
}

bool VisualBuilder::setupQwenViTProfile(
    nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // In Qwen-VL, HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
    int64_t minHW = mBuilderConfig.minImageTokens * 4;
    int64_t maxHW = mBuilderConfig.maxImageTokens * 4;
    int64_t optHW = (mBuilderConfig.minImageTokens + mBuilderConfig.maxImageTokens) / 2 * 4;

    // Infer dimensions from the network
    int64_t inputDim = 0;
    int64_t ropeEmbedSize = 0;

    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        if (strcmp(input->getName(), binding_names::kVisualInput) == 0)
        {
            inputDim = input->getDimensions().d[1];
        }
        else if (strcmp(input->getName(), binding_names::kRotaryPosEmb) == 0)
        {
            ropeEmbedSize = input->getDimensions().d[1];
        }
    }

    if (inputDim == 0)
    {
        LOG_ERROR("Cannot infer inputDim. Do you have proper ONNX input: %s?", binding_names::kVisualInput);
        return false;
    }

    // Cosmos3-Edge (SigLIP2) has no rotary embeddings, so its graph has no rotary_pos_emb input;
    // every other Qwen-family ViT requires it.
    if (ropeEmbedSize == 0 && mModelType != multimodal::ModelType::COSMOS3_EDGE)
    {
        LOG_ERROR("Cannot infer ropeEmbedSize. Do you have proper ONNX input: %s?", binding_names::kRotaryPosEmb);
        return false;
    }

    // Base inputs
    result &= setOptimizationProfile(&profile, binding_names::kVisualInput, createDims({minHW, inputDim}),
        createDims({optHW, inputDim}), createDims({maxHW, inputDim}));
    if (ropeEmbedSize > 0)
    {
        result &= setOptimizationProfile(&profile, binding_names::kRotaryPosEmb, createDims({minHW, ropeEmbedSize}),
            createDims({optHW, ropeEmbedSize}), createDims({maxHW, ropeEmbedSize}));
    }
    // OPT is the image-shaped group count -- TRT picks tactics from it, so widening MAX for
    // per-frame video groups must not drag it along.
    int64_t const optCuGroups = std::max<int64_t>(1, mBuilderConfig.maxImageTokens / mBuilderConfig.minImageTokens);
    int64_t const maxCuGroups = rt::imageUtils::maxCuSeqlenGroups(mBuilderConfig.maxImageTokens);
    result &= setOptimizationProfile(&profile, binding_names::kCuSeqlens, createDims({2}),
        createDims({optCuGroups + 1}), createDims({maxCuGroups + 1}));

    // kv_lengths is required when using TRT-native attention (TRT >= 11).
    // Read the flag from the exporter's config.json.
    mBuilderConfig.useTrtNativeVitAttn = mModelConfig.value("use_trt_native_vit_attn", false);
    if (mBuilderConfig.useTrtNativeVitAttn)
    {
        result &= setOptimizationProfile(&profile, binding_names::kKvLengths, createDims({2}),
            createDims({optCuGroups + 1}), createDims({maxCuGroups + 1}));
    }

    // max_seqlen_carrier is only present when using the ViTAttentionPlugin path (TRT 10).
    // The TRT-native attention path (TRT >= 11) does not emit this input.
    bool hasMaxSeqLenCarrier = false;
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        if (strcmp(network.getInput(i)->getName(), binding_names::kMaxSeqLenCarrier) == 0)
        {
            hasMaxSeqLenCarrier = true;
            break;
        }
    }
    if (hasMaxSeqLenCarrier)
    {
        int32_t maxSeqLen = static_cast<int32_t>(mBuilderConfig.maxImageTokensPerImage * 4);
        result &= setOptimizationProfile(&profile, binding_names::kMaxSeqLenCarrier, createDims({1}),
            createDims({maxSeqLen / 2}), createDims({maxSeqLen}));
    }

    // Additional inputs
    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        // Use maxImageTokens as a safe upper bound for cumulative window sequence lengths.
        result &= setOptimizationProfile(&profile, binding_names::kCuWindowSeqlens, createDims({2}),
            createDims({mBuilderConfig.maxImageTokens}), createDims({mBuilderConfig.maxImageTokens}));
        if (mBuilderConfig.useTrtNativeVitAttn)
        {
            bool hasKvLengthsWindow = false;
            for (int32_t i = 0; i < network.getNbInputs(); ++i)
            {
                if (strcmp(network.getInput(i)->getName(), binding_names::kKvLengthsWindow) == 0)
                {
                    hasKvLengthsWindow = true;
                    break;
                }
            }
            if (hasKvLengthsWindow)
            {
                result &= setOptimizationProfile(&profile, binding_names::kKvLengthsWindow, createDims({2}),
                    createDims({mBuilderConfig.maxImageTokens}), createDims({mBuilderConfig.maxImageTokens}));
            }
        }
        result &= setOptimizationProfile(&profile, binding_names::kWindowIndex, createDims({minHW / 4}),
            createDims({optHW / 4}), createDims({maxHW / 4}));
        result &= setOptimizationProfile(&profile, binding_names::kReverseWindowIndex, createDims({minHW / 4}),
            createDims({optHW / 4}), createDims({maxHW / 4}));
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL || mModelType == multimodal::ModelType::QWEN3_5
        || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER
        || mModelType == multimodal::ModelType::COSMOS3_EDGE)
    {
        result &= setOptimizationProfile(&profile, binding_names::kFastPosEmbIdx, createDims({4, minHW}),
            createDims({4, optHW}), createDims({4, maxHW}));
        result &= setOptimizationProfile(&profile, binding_names::kFastPosEmbWeight, createDims({4, minHW}),
            createDims({4, optHW}), createDims({4, maxHW}));
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile at setupQwenViTProfile().");
    }

    return result;
}

bool VisualBuilder::setupInternPhi4ViTProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // For InternVL and Phi-4MM models, each image block contains 256 tokens (16x16 patch grid)
    // This is model-specific and comes from the vision encoder's patch size configuration
    constexpr int64_t kBlockLength = 256;

    if (mBuilderConfig.minImageTokens % kBlockLength != 0 || mBuilderConfig.maxImageTokens % kBlockLength != 0)
    {
        LOG_ERROR(
            "minImageTokens and maxImageTokens must be divisible by %ld for InternVL/Phi4-MM ViT model.", kBlockLength);
        return false;
    }

    int64_t minNumBlocks = mBuilderConfig.minImageTokens / kBlockLength;
    int64_t maxNumBlocks = mBuilderConfig.maxImageTokens / kBlockLength;
    int64_t optNumBlocks = (minNumBlocks + maxNumBlocks) / 2;

    result &= setOptimizationProfile(&profile, binding_names::kVisualInput,
        createDims({minNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}),
        createDims({optNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}),
        createDims({maxNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}));

    return result;
}

bool VisualBuilder::setupNemotronOmniViTProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Tokens per tile: (H / patch_size * downsample_ratio)^2
    int64_t const patchSize = mModelConfig["patch_size"].get<int64_t>();
    double const downsampleRatio = mModelConfig["downsample_ratio"].get<double>();
    int64_t const tokensPerSide = static_cast<int64_t>(mImageSizeH / patchSize * downsampleRatio);
    int64_t const kTokensPerBlock = tokensPerSide * tokensPerSide;

    int64_t maxNumBlocks = std::max<int64_t>(1, mBuilderConfig.maxImageTokens / kTokensPerBlock);
    int64_t minNumBlocks = std::max<int64_t>(1, mBuilderConfig.minImageTokens / kTokensPerBlock);
    int64_t optNumBlocks = (minNumBlocks + maxNumBlocks) / 2;

    // Engine input is patch embeddings [blocks, patches, vitHidden] (the embedder GEMM runs in the
    // runtime), not pixels; patch count is dynamic (video uses aspect-preserving grids).
    if (!mModelConfig.contains("vit_hidden_size"))
    {
        LOG_ERROR(
            "Nemotron-Omni: vit_hidden_size not found in config.json — re-export the visual model "
            "(the ONNX boundary moved: the engine now consumes patch embeddings, not pixels)");
        return false;
    }
    int64_t const vitHidden = mModelConfig["vit_hidden_size"].get<int64_t>();
    int64_t const scale = static_cast<int64_t>(std::llround(1.0 / downsampleRatio));
    int64_t const maxPatches = (mImageSizeH / patchSize) * (mImageSizeW / patchSize);
    int64_t const minPatches = scale * scale;

    result &= setOptimizationProfile(&profile, binding_names::kVisualInput,
        createDims({minNumBlocks, minPatches, vitHidden}), createDims({optNumBlocks, maxPatches, vitHidden}),
        createDims({maxNumBlocks, maxPatches, vitHidden}));

    result &= setOptimizationProfile(&profile, binding_names::kVisualShuffleIndices,
        createDims({minPatches / (scale * scale), scale * scale}),
        createDims({maxPatches / (scale * scale), scale * scale}),
        createDims({maxPatches / (scale * scale), scale * scale}));

    if (!result)
    {
        LOG_ERROR("Failed to setup Nemotron-Omni ViT optimization profile");
    }
    return result;
}

bool VisualBuilder::setupGemma4ViTProfile(
    nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    auto const& visionConfig = mModelConfig[kVisionConfigKey];
    int64_t const poolingKernelSize = visionConfig.value("pooling_kernel_size", 3);
    int64_t const patchesPerSoftToken = poolingKernelSize * poolingKernelSize;
    int64_t const minSoftTokens = mBuilderConfig.minImageTokens;
    int64_t const maxSoftTokens = mBuilderConfig.maxImageTokens;
    int64_t const optSoftTokens = std::max<int64_t>(minSoftTokens, (minSoftTokens + maxSoftTokens) / 2);
    int64_t const minPatches = minSoftTokens * patchesPerSoftToken;
    int64_t const optPatches = optSoftTokens * patchesPerSoftToken;
    int64_t const maxPatches = maxSoftTokens * patchesPerSoftToken;

    int64_t inputDim = 0;
    int64_t ropeEmbedSize = 0;
    bool hasMaxSeqLenCarrier = false;
    bool hasKvLengths = false;
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        if (strcmp(input->getName(), binding_names::kVisualInput) == 0)
        {
            inputDim = input->getDimensions().d[1];
        }
        else if (strcmp(input->getName(), binding_names::kRotaryPosEmb) == 0)
        {
            ropeEmbedSize = input->getDimensions().d[1];
        }
        else if (strcmp(input->getName(), binding_names::kMaxSeqLenCarrier) == 0)
        {
            hasMaxSeqLenCarrier = true;
        }
        else if (strcmp(input->getName(), binding_names::kKvLengths) == 0)
        {
            hasKvLengths = true;
        }
    }
    if (inputDim == 0)
    {
        LOG_ERROR("Cannot infer inputDim. Do you have proper ONNX input: %s?", binding_names::kVisualInput);
        return false;
    }
    if (ropeEmbedSize == 0)
    {
        LOG_ERROR("Cannot infer ropeEmbedSize. Do you have proper ONNX input: %s?", binding_names::kRotaryPosEmb);
        return false;
    }

    result &= setOptimizationProfile(&profile, binding_names::kVisualInput, createDims({minPatches, inputDim}),
        createDims({optPatches, inputDim}), createDims({maxPatches, inputDim}));
    result &= setOptimizationProfile(&profile, binding_names::kPixelPositionIds, createDims({minPatches, 2}),
        createDims({optPatches, 2}), createDims({maxPatches, 2}));
    result &= setOptimizationProfile(&profile, binding_names::kRotaryPosEmb, createDims({minPatches, ropeEmbedSize}),
        createDims({optPatches, ropeEmbedSize}), createDims({maxPatches, ropeEmbedSize}));
    result &= setOptimizationProfile(&profile, binding_names::kPoolingWeights, createDims({minSoftTokens, minPatches}),
        createDims({optSoftTokens, optPatches}), createDims({maxSoftTokens, maxPatches}));

    int64_t const maxNumImages = std::max<int64_t>(1, maxSoftTokens / mBuilderConfig.minImageTokens);
    result &= setOptimizationProfile(&profile, binding_names::kCuSeqlens, createDims({2}),
        createDims({maxNumImages + 1}), createDims({maxNumImages + 1}));

    mBuilderConfig.useTrtNativeVitAttn = mModelConfig.value("use_trt_native_vit_attn", false);
    if (mBuilderConfig.useTrtNativeVitAttn)
    {
        if (!hasKvLengths)
        {
            LOG_ERROR("Gemma4 config requests TRT-native ViT attention but ONNX is missing kv_lengths");
            return false;
        }
        result &= setOptimizationProfile(&profile, binding_names::kKvLengths, createDims({2}),
            createDims({maxNumImages + 1}), createDims({maxNumImages + 1}));
    }
    if (hasMaxSeqLenCarrier)
    {
        int64_t const maxSeqLen = mBuilderConfig.maxImageTokensPerImage * patchesPerSoftToken;
        result &= setOptimizationProfile(&profile, binding_names::kMaxSeqLenCarrier, createDims({1}),
            createDims({std::max<int64_t>(1, maxSeqLen / 2)}), createDims({maxSeqLen}));
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup Gemma4 ViT optimization profile");
    }
    return result;
}

bool VisualBuilder::setupGemma4UnifiedVisionProfile(
    nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network)
{
    int64_t inputDim = 0;
    int64_t positionDim = 0;
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto const* input = network.getInput(i);
        if (strcmp(input->getName(), binding_names::kVisualInput) == 0)
        {
            auto const dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                inputDim = dims.d[1];
            }
        }
        else if (strcmp(input->getName(), binding_names::kPixelPositionIds) == 0)
        {
            auto const dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                positionDim = dims.d[1];
            }
        }
    }

    constexpr int64_t kUnifiedPatchDim = 48 * 48 * 3;
    if (inputDim != kUnifiedPatchDim || positionDim != 2)
    {
        LOG_ERROR(
            "Gemma4 Unified visual ONNX inputs must be input [N,%ld] and pixel_position_ids [N,2]; "
            "got trailing dimensions %ld and %ld",
            kUnifiedPatchDim, inputDim, positionDim);
        return false;
    }
    if (mBuilderConfig.minImageTokens < 1 || mBuilderConfig.maxImageTokens < mBuilderConfig.minImageTokens)
    {
        LOG_ERROR("Gemma4 Unified visual profile requires 1 <= minImageTokens <= maxImageTokens");
        return false;
    }

    int64_t const minPatches = mBuilderConfig.minImageTokens;
    int64_t const maxPatches = mBuilderConfig.maxImageTokens;
    int64_t const optPatches = minPatches + (maxPatches - minPatches) / 2;
    bool result = true;
    result &= setOptimizationProfile(&profile, binding_names::kVisualInput, createDims({minPatches, inputDim}),
        createDims({optPatches, inputDim}), createDims({maxPatches, inputDim}));
    result &= setOptimizationProfile(&profile, binding_names::kPixelPositionIds, createDims({minPatches, 2}),
        createDims({optPatches, 2}), createDims({maxPatches, 2}));
    if (!result)
    {
        LOG_ERROR("Failed to setup Gemma4 Unified vision optimization profile");
    }
    return result;
}

bool VisualBuilder::copyConfig()
{
    // Save merged config (original model config + builder config) to engine directory
    if (!saveConfigWithBuilderInfo(mEngineDir, mModelConfig, mBuilderConfig.toJson()))
    {
        LOG_ERROR("Failed to save config to engine directory");
        return false;
    }

    // Copy preprocessor_config.json if it exists (needed for image preprocessing at runtime)
    std::string preprocessorConfigPath = (mOnnxDir / "preprocessor_config.json").string();
    if (std::filesystem::exists(preprocessorConfigPath))
    {
        std::string targetPreprocessorConfigPath = (mEngineDir / "preprocessor_config.json").string();
        std::filesystem::copy(
            preprocessorConfigPath, targetPreprocessorConfigPath, std::filesystem::copy_options::overwrite_existing);
        LOG_INFO("Copied preprocessor config to %s", targetPreprocessorConfigPath.c_str());
    }
    else
    {
        LOG_WARNING("No preprocessor config found at %s", preprocessorConfigPath.c_str());
    }

    return true;
}

} // namespace builder
} // namespace trt_edgellm
