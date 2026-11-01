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

#include "audioBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

Json AudioBuilderConfig::toJson() const noexcept
{
    Json json;
    // Audio encoder config
    json["min_time_steps"] = minTimeSteps;
    json["max_time_steps"] = maxTimeSteps;
    json["use_trt_native_audio_attn"] = useTrtNativeAudioAttn;
    // Code2Wav config
    json["min_code_len"] = minCodeLen;
    json["opt_code_len"] = optCodeLen;
    json["max_code_len"] = maxCodeLen;
    return json;
}

AudioBuilderConfig AudioBuilderConfig::fromJson(Json const& json)
{
    AudioBuilderConfig config;
    // Audio encoder config
    if (json.contains("min_time_steps"))
    {
        config.minTimeSteps = json["min_time_steps"];
    }
    if (json.contains("max_time_steps"))
    {
        config.maxTimeSteps = json["max_time_steps"];
    }
    if (json.contains("use_trt_native_audio_attn"))
    {
        config.useTrtNativeAudioAttn = json["use_trt_native_audio_attn"];
    }
    // Code2Wav config
    if (json.contains("min_code_len"))
    {
        config.minCodeLen = json["min_code_len"];
    }
    if (json.contains("opt_code_len"))
    {
        config.optCodeLen = json["opt_code_len"];
    }
    if (json.contains("max_code_len"))
    {
        config.maxCodeLen = json["max_code_len"];
    }
    return config;
}

std::string AudioBuilderConfig::toString() const
{
    std::ostringstream oss;
    oss << "AudioBuilderConfig:"
        << " AudioEncoder[min=" << minTimeSteps << ",max=" << maxTimeSteps
        << ",use_trt_native_audio_attn=" << useTrtNativeAudioAttn << "]"
        << "\n  Code2Wav[min=" << minCodeLen << ",opt=" << optCodeLen << ",max=" << maxCodeLen << "]";
    return oss.str();
}

AudioBuilder::AudioBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, AudioBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool AudioBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        LOG_ERROR("Failed to parse model configuration from %s", mOnnxDir.string().c_str());
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Parse ONNX model
    std::string onnxPath = (mOnnxDir / "model.onnx").string();
    auto parser = parseOnnxModel(network.get(), onnxPath);
    if (!parser)
    {
        LOG_ERROR("Failed to parse ONNX model from %s", onnxPath.c_str());
        return false;
    }

    // Determine build type name and engine filename based on auto-detected type
    std::string buildTypeName;
    std::string engineFileName;
    switch (mBuildType)
    {
    case AudioBuildType::AUDIO_ENCODER:
    {
        buildTypeName = "AudioEncoder";
        engineFileName = "audio_encoder.engine";
        break;
    }
    case AudioBuildType::CODE2WAV:
    {
        buildTypeName = "Code2Wav";
        engineFileName = "code2wav.engine";
        break;
    }
    case AudioBuildType::RNNT_DECODER:
    {
        buildTypeName = "RnntDecoder";
        engineFileName = "rnnt_step.engine";
        break;
    }
    default:
    {
        LOG_ERROR("Build type not determined. Check config.json for audio_config or code2wav_config.");
        return false;
    }
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), buildTypeName).c_str());

    // Create builder config
    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config");
        return false;
    }

    // Setup optimization profile based on auto-detected build type
    bool profileSetup = false;
    switch (mBuildType)
    {
    case AudioBuildType::AUDIO_ENCODER:
    {
        profileSetup = setupAudioEncoderProfile(*builder, *config, *network);
        break;
    }
    case AudioBuildType::CODE2WAV:
    {
        profileSetup = setupCode2WavProfile(*builder, *config, *network);
        break;
    }
    case AudioBuildType::RNNT_DECODER:
    {
        // The RNN-T step engine has fully static I/O (single decode step), so
        // it needs no optimization profile.
        profileSetup = true;
        break;
    }
    default:
    {
        break;
    }
    }

    if (!profileSetup)
    {
        LOG_ERROR("Failed to setup audio optimization profile. Check input dimensions and model configuration.");
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
        LOG_INFO("Created directory %s for saving %s engine.", mEngineDir.string().c_str(), buildTypeName.c_str());
    }

    // Build and save engine
    std::string engineFilePath = (mEngineDir / engineFileName).string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        LOG_ERROR("Failed to build and serialize engine to %s", engineFilePath.c_str());
        return false;
    }

    // Copy config
    if (!copyConfig())
    {
        LOG_ERROR("Failed to copy config to engine directory");
        return false;
    }

    LOG_INFO("Successfully built %s engine: %s", buildTypeName.c_str(), engineFilePath.c_str());
    return true;
}

bool AudioBuilder::parseConfig()
{
    std::string configPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(configPath, mModelConfig))
    {
        LOG_ERROR("Failed to load config from %s", configPath.c_str());
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // Read model type (optional for Code2Wav which may not have a model_type)
    std::string modelTypeStr;
    if (mModelConfig.contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kModelTypeKey].get<std::string>();
    }

    if (!modelTypeStr.empty())
    {
        mModelType = multimodal::stringToModelType(modelTypeStr);
    }
    else
    {
        // Default to CODE2WAV when model_type not specified (legacy Code2Wav exports)
        mModelType = multimodal::ModelType::QWEN3_OMNI_CODE2WAV;
        LOG_INFO("model_type not specified, defaulting to QWEN3_OMNI_CODE2WAV");
    }

    // Auto-detect build type from config.json contents
    // Code2Wav config can be either nested in "code2wav_config" or at top-level with "num_quantizers"
    // Nemotron-3.5-ASR names its encoder sub-config "encoder_config"; its RNN-T
    // step engine carries an "rnnt_decoder_config" marker instead.
    bool const hasRnntDecoderConfig = mModelConfig.contains("rnnt_decoder_config");
    bool const hasAudioConfig = mModelConfig.contains("audio_config") || mModelConfig.contains("sound_config")
        || mModelConfig.contains("encoder_config");
    bool const hasCode2WavConfig = mModelConfig.contains("code2wav_config")
        || (mModelConfig.contains("num_quantizers") && mModelConfig.contains("upsample_rates"));

    // Determine build type based on config presence. RNN-T step and Code2Wav
    // markers take priority over the generic encoder sub-config.
    if (hasRnntDecoderConfig)
    {
        mBuildType = AudioBuildType::RNNT_DECODER;
    }
    else if (hasCode2WavConfig)
    {
        mBuildType = AudioBuildType::CODE2WAV;
    }
    else if (hasAudioConfig)
    {
        mBuildType = AudioBuildType::AUDIO_ENCODER;
    }
    else
    {
        mBuildType = AudioBuildType::UNKNOWN;
    }

    // Log and validate build type, and append subdirectory for AudioEncoder
    switch (mBuildType)
    {
    case AudioBuildType::CODE2WAV:
        LOG_INFO("Auto-detected build type: Code2Wav (found code2wav configuration in config.json)");
        mEngineDir = mEngineDir / "code2wav";
        break;
    case AudioBuildType::AUDIO_ENCODER:
        LOG_INFO("Auto-detected build type: AudioEncoder (found audio_config in config.json)");
        mEngineDir = mEngineDir / "audio";
        break;
    case AudioBuildType::RNNT_DECODER:
        LOG_INFO("Auto-detected build type: RNN-T decoder step (found rnnt_decoder_config in config.json)");
        mEngineDir = mEngineDir / "rnnt_decoder";
        break;
    case AudioBuildType::UNKNOWN:
    default:
        LOG_ERROR(
            "Cannot determine build type from config.json. Expected one of: 'audio_config'/'encoder_config' for "
            "an audio encoder, 'num_quantizers'+'upsample_rates' for code2wav, or 'rnnt_decoder_config' for the "
            "RNN-T step engine");
        return false;
    }

    // Parse model-specific configuration based on detected build type
    switch (mBuildType)
    {
    case AudioBuildType::CODE2WAV: return parseCode2WavConfig();
    case AudioBuildType::RNNT_DECODER:
        // Fully static-shape engine; nothing model-specific to read for the build.
        return true;
    case AudioBuildType::AUDIO_ENCODER:
    {
        bool parseOk = false;
        if (mModelType == multimodal::ModelType::GEMMA4_UNIFIED_AUDIO)
        {
            parseOk = parseGemma4UnifiedAudioConfig();
        }
        else if (mModelType == multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER)
        {
            parseOk = parseNemotronOmniAudioConfig();
        }
        else if (mModelType == multimodal::ModelType::NEMOTRON3_5_ASR_AUDIO_ENCODER)
        {
            parseOk = parseNemotron35AsrAudioConfig();
        }
        else
        {
            parseOk = parseAudioEncoderConfig();
        }
        if (parseOk && mModelConfig.value("use_trt_native_audio_attn", false))
        {
            logTrtNativeAttentionPath("AudioAttention");
        }
        return parseOk;
    }
    default: LOG_ERROR("Unknown build type"); return false;
    }
}

bool AudioBuilder::parseGemma4UnifiedAudioConfig()
{
    if (!mModelConfig.contains("audio_config"))
    {
        LOG_ERROR("audio_config not found in config.json for Gemma4 Unified audio");
        return false;
    }
    auto const& audioConfig = mModelConfig["audio_config"];
    mAudioFeatureDim = audioConfig.value("audio_samples_per_token", audioConfig.value("audio_embed_dim", int32_t{0}));
    if (mAudioFeatureDim != 640)
    {
        LOG_ERROR(
            "Gemma4 Unified audio requires audio_samples_per_token/audio_embed_dim=640, got %d", mAudioFeatureDim);
        return false;
    }
    if (mBuilderConfig.maxTimeSteps < 1)
    {
        LOG_ERROR("Gemma4 Unified maxTimeSteps must be at least one raw PCM frame");
        return false;
    }
    LOG_INFO(
        "Gemma4 Unified AudioEncoder config: frame_size=%d samples; maxTimeSteps=%ld means raw 640-sample "
        "PCM frames (minimum profile length is always 1)",
        mAudioFeatureDim, mBuilderConfig.maxTimeSteps);
    return true;
}

bool AudioBuilder::parseAudioEncoderConfig()
{
    auto const& audioConfig = mModelConfig["audio_config"];

    // mel_bins: HF uses "num_mel_bins", also accept "mel_bins" for backward compat
    if (audioConfig.contains("num_mel_bins"))
    {
        mMelBins = audioConfig["num_mel_bins"].get<int32_t>();
    }
    else if (audioConfig.contains("mel_bins"))
    {
        mMelBins = audioConfig["mel_bins"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("audio_config.num_mel_bins (or mel_bins) not found in config.json");
        return false;
    }

    // n_window: required for Qwen3-Omni audio encoder (chunked feature format).
    // Gemma4 and Nemotron-Omni use [1, seq_len, mel_bins] input and don't need n_window.
    if (audioConfig.contains("n_window"))
    {
        // Feature tensor uses n_window * 2 (implementation detail of Qwen3-Omni audio encoder)
        mNWindowDim = audioConfig["n_window"].get<int32_t>() * 2;
    }
    else if (mModelType == multimodal::ModelType::QWEN3_OMNI_AUDIO_ENCODER)
    {
        LOG_ERROR("audio_config.n_window not found in config.json (required for Qwen3-Omni)");
        return false;
    }

    LOG_INFO("AudioEncoder config: mel_bins=%d, n_window_dim=%d", mMelBins, mNWindowDim);
    return true;
}

bool AudioBuilder::parseCode2WavConfig()
{
    // Code2Wav config can be nested in "code2wav_config" or at top-level
    // Support both formats for flexibility
    Json const& cfg = mModelConfig.contains("code2wav_config") ? mModelConfig["code2wav_config"] : mModelConfig;

    // num_quantizers must exist
    if (!cfg.contains("num_quantizers"))
    {
        LOG_ERROR("num_quantizers not found in config.json");
        return false;
    }
    mNumQuantizers = cfg["num_quantizers"].get<int32_t>();

    // Calculate total upsample rate from upsample_rates and upsampling_ratios
    // These arrays define the decoder's upsampling factors
    int64_t rate = 1;
    if (cfg.contains("upsample_rates"))
    {
        for (auto const& r : cfg["upsample_rates"])
        {
            rate *= r.get<int64_t>();
        }
    }
    else
    {
        LOG_ERROR("upsample_rates not found in config.json");
        return false;
    }

    if (cfg.contains("upsampling_ratios"))
    {
        for (auto const& r : cfg["upsampling_ratios"])
        {
            rate *= r.get<int64_t>();
        }
    }
    else
    {
        LOG_ERROR("upsampling_ratios not found in config.json");
        return false;
    }
    mUpsampleRate = static_cast<int32_t>(rate);

    LOG_INFO("Code2Wav config: num_quantizers=%d, upsample_rate=%d", mNumQuantizers, mUpsampleRate);
    return true;
}

bool AudioBuilder::parseNemotronOmniAudioConfig()
{
    // Nemotron-Omni uses sound_config instead of audio_config
    auto const& soundConfig = mModelConfig["sound_config"];

    if (!soundConfig.contains("num_mel_bins"))
    {
        LOG_ERROR("sound_config.num_mel_bins not found in config.json");
        return false;
    }
    mMelBins = soundConfig["num_mel_bins"].get<int32_t>();

    LOG_INFO("Nemotron-Omni AudioEncoder config: mel_bins=%d", mMelBins);
    return true;
}

bool AudioBuilder::parseNemotron35AsrAudioConfig()
{
    // Nemotron-3.5-ASR names its encoder sub-config "encoder_config".
    if (!mModelConfig.contains("encoder_config"))
    {
        LOG_ERROR("encoder_config not found in config.json for Nemotron-3.5-ASR audio encoder");
        return false;
    }
    auto const& encoderConfig = mModelConfig["encoder_config"];
    mMelBins = encoderConfig.value("num_mel_bins", int32_t{128});

    LOG_INFO("Nemotron-3.5-ASR AudioEncoder config: mel_bins=%d", mMelBins);
    return true;
}

bool AudioBuilder::setupAudioEncoderProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    LOG_INFO("Setting up optimization profile for Audio Encoder");

    auto* profile = builder.createOptimizationProfile();
    bool result = true;

    // Dispatch to model-specific setup based on model type
    switch (mModelType)
    {
    case multimodal::ModelType::QWEN3_OMNI_AUDIO_ENCODER:
    case multimodal::ModelType::QWEN3_OMNI_NEXT_AUDIO_ENCODER:
        result = setupQwen3OmniAudioEncoderProfile(*profile);
        break;
    case multimodal::ModelType::GEMMA4_UNIFIED_AUDIO: result = setupGemma4UnifiedAudioEncoderProfile(*profile); break;
    case multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER:
        result = setupNemotronOmniAudioEncoderProfile(*profile);
        break;
    case multimodal::ModelType::NEMOTRON3_5_ASR_AUDIO_ENCODER:
        result = setupNemotron35AsrEncoderProfile(*profile);
        break;
    case multimodal::ModelType::GEMMA4_AUDIO_ENCODER: result = setupGemma4AudioEncoderProfile(*profile); break;
    default: LOG_ERROR("Unsupported model type for audio encoder: %d", static_cast<int>(mModelType)); return false;
    }

    if (!result)
    {
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(profile, "audio_encoder_profile", &network).c_str());
    config.addOptimizationProfile(profile);
    return true;
}

bool AudioBuilder::setupGemma4UnifiedAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile)
{
    constexpr int64_t kBatch = 1;
    constexpr int64_t kMinFrames = 1;
    int64_t const maxFrames = mBuilderConfig.maxTimeSteps;
    int64_t const optFrames = kMinFrames + (maxFrames - kMinFrames) / 2;

    bool const result = setOptimizationProfile(&profile, binding_names::kAudioInputFeatures,
        createDims({kBatch, kMinFrames, mAudioFeatureDim}), createDims({kBatch, optFrames, mAudioFeatureDim}),
        createDims({kBatch, maxFrames, mAudioFeatureDim}));
    if (!result)
    {
        LOG_ERROR("Failed to setup Gemma4 Unified audio encoder profile");
    }
    return result;
}

bool AudioBuilder::setupQwen3OmniAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Calculate chunk dimensions using ceiling division
    int64_t minChunks = static_cast<int64_t>(divUp(mBuilderConfig.minTimeSteps, mNWindowDim));
    int64_t maxChunks = static_cast<int64_t>(divUp(mBuilderConfig.maxTimeSteps, mNWindowDim));
    int64_t optChunks = (minChunks + maxChunks) / 2;

    // Calculate attention elements for Qwen3-Omni audio encoder
    // Each full chunk has 100 frames -> 13 frames after CNN (3 Conv layers with stride=2).
    // But it's possible for a tail chunk in a batch to have only 1 frame after CNN.
    // In multi-batch case, minElems = minChunks since each chunk can be its own batch
    // with only 1 frame after CNN. So minElems=minChunks is the safe lower bound.
    constexpr int64_t kAttentionElementsPerChunk = 13;
    int64_t minElems = minChunks; // Safe lower bound for multi-batch case
    int64_t maxElems = maxChunks * kAttentionElementsPerChunk;
    int64_t optElems = (minElems + maxElems) / 2;

    // Qwen3-Omni audio encoder inputs:
    //   padded_feature: [num_chunks, mel_bins, n_window_dim]
    //   padded_mask_after_cnn_indices: [num_attention_elems, 2]
    //   attention_mask: [num_attention_elems, num_attention_elems]
    result &= setOptimizationProfile(&profile, "padded_feature", createDims({minChunks, mMelBins, mNWindowDim}),
        createDims({optChunks, mMelBins, mNWindowDim}), createDims({maxChunks, mMelBins, mNWindowDim}));

    result &= setOptimizationProfile(&profile, "padded_mask_after_cnn_indices", createDims({minElems, 2}),
        createDims({optElems, 2}), createDims({maxElems, 2}));

    result &= setOptimizationProfile(&profile, "attention_mask", createDims({minElems, minElems}),
        createDims({optElems, optElems}), createDims({maxElems, maxElems}));

    mBuilderConfig.useTrtNativeAudioAttn = mModelConfig.value("use_trt_native_audio_attn", false);
    if (mBuilderConfig.useTrtNativeAudioAttn)
    {
        int64_t const maxBatchSize = std::max<int64_t>(1, mModelConfig.value("max_batch_size", 16));
        result &= setOptimizationProfile(
            &profile, "cu_seqlens", createDims({2}), createDims({2}), createDims({maxBatchSize + 1}));
        result &= setOptimizationProfile(
            &profile, "kv_lengths", createDims({2}), createDims({2}), createDims({maxBatchSize + 1}));
    }
    if (!result)
    {
        LOG_ERROR("Failed to setup Qwen3-Omni audio encoder profile");
    }
    return result;
}

bool AudioBuilder::setupCode2WavProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* profile = builder.createOptimizationProfile();
    bool result = true;

    // Dispatch to model-specific setup based on model type
    switch (mModelType)
    {
    case multimodal::ModelType::QWEN3_OMNI_CODE2WAV:
    {
        result = setupQwen3OmniCode2WavProfile(*profile, network);
        break;
    }
    default:
    {
        LOG_ERROR("Unsupported model type for Code2Wav: %d", static_cast<int>(mModelType));
        return false;
    }
    }

    if (!result)
    {
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(profile, "code2wav_profile", &network).c_str());
    config.addOptimizationProfile(profile);
    return true;
}

bool AudioBuilder::setupQwen3OmniCode2WavProfile(
    nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Infer num_quantizers from ONNX network input shape for validation
    int32_t networkNumQuantizers = 0;
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        if (strcmp(input->getName(), "codes") == 0)
        {
            // codes shape: [batch, num_quantizers, code_len]
            networkNumQuantizers = input->getDimensions().d[1];
            break;
        }
    }

    if (networkNumQuantizers > 0 && networkNumQuantizers != mNumQuantizers)
    {
        LOG_WARNING("ONNX network num_quantizers (%d) differs from config (%d). Using network value.",
            networkNumQuantizers, mNumQuantizers);
        mNumQuantizers = networkNumQuantizers;
    }

    if (mNumQuantizers <= 0)
    {
        LOG_ERROR("Cannot determine num_quantizers. Check config.json and ONNX model.");
        return false;
    }

    // Input: codes [batch, num_quantizers, code_len]
    // - batch: 1 (single audio generation)
    // - num_quantizers: from config (e.g., 16 for Qwen3-Omni)
    // - code_len: variable, supports dynamic length via CausalConvNet auto-padding
    //   Audio duration = code_len * (upsample_rate / sample_rate)
    //   For Qwen3-Omni (24kHz, upsample_rate=1920): 1 frame = 80ms
    //   Runtime uses chunked inference with chunk_size=300 for memory efficiency
    constexpr int64_t kBatch = 1;
    int64_t const numQuantizers = static_cast<int64_t>(mNumQuantizers);

    result &= setOptimizationProfile(&profile, "codes", createDims({kBatch, numQuantizers, mBuilderConfig.minCodeLen}),
        createDims({kBatch, numQuantizers, mBuilderConfig.optCodeLen}),
        createDims({kBatch, numQuantizers, mBuilderConfig.maxCodeLen}));

    if (!result)
    {
        LOG_ERROR("Failed to setup Qwen3-Omni Code2Wav profile");
    }
    return result;
}

bool AudioBuilder::setupNemotronOmniAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Nemotron-Omni Parakeet audio encoder runs at batch=1: the C++ runtime
    // encodes each audio clip in a request batch sequentially. Cross-clip
    // batching with padding would corrupt the Conformer's depthwise conv on
    // short-clip boundaries (the local kernel reads padded zeros from the
    // longer batchmate's tail).
    //
    // Inputs:
    //   input_features: [1, seq_len, mel_bins]
    // seq_len is mel-spectrogram frames (~16 kHz / 160 hop_length); the
    // 3× stride-2 subsampling requires divisibility by 8.

    constexpr int64_t kSubFactor = 8;
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const minSeqLen = alignUp(mBuilderConfig.minTimeSteps, kSubFactor);
    int64_t const maxSeqLen = alignUp(mBuilderConfig.maxTimeSteps, kSubFactor);
    int64_t const optSeqLen = alignUp((minSeqLen + maxSeqLen) / 2, kSubFactor);
    constexpr int64_t kBatch = 1;

    result &= setOptimizationProfile(&profile, "input_features", createDims({kBatch, minSeqLen, mMelBins}),
        createDims({kBatch, optSeqLen, mMelBins}), createDims({kBatch, maxSeqLen, mMelBins}));

    if (!result)
    {
        LOG_ERROR("Failed to setup Nemotron-Omni audio encoder profile");
    }
    return result;
}

bool AudioBuilder::setupNemotron35AsrEncoderProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Nemotron-3.5-ASR FastConformer encoder runs at batch=1 (same rationale as
    // Nemotron-Omni: the depthwise conv is a local operator, so cross-clip
    // batching with padding corrupts short-clip boundaries).
    //
    // Inputs:
    //   input_features: [1, seq_len, mel_bins]  (mel frames; dynamic)
    //   prompt_ids:     [1]                       (language-prompt index; static)
    // seq_len is mel-spectrogram frames; the 3× stride-2 causal subsampling
    // requires divisibility by 8.

    constexpr int64_t kSubFactor = 8;
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const minSeqLen = alignUp(mBuilderConfig.minTimeSteps, kSubFactor);
    int64_t const maxSeqLen = alignUp(mBuilderConfig.maxTimeSteps, kSubFactor);
    int64_t const optSeqLen = alignUp((minSeqLen + maxSeqLen) / 2, kSubFactor);
    constexpr int64_t kBatch = 1;

    result &= setOptimizationProfile(&profile, "input_features", createDims({kBatch, minSeqLen, mMelBins}),
        createDims({kBatch, optSeqLen, mMelBins}), createDims({kBatch, maxSeqLen, mMelBins}));

    // prompt_ids is a fixed [1] index: min == opt == max.
    result &= setOptimizationProfile(
        &profile, "prompt_ids", createDims({kBatch}), createDims({kBatch}), createDims({kBatch}));

    if (!result)
    {
        LOG_ERROR("Failed to setup Nemotron-3.5-ASR audio encoder profile");
    }
    return result;
}

bool AudioBuilder::setupGemma4AudioEncoderProfile(nvinfer1::IOptimizationProfile& profile)
{
    bool result = true;

    // Gemma4 audio encoder runs at batch=1 (same as Nemotron-Omni): chunked
    // local attention and depthwise convolution are local operators, so
    // cross-clip batching with padding may corrupt outputs.
    //
    // Inputs:
    //   input_features: [1, seq_len, mel_bins]
    //   input_features_mask: [1, seq_len]       (bool mask before subsampling)
    //
    // seq_len is mel-spectrogram frames (~16 kHz / 160 hop_length); the
    // 2× stride-2 subsampling requires divisibility by 4.

    constexpr int64_t kSubFactor = 4;
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const minSeqLen = alignUp(mBuilderConfig.minTimeSteps, kSubFactor);
    int64_t const maxSeqLen = alignUp(mBuilderConfig.maxTimeSteps, kSubFactor);
    int64_t const optSeqLen = alignUp((minSeqLen + maxSeqLen) / 2, kSubFactor);
    constexpr int64_t kBatch = 1;

    result &= setOptimizationProfile(&profile, "input_features", createDims({kBatch, minSeqLen, mMelBins}),
        createDims({kBatch, optSeqLen, mMelBins}), createDims({kBatch, maxSeqLen, mMelBins}));

    result &= setOptimizationProfile(&profile, "input_features_mask", createDims({kBatch, minSeqLen}),
        createDims({kBatch, optSeqLen}), createDims({kBatch, maxSeqLen}));

    if (!result)
    {
        LOG_ERROR("Failed to setup Gemma4 audio encoder profile");
    }
    return result;
}

bool AudioBuilder::copyConfig()
{
    return saveConfigWithBuilderInfo(mEngineDir, mModelConfig, mBuilderConfig.toJson());
}

} // namespace builder
} // namespace trt_edgellm
