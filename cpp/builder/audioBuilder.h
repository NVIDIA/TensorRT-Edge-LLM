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

#pragma once

#include "multimodal/common/modelTypes.h"
#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for audio model building.
//! Contains user-specified build parameters for optimization profiles.
//! Model-specific parameters (mel_bins, num_quantizers, etc.) are automatically
//! read from config.json - do NOT specify them here.
struct AudioBuilderConfig
{
    // Audio encoder profile config (used when building audio_encoder)
    int64_t minTimeSteps{100};         //!< Minimum audio time steps
    int64_t maxTimeSteps{6000};        //!< Maximum audio time steps
    bool useTrtNativeAudioAttn{false}; //!< Use TRT IAttention

    // Code2Wav profile config (used when building code2wav)
    int64_t minCodeLen{1};    //!< Minimum code sequence length in frames
    int64_t optCodeLen{300};  //!< Optimal code sequence length (default matches Qwen3-Omni chunked decode)
    int64_t maxCodeLen{2000}; //!< Maximum code sequence length in frames

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const noexcept;

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return AudioBuilderConfig object with parsed parameters
    static AudioBuilderConfig fromJson(Json const& json);

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const;
};

//! Enum for audio model build type (auto-detected from config.json)
enum class AudioBuildType
{
    UNKNOWN,       //!< Not yet determined
    AUDIO_ENCODER, //!< Audio encoder (speech-to-embeddings)
    CODE2WAV,      //!< Code2Wav vocoder (codes-to-waveform)
    RNNT_DECODER   //!< Nemotron-3.5-ASR RNN-T step engine (static shapes, no profile)
};

//! Builder class for audio-related TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for audio encoders and Code2Wav vocoders used in multimodal models (e.g., Qwen3-Omni).
//!
//! Build type is auto-detected from config.json:
//! - If "audio_config" exists → builds audio_encoder.engine
//! - If "code2wav_config" exists → builds code2wav.engine
class AudioBuilder
{
public:
    //! Constructor for AudioBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    //! @throws std::filesystem::filesystem_error if path operations fail
    AudioBuilder(
        std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, AudioBuilderConfig const& config);

    //! Destructor.
    ~AudioBuilder() noexcept = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    //! @throws std::runtime_error if critical build errors occur (file I/O, TensorRT API failures)
    //! @throws nlohmann::json::exception if JSON parsing fails
    bool build();

private:
    std::filesystem::path mOnnxDir;                                   //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir;                                 //!< Directory for saving built engine
    AudioBuilderConfig mBuilderConfig;                                //!< Build configuration
    multimodal::ModelType mModelType{multimodal::ModelType::UNKNOWN}; //!< Model type from config.json
    AudioBuildType mBuildType{AudioBuildType::UNKNOWN};               //!< Build type (auto-detected from config.json)

    //! Parse the model configuration from config.json.
    //! Auto-detects build type and extracts model-specific dimensions.
    //! @return true if parsing was successful, false otherwise
    //! @throws nlohmann::json::exception if JSON parsing fails
    bool parseConfig();

    //! Parse audio encoder specific configuration from audio_config.
    //! @return true if parsing was successful, false otherwise
    bool parseAudioEncoderConfig();

    //! Parse Code2Wav specific configuration from code2wav_config.
    //! @return true if parsing was successful, false otherwise
    bool parseCode2WavConfig();

    //! Parse Nemotron-Omni audio encoder configuration from sound_config.
    //! @return true if parsing was successful, false otherwise
    bool parseNemotronOmniAudioConfig();

    //! Parse Nemotron-3.5-ASR FastConformer encoder configuration from
    //! encoder_config (reads num_mel_bins).
    //! @return true if parsing was successful, false otherwise
    bool parseNemotron35AsrAudioConfig();

    //! Parse encoder-free Gemma4 Unified framed-PCM audio configuration.
    bool parseGemma4UnifiedAudioConfig();

    //! Set up optimization profile for audio encoder.
    //! Creates optimization profile with appropriate dynamic shapes for audio inputs.
    //! @param builder TensorRT builder object (must not be null)
    //! @param config TensorRT builder config object (must not be null)
    //! @param network TensorRT network definition (must not be null)
    //! @return true if setup was successful, false otherwise
    bool setupAudioEncoderProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Set up Qwen3-Omni audio encoder profile.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupQwen3OmniAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up Nemotron-Omni Parakeet audio encoder profile.
    //! Configures inputs: input_features [batch, seq_len, mel_bins] + attention_mask [batch, seq_len].
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupNemotronOmniAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up Nemotron-3.5-ASR FastConformer encoder profile.
    //! Configures inputs: input_features [1, seq_len, mel_bins] (dynamic seq) and
    //! prompt_ids [1] (static language-prompt index).
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupNemotron35AsrEncoderProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up Gemma4 audio encoder profile.
    //! Configures input: input_features [1, seq_len, mel_bins] with subsamplingFactor=4.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupGemma4AudioEncoderProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up Gemma4 Unified input_features [1, num_raw_pcm_frames, 640].
    bool setupGemma4UnifiedAudioEncoderProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up optimization profile for Code2Wav vocoder.
    //! Creates optimization profile with appropriate dynamic shapes for code inputs.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupCode2WavProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Set up Qwen3-Omni Code2Wav profile.
    //! Configures input "codes" with shape [1, num_quantizers, code_len].
    //! @param profile Optimization profile to configure
    //! @param network Network definition for input dimension inference
    //! @return true if setup was successful, false otherwise
    bool setupQwen3OmniCode2WavProfile(
        nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    //! @throws std::runtime_error if file I/O operations fail
    //! @throws nlohmann::json::exception if JSON serialization fails
    bool copyConfig();

    // Model-specific configuration extracted from config.json (initialized to 0, must be read)
    // Audio encoder config (from audio_config)
    int32_t mMelBins{0};         //!< Number of Mel-frequency bins
    int32_t mNWindowDim{0};      //!< Window dimension for feature tensor (n_window * 2)
    int32_t mAudioFeatureDim{0}; //!< Raw PCM samples per frame for Gemma4 Unified

    // Code2Wav config (from code2wav_config)
    int32_t mNumQuantizers{0}; //!< Number of RVQ quantizers
    int32_t mUpsampleRate{0};  //!< Total upsample rate (samples per code frame)

    Json mModelConfig; //!< Parsed model configuration
};

} // namespace builder
} // namespace trt_edgellm
