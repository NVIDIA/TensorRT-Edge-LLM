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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/wfmRuntimeUtils.h"

#include "common/logger.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace trt_edgellm
{
namespace rt
{
namespace
{
std::vector<int32_t> readIntVector(nlohmann::json const& value)
{
    std::vector<int32_t> out;
    if (!value.is_array())
    {
        return out;
    }
    out.reserve(value.size());
    for (auto const& item : value)
    {
        out.push_back(item.get<int32_t>());
    }
    return out;
}

bool videoShapeMatchesConfig(VideoBuffer const& video, CosmosEngineConfig const& config) noexcept
{
    if (!video.buffer)
    {
        LOG_ERROR("WFM request missing pixel buffer.");
        return false;
    }

    if (video.batch != 1)
    {
        LOG_ERROR(
            "Cosmos Edge runtime currently supports batch=1 (got batch=%lld).", static_cast<long long>(video.batch));
        return false;
    }

    if (video.channels != 3)
    {
        LOG_ERROR("Cosmos Edge expects 3 RGB channels (got channels=%lld).", static_cast<long long>(video.channels));
        return false;
    }

    if (video.numFrames != config.numFrames || video.height != config.height || video.width != config.width)
    {
        LOG_ERROR("Video shape [1,3,%lld,%lld,%lld] does not match export config [1,3,%d,%d,%d].",
            static_cast<long long>(video.numFrames), static_cast<long long>(video.height),
            static_cast<long long>(video.width), config.numFrames, config.height, config.width);
        return false;
    }

    auto const shape = video.buffer->getShape();
    if (shape.getNumDims() == 5)
    {
        // [B, C, T, H, W]
        bool const ok = shape[0] == video.batch && shape[1] == video.channels && shape[2] == video.numFrames
            && shape[3] == video.height && shape[4] == video.width;
        if (!ok)
        {
            LOG_ERROR("Pixel tensor shape does not match VideoBuffer metadata.");
            return false;
        }
    }
    else
    {
        LOG_ERROR("Pixel tensor must be 5D [B, C, T, H, W]; got %d dims.", shape.getNumDims());
        return false;
    }

    return true;
}

bool audioShapeMatchesConfig(AudioBuffer const& audio, CosmosEngineConfig const& config) noexcept
{
    if (!audio.buffer)
    {
        return true;
    }

    if (audio.batch != 1)
    {
        LOG_ERROR(
            "Cosmos sound runtime currently supports batch=1 (got batch=%lld).", static_cast<long long>(audio.batch));
        return false;
    }

    if (audio.sampleRate != config.sampleRate)
    {
        LOG_ERROR(
            "Audio sample rate %d does not match export config sample_rate=%d.", audio.sampleRate, config.sampleRate);
        return false;
    }

    auto const shape = audio.buffer->getShape();
    if (shape.getNumDims() == 3)
    {
        bool const ok = shape[0] == audio.batch && shape[2] == audio.numSamples;
        if (!ok)
        {
            LOG_ERROR("Waveform tensor shape does not match AudioBuffer metadata.");
            return false;
        }
    }
    else if (shape.getNumDims() == 2)
    {
        bool const ok = shape[0] == audio.batch && shape[1] == audio.numSamples;
        if (!ok)
        {
            LOG_ERROR("Waveform tensor shape does not match AudioBuffer metadata.");
            return false;
        }
    }
    else
    {
        LOG_ERROR("Waveform tensor must be 2D [B,T] or 3D [B,1,T]; got %d dims.", shape.getNumDims());
        return false;
    }

    return true;
}

} // namespace

CosmosEngineConfig loadCosmosEngineConfig(std::filesystem::path const& configPath)
{
    std::ifstream configFile(configPath);
    if (!configFile)
    {
        throw std::runtime_error("Failed to open Cosmos config: " + configPath.string());
    }

    nlohmann::json configJson;
    configFile >> configJson;

    CosmosEngineConfig config;
    config.numFrames = configJson.value("num_frames", 0);
    config.height = configJson.value("height", 0);
    config.width = configJson.value("width", 0);
    config.fps = configJson.value("fps", 24.F);
    config.hiddenSize = configJson.value("hidden_size", 0);
    config.numInferenceSteps = configJson.value("num_inference_steps", 35);
    config.maxBatchSize = configJson.value("max_batch_size", 1);
    config.numTrainTimesteps = configJson.value("num_train_timesteps", 1000);
    config.flowShift = configJson.value("flow_shift", 5.F);
    config.schedulerSolverOrder = configJson.value("scheduler_solver_order", 2);
    config.headDim = configJson.value("head_dim", 128);
    config.ropeTheta = configJson.value("rope_theta", 1000000.F);
    config.enableFpsModulation = configJson.value("enable_fps_modulation", false);
    config.unified3dMropeTemporalModalityMargin = configJson.value("unified_3d_mrope_temporal_modality_margin", 15000);
    config.unified3dMropeResetSpatialIds = configJson.value("unified_3d_mrope_reset_spatial_ids", true);
    config.baseFps = configJson.value("base_fps", 24.F);
    config.temporalCompressionFactor = configJson.value("temporal_compression_factor", 4);
    config.visionStartToken = configJson.value("vision_start_token", std::string{"<|vision_start|>"});
    config.useChatTemplate = configJson.value("use_chat_template", true);
    config.useSystemPrompt = configJson.value("use_system_prompt", true);
    config.systemPromptVideo = configJson.value(
        "system_prompt_video", std::string{"You are a helpful assistant who will generate videos from a give prompt."});
    config.enableSound = configJson.value("enable_sound", false);
    config.soundDim = configJson.value("sound_dim", 64);
    config.sampleRate = configJson.value("sample_rate", 48000);
    config.soundLatentFps = configJson.value("sound_latent_fps", 25.F);

    if (configJson.contains("mrope_section") && configJson.at("mrope_section").is_array())
    {
        for (std::size_t i = 0; i < 3U && i < configJson.at("mrope_section").size(); ++i)
        {
            config.mropeSection[i] = configJson.at("mrope_section").at(i).get<int32_t>();
        }
    }

    if (config.numFrames <= 0 || config.height <= 0 || config.width <= 0 || config.hiddenSize <= 0)
    {
        throw std::runtime_error("Invalid Cosmos config in " + configPath.string());
    }

    return config;
}

CosmosPackedStatic loadCosmosPackedStatic(std::filesystem::path const& packingPath)
{
    std::ifstream packingFile(packingPath);
    if (!packingFile)
    {
        throw std::runtime_error("Failed to open Cosmos packing_static: " + packingPath.string());
    }

    nlohmann::json packingJson;
    packingFile >> packingJson;

    CosmosPackedStatic packed;
    packed.undLen = packingJson.value("und_len", 0);
    packed.sequenceLength = packingJson.value("sequence_length", 0);
    packed.numNoisyTokens = packingJson.value("num_noisy_tokens", 0);
    packed.inputIds = readIntVector(packingJson["input_ids"]);
    packed.textIndexes = readIntVector(packingJson["text_indexes"]);
    packed.positionIds = readIntVector(packingJson["position_ids"]);
    packed.visionSequenceIndexes = readIntVector(packingJson["vision_sequence_indexes"]);
    packed.visionMseLossIndexes = readIntVector(packingJson["vision_mse_loss_indexes"]);
    packed.visionNoisyFrameIndexes = readIntVector(packingJson["vision_noisy_frame_indexes"]);
    packed.visionTokenShape = readIntVector(packingJson["vision_token_shapes"].at(0));

    if (packingJson.contains("sound_token_shapes") && packingJson.at("sound_token_shapes").is_array()
        && !packingJson.at("sound_token_shapes").empty())
    {
        packed.soundTokenShape = readIntVector(packingJson["sound_token_shapes"].at(0));
    }
    if (packingJson.contains("sound_sequence_indexes"))
    {
        packed.soundSequenceIndexes = readIntVector(packingJson["sound_sequence_indexes"]);
    }
    if (packingJson.contains("sound_mse_loss_indexes"))
    {
        packed.soundMseLossIndexes = readIntVector(packingJson["sound_mse_loss_indexes"]);
    }
    if (packingJson.contains("sound_noisy_slot_indexes"))
    {
        packed.soundNoisySlotIndexes = readIntVector(packingJson["sound_noisy_slot_indexes"]);
    }

    if (packed.sequenceLength <= 0 || packed.visionTokenShape.size() != 3U)
    {
        throw std::runtime_error("Invalid packing_static contents in " + packingPath.string());
    }

    return packed;
}

bool examineWFMRequest(
    WFMGenerationRequest const& request, CosmosEngineConfig const& config, CosmosPackedStatic const& packed) noexcept
{
    if (request.prompt.empty())
    {
        LOG_ERROR("WFM request prompt is empty.");
        return false;
    }

    int32_t const steps = request.numInferenceSteps > 0 ? request.numInferenceSteps : config.numInferenceSteps;
    if (steps <= 0)
    {
        LOG_ERROR("WFM request numInferenceSteps must be > 0.");
        return false;
    }

    if (!videoShapeMatchesConfig(request.pixels, config))
    {
        return false;
    }

    if (request.inputWaveform.buffer && !audioShapeMatchesConfig(request.inputWaveform, config))
    {
        return false;
    }

    if (request.generateSound && !config.enableSound)
    {
        LOG_ERROR("WFM request generateSound=true but config enable_sound=false.");
        return false;
    }

    if (packed.undLen <= 0 || packed.sequenceLength <= packed.undLen)
    {
        LOG_ERROR("packing_static has invalid und_len=%d sequence_length=%d.", packed.undLen, packed.sequenceLength);
        return false;
    }

    LOG_INFO(
        "WFM request ok: prompt_len=%zu, pixels=[1,3,%d,%d,%d], generate_sound=%s, steps=%d, und_len=%d, seq_len=%d",
        request.prompt.size(), config.numFrames, config.height, config.width, request.generateSound ? "yes" : "no",
        steps, packed.undLen, packed.sequenceLength);

    return true;
}

} // namespace rt
} // namespace trt_edgellm
