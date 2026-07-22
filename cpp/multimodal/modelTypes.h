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

#include <string>

namespace trt_edgellm
{
namespace multimodal
{

//! Enum for supported multimodal model types
enum class ModelType
{
    QWEN2_VL,                     //!< Qwen2-VL model
    QWEN2_5_VL,                   //!< Qwen2.5-VL model
    QWEN3_VL,                     //!< Qwen3-VL model
    QWEN3_5,                      //!< Qwen3.5 model
    QWEN3_OMNI_AUDIO_ENCODER,     //!< Qwen3-Omni audio encoder (speech-to-embeddings)
    QWEN3_OMNI_VISION_ENCODER,    //!< Qwen3-Omni vision encoder (image-to-embeddings)
    QWEN3_OMNI_CODE2WAV,          //!< Qwen3-Omni Code2Wav vocoder (codes-to-waveform)
    INTERNVL,                     //!< InternVL model
    PHI4MM,                       //!< Phi-4MM model
    GEMMA4_VISION,                //!< Gemma4 vision encoder
    NEMOTRON_OMNI_VISION_ENCODER, //!< Nemotron-Omni vision encoder
    NEMOTRON_OMNI_AUDIO_ENCODER,  //!< Nemotron-Omni audio encoder
    VIT,                          //!< Model-agnostic fixed-shape ViT (VisualFixedInput export)
    UNKNOWN                       //!< Unknown or unsupported model type
};

//! Convert string to ModelType enum
//! @param modelTypeStr String representation of model type
//! @return Corresponding ModelType enum value
inline ModelType stringToModelType(std::string const& modelTypeStr)
{
    if (modelTypeStr == "qwen2_vl" || modelTypeStr == "qwen2_vl_vision")
        return ModelType::QWEN2_VL;
    if (modelTypeStr == "qwen2_5_vl" || modelTypeStr == "qwen2_5_vl_vision")
        return ModelType::QWEN2_5_VL;
    if (modelTypeStr == "qwen3_vl" || modelTypeStr == "qwen3_vl_vision")
        return ModelType::QWEN3_VL;
    if (modelTypeStr == "qwen3_5" || modelTypeStr == "qwen3_5_vision")
        return ModelType::QWEN3_5;
    if (modelTypeStr == "qwen3_omni" || modelTypeStr == "qwen3_omni_thinker" || modelTypeStr == "qwen3_omni_text"
        || modelTypeStr == "qwen3_asr_thinker" || modelTypeStr == "qwen3_omni_audio_encoder")
        return ModelType::QWEN3_OMNI_AUDIO_ENCODER;
    if (modelTypeStr == "qwen3_omni_vision_encoder")
        return ModelType::QWEN3_OMNI_VISION_ENCODER;
    if (modelTypeStr == "qwen3_omni_code2wav" || modelTypeStr == "qwen3_tts_code2wav")
        return ModelType::QWEN3_OMNI_CODE2WAV;
    if (modelTypeStr == "internvl" || modelTypeStr == "internvl_vision")
        return ModelType::INTERNVL;
    if (modelTypeStr == "phi4mm")
        return ModelType::PHI4MM;
    if (modelTypeStr == "gemma4" || modelTypeStr == "gemma4_vision")
        return ModelType::GEMMA4_VISION;
    if (modelTypeStr == "nemotron_omni_vision_encoder")
        return ModelType::NEMOTRON_OMNI_VISION_ENCODER;
    if (modelTypeStr == "parakeet")
        return ModelType::NEMOTRON_OMNI_AUDIO_ENCODER;
    if (modelTypeStr == "vit")
        return ModelType::VIT;
    return ModelType::UNKNOWN;
}

} // namespace multimodal
} // namespace trt_edgellm
