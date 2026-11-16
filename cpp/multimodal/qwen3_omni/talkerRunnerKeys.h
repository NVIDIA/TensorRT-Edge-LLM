/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

/*!
 * @brief What a stage graph calls the talker stage: its runner name, port, inputs, sampling keys, and the config keys
 * of the code predictor and clone encoder it hosts.
 */
namespace talker
{

inline constexpr char const* kRunnerName = "qwen3_omni.talker";
inline constexpr char const* kCodesPort = "codes";

// Inputs. A deployment binds kPromptInput or kTextInput; the rest are optional.
inline constexpr char const* kHiddenInput = "hidden";
inline constexpr char const* kEmbedsInput = "embeds";
inline constexpr char const* kTextInput = "text_ids";
inline constexpr char const* kPromptInput = "prompt_ids";
inline constexpr char const* kVoiceInput = "voice";
inline constexpr char const* kSamplingInput = "sampling";
inline constexpr char const* kSpeakerCodesInput = "speaker_codes";
inline constexpr char const* kStyleInput = "style";
inline constexpr char const* kRefAudioInput = "ref_audio";

// Keys of the sampling input.
inline constexpr char const* kTalkerTemperatureKey = "talker_temperature";
inline constexpr char const* kTalkerTopKKey = "talker_top_k";
inline constexpr char const* kTalkerTopPKey = "talker_top_p";
inline constexpr char const* kRepetitionPenaltyKey = "repetition_penalty";
inline constexpr char const* kCpTemperatureKey = "cp_temperature";
inline constexpr char const* kCpTopKKey = "cp_top_k";
inline constexpr char const* kCpTopPKey = "cp_top_p";
inline constexpr char const* kMaxAudioLengthKey = "max_audio_length";

inline constexpr char const* kCodePredictorArtifactKey = "code_predictor_artifact";
inline constexpr char const* kCloneEncoderArtifactKey = "clone_encoder_artifact";
inline constexpr char const* kCpSpecVerifySizeKey = "cp_spec_verify_size";
inline constexpr char const* kCodePredictorCheckpointKey = "code_predictor_checkpoint";

} // namespace talker
} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
