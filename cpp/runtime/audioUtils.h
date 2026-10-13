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

#include "common/tensor.h"
#include "runtime/audioLoader.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

//! Audio input container. Holds raw mono FP32 PCM (host), either decoded
//! from ``.wav`` / ``.mp3`` / ``.flac`` by ``audioLoader`` or wrapped around
//! samples the caller holds. The audio runner reads ``pcm``, extracts mel
//! internally per its ``audio/config.json``, and consumes the resulting GPU mel.
//!
//! TTS-side output fields (``waveform`` / ``codebookCodes``) are
//! documented inline.
struct AudioData
{
    //! \brief Raw audio waveform: mono FP32, host. Sample rate matches the
    //! runner's MelExtractor expectation (16 kHz for whisper / parakeet).
    std::shared_ptr<rt::audio::AudioPCM> pcm;

    //! \brief Path to pre-computed mel-spectrogram file (e.g. safetensors).
    //! Used by Gemma4 audio runner which bypasses PCM decoding.
    std::string melSpectrogramPath;
    std::string melSpectrogramFormat; //!< Format of mel file (e.g. "safetensors")

    // For audio output: generated waveform
    std::shared_ptr<Tensor> waveform; //!< Waveform samples [1, numSamples], FP16, range [-1, 1], CPU
    int32_t sampleRate{24000};        //!< Sample rate in Hz
    int32_t numChannels{1};           //!< Number of audio channels (typically 1 for mono)

    // For audio output: codebook codes (if waveform generation is not available)
    std::vector<std::vector<int32_t>> codebookCodes; //!< RVQ codebook codes [numCodebooks][seqLen]
    bool hasWaveform{false};                         //!< True if waveform contains valid data
};

//! Decode raw audio bytes (wav / mp3 / flac via miniaudio) into an
//! ``AudioData`` container ready for the audio runner. Wraps
//! ``audio::loadAudioBytes`` + ``AudioData`` field plumbing so callers don't
//! repeat the staging boilerplate. Mirrors ``imageUtils::loadRgbImageFromEncodedBytes``.
//!
//! On failure ``out.pcm`` is left null, including when ``out`` arrived holding an earlier decode.
bool loadAudioDataFromBytes(uint8_t const* bytes, size_t size, int32_t targetSampleRate, AudioData& out);

//! Load a local audio file (wav / mp3 / flac via miniaudio) into an
//! ``AudioData`` container. Mirrors ``imageUtils::loadRgbImageFromFile``.
//!
//! On failure ``out.pcm`` is left null, including when ``out`` arrived holding an earlier decode.
bool loadAudioDataFromFile(std::filesystem::path const& path, int32_t targetSampleRate, AudioData& out);

//! Wrap mono FP32 PCM the caller already holds. Decodes, copies and resamples
//! nothing. Mirrors ``imageUtils::wrapImageBuffer``.
//!
//! The samples shall stay valid and unwritten until ``handleRequest`` returns;
//! page-locked memory lets the online GPU fbank upload transfer asynchronously.
//!
//! \param samples ``[N]`` Float host tensor, owning or not, 1 <= N <= INT32_MAX.
//! \param sampleRate Rate of the samples; the audio runner rejects a clip whose
//!                   rate differs from its own configuration.
//! \throws std::runtime_error if \p samples is not that, or \p sampleRate is not positive.
AudioData wrapPcm(std::shared_ptr<rt::Tensor> samples, int32_t sampleRate);

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
