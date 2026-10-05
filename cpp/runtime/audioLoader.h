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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{
namespace audio
{

//! PCM container.
//!
//! Always mono, float32 in [-1, 1]; the decoder mixes multi-channel input
//! down. ``sampleRate`` is the rate the samples are at: the decoder resamples
//! to the rate it was asked for, ``audioUtils::wrapPcm`` records the rate the
//! caller states. ``samples`` stays readable for as long as this container
//! holds it: a decoder attaches storage it allocated, ``wrapPcm`` attaches
//! memory the caller keeps valid for the request.
struct AudioPCM
{
    std::shared_ptr<Tensor> samples; //!< ``[N]`` Float, host.
    int32_t sampleRate{16000};
    int32_t numChannels{1};

    //! Sample count in the single channel; zero when no samples are attached.
    int64_t numSamples() const noexcept;
};

//! Load raw audio bytes (wav / mp3 / flac) into mono float32 PCM
//! resampled to ``targetSampleRate``. Internally decodes the container
//! via vendored miniaudio (auto-detects format from magic bytes) and
//! mixes multi-channel to mono.
//!
//! ``http(s)://`` URLs are not handled here — bytes are expected to be a
//! complete container blob already.
//!
//! \param bytes Pointer to the encoded audio bytes.
//! \param size Length of \p bytes in bytes.
//! \param targetSampleRate Output sample rate (e.g. 16000 for Whisper).
//! \param out Populated on success; on failure ``out`` is left in an
//!            unspecified state (do not consume its contents).
//! \return true on success, false on decode failure.
bool loadAudioBytes(uint8_t const* bytes, size_t size, int32_t targetSampleRate, AudioPCM& out);

//! Load a local audio file (wav / mp3 / flac) into mono float32 PCM
//! resampled to ``targetSampleRate``.
//!
//! Used by the CLI / offline path. The HTTP server uses loadAudioBytes.
//!
//! \param path Local filesystem path.
//! \param targetSampleRate Output sample rate.
//! \param out Populated on success; on failure ``out`` is left in an
//!            unspecified state (do not consume its contents).
//! \return true on success, false on open/decode failure.
bool loadAudioFile(std::filesystem::path const& path, int32_t targetSampleRate, AudioPCM& out);

} // namespace audio
} // namespace rt
} // namespace trt_edgellm
