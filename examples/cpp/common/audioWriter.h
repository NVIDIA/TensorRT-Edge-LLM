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

#include <cstdint>
#include <fstream>
#include <string>
#include <tensorrt_edgellm/runtime/audioUtils.h>

//! Default PCM sample rate for Qwen3-Omni TTS output (Hz).
constexpr int32_t kDefaultAudioSampleRate = 24000;

/*!
 * @brief Save audio data to WAV file
 *
 * Saves audio waveform to a standard WAV file format.
 * Supports mono audio with 16-bit PCM encoding.
 *
 * @param filepath Output file path (should end with .wav)
 * @param audio Audio data containing samples and metadata
 * @return True if save succeeded, false otherwise
 */
bool saveAudioToWav(std::string const& filepath, trt_edgellm::rt::audioUtils::AudioData const& audio);

/*!
 * @brief Streaming WAV writer for incremental chunk-based audio output
 *
 * Opens file and writes WAV header on construction. Each appendChunk() call
 * appends PCM samples. finalize() patches the header with correct sizes.
 * Destructor calls finalize() automatically if not called explicitly.
 */
class StreamingAudioWriter
{
public:
    //! @brief Open file and write initial WAV header (sizes set to 0, patched on finalize)
    //! @param filepath Output .wav path
    //! @param sampleRate Audio sample rate (default kDefaultAudioSampleRate)
    //! @return True if file opened successfully
    bool open(std::string const& filepath, int32_t sampleRate = kDefaultAudioSampleRate);

    //! @brief Append one chunk of audio samples
    //! @param audio AudioData with waveform tensor (FP16 or FP32, mono)
    //! @return True if write succeeded
    bool appendChunk(trt_edgellm::rt::audioUtils::AudioData const& audio);

    //! @brief Append one chunk already in the file's own sample format
    //! @param pcm16 Little-endian int16 samples, interleaved when there are several channels, exactly
    //!              as `AudioChunk::pcm16` carries them. A trailing odd byte is dropped rather than
    //!              written, since half a sample would shift every later sample by one byte.
    //! @return True if write succeeded
    bool appendPcm16(std::string const& pcm16);

    //! @brief Set the sample rate and channel count the header declares; finalize() writes them
    void setFormat(int32_t sampleRate, int32_t numChannels);

    //! @brief Patch WAV header with final sizes and close file
    void finalize();

    ~StreamingAudioWriter();

    int64_t totalSamplesWritten() const
    {
        return mTotalSamples;
    }

private:
    std::ofstream mFile;
    int32_t mSampleRate{kDefaultAudioSampleRate};
    int32_t mNumChannels{1};
    int64_t mTotalSamples{0}; //!< int16 values written, across all channels.
    bool mFinalized{false};
};
