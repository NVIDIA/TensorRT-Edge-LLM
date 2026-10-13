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

#include "audioLoader.h"

#include "common/logger.h"

#include <utility>
#include <vector>

// miniaudio: single-header library. The implementation lives in this TU only.
// Disable subsystems we don't need (playback / capture / device I/O / null
// backend) to keep object size small. We only use the decoder API.
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace trt_edgellm
{
namespace rt
{
namespace audio
{

namespace
{

//! RAII guard so any early-return from a decode helper cleans up the decoder.
class DecoderGuard
{
public:
    explicit DecoderGuard(ma_decoder& d) noexcept
        : mDecoder(d)
    {
    }
    ~DecoderGuard()
    {
        ma_decoder_uninit(&mDecoder);
    }
    DecoderGuard(DecoderGuard const&) = delete;
    DecoderGuard& operator=(DecoderGuard const&) = delete;

private:
    ma_decoder& mDecoder;
};

//! Longest decoded audio accepted, in seconds. Compressed containers can
//! expand far beyond their byte size (a small FLAC can decode to GBs), so
//! cap the decoded duration rather than the input bytes.
constexpr ma_uint64 kMaxDecodedSeconds = 600;

//! Decoded samples and the tensor that views them. The vector stays pageable: an owning host Tensor
//! would page-lock, and whether the samples are worth page-locking is the caller's decision, made by
//! handing wrapPcm memory of its choosing.
struct DecodedPcm
{
    explicit DecodedPcm(std::vector<float> samples)
        : storage(std::move(samples))
        , view(storage.data(), Coords{static_cast<int64_t>(storage.size())}, DeviceType::kCPU,
              nvinfer1::DataType::kFLOAT, "decodedPcm")
    {
    }

    std::vector<float> storage;
    Tensor view; //!< Non-owning view of ``storage``; valid for as long as this object lives.
};

//! Drain an initialised decoder into ``out``.
//!
//! WAV / MP3 / FLAC headers advertise total frame count, so query it once
//! and do a single allocation + read. Matches the HF / whisper.cpp /
//! dr_wav pattern.
bool drainDecoder(ma_decoder& decoder, int32_t sampleRate, AudioPCM& out)
{
    ma_uint64 totalFrames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &totalFrames) != MA_SUCCESS || totalFrames == 0)
    {
        LOG_ERROR("ma_decoder_get_length_in_pcm_frames returned 0 or failed");
        return false;
    }
    if (totalFrames > kMaxDecodedSeconds * static_cast<ma_uint64>(sampleRate))
    {
        LOG_ERROR("audio decodes to %llu frames (> %llu s at %d Hz); refusing to buffer",
            static_cast<unsigned long long>(totalFrames), static_cast<unsigned long long>(kMaxDecodedSeconds),
            sampleRate);
        return false;
    }
    std::vector<float> samples(static_cast<size_t>(totalFrames));

    ma_uint64 framesRead = 0;
    ma_result const status = ma_decoder_read_pcm_frames(&decoder, samples.data(), totalFrames, &framesRead);
    if (status != MA_SUCCESS && status != MA_AT_END)
    {
        LOG_ERROR("ma_decoder_read_pcm_frames failed: %d", static_cast<int>(status));
        return false;
    }
    // Headers can advertise more frames than the stream delivers.
    samples.resize(static_cast<size_t>(framesRead));

    // Aliasing constructor: the pointer is the view, the ownership is the whole DecodedPcm.
    auto decoded = std::make_shared<DecodedPcm>(std::move(samples));
    out.samples = std::shared_ptr<Tensor>(decoded, &decoded->view);
    return true;
}

//! Build a decoder config that asks miniaudio to deliver mono float32 at
//! ``targetSampleRate``. miniaudio handles the channel mixdown and resample
//! internally on every read.
ma_decoder_config makeConfig(int32_t targetSampleRate)
{
    return ma_decoder_config_init(ma_format_f32, 1 /* mono */, static_cast<ma_uint32>(targetSampleRate));
}

} // namespace

int64_t AudioPCM::numSamples() const noexcept
{
    return samples ? samples->getShape().volume() : 0;
}

bool loadAudioBytes(uint8_t const* bytes, size_t size, int32_t targetSampleRate, AudioPCM& out)
{
    if (bytes == nullptr || size == 0)
    {
        LOG_ERROR("loadAudioBytes: empty input");
        return false;
    }

    ma_decoder_config const config = makeConfig(targetSampleRate);
    ma_decoder decoder;
    ma_result const initStatus = ma_decoder_init_memory(bytes, size, &config, &decoder);
    if (initStatus != MA_SUCCESS)
    {
        LOG_ERROR("ma_decoder_init_memory failed: %d", static_cast<int>(initStatus));
        return false;
    }
    DecoderGuard const guard(decoder);

    if (!drainDecoder(decoder, targetSampleRate, out))
    {
        return false;
    }
    out.sampleRate = targetSampleRate;
    out.numChannels = 1;
    return true;
}

bool loadAudioFile(std::filesystem::path const& path, int32_t targetSampleRate, AudioPCM& out)
{
    if (path.empty())
    {
        LOG_ERROR("loadAudioFile: empty path");
        return false;
    }

    ma_decoder_config const config = makeConfig(targetSampleRate);
    ma_decoder decoder;
    ma_result const initStatus = ma_decoder_init_file(path.string().c_str(), &config, &decoder);
    if (initStatus != MA_SUCCESS)
    {
        LOG_ERROR("ma_decoder_init_file('%s') failed: %d", path.string().c_str(), static_cast<int>(initStatus));
        return false;
    }
    DecoderGuard const guard(decoder);

    if (!drainDecoder(decoder, targetSampleRate, out))
    {
        return false;
    }
    out.sampleRate = targetSampleRate;
    out.numChannels = 1;
    return true;
}

} // namespace audio
} // namespace rt
} // namespace trt_edgellm
