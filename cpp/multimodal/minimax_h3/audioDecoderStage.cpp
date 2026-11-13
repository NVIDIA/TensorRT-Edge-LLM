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

#include "multimodal/minimax_h3/minimaxH3Stages.h"

#include "common/checkMacros.h"
#include "kernels/minimaxH3Kernels/minimaxH3Kernels.h"
#include "profiling/timer.h"
#include "runtime/streaming.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

namespace
{

constexpr char const* kRunnerName = "minimax_h3.audio_decoder";
constexpr char const* kLatentsInput = "latents";
constexpr char const* kAudioPort = "audio";
std::string const kTimerStage = "minimax_h3_audio_decode";
constexpr int32_t kCHANNELS{2};

bool listenerGone(multiStage::StageExecContext const& ctx)
{
    return ctx.audioSink != nullptr && ctx.audioSink->isCancelled();
}

} // namespace

AudioDecoderStage::AudioDecoderStage(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
{
}

void AudioDecoderStage::preallocate()
{
    mEngine = std::make_unique<MiniMaxH3Engine>(mArtifactDir, "audio_decoder", mStream);
    auto const& config = mEngine->config();
    mAudioChannels = config.at("latent_channels");
    mMaxFrames = config.at("max_frames");
    mHopLength = config.at("hop_length");
    mSampleRate = config.at("sampling_rate");
    int64_t const maxSamples = static_cast<int64_t>(mMaxFrames) * mHopLength;
    mLatents = rt::Tensor(
        {kCHANNELS, mAudioChannels, mMaxFrames}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::audio_vae_in");
    mWaveform = rt::Tensor({kCHANNELS, maxSamples}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::waveform");
    mWaveformHost
        = rt::Tensor({kCHANNELS, maxSamples}, rt::DeviceType::kCPU, DataType::kFLOAT, "minimax_h3::waveform_host");
}

bool AudioDecoderStage::admit(multiStage::StageExecContext const&, std::string&)
{
    // The latents' extent is known only once the denoiser has planned the request.
    return true;
}

void AudioDecoderStage::prepare(multiStage::StageExecContext& ctx)
{
    auto const& latents = ctx.in->at(kLatentsInput);
    ELLM_CHECK(latents.shape.size() == 3 && latents.shape[0] == kCHANNELS && latents.shape[1] <= mMaxFrames
            && latents.shape[2] == mAudioChannels,
        "MiniMax-H3 audio decoder takes [2, at most " + std::to_string(mMaxFrames) + ", "
            + std::to_string(mAudioChannels) + "] latents");
    mSamples = 0;
}

multiStage::StageStatus AudioDecoderStage::step(multiStage::StageExecContext& ctx)
{
    if (ctx.slot->abandon.requested() || listenerGone(ctx))
    {
        return multiStage::StageStatus::kDone;
    }
    auto const& latents = ctx.in->at(kLatentsInput);
    auto const frames = static_cast<int32_t>(latents.shape[1]);
    int64_t const samples = static_cast<int64_t>(frames) * mHopLength;
    ELLM_CHECK(mLatents.reshape({kCHANNELS, mAudioChannels, frames}) && mWaveform.reshape({kCHANNELS, samples})
            && mWaveformHost.reshape({kCHANNELS, samples}),
        "Request does not fit the audio decoder buffers");
    {
        TIME_STAGE(kTimerStage, ctx.stream);
        kernel::minimax_h3::audioRowsToLatents(
            static_cast<float const*>(latents.data), mLatents.dataPointer<float>(), frames, mAudioChannels, ctx.stream);
        mEngine->bind("audio_latents", mLatents);
        mEngine->bind("waveform", mWaveform);
        mEngine->enqueue(ctx.stream);
        CUDA_CHECK(cudaMemcpyAsync(mWaveformHost.rawPointer(), mWaveform.rawPointer(),
            static_cast<size_t>(kCHANNELS * samples) * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    mSamples = samples;

    if (ctx.audioSink != nullptr)
    {
        AudioChunk chunk;
        chunk.sampleRate = mSampleRate;
        chunk.numChannels = kCHANNELS;
        chunk.pcm16.resize(static_cast<size_t>(kCHANNELS * samples) * sizeof(int16_t));
        auto* out = reinterpret_cast<int16_t*>(chunk.pcm16.data());
        float const* planar = mWaveformHost.dataPointer<float>();
        for (int64_t sample = 0; sample < samples; ++sample)
        {
            for (int32_t channel = 0; channel < kCHANNELS; ++channel)
            {
                float const value = std::clamp(planar[channel * samples + sample], -1.0F, 1.0F) * 32767.0F;
                out[sample * kCHANNELS + channel] = static_cast<int16_t>(std::lround(value));
            }
        }
        ctx.audioSink->push(std::move(chunk));
    }
    return multiStage::StageStatus::kDone;
}

void AudioDecoderStage::finish(multiStage::StageExecContext& ctx, multiStage::FinishReason)
{
    // An empty final chunk and an explicit end of stream on every path, so a listener never waits on a
    // request that stopped.
    if (ctx.audioSink != nullptr)
    {
        AudioChunk chunk;
        chunk.isFinal = true;
        chunk.sampleRate = mSampleRate;
        chunk.numChannels = kCHANNELS;
        ctx.audioSink->push(std::move(chunk));
        ctx.audioSink->finish();
    }
}

void AudioDecoderStage::reset()
{
    mSamples = 0;
}

nlohmann::json AudioDecoderStage::diagnostics(multiStage::RequestSlot const&) const
{
    return {{"samples", mSamples}, {"sample_rate", mSampleRate}, {"channels", kCHANNELS}};
}

std::unique_ptr<multiStage::IStage> createAudioDecoderStage(multiStage::StageInitContext const& init)
{
    return std::make_unique<AudioDecoderStage>(init);
}

multiStage::RunnerDecl audioDecoderDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kEngine;
    decl.ports = {{kAudioPort, multiStage::PortKind::kAudio}};
    decl.inputs = {kLatentsInput};
    return decl;
}

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
