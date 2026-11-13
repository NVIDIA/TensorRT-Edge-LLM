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

#include <cmath>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "builder/audioBuilder.h"
#include "builder/llmBuilder.h"
#include "builder/visualBuilder.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "multimodal/qwen3_omni/code2WavRunner.h"
#include "profiling/metrics.h"
#include "runtime/audioLoader.h"
#include "runtime/audioUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/melSpectrogram.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiStage/multiStageRuntime.h"
#include "runtime/state/sharedResources.h"
#ifdef EDGELLM_ENABLE_NEMOTRON_ASR
#include "runtime/nemotronAsrRuntime.h"
#endif
#include "runtime/qwen3OmniTTSRuntime.h"
#include "runtime/streaming.h"
#include "scheduler/requestEngine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = nanobind;

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

//! RAII wrapper for CUDA stream management.
class CudaStreamWrapper
{
public:
    CudaStreamWrapper()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CudaStreamWrapper()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    CudaStreamWrapper(CudaStreamWrapper const&) = delete;
    CudaStreamWrapper& operator=(CudaStreamWrapper const&) = delete;

    CudaStreamWrapper(CudaStreamWrapper&& other) noexcept
        : mStream(other.mStream)
    {
        other.mStream = nullptr;
    }

    CudaStreamWrapper& operator=(CudaStreamWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (mStream != nullptr)
            {
                cudaStreamDestroy(mStream);
            }
            mStream = other.mStream;
            other.mStream = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

//! Talker / streaming knobs for one Omni audio request. Defaults match
//! examples/cpp/llm_inference.cpp.
struct OmniAudioParams
{
    float talkerTemperature{0.9f};
    int32_t talkerTopK{50};
    float talkerTopP{1.0f};
    float repetitionPenalty{1.05f};
    int32_t maxAudioLength{4096};
    std::string speakerName{""};
    int32_t codecChunkFrames{10};      //!< Vocode every N Talker frames.
    int32_t talkerPrefillThreshold{4}; //!< Thinker tokens before Talker prefill.
};

//! Convert a CPU-resident FP16/FP32 waveform tensor ([-1, 1]) to int16 PCM bytes.
std::string waveformToPcm16(rt::Tensor const& waveform)
{
    constexpr float kPcm16MaxAmplitude = 32767.0f;
    ELLM_CHECK(waveform.getDeviceType() == rt::DeviceType::kCPU, "waveform must be CPU-resident");
    int64_t const numDims = waveform.getShape().getNumDims();
    int64_t const numSamples = waveform.getShape()[numDims - 1];
    bool const isFP32 = (waveform.getDataType() == trt_edgellm::DataType::kFLOAT);
    float const* fp32Data = isFP32 ? static_cast<float const*>(waveform.rawPointer()) : nullptr;
    __half const* fp16Data = isFP32 ? nullptr : static_cast<__half const*>(waveform.rawPointer());

    std::string out(static_cast<size_t>(numSamples) * sizeof(int16_t), '\0');
    auto* dst = reinterpret_cast<int16_t*>(out.data());
    for (int64_t i = 0; i < numSamples; ++i)
    {
        float const sample = isFP32 ? fp32Data[i] : __half2float(fp16Data[i]);
        float const clamped = std::max(-1.0f, std::min(1.0f, sample));
        dst[i] = static_cast<int16_t>(clamped * kPcm16MaxAmplitude);
    }
    return out;
}

//! Build the RVQ-chunk callback shared by the Omni streaming and standalone
//! TTS paths: transpose to layer-major, vocode, push PCM. `vocodeFailed` must
//! outlive the generation call; once set, later chunks are skipped and the
//! caller raises after generation returns.
Qwen3OmniTTSRuntime::AudioChunkCallback makeChunkVocodeCallback(Code2WavRunner& code2wav, cudaStream_t stream,
    std::shared_ptr<AudioStreamChannel> const& audioChannel, bool& vocodeFailed)
{
    return [&code2wav, stream, audioChannel, &vocodeFailed](
               std::vector<std::vector<int32_t>> const& chunkCodes, bool isFinal) {
        AudioChunk chunk;
        chunk.isFinal = isFinal;
        bool const skip = audioChannel->isCancelled() || vocodeFailed;
        if (!skip && !chunkCodes.empty() && !chunkCodes[0].empty())
        {
            size_t const numFrames = chunkCodes.size();
            size_t const numLayers = chunkCodes[0].size();
            std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
            for (size_t f = 0; f < numFrames; ++f)
            {
                for (size_t l = 0; l < numLayers; ++l)
                {
                    transposed[l][f] = chunkCodes[f][l];
                }
            }
            audioUtils::AudioData chunkAudio;
            if (code2wav.generateWaveform(transposed, chunkAudio, stream) && chunkAudio.hasWaveform)
            {
                chunk.pcm16 = waveformToPcm16(*chunkAudio.waveform);
                chunk.numFrames = static_cast<int32_t>(numFrames);
            }
            else
            {
                LOG_ERROR("Code2Wav vocoding failed on a %zu-frame chunk; aborting audio stream", numFrames);
                vocodeFailed = true;
            }
        }
        audioChannel->push(std::move(chunk));
        if (isFinal)
        {
            audioChannel->finish();
        }
    };
}

//! Run one standalone TTS request (Qwen3-TTS-style: text → Talker →
//! CodePredictor → Code2Wav → channel, no Thinker pass). Returns the number
//! of codec frames generated. Shared by PyLLMRuntime and PyTTSRuntime.
int32_t runStandaloneTTS(Qwen3OmniTTSRuntime& ttsRuntime, Code2WavRunner& code2wav, cudaStream_t stream,
    std::string const& text, OmniAudioParams const& params, std::shared_ptr<AudioStreamChannel> const& audioChannel)
{
    ELLM_CHECK(audioChannel != nullptr, "audio_channel must not be null");
    ELLM_CHECK(!audioChannel->isFinished(), "audio_channel is already finished");
    ELLM_CHECK(params.codecChunkFrames > 0, "codec_chunk_frames must be > 0");

    Qwen3OmniTTSRuntime::TalkerGenerationRequest talkerReq;
    talkerReq.maxAudioLength = params.maxAudioLength;
    talkerReq.talkerTemperature = params.talkerTemperature;
    talkerReq.talkerTopK = params.talkerTopK;
    talkerReq.talkerTopP = params.talkerTopP;
    talkerReq.repetitionPenalty = params.repetitionPenalty;
    talkerReq.speakerName = params.speakerName;
    // TTS reads the text as assistant speech (see docs/…/tts.md input format).
    Message message;
    message.role = "assistant";
    message.contents.push_back({"text", text});
    talkerReq.messages.push_back(std::move(message));
    talkerReq.streamingChunkFrames = params.codecChunkFrames;
    bool vocodeFailed = false;
    talkerReq.onChunkReady = makeChunkVocodeCallback(code2wav, stream, audioChannel, vocodeFailed);

    Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResponse;
    bool success = false;
    try
    {
        success = ttsRuntime.handleAudioGeneration(talkerReq, talkerResponse, stream);
    }
    catch (...)
    {
        audioChannel->finish();
        throw;
    }
    // Covers failure paths where the isFinal callback never fired.
    audioChannel->finish();
    ELLM_CHECK(success, "TTS generation failed");
    ELLM_CHECK(!vocodeFailed, "Code2Wav vocoding failed mid-stream; audio output is incomplete");
    return talkerResponse.batchRvqCodes.empty() ? 0 : static_cast<int32_t>(talkerResponse.batchRvqCodes[0].size());
}

// Called synchronously by constructors while the Python caller holds the GIL.
void checkLoadingSignals()
{
    if (PyErr_CheckSignals() != 0)
    {
        throw py::python_error();
    }
}

//! Unified Python wrapper for LLMInferenceRuntime.
//! Supports both vanilla decoding and the runtime's speculative decoders
//! through constructor overloading — mirrors the C++ unified runtime.
class PyLLMRuntime
{
public:
    //! Vanilla constructor (no speculative decoding).
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, std::string const& checkpointDir,
        ContextCacheConfig const& contextCacheConfig, PleRuntimeConfig const& pleConfig)
    {
        mPluginHandle = loadEdgellmPluginLib();
        DeploymentOptions options{};
        options.engineDir = engineDir;
        options.multimodalEngineDir = multimodalEngineDir;
        options.loraWeightsMap = loraWeightsMap;
        options.contextCacheConfig = contextCacheConfig;
        options.pleConfig = pleConfig;
        options.checkpointDir = checkpointDir;
        auto artifacts = ModelArtifacts::loadFromEngineDir(options.engineDir, options.draftingConfig,
            options.checkpointDir, options.draftCheckpointDir, options.contextCacheConfig.enabled, mStream.get(),
            options.runtimeCachePath, options.offloadEmbedding, checkLoadingSignals);
        mRuntime = std::make_unique<LLMInferenceRuntime>(std::move(artifacts), std::move(options), mStream.get());
    }

    //! Speculative decoding constructor.
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, int32_t draftTopK,
        std::optional<int32_t> draftingStep, std::optional<int32_t> proposalTokens, std::optional<int32_t> verifyTokens,
        std::string const& checkpointDir, std::string const& draftCheckpointDir,
        ContextCacheConfig const& contextCacheConfig, PleRuntimeConfig const& pleConfig)
    {
        mPluginHandle = loadEdgellmPluginLib();
        SpecDecodeDraftingConfig draftingConfig;
        draftingConfig.draftTopK = draftTopK;
        draftingConfig.draftingStep = draftingStep;
        draftingConfig.proposalTokens = proposalTokens;
        draftingConfig.verifyTokens = verifyTokens;
        DeploymentOptions options{};
        options.engineDir = engineDir;
        options.multimodalEngineDir = multimodalEngineDir;
        options.loraWeightsMap = loraWeightsMap;
        options.draftingConfig = draftingConfig;
        options.contextCacheConfig = contextCacheConfig;
        options.pleConfig = pleConfig;
        options.checkpointDir = checkpointDir;
        options.draftCheckpointDir = draftCheckpointDir;
        auto artifacts = ModelArtifacts::loadFromEngineDir(options.engineDir, options.draftingConfig,
            options.checkpointDir, options.draftCheckpointDir, options.contextCacheConfig.enabled, mStream.get(),
            options.runtimeCachePath, options.offloadEmbedding, checkLoadingSignals);
        mRuntime = std::make_unique<LLMInferenceRuntime>(std::move(artifacts), std::move(options), mStream.get());
    }

    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool const success = mRuntime->handleRequest(request, response, mStream.get());
        ELLM_CHECK(success,
            mRuntime->unavailableReason().empty() ? "Failed to handle generation request"
                                                  : mRuntime->unavailableReason());
        return response;
    }

    std::string unavailableReason() const
    {
        return mRuntime->unavailableReason();
    }

    //! Load the Qwen3-Omni audio-output stack (Talker + CodePredictor + Code2Wav).
    //! tokenizerDir is normally the Thinker engine dir.
    void loadOmni(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& code2wavEngineDir, std::string const& tokenizerDir, std::string const& checkpointDir)
    {
        // Voice-clone reference encoders are not wired into the server yet.
        mTtsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(talkerEngineDir, codePredictorEngineDir, tokenizerDir,
            /*cloneEncoderDir=*/"", mStream.get(), checkpointDir);
        mCode2wavRunner = std::make_unique<Code2WavRunner>(code2wavEngineDir, mStream.get(), checkpointDir);
        if (!mTtsRuntime->captureDecodingCUDAGraph(mStream.get()))
        {
            LOG_WARNING("CUDA graph capture failed for TTS decoding, proceeding without.");
        }
    }

    //! Thinker-Talker streaming generation. Text streams through the request's
    //! stream_channels as usual; vocoded PCM chunks stream through audioChannel.
    //! Runs synchronously — call from a worker thread and pop both channels.
    LLMGenerationResponse handleRequestStreamingAudio(LLMGenerationRequest& request,
        std::shared_ptr<AudioStreamChannel> const& audioChannel, OmniAudioParams const& params)
    {
        ELLM_CHECK(
            mTtsRuntime != nullptr && mCode2wavRunner != nullptr, "Omni runtime not loaded. Call load_omni() first.");
        ELLM_CHECK(audioChannel != nullptr, "audio_channel must not be null");
        ELLM_CHECK(!audioChannel->isFinished(), "audio_channel is already finished");

        request.generateAudio = true;

        Qwen3OmniTTSRuntime::OmniGenerationRequest omniReq;
        omniReq.talkerTemperature = params.talkerTemperature;
        omniReq.talkerTopK = params.talkerTopK;
        omniReq.talkerTopP = params.talkerTopP;
        omniReq.repetitionPenalty = params.repetitionPenalty;
        omniReq.maxAudioLength = params.maxAudioLength;
        omniReq.speakerName = params.speakerName;

        Qwen3OmniTTSRuntime::ThinkerTalkerStreamingConfig streamCfg;
        streamCfg.talkerPrefillThreshold = params.talkerPrefillThreshold;
        streamCfg.codecChunkFrames = params.codecChunkFrames;
        bool vocodeFailed = false;
        streamCfg.onAudioChunkReady
            = makeChunkVocodeCallback(*mCode2wavRunner, mStream.get(), audioChannel, vocodeFailed);

        LLMGenerationResponse thinkerResponse;
        Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResponse;
        bool success = false;
        try
        {
            success = mTtsRuntime->handleStreamingGeneration(
                *mRuntime, request, thinkerResponse, streamCfg, omniReq, talkerResponse, mStream.get());
        }
        catch (...)
        {
            audioChannel->finish();
            throw;
        }
        // Covers failure paths where the isFinal callback never fired.
        audioChannel->finish();
        ELLM_CHECK(success, "Streaming Omni generation failed");
        ELLM_CHECK(!vocodeFailed, "Code2Wav vocoding failed mid-stream; audio output is incomplete");
        // success=false with zero frames is the legit degenerate case (thinker
        // reply shorter than the prefill threshold) — deliver text with empty
        // audio. With frames already emitted it means the Talker errored and
        // the audio is truncated, which must not pass silently.
        bool const talkerTruncated = !talkerResponse.success && !talkerResponse.numFramesPerSample.empty()
            && talkerResponse.numFramesPerSample[0] > 0;
        ELLM_CHECK(!talkerTruncated, "Talker generation failed mid-stream; audio output is truncated");
        return thinkerResponse;
    }

    //! Standalone TTS on the loaded Omni stack: synthesize speech for `text`
    //! directly, without a Thinker generation pass.
    int32_t handleRequestTTS(
        std::string const& text, OmniAudioParams const& params, std::shared_ptr<AudioStreamChannel> const& audioChannel)
    {
        ELLM_CHECK(
            mTtsRuntime != nullptr && mCode2wavRunner != nullptr, "Omni runtime not loaded. Call load_omni() first.");
        return runStandaloneTTS(*mTtsRuntime, *mCode2wavRunner, mStream.get(), text, params, audioChannel);
    }

    std::vector<std::string> getSpeakerNames() const
    {
        ELLM_CHECK(mTtsRuntime != nullptr, "Omni runtime not loaded. Call load_omni() first.");
        return mTtsRuntime->getSpeakerNames();
    }

    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const
    {
        return mRuntime->countPromptTokens(request);
    }

    bool captureDecodingCudaGraph()
    {
        return mRuntime->captureDecodingCUDAGraph(mStream.get());
    }

    bool hasDraftModel() const
    {
        return mRuntime->hasDraftModel();
    }

    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mRuntime->getPrefillMetrics();
    }

    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mRuntime->getGenerationMetrics();
    }

    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const
    {
        return mRuntime->getSpecDecodeGenerationMetrics();
    }

    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mRuntime->getMultimodalMetrics();
    }

    std::optional<ContextCacheMetrics> getContextCacheMetrics() const
    {
        return mRuntime->getContextCacheMetrics();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    std::unique_ptr<Qwen3OmniTTSRuntime> mTtsRuntime;
    std::unique_ptr<Code2WavRunner> mCode2wavRunner;
};

//! What one request produced at the terminals of a stage graph.
struct MultiStageOutput
{
    std::string text;
    std::string audioPcm16; //!< Little-endian int16 samples, interleaved across channels.
    int32_t audioSampleRate{0};
    int32_t audioChannels{0};
    std::string videoRgb8; //!< Packed [frames, height, width, 3] 8-bit RGB.
    int32_t videoFrames{0};
    int32_t videoHeight{0};
    int32_t videoWidth{0};
    int32_t videoFps{0};
    std::string stages; //!< Per-stage report as JSON.
};

//! Runs one request at a time through the stage graph in a directory's multistage_config.json.
class PyMultiStageRuntime
{
public:
    explicit PyMultiStageRuntime(std::string const& configDir)
    {
        mPluginHandle = loadEdgellmPluginLib();
        mRuntime.load(configDir, mShared);
    }

    MultiStageOutput generate(
        std::string const& prompt, std::string const& style, std::string const& sampling, bool audio, bool video)
    {
        namespace ms = multiStage;
        auto const bindText = [](ms::PortBindings& payload, char const* field, std::string const& value) {
            if (!value.empty())
            {
                payload.bind(field,
                    ms::PortValue{const_cast<char*>(value.data()), {static_cast<int64_t>(value.size())},
                        ms::PortKind::kText, /*elementBytes=*/1, /*complete=*/true});
            }
        };
        ms::PortBindings payload;
        bindText(payload, "request:messages", prompt);
        bindText(payload, "request:style", style);
        bindText(payload, "request:sampling", sampling);

        // The channels are unbounded, so they hold every chunk until run() returns.
        ms::Sinks sinks;
        sinks.text = StreamChannel::create();
        AudioStreamChannel audioChannel;
        VideoStreamChannel videoChannel;
        sinks.audio = audio ? &audioChannel : nullptr;
        sinks.video = video ? &videoChannel : nullptr;

        std::string error;
        if (mRuntime.submit(payload, sinks, error) == nullptr)
        {
            throw std::runtime_error("request refused: " + error);
        }
        auto const responses = mRuntime.run();
        mRuntime.take();
        auto const& response = responses.front();
        if (!response.ok)
        {
            throw std::runtime_error("request failed: " + response.error);
        }

        MultiStageOutput output;
        output.stages = response.stageDiagnostics.dump();
        while (auto chunk = sinks.text->waitPop(std::chrono::milliseconds{0}))
        {
            output.text += chunk->text;
        }
        while (auto chunk = audioChannel.waitPop(std::chrono::milliseconds{0}))
        {
            output.audioPcm16 += chunk->pcm16;
            output.audioSampleRate = chunk->sampleRate;
            output.audioChannels = chunk->numChannels;
        }
        while (auto chunk = videoChannel.waitPop(std::chrono::milliseconds{0}))
        {
            output.videoRgb8 += chunk->rgb8;
            output.videoFrames += chunk->numFrames;
            output.videoHeight = chunk->height;
            output.videoWidth = chunk->width;
            output.videoFps = chunk->fps;
        }
        return output;
    }

private:
    std::unique_ptr<void, DlDeleter> mPluginHandle; //!< Outlives every engine the runtime loads.
    SharedResources mShared;                        //!< Outlives every stage that borrowed it.
    multiStage::MultiStageRuntime mRuntime;
};

//! TTS-only runtime for Qwen3-TTS-style deployments: Talker + CodePredictor +
//! Code2Wav, no Thinker engine. An empty tokenizer_dir falls back to the
//! talker dir, which carries the tokenizer files in the standard export
//! layout.
class PyTTSRuntime
{
public:
    PyTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& code2wavEngineDir, std::string const& tokenizerDir, std::string const& checkpointDir)
    {
        mPluginHandle = loadEdgellmPluginLib();
        std::string const& tokenizer = tokenizerDir.empty() ? talkerEngineDir : tokenizerDir;
        mTtsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(
            talkerEngineDir, codePredictorEngineDir, tokenizer, /*cloneEncoderDir=*/"", mStream.get(), checkpointDir);
        mCode2wavRunner = std::make_unique<Code2WavRunner>(code2wavEngineDir, mStream.get(), checkpointDir);
        if (!mTtsRuntime->captureDecodingCUDAGraph(mStream.get()))
        {
            LOG_WARNING("CUDA graph capture failed for TTS decoding, proceeding without.");
        }
    }

    int32_t handleRequestTTS(
        std::string const& text, OmniAudioParams const& params, std::shared_ptr<AudioStreamChannel> const& audioChannel)
    {
        return runStandaloneTTS(*mTtsRuntime, *mCode2wavRunner, mStream.get(), text, params, audioChannel);
    }

    std::vector<std::string> getSpeakerNames() const
    {
        return mTtsRuntime->getSpeakerNames();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    std::unique_ptr<Qwen3OmniTTSRuntime> mTtsRuntime;
    std::unique_ptr<Code2WavRunner> mCode2wavRunner;
};

#ifdef EDGELLM_ENABLE_NEMOTRON_ASR
class PyNemotronAsrRuntime
{
public:
    PyNemotronAsrRuntime(std::string const& engineDir, std::string const& tokenizerDir)
    {
        mPluginHandle = loadEdgellmPluginLib();
        std::string const& tokenizer = tokenizerDir.empty() ? engineDir : tokenizerDir;
        mRuntime = std::make_unique<NemotronAsrRuntime>(engineDir, tokenizer, mStream.get());
    }

    NemotronAsrRuntime::Result transcribe(std::string const& audioBytes, int32_t promptId)
    {
        constexpr int32_t kTargetSampleRate = 16000;
        audio::AudioPCM pcm;
        if (!audio::loadAudioBytes(
                reinterpret_cast<uint8_t const*>(audioBytes.data()), audioBytes.size(), kTargetSampleRate, pcm))
        {
            throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
        }
        int32_t const resolvedPromptId = promptId >= 0 ? promptId : mRuntime->defaultPromptId();
        return mRuntime->transcribe(pcm, resolvedPromptId, mStream.get());
    }

    int32_t defaultPromptId() const
    {
        return mRuntime->defaultPromptId();
    }

    int64_t maxMelFrames() const
    {
        return mRuntime->maxMelFrames();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    std::unique_ptr<NemotronAsrRuntime> mRuntime;
};
#endif

//! The scheduling options as the bindings name them.
//! @throws std::invalid_argument for a name the scheduler does not know, listing the accepted ones.
scheduler::SchedulingOptions toSchedulingOptions(std::string const& ifbSchedulingMode, int32_t maxNumTokens,
    int32_t maxChunkTokens, std::string const& schedulingPolicy, std::string const& preemption,
    int32_t priorityReservedSlots, int32_t maxPauseSteps)
{
    scheduler::SchedulingOptions options;
    options.ifbSchedulingMode = scheduler::parseEnum<scheduler::IfbSchedulingMode>(ifbSchedulingMode);
    options.maxNumTokens = maxNumTokens;
    options.maxChunkTokens = maxChunkTokens;
    options.schedulingPolicy = scheduler::parseEnum<scheduler::SchedulingPolicy>(schedulingPolicy);
    options.preemption = scheduler::parseEnum<scheduler::PreemptionMode>(preemption);
    options.priorityReservedSlots = priorityReservedSlots;
    options.maxPauseSteps = maxPauseSteps;
    return options;
}

//! Python-facing owner of the IFB request engine.
//!
//! Builds the runtime the same way PyLLMRuntime's constructors do -- the speculative one when a
//! drafting shape is given -- then hands it to a RequestEngine together with the CUDA stream; the
//! engine owns the runtime, this class owns the engine and the stream.
class PyRequestEngine
{
public:
    PyRequestEngine(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, std::string const& checkpointDir,
        ContextCacheConfig const& contextCacheConfig, int32_t maxBatchSize, int32_t maxPendingRequests,
        int32_t commandQueueCapacity, bool captureCudaGraph, PleRuntimeConfig const& pleConfig,
        std::string const& ifbSchedulingMode, int32_t maxNumTokens, int32_t maxChunkTokens,
        std::string const& schedulingPolicy, std::string const& preemption, int32_t priorityReservedSlots,
        int32_t maxPauseSteps, std::optional<int32_t> draftTopK, std::optional<int32_t> draftingStep,
        std::optional<int32_t> proposalTokens, std::optional<int32_t> verifyTokens,
        std::string const& draftCheckpointDir)
    {
        scheduler::EngineConfig config;
        config.maxBatchSize = maxBatchSize;
        config.maxPendingRequests = maxPendingRequests;
        config.commandQueueCapacity = commandQueueCapacity;
        // Parsed before the runtime is built so a misspelt name fails fast.
        config.scheduling = toSchedulingOptions(ifbSchedulingMode, maxNumTokens, maxChunkTokens, schedulingPolicy,
            preemption, priorityReservedSlots, maxPauseSteps);

        mPluginHandle = loadEdgellmPluginLib();
        // Single rank only in this release: tensor-parallel deployments stay on the blocking path
        // until the session's step decisions reach the other ranks (see the IFB support matrix).
        DeploymentOptions options{};
        options.engineDir = engineDir;
        options.multimodalEngineDir = multimodalEngineDir;
        options.loraWeightsMap = loraWeightsMap;
        options.contextCacheConfig = contextCacheConfig;
        options.checkpointDir = checkpointDir;
        options.pleConfig = pleConfig;
        // A speculative deployment: the session seats each sequence's draft state as it admits it.
        if (draftTopK.has_value())
        {
            SpecDecodeDraftingConfig draftingConfig;
            draftingConfig.draftTopK = *draftTopK;
            draftingConfig.draftingStep = draftingStep;
            draftingConfig.proposalTokens = proposalTokens;
            draftingConfig.verifyTokens = verifyTokens;
            options.draftingConfig = draftingConfig;
            options.draftCheckpointDir = draftCheckpointDir;
        }
        auto artifacts = ModelArtifacts::loadFromEngineDir(options.engineDir, options.draftingConfig,
            options.checkpointDir, options.draftCheckpointDir, options.contextCacheConfig.enabled, mStream.get(),
            options.runtimeCachePath, options.offloadEmbedding, checkLoadingSignals);
        auto runtime = std::make_unique<LLMInferenceRuntime>(std::move(artifacts), std::move(options), mStream.get());
        // Capture happens here because after the constructor hands the runtime over, nothing but
        // the actor may touch it, and the actor has no pre-first-request hook to capture from.
        if (captureCudaGraph && !runtime->captureDecodingCUDAGraph(mStream.get()))
        {
            LOG_WARNING("CUDA graph capture failed for decoding, proceeding without.");
        }
        mTokenizerView = runtime.get();
        mEngine = std::make_unique<scheduler::RequestEngine>(std::move(runtime), mStream.get(), config);
    }

    scheduler::RequestHandle submit(LLMGenerationRequest request)
    {
        return mEngine->submit(std::move(request));
    }

    void cancel(scheduler::RequestId id)
    {
        mEngine->cancel(id);
    }

    void shutdown(scheduler::ShutdownMode mode)
    {
        mEngine->shutdown(mode);
    }

    int32_t queued() const noexcept
    {
        return mEngine->queued();
    }

    int32_t resident() const noexcept
    {
        return mEngine->resident();
    }

    bool running() const noexcept
    {
        return mEngine->running();
    }

    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const
    {
        return mTokenizerView->countPromptTokens(request);
    }

    scheduler::EngineMetrics metrics() const noexcept
    {
        return mEngine->metrics();
    }

    std::optional<ContextCacheMetrics> contextCacheMetrics() const
    {
        return mEngine->contextCacheMetrics();
    }

    scheduler::ResolvedSchedulingPolicy resolvedPolicy() const
    {
        return mEngine->resolvedPolicy();
    }

    scheduler::EngineStatus status() const
    {
        return mEngine->status();
    }

    ~PyRequestEngine()
    {
        // The binding runs this with the GIL held, and the engine's destructor joins the actor, which
        // may be inside a generation for a while. Release the GIL for that wait so the other Python
        // threads (and the asyncio loop) keep running; the actor itself never takes the GIL.
        py::gil_scoped_release release;
        mEngine.reset();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    //! The engine owns the runtime and only the actor may drive it; this view exists for exactly
    //! one call, countPromptTokens, which is const all the way down (template + encode on a
    //! tokenizer with no mutable state), so it can run concurrently with the actor's own
    //! tokenization. Valid as long as mEngine is, which owns the pointee.
    LLMInferenceRuntime const* mTokenizerView{nullptr};
    //! Last member on purpose: its destructor joins the actor, which is still driving mStream,
    //! so it has to run before the stream (and the plugin library) are torn down.
    std::unique_ptr<scheduler::RequestEngine> mEngine;
};

imageUtils::ImageData loadImageFromPath(std::string const& path)
{
    return imageUtils::loadRgbImageFromFile(path);
}

imageUtils::ImageData loadImageFromBytes(py::bytes const& data)
{
    std::string dataStr(data.c_str(), data.size());
    return imageUtils::loadRgbImageFromEncodedBytes(
        reinterpret_cast<unsigned char const*>(dataStr.data()), dataStr.size());
}

//! \brief Build an AudioData from raw encoded audio bytes (wav / mp3 / flac).
//! Decodes via vendored miniaudio (16 kHz mono FP32) and hands raw PCM off to
//! the runner; mel extraction happens inside the audio runner per its
//! ``audio/config.json``.
//! The binding copies Python bytes before releasing the GIL for this pure-C++
//! decode path.
audioUtils::AudioData loadAudioBufferFromBytes(std::string const& dataStr)
{
    constexpr int32_t kTargetSampleRate = 16000;
    audioUtils::AudioData audio;
    if (!audioUtils::loadAudioDataFromBytes(
            reinterpret_cast<uint8_t const*>(dataStr.data()), dataStr.size(), kTargetSampleRate, audio))
    {
        throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
    }
    return audio;
}

//! \brief Decode audio bytes + extract mel to numpy.float32 (host-resident).
//! Diagnostic helper exposing the C++ ``MelExtractor`` output directly — the
//! production path (``loadAudioBufferFromBytes`` -> runner) only returns PCM
//! at the Python boundary and runs mel extraction inside the audio runner.
//! Returns the FE's natural 2-D layout.
py::ndarray<py::numpy, float, py::c_contig> extractMelToNumpy(py::bytes const& data, std::string const& feType)
{
    audio::MelExtractor extractor = audio::makeExtractorByName(feType);
    int32_t const targetSampleRate = extractor.config().sampleRate;

    audio::AudioPCM pcm;
    if (!audio::loadAudioBytes(reinterpret_cast<uint8_t const*>(data.c_str()), data.size(), targetSampleRate, pcm))
    {
        throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
    }

    Tensor hostMel;
    if (!extractor.extract(pcm, hostMel))
    {
        throw std::runtime_error("Mel extraction failed");
    }

    Coords const& shape = hostMel.getShape();
    std::vector<size_t> npShape;
    npShape.reserve(static_cast<size_t>(shape.getNumDims()));
    for (int32_t d = 0; d < shape.getNumDims(); ++d)
    {
        npShape.push_back(static_cast<size_t>(shape[d]));
    }
    size_t const byteSize = static_cast<size_t>(shape.volume()) * sizeof(float);
    float* dataPtr = new float[static_cast<size_t>(shape.volume())];
    py::capsule owner(dataPtr, [](void* value) noexcept { delete[] static_cast<float*>(value); });
    py::ndarray<py::numpy, float, py::c_contig> out(dataPtr, npShape.size(), npShape.data(), owner);
    std::memcpy(dataPtr, hostMel.dataPointer<float>(), byteSize);
    return out;
}

//! Build an ImageData (video frame stack) over a contiguous (T, H, W, 3) uint8 numpy array.
struct VideoRegistrationRegistry
{
    struct Entry
    {
        size_t size;
        size_t users;
    };

    std::mutex mutex;
    std::map<uintptr_t, Entry> entries;

    static VideoRegistrationRegistry& instance()
    {
        static auto* registry = new VideoRegistrationRegistry;
        return *registry;
    }

    void release(uintptr_t begin) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = entries.find(begin);
        if (--it->second.users != 0)
        {
            return;
        }
        cudaError_t const status = cudaHostUnregister(reinterpret_cast<void*>(begin));
        if (status != cudaSuccess)
        {
            LOG_WARNING("Failed to unregister a NumPy video buffer: %s", cudaGetErrorString(status));
            (void) cudaGetLastError();
        }
        entries.erase(it);
    }
};

struct VideoRegistrationLease
{
    uintptr_t begin = 0;

    VideoRegistrationLease() = default;
    explicit VideoRegistrationLease(uintptr_t value)
        : begin(value)
    {
    }
    VideoRegistrationLease(VideoRegistrationLease const&) = delete;
    VideoRegistrationLease& operator=(VideoRegistrationLease const&) = delete;
    VideoRegistrationLease(VideoRegistrationLease&& other) noexcept
        : begin(std::exchange(other.begin, 0))
    {
    }
    ~VideoRegistrationLease()
    {
        release();
    }

    void release() noexcept
    {
        if (begin != 0)
        {
            VideoRegistrationRegistry::instance().release(std::exchange(begin, 0));
        }
    }
};

VideoRegistrationLease registerVideoStorage(void* data, size_t size)
{
    auto& registry = VideoRegistrationRegistry::instance();
    std::lock_guard<std::mutex> lock(registry.mutex);
    uintptr_t const begin = reinterpret_cast<uintptr_t>(data);
    uintptr_t const end = begin + size;
    for (auto& [registeredBegin, entry] : registry.entries)
    {
        uintptr_t const registeredEnd = registeredBegin + entry.size;
        if (begin < registeredEnd && registeredBegin < end)
        {
            check::check(registeredBegin <= begin && end <= registeredEnd,
                "Overlapping NumPy video views must share a contiguous base allocation");
            ++entry.users;
            return VideoRegistrationLease{registeredBegin};
        }
    }

    cudaPointerAttributes attributes{};
    cudaError_t const status = cudaPointerGetAttributes(&attributes, data);
    bool const needsRegistration = status != cudaSuccess || attributes.type == cudaMemoryTypeUnregistered;
    if (status != cudaSuccess)
    {
        (void) cudaGetLastError();
    }
    if (!needsRegistration)
    {
        return {};
    }
    CUDA_CHECK(cudaHostRegister(data, size, cudaHostRegisterMapped | cudaHostRegisterPortable));
    try
    {
        registry.entries.emplace(begin, VideoRegistrationRegistry::Entry{size, 1});
    }
    catch (...)
    {
        (void) cudaHostUnregister(data);
        throw;
    }
    return VideoRegistrationLease{begin};
}

imageUtils::ImageData loadVideoFromArray(
    py::ndarray<py::numpy, uint8_t const, py::c_contig> array, double fps, std::vector<double> const& timestamps)
{
    check::check(array.ndim() == 4, "video array must be 4D (T, H, W, 3)");
    check::check(array.shape(3) == 3, "video array must have 3 channels (last dim == 3)");
    check::check(std::isfinite(fps) && fps > 0.0, "fps must be a positive finite number");
    int64_t const T = static_cast<int64_t>(array.shape(0));
    int64_t const H = static_cast<int64_t>(array.shape(1));
    int64_t const W = static_cast<int64_t>(array.shape(2));
    int64_t const C = static_cast<int64_t>(array.shape(3));
    check::check(T > 0, "video array must have at least one frame (T > 0)");

    check::check(timestamps.empty() || static_cast<int64_t>(timestamps.size()) == T,
        "timestamps must be empty or have one entry per frame");
    for (double const ts : timestamps)
    {
        check::check(std::isfinite(ts), "timestamps must be finite");
    }
    void* const data = const_cast<uint8_t*>(array.data());
    auto tensor = std::make_unique<rt::Tensor>(
        data, Coords{T, H, W, C}, rt::DeviceType::kCPU, trt_edgellm::DataType::kUINT8, "nanobind::loadVideoFromArray");
    auto owner = array;

    py::object root = array.cast();
    while (true)
    {
        py::object base = py::getattr(root, "base", py::none());
        if (base.is_none() || !py::hasattr(base, "ctypes") || !py::hasattr(base, "flags")
            || !py::cast<bool>(base.attr("flags").attr("c_contiguous")))
        {
            break;
        }
        root = std::move(base);
    }
    void* const storage = reinterpret_cast<void*>(py::cast<uintptr_t>(root.attr("ctypes").attr("data")));
    size_t const storageSize = py::cast<size_t>(root.attr("nbytes"));
    auto registration = registerVideoStorage(storage, storageSize);

    auto release
        = [owner = std::move(owner), registration = std::move(registration)](rt::Tensor* value) mutable noexcept {
              delete value;
              registration.release();
              py::gil_scoped_acquire gil;
              owner = decltype(owner){};
          };
    auto buffer = std::shared_ptr<rt::Tensor>(tensor.release(), std::move(release));

    imageUtils::ImageLayout layout{};
    layout.format = imageUtils::ImageFormat::kRGB8;
    layout.pitchBytes[0] = W * C;
    layout.frameStrideBytes[0] = H * W * C;
    imageUtils::ImageData video = imageUtils::wrapImageBuffer(std::move(buffer), layout, W, H, T, true);
    video.fps = fps;
    video.timestamps = timestamps;
    return video;
}

} // anonymous namespace

NB_MODULE(_edgellm_runtime, m)
{
    m.doc() = "TensorRT Edge LLM Python Bindings";

    // ========================================================================
    // Profiling
    // ========================================================================
    m.def("set_profiling_enabled", &setProfilingEnabled, py::arg("enabled"),
        "Enable or disable aggregate counters and GPU stage timing on this thread");
    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if aggregate profiling is enabled on this thread");

    // ========================================================================
    // Metrics
    // ========================================================================
    py::class_<metrics::LLMPrefillMetrics>(m, "LLMPrefillMetrics")
        .def_ro("reused_tokens", &metrics::LLMPrefillMetrics::reusedTokens)
        .def_ro("computed_tokens", &metrics::LLMPrefillMetrics::computedTokens)
        .def("get_total_runs", &metrics::LLMPrefillMetrics::getTotalRuns);

    py::class_<metrics::LLMGenerationMetrics>(m, "LLMGenerationMetrics")
        .def_ro("generated_tokens", &metrics::LLMGenerationMetrics::generatedTokens)
        .def("get_total_runs", &metrics::LLMGenerationMetrics::getTotalRuns);

    py::class_<metrics::SpecDecodeGenerationMetrics>(m, "SpecDecodeGenerationMetrics")
        .def_ro("total_iterations", &metrics::SpecDecodeGenerationMetrics::totalIterations)
        .def_ro("total_generated_tokens", &metrics::SpecDecodeGenerationMetrics::totalGeneratedTokens)
        .def_ro("total_accepted_draft_tokens", &metrics::SpecDecodeGenerationMetrics::totalAcceptedDraftTokens)
        .def_ro("total_proposed_draft_tokens", &metrics::SpecDecodeGenerationMetrics::totalProposedDraftTokens)
        .def("get_total_runs", &metrics::SpecDecodeGenerationMetrics::getTotalRuns);

    py::class_<metrics::MultimodalMetrics>(m, "MultimodalMetrics")
        .def_ro("total_images", &metrics::MultimodalMetrics::totalImages)
        .def_ro("total_image_tokens", &metrics::MultimodalMetrics::totalImageTokens)
        .def("get_total_runs", &metrics::MultimodalMetrics::getTotalRuns);

    py::class_<ContextCachePoolMetrics>(m, "ContextCachePoolMetrics")
        .def_ro("free", &ContextCachePoolMetrics::free)
        .def_ro("capacity", &ContextCachePoolMetrics::capacity);

    py::class_<ContextCacheMetrics>(m, "ContextCacheMetrics")
        .def_ro("admitted_sequences", &ContextCacheMetrics::admittedSequences)
        .def_ro("hit_sequences", &ContextCacheMetrics::hitSequences)
        .def_ro("media_aware_sequences", &ContextCacheMetrics::mediaAwareSequences)
        .def_ro("lookup_bypass_sequences", &ContextCacheMetrics::lookupBypassSequences)
        .def_ro("forced_cold_sequences", &ContextCacheMetrics::forcedColdSequences)
        .def_ro("standard_plans", &ContextCacheMetrics::standardPlans)
        .def_ro("no_reusable_prefix_plans", &ContextCacheMetrics::noReusablePrefixPlans)
        .def_ro("full_input_rewind_plans", &ContextCacheMetrics::fullInputRewindPlans)
        .def_ro("matched_tokens", &ContextCacheMetrics::matchedTokens)
        .def_ro("reused_tokens", &ContextCacheMetrics::reusedTokens)
        .def_ro("publication_attempts", &ContextCacheMetrics::publicationAttempts)
        .def_ro("committed_publications", &ContextCacheMetrics::committedPublications)
        .def_ro("existing_publications", &ContextCacheMetrics::existingPublications)
        .def_ro("published_endpoints", &ContextCacheMetrics::publishedEndpoints)
        .def_ro("hybrid_restores", &ContextCacheMetrics::hybridRestores)
        .def_ro("hybrid_snapshot_pressure_skips", &ContextCacheMetrics::hybridSnapshotPressureSkips)
        .def_ro("hybrid_capture_synchronizations", &ContextCacheMetrics::hybridCaptureSynchronizations)
        .def_ro("spec_full_page_replays", &ContextCacheMetrics::specFullPageReplays)
        .def_ro("spec_pair_publications", &ContextCacheMetrics::specPairPublications)
        .def_ro("planning_nanoseconds", &ContextCacheMetrics::planningNanoseconds)
        .def_ro("current_records", &ContextCacheMetrics::currentRecords)
        .def_ro("base_kv_pages", &ContextCacheMetrics::baseKvPages)
        .def_ro("draft_kv_pages", &ContextCacheMetrics::draftKvPages)
        .def_ro("recurrent_snapshots", &ContextCacheMetrics::recurrentSnapshots)
        .def_ro("partial_kv_snapshots", &ContextCacheMetrics::partialKvSnapshots)
        .def_ro("swa_kv_pages", &ContextCacheMetrics::swaKvPages)
        .def_ro("evicted_records", &ContextCacheMetrics::evictedRecords)
        .def_ro("reclaimed_base_kv_pages", &ContextCacheMetrics::reclaimedBaseKvPages)
        .def_ro("reclaimed_draft_kv_pages", &ContextCacheMetrics::reclaimedDraftKvPages)
        .def_ro("reclaimed_recurrent_snapshots", &ContextCacheMetrics::reclaimedRecurrentSnapshots)
        .def_ro("reclaimed_partial_kv_snapshots", &ContextCacheMetrics::reclaimedPartialKvSnapshots)
        .def_ro("reclaimed_swa_kv_pages", &ContextCacheMetrics::reclaimedSwaKvPages);

    py::enum_<PleResidency>(m, "PleResidency")
        .value("RESIDENT", PleResidency::kRESIDENT)
        .value("CACHED", PleResidency::kCACHED);
    py::class_<PleRuntimeConfig>(m, "PleRuntimeConfig")
        .def(py::init<>())
        .def_rw("residency", &PleRuntimeConfig::residency)
        .def_rw("cache_slots", &PleRuntimeConfig::cacheSlots)
        .def_rw("file_path", &PleRuntimeConfig::filePath);

    py::class_<ContextCacheConfig>(m, "ContextCacheConfig")
        .def(py::init<>())
        .def_rw("enabled", &ContextCacheConfig::enabled)
        .def_rw("max_records", &ContextCacheConfig::maxRecords)
        .def_rw("max_snapshots", &ContextCacheConfig::maxSnapshots);

    // ========================================================================
    // Image utilities
    // ========================================================================
    py::class_<imageUtils::ImageData>(m, "ImageData")
        .def(py::init<>())
        .def_ro("width", &imageUtils::ImageData::width)
        .def_ro("height", &imageUtils::ImageData::height)
        .def_ro("channels", &imageUtils::ImageData::channels)
        .def_ro("frames", &imageUtils::ImageData::frames)
        .def_ro("is_video", &imageUtils::ImageData::isVideo)
        .def_ro("timestamps", &imageUtils::ImageData::timestamps)
        .def_rw("fps", &imageUtils::ImageData::fps)
        .def_rw("do_resize", &imageUtils::ImageData::doResize);

    m.def("load_image_from_path", &loadImageFromPath, py::arg("path"), "Load image from file path");
    m.def("load_image_from_bytes", &loadImageFromBytes, py::arg("data"), "Load image from bytes");
    m.def(
        "load_video_from_paths",
        [](std::vector<std::string> const& framePaths, double fps, std::vector<double> const& timestamps) {
            check::check(std::isfinite(fps) && fps > 0.0, "fps must be a positive finite number");
            check::check(timestamps.empty() || timestamps.size() == framePaths.size(),
                "timestamps must be empty or have one entry per frame");
            for (double const ts : timestamps)
            {
                check::check(std::isfinite(ts), "timestamps must be finite");
            }
            imageUtils::ImageData video = imageUtils::loadRgbVideoFromFrames(framePaths, fps);
            video.timestamps = timestamps;
            return video;
        },
        py::arg("frame_paths"), py::arg("fps") = 1.0, py::arg("timestamps") = std::vector<double>{},
        py::call_guard<py::gil_scoped_release>(),
        "Load a video by stacking identically-sized image files into one (T, H, W, 3) ImageData");
    // The retained ndarray keeps its zero-copy buffer alive for ImageData.
    m.def("load_video_from_array", &loadVideoFromArray, py::arg("array"), py::arg("fps") = 1.0,
        py::arg("timestamps") = std::vector<double>{},
        "Wrap a contiguous (T, H, W, 3) uint8 numpy array as video ImageData without copying");

    // ========================================================================
    // Audio utilities
    // ========================================================================
    py::class_<audioUtils::AudioData>(m, "AudioData")
        .def(py::init<>())
        .def_rw("sample_rate", &audioUtils::AudioData::sampleRate)
        .def_prop_ro(
            "num_samples",
            [](audioUtils::AudioData const& audio) { return audio.pcm ? audio.pcm->numSamples() : int64_t{0}; },
            "Number of decoded PCM samples (0 when no PCM is attached)");

    m.def(
        "load_audio_buffer_from_bytes",
        [](py::bytes const& data) {
            std::string dataStr(data.c_str(), data.size());
            py::gil_scoped_release release;
            return loadAudioBufferFromBytes(dataStr);
        },
        py::arg("data"),
        "Build an AudioData from raw encoded audio bytes (wav/mp3/flac). "
        "Decodes to mono FP32 PCM @ 16 kHz via miniaudio; the audio runner "
        "extracts mel internally per its audio/config.json.");

    m.def("extract_mel_to_numpy", &extractMelToNumpy, py::arg("data"), py::arg("fe_type"),
        "Decode audio bytes (wav/mp3/flac) and return the host float32 "
        "mel-spectrogram directly to numpy (no f16 cast, no GPU upload). "
        "Test / diagnostic helper for comparing the C++ pipeline against HF "
        "feature extractors at full float32 precision.");

    // ========================================================================
    // Message structures
    // ========================================================================
    py::class_<Message::MessageContent>(m, "MessageContent")
        .def(py::init<>())
        .def(
            "__init__",
            [](Message::MessageContent* self, std::string const& type, std::string const& content) {
                new (self) Message::MessageContent{};
                self->type = type;
                self->content = content;
            },
            py::arg("type"), py::arg("content"))
        .def_rw("type", &Message::MessageContent::type)
        .def_rw("content", &Message::MessageContent::content);

    py::class_<Message::ToolCall>(m, "MessageToolCall")
        .def(py::init<>())
        .def_rw("id", &Message::ToolCall::id)
        .def_rw("type", &Message::ToolCall::type)
        .def_rw("name", &Message::ToolCall::name)
        .def_rw("arguments", &Message::ToolCall::arguments);

    py::class_<Message>(m, "Message")
        .def(py::init<>())
        .def(
            "__init__",
            [](Message* self, std::string const& role, std::vector<Message::MessageContent> const& contents) {
                new (self) Message{};
                self->role = role;
                self->contents = contents;
            },
            py::arg("role"), py::arg("contents"))
        .def_rw("role", &Message::role)
        .def_rw("contents", &Message::contents)
        .def_rw("reasoning_content", &Message::reasoningContent)
        .def_rw("has_reasoning_content", &Message::hasReasoningContent)
        .def_rw("tool_calls", &Message::toolCalls)
        .def_rw("has_tool_calls", &Message::hasToolCalls)
        .def_rw("tool_call_id", &Message::toolCallId)
        .def_rw("name", &Message::name)
        .def_rw("has_content", &Message::hasContent)
        .def_rw("content_is_array", &Message::contentIsArray)
        .def_rw("content_is_null", &Message::contentIsNull);

    py::class_<ToolDefinition>(m, "ToolDefinition")
        .def(py::init<>())
        .def_rw("name", &ToolDefinition::name)
        .def_rw("description", &ToolDefinition::description)
        .def_rw("parameters", &ToolDefinition::parameters)
        .def_rw("strict", &ToolDefinition::strict)
        .def_rw("has_description", &ToolDefinition::hasDescription)
        .def_rw("has_parameters", &ToolDefinition::hasParameters)
        .def_rw("has_strict", &ToolDefinition::hasStrict);

    py::enum_<ToolChoice::Mode>(m, "ToolChoiceMode")
        .value("NONE", ToolChoice::Mode::kNone)
        .value("AUTO", ToolChoice::Mode::kAuto)
        .value("REQUIRED", ToolChoice::Mode::kRequired)
        .value("FUNCTION", ToolChoice::Mode::kFunction);

    py::class_<ToolChoice>(m, "ToolChoice")
        .def(py::init<>())
        .def_rw("mode", &ToolChoice::mode)
        .def_rw("function_name", &ToolChoice::functionName);

    m.def(
        "create_text_message",
        [](std::string const& role, std::string const& text) {
            Message msg;
            msg.role = role;
            Message::MessageContent content;
            content.type = "text";
            content.content = text;
            msg.contents.push_back(content);
            return msg;
        },
        py::arg("role"), py::arg("text"), "Create a simple text message");

    // ========================================================================
    // Request / Response
    // ========================================================================
    // One top-K logprob entry. `piece` is exposed as bytes (raw token bytes may
    // not be valid UTF-8 for byte-level BPE tokens); decode on the Python side
    // with errors="replace" for a display string.
    py::class_<LogprobEntry>(m, "LogprobEntry")
        .def_ro("token_id", &LogprobEntry::tokenId)
        .def_ro("logprob", &LogprobEntry::logprob)
        .def_prop_ro("piece", [](LogprobEntry const& e) { return py::bytes(e.piece.data(), e.piece.size()); });

    py::class_<LLMGenerationRequest::FormattedRequest>(m, "FormattedRequest")
        .def(py::init<>())
        .def_rw("formatted_system_prompt", &LLMGenerationRequest::FormattedRequest::formattedSystemPrompt)
        .def_rw("formatted_complete_request", &LLMGenerationRequest::FormattedRequest::formattedCompleteRequest);

    py::enum_<GuideType>(m, "GuideType", "Grammar dialect carried by GuidedDecodingParams.")
        .value("JSON_OBJECT", GuideType::kJsonObject)
        .value("JSON_SCHEMA", GuideType::kJsonSchema)
        .value("REGEX", GuideType::kRegex)
        .value("EBNF", GuideType::kEbnf)
        .value("STRUCTURAL_TAG", GuideType::kStructuralTag)
        .value("CHOICE", GuideType::kChoice);

    py::class_<GuidedDecodingParams>(m, "GuidedDecodingParams",
        "Grammar constraint for one request. `guide` holds the schema / pattern / grammar / choice "
        "list and is empty only for JSON_OBJECT. `jump_forward` appends grammar-forced tokens "
        "without forward passes.")
        .def(py::init<>())
        .def(
            "__init__",
            [](GuidedDecodingParams* self, GuideType type, std::string guide, bool jumpForward) {
                new (self) GuidedDecodingParams{};
                self->type = type;
                self->guide = std::move(guide);
                self->jumpForward = jumpForward;
            },
            py::arg("type"), py::arg("guide") = std::string{}, py::arg("jump_forward") = false)
        .def_rw("type", &GuidedDecodingParams::type)
        .def_rw("guide", &GuidedDecodingParams::guide)
        .def_rw("jump_forward", &GuidedDecodingParams::jumpForward);

    m.def(
        "validate_guided_decoding_params",
        [](GuidedDecodingParams const& params) {
            std::string failReason;
            bool const valid = rt::validateGuidedDecodingParams(params, failReason);
            return std::make_pair(valid, failReason);
        },
        py::arg("params"),
        "Reject guides XGrammar accepts but does not enforce, and oversized ones. Returns "
        "(is_valid, reason) so a server can answer 400 instead of failing the request later.");

    py::class_<LLMGenerationRequest::Request>(m, "Request")
        .def(py::init<>())
        .def(
            "__init__",
            [](LLMGenerationRequest::Request* self, std::vector<Message> const& messages) {
                new (self) LLMGenerationRequest::Request{};
                self->messages = messages;
            },
            py::arg("messages"))
        .def_rw("messages", &LLMGenerationRequest::Request::messages)
        .def_rw("image_buffers", &LLMGenerationRequest::Request::imageBuffers)
        .def_rw("audio_buffers", &LLMGenerationRequest::Request::audioBuffers)
        .def_rw("stop_strings", &LLMGenerationRequest::Request::stopStrings)
        .def_rw("logit_bias", &LLMGenerationRequest::Request::logitBias)
        .def_rw("guided_decoding", &LLMGenerationRequest::Request::guidedDecoding)
        .def_rw("sampling_seed", &LLMGenerationRequest::Request::samplingSeed);

    // ========================================================================
    // Streaming
    // ========================================================================
    py::enum_<FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", FinishReason::kNotFinished)
        .value("END_ID", FinishReason::kEndId)
        .value("LENGTH", FinishReason::kLength)
        .value("CANCELLED", FinishReason::kCancelled)
        .value("ERROR", FinishReason::kError)
        .value("STOP_WORDS", FinishReason::kStopWords);

    py::class_<StreamMetrics>(m, "StreamMetrics")
        .def_ro("processed_prompt_tokens", &StreamMetrics::processedPromptTokens)
        .def_ro("generated_tokens", &StreamMetrics::generatedTokens)
        .def_ro("decoded_tokens", &StreamMetrics::decodedTokens)
        .def_ro("prompt_seconds", &StreamMetrics::promptSeconds)
        .def_ro("generation_seconds", &StreamMetrics::generationSeconds)
        .def_ro("prefill_start_ns", &StreamMetrics::prefillStartNs);

    py::class_<StreamChunk>(m, "StreamChunk")
        .def(py::init<>())
        .def_ro("token_ids", &StreamChunk::tokenIds)
        .def_ro("text", &StreamChunk::text)
        .def_ro("prompt_token_count", &StreamChunk::promptTokenCount)
        .def_ro("reused_prompt_token_count", &StreamChunk::reusedPromptTokenCount)
        .def_ro("metrics", &StreamChunk::metrics)
        .def_ro("finished", &StreamChunk::finished)
        .def_ro("reason", &StreamChunk::reason)
        .def_ro("logprobs", &StreamChunk::logprobs);

    py::class_<StreamChannel>(m, "StreamChannel")
        .def_static("create", &StreamChannel::create, py::arg("collect_metrics") = false)
        .def("try_pop", &StreamChannel::tryPop)
        .def(
            "wait_pop",
            [](StreamChannel& self, int64_t timeoutMs) { return self.waitPop(std::chrono::milliseconds{timeoutMs}); },
            py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
        .def("is_finished", &StreamChannel::isFinished)
        .def("get_reason", &StreamChannel::getReason)
        .def("is_cancelled", &StreamChannel::isCancelled)
        .def("cancel", &StreamChannel::cancel)
        .def("set_stream_interval", &StreamChannel::setStreamInterval, py::arg("n"))
        .def("get_stream_interval", &StreamChannel::getStreamInterval)
        .def("set_skip_special_tokens", &StreamChannel::setSkipSpecialTokens, py::arg("skip"))
        .def("get_skip_special_tokens", &StreamChannel::getSkipSpecialTokens);

    // ========================================================================
    // Omni audio streaming
    // ========================================================================
    py::class_<AudioChunk>(m, "AudioChunk")
        .def_prop_ro(
            "pcm16", [](AudioChunk const& c) { return py::bytes(c.pcm16.data(), c.pcm16.size()); },
            "Little-endian int16 PCM samples, interleaved across channels")
        .def_ro("is_final", &AudioChunk::isFinal)
        .def_ro("num_frames", &AudioChunk::numFrames)
        .def_ro("sample_rate", &AudioChunk::sampleRate)
        .def_ro("num_channels", &AudioChunk::numChannels);

    py::class_<AudioStreamChannel>(m, "AudioStreamChannel")
        .def(py::init<>())
        .def("try_pop", [](AudioStreamChannel& self) { return self.waitPop(std::chrono::milliseconds{0}); })
        .def(
            "wait_pop",
            [](AudioStreamChannel& self, int64_t timeoutMs) {
                return self.waitPop(std::chrono::milliseconds{timeoutMs});
            },
            py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
        .def("is_finished", [](AudioStreamChannel const& self) { return self.isFinished(); })
        .def("is_cancelled", [](AudioStreamChannel const& self) { return self.isCancelled(); })
        .def("cancel", [](AudioStreamChannel& self) { self.cancel(); });

    py::class_<OmniAudioParams>(m, "OmniAudioParams")
        .def(py::init<>())
        .def_rw("talker_temperature", &OmniAudioParams::talkerTemperature)
        .def_rw("talker_top_k", &OmniAudioParams::talkerTopK)
        .def_rw("talker_top_p", &OmniAudioParams::talkerTopP)
        .def_rw("repetition_penalty", &OmniAudioParams::repetitionPenalty)
        .def_rw("max_audio_length", &OmniAudioParams::maxAudioLength)
        .def_rw("speaker_name", &OmniAudioParams::speakerName)
        .def_rw("codec_chunk_frames", &OmniAudioParams::codecChunkFrames)
        .def_rw("talker_prefill_threshold", &OmniAudioParams::talkerPrefillThreshold);

    // ========================================================================
    // Request / Response (continued)
    // ========================================================================
    py::enum_<ContextCacheLookupPolicy>(m, "ContextCacheLookupPolicy")
        .value("USE_CACHE", ContextCacheLookupPolicy::kUseCache)
        .value("BYPASS", ContextCacheLookupPolicy::kBypass);

    py::enum_<ContextCacheCommitPolicy>(m, "ContextCacheCommitPolicy")
        .value("INCLUDING_GENERATED_TOKENS", ContextCacheCommitPolicy::kIncludingGeneratedTokens)
        .value("PREFILL_STATE_ONLY", ContextCacheCommitPolicy::kPrefillStateOnly);

    py::enum_<SpecProposalSampling>(m, "SpecProposalSampling")
        .value("AUTO", SpecProposalSampling::kAuto)
        .value("GREEDY", SpecProposalSampling::kGreedy)
        .value("PROBABILISTIC", SpecProposalSampling::kProbabilistic);

    py::class_<LLMGenerationRequest>(m, "LLMGenerationRequest")
        .def(py::init<>())
        .def_rw("requests", &LLMGenerationRequest::requests)
        .def_rw("pre_tokenized_input_ids", &LLMGenerationRequest::preTokenizedInputIds)
        .def_rw("formatted_requests", &LLMGenerationRequest::formattedRequests)
        .def_rw("temperature", &LLMGenerationRequest::temperature)
        .def_rw("top_p", &LLMGenerationRequest::topP)
        .def_rw("top_k", &LLMGenerationRequest::topK)
        .def_rw("sampling_seed", &LLMGenerationRequest::samplingSeed)
        .def_rw("spec_proposal_sampling", &LLMGenerationRequest::proposalSampling)
        .def_rw("max_generate_length", &LLMGenerationRequest::maxGenerateLength)
        .def_rw("lora_weights_name", &LLMGenerationRequest::loraWeightsName)
        .def_rw("apply_chat_template", &LLMGenerationRequest::applyChatTemplate)
        .def_rw("add_generation_prompt", &LLMGenerationRequest::addGenerationPrompt)
        .def_rw("enable_thinking", &LLMGenerationRequest::enableThinking)
        .def_rw("reasoning_effort", &LLMGenerationRequest::reasoningEffort)
        .def_rw("tools", &LLMGenerationRequest::tools)
        .def_rw("tool_choice", &LLMGenerationRequest::toolChoice)
        .def_rw("parallel_tool_calls", &LLMGenerationRequest::parallelToolCalls)
        .def_rw("disable_spec_decode", &LLMGenerationRequest::disableSpecDecode)
        .def_rw("skip_special_tokens", &LLMGenerationRequest::skipSpecialTokens)
        .def_rw("ignore_eos", &LLMGenerationRequest::ignoreEos)
        .def_rw("recurrent_capture_interval", &LLMGenerationRequest::recurrentCaptureInterval)
        .def_rw("stream_channels", &LLMGenerationRequest::streamChannels)
        .def_rw("num_logprobs", &LLMGenerationRequest::numLogprobs)
        .def_rw("context_cache_lookup_policy", &LLMGenerationRequest::contextCacheLookupPolicy)
        .def_rw("context_cache_commit_policy", &LLMGenerationRequest::contextCacheCommitPolicy)
        .def_rw("context_cache_replay_tail_length", &LLMGenerationRequest::contextCacheReplayTailLength)
        .def_rw("priority", &LLMGenerationRequest::priority,
            "Scheduling priority; lower is served first. Honoured by a RequestEngine with "
            "scheduling_policy='priority'.");

    py::class_<LLMGenerationResponse>(m, "LLMGenerationResponse")
        .def(py::init<>())
        .def_rw("output_ids", &LLMGenerationResponse::outputIds)
        .def_rw("output_texts", &LLMGenerationResponse::outputTexts)
        .def_rw("logprobs", &LLMGenerationResponse::logprobs)
        .def_ro("finish_reasons", &LLMGenerationResponse::finishReasons)
        .def_ro("prompt_token_counts", &LLMGenerationResponse::inputTokenCounts)
        .def_ro("reused_prompt_token_counts", &LLMGenerationResponse::reusedInputTokenCounts);

    // ========================================================================
    // Runtime: unified (vanilla + Eagle speculative decoding)
    // ========================================================================
    py::class_<PyLLMRuntime>(m, "LLMRuntime", "Unified LLM inference runtime with optional speculative decoding.")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 std::string const&, ContextCacheConfig const&, PleRuntimeConfig const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{},
            py::arg("checkpoint_dir") = "", py::arg("context_cache_config") = ContextCacheConfig{},
            py::arg("ple_config") = PleRuntimeConfig{}, "Construct for vanilla (non-speculative) decoding")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 int32_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::string const&,
                 std::string const&, ContextCacheConfig const&, PleRuntimeConfig const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir"), py::arg("lora_weights_map"),
            py::arg("draft_top_k"), py::arg("draft_step"), py::arg("proposal_tokens"), py::arg("verify_tokens"),
            py::arg("checkpoint_dir") = "", py::arg("draft_checkpoint_dir") = "",
            py::arg("context_cache_config") = ContextCacheConfig{}, py::arg("ple_config") = PleRuntimeConfig{},
            "Construct for speculative decoding")
        .def("handle_request", &PyLLMRuntime::handleRequest, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(), "Process a generation request and return the response")
        .def_prop_ro("unavailable_reason", &PyLLMRuntime::unavailableReason)
        .def("load_omni", &PyLLMRuntime::loadOmni, py::arg("talker_engine_dir"), py::arg("code_predictor_engine_dir"),
            py::arg("code2wav_engine_dir"), py::arg("tokenizer_dir"), py::arg("checkpoint_dir") = "",
            "Load the Qwen3-Omni audio-output stack (Talker + CodePredictor + Code2Wav)")
        .def("handle_request_streaming_audio", &PyLLMRuntime::handleRequestStreamingAudio, py::arg("request"),
            py::arg("audio_channel"), py::arg("params"), py::call_guard<py::gil_scoped_release>(),
            "Thinker-Talker streaming generation with PCM chunks pushed to audio_channel")
        .def("handle_request_tts", &PyLLMRuntime::handleRequestTTS, py::arg("text"), py::arg("params"),
            py::arg("audio_channel"), py::call_guard<py::gil_scoped_release>(),
            "Standalone TTS on the loaded Omni stack; returns the number of codec frames generated")
        .def("get_speaker_names", &PyLLMRuntime::getSpeakerNames)
        .def("count_prompt_tokens", &PyLLMRuntime::countPromptTokens, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(), "Handle an explicit text token-count request")
        .def("capture_decoding_cuda_graph", &PyLLMRuntime::captureDecodingCudaGraph,
            "Capture CUDA graphs for optimized decoding")
        .def("has_draft_model", &PyLLMRuntime::hasDraftModel, "Check if speculative decoding draft model is loaded")
        .def("get_prefill_metrics", &PyLLMRuntime::getPrefillMetrics, py::rv_policy::reference_internal)
        .def("get_generation_metrics", &PyLLMRuntime::getGenerationMetrics, py::rv_policy::reference_internal)
        .def("get_spec_decode_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::rv_policy::reference_internal)
        .def("get_eagle_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::rv_policy::reference_internal) // deprecated alias
        .def("get_multimodal_metrics", &PyLLMRuntime::getMultimodalMetrics)
        .def("get_context_cache_metrics", &PyLLMRuntime::getContextCacheMetrics);

    // ========================================================================
    // Request engine: in-flight batching over the same runtime
    // ========================================================================

    // submit() refusing is backpressure, not failure; give it its own exception type so the
    // server can translate it into "busy" without pattern-matching RuntimeError strings.
    py::exception<scheduler::SubmitError>(m, "SubmitError");

    py::enum_<scheduler::FailureKind>(m, "RequestFailureKind", "Why a request failed, for a caller that acts on it.")
        .value("OTHER", scheduler::FailureKind::kOther)
        .value("COMPACT_STEP_REFUSED", scheduler::FailureKind::kCompactStepRefused,
            "A plugin refused a compact step; the engine serves on without compact steps and the request can be "
            "retried.")
        .value("ENGINE_STOPPED", scheduler::FailureKind::kEngineStopped,
            "The engine stopped serving and fails every request until it is recreated.");

    // RequestHandle.get() of a failed request raises RequestFailedError (a RuntimeError) whose kind attribute is the
    // RequestFailureKind, so the server maps failures to HTTP statuses without reading the message.
    // The module attribute keeps the type alive for the translator's payload.
    py::exception<scheduler::RequestFailedError> const requestFailedError(m, "RequestFailedError", PyExc_RuntimeError);
    py::register_exception_translator(
        [](std::exception_ptr const& error, void* type) {
            try
            {
                std::rethrow_exception(error);
            }
            catch (scheduler::RequestFailedError const& failure)
            {
                py::object const exception = py::handle(static_cast<PyObject*>(type))(failure.what());
                exception.attr("kind") = py::cast(failure.kind());
                PyErr_SetObject(static_cast<PyObject*>(type), exception.ptr());
            }
        },
        requestFailedError.ptr());

    m.def(
        "validate_scheduling_options",
        [](std::string const& ifbSchedulingMode, int32_t maxNumTokens, int32_t maxChunkTokens,
            std::string const& schedulingPolicy, std::string const& preemption, int32_t priorityReservedSlots,
            int32_t maxPauseSteps, int32_t maxBatchSize) {
            scheduler::validateSchedulingOptions(
                toSchedulingOptions(ifbSchedulingMode, maxNumTokens, maxChunkTokens, schedulingPolicy, preemption,
                    priorityReservedSlots, maxPauseSteps),
                maxBatchSize);
        },
        py::arg("ifb_scheduling_mode"), py::arg("max_num_tokens"), py::arg("max_chunk_tokens"),
        py::arg("scheduling_policy"), py::arg("preemption"), py::arg("priority_reserved_slots"),
        py::arg("max_pause_steps"), py::arg("max_batch_size"),
        "Check the RequestEngine scheduling options that need no engine, as a RequestEngine built for "
        "max_batch_size checks them at construction. Raises ValueError naming the option and the alternative.");

    py::class_<scheduler::EngineMetrics>(m, "EngineMetrics",
        "A point-in-time copy of the engine's counters; per-field accurate, not a single atomic cut.")
        .def_ro("submitted", &scheduler::EngineMetrics::submitted)
        .def_ro("refused", &scheduler::EngineMetrics::refused)
        .def_ro("stalls_incompatible", &scheduler::EngineMetrics::stallsIncompatible)
        .def_ro("stalls_guided", &scheduler::EngineMetrics::stallsGuided)
        .def_ro("stalls_exclusive", &scheduler::EngineMetrics::stallsExclusive)
        .def_ro("stalls_no_capacity", &scheduler::EngineMetrics::stallsNoCapacity)
        .def_ro("admitted_mid_flight", &scheduler::EngineMetrics::admittedMidFlight)
        .def_ro("completed", &scheduler::EngineMetrics::completed)
        .def_ro("cancelled", &scheduler::EngineMetrics::cancelled)
        .def_ro("failed", &scheduler::EngineMetrics::failed)
        .def_ro("queue_latency_total_us", &scheduler::EngineMetrics::queueLatencyTotalUs)
        .def_ro("queue_latency_max_us", &scheduler::EngineMetrics::queueLatencyMaxUs)
        .def_ro("queue_latency_count", &scheduler::EngineMetrics::queueLatencyCount)
        .def_ro("decode_steps", &scheduler::EngineMetrics::decodeSteps)
        .def_ro("decode_graph_steps", &scheduler::EngineMetrics::decodeGraphSteps)
        .def_ro("prompt_steps", &scheduler::EngineMetrics::promptSteps)
        .def_ro("mixed_steps", &scheduler::EngineMetrics::mixedSteps)
        .def_ro("decode_subset_steps", &scheduler::EngineMetrics::decodeSubsetSteps)
        .def_ro("step_sequences", &scheduler::EngineMetrics::stepSequences)
        .def_ro("step_prompt_chunks", &scheduler::EngineMetrics::stepPromptChunks)
        .def_ro("partial_prompt_chunks", &scheduler::EngineMetrics::partialPromptChunks)
        .def_ro("step_decode_rows", &scheduler::EngineMetrics::stepDecodeRows)
        .def_ro("step_tokens", &scheduler::EngineMetrics::stepTokens)
        .def_ro("step_sampled_rows", &scheduler::EngineMetrics::stepSampledRows)
        .def_ro("step_time_total_us", &scheduler::EngineMetrics::stepTimeTotalUs)
        .def_ro("step_time_max_us", &scheduler::EngineMetrics::stepTimeMaxUs)
        .def_ro("paused_resident_steps", &scheduler::EngineMetrics::pausedResidentSteps)
        .def_ro("compact_refusals", &scheduler::EngineMetrics::compactRefusals);

    py::class_<scheduler::EngineStatus>(m, "EngineStatus", "Whether the engine still serves requests.")
        .def_ro("serving", &scheduler::EngineStatus::serving,
            "False once the engine opens no session again; every later request then fails with unavailable_reason.")
        .def_ro("unavailable_reason", &scheduler::EngineStatus::unavailableReason,
            "Why the engine stopped serving; empty while it serves.")
        .def_ro("compact_unavailable_reason", &scheduler::EngineStatus::compactUnavailableReason,
            "Why compact execution is unavailable; empty when it is available.");

    py::class_<scheduler::RequestStats>(m, "RequestStats",
        "Scheduling statistics of one request; available from RequestHandle.stats() once it is ready.")
        .def_ro("admitted", &scheduler::RequestStats::admitted)
        .def_ro("queue_time_us", &scheduler::RequestStats::queueTimeUs)
        .def_ro("time_to_first_token_us", &scheduler::RequestStats::timeToFirstTokenUs)
        .def_ro("prompt_tokens", &scheduler::RequestStats::promptTokens)
        .def_ro("reused_prompt_tokens", &scheduler::RequestStats::reusedPromptTokens)
        .def_ro("prompt_chunks", &scheduler::RequestStats::promptChunks)
        .def_ro("paused_steps", &scheduler::RequestStats::pausedSteps)
        .def_ro("paused_time_us", &scheduler::RequestStats::pausedTimeUs);

    py::class_<scheduler::ResolvedSchedulingPolicy>(
        m, "ResolvedSchedulingPolicy", "The scheduling policy a RequestEngine runs with.")
        .def_prop_ro("mode", [](scheduler::ResolvedSchedulingPolicy const& p) { return scheduler::toString(p.mode); })
        .def_prop_ro(
            "mode_source",
            [](scheduler::ResolvedSchedulingPolicy const& p) { return scheduler::toString(p.modeSource); },
            "\"requested\" when the configured mode is kept, \"auto\" when auto chose it.")
        .def_ro("reason", &scheduler::ResolvedSchedulingPolicy::reason,
            "Why auto chose the mode; empty for a requested mode.")
        .def_prop_ro("scheduling_policy",
            [](scheduler::ResolvedSchedulingPolicy const& p) { return scheduler::toString(p.schedulingPolicy); })
        .def_prop_ro("preemption",
            [](scheduler::ResolvedSchedulingPolicy const& p) { return scheduler::toString(p.preemption); })
        .def_prop_ro(
            "max_num_tokens", [](scheduler::ResolvedSchedulingPolicy const& p) { return p.budgets.maxNumTokens; })
        .def_prop_ro(
            "max_chunk_tokens", [](scheduler::ResolvedSchedulingPolicy const& p) { return p.budgets.maxChunkTokens; })
        .def_ro("priority_reserved_slots", &scheduler::ResolvedSchedulingPolicy::priorityReservedSlots)
        .def_ro("max_pause_steps", &scheduler::ResolvedSchedulingPolicy::maxPauseSteps)
        .def_ro("compact_execution", &scheduler::ResolvedSchedulingPolicy::compactExecution)
        .def_prop_ro(
            "execution", [](scheduler::ResolvedSchedulingPolicy const& p) { return scheduler::executionName(p); },
            "\"compact\" when steps may mix prompt chunks and decode rows, else \"single-cohort\".")
        .def_ro("warnings", &scheduler::ResolvedSchedulingPolicy::warnings,
            "Configuration accepted but not applied as given, one sentence each.");

    py::enum_<scheduler::ShutdownMode>(m, "ShutdownMode")
        .value("DRAIN", scheduler::ShutdownMode::kDrain)
        .value("CANCEL", scheduler::ShutdownMode::kCancel);

    // GIL rules: every blocking call (get(), shutdown(), and StreamChannel.wait_pop() above)
    // releases the GIL, and the actor thread never takes it (no Python callbacks cross this
    // boundary). submit() and cancel() only post to the command queue, but they still release:
    // they can contend on the admission gate, and nothing they touch is Python state.
    py::class_<scheduler::RequestHandle>(
        m, "RequestHandle", "A claim on one in-flight request. Move-only in C++; treat it as the single consumer.")
        .def_prop_ro("id", &scheduler::RequestHandle::id)
        .def("stream", &scheduler::RequestHandle::streamShared,
            "The token channel for this request; tokens appear as they are produced.")
        .def("get", &scheduler::RequestHandle::get, py::call_guard<py::gil_scoped_release>(),
            "Block until the request is terminal, then return its response. Raises RequestFailedError when it failed "
            "and RuntimeError when it was cancelled.")
        .def("ready", &scheduler::RequestHandle::ready, "True once get() would return without blocking.")
        .def("stats", &scheduler::RequestHandle::stats,
            "The request's RequestStats once ready() (queue time, TTFT, prompt chunks, reuse, pause); None before.")
        .def("cancel", &scheduler::RequestHandle::cancel, py::call_guard<py::gil_scoped_release>(),
            "Ask the engine to stop this request. Non-blocking and idempotent.")
        .def("valid", &scheduler::RequestHandle::valid);

    py::class_<PyRequestEngine>(m, "RequestEngine",
        "Owns a runtime and the actor thread that drives it; submissions from any thread, "
        "responses through RequestHandle.")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 std::string const&, ContextCacheConfig const&, int32_t, int32_t, int32_t, bool,
                 PleRuntimeConfig const&, std::string const&, int32_t, int32_t, std::string const&, std::string const&,
                 int32_t, int32_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>,
                 std::optional<int32_t>, std::string const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{},
            py::arg("checkpoint_dir") = "", py::arg("context_cache_config") = ContextCacheConfig{},
            py::arg("max_batch_size") = 1, py::arg("max_pending_requests") = 256,
            py::arg("command_queue_capacity") = 64, py::arg("capture_cuda_graph") = true,
            py::arg("ple_config") = PleRuntimeConfig{}, py::arg("ifb_scheduling_mode") = "auto",
            py::arg("max_num_tokens") = 0, py::arg("max_chunk_tokens") = 0, py::arg("scheduling_policy") = "fcfs",
            py::arg("preemption") = "none", py::arg("priority_reserved_slots") = 0, py::arg("max_pause_steps") = 0,
            py::arg("draft_top_k") = py::none(), py::arg("draft_step") = py::none(),
            py::arg("proposal_tokens") = py::none(), py::arg("verify_tokens") = py::none(),
            py::arg("draft_checkpoint_dir") = "",
            "ifb_scheduling_mode: auto | mixed | prefillFirst | singleCohort; scheduling_policy: fcfs | priority; "
            "preemption: none | pause (pause requires scheduling_policy=priority and the mixed mode); "
            "max_num_tokens / max_chunk_tokens: mixed-step budgets, 0 for the engine-derived default; "
            "priority_reserved_slots: slots kept for requests with priority < 0 (requires "
            "scheduling_policy=priority); max_pause_steps: under preemption=pause, the most steps in a row a "
            "request stays paused, 0 for no limit. Speculative deployments pass their drafting shape "
            "(draft_top_k and the rest, as LLMRuntime's speculative constructor takes them). Raises ValueError "
            "when the deployment cannot serve the configuration.")
        .def("submit", &PyRequestEngine::submit, py::arg("request"), py::call_guard<py::gil_scoped_release>(),
            "Queue a request. Returns immediately with a RequestHandle; raises SubmitError when refused.")
        .def("cancel", &PyRequestEngine::cancel, py::arg("request_id"), py::call_guard<py::gil_scoped_release>(),
            "Ask the actor to terminate a request by id.")
        .def("shutdown", &PyRequestEngine::shutdown, py::arg("mode") = scheduler::ShutdownMode::kDrain,
            py::call_guard<py::gil_scoped_release>(),
            "Stop the actor and join it. DRAIN lets queued requests finish; CANCEL abandons them.")
        .def("count_prompt_tokens", &PyRequestEngine::countPromptTokens, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(),
            "Tokenize and count without generating; safe alongside in-flight requests.")
        .def("metrics", &PyRequestEngine::metrics,
            "Scheduling counters: queue latency, admission stalls by reason, outcome distribution, steps by kind, "
            "step sizes, host step time, CUDA-graph decode steps and refused compact steps.")
        .def("context_cache_metrics", &PyRequestEngine::contextCacheMetrics,
            "Context-cache counters as of the last admission or step, or None when reuse is disabled.")
        .def_prop_ro("resolved_policy", &PyRequestEngine::resolvedPolicy,
            "The scheduling policy in effect: mode, reason, budgets. Resolved at construction, and again when a "
            "plugin refuses a compact step outside an explicit mixed mode.")
        .def("status", &PyRequestEngine::status,
            "EngineStatus: whether the engine still serves requests, and why compact execution is unavailable.")
        .def_prop_ro("queued", &PyRequestEngine::queued)
        .def_prop_ro("resident", &PyRequestEngine::resident)
        .def_prop_ro("running", &PyRequestEngine::running);

    py::class_<MultiStageOutput>(m, "MultiStageOutput", "What one request produced at the terminals of a stage graph")
        .def_ro("text", &MultiStageOutput::text)
        .def_prop_ro(
            "audio_pcm16",
            [](MultiStageOutput const& o) { return py::bytes(o.audioPcm16.data(), o.audioPcm16.size()); },
            "Little-endian int16 samples, interleaved across channels")
        .def_ro("audio_sample_rate", &MultiStageOutput::audioSampleRate)
        .def_ro("audio_channels", &MultiStageOutput::audioChannels)
        .def_prop_ro(
            "video_rgb8", [](MultiStageOutput const& o) { return py::bytes(o.videoRgb8.data(), o.videoRgb8.size()); },
            "Packed [frames, height, width, 3] 8-bit RGB")
        .def_ro("video_frames", &MultiStageOutput::videoFrames)
        .def_ro("video_height", &MultiStageOutput::videoHeight)
        .def_ro("video_width", &MultiStageOutput::videoWidth)
        .def_ro("video_fps", &MultiStageOutput::videoFps)
        .def_ro("stages", &MultiStageOutput::stages, "Per-stage report as JSON");

    py::class_<PyMultiStageRuntime>(
        m, "MultiStageRuntime", "Runs one request at a time through the stage graph of an engine directory")
        .def(py::init<std::string const&>(), py::arg("config_dir"), py::call_guard<py::gil_scoped_release>())
        .def("generate", &PyMultiStageRuntime::generate, py::arg("prompt"), py::arg("style") = "",
            py::arg("sampling") = "", py::arg("audio") = true, py::arg("video") = true,
            py::call_guard<py::gil_scoped_release>(),
            "Run one prompt; style and sampling are JSON objects; audio and video attach those terminals");

    py::class_<PyTTSRuntime>(
        m, "TTSRuntime", "TTS-only runtime (Qwen3-TTS-style): Talker + CodePredictor + Code2Wav, no Thinker engine")
        .def(py::init<std::string const&, std::string const&, std::string const&, std::string const&,
                 std::string const&>(),
            py::arg("talker_engine_dir"), py::arg("code_predictor_engine_dir"), py::arg("code2wav_engine_dir"),
            py::arg("tokenizer_dir") = "", py::arg("checkpoint_dir") = "", py::call_guard<py::gil_scoped_release>())
        .def("handle_request_tts", &PyTTSRuntime::handleRequestTTS, py::arg("text"), py::arg("params"),
            py::arg("audio_channel"), py::call_guard<py::gil_scoped_release>(),
            "Synthesize speech for text; PCM chunks stream to audio_channel; returns codec frame count")
        .def("get_speaker_names", &PyTTSRuntime::getSpeakerNames);

#ifdef EDGELLM_ENABLE_NEMOTRON_ASR
    py::class_<NemotronAsrRuntime::Result>(m, "NemotronAsrResult")
        .def_ro("tokens", &NemotronAsrRuntime::Result::tokens)
        .def_ro("text", &NemotronAsrRuntime::Result::text)
        .def_ro("num_mel_frames", &NemotronAsrRuntime::Result::numMelFrames)
        .def_ro("num_encoder_frames", &NemotronAsrRuntime::Result::numEncoderFrames)
        .def_ro("num_decode_steps", &NemotronAsrRuntime::Result::numDecodeSteps);

    py::class_<PyNemotronAsrRuntime>(m, "NemotronAsrRuntime")
        .def(py::init<std::string const&, std::string const&>(), py::arg("engine_dir"), py::arg("tokenizer_dir") = "",
            py::call_guard<py::gil_scoped_release>())
        .def("transcribe", &PyNemotronAsrRuntime::transcribe, py::arg("audio_bytes"), py::arg("prompt_id") = -1,
            py::call_guard<py::gil_scoped_release>())
        .def("default_prompt_id", &PyNemotronAsrRuntime::defaultPromptId)
        .def_prop_ro("max_mel_frames", &PyNemotronAsrRuntime::maxMelFrames);
#endif

    // ========================================================================
    // Builder: LLM
    // ========================================================================
    py::class_<builder::LLMBuilderConfig>(
        m, "LLMBuilderConfig", "Configuration for building TensorRT LLM engines from ONNX.")
        .def(py::init<>())
        .def_rw("max_input_len", &builder::LLMBuilderConfig::maxInputLen)
        .def_rw("spec_draft", &builder::LLMBuilderConfig::specDraft)
        .def_rw("spec_base", &builder::LLMBuilderConfig::specBase)
        .def_rw("eagle_draft", &builder::LLMBuilderConfig::specDraft) // deprecated alias
        .def_rw("eagle_base", &builder::LLMBuilderConfig::specBase)   // deprecated alias
        .def_rw("max_batch_size", &builder::LLMBuilderConfig::maxBatchSize)
        .def_rw("max_lora_rank", &builder::LLMBuilderConfig::maxLoraRank)
        .def_rw("max_kv_cache_capacity", &builder::LLMBuilderConfig::maxKVCacheCapacity)
        .def_rw("max_spec_verify_tokens", &builder::LLMBuilderConfig::maxSpecVerifyTokens)
        .def_rw("max_spec_draft_tokens", &builder::LLMBuilderConfig::maxSpecDraftTokens)
        .def("__repr__", &builder::LLMBuilderConfig::toString);

    py::class_<builder::LLMBuilder>(m, "LLMBuilder", "Build a TensorRT engine from an ONNX directory.")
        .def(py::init<std::filesystem::path const&, std::filesystem::path const&, builder::LLMBuilderConfig const&>(),
            py::arg("onnx_dir"), py::arg("engine_dir"), py::arg("config"))
        .def("build", &builder::LLMBuilder::build, "Build the TensorRT engine. Returns True on success.");

    // ========================================================================
    // Builder: Visual
    // ========================================================================
    py::class_<builder::VisualBuilderConfig>(
        m, "VisualBuilderConfig", "Configuration for building TensorRT visual encoder engines from ONNX.")
        .def(py::init<>())
        .def_rw("min_image_tokens", &builder::VisualBuilderConfig::minImageTokens)
        .def_rw("max_image_tokens", &builder::VisualBuilderConfig::maxImageTokens)
        .def_rw("max_image_tokens_per_image", &builder::VisualBuilderConfig::maxImageTokensPerImage)
        .def("__repr__", &builder::VisualBuilderConfig::toString);

    py::class_<builder::VisualBuilder>(
        m, "VisualBuilder", "Build a TensorRT engine for a visual encoder from an ONNX directory.")
        .def(
            py::init<std::filesystem::path const&, std::filesystem::path const&, builder::VisualBuilderConfig const&>(),
            py::arg("onnx_dir"), py::arg("engine_dir"), py::arg("config"))
        .def("build", &builder::VisualBuilder::build, "Build the TensorRT visual engine. Returns True on success.");

    // ========================================================================
    // Builder: Audio
    // ========================================================================
    py::class_<builder::AudioBuilderConfig>(
        m, "AudioBuilderConfig", "Configuration for building TensorRT audio encoder engines from ONNX.")
        .def(py::init<>())
        .def_rw("min_time_steps", &builder::AudioBuilderConfig::minTimeSteps)
        .def_rw("max_time_steps", &builder::AudioBuilderConfig::maxTimeSteps)
        .def_rw("use_trt_native_audio_attn", &builder::AudioBuilderConfig::useTrtNativeAudioAttn)
        .def("__repr__", &builder::AudioBuilderConfig::toString);

    py::class_<builder::AudioBuilder>(
        m, "AudioBuilder", "Build a TensorRT engine for an audio encoder from an ONNX directory.")
        .def(py::init<std::filesystem::path const&, std::filesystem::path const&, builder::AudioBuilderConfig const&>(),
            py::arg("onnx_dir"), py::arg("engine_dir"), py::arg("config"))
        .def("build", &builder::AudioBuilder::build, "Build the TensorRT audio engine. Returns True on success.");

    // ========================================================================
    // Convenience: create_generation_request
    // ========================================================================
    m.def(
        "create_generation_request",
        [](std::vector<std::vector<Message>> const& batchMessages, float temperature, float topP, int64_t topK,
            int64_t maxGenerateLength, bool applyChatTemplate, bool addGenerationPrompt, bool enableThinking,
            std::string const& loraWeightsName, bool disableSpecDecode,
            std::unordered_map<int32_t, float> const& logitBias, int32_t numLogprobs,
            std::optional<GuidedDecodingParams> const& guidedDecoding) {
            LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = topP;
            request.topK = topK;
            request.maxGenerateLength = maxGenerateLength;
            request.applyChatTemplate = applyChatTemplate;
            request.addGenerationPrompt = addGenerationPrompt;
            request.enableThinking = enableThinking;
            request.loraWeightsName = loraWeightsName;
            request.disableSpecDecode = disableSpecDecode;
            request.numLogprobs = numLogprobs;

            for (auto const& messages : batchMessages)
            {
                LLMGenerationRequest::Request req;
                req.messages = messages;
                req.logitBias = logitBias;
                req.guidedDecoding = guidedDecoding;
                request.requests.push_back(std::move(req));
            }
            return request;
        },
        py::arg("batch_messages"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f, py::arg("top_k") = 50,
        py::arg("max_generate_length") = 256, py::arg("apply_chat_template") = true,
        py::arg("add_generation_prompt") = true, py::arg("enable_thinking") = false, py::arg("lora_weights_name") = "",
        py::arg("disable_spec_decode") = false, py::arg("logit_bias") = std::unordered_map<int32_t, float>{},
        py::arg("num_logprobs") = 0, py::arg("guided_decoding") = std::optional<GuidedDecodingParams>{},
        "Create a generation request from a batch of message lists.");
}
