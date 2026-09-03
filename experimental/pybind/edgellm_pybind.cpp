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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

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
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/melSpectrogram.h"
#ifdef EDGELLM_ENABLE_NEMOTRON_ASR
#include "runtime/nemotronAsrRuntime.h"
#endif
#include "runtime/qwen3OmniTTSRuntime.h"
#include "runtime/streaming.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

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
//! examples/llm/llm_inference.cpp.
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
    bool const isFP32 = (waveform.getDataType() == nvinfer1::DataType::kFLOAT);
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

//! Unified Python wrapper for LLMInferenceRuntime.
//! Supports both vanilla decoding and the runtime's speculative decoders
//! through constructor overloading — mirrors the C++ unified runtime.
class PyLLMRuntime
{
public:
    //! Vanilla constructor (no speculative decoding). The trailing
    //! contextCacheConfig defaults to a disabled config, leaving the existing
    //! identity-page runtime path unchanged when callers omit it.
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, std::string const& checkpointDir,
        ContextCacheConfig const& contextCacheConfig)
    {
        mPluginHandle = loadEdgellmPluginLib();
        mRuntime = std::make_unique<LLMInferenceRuntime>(
            engineDir, multimodalEngineDir, loraWeightsMap, mStream.get(), contextCacheConfig, checkpointDir);
    }

    //! Speculative decoding constructor.
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, int32_t draftTopK, int32_t draftStep,
        int32_t verifyTreeSize, std::string const& checkpointDir, std::string const& draftCheckpointDir,
        ContextCacheConfig const& contextCacheConfig, int32_t dflashBlockSize)
    {
        mPluginHandle = loadEdgellmPluginLib();
        SpecDecodeDraftingConfig draftingConfig{draftTopK, draftStep, verifyTreeSize};
        draftingConfig.dflashBlockSize = dflashBlockSize;
        mRuntime = std::make_unique<LLMInferenceRuntime>(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig,
            mStream.get(), contextCacheConfig, checkpointDir, draftCheckpointDir);
    }

    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool const success = mRuntime->handleRequest(request, response, mStream.get());
        ELLM_CHECK(success, "Failed to handle generation request");
        return response;
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

    bool saveSystemPromptKVCache(std::string const& prompt, std::string const& loraWeightsName)
    {
        return mRuntime->genAndSaveSystemPromptKVCache(prompt, loraWeightsName, mStream.get());
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

imageUtils::ImageData loadImageFromPath(std::string const& path)
{
    return imageUtils::loadImageFromFile(path);
}

imageUtils::ImageData loadImageFromBytes(py::bytes const& data)
{
    std::string dataStr = data;
    return imageUtils::loadImageFromMemory(reinterpret_cast<unsigned char const*>(dataStr.data()), dataStr.size());
}

//! \brief Build an AudioData from raw encoded audio bytes (wav / mp3 / flac).
//! Decodes via vendored miniaudio (16 kHz mono FP32) and hands raw PCM off to
//! the runner; mel extraction happens inside the audio runner per its
//! ``audio/config.json``.
//! Takes std::string (not py::bytes) so the argument caster copies the bytes
//! while the GIL is held; the pure-C++ body then runs under gil_scoped_release.
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
py::array_t<float> extractMelToNumpy(py::bytes data, std::string const& feType)
{
    audio::MelExtractor extractor = audio::makeExtractorByName(feType);
    int32_t const targetSampleRate = extractor.config().sampleRate;

    char const* rawPtr = nullptr;
    Py_ssize_t rawSize = 0;
    if (PyBytes_AsStringAndSize(data.ptr(), const_cast<char**>(&rawPtr), &rawSize) != 0)
    {
        throw py::error_already_set();
    }

    audio::AudioPCM pcm;
    if (!audio::loadAudioBytes(
            reinterpret_cast<uint8_t const*>(rawPtr), static_cast<size_t>(rawSize), targetSampleRate, pcm))
    {
        throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
    }

    Tensor hostMel;
    if (!extractor.extract(pcm, hostMel))
    {
        throw std::runtime_error("Mel extraction failed");
    }

    Coords const& shape = hostMel.getShape();
    std::vector<ssize_t> npShape;
    npShape.reserve(static_cast<size_t>(shape.getNumDims()));
    for (int32_t d = 0; d < shape.getNumDims(); ++d)
    {
        npShape.push_back(static_cast<ssize_t>(shape[d]));
    }
    py::array_t<float> out(npShape);
    std::memcpy(out.mutable_data(), hostMel.dataPointer<float>(), static_cast<size_t>(shape.volume()) * sizeof(float));
    return out;
}

//! Build an ImageData (video frame stack) from a (T, H, W, 3) uint8 numpy array. forcecast makes a contiguous
//! uint8 copy if needed, so the buffer is always row-major and directly copyable into the device-bound tensor.
imageUtils::ImageData loadVideoFromArray(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const& array,
    double fps, std::vector<double> const& timestamps)
{
    py::buffer_info info = array.request();
    check::check(info.ndim == 4, "video array must be 4D (T, H, W, 3)");
    check::check(info.shape[3] == 3, "video array must have 3 channels (last dim == 3)");
    check::check(std::isfinite(fps) && fps > 0.0, "fps must be a positive finite number");
    int64_t const T = info.shape[0];
    int64_t const H = info.shape[1];
    int64_t const W = info.shape[2];
    int64_t const C = info.shape[3];
    check::check(T > 0, "video array must have at least one frame (T > 0)");

    rt::Tensor stacked(
        {T, H, W, C}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "pybind::loadVideoFromArray::stacked");
    std::memcpy(stacked.dataPointer<unsigned char>(), info.ptr, static_cast<size_t>(T * H * W * C));

    check::check(timestamps.empty() || static_cast<int64_t>(timestamps.size()) == T,
        "timestamps must be empty or have one entry per frame");
    for (double const ts : timestamps)
    {
        check::check(std::isfinite(ts), "timestamps must be finite");
    }
    imageUtils::ImageData video(std::move(stacked));
    video.fps = fps;
    video.isVideo = true;
    video.timestamps = timestamps;
    return video;
}

} // anonymous namespace

PYBIND11_MODULE(_edgellm_runtime, m)
{
    m.doc() = "TensorRT Edge LLM Python Bindings";

    // ========================================================================
    // Profiling
    // ========================================================================
    m.def("set_profiling_enabled", &setProfilingEnabled, py::arg("enabled"),
        "Enable or disable profiling data collection");
    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if profiling is currently enabled");

    // ========================================================================
    // Metrics
    // ========================================================================
    py::class_<metrics::LLMPrefillMetrics>(m, "LLMPrefillMetrics")
        .def_readonly("reused_tokens", &metrics::LLMPrefillMetrics::reusedTokens)
        .def_readonly("computed_tokens", &metrics::LLMPrefillMetrics::computedTokens)
        .def("get_total_runs", &metrics::LLMPrefillMetrics::getTotalRuns);

    py::class_<metrics::LLMGenerationMetrics>(m, "LLMGenerationMetrics")
        .def_readonly("generated_tokens", &metrics::LLMGenerationMetrics::generatedTokens)
        .def("get_total_runs", &metrics::LLMGenerationMetrics::getTotalRuns);

    py::class_<metrics::SpecDecodeGenerationMetrics>(m, "SpecDecodeGenerationMetrics")
        .def_readonly("total_iterations", &metrics::SpecDecodeGenerationMetrics::totalIterations)
        .def_readonly("total_generated_tokens", &metrics::SpecDecodeGenerationMetrics::totalGeneratedTokens)
        .def("get_total_runs", &metrics::SpecDecodeGenerationMetrics::getTotalRuns);

    py::class_<metrics::MultimodalMetrics>(m, "MultimodalMetrics")
        .def_readonly("total_images", &metrics::MultimodalMetrics::totalImages)
        .def_readonly("total_image_tokens", &metrics::MultimodalMetrics::totalImageTokens)
        .def("get_total_runs", &metrics::MultimodalMetrics::getTotalRuns);

    py::class_<ContextCachePoolMetrics>(m, "ContextCachePoolMetrics")
        .def_readonly("free", &ContextCachePoolMetrics::free)
        .def_readonly("capacity", &ContextCachePoolMetrics::capacity);

    py::class_<ContextCacheMetrics>(m, "ContextCacheMetrics")
        .def_readonly("admitted_sequences", &ContextCacheMetrics::admittedSequences)
        .def_readonly("hit_sequences", &ContextCacheMetrics::hitSequences)
        .def_readonly("lookup_bypass_sequences", &ContextCacheMetrics::lookupBypassSequences)
        .def_readonly("forced_cold_sequences", &ContextCacheMetrics::forcedColdSequences)
        .def_readonly("standard_plans", &ContextCacheMetrics::standardPlans)
        .def_readonly("no_reusable_prefix_plans", &ContextCacheMetrics::noReusablePrefixPlans)
        .def_readonly("full_input_rewind_plans", &ContextCacheMetrics::fullInputRewindPlans)
        .def_readonly("matched_tokens", &ContextCacheMetrics::matchedTokens)
        .def_readonly("reused_tokens", &ContextCacheMetrics::reusedTokens)
        .def_readonly("publication_attempts", &ContextCacheMetrics::publicationAttempts)
        .def_readonly("committed_publications", &ContextCacheMetrics::committedPublications)
        .def_readonly("existing_publications", &ContextCacheMetrics::existingPublications)
        .def_readonly("published_endpoints", &ContextCacheMetrics::publishedEndpoints)
        .def_readonly("hybrid_restores", &ContextCacheMetrics::hybridRestores)
        .def_readonly("hybrid_snapshot_pressure_skips", &ContextCacheMetrics::hybridSnapshotPressureSkips)
        .def_readonly("hybrid_capture_synchronizations", &ContextCacheMetrics::hybridCaptureSynchronizations)
        .def_readonly("spec_full_page_replays", &ContextCacheMetrics::specFullPageReplays)
        .def_readonly("spec_pair_publications", &ContextCacheMetrics::specPairPublications)
        .def_readonly("planning_nanoseconds", &ContextCacheMetrics::planningNanoseconds)
        .def_readonly("current_records", &ContextCacheMetrics::currentRecords)
        .def_readonly("base_kv_pages", &ContextCacheMetrics::baseKvPages)
        .def_readonly("draft_kv_pages", &ContextCacheMetrics::draftKvPages)
        .def_readonly("recurrent_snapshots", &ContextCacheMetrics::recurrentSnapshots)
        .def_readonly("partial_kv_snapshots", &ContextCacheMetrics::partialKvSnapshots)
        .def_readonly("evicted_records", &ContextCacheMetrics::evictedRecords)
        .def_readonly("reclaimed_base_kv_pages", &ContextCacheMetrics::reclaimedBaseKvPages)
        .def_readonly("reclaimed_draft_kv_pages", &ContextCacheMetrics::reclaimedDraftKvPages)
        .def_readonly("reclaimed_recurrent_snapshots", &ContextCacheMetrics::reclaimedRecurrentSnapshots)
        .def_readonly("reclaimed_partial_kv_snapshots", &ContextCacheMetrics::reclaimedPartialKvSnapshots);

    py::class_<ContextCacheConfig>(m, "ContextCacheConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &ContextCacheConfig::enabled)
        .def_readwrite("max_records", &ContextCacheConfig::maxRecords)
        .def_readwrite("recurrent_snapshot_pool_bytes", &ContextCacheConfig::recurrentSnapshotPoolBytes)
        .def_readwrite("partial_kv_snapshot_pool_bytes", &ContextCacheConfig::partialKvSnapshotPoolBytes);

    // ========================================================================
    // Image utilities
    // ========================================================================
    py::class_<imageUtils::ImageData>(m, "ImageData")
        .def(py::init<>())
        .def_readonly("width", &imageUtils::ImageData::width)
        .def_readonly("height", &imageUtils::ImageData::height)
        .def_readonly("channels", &imageUtils::ImageData::channels)
        .def_readonly("frames", &imageUtils::ImageData::frames)
        .def_readonly("is_video", &imageUtils::ImageData::isVideo)
        .def_readonly("timestamps", &imageUtils::ImageData::timestamps)
        .def_readwrite("fps", &imageUtils::ImageData::fps)
        .def_readwrite("do_resize", &imageUtils::ImageData::doResize);

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
            imageUtils::ImageData video = imageUtils::loadVideoFromFrames(framePaths, fps);
            video.timestamps = timestamps;
            return video;
        },
        py::arg("frame_paths"), py::arg("fps") = 1.0, py::arg("timestamps") = std::vector<double>{},
        py::call_guard<py::gil_scoped_release>(),
        "Load a video by stacking identically-sized image files into one (T, H, W, 3) ImageData");
    // No gil_scoped_release for load_video_from_array: its body reads the
    // py::array buffer, which requires the GIL.
    m.def("load_video_from_array", &loadVideoFromArray, py::arg("array"), py::arg("fps") = 1.0,
        py::arg("timestamps") = std::vector<double>{}, "Build a video ImageData from a (T, H, W, 3) uint8 numpy array");

    // ========================================================================
    // Audio utilities
    // ========================================================================
    py::class_<audioUtils::AudioData>(m, "AudioData")
        .def(py::init<>())
        .def_readwrite("sample_rate", &audioUtils::AudioData::sampleRate)
        .def_property_readonly(
            "num_samples",
            [](audioUtils::AudioData const& audio) {
                return audio.pcm ? static_cast<int64_t>(audio.pcm->samples.size()) : int64_t{0};
            },
            "Number of decoded PCM samples (0 when no PCM is attached)");

    m.def("load_audio_buffer_from_bytes", &loadAudioBufferFromBytes, py::arg("data"),
        py::call_guard<py::gil_scoped_release>(),
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
        .def(py::init([](std::string const& type, std::string const& content) {
            Message::MessageContent mc;
            mc.type = type;
            mc.content = content;
            return mc;
        }),
            py::arg("type"), py::arg("content"))
        .def_readwrite("type", &Message::MessageContent::type)
        .def_readwrite("content", &Message::MessageContent::content);

    py::class_<Message>(m, "Message")
        .def(py::init<>())
        .def(py::init([](std::string const& role, std::vector<Message::MessageContent> const& contents) {
            Message msg;
            msg.role = role;
            msg.contents = contents;
            return msg;
        }),
            py::arg("role"), py::arg("contents"))
        .def_readwrite("role", &Message::role)
        .def_readwrite("contents", &Message::contents);

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
        .def_readonly("token_id", &LogprobEntry::tokenId)
        .def_readonly("logprob", &LogprobEntry::logprob)
        .def_property_readonly("piece", [](LogprobEntry const& e) { return py::bytes(e.piece); });

    py::class_<LLMGenerationRequest::FormattedRequest>(m, "FormattedRequest")
        .def(py::init<>())
        .def_readwrite("formatted_system_prompt", &LLMGenerationRequest::FormattedRequest::formattedSystemPrompt)
        .def_readwrite("formatted_complete_request", &LLMGenerationRequest::FormattedRequest::formattedCompleteRequest);

    py::class_<LLMGenerationRequest::Request>(m, "Request")
        .def(py::init<>())
        .def(py::init([](std::vector<Message> const& messages) {
            LLMGenerationRequest::Request req;
            req.messages = messages;
            return req;
        }),
            py::arg("messages"))
        .def_readwrite("messages", &LLMGenerationRequest::Request::messages)
        .def_readwrite("image_buffers", &LLMGenerationRequest::Request::imageBuffers)
        .def_readwrite("audio_buffers", &LLMGenerationRequest::Request::audioBuffers)
        .def_readwrite("stop_strings", &LLMGenerationRequest::Request::stopStrings)
        .def_readwrite("logit_bias", &LLMGenerationRequest::Request::logitBias);

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

    py::class_<StreamChunk>(m, "StreamChunk")
        .def(py::init<>())
        .def_readonly("token_ids", &StreamChunk::tokenIds)
        .def_readonly("text", &StreamChunk::text)
        .def_readonly("prompt_token_count", &StreamChunk::promptTokenCount)
        .def_readonly("finished", &StreamChunk::finished)
        .def_readonly("reason", &StreamChunk::reason)
        .def_readonly("logprobs", &StreamChunk::logprobs);

    py::class_<StreamChannel, std::shared_ptr<StreamChannel>>(m, "StreamChannel")
        .def_static("create", &StreamChannel::create)
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
        .def_property_readonly(
            "pcm16", [](AudioChunk const& c) { return py::bytes(c.pcm16); }, "Little-endian int16 mono PCM samples")
        .def_readonly("is_final", &AudioChunk::isFinal)
        .def_readonly("num_frames", &AudioChunk::numFrames);

    py::class_<AudioStreamChannel, std::shared_ptr<AudioStreamChannel>>(m, "AudioStreamChannel")
        .def(py::init<>())
        .def("try_pop", [](AudioStreamChannel& self) { return self.waitPop(std::chrono::milliseconds{0}); })
        .def(
            "wait_pop",
            [](AudioStreamChannel& self, int64_t timeoutMs) {
                return self.waitPop(std::chrono::milliseconds{timeoutMs});
            },
            py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
        .def("is_finished", &AudioStreamChannel::isFinished)
        .def("is_cancelled", &AudioStreamChannel::isCancelled)
        .def("cancel", &AudioStreamChannel::cancel);

    py::class_<OmniAudioParams>(m, "OmniAudioParams")
        .def(py::init<>())
        .def_readwrite("talker_temperature", &OmniAudioParams::talkerTemperature)
        .def_readwrite("talker_top_k", &OmniAudioParams::talkerTopK)
        .def_readwrite("talker_top_p", &OmniAudioParams::talkerTopP)
        .def_readwrite("repetition_penalty", &OmniAudioParams::repetitionPenalty)
        .def_readwrite("max_audio_length", &OmniAudioParams::maxAudioLength)
        .def_readwrite("speaker_name", &OmniAudioParams::speakerName)
        .def_readwrite("codec_chunk_frames", &OmniAudioParams::codecChunkFrames)
        .def_readwrite("talker_prefill_threshold", &OmniAudioParams::talkerPrefillThreshold);

    // ========================================================================
    // Request / Response (continued)
    // ========================================================================
    py::enum_<ContextCacheLookupPolicy>(m, "ContextCacheLookupPolicy")
        .value("USE_CACHE", ContextCacheLookupPolicy::kUseCache)
        .value("BYPASS", ContextCacheLookupPolicy::kBypass);

    py::enum_<ContextCacheCommitPolicy>(m, "ContextCacheCommitPolicy")
        .value("INCLUDING_GENERATED_TOKENS", ContextCacheCommitPolicy::kIncludingGeneratedTokens)
        .value("PREFILL_STATE_ONLY", ContextCacheCommitPolicy::kPrefillStateOnly);

    py::class_<LLMGenerationRequest>(m, "LLMGenerationRequest")
        .def(py::init<>())
        .def_readwrite("requests", &LLMGenerationRequest::requests)
        .def_readwrite("formatted_requests", &LLMGenerationRequest::formattedRequests)
        .def_readwrite("temperature", &LLMGenerationRequest::temperature)
        .def_readwrite("top_p", &LLMGenerationRequest::topP)
        .def_readwrite("top_k", &LLMGenerationRequest::topK)
        .def_readwrite("max_generate_length", &LLMGenerationRequest::maxGenerateLength)
        .def_readwrite("lora_weights_name", &LLMGenerationRequest::loraWeightsName)
        .def_readwrite("save_system_prompt_kv_cache", &LLMGenerationRequest::saveSystemPromptKVCache)
        .def_readwrite("apply_chat_template", &LLMGenerationRequest::applyChatTemplate)
        .def_readwrite("add_generation_prompt", &LLMGenerationRequest::addGenerationPrompt)
        .def_readwrite("enable_thinking", &LLMGenerationRequest::enableThinking)
        .def_readwrite("disable_spec_decode", &LLMGenerationRequest::disableSpecDecode)
        .def_readwrite("recurrent_capture_interval", &LLMGenerationRequest::recurrentCaptureInterval)
        .def_readwrite("stream_channels", &LLMGenerationRequest::streamChannels)
        .def_readwrite("num_logprobs", &LLMGenerationRequest::numLogprobs)
        .def_readwrite("context_cache_lookup_policy", &LLMGenerationRequest::contextCacheLookupPolicy)
        .def_readwrite("context_cache_commit_policy", &LLMGenerationRequest::contextCacheCommitPolicy)
        .def_readwrite("context_cache_replay_tail_length", &LLMGenerationRequest::contextCacheReplayTailLength);

    py::class_<LLMGenerationResponse>(m, "LLMGenerationResponse")
        .def(py::init<>())
        .def_readwrite("output_ids", &LLMGenerationResponse::outputIds)
        .def_readwrite("output_texts", &LLMGenerationResponse::outputTexts)
        .def_readwrite("logprobs", &LLMGenerationResponse::logprobs)
        .def_readonly("finish_reasons", &LLMGenerationResponse::finishReasons)
        .def_readonly("prompt_token_counts", &LLMGenerationResponse::inputTokenCounts);

    // ========================================================================
    // Runtime: unified (vanilla + Eagle speculative decoding)
    // ========================================================================
    py::class_<PyLLMRuntime>(m, "LLMRuntime", "Unified LLM inference runtime with optional speculative decoding.")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 std::string const&, ContextCacheConfig const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{},
            py::arg("checkpoint_dir") = "", py::arg("context_cache_config") = ContextCacheConfig{},
            "Construct for vanilla (non-speculative) decoding")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 int32_t, int32_t, int32_t, std::string const&, std::string const&, ContextCacheConfig const&,
                 int32_t>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir"), py::arg("lora_weights_map"),
            py::arg("draft_top_k"), py::arg("draft_step"), py::arg("verify_tree_size"), py::arg("checkpoint_dir") = "",
            py::arg("draft_checkpoint_dir") = "", py::arg("context_cache_config") = ContextCacheConfig{},
            py::arg("dflash_block_size") = 0, "Construct for speculative decoding")
        .def("handle_request", &PyLLMRuntime::handleRequest, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(), "Process a generation request and return the response")
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
        .def("save_system_prompt_kv_cache", &PyLLMRuntime::saveSystemPromptKVCache, py::arg("prompt"),
            py::arg("lora_weights_name") = "", "Pre-generate and cache system prompt KV cache")
        .def("has_draft_model", &PyLLMRuntime::hasDraftModel, "Check if speculative decoding draft model is loaded")
        .def("get_prefill_metrics", &PyLLMRuntime::getPrefillMetrics, py::return_value_policy::reference_internal)
        .def("get_generation_metrics", &PyLLMRuntime::getGenerationMetrics, py::return_value_policy::reference_internal)
        .def("get_spec_decode_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::return_value_policy::reference_internal)
        .def("get_eagle_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::return_value_policy::reference_internal) // deprecated alias
        .def("get_multimodal_metrics", &PyLLMRuntime::getMultimodalMetrics)
        .def("get_context_cache_metrics", &PyLLMRuntime::getContextCacheMetrics);

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
        .def_readonly("tokens", &NemotronAsrRuntime::Result::tokens)
        .def_readonly("text", &NemotronAsrRuntime::Result::text)
        .def_readonly("num_mel_frames", &NemotronAsrRuntime::Result::numMelFrames)
        .def_readonly("num_encoder_frames", &NemotronAsrRuntime::Result::numEncoderFrames)
        .def_readonly("num_decode_steps", &NemotronAsrRuntime::Result::numDecodeSteps);

    py::class_<PyNemotronAsrRuntime>(m, "NemotronAsrRuntime")
        .def(py::init<std::string const&, std::string const&>(), py::arg("engine_dir"), py::arg("tokenizer_dir") = "",
            py::call_guard<py::gil_scoped_release>())
        .def("transcribe", &PyNemotronAsrRuntime::transcribe, py::arg("audio_bytes"), py::arg("prompt_id") = -1,
            py::call_guard<py::gil_scoped_release>())
        .def("default_prompt_id", &PyNemotronAsrRuntime::defaultPromptId)
        .def_property_readonly("max_mel_frames", &PyNemotronAsrRuntime::maxMelFrames);
#endif

    // ========================================================================
    // Builder: LLM
    // ========================================================================
    py::class_<builder::LLMBuilderConfig>(
        m, "LLMBuilderConfig", "Configuration for building TensorRT LLM engines from ONNX.")
        .def(py::init<>())
        .def_readwrite("max_input_len", &builder::LLMBuilderConfig::maxInputLen)
        .def_readwrite("spec_draft", &builder::LLMBuilderConfig::specDraft)
        .def_readwrite("spec_base", &builder::LLMBuilderConfig::specBase)
        .def_readwrite("eagle_draft", &builder::LLMBuilderConfig::specDraft) // deprecated alias
        .def_readwrite("eagle_base", &builder::LLMBuilderConfig::specBase)   // deprecated alias
        .def_readwrite("max_batch_size", &builder::LLMBuilderConfig::maxBatchSize)
        .def_readwrite("max_lora_rank", &builder::LLMBuilderConfig::maxLoraRank)
        .def_readwrite("max_kv_cache_capacity", &builder::LLMBuilderConfig::maxKVCacheCapacity)
        .def_readwrite("max_verify_tree_size", &builder::LLMBuilderConfig::maxVerifyTreeSize)
        .def_readwrite("max_draft_tree_size", &builder::LLMBuilderConfig::maxDraftTreeSize)
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
        .def_readwrite("min_image_tokens", &builder::VisualBuilderConfig::minImageTokens)
        .def_readwrite("max_image_tokens", &builder::VisualBuilderConfig::maxImageTokens)
        .def_readwrite("max_image_tokens_per_image", &builder::VisualBuilderConfig::maxImageTokensPerImage)
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
        .def_readwrite("min_time_steps", &builder::AudioBuilderConfig::minTimeSteps)
        .def_readwrite("max_time_steps", &builder::AudioBuilderConfig::maxTimeSteps)
        .def_readwrite("use_trt_native_audio_attn", &builder::AudioBuilderConfig::useTrtNativeAudioAttn)
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
            std::string const& loraWeightsName, bool saveSystemPromptKvCache, bool disableSpecDecode,
            std::unordered_map<int32_t, float> const& logitBias, int32_t numLogprobs) {
            LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = topP;
            request.topK = topK;
            request.maxGenerateLength = maxGenerateLength;
            request.applyChatTemplate = applyChatTemplate;
            request.addGenerationPrompt = addGenerationPrompt;
            request.enableThinking = enableThinking;
            request.loraWeightsName = loraWeightsName;
            request.saveSystemPromptKVCache = saveSystemPromptKvCache;
            request.disableSpecDecode = disableSpecDecode;
            request.numLogprobs = numLogprobs;

            for (auto const& messages : batchMessages)
            {
                LLMGenerationRequest::Request req;
                req.messages = messages;
                req.logitBias = logitBias;
                request.requests.push_back(std::move(req));
            }
            return request;
        },
        py::arg("batch_messages"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f, py::arg("top_k") = 50,
        py::arg("max_generate_length") = 256, py::arg("apply_chat_template") = true,
        py::arg("add_generation_prompt") = true, py::arg("enable_thinking") = false, py::arg("lora_weights_name") = "",
        py::arg("save_system_prompt_kv_cache") = false, py::arg("disable_spec_decode") = false,
        py::arg("logit_bias") = std::unordered_map<int32_t, float>{}, py::arg("num_logprobs") = 0,
        "Create a generation request from a batch of message lists.");
}
