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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "builder/llmBuilder.h"
#include "builder/visualBuilder.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "runtime/audioLoader.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/melSpectrogram.h"

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
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

//! Unified Python wrapper for LLMInferenceRuntime.
//! Supports both vanilla decoding (no draft model) and Eagle speculative decoding
//! through constructor overloading — mirrors the C++ unified runtime.
class PyLLMRuntime
{
public:
    //! Vanilla constructor (no speculative decoding).
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        ContextCacheConfig const& contextCacheConfig)
    {
        mPluginHandle = loadEdgellmPluginLib();
        mRuntime = std::make_unique<LLMInferenceRuntime>(
            engineDir, multimodalEngineDir, loraWeightsMap, mStream.get(), contextCacheConfig);
    }

    //! Eagle speculative decoding constructor.
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, int32_t draftTopK, int32_t draftStep,
        int32_t verifyTreeSize, ContextCacheConfig const& contextCacheConfig)
    {
        mPluginHandle = loadEdgellmPluginLib();
        SpecDecodeDraftingConfig draftingConfig{draftTopK, draftStep, verifyTreeSize};
        mRuntime = std::make_unique<LLMInferenceRuntime>(
            engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, mStream.get(), contextCacheConfig);
    }

    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool const success = mRuntime->handleRequest(request, response, mStream.get());
        ELLM_CHECK(success, "Failed to handle generation request");
        return response;
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

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
};

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
audioUtils::AudioData loadAudioBufferFromBytes(py::bytes data)
{
    constexpr int32_t kTargetSampleRate = 16000;
    std::string dataStr = data;
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
imageUtils::ImageData loadVideoFromArray(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const& array, double fps)
{
    py::buffer_info info = array.request();
    check::check(info.ndim == 4, "video array must be 4D (T, H, W, 3)");
    check::check(info.shape[3] == 3, "video array must have 3 channels (last dim == 3)");
    check::check(fps > 0.0, "fps must be positive");
    int64_t const T = info.shape[0];
    int64_t const H = info.shape[1];
    int64_t const W = info.shape[2];
    int64_t const C = info.shape[3];
    check::check(T > 0, "video array must have at least one frame (T > 0)");

    rt::Tensor stacked(
        {T, H, W, C}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "pybind::loadVideoFromArray::stacked");
    std::memcpy(stacked.dataPointer<unsigned char>(), info.ptr, static_cast<size_t>(T * H * W * C));

    imageUtils::ImageData video(std::move(stacked));
    video.fps = fps;
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
        .def("get_total_runs", [](metrics::LLMPrefillMetrics const& metrics) { return metrics.getTotalRuns(); });

    py::class_<metrics::LLMGenerationMetrics>(m, "LLMGenerationMetrics")
        .def_readonly("generated_tokens", &metrics::LLMGenerationMetrics::generatedTokens)
        .def("get_total_runs", [](metrics::LLMGenerationMetrics const& metrics) { return metrics.getTotalRuns(); });

    py::class_<metrics::SpecDecodeGenerationMetrics>(m, "SpecDecodeGenerationMetrics")
        .def_readonly("total_iterations", &metrics::SpecDecodeGenerationMetrics::totalIterations)
        .def_readonly("total_generated_tokens", &metrics::SpecDecodeGenerationMetrics::totalGeneratedTokens)
        .def_readonly("total_accepted_tokens", &metrics::SpecDecodeGenerationMetrics::totalAcceptedTokens)
        .def("get_total_runs",
            [](metrics::SpecDecodeGenerationMetrics const& metrics) { return metrics.getTotalRuns(); });

    py::class_<metrics::MultimodalMetrics>(m, "MultimodalMetrics")
        .def_readonly("total_images", &metrics::MultimodalMetrics::totalImages)
        .def_readonly("total_image_tokens", &metrics::MultimodalMetrics::totalImageTokens)
        .def("get_total_runs", [](metrics::MultimodalMetrics const& metrics) { return metrics.getTotalRuns(); });

    // ========================================================================
    // Image utilities
    // ========================================================================
    py::class_<imageUtils::ImageData>(m, "ImageData")
        .def(py::init<>())
        .def_readonly("width", &imageUtils::ImageData::width)
        .def_readonly("height", &imageUtils::ImageData::height)
        .def_readonly("channels", &imageUtils::ImageData::channels)
        .def_readonly("frames", &imageUtils::ImageData::frames)
        .def_readwrite("fps", &imageUtils::ImageData::fps);

    m.def("load_image_from_path", &loadImageFromPath, py::arg("path"), "Load image from file path");
    m.def("load_image_from_bytes", &loadImageFromBytes, py::arg("data"), "Load image from bytes");
    m.def("load_video_from_paths", &imageUtils::loadVideoFromFrames, py::arg("frame_paths"), py::arg("fps") = 1.0,
        "Load a video by stacking identically-sized image files into one (T, H, W, 3) ImageData");
    m.def("load_video_from_array", &loadVideoFromArray, py::arg("array"), py::arg("fps") = 1.0,
        "Build a video ImageData from a (T, H, W, 3) uint8 numpy array");

    // ========================================================================
    // Audio utilities
    // ========================================================================
    py::class_<audioUtils::AudioData>(m, "AudioData")
        .def(py::init<>())
        .def_readwrite("sample_rate", &audioUtils::AudioData::sampleRate);

    m.def("load_audio_buffer_from_bytes", &loadAudioBufferFromBytes, py::arg("data"),
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

    py::enum_<CommitPolicy>(m, "ContextCacheCommitPolicy")
        .value("INCLUDING_GENERATED_TOKENS", CommitPolicy::kIncludingGeneratedTokens)
        .value("PREFILL_STATE_ONLY", CommitPolicy::kPrefillStateOnly);

    py::class_<ContextCacheConfig>(m, "ContextCacheConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &ContextCacheConfig::enabled)
        .def_readwrite("max_records", &ContextCacheConfig::maxRecords)
        .def_readwrite("recurrent_snapshot_pool_bytes", &ContextCacheConfig::recurrentSnapshotPoolBytes)
        .def_readwrite("partial_kv_snapshot_pool_bytes", &ContextCacheConfig::partialKvSnapshotPoolBytes)
        .def_readwrite("media_artifact_pool_bytes", &ContextCacheConfig::mediaArtifactPoolBytes)
        .def_readwrite("max_media_artifacts", &ContextCacheConfig::maxMediaArtifacts);

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
    // Request / Response (continued)
    // ========================================================================
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
        .def_readwrite("disable_context_reuse", &LLMGenerationRequest::disableContextReuse)
        .def_readwrite("context_cache_isolation_key", &LLMGenerationRequest::contextCacheIsolationKey)
        .def_readwrite("context_cache_commit_policy", &LLMGenerationRequest::contextCacheCommitPolicy)
        .def_readwrite("recurrent_capture_interval", &LLMGenerationRequest::recurrentCaptureInterval)
        .def_readwrite("context_cache_replay_tail_length", &LLMGenerationRequest::contextCacheReplayTailLength)
        .def_readwrite("stream_channels", &LLMGenerationRequest::streamChannels)
        .def_readwrite("num_logprobs", &LLMGenerationRequest::numLogprobs);

    py::class_<LLMGenerationResponse>(m, "LLMGenerationResponse")
        .def(py::init<>())
        .def_readwrite("output_ids", &LLMGenerationResponse::outputIds)
        .def_readwrite("output_texts", &LLMGenerationResponse::outputTexts)
        .def_readwrite("logprobs", &LLMGenerationResponse::logprobs)
        .def_readonly("finish_reasons", &LLMGenerationResponse::finishReasons);

    // ========================================================================
    // Runtime: unified (vanilla + Eagle speculative decoding)
    // ========================================================================
    py::class_<PyLLMRuntime>(m, "LLMRuntime",
        "Unified LLM inference runtime. Supports both vanilla decoding and Eagle speculative decoding.")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 ContextCacheConfig const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{},
            py::arg("context_cache_config") = ContextCacheConfig{}, "Construct for vanilla (non-speculative) decoding")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 int32_t, int32_t, int32_t, ContextCacheConfig const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir"), py::arg("lora_weights_map"),
            py::arg("draft_top_k"), py::arg("draft_step"), py::arg("verify_tree_size"),
            py::arg("context_cache_config") = ContextCacheConfig{}, "Construct for Eagle speculative decoding")
        .def("handle_request", &PyLLMRuntime::handleRequest, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(), "Process a generation request and return the response")
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
        .def("get_multimodal_metrics", &PyLLMRuntime::getMultimodalMetrics);

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
        .def_readwrite("max_kv_pool_pages", &builder::LLMBuilderConfig::maxKVPoolPages)
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
