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

#include "multimodal/qwen3_omni/audioRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "multimodal/common/audioUtils.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{
namespace
{
bool prepareSeqlensInput(rt::Tensor& seqlensTensor, nvinfer1::IExecutionContext& context, char const* bindingName,
    rt::Tensor const& seqlensHost, int64_t seqlensSize, int64_t seqlensSizeInBytes, cudaStream_t stream,
    char const* tensorLabel)
{
    if (!seqlensTensor.reshape({seqlensSize}))
    {
        LOG_ERROR("Failed to reshape %s", tensorLabel);
        return false;
    }

    if (!context.setInputShape(bindingName, seqlensTensor.getShape().getTRTDims()))
    {
        LOG_ERROR("Failed to set %s input shape", tensorLabel);
        return false;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        seqlensTensor.rawPointer(), seqlensHost.rawPointer(), seqlensSizeInBytes, cudaMemcpyHostToDevice, stream));

    return true;
}
} // namespace

Qwen3OmniAudioRunner::Qwen3OmniAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    // Initialize audio encoder (engineDir is already the audio subdirectory)
    std::string audioEnginePath = engineDir + "/audio_encoder.engine";
    if (std::filesystem::exists(audioEnginePath))
    {
        LOG_INFO("Loading audio encoder from %s", audioEnginePath.c_str());
        try
        {
            mAudioEngine = deserializeCudaEngineFromFile(*mRuntime, audioEnginePath);

            mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
                mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
            ELLM_CHECK(mAudioContext, "Failed to create audio encoder context");

            bool const profileSet = mAudioContext->setOptimizationProfileAsync(0, stream);
            ELLM_CHECK(profileSet, "Failed to set optimization profile for audio encoder");

            setNonBlockingAuxStreams(mAudioContext.get(), mAudioEngine.get(), mAuxStreams);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to load audio encoder: %s", e.what());
            throw;
        }
    }
    else
    {
        throw std::runtime_error("Audio encoder not found at " + audioEnginePath);
    }

    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    // Best-effort init of the online GPU fbank. Failure is non-fatal: the
    // runner falls back to the CPU MelExtractor (mFeMel, already bound by
    // validateAndFillConfig above) for PCM→mel. initFbankResources issues H2D
    // copies via CUDA_CHECK, which throws on failure (e.g. device OOM), so
    // catch here to keep construction non-fatal.
    try
    {
        mFbankReady = initFbankResources(stream);
        if (!mFbankReady)
        {
            // initFbankResources already logged the specific reason; add the consequence.
            LOG_WARNING("Online GPU fbank unavailable; using CPU MelExtractor for PCM→mel.");
        }
    }
    catch (std::exception const& e)
    {
        mFbankReady = false;
        LOG_WARNING("Online GPU fbank init failed (%s); using CPU MelExtractor for PCM→mel.", e.what());
    }

    LOG_INFO("Qwen3OmniAudioRunner initialized successfully");
}

bool Qwen3OmniAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    // Read config.json
    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    // Parse audio configuration
    if (jsonConfig.contains("audio_config"))
    {
        auto audioConfig = jsonConfig["audio_config"];
        mConfig.melBins = audioConfig.value("num_mel_bins", 128);
        mConfig.audioFeatureDim = audioConfig.value("output_dim", 2560);
        mConfig.nWindow = audioConfig.value("n_window", 50);
        mConfig.nWindowInfer = audioConfig.value("n_window_infer", 200);
    }
    else
    {
        LOG_WARNING("audio_config not found in config.json, using default values");
    }

    // Convolutional downsample-stack depth — derived from the top-level model_type.
    // Qwen3-Omni / Qwen3-ASR use 3 stride-2 Conv2D stages; Qwen3-Next Omni adds a 4th stage
    // (``conv2d4`` in HF ``Qwen3OmniNextAudioEncoder``). Defaulting to 3 keeps existing
    // exports unchanged.
    {
        std::string const topLevelModelType = jsonConfig.value("model_type", "");
        if (topLevelModelType == "qwen3_omni_next_audio_encoder")
        {
            mConfig.numConvDownsampleStages = 4;
        }
        else
        {
            mConfig.numConvDownsampleStages = 3;
        }
        LOG_INFO("Audio encoder: model_type='%s', conv downsample stages=%d", topLevelModelType.c_str(),
            mConfig.numConvDownsampleStages);
    }

    // Parse audio special token IDs from top-level config (may differ between Qwen3-Omni and Qwen3-ASR)
    if (jsonConfig.contains("audio_token_id"))
    {
        mConfig.audioTokenId = jsonConfig["audio_token_id"].get<int32_t>();
    }
    LOG_DEBUG("Audio token IDs: audio_pad=%d", mConfig.audioTokenId);

    // Parse rope_theta for MRope initialization (from text_config or top-level)
    if (jsonConfig.contains("text_config") && jsonConfig["text_config"].contains("rope_theta"))
    {
        mConfig.mropeTheta = jsonConfig["text_config"]["rope_theta"].get<float>();
    }
    else if (jsonConfig.contains("rope_theta"))
    {
        mConfig.mropeTheta = jsonConfig["rope_theta"].get<float>();
    }

    // Pick mel feature-extractor family from the audio engine's top-level
    // ``model_type`` (export.py writes ``qwen3_asr_thinker`` /
    // ``qwen3_omni_audio_encoder`` here). Runner owns mel extraction;
    // callers hand off raw PCM.
    std::string const audioModelType = jsonConfig.value("model_type", "");
    if (audioModelType == "qwen3_asr_thinker" || audioModelType == "qwen3_omni_audio_encoder"
        || audioModelType == "qwen3_omni_next_audio_encoder")
    {
        mFeMel = rt::audio::makeWhisperExtractor();
    }
    else
    {
        LOG_ERROR(
            "Unsupported audio model_type %s for Qwen3OmniAudioRunner (expected qwen3_asr_thinker / "
            "qwen3_omni_audio_encoder / qwen3_omni_next_audio_encoder)",
            audioModelType.c_str());
        return false;
    }

    return true;
}

bool Qwen3OmniAudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    if (!mAudioEngine || !mAudioContext)
    {
        LOG_ERROR("Cannot allocate buffers - audio encoder not loaded");
        return false;
    }

    // Get max shapes from optimization profile using tensor names directly
    nvinfer1::Dims paddedFeaturesShapeMax
        = mAudioEngine->getProfileShape(binding_names::kAudioPaddedFeatures, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims paddedMaskIndicesShapeMax
        = mAudioEngine->getProfileShape(binding_names::kAudioPaddedMaskIndices, 0, nvinfer1::OptProfileSelector::kMAX);

    // Extract dimensions
    int64_t const melBins = mConfig.melBins;
    int64_t const audioFeatureDim = mConfig.audioFeatureDim;       // Use config value, not output shape
    int64_t const maxAudioTokens = paddedMaskIndicesShapeMax.d[0]; // Max attention elements = max audio tokens

    // Allocate tensors
    int64_t const maxNumChunks = paddedFeaturesShapeMax.d[0];
    int64_t const maxChunkLen = paddedFeaturesShapeMax.d[2];
    mPaddedFeature = rt::Tensor({maxNumChunks, melBins, maxChunkLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    int64_t const maxLenAfterCNN = (maxChunkLen - 1) / 2 + 1;
    mPaddedMaskAfterCNN = rt::Tensor({maxNumChunks, maxLenAfterCNN}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);

    int64_t const maxValidElements = paddedMaskIndicesShapeMax.d[0];
    mPaddedMaskIndices = rt::Tensor({maxValidElements, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);

    // Sized for one full-length clip; preprocess() grows it via
    // resizeEmbeddingForRows() when a batch packs more clips.
    mAudioEmbedding = rt::Tensor({maxAudioTokens, audioFeatureDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    int64_t maxSeqlensHostCapacity = 0;
    mHasCuSeqlens = isEngineInput(*mAudioEngine, binding_names::kCuSeqlens);
    if (mHasCuSeqlens)
    {
        nvinfer1::Dims const cuSeqlensShapeMax
            = mAudioEngine->getProfileShape(binding_names::kCuSeqlens, 0, nvinfer1::OptProfileSelector::kMAX);
        int64_t const maxCuSeqlens = cuSeqlensShapeMax.d[0];
        maxSeqlensHostCapacity = std::max(maxSeqlensHostCapacity, maxCuSeqlens);
        mCuSeqlens = rt::Tensor(
            {maxCuSeqlens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Qwen3OmniAudioRunner::mCuSeqlens");
        if (!mAudioContext->setTensorAddress(binding_names::kCuSeqlens, mCuSeqlens.rawPointer()))
        {
            LOG_ERROR("Failed to set cu_seqlens input address");
            return false;
        }
    }

    mHasKvLengths = isEngineInput(*mAudioEngine, binding_names::kKvLengths);
    if (mHasKvLengths)
    {
        nvinfer1::Dims const kvLengthsShapeMax
            = mAudioEngine->getProfileShape(binding_names::kKvLengths, 0, nvinfer1::OptProfileSelector::kMAX);
        int64_t const maxKvLengths = kvLengthsShapeMax.d[0];
        maxSeqlensHostCapacity = std::max(maxSeqlensHostCapacity, maxKvLengths);
        mKvLengths = rt::Tensor(
            {maxKvLengths}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Qwen3OmniAudioRunner::mKvLengths");
        if (!mAudioContext->setTensorAddress(binding_names::kKvLengths, mKvLengths.rawPointer()))
        {
            LOG_ERROR("Failed to set kv_lengths input address");
            return false;
        }
    }

    if (maxSeqlensHostCapacity > 0)
    {
        mCuSeqlensHost = rt::Tensor({maxSeqlensHostCapacity}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
            "Qwen3OmniAudioRunner::mCuSeqlensHost");
    }

    return true;
}

bool Qwen3OmniAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, [[maybe_unused]] bool imageOnly,
    [[maybe_unused]] bool skipEncoderWork)
{
    if (!mropeCosSinOut.has_value())
    {
        LOG_ERROR("mropeCosSinOut is required (this runner is MRope-only).");
        return false;
    }

    std::vector<int64_t> audioTokenLengths;

    // Pass 1: derive every clip's encoded length without touching the encoder, so
    // mAudioEmbedding can be sized for the whole batch before anything is written.
    // The multimodal indices generated downstream are numbered across the batch and
    // address this buffer directly, so all clips have to stay resident side by side.
    struct ClipPlan
    {
        rt::audio::AudioPCM const* pcm{nullptr}; //!< Non-null when the GPU fbank serves this clip
        rt::Tensor hostMel{};                    //!< CPU-extracted mel, kept so pass 2 need not redo it
        int64_t numTokens{0};
    };
    std::vector<ClipPlan> plans;
    int64_t totalRows = 0;
    for (auto const& req : request.requests)
    {
        // textPreprocess expands *every* <|audio_pad|> in a request to this request's
        // entry, so a multi-clip request would repeat the per-request total at each
        // placeholder. Supporting it needs per-clip lengths threaded through
        // textPreprocess; reject it here rather than mis-expand silently.
        if (req.audioBuffers.size() > 1)
        {
            LOG_ERROR("At most one audio clip per request is supported; got %zu", req.audioBuffers.size());
            return false;
        }

        int64_t requestRows = 0;
        for (auto const& audio : req.audioBuffers)
        {
            if (!audio.pcm)
            {
                LOG_ERROR(
                    "AudioData.pcm is null; populate via load_audio_buffer_from_bytes "
                    "(server) or requestFileParser (CLI).");
                return false;
            }
            ClipPlan plan;
            int32_t numFrames = 0;
            if (gpuFbankViable(*audio.pcm, numFrames))
            {
                plan.pcm = audio.pcm.get();
            }
            else if (mFeMel.extract(*audio.pcm, plan.hostMel))
            {
                numFrames = static_cast<int32_t>(plan.hostMel.getShape()[1]);
            }
            else
            {
                LOG_ERROR("Mel extraction failed");
                return false;
            }
            plan.numTokens = audioUtils::computeFeatExtractOutputLength(
                numFrames, mConfig.nWindow, mConfig.numConvDownsampleStages);
            requestRows += plan.numTokens;
            plans.push_back(std::move(plan));
        }
        audioTokenLengths.push_back(requestRows);
        totalRows += requestRows;
    }

    if (!resizeEmbeddingForRows(totalRows))
    {
        LOG_ERROR("Failed to size mAudioEmbedding for %ld rows", static_cast<long>(totalRows));
        return false;
    }

    // Pass 2: encode each clip into the row slot pass 1 reserved for it.
    int64_t rowOffset = 0;
    for (auto& plan : plans)
    {
        rt::Tensor melSpec;
        if (plan.pcm != nullptr && tryOnlineGpuFbank(*plan.pcm, melSpec, stream))
        {
            // GPU mel produced in place.
        }
        else
        {
            if (plan.pcm != nullptr)
            {
                // Pass 1 only checked the gate, so the fbank can still fail here. The CPU
                // MelExtractor's Whisper framing reduces to the same recurrence
                // computeNumMelFrames used in pass 1, so the row reservation still holds
                // (and encodeClip cross-checks it either way).
                LOG_WARNING("Online GPU fbank failed; falling back to CPU MelExtractor for this clip.");
                if (!mFeMel.extract(*plan.pcm, plan.hostMel))
                {
                    LOG_ERROR("Mel extraction failed");
                    return false;
                }
            }
            if (!audioUtils::uploadHostMelFp32ToFp16Gpu(plan.hostMel, melSpec, stream, "Qwen3OmniAudioRunner::mel"))
            {
                return false;
            }
        }
        if (!encodeClip(melSpec, rowOffset, plan.numTokens, stream))
        {
            return false;
        }
        rowOffset += plan.numTokens;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 2: Tokenize and replace audio tokens (similar to QwenViTRunner::textPreprocess)
    textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);

    // Step 3: Initialize sequential MRope cache if applicable.
    // For audio+text only (no vision), all 3 MRope dimensions (T, H, W) use identical sequential positions.
    // When a vision runner is also present, QwenViTRunner::preprocess will overwrite the MRope cache
    // with vision-aware position IDs, so this initialization is harmlessly overwritten.
    int64_t const activeBatchSize = static_cast<int64_t>(request.requests.size());
    if (!initializeSequentialMRopeCache(activeBatchSize, mropeCosSinOut.value().get(), stream))
    {
        LOG_ERROR("Failed to initialize sequential MRope cache for audio input.");
        return false;
    }

    mLastMediaTokenLengths = audioTokenLengths;
    return true;
}

void Qwen3OmniAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(audioTokenLengths.size() == request.requests.size(),
        "audioTokenLengths.size() != request.requests.size(), " + std::to_string(audioTokenLengths.size())
            + " != " + std::to_string(request.requests.size()));

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;

        // Check if already tokenized (incremental mode)
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            // Already tokenized by another runner, use existing tokens
            ids = batchInputIds[i];
        }
        else
        {
            // First runner to process, tokenize the request
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        // Insert audio tokens if present
        if (audioTokenLengths[i] > 0)
        {
            std::vector<int32_t> newIds;
            for (size_t j = 0; j < ids.size(); ++j)
            {
                if (ids[j] == mConfig.audioTokenId)
                {
                    // Expand the <|audio_pad|> placeholder to N×<|audio_pad|> in place. The <|audio_start|>/
                    // <|audio_end|> markers come from the (unified) chat template, so we do NOT re-add them here.
                    int64_t numAudioTokens = audioTokenLengths[i];
                    for (int64_t k = 0; k < numAudioTokens; ++k)
                    {
                        newIds.push_back(mConfig.audioTokenId);
                    }
                }
                else
                {
                    newIds.push_back(ids[j]);
                }
            }

            // Update batchInputIds
            if (i < batchInputIds.size())
            {
                batchInputIds[i] = std::move(newIds);
            }
            else
            {
                batchInputIds.emplace_back(std::move(newIds));
            }
        }
        else
        {
            // No audio in this request, keep original tokens
            if (i < batchInputIds.size())
            {
                batchInputIds[i] = std::move(ids);
            }
            else
            {
                batchInputIds.emplace_back(std::move(ids));
            }
        }
    }
}

namespace
{

// The GPU fbank kernel suite hard-codes the Whisper choices (N=400 FFT, periodic
// Hann, reflect-pad, log10 + max-floor, whisper-clamp post-norm, [nMel,T] layout,
// drop-last-frame, no pre-emphasis). Validate the bound MelExtractor's config is
// Whisper-spec before enabling online fbank; any mismatch (e.g. a Parakeet
// extractor) cleanly falls back to the CPU path instead of silently producing a
// wrong spectrogram. The config is the single source of truth — these are the
// only places the "expected Whisper spec" literals live.
bool whisperFbankConfigOk(audio::MelExtractorConfig const& cfg, std::string& reason)
{
    auto fail = [&reason](char const* msg) {
        reason = msg;
        return false;
    };
    // Shape: the kernel FFT is N=400-specific and the encoder takes [1, 128, T].
    if (!(cfg.nFFT == 400 && cfg.winLength == cfg.nFFT && cfg.nMel == 128))
        return fail("needs nFFT==400, winLength==nFFT, nMel==128.");
    // Windowing / framing.
    if (!(cfg.windowType == audio::WindowType::kHannPeriodic && cfg.framePadding == audio::FramePadding::kCenterReflect
            && cfg.preemphCoeff == 0.0f))
        return fail("needs periodic Hann window, centre-reflect framing, no pre-emphasis.");
    // Log + post-normalize.
    if (!(cfg.logType == audio::LogType::kLog10 && cfg.logFloorMode == audio::LogFloorMode::kMax
            && cfg.postNormalize == audio::PostNormalize::kWhisperClamp))
        return fail("needs log10 + max-floor + Whisper-clamp post-normalize.");
    // Layout + dynamic T.
    if (!(cfg.layout == audio::MelLayout::kMelTime && cfg.dropLastStftFrame
            && cfg.timePadding == audio::TimePadding::kNone))
        return fail("needs [nMel, T] layout, drop-last-frame, no static time padding.");
    return true;
}

} // anonymous namespace

bool Qwen3OmniAudioRunner::initFbankResources(cudaStream_t stream)
{
    // The bound CPU MelExtractor's config is the single source of truth for every
    // STFT/mel param AND the mel filter weights. The GPU kernels hard-code the
    // Whisper choices, so validate the config is Whisper-spec before enabling —
    // otherwise fall back to CPU.
    audio::MelExtractorConfig const& cfg = mFeMel.config();
    std::string reason;
    if (!whisperFbankConfigOk(cfg, reason))
    {
        LOG_WARNING("Online GPU fbank disabled (extractor not Whisper-spec): %s", reason.c_str());
        return false;
    }
    int32_t const nFft = cfg.nFFT;
    int32_t const nMel = cfg.nMel;
    int32_t const nFreq = nFft / 2 + 1;

    // Single source of truth for the mel filter: reuse the exact Slaney weights
    // the CPU MelExtractor (mFeMel) was built with, so the GPU fbank and the CPU
    // fallback can never drift apart. No external mel_filter.bin to ship or keep
    // in sync. buildMelFilterBank already emits the [nMel, nBins] (nMel-major,
    // freq-contiguous) layout the GEMM A-matrix wants, so this is a pure FP16
    // cast + K-pad — no transpose.
    std::vector<float> const& melFilterHost = mFeMel.melFilterBank(); // [nMel, nBins] row-major F32
    if (melFilterHost.size() != static_cast<size_t>(nMel) * nFreq)
    {
        LOG_WARNING("Online fbank disabled: mel filter size %zu != %d x %d.", melFilterHost.size(), nMel, nFreq);
        return false;
    }

    // Single source of truth for the window: reuse the exact periodic-Hann taps
    // the CPU MelExtractor built (buildWindow), so the GPU fbank and the CPU
    // fallback can never drift. The config gate guarantees winLength == nFft.
    std::vector<float> const& windowHost = mFeMel.window();
    if (windowHost.size() != static_cast<size_t>(nFft))
    {
        LOG_WARNING("Online GPU fbank disabled: window size %zu != nFft=%d.", windowHost.size(), nFft);
        return false;
    }

#ifdef CUTE_DSL_GEMM_ENABLED
    // Idempotent + thread-safe; returns false (with its own LOG_ERROR) when no
    // GEMM variant is compiled for the current SM, so online fbank fails early
    // here rather than dispatching to an unloaded module at the first GEMM call.
    if (!CuteDslGemmRunner::loadKernelModule())
    {
        LOG_ERROR("CuteDslGemmRunner::loadKernelModule failed — online fbank GEMM unavailable on this device.");
        return false;
    }
#else
    LOG_WARNING(
        "Online GPU fbank disabled: built without CuTe DSL GEMM (rebuild with -DENABLE_CUTE_DSL=gemm). "
        "Using CPU MelExtractor.");
    return false;
#endif

    // Upper bounds from the engine kMAX profile, mirroring allocateBuffer: the
    // encoder consumes at most maxNumChunks × maxChunkLen mel frames, and
    // T_out == floor(N / hopLength) for the Whisper-gated geometry
    // (2 * padLength == nFft), so the largest PCM the GPU path accepts is the
    // largest N with T_out <= maxFrames. Longer clips exceed the encoder
    // profile on any path; tryOnlineGpuFbank routes them to the CPU fallback.
    nvinfer1::Dims const paddedFeaturesShapeMax
        = mAudioEngine->getProfileShape(binding_names::kAudioPaddedFeatures, 0, nvinfer1::OptProfileSelector::kMAX);
    int64_t const maxFrames = paddedFeaturesShapeMax.d[0] * paddedFeaturesShapeMax.d[2];
    if (maxFrames <= 0)
    {
        LOG_WARNING("Online GPU fbank disabled: engine kMAX profile gives non-positive max mel frames (%ld).",
            static_cast<long>(maxFrames));
        return false;
    }
    int64_t const maxPcmSamples = (maxFrames + 1) * cfg.hopLength - 1;
    int64_t const maxFramesFull = maxFrames + 1; // fbankWhisper frames T_full = T_out + 1
    int64_t const nPadMax = audioUtils::fbankNPad(static_cast<int32_t>(maxFrames));

    // Allocate every device buffer the online fbank path uses, sized at the
    // bounds above.
    mFbankResources.melFilterFp16Kmajor
        = rt::Tensor({nMel, audioUtils::kFbankKPad}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mFbankResources.hannWindow = rt::Tensor({nFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mFbankResources.fftTwiddle = rt::Tensor({nFft, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mFbankResources.framedF32 = rt::Tensor({maxFramesFull, nFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mFbankResources.magFp16
        = rt::Tensor({nPadMax, audioUtils::kFbankKPad}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mFbankResources.melPowerFp16 = rt::Tensor({nMel, nPadMax}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mFbankResources.maxLogScalar = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mPcmF32Device = rt::Tensor({maxPcmSamples}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    mMelSpecDevice = rt::Tensor({1, nMel, maxFrames}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Fill the persistent weight / table tensors (H2D). Mel filter: cast +
    // K-pad the Slaney weights into the [nMel, K_pad] F16 K-major GEMM
    // A-matrix layout.
    if (!audioUtils::fillMelFilterFp16Kmajor(
            melFilterHost.data(), nMel, nFreq, audioUtils::kFbankKPad, mFbankResources.melFilterFp16Kmajor, stream))
    {
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(mFbankResources.hannWindow.rawPointer(), windowHost.data(),
        windowHost.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    std::vector<float> twiddleHost;
    audioUtils::makeFftTwiddleHost(nFft, twiddleHost);
    CUDA_CHECK(cudaMemcpyAsync(mFbankResources.fftTwiddle.rawPointer(), twiddleHost.data(),
        twiddleHost.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Publish the scalar params derived from cfg — the single source of truth
    // for fbankWhisper / computeNumMelFrames (no duplicated constants downstream).
    mFbankResources.nFft = nFft;
    mFbankResources.hopLength = cfg.hopLength;
    mFbankResources.padLength = nFft / 2;
    mFbankResources.nMel = nMel;
    mFbankResources.nFreq = nFreq;
    mFbankResources.maxFrames = static_cast<int32_t>(maxFrames);
    mFbankResources.melFloor = cfg.logFloor;

    LOG_INFO(
        "Online GPU fbank ready (mel filter %d×%d Slaney reused from CPU MelExtractor → [%d, %d] F16 K-major; "
        "buffers pre-allocated for up to %ld mel frames).",
        nMel, nFreq, nMel, audioUtils::kFbankKPad, static_cast<long>(maxFrames));
    return true;
}

bool Qwen3OmniAudioRunner::gpuFbankViable(rt::audio::AudioPCM const& pcm, int32_t& numFramesOut) const
{
    // The CPU MelExtractor rejects a rate mismatch, so the GPU path must too, or
    // it would silently produce a wrong-but-plausible spectrogram.
    if (!mFbankReady || mConfig.melBins != mFbankResources.nMel || pcm.sampleRate != mFeMel.config().sampleRate)
    {
        return false;
    }

    // Clips too short for the GPU framing (the CPU MelExtractor clamps to >= 1
    // frame) and clips beyond the engine kMAX profile the fbank buffers were
    // pre-allocated for fall back to the CPU path without touching the GPU.
    int32_t const numFrames = audioUtils::computeNumMelFrames(static_cast<int64_t>(pcm.samples.size()),
        mFbankResources.nFft, mFbankResources.hopLength, mFbankResources.padLength);
    if (numFrames <= 0 || numFrames > mFbankResources.maxFrames)
    {
        return false;
    }
    numFramesOut = numFrames;
    return true;
}

bool Qwen3OmniAudioRunner::resizeEmbeddingForRows(int64_t rows)
{
    int64_t const hidden = mConfig.audioFeatureDim;
    if (rows * hidden * static_cast<int64_t>(sizeof(__half)) <= mAudioEmbedding.getMemoryCapacity())
    {
        return mAudioEmbedding.reshape({rows, hidden});
    }
    LOG_WARNING("Qwen3OmniAudioRunner: mAudioEmbedding reallocation at runtime (%ld rows)", static_cast<long>(rows));
    mAudioEmbedding = rt::Tensor({rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    return true;
}

bool Qwen3OmniAudioRunner::tryOnlineGpuFbank(rt::audio::AudioPCM const& pcm, rt::Tensor& melSpec, cudaStream_t stream)
{
    // Gate: online fbank ready (config validated Whisper-spec, params published
    // into mFbankResources), the engine mel width agrees, and the sample rate
    // matches what the kernels assume. The CPU MelExtractor rejects a rate
    // mismatch, so the GPU path must too, or it would silently produce a
    // wrong-but-plausible spectrogram.
    int32_t numFrames = 0;
    if (!gpuFbankViable(pcm, numFrames))
    {
        return false;
    }

    // Upload host FP32 PCM [-1, 1] into the pre-allocated [N] GPU staging tensor
    // (metadata-only reshape; the maxFrames gate above bounds N). pcm.samples is
    // owned by the request and outlives encodeClip, covering the async
    // fbank launches and the cudaStreamSynchronize at the end of preprocess().
    if (!audioUtils::uploadHostPcmF32ToGpu(pcm.samples, mPcmF32Device, stream))
    {
        return false;
    }

    // Non-owning view of the pre-allocated backing store at this clip's width.
    // A move-assignment into melSpec (the CPU fallback path) cannot free the
    // backing store, and every consumer of melSpec runs within this
    // preprocess() call — inside mMelSpecDevice's lifetime.
    melSpec = rt::Tensor(mMelSpecDevice.rawPointer(),
        {1, static_cast<int64_t>(mFbankResources.nMel), static_cast<int64_t>(numFrames)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    if (!audioUtils::fbankWhisper(mPcmF32Device, mFbankResources, melSpec, stream))
    {
        LOG_WARNING("Online GPU fbank failed; falling back to CPU MelExtractor for this clip.");
        return false;
    }
    return true;
}

bool Qwen3OmniAudioRunner::encodeClip(
    rt::Tensor const& melSpec, int64_t destRowOffset, int64_t expectedRows, cudaStream_t stream)
{
    if (!mAudioEngine || !mAudioContext)
    {
        LOG_ERROR("Audio encoder not loaded");
        return false;
    }

    int64_t const timeSteps = melSpec.getShape()[2];
    LOG_DEBUG("Mel-spectrogram shape: [%ld, %ld, %ld]", melSpec.getShape()[0], melSpec.getShape()[1], timeSteps);

    // Preprocess for audio encoder. ``numConvDownsampleStages`` selects the post-CNN
    // length recurrence (3 stages for Qwen3-Omni, 4 for Qwen3-Next Omni).
    std::vector<int64_t> afterCNNLens;
    if (!audioUtils::preprocessAudioForEncoder(melSpec, mConfig.nWindow, mPaddedFeature, mPaddedMaskAfterCNN,
            afterCNNLens, stream, mConfig.numConvDownsampleStages))
    {
        LOG_ERROR("Failed to preprocess audio for encoder");
        return false;
    }

    // Convert mask to indices
    if (!audioUtils::convertMaskToIndices(mPaddedMaskAfterCNN, mPaddedMaskIndices, stream))
    {
        LOG_ERROR("Failed to convert mask to indices");
        return false;
    }

    LOG_DEBUG("Mask shape: [%ld, %ld], Indices shape: [%ld, %ld]", mPaddedMaskAfterCNN.getShape()[0],
        mPaddedMaskAfterCNN.getShape()[1], mPaddedMaskIndices.getShape()[0], mPaddedMaskIndices.getShape()[1]);

    // Create attention mask with merged windows (matching PyTorch cu_seqlens logic)
    if (!audioUtils::createChunkwiseAttentionMask(
            afterCNNLens, mConfig.nWindow, mConfig.nWindowInfer, mAudioAttentionMask, stream))
    {
        LOG_ERROR("Failed to create attention mask");
        return false;
    }

    LOG_DEBUG(
        "Created attention mask [%ld, %ld]", mAudioAttentionMask.getShape()[0], mAudioAttentionMask.getShape()[1]);

    // Calculate total audio tokens
    int64_t const totalAudioTokens = mPaddedMaskIndices.getShape()[0];

    if (mHasCuSeqlens || mHasKvLengths)
    {
        // NOTE: Currently, audio encoder always runs at batch size of 1.
        // Thus, we always set the seqlens size to 2 and set values to {0, totalAudioTokens}.
        int64_t const seqlensSize = 2;
        int64_t const seqlensSizeInBytes = seqlensSize * static_cast<int64_t>(sizeof(int32_t));
        if (mCuSeqlensHost.getMemoryCapacity() < seqlensSizeInBytes)
        {
            LOG_ERROR("cu_seqlens host capacity too small: need=%ld bytes, capacity=%ld bytes", seqlensSizeInBytes,
                mCuSeqlensHost.getMemoryCapacity());
            return false;
        }

        if (!mCuSeqlensHost.reshape({seqlensSize}))
        {
            LOG_ERROR("Failed to reshape host cu_seqlens buffer");
            return false;
        }
        int32_t* seqlensData = mCuSeqlensHost.dataPointer<int32_t>();
        seqlensData[0] = 0;
        seqlensData[1] = static_cast<int32_t>(totalAudioTokens);

        if (mHasCuSeqlens)
        {
            if (!prepareSeqlensInput(mCuSeqlens, *mAudioContext, binding_names::kCuSeqlens, mCuSeqlensHost, seqlensSize,
                    seqlensSizeInBytes, stream, "cu_seqlens"))
            {
                return false;
            }
        }

        if (mHasKvLengths)
        {
            if (!prepareSeqlensInput(mKvLengths, *mAudioContext, binding_names::kKvLengths, mCuSeqlensHost, seqlensSize,
                    seqlensSizeInBytes, stream, "kv_lengths"))
            {
                return false;
            }
        }
    }

    // The caller laid out row slots from computeFeatExtractOutputLength; a disagreement
    // here would silently shift every later clip in the batch.
    if (totalAudioTokens != expectedRows)
    {
        LOG_ERROR("Audio encoder produced %ld rows but %ld were reserved for this clip",
            static_cast<long>(totalAudioTokens), static_cast<long>(expectedRows));
        return false;
    }
    __half* const audioOutPtr = mAudioEmbedding.dataPointer<__half>() + destRowOffset * mConfig.audioFeatureDim;

    // Set input shapes
    if (!mAudioContext->setInputShape(binding_names::kAudioPaddedFeatures, mPaddedFeature.getShape().getTRTDims()))
    {
        LOG_ERROR("Failed to set padded features input shape");
        return false;
    }

    if (!mAudioContext->setInputShape(
            binding_names::kAudioPaddedMaskIndices, mPaddedMaskIndices.getShape().getTRTDims()))
    {
        LOG_ERROR("Failed to set padded mask indices input shape");
        return false;
    }

    if (!mAudioContext->setInputShape(binding_names::kAudioAttentionMask, mAudioAttentionMask.getShape().getTRTDims()))
    {
        LOG_ERROR("Failed to set attention mask input shape");
        return false;
    }

    // Set tensor addresses
    if (!mAudioContext->setTensorAddress(binding_names::kAudioPaddedFeatures, mPaddedFeature.rawPointer()))
    {
        LOG_ERROR("Failed to set padded features input address");
        return false;
    }

    if (!mAudioContext->setTensorAddress(binding_names::kAudioPaddedMaskIndices, mPaddedMaskIndices.rawPointer()))
    {
        LOG_ERROR("Failed to set padded mask indices input address");
        return false;
    }

    if (!mAudioContext->setTensorAddress(binding_names::kAudioAttentionMask, mAudioAttentionMask.rawPointer()))
    {
        LOG_ERROR("Failed to set attention mask input address");
        return false;
    }

    if (!mAudioContext->setTensorAddress(binding_names::kAudioOutput, audioOutPtr))
    {
        LOG_ERROR("Failed to set audio output address");
        return false;
    }

    // Execute audio encoder
    LOG_DEBUG(
        "Executing audio encoder with shapes: "
        "input=[%ld,%ld,%ld], indices=[%ld,2], mask=[%ld,%ld], output=[%ld,%ld]",
        mPaddedFeature.getShape()[0], mPaddedFeature.getShape()[1], mPaddedFeature.getShape()[2],
        mPaddedMaskIndices.getShape()[0], mAudioAttentionMask.getShape()[0], mAudioAttentionMask.getShape()[1],
        totalAudioTokens, mConfig.audioFeatureDim);

    {
        TIME_STAGE(metrics::StageNames::kAUDIO_ENCODER, stream);

        if (!mAudioContext->enqueueV3(stream))
        {
            LOG_ERROR("Audio encoder inference failed");
            return false;
        }
    }

    LOG_DEBUG("Audio encoder inference completed");
    mMultimodalMetrics.recordRun(0, 0, 1, totalAudioTokens);
    return true;
}

bool Qwen3OmniAudioRunner::infer([[maybe_unused]] cudaStream_t stream)
{
    LOG_DEBUG("No-op (inference already done in preprocess)");
    return true;
}

rt::Tensor& Qwen3OmniAudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

bool Qwen3OmniAudioRunner::preprocessSystemPrompt(std::string const& systemPrompt,
    [[maybe_unused]] tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut,
    cudaStream_t stream)
{
    if (systemPrompt.empty())
    {
        return true;
    }

    if (!mropeCosSinOut.has_value())
    {
        LOG_ERROR("mropeCosSinOut is required for non-empty system prompts.");
        return false;
    }

    // For audio-only MRope models (e.g. Qwen3-ASR), initialize sequential MRope cache
    // for system prompt since no vision runner will fill it.
    // Batch size is always 1 for system prompt KVCache generation.
    return initializeSequentialMRopeCache(1, mropeCosSinOut.value().get(), stream);
}

bool Qwen3OmniAudioRunner::initializeSequentialMRopeCache(
    int64_t activeBatchSize, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    if (mConfig.mropeTheta <= 0.0F)
    {
        // Not an MRope model, nothing to do.
        return true;
    }

    auto const ropeShape = ropeRotaryCosSinDevice.getShape();
    int64_t const maxPositionEmbeddings = ropeShape[1];
    int64_t const rotaryDim = ropeShape[2];

    // Allocate host position IDs: [activeBatchSize, 3, maxPositionEmbeddings]
    // For audio+text only, all 3 dimensions (T, H, W) use identical sequential positions [0, 1, 2, ...]
    int64_t const numElements = activeBatchSize * 3 * maxPositionEmbeddings;
    std::vector<int64_t> positionIdsHost(numElements);

    for (int64_t b = 0; b < activeBatchSize; ++b)
    {
        int64_t batchOffset = b * 3 * maxPositionEmbeddings;
        for (int64_t dim = 0; dim < 3; ++dim)
        {
            for (int64_t pos = 0; pos < maxPositionEmbeddings; ++pos)
            {
                positionIdsHost[batchOffset + dim * maxPositionEmbeddings + pos] = pos;
            }
        }
    }

    // Copy position IDs to device
    rt::Tensor positionIdsDevice({activeBatchSize, 3, maxPositionEmbeddings}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "sequentialMRopePositionIds");
    CUDA_CHECK(cudaMemcpyAsync(positionIdsDevice.rawPointer(), positionIdsHost.data(), numElements * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    // Reshape the cos/sin cache and fill it
    check::check(
        ropeRotaryCosSinDevice.reshape({activeBatchSize, maxPositionEmbeddings, rotaryDim}), "Tensor reshape failed");

    // All audio+text MRope models (Qwen3-ASR, Qwen3-Omni) use interleaved layout.
    bool constexpr interleaved = true;
    try
    {
        kernel::initializeMRopeCosSin(ropeRotaryCosSinDevice.dataPointer<float>(),
            positionIdsDevice.dataPointer<int64_t>(), mConfig.mropeTheta, rotaryDim, maxPositionEmbeddings,
            activeBatchSize, interleaved, mConfig.mropeSectionH, mConfig.mropeSectionW, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("initializeSequentialMRopeCache: kernel launch failed: %s", e.what());
        return false;
    }

    // Synchronize to ensure kernel completes before local positionIdsDevice is freed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("Initialized sequential MRope cos/sin cache for %ld batches, %ld positions", activeBatchSize,
        maxPositionEmbeddings);

    return true;
}

} // namespace rt
} // namespace trt_edgellm
