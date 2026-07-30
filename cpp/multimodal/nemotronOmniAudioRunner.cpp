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

#include "nemotronOmniAudioRunner.h"
#include "audioUtils.h"
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

namespace
{

// Validate the bound CPU MelExtractor config is parakeet-spec — the GPU kernels
// hard-code the parakeet geometry/algorithm, so a non-parakeet extractor must
// fall back to CPU rather than produce a wrong-but-plausible spectrogram. Values
// mirror makeParakeetExtractor (melSpectrogram.cpp).
bool parakeetFbankConfigOk(audio::MelExtractorConfig const& cfg, std::string& reason)
{
    auto fail = [&reason](char const* msg) {
        reason = msg;
        return false;
    };
    // Shape: the kernel FFT is N=512-specific (winLength 400 centred in 512), the
    // encoder takes [1, T, 128].
    if (!(cfg.nFFT == 512 && cfg.winLength == 400 && cfg.nMel == 128))
        return fail("needs nFFT==512, winLength==400, nMel==128.");
    // Windowing / framing. windowCentredInFft must hold: the GPU embeds the 400-tap
    // window centred at (nFft-winLength)/2 and frames the full nFft span, so a
    // left-aligned config would diverge from the kernel. Compare preemph with a
    // tolerance — an exact float == would silently reject a config whose 0.97 has any
    // representation drift and fall back to CPU. preemphPostScale must be exactly 0:
    // a non-zero value switches the CPU extractor to a per-frame pre-emphasis variant
    // the GPU kernel does not implement (it is full-waveform only).
    if (!(cfg.windowType == audio::WindowType::kHannSymmetric && cfg.framePadding == audio::FramePadding::kCenterZero
            && cfg.windowCentredInFft && std::fabs(cfg.preemphCoeff - 0.97f) < 1e-6f && cfg.preemphPostScale == 0.0f))
        return fail(
            "needs symmetric Hann window centred in nFFT, centre-zero framing, full-waveform pre-emphasis 0.97 "
            "(no per-frame post-scale).");
    // Log + post-normalize.
    if (!(cfg.logType == audio::LogType::kLn && cfg.logFloorMode == audio::LogFloorMode::kAdd
            && cfg.postNormalize == audio::PostNormalize::kPerFeatureMeanStd))
        return fail("needs natural log + add-floor + per-feature mean/std post-normalize.");
    // Layout + dynamic T. dropLastStftFrame is true for parakeet (features_lengths
    // = N // hop); the GPU kernel frames exactly floor(N/hop) directly, which is
    // bit-equivalent to framing floor(N/hop)+1 and dropping the last.
    if (!(cfg.layout == audio::MelLayout::kTimeMel && cfg.dropLastStftFrame
            && cfg.timePadding == audio::TimePadding::kNone))
        return fail("needs [T, nMel] time-first layout, drop-last-frame, no static time padding.");
    return true;
}

} // namespace

NemotronOmniAudioRunner::NemotronOmniAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    // Audio runner does not use the visual engine from base class.
    // Load the audio engine instead.
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    std::string const enginePath = engineDir + "/audio_encoder.engine";
    mAudioEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    bool const profileSet = mAudioContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile for audio engine");

    setNonBlockingAuxStreams(mAudioContext.get(), mAudioEngine.get(), mAuxStreams);

    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "NemotronOmniAudioRunner: Failed to validate config");
    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "NemotronOmniAudioRunner: Failed to allocate buffer");

    // Best-effort init of the online GPU fbank. Failure is non-fatal: the runner
    // falls back to the CPU MelExtractor (mFeMel, already bound by
    // validateAndFillConfig) for PCM→mel. initFbankResources issues H2D copies
    // via CUDA_CHECK, which throws on failure (e.g. device OOM), so catch here to
    // keep construction non-fatal.
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
}

bool NemotronOmniAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

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

    mModelType = multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER;

    if (!jsonConfig.contains("sound_config"))
    {
        LOG_ERROR("sound_config not found in config.json");
        return false;
    }
    auto const& soundConfig = jsonConfig["sound_config"];

    if (!soundConfig.contains("num_mel_bins"))
    {
        LOG_ERROR("sound_config.num_mel_bins not found in config.json");
        return false;
    }
    mConfig.melBins = soundConfig["num_mel_bins"].get<int32_t>();

    if (soundConfig.contains("subsampling_factor"))
    {
        mConfig.subsamplingFactor = soundConfig["subsampling_factor"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("sound_config.subsampling_factor not found in config.json");
        return false;
    }

    if (!jsonConfig.contains("sound_context_token_id"))
    {
        LOG_ERROR("sound_context_token_id not found in config.json");
        return false;
    }
    mConfig.soundContextTokenId = jsonConfig["sound_context_token_id"].get<int32_t>();

    nvinfer1::Dims const inputShapeMax
        = mAudioEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxSeqLen = inputShapeMax.d[1];

    nvinfer1::Dims const outputShape = mAudioEngine->getTensorShape("last_hidden_state");
    mConfig.audioFeatureDim = outputShape.d[2];

    LOG_INFO("NemotronOmniAudioRunner: melBins=%d, maxSeqLen=%ld, hiddenDim=%d, soundTokenId=%d", mConfig.melBins,
        mMaxSeqLen, mConfig.audioFeatureDim, mConfig.soundContextTokenId);

    // Pick mel feature-extractor family from the audio engine's top-level
    // ``model_type`` (export.py writes ``parakeet`` here for Nemotron-Omni).
    // Runner owns mel extraction; callers hand off raw PCM.
    std::string const audioModelType = jsonConfig.value("model_type", "");
    if (audioModelType == "parakeet")
    {
        mFeMel = rt::audio::makeParakeetExtractor();
    }
    else
    {
        LOG_ERROR(
            "Unsupported audio model_type %s for NemotronOmniAudioRunner (expected parakeet)", audioModelType.c_str());
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    bool status{true};

    mInputFeatures = rt::Tensor({1, mMaxSeqLen, mConfig.melBins}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniAudioRunner::mInputFeatures");
    status &= mAudioContext->setTensorAddress("input_features", mInputFeatures.rawPointer());

    // Initial output buffer is sized for one full-length clip. preprocess()
    // grows it via resizeEmbeddingForRows() when a batch packs more clips.
    // last_hidden_state is rebound per clip with the row offset, so no
    // setTensorAddress here.
    int64_t const maxEncodedSeqLen = mMaxSeqLen / mConfig.subsamplingFactor;
    mAudioEmbedding = rt::Tensor({maxEncodedSeqLen, static_cast<int64_t>(mConfig.audioFeatureDim)},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mAudioEmbedding");

    if (!status)
    {
        LOG_ERROR("Failed to set tensor addresses for audio engine");
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::resizeEmbeddingForRows(int64_t rows)
{
    int64_t const hidden = mConfig.audioFeatureDim;
    int64_t const requiredBytes = rows * hidden * static_cast<int64_t>(sizeof(__half));
    if (requiredBytes <= mAudioEmbedding.getMemoryCapacity())
    {
        return mAudioEmbedding.reshape({rows, hidden});
    }
    // Existing capacity isn't enough — reallocate. Encoded rows haven't been
    // written yet at this point in preprocess(), so dropping the old buffer
    // is safe.
    mAudioEmbedding = rt::Tensor(
        {rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mAudioEmbedding");
    return true;
}

bool NemotronOmniAudioRunner::encodeSingleClip(
    rt::Tensor const& mel, int64_t destRowOffset, int64_t& encodedRowsOut, cudaStream_t stream)
{
    // Mel shape: [1, rawSeqLen, mel_bins]. Pad rawSeqLen up to a multiple of
    // subsamplingFactor so the 3× stride-2 subsampling produces an integer
    // number of output rows.
    int64_t const rawSeqLen = mel.getShape()[1];
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const paddedSeqLen = alignUp(rawSeqLen, mConfig.subsamplingFactor);
    int64_t const encodedSeqLen = paddedSeqLen / mConfig.subsamplingFactor;

    check::check(
        mInputFeatures.reshape({1, paddedSeqLen, static_cast<int64_t>(mConfig.melBins)}), "Tensor reshape failed");
    if (paddedSeqLen > rawSeqLen)
    {
        CUDA_CHECK(cudaMemsetAsync(mInputFeatures.rawPointer(), 0, mInputFeatures.getMemoryCapacity(), stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mInputFeatures.rawPointer(), mel.rawPointer(), mel.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));

    // Bind the encoder output to this clip's slot in mAudioEmbedding so each
    // enqueueV3 writes directly to its destination — no scatter step needed.
    auto* const embeddingBase = static_cast<std::byte*>(mAudioEmbedding.rawPointer());
    int64_t const destByteOffset
        = destRowOffset * static_cast<int64_t>(mConfig.audioFeatureDim) * static_cast<int64_t>(sizeof(__half));

    bool status{true};
    status &= mAudioContext->setInputShape("input_features", mInputFeatures.getShape().getTRTDims());
    status &= mAudioContext->setTensorAddress("last_hidden_state", embeddingBase + destByteOffset);
    if (!status)
    {
        LOG_ERROR("encodeSingleClip: Failed to set encoder bindings");
        return false;
    }

    if (!mAudioContext->enqueueV3(stream))
    {
        LOG_ERROR("encodeSingleClip: Failed to enqueue encoder");
        return false;
    }

    mMultimodalMetrics.recordRun(1, encodedSeqLen);
    encodedRowsOut = encodedSeqLen;
    return true;
}

bool NemotronOmniAudioRunner::initFbankResources(cudaStream_t stream)
{
    // ── 1. Validation: config is parakeet-spec, and fetch the CPU MelExtractor's
    //    Slaney weights / symmetric-Hann taps (single source of truth so the GPU
    //    fbank and CPU fallback can never drift), shape-checked via .size(). ──
    audio::MelExtractorConfig const& cfg = mFeMel.config();
    std::string reason;
    if (!parakeetFbankConfigOk(cfg, reason))
    {
        LOG_WARNING("Online GPU fbank disabled (extractor not parakeet-spec): %s", reason.c_str());
        return false;
    }
    int32_t const nFft = cfg.nFFT;
    int32_t const nMel = cfg.nMel;
    int32_t const nFreq = nFft / 2 + 1;

    std::vector<float> const& melFilterHost = mFeMel.melFilterBank(); // [nMel, nFreq] row-major, freq-contiguous
    if (melFilterHost.size() != static_cast<size_t>(nMel) * nFreq)
    {
        LOG_WARNING("Online GPU fbank disabled: mel filter size %zu != nMel(%d) x nFreq(%d).", melFilterHost.size(),
            nMel, nFreq);
        return false;
    }
    std::vector<float> const& windowHost = mFeMel.window(); // winLength taps, embedded centred in nFft below
    if (windowHost.size() != static_cast<size_t>(cfg.winLength) || cfg.winLength > nFft)
    {
        LOG_WARNING("Online GPU fbank disabled: window size %zu invalid for winLength=%d, nFft=%d.", windowHost.size(),
            cfg.winLength, nFft);
        return false;
    }

    // ── 2. GEMM module load — before any allocation, so an unsupported SM or a
    //    build without the FP32-out mel GEMM variant uses the CPU MelExtractor
    //    without reserving device memory. The umbrella macro alone is not enough:
    //    loadKernelModule() succeeds with any compiled GEMM variant, while the
    //    per-clip pipeline needs gemm_blackwell_small_fp16in_fp32out specifically
    //    (runFp16inFp32out has no in-variant fallback). ──
#if defined(CUTE_DSL_GEMM_ENABLED) && defined(CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED)
    if (!CuteDslGemmRunner::loadKernelModule())
    {
        LOG_ERROR("CuteDslGemmRunner::loadKernelModule failed — online fbank GEMM unavailable on this device.");
        return false;
    }
#else
    LOG_WARNING(
        "Online GPU fbank disabled: FP16-in/FP32-out mel GEMM variant not compiled (need -DENABLE_CUTE_DSL=gemm "
        "with an artifact providing gemm_blackwell_small_fp16in_fp32out). Using CPU MelExtractor.");
    return false;
#endif

    // ── 3. Bound derivation from the engine "input_features" kMAX profile.
    //    mMaxSeqLen (validateAndFillConfig) is the max raw mel time steps; parakeet
    //    T_out == floor(N/hop), so the longest PCM the GPU path accepts is the
    //    largest N with T_out <= maxFrames. The pass-1 gate routes longer clips to
    //    the CPU fallback. ──
    int64_t const maxFrames = mMaxSeqLen;
    if (maxFrames <= 0)
    {
        LOG_WARNING("Online GPU fbank disabled: engine kMAX profile gives non-positive max mel frames (%ld).",
            static_cast<long>(maxFrames));
        return false;
    }
    int64_t const maxPcmSamples = (maxFrames + 1) * cfg.hopLength - 1;
    int32_t const nPadMax = audioUtils::fbankNPad(static_cast<int32_t>(maxFrames));
    int32_t const kPad = audioUtils::fbankKPadParakeet(nFreq);

    // ── 4. Allocation block — the ONLY site fbank device tensors are constructed;
    //    every buffer is sized at the kMAX bound, per-clip only reshapes. ──
    mFbankResourcesParakeet.melFilterFp16Kmajor = rt::Tensor(
        {nMel, kPad}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::melFilterFp16Kmajor");
    mFbankResourcesParakeet.windowF32
        = rt::Tensor({nFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::windowF32");
    mFbankResourcesParakeet.fftTwiddle = rt::Tensor(
        {nFft, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::fftTwiddle");
    mFbankResourcesParakeet.framedF32 = rt::Tensor(
        {maxFrames, nFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::framedF32");
    mFbankResourcesParakeet.magFp16 = rt::Tensor({static_cast<int64_t>(nPadMax), kPad}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::magFp16");
    mFbankResourcesParakeet.melPowerF32 = rt::Tensor({nMel, static_cast<int64_t>(nPadMax)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::melPowerF32");
    mFbankResourcesParakeet.mean
        = rt::Tensor({nMel}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::mean");
    mFbankResourcesParakeet.invDenom
        = rt::Tensor({nMel}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::invDenom");
    mPcmF32Device = rt::Tensor(
        {maxPcmSamples}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniAudioRunner::mPcmF32Device");
    // Time-first [1, maxFrames, nMel] (parakeet layout; NOT whisper's [1, nMel, T]).
    mMelSpecDevice = rt::Tensor({1, maxFrames, static_cast<int64_t>(nMel)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mMelSpecDevice");

    // ── 5. Fill-only block: H2D the persistent weight / table tensors, and zero
    //    the mag K-pad region once (see the per-clip-memset elimination below). ──
    // Mel filter: cast + K-pad the Slaney weights into the [nMel, kPad] F16 K-major
    // GEMM A-matrix layout (nMel-major, freq-contiguous — pure FP16 cast, no transpose).
    if (!audioUtils::fillMelFilterFp16Kmajor(
            melFilterHost.data(), nMel, nFreq, kPad, mFbankResourcesParakeet.melFilterFp16Kmajor, stream))
    {
        return false;
    }
    std::vector<float> windowEmbedded;
    audioUtils::makeCentredWindowHost(windowHost, nFft, windowEmbedded);
    CUDA_CHECK(cudaMemcpyAsync(mFbankResourcesParakeet.windowF32.rawPointer(), windowEmbedded.data(),
        static_cast<size_t>(nFft) * sizeof(float), cudaMemcpyHostToDevice, stream));
    std::vector<float> twiddleHost;
    audioUtils::makeFftTwiddleHost(nFft, twiddleHost);
    CUDA_CHECK(cudaMemcpyAsync(mFbankResourcesParakeet.fftTwiddle.rawPointer(), twiddleHost.data(),
        twiddleHost.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    // Zero magFp16 ONCE over the full [nPadMax, kPad] buffer. Per clip fbankParakeet
    // only shrinks the view (no re-allocation) and stftR2C512FusedMagsq writes only
    // the active [0, T_out) × [0, nFreq) window, so the K-pad columns [nFreq, kPad)
    // and unused rows stay zero for every clip — the AOT GEMM (no K-residue handling)
    // is exact without a per-clip memset.
    CUDA_CHECK(cudaMemsetAsync(
        mFbankResourcesParakeet.magFp16.rawPointer(), 0, static_cast<size_t>(nPadMax) * kPad * sizeof(__half), stream));

    // windowEmbedded / twiddleHost are stack-local pageable host buffers feeding the
    // async H2D copies above; a pageable async H2D does not guarantee the source is
    // consumed before the call returns, so synchronise before they go out of scope.
    // Runs once at construction, so the sync is negligible.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 6. Publish the scalar params from cfg (single source of truth for
    //    fbankParakeet / computeNumMelFramesParakeet). normEps is not a
    //    MelExtractorConfig field — it is the fixed parakeet spec constant
    //    kParakeetZScoreEps, matching the CPU MelExtractor's per-feature epsilon. ──
    mFbankResourcesParakeet.nFft = nFft;
    mFbankResourcesParakeet.hopLength = cfg.hopLength;
    mFbankResourcesParakeet.centerPad = nFft / 2;
    mFbankResourcesParakeet.nMel = nMel;
    mFbankResourcesParakeet.nFreq = nFreq;
    mFbankResourcesParakeet.maxFrames = static_cast<int32_t>(maxFrames);
    mFbankResourcesParakeet.preemph = cfg.preemphCoeff;
    mFbankResourcesParakeet.logGuard = cfg.logFloor;
    mFbankResourcesParakeet.normEps = audioUtils::kParakeetZScoreEps;

    LOG_INFO(
        "Online GPU fbank ready (mel filter %dx%d Slaney reused from CPU MelExtractor; K-pad %d, FP32-out GEMM; "
        "buffers pre-allocated for up to %ld mel frames).",
        nMel, nFreq, kPad, static_cast<long>(maxFrames));
    return true;
}

bool NemotronOmniAudioRunner::gpuFbankViable(rt::audio::AudioPCM const& pcm) const
{
    // Online fbank ready (config validated parakeet-spec, params published into
    // mFbankResourcesParakeet), the engine mel width agrees, and the sample rate
    // matches what the kernels assume (the CPU MelExtractor rejects a rate
    // mismatch, so the GPU path must too).
    return mFbankReady && mConfig.melBins == mFbankResourcesParakeet.nMel
        && pcm.sampleRate == mFeMel.config().sampleRate;
}

bool NemotronOmniAudioRunner::runGpuFbankClip(
    rt::audio::AudioPCM const& pcm, int64_t const numFrames, rt::Tensor& melSpec, cudaStream_t stream)
{
    // Upload host FP32 PCM [-1, 1] into the pre-allocated [N] staging tensor
    // (metadata-only reshape; the pass-1 maxFrames gate bounds N). pcm.samples is
    // owned by the request and outlives preprocess, covering the async fbank launches.
    if (!audioUtils::uploadHostPcmF32ToGpu(pcm.samples, mPcmF32Device, stream))
    {
        return false;
    }
    // Non-owning view of the pre-allocated backing store at this clip's width,
    // time-first [1, T, nMel] (parakeet layout; numFrames == fbankParakeet's internal
    // floor(N/hop), so the shape contract holds). A move-assignment into melSpec on
    // the CPU-fallback path (produceClipMel → uploadHostMelFp32ToFp16Gpu) rebinds it
    // to a freshly-owned tensor and cannot free mMelSpecDevice; every consumer of
    // melSpec runs within this encodePreparedClips call, inside mMelSpecDevice's lifetime.
    melSpec
        = rt::Tensor(mMelSpecDevice.rawPointer(), {1, numFrames, static_cast<int64_t>(mFbankResourcesParakeet.nMel)},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    return audioUtils::fbankParakeet(mPcmF32Device, mFbankResourcesParakeet, melSpec, stream);
}

bool NemotronOmniAudioRunner::produceClipMel(ClipPlan& plan, rt::Tensor& melSpec, cudaStream_t stream)
{
    if (plan.pcm)
    {
        if (runGpuFbankClip(*plan.pcm, plan.numFrames, melSpec, stream))
        {
            return true;
        }
        // GPU fbank failed at runtime: re-extract on the CPU. It yields the same
        // floor(N/hop) frame count, so the pass-1 row sizing still holds.
        LOG_WARNING("Online GPU fbank failed; falling back to CPU MelExtractor for this clip.");
        if (!mFeMel.extract(*plan.pcm, plan.hostMel))
        {
            LOG_ERROR("Mel extraction failed");
            return false;
        }
    }
    // CPU mel (pass-1 extracted, or the GPU fallback above): FP32 host mel →
    // FP16 GPU mel ([1, T, mel_bins]).
    return audioUtils::uploadHostMelFp32ToFp16Gpu(plan.hostMel, melSpec, stream, "NemotronOmniAudioRunner::mel");
}

bool NemotronOmniAudioRunner::inspectAllClips(
    rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths)
{
    // Choose each clip's fbank path and derive its encoded row count without running the encoder. Online fbank uses
    // the closed-form frame count; CPU fallback retains the extracted host mel for infer().
    mPreparedClipPlans.clear();
    mPreparedTokenLengths.clear();
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };

    // Pass 1: determine each clip's mel frame count T and size mAudioEmbedding.
    for (auto const& req : request.requests)
    {
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
            int64_t numFrames = 0;
            if (gpuFbankViable(*audio.pcm))
            {
                // GPU path: T = floor(N / hop), no kernel needed to size the buffer.
                int32_t const t = audioUtils::computeNumMelFramesParakeet(
                    static_cast<int64_t>(audio.pcm->samples.size()), mFbankResourcesParakeet.hopLength);
                // The fbank buffers were pre-allocated for up to maxFrames frames;
                // clips too short for the framing (t <= 0) and clips beyond that
                // bound (t > maxFrames) fall back to the CPU MelExtractor without
                // touching the GPU. Both paths yield the same floor(N/hop) T.
                if (t > 0 && t <= mFbankResourcesParakeet.maxFrames)
                {
                    plan.pcm = audio.pcm;
                    numFrames = t;
                }
            }
            if (!plan.pcm)
            {
                // CPU path (gate miss, clip too short for the GPU framing, or over-bound).
                if (!mFeMel.extract(*audio.pcm, plan.hostMel))
                {
                    LOG_ERROR("Mel extraction failed");
                    return false;
                }
                numFrames = plan.hostMel.getShape()[0];
            }
            plan.numFrames = numFrames;
            // Parakeet layout: mel is [T, mel_bins]; encoded length is
            // ceil(T / subsamplingFactor).
            int64_t const encodedSeqLen = alignUp(numFrames, mConfig.subsamplingFactor) / mConfig.subsamplingFactor;
            audioTokenLengths.push_back(encodedSeqLen);
            mPreparedTokenLengths.push_back(encodedSeqLen);
            mPreparedClipPlans.push_back(std::move(plan));
        }
    }
    return true;
}

bool NemotronOmniAudioRunner::encodePreparedClips(cudaStream_t stream)
{
    int64_t totalEncodedRows{};
    for (int64_t const length : mPreparedTokenLengths)
    {
        totalEncodedRows += length;
    }

    if (totalEncodedRows == 0)
    {
        // No audio in the batch: leave a 0-row view so downstream multimodal
        // scatter has nothing to inject.
        return mAudioEmbedding.reshape({0, static_cast<int64_t>(mConfig.audioFeatureDim)});
    }

    if (!resizeEmbeddingForRows(totalEncodedRows))
    {
        LOG_ERROR("Failed to size mAudioEmbedding for %ld rows", totalEncodedRows);
        return false;
    }

    // Pass 2: produce each clip's [1, T, mel_bins] FP16 mel (GPU fbank or CPU
    // upload) and encode it into its row slot.
    int64_t rowOffset = 0;
    for (auto& plan : mPreparedClipPlans)
    {
        rt::Tensor melSpec;
        if (!produceClipMel(plan, melSpec, stream))
        {
            return false;
        }
        int64_t encodedRows = 0;
        if (!encodeSingleClip(melSpec, rowOffset, encodedRows, stream))
        {
            return false;
        }
        rowOffset += encodedRows;
    }
    ELLM_CHECK(rowOffset == totalEncodedRows, "Nemotron audio encoder produced an unexpected row count");
    return true;
}

void NemotronOmniAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    // Repeat each ``<so_embedding>`` placeholder ``audioTokenLengths[i]``
    // times so the runtime multimodal kernel can inject one audio frame per
    // copy. Reuse ``batchInputIds`` if the ViT runner already tokenized
    // (combined image+audio); otherwise tokenize from scratch.
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int audioIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (auto const& id : ids)
        {
            if (id == mConfig.soundContextTokenId)
            {
                int64_t const numAudioTokens = audioTokenLengths.at(audioIndex);
                for (int64_t k = 0; k < numAudioTokens; ++k)
                {
                    newIds.push_back(mConfig.soundContextTokenId);
                }
                ++audioIndex;
            }
            else
            {
                newIds.push_back(id);
            }
        }
        if (alreadyTokenized)
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

bool NemotronOmniAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, [[maybe_unused]] bool imageOnly)
{
    std::vector<int64_t> audioTokenLengths;

    try
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);
        if (!inspectAllClips(request, audioTokenLengths))
        {
            return false;
        }
        textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed: %s", e.what());
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::infer(cudaStream_t stream)
{
    bool const success = encodePreparedClips(stream);
    discardPreparedInput();
    return success;
}

bool NemotronOmniAudioRunner::prepareArtifactSubset([[maybe_unused]] rt::LLMGenerationRequest const& request,
    std::vector<size_t> const& originalItemIndices, [[maybe_unused]] cudaStream_t stream)
{
    std::vector<ClipPlan> selectedPlans;
    std::vector<int64_t> selectedLengths;
    selectedPlans.reserve(originalItemIndices.size());
    selectedLengths.reserve(originalItemIndices.size());
    size_t previous = 0;
    bool first = true;
    for (size_t const index : originalItemIndices)
    {
        ELLM_CHECK(index < mPreparedClipPlans.size(), "Selected audio artifact index is out of range");
        ELLM_CHECK(first || index > previous, "Selected audio artifact indices must be unique and ordered");
        selectedPlans.push_back(std::move(mPreparedClipPlans[index]));
        selectedLengths.push_back(mPreparedTokenLengths[index]);
        previous = index;
        first = false;
    }
    mPreparedClipPlans = std::move(selectedPlans);
    mPreparedTokenLengths = std::move(selectedLengths);
    return true;
}

std::vector<int64_t> NemotronOmniAudioRunner::preparedArtifactRowCounts() const
{
    return mPreparedTokenLengths;
}

void NemotronOmniAudioRunner::discardPreparedInput() noexcept
{
    mPreparedClipPlans.clear();
    mPreparedTokenLengths.clear();
}

rt::Tensor& NemotronOmniAudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

} // namespace rt
} // namespace trt_edgellm
