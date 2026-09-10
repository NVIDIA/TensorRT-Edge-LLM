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

#include "runtime/nemotronAsrRuntime.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/common/argmaxKernel.h"

#include <chrono>
#include <cuda_fp16.h>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>

namespace trt_edgellm
{
namespace rt
{

namespace
{
using Json = nlohmann::json;

constexpr char const* kEncoderEngineFile = "audio_encoder.engine";
constexpr char const* kStepEngineFile = "rnnt_step.engine";
constexpr char const* kConfigFile = "config.json";

struct EnginePaths
{
    std::filesystem::path encoder;
    std::filesystem::path step;
};

EnginePaths resolveEnginePaths(std::filesystem::path const& root)
{
    EnginePaths const direct{root / "audio" / kEncoderEngineFile, root / "rnnt" / kStepEngineFile};
    EnginePaths const flat{root / kEncoderEngineFile, root / kStepEngineFile};
    bool const hasDirectEncoder = std::filesystem::is_regular_file(direct.encoder);
    bool const hasDirectStep = std::filesystem::is_regular_file(direct.step);
    bool const hasFlatEncoder = std::filesystem::is_regular_file(flat.encoder);
    bool const hasFlatStep = std::filesystem::is_regular_file(flat.step);
    if (hasDirectEncoder && hasDirectStep && !hasFlatEncoder && !hasFlatStep)
    {
        return direct;
    }
    check::check(hasFlatEncoder && hasFlatStep && !hasDirectEncoder && !hasDirectStep,
        "Nemotron-3.5-ASR bundle must contain exactly one complete engine layout: "
        "audio/audio_encoder.engine plus rnnt/rnnt_step.engine, or both engines at its root");
    return flat;
}

//! Encoder output frames for a mel length: three causal stride-2 stages,
//! each ``floor(L/2) + 1``.
int64_t encoderFramesForMel(int64_t melFrames) noexcept
{
    int64_t len = melFrames;
    for (int i = 0; i < 3; ++i)
    {
        len = len / 2 + 1;
    }
    return len;
}

void checkTensorDataType(
    nvinfer1::ICudaEngine const& engine, char const* name, nvinfer1::DataType expected, char const* engineLabel)
{
    check::check(engine.getTensorDataType(name) == expected,
        std::string(engineLabel) + ": tensor '" + name + "' has dtype "
            + getDataTypeString(engine.getTensorDataType(name)) + ", expected " + getDataTypeString(expected));
}

void checkTensorShape(nvinfer1::ICudaEngine const& engine, char const* name, std::initializer_list<int64_t> expected,
    char const* engineLabel)
{
    auto const shape = engine.getTensorShape(name);
    check::check(shape.nbDims == static_cast<int32_t>(expected.size()),
        std::string(engineLabel) + ": tensor '" + name + "' has unexpected rank");
    int32_t axis = 0;
    for (int64_t const dimension : expected)
    {
        check::check(dimension < 0 || shape.d[axis] == dimension,
            std::string(engineLabel) + ": tensor '" + name + "' has unexpected dimension " + std::to_string(axis));
        ++axis;
    }
}

} // namespace

NemotronAsrRuntime::NemotronAsrRuntime(
    std::filesystem::path const& engineDir, std::filesystem::path const& tokenizerDir, cudaStream_t stream)
{
    loadConfig(engineDir / kConfigFile);

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    check::check(mRuntime != nullptr, "Failed to create TRT runtime");

    auto const enginePaths = resolveEnginePaths(engineDir);
    mEncoderEngine = deserializeCudaEngineFromFile(*mRuntime, enginePaths.encoder);
    check::check(mEncoderEngine != nullptr, "Failed to load " + enginePaths.encoder.string());
    mEncoderContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEncoderEngine->createExecutionContext());
    check::check(mEncoderContext != nullptr, "Failed to create encoder execution context");

    mStepEngine = deserializeCudaEngineFromFile(*mRuntime, enginePaths.step);
    check::check(mStepEngine != nullptr, "Failed to load " + enginePaths.step.string());
    mStepContext = std::unique_ptr<nvinfer1::IExecutionContext>(mStepEngine->createExecutionContext());
    check::check(mStepContext != nullptr, "Failed to create step execution context");

    // Validate the I/O contract this runtime hardcodes.
    checkTensorDataType(*mEncoderEngine, "input_features", nvinfer1::DataType::kHALF, "encoder");
    checkTensorDataType(*mEncoderEngine, "prompt_ids", nvinfer1::DataType::kINT64, "encoder");
    checkTensorDataType(*mEncoderEngine, "encoder_frames", nvinfer1::DataType::kHALF, "encoder");
    checkTensorShape(*mEncoderEngine, "input_features", {1, -1, mMelBins}, "encoder");
    checkTensorShape(*mEncoderEngine, "prompt_ids", {1}, "encoder");
    checkTensorShape(*mEncoderEngine, "encoder_frames", {1, -1, mDecoderHiddenSize}, "encoder");
    checkTensorDataType(*mStepEngine, "decoder_input_ids", nvinfer1::DataType::kINT64, "rnnt_step");
    for (char const* name :
        {"hidden_state", "cell_state", "encoder_frame", "logits", "present_hidden_state", "present_cell_state"})
    {
        checkTensorDataType(*mStepEngine, name, nvinfer1::DataType::kHALF, "rnnt_step");
    }
    checkTensorShape(*mStepEngine, "decoder_input_ids", {1, 1}, "rnnt_step");
    checkTensorShape(*mStepEngine, "hidden_state", {mNumDecoderLayers, 1, mDecoderHiddenSize}, "rnnt_step");
    checkTensorShape(*mStepEngine, "cell_state", {mNumDecoderLayers, 1, mDecoderHiddenSize}, "rnnt_step");
    checkTensorShape(*mStepEngine, "encoder_frame", {1, mDecoderHiddenSize}, "rnnt_step");
    checkTensorShape(*mStepEngine, "logits", {1, mVocabSize}, "rnnt_step");
    checkTensorShape(*mStepEngine, "present_hidden_state", {mNumDecoderLayers, 1, mDecoderHiddenSize}, "rnnt_step");
    checkTensorShape(*mStepEngine, "present_cell_state", {mNumDecoderLayers, 1, mDecoderHiddenSize}, "rnnt_step");

    check::check(
        mEncoderEngine->getNbOptimizationProfiles() == 1, "encoder must contain exactly one optimization profile");
    auto const minProfile = mEncoderEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMIN);
    auto const maxProfile = mEncoderEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMAX);
    check::check(minProfile.nbDims == 3 && maxProfile.nbDims == 3 && minProfile.d[0] == 1 && maxProfile.d[0] == 1
            && minProfile.d[2] == mMelBins && maxProfile.d[2] == mMelBins && minProfile.d[1] > 0
            && minProfile.d[1] <= maxProfile.d[1],
        "encoder input_features profile does not match the batch-1 mel contract");
    mMinMelFrames = minProfile.d[1];
    mMaxMelFrames = maxProfile.d[1];
    mMaxEncoderFrames = encoderFramesForMel(mMaxMelFrames);

    mMelExtractor = audio::makeNemotronAsrExtractor();

    check::check(mTokenizer.loadFromHF(tokenizerDir, /*requireChatTemplate=*/false),
        "Failed to load tokenizer from " + tokenizerDir.string());
    check::check(mTokenizer.getNumVocab() >= mVocabSize - 1,
        "tokenizer vocab smaller than model vocab (wrong tokenizer directory?)");

    allocateBuffers(stream);

    LOG_INFO("NemotronAsrRuntime ready: maxMelFrames=%ld (%.1fs audio), vocab=%d, blank=%d, L=%d, H=%d", mMaxMelFrames,
        static_cast<double>(mMaxMelFrames) * 0.01, mVocabSize, mBlankTokenId, mNumDecoderLayers, mDecoderHiddenSize);
}

void NemotronAsrRuntime::loadConfig(std::filesystem::path const& configPath)
{
    std::ifstream configStream(configPath);
    check::check(configStream.is_open(), "Failed to open " + configPath.string());
    Json config;
    try
    {
        config = Json::parse(configStream);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error("Failed to parse " + configPath.string() + ": " + e.what());
    }

    check::check(
        config.value("model_type", "") == "nemotron3_5_asr", "config.json model_type is not 'nemotron3_5_asr'");
    mBlankTokenId = config.at("blank_token_id").get<int32_t>();
    mVocabSize = config.at("vocab_size").get<int32_t>();
    mDecoderHiddenSize = config.at("decoder_hidden_size").get<int32_t>();
    mNumDecoderLayers = config.value("num_decoder_layers", 2);
    mMaxSymbolsPerStep = config.value("max_symbols_per_step", 10);
    mDefaultPromptId = config.value("default_prompt_id", 101);
    Json const& encoderConfig = config.at("encoder_config");
    mMelBins = encoderConfig.value("num_mel_bins", 128);
    mNumPrompts = config.value("num_prompts", 128);

    check::check(mVocabSize > 0 && mBlankTokenId >= 0 && mBlankTokenId < mVocabSize,
        "config.json has an invalid RNN-T vocabulary or blank_token_id");
    check::check(mDecoderHiddenSize > 0 && mNumDecoderLayers > 0 && mMaxSymbolsPerStep > 0,
        "config.json has invalid RNN-T decoder dimensions");
    check::check(mMelBins > 0 && mNumPrompts > 0 && mDefaultPromptId >= 0 && mDefaultPromptId < mNumPrompts,
        "config.json has an invalid mel or language-prompt contract");
    check::check(
        encoderConfig.value("subsampling_factor", 8) == 8 && encoderConfig.value("subsampling_conv_stride", 2) == 2,
        "Nemotron-3.5-ASR runtime requires factor-eight stride-two encoder subsampling");
}

void NemotronAsrRuntime::allocateBuffers(cudaStream_t)
{
    mInputFeatures = Tensor(
        {1, mMaxMelFrames, mMelBins}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronAsrRuntime::inputFeatures");
    mPromptIds = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "NemotronAsrRuntime::promptIds");
    mEncoderFrames = Tensor({1, mMaxEncoderFrames, mDecoderHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronAsrRuntime::encoderFrames");
    mEncoderFrameStaging = Tensor(
        {1, mDecoderHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronAsrRuntime::frameStaging");
    mDecoderInputIds
        = Tensor({1, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "NemotronAsrRuntime::decoderInputIds");
    for (int i = 0; i < 2; ++i)
    {
        mHiddenState[i] = Tensor({mNumDecoderLayers, 1, mDecoderHiddenSize}, DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "NemotronAsrRuntime::hiddenState");
        mCellState[i] = Tensor({mNumDecoderLayers, 1, mDecoderHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
            "NemotronAsrRuntime::cellState");
    }
    mLogits = Tensor({1, mVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronAsrRuntime::logits");
    mTokenOut = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "NemotronAsrRuntime::tokenOut");

    mMelHostFp16 = Tensor(
        {mMaxMelFrames, mMelBins}, DeviceType::kCPU, nvinfer1::DataType::kHALF, "NemotronAsrRuntime::melHostFp16");
    mTokenHostI64 = Tensor({1}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "NemotronAsrRuntime::tokenHostI64");
    mPromptHostI64 = Tensor({1}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "NemotronAsrRuntime::promptHostI64");
    mTokenHostI32 = Tensor({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "NemotronAsrRuntime::tokenHostI32");
}

NemotronAsrRuntime::~NemotronAsrRuntime() noexcept
{
    if (mStepGraphExec != nullptr)
    {
        static_cast<void>(cudaGraphExecDestroy(mStepGraphExec));
    }
    if (mStepGraph != nullptr)
    {
        static_cast<void>(cudaGraphDestroy(mStepGraph));
    }
}

int64_t NemotronAsrRuntime::runEncoder(
    audio::AudioPCM const& pcm, int32_t promptId, cudaStream_t stream, int64_t& numMelFrames, Timings* timings)
{
    using Clock = std::chrono::steady_clock;
    auto elapsedMs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    check::check(pcm.sampleRate == mMelExtractor.config().sampleRate,
        "PCM sample rate mismatch (load audio with targetSampleRate=16000)");

    auto const melStart = Clock::now();
    Tensor melHost;
    check::check(mMelExtractor.extract(pcm, melHost), "Mel extraction failed");
    if (timings != nullptr)
    {
        timings->melMs = elapsedMs(melStart, Clock::now());
    }
    auto const encStart = Clock::now();
    auto const melShape = melHost.getShape(); // [T_mel, melBins]
    int64_t const melFrames = melShape[0];
    numMelFrames = melFrames;
    check::check(melFrames > 0, "Audio too short: zero mel frames");
    check::check(melFrames >= mMinMelFrames,
        "Audio too short: " + std::to_string(melFrames) + " mel frames is below engine min "
            + std::to_string(mMinMelFrames));
    check::check(melFrames <= mMaxMelFrames,
        "Audio too long: " + std::to_string(melFrames) + " mel frames exceeds engine max "
            + std::to_string(mMaxMelFrames));
    check::check(promptId >= 0 && promptId < mNumPrompts,
        "Language prompt id " + std::to_string(promptId) + " is outside [0, " + std::to_string(mNumPrompts) + ")");

    // fp32 host mel → fp16 staging → device.
    float const* melData = static_cast<float const*>(melHost.rawPointer());
    size_t const melCount = static_cast<size_t>(melFrames) * mMelBins;
    __half* staging = static_cast<__half*>(mMelHostFp16.rawPointer());
    for (size_t i = 0; i < melCount; ++i)
    {
        staging[i] = __float2half(melData[i]);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mInputFeatures.rawPointer(), staging, melCount * sizeof(__half), cudaMemcpyHostToDevice, stream));

    *mPromptHostI64.dataPointer<int64_t>() = promptId;
    CUDA_CHECK(cudaMemcpyAsync(
        mPromptIds.rawPointer(), mPromptHostI64.rawPointer(), sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    nvinfer1::Dims melDims;
    melDims.nbDims = 3;
    melDims.d[0] = 1;
    melDims.d[1] = melFrames;
    melDims.d[2] = mMelBins;
    check::check(mEncoderContext->setInputShape("input_features", melDims), "encoder setInputShape failed");
    check::check(mEncoderContext->setTensorAddress("input_features", mInputFeatures.rawPointer()),
        "encoder bind input_features failed");
    check::check(
        mEncoderContext->setTensorAddress("prompt_ids", mPromptIds.rawPointer()), "encoder bind prompt_ids failed");
    check::check(mEncoderContext->setTensorAddress("encoder_frames", mEncoderFrames.rawPointer()),
        "encoder bind encoder_frames failed");

    auto const outDims = mEncoderContext->getTensorShape("encoder_frames");
    check::check(outDims.nbDims == 3 && outDims.d[0] == 1 && outDims.d[2] == mDecoderHiddenSize,
        "unexpected encoder_frames shape");
    int64_t const numFrames = outDims.d[1];
    check::check(numFrames == encoderFramesForMel(melFrames), "encoder output frame count mismatch");

    check::check(mEncoderContext->enqueueV3(stream), "encoder enqueue failed");

    if (timings != nullptr)
    {
        // Isolate the encoder phase: wait for the enqueued forward to complete
        // so encoderMs is the real GPU time, not just launch overhead.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timings->encoderMs = elapsedMs(encStart, Clock::now());
    }
    return numFrames;
}

NemotronAsrRuntime::Result NemotronAsrRuntime::decodeGreedy(int64_t numFrames, cudaStream_t stream, Timings* timings)
{
    Result result;
    result.numEncoderFrames = numFrames;

    // Initial state: y = blank, LSTM h = c = 0. State buffer [0] holds the
    // CURRENT prediction-network state; the engine always writes present
    // states into buffer [1] and the host copies [1] → [0] only on emit.
    size_t const stateBytes = static_cast<size_t>(mNumDecoderLayers) * mDecoderHiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(mHiddenState[0].rawPointer(), 0, stateBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(mCellState[0].rawPointer(), 0, stateBytes, stream));
    *mTokenHostI64.dataPointer<int64_t>() = mBlankTokenId;
    CUDA_CHECK(cudaMemcpyAsync(
        mDecoderInputIds.rawPointer(), mTokenHostI64.rawPointer(), sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    if (!mStepBindingsSet)
    {
        // All step-engine addresses are fixed for the runtime's lifetime.
        bool bindOk = mStepContext->setTensorAddress("decoder_input_ids", mDecoderInputIds.rawPointer());
        bindOk &= mStepContext->setTensorAddress("encoder_frame", mEncoderFrameStaging.rawPointer());
        bindOk &= mStepContext->setTensorAddress("hidden_state", mHiddenState[0].rawPointer());
        bindOk &= mStepContext->setTensorAddress("cell_state", mCellState[0].rawPointer());
        bindOk &= mStepContext->setTensorAddress("present_hidden_state", mHiddenState[1].rawPointer());
        bindOk &= mStepContext->setTensorAddress("present_cell_state", mCellState[1].rawPointer());
        bindOk &= mStepContext->setTensorAddress("logits", mLogits.rawPointer());
        check::check(bindOk, "rnnt_step binds failed");
        mStepBindingsSet = true;
    }

    if (!mStepGraphCaptureAttempted)
    {
        mStepGraphCaptureAttempted = true;
        // Warmup (initializes TRT internal state), then capture one full step:
        // engine enqueue + argmax. Bindings are fixed so one graph serves all
        // steps. On capture failure fall back to per-step enqueue.
        CUDA_CHECK(cudaMemsetAsync(
            mEncoderFrameStaging.rawPointer(), 0, static_cast<size_t>(mDecoderHiddenSize) * sizeof(__half), stream));
        check::check(mStepContext->enqueueV3(stream), "rnnt_step warmup enqueue failed");
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaError_t const beginStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (beginStatus == cudaSuccess)
        {
            bool const enqueued = mStepContext->enqueueV3(stream);
            if (enqueued)
            {
                kernel::invokeRowwiseArgmax<__half>(static_cast<__half const*>(mLogits.rawPointer()), /*rows=*/1,
                    mVocabSize, static_cast<int32_t*>(mTokenOut.rawPointer()), stream);
            }
            cudaGraph_t capturedGraph{nullptr};
            cudaError_t const endStatus = cudaStreamEndCapture(stream, &capturedGraph);
            if (enqueued && endStatus == cudaSuccess && capturedGraph != nullptr)
            {
                cudaGraphExec_t capturedExec{nullptr};
                cudaError_t const instantiateStatus = instantiateCudaGraph(&capturedExec, capturedGraph);
                if (instantiateStatus == cudaSuccess)
                {
                    mStepGraph = capturedGraph;
                    mStepGraphExec = capturedExec;
                    LOG_INFO("RNN-T step CUDA graph captured.");
                }
                else
                {
                    static_cast<void>(cudaGraphDestroy(capturedGraph));
                    static_cast<void>(cudaGetLastError());
                    LOG_WARNING("RNN-T step CUDA graph instantiation failed; falling back to per-step enqueue.");
                }
            }
            else
            {
                if (capturedGraph != nullptr)
                {
                    static_cast<void>(cudaGraphDestroy(capturedGraph));
                }
                static_cast<void>(cudaGetLastError());
                LOG_WARNING("RNN-T step CUDA graph capture failed; falling back to per-step enqueue.");
            }
        }
        else
        {
            static_cast<void>(cudaGetLastError());
            LOG_WARNING("RNN-T step CUDA graph capture could not start; falling back to per-step enqueue.");
        }
        // The warmup only wrote the present buffers [1] and the logits, both
        // of which the first real step overwrites; current state [0] and
        // decoder ids are inputs and remain as initialized above.
    }

    uint8_t const* framesBase = static_cast<uint8_t const*>(mEncoderFrames.rawPointer());
    size_t const frameBytes = static_cast<size_t>(mDecoderHiddenSize) * sizeof(__half);

    // Timer starts after any one-time graph capture above so decodeMs is the
    // pure step-loop time. The loop syncs the stream every step, so host
    // wall-clock already reflects GPU work.
    auto const decodeStart = std::chrono::steady_clock::now();
    int64_t t = 0;
    int32_t symbols = 0;
    bool pendingStateUpdate = false;
    while (t < numFrames)
    {
        // Stage the current encoder frame at the step engine's fixed input.
        CUDA_CHECK(cudaMemcpyAsync(mEncoderFrameStaging.rawPointer(), framesBase + static_cast<size_t>(t) * frameBytes,
            frameBytes, cudaMemcpyDeviceToDevice, stream));

        if (mStepGraphExec != nullptr)
        {
            CUDA_CHECK(cudaGraphLaunch(mStepGraphExec, stream));
        }
        else
        {
            check::check(mStepContext->enqueueV3(stream), "rnnt_step enqueue failed");
            kernel::invokeRowwiseArgmax<__half>(static_cast<__half const*>(mLogits.rawPointer()), /*rows=*/1,
                mVocabSize, static_cast<int32_t*>(mTokenOut.rawPointer()), stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mTokenHostI32.rawPointer(), mTokenOut.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        pendingStateUpdate = false;
        ++result.numDecodeSteps;
        int32_t const best = *mTokenHostI32.dataPointer<int32_t>();

        if (best == mBlankTokenId)
        {
            // Blank: advance the frame pointer; present states are discarded
            // — blank never updates the prediction network.
            ++t;
            symbols = 0;
        }
        else
        {
            // Emit: adopt present states (copy back over current), feed the
            // token back, stay on the frame.
            result.tokens.push_back(best);
            CUDA_CHECK(cudaMemcpyAsync(mHiddenState[0].rawPointer(), mHiddenState[1].rawPointer(), stateBytes,
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                mCellState[0].rawPointer(), mCellState[1].rawPointer(), stateBytes, cudaMemcpyDeviceToDevice, stream));
            *mTokenHostI64.dataPointer<int64_t>() = best;
            CUDA_CHECK(cudaMemcpyAsync(mDecoderInputIds.rawPointer(), mTokenHostI64.rawPointer(), sizeof(int64_t),
                cudaMemcpyHostToDevice, stream));
            pendingStateUpdate = true;
            if (++symbols >= mMaxSymbolsPerStep)
            {
                ++t;
                symbols = 0;
            }
        }
    }

    if (pendingStateUpdate)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (timings != nullptr)
    {
        timings->decodeMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decodeStart).count();
    }

    std::vector<tokenizer::Rank> ranks(result.tokens.begin(), result.tokens.end());
    result.text = mTokenizer.decode(ranks, /*skipSpecialTokens=*/false);
    return result;
}

NemotronAsrRuntime::Result NemotronAsrRuntime::transcribe(
    audio::AudioPCM const& pcm, int32_t promptId, cudaStream_t stream, Timings* timings)
{
    int64_t numMelFrames = 0;
    int64_t const numFrames = runEncoder(pcm, promptId, stream, numMelFrames, timings);
    Result result = decodeGreedy(numFrames, stream, timings);
    result.numMelFrames = numMelFrames;
    return result;
}

} // namespace rt
} // namespace trt_edgellm
