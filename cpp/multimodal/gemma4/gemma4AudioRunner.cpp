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

#include "multimodal/gemma4/gemma4AudioRunner.h"
#include "common/checkMacros.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <cstdint>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace
{
inline int64_t alignUp(int64_t x, int64_t a)
{
    return (x + a - 1) / a * a;
}
} // namespace

namespace trt_edgellm
{
namespace rt
{

Gemma4AudioRunner::Gemma4AudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    std::string const enginePath = engineDir + "/audio_encoder.engine";
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mAudioEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    bool const profileSet = mAudioContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile for Gemma4 audio engine");

    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "Gemma4AudioRunner: Failed to validate config");
    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "Gemma4AudioRunner: Failed to allocate buffer");

    // Initialize Whisper-style mel extractor for PCM→mel fallback.
    // Gemma4 audio uses 128 mel bins, 16kHz, same as Whisper FE, but expects
    // [1, T, mel_bins] layout. The extractor outputs [mel_bins, T] which we
    // transpose during upload.
    mMelExtractor = rt::audio::makeGemma4AudioExtractor();
}

bool Gemma4AudioRunner::validateAndFillConfig(std::string const& engineDir)
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

    mModelType = multimodal::ModelType::GEMMA4_AUDIO_ENCODER;

    // Read audio_config section
    if (!jsonConfig.contains("audio_config"))
    {
        LOG_ERROR("audio_config not found in config.json");
        return false;
    }
    auto const& audioConfig = jsonConfig["audio_config"];

    if (audioConfig.contains("num_mel_bins"))
    {
        mConfig.melBins = audioConfig["num_mel_bins"].get<int32_t>();
    }

    // Gemma4 audio uses 2x stride-2 conv → 4x temporal downsample
    mConfig.subsamplingFactor = 4;

    if (jsonConfig.contains("audio_token_id"))
    {
        mConfig.audioTokenId = jsonConfig["audio_token_id"].get<int32_t>();
        // boa/eoa delimit each audio span so the model can locate it (mirrors
        // the HF processor layout).
        mConfig.beginAudioTokenId = jsonConfig.value("boa_token_id", -1);
        mConfig.endAudioTokenId = jsonConfig.value("eoa_token_id", -1);
        if (mConfig.beginAudioTokenId < 0 || mConfig.endAudioTokenId < 0)
        {
            LOG_WARNING(
                "Gemma4 audio config has no boa_token_id/eoa_token_id; audio spans will not be delimited and "
                "may be misread. Re-export the model to add them.");
        }
    }
    else
    {
        LOG_ERROR("audio_token_id not found in config.json");
        return false;
    }

    // Optional: max audio clips per request (for pre-allocating embedding buffer)
    if (jsonConfig.contains("builder_config"))
    {
        auto const& builderConfig = jsonConfig["builder_config"];
        if (builderConfig.contains("max_audio_clips_per_request"))
        {
            mConfig.maxAudioClipsPerRequest = builderConfig["max_audio_clips_per_request"].get<int32_t>();
        }
    }

    // Get max sequence length from engine profile
    nvinfer1::Dims const inputShapeMax
        = mAudioEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxSeqLen = inputShapeMax.d[1];

    // Get output feature dimension from engine
    nvinfer1::Dims const outputShape = mAudioEngine->getTensorShape("last_hidden_state");
    mConfig.audioFeatureDim = outputShape.d[2];

    LOG_INFO("Gemma4AudioRunner: melBins=%d, maxSeqLen=%ld, hiddenDim=%d, audioTokenId=%d", mConfig.melBins, mMaxSeqLen,
        mConfig.audioFeatureDim, mConfig.audioTokenId);

    return true;
}

bool Gemma4AudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    bool status{true};

    mInputFeatures = rt::Tensor({1, mMaxSeqLen, static_cast<int64_t>(mConfig.melBins)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4AudioRunner::mInputFeatures");
    status &= mAudioContext->setTensorAddress("input_features", mInputFeatures.rawPointer());

    // Valid mask for the downsampled sequence
    int64_t const maxDownSeqLen = mMaxSeqLen / mConfig.subsamplingFactor;
    mValidMask = rt::Tensor(
        {1, maxDownSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kBOOL, "Gemma4AudioRunner::mValidMask");
    status &= mAudioContext->setTensorAddress("valid", mValidMask.rawPointer());

    // Output embedding buffer — pre-allocate for max possible total rows across all clips.
    // Each clip produces at most maxDownSeqLen rows; we size for maxAudioClipsPerRequest clips
    // to avoid dynamic reallocation during inference.
    int64_t const maxTotalRows = maxDownSeqLen * static_cast<int64_t>(mConfig.maxAudioClipsPerRequest);
    mAudioEmbedding = rt::Tensor({maxTotalRows, static_cast<int64_t>(mConfig.audioFeatureDim)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4AudioRunner::mAudioEmbedding");

    if (!status)
    {
        LOG_ERROR("Failed to set tensor addresses for Gemma4 audio engine");
        return false;
    }

    return true;
}

bool Gemma4AudioRunner::resizeEmbeddingForRows(int64_t rows)
{
    int64_t const hidden = mConfig.audioFeatureDim;
    int64_t const requiredBytes = rows * hidden * static_cast<int64_t>(sizeof(__half));
    if (requiredBytes <= mAudioEmbedding.getMemoryCapacity())
    {
        return mAudioEmbedding.reshape({rows, hidden});
    }
    // Should not reach here if maxAudioClipsPerRequest is configured correctly.
    LOG_WARNING(
        "Gemma4AudioRunner: mAudioEmbedding reallocation at runtime (%ld rows). "
        "Consider increasing maxAudioClipsPerRequest (currently %d).",
        rows, mConfig.maxAudioClipsPerRequest);
    mAudioEmbedding = rt::Tensor(
        {rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Gemma4AudioRunner::mAudioEmbedding");
    return true;
}

bool Gemma4AudioRunner::loadMelSpectrogramFromFile(
    std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream)
{
    if (format != "safetensors")
    {
        LOG_ERROR("Only safetensors format is supported. Got: %s", format.c_str());
        return false;
    }

    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(filePath, tensors, stream))
    {
        LOG_ERROR("Failed to load mel-spectrogram from: %s", filePath.c_str());
        return false;
    }

    check::check(tensors.size() == 1, "Mel-spectrogram safetensors should contain exactly one tensor");
    check::check(tensors[0].getDataType() == nvinfer1::DataType::kHALF, "Mel-spectrogram must be FP16");

    melSpectrogram = std::move(tensors[0]);
    return true;
}

bool Gemma4AudioRunner::extractMelFromPcm(rt::audio::AudioPCM const& pcm, rt::Tensor& melGpu, cudaStream_t stream)
{
    // HF Gemma4AudioFeatureExtractor pads the raw waveform to a multiple of 128 samples
    // before mel extraction (pad_to_multiple_of=128). Replicate this to match frame counts.
    static constexpr int32_t kPadToMultipleOf = 128;
    rt::audio::AudioPCM paddedPcm = pcm;
    size_t const rawLen = paddedPcm.samples.size();
    size_t const paddedLen = ((rawLen + kPadToMultipleOf - 1) / kPadToMultipleOf) * kPadToMultipleOf;
    if (paddedLen > rawLen)
    {
        paddedPcm.samples.resize(paddedLen, 0.0f);
    }

    // Extract mel-spectrogram on CPU. Output shape: [mel_bins, T] (kMelTime layout).
    rt::Tensor hostMel;
    if (!mMelExtractor.extract(paddedPcm, hostMel))
    {
        LOG_ERROR("Gemma4AudioRunner: mel extraction from PCM failed");
        return false;
    }

    // Transpose [mel_bins, T] → [T, mel_bins] and cast FP32→FP16, then upload as [1, T, mel_bins].
    int64_t const melBins = hostMel.getShape()[0];
    int64_t const timeSteps = hostMel.getShape()[1];
    int64_t const numel = melBins * timeSteps;

    std::vector<__half> transposedHalf(static_cast<size_t>(numel));
    float const* src = hostMel.dataPointer<float>();
    for (int64_t t = 0; t < timeSteps; ++t)
    {
        for (int64_t m = 0; m < melBins; ++m)
        {
            transposedHalf[static_cast<size_t>(t * melBins + m)] = __float2half(src[m * timeSteps + t]);
        }
    }

    melGpu = rt::Tensor(
        {1, timeSteps, melBins}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Gemma4AudioRunner::melFromPcm");
    CUDA_CHECK(cudaMemcpyAsync(melGpu.rawPointer(), transposedHalf.data(), static_cast<size_t>(numel) * sizeof(__half),
        cudaMemcpyHostToDevice, stream));

    LOG_INFO("Gemma4AudioRunner: extracted mel from PCM (%zu samples @ %d Hz) -> [1, %ld, %ld]", pcm.samples.size(),
        pcm.sampleRate, timeSteps, melBins);
    return true;
}

bool Gemma4AudioRunner::encodeSingleClip(
    rt::Tensor const& mel, int64_t destRowOffset, int64_t& encodedRowsOut, cudaStream_t stream)
{
    // Mel shape: [1, rawSeqLen, mel_bins].
    // Gemma4 uses 2x stride-2 conv → output seq_len = rawSeqLen / 4
    int64_t const rawSeqLen = mel.getShape()[1];
    int64_t const paddedSeqLen = alignUp(rawSeqLen, mConfig.subsamplingFactor);
    int64_t const encodedSeqLen = paddedSeqLen / mConfig.subsamplingFactor;

    // Set up input features (zero-padded)
    check::check(
        mInputFeatures.reshape({1, paddedSeqLen, static_cast<int64_t>(mConfig.melBins)}), "Tensor reshape failed");
    if (paddedSeqLen > rawSeqLen)
    {
        CUDA_CHECK(cudaMemsetAsync(mInputFeatures.rawPointer(), 0, mInputFeatures.getMemoryCapacity(), stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mInputFeatures.rawPointer(), mel.rawPointer(), mel.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));

    // Set up valid mask — true for real encoded positions, false for padding.
    check::check(mValidMask.reshape({1, encodedSeqLen}), "Valid mask reshape failed");
    // Ceil, matching the HF encoder mask: a trailing partial group still
    // carries real content and yields a valid soft token.
    int64_t const validPositions = encodedSeqLen;
    CUDA_CHECK(cudaMemsetAsync(mValidMask.rawPointer(), 0, encodedSeqLen * sizeof(bool), stream));
    if (validPositions > 0)
    {
        CUDA_CHECK(cudaMemsetAsync(mValidMask.rawPointer(), 1, validPositions * sizeof(bool), stream));
    }

    // Bind encoder output to the correct offset in mAudioEmbedding
    auto* const embeddingBase = static_cast<std::byte*>(mAudioEmbedding.rawPointer());
    int64_t const destByteOffset
        = destRowOffset * static_cast<int64_t>(mConfig.audioFeatureDim) * static_cast<int64_t>(sizeof(__half));

    bool status{true};
    status &= mAudioContext->setInputShape("input_features", mInputFeatures.getShape().getTRTDims());
    status &= mAudioContext->setInputShape("valid", mValidMask.getShape().getTRTDims());
    status &= mAudioContext->setTensorAddress("last_hidden_state", embeddingBase + destByteOffset);
    if (!status)
    {
        LOG_ERROR("Gemma4AudioRunner::encodeSingleClip: Failed to set encoder bindings");
        return false;
    }

    if (!mAudioContext->enqueueV3(stream))
    {
        LOG_ERROR("Gemma4AudioRunner::encodeSingleClip: Failed to enqueue encoder");
        return false;
    }

    mMultimodalMetrics.recordRun(1, encodedSeqLen);
    // Return valid (non-padding) positions so downstream token expansion matches
    // the actual encoded content. The encoder's valid mask zeroes padding outputs.
    encodedRowsOut = validPositions;
    return true;
}

bool Gemma4AudioRunner::encodeAllClips(
    rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths, cudaStream_t stream)
{
    // Two-pass: load all clips first, then encode sequentially
    std::vector<rt::Tensor> mels;
    int64_t totalEncodedRows = 0;

    for (auto const& req : request.requests)
    {
        for (auto const& audio : req.audioBuffers)
        {
            rt::Tensor mel;
            if (!audio.melSpectrogramPath.empty())
            {
                // Pre-computed mel-spectrogram path (safetensors).
                if (!loadMelSpectrogramFromFile(audio.melSpectrogramPath, audio.melSpectrogramFormat, mel, stream))
                {
                    LOG_ERROR("Failed to load mel-spectrogram from %s", audio.melSpectrogramPath.c_str());
                    return false;
                }
            }
            else if (audio.pcm)
            {
                // Fallback: extract mel from raw PCM audio data.
                if (!extractMelFromPcm(*audio.pcm, mel, stream))
                {
                    LOG_ERROR("Failed to extract mel-spectrogram from PCM audio");
                    return false;
                }
            }
            else
            {
                LOG_ERROR(
                    "Gemma4 audio runner requires either melSpectrogramPath or PCM audio data; "
                    "neither is available.");
                return false;
            }
            int64_t const encodedSeqLen
                = alignUp(mel.getShape()[1], mConfig.subsamplingFactor) / mConfig.subsamplingFactor;
            totalEncodedRows += encodedSeqLen;
            mels.emplace_back(std::move(mel));
        }
    }

    if (totalEncodedRows == 0)
    {
        return mAudioEmbedding.reshape({0, static_cast<int64_t>(mConfig.audioFeatureDim)});
    }

    if (!resizeEmbeddingForRows(totalEncodedRows))
    {
        LOG_ERROR("Failed to size mAudioEmbedding for %ld rows", totalEncodedRows);
        return false;
    }

    int64_t rowOffset = 0;
    for (auto const& mel : mels)
    {
        int64_t encodedRows = 0;
        if (!encodeSingleClip(mel, rowOffset, encodedRows, stream))
        {
            return false;
        }
        audioTokenLengths.push_back(encodedRows);
        rowOffset += encodedRows;
    }
    return true;
}

void Gemma4AudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int64_t audioIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        // Move tokens out of batchInputIds[i] into local `ids` for expansion.
        // The moved-from slot is reassigned with expanded tokens below.
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        bool const wrapAudio = mConfig.beginAudioTokenId >= 0 && mConfig.endAudioTokenId >= 0;
        // Per-request media window (one entry per clip): a placeholder may only consume this request's own clips (batch
        // isolation).
        int64_t const mediaEnd = audioIndex + static_cast<int64_t>(request.requests[i].audioBuffers.size());

        std::vector<int32_t> newIds;
        size_t expandedSize = 0;
        int64_t tmpAudioIdx = audioIndex;
        for (auto const& id : ids)
        {
            expandedSize += (id == mConfig.audioTokenId && tmpAudioIdx < mediaEnd)
                ? audioTokenLengths.at(tmpAudioIdx++) + (wrapAudio ? 2 : 0)
                : 1;
        }
        newIds.reserve(expandedSize);
        for (size_t tokenIndex = 0; tokenIndex < ids.size(); ++tokenIndex)
        {
            int32_t const id = ids[tokenIndex];
            if (id == mConfig.audioTokenId)
            {
                ELLM_CHECK(audioIndex < mediaEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: Gemma4AudioRunner::textPreprocess() placeholder count exceeds this "
                    "request's clip count");
                bool const alreadyHasBegin
                    = wrapAudio && tokenIndex > 0 && ids[tokenIndex - 1] == mConfig.beginAudioTokenId;
                bool const alreadyHasEnd
                    = wrapAudio && tokenIndex + 1 < ids.size() && ids[tokenIndex + 1] == mConfig.endAudioTokenId;
                if (wrapAudio && !alreadyHasBegin)
                {
                    newIds.push_back(mConfig.beginAudioTokenId);
                }
                int64_t const numAudioTokens = audioTokenLengths.at(audioIndex);
                for (int64_t k = 0; k < numAudioTokens; ++k)
                {
                    newIds.push_back(mConfig.audioTokenId);
                }
                if (wrapAudio && !alreadyHasEnd)
                {
                    newIds.push_back(mConfig.endAudioTokenId);
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
        ELLM_CHECK(audioIndex == mediaEnd,
            "EDGELLM_BAD_MEDIA_COUNT: Gemma4AudioRunner::textPreprocess() placeholder count is smaller than this "
            "request's clip count");
    }
}

bool Gemma4AudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, [[maybe_unused]] bool imageOnly,
    [[maybe_unused]] bool skipEncoderWork)
{
    std::vector<int64_t> audioTokenLengths;

    try
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);
        if (!encodeAllClips(request, audioTokenLengths, stream))
        {
            return false;
        }
        textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        bool const actionable = isCallerActionable(e);
        if (!actionable)
        {
            LOG_ERROR("Gemma4AudioRunner::preprocess failed: %s", e.what());
        }
        // Drain in-flight encoder work still reading the request's PCM, so the caller can safely
        // release it -- including when the error propagates.
        cudaStreamSynchronize(stream);
        if (actionable)
        {
            throw;
        }
        return false;
    }

    mLastMediaTokenLengths = audioTokenLengths;
    return true;
}

bool Gemma4AudioRunner::infer([[maybe_unused]] cudaStream_t stream)
{
    // Encoder is invoked per-clip from preprocess(); nothing left to do here.
    return true;
}

rt::Tensor& Gemma4AudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

} // namespace rt
} // namespace trt_edgellm
