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

#include "multimodal/gemma4/gemma4UnifiedAudioRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/mmapReader.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

Gemma4UnifiedAudioRunner::Gemma4UnifiedAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    std::string const enginePath = engineDir + "/audio_encoder.engine";
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mAudioEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    ELLM_CHECK(mAudioEngine != nullptr, "Failed to deserialize Gemma4 Unified audio engine");
    mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mAudioContext != nullptr, "Failed to create Gemma4 Unified audio execution context");
    ELLM_CHECK(mAudioContext->setOptimizationProfileAsync(0, stream),
        "Failed to select Gemma4 Unified audio optimization profile");
    ELLM_CHECK(validateAndFillConfig(engineDir), "Gemma4UnifiedAudioRunner: invalid configuration or engine");
    ELLM_CHECK(allocateBuffer(stream), "Gemma4UnifiedAudioRunner: failed to allocate buffers");
}

bool Gemma4UnifiedAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json config;
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configStream(configPath);
    if (!configStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }
    try
    {
        config = Json::parse(configStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse %s: %s", configPath.c_str(), e.what());
        return false;
    }

    std::string const modelType = config.value("model_type", "");
    mModelType = multimodal::stringToModelType(modelType);
    if (mModelType != multimodal::ModelType::GEMMA4_UNIFIED_AUDIO)
    {
        LOG_ERROR("Invalid model type for Gemma4 Unified audio: %s", modelType.c_str());
        return false;
    }
    if (!config.contains("audio_config") || !config.contains("audio_token_id") || !config.contains("boa_token_id")
        || !config.contains("eoa_token_index"))
    {
        LOG_ERROR(
            "Gemma4 Unified audio config requires audio_config, audio_token_id, boa_token_id, and "
            "eoa_token_index");
        return false;
    }
    auto const& audioConfig = config["audio_config"];
    mConfig.samplesPerFrame
        = audioConfig.value("audio_samples_per_token", audioConfig.value("audio_embed_dim", int64_t{0}));
    mConfig.audioTokenId = config["audio_token_id"].get<int32_t>();
    mConfig.beginAudioTokenId = config["boa_token_id"].get<int32_t>();
    mConfig.endAudioTokenId = config["eoa_token_index"].get<int32_t>();
    Json const& featureExtractor = config.contains("feature_extractor") ? config["feature_extractor"] : audioConfig;
    mConfig.sampleRate = featureExtractor.value("sampling_rate", int32_t{16000});
    int64_t const featureSize = featureExtractor.value("feature_size", mConfig.samplesPerFrame);
    float const paddingValue = featureExtractor.value("padding_value", 0.0F);

    if (mAudioEngine->getTensorIOMode(binding_names::kAudioInputFeatures) != nvinfer1::TensorIOMode::kINPUT
        || mAudioEngine->getTensorIOMode(binding_names::kAudioOutput) != nvinfer1::TensorIOMode::kOUTPUT)
    {
        LOG_ERROR("Gemma4 Unified audio engine is missing input_features or last_hidden_state");
        return false;
    }
    if (mAudioEngine->getTensorDataType(binding_names::kAudioInputFeatures) != nvinfer1::DataType::kHALF
        || mAudioEngine->getTensorDataType(binding_names::kAudioOutput) != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("Gemma4 Unified audio engine requires FP16 input and output");
        return false;
    }
    nvinfer1::Dims const inputMin
        = mAudioEngine->getProfileShape(binding_names::kAudioInputFeatures, 0, nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims const inputMax
        = mAudioEngine->getProfileShape(binding_names::kAudioInputFeatures, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const outputShape = mAudioEngine->getTensorShape(binding_names::kAudioOutput);
    if (inputMin.nbDims != 3 || inputMax.nbDims != 3 || outputShape.nbDims != 3)
    {
        LOG_ERROR("Gemma4 Unified audio bindings have unexpected ranks");
        return false;
    }
    mConfig.minFrames = inputMin.d[1];
    mConfig.maxFrames = inputMax.d[1];
    mConfig.outputHiddenSize = outputShape.d[2];
    if (inputMin.d[0] != 1 || inputMax.d[0] != 1 || outputShape.d[0] != 1)
    {
        LOG_ERROR("Gemma4 Unified audio bindings must have batch size 1, got input [%ld,%ld] and output %ld",
            inputMin.d[0], inputMax.d[0], outputShape.d[0]);
        return false;
    }
    if (inputMin.d[2] != 640 || inputMax.d[2] != 640)
    {
        LOG_ERROR("Gemma4 Unified audio input_features must have frame size 640, got [%ld,%ld]", inputMin.d[2],
            inputMax.d[2]);
        return false;
    }
    if (mConfig.samplesPerFrame != 640 || featureSize != 640)
    {
        LOG_ERROR("Gemma4 Unified audio requires 640 samples per frame, got samples_per_frame=%ld, feature_size=%ld",
            mConfig.samplesPerFrame, featureSize);
        return false;
    }
    if (paddingValue != 0.0F)
    {
        LOG_ERROR("Gemma4 Unified audio requires padding_value=0, got %f", static_cast<double>(paddingValue));
        return false;
    }
    if (mConfig.sampleRate != 16000)
    {
        LOG_ERROR("Gemma4 Unified audio requires a 16-kHz sampling_rate, got %d", mConfig.sampleRate);
        return false;
    }
    if (mConfig.minFrames != 1 || mConfig.maxFrames < mConfig.minFrames)
    {
        LOG_ERROR("Gemma4 Unified audio requires a one-frame profile minimum, got frames=[%ld,%ld]", mConfig.minFrames,
            mConfig.maxFrames);
        return false;
    }
    if (mConfig.outputHiddenSize != 3840)
    {
        LOG_ERROR(
            "Gemma4 Unified audio last_hidden_state must have hidden size 3840, got %ld", mConfig.outputHiddenSize);
        return false;
    }
    LOG_INFO("Gemma4 Unified audio: profile raw PCM frames=[%ld,%ld], samples_per_frame=%ld, hidden=%ld",
        mConfig.minFrames, mConfig.maxFrames, mConfig.samplesPerFrame, mConfig.outputHiddenSize);
    return true;
}

bool Gemma4UnifiedAudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    mInputFeatures = rt::Tensor({1, mConfig.maxFrames, mConfig.samplesPerFrame}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4UnifiedAudioRunner::mInputFeatures");
    mInputFeaturesHost = rt::Tensor({1, mConfig.maxFrames, mConfig.samplesPerFrame}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kHALF, "Gemma4UnifiedAudioRunner::mInputFeaturesHost");
    mOutputEmbedding = rt::Tensor({mConfig.maxFrames, mConfig.outputHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4UnifiedAudioRunner::mOutputEmbedding");
    bool status = true;
    status &= mAudioContext->setTensorAddress(binding_names::kAudioInputFeatures, mInputFeatures.rawPointer());
    status &= mAudioContext->setTensorAddress(binding_names::kAudioOutput, mOutputEmbedding.rawPointer());
    if (!status)
    {
        LOG_ERROR("Failed to set Gemma4 Unified audio tensor addresses");
        return false;
    }
    return true;
}

void Gemma4UnifiedAudioRunner::frameAudioTokenLengthsOnly(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& audioTokenLengths, std::vector<int64_t>& audiosPerRequest)
{
    int64_t totalFrames = 0;
    for (auto const& req : request.requests)
    {
        audiosPerRequest.push_back(static_cast<int64_t>(req.audioBuffers.size()));
        for (auto const& audio : req.audioBuffers)
        {
            ELLM_CHECK(audio.pcm != nullptr, "Gemma4 Unified AudioData.pcm is null");
            int64_t const sampleCount = static_cast<int64_t>(audio.pcm->samples.size());
            int64_t const frameCount = (sampleCount + mConfig.samplesPerFrame - 1) / mConfig.samplesPerFrame;
            totalFrames += frameCount;
            audioTokenLengths.push_back(frameCount);
        }
    }
    if (totalFrames > 0)
    {
        check::check(mOutputEmbedding.reshape({totalFrames, mConfig.outputHiddenSize}), "Tensor reshape failed");
    }
}

void Gemma4UnifiedAudioRunner::frameAudio(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& audioTokenLengths, std::vector<int64_t>& audiosPerRequest, cudaStream_t stream)
{
    check::check(mInputFeaturesHost.reshape({1, mConfig.maxFrames, mConfig.samplesPerFrame}), "Tensor reshape failed");
    half* const framedPcm = mInputFeaturesHost.dataPointer<half>();
    int64_t totalFrames = 0;
    for (auto const& req : request.requests)
    {
        audiosPerRequest.push_back(static_cast<int64_t>(req.audioBuffers.size()));
        for (auto const& audio : req.audioBuffers)
        {
            ELLM_CHECK(audio.pcm != nullptr, "Gemma4 Unified AudioData.pcm is null");
            ELLM_CHECK(audio.pcm->sampleRate == mConfig.sampleRate && audio.pcm->numChannels == 1,
                "Gemma4 Unified audio requires mono 16-kHz PCM");
            ELLM_CHECK(!audio.pcm->samples.empty(), "Gemma4 Unified audio clip is empty");
            int64_t const sampleCount = static_cast<int64_t>(audio.pcm->samples.size());
            int64_t const frameCount = (sampleCount + mConfig.samplesPerFrame - 1) / mConfig.samplesPerFrame;
            ELLM_CHECK(totalFrames + frameCount <= mConfig.maxFrames,
                "Gemma4 Unified audio request exceeds engine profile maximum");

            int64_t const frameStart = totalFrames * mConfig.samplesPerFrame;
            int64_t const framedSampleCount = frameCount * mConfig.samplesPerFrame;
            std::fill(framedPcm + frameStart, framedPcm + frameStart + framedSampleCount, __float2half(0.0F));
            for (int64_t i = 0; i < sampleCount; ++i)
            {
                framedPcm[frameStart + i] = __float2half(audio.pcm->samples[static_cast<size_t>(i)]);
            }
            totalFrames += frameCount;
            audioTokenLengths.push_back(frameCount);
        }
    }

    ELLM_CHECK(totalFrames >= mConfig.minFrames && totalFrames <= mConfig.maxFrames,
        "Gemma4 Unified framed audio is outside the engine profile");
    check::check(mInputFeaturesHost.reshape({1, totalFrames, mConfig.samplesPerFrame}), "Tensor reshape failed");
    check::check(mInputFeatures.reshape({1, totalFrames, mConfig.samplesPerFrame}), "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalFrames, mConfig.outputHiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mInputFeatures.rawPointer(), mInputFeaturesHost.rawPointer(),
        totalFrames * mConfig.samplesPerFrame * static_cast<int64_t>(sizeof(half)), cudaMemcpyHostToDevice, stream));

    int64_t const audioCount = std::accumulate(audiosPerRequest.begin(), audiosPerRequest.end(), int64_t{0});
    mMultimodalMetrics.recordRun(0, 0, audioCount, totalFrames);
}

void Gemma4UnifiedAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, std::vector<int64_t> const& audioTokenLengths,
    std::vector<int64_t> const& audiosPerRequest, tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(audiosPerRequest.size() == request.requests.size(), "Invalid Gemma4 Unified audio request metadata");
    size_t audioIndex = 0;
    for (size_t requestIndex = 0; requestIndex < request.requests.size(); ++requestIndex)
    {
        std::vector<int32_t> ids = requestIndex < batchedInputIds.size() && !batchedInputIds[requestIndex].empty()
            ? batchedInputIds[requestIndex]
            : tokenizer->encode(request.formattedRequests[requestIndex].formattedCompleteRequest);
        check::check(!ids.empty(), "Gemma4 Unified audio failed to tokenize text");
        size_t const expectedEnd = audioIndex + static_cast<size_t>(audiosPerRequest[requestIndex]);
        std::vector<int32_t> expanded;
        for (size_t tokenIndex = 0; tokenIndex < ids.size(); ++tokenIndex)
        {
            int32_t const token = ids[tokenIndex];
            if (token == mConfig.audioTokenId)
            {
                ELLM_CHECK(audioIndex < expectedEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: too many audio placeholders in Gemma4 Unified prompt");
                bool const alreadyHasBegin = tokenIndex > 0 && ids[tokenIndex - 1] == mConfig.beginAudioTokenId;
                bool const alreadyHasEnd
                    = tokenIndex + 1 < ids.size() && ids[tokenIndex + 1] == mConfig.endAudioTokenId;
                if (!alreadyHasBegin)
                {
                    expanded.push_back(mConfig.beginAudioTokenId);
                }
                expanded.insert(expanded.end(), static_cast<size_t>(audioTokenLengths.at(audioIndex)), token);
                if (!alreadyHasEnd)
                {
                    expanded.push_back(mConfig.endAudioTokenId);
                }
                ++audioIndex;
            }
            else
            {
                expanded.push_back(token);
            }
        }
        ELLM_CHECK(audioIndex == expectedEnd,
            "EDGELLM_BAD_MEDIA_COUNT: audio placeholder count does not match Gemma4 Unified clip count");
        if (requestIndex < batchedInputIds.size())
        {
            batchedInputIds[requestIndex] = std::move(expanded);
        }
        else
        {
            batchedInputIds.push_back(std::move(expanded));
        }
    }
    ELLM_CHECK(audioIndex == audioTokenLengths.size(), "Unused Gemma4 Unified audio embeddings");
}

bool Gemma4UnifiedAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly, bool skipEncoderWork)
{
    try
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);
        std::vector<int64_t> audioTokenLengths;
        std::vector<int64_t> audiosPerRequest;
        if (skipEncoderWork)
        {
            frameAudioTokenLengthsOnly(request, audioTokenLengths, audiosPerRequest);
        }
        else
        {
            frameAudio(request, audioTokenLengths, audiosPerRequest, stream);
        }
        if (!imageOnly)
        {
            textPreprocess(request, batchedInputIds, audioTokenLengths, audiosPerRequest, tokenizer);
        }
        mLastMediaTokenLengths = audioTokenLengths;
        return true;
    }
    catch (std::exception const& e)
    {
        bool const actionable = isCallerActionable(e);
        if (!actionable)
        {
            LOG_ERROR("Gemma4 Unified audio preprocessing failed: %s", e.what());
        }
        // Drain the async H2D copy still reading the framing staging buffer, so the next request
        // cannot overwrite it mid-flight -- including when the error propagates.
        cudaStreamSynchronize(stream);
        if (actionable)
        {
            throw;
        }
        return false;
    }
}

bool Gemma4UnifiedAudioRunner::infer(cudaStream_t stream) noexcept
{
    TIME_STAGE(metrics::StageNames::kAUDIO_ENCODER, stream);
    if (!mAudioContext->setInputShape(binding_names::kAudioInputFeatures, mInputFeatures.getShape().getTRTDims()))
    {
        LOG_ERROR("Failed to set Gemma4 Unified audio input shape");
        return false;
    }
    if (!mAudioContext->enqueueV3(stream))
    {
        LOG_ERROR("Failed to enqueue Gemma4 Unified audio engine");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
