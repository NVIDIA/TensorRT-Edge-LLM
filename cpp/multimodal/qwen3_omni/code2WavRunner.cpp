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

#include "multimodal/qwen3_omni/code2WavRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{
namespace
{

std::filesystem::path resolveCheckpointDir(std::filesystem::path const& checkpointDir)
{
    if (checkpointDir.empty())
    {
        return checkpointDir;
    }

    std::filesystem::path const speechTokenizerDir = checkpointDir / "speech_tokenizer";
    return std::filesystem::is_directory(speechTokenizerDir) ? speechTokenizerDir : checkpointDir;
}

} // namespace

Code2WavRunner::Code2WavRunner(std::string const& engineDir, cudaStream_t stream, std::string const& checkpointDir)
{
    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    std::string const code2wavEnginePath = engineDir + "/code2wav.engine";
    ELLM_CHECK(std::filesystem::exists(code2wavEnginePath), "Code2Wav engine not found at " + code2wavEnginePath);

    try
    {
        mCode2WavEngine = deserializeCudaEngineFromFile(*mRuntime, code2wavEnginePath);

        mCode2WavContext = std::unique_ptr<nvinfer1::IExecutionContext>(mCode2WavEngine->createExecutionContext());
        ELLM_CHECK(mCode2WavContext, "Failed to create Code2Wav execution context");

        bool const profileSet = mCode2WavContext->setOptimizationProfileAsync(0, stream);
        ELLM_CHECK(profileSet, "Failed to set optimization profile");

        setNonBlockingAuxStreams(mCode2WavContext.get(), mCode2WavEngine.get(), mAuxStreams);

        mExternalWeights.load(std::filesystem::path(engineDir), std::filesystem::path(engineDir) / "config.json",
            stream, resolveCheckpointDir(checkpointDir));
        mExternalWeights.bindToContext(*mCode2WavEngine, *mCode2WavContext, "code2wav");

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Code2Wav engine: %s", e.what());
        throw;
    }

    bool const bufferAllocated = allocateBuffer();
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    LOG_INFO("Code2Wav runner initialized successfully");
}

bool Code2WavRunner::validateAndFillConfig(std::string const& engineDir)
{
    std::string const configPath = engineDir + "/config.json";
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

    Json const& cfg = jsonConfig.contains("code2wav_config") ? jsonConfig["code2wav_config"] : jsonConfig;

    if (!cfg.contains("num_quantizers"))
    {
        LOG_ERROR("num_quantizers not found in config.json");
        return false;
    }
    mConfig.numQuantizers = cfg["num_quantizers"].get<int32_t>();

    mConfig.codebookSize = cfg.value("codebook_size", mConfig.codebookSize);
    mConfig.hiddenSize = cfg.value("hidden_size", mConfig.hiddenSize);
    mConfig.decoderDim = cfg.value("decoder_dim", mConfig.decoderDim);

    int64_t rate = 1;
    if (cfg.contains("upsample_rates"))
    {
        for (auto const& r : cfg["upsample_rates"])
        {
            rate *= r.get<int64_t>();
        }
    }
    if (cfg.contains("upsampling_ratios"))
    {
        for (auto const& r : cfg["upsampling_ratios"])
        {
            rate *= r.get<int64_t>();
        }
    }
    if (rate > 1)
    {
        mConfig.upsampleRate = math::cast<int32_t>(rate);
    }
    else
    {
        LOG_ERROR("Failed to calculate upsample_rate from config");
        return false;
    }

    if (jsonConfig.contains("builder_config"))
    {
        auto const& builderConfig = jsonConfig["builder_config"];
        mConfig.chunkSize = builderConfig.value("opt_code_len", mConfig.chunkSize);
    }

    LOG_INFO("Code2Wav config: numQuantizers=%d, codebookSize=%d, upsampleRate=%d, chunkSize=%d", mConfig.numQuantizers,
        mConfig.codebookSize, mConfig.upsampleRate, mConfig.chunkSize);

    return true;
}

bool Code2WavRunner::allocateBuffer()
{
    if (!mCode2WavEngine || !mCode2WavContext)
    {
        LOG_ERROR("Cannot allocate buffers - engine not loaded");
        return false;
    }

    nvinfer1::Dims const codesShapeMax
        = mCode2WavEngine->getProfileShape(binding_names::kCode2WavCodes, 0, nvinfer1::OptProfileSelector::kMAX);

    int64_t const maxSeqLen = codesShapeMax.d[2];
    int64_t const maxWaveformLen = maxSeqLen * mConfig.upsampleRate;

    // Detect engine's actual waveform output dtype (FP16 for Qwen3-Omni code2wav,
    // FP32 for Qwen3-TTS tokenizer_decoder). Allocating the wrong dtype reinterprets bytes
    // and produces garbled audio.
    mWaveformDtype = mCode2WavEngine->getTensorDataType(binding_names::kCode2WavWaveform);
    LOG_INFO("Code2Wav waveform output dtype: %s",
        mWaveformDtype == nvinfer1::DataType::kHALF        ? "FP16"
            : mWaveformDtype == nvinfer1::DataType::kFLOAT ? "FP32"
                                                           : "UNKNOWN");

    mInputCodesDevice
        = rt::Tensor({1, mConfig.numQuantizers, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    mOutputWaveform = rt::Tensor({1, 1, maxWaveformLen}, rt::DeviceType::kGPU, mWaveformDtype);
    mInputCodesHost
        = rt::Tensor({1, mConfig.numQuantizers, maxSeqLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64);

    bool setTensorAddressStatus = true;
    setTensorAddressStatus
        &= mCode2WavContext->setTensorAddress(binding_names::kCode2WavCodes, mInputCodesDevice.rawPointer());
    setTensorAddressStatus
        &= mCode2WavContext->setTensorAddress(binding_names::kCode2WavWaveform, mOutputWaveform.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor addresses");
        return false;
    }

    LOG_INFO("Buffers allocated: maxSeqLen=%ld, maxWaveformLen=%ld", maxSeqLen, maxWaveformLen);
    return true;
}

bool Code2WavRunner::infer(cudaStream_t stream)
{
    TIME_STAGE(metrics::StageNames::kCODE2WAV, stream);

    nvinfer1::Dims const codeDims = mInputCodesDevice.getTRTDims();
    if (!mCode2WavContext->setInputShape(binding_names::kCode2WavCodes, codeDims))
    {
        LOG_ERROR("Failed to set input shape");
        return false;
    }

    if (!mCode2WavContext->enqueueV3(stream))
    {
        LOG_ERROR("Inference failed");
        return false;
    }

    return true;
}

bool Code2WavRunner::prepareCodes(std::vector<std::vector<int32_t>> const& codes, cudaStream_t stream)
{
    if (codes.empty() || codes[0].empty())
    {
        LOG_ERROR("Empty codes provided");
        return false;
    }

    int64_t const numLayers = math::cast<int64_t>(codes.size());
    int64_t const seqLen = math::cast<int64_t>(codes[0].size());

    if (numLayers != mConfig.numQuantizers)
    {
        LOG_ERROR("Expected %d quantizer layers, got %ld", mConfig.numQuantizers, numLayers);
        return false;
    }

    for (size_t i = 1; i < codes.size(); ++i)
    {
        if (math::cast<int64_t>(codes[i].size()) != seqLen)
        {
            LOG_ERROR("Inconsistent code lengths: layer 0 has %ld, layer %zu has %zu", seqLen, i, codes[i].size());
            return false;
        }
    }

    if (!mInputCodesDevice.reshape({1, numLayers, seqLen}))
    {
        LOG_ERROR("Failed to reshape input codes tensor");
        return false;
    }

    int64_t* const hostData = static_cast<int64_t*>(mInputCodesHost.rawPointer());
    for (int64_t layer = 0; layer < numLayers; ++layer)
    {
        for (int64_t t = 0; t < seqLen; ++t)
        {
            hostData[layer * seqLen + t] = math::cast<int64_t>(codes[layer][t]);
        }
    }

    size_t const copySize = math::cast<size_t>(numLayers * seqLen) * sizeof(int64_t);
    CUDA_CHECK(cudaMemcpyAsync(mInputCodesDevice.rawPointer(), hostData, copySize, cudaMemcpyHostToDevice, stream));

    return true;
}

bool Code2WavRunner::runChunkedInference(
    std::vector<std::vector<int32_t>> const& codes, rt::Tensor& outputWaveform, cudaStream_t stream)
{
    int64_t const totalLen = math::cast<int64_t>(codes[0].size());
    int64_t const chunkSize = mConfig.chunkSize;
    int64_t const contextSize = mConfig.leftContextSize;

    std::vector<rt::Tensor> waveformChunks;
    std::vector<std::vector<int32_t>> chunkCodes(mConfig.numQuantizers);
    int64_t startIdx = 0;
    int32_t chunkIdx = 0;

    while (startIdx < totalLen)
    {
        int64_t const endIdx = std::min(startIdx + chunkSize, totalLen);
        int64_t const actualContextSize = (startIdx >= contextSize) ? contextSize : startIdx;
        int64_t const actualChunkLen = (endIdx - startIdx) + actualContextSize;

        // Extract chunk with context
        for (int32_t layer = 0; layer < mConfig.numQuantizers; ++layer)
        {
            chunkCodes[layer].assign(
                codes[layer].begin() + (startIdx - actualContextSize), codes[layer].begin() + endIdx);
        }

        if (!prepareCodes(chunkCodes, stream))
        {
            return false;
        }

        if (!infer(stream))
        {
            return false;
        }

        // Get actual output length from engine (shorter than frames * upsampleRate
        // due to CausalConvNet padding loss in Code2Wav)
        nvinfer1::Dims const outDims = mCode2WavContext->getTensorShape(binding_names::kCode2WavWaveform);
        int64_t const actualOutputLen = outDims.d[outDims.nbDims - 1];

        // Trim context from the beginning (matches PyTorch: wav[..., context * upsample :])
        int64_t const contextSamples = actualContextSize * mConfig.upsampleRate;
        int64_t const validLen = actualOutputLen - contextSamples;

        LOG_DEBUG("Chunk %d: codes_len=%ld, context=%ld, actual_output=%ld, context_samples=%ld, valid_len=%ld",
            chunkIdx, actualChunkLen, actualContextSize, actualOutputLen, contextSamples, validLen);

        rt::Tensor chunkWaveform({1, 1, validLen}, rt::DeviceType::kGPU, mWaveformDtype);
        CUDA_CHECK(cudaMemcpyAsync(chunkWaveform.rawPointer(),
            static_cast<char*>(mOutputWaveform.rawPointer()) + contextSamples * rt::utils::getTypeSize(mWaveformDtype),
            math::cast<size_t>(validLen) * rt::utils::getTypeSize(mWaveformDtype), cudaMemcpyDeviceToDevice, stream));

        waveformChunks.push_back(std::move(chunkWaveform));
        startIdx = endIdx;
        ++chunkIdx;
    }

    LOG_INFO("Processed %d chunks", chunkIdx);

    int64_t totalSamples = 0;
    for (auto const& chunk : waveformChunks)
    {
        totalSamples += chunk.getShape()[2];
    }

    outputWaveform = rt::Tensor({1, 1, totalSamples}, rt::DeviceType::kGPU, mWaveformDtype);

    int64_t offset = 0;
    for (auto const& chunk : waveformChunks)
    {
        int64_t const chunkLen = chunk.getShape()[2];
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(outputWaveform.rawPointer()) + offset * rt::utils::getTypeSize(mWaveformDtype),
            chunk.rawPointer(), math::cast<size_t>(chunkLen) * rt::utils::getTypeSize(mWaveformDtype),
            cudaMemcpyDeviceToDevice, stream));
        offset += chunkLen;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool Code2WavRunner::generateWaveform(
    std::vector<std::vector<int32_t>> const& codes, rt::audioUtils::AudioData& outputAudio, cudaStream_t stream)
{
    if (codes.empty() || codes[0].empty())
    {
        LOG_ERROR("Empty codes provided");
        return false;
    }

    int64_t const seqLen = math::cast<int64_t>(codes[0].size());
    // Query engine's actual max profile (not mInputCodesDevice.getShape() which is mutated by prepareCodes).
    nvinfer1::Dims const codesShapeMax
        = mCode2WavEngine->getProfileShape(binding_names::kCode2WavCodes, 0, nvinfer1::OptProfileSelector::kMAX);
    int64_t const maxCodeLen = codesShapeMax.d[2];
    int64_t waveformLen = 0;

    // Use direct inference if sequence fits in engine's max capacity
    if (seqLen <= maxCodeLen)
    {
        LOG_DEBUG("Direct inference: seqLen=%ld", seqLen);

        if (!prepareCodes(codes, stream))
        {
            return false;
        }

        if (!infer(stream))
        {
            return false;
        }

        // Get actual output shape from the engine. Code2Wav output can be shorter than
        // seqLen * upsampleRate because the model trims samples at the boundaries.
        nvinfer1::Dims const outDims = mCode2WavContext->getTensorShape(binding_names::kCode2WavWaveform);
        int64_t const engineWaveformLen = outDims.d[outDims.nbDims - 1];
        int64_t const expectedWaveformLen = seqLen * mConfig.upsampleRate;
        // Some vocoder ONNX (e.g. Qwen3-TTS tokenizer_decoder) always return the max-profile output
        // length even though only the first `seqLen * upsampleRate` samples are valid audio.
        // Trim to the expected length so downstream consumers don't pick up stale buffer content.
        waveformLen = std::min(engineWaveformLen, expectedWaveformLen);

        outputAudio.waveform
            = std::make_shared<rt::Tensor>(rt::Tensor({1, waveformLen}, rt::DeviceType::kCPU, mWaveformDtype));
        CUDA_CHECK(cudaMemcpyAsync(outputAudio.waveform->rawPointer(), mOutputWaveform.rawPointer(),
            math::cast<size_t>(waveformLen) * rt::utils::getTypeSize(mWaveformDtype), cudaMemcpyDeviceToHost, stream));
    }
    else
    {
        // Chunked inference for sequences exceeding engine's max capacity
        LOG_INFO("Using chunked inference for long sequence (len=%ld, max=%ld)", seqLen, maxCodeLen);
        rt::Tensor finalWaveform;
        if (!runChunkedInference(codes, finalWaveform, stream))
        {
            return false;
        }

        auto const& finalShape = finalWaveform.getShape();
        waveformLen = finalShape[finalShape.getNumDims() - 1];

        outputAudio.waveform
            = std::make_shared<rt::Tensor>(rt::Tensor({1, waveformLen}, rt::DeviceType::kCPU, mWaveformDtype));
        CUDA_CHECK(cudaMemcpyAsync(outputAudio.waveform->rawPointer(), finalWaveform.rawPointer(),
            math::cast<size_t>(waveformLen) * rt::utils::getTypeSize(mWaveformDtype), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    outputAudio.sampleRate = mConfig.sampleRate;
    outputAudio.numChannels = 1;
    outputAudio.hasWaveform = true;

    return true;
}

} // namespace rt
} // namespace trt_edgellm
