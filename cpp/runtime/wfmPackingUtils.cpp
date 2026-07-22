/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/wfmRuntimeUtils.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "tokenizer/tokenizer.h"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

std::pair<std::vector<int64_t>, int64_t> get3dMropeIdsTextTokens(
    int32_t numTokens, int64_t temporalOffset, bool useFloatPositions)
{
    std::vector<int64_t> ids(static_cast<std::size_t>(numTokens) * 3U);
    for (int32_t i = 0; i < numTokens; ++i)
    {
        int64_t const value = temporalOffset + i;
        for (int32_t axis = 0; axis < 3; ++axis)
        {
            ids[static_cast<std::size_t>(axis) * static_cast<std::size_t>(numTokens) + static_cast<std::size_t>(i)]
                = value;
        }
    }
    (void) useFloatPositions;
    return {ids, temporalOffset + numTokens};
}

std::pair<std::vector<int64_t>, int64_t> get3dMropeIdsVaeTokens(int32_t gridT, int32_t gridH, int32_t gridW,
    int64_t temporalOffset, bool resetSpatialIndices, float fps, float baseFps, int32_t temporalCompressionFactor,
    bool enableFpsModulation)
{
    int32_t const numTokens = gridT * gridH * gridW;
    std::vector<int64_t> ids(static_cast<std::size_t>(numTokens) * 3U);
    bool const fpsModulationEnabled = enableFpsModulation && gridT > 1 && fps > 0.F;

    int32_t tokenIdx = 0;
    for (int32_t t = 0; t < gridT; ++t)
    {
        for (int32_t h = 0; h < gridH; ++h)
        {
            for (int32_t w = 0; w < gridW; ++w)
            {
                int64_t tIndex = 0;
                if (fpsModulationEnabled)
                {
                    float const tps = fps / static_cast<float>(temporalCompressionFactor);
                    float const baseTps = baseFps / static_cast<float>(temporalCompressionFactor);
                    float const scaledT = (static_cast<float>(t) / tps) * baseTps + static_cast<float>(temporalOffset);
                    tIndex = static_cast<int64_t>(std::floor(scaledT));
                }
                else
                {
                    tIndex = static_cast<int64_t>(t) + temporalOffset;
                }

                int64_t hIndex = h;
                int64_t wIndex = w;
                if (!resetSpatialIndices)
                {
                    hIndex += temporalOffset;
                    wIndex += temporalOffset;
                }

                ids[static_cast<std::size_t>(tokenIdx)] = tIndex;
                ids[static_cast<std::size_t>(numTokens) + static_cast<std::size_t>(tokenIdx)] = hIndex;
                ids[static_cast<std::size_t>(numTokens) * 2U + static_cast<std::size_t>(tokenIdx)] = wIndex;
                ++tokenIdx;
            }
        }
    }

    int64_t maxPosition = 0;
    for (auto const value : ids)
    {
        maxPosition = std::max(maxPosition, value);
    }
    return {ids, maxPosition + 1};
}

void applyInterleavedMrope(std::array<std::vector<float>, 3>& freqs, std::array<int32_t, 3> const& mropeSection)
{
    std::vector<float> merged = freqs[0];
    for (int32_t dim = 1; dim <= 2; ++dim)
    {
        int32_t const length = mropeSection[static_cast<std::size_t>(dim)] * 3;
        for (int32_t j = dim; j < length; j += 3)
        {
            merged[static_cast<std::size_t>(j)] = freqs[static_cast<std::size_t>(dim)][static_cast<std::size_t>(j)];
        }
    }
    freqs[0] = std::move(merged);
}

void computeCosmosRotaryEmbeddings(std::vector<int64_t> const& positionIds, int32_t sequenceLength,
    CosmosEngineConfig const& config, std::vector<float>& cosOut, std::vector<float>& sinOut)
{
    int32_t const headDim = config.headDim;
    int32_t const halfDim = headDim / 2;
    cosOut.assign(static_cast<std::size_t>(sequenceLength) * static_cast<std::size_t>(headDim), 0.F);
    sinOut.assign(static_cast<std::size_t>(sequenceLength) * static_cast<std::size_t>(headDim), 0.F);

    std::vector<float> invFreq(static_cast<std::size_t>(halfDim));
    for (int32_t i = 0; i < halfDim; ++i)
    {
        invFreq[static_cast<std::size_t>(i)]
            = 1.F / std::pow(config.ropeTheta, static_cast<float>(i * 2) / static_cast<float>(headDim));
    }

    for (int32_t token = 0; token < sequenceLength; ++token)
    {
        std::array<std::vector<float>, 3> freqs;
        for (int32_t axis = 0; axis < 3; ++axis)
        {
            freqs[static_cast<std::size_t>(axis)].resize(static_cast<std::size_t>(halfDim));
            int64_t const pos
                = positionIds[static_cast<std::size_t>(axis) * static_cast<std::size_t>(sequenceLength)
                    + static_cast<std::size_t>(token)];
            float const posF = static_cast<float>(pos);
            for (int32_t j = 0; j < halfDim; ++j)
            {
                freqs[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)] = invFreq[static_cast<std::size_t>(j)] * posF;
            }
        }

        applyInterleavedMrope(freqs, config.mropeSection);
        auto const& merged = freqs[0];

        for (int32_t j = 0; j < halfDim; ++j)
        {
            float const angle = merged[static_cast<std::size_t>(j)];
            float const c = std::cos(angle);
            float const s = std::sin(angle);
            cosOut[static_cast<std::size_t>(token) * static_cast<std::size_t>(headDim) + static_cast<std::size_t>(j)]
                = c;
            cosOut[static_cast<std::size_t>(token) * static_cast<std::size_t>(headDim) + static_cast<std::size_t>(j)
                + static_cast<std::size_t>(halfDim)]
                = c;
            sinOut[static_cast<std::size_t>(token) * static_cast<std::size_t>(headDim) + static_cast<std::size_t>(j)]
                = s;
            sinOut[static_cast<std::size_t>(token) * static_cast<std::size_t>(headDim) + static_cast<std::size_t>(j)
                + static_cast<std::size_t>(halfDim)]
                = s;
        }
    }
}

void uploadHostFloatToHalfGpu(rt::Tensor& dst, std::vector<float> const& host, cudaStream_t stream)
{
    std::vector<half> converted(host.size());
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        converted[i] = __float2half(host[i]);
    }
    CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), converted.data(), converted.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));
}

std::string formatCosmosUserPrompt(std::string const& prompt, CosmosEngineConfig const& config)
{
    std::string text = prompt;
    while (!text.empty() && text.back() == '.')
    {
        text.pop_back();
    }

    if (config.numFrames == 1)
    {
        return text + ". This image is of " + std::to_string(config.height) + "x" + std::to_string(config.width)
            + " resolution.";
    }

    float const duration = static_cast<float>(config.numFrames) / config.fps;
    std::ostringstream oss;
    oss << text << ". The video is " << std::fixed << std::setprecision(1) << duration << " seconds long and is of "
        << static_cast<int>(config.fps) << " FPS."
        << ". This video is of " << config.height << "x" << config.width << " resolution.";
    return oss.str();
}

} // namespace

std::vector<int32_t> tokenizeCosmosPrompt(
    tokenizer::Tokenizer const& tokenizer, std::string const& prompt, CosmosEngineConfig const& config)
{
    std::vector<int32_t> inputIds;
    std::string const userPrompt = formatCosmosUserPrompt(prompt, config);
    if (config.useChatTemplate)
    {
        LLMGenerationRequest::Request request;
        if (config.useSystemPrompt)
        {
            Message::MessageContent systemContent;
            systemContent.type = "text";
            systemContent.content = config.systemPromptVideo;

            Message systemMessage;
            systemMessage.role = "system";
            systemMessage.contents.push_back(std::move(systemContent));
            request.messages.push_back(std::move(systemMessage));
        }

        Message::MessageContent userContent;
        userContent.type = "text";
        userContent.content = userPrompt;

        Message userMessage;
        userMessage.role = "user";
        userMessage.contents.push_back(std::move(userContent));
        request.messages.push_back(std::move(userMessage));

        LLMGenerationRequest::FormattedRequest formatted;
        if (!tokenizer.applyChatTemplate(request, formatted, true, true, false))
        {
            throw std::runtime_error("Cosmos chat template application failed.");
        }

        inputIds = tokenizer.encode(formatted.formattedCompleteRequest, false, false);
    }
    else
    {
        inputIds = tokenizer.encode(userPrompt, false, false);
    }

    if (inputIds.empty())
    {
        throw std::runtime_error("Cosmos prompt tokenization produced an empty input_ids sequence.");
    }

    inputIds.push_back(tokenizer.getEosId());

    int32_t const visionStartId = tokenizer.getTokenId(config.visionStartToken);
    if (visionStartId < 0)
    {
        throw std::runtime_error("Cosmos tokenizer is missing vision start token: " + config.visionStartToken);
    }
    inputIds.push_back(visionStartId);
    return inputIds;
}

std::vector<int64_t> buildCosmosPositionIds(
    int32_t undLen, CosmosPackedStatic const& packed, CosmosEngineConfig const& config)
{
    if (packed.visionTokenShape.size() != 3U)
    {
        throw std::runtime_error("Cosmos packing_static is missing vision_token_shapes");
    }

    auto [textIds, nextOffset] = get3dMropeIdsTextTokens(undLen, 0, config.enableFpsModulation);
    int64_t const visionTemporalOffset
        = nextOffset + static_cast<int64_t>(config.unified3dMropeTemporalModalityMargin);

    auto [visionIds, unusedNext] = get3dMropeIdsVaeTokens(packed.visionTokenShape[0], packed.visionTokenShape[1],
        packed.visionTokenShape[2], visionTemporalOffset, config.unified3dMropeResetSpatialIds, config.fps,
        config.baseFps, config.temporalCompressionFactor, config.enableFpsModulation);
    (void) unusedNext;

    int32_t const genLen = packed.visionTokenShape[0] * packed.visionTokenShape[1] * packed.visionTokenShape[2];
    int32_t const sequenceLength = undLen + genLen;
    std::vector<int64_t> positionIds(static_cast<std::size_t>(sequenceLength) * 3U);

    for (int32_t axis = 0; axis < 3; ++axis)
    {
        for (int32_t i = 0; i < undLen; ++i)
        {
            positionIds[static_cast<std::size_t>(axis) * static_cast<std::size_t>(sequenceLength)
                + static_cast<std::size_t>(i)]
                = textIds[static_cast<std::size_t>(axis) * static_cast<std::size_t>(undLen) + static_cast<std::size_t>(i)];
        }
        for (int32_t i = 0; i < genLen; ++i)
        {
            positionIds[static_cast<std::size_t>(axis) * static_cast<std::size_t>(sequenceLength)
                + static_cast<std::size_t>(undLen) + static_cast<std::size_t>(i)]
                = visionIds[static_cast<std::size_t>(axis) * static_cast<std::size_t>(genLen) + static_cast<std::size_t>(i)];
        }
    }

    return positionIds;
}

bool prepareCosmosTextPhase0(tokenizer::Tokenizer const& tokenizer, EmbeddingData const& embedding,
    CosmosEngineConfig const& config, CosmosPackedStatic const& packed, std::string const& prompt,
    CosmosTextPhase0& phase0, cudaStream_t stream)
{
    try
    {
        phase0.inputIds = tokenizeCosmosPrompt(tokenizer, prompt, config);
        phase0.undLen = static_cast<int32_t>(phase0.inputIds.size());
        phase0.genLen = packed.visionTokenShape[0] * packed.visionTokenShape[1] * packed.visionTokenShape[2];
        phase0.sequenceLength = phase0.undLen + phase0.genLen;

        if (phase0.undLen != packed.undLen)
        {
            LOG_WARNING(
                "Runtime und_len=%d differs from packing_static und_len=%d; vision indexes must be rebuilt before denoise.",
                phase0.undLen, packed.undLen);
        }

        auto const positionIds = buildCosmosPositionIds(phase0.undLen, packed, config);

        rt::Tensor hostInputIds({1, phase0.undLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        std::memcpy(hostInputIds.rawPointer(), phase0.inputIds.data(),
            static_cast<std::size_t>(phase0.undLen) * sizeof(int32_t));

        rt::Tensor gpuInputIds({1, phase0.undLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        CUDA_CHECK(cudaMemcpyAsync(gpuInputIds.rawPointer(), hostInputIds.rawPointer(),
            static_cast<std::size_t>(phase0.undLen) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        rt::Tensor embedOutput({1, phase0.undLen, config.hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        kernel::embeddingLookup(gpuInputIds, embedding.table, embedding.scalesAsOptional(), embedOutput, stream);

        if (!phase0.undSeq.reshape({phase0.undLen, config.hiddenSize}))
        {
            LOG_ERROR("Failed to reshape und_seq to [%d, %d].", phase0.undLen, config.hiddenSize);
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(phase0.undSeq.rawPointer(), embedOutput.rawPointer(),
            static_cast<std::size_t>(phase0.undLen) * static_cast<std::size_t>(config.hiddenSize) * sizeof(half),
            cudaMemcpyDeviceToDevice, stream));

        std::vector<float> cosFull;
        std::vector<float> sinFull;
        computeCosmosRotaryEmbeddings(positionIds, phase0.sequenceLength, config, cosFull, sinFull);

        if (!phase0.cosUnd.reshape({phase0.undLen, config.headDim}) || !phase0.sinUnd.reshape({phase0.undLen, config.headDim})
            || !phase0.cosGen.reshape({phase0.genLen, config.headDim})
            || !phase0.sinGen.reshape({phase0.genLen, config.headDim}))
        {
            LOG_ERROR("Failed to reshape Cosmos rotary tensors.");
            return false;
        }

        std::vector<float> cosUndHost(static_cast<std::size_t>(phase0.undLen) * static_cast<std::size_t>(config.headDim));
        std::vector<float> sinUndHost(static_cast<std::size_t>(phase0.undLen) * static_cast<std::size_t>(config.headDim));
        std::vector<float> cosGenHost(static_cast<std::size_t>(phase0.genLen) * static_cast<std::size_t>(config.headDim));
        std::vector<float> sinGenHost(static_cast<std::size_t>(phase0.genLen) * static_cast<std::size_t>(config.headDim));

        for (int32_t i = 0; i < phase0.undLen; ++i)
        {
            std::memcpy(cosUndHost.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(config.headDim),
                cosFull.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(config.headDim),
                static_cast<std::size_t>(config.headDim) * sizeof(float));
            std::memcpy(sinUndHost.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(config.headDim),
                sinFull.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(config.headDim),
                static_cast<std::size_t>(config.headDim) * sizeof(float));
        }
        for (int32_t i = 0; i < phase0.genLen; ++i)
        {
            std::size_t const src = static_cast<std::size_t>(phase0.undLen + i) * static_cast<std::size_t>(config.headDim);
            std::size_t const dst = static_cast<std::size_t>(i) * static_cast<std::size_t>(config.headDim);
            std::memcpy(cosGenHost.data() + dst, cosFull.data() + src, static_cast<std::size_t>(config.headDim) * sizeof(float));
            std::memcpy(sinGenHost.data() + dst, sinFull.data() + src, static_cast<std::size_t>(config.headDim) * sizeof(float));
        }

        uploadHostFloatToHalfGpu(phase0.cosUnd, cosUndHost, stream);
        uploadHostFloatToHalfGpu(phase0.sinUnd, sinUndHost, stream);
        uploadHostFloatToHalfGpu(phase0.cosGen, cosGenHost, stream);
        uploadHostFloatToHalfGpu(phase0.sinGen, sinGenHost, stream);

        LOG_INFO("Cosmos Phase 0 ready: und_len=%d, gen_len=%d, seq_len=%d, und_seq=[%d,%d], rotary head_dim=%d",
            phase0.undLen, phase0.genLen, phase0.sequenceLength, phase0.undLen, config.hiddenSize, config.headDim);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("prepareCosmosTextPhase0 failed: %s", e.what());
        return false;
    }
}

bool seedNoisyVisionLatents(rt::Tensor const& cleanLatents, rt::Tensor& noisyLatents,
    std::vector<int32_t> const& noisyFrameIndexes, int32_t seed, float noiseScale, cudaStream_t stream)
{
    if (cleanLatents.getDeviceType() != rt::DeviceType::kGPU)
    {
        LOG_ERROR("seedNoisyVisionLatents: clean latents must be on GPU.");
        return false;
    }
    if (noisyLatents.getDeviceType() != rt::DeviceType::kGPU)
    {
        LOG_ERROR("seedNoisyVisionLatents: noisy latents must be on GPU.");
        return false;
    }

    auto const shape = cleanLatents.getShape();
    if (shape.getNumDims() != 5)
    {
        LOG_ERROR("seedNoisyVisionLatents: expected rank-5 latents [B,C,T,H,W], got %d dims.",
            shape.getNumDims());
        return false;
    }
    if (noisyLatents.getShape() != shape)
    {
        if (!noisyLatents.reshape(shape))
        {
            LOG_ERROR("seedNoisyVisionLatents: failed to reshape noisy latents to %s.",
                shape.formatString().c_str());
            return false;
        }
    }
    if (cleanLatents.getDataType() != noisyLatents.getDataType())
    {
        LOG_ERROR("seedNoisyVisionLatents: clean/noisy latent dtypes must match.");
        return false;
    }

    nvinfer1::DataType const dtype = cleanLatents.getDataType();
    if (dtype != nvinfer1::DataType::kHALF && dtype != nvinfer1::DataType::kFLOAT)
    {
        LOG_ERROR("seedNoisyVisionLatents: unsupported latent dtype.");
        return false;
    }

    int64_t const batch = shape[0];
    int64_t const channels = shape[1];
    int64_t const latentT = shape[2];
    int64_t const height = shape[3];
    int64_t const width = shape[4];
    int64_t const framePlaneElems = height * width;
    int64_t const temporalStride = framePlaneElems;
    std::size_t const elemSize = (dtype == nvinfer1::DataType::kHALF) ? sizeof(half) : sizeof(float);
    std::size_t const tensorBytes = static_cast<std::size_t>(shape.volume()) * elemSize;

    CUDA_CHECK(cudaMemcpyAsync(noisyLatents.rawPointer(), cleanLatents.rawPointer(), tensorBytes,
        cudaMemcpyDeviceToDevice, stream));

    std::vector<float> noiseHost(static_cast<std::size_t>(shape.volume()));
    std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<float> dist(0.F, 1.F);
    for (float& value : noiseHost)
    {
        value = dist(generator) * noiseScale;
    }

    std::vector<std::uint8_t> planeBytes(static_cast<std::size_t>(framePlaneElems) * elemSize);
    for (int32_t frameIdx : noisyFrameIndexes)
    {
        if (frameIdx < 0 || frameIdx >= latentT)
        {
            LOG_ERROR("seedNoisyVisionLatents: invalid noisy frame index %d (latent T=%ld).", frameIdx, latentT);
            return false;
        }

        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                int64_t const linearOffset
                    = ((b * channels + c) * latentT + frameIdx) * temporalStride;
                for (int64_t elem = 0; elem < framePlaneElems; ++elem)
                {
                    float const value = noiseHost[static_cast<std::size_t>(linearOffset + elem)];
                    if (dtype == nvinfer1::DataType::kHALF)
                    {
                        reinterpret_cast<half*>(planeBytes.data())[static_cast<std::size_t>(elem)] = __float2half(value);
                    }
                    else
                    {
                        reinterpret_cast<float*>(planeBytes.data())[static_cast<std::size_t>(elem)] = value;
                    }
                }

                std::size_t const dstOffset = static_cast<std::size_t>(linearOffset) * elemSize;
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(noisyLatents.rawPointer()) + dstOffset,
                    planeBytes.data(), planeBytes.size(), cudaMemcpyHostToDevice, stream));
            }
        }
    }

    LOG_INFO("seedNoisyVisionLatents: cloned clean latents and noised %zu frame(s) with seed=%d.",
        noisyFrameIndexes.size(), seed);
    return true;
}

bool seedNoisySoundLatents(rt::Tensor const& cleanLatents, rt::Tensor& noisyLatents,
    std::vector<int32_t> const& noisySlotIndexes, int32_t seed, float noiseScale, cudaStream_t stream)
{
    if (cleanLatents.getDeviceType() != rt::DeviceType::kGPU)
    {
        LOG_ERROR("seedNoisySoundLatents: clean latents must be on GPU.");
        return false;
    }
    if (noisyLatents.getDeviceType() != rt::DeviceType::kGPU)
    {
        LOG_ERROR("seedNoisySoundLatents: noisy latents must be on GPU.");
        return false;
    }

    auto const shape = cleanLatents.getShape();
    if (shape.getNumDims() != 3)
    {
        LOG_ERROR("seedNoisySoundLatents: expected rank-3 latents [B,C,T], got %d dims.", shape.getNumDims());
        return false;
    }
    if (noisyLatents.getShape() != shape)
    {
        if (!noisyLatents.reshape(shape))
        {
            LOG_ERROR("seedNoisySoundLatents: failed to reshape noisy latents to %s.",
                shape.formatString().c_str());
            return false;
        }
    }
    if (cleanLatents.getDataType() != noisyLatents.getDataType())
    {
        LOG_ERROR("seedNoisySoundLatents: clean/noisy latent dtypes must match.");
        return false;
    }

    nvinfer1::DataType const dtype = cleanLatents.getDataType();
    if (dtype != nvinfer1::DataType::kHALF && dtype != nvinfer1::DataType::kFLOAT)
    {
        LOG_ERROR("seedNoisySoundLatents: unsupported latent dtype.");
        return false;
    }

    int64_t const batch = shape[0];
    int64_t const channels = shape[1];
    int64_t const latentT = shape[2];
    std::size_t const elemSize = (dtype == nvinfer1::DataType::kHALF) ? sizeof(half) : sizeof(float);
    std::size_t const tensorBytes = static_cast<std::size_t>(shape.volume()) * elemSize;

    CUDA_CHECK(cudaMemcpyAsync(noisyLatents.rawPointer(), cleanLatents.rawPointer(), tensorBytes,
        cudaMemcpyDeviceToDevice, stream));

    std::vector<int32_t> slots = noisySlotIndexes;
    if (slots.empty())
    {
        slots.reserve(static_cast<std::size_t>(latentT));
        for (int64_t t = 0; t < latentT; ++t)
        {
            slots.push_back(static_cast<int32_t>(t));
        }
    }

    std::vector<float> noiseHost(static_cast<std::size_t>(shape.volume()));
    std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<float> dist(0.F, 1.F);
    for (float& value : noiseHost)
    {
        value = dist(generator) * noiseScale;
    }

    std::vector<std::uint8_t> slotBytes(static_cast<std::size_t>(channels) * elemSize);
    for (int32_t slotIdx : slots)
    {
        if (slotIdx < 0 || slotIdx >= latentT)
        {
            LOG_ERROR("seedNoisySoundLatents: invalid noisy slot index %d (latent T=%ld).", slotIdx, latentT);
            return false;
        }

        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                int64_t const linearOffset = (b * channels + c) * latentT + slotIdx;
                float const value = noiseHost[static_cast<std::size_t>(linearOffset)];
                if (dtype == nvinfer1::DataType::kHALF)
                {
                    reinterpret_cast<half*>(slotBytes.data())[static_cast<std::size_t>(c)] = __float2half(value);
                }
                else
                {
                    reinterpret_cast<float*>(slotBytes.data())[static_cast<std::size_t>(c)] = value;
                }
            }

            for (int64_t c = 0; c < channels; ++c)
            {
                std::size_t const dstOffset
                    = static_cast<std::size_t>((b * channels + c) * latentT + slotIdx) * elemSize;
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(noisyLatents.rawPointer()) + dstOffset,
                    slotBytes.data() + static_cast<std::size_t>(c) * elemSize, elemSize, cudaMemcpyHostToDevice,
                    stream));
            }
        }
    }

    LOG_INFO("seedNoisySoundLatents: cloned clean latents and noised %zu slot(s) with seed=%d.", slots.size(), seed);
    return true;
}

} // namespace rt
} // namespace trt_edgellm
