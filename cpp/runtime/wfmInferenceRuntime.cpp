/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/wfmInferenceRuntime.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <filesystem>
#include <optional>

namespace trt_edgellm
{
namespace rt
{
namespace
{

bool denoiseComponentsPresent(std::filesystem::path const& root)
{
    return std::filesystem::exists(root / "embed" / "config.json")
        && std::filesystem::exists(root / "mot_backbone" / "config.json")
        && std::filesystem::exists(root / "denoise_head" / "config.json");
}

bool componentPresent(std::filesystem::path const& root, char const* name)
{
    return std::filesystem::exists(root / name / "config.json");
}

std::optional<rt::Coords> soundLatentShapeFromPacked(
    CosmosEngineConfig const& config, CosmosPackedStatic const& packed)
{
    if (packed.soundTokenShape.empty())
    {
        return std::nullopt;
    }

    if (packed.soundTokenShape.size() == 1)
    {
        return rt::Coords({1, config.soundDim, packed.soundTokenShape[0]});
    }
    if (packed.soundTokenShape.size() == 2)
    {
        return rt::Coords({1, packed.soundTokenShape[0], packed.soundTokenShape[1]});
    }

    return std::nullopt;
}

} // namespace

WFMInferenceRuntime::WFMInferenceRuntime(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
{
    std::filesystem::path const root(engineDir);
    mConfig = loadCosmosEngineConfig(root / "config.json");
    mPackedStatic = loadCosmosPackedStatic(root / "packing_static.json");

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    std::filesystem::path tokenizerDir = root / "tokenizer";
    if (!std::filesystem::exists(tokenizerDir / "tokenizer.json"))
    {
        tokenizerDir = root;
    }
    if (!mTokenizer->loadFromHF(tokenizerDir))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to load tokenizer from " + tokenizerDir.string());
    }

    mEmbedding = loadEmbeddingTable(root / "embedding.safetensors", stream);

    int32_t const maxUndLen = mPackedStatic.sequenceLength;
    int32_t const maxGenLen = mPackedStatic.visionTokenShape[0] * mPackedStatic.visionTokenShape[1]
        * mPackedStatic.visionTokenShape[2];
    mTextPhase0.undSeq = rt::Tensor({maxUndLen, mConfig.hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "WFMInferenceRuntime::undSeq");
    mTextPhase0.cosUnd = rt::Tensor({maxUndLen, mConfig.headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "WFMInferenceRuntime::cosUnd");
    mTextPhase0.sinUnd = rt::Tensor({maxUndLen, mConfig.headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "WFMInferenceRuntime::sinUnd");
    mTextPhase0.cosGen = rt::Tensor({maxGenLen, mConfig.headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "WFMInferenceRuntime::cosGen");
    mTextPhase0.sinGen = rt::Tensor({maxGenLen, mConfig.headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "WFMInferenceRuntime::sinGen");

    std::string const visualEncodeDir = (root / "visual_encode").string();
    std::string const visualDecodeDir = (root / "visual_decode").string();
    mVaeEncodeRunner = std::make_unique<VaeEncodeRunner>(visualEncodeDir, stream);
    mVaeDecodeRunner = std::make_unique<VaeDecodeRunner>(visualDecodeDir, stream);

    if (componentPresent(root, "audio_encode"))
    {
        mAudioEncodeRunner = std::make_unique<AudioEncodeRunner>((root / "audio_encode").string(), stream);
    }
    else if (mConfig.enableSound)
    {
        LOG_WARNING("WFMInferenceRuntime: enable_sound=true but audio_encode/ is missing.");
    }

    if (componentPresent(root, "audio_decode"))
    {
        mAudioDecodeRunner = std::make_unique<AudioDecodeRunner>((root / "audio_decode").string(), stream);
    }
    else if (mConfig.enableSound)
    {
        LOG_WARNING("WFMInferenceRuntime: enable_sound=true but audio_decode/ is missing.");
    }

    if (denoiseComponentsPresent(root))
    {
        mDenoiseRunner = std::make_unique<CosmosDenoiseRunner>(root, mConfig, stream);
    }
    else
    {
        LOG_WARNING(
            "WFMInferenceRuntime: denoise engines incomplete under %s; denoise path will be unavailable.",
            engineDir.c_str());
    }

    int64_t sharedContextMemorySize = std::max(mVaeEncodeRunner->getRequiredContextMemorySize(),
        mVaeDecodeRunner->getRequiredContextMemorySize());
    if (mAudioEncodeRunner)
    {
        sharedContextMemorySize
            = std::max(sharedContextMemorySize, mAudioEncodeRunner->getRequiredContextMemorySize());
    }
    if (mAudioDecodeRunner)
    {
        sharedContextMemorySize
            = std::max(sharedContextMemorySize, mAudioDecodeRunner->getRequiredContextMemorySize());
    }
    if (mDenoiseRunner)
    {
        sharedContextMemorySize
            = std::max(sharedContextMemorySize, mDenoiseRunner->getRequiredContextMemorySize());
    }
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "WFMInferenceRuntime::mSharedExecContextMemory");
    if (!mVaeEncodeRunner->setContextMemory(mSharedExecContextMemory))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to set shared context memory for visual_encode");
    }
    if (!mVaeDecodeRunner->setContextMemory(mSharedExecContextMemory))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to set shared context memory for visual_decode");
    }
    if (mAudioEncodeRunner && !mAudioEncodeRunner->setContextMemory(mSharedExecContextMemory))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to set shared context memory for audio_encode");
    }
    if (mAudioDecodeRunner && !mAudioDecodeRunner->setContextMemory(mSharedExecContextMemory))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to set shared context memory for audio_decode");
    }
    if (mDenoiseRunner && !mDenoiseRunner->setContextMemory(mSharedExecContextMemory))
    {
        throw std::runtime_error("WFMInferenceRuntime: failed to set shared context memory for denoise");
    }

    mOutputVideo = std::make_shared<rt::Tensor>(mVaeDecodeRunner->getPixelShape(), rt::DeviceType::kGPU,
        mVaeDecodeRunner->getPixels().getDataType(), "WFMInferenceRuntime::mOutputVideo");

    mNoisyLatents = rt::Tensor(mVaeEncodeRunner->getLatentShape(), rt::DeviceType::kGPU,
        mVaeEncodeRunner->getLatents().getDataType(), "WFMInferenceRuntime::mNoisyLatents");

    if (mAudioDecodeRunner)
    {
        mOutputWaveform = std::make_shared<rt::Tensor>(mAudioDecodeRunner->getWaveformShape(), rt::DeviceType::kGPU,
            mAudioDecodeRunner->getWaveform().getDataType(), "WFMInferenceRuntime::mOutputWaveform");
    }

    rt::Coords soundLatentShape{};
    nvinfer1::DataType soundLatentDtype{nvinfer1::DataType::kHALF};
    if (mAudioEncodeRunner)
    {
        soundLatentShape = mAudioEncodeRunner->getSoundLatentShape();
        soundLatentDtype = mAudioEncodeRunner->getSoundLatents().getDataType();
    }
    else if (auto packedShape = soundLatentShapeFromPacked(mConfig, mPackedStatic))
    {
        soundLatentShape = *packedShape;
    }
    else if (mAudioDecodeRunner)
    {
        soundLatentShape = mAudioDecodeRunner->getSoundLatentShape();
        soundLatentDtype = mAudioDecodeRunner->getSoundLatents().getDataType();
    }

    if (soundLatentShape.getNumDims() == 3)
    {
        mNoisySoundLatents = rt::Tensor(soundLatentShape, rt::DeviceType::kGPU, soundLatentDtype,
            "WFMInferenceRuntime::mNoisySoundLatents");
        mZeroSoundLatents = rt::Tensor(soundLatentShape, rt::DeviceType::kGPU, soundLatentDtype,
            "WFMInferenceRuntime::mZeroSoundLatents");
        CUDA_CHECK(cudaMemsetAsync(mZeroSoundLatents.rawPointer(), 0,
            static_cast<size_t>(mZeroSoundLatents.getShape().volume())
                * rt::utils::getTypeSize(mZeroSoundLatents.getDataType()),
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    LOG_INFO(
        "WFMInferenceRuntime loaded from %s (frames=%d, %dx%d, hidden=%d, seq_len=%d, denoise=%s, sound=%s, shared_ctx=%zu bytes)",
        engineDir.c_str(), mConfig.numFrames, mConfig.height, mConfig.width, mConfig.hiddenSize,
        mPackedStatic.sequenceLength, mDenoiseRunner ? "yes" : "no",
        mConfig.enableSound ? "enabled" : "disabled", static_cast<size_t>(sharedContextMemorySize));
}

bool WFMInferenceRuntime::examineRequest(WFMGenerationRequest const& request) const noexcept
{
    return examineWFMRequest(request, mConfig, mPackedStatic);
}

bool WFMInferenceRuntime::prepareTextPhase0(std::string const& prompt, cudaStream_t stream)
{
    return prepareCosmosTextPhase0(*mTokenizer, mEmbedding, mConfig, mPackedStatic, prompt, mTextPhase0, stream);
}

bool WFMInferenceRuntime::allocateZeroSoundLatents(
    rt::Coords const& shape, nvinfer1::DataType dtype, cudaStream_t stream)
{
    mZeroSoundLatents = rt::Tensor(shape, rt::DeviceType::kGPU, dtype, "WFMInferenceRuntime::mZeroSoundLatents");
    mNoisySoundLatents = rt::Tensor(shape, rt::DeviceType::kGPU, dtype, "WFMInferenceRuntime::mNoisySoundLatents");
    CUDA_CHECK(cudaMemsetAsync(mZeroSoundLatents.rawPointer(), 0,
        static_cast<size_t>(shape.volume()) * rt::utils::getTypeSize(dtype), stream));
    return true;
}

bool WFMInferenceRuntime::handleRequest(
    WFMGenerationRequest const& request, WFMGenerationResponse& response, cudaStream_t stream)
{
    response = WFMGenerationResponse{};

    if (!examineRequest(request))
    {
        LOG_ERROR("WFMInferenceRuntime: request validation failed.");
        return false;
    }

    if (!prepareTextPhase0(request.prompt, stream))
    {
        LOG_ERROR("WFMInferenceRuntime: Phase 0 text preparation failed.");
        return false;
    }

    if (!mVaeEncodeRunner->copyPixelsFrom(request.pixels, stream))
    {
        LOG_ERROR("WFMInferenceRuntime: failed to copy pixels into visual_encode runner.");
        return false;
    }

    if (!mVaeEncodeRunner->encode(stream))
    {
        LOG_ERROR("WFMInferenceRuntime: visual_encode TRT inference failed.");
        return false;
    }

    if (!seedNoisyVisionLatents(mVaeEncodeRunner->getLatents(), mNoisyLatents,
            mPackedStatic.visionNoisyFrameIndexes, request.seed, 1.F, stream))
    {
        LOG_ERROR("WFMInferenceRuntime: failed to seed noisy vision latents.");
        return false;
    }

    rt::Tensor const* cleanSoundLatents = nullptr;
    if (request.inputWaveform.buffer)
    {
        if (!mAudioEncodeRunner)
        {
            LOG_ERROR("WFMInferenceRuntime: input waveform provided but audio_encode engine is not loaded.");
            return false;
        }
        if (!mAudioEncodeRunner->copyWaveformFrom(request.inputWaveform, stream))
        {
            LOG_ERROR("WFMInferenceRuntime: failed to copy waveform into audio_encode runner.");
            return false;
        }
        if (!mAudioEncodeRunner->encode(stream))
        {
            LOG_ERROR("WFMInferenceRuntime: audio_encode TRT inference failed.");
            return false;
        }
        cleanSoundLatents = &mAudioEncodeRunner->getSoundLatents();
        if (mNoisySoundLatents.getShape().volume() == 0)
        {
            if (!allocateZeroSoundLatents(cleanSoundLatents->getShape(), cleanSoundLatents->getDataType(), stream))
            {
                LOG_ERROR("WFMInferenceRuntime: failed to allocate sound latent buffers.");
                return false;
            }
        }
    }

    bool const runSoundDenoise = request.generateSound && mConfig.enableSound;
    if (runSoundDenoise)
    {
        if (!mDenoiseRunner || !mDenoiseRunner->hasSoundPath())
        {
            LOG_ERROR("WFMInferenceRuntime: generateSound requested but denoise_head_sound is unavailable.");
            return false;
        }
        if (mNoisySoundLatents.getShape().volume() == 0)
        {
            LOG_ERROR("WFMInferenceRuntime: sound latent buffers are not allocated.");
            return false;
        }

        rt::Tensor const& soundSeedSource = cleanSoundLatents != nullptr ? *cleanSoundLatents : mZeroSoundLatents;
        if (!seedNoisySoundLatents(soundSeedSource, mNoisySoundLatents, mPackedStatic.soundNoisySlotIndexes,
                request.seed, 1.F, stream))
        {
            LOG_ERROR("WFMInferenceRuntime: failed to seed noisy sound latents.");
            return false;
        }
    }

    if (mDenoiseRunner)
    {
        int32_t const numSteps
            = request.numInferenceSteps > 0 ? request.numInferenceSteps : mConfig.numInferenceSteps;

        CosmosDenoiseLatents denoiseLatents{};
        denoiseLatents.vision = &mNoisyLatents;
        if (runSoundDenoise)
        {
            denoiseLatents.sound = &mNoisySoundLatents;
        }

        if (!mDenoiseRunner->sampleLatents(mTextPhase0, denoiseLatents, numSteps, stream))
        {
            LOG_ERROR("WFMInferenceRuntime: denoise failed.");
            return false;
        }
    }
    else
    {
        LOG_WARNING(
            "WFMInferenceRuntime: CosmosDenoiseRunner unavailable; skipping denoise (encode->seed->decode smoke path).");
    }

    if (!mVaeDecodeRunner->copyLatentsFrom(mNoisyLatents, stream))
    {
        LOG_ERROR("WFMInferenceRuntime: failed to copy latents into visual_decode runner.");
        return false;
    }

    if (!mVaeDecodeRunner->decode(stream))
    {
        LOG_ERROR("WFMInferenceRuntime: visual_decode TRT inference failed.");
        return false;
    }

    if (!mVaeDecodeRunner->copyPixelsTo(*mOutputVideo, stream))
    {
        LOG_ERROR("WFMInferenceRuntime: failed to copy decoded pixels into output buffer.");
        return false;
    }

    response.outputVideo.buffer = mOutputVideo;
    response.outputVideo.batch = 1;
    response.outputVideo.channels = 3;
    response.outputVideo.numFrames = mConfig.numFrames;
    response.outputVideo.height = mConfig.height;
    response.outputVideo.width = mConfig.width;

    if (runSoundDenoise && mAudioDecodeRunner && mOutputWaveform)
    {
        if (!mAudioDecodeRunner->copySoundLatentsFrom(mNoisySoundLatents, stream))
        {
            LOG_ERROR("WFMInferenceRuntime: failed to copy sound latents into audio_decode runner.");
            return false;
        }
        if (!mAudioDecodeRunner->decode(stream))
        {
            LOG_ERROR("WFMInferenceRuntime: audio_decode TRT inference failed.");
            return false;
        }
        if (!mAudioDecodeRunner->copyWaveformTo(*mOutputWaveform, stream))
        {
            LOG_ERROR("WFMInferenceRuntime: failed to copy decoded waveform into output buffer.");
            return false;
        }

        response.outputWaveform.buffer = mOutputWaveform;
        response.outputWaveform.batch = 1;
        response.outputWaveform.sampleRate = mConfig.sampleRate;
        auto const& waveformShape = mOutputWaveform->getShape();
        if (waveformShape.getNumDims() == 3)
        {
            response.outputWaveform.numSamples = waveformShape[2];
        }
        else if (waveformShape.getNumDims() == 2)
        {
            response.outputWaveform.numSamples = waveformShape[1];
        }
    }

    LOG_INFO("WFMInferenceRuntime: inference complete (video=%s, sound=%s, denoise=%s).",
        mOutputVideo->getShape().formatString().c_str(),
        response.outputWaveform.buffer ? mOutputWaveform->getShape().formatString().c_str() : "n/a",
        mDenoiseRunner ? "yes" : "no");
    return true;
}

} // namespace rt
} // namespace trt_edgellm
