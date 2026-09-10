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

#include "multimodal/qwen3_omni/cloneEncoderRunner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"

#include <algorithm>
#include <filesystem>

namespace trt_edgellm
{
namespace rt
{

namespace
{

std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(nvinfer1::IRuntime& runtime, std::string const& path)
{
    auto mmapReader = std::make_unique<file_io::MmapReader>(path);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime.deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    ELLM_CHECK(engine, "Failed to deserialize engine: " + path);
    return engine;
}

} // namespace

CloneEncoderRunner::CloneEncoderRunner(std::string const& engineDir, cudaStream_t stream)
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    std::string const speakerPath = engineDir + "/speaker_encoder.engine";
    ELLM_CHECK(std::filesystem::exists(speakerPath), "speaker_encoder.engine not found in " + engineDir);
    mSpeakerEngine = loadEngine(*mRuntime, speakerPath);
    // Both encoders run strictly serially on the same stream, so their execution contexts
    // share one device-memory block (sized for the larger engine) instead of each holding
    // its own allocation.
    mSpeakerContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mSpeakerEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mSpeakerContext, "Failed to create speaker encoder context");

    auto const spkOutDims = mSpeakerEngine->getTensorShape("speaker_embedding");
    mXvecDim = spkOutDims.d[spkOutDims.nbDims - 1];
    auto const spkMaxDims = mSpeakerEngine->getProfileShape("wav", 0, nvinfer1::OptProfileSelector::kMAX);
    mSpeakerMaxSamples = spkMaxDims.d[1];

    int64_t maxSamples = mSpeakerMaxSamples;
    int64_t contextMemBytes = static_cast<int64_t>(mSpeakerEngine->getDeviceMemorySizeV2());
    std::string const tokenizerPath = engineDir + "/speech_tokenizer_encoder.engine";
    if (std::filesystem::exists(tokenizerPath))
    {
        mTokenizerEngine = loadEngine(*mRuntime, tokenizerPath);
        mTokenizerContext = std::unique_ptr<nvinfer1::IExecutionContext>(
            mTokenizerEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
        ELLM_CHECK(mTokenizerContext, "Failed to create tokenizer encoder context");
        contextMemBytes = std::max(contextMemBytes, static_cast<int64_t>(mTokenizerEngine->getDeviceMemorySizeV2()));

        auto const wavDims = mTokenizerEngine->getTensorShape("wav");
        auto const codeDims = mTokenizerEngine->getTensorShape("codes");
        mBucketSamples = wavDims.d[1];
        mBucketFrames = codeDims.d[0];
        mNumQuantizers = static_cast<int32_t>(codeDims.d[1]);
        check::check(
            mBucketFrames > 0 && mBucketSamples % mBucketFrames == 0, "Unexpected tokenizer encoder bucket geometry");
        mDownsampleRate = mBucketSamples / mBucketFrames;
        maxSamples = std::max(maxSamples, mBucketSamples);

        mCodesBuffer = rt::Tensor(
            {mBucketFrames, mNumQuantizers}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "cloneRefCodes");
    }
    else
    {
        LOG_INFO("speech_tokenizer_encoder.engine absent; ICL cloning disabled (x-vector only)");
    }

    mSharedContextMemory
        = rt::Tensor({contextMemBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "cloneCtxMem");
    mSpeakerContext->setDeviceMemoryV2(mSharedContextMemory.rawPointer(), contextMemBytes);
    ELLM_CHECK(mSpeakerContext->setOptimizationProfileAsync(0, stream), "Failed to set speaker profile");
    if (mTokenizerContext)
    {
        mTokenizerContext->setDeviceMemoryV2(mSharedContextMemory.rawPointer(), contextMemBytes);
        ELLM_CHECK(mTokenizerContext->setOptimizationProfileAsync(0, stream), "Failed to set tokenizer profile");
    }

    mWavBuffer = rt::Tensor({1, maxSamples}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cloneWav");
    mXvecFp32 = rt::Tensor({1, mXvecDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cloneXvecF32");

    LOG_INFO(
        "CloneEncoderRunner initialized: xvecDim=%ld, bucket=%ld samples/%ld frames, %d quantizers, "
        "sharedCtxMem=%.1f MB",
        mXvecDim, mBucketSamples, mBucketFrames, mNumQuantizers, static_cast<double>(contextMemBytes) / (1 << 20));
}

bool CloneEncoderRunner::extractSpeakerEmbedding(
    std::vector<float> const& wav24k, rt::Tensor& xvecOut, cudaStream_t stream)
{
    check::check(!wav24k.empty(), "extractSpeakerEmbedding: empty waveform");
    check::check(xvecOut.getShape().volume() == mXvecDim, "x-vector output dim mismatch");
    int64_t const numSamples = std::min<int64_t>(static_cast<int64_t>(wav24k.size()), mSpeakerMaxSamples);
    if (numSamples < static_cast<int64_t>(wav24k.size()))
    {
        LOG_WARNING("Reference audio truncated to %ld samples for speaker encoder", numSamples);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mWavBuffer.rawPointer(), wav24k.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice, stream));

    ELLM_CHECK(mSpeakerContext->setInputShape("wav", nvinfer1::Dims2{1, numSamples}), "setInputShape failed");
    ELLM_CHECK(mSpeakerContext->setTensorAddress("wav", mWavBuffer.rawPointer()), "bind wav failed");
    ELLM_CHECK(mSpeakerContext->setTensorAddress("speaker_embedding", mXvecFp32.rawPointer()), "bind out failed");
    ELLM_CHECK(mSpeakerContext->enqueueV3(stream), "speaker encoder enqueue failed");

    // Engine emits FP32; the prefill consumes FP16. Cast on device.
    kernel::invokeCastFp32ToFp16(
        static_cast<float const*>(mXvecFp32.rawPointer()), static_cast<half*>(xvecOut.rawPointer()), mXvecDim, stream);
    return true;
}

bool CloneEncoderRunner::encodeReferenceCodes(std::vector<float> const& wav24k, int32_t& numFrames, cudaStream_t stream)
{
    check::check(hasTokenizerEncoder(), "speech_tokenizer_encoder.engine not loaded");
    int64_t const numSamples = std::min<int64_t>(static_cast<int64_t>(wav24k.size()), mBucketSamples);
    if (numSamples < static_cast<int64_t>(wav24k.size()))
    {
        LOG_WARNING("Reference audio truncated to %.1fs for the codec encoder bucket",
            static_cast<double>(mBucketSamples) / 24000.0);
    }
    numFrames = static_cast<int32_t>(numSamples / mDownsampleRate); // complete frames only
    check::check(numFrames > 0, "Reference audio shorter than one codec frame");

    // Zero-fill the static bucket, then copy the reference in. The encoder is causal, so the
    // zero tail cannot affect the first numFrames rows.
    CUDA_CHECK(cudaMemsetAsync(mWavBuffer.rawPointer(), 0, mBucketSamples * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mWavBuffer.rawPointer(), wav24k.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice, stream));

    ELLM_CHECK(mTokenizerContext->setTensorAddress("wav", mWavBuffer.rawPointer()), "bind wav failed");
    ELLM_CHECK(mTokenizerContext->setTensorAddress("codes", mCodesBuffer.rawPointer()), "bind codes failed");
    ELLM_CHECK(mTokenizerContext->enqueueV3(stream), "tokenizer encoder enqueue failed");
    // Codes remain on device; downstream consumers read refCodesDevice() directly.
    return true;
}

} // namespace rt
} // namespace trt_edgellm
