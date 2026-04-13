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

/*
 * AlpamayoExpertRunner: runs TRT Diffusion Expert inline, KV cache stays on GPU.
 */

#include "alpamayoExpertRunner.h"

#include "kernels/alpamayoExpertKernels/alpamayoExpertKernels.h"

#include <cmath>
#include <fstream>

namespace trt_edgellm
{
namespace rt
{

AlpamayoExpertRunner::AlpamayoExpertRunner(
    std::string const& enginePath, AlpamayoExpertConfig const& config, cudaStream_t stream)
    : mConfig(config)
{
    loadEngine(enginePath, stream);

    auto maxShape = mEngine->getProfileShape("kv_cache", 0, nvinfer1::OptProfileSelector::kMAX);
    int32_t maxKVSeqLen = maxShape.d[3];

    allocateBuffers(maxKVSeqLen, stream);

    curandCreateGenerator(&mRNG, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(mRNG, mConfig.seed);
    curandSetStream(mRNG, stream);

    LOG_INFO("AlpamayoExpertRunner: %d candidates, %d steps, maxKVSeq=%d, engine=%s", mConfig.numCandidates,
        mConfig.numDiffusionSteps, maxKVSeqLen, enginePath.c_str());
}

AlpamayoExpertRunner::~AlpamayoExpertRunner() noexcept
{
    if (mRNG)
    {
        curandDestroyGenerator(mRNG);
    }
}

void AlpamayoExpertRunner::loadEngine(std::string const& path, cudaStream_t stream)
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        throw std::runtime_error("Cannot open Expert engine: " + path);
    }
    size_t const fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(fileSize);
    file.read(buf.data(), fileSize);

    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(buf.data(), fileSize));
    if (!mEngine)
    {
        throw std::runtime_error("Failed to deserialize Expert engine");
    }

    int64_t const execBytes = mEngine->getDeviceMemorySizeV2();
    mExecContextMemory
        = rt::Tensor({execBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "ExpertRunner::execMem");

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    mContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(), execBytes);

    LOG_INFO("Expert engine loaded: exec ctx %zu MB", execBytes / (1024 * 1024));
}

void AlpamayoExpertRunner::allocateBuffers(int32_t maxKVSeqLen, cudaStream_t stream)
{
    int32_t const B = mConfig.numCandidates;
    int32_t const T = mConfig.numDiffusionTokens;
    int32_t const A = mConfig.actionDim;
    int64_t const L2 = mConfig.numLayers * 2;
    int32_t const H = mConfig.numKVHeads;
    int32_t const D = mConfig.headDim;

    mKVCacheReshaped = rt::Tensor({L2, B, H, static_cast<int64_t>(maxKVSeqLen), D}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "ExpertRunner::kv");
    mNoisyAction = rt::Tensor({B, T, A}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ExpertRunner::x");
    mTimestep = rt::Tensor({B, 1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ExpertRunner::t");
    mPositionIds = rt::Tensor({3, static_cast<int64_t>(B), static_cast<int64_t>(T)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "ExpertRunner::posIds");
    mAttentionMask = rt::Tensor({B, 1, T, static_cast<int64_t>(maxKVSeqLen + T)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "ExpertRunner::mask");
    mPredVelocity = rt::Tensor({B, T, A}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ExpertRunner::v");
}

int32_t AlpamayoExpertRunner::findTrajFutureStart(
    std::vector<int32_t> const& outputIds, int32_t prefillSeqLen) const noexcept
{
    for (size_t i = 0; i < outputIds.size(); ++i)
    {
        if (outputIds[i] == mConfig.trajFutureStartTokenId)
        {
            return prefillSeqLen + static_cast<int32_t>(i) + 1;
        }
    }
    return -1;
}

void AlpamayoExpertRunner::runDiffusion(
    rt::Tensor const& vlmKVCache, int32_t kvSeqLen, int32_t ropeDelta, rt::Tensor& outputActions, cudaStream_t stream)
{
    int32_t const B = mConfig.numCandidates;
    int32_t const T = mConfig.numDiffusionTokens;
    int32_t const A = mConfig.actionDim;
    int32_t const L = mConfig.numLayers;
    int32_t const H = mConfig.numKVHeads;
    int32_t const D = mConfig.headDim;
    int32_t const L2 = L * 2;
    int32_t const nSteps = mConfig.numDiffusionSteps;
    int32_t const actionElements = B * T * A;
    int32_t const totalMaskLen = kvSeqLen + T;

    // 1. KV cache reshape + repeat: (L,2,H,S,D) fp16 → (L*2,B,H,S,D) fp32
    check::check(mKVCacheReshaped.reshape({L2, B, H, static_cast<int64_t>(kvSeqLen), D}), "KV reshape failed");
    kernel::kvCacheReshapeRepeat(mKVCacheReshaped.dataPointer<float>(),
        reinterpret_cast<half const*>(vlmKVCache.rawPointer()), L, H, kvSeqLen, D, B, stream);

    // 2. Position IDs: basePos = kvSeqLen + ropeDelta
    int64_t basePos = static_cast<int64_t>(kvSeqLen) + ropeDelta;
    kernel::buildPositionIds(mPositionIds.dataPointer<int64_t>(), B, T, basePos, stream);

    // 3. Attention mask: all zeros (non-causal expert)
    check::check(mAttentionMask.reshape({B, 1, T, static_cast<int64_t>(totalMaskLen)}), "Mask reshape failed");
    CUDA_CHECK(cudaMemsetAsync(
        mAttentionMask.rawPointer(), 0, static_cast<size_t>(B) * T * totalMaskLen * sizeof(float), stream));

    // 4. Initial noise x₀ ~ N(0, I)
    check::check(actionElements % 2 == 0, "curandGenerateNormal requires even element count");
    curandGenerateNormal(mRNG, mNoisyAction.dataPointer<float>(), actionElements, 0.0f, 1.0f);

    // 5. Set TRT input shapes
    nvinfer1::Dims dims;

    dims.nbDims = 3;
    dims.d[0] = B;
    dims.d[1] = T;
    dims.d[2] = A;
    mContext->setInputShape("noisy_action", dims);

    dims.d[0] = B;
    dims.d[1] = 1;
    dims.d[2] = 1;
    mContext->setInputShape("timestep", dims);

    dims.d[0] = 3;
    dims.d[1] = B;
    dims.d[2] = T;
    mContext->setInputShape("position_ids", dims);

    dims.nbDims = 4;
    dims.d[0] = B;
    dims.d[1] = 1;
    dims.d[2] = T;
    dims.d[3] = totalMaskLen;
    mContext->setInputShape("attention_mask", dims);

    dims.nbDims = 5;
    dims.d[0] = L2;
    dims.d[1] = B;
    dims.d[2] = H;
    dims.d[3] = kvSeqLen;
    dims.d[4] = D;
    mContext->setInputShape("kv_cache", dims);

    mContext->setTensorAddress("noisy_action", mNoisyAction.rawPointer());
    mContext->setTensorAddress("timestep", mTimestep.rawPointer());
    mContext->setTensorAddress("position_ids", mPositionIds.rawPointer());
    mContext->setTensorAddress("attention_mask", mAttentionMask.rawPointer());
    mContext->setTensorAddress("kv_cache", mKVCacheReshaped.rawPointer());
    mContext->setTensorAddress("pred_velocity", mPredVelocity.rawPointer());

    // 6. Flow matching: Euler integration (no CUDA Graph — kvSeqLen varies between requests)
    float const dt = 1.0f / static_cast<float>(nSteps);

    for (int32_t step = 0; step < nSteps; ++step)
    {
        float const tVal = static_cast<float>(step) * dt;
        kernel::fillTimestep(mTimestep.dataPointer<float>(), B, tVal, stream);

        auto ok = mContext->enqueueV3(stream);
        check::check(ok, "Expert enqueueV3 failed at diffusion step");

        kernel::eulerUpdate(
            mNoisyAction.dataPointer<float>(), mPredVelocity.dataPointer<float>(), dt, actionElements, stream);
    }

    // 7. Copy result to output
    CUDA_CHECK(cudaMemcpyAsync(outputActions.rawPointer(), mNoisyAction.rawPointer(), actionElements * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    LOG_INFO("Expert diffusion done: %d steps, %d candidates, kvSeq=%d", nSteps, B, kvSeqLen);
}

} // namespace rt
} // namespace trt_edgellm
