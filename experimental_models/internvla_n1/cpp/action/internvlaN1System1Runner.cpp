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

#include "action/internvlaN1System1Runner.h"
#include "action/internvlaN1Kernels.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace internvla_n1
{

namespace
{
constexpr char const* kMemoryEngine = "memory_bf16.engine";
constexpr char const* kDitEngine = "traj_dit_bf16.engine";
constexpr char const* kImages = "images";
constexpr char const* kMemoryTokens = "memory_tokens";
constexpr char const* kLatents = "latents";
constexpr char const* kTimestep = "timestep";
constexpr char const* kZLatents = "z_latents";
constexpr char const* kOutput = "output";

std::vector<int64_t> dimsToVector(Dims const& dims)
{
    return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
}

std::vector<int64_t> shapeOf(rt::Tensor const& tensor)
{
    return dimsToVector(tensor.getShape().getTRTDims());
}
} // namespace

cudaStream_t InternVLAN1System1Runner::makeControlStream()
{
    int least = 0;
    int greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
    return stream;
}

InternVLAN1System1Runner::InternVLAN1System1Runner(
    std::string const& engineDir, Config const& config, cudaStream_t stream)
    : mConfig(config)
    , mScheduler(config.numInferenceSteps)
{
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "InternVLAN1System1Runner: failed to create TensorRT runtime");

    loadEngine(engineDir + "/" + kMemoryEngine, mMemoryEngine, mMemoryContext, stream);
    loadEngine(engineDir + "/" + kDitEngine, mDitEngine, mDitContext, stream);

    // Both contexts are USER_MANAGED, so the scratch has to be supplied before the first
    // enqueue. One pool serves both: within a plan they run one after the other.
    int64_t const contextBytes = getRequiredContextMemorySize();
    mContextMemory = rt::Tensor(rt::Coords(std::vector<int64_t>{contextBytes}), rt::DeviceType::kGPU, DataType::kUINT8,
        "internvla_n1::contextMemory");
    mMemoryContext->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);
    mDitContext->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);

    int64_t const batch = mConfig.numSampleTrajs;
    int64_t const doubled = 2 * batch;
    int64_t const wp = mConfig.predictStepNums;
    int64_t const dim = mConfig.actionDim;

    mLatents = rt::Tensor(rt::Coords(std::vector<int64_t>{batch, wp, dim}), rt::DeviceType::kGPU, DataType::kFLOAT,
        "internvla_n1::latents");
    mDoubledLatents = rt::Tensor(rt::Coords(std::vector<int64_t>{doubled, wp, dim}), rt::DeviceType::kGPU,
        DataType::kFLOAT, "internvla_n1::doubledLatents");
    mModelOutput = rt::Tensor(rt::Coords(std::vector<int64_t>{doubled, wp, dim}), rt::DeviceType::kGPU,
        DataType::kFLOAT, "internvla_n1::modelOutput");
    mTimesteps = rt::Tensor(
        rt::Coords(std::vector<int64_t>{doubled}), rt::DeviceType::kGPU, DataType::kINT64, "internvla_n1::timestep");
}

void InternVLAN1System1Runner::loadEngine(std::string const& path, std::unique_ptr<ICudaEngine>& engine,
    std::unique_ptr<IExecutionContext>& context, cudaStream_t stream)
{
    engine = deserializeCudaEngineFromFile(*mRuntime, path);
    ELLM_CHECK(engine, "InternVLAN1System1Runner: failed to load " + path);
    // kUSER_MANAGED so the caller decides where the scratch lives. System 1 keeps its own pool
    // rather than joining the LLM's shared one -- see the class note.
    context = std::unique_ptr<IExecutionContext>(
        engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(context, "InternVLAN1System1Runner: failed to create context for " + path);
    ELLM_CHECK(context->setOptimizationProfileAsync(0, stream),
        "InternVLAN1System1Runner: failed to set the optimization profile for " + path);
}

int64_t InternVLAN1System1Runner::getRequiredContextMemorySize() const
{
    int64_t const memorySize = mMemoryEngine ? mMemoryEngine->getDeviceMemorySizeV2() : 0;
    int64_t const ditSize = mDitEngine ? mDitEngine->getDeviceMemorySizeV2() : 0;
    // The two engines run one after the other within a plan, so the pool only has to hold the
    // larger of them.
    return std::max(memorySize, ditSize);
}

rt::Tensor& InternVLAN1System1Runner::encodeMemory(rt::Tensor const& images, cudaStream_t stream)
{
    auto const shape = shapeOf(images);
    ELLM_CHECK(shape.size() == 4U, "InternVLAN1System1Runner::encodeMemory: expected [frames, 3, H, W]");

    Dims imageDims{};
    imageDims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        imageDims.d[i] = shape[i];
    }
    ELLM_CHECK(mMemoryContext->setInputShape(kImages, imageDims),
        "InternVLAN1System1Runner::encodeMemory: failed to set the frame shape");
    ELLM_CHECK(mMemoryContext->setTensorAddress(kImages, const_cast<void*>(images.rawPointer())),
        "InternVLAN1System1Runner::encodeMemory: failed to bind images");

    // The token count is fixed by the resampler, but read it from the engine rather than the
    // config so the two cannot disagree.
    auto const outShape = dimsToVector(mMemoryContext->getTensorShape(kMemoryTokens));
    if (shapeOf(mMemoryTokens) != outShape)
    {
        mMemoryTokens
            = rt::Tensor(rt::Coords(outShape), rt::DeviceType::kGPU, DataType::kFLOAT, "internvla_n1::memoryTokens");
    }
    ELLM_CHECK(mMemoryContext->setTensorAddress(kMemoryTokens, mMemoryTokens.rawPointer()),
        "InternVLAN1System1Runner::encodeMemory: failed to bind memory_tokens");
    ELLM_CHECK(mMemoryContext->enqueueV3(stream), "InternVLAN1System1Runner::encodeMemory: enqueue failed");
    return mMemoryTokens;
}

rt::Tensor& InternVLAN1System1Runner::sampleTrajectory(
    rt::Tensor const& conditioning, rt::Tensor const& noise, cudaStream_t stream)
{
    auto const condShape = shapeOf(conditioning);
    ELLM_CHECK(condShape.size() == 3U && condShape[0] == 2,
        "InternVLAN1System1Runner::sampleTrajectory: conditioning must be [2, condLen, latentDim]");

    // Both contexts are USER_MANAGED, so the scratch has to be supplied before the first
    // enqueue. One pool serves both: within a plan they run one after the other.
    int64_t const contextBytes = getRequiredContextMemorySize();
    mContextMemory = rt::Tensor(rt::Coords(std::vector<int64_t>{contextBytes}), rt::DeviceType::kGPU, DataType::kUINT8,
        "internvla_n1::contextMemory");
    mMemoryContext->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);
    mDitContext->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);

    int64_t const batch = mConfig.numSampleTrajs;
    int64_t const doubled = 2 * batch;
    int64_t const perBatch = mLatents.getShape().volume();

    // Expand [null, real] to one row per trajectory, matching the doubled latent layout.
    std::vector<int64_t> const doubledCondShape{doubled, condShape[1], condShape[2]};
    if (shapeOf(mDoubledCond) != doubledCondShape)
    {
        mDoubledCond = rt::Tensor(
            rt::Coords(doubledCondShape), rt::DeviceType::kGPU, DataType::kFLOAT, "internvla_n1::doubledCond");
    }
    int64_t const rowElems = condShape[1] * condShape[2];
    auto const* condData = static_cast<float const*>(conditioning.rawPointer());
    auto* doubledCondData = static_cast<float*>(mDoubledCond.rawPointer());
    for (int64_t half = 0; half < 2; ++half)
    {
        for (int64_t i = 0; i < batch; ++i)
        {
            CUDA_CHECK(cudaMemcpyAsync(doubledCondData + (half * batch + i) * rowElems, condData + half * rowElems,
                static_cast<size_t>(rowElems) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mLatents.rawPointer(), noise.rawPointer(), static_cast<size_t>(perBatch) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    Dims condDims{};
    condDims.nbDims = 3;
    condDims.d[0] = doubled;
    condDims.d[1] = condShape[1];
    condDims.d[2] = condShape[2];
    ELLM_CHECK(mDitContext->setInputShape(kZLatents, condDims),
        "InternVLAN1System1Runner::sampleTrajectory: failed to set the z_latents shape");
    ELLM_CHECK(mDitContext->setTensorAddress(kZLatents, mDoubledCond.rawPointer())
            && mDitContext->setTensorAddress(kLatents, mDoubledLatents.rawPointer())
            && mDitContext->setTensorAddress(kTimestep, mTimesteps.rawPointer())
            && mDitContext->setTensorAddress(kOutput, mModelOutput.rawPointer()),
        "InternVLAN1System1Runner::sampleTrajectory: failed to bind the expert tensors");

    for (int32_t step = 0; step < mScheduler.numInferenceSteps(); ++step)
    {
        launchDuplicateBatch(static_cast<float*>(mDoubledLatents.rawPointer()),
            static_cast<float const*>(mLatents.rawPointer()), perBatch, stream);

        // The timestep is one scalar broadcast across the batch; the engine takes it per-row.
        //
        // This copies from host and synchronizes, once per step. Two cheaper shapes were tried
        // and both broke: binding an offset into a pre-uploaded schedule silently reuses step 0
        // (TensorRT resolves the address once), and a device-to-device copy into the bound
        // buffer fails with an invalid argument. The synchronize is not decoration -- the host
        // vector dies at the end of this iteration, and an async copy outliving its source is a
        // use-after-free. Cost is measured: 46.4 ms for the whole loop, so it is not the
        // bottleneck.
        std::vector<int64_t> timestepRow(static_cast<size_t>(doubled), mScheduler.timestepIndexAt(step));
        CUDA_CHECK(cudaMemcpyAsync(mTimesteps.rawPointer(), timestepRow.data(), timestepRow.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        ELLM_CHECK(mDitContext->enqueueV3(stream), "InternVLAN1System1Runner::sampleTrajectory: expert enqueue failed");

        launchGuidedEulerStep(static_cast<float*>(mLatents.rawPointer()),
            static_cast<float const*>(mModelOutput.rawPointer()), perBatch, mConfig.guidanceScale,
            mScheduler.dtAt(step), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return mLatents;
}

} // namespace internvla_n1
} // namespace trt_edgellm
