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

#include "multimodal/cosmos3VaeEncoderRunner.h"
#include "common/cosmos3Bindings.h"
#include "common/trtInterop.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "profiling/nvtx_wrapper.h"

#include <stdexcept>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace trt_edgellm
{
namespace cosmos3
{

Cosmos3VaeEncoderRunner::Cosmos3VaeEncoderRunner(std::string const& engineDir, cudaStream_t stream)
{
    std::string const enginePath = engineDir + "/vae_encoder.engine";
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(TrtLogger::instance()));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "Failed to create execution context");
    ELLM_CHECK(mContext->setOptimizationProfileAsync(0, stream), "Failed to set optimization profile");

    // Allocate the conditioning-latent output once at the engine's MAXIMUM profile shape (the pixel
    // grid is contract-fixed; the batch axis max comes from the builder's --max-batch-size) and bind
    // its address once. Per-request shapes are metadata-only reshapes within this capacity.
    Dims const maxImage = mEngine->getProfileShape(binding_names::kVaeImage, 0, OptProfileSelector::kMAX);
    ELLM_CHECK(mContext->setInputShape(binding_names::kVaeImage, maxImage),
        "Cosmos3VaeEncoderRunner: failed to set the profile pixel shape");
    Dims const out = mContext->getTensorShape(binding_names::kCondLatent);
    mLatentShape.assign(out.d, out.d + out.nbDims);
    mCondLatentDevice = rt::Tensor(rt::Coords(std::vector<int64_t>(out.d, out.d + out.nbDims)), rt::DeviceType::kGPU,
        DataType::kFLOAT, "cosmos3::condLatent");
    ELLM_CHECK(mContext->setTensorAddress(binding_names::kCondLatent, mCondLatentDevice.rawPointer()),
        "Cosmos3VaeEncoderRunner: failed to bind cond_latent");
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int64_t Cosmos3VaeEncoderRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool Cosmos3VaeEncoderRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (sharedContextMemory.getMemoryCapacity() < getRequiredContextMemorySize())
    {
        return false;
    }
    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

rt::Tensor& Cosmos3VaeEncoderRunner::encode(rt::Tensor const& pixelValues, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(vaeRange, "cosmos3::vae_encode");

    Dims inDims{};
    inDims.nbDims = pixelValues.getShape().getNumDims();
    for (int32_t i = 0; i < inDims.nbDims; ++i)
    {
        inDims.d[i] = pixelValues.getShape()[i];
    }
    ELLM_CHECK(mContext->setInputShape(binding_names::kVaeImage, inDims),
        "Cosmos3VaeEncoderRunner: failed to set pixel_values input shape");

    // The latent output is preallocated at construction for the fixed policy pixel shape; a different
    // input shape must still fit the preallocated capacity (metadata-only reshape).
    Dims const out = mContext->getTensorShape(binding_names::kCondLatent);
    rt::Coords const outCoords(std::vector<int64_t>(out.d, out.d + out.nbDims));
    ELLM_CHECK(mCondLatentDevice.reshape(outCoords), "Cosmos3VaeEncoderRunner: cond_latent exceeds capacity");
    mLatentShape.assign(out.d, out.d + out.nbDims);

    ELLM_CHECK(mContext->setInputTensorAddress(binding_names::kVaeImage, pixelValues.rawPointer()),
        "Cosmos3VaeEncoderRunner: failed to bind pixel_values");
    if (!mContext->enqueueV3(stream))
    {
        throw std::runtime_error("Cosmos3VaeEncoderRunner: enqueueV3 failed");
    }
    return mCondLatentDevice;
}

} // namespace cosmos3
} // namespace trt_edgellm
