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

#include "multimodal/pi05VisualRunner.h"

#include "common/pi05Bindings.h"

#include <stdexcept>

using nvinfer1::DataType;
using nvinfer1::Dims;

namespace trt_edgellm
{
namespace pi05
{

Pi05VisualRunner::Pi05VisualRunner(std::string const& engineDir, Pi05PolicyConfig const& config, cudaStream_t stream)
    : mConfig(config)
    , mStream(stream)
{
    mEngine.load(engineDir, "visual");

    // The tower's batch axis counts views, not requests: a batched request with its
    // own cameras needs a profile built for batch * views.
    Dims const viewsMax
        = mEngine.engine->getProfileShape(binding_names::kPixelValues, 0, nvinfer1::OptProfileSelector::kMAX);
    mImageFeatures = rt::Tensor({viewsMax.d[0], mConfig.numImageTokens, mConfig.hiddenSize}, rt::DeviceType::kGPU,
        DataType::kHALF, "pi05::imageFeatures");

    CUDA_CHECK(cudaEventCreate(&mBegin));
    CUDA_CHECK(cudaEventCreate(&mEnd));
}

Pi05VisualRunner::~Pi05VisualRunner() noexcept
{
    for (cudaEvent_t event : {mBegin, mEnd})
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

rt::Tensor const& Pi05VisualRunner::encode(rt::Tensor const& pixelValues)
{
    cudaEventRecord(mBegin, mStream);

    rt::Coords const shape = pixelValues.getShape();
    int32_t const views = static_cast<int32_t>(shape[0]);

    bool ok = mEngine.context->setInputShape(
        binding_names::kPixelValues, Dims{4, {views, 3, mConfig.imageSize, mConfig.imageSize}});
    ok &= mEngine.context->setTensorAddress(binding_names::kPixelValues, const_cast<void*>(pixelValues.rawPointer()));
    ok &= mImageFeatures.reshape({views, mConfig.numImageTokens, mConfig.hiddenSize});
    ok &= mEngine.context->setTensorAddress(binding_names::kImageFeatures, mImageFeatures.rawPointer());
    if (!ok || !mEngine.context->enqueueV3(mStream))
    {
        throw std::runtime_error("pi0.5 visual engine execution failed");
    }

    cudaEventRecord(mEnd, mStream);
    return mImageFeatures;
}

float Pi05VisualRunner::getElapsedMs() const noexcept
{
    float elapsed = 0.0F;
    if (cudaEventElapsedTime(&elapsed, mBegin, mEnd) != cudaSuccess)
    {
        return 0.0F;
    }
    return elapsed;
}

} // namespace pi05
} // namespace trt_edgellm
