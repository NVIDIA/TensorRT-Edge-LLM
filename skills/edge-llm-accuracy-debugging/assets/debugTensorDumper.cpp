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

#include "debug/debugTensorDumper.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtInterop.h"

#include <NvInferRuntime.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{

#ifdef SUPPORTS_DEBUG_TENSOR

DebugTensorDumper::DebugTensorDumper(std::filesystem::path outputDir)
    : mOutputDir(std::move(outputDir))
{
    if (!std::filesystem::exists(mOutputDir))
    {
        std::filesystem::create_directories(mOutputDir);
    }
    LOG_INFO("Output directory: %s", mOutputDir.string().c_str());
}

void DebugTensorDumper::beginStep(std::string const& label)
{
    mStepLabel = label;
    mPendingTensors.clear();
    mSeenNames.clear();
}

void DebugTensorDumper::flush(cudaStream_t stream)
{
    if (mPendingTensors.empty())
    {
        return;
    }

    std::ostringstream name;
    name << "step_" << std::setw(3) << std::setfill('0') << mStepIndex << "_" << mStepLabel << ".safetensors";
    std::filesystem::path const outPath = mOutputDir / name.str();

    if (rt::safetensors::saveSafetensors(outPath, mPendingTensors, stream))
    {
        LOG_INFO("Step %d (%s): wrote %zu tensors → %s", mStepIndex, mStepLabel.c_str(), mPendingTensors.size(),
            outPath.string().c_str());
    }
    else
    {
        LOG_WARNING("Step %d (%s): failed to write safetensors file", mStepIndex, mStepLabel.c_str());
    }

    mPendingTensors.clear();
    mSeenNames.clear();
    ++mStepIndex;
}

bool DebugTensorDumper::processDebugTensor(void const* addr, nvinfer1::TensorLocation location, nvinfer1::DataType type,
    nvinfer1::Dims const& shape, char const* name, cudaStream_t /*stream*/)
{
    if (!addr || !name)
    {
        return false;
    }

    if (!mSeenNames.insert(name).second)
    {
        return true;
    }

    int64_t numElements = 1;
    for (int32_t i = 0; i < shape.nbDims; ++i)
    {
        if (shape.d[i] <= 0)
        {
            return true;
        }
        numElements *= shape.d[i];
    }

    size_t nbytes;
    try
    {
        nbytes = static_cast<size_t>(numElements) * rt::utils::getTypeSize(fromTrt(type));
    }
    catch (std::runtime_error const& e)
    {
        LOG_WARNING("Skipping tensor '%s': %s", name, e.what());
        return true;
    }
    if (nbytes == 0)
    {
        return true;
    }

    rt::Tensor tensor(rt::toCoords(shape), rt::DeviceType::kCPU, fromTrt(type), name);

    if (location == nvinfer1::TensorLocation::kDEVICE)
    {
        cudaError_t err = cudaMemcpy(tensor.rawPointer(), addr, nbytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            LOG_WARNING("cudaMemcpy failed for tensor '%s': %s", name, cudaGetErrorString(err));
            return false;
        }
    }
    else
    {
        std::memcpy(tensor.rawPointer(), addr, nbytes);
    }

    LOG_INFO("  %s  shape=%s", name, tensor.getShape().formatString().c_str());

    mPendingTensors.push_back(std::move(tensor));
    return true;
}

#endif // SUPPORTS_DEBUG_TENSOR

} // namespace trt_edgellm
