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

#include "common/pi05Common.h"

#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>

namespace trt_edgellm
{
namespace pi05
{

void Pi05Component::load(std::string const& engineDir, std::string const& component)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / component / (component + ".engine");
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime for pi0.5 " + component);
    }
    engine = deserializeCudaEngineFromFile(*runtime, enginePath.string());
    if (!engine)
    {
        throw std::runtime_error("Failed to load pi0.5 engine: " + enginePath.string());
    }
    context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!context)
    {
        throw std::runtime_error("Failed to create execution context for pi0.5 " + component);
    }
    setNonBlockingAuxStreams(context.get(), engine.get(), auxStreams);
    LOG_INFO("Loaded pi0.5 %s engine: %s (context memory %.1f MiB)", component.c_str(), enginePath.string().c_str(),
        static_cast<double>(engine->getDeviceMemorySizeV2()) / (1024.0 * 1024.0));
}

int64_t Pi05Component::getRequiredContextMemorySize() const
{
    return engine ? engine->getDeviceMemorySizeV2() : 0;
}

bool Pi05Component::setContextMemory(rt::Tensor& pool)
{
    if (!context)
    {
        return true;
    }
    if (pool.getMemoryCapacity() < getRequiredContextMemorySize())
    {
        return false;
    }
    context->setDeviceMemoryV2(pool.rawPointer(), pool.getMemoryCapacity());
    return true;
}

void stageRopeInputs(rt::Tensor const& ropeCache, int32_t headDim, int32_t batch, int32_t len, int32_t posOffset,
    rt::Tensor& cosSin, rt::Tensor& posIds, rt::Tensor& posIdsHost, cudaStream_t stream)
{
    // attention_pos_id indexes INTO the bound cos/sin, not into absolute positions:
    // the tower's offset is carried by which cache rows are copied here.
    auto* ids = posIdsHost.dataPointer<int32_t>();
    for (int32_t i = 0; i < len; ++i)
    {
        ids[i] = i;
    }
    for (int32_t b = 1; b < batch; ++b)
    {
        std::copy_n(ids, len, ids + static_cast<size_t>(b) * len);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        posIds.rawPointer(), ids, static_cast<size_t>(batch) * len * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // The engine input is packed at the bound length, so the cache rows are replicated
    // rather than aliased once the batch exceeds one.
    size_t const entryElems = static_cast<size_t>(len) * headDim;
    float const* src = ropeCache.dataPointer<float>() + static_cast<size_t>(posOffset) * headDim;
    for (int32_t b = 0; b < batch; ++b)
    {
        CUDA_CHECK(cudaMemcpyAsync(cosSin.dataPointer<float>() + static_cast<size_t>(b) * entryElems, src,
            entryElems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace pi05
} // namespace trt_edgellm
