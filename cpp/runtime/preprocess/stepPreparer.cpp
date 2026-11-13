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

#include "runtime/preprocess/stepPreparer.h"

#include "common/checkMacros.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"

namespace trt_edgellm
{
namespace rt
{

StepPreparer::StepPreparer(LLMEngineConfig const& config)
    : mConfig(config)
{
    // Pre-allocate host scratch for selectTokenIndices (pinned memory avoids
    // repeated allocation on the hot path).
    mHostSelectTokenIndices = Tensor(
        {mConfig.maxSupportedBatchSize, 1}, DeviceType::kCPU, trt_edgellm::DataType::kINT64, "hostSelectTokenIndices");
}

void StepPreparer::prepare(
    InferencePhase phase, int32_t batchSize, HybridCacheManager& kvCache, PipelineIO& io, cudaStream_t stream)
{
    // --- Reshape IO tensors for the current batch ---
    check::check(io.selectTokenIndices.reshape({batchSize, 1}), "selectTokenIndices reshape failed");
    check::check(io.contextLengths.reshape({batchSize}), "contextLengths reshape failed");

    if (phase == InferencePhase::kPrefill)
    {
        // -- selectTokenIndices: last real token position for each batch element --
        check::check(mHostSelectTokenIndices.reshape({batchSize, 1}), "hostSelectTokenIndices reshape failed");
        int64_t* selectData = mHostSelectTokenIndices.dataPointer<int64_t>();
        int32_t const* ctxData = io.hostContextLengths.dataPointer<int32_t>();
        for (int32_t i = 0; i < batchSize; ++i)
        {
            selectData[i] = static_cast<int64_t>(ctxData[i] - 1);
        }
        CUDA_CHECK(cudaMemcpyAsync(io.selectTokenIndices.rawPointer(), mHostSelectTokenIndices.rawPointer(),
            batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        // -- contextLengths: H2D copy from host --
        CUDA_CHECK(cudaMemcpyAsync(io.contextLengths.rawPointer(), io.hostContextLengths.rawPointer(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    else
    {
        // -- selectTokenIndices: always zero for decode (single token) --
        CUDA_CHECK(cudaMemsetAsync(io.selectTokenIndices.rawPointer(), 0, batchSize * sizeof(int64_t), stream));

        // -- contextLengths: KV cache lengths + 1 --
        Tensor& kvLengths = kvCache.getKVCacheLengths();
        CUDA_CHECK(cudaMemcpyAsync(io.contextLengths.rawPointer(), kvLengths.rawPointer(), batchSize * sizeof(int32_t),
            cudaMemcpyDeviceToDevice, stream));
        constexpr int32_t kDecodeIncrement{1};
        kernel::incrementLengthTensor(io.contextLengths, kDecodeIncrement, stream);
    }
}

} // namespace rt
} // namespace trt_edgellm
