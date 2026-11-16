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

#include "runtime/commitPolicy/recurrentCommitPolicy.h"
#include "common/checkMacros.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

namespace trt_edgellm
{
namespace rt
{

RecurrentCommitPolicy::RecurrentCommitPolicy(Config config, std::vector<rt::Tensor>& liveRecurrent,
    std::vector<rt::Tensor>& liveConv, std::string const& name, cudaStream_t stream)
    : mConfig(config)
    , mLiveRecurrent(liveRecurrent)
    , mLiveConv(liveConv)
{
    if (mConfig.numRecurrentLayers == 0 || mConfig.maxIntermediateSeqLen == 0)
    {
        return;
    }

    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.convDim > 0, "convDim must be positive.");
    check::check(mConfig.convKernel > 0, "convKernel must be positive.");

    size_t const convElemSize = rt::utils::getTypeSize(mConfig.convStateType);
    mIntermediateConvStates.reserve(mConfig.numRecurrentLayers);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        int64_t const intermConvVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.maxIntermediateSeqLen
            * mConfig.convDim * mConfig.convKernel;
        size_t const intermConvBytes = static_cast<size_t>(intermConvVolume) * convElemSize;
        mIntermediateConvStates.emplace_back(
            rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.convDim, mConfig.convKernel},
                DeviceType::kGPU, mConfig.convStateType, name + "::intermediateConvState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mIntermediateConvStates.back().rawPointer(), 0, intermConvBytes, stream));
    }
}

void RecurrentCommitPolicy::buildMtpLayerInfos(std::vector<void*> const& recurrentSrcPointers, cudaStream_t stream)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }
    check::check(static_cast<int32_t>(recurrentSrcPointers.size()) == mConfig.numRecurrentLayers,
        "recurrentSrcPointers must carry one entry per recurrent layer.");

    std::vector<kernel::MtpLayerInfo> hostInfos(mConfig.numRecurrentLayers);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        hostInfos[i] = {
            mLiveRecurrent[i].rawPointer(),
            recurrentSrcPointers[i],
            mLiveConv[i].rawPointer(),
            mIntermediateConvStates[i].rawPointer(),
        };
    }
    size_t const infoBytes = hostInfos.size() * sizeof(kernel::MtpLayerInfo);
    mDeviceMtpLayerInfos = rt::Tensor(
        {static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8, "RecurrentCommitPolicy::mtpLayerInfos");
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceMtpLayerInfos.rawPointer(), hostInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));
    // Sync before hostInfos, a pageable host vector, falls out of scope.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int32_t RecurrentCommitPolicy::numLayers() const noexcept
{
    return mConfig.numRecurrentLayers;
}

bool RecurrentCommitPolicy::hasIntermediateConvStates() const noexcept
{
    return !mIntermediateConvStates.empty();
}

void RecurrentCommitPolicy::reshapeIntermediate(int32_t activeBatchSize, int32_t seqLen)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }
    reshapeRecurrentSide(activeBatchSize, seqLen);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        check::check(mIntermediateConvStates[i].reshape({activeBatchSize, seqLen, mConfig.convDim, mConfig.convKernel}),
            "Intermediate conv state reshape failed");
    }
}

rt::Tensor& RecurrentCommitPolicy::intermediateConvState(int32_t recurrentLayerIdx)
{
    return mIntermediateConvStates[recurrentLayerIdx];
}

} // namespace rt
} // namespace trt_edgellm
