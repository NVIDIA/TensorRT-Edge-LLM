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

#include "runtime/commitPolicy/mamba2CommitPolicy.h"
#include "common/checkMacros.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

namespace trt_edgellm
{
namespace rt
{

Mamba2CommitPolicy::Mamba2CommitPolicy(
    Config config, std::vector<rt::Tensor>& liveRecurrent, std::vector<rt::Tensor>& liveConv, cudaStream_t stream)
    : RecurrentCommitPolicy(config, liveRecurrent, liveConv, "Mamba2CommitPolicy", stream)
{
    if (mConfig.numRecurrentLayers == 0 || mConfig.maxIntermediateSeqLen == 0)
    {
        return;
    }
    check::check(
        mConfig.recurrentStateNumGroups > 0, "recurrentStateNumGroups must be positive for Mamba spec-verify.");

    mReplayDaStates.reserve(mConfig.numRecurrentLayers);
    mReplayUStates.reserve(mConfig.numRecurrentLayers);
    mReplayBStates.reserve(mConfig.numRecurrentLayers);
    mReplayDtStates.reserve(mConfig.numRecurrentLayers);

    int64_t const stashSeq = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.maxIntermediateSeqLen;
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        auto const idx = std::to_string(i);
        // Replay stash (FP32): dA/dt [B,S,H], x in u [B,S,H,dim], B [B,S,ngroups,dstate].
        int64_t const daVolume = stashSeq * mConfig.recurrentStateNumHeads;
        int64_t const uVolume = daVolume * mConfig.recurrentStateHeadDim;
        int64_t const bVolume = stashSeq * mConfig.recurrentStateNumGroups * mConfig.recurrentStateSize;

        mReplayDaStates.emplace_back(
            rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads},
                DeviceType::kGPU, DataType::kFLOAT, "Mamba2CommitPolicy::replayDa_" + idx));
        mReplayUStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen,
                                                   mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim},
            DeviceType::kGPU, DataType::kFLOAT, "Mamba2CommitPolicy::replayU_" + idx));
        mReplayBStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen,
                                                   mConfig.recurrentStateNumGroups, mConfig.recurrentStateSize},
            DeviceType::kGPU, DataType::kFLOAT, "Mamba2CommitPolicy::replayB_" + idx));
        mReplayDtStates.emplace_back(
            rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads},
                DeviceType::kGPU, DataType::kFLOAT, "Mamba2CommitPolicy::replayDt_" + idx));
        CUDA_CHECK(cudaMemsetAsync(mReplayDaStates.back().rawPointer(), 0, daVolume * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(mReplayUStates.back().rawPointer(), 0, uVolume * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(mReplayBStates.back().rawPointer(), 0, bVolume * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(mReplayDtStates.back().rawPointer(), 0, daVolume * sizeof(float), stream));
    }

    // The recurrent state commits via replay reconstruction, not scatter, so the MtpLayerInfo
    // recurrent source is null; only the conv source and the dst pools are read from it.
    buildMtpLayerInfos(std::vector<void*>(mConfig.numRecurrentLayers, nullptr), stream);

    std::vector<mamba_ssm::MambaReplayLayerInfo> replayInfos(mConfig.numRecurrentLayers);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        replayInfos[i] = {mLiveRecurrent[i].rawPointer(), mReplayDaStates[i].rawPointer(),
            mReplayUStates[i].rawPointer(), mReplayBStates[i].rawPointer(), mReplayDtStates[i].rawPointer()};
    }
    size_t const replayInfoBytes = replayInfos.size() * sizeof(mamba_ssm::MambaReplayLayerInfo);
    mDeviceReplayLayerInfos = rt::Tensor({static_cast<int64_t>(replayInfoBytes)}, DeviceType::kGPU, DataType::kINT8,
        "Mamba2CommitPolicy::mambaReplayLayerInfos");
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceReplayLayerInfos.rawPointer(), replayInfos.data(), replayInfoBytes, cudaMemcpyHostToDevice, stream));
    // Sync before replayInfos, a pageable host vector, falls out of scope.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool Mamba2CommitPolicy::hasIntermediateRecurrentStates() const noexcept
{
    return !mReplayUStates.empty();
}

bool Mamba2CommitPolicy::usesReplay() const noexcept
{
    return true;
}

void Mamba2CommitPolicy::reshapeRecurrentSide(int32_t activeBatchSize, int32_t seqLen)
{
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        check::check(mReplayDaStates[i].reshape({activeBatchSize, seqLen, mConfig.recurrentStateNumHeads}),
            "Replay dA reshape failed");
        check::check(mReplayUStates[i].reshape(
                         {activeBatchSize, seqLen, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim}),
            "Replay u reshape failed");
        check::check(mReplayBStates[i].reshape(
                         {activeBatchSize, seqLen, mConfig.recurrentStateNumGroups, mConfig.recurrentStateSize}),
            "Replay B reshape failed");
        check::check(mReplayDtStates[i].reshape({activeBatchSize, seqLen, mConfig.recurrentStateNumHeads}),
            "Replay dt reshape failed");
    }
}

void Mamba2CommitPolicy::commitLinear(
    rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, cudaStream_t stream)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }

    int32_t const verifySize = static_cast<int32_t>(mReplayUStates[0].getShape()[1]);
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    int32_t const activeBatchSize = static_cast<int32_t>(acceptLengths.getShape()[0]);
    check::check(stateIndices.getDeviceType() == DeviceType::kGPU && stateIndices.getDataType() == DataType::kINT32
            && stateIndices.getShape().getNumDims() == 1 && stateIndices.getShape()[0] == activeBatchSize,
        "stateIndices must be a GPU INT32 tensor with shape [activeBatchSize].");
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    int32_t const* const acceptPtr = acceptLengths.dataPointer<int32_t>();

    // Reconstruct the accepted recurrent state by replaying the SSD scan from the read-only committed
    // state, consuming the per-token replay stash the verify pass wrote.
    auto const* const replayInfos
        = static_cast<mamba_ssm::MambaReplayLayerInfo const*>(mDeviceReplayLayerInfos.rawPointer());
    mamba_ssm::invokeMambaReplayReconstructBatched(replayInfos, mConfig.numRecurrentLayers, mLiveRecurrent[0],
        mReplayUStates[0], mReplayBStates[0], mReplayDtStates[0], acceptLengths, stateIndices, activeBatchSize, stream);
    kernel::mtpScatterConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize, mConfig.maxBatchSize,
        verifySize, convElements, acceptPtr, stateIndices.dataPointer<int32_t>(), stream);
}

void Mamba2CommitPolicy::commitTree(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths,
    rt::Tensor const& stateIndices, int32_t /*verifySize*/, cudaStream_t stream)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }
    check::check(acceptedNodeIds.getDeviceType() == DeviceType::kGPU, "acceptedNodeIds must be a GPU tensor.");
    check::check(acceptLengths.getDeviceType() == DeviceType::kGPU, "acceptLengths must be a GPU tensor.");
    check::check(acceptedNodeIds.getDataType() == DataType::kINT32, "acceptedNodeIds must have INT32 data type.");
    check::check(acceptLengths.getDataType() == DataType::kINT32, "acceptLengths must have INT32 data type.");

    auto const idsShape = acceptedNodeIds.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    check::check(idsShape.getNumDims() == 2, "acceptedNodeIds must have shape [B, maxAcceptLen].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths must have shape [B].");
    int32_t const activeBatchSize = static_cast<int32_t>(idsShape[0]);
    check::check(acceptLengthsShape[0] == activeBatchSize, "acceptedNodeIds and acceptLengths batch sizes differ.");
    int32_t const maxAcceptLen = static_cast<int32_t>(idsShape[1]);
    if (activeBatchSize == 0 || maxAcceptLen == 0)
    {
        return;
    }
    check::check(stateIndices.getDeviceType() == DeviceType::kGPU && stateIndices.getDataType() == DataType::kINT32
            && stateIndices.getShape().getNumDims() == 1 && stateIndices.getShape()[0] == activeBatchSize,
        "stateIndices must be a GPU INT32 tensor with shape [activeBatchSize].");

    auto const replayShape = mReplayUStates[0].getShape();
    check::check(replayShape[0] == activeBatchSize,
        "Mamba replay states must be reshaped to the active batch size before tree commit.");
    int32_t const verifyTreeSize = static_cast<int32_t>(replayShape[1]);
    check::check(maxAcceptLen <= verifyTreeSize, "maxAcceptLen must not exceed the Mamba replay tree size.");
    auto const* const replayInfos
        = static_cast<mamba_ssm::MambaReplayLayerInfo const*>(mDeviceReplayLayerInfos.rawPointer());
    mamba_ssm::invokeMambaReplayReconstructBatched(replayInfos, mConfig.numRecurrentLayers, mLiveRecurrent[0],
        mReplayUStates[0], mReplayBStates[0], mReplayDtStates[0], acceptLengths, stateIndices, activeBatchSize, stream,
        &acceptedNodeIds, maxAcceptLen);

    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    kernel::mtpScatterAcceptedTreeConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize,
        mConfig.maxBatchSize, verifyTreeSize, convElements, acceptedNodeIds.dataPointer<int32_t>(), maxAcceptLen,
        acceptLengths.dataPointer<int32_t>(), stateIndices.dataPointer<int32_t>(), stream);
}

rt::Tensor& Mamba2CommitPolicy::replayDaState(int32_t recurrentLayerIdx)
{
    return mReplayDaStates[recurrentLayerIdx];
}

rt::Tensor& Mamba2CommitPolicy::replayUState(int32_t recurrentLayerIdx)
{
    return mReplayUStates[recurrentLayerIdx];
}

rt::Tensor& Mamba2CommitPolicy::replayBState(int32_t recurrentLayerIdx)
{
    return mReplayBStates[recurrentLayerIdx];
}

rt::Tensor& Mamba2CommitPolicy::replayDtState(int32_t recurrentLayerIdx)
{
    return mReplayDtStates[recurrentLayerIdx];
}

} // namespace rt
} // namespace trt_edgellm
