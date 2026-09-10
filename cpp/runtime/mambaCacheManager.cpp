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

#include "runtime/mambaCacheManager.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

MambaCacheManager::MambaCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
{
    if (mConfig.numRecurrentLayers == 0)
    {
        return;
    }

    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.recurrentStateNumHeads > 0, "recurrentStateNumHeads must be positive.");
    check::check(mConfig.recurrentStateHeadDim > 0, "recurrentStateHeadDim must be positive.");
    check::check(mConfig.recurrentStateSize > 0, "recurrentStateSize must be positive.");
    check::check(mConfig.convDim > 0, "convDim must be positive.");
    check::check(mConfig.convKernel > 0, "convKernel must be positive.");

    size_t totalBytes = 0;
    size_t const recurrentElemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mConfig.convStateType);

    mRecurrentStates.reserve(mConfig.numRecurrentLayers);
    mConvStates.reserve(mConfig.numRecurrentLayers);

    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        // Allocate recurrent state: [maxBatchSize, numHeads, headDim, stateSize]
        int64_t const recurrentVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.recurrentStateNumHeads
            * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
        size_t const recurrentBytes = static_cast<size_t>(recurrentVolume) * recurrentElemSize;

        mRecurrentStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.recurrentStateNumHeads,
                                                     mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
            DeviceType::kGPU, mConfig.recurrentStateType, "MambaCacheManager::recurrentState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mRecurrentStates.back().rawPointer(), 0, recurrentBytes, stream));

        // Allocate conv state: [maxBatchSize, convDim, convKernel]
        int64_t const convVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.convDim * mConfig.convKernel;
        size_t const convBytes = static_cast<size_t>(convVolume) * convElemSize;

        mConvStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.convDim, mConfig.convKernel},
            DeviceType::kGPU, mConfig.convStateType, "MambaCacheManager::convState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mConvStates.back().rawPointer(), 0, convBytes, stream));

        totalBytes += recurrentBytes + convBytes;
    }

    // Allocate spec-verify intermediate state buffers when enabled (maxIntermediateSeqLen > 0). The
    // recurrent-state commit path is selected by the engine's explicit spec-verify mode: replay =
    // Mamba; snapshot = GDN/DDTree, which keeps the per-token full-state snapshot committed by the scatter.
    // The conv state always uses per-token snapshots committed by the batched scatter.
    bool const useReplay = mConfig.specVerifyUsesReplay;
    if (mConfig.maxIntermediateSeqLen > 0)
    {
        mIntermediateConvStates.reserve(mConfig.numRecurrentLayers);
        if (useReplay)
        {
            check::check(
                mConfig.recurrentStateNumGroups > 0, "recurrentStateNumGroups must be positive for Mamba spec-verify.");
            mReplayDaStates.reserve(mConfig.numRecurrentLayers);
            mReplayUStates.reserve(mConfig.numRecurrentLayers);
            mReplayBStates.reserve(mConfig.numRecurrentLayers);
        }
        else
        {
            mIntermediateRecurrentStates.reserve(mConfig.numRecurrentLayers);
        }

        int64_t const stashSeq = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.maxIntermediateSeqLen;
        for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
        {
            auto const idx = std::to_string(i);
            if (useReplay)
            {
                // Replay stash (FP32): dA [B,S,H], u [B,S,H,dim], B [B,S,ngroups,dstate].
                int64_t const daVolume = stashSeq * mConfig.recurrentStateNumHeads;
                int64_t const uVolume = daVolume * mConfig.recurrentStateHeadDim;
                int64_t const bVolume = stashSeq * mConfig.recurrentStateNumGroups * mConfig.recurrentStateSize;

                mReplayDaStates.emplace_back(
                    rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads},
                        DeviceType::kGPU, DataType::kFLOAT, "MambaCacheManager::replayDa_" + idx));
                mReplayUStates.emplace_back(
                    rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads,
                                   mConfig.recurrentStateHeadDim},
                        DeviceType::kGPU, DataType::kFLOAT, "MambaCacheManager::replayU_" + idx));
                mReplayBStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen,
                                                           mConfig.recurrentStateNumGroups, mConfig.recurrentStateSize},
                    DeviceType::kGPU, DataType::kFLOAT, "MambaCacheManager::replayB_" + idx));
                CUDA_CHECK(cudaMemsetAsync(mReplayDaStates.back().rawPointer(), 0, daVolume * sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(mReplayUStates.back().rawPointer(), 0, uVolume * sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(mReplayBStates.back().rawPointer(), 0, bVolume * sizeof(float), stream));
                totalBytes += static_cast<size_t>(daVolume + uVolume + bVolume) * sizeof(float);
            }
            else
            {
                int64_t const intermRecVolume = stashSeq * mConfig.recurrentStateNumHeads
                    * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
                size_t const intermRecBytes = static_cast<size_t>(intermRecVolume) * recurrentElemSize;
                mIntermediateRecurrentStates.emplace_back(
                    rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads,
                                   mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
                        DeviceType::kGPU, mConfig.recurrentStateType,
                        "MambaCacheManager::intermediateRecurrentState_" + idx));
                CUDA_CHECK(
                    cudaMemsetAsync(mIntermediateRecurrentStates.back().rawPointer(), 0, intermRecBytes, stream));
                totalBytes += intermRecBytes;
            }

            int64_t const intermConvVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.maxIntermediateSeqLen
                * mConfig.convDim * mConfig.convKernel;
            size_t const intermConvBytes = static_cast<size_t>(intermConvVolume) * convElemSize;

            mIntermediateConvStates.emplace_back(
                rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.convDim, mConfig.convKernel},
                    DeviceType::kGPU, mConfig.convStateType, "MambaCacheManager::intermediateConvState_" + idx));
            CUDA_CHECK(cudaMemsetAsync(mIntermediateConvStates.back().rawPointer(), 0, intermConvBytes, stream));
            totalBytes += intermConvBytes;
        }

        // Build the MtpLayerInfo array for the batched conv-state (and, for GDN, recurrent) scatter;
        // pointers are stable, so upload runs once. In the Mamba replay path the recurrent source is
        // null (recurrent commits via replay reconstruction, not scatter).
        std::vector<kernel::MtpLayerInfo> hostInfos(mConfig.numRecurrentLayers);
        for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
        {
            hostInfos[i] = {
                mRecurrentStates[i].rawPointer(),
                useReplay ? nullptr : mIntermediateRecurrentStates[i].rawPointer(),
                mConvStates[i].rawPointer(),
                mIntermediateConvStates[i].rawPointer(),
            };
        }
        size_t const infoBytes = hostInfos.size() * sizeof(kernel::MtpLayerInfo);
        mDeviceMtpLayerInfos = rt::Tensor(
            {static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8, "MambaCacheManager::mtpLayerInfos");
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceMtpLayerInfos.rawPointer(), hostInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));
        if (useReplay)
        {
            std::vector<mamba_ssm::MambaReplayLayerInfo> replayInfos(mConfig.numRecurrentLayers);
            for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
            {
                replayInfos[i] = {mRecurrentStates[i].rawPointer(), mReplayDaStates[i].rawPointer(),
                    mReplayUStates[i].rawPointer(), mReplayBStates[i].rawPointer()};
            }
            size_t const replayInfoBytes = replayInfos.size() * sizeof(mamba_ssm::MambaReplayLayerInfo);
            mDeviceMambaReplayLayerInfos = rt::Tensor({static_cast<int64_t>(replayInfoBytes)}, DeviceType::kGPU,
                DataType::kINT8, "MambaCacheManager::mambaReplayLayerInfos");
            CUDA_CHECK(cudaMemcpyAsync(mDeviceMambaReplayLayerInfos.rawPointer(), replayInfos.data(), replayInfoBytes,
                cudaMemcpyHostToDevice, stream));
        }
        // Sync before hostInfos and replayInfos go out of scope.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        LOG_INFO("MambaCacheManager: allocated spec-verify intermediate state buffers (%d layers, maxSeqLen=%d)",
            mConfig.numRecurrentLayers, mConfig.maxIntermediateSeqLen);
    }

    LOG_DEBUG("MambaCacheManager(layers=%d) allocated %.2f MB total GPU memory", mConfig.numRecurrentLayers,
        static_cast<float>(totalBytes) / (1024.0f * 1024.0f));
}

MambaCacheManager::~MambaCacheManager() noexcept {}

MambaCacheManager::MambaCacheManager(MambaCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mRecurrentStates = std::move(other.mRecurrentStates);
    mConvStates = std::move(other.mConvStates);
    mIntermediateRecurrentStates = std::move(other.mIntermediateRecurrentStates);
    mIntermediateConvStates = std::move(other.mIntermediateConvStates);
    mReplayDaStates = std::move(other.mReplayDaStates);
    mReplayUStates = std::move(other.mReplayUStates);
    mReplayBStates = std::move(other.mReplayBStates);
    mDeviceMtpLayerInfos = std::move(other.mDeviceMtpLayerInfos);
    mDeviceMambaReplayLayerInfos = std::move(other.mDeviceMambaReplayLayerInfos);

    other.mConfig = Config{};
}

MambaCacheManager& MambaCacheManager::operator=(MambaCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mRecurrentStates = std::move(other.mRecurrentStates);
        mConvStates = std::move(other.mConvStates);
        mIntermediateRecurrentStates = std::move(other.mIntermediateRecurrentStates);
        mIntermediateConvStates = std::move(other.mIntermediateConvStates);
        mReplayDaStates = std::move(other.mReplayDaStates);
        mReplayUStates = std::move(other.mReplayUStates);
        mReplayBStates = std::move(other.mReplayBStates);
        mDeviceMtpLayerInfos = std::move(other.mDeviceMtpLayerInfos);
        mDeviceMambaReplayLayerInfos = std::move(other.mDeviceMambaReplayLayerInfos);

        other.mConfig = Config{};
    }
    return *this;
}

rt::Tensor& MambaCacheManager::getRecurrentState(int32_t recurrentLayerIdx) noexcept
{
    return mRecurrentStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getConvState(int32_t recurrentLayerIdx) noexcept
{
    return mConvStates[recurrentLayerIdx];
}

void MambaCacheManager::clearStates(cudaStream_t stream)
{
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        CUDA_CHECK(
            cudaMemsetAsync(mRecurrentStates[i].rawPointer(), 0, mRecurrentStates[i].getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(mConvStates[i].rawPointer(), 0, mConvStates[i].getMemoryCapacity(), stream));
    }
}

std::vector<rt::Tensor> MambaCacheManager::captureRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numRecurrentLayers == 0)
    {
        return result;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    int64_t const perBatchElems
        = mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numRecurrentLayers);
    for (int32_t layer = 0; layer < mConfig.numRecurrentLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mRecurrentStates[layer].rawPointer())
            + static_cast<size_t>(batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
            DeviceType::kGPU, mConfig.recurrentStateType,
            "MambaCacheManager::capturedRecurrentState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

std::vector<rt::Tensor> MambaCacheManager::captureConvStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numRecurrentLayers == 0)
    {
        return result;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.convStateType);
    int64_t const perBatchElems = mConfig.convDim * mConfig.convKernel;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numRecurrentLayers);
    for (int32_t layer = 0; layer < mConfig.numRecurrentLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mConvStates[layer].rawPointer())
            + static_cast<size_t>(batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.convDim, mConfig.convKernel}, DeviceType::kGPU, mConfig.convStateType,
            "MambaCacheManager::capturedConvState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

void MambaCacheManager::reshapeIntermediateStates(int32_t activeBatchSize, int32_t seqLen)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }
    bool const useReplay = !mReplayUStates.empty();
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        if (useReplay)
        {
            check::check(mReplayDaStates[i].reshape({activeBatchSize, seqLen, mConfig.recurrentStateNumHeads}),
                "Replay dA reshape failed");
            check::check(mReplayUStates[i].reshape(
                             {activeBatchSize, seqLen, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim}),
                "Replay u reshape failed");
            check::check(mReplayBStates[i].reshape(
                             {activeBatchSize, seqLen, mConfig.recurrentStateNumGroups, mConfig.recurrentStateSize}),
                "Replay B reshape failed");
        }
        else
        {
            check::check(
                mIntermediateRecurrentStates[i].reshape({activeBatchSize, seqLen, mConfig.recurrentStateNumHeads,
                    mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize}),
                "Intermediate recurrent state reshape failed");
        }
        check::check(mIntermediateConvStates[i].reshape({activeBatchSize, seqLen, mConfig.convDim, mConfig.convKernel}),
            "Intermediate conv state reshape failed");
    }
}

rt::Tensor& MambaCacheManager::getReplayDaState(int32_t recurrentLayerIdx) noexcept
{
    return mReplayDaStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getReplayUState(int32_t recurrentLayerIdx) noexcept
{
    return mReplayUStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getReplayBState(int32_t recurrentLayerIdx) noexcept
{
    return mReplayBStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getIntermediateRecurrentState(int32_t recurrentLayerIdx) noexcept
{
    return mIntermediateRecurrentStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getIntermediateConvState(int32_t recurrentLayerIdx) noexcept
{
    return mIntermediateConvStates[recurrentLayerIdx];
}

bool MambaCacheManager::hasIntermediateRecurrentStates() const noexcept
{
    return !mReplayUStates.empty() || !mIntermediateRecurrentStates.empty();
}

bool MambaCacheManager::recurrentUsesReplay() const noexcept
{
    return !mReplayUStates.empty();
}

bool MambaCacheManager::hasIntermediateConvStates() const noexcept
{
    return !mIntermediateConvStates.empty();
}

int32_t MambaCacheManager::numLayers() const noexcept
{
    return mConfig.numRecurrentLayers;
}

MambaCacheManager::Config const& MambaCacheManager::getConfig() const noexcept
{
    return mConfig;
}

void MambaCacheManager::scatterAcceptedLinearStates(rt::Tensor const& acceptLengths, cudaStream_t stream)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }

    bool const useReplay = !mReplayUStates.empty();
    int32_t const verifySize = static_cast<int32_t>(
        useReplay ? mReplayUStates[0].getShape()[1] : mIntermediateRecurrentStates[0].getShape()[1]);
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    int32_t const activeBatchSize = static_cast<int32_t>(acceptLengths.getShape()[0]);
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    int32_t const* const acceptPtr = acceptLengths.dataPointer<int32_t>();

    if (useReplay)
    {
        // Mamba: reconstruct the accepted recurrent state by replaying the SSD scan from the read-only
        // committed state, consuming the per-token replay stash the verify pass wrote.
        auto const* const replayInfos
            = static_cast<mamba_ssm::MambaReplayLayerInfo const*>(mDeviceMambaReplayLayerInfos.rawPointer());
        mamba_ssm::invokeMambaReplayReconstructBatched(replayInfos, mConfig.numRecurrentLayers, mRecurrentStates[0],
            mReplayUStates[0], mReplayBStates[0], acceptLengths, activeBatchSize, stream);
    }
    else
    {
        // GDN linear MTP: commit the accepted per-token recurrent snapshot via the batched scatter.
        int32_t const recElements
            = mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
        kernel::mtpScatterRecurrentStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize, verifySize,
            recElements, acceptPtr, stream, mConfig.recurrentStateType == DataType::kHALF);
    }
    // Conv state: commit the accepted per-token snapshot via the batched scatter.
    kernel::mtpScatterConvStates(
        layerInfos, mConfig.numRecurrentLayers, activeBatchSize, verifySize, convElements, acceptPtr, stream);
}

void MambaCacheManager::scatterAcceptedTreeStates(
    rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream)
{
    if (mIntermediateRecurrentStates.empty())
    {
        return;
    }

    check::check(!mIntermediateConvStates.empty(), "Intermediate conv states are not allocated.");
    check::check(mConfig.recurrentStateType == DataType::kFLOAT,
        "Accepted tree recurrent state scatter supports FP32 recurrent states only.");
    check::check(
        mConfig.convStateType == DataType::kHALF, "Accepted tree conv state scatter supports FP16 conv states only.");
    check::check(
        acceptedStateNodeIds.getDeviceType() == DeviceType::kGPU, "acceptedStateNodeIds must be a GPU tensor.");
    check::check(acceptLengths.getDeviceType() == DeviceType::kGPU, "acceptLengths must be a GPU tensor.");
    check::check(
        acceptedStateNodeIds.getDataType() == DataType::kINT32, "acceptedStateNodeIds must have INT32 data type.");
    check::check(acceptLengths.getDataType() == DataType::kINT32, "acceptLengths must have INT32 data type.");

    auto const acceptedStateNodeIdsShape = acceptedStateNodeIds.getShape();
    auto const acceptLengthsShape = acceptLengths.getShape();
    check::check(
        acceptedStateNodeIdsShape.getNumDims() == 2, "acceptedStateNodeIds must have shape [B, maxAcceptLen].");
    check::check(acceptLengthsShape.getNumDims() == 1, "acceptLengths must have shape [B].");

    int32_t const activeBatchSize = static_cast<int32_t>(acceptedStateNodeIdsShape[0]);
    int32_t const maxAcceptLen = static_cast<int32_t>(acceptedStateNodeIdsShape[1]);
    check::check(
        acceptLengthsShape[0] == activeBatchSize, "acceptedStateNodeIds and acceptLengths batch sizes differ.");
    if (activeBatchSize == 0 || maxAcceptLen == 0)
    {
        return;
    }

    auto const intermediateShape = mIntermediateRecurrentStates[0].getShape();
    check::check(intermediateShape[0] == activeBatchSize,
        "Intermediate recurrent states must be reshaped to the active batch size before accepted tree scatter.");
    int32_t const verifyTreeSize = static_cast<int32_t>(intermediateShape[1]);
    int32_t const recElements
        = mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    int32_t const* const acceptedStateNodeIdsPtr = acceptedStateNodeIds.dataPointer<int32_t>();
    int32_t const* const acceptLengthsPtr = acceptLengths.dataPointer<int32_t>();

    kernel::mtpScatterAcceptedTreeRecurrentStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize,
        verifyTreeSize, recElements, acceptedStateNodeIdsPtr, maxAcceptLen, acceptLengthsPtr, stream);
    kernel::mtpScatterAcceptedTreeConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize, verifyTreeSize,
        convElements, acceptedStateNodeIdsPtr, maxAcceptLen, acceptLengthsPtr, stream);
}

void MambaCacheManager::replayCommitAcceptedTreeStates(
    rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream)
{
    if (mIntermediateRecurrentStates.empty())
    {
        check::check(mReplayUStates.empty(),
            "Tree MTP is not supported for Mamba SSD replay recurrent states; chunk-form tree replay commit requires "
            "GDN intermediate recurrent states.");
        return;
    }

    check::check(!mIntermediateConvStates.empty(), "Intermediate conv states are not allocated.");
    check::check(mConfig.recurrentStateType == DataType::kFLOAT, "Replay commit supports FP32 recurrent states only.");
    check::check(
        acceptedStateNodeIds.getDataType() == DataType::kINT32, "acceptedStateNodeIds must have INT32 data type.");
    check::check(acceptLengths.getDataType() == DataType::kINT32, "acceptLengths must have INT32 data type.");

    auto const idsShape = acceptedStateNodeIds.getShape();
    int32_t const activeBatchSize = static_cast<int32_t>(idsShape[0]);
    int32_t const maxAcceptLen = static_cast<int32_t>(idsShape[1]);
    if (activeBatchSize == 0 || maxAcceptLen == 0)
    {
        return;
    }
    check::check(maxAcceptLen <= kernel::kGDN_TREE_CHUNK_MAX_ACCEPT,
        format::fmtstr("replayCommitAcceptedTreeStates: maxAcceptLen (%d) exceeds kGDN_TREE_CHUNK_MAX_ACCEPT (%d); "
                       "blockSize must not exceed %d for the chunk-form replay kernel.",
            maxAcceptLen, kernel::kGDN_TREE_CHUNK_MAX_ACCEPT, kernel::kGDN_TREE_CHUNK_MAX_ACCEPT));

    auto const intermediateShape = mIntermediateRecurrentStates[0].getShape();
    check::check(intermediateShape[0] == activeBatchSize,
        "Intermediate recurrent states must be reshaped to the active batch size before replay commit.");
    int32_t const verifyTreeSize = static_cast<int32_t>(intermediateShape[1]);

    // GDN dims: hv/dk/dv live in the config; the k-head count derives from
    // the fused-QKV conv width: convDim = 2*h*dk + hv*dv.
    int32_t const hv = mConfig.recurrentStateNumHeads;
    int32_t const dk = mConfig.recurrentStateHeadDim;
    int32_t const dv = mConfig.recurrentStateSize;
    check::check(dk == 128 && dv == 128, "Replay commit requires dk == dv == 128.");
    int32_t const h = (mConfig.convDim / dk - hv) / 2;
    check::check(h > 0 && hv % h == 0, "Replay commit: derived k-head count is inconsistent.");

    // Stash rows sit at the head of each per-batch intermediate row; the
    // stride is the (now unused) checkpoint row size. Must match the layout
    // the chunk-form verify wrote in gatedDeltaNetPlugin.
    size_t const stashBatchStrideBytes
        = static_cast<size_t>(verifyTreeSize) * hv * static_cast<size_t>(dk) * dv * sizeof(float);

    int32_t const* const idsPtr = acceptedStateNodeIds.dataPointer<int32_t>();
    int32_t const* const lensPtr = acceptLengths.dataPointer<int32_t>();
    auto const* const layerInfosForReplay = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    CUDA_CHECK(kernel::gdnTreeReplayCommitBatched(layerInfosForReplay, mConfig.numRecurrentLayers,
        stashBatchStrideBytes, idsPtr, lensPtr, activeBatchSize, maxAcceptLen, verifyTreeSize, h, hv, stream));

    // Conv states keep the checkpoint scatter (conv checkpoints remain valid
    // in chunk mode — the conv plugin path is unchanged).
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    kernel::mtpScatterAcceptedTreeConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize, verifyTreeSize,
        convElements, idsPtr, maxAcceptLen, lensPtr, stream);
}

} // namespace rt
} // namespace trt_edgellm
