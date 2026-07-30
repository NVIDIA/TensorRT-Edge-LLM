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
    int32_t const h = kernel::gdnTreeChunkNumKeyHeads(
        mConfig.convDim, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize);
    int64_t compactElements = 0;
    if (mConfig.maxIntermediateSeqLen > 0)
    {
        check::check(mConfig.maxIntermediateSeqLen <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
            "Speculative verify length exceeds the compact GDN replay-buffer limit.");
        check::check(mConfig.recurrentStateType == DataType::kFLOAT,
            "Compact GDN replay buffers require FP32 recurrent state storage.");
        check::check(
            mConfig.convStateType == DataType::kHALF, "Compact GDN replay requires FP16 conv checkpoint storage.");
        check::check(mConfig.recurrentStateHeadDim == 128 && mConfig.recurrentStateSize == 128,
            "Compact GDN replay requires recurrent head and state dimensions of 128.");
        check::check(h > 0 && mConfig.recurrentStateNumHeads % h == 0,
            "Compact GDN replay: fused QKV convolution dimensions are inconsistent.");
        compactElements = kernel::gdnTreeChunkBufferElements(h, mConfig.recurrentStateNumHeads);
    }

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

    // Allocate spec-verify intermediate state buffers when enabled (maxIntermediateSeqLen > 0).
    if (mConfig.maxIntermediateSeqLen > 0)
    {
        mIntermediateRecurrentStates.reserve(mConfig.numRecurrentLayers);
        mIntermediateConvStates.reserve(mConfig.numRecurrentLayers);

        for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
        {
            int64_t const intermRecVolume = static_cast<int64_t>(mConfig.maxBatchSize) * compactElements;
            size_t const intermRecBytes = static_cast<size_t>(intermRecVolume) * recurrentElemSize;

            mIntermediateRecurrentStates.emplace_back(
                rt::Tensor({mConfig.maxBatchSize, compactElements}, DeviceType::kGPU, mConfig.recurrentStateType,
                    "MambaCacheManager::intermediateRecurrentState_" + std::to_string(i)));
            CUDA_CHECK(cudaMemsetAsync(mIntermediateRecurrentStates.back().rawPointer(), 0, intermRecBytes, stream));

            int64_t const intermConvVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.maxIntermediateSeqLen
                * mConfig.convDim * mConfig.convKernel;
            size_t const intermConvBytes = static_cast<size_t>(intermConvVolume) * convElemSize;

            mIntermediateConvStates.emplace_back(
                rt::Tensor({mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.convDim, mConfig.convKernel},
                    DeviceType::kGPU, mConfig.convStateType,
                    "MambaCacheManager::intermediateConvState_" + std::to_string(i)));
            CUDA_CHECK(cudaMemsetAsync(mIntermediateConvStates.back().rawPointer(), 0, intermConvBytes, stream));

            totalBytes += intermRecBytes + intermConvBytes;
        }

        // Build the MtpLayerInfo array for batched spec-verify state scatter; pointers are stable, so
        // upload runs once.
        std::vector<kernel::MtpLayerInfo> hostInfos(mConfig.numRecurrentLayers);
        for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
        {
            hostInfos[i] = {
                mRecurrentStates[i].rawPointer(),
                mIntermediateRecurrentStates[i].rawPointer(),
                mConvStates[i].rawPointer(),
                mIntermediateConvStates[i].rawPointer(),
            };
        }
        size_t const infoBytes = hostInfos.size() * sizeof(kernel::MtpLayerInfo);
        mDeviceMtpLayerInfos = rt::Tensor(
            {static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8, "MambaCacheManager::mtpLayerInfos");
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceMtpLayerInfos.rawPointer(), hostInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));
        // Sync before hostInfos goes out of scope.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        LOG_INFO(
            "MambaCacheManager: allocated compact spec-verify replay buffers (%d layers, maxSeqLen=%d, %.2f MB "
            "per recurrent layer/batch row)",
            mConfig.numRecurrentLayers, mConfig.maxIntermediateSeqLen,
            static_cast<float>(kernel::gdnTreeChunkBufferBytes(h, mConfig.recurrentStateNumHeads))
                / (1024.0f * 1024.0f));
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
    mActiveIntermediateSeqLen = other.mActiveIntermediateSeqLen;
    mDeviceMtpLayerInfos = std::move(other.mDeviceMtpLayerInfos);

    other.mConfig = Config{};
    other.mActiveIntermediateSeqLen = 0;
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
        mActiveIntermediateSeqLen = other.mActiveIntermediateSeqLen;
        mDeviceMtpLayerInfos = std::move(other.mDeviceMtpLayerInfos);

        other.mConfig = Config{};
        other.mActiveIntermediateSeqLen = 0;
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
    if (mIntermediateRecurrentStates.empty())
    {
        return;
    }
    check::check(activeBatchSize > 0 && activeBatchSize <= mConfig.maxBatchSize,
        "Intermediate state active batch size is out of range.");
    check::check(
        seqLen > 0 && seqLen <= mConfig.maxIntermediateSeqLen, "Intermediate state sequence length is out of range.");
    int32_t const h = kernel::gdnTreeChunkNumKeyHeads(
        mConfig.convDim, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize);
    int64_t const compactElements = kernel::gdnTreeChunkBufferElements(h, mConfig.recurrentStateNumHeads);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        check::check(mIntermediateRecurrentStates[i].reshape({activeBatchSize, compactElements}),
            "Intermediate recurrent state reshape failed");
        check::check(mIntermediateConvStates[i].reshape({activeBatchSize, seqLen, mConfig.convDim, mConfig.convKernel}),
            "Intermediate conv state reshape failed");
    }
    mActiveIntermediateSeqLen = seqLen;
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
    return !mIntermediateRecurrentStates.empty();
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

void MambaCacheManager::replayCommitAcceptedTreeStates(
    rt::Tensor const& acceptedStateNodeIds, rt::Tensor const& acceptLengths, cudaStream_t stream)
{
    if (mIntermediateRecurrentStates.empty())
    {
        return;
    }

    check::check(!mIntermediateConvStates.empty(), "Intermediate conv states are not allocated.");
    check::check(mConfig.recurrentStateType == DataType::kFLOAT, "Replay commit supports FP32 recurrent states only.");
    check::check(mConfig.convStateType == DataType::kHALF, "Replay commit supports FP16 conv states only.");
    check::check(
        acceptedStateNodeIds.getDeviceType() == DeviceType::kGPU, "acceptedStateNodeIds must be a GPU tensor.");
    check::check(acceptLengths.getDeviceType() == DeviceType::kGPU, "acceptLengths must be a GPU tensor.");
    check::check(
        acceptedStateNodeIds.getDataType() == DataType::kINT32, "acceptedStateNodeIds must have INT32 data type.");
    check::check(acceptLengths.getDataType() == DataType::kINT32, "acceptLengths must have INT32 data type.");

    auto const idsShape = acceptedStateNodeIds.getShape();
    auto const lengthsShape = acceptLengths.getShape();
    check::check(idsShape.getNumDims() == 2, "acceptedStateNodeIds must have shape [B, maxAcceptLen].");
    check::check(lengthsShape.getNumDims() == 1, "acceptLengths must have shape [B].");
    int32_t const activeBatchSize = static_cast<int32_t>(idsShape[0]);
    int32_t const maxAcceptLen = static_cast<int32_t>(idsShape[1]);
    check::check(lengthsShape[0] == activeBatchSize, "acceptedStateNodeIds and acceptLengths batch sizes differ.");
    if (activeBatchSize == 0 || maxAcceptLen == 0)
    {
        return;
    }

    auto const intermediateShape = mIntermediateRecurrentStates[0].getShape();
    check::check(intermediateShape.getNumDims() == 2 && intermediateShape[0] == activeBatchSize,
        "Intermediate recurrent states must be reshaped to the active batch size before replay commit.");
    check::check(mActiveIntermediateSeqLen > 0,
        "Intermediate states must be reshaped to the active verify length before replay commit.");
    int32_t const verifyTreeSize = mActiveIntermediateSeqLen;
    auto const convShape = mIntermediateConvStates[0].getShape();
    check::check(convShape.getNumDims() == 4 && convShape[0] == activeBatchSize && convShape[1] == verifyTreeSize,
        "Intermediate conv states must be reshaped to the active batch and verify length before replay commit.");

    // GDN dims: hv/dk/dv live in the config; the k-head count derives from
    // the fused-QKV conv width: convDim = 2*h*dk + hv*dv.
    int32_t const hv = mConfig.recurrentStateNumHeads;
    int32_t const dk = mConfig.recurrentStateHeadDim;
    int32_t const dv = mConfig.recurrentStateSize;
    check::check(dk == 128 && dv == 128, "Replay commit requires dk == dv == 128.");
    int32_t const h = kernel::gdnTreeChunkNumKeyHeads(mConfig.convDim, hv, dk, dv);
    check::check(h > 0 && hv % h == 0, "Replay commit: derived k-head count is inconsistent.");

    size_t const stashBatchStrideBytes = kernel::gdnTreeChunkBufferBytes(h, hv);

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
