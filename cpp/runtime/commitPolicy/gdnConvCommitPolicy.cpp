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

#include "runtime/commitPolicy/gdnConvCommitPolicy.h"
#include "common/checkMacros.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

namespace trt_edgellm
{
namespace rt
{

GdnConvCommitPolicy::GdnConvCommitPolicy(
    Config config, std::vector<rt::Tensor>& liveRecurrent, std::vector<rt::Tensor>& liveConv, cudaStream_t stream)
    : RecurrentCommitPolicy(config, liveRecurrent, liveConv, "GdnConvCommitPolicy", stream)
{
    if (mConfig.numRecurrentLayers == 0 || mConfig.maxIntermediateSeqLen == 0)
    {
        return;
    }

    size_t const recurrentElemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    rt::Coords const stashShape = kernel::gdnTreeVerifyStashShape(mConfig.recurrentStateType == DataType::kFLOAT,
        mConfig.convDim, mConfig.maxBatchSize, mConfig.maxIntermediateSeqLen, mConfig.recurrentStateNumHeads,
        mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize);
    mIntermediateRecurrentStates.reserve(mConfig.numRecurrentLayers);
    std::vector<void*> recurrentSrcPointers(mConfig.numRecurrentLayers);
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        size_t const intermRecBytes = static_cast<size_t>(stashShape.volume()) * recurrentElemSize;
        mIntermediateRecurrentStates.emplace_back(rt::Tensor(stashShape, DeviceType::kGPU, mConfig.recurrentStateType,
            "GdnConvCommitPolicy::intermediateRecurrentState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mIntermediateRecurrentStates.back().rawPointer(), 0, intermRecBytes, stream));
        recurrentSrcPointers[i] = mIntermediateRecurrentStates.back().rawPointer();
    }

    buildMtpLayerInfos(recurrentSrcPointers, stream);
}

bool GdnConvCommitPolicy::hasIntermediateRecurrentStates() const noexcept
{
    return !mIntermediateRecurrentStates.empty();
}

bool GdnConvCommitPolicy::usesReplay() const noexcept
{
    return false;
}

void GdnConvCommitPolicy::reshapeRecurrentSide(int32_t activeBatchSize, int32_t seqLen)
{
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        auto& stash = mIntermediateRecurrentStates[i];
        rt::Coords activeShape = stash.getShape();
        activeShape[0] = activeBatchSize;
        activeShape[1] = seqLen;
        check::check(stash.reshape(activeShape), "Intermediate recurrent state reshape failed");
    }
}

void GdnConvCommitPolicy::commitLinear(
    rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, cudaStream_t stream)
{
    if (mIntermediateConvStates.empty())
    {
        return;
    }

    int32_t const verifyTokenCount = static_cast<int32_t>(mIntermediateRecurrentStates[0].getShape()[1]);
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    int32_t const activeBatchSize = static_cast<int32_t>(acceptLengths.getShape()[0]);
    check::check(stateIndices.getDeviceType() == DeviceType::kGPU && stateIndices.getDataType() == DataType::kINT32
            && stateIndices.getShape().getNumDims() == 1 && stateIndices.getShape()[0] == activeBatchSize,
        "stateIndices must be a GPU INT32 tensor with shape [activeBatchSize].");
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    int32_t const* const acceptPtr = acceptLengths.dataPointer<int32_t>();

    // Advance the persistent recurrent state along the accepted chain prefix by replaying the per-node
    // stash the chunk-form verify wrote at the head of each layer's snapshot buffer; a null accepted-index
    // list selects the in-order chain path.
    check::check(mConfig.recurrentStateType == DataType::kFLOAT,
        "GDN recurrent replay commit supports FP32 recurrent states only.");
    check::check(verifyTokenCount <= kernel::kGDN_TREE_CHUNK_MAX_ACCEPT,
        format::fmtstr("GdnConvCommitPolicy::commitLinear: verify token count (%d) exceeds "
                       "kGDN_TREE_CHUNK_MAX_ACCEPT (%d).",
            verifyTokenCount, kernel::kGDN_TREE_CHUNK_MAX_ACCEPT));

    // GDN dims: hv/dk/dv live in the config; the k-head count derives from the fused-QKV conv width:
    // convDim = 2*h*dk + hv*dv.
    int32_t const hv = mConfig.recurrentStateNumHeads;
    int32_t const dk = mConfig.recurrentStateHeadDim;
    int32_t const dv = mConfig.recurrentStateSize;
    check::check(dk == 128 && dv == 128, "GDN recurrent replay commit requires dk == dv == 128.");
    int32_t const h = (mConfig.convDim / dk - hv) / 2;
    check::check(h > 0 && hv % h == 0, "GDN recurrent replay commit: derived k-head count is inconsistent.");
    size_t const stashBatchStrideBytes = kernel::gdnTreeVerifyRowBytes(h, hv, verifyTokenCount);
    CUDA_CHECK(kernel::gdnTreeReplayCommitBatched(layerInfos, mConfig.numRecurrentLayers, stashBatchStrideBytes,
        nullptr, acceptPtr, stateIndices.dataPointer<int32_t>(), activeBatchSize, mConfig.maxBatchSize,
        verifyTokenCount, verifyTokenCount, h, hv, stream));

    kernel::mtpScatterConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize, mConfig.maxBatchSize,
        verifyTokenCount, convElements, acceptPtr, stateIndices.dataPointer<int32_t>(), stream);
}

void GdnConvCommitPolicy::commitTree(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths,
    rt::Tensor const& stateIndices, int32_t /*verifySize*/, cudaStream_t stream)
{
    if (mIntermediateRecurrentStates.empty())
    {
        return;
    }

    check::check(!mIntermediateConvStates.empty(), "Intermediate conv states are not allocated.");
    check::check(mConfig.recurrentStateType == DataType::kFLOAT,
        "Accepted tree recurrent state commit supports FP32 recurrent states only.");
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

    auto const intermediateShape = mIntermediateRecurrentStates[0].getShape();
    check::check(intermediateShape[0] == activeBatchSize,
        "Intermediate recurrent states must be reshaped to the active batch size before tree commit.");
    int32_t const verifyTreeSize = static_cast<int32_t>(intermediateShape[1]);

    check::check(maxAcceptLen <= kernel::kGDN_TREE_CHUNK_MAX_ACCEPT,
        format::fmtstr("GdnConvCommitPolicy::commitTree: maxAcceptLen (%d) exceeds kGDN_TREE_CHUNK_MAX_ACCEPT "
                       "(%d); blockSize must not exceed %d for the chunk-form replay kernel.",
            maxAcceptLen, kernel::kGDN_TREE_CHUNK_MAX_ACCEPT, kernel::kGDN_TREE_CHUNK_MAX_ACCEPT));

    // The verify pass stashed the per-node replay quantities at the head of each layer's snapshot buffer;
    // advance the persistent state along the accepted path from there.
    // GDN dims: hv/dk/dv live in the config; the k-head count derives from the fused-QKV conv width:
    // convDim = 2*h*dk + hv*dv.
    int32_t const hv = mConfig.recurrentStateNumHeads;
    int32_t const dk = mConfig.recurrentStateHeadDim;
    int32_t const dv = mConfig.recurrentStateSize;
    check::check(dk == 128 && dv == 128, "GDN tree replay commit requires dk == dv == 128.");
    int32_t const h = (mConfig.convDim / dk - hv) / 2;
    check::check(h > 0 && hv % h == 0, "GDN tree replay commit: derived k-head count is inconsistent.");

    // Must match the stash row stride the verify pass wrote in gatedDeltaNetPlugin.
    size_t const stashBatchStrideBytes = kernel::gdnTreeVerifyRowBytes(h, hv, verifyTreeSize);

    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    CUDA_CHECK(kernel::gdnTreeReplayCommitBatched(layerInfos, mConfig.numRecurrentLayers, stashBatchStrideBytes,
        acceptedNodeIds.dataPointer<int32_t>(), acceptLengths.dataPointer<int32_t>(),
        stateIndices.dataPointer<int32_t>(), activeBatchSize, mConfig.maxBatchSize, maxAcceptLen, verifyTreeSize, h, hv,
        stream));

    scatterAcceptedTreeConvStates(acceptedNodeIds, acceptLengths, stateIndices, verifyTreeSize, maxAcceptLen, stream);
}

void GdnConvCommitPolicy::scatterAcceptedTreeConvStates(rt::Tensor const& acceptedNodeIds,
    rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, int32_t verifyTreeSize, int32_t maxAcceptLen,
    cudaStream_t stream)
{
    int32_t const convElements = mConfig.convDim * mConfig.convKernel;
    int32_t const activeBatchSize = static_cast<int32_t>(acceptedNodeIds.getShape()[0]);
    auto const* const layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceMtpLayerInfos.rawPointer());
    kernel::mtpScatterAcceptedTreeConvStates(layerInfos, mConfig.numRecurrentLayers, activeBatchSize,
        mConfig.maxBatchSize, verifyTreeSize, convElements, acceptedNodeIds.dataPointer<int32_t>(), maxAcceptLen,
        acceptLengths.dataPointer<int32_t>(), stateIndices.dataPointer<int32_t>(), stream);
}

rt::Tensor& GdnConvCommitPolicy::intermediateRecurrentState(int32_t recurrentLayerIdx)
{
    return mIntermediateRecurrentStates[recurrentLayerIdx];
}

} // namespace rt
} // namespace trt_edgellm
