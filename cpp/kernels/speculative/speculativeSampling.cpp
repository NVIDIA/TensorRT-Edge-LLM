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

#include "kernels/speculative/speculativeSampling.h"

#include "common/checkMacros.h"

#include "kernels/speculative/speculativeSamplingKernels.h"

#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kMaxSparseSupport{128};

bool isGpuTensor(rt::Tensor const& tensor) noexcept
{
    return tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.rawPointer() != nullptr;
}
} // namespace

void speculativeSparseAccept(rt::Tensor const& targetSupportProbabilities, rt::Tensor const& targetSupportIds,
    rt::Tensor const& proposalSupportProbabilities, rt::Tensor const& proposalSupportIds,
    rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, rt::Tensor* acceptedTokenIndices, cudaStream_t stream,
    rt::Tensor const* maxAcceptLengths)
{
    rt::Coords const targetShape = targetSupportProbabilities.getShape();
    rt::Coords const proposalShape = proposalSupportProbabilities.getShape();
    check::check(targetShape.getNumDims() == 3 && proposalShape.getNumDims() == 3,
        "Sparse speculative sampling requires rank-3 target and proposal distributions");
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    check::check(targetShape[0] > 0 && targetShape[0] <= kInt32Max && targetShape[1] > 0 && targetShape[1] <= kInt32Max
            && targetShape[2] > 0 && targetShape[2] <= kInt32Max && proposalShape[1] > 0
            && proposalShape[1] <= (kInt32Max - 1) / 2 && proposalShape[2] > 0 && proposalShape[2] <= kInt32Max,
        "Sparse speculative sampling dimensions exceed int32 launch limits");
    int32_t const batchSize = static_cast<int32_t>(targetShape[0]);
    int32_t const verifyLen = static_cast<int32_t>(targetShape[1]);
    int32_t const targetSupportSize = static_cast<int32_t>(targetShape[2]);
    int32_t const proposalStride = static_cast<int32_t>(proposalShape[1]);
    int32_t const proposalSupportSize = static_cast<int32_t>(proposalShape[2]);
    int32_t const verifyProposalLen = verifyLen - 1;
    check::check(batchSize > 0 && proposalShape[0] == batchSize && proposalStride > 0 && verifyProposalLen >= 0
            && verifyProposalLen <= proposalStride && targetSupportSize > 0 && targetSupportSize <= kMaxSparseSupport
            && proposalSupportSize > 0 && proposalSupportSize <= kMaxSparseSupport
            && targetSupportIds.getShape() == targetShape && proposalSupportIds.getShape() == proposalShape
            && proposalTokenIds.getShape() == rt::Coords{batchSize, proposalStride}
            && proposalLengths.getShape() == rt::Coords{batchSize}
            && acceptUniforms.getShape() == rt::Coords{batchSize, 2 * proposalStride + 1}
            && acceptedTokenIds.getShape() == rt::Coords{batchSize, verifyLen}
            && acceptLength.getShape() == rt::Coords{batchSize}
            && (acceptedTokenIndices == nullptr || acceptedTokenIndices->getShape() == acceptedTokenIds.getShape())
            && (maxAcceptLengths == nullptr || maxAcceptLengths->getShape() == acceptLength.getShape()),
        "Sparse speculative sampling tensor shapes do not match");
    check::check(targetSupportProbabilities.getDataType() == nvinfer1::DataType::kFLOAT
            && proposalSupportProbabilities.getDataType() == nvinfer1::DataType::kFLOAT
            && acceptUniforms.getDataType() == nvinfer1::DataType::kFLOAT
            && targetSupportIds.getDataType() == nvinfer1::DataType::kINT32
            && proposalSupportIds.getDataType() == nvinfer1::DataType::kINT32
            && proposalTokenIds.getDataType() == nvinfer1::DataType::kINT32
            && proposalLengths.getDataType() == nvinfer1::DataType::kINT32
            && acceptedTokenIds.getDataType() == nvinfer1::DataType::kINT32
            && acceptLength.getDataType() == nvinfer1::DataType::kINT32
            && (acceptedTokenIndices == nullptr || acceptedTokenIndices->getDataType() == nvinfer1::DataType::kINT32)
            && (maxAcceptLengths == nullptr || maxAcceptLengths->getDataType() == nvinfer1::DataType::kINT32),
        "Sparse speculative sampling tensor dtypes do not match");
    check::check(isGpuTensor(targetSupportProbabilities) && isGpuTensor(targetSupportIds)
            && isGpuTensor(proposalSupportProbabilities) && isGpuTensor(proposalSupportIds)
            && isGpuTensor(proposalTokenIds) && isGpuTensor(proposalLengths) && isGpuTensor(acceptUniforms)
            && isGpuTensor(acceptedTokenIds) && isGpuTensor(acceptLength)
            && (acceptedTokenIndices == nullptr || isGpuTensor(*acceptedTokenIndices))
            && (maxAcceptLengths == nullptr || isGpuTensor(*maxAcceptLengths)),
        "Sparse speculative sampling requires GPU tensors");
    detail::launchSparseAccept(targetSupportProbabilities.dataPointer<float>(), targetSupportIds.dataPointer<int32_t>(),
        proposalSupportProbabilities.dataPointer<float>(), proposalSupportIds.dataPointer<int32_t>(),
        proposalTokenIds.dataPointer<int32_t>(), proposalLengths.dataPointer<int32_t>(),
        acceptUniforms.dataPointer<float>(), acceptedTokenIds.dataPointer<int32_t>(),
        acceptLength.dataPointer<int32_t>(),
        acceptedTokenIndices == nullptr ? nullptr : acceptedTokenIndices->dataPointer<int32_t>(), batchSize,
        proposalStride, verifyProposalLen, targetSupportSize, proposalSupportSize, stream,
        maxAcceptLengths == nullptr ? nullptr : maxAcceptLengths->dataPointer<int32_t>());
}

void speculativeDenseTargetAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& proposalSupportProbabilities,
    rt::Tensor const& proposalSupportIds, rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalLengths,
    rt::Tensor const& acceptUniforms, rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength,
    rt::Tensor* acceptedTokenIndices, cudaStream_t stream, rt::Tensor const* maxAcceptLengths)
{
    rt::Coords const targetShape = targetProbabilities.getShape();
    rt::Coords const proposalShape = proposalSupportProbabilities.getShape();
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    check::check(targetShape.getNumDims() == 3 && proposalShape.getNumDims() == 3 && targetShape[0] > 0
            && targetShape[0] <= kInt32Max && targetShape[1] > 0 && targetShape[1] <= kInt32Max && targetShape[2] > 0
            && targetShape[2] <= kInt32Max && proposalShape[1] > 0 && proposalShape[1] <= (kInt32Max - 1) / 2
            && proposalShape[2] > 0 && proposalShape[2] <= kMaxSparseSupport,
        "Dense-target speculative sampling dimensions exceed launch limits");
    int32_t const batchSize = static_cast<int32_t>(targetShape[0]);
    int32_t const verifyLen = static_cast<int32_t>(targetShape[1]);
    int32_t const vocabSize = static_cast<int32_t>(targetShape[2]);
    int32_t const proposalStride = static_cast<int32_t>(proposalShape[1]);
    int32_t const proposalSupportSize = static_cast<int32_t>(proposalShape[2]);
    int32_t const verifyProposalLen = verifyLen - 1;
    check::check(proposalShape[0] == batchSize && verifyProposalLen >= 0 && verifyProposalLen <= proposalStride
            && proposalSupportIds.getShape() == proposalShape
            && proposalTokenIds.getShape() == rt::Coords{batchSize, proposalStride}
            && proposalLengths.getShape() == rt::Coords{batchSize}
            && acceptUniforms.getShape() == rt::Coords{batchSize, 2 * proposalStride + 1}
            && acceptedTokenIds.getShape() == rt::Coords{batchSize, verifyLen}
            && acceptLength.getShape() == rt::Coords{batchSize}
            && (acceptedTokenIndices == nullptr || acceptedTokenIndices->getShape() == acceptedTokenIds.getShape())
            && (maxAcceptLengths == nullptr || maxAcceptLengths->getShape() == acceptLength.getShape()),
        "Dense-target speculative sampling tensor shapes do not match");
    check::check(targetProbabilities.getDataType() == nvinfer1::DataType::kFLOAT
            && proposalSupportProbabilities.getDataType() == nvinfer1::DataType::kFLOAT
            && acceptUniforms.getDataType() == nvinfer1::DataType::kFLOAT
            && proposalSupportIds.getDataType() == nvinfer1::DataType::kINT32
            && proposalTokenIds.getDataType() == nvinfer1::DataType::kINT32
            && proposalLengths.getDataType() == nvinfer1::DataType::kINT32
            && acceptedTokenIds.getDataType() == nvinfer1::DataType::kINT32
            && acceptLength.getDataType() == nvinfer1::DataType::kINT32
            && (acceptedTokenIndices == nullptr || acceptedTokenIndices->getDataType() == nvinfer1::DataType::kINT32)
            && (maxAcceptLengths == nullptr || maxAcceptLengths->getDataType() == nvinfer1::DataType::kINT32),
        "Dense-target speculative sampling tensor dtypes do not match");
    check::check(isGpuTensor(targetProbabilities) && isGpuTensor(proposalSupportProbabilities)
            && isGpuTensor(proposalSupportIds) && isGpuTensor(proposalTokenIds) && isGpuTensor(proposalLengths)
            && isGpuTensor(acceptUniforms) && isGpuTensor(acceptedTokenIds) && isGpuTensor(acceptLength)
            && (acceptedTokenIndices == nullptr || isGpuTensor(*acceptedTokenIndices))
            && (maxAcceptLengths == nullptr || isGpuTensor(*maxAcceptLengths)),
        "Dense-target speculative sampling requires GPU tensors");
    detail::launchDenseTargetAccept(targetProbabilities.dataPointer<float>(),
        proposalSupportProbabilities.dataPointer<float>(), proposalSupportIds.dataPointer<int32_t>(),
        proposalTokenIds.dataPointer<int32_t>(), proposalLengths.dataPointer<int32_t>(),
        acceptUniforms.dataPointer<float>(), acceptedTokenIds.dataPointer<int32_t>(),
        acceptLength.dataPointer<int32_t>(),
        acceptedTokenIndices == nullptr ? nullptr : acceptedTokenIndices->dataPointer<int32_t>(), batchSize,
        proposalStride, verifyProposalLen, vocabSize, proposalSupportSize, stream,
        maxAcceptLengths == nullptr ? nullptr : maxAcceptLengths->dataPointer<int32_t>());
}

void speculativeNormalizeTopKTopP(
    rt::Tensor const& topKValues, rt::Tensor& probabilities, float temperature, float topP, cudaStream_t stream)
{
    rt::Coords const shape = topKValues.getShape();
    check::check(shape.getNumDims() == 2 && shape[0] > 0 && shape[0] <= std::numeric_limits<int32_t>::max()
            && shape[1] > 0 && shape[1] <= kMaxSparseSupport && probabilities.getShape() == shape
            && topKValues.getDataType() == nvinfer1::DataType::kFLOAT
            && probabilities.getDataType() == nvinfer1::DataType::kFLOAT && isGpuTensor(topKValues)
            && isGpuTensor(probabilities),
        "Sparse top-k/top-p normalization tensor contract does not match");
    detail::launchNormalizeTopKTopP(topKValues.dataPointer<float>(), probabilities.dataPointer<float>(),
        static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[1]), temperature, topP, stream);
}

} // namespace kernel
} // namespace trt_edgellm
