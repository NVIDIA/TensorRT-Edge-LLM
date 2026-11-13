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

#pragma once
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/speculative/ddtreeKernels.h"
using namespace trt_edgellm;
struct DDTreeBuildScratch
{
    rt::Tensor draftLogits;
    rt::Tensor lastAcceptedTokens;
    rt::Tensor baseKVCacheLengths;

    rt::Tensor treeTokenIds;
    rt::Tensor treeDepths;
    rt::Tensor treeParentIds;
    rt::Tensor treeNodeScores;
    rt::Tensor validCounts;
    rt::Tensor verifyTokenIds;
    rt::Tensor specDecodePositionIds;
    rt::Tensor packedAttentionMask;
    rt::Tensor verifyTreeMask;
    rt::Tensor contextLengths;
    rt::Tensor selectTokenIndices;
    rt::Tensor workspace;
};

//! Allocate all DDTree build inputs, outputs, and the exact production workspace size.
inline DDTreeBuildScratch allocateDDTreeBuildScratch(
    int32_t batchSize, int32_t blockSize, int32_t draftVocabSize, int32_t verifySize, int32_t candidateTopK)
{
    ELLM_CHECK(batchSize > 0 && blockSize > 1 && draftVocabSize > 0 && verifySize > 0 && candidateTopK > 0,
        "DDTree build requires positive dimensions and candidateTopK, with blockSize > 1");
    ELLM_CHECK(verifySize <= kernel::kDDTreeMaxVerifySize,
        "DDTree verifySize exceeds the production kernel limit of " + std::to_string(kernel::kDDTreeMaxVerifySize));
    ELLM_CHECK(candidateTopK <= kernel::kDDTreeMaxCandidateTopK,
        "DDTree candidateTopK exceeds the production kernel limit of "
            + std::to_string(kernel::kDDTreeMaxCandidateTopK));
    ELLM_CHECK(candidateTopK <= draftVocabSize, "DDTree candidateTopK must not exceed draftVocabSize");

    DDTreeBuildScratch scratch;
    scratch.draftLogits = rt::Tensor({batchSize, blockSize, draftVocabSize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DFlashBench::ddtreeDraftLogits");
    scratch.lastAcceptedTokens = rt::Tensor(
        {batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::lastAcceptedTokens");
    scratch.baseKVCacheLengths = rt::Tensor(
        {batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::baseKVCacheLengths");

    rt::Coords const nodeShape{batchSize, verifySize};
    scratch.treeTokenIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::treeTokenIds");
    scratch.treeDepths
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::treeDepths");
    scratch.treeParentIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::treeParentIds");
    scratch.treeNodeScores
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlashBench::treeNodeScores");
    scratch.validCounts
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::validCounts");
    scratch.verifyTokenIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::verifyTokenIds");
    scratch.specDecodePositionIds = rt::Tensor(
        nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::specDecodePositionIds");
    scratch.packedAttentionMask = rt::Tensor({batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::packedAttentionMask");
    scratch.verifyTreeMask = rt::Tensor({batchSize, verifySize, verifySize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kINT8, "DFlashBench::verifyTreeMask");
    scratch.contextLengths
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashBench::contextLengths");
    scratch.selectTokenIndices
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT64, "DFlashBench::selectTokenIndices");

    size_t const workspaceSize
        = kernel::getDDTreeBuildWorkspaceSize(batchSize, blockSize, verifySize, draftVocabSize, candidateTopK);
    ELLM_CHECK(workspaceSize > 0, "DDTree build workspace size must be positive");
    scratch.workspace = rt::Tensor({static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "DFlashBench::ddtreeWorkspace");
    return scratch;
}
