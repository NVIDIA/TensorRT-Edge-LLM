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

#include "common/checkMacros.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/speculativeSampling.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(ForceAcceptanceTest, ProbabilityRoutesUseForcedDraftPrefixAndTargetBonus)
{
    // Given a naturally rejected draft, when acceptance is forced to include it, then sparse-target,
    // dense-target and DSpark probability routes all emit the draft followed by a target bonus.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    rt::Tensor sparseTarget({1, 2, 2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor sparseTargetIds({1, 2, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor sparseProposal({1, 1, 2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor sparseProposalIds({1, 1, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor denseTarget({1, 2, 2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor denseProposal({1, 1, 2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor tokens({1, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({1, 3}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor output({1, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor rows({1, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<float>(sparseTarget, {0.75F, 0.25F, 0.25F, 0.75F});
    copyHostToDevice<int32_t>(sparseTargetIds, {10, 11, 20, 21});
    copyHostToDevice<float>(sparseProposal, {0.0F, 1.0F});
    copyHostToDevice<int32_t>(sparseProposalIds, {10, 11});
    copyHostToDevice<float>(denseTarget, {0.75F, 0.25F, 0.25F, 0.75F});
    copyHostToDevice<float>(denseProposal, {0.0F, 1.0F});
    copyHostToDevice<int32_t>(tokens, {1});
    copyHostToDevice<int32_t>(proposalLengths, {1});
    copyHostToDevice<float>(uniforms, {0.9F, 0.1F, 0.6F});

    kernel::speculativeSparseAccept(sparseTarget, sparseTargetIds, sparseProposal, sparseProposalIds, tokens,
        proposalLengths, uniforms, output, lengths, &rows, stream, nullptr, 2);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(output), (std::vector<int32_t>{1, 21}));

    kernel::speculativeDenseTargetAccept(denseTarget, sparseProposal, sparseProposalIds, tokens, proposalLengths,
        uniforms, output, lengths, &rows, stream, nullptr, 2);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(output), (std::vector<int32_t>{1, 1}));

    kernel::dsparkProbabilisticAccept(
        denseTarget, denseProposal, tokens, proposalLengths, uniforms, output, lengths, 1, 1, 1, 2, stream, 2);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(output), (std::vector<int32_t>{1, 1}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
