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

#pragma once

#include "runtime/commitPolicy/recurrentCommitPolicy.h"

namespace trt_edgellm
{
namespace rt
{

//! Mamba2 commit: reconstruct the accepted recurrent state by replaying the SSD scan from the
//! committed state over the per-token dA/x/B/dt stash the verify pass wrote. Owns the replay stash
//! and the B/C group count the stash extent needs.
class Mamba2CommitPolicy final : public RecurrentCommitPolicy
{
public:
    Mamba2CommitPolicy(
        Config config, std::vector<rt::Tensor>& liveRecurrent, std::vector<rt::Tensor>& liveConv, cudaStream_t stream);

    bool hasIntermediateRecurrentStates() const noexcept override;
    bool usesReplay() const noexcept override;

    void commitLinear(rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, cudaStream_t stream) override;
    void commitTree(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices,
        int32_t verifySize, cudaStream_t stream) override;

    //! Replay-stash side buffers for one layer, shaped by the last `reshapeIntermediate` (all FP32):
    //!   dA/dt [activeBatch, seqLen, H], x in u [activeBatch, seqLen, H, headDim],
    //!   B [activeBatch, seqLen, numGroups, stateSize].
    rt::Tensor& replayDaState(int32_t recurrentLayerIdx);
    rt::Tensor& replayUState(int32_t recurrentLayerIdx);
    rt::Tensor& replayBState(int32_t recurrentLayerIdx);
    rt::Tensor& replayDtState(int32_t recurrentLayerIdx);

protected:
    void reshapeRecurrentSide(int32_t activeBatchSize, int32_t seqLen) override;

private:
    std::vector<rt::Tensor> mReplayDaStates;
    std::vector<rt::Tensor> mReplayUStates;
    std::vector<rt::Tensor> mReplayBStates;
    std::vector<rt::Tensor> mReplayDtStates;
    rt::Tensor mDeviceReplayLayerInfos;
};

} // namespace rt
} // namespace trt_edgellm
