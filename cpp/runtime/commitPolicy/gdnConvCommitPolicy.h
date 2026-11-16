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

//! GDN commit: snapshot the per-token full recurrent state, then advance the persistent state along the
//! accepted path by replaying the per-node stash the verify pass wrote at the head of each layer's
//! snapshot buffer. Convolution state commits by scatter.
class GdnConvCommitPolicy final : public RecurrentCommitPolicy
{
public:
    GdnConvCommitPolicy(
        Config config, std::vector<rt::Tensor>& liveRecurrent, std::vector<rt::Tensor>& liveConv, cudaStream_t stream);

    bool hasIntermediateRecurrentStates() const noexcept override;
    bool usesReplay() const noexcept override;

    void commitLinear(rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, cudaStream_t stream) override;
    void commitTree(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices,
        int32_t verifySize, cudaStream_t stream) override;

    //! Verify-pass recurrent snapshot for one layer, shaped by the last `reshapeIntermediate`:
    //! [activeBatch, seqLen, H, headDim, stateSize].
    rt::Tensor& intermediateRecurrentState(int32_t recurrentLayerIdx);

protected:
    void reshapeRecurrentSide(int32_t activeBatchSize, int32_t seqLen) override;

private:
    //! Scatter the accepted per-node convolution snapshot; shared by both tree-commit branches.
    void scatterAcceptedTreeConvStates(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths,
        rt::Tensor const& stateIndices, int32_t verifyTreeSize, int32_t maxAcceptLen, cudaStream_t stream);

    std::vector<rt::Tensor> mIntermediateRecurrentStates;
};

} // namespace rt
} // namespace trt_edgellm
