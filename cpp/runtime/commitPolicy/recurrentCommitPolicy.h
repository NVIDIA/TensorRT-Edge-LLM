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

#include "common/tensor.h"
#include "runtime/recurrentStateTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! How one recurrent-mixer kind commits its speculative-verify state. A layer's mixer lineage fixes
//! both the side buffers the verify pass writes and the math that folds the accepted path back into
//! the persistent slab, so the commit path is a property of the kind, not a runtime flag:
//!   - mamba2 stashes the per-token replay inputs (dA/x/B/dt) and reconstructs the accepted state by
//!     replaying the SSD scan from the committed state;
//!   - gdn snapshots the per-token full recurrent state and commits the accepted endpoint by scatter,
//!     except for tree sizes the chunk-form verify handles, which commit by replaying the stash.
//! Both kinds snapshot the convolution state per token and commit it by scatter.
//!
//! The per-kind commit lives in its own translation unit (mamba2CommitPolicy / gdnConvCommitPolicy);
//! this file holds only the shared base and the convolution side-buffer machinery both reuse, so a
//! future mixer or attention-side commit adds a sibling file without touching the others.
//!
//! The policy owns the spec-verify side buffers and the device pointer bundles the commit kernels
//! read; it does not own the persistent recurrent/conv slabs, which it reaches through references
//! supplied at construction and whose lifetime the caller guarantees for the policy's lifetime.
class RecurrentCommitPolicy
{
public:
    //! Geometry of the recurrent axis this policy commits. `maxIntermediateSeqLen` is the verify-pass
    //! token capacity; a zero value means no side buffers are allocated and every commit is a no-op.
    struct Config
    {
        int32_t numRecurrentLayers{0};
        int32_t maxBatchSize{};
        int32_t recurrentStateNumHeads{0};
        int32_t recurrentStateHeadDim{0};
        int32_t recurrentStateSize{0};
        int32_t convDim{0};
        int32_t convKernel{0};
        int32_t maxIntermediateSeqLen{0};
        trt_edgellm::DataType recurrentStateType{trt_edgellm::DataType::kHALF};
        trt_edgellm::DataType convStateType{trt_edgellm::DataType::kHALF};
        //! B/C group count. Consumed only by the mamba2 replay stash (the replay-B extent); gdn leaves it 0.
        int32_t recurrentStateNumGroups{0};
    };

    virtual ~RecurrentCommitPolicy() = default;

    RecurrentCommitPolicy(RecurrentCommitPolicy const&) = delete;
    RecurrentCommitPolicy& operator=(RecurrentCommitPolicy const&) = delete;
    RecurrentCommitPolicy(RecurrentCommitPolicy&&) = delete;
    RecurrentCommitPolicy& operator=(RecurrentCommitPolicy&&) = delete;

    //! @brief Number of recurrent layers this policy commits.
    int32_t numLayers() const noexcept;

    //! @brief Whether verify-pass recurrent side buffers were allocated (spec-verify enabled).
    virtual bool hasIntermediateRecurrentStates() const noexcept = 0;

    //! @brief Whether verify-pass convolution side buffers were allocated (spec-verify enabled).
    bool hasIntermediateConvStates() const noexcept;

    //! @brief Whether the recurrent commit reconstructs by replay (mamba2) rather than scatter (gdn).
    virtual bool usesReplay() const noexcept = 0;

    //! Reshape the verify-pass side buffers from their [maxBatch, maxIntermediateSeqLen, ...] allocation
    //! to the [activeBatchSize, seqLen, ...] the engine wrote, so stride-based reads address correctly.
    //! No-op when side buffers are not allocated.
    void reshapeIntermediate(int32_t activeBatchSize, int32_t seqLen);

    //! Verify-pass convolution side buffer for one layer, shaped by the last `reshapeIntermediate`.
    rt::Tensor& intermediateConvState(int32_t recurrentLayerIdx);

    //! Commit the single accepted recurrent/conv state per sequence after linear (chain) verification.
    //! @param acceptLengths [activeBatchSize] INT32 GPU accepted-token count per sequence.
    //! @param stateIndices  [activeBatchSize] INT32 GPU active-row to resident-slot map.
    virtual void commitLinear(rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices, cudaStream_t stream) = 0;

    //! Commit the accepted recurrent/conv state per sequence after tree verification.
    //! @param acceptedNodeIds [activeBatchSize, maxAcceptLen] INT32 GPU verify-node id per accepted token.
    //! @param acceptLengths   [activeBatchSize] INT32 GPU accepted-token count per sequence.
    //! @param stateIndices    [activeBatchSize] INT32 GPU active-row to resident-slot map.
    //! @param verifySize      The verify-tree node count; selects snapshot vs chunk-replay for gdn.
    virtual void commitTree(rt::Tensor const& acceptedNodeIds, rt::Tensor const& acceptLengths,
        rt::Tensor const& stateIndices, int32_t verifySize, cudaStream_t stream) = 0;

protected:
    //! @param config          Recurrent-axis geometry.
    //! @param liveRecurrent   Per-layer persistent recurrent slabs [maxBatch, H, dk, dv]; not owned.
    //! @param liveConv        Per-layer persistent conv slabs [maxBatch, convDim, convKernel]; not owned.
    //! @param name            Prefix for the side-buffer tensor names surfaced in allocation diagnostics.
    //! @param stream          CUDA stream for the zeroing memsets.
    RecurrentCommitPolicy(Config config, std::vector<rt::Tensor>& liveRecurrent, std::vector<rt::Tensor>& liveConv,
        std::string const& name, cudaStream_t stream);

    //! Upload the per-layer `MtpLayerInfo` bundle once construction has the recurrent source pointers.
    //! @param recurrentSrcPointers Per-layer verify-pass recurrent source; null entries for the replay
    //!        path, where the recurrent state commits by reconstruction rather than scatter.
    void buildMtpLayerInfos(std::vector<void*> const& recurrentSrcPointers, cudaStream_t stream);

    //! Reshape the kind-specific recurrent side buffers; the base handles the shared conv buffer.
    virtual void reshapeRecurrentSide(int32_t activeBatchSize, int32_t seqLen) = 0;

    Config mConfig{};
    std::vector<rt::Tensor>& mLiveRecurrent;
    std::vector<rt::Tensor>& mLiveConv;
    std::vector<rt::Tensor> mIntermediateConvStates;
    rt::Tensor mDeviceMtpLayerInfos;
};

} // namespace rt
} // namespace trt_edgellm
