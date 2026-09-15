/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common/tensor.h"
#include "runtime/state/sequenceSlots.h"
#include <array>
#include <memory>

namespace trt_edgellm
{
namespace rt
{
class LLMRankRuntime;

//! Exclusive, single-threaded text-only step session borrowing one rank runtime and its stream.
//! Runtime and stream must outlive the session. Do not invoke other runtime APIs during this lease.
//! Engine/CUDA failure poisons the lease; recreate the parent runtime before further use.
class SequenceStepRuntime
{
public:
    SequenceStepRuntime(LLMRankRuntime& runtime, cudaStream_t stream, bool captureGraphs = false);
    ~SequenceStepRuntime();
    SequenceStepRuntime(SequenceStepRuntime const&) = delete;
    SequenceStepRuntime& operator=(SequenceStepRuntime const&) = delete;

    SequenceHandle acquire(uint64_t requestId, std::vector<int32_t> prompt, SequenceOptions options);
    SequenceState const& state(SequenceHandle handle) const;
    //! Enqueue a prompt span or pending output tokens. No sampling, output publication or CPU state commit occurs.
    //! Logits are borrowed until the next begin call; they become readable after completion on the supplied stream.
    //! Low-level span control for diagnostics; arbitrary partitions are not numerically qualified.
    Tensor const& beginPrefill(SequenceHandle handle, int32_t count);
    //! Enqueue at most 128 true prompt tokens; completeStep publishes progress. Sample only at kAwaitingSample.
    Tensor const& beginPrefillChunk(SequenceHandle handle);
    Tensor const& beginDecode(std::array<SequenceHandle, 2> const& handles, int32_t count);
    //! Wait for forward completion before publishing endpoints and reusing pinned staging or physical slots.
    void completeStep();
    void acceptToken(SequenceHandle handle, int32_t token, uint64_t randomDraws = 0);
    void finish(SequenceHandle handle);
    void release(SequenceHandle handle);
    bool healthy() const noexcept;
    //! Retain the parent lease after an external sampling/worker failure. Recreate the parent to recover.
    void poison() noexcept;

private:
    struct Lease;
    struct View;
    void requireIdle() const;
    void captureDecodeViews();
    Tensor const& enqueue(std::array<SequenceHandle, 2> const& handles, int32_t count, int32_t span, bool decode);
    LLMRankRuntime& mRuntime;
    cudaStream_t mStream;
    std::unique_ptr<Lease> mLease;
    SequenceSlots mSlots;
    std::array<std::unique_ptr<View>, 3> mViews;
    Tensor mHostStarts;
    Tensor mDeviceStarts;
    Tensor mHostSelect;
    cudaEvent_t mComplete{};
    bool mPending{};
    bool mRopeInitialized{};
    bool mDecode{};
    int32_t mCount{};
    int32_t mSpan{};
    std::array<SequenceHandle, 2> mHandles{};
};
} // namespace rt
} // namespace trt_edgellm
