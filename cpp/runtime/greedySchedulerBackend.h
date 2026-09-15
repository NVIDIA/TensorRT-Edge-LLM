/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "runtime/continuousScheduler.h"
#include "runtime/sequenceStepRuntime.h"
namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
}
namespace rt
{
//! Independent CPU policy over one sampling-free TensorRT runtime. All forward methods belong to the scheduler worker.
class SamplingSchedulerBackend : public SchedulerBackend
{
public:
    SamplingSchedulerBackend(LLMRankRuntime& runtime, cudaStream_t stream, int32_t vocabularySize,
        tokenizer::Tokenizer const* tokenizer = nullptr, bool captureGraphs = false);
    void start() override;
    void invalidate() noexcept override
    {
        mSteps.poison();
    }
    bool validToken(int32_t token) const override
    {
        return token >= 0 && token < mVocabulary;
    }
    int32_t vocabularySize() const override
    {
        return mVocabulary;
    }
    SequenceOptions normalizeOptions(SequenceOptions options) const override;
    SequenceHandle acquire(uint64_t id, std::vector<int32_t> prompt, SequenceOptions options) override;
    SequenceState const& state(SequenceHandle handle) const override;
    void prefill(SequenceHandle handle) override;
    void decode(std::array<SequenceHandle, 2> const& handles, int32_t count) override;
    void release(SequenceHandle handle) override;
    SequenceSample lastSample(SequenceHandle handle) const override;
    std::string_view text(SequenceHandle handle) const override;
    std::vector<SequenceSample> const& logprobs(SequenceHandle handle) const override;
    SequenceFinish finishReason(SequenceHandle handle) const override;
    void finalize(SequenceHandle handle) override;

private:
    void sample(Tensor const& logits, std::array<SequenceHandle, 2> const& handles, int32_t count);
    SequencePolicy const& policy(SequenceHandle handle) const;
    SequenceStepRuntime mSteps;
    cudaStream_t mStream;
    int32_t mVocabulary;
    int mDevice{};
    Tensor mHostLogits;
    SequenceSampler mSampler;
    std::array<SequencePolicy, 2> mPolicies;
    std::vector<std::string> mPieces;
    size_t mMaxPieceBytes{};
    std::vector<int32_t> mEos;
    int32_t mPrimaryEos{-1}, mThinkStart{-1}, mThinkEnd{-1};
};
//! P4 source compatibility name; the options-based API applies full independent policy.
using GreedySchedulerBackend = SamplingSchedulerBackend;
} // namespace rt
} // namespace trt_edgellm
