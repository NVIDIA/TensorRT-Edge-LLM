/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/greedySchedulerBackend.h"
#include "common/checkMacros.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <stdexcept>
namespace trt_edgellm
{
namespace rt
{
SamplingSchedulerBackend::SamplingSchedulerBackend(LLMRankRuntime& runtime, cudaStream_t stream, int32_t vocabularySize,
    tokenizer::Tokenizer const* tokenizer, bool captureGraphs)
    : mSteps(runtime, stream, captureGraphs)
    , mStream(stream)
    , mVocabulary(vocabularySize)
    , mHostLogits({2, vocabularySize}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
    , mSampler(vocabularySize)
{
    CUDA_CHECK(cudaGetDevice(&mDevice));
    if (tokenizer)
    {
        mPieces.reserve(vocabularySize);
        for (int32_t id = 0; id < vocabularySize; ++id)
        {
            mPieces.push_back(tokenizer->idToPiece(id, true));
            mMaxPieceBytes = std::max(mMaxPieceBytes, mPieces.back().size());
        }
        mEos = tokenizer->getEosIds();
        mPrimaryEos = tokenizer->getEosId();
        mThinkStart = tokenizer->getTokenId("<think>");
        mThinkEnd = tokenizer->getTokenId("</think>");
    }
}
void SamplingSchedulerBackend::start()
{
    CUDA_CHECK(cudaSetDevice(mDevice));
}
SequenceOptions SamplingSchedulerBackend::normalizeOptions(SequenceOptions options) const
{
    if (mPieces.empty() && (!options.stopStrings.empty() || options.enableThinking))
    {
        throw std::invalid_argument("Text policy requires tokenizer metadata");
    }
    if (options.eosTokenIds.empty())
    {
        options.eosTokenIds = mEos;
    }
    if (options.primaryEosTokenId == -1)
    {
        options.primaryEosTokenId = mPrimaryEos;
    }
    if (options.thinkingStartTokenId == -1)
    {
        options.thinkingStartTokenId = mThinkStart;
    }
    if (options.thinkingEndTokenId == -1)
    {
        options.thinkingEndTokenId = mThinkEnd;
    }
    return options;
}
SequenceHandle SamplingSchedulerBackend::acquire(uint64_t id, std::vector<int32_t> prompt, SequenceOptions options)
{
    auto handle = mSteps.acquire(id, std::move(prompt), options);
    mPolicies[handle.slot].reset(std::move(options), mMaxPieceBytes);
    return handle;
}
SequenceState const& SamplingSchedulerBackend::state(SequenceHandle handle) const
{
    return mSteps.state(handle);
}
SequencePolicy const& SamplingSchedulerBackend::policy(SequenceHandle handle) const
{
    mSteps.state(handle);
    return mPolicies[handle.slot];
}
SequenceSample SamplingSchedulerBackend::lastSample(SequenceHandle handle) const
{
    return policy(handle).last();
}
std::string_view SamplingSchedulerBackend::text(SequenceHandle handle) const
{
    return policy(handle).text();
}
std::vector<SequenceSample> const& SamplingSchedulerBackend::logprobs(SequenceHandle handle) const
{
    return policy(handle).logprobs();
}
SequenceFinish SamplingSchedulerBackend::finishReason(SequenceHandle handle) const
{
    return policy(handle).finish();
}
void SamplingSchedulerBackend::finalize(SequenceHandle handle)
{
    mSteps.state(handle);
    mPolicies[handle.slot].finalize();
}
void SamplingSchedulerBackend::sample(Tensor const& logits, std::array<SequenceHandle, 2> const& handles, int32_t count)
{
    if (logits.getDataType() != nvinfer1::DataType::kFLOAT || logits.getShape()[0] != count
        || logits.getShape()[1] != mVocabulary)
    {
        throw std::runtime_error("Unexpected scheduler logits");
    }
    CUDA_CHECK(cudaMemcpyAsync(mHostLogits.rawPointer(), logits.rawPointer(), count * mVocabulary * sizeof(float),
        cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    auto const* data = mHostLogits.dataPointer<float>();
    for (int32_t row = 0; row < count; ++row)
    {
        auto handle = handles[row];
        auto const& sequence = state(handle);
        auto sample = mSampler.sample(data + row * mVocabulary, sequence.options(), sequence.randomCounter());
        mSteps.acceptToken(handle, sample.token, sample.randomCounter - sequence.randomCounter());
        mPolicies[handle.slot].accept(sample, mPieces.empty() ? std::string_view{} : mPieces[sample.token]);
        if (mPolicies[handle.slot].finish() != SequenceFinish::kNone)
        {
            mSteps.finish(handle);
        }
    }
}
void SamplingSchedulerBackend::prefill(SequenceHandle handle)
{
    auto const& logits = mSteps.beginPrefillChunk(handle);
    mSteps.completeStep();
    if (state(handle).phase() == SequencePhase::kAwaitingSample)
    {
        sample(logits, {handle, {}}, 1);
    }
}
void SamplingSchedulerBackend::decode(std::array<SequenceHandle, 2> const& handles, int32_t count)
{
    auto const& logits = mSteps.beginDecode(handles, count);
    mSteps.completeStep();
    sample(logits, handles, count);
}
void SamplingSchedulerBackend::release(SequenceHandle handle)
{
    mSteps.finish(handle);
    mSteps.release(handle);
}
} // namespace rt
} // namespace trt_edgellm
