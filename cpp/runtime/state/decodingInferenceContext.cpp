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

#include "runtime/state/decodingInferenceContext.h"

#include "common/checkMacros.h"
#include "runtime/debug/layerDebugger.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

namespace
{

void exchangeExecutionRows(DecodingInferenceContext& context, size_t a, size_t b) noexcept
{
    using std::swap;
    swap(context.systemPrompts[a], context.systemPrompts[b]);
    swap(context.rawBatchedInputIds[a], context.rawBatchedInputIds[b]);
    swap(context.tokenIds[a], context.tokenIds[b]);
    swap(context.currentGenerateLengths[a], context.currentGenerateLengths[b]);
    swap(context.effectivePrefillLengths[a], context.effectivePrefillLengths[b]);
    swap(context.prefillStartLengths[a], context.prefillStartLengths[b]);
    swap(context.committedLengths[a], context.committedLengths[b]);
    swap(context.requestIds[a], context.requestIds[b]);
    swap(context.residentRefs[a], context.residentRefs[b]);
    swap(context.finishedStates[a], context.finishedStates[b]);
    swap(context.batchIndexMapping[a], context.batchIndexMapping[b]);
    swap(context.thinkingDone[a], context.thinkingDone[b]);
    swap(context.guidedReasoningEnded[a], context.guidedReasoningEnded[b]);
    swap(context.callbackEmittedTokenCounts[a], context.callbackEmittedTokenCounts[b]);
    swap(context.slotStreams[a], context.slotStreams[b]);
    swap(context.stopStringsPerSlot[a], context.stopStringsPerSlot[b]);
    swap(context.samplingSeeds[a], context.samplingSeeds[b]);
    swap(context.acceptedDraftTokens[a], context.acceptedDraftTokens[b]);
    swap(context.proposedDraftTokens[a], context.proposedDraftTokens[b]);
    swap(context.logitBiasPerSlot[a], context.logitBiasPerSlot[b]);
    swap(context.stepLogprobs[a], context.stepLogprobs[b]);
    if (!context.prunedPrefillTokens.empty())
    {
        swap(context.prunedPrefillTokens[a], context.prunedPrefillTokens[b]);
    }
}

} // namespace

// Out-of-line so the unique_ptr<LayerDebugger> member can hold an incomplete type
// in the header; LayerDebugger is complete here. Move ops are defined too: a
// user-declared destructor suppresses the implicit move, but the context is
// returned by value (e.g. tests' makeContext), so it must stay movable. It
// remains non-copyable through the unique_ptr member.
DecodingInferenceContext::DecodingInferenceContext() = default;
DecodingInferenceContext::DecodingInferenceContext(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext& DecodingInferenceContext::operator=(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext::~DecodingInferenceContext() = default;

int32_t DecodingInferenceContext::appendSlot(SlotSeed seed, SequenceIdentity identity)
{
    ELLM_CHECK(activeBatchSize > 0, "appendSlot joins a running batch; an empty one is built by initialize().");
    ELLM_CHECK(!seed.promptTokenIds.empty(), "appendSlot requires a non-empty prompt.");
    ELLM_CHECK(seed.prefillStart >= 0 && seed.prefillStart < static_cast<int32_t>(seed.promptTokenIds.size()),
        "appendSlot: the reused prefix must leave at least one token to prefill.");
    ELLM_CHECK(perSlotSizesConsistent(), "appendSlot found the batch already inconsistent.");
    ELLM_CHECK(identity.requestId != 0, "appendSlot requires a non-zero request ID.");
    ELLM_CHECK(residentSlots.contains(identity.resident), "appendSlot requires a live reserved resident slot.");
    for (size_t slot = 0; slot < requestIds.size(); ++slot)
    {
        ELLM_CHECK(requestIds[slot] != identity.requestId, "appendSlot: request ID is already resident.");
        ELLM_CHECK(residentRefs[slot] != identity.resident, "appendSlot: resident slot is already in use.");
    }
    for (int32_t const original : batchIndexMapping)
    {
        ELLM_CHECK(original != seed.originalIndex, "appendSlot: original index is held by a live slot.");
    }
    ELLM_CHECK(completedBatches.find(seed.originalIndex) == completedBatches.end(),
        "appendSlot: original index already has a collected result.");

    size_t maxStopLen = 0;
    for (auto const& stop : seed.stopStrings)
    {
        maxStopLen = std::max(maxStopLen, stop.size());
    }

    // The working history holds only what this batch must compute: everything from prefillStart
    // on. Founder slots under a context cache are seeded the same way (setUpForPrefillExecution),
    // so streaming offsets and generate-length math see one shape regardless of how a slot joined.
    size_t const promptLength = seed.promptTokenIds.size() - static_cast<size_t>(seed.prefillStart);

    std::vector<int32_t> rawInput = seed.promptTokenIds;
    std::vector<int32_t> workingTokens(seed.promptTokenIds.begin() + seed.prefillStart, seed.promptTokenIds.end());
    rt::LogprobsSlot logprobs;
    if (numLogprobs > 0)
    {
        logprobs.data.resize(stepLogprobs.front().data.size());
    }
    SlotStreamState slotState;
    slotState.channel = seed.channel;
    slotState.sentTokenCount = promptLength;
    slotState.lastEmittedTokenCount = promptLength;
    slotState.maxStopLen = maxStopLen;

    // All allocations precede the stream-channel attachment, which is the externally visible
    // commit point. Once attached, the prepared values below move into reserved storage without
    // allocating, so an admission either leaves the old batch intact or appends one full row.
    size_t const nextSize = static_cast<size_t>(activeBatchSize) + 1;
    systemPrompts.reserve(nextSize);
    rawBatchedInputIds.reserve(nextSize);
    tokenIds.reserve(nextSize);
    currentGenerateLengths.reserve(nextSize);
    samplingSeeds.reserve(nextSize);
    acceptedDraftTokens.reserve(nextSize);
    proposedDraftTokens.reserve(nextSize);
    effectivePrefillLengths.reserve(nextSize);
    prefillStartLengths.reserve(nextSize);
    committedLengths.reserve(nextSize);
    requestIds.reserve(nextSize);
    residentRefs.reserve(nextSize);
    finishedStates.reserve(nextSize);
    batchIndexMapping.reserve(nextSize);
    thinkingDone.reserve(nextSize);
    guidedReasoningEnded.reserve(nextSize);
    callbackEmittedTokenCounts.reserve(nextSize);
    slotStreams.reserve(nextSize);
    stopStringsPerSlot.reserve(nextSize);
    logitBiasPerSlot.reserve(nextSize);
    stepLogprobs.reserve(nextSize);
    if (!prunedPrefillTokens.empty())
    {
        prunedPrefillTokens.reserve(nextSize);
    }
    if (seed.channel)
    {
        attachStreamChannel(seed.channel, seed.originalIndex);
    }

    systemPrompts.push_back(std::move(seed.systemPrompt));
    rawBatchedInputIds.push_back(std::move(rawInput));
    tokenIds.push_back(std::move(workingTokens));
    currentGenerateLengths.push_back(0);
    samplingSeeds.push_back(seed.samplingSeed);
    // Spec decode never takes admissions, but the counters are sized for every deployment.
    acceptedDraftTokens.push_back(0);
    proposedDraftTokens.push_back(0);
    effectivePrefillLengths.push_back(0);
    prefillStartLengths.push_back(seed.prefillStart);
    committedLengths.push_back(seed.prefillStart);
    requestIds.push_back(identity.requestId);
    residentRefs.push_back(identity.resident);
    finishedStates.push_back(0);
    batchIndexMapping.push_back(seed.originalIndex);
    // Reasoning trackers: a fresh slot has produced no tokens, so neither latch is set. Guided
    // admission itself is refused upstream (the GuidedDecoder's matchers are slot-numbered and the
    // seating swaps do not cover them), but the vectors stay rectangular either way.
    thinkingDone.push_back(0);
    guidedReasoningEnded.push_back(0);

    // Seeded to the prompt length so streaming and callbacks emit only generated tokens, exactly as
    // a slot set up at request start is seeded.
    callbackEmittedTokenCounts.push_back(static_cast<int32_t>(promptLength));
    slotStreams.push_back(std::move(slotState));

    stopStringsPerSlot.push_back(std::move(seed.stopStrings));

    logitBiasPerSlot.push_back(std::move(seed.logitBias));
    hasLogitBias = hasLogitBias || !logitBiasPerSlot.back().empty();
    // The GPU-side table is indexed by slot, so any bias anywhere means the grown table must go up
    // again -- mirroring what eviction does after it compacts.
    logitBiasGpuDirty = hasLogitBias;

    // Capacity is deployment-dependent (spec decode can accept several tokens per step), so it is
    // inherited from a slot that was sized by the code that knows -- the reason this method refuses
    // an empty batch.
    stepLogprobs.push_back(std::move(logprobs));

    // Conditional vector: sized only while visual-token pruning is active. Grown when present so a
    // per-slot read stays aligned, left empty otherwise.
    if (!prunedPrefillTokens.empty())
    {
        prunedPrefillTokens.push_back(0);
    }

    ++activeBatchSize;
    ELLM_CHECK(perSlotSizesConsistent(), "appendSlot left the batch inconsistent; a vector was missed.");
    return activeBatchSize - 1;
}

void DecodingInferenceContext::swapExecutionRows(int32_t rowA, int32_t rowB)
{
    ELLM_CHECK(rowA >= 0 && rowA < activeBatchSize && rowB >= 0 && rowB < activeBatchSize,
        "swapExecutionRows: row is out of range.");
    ELLM_CHECK(perSlotSizesConsistent(), "swapExecutionRows found the batch inconsistent.");
    if (rowA == rowB)
    {
        return;
    }

    exchangeExecutionRows(*this, static_cast<size_t>(rowA), static_cast<size_t>(rowB));
    // hasLogitBias / logitBiasGpuDirty are batch-wide, but the GPU-side table is slot-indexed, so a
    // swap that moved any bias must be pushed up again before the next step reads it.
    if (hasLogitBias)
    {
        logitBiasGpuDirty = true;
    }
}

void DecodingInferenceContext::restoreExecutionRows(int32_t rowA, int32_t rowB) noexcept
{
    exchangeExecutionRows(*this, static_cast<size_t>(rowA), static_cast<size_t>(rowB));
    if (hasLogitBias)
    {
        logitBiasGpuDirty = true;
    }
}

bool DecodingInferenceContext::perSlotSizesConsistent() const noexcept
{
    auto const expected = static_cast<size_t>(activeBatchSize);
    return systemPrompts.size() == expected && rawBatchedInputIds.size() == expected && tokenIds.size() == expected
        && currentGenerateLengths.size() == expected && samplingSeeds.size() == expected
        && acceptedDraftTokens.size() == expected && proposedDraftTokens.size() == expected
        && effectivePrefillLengths.size() == expected && prefillStartLengths.size() == expected
        && committedLengths.size() == expected && requestIds.size() == expected && residentRefs.size() == expected
        && finishedStates.size() == expected && batchIndexMapping.size() == expected && thinkingDone.size() == expected
        && guidedReasoningEnded.size() == expected && callbackEmittedTokenCounts.size() == expected
        && slotStreams.size() == expected && stopStringsPerSlot.size() == expected
        && logitBiasPerSlot.size() == expected && stepLogprobs.size() == expected
        && (prunedPrefillTokens.empty() || prunedPrefillTokens.size() == expected);
}

void DecodingInferenceContext::initialize(int32_t batchSize, int32_t maxGenLength,
    rt::OptionalInputTensor const& visual, rt::OptionalInputTensors const& deepstack, std::string const& loraName,
    cudaStream_t cudaStream, int32_t residentCapacity)
{
    int32_t const poolCapacity = residentCapacity == 0 ? batchSize : residentCapacity;
    ELLM_CHECK(poolCapacity >= batchSize, "Resident slot pool cannot hold the initial batch");
    residentSlots.reset(poolCapacity);
    systemPrompts.resize(batchSize);
    rawBatchedInputIds.reserve(batchSize);
    tokenIds.resize(batchSize);
    currentGenerateLengths.resize(batchSize, 0);
    samplingSeeds.resize(batchSize, kDefaultSamplingSeed);
    effectivePrefillLengths.resize(batchSize, 0);
    prefillStartLengths.resize(batchSize, 0);
    committedLengths.resize(batchSize, 0);
    requestIds.resize(batchSize);
    residentRefs.resize(batchSize);
    finishedStates.resize(batchSize, 0);
    thinkingDone.clear();
    thinkingDone.resize(batchSize, 0);
    guidedReasoningEnded.clear();
    guidedReasoningEnded.resize(batchSize, 0);
    acceptedDraftTokens.assign(batchSize, 0);
    proposedDraftTokens.assign(batchSize, 0);
    slotStreams.clear();
    slotStreams.resize(batchSize);
    stopStringsPerSlot.clear();
    stopStringsPerSlot.resize(batchSize);
    logitBiasPerSlot.clear();
    logitBiasPerSlot.resize(batchSize);
    hasLogitBias = false;
    logitBiasGpuDirty = false;
    hasGuidedDecoding = false;
    guidedUnsatisfiableSlots.clear();
    guidedMaskSuppressedPerSlot.clear();
    enableThinking = false;
    callbackEmittedTokenCounts.clear();
    callbackEmittedTokenCounts.resize(batchSize, 0);
    shouldStopAfterAcceptedToken = {};

    batchIndexMapping.resize(batchSize);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        batchIndexMapping[i] = i;
        requestIds[i] = static_cast<RequestId>(i) + 1;
        auto const resident = residentSlots.acquire();
        ELLM_CHECK(resident.has_value(), "Resident slot allocation failed for the initial batch");
        residentRefs[i] = *resident;
    }
    nextStepId = 1;

    completedBatches.clear();

    // Initialize per-batch logprobs accumulator (populated only when numLogprobs > 0)
    stepLogprobs.clear();
    stepLogprobs.resize(batchSize);

    visualEmbeddings = visual;
    deepstackFeatures = deepstack;
    generationRound = 0;
    maxGenerateLength = maxGenLength;
    diffusionMaxDenoisingSteps = 0;
    activeBatchSize = batchSize;
    loraWeightsName = loraName;
    stream = cudaStream;
}

void DecodingInferenceContext::initializeRaggedScratch(RaggedEngineContract const& contract)
{
    scheduledStep.sequences.reserve(static_cast<size_t>(contract.maxNumSequences));
    completionSequences.reserve(static_cast<size_t>(contract.maxNumSequences));
    raggedBatchBuilder.emplace(contract);
    raggedBatchBuilder->reserve(raggedExecutionBatch);
}

} // namespace rt
} // namespace trt_edgellm
