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

// appendSlot: joining a sequence to a batch that is already running. What matters is symmetry with
// eviction -- both must touch the same per-slot vectors -- and that slots already in flight are not
// disturbed. Everything here is host-only; no kernel runs and no device memory is touched.

#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/streaming.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

//! A running single-slot batch, as handleRequest would have built it for one sequence.
DecodingInferenceContext makeRunningBatch()
{
    DecodingInferenceContext context;
    context.initialize(/*batchSize=*/1, /*maxGenLength=*/64, /*visual=*/{}, /*deepstack=*/{}, /*loraName=*/"",
        /*cudaStream=*/nullptr, /*residentCapacity=*/4);
    context.rawBatchedInputIds.push_back({11, 12, 13});
    context.tokenIds[0] = {11, 12, 13, 100, 101}; // prompt plus two generated tokens
    context.currentGenerateLengths[0] = 2;
    context.batchIndexMapping[0] = 0;
    return context;
}

SlotSeed makeSeed(int32_t originalIndex = 1)
{
    SlotSeed seed;
    seed.systemPrompt = "sys";
    seed.promptTokenIds = {21, 22};
    seed.stopStrings = {"STOP", "longer-stop"};
    seed.originalIndex = originalIndex;
    return seed;
}

int32_t appendTestSlot(DecodingInferenceContext& context, SlotSeed seed, RequestId requestId)
{
    std::optional<ResidentRef> const resident = context.residentSlots.acquire();
    if (!resident.has_value())
    {
        throw std::runtime_error("test resident pool is full");
    }
    try
    {
        return context.appendSlot(std::move(seed), SequenceIdentity{requestId, *resident});
    }
    catch (...)
    {
        EXPECT_TRUE(context.residentSlots.release(*resident));
        throw;
    }
}

} // namespace

TEST(DecodingInferenceContextAppendTests, GrowsEveryVectorWithoutDisturbingResidents)
{
    DecodingInferenceContext context = makeRunningBatch();
    ASSERT_TRUE(context.perSlotSizesConsistent());

    int32_t const slot = appendTestSlot(context, makeSeed(), /*requestId=*/701);
    EXPECT_EQ(slot, 1);
    EXPECT_EQ(context.activeBatchSize, 2);
    EXPECT_TRUE(context.perSlotSizesConsistent());

    // The new slot starts at its prompt, exactly as a slot built at request start does.
    EXPECT_EQ(context.tokenIds[1], (std::vector<int32_t>{21, 22}));
    EXPECT_EQ(context.rawBatchedInputIds[1], (std::vector<int32_t>{21, 22}));
    EXPECT_EQ(context.currentGenerateLengths[1], 0);
    EXPECT_EQ(context.effectivePrefillLengths[1], 0);
    EXPECT_EQ(context.prefillStartLengths[1], 0);
    EXPECT_EQ(context.committedLengths[1], 0);
    EXPECT_EQ(context.requestIds[1], 701U);
    EXPECT_NE(context.residentRefs[1], context.residentRefs[0]);
    EXPECT_TRUE(context.residentSlots.contains(context.residentRefs[1]));
    EXPECT_EQ(context.finishedStates[1], 0);
    EXPECT_EQ(context.batchIndexMapping[1], 1);
    EXPECT_EQ(context.systemPrompts[1], "sys");

    // Delivery counters are seeded to the prompt so streaming emits only generated tokens.
    EXPECT_EQ(context.callbackEmittedTokenCounts[1], 2);
    EXPECT_EQ(context.slotStreams[1].sentTokenCount, 2U);
    EXPECT_EQ(context.slotStreams[1].maxStopLen, std::string("longer-stop").size());

    // And the same append left the resident slot exactly as it was.
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{11, 12, 13, 100, 101}));
    EXPECT_EQ(context.currentGenerateLengths[0], 2);
    EXPECT_EQ(context.batchIndexMapping[0], 0);
    EXPECT_EQ(context.finishedStates[0], 0);
}

TEST(DecodingInferenceContextAppendTests, RejectsWhatWouldCorruptTheBatch)
{
    // An empty batch is initialize()'s job, and the appended slot inherits sizing (logprobs
    // capacity) from an existing one, so there must be one.
    DecodingInferenceContext empty;
    empty.initialize(1, 64, {}, {}, "", nullptr);
    empty.activeBatchSize = 0;
    EXPECT_THROW(appendTestSlot(empty, makeSeed(), /*requestId=*/701), std::runtime_error);

    DecodingInferenceContext context = makeRunningBatch();

    SlotSeed noPrompt = makeSeed();
    noPrompt.promptTokenIds.clear();
    EXPECT_THROW(appendTestSlot(context, noPrompt, /*requestId=*/701), std::runtime_error);

    // Original index 0 is held by the live slot; a duplicate would file two sequences' results in
    // one bucket.
    EXPECT_THROW(appendTestSlot(context, makeSeed(/*originalIndex=*/0), /*requestId=*/701), std::runtime_error);

    // Same rule against results already collected.
    context.completedBatches[7] = {};
    EXPECT_THROW(appendTestSlot(context, makeSeed(/*originalIndex=*/7), /*requestId=*/701), std::runtime_error);

    // A reused prefix covering the whole prompt would leave nothing to prefill.
    SlotSeed allReused = makeSeed();
    allReused.prefillStart = static_cast<int32_t>(allReused.promptTokenIds.size());
    EXPECT_THROW(appendTestSlot(context, allReused, /*requestId=*/701), std::runtime_error);

    // Every rejection above must leave the batch exactly as it was.
    EXPECT_EQ(context.activeBatchSize, 1);
    EXPECT_TRUE(context.perSlotSizesConsistent());
}

TEST(DecodingInferenceContextAppendTests, RejectsAChannelAttachedElsewhereWithoutModifyingAnything)
{
    DecodingInferenceContext context = makeRunningBatch();

    auto channel = StreamChannel::create();
    attachStreamChannel(channel, /*originalIdx=*/5); // someone else owns it

    SlotSeed seed = makeSeed();
    seed.channel = channel;
    EXPECT_THROW(appendTestSlot(context, seed, /*requestId=*/701), std::runtime_error);
    EXPECT_EQ(context.activeBatchSize, 1);
    EXPECT_TRUE(context.perSlotSizesConsistent());
}

TEST(DecodingInferenceContextAppendTests, InheritsLogprobsCapacityFromAnExistingSlot)
{
    DecodingInferenceContext context = makeRunningBatch();
    context.numLogprobs = 4;
    context.stepLogprobs[0].data.resize(64 * 4);

    int32_t const slot = appendTestSlot(context, makeSeed(), /*requestId=*/701);
    EXPECT_EQ(context.stepLogprobs[slot].data.size(), context.stepLogprobs[0].data.size());
    EXPECT_EQ(context.stepLogprobs[slot].numSteps, 0);
}

TEST(DecodingInferenceContextAppendTests, TracksTheConditionalPruningVector)
{
    // Off: the vector stays empty, matching the "missing entry means 0" convention.
    DecodingInferenceContext context = makeRunningBatch();
    appendTestSlot(context, makeSeed(), /*requestId=*/701);
    EXPECT_TRUE(context.prunedPrefillTokens.empty());

    // On: it must grow with the batch or later per-slot reads shift by one.
    DecodingInferenceContext pruned = makeRunningBatch();
    pruned.prunedPrefillTokens.assign(1, 9);
    appendTestSlot(pruned, makeSeed(), /*requestId=*/701);
    ASSERT_EQ(pruned.prunedPrefillTokens.size(), 2U);
    EXPECT_EQ(pruned.prunedPrefillTokens[0], 9);
    EXPECT_EQ(pruned.prunedPrefillTokens[1], 0);
}

TEST(DecodingInferenceContextAppendTests, RaisesTheLogitBiasFlagsExactlyWhenBiasArrives)
{
    DecodingInferenceContext context = makeRunningBatch();

    appendTestSlot(context, makeSeed(1), /*requestId=*/701);
    EXPECT_FALSE(context.hasLogitBias) << "a slot without bias must not raise the flag";

    SlotSeed biased = makeSeed(2);
    biased.logitBias = {{42, -1.0F}};
    appendTestSlot(context, biased, /*requestId=*/702);
    EXPECT_TRUE(context.hasLogitBias);
    EXPECT_TRUE(context.logitBiasGpuDirty) << "the slot-indexed GPU table must be re-uploaded after growth";
}

TEST(DecodingInferenceContextAppendTests, AReusedPrefixSeedsOnlyTheSuffixIntoTheWorkingHistory)
{
    // A context-cache hit means the tokens before prefillStart already have KV state; the working
    // history and every count derived from it must start at the suffix, while the raw input keeps
    // the full prompt for identity and result assembly.
    DecodingInferenceContext context = makeRunningBatch();
    SlotSeed seed = makeSeed();
    seed.promptTokenIds = {21, 22, 23, 24};
    seed.prefillStart = 3;
    int32_t const slot = appendTestSlot(context, seed, /*requestId=*/701);

    EXPECT_EQ(context.rawBatchedInputIds[static_cast<size_t>(slot)], (std::vector<int32_t>{21, 22, 23, 24}));
    EXPECT_EQ(context.tokenIds[static_cast<size_t>(slot)], (std::vector<int32_t>{24}));
    EXPECT_EQ(context.callbackEmittedTokenCounts[static_cast<size_t>(slot)], 1)
        << "callback and streaming offsets must count from the suffix, or the first generated "
           "tokens would be swallowed as prompt";
    EXPECT_EQ(context.slotStreams[static_cast<size_t>(slot)].sentTokenCount, 1U);
}

TEST(DecodingInferenceContextAppendTests, SurvivesTheEvictionThatFollowsIt)
{
    // The scenario the symmetry exists for: B joins while A runs, A finishes and is evicted, and
    // B's state must come through the compaction intact. A vector missed by either side shows up
    // here as B inheriting A's values.
    DecodingInferenceContext context = makeRunningBatch();
    // Visual-token pruning active: the conditional vector is sized, so eviction must move it too.
    context.prunedPrefillTokens = {3};
    SlotSeed seed = makeSeed();
    seed.channel = StreamChannel::create();
    seed.samplingSeed = 77;
    int32_t const slotB = appendTestSlot(context, seed, /*requestId=*/701);
    context.prunedPrefillTokens[static_cast<size_t>(slotB)] = 5; // as B's seated prefill would record

    context.finishedStates[0] = 1; // A is done

    std::vector<int32_t> const mapping = buildBatchMapping(context.finishedStates);
    ASSERT_EQ(mapping[0], -1);
    ASSERT_EQ(mapping[static_cast<size_t>(slotB)], 0);

    // The same moves performBatchEvict makes for the host-side per-slot state.
    compactVector(mapping, context.finishedStates);
    compactVector(mapping, context.currentGenerateLengths);
    compactVector(mapping, context.samplingSeeds);
    compactVector(mapping, context.acceptedDraftTokens);
    compactVector(mapping, context.proposedDraftTokens);
    compactVector(mapping, context.tokenIds);
    compactVector(mapping, context.systemPrompts);
    compactVector(mapping, context.rawBatchedInputIds);
    compactVector(mapping, context.effectivePrefillLengths);
    compactVector(mapping, context.prefillStartLengths);
    compactVector(mapping, context.committedLengths);
    compactVector(mapping, context.requestIds);
    compactVector(mapping, context.residentRefs);
    if (!context.prunedPrefillTokens.empty())
    {
        compactVector(mapping, context.prunedPrefillTokens);
    }
    compactVector(mapping, context.batchIndexMapping);
    compactVector(mapping, context.thinkingDone);
    compactVector(mapping, context.guidedReasoningEnded);
    compactVector(mapping, context.callbackEmittedTokenCounts);
    compactVector(mapping, context.slotStreams);
    compactVector(mapping, context.stopStringsPerSlot);
    compactVector(mapping, context.logitBiasPerSlot);
    compactVector(mapping, context.stepLogprobs);
    context.activeBatchSize = 1;

    EXPECT_TRUE(context.perSlotSizesConsistent());
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{21, 22})) << "B inherited A's tokens";
    EXPECT_EQ(context.batchIndexMapping[0], 1);
    EXPECT_EQ(context.systemPrompts[0], "sys");
    EXPECT_EQ(context.samplingSeeds[0], 77u) << "B inherited A's sampling seed";
    EXPECT_NE(context.slotStreams[0].channel, nullptr) << "B's delivery channel was lost in the move";
    EXPECT_EQ(context.prunedPrefillTokens[0], 5) << "B inherited A's pruning count";
}

// swapExecutionRows presents one logical sequence at row zero without moving resident state.

TEST(DecodingInferenceContextSwapTests, ExchangesEveryPerSlotFieldAndIsItsOwnInverse)
{
    DecodingInferenceContext context = makeRunningBatch();
    SlotSeed seed = makeSeed();
    seed.channel = StreamChannel::create();
    seed.logitBias = {{7, 2.0F}};
    appendTestSlot(context, seed, /*requestId=*/701);
    context.thinkingDone[0] = 1; // A left its thinking block; B has not

    context.swapExecutionRows(0, 1);
    // B's identity, history and delivery state now sit at slot 0; A's at slot 1.
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{21, 22}));
    EXPECT_EQ(context.batchIndexMapping[0], 1);
    EXPECT_NE(context.slotStreams[0].channel, nullptr);
    EXPECT_EQ(context.currentGenerateLengths[1], 2);
    EXPECT_EQ(context.batchIndexMapping[1], 0);
    EXPECT_EQ(context.logitBiasPerSlot[0].count(7), 1U);
    EXPECT_EQ(context.thinkingDone[0], 0);
    EXPECT_EQ(context.thinkingDone[1], 1) << "the thinking latch stayed behind in the swap";
    EXPECT_TRUE(context.logitBiasGpuDirty) << "the slot-indexed GPU bias table must be re-uploaded";
    EXPECT_TRUE(context.perSlotSizesConsistent());

    // The restore is the same call.
    context.swapExecutionRows(0, 1);
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{11, 12, 13, 100, 101}));
    EXPECT_EQ(context.batchIndexMapping[0], 0);
    EXPECT_EQ(context.tokenIds[1], (std::vector<int32_t>{21, 22}));
    EXPECT_EQ(context.slotStreams[0].channel, nullptr);
    EXPECT_NE(context.slotStreams[1].channel, nullptr);
}

TEST(DecodingInferenceContextSwapTests, RejectsOutOfRangeAndIgnoresSelfSwap)
{
    DecodingInferenceContext context = makeRunningBatch();
    EXPECT_THROW(context.swapExecutionRows(0, 1), std::runtime_error); // batch of 1 has no row 1
    EXPECT_THROW(context.swapExecutionRows(-1, 0), std::runtime_error);

    std::vector<int32_t> const before = context.tokenIds[0];
    context.swapExecutionRows(0, 0);
    EXPECT_EQ(context.tokenIds[0], before);
}
