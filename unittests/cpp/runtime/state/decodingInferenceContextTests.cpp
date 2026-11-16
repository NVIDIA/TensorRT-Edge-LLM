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
#include "sampler/sampling.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;
using trt_edgellm::RowSamplingParams;

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
    seed.maxGenerateLength = 32;
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

TEST(DecodingInferenceContextAppendTests, GrowsFirstSlotOfEmptyBatch)
{
    // A context initialized for zero sequences is a legitimate empty batch -- given a resident
    // capacity. With the pool defaulting to the batch size there would be nothing to hand out, and
    // every later admission would be refused for capacity with no error naming the cause, so that
    // is refused up front. Its first slot then arrives through appendSlot exactly like any later
    // one, with nothing to inherit from: its logprobs accumulator is sized for its own budget and
    // width, and a seed without a generate budget is refused at the seam before anything is
    // modified. (Per-field growth is GrowsEveryVectorWithoutDisturbingResidents'.)
    DecodingInferenceContext context;
    EXPECT_THROW(context.initialize(/*batchSize=*/0, /*maxGenLength=*/64, /*visual=*/{}, /*deepstack=*/{},
                     /*loraName=*/"", /*cudaStream=*/nullptr),
        std::runtime_error);
    context.initialize(/*batchSize=*/0, /*maxGenLength=*/64, /*visual=*/{}, /*deepstack=*/{}, /*loraName=*/"",
        /*cudaStream=*/nullptr, /*residentCapacity=*/4);
    ASSERT_EQ(context.activeBatchSize, 0);
    ASSERT_TRUE(context.perSlotSizesConsistent());
    ASSERT_EQ(context.residentSlots.available(), 4);

    SlotSeed noBudget = makeSeed(/*originalIndex=*/0);
    noBudget.maxGenerateLength = 0;
    EXPECT_THROW(appendTestSlot(context, noBudget, /*requestId=*/701), std::runtime_error);
    EXPECT_EQ(context.activeBatchSize, 0);
    EXPECT_TRUE(context.perSlotSizesConsistent());
    EXPECT_EQ(context.residentSlots.available(), 4) << "a refused append holds no slot";

    context.logprobsStepOvershoot = 2;
    SlotSeed seed = makeSeed(/*originalIndex=*/0);
    seed.channel = StreamChannel::create();
    seed.maxGenerateLength = 16;
    seed.numLogprobs = 4;
    int32_t const slot = appendTestSlot(context, seed, /*requestId=*/701);

    EXPECT_EQ(slot, 0);
    EXPECT_EQ(context.activeBatchSize, 1);
    EXPECT_TRUE(context.perSlotSizesConsistent());
    EXPECT_EQ(context.residentRefs[0], (ResidentRef{0, 2})) << "the refused append's reservation spent epoch 1";
    EXPECT_EQ(context.slotStreams[0].channel, seed.channel);
    EXPECT_EQ(context.residentSlots.available(), 3);
    EXPECT_EQ(context.stepLogprobs[slot].data.size(), (16U + 2U) * 4U) << "sized for the slot's own budget and width";
    EXPECT_EQ(context.stepLogprobs[slot].numSteps, 0);
    EXPECT_EQ(context.numLogprobsPerSlot[slot], 4);
    EXPECT_EQ(context.maxNumLogprobs(), 4);

    // And a second one lands beside it: the empty start left the batch no different from one
    // that initialize() had populated. Without logprobs of its own it holds no accumulator.
    int32_t const second = appendTestSlot(context, makeSeed(/*originalIndex=*/1), /*requestId=*/702);
    EXPECT_EQ(second, 1);
    EXPECT_EQ(context.activeBatchSize, 2);
    EXPECT_TRUE(context.perSlotSizesConsistent());
    EXPECT_TRUE(context.stepLogprobs[second].data.empty());
    EXPECT_EQ(context.maxNumLogprobs(), 4);
}

// Given a running batch built at request start with a budget of 64 and no logprobs
// When slots with their own sampling, seed, budgets of 32 and 16 and logprobs widths are appended, a seed without
//      a budget is offered, and the budget of a row outside the batch is read
// Then each slot keeps its own settings and an accumulator sized for its budget and width, the resident keeps
//      the batch's, the budget-less seed is refused with the batch intact, and the out-of-batch read throws
TEST(DecodingInferenceContextAppendTests, EachSlotKeepsItsOwnSettings)
{
    DecodingInferenceContext context = makeRunningBatch();
    SlotSeed seed = makeSeed();
    seed.sampling = RowSamplingParams{0.7F, 40, 0.9F};
    seed.samplingSeed = 77;
    seed.requestStableSampling = true;
    seed.numLogprobs = 4;
    SlotSeed shortSeed = makeSeed(/*originalIndex=*/2);
    shortSeed.maxGenerateLength = 16;
    shortSeed.numLogprobs = 3;
    int32_t const slot = appendTestSlot(context, seed, /*requestId=*/701);
    int32_t const shortSlot = appendTestSlot(context, shortSeed, /*requestId=*/702);

    EXPECT_EQ(context.samplingParams[slot], (RowSamplingParams{0.7F, 40, 0.9F}));
    EXPECT_EQ(context.samplingSeeds[slot], 77U);
    EXPECT_EQ(context.requestStableSampling[slot], 1);
    EXPECT_EQ(context.requestStableSampling[shortSlot], 0) << "a seed without an explicit seed draws the shared stream";
    EXPECT_EQ(context.numLogprobsPerSlot[slot], 4);
    EXPECT_EQ(context.generateBudget(0), 64);
    EXPECT_EQ(context.generateBudget(slot), 32);
    EXPECT_EQ(context.generateBudget(shortSlot), 16);
    EXPECT_EQ(context.stepLogprobs[slot].data.size(), 32U * 4U);
    EXPECT_EQ(context.stepLogprobs[shortSlot].data.size(), 16U * 3U);
    EXPECT_EQ(context.samplingParams[0], RowSamplingParams{}) << "the resident kept its batch's settings";
    EXPECT_EQ(context.numLogprobsPerSlot[0], 0);
    EXPECT_TRUE(context.stepLogprobs[0].data.empty());

    SlotSeed noBudget = makeSeed(/*originalIndex=*/3);
    noBudget.maxGenerateLength = 0;
    EXPECT_THROW(appendTestSlot(context, noBudget, /*requestId=*/703), std::runtime_error);
    EXPECT_EQ(context.activeBatchSize, 3);
    EXPECT_TRUE(context.perSlotSizesConsistent());
    EXPECT_THROW(context.generateBudget(context.activeBatchSize), std::runtime_error);
    EXPECT_THROW(context.generateBudget(-1), std::runtime_error);
}

TEST(DecodingInferenceContextAppendTests, UniformParamsNeedAgreeingSlots)
{
    // Per-launch decoders read the batch through this accessor; a mixed batch must fail there.
    DecodingInferenceContext empty;
    EXPECT_EQ(empty.uniformSamplingParams(), RowSamplingParams{});

    DecodingInferenceContext context = makeRunningBatch();
    context.samplingParams[0] = RowSamplingParams{0.5F, 10, 1.0F};
    SlotSeed agreeing = makeSeed();
    agreeing.sampling = RowSamplingParams{0.5F, 10, 1.0F};
    agreeing.maxGenerateLength = 64;
    appendTestSlot(context, agreeing, /*requestId=*/701);
    EXPECT_EQ(context.uniformSamplingParams(), (RowSamplingParams{0.5F, 10, 1.0F}));

    SlotSeed disagreeing = makeSeed(/*originalIndex=*/2);
    disagreeing.sampling = RowSamplingParams{0.9F, 10, 1.0F};
    appendTestSlot(context, disagreeing, /*requestId=*/702);
    EXPECT_THROW(context.uniformSamplingParams(), std::runtime_error);
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
    seed.requestStableSampling = true;
    seed.sampling = RowSamplingParams{0.7F, 40, 0.9F};
    seed.maxGenerateLength = 9;
    seed.numLogprobs = 2;
    seed.ignoreEos = true;
    seed.jumpForward = true;
    int32_t const slotB = appendTestSlot(context, seed, /*requestId=*/701);
    context.prunedPrefillTokens[static_cast<size_t>(slotB)] = 5; // as B's seated prefill would record
    context.pendingTokens[static_cast<size_t>(slotB)] = 3;       // as a jump would leave it
    context.jumpForwardStats[static_cast<size_t>(slotB)].jumps = 2;

    context.finishedStates[0] = 1; // A is done

    std::vector<int32_t> const mapping = buildBatchMapping(context.finishedStates);
    ASSERT_EQ(mapping[0], -1);
    ASSERT_EQ(mapping[static_cast<size_t>(slotB)], 0);

    // The same moves performBatchEvict makes for the host-side per-slot state.
    compactVector(mapping, context.finishedStates);
    compactVector(mapping, context.currentGenerateLengths);
    compactVector(mapping, context.samplingParams);
    compactVector(mapping, context.samplingSeeds);
    compactVector(mapping, context.requestStableSampling);
    compactVector(mapping, context.maxGenerateLengths);
    compactVector(mapping, context.numLogprobsPerSlot);
    compactVector(mapping, context.ignoreEos);
    compactVector(mapping, context.acceptedDraftTokens);
    compactVector(mapping, context.proposedDraftTokens);
    compactVector(mapping, context.decodeRounds);
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
    compactVector(mapping, context.jumpForwardEnabled);
    compactVector(mapping, context.pendingTokens);
    compactVector(mapping, context.jumpForwardStats);
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
    EXPECT_EQ(context.requestStableSampling[0], 1) << "B inherited A's sampling stream";
    EXPECT_EQ(context.samplingParams[0], (RowSamplingParams{0.7F, 40, 0.9F})) << "B inherited A's sampling";
    EXPECT_EQ(context.maxGenerateLengths[0], 9) << "B inherited A's budget";
    EXPECT_EQ(context.numLogprobsPerSlot[0], 2) << "B inherited A's logprobs width";
    EXPECT_EQ(context.ignoreEos[0], 1) << "B inherited A's EOS policy";
    EXPECT_NE(context.slotStreams[0].channel, nullptr) << "B's delivery channel was lost in the move";
    EXPECT_EQ(context.prunedPrefillTokens[0], 5) << "B inherited A's pruning count";
    EXPECT_EQ(context.jumpForwardEnabled[0], 1) << "B inherited A's jump-forward flag";
    EXPECT_EQ(context.pendingTokens[0], 3) << "B inherited A's pending run";
    EXPECT_EQ(context.jumpForwardStats[0].jumps, 2) << "B inherited A's jump counters";
}

// Given an empty batch whose runtime never recorded the logprobs step overshoot
// When a seed asking for logprobs is appended with a stream channel
// Then the append is refused before the channel is attached, so the same channel is accepted once the
//      width is in range
TEST(DecodingInferenceContextAppendTests, RefusedAppendLeavesChannelUnattached)
{
    DecodingInferenceContext context;
    context.initialize(/*batchSize=*/0, /*maxGenLength=*/64, /*visual=*/{}, /*deepstack=*/{}, /*loraName=*/"",
        /*cudaStream=*/nullptr, /*residentCapacity=*/2);
    SlotSeed seed = makeSeed(/*originalIndex=*/0);
    seed.channel = StreamChannel::create();
    seed.numLogprobs = static_cast<int32_t>(trt_edgellm::kMaxLogprobsK) + 1;
    EXPECT_THROW(appendTestSlot(context, seed, /*requestId=*/701), std::runtime_error);
    EXPECT_EQ(context.activeBatchSize, 0);
    EXPECT_TRUE(context.perSlotSizesConsistent());

    seed.numLogprobs = 2;
    int32_t const slot = appendTestSlot(context, seed, /*requestId=*/701);
    EXPECT_EQ(context.slotStreams[slot].channel, seed.channel) << "the refused append had left the channel attached";
    EXPECT_EQ(context.stepLogprobs[slot].data.size(), 32U * 2U);
}

// swapExecutionRows presents one logical sequence at row zero without moving resident state.

TEST(DecodingInferenceContextSwapTests, ExchangesEveryPerSlotFieldAndIsItsOwnInverse)
{
    DecodingInferenceContext context = makeRunningBatch();
    SlotSeed seed = makeSeed();
    seed.channel = StreamChannel::create();
    seed.logitBias = {{7, 2.0F}};
    seed.sampling = RowSamplingParams{0.7F, 40, 0.9F};
    seed.maxGenerateLength = 9;
    seed.jumpForward = true;
    appendTestSlot(context, seed, /*requestId=*/701);
    EXPECT_TRUE(context.hasJumpForward) << "a slot that asks for jump-forward turns the batch gate on";
    EXPECT_EQ(context.pendingTokens[1], 0) << "a fresh slot has no pending token until its prefill samples one";
    context.thinkingDone[0] = 1; // A left its thinking block; B has not
    context.pendingTokens[1] = 4;

    context.swapExecutionRows(0, 1);
    EXPECT_EQ(context.jumpForwardEnabled[0], 1);
    EXPECT_EQ(context.pendingTokens[0], 4);
    EXPECT_EQ(context.jumpForwardEnabled[1], 0);
    // B's identity, history and delivery state now sit at slot 0; A's at slot 1.
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{21, 22}));
    EXPECT_EQ(context.batchIndexMapping[0], 1);
    EXPECT_NE(context.slotStreams[0].channel, nullptr);
    EXPECT_EQ(context.currentGenerateLengths[1], 2);
    EXPECT_EQ(context.batchIndexMapping[1], 0);
    EXPECT_EQ(context.logitBiasPerSlot[0].count(7), 1U);
    EXPECT_EQ(context.thinkingDone[0], 0);
    EXPECT_EQ(context.thinkingDone[1], 1) << "the thinking latch stayed behind in the swap";
    EXPECT_EQ(context.samplingParams[0], (RowSamplingParams{0.7F, 40, 0.9F})) << "B's sampling moved with it";
    EXPECT_EQ(context.maxGenerateLengths[0], 9);
    EXPECT_EQ(context.maxGenerateLengths[1], 64);
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
