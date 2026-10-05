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

#include "runtime/decoding/decoderUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

using namespace trt_edgellm;

namespace
{

rt::DecodingInferenceContext makeContext(int32_t batchSize)
{
    rt::DecodingInferenceContext context;
    context.initialize(batchSize, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    return context;
}

//! Sized by the context so an eviction reindexes it with every other per-slot vector; a
//! `handleRequest` local would be renumbered against them and silently mask the wrong slot.
TEST(DecodingContextCompactionTest, ThinkingDoneIsSizedByInitialize)
{
    auto context = makeContext(3);
    EXPECT_EQ(context.thinkingDone.size(), 3U);
    EXPECT_EQ(context.thinkingDone, (std::vector<int8_t>{0, 0, 0}));
    EXPECT_EQ(context.guidedReasoningEnded.size(), 3U);
}

//! Evicting the middle slot of a 3-batch must keep slots 0 and 2 paired with their own
//! thinking state. Before the fix, slot 2's state stayed at index 2 (out of range of the
//! compacted batch) and slot 2 inherited whatever the evicted slot 1 had left behind.
TEST(DecodingContextCompactionTest, ThinkingDoneFollowsItsRequestAcrossEviction)
{
    auto context = makeContext(3);
    // Slot 0 finished thinking, slot 1 (about to be evicted) did not, slot 2 finished.
    context.thinkingDone = {1, 0, 1};
    context.finishedStates = {0, 1, 0};
    context.currentGenerateLengths = {5, 9, 7};

    auto const batchMapping = rt::buildBatchMapping(context.finishedStates);
    ASSERT_EQ(batchMapping, (std::vector<int32_t>{0, -1, 1}));

    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.thinkingDone);
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    context.activeBatchSize = 2;

    // Surviving slots keep their own thinking state; nothing shifted in from the evicted slot.
    EXPECT_EQ(context.thinkingDone, (std::vector<int8_t>{1, 1}));
    // Cross-check against another per-slot vector compacted with the same mapping.
    EXPECT_EQ(context.currentGenerateLengths, (std::vector<int32_t>{5, 7}));
    EXPECT_EQ(context.thinkingDone.size(), context.currentGenerateLengths.size());
}

//! The failure this guards against is an index-space mismatch, which shows up as a length
//! mismatch between the per-slot vectors after an eviction.
TEST(DecodingContextCompactionTest, ThinkingDoneStaysInStepWithOtherPerSlotVectors)
{
    auto context = makeContext(4);
    context.thinkingDone = {0, 1, 1, 0};
    context.guidedReasoningEnded = {0, 1, 0, 1};
    context.finishedStates = {1, 0, 0, 1};

    auto const batchMapping = rt::buildBatchMapping(context.finishedStates);
    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.thinkingDone);
    rt::compactVector(batchMapping, context.guidedReasoningEnded);
    rt::compactVector(batchMapping, context.stopStringsPerSlot);
    rt::compactVector(batchMapping, context.logitBiasPerSlot);
    rt::compactVector(batchMapping, context.slotStreams);

    EXPECT_EQ(context.thinkingDone, (std::vector<int8_t>{1, 1}));
    // Distinct from thinkingDone on purpose: only the reasoning-end marker sets this one, so
    // an eviction that renumbered it against the rest would mask the wrong slot.
    EXPECT_EQ(context.guidedReasoningEnded, (std::vector<int8_t>{1, 0}));
    EXPECT_EQ(context.thinkingDone.size(), context.finishedStates.size());
    EXPECT_EQ(context.guidedReasoningEnded.size(), context.finishedStates.size());
    EXPECT_EQ(context.thinkingDone.size(), context.stopStringsPerSlot.size());
    EXPECT_EQ(context.thinkingDone.size(), context.logitBiasPerSlot.size());
    EXPECT_EQ(context.thinkingDone.size(), context.slotStreams.size());
}

//! A slot marked finished earlier in the same step (cancellation, or a guided-decoding
//! failure) is evicted at the end of it, so a token appended to it is never fed back and only
//! pollutes the output. Prefill and collectLogprobsFromHost both skip such slots; decode used
//! to append unconditionally, which left tokenIds one ahead of stepLogprobs.
TEST(AppendSampledTokensTest, SkipsSlotsAlreadyFinishedThisStep)
{
    auto context = makeContext(3);
    context.finishedStates = {0, 1, 0};
    context.currentGenerateLengths = {4, 4, 4};

    std::vector<int32_t> const sampled{11, 22, 33};
    rt::decoder_utils::appendSampledTokens(context, sampled.data(), 3);

    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{11}));
    EXPECT_TRUE(context.tokenIds[1].empty()) << "a finished slot must not take another token";
    EXPECT_EQ(context.tokenIds[2], (std::vector<int32_t>{33}));
    EXPECT_EQ(context.currentGenerateLengths, (std::vector<int32_t>{5, 4, 5}));
}

//! With nothing finished, every active slot advances by exactly one token.
TEST(AppendSampledTokensTest, AppendsOneTokenPerActiveSlot)
{
    auto context = makeContext(2);
    context.finishedStates = {0, 0};
    context.currentGenerateLengths = {0, 7};

    std::vector<int32_t> const sampled{5, 6};
    rt::decoder_utils::appendSampledTokens(context, sampled.data(), 2);

    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{5}));
    EXPECT_EQ(context.tokenIds[1], (std::vector<int32_t>{6}));
    EXPECT_EQ(context.currentGenerateLengths, (std::vector<int32_t>{1, 8}));
}

} // namespace
