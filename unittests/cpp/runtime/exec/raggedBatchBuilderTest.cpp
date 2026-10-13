/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/exec/raggedBatchBuilder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

RaggedEngineContract makeContract()
{
    return {TokenLayoutBackend::kEntryPaddedCompatibility, 8, 16, 128, 8, false};
}

ScheduledSequence makeSequence(RequestId requestId, CacheSlot slot, uint64_t epoch, SequenceWork work,
    std::vector<int32_t> const& tokens, int32_t pastLength = 0, bool selectLogits = true)
{
    int32_t const size = static_cast<int32_t>(tokens.size());
    return {SequenceIdentity{requestId, ResidentRef{slot, epoch}}, work, size, pastLength,
        HostTokenRange{tokens.data(), size, 0, size}, selectLogits};
}

RaggedExecutionBatch buildBatch(RaggedBatchBuilder& builder, StepId id, std::vector<ScheduledSequence> sequences)
{
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    builder.buildInto(ScheduledStep{id, std::move(sequences)}, batch);
    return batch;
}

std::vector<int32_t> hostTokens(RaggedExecutionBatch const& batch)
{
    int32_t const* data = batch.hostTokenIds.dataPointer<int32_t>();
    return {data, data + batch.shape.physicalTokens};
}

TEST(RaggedBatchBuilderTest, LowersGoldenUnevenContextBatchWithoutOwnerMap)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<std::vector<int32_t>> const tokens{{10, 11, 12}, {20}, {30, 31}};
    auto batch = buildBatch(builder, 55,
        {makeSequence(101, 2, 7, SequenceWork::kContext, tokens[0]),
            makeSequence(102, 0, 4, SequenceWork::kContext, tokens[1]),
            makeSequence(103, 5, 9, SequenceWork::kContext, tokens[2])});

    EXPECT_EQ(batch.stepId, 55U);
    EXPECT_EQ(batch.shape.validTokens, 6);
    EXPECT_EQ(batch.shape.physicalTokens, 9);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 3, 6, 9}));
    EXPECT_EQ(batch.queryLengths, (std::vector<int32_t>{3, 1, 2}));
    EXPECT_EQ(batch.stateIndices, (std::vector<int32_t>{2, 0, 5}));
    EXPECT_EQ(hostTokens(batch), (std::vector<int32_t>{10, 11, 12, 20, 0, 0, 30, 31, 0}));
    EXPECT_EQ(batch.positions, (std::vector<int32_t>{0, 1, 2, 0, -1, -1, 0, 1, -1}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{2, 3, 7}));
    EXPECT_EQ(batch.logitsToSequence, (std::vector<int32_t>{0, 1, 2}));
    ASSERT_EQ(batch.sequenceOrder.size(), 3U);
    EXPECT_EQ(batch.sequenceOrder[0].requestId, 101U);
    EXPECT_EQ(batch.sequenceOrder[2].resident, (ResidentRef{5, 9}));
    EXPECT_NO_THROW(RaggedBatchBuilder::validateExecutionBatch(batch, makeContract()));
}

TEST(RaggedBatchBuilderTest, ReusesPinnedTokenAndVectorStorage)
{
    RaggedBatchBuilder builder{makeContract()};
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    void* const tokenStorage = batch.hostTokenIds.rawPointer();
    auto const* const sequenceStorage = batch.sequenceOrder.data();
    size_t const sequenceCapacity = batch.sequenceOrder.capacity();

    std::vector<int32_t> const context{1, 2, 3};
    builder.buildInto(ScheduledStep{1, {makeSequence(1, 2, 1, SequenceWork::kContext, context)}}, batch);
    std::vector<int32_t> const decode{4};
    builder.buildInto(ScheduledStep{2, {makeSequence(2, 5, 1, SequenceWork::kDecode, decode, 3)}}, batch);

    EXPECT_EQ(batch.hostTokenIds.rawPointer(), tokenStorage);
    EXPECT_EQ(batch.sequenceOrder.data(), sequenceStorage);
    EXPECT_EQ(batch.sequenceOrder.capacity(), sequenceCapacity);
}

TEST(RaggedBatchBuilderTest, ReserveRemainsIdempotentAfterLogicalReshape)
{
    RaggedBatchBuilder builder{makeContract()};
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    void* const tokenStorage = batch.hostTokenIds.rawPointer();
    int64_t const tokenCapacity = batch.hostTokenIds.getMemoryCapacity();
    size_t const sequenceCapacity = batch.sequenceOrder.capacity();

    std::vector<int32_t> const tokens{1, 2, 3};
    builder.buildInto(ScheduledStep{1, {makeSequence(1, 2, 1, SequenceWork::kContext, tokens)}}, batch);

    EXPECT_NO_THROW(builder.reserve(batch));
    EXPECT_EQ(batch.hostTokenIds.rawPointer(), tokenStorage);
    EXPECT_EQ(batch.hostTokenIds.getMemoryCapacity(), tokenCapacity);
    EXPECT_EQ(batch.sequenceOrder.capacity(), sequenceCapacity);
}

TEST(RaggedBatchBuilderTest, PreservesCanonicalScheduledOrderAndSparseLogitRouting)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<int32_t> const a{1};
    std::vector<int32_t> const b{2};
    std::vector<int32_t> const c{3};
    auto batch = buildBatch(builder, 3,
        {makeSequence(103, 5, 1, SequenceWork::kDecode, c, 7),
            makeSequence(101, 2, 1, SequenceWork::kDecode, a, 8, false),
            makeSequence(102, 0, 1, SequenceWork::kDecode, b, 9)});

    EXPECT_EQ(batch.sequenceOrder[0].requestId, 103U);
    EXPECT_EQ(batch.sequenceOrder[1].requestId, 101U);
    EXPECT_EQ(batch.sequenceOrder[2].requestId, 102U);
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{0, 2}));
    EXPECT_EQ(batch.logitsToSequence, (std::vector<int32_t>{0, 2}));
}

TEST(RaggedBatchBuilderTest, RuntimeAdapterRequiresLiveContextOrderAndDenseLogits)
{
    std::vector<int32_t> const a{1};
    std::vector<int32_t> const b{2};
    ScheduledStep canonical{3,
        {makeSequence(101, 2, 7, SequenceWork::kDecode, a, 8), makeSequence(102, 5, 9, SequenceWork::kDecode, b, 9)}};
    std::vector<RequestId> const requestIds{101, 102};
    std::vector<ResidentRef> const residents{{2, 7}, {5, 9}};

    EXPECT_NO_THROW(RaggedBatchBuilder::validateRuntimeAdapterStep(canonical, 2, requestIds, residents));

    ScheduledStep projected{4, {canonical.sequences[1]}};
    std::vector<RequestId> const projectedRequestIds{102, 101};
    std::vector<ResidentRef> const projectedResidents{{5, 9}, {2, 7}};
    EXPECT_NO_THROW(
        RaggedBatchBuilder::validateRuntimeAdapterStep(projected, 1, projectedRequestIds, projectedResidents));

    ScheduledStep reordered = canonical;
    std::swap(reordered.sequences[0], reordered.sequences[1]);
    EXPECT_THROW(
        RaggedBatchBuilder::validateRuntimeAdapterStep(reordered, 2, requestIds, residents), std::runtime_error);

    ScheduledStep sparse = canonical;
    sparse.sequences[1].selectLastTokenLogits = false;
    EXPECT_THROW(RaggedBatchBuilder::validateRuntimeAdapterStep(sparse, 2, requestIds, residents), std::runtime_error);

    EXPECT_THROW(
        RaggedBatchBuilder::validateRuntimeAdapterStep(canonical, 1, requestIds, residents), std::runtime_error);
}

TEST(RaggedBatchBuilderTest, FinalizesSubsetSelectionWithDensePositions)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<int32_t> const tokens{10, 11, 12, 13, 14};
    ScheduledStep step{4, {makeSequence(101, 2, 7, SequenceWork::kContext, tokens, 6)}};
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    builder.buildInto(step, batch);
    std::vector<int32_t> const starts{0, 3};
    std::vector<int32_t> const keep{0, 2, 4};

    builder.finalizeSubsetSelection(step, starts, keep, batch);

    EXPECT_EQ(hostTokens(batch), (std::vector<int32_t>{10, 12, 14}));
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 3}));
    EXPECT_EQ(batch.queryLengths, (std::vector<int32_t>{3}));
    EXPECT_EQ(batch.positions, (std::vector<int32_t>{6, 7, 8}));
    EXPECT_EQ(batch.attentionSequenceLengths, (std::vector<int32_t>{9}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{2}));
}

TEST(RaggedBatchBuilderTest, RejectsInvalidIdentityRangesMixedWorkAndKeepMaps)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<int32_t> const one{1};
    std::vector<int32_t> const two{2, 3};
    EXPECT_THROW(
        buildBatch(builder, 5,
            {makeSequence(1, 0, 1, SequenceWork::kDecode, one), makeSequence(1, 1, 1, SequenceWork::kDecode, one)}),
        std::runtime_error);
    EXPECT_THROW(
        buildBatch(builder, 5,
            {makeSequence(1, 0, 1, SequenceWork::kContext, two), makeSequence(2, 1, 1, SequenceWork::kDecode, one)}),
        std::runtime_error);

    auto invalid = makeSequence(1, 0, 1, SequenceWork::kContext, two);
    invalid.queryTokens.extent = 1;
    EXPECT_THROW(buildBatch(builder, 5, {invalid}), std::runtime_error);

    auto invalidEpoch = makeSequence(3, 2, 0, SequenceWork::kContext, two);
    EXPECT_THROW(buildBatch(builder, 5, {invalidEpoch}), std::runtime_error);

    ScheduledStep step{5, {makeSequence(1, 0, 1, SequenceWork::kContext, two)}};
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    builder.buildInto(step, batch);
    std::vector<int32_t> const starts{0, 2};
    std::vector<int32_t> const duplicate{1, 1};
    EXPECT_THROW(builder.finalizeSubsetSelection(step, starts, duplicate, batch), std::runtime_error);
    step.id = 6;
    std::vector<int32_t> const keep{0};
    std::vector<int32_t> const oneStart{0, 1};
    EXPECT_THROW(builder.finalizeSubsetSelection(step, oneStart, keep, batch), std::runtime_error);
}

TEST(RaggedBatchBuilderTest, RejectsDuplicateAndOutOfRangeResidentSlotsIndependentlyOfRequestIdentity)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<int32_t> const token{1};

    EXPECT_THROW(buildBatch(builder, 5,
                     {makeSequence(101, 3, 1, SequenceWork::kDecode, token),
                         makeSequence(202, 3, 7, SequenceWork::kDecode, token)}),
        std::runtime_error);
    EXPECT_THROW(buildBatch(builder, 6, {makeSequence(303, -1, 1, SequenceWork::kDecode, token)}), std::runtime_error);
    EXPECT_THROW(buildBatch(builder, 7, {makeSequence(404, 8, 1, SequenceWork::kDecode, token)}), std::runtime_error);
}

TEST(RaggedBatchBuilderTest, ValidatesCompletionIdentityEpochAndPastLength)
{
    RaggedBatchBuilder builder{makeContract()};
    std::vector<int32_t> const a{1};
    std::vector<int32_t> const b{2};
    auto batch = buildBatch(builder, 69,
        {makeSequence(1101, 2, 7, SequenceWork::kDecode, a, 4), makeSequence(1102, 0, 9, SequenceWork::kDecode, b, 6)});
    std::vector<CompletionSequence> current{{1101, {2, 7}, 4}, {1102, {0, 9}, 6}};

    EXPECT_NO_THROW(batch.validateCommitSnapshot(69, current));
    current[0].resident.epoch = 8;
    EXPECT_THROW(batch.validateCommitSnapshot(69, current), std::runtime_error);
}

} // namespace
