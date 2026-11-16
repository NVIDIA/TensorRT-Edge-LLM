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

#include "runtime/exec/raggedBatchBuilder.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <limits>
#include <string>

namespace trt_edgellm
{
namespace rt
{

void IdentitySetScratch::reserve(size_t maxItems)
{
    size_t tableSize = 1;
    while (tableSize < maxItems * 2)
    {
        tableSize *= 2;
    }
    if (mKeys.size() >= tableSize)
    {
        return;
    }
    mKeys.resize(tableSize);
    mGenerations.assign(tableSize, 0);
    mGeneration = 1;
    mMask = tableSize - 1;
}

void IdentitySetScratch::clear()
{
    ++mGeneration;
    if (mGeneration == 0)
    {
        std::fill(mGenerations.begin(), mGenerations.end(), 0);
        mGeneration = 1;
    }
}

bool IdentitySetScratch::insert(uint64_t value)
{
    ELLM_CHECK(!mKeys.empty(), "identity scratch table is not reserved");
    uint64_t hash = value + 0x9E3779B97F4A7C15ULL;
    hash = (hash ^ (hash >> 30)) * 0xBF58476D1CE4E5B9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94D049BB133111EBULL;
    hash ^= hash >> 31;
    size_t index = static_cast<size_t>(hash) & mMask;
    while (mGenerations[index] == mGeneration)
    {
        if (mKeys[index] == value)
        {
            return false;
        }
        index = (index + 1) & mMask;
    }
    mKeys[index] = value;
    mGenerations[index] = mGeneration;
    return true;
}

namespace
{

constexpr int32_t kPaddingTokenId = 0;
constexpr int32_t kPaddingSentinel = -1;

void validateContract(RaggedEngineContract const& contract)
{
    bool const compact = contract.backend == TokenLayoutBackend::kNativeCompactRagged;
    ELLM_CHECK(contract.backend == TokenLayoutBackend::kEntryPaddedCompatibility || compact,
        "unsupported token layout backend");
    ELLM_CHECK(contract.maxNumSequences > 0, "maxNumSequences must be positive");
    ELLM_CHECK(contract.maxQueryLength > 0, "maxQueryLength must be positive");
    ELLM_CHECK(contract.maxPhysicalTokens > 0, "maxPhysicalTokens must be positive");
    ELLM_CHECK(contract.recurrentPoolRows > 0, "recurrentPoolRows must be positive");
    if (compact)
    {
        ELLM_CHECK(contract.mixedStepSupported, "the compact ragged backend requires mixed-step support");
    }
    else
    {
        ELLM_CHECK(!contract.mixedStepSupported, "mixed execution is not supported by the entry-padded ragged backend");
    }
}

int32_t checkedInt32(int64_t value, char const* description)
{
    ELLM_CHECK(value >= 0 && value <= std::numeric_limits<int32_t>::max(),
        std::string(description) + " exceeds int32 capacity");
    return static_cast<int32_t>(value);
}

void checkScheduledSequence(ScheduledSequence const& sequence, RaggedEngineContract const& contract,
    IdentitySetScratch& requestIdentities, IdentitySetScratch& residentSlots)
{
    ELLM_CHECK(sequence.identity.requestId != 0, "request ID zero is reserved");
    ELLM_CHECK(sequence.identity.resident.epoch != 0, "resident epoch zero is reserved");
    ELLM_CHECK(sequence.identity.resident.slot >= 0 && sequence.identity.resident.slot < contract.recurrentPoolRows,
        "resident slot is outside the configured pool");
    ELLM_CHECK(requestIdentities.insert(sequence.identity.requestId), "duplicate request ID in scheduled step");
    ELLM_CHECK(residentSlots.insert(static_cast<uint64_t>(sequence.identity.resident.slot)),
        "duplicate resident slot in scheduled step");
    ELLM_CHECK(sequence.queryLength > 0, "query length must be positive");
    ELLM_CHECK(sequence.queryLength <= contract.maxQueryLength, "query length exceeds engine capacity");
    ELLM_CHECK(sequence.pastLength >= 0, "past length must not be negative");
    ELLM_CHECK(sequence.queryTokens.data != nullptr, "token range data must not be null");
    ELLM_CHECK(sequence.queryTokens.extent >= 0 && sequence.queryTokens.begin >= 0 && sequence.queryTokens.count > 0,
        "token range bounds must be non-negative and non-empty");
    ELLM_CHECK(sequence.queryTokens.count == sequence.queryLength, "token range count must equal query length");
    ELLM_CHECK(
        static_cast<int64_t>(sequence.queryTokens.begin) + sequence.queryTokens.count <= sequence.queryTokens.extent,
        "token range exceeds its source extent");
    ELLM_CHECK(static_cast<int64_t>(sequence.pastLength) + sequence.queryLength <= std::numeric_limits<int32_t>::max(),
        "resulting sequence length exceeds int32 capacity");
}

} // namespace

RaggedBatchBuilder::RaggedBatchBuilder(RaggedEngineContract contract)
    : mContract(contract)
{
    validateContract(mContract);
}

void RaggedBatchBuilder::reserve(RaggedExecutionBatch& batch)
{
    validateContract(mContract);
    size_t const maxSequences = static_cast<size_t>(mContract.maxNumSequences);
    size_t const maxPhysicalTokens = static_cast<size_t>(mContract.maxPhysicalTokens);
    int64_t const requiredTokenBytes
        = static_cast<int64_t>(mContract.maxPhysicalTokens) * static_cast<int64_t>(sizeof(int32_t));
    if (batch.hostTokenIds.isEmpty())
    {
        batch.hostTokenIds = Tensor({mContract.maxPhysicalTokens}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "RaggedExecutionBatch::hostTokenIds");
    }
    else
    {
        ELLM_CHECK(batch.hostTokenIds.getDataType() == trt_edgellm::DataType::kINT32
                && batch.hostTokenIds.getDeviceType() == DeviceType::kCPU
                && batch.hostTokenIds.getMemoryCapacity() >= requiredTokenBytes,
            "existing host token storage does not satisfy the ragged contract");
    }
    batch.sequenceOrder.reserve(maxSequences);
    batch.positions.reserve(maxPhysicalTokens);
    batch.queryStartOffsets.reserve(maxSequences + 1);
    batch.queryLengths.reserve(maxSequences);
    batch.pastLengths.reserve(maxSequences);
    batch.attentionSequenceLengths.reserve(maxSequences);
    batch.stateIndices.reserve(maxSequences);
    batch.logitsIndices.reserve(maxSequences);
    batch.logitsToSequence.reserve(maxSequences);
    batch.sequenceWorks.reserve(maxSequences);
    mRequestIdentityScratch.reserve(maxSequences);
    mResidentSlotScratch.reserve(maxSequences);
}

void RaggedBatchBuilder::buildInto(ScheduledStep const& step, RaggedExecutionBatch& batch)
{
    validateContract(mContract);
    if (mContract.backend == TokenLayoutBackend::kNativeCompactRagged)
    {
        buildCompactInto(step, batch);
        return;
    }
    size_t const maxSequences = static_cast<size_t>(mContract.maxNumSequences);
    mRequestIdentityScratch.reserve(maxSequences);
    mResidentSlotScratch.reserve(maxSequences);
    ELLM_CHECK(!step.sequences.empty(), "scheduled step must contain at least one sequence");
    ELLM_CHECK(step.sequences.data() != nullptr, "scheduled sequence view must not be null");
    ELLM_CHECK(step.sequences.size() <= static_cast<size_t>(mContract.maxNumSequences),
        "scheduled step exceeds sequence capacity");

    int32_t const numSequences = checkedInt32(static_cast<int64_t>(step.sequences.size()), "sequence count");
    int32_t queryWidth = 0;
    int64_t validTokens = 0;
    int32_t numContextSequences = 0;
    int64_t numContextTokens = 0;
    int32_t numDecodeSequences = 0;
    mRequestIdentityScratch.clear();
    mResidentSlotScratch.clear();

    for (int32_t sequenceIndex = 0; sequenceIndex < numSequences; ++sequenceIndex)
    {
        ScheduledSequence const& sequence = step.sequences[static_cast<size_t>(sequenceIndex)];
        checkScheduledSequence(sequence, mContract, mRequestIdentityScratch, mResidentSlotScratch);

        if (sequence.work == SequenceWork::kContext)
        {
            ++numContextSequences;
            numContextTokens += sequence.queryLength;
        }
        else
        {
            ELLM_CHECK(sequence.work == SequenceWork::kDecode, "unknown sequence work kind");
            ELLM_CHECK(sequence.queryLength == 1, "vanilla decode requires query length one");
            ++numDecodeSequences;
        }
        queryWidth = std::max(queryWidth, sequence.queryLength);
        validTokens += sequence.queryLength;
        ELLM_CHECK(validTokens <= std::numeric_limits<int32_t>::max(), "valid token count exceeds int32 capacity");
        ELLM_CHECK(
            numContextTokens <= std::numeric_limits<int32_t>::max(), "context token count exceeds int32 capacity");
    }

    ELLM_CHECK(numContextSequences == 0 || numDecodeSequences == 0, "mixed context and decode steps are not supported");
    int64_t const physicalTokens64 = static_cast<int64_t>(numSequences) * queryWidth;
    int32_t const physicalTokens = checkedInt32(physicalTokens64, "physical token count");
    ELLM_CHECK(physicalTokens <= mContract.maxPhysicalTokens, "scheduled step exceeds physical token capacity");

    batch.layout = mContract.backend;
    batch.placeholderLogitsRows = 0;
    batch.sequenceOrder.clear();
    batch.positions.clear();
    batch.queryStartOffsets.clear();
    batch.queryLengths.clear();
    batch.pastLengths.clear();
    batch.attentionSequenceLengths.clear();
    batch.stateIndices.clear();
    batch.logitsIndices.clear();
    batch.logitsToSequence.clear();
    batch.sequenceWorks.clear();
    batch.stepId = step.id;
    batch.shape = RaggedStepShape{numSequences, checkedInt32(validTokens, "valid token count"), physicalTokens,
        queryWidth, numContextSequences, checkedInt32(numContextTokens, "context token count"), 0};
    check::check(batch.hostTokenIds.reshape({physicalTokens}), "host token storage reshape failed");
    std::fill_n(batch.hostTokenIds.dataPointer<int32_t>(), physicalTokens, kPaddingTokenId);
    batch.positions.assign(static_cast<size_t>(physicalTokens), kPaddingSentinel);
    batch.queryStartOffsets.resize(static_cast<size_t>(numSequences) + 1);

    for (int32_t forwardIndex = 0; forwardIndex < numSequences; ++forwardIndex)
    {
        ScheduledSequence const& sequence = step.sequences[static_cast<size_t>(forwardIndex)];

        int32_t const physicalStart = forwardIndex * queryWidth;
        batch.queryStartOffsets[static_cast<size_t>(forwardIndex)] = physicalStart;
        batch.queryLengths.push_back(sequence.queryLength);
        batch.pastLengths.push_back(sequence.pastLength);
        batch.attentionSequenceLengths.push_back(sequence.pastLength + sequence.queryLength);
        batch.stateIndices.push_back(sequence.identity.resident.slot);
        batch.sequenceOrder.push_back(sequence.identity);
        batch.sequenceWorks.push_back(sequence.work);

        for (int32_t tokenIndex = 0; tokenIndex < sequence.queryLength; ++tokenIndex)
        {
            int32_t const physicalIndex = physicalStart + tokenIndex;
            batch.hostTokenIds.dataPointer<int32_t>()[physicalIndex]
                = sequence.queryTokens.data[sequence.queryTokens.begin + tokenIndex];
            batch.positions[static_cast<size_t>(physicalIndex)] = sequence.pastLength + tokenIndex;
        }
        if (sequence.selectLastTokenLogits)
        {
            batch.logitsIndices.push_back(physicalStart + sequence.queryLength - 1);
            batch.logitsToSequence.push_back(forwardIndex);
        }
    }
    batch.queryStartOffsets[static_cast<size_t>(numSequences)] = physicalTokens;
    batch.shape.numLogits = checkedInt32(static_cast<int64_t>(batch.logitsIndices.size()), "logits count");

#ifndef NDEBUG
    validateExecutionBatch(batch, mContract);
#endif
}

void RaggedBatchBuilder::buildCompactInto(ScheduledStep const& step, RaggedExecutionBatch& batch)
{
    size_t const maxSequences = static_cast<size_t>(mContract.maxNumSequences);
    mRequestIdentityScratch.reserve(maxSequences);
    mResidentSlotScratch.reserve(maxSequences);
    ELLM_CHECK(!step.sequences.empty(), "scheduled step must contain at least one sequence");
    ELLM_CHECK(step.sequences.size() <= maxSequences, "scheduled step exceeds sequence capacity");

    int32_t const numSequences = checkedInt32(static_cast<int64_t>(step.sequences.size()), "sequence count");
    int32_t queryWidth = 0;
    int64_t validTokens = 0;
    int32_t numContextSequences = 0;
    int64_t numContextTokens = 0;
    bool decodeSeen = false;
    mRequestIdentityScratch.clear();
    mResidentSlotScratch.clear();
    for (ScheduledSequence const& sequence : step.sequences)
    {
        checkScheduledSequence(sequence, mContract, mRequestIdentityScratch, mResidentSlotScratch);
        if (sequence.work == SequenceWork::kContext)
        {
            ELLM_CHECK(!decodeSeen, "compact steps require every context sequence before every decode sequence");
            ++numContextSequences;
            numContextTokens += sequence.queryLength;
        }
        else
        {
            ELLM_CHECK(sequence.work == SequenceWork::kDecode, "unknown sequence work kind");
            ELLM_CHECK(sequence.queryLength == 1, "vanilla decode requires query length one");
            decodeSeen = true;
        }
        queryWidth = std::max(queryWidth, sequence.queryLength);
        validTokens += sequence.queryLength;
        ELLM_CHECK(validTokens <= mContract.maxPhysicalTokens, "scheduled step exceeds physical token capacity");
    }
    int32_t const numTokens = static_cast<int32_t>(validTokens);

    batch.layout = TokenLayoutBackend::kNativeCompactRagged;
    batch.stepId = step.id;
    batch.shape = RaggedStepShape{numSequences, numTokens, numTokens, queryWidth, numContextSequences,
        static_cast<int32_t>(numContextTokens), numSequences};
    batch.sequenceOrder.clear();
    batch.queryLengths.clear();
    batch.pastLengths.clear();
    batch.attentionSequenceLengths.clear();
    batch.stateIndices.clear();
    batch.logitsIndices.clear();
    batch.logitsToSequence.clear();
    batch.sequenceWorks.clear();
    check::check(batch.hostTokenIds.reshape({numTokens}), "host token storage reshape failed");
    int32_t* const tokenIds = batch.hostTokenIds.dataPointer<int32_t>();
    batch.positions.resize(static_cast<size_t>(numTokens));
    batch.queryStartOffsets.resize(static_cast<size_t>(numSequences) + 1);

    int32_t rowStart = 0;
    for (int32_t row = 0; row < numSequences; ++row)
    {
        ScheduledSequence const& sequence = step.sequences[static_cast<size_t>(row)];
        batch.queryStartOffsets[static_cast<size_t>(row)] = rowStart;
        batch.queryLengths.push_back(sequence.queryLength);
        batch.pastLengths.push_back(sequence.pastLength);
        batch.attentionSequenceLengths.push_back(sequence.pastLength + sequence.queryLength);
        batch.stateIndices.push_back(sequence.identity.resident.slot);
        batch.sequenceOrder.push_back(sequence.identity);
        batch.sequenceWorks.push_back(sequence.work);
        std::copy_n(sequence.queryTokens.data + sequence.queryTokens.begin, sequence.queryLength, tokenIds + rowStart);
        for (int32_t token = 0; token < sequence.queryLength; ++token)
        {
            batch.positions[static_cast<size_t>(rowStart + token)] = sequence.pastLength + token;
        }
        rowStart += sequence.queryLength;
        if (sequence.selectLastTokenLogits)
        {
            batch.logitsIndices.push_back(rowStart - 1);
            batch.logitsToSequence.push_back(row);
        }
    }
    batch.queryStartOffsets[static_cast<size_t>(numSequences)] = numTokens;
    batch.placeholderLogitsRows = numSequences - static_cast<int32_t>(batch.logitsToSequence.size());
    for (int32_t row = 0; row < numSequences; ++row)
    {
        if (!step.sequences[static_cast<size_t>(row)].selectLastTokenLogits)
        {
            batch.logitsIndices.push_back(batch.queryStartOffsets[static_cast<size_t>(row) + 1] - 1);
        }
    }

#ifndef NDEBUG
    validateExecutionBatch(batch, mContract);
#endif
}

void RaggedBatchBuilder::finalizeSubsetSelection(ScheduledStep const& sourceStep,
    std::vector<int32_t> const& keepStartOffsets, std::vector<int32_t> const& concatenatedKeepIndices,
    RaggedExecutionBatch& batch) const
{
    validateContract(mContract);
    int32_t const numSequences = batch.shape.numSequences;
    ELLM_CHECK(batch.layout == TokenLayoutBackend::kEntryPaddedCompatibility,
        "subset finalization requires the entry-padded backend");
    ELLM_CHECK(sourceStep.id == batch.stepId, "subset source step ID does not match the provisional batch");
    ELLM_CHECK(static_cast<int32_t>(sourceStep.sequences.size()) == numSequences,
        "subset source step does not match the provisional batch");
    ELLM_CHECK(
        keepStartOffsets.size() == static_cast<size_t>(numSequences) + 1, "subset keep offsets have the wrong extent");
    ELLM_CHECK(keepStartOffsets[0] == 0, "subset keep offsets must start at zero");
    ELLM_CHECK(keepStartOffsets[numSequences] >= 0
            && static_cast<size_t>(keepStartOffsets[numSequences]) == concatenatedKeepIndices.size(),
        "subset keep offsets do not cover the keep-index storage");

    int32_t const oldQueryWidth = batch.shape.queryWidth;
    int32_t newQueryWidth = 0;
    int64_t validTokens = 0;
    int64_t contextTokens = 0;
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        size_t const sequenceIndex = static_cast<size_t>(sequence);
        ScheduledSequence const& source = sourceStep.sequences[sequenceIndex];
        ELLM_CHECK(source.identity.requestId == batch.sequenceOrder[sequenceIndex].requestId
                && source.identity.resident == batch.sequenceOrder[sequenceIndex].resident
                && source.work == batch.sequenceWorks[sequenceIndex]
                && source.pastLength == batch.pastLengths[sequenceIndex]
                && source.queryLength == batch.queryLengths[sequenceIndex],
            "subset source step does not describe the provisional execution snapshot");
        int32_t const begin = keepStartOffsets[sequence];
        int32_t const end = keepStartOffsets[sequence + 1];
        ELLM_CHECK(begin >= 0 && end > begin && end <= keepStartOffsets[numSequences],
            "each subset sequence must retain a non-empty ordered keep range");
        int32_t const oldLength = batch.queryLengths[static_cast<size_t>(sequence)];
        int32_t previous = -1;
        for (int32_t cursor = begin; cursor < end; ++cursor)
        {
            int32_t const index = concatenatedKeepIndices[static_cast<size_t>(cursor)];
            ELLM_CHECK(index > previous && index < oldLength,
                "subset keep indices must be strictly increasing and inside the source query");
            previous = index;
        }
        int32_t const kept = end - begin;
        newQueryWidth = std::max(newQueryWidth, kept);
        validTokens += kept;
        if (batch.sequenceWorks[static_cast<size_t>(sequence)] == SequenceWork::kContext)
        {
            contextTokens += kept;
        }
    }

    int32_t const physicalTokens
        = checkedInt32(static_cast<int64_t>(numSequences) * newQueryWidth, "final subset physical token count");
    ELLM_CHECK(physicalTokens <= mContract.maxPhysicalTokens, "final subset exceeds physical token capacity");
    int32_t* tokenIds = batch.hostTokenIds.dataPointer<int32_t>();
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        int32_t const keepBegin = keepStartOffsets[sequence];
        int32_t const keepEnd = keepStartOffsets[sequence + 1];
        int32_t const sourceStart = sequence * oldQueryWidth;
        int32_t const destinationStart = sequence * newQueryWidth;
        for (int32_t cursor = keepBegin; cursor < keepEnd; ++cursor)
        {
            int32_t const destination = destinationStart + cursor - keepBegin;
            int32_t const source = sourceStart + concatenatedKeepIndices[static_cast<size_t>(cursor)];
            tokenIds[destination] = tokenIds[source];
        }
    }

    batch.positions.assign(static_cast<size_t>(physicalTokens), kPaddingSentinel);
    batch.logitsIndices.clear();
    batch.logitsToSequence.clear();
    batch.shape.validTokens = checkedInt32(validTokens, "final subset valid token count");
    batch.shape.physicalTokens = physicalTokens;
    batch.shape.queryWidth = newQueryWidth;
    batch.shape.numContextTokens = checkedInt32(contextTokens, "final subset context token count");
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        size_t const index = static_cast<size_t>(sequence);
        int32_t const kept = keepStartOffsets[sequence + 1] - keepStartOffsets[sequence];
        int32_t const start = sequence * newQueryWidth;
        batch.queryStartOffsets[index] = start;
        batch.queryLengths[index] = kept;
        batch.attentionSequenceLengths[index] = batch.pastLengths[index] + kept;
        for (int32_t token = 0; token < kept; ++token)
        {
            batch.positions[static_cast<size_t>(start + token)] = batch.pastLengths[index] + token;
        }
        std::fill(tokenIds + start + kept, tokenIds + start + newQueryWidth, kPaddingTokenId);
        if (sourceStep.sequences[index].selectLastTokenLogits)
        {
            batch.logitsIndices.push_back(start + kept - 1);
            batch.logitsToSequence.push_back(sequence);
        }
    }
    batch.queryStartOffsets[static_cast<size_t>(numSequences)] = physicalTokens;
    batch.shape.numLogits = checkedInt32(batch.logitsIndices.size(), "final subset logits count");
    check::check(batch.hostTokenIds.reshape({physicalTokens}), "final host token storage reshape failed");

#ifndef NDEBUG
    validateExecutionBatch(batch, mContract);
#endif
}

void RaggedBatchBuilder::validateRuntimeAdapterStep(ScheduledStep const& step, int32_t activeBatchSize,
    std::vector<RequestId> const& requestIds, std::vector<ResidentRef> const& residentRefs)
{
    ELLM_CHECK(activeBatchSize >= 0 && static_cast<size_t>(activeBatchSize) <= requestIds.size()
            && static_cast<size_t>(activeBatchSize) <= residentRefs.size(),
        "Runtime ragged active batch exceeds resident context storage");
    ELLM_CHECK(step.sequences.size() == static_cast<size_t>(activeBatchSize),
        "Runtime ragged step must contain one sequence per active execution row");
    for (size_t index = 0; index < step.sequences.size(); ++index)
    {
        ScheduledSequence const& sequence = step.sequences[index];
        ELLM_CHECK(
            sequence.identity.requestId == requestIds[index] && sequence.identity.resident == residentRefs[index],
            "Runtime ragged step order must match the live context order");
        ELLM_CHECK(sequence.selectLastTokenLogits, "MR1 runtime requires one selected logits row per active sequence");
    }
}

namespace
{

void validateCompactExecutionBatch(RaggedExecutionBatch const& batch, RaggedEngineContract const& contract)
{
    RaggedStepShape const& shape = batch.shape;
    ELLM_CHECK(batch.layout == contract.backend, "execution layout does not match the engine contract");
    ELLM_CHECK(shape.numSequences > 0 && shape.numSequences <= contract.maxNumSequences,
        "execution sequence count is outside engine capacity");
    ELLM_CHECK(shape.physicalTokens == shape.validTokens && shape.validTokens >= shape.numSequences
            && shape.physicalTokens <= contract.maxPhysicalTokens,
        "compact execution token shape is invalid");
    ELLM_CHECK(shape.numContextSequences >= 0 && shape.numContextSequences <= shape.numSequences,
        "compact context sequence count is outside the execution batch");
    ELLM_CHECK(shape.numContextTokens == shape.validTokens - (shape.numSequences - shape.numContextSequences),
        "compact context tokens do not leave one row per decode sequence");

    size_t const numSequences = static_cast<size_t>(shape.numSequences);
    ELLM_CHECK(!batch.hostTokenIds.isEmpty() && batch.hostTokenIds.getDeviceType() == DeviceType::kCPU
            && batch.hostTokenIds.getDataType() == trt_edgellm::DataType::kINT32
            && batch.hostTokenIds.getShape().volume() == shape.physicalTokens,
        "host token ID storage has the wrong shape or type");
    ELLM_CHECK(
        batch.positions.size() == static_cast<size_t>(shape.physicalTokens), "position storage has the wrong size");
    ELLM_CHECK(batch.queryStartOffsets.size() == numSequences + 1, "query offsets have the wrong size");
    ELLM_CHECK(batch.queryLengths.size() == numSequences, "query lengths have the wrong size");
    ELLM_CHECK(batch.pastLengths.size() == numSequences, "past lengths have the wrong size");
    ELLM_CHECK(batch.attentionSequenceLengths.size() == numSequences, "attention sequence lengths have the wrong size");
    ELLM_CHECK(batch.stateIndices.size() == numSequences, "state indices have the wrong size");
    ELLM_CHECK(batch.sequenceOrder.size() == numSequences, "sequence identity snapshot has the wrong size");
    ELLM_CHECK(batch.sequenceWorks.size() == numSequences, "sequence work snapshot has the wrong size");
    ELLM_CHECK(batch.queryStartOffsets.front() == 0, "compact query offsets must start at zero");

    int32_t queryWidth = 0;
    for (size_t index = 0; index < numSequences; ++index)
    {
        int32_t const queryLength = batch.queryLengths[index];
        int32_t const start = batch.queryStartOffsets[index];
        ELLM_CHECK(queryLength > 0 && queryLength <= contract.maxQueryLength, "query length is invalid");
        ELLM_CHECK(
            batch.queryStartOffsets[index + 1] == start + queryLength && start + queryLength <= shape.physicalTokens,
            "compact query offsets are not the prefix sum of query lengths");
        bool const contextRow = static_cast<int32_t>(index) < shape.numContextSequences;
        ELLM_CHECK(contextRow ? batch.sequenceWorks[index] == SequenceWork::kContext
                              : batch.sequenceWorks[index] == SequenceWork::kDecode && queryLength == 1,
            "compact rows must be context sequences followed by single-token decode sequences");
        ELLM_CHECK(batch.pastLengths[index] >= 0, "past length is invalid");
        ELLM_CHECK(
            static_cast<int64_t>(batch.pastLengths[index]) + queryLength == batch.attentionSequenceLengths[index],
            "attention sequence length must equal past plus query length");
        ELLM_CHECK(
            batch.stateIndices[index] == batch.sequenceOrder[index].resident.slot, "state and resident slots disagree");
        ELLM_CHECK(batch.stateIndices[index] >= 0 && batch.stateIndices[index] < contract.recurrentPoolRows,
            "state index is outside the configured pool");
        auto const previousStates = batch.stateIndices.begin() + static_cast<ptrdiff_t>(index);
        ELLM_CHECK(std::find(batch.stateIndices.begin(), previousStates, batch.stateIndices[index]) == previousStates,
            "duplicate state index in compact execution batch");
        auto const previousIdentities = batch.sequenceOrder.begin() + static_cast<ptrdiff_t>(index);
        ELLM_CHECK(std::find_if(batch.sequenceOrder.begin(), previousIdentities,
                       [&](SequenceIdentity const& identity) {
                           return identity.requestId == batch.sequenceOrder[index].requestId;
                       })
                == previousIdentities,
            "duplicate request identity snapshot");
        for (int32_t token = 0; token < queryLength; ++token)
        {
            ELLM_CHECK(batch.positions[static_cast<size_t>(start + token)] == batch.pastLengths[index] + token,
                "valid token has the wrong absolute position");
        }
        queryWidth = std::max(queryWidth, queryLength);
    }
    ELLM_CHECK(batch.queryStartOffsets.back() == shape.physicalTokens, "terminal query offset is invalid");
    ELLM_CHECK(queryWidth == shape.queryWidth, "compact query width is inconsistent");

    ELLM_CHECK(
        shape.numLogits == shape.numSequences && static_cast<int32_t>(batch.logitsIndices.size()) == shape.numLogits,
        "compact execution needs one logits row per sequence");
    ELLM_CHECK(batch.placeholderLogitsRows >= 0
            && batch.logitsToSequence.size() + static_cast<size_t>(batch.placeholderLogitsRows)
                == batch.logitsIndices.size(),
        "logits routing storage has the wrong size");
    auto const lastRow = [&batch](size_t sequence) {
        return static_cast<int64_t>(batch.queryStartOffsets[sequence]) + batch.queryLengths[sequence] - 1;
    };
    int32_t previousSequence = -1;
    for (size_t logitsRow = 0; logitsRow < batch.logitsToSequence.size(); ++logitsRow)
    {
        int32_t const sequenceIndex = batch.logitsToSequence[logitsRow];
        ELLM_CHECK(sequenceIndex > previousSequence && sequenceIndex < shape.numSequences,
            "compact logits rows must follow execution-row order without duplicates");
        previousSequence = sequenceIndex;
        ELLM_CHECK(batch.logitsIndices[logitsRow] == lastRow(static_cast<size_t>(sequenceIndex)),
            "logits index does not select the final valid row");
    }
    size_t routed = 0;
    size_t placeholder = batch.logitsToSequence.size();
    for (size_t sequence = 0; sequence < numSequences; ++sequence)
    {
        if (routed < batch.logitsToSequence.size() && batch.logitsToSequence[routed] == static_cast<int32_t>(sequence))
        {
            ++routed;
            continue;
        }
        ELLM_CHECK(batch.logitsIndices[placeholder] == lastRow(sequence),
            "a placeholder logits row must select the final row of a sequence that samples nothing");
        ++placeholder;
    }
}

} // namespace

void RaggedBatchBuilder::validateExecutionBatch(RaggedExecutionBatch const& batch, RaggedEngineContract const& contract)
{
    validateContract(contract);
    if (contract.backend == TokenLayoutBackend::kNativeCompactRagged)
    {
        validateCompactExecutionBatch(batch, contract);
        return;
    }
    RaggedStepShape const& shape = batch.shape;
    ELLM_CHECK(batch.layout == contract.backend, "execution layout does not match the engine contract");
    ELLM_CHECK(shape.numSequences > 0 && shape.numSequences <= contract.maxNumSequences,
        "execution sequence count is outside engine capacity");
    ELLM_CHECK(shape.queryWidth > 0 && shape.queryWidth <= contract.maxQueryLength,
        "execution query width is outside engine capacity");
    int32_t const expectedPhysical
        = checkedInt32(static_cast<int64_t>(shape.numSequences) * shape.queryWidth, "execution physical token count");
    ELLM_CHECK(shape.physicalTokens == expectedPhysical && expectedPhysical <= contract.maxPhysicalTokens,
        "execution physical token shape is invalid");

    size_t const numSequences = static_cast<size_t>(shape.numSequences);
    size_t const physicalTokens = static_cast<size_t>(shape.physicalTokens);
    ELLM_CHECK(!batch.hostTokenIds.isEmpty() && batch.hostTokenIds.getDeviceType() == DeviceType::kCPU
            && batch.hostTokenIds.getDataType() == trt_edgellm::DataType::kINT32
            && batch.hostTokenIds.getShape().volume() == shape.physicalTokens,
        "host token ID storage has the wrong shape or type");
    int32_t const* tokenIds = batch.hostTokenIds.dataPointer<int32_t>();
    ELLM_CHECK(batch.positions.size() == physicalTokens, "position storage has the wrong size");
    ELLM_CHECK(batch.queryStartOffsets.size() == numSequences + 1, "query offsets have the wrong size");
    ELLM_CHECK(batch.queryLengths.size() == numSequences, "query lengths have the wrong size");
    ELLM_CHECK(batch.pastLengths.size() == numSequences, "past lengths have the wrong size");
    ELLM_CHECK(batch.attentionSequenceLengths.size() == numSequences, "attention sequence lengths have the wrong size");
    ELLM_CHECK(batch.stateIndices.size() == numSequences, "state indices have the wrong size");
    ELLM_CHECK(batch.sequenceOrder.size() == numSequences, "sequence identity snapshot has the wrong size");
    ELLM_CHECK(batch.sequenceWorks.size() == numSequences, "sequence work snapshot has the wrong size");

    int64_t validTokens = 0;
    int32_t numContextSequences = 0;
    int64_t numContextTokens = 0;
    for (int32_t sequenceIndex = 0; sequenceIndex < shape.numSequences; ++sequenceIndex)
    {
        size_t const index = static_cast<size_t>(sequenceIndex);
        int32_t const start = sequenceIndex * shape.queryWidth;
        int32_t const queryLength = batch.queryLengths[index];
        ELLM_CHECK(batch.queryStartOffsets[index] == start, "query offset is not an entry-padded physical start");
        ELLM_CHECK(queryLength > 0 && queryLength <= shape.queryWidth, "query length is invalid");
        ELLM_CHECK(batch.pastLengths[index] >= 0, "past length is invalid");
        ELLM_CHECK(
            static_cast<int64_t>(batch.pastLengths[index]) + queryLength <= batch.attentionSequenceLengths[index],
            "attention sequence length cannot precede the visible query endpoint");
        ELLM_CHECK(
            batch.stateIndices[index] == batch.sequenceOrder[index].resident.slot, "state and resident slots disagree");
        ELLM_CHECK(batch.stateIndices[index] >= 0 && batch.stateIndices[index] < contract.recurrentPoolRows,
            "state index is outside the configured pool");
        ELLM_CHECK(std::find_if(batch.sequenceOrder.begin(), batch.sequenceOrder.begin() + sequenceIndex,
                       [&](SequenceIdentity const& identity) {
                           return identity.requestId == batch.sequenceOrder[index].requestId;
                       })
                == batch.sequenceOrder.begin() + sequenceIndex,
            "duplicate request identity snapshot");
        ELLM_CHECK(std::find_if(batch.sequenceOrder.begin(), batch.sequenceOrder.begin() + sequenceIndex,
                       [&](SequenceIdentity const& identity) {
                           return identity.resident.slot == batch.sequenceOrder[index].resident.slot;
                       })
                == batch.sequenceOrder.begin() + sequenceIndex,
            "duplicate resident identity snapshot");

        if (batch.sequenceWorks[index] == SequenceWork::kContext)
        {
            ++numContextSequences;
            numContextTokens += queryLength;
        }
        else
        {
            ELLM_CHECK(batch.sequenceWorks[index] == SequenceWork::kDecode && queryLength == 1,
                "decode execution row must have query length one");
        }
        validTokens += queryLength;

        for (int32_t tokenIndex = 0; tokenIndex < shape.queryWidth; ++tokenIndex)
        {
            size_t const physicalIndex = static_cast<size_t>(start + tokenIndex);
            if (tokenIndex < queryLength)
            {
                ELLM_CHECK(batch.positions[physicalIndex] == batch.pastLengths[index] + tokenIndex,
                    "valid token has the wrong absolute position");
            }
            else
            {
                ELLM_CHECK(tokenIds[physicalIndex] == kPaddingTokenId, "padding token does not use the safe value");
                ELLM_CHECK(batch.positions[physicalIndex] == kPaddingSentinel, "padding token has a live position");
            }
        }
    }
    ELLM_CHECK(batch.queryStartOffsets.back() == shape.physicalTokens, "terminal query offset is invalid");
    ELLM_CHECK(validTokens == shape.validTokens, "valid token shape is inconsistent");
    ELLM_CHECK(numContextSequences == shape.numContextSequences, "context sequence shape is inconsistent");
    ELLM_CHECK(numContextTokens == shape.numContextTokens, "context token shape is inconsistent");
    ELLM_CHECK(
        numContextSequences == 0 || numContextSequences == shape.numSequences, "mixed execution step is not supported");

    ELLM_CHECK(
        static_cast<int32_t>(batch.logitsIndices.size()) == shape.numLogits, "logits index storage has the wrong size");
    ELLM_CHECK(
        batch.logitsToSequence.size() == batch.logitsIndices.size(), "logits routing storage has the wrong size");
    for (size_t logitsRow = 0; logitsRow < batch.logitsIndices.size(); ++logitsRow)
    {
        int32_t const sequenceIndex = batch.logitsToSequence[logitsRow];
        ELLM_CHECK(
            sequenceIndex >= 0 && sequenceIndex < shape.numSequences, "logits owner is outside the execution batch");
        size_t const sequence = static_cast<size_t>(sequenceIndex);
        ELLM_CHECK(std::find(batch.logitsToSequence.begin(),
                       batch.logitsToSequence.begin() + static_cast<ptrdiff_t>(logitsRow), sequenceIndex)
                == batch.logitsToSequence.begin() + static_cast<ptrdiff_t>(logitsRow),
            "logits route contains a duplicate sequence");
        int32_t const expectedIndex = batch.queryStartOffsets[sequence] + batch.queryLengths[sequence] - 1;
        ELLM_CHECK(batch.logitsIndices[logitsRow] == expectedIndex, "logits index does not select the final valid row");
        ELLM_CHECK(batch.positions[static_cast<size_t>(expectedIndex)] >= 0, "logits index selects padding");
    }
}

void RaggedExecutionBatch::validateCommitSnapshot(
    StepId completedStepId, std::vector<CompletionSequence> const& current) const
{
    ELLM_CHECK(completedStepId == stepId, "completion step ID is stale");
    ELLM_CHECK(current.size() == sequenceOrder.size(), "completion identity count does not match execution batch");
    for (size_t index = 0; index < current.size(); ++index)
    {
        ELLM_CHECK(current[index].requestId == sequenceOrder[index].requestId, "completion request ID is stale");
        ELLM_CHECK(current[index].resident == sequenceOrder[index].resident, "completion resident identity is stale");
        ELLM_CHECK(current[index].pastLength == pastLengths[index], "completion past length is stale");
    }
}

} // namespace rt
} // namespace trt_edgellm
