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

#include "runtime/decoding/specCommonStateTracker.h"

#include "common/checkMacros.h"
#include "runtime/state/decodingInferenceContext.h"

#include <cstddef>
#include <limits>

namespace trt_edgellm::rt
{

void SpecCommonStateTracker::initialize(DecodingInferenceContext const& context)
{
    ELLM_CHECK(context.activeBatchSize >= 0
            && context.rawBatchedInputIds.size() == static_cast<size_t>(context.activeBatchSize)
            && context.effectivePrefillLengths.size() == static_cast<size_t>(context.activeBatchSize),
        "Speculative common-state initialization does not match the active batch");
    mDraftPrefillOutputsPending = false;
    mCommonMaterializedStateLengths.resize(static_cast<size_t>(context.activeBatchSize));
    mPendingDraftAcceptLengths.assign(static_cast<size_t>(context.activeBatchSize), 0);
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        size_t const inputLength = context.rawBatchedInputIds[static_cast<size_t>(slot)].size();
        int32_t const effectivePrefillLength = context.effectivePrefillLengths[static_cast<size_t>(slot)];
        // Visual-token pruning shortens the materialized KV below the raw input length; empty
        // prunedPrefillTokens means no pruning ran.
        int32_t const prunedTokens = slot < static_cast<int32_t>(context.prunedPrefillTokens.size())
            ? context.prunedPrefillTokens[static_cast<size_t>(slot)]
            : 0;
        ELLM_CHECK(inputLength <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) && prunedTokens >= 0
                && static_cast<size_t>(prunedTokens) <= inputLength && effectivePrefillLength >= 0
                && static_cast<size_t>(effectivePrefillLength) <= inputLength - static_cast<size_t>(prunedTokens),
            "Speculative prefill length is outside the input sequence");
        mCommonMaterializedStateLengths[static_cast<size_t>(slot)] = static_cast<int32_t>(inputLength) - prunedTokens;
    }
}

void SpecCommonStateTracker::reserve(int32_t rows)
{
    mCommonMaterializedStateLengths.reserve(static_cast<size_t>(rows));
    mPendingDraftAcceptLengths.reserve(static_cast<size_t>(rows));
    mCompactedValues.reserve(static_cast<size_t>(rows));
    mCompactedSlotFilled.reserve(static_cast<size_t>(rows));
}

void SpecCommonStateTracker::appendRow(int32_t row)
{
    ELLM_CHECK(row >= 0 && static_cast<size_t>(row) <= mCommonMaterializedStateLengths.size()
            && mPendingDraftAcceptLengths.size() == mCommonMaterializedStateLengths.size(),
        "Speculative common-state tracking does not cover the rows before the appended one");
    // A tracker left over from a blocking request holds rows the session does not: the batch starts over.
    mCommonMaterializedStateLengths.resize(static_cast<size_t>(row));
    mPendingDraftAcceptLengths.resize(static_cast<size_t>(row));
    mCommonMaterializedStateLengths.push_back(0);
    mPendingDraftAcceptLengths.push_back(0);
}

void SpecCommonStateTracker::seatRow(int32_t row, int32_t materializedLength, int32_t pendingLength)
{
    ELLM_CHECK(row >= 0 && static_cast<size_t>(row) < mCommonMaterializedStateLengths.size(),
        "Speculative common-state tracking has no row for the seated sequence");
    ELLM_CHECK(materializedLength >= 0 && pendingLength >= 0, "Speculative seated lengths must be non-negative");
    mCommonMaterializedStateLengths[static_cast<size_t>(row)] = materializedLength;
    mPendingDraftAcceptLengths[static_cast<size_t>(row)] = pendingLength;
}

void SpecCommonStateTracker::materializePending(int32_t activeBatchSize)
{
    // Pending lengths start at zero, so the first round after a prefill materializes nothing new -- but a sequence
    // seated mid-flight carries its held-back prompt position into the round it first decodes in.
    if (mCommonMaterializedStateLengths.empty())
    {
        return;
    }
    ELLM_CHECK(activeBatchSize >= 0 && mCommonMaterializedStateLengths.size() == static_cast<size_t>(activeBatchSize)
            && mPendingDraftAcceptLengths.size() == static_cast<size_t>(activeBatchSize),
        "Speculative common-state tracking does not match the active batch");
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        int32_t const pending = mPendingDraftAcceptLengths[static_cast<size_t>(slot)];
        ELLM_CHECK(pending >= 0
                && mCommonMaterializedStateLengths[static_cast<size_t>(slot)]
                    <= std::numeric_limits<int32_t>::max() - pending,
            "Speculative common-state length overflow");
        mCommonMaterializedStateLengths[static_cast<size_t>(slot)] += pending;
    }
}

void SpecCommonStateTracker::recordAccepted(int32_t const* hostAcceptLengths, int32_t activeBatchSize)
{
    if (mCommonMaterializedStateLengths.empty())
    {
        return;
    }
    ELLM_CHECK(hostAcceptLengths != nullptr && activeBatchSize >= 0
            && mCommonMaterializedStateLengths.size() == static_cast<size_t>(activeBatchSize),
        "Speculative accepted lengths do not match the active batch");
    mPendingDraftAcceptLengths.assign(hostAcceptLengths, hostAcceptLengths + activeBatchSize);
}

void SpecCommonStateTracker::compact(
    std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch)
{
    ELLM_CHECK(oldActiveBatch >= 0 && newActiveBatch >= 0 && newActiveBatch <= oldActiveBatch
            && batchMapping.size() == static_cast<size_t>(oldActiveBatch),
        "Speculative batch mapping does not match the active batch");
    auto compactVector = [&](std::vector<int32_t>& values) {
        if (values.empty())
        {
            return;
        }
        ELLM_CHECK(values.size() == static_cast<size_t>(oldActiveBatch),
            "Speculative common-state tracking does not match the old active batch");
        // Staged through reserved scratch and copied back, so the tracked vectors keep their reserved capacity.
        mCompactedValues.assign(static_cast<size_t>(newActiveBatch), 0);
        mCompactedSlotFilled.assign(static_cast<size_t>(newActiveBatch), 0);
        for (int32_t oldSlot = 0; oldSlot < oldActiveBatch; ++oldSlot)
        {
            int32_t const newSlot = batchMapping[static_cast<size_t>(oldSlot)];
            ELLM_CHECK(newSlot >= -1 && newSlot < newActiveBatch, "Speculative batch mapping is invalid");
            if (newSlot >= 0)
            {
                ELLM_CHECK(
                    !mCompactedSlotFilled[static_cast<size_t>(newSlot)], "Speculative batch mapping is not one-to-one");
                mCompactedValues[static_cast<size_t>(newSlot)] = values[static_cast<size_t>(oldSlot)];
                mCompactedSlotFilled[static_cast<size_t>(newSlot)] = 1;
            }
        }
        for (char const filled : mCompactedSlotFilled)
        {
            ELLM_CHECK(filled != 0, "Speculative batch mapping does not cover the new active batch");
        }
        values.assign(mCompactedValues.begin(), mCompactedValues.end());
    };
    compactVector(mCommonMaterializedStateLengths);
    compactVector(mPendingDraftAcceptLengths);
    if (newActiveBatch == 0)
    {
        mDraftPrefillOutputsPending = false;
    }
}

void SpecCommonStateTracker::reset() noexcept
{
    mCommonMaterializedStateLengths.clear();
    mPendingDraftAcceptLengths.clear();
    mDraftPrefillOutputsPending = false;
}

void SpecCommonStateTracker::markDraftPrefillOutputsPending() noexcept
{
    mDraftPrefillOutputsPending = true;
}

void SpecCommonStateTracker::consumeDraftPrefillOutputs() noexcept
{
    mDraftPrefillOutputsPending = false;
}

bool SpecCommonStateTracker::draftPrefillOutputsPending() const noexcept
{
    return mDraftPrefillOutputsPending;
}

bool SpecCommonStateTracker::shouldUsePendingPrefillProposal(int32_t generationRound) const noexcept
{
    return generationRound == 0 && mDraftPrefillOutputsPending;
}

std::vector<int32_t> const& SpecCommonStateTracker::commonMaterializedStateLengths() const noexcept
{
    return mCommonMaterializedStateLengths;
}

} // namespace trt_edgellm::rt
