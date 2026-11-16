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

#pragma once

#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

struct DecodingInferenceContext;

//! Tracks the greatest per-slot prefix materialized by both target and draft models.
class SpecCommonStateTracker
{
public:
    void initialize(DecodingInferenceContext const& context);
    //! Allocate room for @p rows rows up front, so appendRow() and compact() within it never allocate.
    void reserve(int32_t rows);
    //! A sequence joins a running batch at @p row, its last row: track it from nothing until seatRow().
    void appendRow(int32_t row);
    //! The sequence at @p row seated its draft: both models hold its first @p materializedLength positions, and its
    //! draft takes the next @p pendingLength at the next decode step, like a resident's accepted span.
    void seatRow(int32_t row, int32_t materializedLength, int32_t pendingLength);
    void materializePending(int32_t activeBatchSize);
    void recordAccepted(int32_t const* hostAcceptLengths, int32_t activeBatchSize);
    void compact(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch);
    void reset() noexcept;

    void markDraftPrefillOutputsPending() noexcept;
    void consumeDraftPrefillOutputs() noexcept;
    bool draftPrefillOutputsPending() const noexcept;
    bool shouldUsePendingPrefillProposal(int32_t generationRound) const noexcept;
    std::vector<int32_t> const& commonMaterializedStateLengths() const noexcept;

private:
    std::vector<int32_t> mCommonMaterializedStateLengths;
    std::vector<int32_t> mPendingDraftAcceptLengths;
    //! compact() scratch: the surviving values in their new slots, and which new slots the mapping filled.
    std::vector<int32_t> mCompactedValues;
    std::vector<char> mCompactedSlotFilled;
    bool mDraftPrefillOutputsPending{};
};

} // namespace trt_edgellm::rt
