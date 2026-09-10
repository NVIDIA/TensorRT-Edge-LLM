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

#include "runtime/state/decodingInferenceContext.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm::rt;

TEST(SpecCommonStateTrackerTests, InitializesFromFullInputsAndTracksAcceptedTokens)
{
    DecodingInferenceContext context;
    context.activeBatchSize = 2;
    context.rawBatchedInputIds = {{1, 2, 3}, {4, 5, 6, 7, 8}};
    context.effectivePrefillLengths = {2, 5};

    SpecCommonStateTracker tracker;
    tracker.initialize(context);
    EXPECT_EQ(tracker.commonMaterializedStateLengths(), std::vector<int32_t>({3, 5}));
    EXPECT_FALSE(tracker.draftPrefillOutputsPending());

    tracker.markDraftPrefillOutputsPending();
    EXPECT_TRUE(tracker.shouldUsePendingPrefillProposal(0));
    tracker.consumeDraftPrefillOutputs();
    int32_t const accepted[]{2, 1};
    tracker.recordAccepted(accepted, 2);
    tracker.materializePending(1, 2);
    EXPECT_EQ(tracker.commonMaterializedStateLengths(), std::vector<int32_t>({5, 6}));
}

TEST(SpecCommonStateTrackerTests, CompactsAllStateAndPreservesPendingPrefill)
{
    DecodingInferenceContext context;
    context.activeBatchSize = 3;
    context.rawBatchedInputIds = {{1}, {2, 3}, {4, 5, 6}};
    context.effectivePrefillLengths = {1, 2, 3};

    SpecCommonStateTracker tracker;
    tracker.initialize(context);
    int32_t const accepted[]{3, 4, 5};
    tracker.recordAccepted(accepted, 3);
    tracker.markDraftPrefillOutputsPending();
    tracker.compact({1, -1, 0}, 3, 2);

    EXPECT_EQ(tracker.commonMaterializedStateLengths(), std::vector<int32_t>({3, 1}));
    tracker.materializePending(1, 2);
    EXPECT_EQ(tracker.commonMaterializedStateLengths(), std::vector<int32_t>({8, 4}));
    EXPECT_TRUE(tracker.draftPrefillOutputsPending());

    tracker.compact({-1, -1}, 2, 0);
    EXPECT_TRUE(tracker.commonMaterializedStateLengths().empty());
    EXPECT_FALSE(tracker.draftPrefillOutputsPending());
}

TEST(SpecCommonStateTrackerTests, RejectsInvalidLengthsMappingsAndOverflow)
{
    SpecCommonStateTracker tracker;
    DecodingInferenceContext context;
    context.activeBatchSize = 1;
    context.rawBatchedInputIds = {{1, 2}};
    context.effectivePrefillLengths = {3};
    EXPECT_THROW(tracker.initialize(context), std::runtime_error);

    context.effectivePrefillLengths = {2};
    tracker.initialize(context);
    EXPECT_THROW(tracker.compact({1}, 1, 1), std::runtime_error);

    int32_t const accepted[]{std::numeric_limits<int32_t>::max()};
    tracker.recordAccepted(accepted, 1);
    EXPECT_THROW(tracker.materializePending(1, 1), std::runtime_error);
}
