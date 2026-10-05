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

#include "scheduler/requestCoordinator.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;

namespace
{

//! A SteppedExecution that only materializes; the table never calls anything else.
class MaterializeOnly final : public SteppedExecution
{
public:
    //! Throw on the Nth materialize call (1-based); 0 never throws.
    explicit MaterializeOnly(int32_t throwOnCall = 0)
        : mThrowOnCall(throwOnCall)
    {
    }

    std::vector<ResidentRef> residents() const override
    {
        return {};
    }

    AdmissionResult admit(LLMGenerationRequest const&, int32_t, RequestId) override
    {
        return {AdmissionResult::Status::kRejected, {}, "not under test"};
    }

    StepResult prefill(ImmutablePrefillBatch const&) override
    {
        return {};
    }

    StepResult decode(ImmutableDecodeBatch const&) override
    {
        return {};
    }

    LLMGenerationResponse materialize(BatchResult const& result, std::vector<std::string> const&) const override
    {
        if (++mCalls == mThrowOnCall)
        {
            throw std::runtime_error("tokenizer exploded");
        }
        LLMGenerationResponse response;
        response.outputIds = {result.tokenIds};
        return response;
    }

    void abort() noexcept override {}

    bool finish(LLMGenerationResponse&) override
    {
        return true;
    }

    int32_t calls() const noexcept
    {
        return mCalls;
    }

private:
    int32_t mThrowOnCall;
    mutable int32_t mCalls{0};
};

std::shared_ptr<RequestRecord> makeRecord(RequestId id)
{
    auto record = std::make_shared<RequestRecord>();
    record->id = id;
    return record;
}

BatchResult finishedWith(FinishReason reason, std::vector<int32_t> tokens = {7})
{
    BatchResult result;
    result.tokenIds = std::move(tokens);
    result.terminalReason = reason;
    return result;
}

} // namespace

TEST(RequestCoordinatorTests, CommitRetiresTheFinishedAndKeepsTheSurvivors)
{
    MaterializeOnly stepped;
    RequestCoordinator table;
    auto const a = makeRecord(1);
    auto const b = makeRecord(2);
    table.add(ResidentRef{0, 0}, a, {});
    table.add(ResidentRef{1, 0}, b, {});

    StepResult step;
    step.ok = true;
    step.finished.emplace_back(ResidentRef{0, 0}, finishedWith(FinishReason::kLength, {3, 4}));
    auto outcomes = table.commit(step, stepped);

    ASSERT_EQ(outcomes.size(), 1U);
    EXPECT_EQ(outcomes[0].record.get(), a.get());
    EXPECT_EQ(outcomes[0].reason, FinishReason::kLength);
    ASSERT_TRUE(outcomes[0].response.has_value());
    EXPECT_EQ(outcomes[0].response->outputIds.front(), (std::vector<int32_t>{3, 4}));
    EXPECT_FALSE(table.empty()) << "B is still resident";

    auto rest = table.failEverything();
    ASSERT_EQ(rest.size(), 1U);
    EXPECT_EQ(rest[0].record.get(), b.get());
    EXPECT_TRUE(table.empty());
}

TEST(RequestCoordinatorTests, ACancelledOrFailedResidentIsNotMaterialized)
{
    MaterializeOnly stepped;
    RequestCoordinator table;
    table.add(ResidentRef{0, 0}, makeRecord(1), {});
    table.add(ResidentRef{1, 0}, makeRecord(2), {});

    StepResult step;
    step.ok = true;
    step.finished.emplace_back(ResidentRef{0, 0}, finishedWith(FinishReason::kCancelled));
    step.finished.emplace_back(ResidentRef{1, 0}, finishedWith(FinishReason::kError));
    auto outcomes = table.commit(step, stepped);

    ASSERT_EQ(outcomes.size(), 2U);
    EXPECT_FALSE(outcomes[0].response.has_value());
    EXPECT_FALSE(outcomes[1].response.has_value());
    EXPECT_EQ(stepped.calls(), 0);
    EXPECT_TRUE(table.empty());
}

TEST(RequestCoordinatorTests, CommitIsAllOrNothingWhenMaterializeThrows)
{
    // Two residents finish in one step and the second materialize throws. Before the fix the first
    // record had already been moved out and erased: its outcome died with the unwind (its caller
    // would hang forever) and the second entry was left holding a null record for failEverything()
    // to hand to the engine.
    MaterializeOnly stepped(/*throwOnCall=*/2);
    RequestCoordinator table;
    auto const a = makeRecord(1);
    auto const b = makeRecord(2);
    table.add(ResidentRef{0, 0}, a, {});
    table.add(ResidentRef{1, 0}, b, {});

    StepResult step;
    step.ok = true;
    step.finished.emplace_back(ResidentRef{0, 0}, finishedWith(FinishReason::kLength));
    step.finished.emplace_back(ResidentRef{1, 0}, finishedWith(FinishReason::kLength));
    EXPECT_THROW(table.commit(step, stepped), std::runtime_error);

    // The table is untouched: both records are still owned, so the batch failure that follows can
    // give every caller a terminal outcome.
    auto outcomes = table.failEverything();
    ASSERT_EQ(outcomes.size(), 2U);
    for (auto const& outcome : outcomes)
    {
        ASSERT_NE(outcome.record, nullptr);
        EXPECT_EQ(outcome.reason, FinishReason::kError);
    }
    EXPECT_TRUE(table.empty());
}

TEST(RequestCoordinatorTests, CommitRejectsARefThatIsNotTheLiveOwnerWithoutTouchingTheTable)
{
    MaterializeOnly stepped;
    RequestCoordinator table;
    auto const a = makeRecord(1);
    table.add(ResidentRef{0, 1}, a, {});

    // Same slot, stale epoch: the handle has been reused since this ref was taken.
    StepResult step;
    step.ok = true;
    step.finished.emplace_back(ResidentRef{0, 0}, finishedWith(FinishReason::kLength));
    EXPECT_THROW(table.commit(step, stepped), std::runtime_error);
    EXPECT_EQ(stepped.calls(), 0) << "nothing is materialized once a ref fails validation";

    auto outcomes = table.failEverything();
    ASSERT_EQ(outcomes.size(), 1U);
    EXPECT_EQ(outcomes[0].record.get(), a.get());
}
