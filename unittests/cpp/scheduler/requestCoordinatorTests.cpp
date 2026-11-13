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

std::vector<ScheduledSequenceDescriptor> decodeStepOf(RequestCoordinator const& table, std::vector<ResidentRef> refs)
{
    std::vector<ScheduledSequenceDescriptor> plan;
    table.decodeStep(refs, plan);
    return plan;
}

//! A RuntimeSession that only materializes; the table never calls anything else.
class MaterializeOnly final : public RuntimeSession
{
public:
    //! Throw on the Nth materialize call (1-based); 0 never throws.
    explicit MaterializeOnly(int32_t throwOnCall = 0)
        : mThrowOnCall(throwOnCall)
    {
    }

    AdmissionResult admit(LLMGenerationRequest const&, RequestId, std::shared_ptr<std::atomic<bool> const>) override
    {
        return {AdmissionResult::Status::kRejected, {}, "not under test"};
    }

    StepResult execute(std::vector<ScheduledSequenceDescriptor> const&) override
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

    std::vector<ResidentRef> residents() const override
    {
        return {};
    }

    bool idle() const noexcept override
    {
        return true;
    }

    bool exclusiveResident() const noexcept override
    {
        return false;
    }

    bool requiresExclusiveAdmission(LLMGenerationRequest const&) const noexcept override
    {
        return false;
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

std::unique_ptr<LLMGenerationRequest> makeRequest()
{
    auto request = std::make_unique<LLMGenerationRequest>();
    request->requests.resize(1);
    return request;
}

AdmissionResult admitted(ResidentRef ref, int32_t promptLength = 4, int32_t prefillStart = 0)
{
    return AdmissionResult{AdmissionResult::Status::kAdmitted, ref, {}, promptLength, prefillStart};
}

BatchResult finishedWith(FinishReason reason, std::vector<int32_t> tokens = {7})
{
    BatchResult result;
    result.tokenIds = std::move(tokens);
    result.terminalReason = reason;
    return result;
}

//! What a session reports for a step that ran exactly the plan it was given: participants are the
//! plan's identities and the execution record is the plan itself.
StepResult ranAsPlanned(std::vector<ScheduledSequenceDescriptor> plan)
{
    StepResult result;
    result.ok = true;
    result.work = plan.front().work;
    result.stepId = 1;
    for (auto const& sequence : plan)
    {
        result.participants.push_back(sequence.identity);
    }
    result.execution = std::move(plan);
    return result;
}

} // namespace

TEST(RequestCoordinatorTests, BuildsStepsFromAdmissionAndLedger)
{
    MaterializeOnly session;
    RequestCoordinator table;
    ResidentRef const a{0, 1};
    ResidentRef const b{1, 1};
    table.add(a, makeRecord(11), makeRequest(), admitted(a, /*promptLength=*/6, /*prefillStart=*/0));
    table.add(b, makeRecord(12), makeRequest(), admitted(b, /*promptLength=*/6, /*prefillStart=*/4));

    // The context step covers exactly the uncommitted prompt: everything for a cold sequence, the
    // tail past the reused prefix for a cache hit.
    ScheduledSequenceDescriptor const prefillA = table.contextStep(a);
    EXPECT_EQ(prefillA.identity, (SequenceIdentity{11, a}));
    EXPECT_EQ(prefillA.work, SequenceWork::kContext);
    EXPECT_EQ(prefillA.queryLength, 6);
    EXPECT_EQ(prefillA.pastLength, 0);
    ScheduledSequenceDescriptor const prefillB = table.contextStep(b);
    EXPECT_EQ(prefillB.queryLength, 2);
    EXPECT_EQ(prefillB.pastLength, 4);

    // A resident cannot decode before its context step committed the prompt.
    EXPECT_THROW(decodeStepOf(table, {a}), std::runtime_error);

    EXPECT_TRUE(table.commit(ranAsPlanned({prefillA}), session).empty());
    EXPECT_TRUE(table.commit(ranAsPlanned({prefillB}), session).empty());
    EXPECT_THROW(table.contextStep(a), std::runtime_error) << "the prompt is committed; no context step is left";

    // Decode steps advance the committed length by one per step, per participant.
    std::vector<ScheduledSequenceDescriptor> decode = decodeStepOf(table, {a, b});
    ASSERT_EQ(decode.size(), 2U);
    EXPECT_EQ(decode[0].work, SequenceWork::kDecode);
    EXPECT_EQ(decode[0].queryLength, 1);
    EXPECT_EQ(decode[0].pastLength, 6);
    EXPECT_EQ(decode[1].pastLength, 6);
    EXPECT_TRUE(table.commit(ranAsPlanned(decode), session).empty());
    decode = decodeStepOf(table, {a, b});
    EXPECT_EQ(decode[0].pastLength, 7);
    EXPECT_EQ(decode[1].pastLength, 7);

    // The plan follows the session's order, whatever it is.
    decode = decodeStepOf(table, {b, a});
    EXPECT_EQ(decode[0].identity.requestId, 12U);
    EXPECT_EQ(decode[1].identity.requestId, 11U);
}

TEST(RequestCoordinatorTests, LedgerAdvancesByExecutionNotPlan)
{
    // Under a visual-token pruner a context step adds fewer positions than the prompt range it was
    // asked to cover. The ledger must take the runtime's word for it: the next decode step is built
    // from the positions that exist, and the context step counts as committed even though the
    // committed length never reaches the prompt length.
    MaterializeOnly session;
    RequestCoordinator table;
    ResidentRef const a{0, 1};
    table.add(a, makeRecord(11), makeRequest(), admitted(a, /*promptLength=*/6, /*prefillStart=*/0));

    ScheduledSequenceDescriptor const planned = table.contextStep(a);
    EXPECT_EQ(planned.queryLength, 6);
    ScheduledSequenceDescriptor pruned = planned;
    pruned.queryLength = 4; // two media tokens pruned before the engine ran
    EXPECT_TRUE(table.commit(ranAsPlanned({pruned}), session).empty());

    EXPECT_THROW(table.contextStep(a), std::runtime_error) << "the context step ran; it is not offered again";
    std::vector<ScheduledSequenceDescriptor> const decode = decodeStepOf(table, {a});
    ASSERT_EQ(decode.size(), 1U);
    EXPECT_EQ(decode.front().pastLength, 4) << "the decode step starts where the runtime's state ends";

    // A participant the runtime executed nothing for (no execution record) does not advance.
    StepResult nothingRan;
    nothingRan.ok = true;
    nothingRan.work = SequenceWork::kDecode;
    nothingRan.participants = {decode.front().identity};
    EXPECT_TRUE(table.commit(nothingRan, session).empty());
    EXPECT_EQ(decodeStepOf(table, {a}).front().pastLength, 4);
}

TEST(RequestCoordinatorTests, DivergentLedgerIsRefusedWhole)
{
    // The runtime naming residents the table does not hold, or reporting a step from a past length
    // it does not hold, is the two ledgers diverging. describes() detects the first; the commit
    // refuses the second as a whole and leaves the table untouched; the caller then fails the
    // session, and retireEverything hands its reason to every resident.
    MaterializeOnly session;
    RequestCoordinator table;
    ResidentRef const a{0, 1};
    ResidentRef const b{1, 1};
    table.add(a, makeRecord(11), makeRequest(), admitted(a, /*promptLength=*/4, /*prefillStart=*/0));
    table.add(b, makeRecord(12), makeRequest(), admitted(b, /*promptLength=*/4, /*prefillStart=*/0));
    EXPECT_TRUE(table.commit(ranAsPlanned({table.contextStep(a)}), session).empty());
    EXPECT_TRUE(table.commit(ranAsPlanned({table.contextStep(b)}), session).empty());

    EXPECT_TRUE(table.describes({a, b}));
    EXPECT_TRUE(table.describes({b, a})) << "order is the session's business";
    EXPECT_FALSE(table.describes({a})) << "a resident the session dropped";
    EXPECT_FALSE(table.describes({a, b, ResidentRef{2, 1}})) << "a resident the table never seated";
    EXPECT_FALSE(table.describes({a, ResidentRef{1, 2}})) << "same slot, another epoch: not the resident held";

    std::vector<ScheduledSequenceDescriptor> decode = decodeStepOf(table, {a, b});
    StepResult disagreeing = ranAsPlanned(decode);
    disagreeing.execution[1].pastLength = 9;
    disagreeing.finished.emplace_back(a, finishedWith(FinishReason::kLength));
    EXPECT_THROW(table.commit(disagreeing, session), std::runtime_error);
    // A ref that is not the live owner -- the same slot under a stale epoch -- or a participant the
    // table never seated is the same violation.
    StepResult stale;
    stale.ok = true;
    stale.finished.emplace_back(ResidentRef{0, 0}, finishedWith(FinishReason::kLength));
    EXPECT_THROW(table.commit(stale, session), std::runtime_error);
    StepResult stranger;
    stranger.ok = true;
    stranger.participants = {SequenceIdentity{9, ResidentRef{2, 1}}};
    EXPECT_THROW(table.commit(stranger, session), std::runtime_error);
    EXPECT_EQ(table.size(), 2) << "a refused commit retires nobody";
    EXPECT_EQ(decodeStepOf(table, {a, b})[0].pastLength, 4) << "and advances nobody";
    EXPECT_EQ(session.calls(), 0) << "nothing is materialized once the record fails validation";

    auto outcomes = table.retireEverything(FinishReason::kError, "the device fell over");
    ASSERT_EQ(outcomes.size(), 2U);
    for (auto const& outcome : outcomes)
    {
        EXPECT_EQ(outcome.reason, FinishReason::kError);
        EXPECT_EQ(outcome.errorMessage, "the device fell over");
        EXPECT_FALSE(outcome.response.has_value());
    }
    EXPECT_EQ(session.calls(), 0) << "a retired resident is not materialized";
    EXPECT_EQ(table.size(), 0);
    EXPECT_TRUE(table.describes({}));
}

TEST(RequestCoordinatorTests, CommitRetiresFinishedAndKeepsSurvivors)
{
    // One step retires four residents: a completion, a cancel, a failure inside the step, and a
    // completion whose response cannot be assembled (the second materialize throws). Only the
    // completions are materialized; the one that cannot be fails on its own, with the reason, and
    // the commit goes through for everyone else. The fifth resident survives the step.
    MaterializeOnly session(/*throwOnCall=*/2);
    RequestCoordinator table;
    std::vector<std::shared_ptr<RequestRecord>> records;
    for (int32_t slot = 0; slot < 5; ++slot)
    {
        records.push_back(makeRecord(static_cast<RequestId>(slot + 1)));
        table.add(ResidentRef{slot, 1}, records.back(), makeRequest(), admitted(ResidentRef{slot, 1}));
    }

    StepResult step;
    step.ok = true;
    step.work = SequenceWork::kDecode;
    step.finished.emplace_back(ResidentRef{0, 1}, finishedWith(FinishReason::kLength, {3, 4}));
    step.finished.emplace_back(ResidentRef{1, 1}, finishedWith(FinishReason::kCancelled));
    step.finished.emplace_back(ResidentRef{2, 1}, finishedWith(FinishReason::kError));
    step.finished.emplace_back(ResidentRef{3, 1}, finishedWith(FinishReason::kLength));
    auto outcomes = table.commit(step, session);

    ASSERT_EQ(outcomes.size(), 4U);
    for (size_t i = 0; i < outcomes.size(); ++i)
    {
        EXPECT_EQ(outcomes[i].record.get(), records[i].get()) << "outcome " << i;
    }
    EXPECT_EQ(outcomes[0].reason, FinishReason::kLength);
    ASSERT_TRUE(outcomes[0].response.has_value());
    EXPECT_EQ(outcomes[0].response->outputIds.front(), (std::vector<int32_t>{3, 4}));
    EXPECT_EQ(outcomes[1].reason, FinishReason::kCancelled);
    EXPECT_FALSE(outcomes[1].response.has_value());
    EXPECT_EQ(outcomes[2].reason, FinishReason::kError);
    EXPECT_FALSE(outcomes[2].response.has_value());
    EXPECT_EQ(outcomes[3].reason, FinishReason::kError)
        << "a response that cannot be assembled is that sequence's error";
    EXPECT_FALSE(outcomes[3].response.has_value());
    EXPECT_NE(outcomes[3].errorMessage.find("tokenizer exploded"), std::string::npos) << outcomes[3].errorMessage;
    EXPECT_EQ(session.calls(), 2) << "a cancelled or failed resident is not materialized";

    EXPECT_EQ(table.size(), 1) << "the fifth resident is still resident";
    EXPECT_NE(table.anyRequest(), nullptr);
    auto rest = table.retireEverything(FinishReason::kError);
    ASSERT_EQ(rest.size(), 1U);
    EXPECT_EQ(rest[0].record.get(), records[4].get());
    EXPECT_EQ(rest[0].reason, FinishReason::kError);
    EXPECT_TRUE(table.empty());
    EXPECT_EQ(table.anyRequest(), nullptr);
}
