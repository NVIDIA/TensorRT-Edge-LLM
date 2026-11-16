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

// The runtime session: one generation held open for the runtime's life, into which sequences are
// admitted one at a time and out of which they leave through the steps the caller schedules. What
// is under test is the contract a scheduler builds on -- admission into an empty and into a
// populated session through one path, the plan the session accepts, the results it reports, and
// that draining to nothing never closes anything -- driven against a substitute engine so every
// token is chosen by the test.

#include "fakeEngineExecutor.h"
#include "substituteEngine.h"

#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "profiling/timer.h"
#include "runtime/decoding/requestStableRng.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/runtimeSession.h"
#include "sampler/sampling.h"

#include <atomic>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

class RuntimeSessionTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("runtimeSessionTests", makeTinyVanillaConfig());
    }
};

//! What a scheduler keeps per resident to emit steps: its identity and how far its state is
//! committed. The context step covers [prefillStart, promptLength); every decode step commits one
//! more token. This is exactly the bookkeeping the engine's resident table will carry.
struct Resident
{
    rt::RequestId requestId{};
    rt::ResidentRef ref;
    int32_t committed{0};
};

rt::ScheduledSequenceDescriptor contextStep(Resident const& resident, rt::AdmissionResult const& admitted)
{
    return rt::ScheduledSequenceDescriptor{rt::SequenceIdentity{resident.requestId, resident.ref},
        rt::SequenceWork::kContext, admitted.promptLength - admitted.prefillStart, admitted.prefillStart, true};
}

std::vector<rt::ScheduledSequenceDescriptor> decodeStep(
    rt::RuntimeSession const& session, std::unordered_map<int32_t, Resident> const& residents)
{
    std::vector<rt::ScheduledSequenceDescriptor> plan;
    for (rt::ResidentRef const ref : session.residents())
    {
        Resident const& resident = residents.at(ref.slot);
        plan.push_back(rt::ScheduledSequenceDescriptor{
            rt::SequenceIdentity{resident.requestId, ref}, rt::SequenceWork::kDecode, 1, resident.committed, true});
    }
    return plan;
}

//! Drives the session the way the engine's actor will: admit, run the context step, then decode
//! steps; commit each result into the caller's table.
class SessionDriver
{
public:
    explicit SessionDriver(rt::RuntimeSession& session)
        : mSession(session)
    {
    }

    rt::AdmissionResult admit(rt::LLMGenerationRequest const& request, rt::RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken = nullptr)
    {
        rt::AdmissionResult const admitted = mSession.admit(request, requestId, std::move(cancelToken));
        if (admitted.status == rt::AdmissionResult::Status::kAdmitted)
        {
            mResidents[admitted.ref.slot] = Resident{requestId, admitted.ref, admitted.prefillStart};
            mPendingContext.emplace(admitted);
        }
        return admitted;
    }

    //! The pending admission's context step.
    rt::StepResult context()
    {
        rt::AdmissionResult const admitted = *mPendingContext;
        mPendingContext.reset();
        rt::StepResult result = mSession.execute({contextStep(mResidents.at(admitted.ref.slot), admitted)});
        commit(result);
        return result;
    }

    rt::StepResult decode()
    {
        rt::StepResult result = mSession.execute(decodeStep(mSession, mResidents));
        commit(result);
        return result;
    }

    //! Finished responses keyed by request id.
    std::unordered_map<rt::RequestId, rt::LLMGenerationResponse> const& outcomes() const
    {
        return mOutcomes;
    }

private:
    void commit(rt::StepResult const& result)
    {
        // The ledger advances by what the runtime executed, as the engine's resident table does.
        for (auto const& executed : result.execution)
        {
            auto const it = mResidents.find(executed.identity.resident.slot);
            if (it != mResidents.end() && it->second.ref == executed.identity.resident)
            {
                EXPECT_EQ(executed.pastLength, it->second.committed) << "the runtime ran from another past length";
                it->second.committed += executed.queryLength;
            }
        }
        for (auto const& [ref, snapshot] : result.finished)
        {
            Resident const resident = mResidents.at(ref.slot);
            EXPECT_EQ(resident.ref, ref) << "a finished ref must be the one the caller holds for that slot";
            mOutcomes.emplace(resident.requestId, mSession.materialize(snapshot, {}));
            mResidents.erase(ref.slot);
        }
    }

    rt::RuntimeSession& mSession;
    std::unordered_map<int32_t, Resident> mResidents;
    std::optional<rt::AdmissionResult> mPendingContext;
    std::unordered_map<rt::RequestId, rt::LLMGenerationResponse> mOutcomes;
};

rt::LLMGenerationRequest makeTokenizedRequest(std::vector<int32_t> tokenIds, int64_t maxGenerateLength)
{
    rt::LLMGenerationRequest request = makeGreedyRequest("x", maxGenerateLength);
    request.preTokenizedInputIds = {std::move(tokenIds)};
    return request;
}

} // namespace

// Given a session running one jump-forward sequence whose grammar forces text around two free letters
// When its decode steps run
// Then the steps after a jump report context rows covering the pending run, the caller's ledger advances by them
//      so every later step's past length still matches, and the sequence spells what it would token by token
TEST_F(RuntimeSessionTest, AJumpForwardCatchUpStepReportsItsContextRowsToTheLedger)
{
    constexpr int32_t kH = 'h' - 'b' + 3, kE = 'e' - 'b' + 3, kL = 'l' - 'b' + 3, kO = 'o' - 'b' + 3;
    auto artifacts = makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);

    rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/10);
    request.requests.front().guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hell[a-z]o[a-z]", /*jumpForward=*/true};
    ASSERT_EQ(driver.admit(request, 7).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok); // h

    struct Expected
    {
        rt::SequenceWork work;
        int32_t queryLength;
        int32_t pastLength;
    };
    // e (+ forced l, l); the chunk e,l,l (+ forced o); the chunk ?,o; the decode step that samples EOS.
    std::vector<Expected> const expected{{rt::SequenceWork::kDecode, 1, 1}, {rt::SequenceWork::kContext, 3, 2},
        {rt::SequenceWork::kContext, 2, 5}, {rt::SequenceWork::kDecode, 1, 7}};
    for (size_t step = 0; step < expected.size(); ++step)
    {
        rt::StepResult const result = driver.decode();
        ASSERT_TRUE(result.ok) << "step " << step;
        ASSERT_EQ(result.execution.size(), 1U) << "step " << step;
        EXPECT_EQ(result.execution[0].work, expected[step].work) << "step " << step;
        EXPECT_EQ(result.execution[0].queryLength, expected[step].queryLength) << "step " << step;
        EXPECT_EQ(result.execution[0].pastLength, expected[step].pastLength) << "step " << step;
        EXPECT_EQ(result.finished.size(), step + 1 == expected.size() ? 1U : 0U) << "step " << step;
    }
    ASSERT_EQ(driver.outcomes().count(7), 1U);
    rt::LLMGenerationResponse const& outcome = driver.outcomes().at(7);
    ASSERT_EQ(outcome.outputIds.size(), 1U);
    ASSERT_EQ(outcome.outputIds[0].size(), 8U);
    EXPECT_EQ(outcome.outputIds[0][0], kH);
    EXPECT_EQ(outcome.outputIds[0][1], kE);
    EXPECT_EQ(outcome.outputIds[0][2], kL);
    EXPECT_EQ(outcome.outputIds[0][3], kL);
    EXPECT_EQ(outcome.outputIds[0][5], kO);
    EXPECT_EQ(outcome.finishReasons[0], rt::FinishReason::kEndId);
}

// Given an open session with nothing resident
// When sequences are admitted while others decode, finish, and free their slots
// Then every one runs through the same admit/context/decode path, slots are reused with fresh
//      epochs, results leave through the step that evicts them, and the emptied session admits
//      again without anything being reopened
TEST_F(RuntimeSessionTest, OneAdmissionPathNeverReopens)
{
    using ::testing::_;

    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5, kA4 = 6;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9, kB4 = 10;
    constexpr int32_t kC1 = 11, kC2 = 12, kC3 = 13, kC4 = 14;
    constexpr int32_t kD1 = 15, kD2 = 16, kD3 = 17, kD4 = 18;

    auto engine = makeEngine();
    auto& mock = *engine;
    // Four context steps and nine decode steps, and not one more forward than the tokens require:
    // no hidden prefill inside the open, and no teardown/reopen pass between C and D.
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(4);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(9);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1}));      // A's context step, alone
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));      // decode, A alone
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1}));      // B's context step beside A
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2})); // decode, both resident
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA4, kB3})); // A finishes and is evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC1}));      // C's context step, in A's freed slot
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB4, kC2})); // B finishes and is evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC3}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC4})); // C finishes: the session is empty again
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kD1})); // D's context step into the empty session
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kD2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kD3}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kD4}));
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    ASSERT_TRUE(runtime.supportsSeatedAdmission());

    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    EXPECT_TRUE(session->idle());
    EXPECT_TRUE(session->residents().empty());
    SessionDriver driver(*session);

    // A into the empty session: an ordinary admission followed by the context step the caller
    // schedules; the first token comes out of that step, not out of an opening.
    rt::AdmissionResult const admittedA = driver.admit(makeTokenizedRequest({41}, kMaxGenerateLength), 101);
    ASSERT_EQ(admittedA.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedA.ref, (rt::ResidentRef{0, 1}));
    EXPECT_EQ(admittedA.promptLength, 1);
    EXPECT_EQ(admittedA.prefillStart, 0);
    EXPECT_FALSE(session->idle());
    rt::StepResult const prefillA = driver.context();
    ASSERT_TRUE(prefillA.ok);
    EXPECT_EQ(prefillA.work, rt::SequenceWork::kContext);
    ASSERT_EQ(prefillA.deltas.size(), 1U);
    EXPECT_EQ(prefillA.deltas.front().first, admittedA.ref);
    EXPECT_EQ(prefillA.deltas.front().second.tokenIds, (std::vector<int32_t>{kA1}));
    EXPECT_TRUE(prefillA.finished.empty());
    EXPECT_FALSE(prefillA.publishedPrefix) << "no context cache in this deployment";
    // What ran: the plan's participant, and an execution record covering the whole prompt from
    // position zero, under the runtime's own step id.
    rt::SequenceIdentity const identityA{101, admittedA.ref};
    EXPECT_EQ(prefillA.participants, (std::vector<rt::SequenceIdentity>{identityA}));
    EXPECT_EQ(prefillA.execution,
        (std::vector<rt::ScheduledSequenceDescriptor>{{identityA, rt::SequenceWork::kContext, 1, 0, true}}));
    EXPECT_NE(prefillA.stepId, 0U) << "a step ran";

    rt::StepResult const decodeA = driver.decode();
    ASSERT_TRUE(decodeA.ok);
    EXPECT_EQ(decodeA.execution,
        (std::vector<rt::ScheduledSequenceDescriptor>{{identityA, rt::SequenceWork::kDecode, 1, 1, true}}));
    EXPECT_GT(decodeA.stepId, prefillA.stepId) << "step ids are the session's, monotonic across steps";

    // B beside A: the same admission, the same context step, while A keeps its state.
    rt::AdmissionResult const admittedB = driver.admit(makeTokenizedRequest({42}, kMaxGenerateLength), 102);
    ASSERT_EQ(admittedB.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedB.ref, (rt::ResidentRef{1, 1}));
    rt::StepResult const prefillB = driver.context();
    ASSERT_TRUE(prefillB.ok);
    ASSERT_EQ(prefillB.deltas.size(), 1U);
    EXPECT_EQ(prefillB.deltas.front().second.tokenIds, (std::vector<int32_t>{kB1}));
    EXPECT_EQ(session->residents(), (std::vector<rt::ResidentRef>{admittedA.ref, admittedB.ref}));

    ASSERT_TRUE(driver.decode().ok);
    rt::StepResult const evictsA = driver.decode();
    ASSERT_TRUE(evictsA.ok);
    ASSERT_EQ(evictsA.finished.size(), 1U);
    EXPECT_EQ(evictsA.finished.front().first, admittedA.ref);
    EXPECT_EQ(evictsA.finished.front().second.terminalReason, rt::FinishReason::kLength);
    EXPECT_EQ(session->residents(), (std::vector<rt::ResidentRef>{admittedB.ref}))
        << "B keeps its ref across A's eviction; row compaction is the session's business";

    // C takes A's released slot with a bumped epoch, so A's stale ref can never name C.
    rt::AdmissionResult const admittedC = driver.admit(makeTokenizedRequest({43}, kMaxGenerateLength), 103);
    ASSERT_EQ(admittedC.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedC.ref, (rt::ResidentRef{0, 2}));
    ASSERT_TRUE(driver.context().ok);

    ASSERT_TRUE(driver.decode().ok); // B finishes
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(driver.decode().ok); // C finishes
    EXPECT_TRUE(session->idle());
    EXPECT_TRUE(session->residents().empty());

    // The emptied session is not closed: D is admitted exactly as A was, into slot 0 again.
    rt::AdmissionResult const admittedD = driver.admit(makeTokenizedRequest({44}, kMaxGenerateLength), 104);
    ASSERT_EQ(admittedD.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedD.ref, (rt::ResidentRef{0, 3}));
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(driver.decode().ok);
    EXPECT_TRUE(session->idle());

    auto const& outcomes = driver.outcomes();
    ASSERT_EQ(outcomes.size(), 4U);
    EXPECT_EQ(outcomes.at(101).outputIds.front(), (std::vector<int32_t>{kA1, kA2, kA3, kA4}));
    EXPECT_EQ(outcomes.at(102).outputIds.front(), (std::vector<int32_t>{kB1, kB2, kB3, kB4}));
    EXPECT_EQ(outcomes.at(103).outputIds.front(), (std::vector<int32_t>{kC1, kC2, kC3, kC4}));
    EXPECT_EQ(outcomes.at(104).outputIds.front(), (std::vector<int32_t>{kD1, kD2, kD3, kD4}));
    for (auto const& [requestId, response] : outcomes)
    {
        EXPECT_EQ(response.finishReasons.front(), rt::FinishReason::kLength) << "request " << requestId;
        EXPECT_EQ(response.inputTokenCounts.front(), 1) << "request " << requestId;
    }
}

// Given streaming sequences whose first generated token is already terminal -- EOS sampled at the
//       context step, or a generate budget of one -- and one that goes on, two of them carrying a
//       token callback
// When their context steps run
// Then the terminal ones have their token delivered in a final chunk and their channel finished with
//      the reason in that same step, the same path a decode step's token takes; the one that goes
//      on has its first token on the wire before its first decode step; and a callback sees every
//      token from the context step's on, the finish flag on the last decode step's token (as on
//      the blocking path, not on a context step's), decode rounds counted from zero for each fill
//      of the session, and leaves with its request
TEST_F(RuntimeSessionTest, FirstTokenLeavesTheContextStep)
{
    using ::testing::_;
    constexpr int32_t kEos = 1; // the tiny tokenizer's <eos>
    constexpr int32_t kT1 = 5, kT2 = 6, kT3 = 7;
    auto engine = makeEngine();
    auto& mock = *engine;
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kEos})); // A: EOS at its context step
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT1}));  // B: a budget of one, spent at its context step
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT1}));  // C: goes on
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT2}));  // C's first decode step
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT3}));  // C's second, which spends its budget
    }
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    auto const streaming = [](std::vector<int32_t> prompt, int64_t maxGenerateLength,
                               std::vector<rt::TokenCallbackInfo>* seen = nullptr) {
        rt::LLMGenerationRequest request = makeTokenizedRequest(std::move(prompt), maxGenerateLength);
        request.streamChannels = {rt::StreamChannel::create()};
        if (seen != nullptr)
        {
            request.onTokenGenerated = [seen](rt::TokenCallbackInfo const& info) { seen->push_back(info); };
        }
        return request;
    };
    auto const finalChunk
        = [](rt::StreamChannel& channel, std::vector<int32_t> const& tokens, rt::FinishReason reason) {
              EXPECT_TRUE(channel.isFinished()) << "the channel was left open";
              EXPECT_EQ(channel.getReason(), reason);
              std::optional<rt::StreamChunk> chunk = channel.tryPop();
              ASSERT_TRUE(chunk.has_value()) << "the token never reached the channel";
              EXPECT_EQ(chunk->tokenIds, tokens);
              EXPECT_TRUE(chunk->finished);
              EXPECT_EQ(chunk->reason, reason);
              EXPECT_FALSE(channel.tryPop().has_value());
          };
    // A: EOS at the context step.
    rt::LLMGenerationRequest const eosAtPrefill = streaming({41}, 4);
    EXPECT_FALSE(session->requiresExclusiveAdmission(eosAtPrefill));
    ASSERT_EQ(driver.admit(eosAtPrefill, 1).status, rt::AdmissionResult::Status::kAdmitted);
    rt::StepResult const prefillA = driver.context();
    ASSERT_TRUE(prefillA.ok);
    ASSERT_EQ(prefillA.finished.size(), 1U);
    EXPECT_EQ(prefillA.finished.front().second.terminalReason, rt::FinishReason::kEndId);
    finalChunk(*eosAtPrefill.streamChannels.front(), {kEos}, rt::FinishReason::kEndId);
    EXPECT_TRUE(session->idle());
    // B: a budget of one, with a callback: the one call carries the finish flag. A callback
    // request has the session to itself.
    std::vector<rt::TokenCallbackInfo> seenB;
    rt::LLMGenerationRequest const budgetOfOne = streaming({42}, 1, &seenB);
    EXPECT_TRUE(session->requiresExclusiveAdmission(budgetOfOne));
    ASSERT_EQ(driver.admit(budgetOfOne, 2).status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_TRUE(session->exclusiveResident());
    rt::StepResult const prefillB = driver.context();
    ASSERT_TRUE(prefillB.ok);
    ASSERT_EQ(prefillB.finished.size(), 1U);
    EXPECT_EQ(prefillB.finished.front().second.terminalReason, rt::FinishReason::kLength);
    finalChunk(*budgetOfOne.streamChannels.front(), {kT1}, rt::FinishReason::kLength);
    EXPECT_TRUE(session->idle());
    ASSERT_EQ(seenB.size(), 1U) << "the first token reaches the callback from the context step";
    EXPECT_EQ(seenB.front().tokenId, kT1);
    EXPECT_EQ(seenB.front().batchIdx, 0);
    EXPECT_EQ(seenB.front().generationStep, 0);
    // As on the blocking path: the prefill's callback fires before the finish state is decided, so
    // a sequence that ends at its context step never shows the finish flag to its callback (its
    // channel is the terminal signal).
    EXPECT_FALSE(seenB.front().isFinished);
    // C: goes on, with a callback of its own, filling the emptied session.
    std::vector<rt::TokenCallbackInfo> seenC;
    rt::LLMGenerationRequest const goesOn = streaming({43}, 3, &seenC);
    ASSERT_EQ(driver.admit(goesOn, 3).status, rt::AdmissionResult::Status::kAdmitted);
    rt::StepResult const contextC = driver.context();
    ASSERT_TRUE(contextC.ok);
    EXPECT_TRUE(contextC.finished.empty());
    rt::StreamChannel& channelC = *goesOn.streamChannels.front();
    EXPECT_FALSE(channelC.isFinished());
    std::optional<rt::StreamChunk> const first = channelC.tryPop();
    ASSERT_TRUE(first.has_value()) << "the first token waits for the first decode step";
    EXPECT_EQ(first->tokenIds, (std::vector<int32_t>{kT1}));
    EXPECT_FALSE(first->finished);
    ASSERT_EQ(seenC.size(), 1U);
    EXPECT_EQ(seenC.front().tokenId, kT1);
    EXPECT_FALSE(seenC.front().isFinished);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(driver.decode().ok);
    EXPECT_TRUE(session->idle());
    std::vector<int32_t> rest;
    bool restFinished = false;
    while (std::optional<rt::StreamChunk> chunk = channelC.tryPop())
    {
        rest.insert(rest.end(), chunk->tokenIds.begin(), chunk->tokenIds.end());
        restFinished = chunk->finished;
    }
    EXPECT_EQ(rest, (std::vector<int32_t>{kT2, kT3}));
    EXPECT_TRUE(restFinished) << "the last chunk carries the finish flag";
    EXPECT_TRUE(channelC.isFinished());
    EXPECT_EQ(channelC.getReason(), rt::FinishReason::kLength);
    ASSERT_EQ(seenC.size(), 3U);
    EXPECT_EQ(seenC[1].tokenId, kT2);
    EXPECT_FALSE(seenC[1].isFinished);
    EXPECT_EQ(seenC[2].tokenId, kT3);
    EXPECT_TRUE(seenC[2].isFinished);
    // Decode rounds as the blocking path counts them: the prefill's token and the first decode
    // step's both report round 0, the next decode step round 1 -- counted from zero for this fill,
    // not from where B's left off.
    EXPECT_EQ(seenC[0].generationStep, 0);
    EXPECT_EQ(seenC[1].generationStep, 0);
    EXPECT_EQ(seenC[2].generationStep, 1);
    EXPECT_EQ(seenB.size(), 1U) << "B's callback left with B";
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), (std::vector<int32_t>{kEos}));
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), (std::vector<int32_t>{kT1}));
    EXPECT_EQ(driver.outcomes().at(3).outputIds.front(), (std::vector<int32_t>{kT1, kT2, kT3}));
}

// Given a session that closed with a grammar-constrained sequence seated (as a failed session closes)
// When a fresh session on the same runtime seats plain sequences, evicts one and seats another
//      beside the survivor
// Then nothing of the previous session's grammar reaches them: the runtime's decoder is reset
//      when the session opens, so the eviction finds no matcher and the batch is never constrained
TEST_F(RuntimeSessionTest, NoGrammarInheritedAcrossSessions)
{
    using ::testing::_;

    constexpr int32_t kEos = 1; // the tiny tokenizer's <eos>
    constexpr int32_t kT1 = 5, kT4 = 8;

    auto engine = makeEngine();
    auto& mock = *engine;
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT1}));  // A's context step
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kEos})); // B's context step: EOS, B leaves
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kEos})); // C's context step, seated beside A: EOS
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kT4}));  // A decodes and spends its budget
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    {
        std::unique_ptr<rt::RuntimeSession> doomed = runtime.openSession(mStream);
        ASSERT_NE(doomed, nullptr);
        rt::LLMGenerationRequest guided = makeTokenizedRequest({40}, 2);
        guided.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kJsonObject, ""};
        // Admission compiles the grammar into the runtime's decoder; the session then closes with
        // the sequence seated and never prefilled, as a session that failed closes.
        ASSERT_EQ(doomed->admit(guided, 1).status, rt::AdmissionResult::Status::kAdmitted);
    }

    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(makeTokenizedRequest({41}, 2), 2).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_EQ(driver.admit(makeTokenizedRequest({42}, 2), 3).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    EXPECT_EQ(session->residents().size(), 1U) << "B left; the eviction renumbered the rows";
    // C seated beside A: with the previous session's grammar still on row zero, this is where a
    // constrained batch would refuse company.
    ASSERT_EQ(driver.admit(makeTokenizedRequest({43}, 2), 4).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    EXPECT_TRUE(session->idle());
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), (std::vector<int32_t>{kT1, kT4}));
    EXPECT_EQ(driver.outcomes().at(3).outputIds.front(), (std::vector<int32_t>{kEos}));
    EXPECT_EQ(driver.outcomes().at(4).outputIds.front(), (std::vector<int32_t>{kEos}));
}

// Given requests the session never serves, that this deployment cannot take, that must run alone,
//       or whose batch-wide parameters differ from the residents'
// When they are offered
// Then admission refuses them without modifying the session, with the same distinctions the
//      scheduler's own gate draws (permanent versus runtime-reported), the input-length marker
//      the server maps to 413 rides the exception, and only the batch-wide settings (not sampling
//      or budgets) are checked against the residents
TEST_F(RuntimeSessionTest, RefusesWhatItDoesNotServe)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);

    rt::LLMGenerationRequest twoSequences = makeGreedyRequest("a", 2);
    twoSequences.requests.push_back(twoSequences.requests.front());
    EXPECT_THROW(session->admit(twoSequences, 1), std::invalid_argument);

    rt::LLMGenerationRequest lora = makeGreedyRequest("a", 2);
    lora.loraWeightsName = "adapter";
    EXPECT_THROW(session->admit(lora, 2), std::invalid_argument);

    rt::LLMGenerationRequest speech = makeGreedyRequest("a", 2);
    speech.generateAudio = true;
    EXPECT_THROW(session->admit(speech, 3), std::invalid_argument);

    rt::LLMGenerationRequest trajectory = makeGreedyRequest("a", 2);
    trajectory.requests.front().pastTrajectory.emplace();
    EXPECT_THROW(session->admit(trajectory, 5), std::invalid_argument);

    EXPECT_THROW(session->admit(makeGreedyRequest("a", 2), /*requestId=*/0), std::runtime_error);

    // Longer than the engine was built for: refused before anything is reserved, and the message
    // carries the marker the server keys on.
    std::vector<int32_t> tooLong(33, 1);
    try
    {
        session->admit(makeTokenizedRequest(std::move(tooLong), 2), 6);
        FAIL() << "an over-long prompt must be refused";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("EDGELLM_INPUT_TOO_LONG"), std::string::npos) << error.what();
    }

    // A logit bias outside the vocabulary fails validation exactly as it does on the blocking path.
    rt::LLMGenerationRequest badBias = makeGreedyRequest("a", 2);
    badBias.requests.front().logitBias = {{static_cast<int32_t>(kVocabSize) + 1, 1.0F}};
    EXPECT_THROW(session->admit(badBias, 8), std::runtime_error);

    EXPECT_TRUE(session->idle()) << "a refused admission leaves nothing resident";
    EXPECT_TRUE(session->residents().empty());

    // And the session still admits an ordinary request afterwards.
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(makeGreedyRequest("a", 1), 9).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    EXPECT_TRUE(session->idle());

    // A grammar-constrained request is exclusive: offered beside a resident it is refused, and the
    // resident is untouched.
    rt::LLMGenerationRequest guided = makeTokenizedRequest({42}, 2);
    guided.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kJsonObject, ""};
    EXPECT_TRUE(session->requiresExclusiveAdmission(guided));
    EXPECT_FALSE(session->requiresExclusiveAdmission(makeTokenizedRequest({42}, 2)));
    ASSERT_EQ(driver.admit(makeTokenizedRequest({41}, 2), 10).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    EXPECT_THROW(session->admit(guided, 11), std::runtime_error);
    // So is a request whose batch-wide settings differ from the residents': the caller's
    // compatibility gate should have kept them apart, and the session fails loudly at the seam.
    // Sampling, seed, budget and logprobs width are each sequence's own and take no part in it.
    rt::LLMGenerationRequest thinking = makeTokenizedRequest({42}, 2);
    thinking.enableThinking = true;
    EXPECT_THROW(session->admit(thinking, 12), std::runtime_error);
    EXPECT_EQ(session->residents().size(), 1U) << "a refused admission changes nothing";
    EXPECT_FALSE(session->exclusiveResident());
    ASSERT_TRUE(driver.decode().ok); // the resident finishes (length 2)
    EXPECT_TRUE(session->idle());
}

// Given an open session over a vanilla engine built for B = 2, S_in = 32 whose prefill profile does
//       not admit compact execution
// When the scheduler asks what the deployment can execute
// Then the session reports no compact execution, with that reason, one pending prefill at a time,
//      and the engine's build batch size, input length and the KV page size
TEST_F(RuntimeSessionTest, ReportsNoCompactExecutionAndTheEngineLimits)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session
        = runtime.openSession(mStream, rt::SessionOptions{/*allowCompactExecution=*/true});
    ASSERT_NE(session, nullptr);

    rt::SessionCapabilities const capabilities = session->capabilities();
    EXPECT_FALSE(capabilities.compact);
#if IS_TRT_RTX
    // TensorRT-RTX refuses compact execution before it looks at the prefill profile.
    EXPECT_EQ(capabilities.reason, "TensorRT-RTX engines may run on a GPU other than the one they were built for");
#else
    EXPECT_EQ(capabilities.reason, "the engine's prefill profile does not admit compact execution");
#endif
    EXPECT_EQ(capabilities.maxBatchSize, kMaxBatchSize);
    EXPECT_EQ(capabilities.maxInputLength, 32);
    EXPECT_EQ(capabilities.kvPageSize, 128);
}

// Given a resident decoding greedily with a budget of two tokens
// When a request with temperature 2.0, top-k 40, its own seed and a budget of three is offered beside it
// Then each sequence runs on its own settings: the resident takes the argmax and stops at two, the joiner takes
//      the sampler's draws for its seed (not the argmax) and stops at three
TEST_F(RuntimeSessionTest, OwnSamplingAndBudgetPerSequence)
{
    using ::testing::_;
    constexpr uint64_t kSeed{9U};
    // The token each of B's steps peaks at: its context step, the shared decode (B is row 1), B alone.
    std::vector<int32_t> const peaks{4, 6, 7};

    auto engine = makeEngine();
    auto& mock = *engine;
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({3}));           // A's context step
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({peaks[0]}));    // B's context step beside A
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({5, peaks[1]})); // decode both; A reaches its budget
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({peaks[2]}));    // decode B alone; B reaches its budget
    }

    rt::LLMGenerationRequest sampled = makeTokenizedRequest({42}, 3);
    sampled.temperature = 2.0F;
    sampled.topK = 40;
    sampled.samplingSeed = kSeed;

    // B's expected tokens: the kernels' draws for its seed's request-stable uniform at each position.
    std::vector<int32_t> expected;
    {
        rt::Tensor logits({1, kVocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor selected({1, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor uniform({1}, rt::DeviceType::kGPU, DataType::kFLOAT);
        SamplingParams const params(
            1, static_cast<int32_t>(kVocabSize), sampled.temperature, static_cast<int32_t>(sampled.topK), sampled.topP);
        rt::Tensor workspace(
            {static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(1, static_cast<int32_t>(kVocabSize), params))},
            rt::DeviceType::kGPU, DataType::kINT8);
        for (size_t step = 0; step < peaks.size(); ++step)
        {
            writeLogits(logits, kVocabSize, {peaks[step]}, mStream);
            float const draw = rt::requestStableUniform(kSeed,
                rt::requestStableNextAbsolutePosition(/*fullPromptTokenCount=*/1U, static_cast<uint64_t>(step)),
                rt::SpecRandomPurpose::kTarget, /*lane=*/0U);
            CUDA_CHECK(cudaMemcpyAsync(uniform.rawPointer(), &draw, sizeof(float), cudaMemcpyHostToDevice, mStream));
            topKtopPSamplingFromLogits(
                logits, selected, params, workspace, mStream, /*randomSeed=*/0U, /*randomOffset=*/0U, &uniform);
            int32_t token{};
            CUDA_CHECK(
                cudaMemcpyAsync(&token, selected.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            expected.push_back(token);
        }
    }
    ASSERT_NE(expected, peaks)
        << "the seed's draws all land on the argmax, so this fixture cannot tell a sampled row from a greedy one";

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);

    ASSERT_EQ(driver.admit(makeTokenizedRequest({41}, 2), 1).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_EQ(driver.admit(sampled, 2).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_EQ(session->residents().size(), 2U);

    ASSERT_TRUE(driver.decode().ok);
    ASSERT_EQ(session->residents().size(), 1U) << "A stopped at its own budget of two";
    ASSERT_TRUE(driver.decode().ok);
    EXPECT_TRUE(session->idle()) << "B stopped at its own budget of three";
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), (std::vector<int32_t>{3, 5}));
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), expected);
}

// Given a pending admission
// When the caller schedules a step that disagrees with the session's ledger
// Then the session refuses to run it: a decode before the context step, a context step for the
//      wrong sequence or over the wrong range, a decode naming a sequence that is not resident
TEST_F(RuntimeSessionTest, RefusesPlanDisagreeingWithLedger)
{
    using ::testing::_;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, execute(_)).WillOnce(emit({3}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);

    EXPECT_THROW(session->execute({}), std::runtime_error);

    rt::AdmissionResult const admitted = session->admit(makeTokenizedRequest({41, 42}, 4), 11);
    ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
    rt::SequenceIdentity const identity{11, admitted.ref};

    // Decode before the context step.
    EXPECT_THROW(session->execute({rt::ScheduledSequenceDescriptor{identity, rt::SequenceWork::kDecode, 1, 0, true}}),
        std::runtime_error);
    // The wrong sequence.
    EXPECT_THROW(session->execute({rt::ScheduledSequenceDescriptor{
                     rt::SequenceIdentity{12, admitted.ref}, rt::SequenceWork::kContext, 2, 0, true}}),
        std::runtime_error);
    // The wrong range: the context step covers the whole uncommitted prompt.
    EXPECT_THROW(session->execute({rt::ScheduledSequenceDescriptor{identity, rt::SequenceWork::kContext, 1, 0, true}}),
        std::runtime_error);
    // A second admission before the pending one ran its context step.
    EXPECT_THROW(session->admit(makeTokenizedRequest({43}, 4), 13), std::runtime_error);

    // The correct plan still runs.
    rt::StepResult const context
        = session->execute({rt::ScheduledSequenceDescriptor{identity, rt::SequenceWork::kContext, 2, 0, true}});
    ASSERT_TRUE(context.ok);
    ASSERT_EQ(context.deltas.size(), 1U);
    EXPECT_EQ(context.deltas.front().second.tokenIds, (std::vector<int32_t>{3}));

    // A decode naming a stale committed length, or a sequence that is not resident.
    EXPECT_THROW(session->execute({rt::ScheduledSequenceDescriptor{identity, rt::SequenceWork::kDecode, 1, 1, true}}),
        std::runtime_error);
    EXPECT_THROW(session->execute({rt::ScheduledSequenceDescriptor{
                     rt::SequenceIdentity{99, rt::ResidentRef{1, 1}}, rt::SequenceWork::kDecode, 1, 2, true}}),
        std::runtime_error);
}

// Given a device failure inside a step
// When the exception leaves execute()
// Then the caller owns every resident's outcome, and destroying the session releases what the
//      residents held without a second failure
TEST_F(RuntimeSessionTest, DeviceFailurePropagatesAndSessionCloses)
{
    using ::testing::_;

    auto engine = makeEngine();
    auto& mock = *engine;
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({3}));
        EXPECT_CALL(mock, execute(_)).WillOnce([](cudaStream_t) -> bool {
            throw std::runtime_error("device lost mid-decode");
        });
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(makeTokenizedRequest({41}, 4), 1).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    EXPECT_THROW(driver.decode(), std::runtime_error);
    EXPECT_FALSE(session->idle()) << "the failed resident is the caller's to retire";

    session.reset();
    // The runtime is usable again once the failed session is gone.
    EXPECT_NE(runtime.openSession(mStream), nullptr);
}

// Given a session with a resident decoding
// When a second sequence's seated prefill fails -- the engine reports failure, or throws
// Then that sequence alone is retired with kError, in the step that would have prefilled it; the
//      resident keeps decoding in the same session and nothing is reopened
TEST_F(RuntimeSessionTest, FailedSeatedPrefillRetiresOnlyItself)
{
    using ::testing::_;
    using ::testing::Return;

    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5;
    for (bool const throwing : {false, true})
    {
        SCOPED_TRACE(throwing ? "the prefill throws" : "the prefill reports failure");
        auto engine = makeEngine();
        auto& mock = *engine;
        {
            ::testing::InSequence forwardOrder;
            EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // A's context step
            EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2})); // decode, A alone
            if (throwing)
            {
                EXPECT_CALL(mock, execute(_)).WillOnce([](cudaStream_t) -> bool {
                    throw std::runtime_error("the prefill kernel faulted");
                });
            }
            else
            {
                EXPECT_CALL(mock, execute(_)).WillOnce(Return(false));
            }
            EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3})); // decode, A alone again
        }
        auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        rt::LLMInferenceRuntime runtime{
            std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
        ASSERT_NE(session, nullptr);
        SessionDriver driver(*session);
        ASSERT_EQ(driver.admit(makeTokenizedRequest({40, 41}, 3), 1).status, rt::AdmissionResult::Status::kAdmitted);
        ASSERT_TRUE(driver.context().ok);
        ASSERT_TRUE(driver.decode().ok);

        ASSERT_EQ(driver.admit(makeTokenizedRequest({42, 43}, 3), 2).status, rt::AdmissionResult::Status::kAdmitted);
        rt::StepResult const prefillB = driver.context();
        EXPECT_TRUE(prefillB.ok) << "the step filed B's failure; it did not fail itself";
        EXPECT_TRUE(prefillB.execution.empty());
        ASSERT_EQ(prefillB.finished.size(), 1U);
        EXPECT_EQ(prefillB.finished.front().second.terminalReason, rt::FinishReason::kError);
        EXPECT_EQ(session->residents().size(), 1U) << "A is still resident";

        ASSERT_TRUE(driver.decode().ok);
        EXPECT_TRUE(session->idle());
        EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), (std::vector<int32_t>{kA1, kA2, kA3}));
    }
}

namespace
{

//! Context reuse operates at page granularity (kTOKENS_PER_PAGE == 128): widen the tiny deployment
//! so a sequence longer than one page fits, with room for two such residents.
//! The placeholder token the deployment's context cache keys media runs by; the tiny deployment has
//! no encoder, so a prompt carrying it is one that names media the request does not supply. Out of
//! the range makeTokens() draws from, so no other prompt here names media by accident.
constexpr int32_t kImagePlaceholder = 4000;

Json makeContextReuseSessionConfig()
{
    Json config = makeTinyVanillaConfig();
    config["image_token_id"] = kImagePlaceholder;
    config["builder_config"]["max_input_len"] = 256;
    config["builder_config"]["max_kv_cache_capacity"] = 256;
    config["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kMaxBatchSize, 256);
    return config;
}

class RuntimeSessionContextReuseTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("runtimeSessionContextReuseTests", makeContextReuseSessionConfig());
    }

    static rt::ContextCacheConfig makeCacheConfig()
    {
        rt::ContextCacheConfig cacheConfig;
        cacheConfig.enabled = true;
        cacheConfig.maxRecords = 16;
        cacheConfig.maxSnapshots = 64;
        cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
        return cacheConfig;
    }

    static std::vector<int32_t> makeTokens(int32_t count)
    {
        std::vector<int32_t> tokens(static_cast<size_t>(count));
        std::iota(tokens.begin(), tokens.end(), 1);
        return tokens;
    }
};

} // namespace

// Given a session over a context-reuse deployment
// When one sequence publishes a full page and retires, a later one shares that prefix, and another
//      is cancelled between its admission and its context step
// Then each sequence leases its own pages at admission, the later one's context step covers only
//      the tail past the published page, a sequence that bypasses the lookup for itself computes
//      its whole prompt, the cancelled seat is retired without a forward pass and publishes
//      nothing, and every lease returns when its sequence retires -- with no per-request cache
//      request opened or closed in between
TEST_F(RuntimeSessionContextReuseTest, LeasesPerSequenceAndReusesPrefix)
{
    using ::testing::_;

    constexpr int32_t kPrefix = rt::kTOKENS_PER_PAGE;
    constexpr int32_t kA1 = 3, kA2 = 4, kB1 = 7, kB2 = 8, kC1 = 9, kC2 = 10, kE1 = 11, kE2 = 12;
    auto engine = makeEngine();
    auto& mock = *engine;
    std::vector<int64_t> prefillLengths;
    ON_CALL(mock, prepare(kPrefillProfile, _, _, _))
        .WillByDefault(
            [this, &prefillLengths](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                prefillLengths.push_back(dims.seqLen);
                return true;
            });
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // A's context step: the whole page plus two
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2})); // A finishes
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1})); // B's context step: only the tail past the page
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB2})); // B finishes
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC1})); // C's context step: the whole prompt, by its own policy
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC2})); // C finishes
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kE1})); // E's context step: D's prompt, never computed for D
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kE2})); // E finishes
    }
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};
    ASSERT_TRUE(runtime.supportsSeatedAdmission());
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);

    std::vector<int32_t> promptA = makeTokens(kPrefix + 2);
    rt::AdmissionResult const admittedA = driver.admit(makeTokenizedRequest(promptA, 2), 1);
    ASSERT_EQ(admittedA.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedA.promptLength, kPrefix + 2);
    EXPECT_EQ(admittedA.prefillStart, 0) << "nothing is published yet";
    rt::StepResult const prefillA = driver.context();
    ASSERT_TRUE(prefillA.ok);
    EXPECT_TRUE(prefillA.publishedPrefix) << "A's full page is published with the step that computed it";
    // Beside A, a prompt whose placeholders name media the request does not supply: refused before
    // anything is reserved -- an invalid argument, not the SessionLostError a failed lease is --
    // and A is untouched.
    EXPECT_THROW(driver.admit(makeTokenizedRequest({40, kImagePlaceholder, kImagePlaceholder, 41}, 2), 9),
        std::invalid_argument);
    EXPECT_EQ(session->residents().size(), 1U);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    // B shares A's first page and diverges after it: its lease lands on the published page and its
    // context step computes only what the cache did not cover.
    std::vector<int32_t> promptB = makeTokens(kPrefix + 1);
    promptB.push_back(7001);
    rt::AdmissionResult const admittedB = driver.admit(makeTokenizedRequest(promptB, 2), 2);
    ASSERT_EQ(admittedB.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedB.promptLength, kPrefix + 2);
    EXPECT_EQ(admittedB.prefillStart, kPrefix);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    // C is A's prompt again, but asks to bypass the lookup for itself: the sequence's own policy,
    // under the request's default, keeps the published page out of its lease (the blocking path
    // resolves the same two levels in ContextCacheRequest::begin).
    rt::LLMGenerationRequest bypass = makeTokenizedRequest(promptA, 2);
    bypass.requests.front().contextCacheLookupPolicy = rt::ContextCacheLookupPolicy::kBypass;
    rt::AdmissionResult const admittedC = driver.admit(bypass, 3);
    ASSERT_EQ(admittedC.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedC.prefillStart, 0) << "a bypassing sequence does not consult the published page";
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    // D is cancelled between its admission and its context step (the boundary the engine's actor
    // leaves between the two): the seat is retired without a forward pass -- no execution record,
    // nothing published, the sequence filed as cancelled with its channel finished and its lease
    // aborted -- and E, bringing D's prompt, takes the slot with a fresh epoch and computes it all.
    std::vector<int32_t> promptD(static_cast<size_t>(kPrefix + 2));
    std::iota(promptD.begin(), promptD.end(), 2000);
    rt::LLMGenerationRequest cancelled = makeTokenizedRequest(promptD, 2);
    cancelled.streamChannels = {rt::StreamChannel::create()};
    auto const token = std::make_shared<std::atomic<bool>>(false);
    rt::AdmissionResult const admittedD = driver.admit(cancelled, 4, token);
    ASSERT_EQ(admittedD.status, rt::AdmissionResult::Status::kAdmitted);
    token->store(true);
    rt::StepResult const prefillD = driver.context();
    ASSERT_TRUE(prefillD.ok);
    EXPECT_TRUE(prefillD.execution.empty());
    EXPECT_FALSE(prefillD.publishedPrefix) << "nothing was computed, so nothing is published";
    ASSERT_EQ(prefillD.finished.size(), 1U);
    EXPECT_EQ(prefillD.finished.front().second.terminalReason, rt::FinishReason::kCancelled);
    EXPECT_TRUE(driver.outcomes().at(4).outputIds.front().empty());
    rt::StreamChannel const& channelD = *cancelled.streamChannels.front();
    EXPECT_TRUE(channelD.isFinished()) << "the session finishes the channel of a seat it retires";
    EXPECT_EQ(channelD.getReason(), rt::FinishReason::kCancelled);
    EXPECT_TRUE(session->idle());
    rt::AdmissionResult const admittedE = driver.admit(makeTokenizedRequest(promptD, 2), 5);
    ASSERT_EQ(admittedE.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedE.prefillStart, 0) << "a cancelled seat publishes nothing";
    EXPECT_EQ(admittedE.ref.slot, admittedD.ref.slot) << "the slot is reused";
    EXPECT_GT(admittedE.ref.epoch, admittedD.ref.epoch) << "with a fresh epoch";
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());
    ASSERT_EQ(prefillLengths.size(), 4U);
    EXPECT_EQ(prefillLengths[0], kPrefix + 2);
    EXPECT_EQ(prefillLengths[1], 2) << "B's context step must skip the reused page";
    EXPECT_EQ(prefillLengths[2], kPrefix + 2) << "C's context step must cover its whole prompt";
    EXPECT_EQ(prefillLengths[3], kPrefix + 2) << "E's context step must cover the prompt D never computed";
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), (std::vector<int32_t>{kA1, kA2}));
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), (std::vector<int32_t>{kB1, kB2}));
    EXPECT_EQ(driver.outcomes().at(3).outputIds.front(), (std::vector<int32_t>{kC1, kC2}));
    EXPECT_EQ(driver.outcomes().at(5).outputIds.front(), (std::vector<int32_t>{kE1, kE2}));
    // Closing the session returns the one cache request it held; the coordinator's own destructor
    // proves quiescence (it terminates otherwise), so reaching the end of this test is the check
    // that every lease, D's aborted one included, came back.
}
namespace
{

//! A context-reuse deployment whose input limit is one page plus a little, under a KV capacity of
//! four pages: a prompt can be longer than what one context step may compute.
Json makeShortInputContextReuseConfig()
{
    Json config = makeTinyVanillaConfig();
    config["builder_config"]["max_input_len"] = rt::kTOKENS_PER_PAGE + 8;
    config["builder_config"]["max_kv_cache_capacity"] = 4 * rt::kTOKENS_PER_PAGE;
    config["builder_config"]["max_kv_pool_pages"]
        = rt::computeMinimumKvPoolPages(kMaxBatchSize, 4 * rt::kTOKENS_PER_PAGE);
    return config;
}

class RuntimeSessionShortInputContextReuseTest : public RuntimeSessionContextReuseTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("runtimeSessionShortInputContextReuseTests", makeShortInputContextReuseConfig());
    }
};

} // namespace

// Given a session over a context-reuse deployment whose input limit is shorter than its KV capacity
// When prompts longer than the input limit are offered, with and without a published prefix
//      covering the excess, and one longer than the KV capacity
// Then the input limit binds what the context step computes, not the whole prompt: a prompt whose
//      uncached part fits is admitted, one whose uncached part does not is refused with its
//      reservation given back and nothing resident, and a prompt beyond the KV capacity is refused
//      before any lease is taken -- each with the marker the server maps to 413, as on the blocking path
TEST_F(RuntimeSessionShortInputContextReuseTest, InputLimitBindsUncachedPrompt)
{
    using ::testing::_;

    constexpr int32_t kPrefix = rt::kTOKENS_PER_PAGE;
    constexpr int32_t kInputLimit = kPrefix + 8;
    constexpr int32_t kKvCapacity = 4 * kPrefix;
    constexpr int32_t kA1 = 3, kA2 = 4, kB1 = 5, kB2 = 6, kE1 = 7, kE2 = 8;

    auto engine = makeEngine();
    auto& mock = *engine;
    std::vector<int64_t> prefillLengths;
    ON_CALL(mock, prepare(kPrefillProfile, _, _, _))
        .WillByDefault(
            [this, &prefillLengths](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                prefillLengths.push_back(dims.seqLen);
                return true;
            });
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // A: one page plus two, published
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1})); // B: longer than the limit, its tail under it
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kE1})); // E: after the two refusals
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kE2}));
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);

    auto const refusedWith = [&](std::vector<int32_t> prompt, rt::RequestId requestId, char const* marker) {
        try
        {
            static_cast<void>(session->admit(makeTokenizedRequest(std::move(prompt), 2), requestId));
            ADD_FAILURE() << "request " << requestId << " was admitted";
        }
        catch (std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find("EDGELLM_INPUT_TOO_LONG"), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find(marker), std::string::npos) << error.what();
        }
        EXPECT_TRUE(session->idle()) << "a refused admission leaves nothing resident";
    };

    // A publishes its first page.
    rt::AdmissionResult const admittedA = driver.admit(makeTokenizedRequest(makeTokens(kPrefix + 2), 2), 1);
    ASSERT_EQ(admittedA.status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    // B is longer than the input limit as a whole; past the published page it is not.
    rt::AdmissionResult const admittedB = driver.admit(makeTokenizedRequest(makeTokens(kPrefix + 16), 2), 2);
    ASSERT_EQ(admittedB.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedB.promptLength, kPrefix + 16);
    EXPECT_EQ(admittedB.prefillStart, kPrefix);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    // C's uncached part is one over the limit: refused before any page is leased.
    refusedWith(makeTokens(kPrefix + kInputLimit + 1), 3, "max_input_len");
    // D is longer than the KV capacity: refused before any lease.
    refusedWith(makeTokens(kKvCapacity + 1), 4, "max_kv_cache_capacity");

    // The session is whole, the published page still there.
    rt::AdmissionResult const admittedE = driver.admit(makeTokenizedRequest(makeTokens(kPrefix + 2), 2), 5);
    ASSERT_EQ(admittedE.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedE.prefillStart, kPrefix);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_TRUE(session->idle());

    ASSERT_EQ(prefillLengths.size(), 3U);
    EXPECT_EQ(prefillLengths[0], kPrefix + 2);
    EXPECT_EQ(prefillLengths[1], 16);
    EXPECT_EQ(prefillLengths[2], 2);
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), (std::vector<int32_t>{kA1, kA2}));
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), (std::vector<int32_t>{kB1, kB2}));
    EXPECT_EQ(driver.outcomes().at(5).outputIds.front(), (std::vector<int32_t>{kE1, kE2}));
    // As above, the coordinator's destructor is the check that C's refusal left nothing behind.
}

// Given a session whose context cache has three of its eight pages free, the least recently used
//      conversation's two published pages next in line for eviction
// When a cold prompt that fits the KV capacity but whose uncached part exceeds the input limit is offered
// Then it is refused with the marker the server maps to 413 before anything is evicted, so that
//      conversation's next turn still skips both pages
TEST_F(RuntimeSessionShortInputContextReuseTest, AnOverlongPromptIsRefusedBeforeEvictingAnotherConversation)
{
    constexpr int32_t kPage = rt::kTOKENS_PER_PAGE;
    auto artifacts = makeVanillaArtifacts(mModelDir, makeFakeEngine(makeShortInputContextReuseConfig()), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    rt::RequestId nextId = 1;
    auto const request = [](int32_t length, int32_t variant) {
        return makeTokenizedRequest(promptTokenIds(makePrompt(length, variant)), /*maxGenerateLength=*/1);
    };
    auto const serve = [&](int32_t length, int32_t variant) {
        rt::AdmissionResult const admitted = driver.admit(request(length, variant), nextId++);
        return admitted.status == rt::AdmissionResult::Status::kAdmitted && driver.context().ok && session->idle();
    };

    // Every turn's uncached part fits the input limit: conversation 1 publishes two pages, then conversation 2 three.
    ASSERT_TRUE(serve(kPage + 2, 1));
    ASSERT_TRUE(serve(2 * kPage + 2, 1));
    for (int32_t pages = 1; pages <= 3; ++pages)
    {
        ASSERT_TRUE(serve(pages * kPage + 2, 2));
    }
    auto const before = runtime.getContextCacheMetrics();
    ASSERT_TRUE(before.has_value());

    try
    {
        static_cast<void>(session->admit(request(3 * kPage + 16, 3), nextId++));
        ADD_FAILURE() << "a prompt whose uncached part exceeds the input limit was admitted";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("EDGELLM_INPUT_TOO_LONG"), std::string::npos) << error.what();
    }
    EXPECT_TRUE(session->idle());
    EXPECT_EQ(runtime.getContextCacheMetrics()->evictedRecords, before->evictedRecords);

    rt::AdmissionResult const returning = driver.admit(request(2 * kPage + 7, 1), nextId++);
    ASSERT_EQ(returning.status, rt::AdmissionResult::Status::kAdmitted);
    rt::StepResult const context = driver.context();
    ASSERT_TRUE(context.ok);
    ASSERT_EQ(context.execution.size(), 1U);
    EXPECT_EQ(context.execution[0].queryLength, 7) << "conversation 1's published pages were evicted";
}

// ---------------------------------------------------------------------------
// Profiling under the session. The runtime's timer is reselected on every step and a profiling
// request spans one fill of the session, so captures on another runtime sharing the thread
// neither steal samples nor disturb request identities.
// ---------------------------------------------------------------------------

namespace
{

struct ScopedSessionProfiling
{
    bool previous{getProfilingEnabled()};
    ScopedSessionProfiling()
    {
        setProfilingEnabled(true);
    }
    ~ScopedSessionProfiling()
    {
        setProfilingEnabled(previous);
    }
};

int64_t selectedStageCount(std::string const& stage)
{
    auto const data = gTimer.getTimingData(stage);
    return data ? data->getTotalRuns() : 0;
}

} // namespace

// Given a session on one runtime interleaved with requests and graph captures on another
// When each sequence fills the empty session in turn, and when a second sequence is seated beside a
//      resident decoding under profiling
// Then every sample keeps the session runtime's profile: each fill is a distinct profiling request,
//      the seated prefill's forward carries the running request's identity, and the other runtime's
//      profile sees none of it
TEST_F(RuntimeSessionTest, ProfilingFollowsSessionRuntime)
{
    // Each fill of the empty session is its own profiling request.
    {
        rt::LLMInferenceRuntime runtimeA{makeVanillaArtifacts(mModelDir, makeEngine(), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        rt::LLMInferenceRuntime runtimeB{makeVanillaArtifacts(mModelDir, makeEngine(), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        ScopedSessionProfiling profiling;
        auto request = makeGreedyRequest("a", 2);
        request.ignoreEos = true;

        std::unique_ptr<rt::RuntimeSession> session = runtimeA.openSession(mStream);
        ASSERT_NE(session, nullptr);
        SessionDriver driver(*session);
        std::vector<uint64_t> requestIds;
        for (rt::RequestId const requestId : {rt::RequestId{101}, rt::RequestId{102}})
        {
            static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
            ASSERT_EQ(driver.admit(request, requestId).status, rt::AdmissionResult::Status::kAdmitted);
            static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
            ASSERT_TRUE(driver.context().ok);
            static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
            ASSERT_TRUE(driver.decode().ok);
            ASSERT_TRUE(session->idle()) << "two tokens: the context step's and one decode step's";
            for (auto const& sample : gTimer.getSamples())
            {
                if (sample.stageId == "decode_round" && (requestIds.empty() || requestIds.back() != sample.requestId))
                {
                    requestIds.push_back(sample.requestId);
                }
            }
        }
        EXPECT_EQ(selectedStageCount("llm_prefill"), 2);
        EXPECT_EQ(selectedStageCount("decode_round"), 2);
        EXPECT_EQ(requestIds.size(), 2U) << "each fill of the empty session starts a profiling request";
        static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
        EXPECT_EQ(selectedStageCount("decode_round"), 0) << "the other runtime's profile saw none of it";
    }
    // A seated prefill is timed under the running profiling request.
    {
        rt::LLMInferenceRuntime runtimeA{makeVanillaArtifacts(mModelDir, makeEngine(), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        rt::LLMInferenceRuntime runtimeB{makeVanillaArtifacts(mModelDir, makeEngine(), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        ScopedSessionProfiling profiling;
        auto request = makeGreedyRequest("a", 4);
        request.ignoreEos = true;
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtimeA.handleRequest(request, response, mStream));

        std::unique_ptr<rt::RuntimeSession> session = runtimeA.openSession(mStream);
        ASSERT_NE(session, nullptr);
        SessionDriver driver(*session);
        ASSERT_EQ(driver.admit(request, 101).status, rt::AdmissionResult::Status::kAdmitted);
        ASSERT_TRUE(driver.context().ok);
        auto const foundingSamples = gTimer.getSamples();
        ASSERT_FALSE(foundingSamples.empty());
        uint64_t const requestId = foundingSamples.back().requestId;
        EXPECT_EQ(selectedStageCount("llm_prefill"), 2);

        auto other = makeGreedyRequest("aaa", 2);
        other.ignoreEos = true;
        ASSERT_TRUE(runtimeB.handleRequest(other, response, mStream));
        auto const otherSamples = gTimer.getSamples();
        ASSERT_FALSE(otherSamples.empty());
        ASSERT_NE(requestId, otherSamples.back().requestId);

        rt::LLMGenerationRequest joiner = makeTokenizedRequest({43, 43}, 4);
        joiner.ignoreEos = true;
        rt::AdmissionResult const admitted = driver.admit(joiner, 102);
        ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
        EXPECT_EQ(selectedStageCount("llm_prefill"), 2) << "admission runs no prefill";
        ASSERT_FALSE(gTimer.getSamples().empty());
        EXPECT_EQ(gTimer.getSamples().back().requestId, requestId);

        static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
        ASSERT_TRUE(driver.context().ok);
        EXPECT_EQ(selectedStageCount("llm_prefill"), 3);
        auto const& samples = gTimer.getSamples();
        ASSERT_GT(samples.size(), foundingSamples.size());
        bool foundJoinedForward = false;
        for (size_t i = foundingSamples.size(); i < samples.size(); ++i)
        {
            EXPECT_EQ(samples[i].requestId, requestId);
            EXPECT_EQ(samples[i].round, -1);
            if (samples[i].forward)
            {
                foundJoinedForward = true;
                EXPECT_EQ(samples[i].stageId, "forward/base");
                EXPECT_EQ(samples[i].queryTokens, 2);
                EXPECT_EQ(samples[i].parentStage, "llm_prefill");
            }
        }
        EXPECT_TRUE(foundJoinedForward);
        static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
        EXPECT_EQ(selectedStageCount("llm_prefill"), 1);
        EXPECT_EQ(selectedStageCount("forward/base"), 2);
        EXPECT_EQ(selectedStageCount("decode_round"), 1);
        while (!session->idle())
        {
            ASSERT_TRUE(driver.decode().ok);
        }
    }
}

// Given two prompts whose solo decodes on the blocking path differ
// When the first runs through a session alone and the second is admitted while it is mid-generation
// Then each sequence's response is the blocking path's, field for field: neither the resident's
//      state nor the joiner's seated prefill moved a token; and the session holds the runtime's
//      one-request latch for as long as it is open
TEST_F(RuntimeSessionTest, MatchesBlockingPathAloneAndJoined)
{
    constexpr int64_t kMaxGenerateLength{4};
    std::vector<int32_t> const joinerPrompt{42, 43, 44};
    auto const decodeAlone = [this](rt::LLMGenerationRequest const& request) {
        rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        return response;
    };
    auto const firstRequest = makeGreedyRequest("a", kMaxGenerateLength);
    auto const joinerRequest = makeTokenizedRequest(joinerPrompt, kMaxGenerateLength);
    rt::LLMGenerationResponse const firstAlone = decodeAlone(firstRequest);
    rt::LLMGenerationResponse const joinerAlone = decodeAlone(joinerRequest);
    ASSERT_EQ(firstAlone.outputIds.size(), 1U);
    ASSERT_EQ(firstAlone.outputIds.front().size(), static_cast<size_t>(kMaxGenerateLength));
    ASSERT_NE(firstAlone.outputIds, joinerAlone.outputIds)
        << "the two prompts decode alike, so this fixture cannot tell one sequence's state from the other's";

    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream), mModelDir.string(),
        /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    // The session holds the runtime's one-request latch: the blocking path is refused until it
    // closes, and a second session cannot open beside it.
    rt::LLMGenerationResponse refused;
    EXPECT_FALSE(runtime.handleRequest(firstRequest, refused, mStream));
    EXPECT_EQ(runtime.openSession(mStream), nullptr);
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(firstRequest, 1).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok); // the first sequence is mid-generation by construction
    ASSERT_EQ(driver.admit(joinerRequest, 2).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    while (!session->idle())
    {
        ASSERT_TRUE(driver.decode().ok);
    }
    auto const sameAs
        = [](rt::LLMGenerationResponse const& viaSession, rt::LLMGenerationResponse const& alone, char const* who) {
              EXPECT_EQ(viaSession.outputIds, alone.outputIds) << who;
              EXPECT_EQ(viaSession.outputTexts, alone.outputTexts) << who;
              EXPECT_EQ(viaSession.finishReasons, alone.finishReasons) << who;
              EXPECT_EQ(viaSession.inputTokenCounts, alone.inputTokenCounts) << who;
          };
    ASSERT_EQ(driver.outcomes().count(1), 1U);
    ASSERT_EQ(driver.outcomes().count(2), 1U);
    sameAs(driver.outcomes().at(1), firstAlone, "the resident's answer moved when a second sequence joined");
    sameAs(driver.outcomes().at(2), joinerAlone, "the joiner's answer moved when it was seated into a live batch");
    session.reset();
    EXPECT_NE(runtime.openSession(mStream), nullptr) << "the latch is released with the session";
}

// Given a session serving a resident with a one-token prompt, and a joiner whose 32-token prompt leaves room in the
//       KV capacity of 64 for 31 of the 40 tokens both ask for
// When the joiner is admitted beside the resident and both run to completion
// Then the joiner is admitted rather than refused as transient and stops after 31 tokens, its budget bounded by the
//      capacity behind its own prompt alone, and the resident generates all 40
TEST_F(RuntimeSessionTest, AJoinerBudgetIsBoundedByItsOwnPrompt)
{
    std::vector<int32_t> joinerPrompt(32);
    std::iota(joinerPrompt.begin(), joinerPrompt.end(), 10);
    rt::LLMGenerationRequest resident = makeTokenizedRequest({41}, 40);
    rt::LLMGenerationRequest joiner = makeTokenizedRequest(joinerPrompt, 40);
    resident.ignoreEos = joiner.ignoreEos = true;
    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream), mModelDir.string(),
        /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(resident, 1).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_TRUE(driver.decode().ok);
    ASSERT_EQ(driver.admit(joiner, 2).status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    while (!session->idle())
    {
        ASSERT_TRUE(driver.decode().ok);
    }

    ASSERT_EQ(driver.outcomes().count(1), 1U);
    ASSERT_EQ(driver.outcomes().count(2), 1U);
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front().size(), 40U);
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front().size(), 31U);
    EXPECT_EQ(driver.outcomes().at(2).finishReasons.front(), rt::FinishReason::kLength);
}

// Given an open session over a vanilla engine
// When a request with a maxGenerateLength of 0 or -1 is admitted
// Then it is refused naming maxGenerateLength as not positive, not as a KV capacity shortage, and nothing is resident
TEST_F(RuntimeSessionTest, ANonPositiveGenerateLengthIsRefusedAsSuch)
{
    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream), mModelDir.string(),
        /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    for (int64_t const requested : {int64_t{0}, int64_t{-1}})
    {
        EXPECT_THAT(
            [&] {
                static_cast<void>(
                    session->admit(makeTokenizedRequest({41}, requested), static_cast<rt::RequestId>(1 - requested)));
            },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("maxGenerateLength must be positive; got " + std::to_string(requested))));
        EXPECT_TRUE(session->idle());
    }
}

// --------------------------------------------------------------------------
// Hybrid deployments: recurrent and convolution state lives in a pool with one row per resident
// slot, and the kernels read that row as the state before a sequence's first token.
// --------------------------------------------------------------------------

namespace
{

class RuntimeSessionHybridTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("runtimeSessionHybridTests", makeTinyHybridConfig());
    }
};

//! The two state pools of the deployment's one recurrent layer, as the engine binds them.
std::vector<std::string> hybridStatePools()
{
    return {binding_names::formatRecurrentStateName(0, /*isPast=*/false),
        binding_names::formatConvStateName(0, /*isPast=*/false)};
}

} // namespace

// Given a session that closed with a hybrid sequence still resident, as a failed session closes,
//      so no eviction ever cleared that sequence's state row
// When the next session seats a sequence into the same slot
// Then that sequence's first forward starts from cleared recurrent and convolution state, not
//      from what the previous sequence left there: the admission clears the slot it takes
TEST_F(RuntimeSessionHybridTest, SeatStartsFromClearedRecurrentState)
{
    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kToken{5};

    auto engine = makeFakeEngine(makeTinyHybridConfig());
    auto* const raw = engine.get();
    // Every state entry a forward writes is non-zero, so a cleared row cannot be mistaken for one a forward left.
    raw->forceStateValues([](FakeStateSlot const& slot) { return 1.0 + static_cast<double>(slot.position); });
    raw->forceOutputTokens({kToken, kToken});
    raw->recordBinding(binding_names::kStateIndices);
    for (auto const& pool : hybridStatePools())
    {
        raw->recordStatePool(pool);
    }

    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, std::move(engine), mStream), mModelDir.string(),
        /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    {
        std::unique_ptr<rt::RuntimeSession> abandoned = runtime.openSession(mStream);
        ASSERT_NE(abandoned, nullptr);
        SessionDriver driver(*abandoned);
        ASSERT_EQ(driver.admit(makeTokenizedRequest({40, 41, 42}, kMaxGenerateLength), 1).status,
            rt::AdmissionResult::Status::kAdmitted);
        ASSERT_TRUE(driver.context().ok);
        ASSERT_TRUE(driver.decode().ok);
        // A is still resident when the session closes: nothing evicts it, nothing clears its row.
    }
    size_t const roundsBefore = raw->preparedRounds().size();
    ASSERT_EQ(roundsBefore, 2U) << "A's context step and one decode step";
    int32_t const slotOfA = raw->preparedBinding(binding_names::kStateIndices).back().at(0);
    for (auto const& pool : hybridStatePools())
    {
        EXPECT_NE(raw->preparedStatePool(pool).back().at(static_cast<size_t>(slotOfA)), 0.0)
            << pool << ": A's decode step started from the state its context step left";
    }

    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);
    ASSERT_EQ(driver.admit(makeTokenizedRequest({43, 44}, kMaxGenerateLength), 2).status,
        rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_EQ(raw->preparedRounds().size(), roundsBefore + 1U);
    int32_t const slotOfB = raw->preparedBinding(binding_names::kStateIndices).back().at(0);
    ASSERT_EQ(slotOfB, slotOfA) << "the fresh session's pool hands out the slot A held";
    for (auto const& pool : hybridStatePools())
    {
        EXPECT_EQ(raw->preparedStatePool(pool).back().at(static_cast<size_t>(slotOfB)), 0.0)
            << pool << ": B's first forward must not start from A's state";
    }
}

// Given a hybrid session holding A and B, where A finishes first, so the eviction repacks B into
//      execution row 0 while B keeps resident slot 1, and C then takes resident slot 0 in row 1
// When C's context step and the next decode step run
// Then the engine is handed the resident slots, not the rows (state_indices [0], then [1, 0]), C
//      starts from a cleared state row and B's row still holds B's state: rows and resident slots
//      have diverged, and the state follows the resident
TEST_F(RuntimeSessionHybridTest, StateFollowsSlotNotRow)
{
    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kEos = 1; // the tiny tokenizer's <eos>
    constexpr int32_t kToken{5};

    auto engine = makeFakeEngine(makeTinyHybridConfig());
    auto* const raw = engine.get();
    raw->forceStateValues([](FakeStateSlot const& slot) { return 1.0 + static_cast<double>(slot.position); });
    raw->forceOutputTokensPerRound({
        {kToken},         // A's context step
        {kToken},         // B's context step, projected to one row
        {kEos, kToken},   // decode over [A, B]: A finishes, the eviction repacks B into row 0
        {kToken},         // C's context step, projected to one row
        {kToken, kToken}, // decode over [B, C], from here on
    });
    raw->recordBinding(binding_names::kStateIndices);
    for (auto const& pool : hybridStatePools())
    {
        raw->recordStatePool(pool);
    }

    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, std::move(engine), mStream), mModelDir.string(),
        /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SessionDriver driver(*session);

    rt::AdmissionResult const admittedA = driver.admit(makeTokenizedRequest({40, 41}, kMaxGenerateLength), 1);
    ASSERT_EQ(admittedA.status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    rt::AdmissionResult const admittedB = driver.admit(makeTokenizedRequest({42, 43}, kMaxGenerateLength), 2);
    ASSERT_EQ(admittedB.status, rt::AdmissionResult::Status::kAdmitted);
    ASSERT_TRUE(driver.context().ok);
    ASSERT_EQ(admittedA.ref.slot, 0);
    ASSERT_EQ(admittedB.ref.slot, 1);

    ASSERT_TRUE(driver.decode().ok);
    ASSERT_EQ(driver.outcomes().count(1), 1U) << "A finished on EOS";
    ASSERT_EQ(session->residents(), (std::vector<rt::ResidentRef>{admittedB.ref}));

    rt::AdmissionResult const admittedC = driver.admit(makeTokenizedRequest({44, 45}, kMaxGenerateLength), 3);
    ASSERT_EQ(admittedC.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedC.ref.slot, 0) << "C takes the resident slot A freed";
    ASSERT_TRUE(driver.context().ok);
    // C's context step: one projected row, and the engine is told C's resident slot for it.
    EXPECT_EQ(raw->preparedBinding(binding_names::kStateIndices).back(), (std::vector<int32_t>{0}));
    for (auto const& pool : hybridStatePools())
    {
        auto const& rows = raw->preparedStatePool(pool).back();
        EXPECT_EQ(rows.at(0), 0.0) << pool << ": C must start from a cleared row, not from A's state";
        EXPECT_NE(rows.at(1), 0.0) << pool << ": B's row is untouched by A's eviction and C's admission";
    }

    // Execution rows [B, C] over resident slots [1, 0].
    ASSERT_EQ(session->residents(), (std::vector<rt::ResidentRef>{admittedB.ref, admittedC.ref}));
    ASSERT_TRUE(driver.decode().ok);
    EXPECT_EQ(raw->preparedBinding(binding_names::kStateIndices).back(), (std::vector<int32_t>{1, 0}));
}
