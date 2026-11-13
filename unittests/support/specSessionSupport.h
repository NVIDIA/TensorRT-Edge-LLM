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

// Speculative deployments driven through a runtime session, against substitute engines whose output is a function of
// each row's token and position only. Shared by the session suite and the per-decoder suites: every speculative
// decoder implements one seating contract, so the expectation that a seated sequence decodes exactly as it does
// alone is stated once here and called from each of them.

#include "substituteEngine.h"

#include "runtime/runtimeSession.h"

#include <cuda_fp16.h>

#include <atomic>
#include <memory>
#include <unordered_map>

namespace substitute_engine
{

using namespace trt_edgellm;

//! A token the substitute model never produces itself; it is followed by EOS (id 1), so a prompt ending in it finishes
//! at its first generated token.
constexpr int32_t kEndsSequence{2};

//! The next token after `token` at `position`, as the substitute base model has it. Never the tie-break token, and
//! EOS only after kEndsSequence, so a sequence runs to its length budget unless its prompt says otherwise.
inline int32_t markovNext(int32_t token, int32_t position)
{
    if (token == kEndsSequence)
    {
        return 1;
    }
    constexpr int32_t kSpan = static_cast<int32_t>(kVocabSize) - kFirstComputedToken - 1;
    return kFirstComputedToken + (token * 7 + position * 13) % kSpan;
}

//! The token whose embedding row starts with `firstChannel` (see fakeEmbeddingValue: channel zero is
//! (token + 1) * 3 mod kFakeModulus, and 84 is the inverse of 3 modulo it).
inline int32_t tokenOfEmbedding(float firstChannel)
{
    auto const value = static_cast<int32_t>(std::lround(firstChannel));
    return (value * 84) % kFakeModulus - 1;
}

//! Write logits whose every row decodes to markovNext() of the token and position the row was computed from. A draft
//! model sees each token one position earlier than the base (its prefill pairs token t + 1 with the target's hidden
//! state at t), so it predicts markovNext(token, position + 1) -- except where `(token + position) % 3 == 0`, where
//! it guesses wrong, so its proposals are accepted over varying lengths, each a function of the sequence alone. A
//! forward that binds no per-row positions (a block draft over mask tokens) decodes to the tie-break token.
//!
//! The base also writes, into channel zero of each row's output hidden state, the token it predicts there; that is
//! the token a draft forward pairs with that hidden state. A draft forward that consumes target hidden states (its
//! prefill, or the pass over a round's accepted span: @p consumesTargetHidden) checks the pairing and guesses wrong
//! when it does not hold, so a decoder that feeds a row the wrong carried hidden state loses acceptance on it.
inline void writeMarkovLogits(rt::TensorMap const& map, bool draft, bool consumesTargetHidden, cudaStream_t stream)
{
    rt::Tensor* const logits = map.get(binding_names::kLogits);
    ASSERT_NE(logits, nullptr);
    FakeOutputScale const scale = draft ? FakeOutputScale::kLogProbabilities : FakeOutputScale::kLogits;
    rt::Tensor* const embeds = map.get(binding_names::kInputsEmbeds);
    rt::Tensor* const positions = map.get(binding_names::kPositions);
    rt::Tensor* const logitsIndices = map.get(binding_names::kLogitsIndices);
    int64_t const rows = logits->getShape().volume() / kVocabSize;
    int64_t const tokens = positions == nullptr ? 0 : positions->getShape().volume();
    if (embeds == nullptr || logitsIndices == nullptr || tokens == 0 || embeds->getDataType() != DataType::kHALF
        || logitsIndices->getShape().volume() < rows)
    {
        writeLogits(*logits, kVocabSize, {}, stream, scale);
        return;
    }
    int64_t const hiddenSize = embeds->getShape().volume() / tokens;

    std::vector<half> hostEmbeds(static_cast<size_t>(tokens * hiddenSize));
    std::vector<int32_t> hostPositions(static_cast<size_t>(tokens));
    std::vector<int64_t> hostRows(static_cast<size_t>(rows));
    CUDA_CHECK(cudaMemcpyAsync(
        hostEmbeds.data(), embeds->rawPointer(), hostEmbeds.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostPositions.data(), positions->rawPointer(), hostPositions.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostRows.data(), logitsIndices->rawPointer(), hostRows.size() * sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> tokenIds(static_cast<size_t>(tokens), -1);
    for (int64_t token = 0; token < tokens; ++token)
    {
        if (hostPositions[static_cast<size_t>(token)] >= 0)
        {
            tokenIds[static_cast<size_t>(token)]
                = tokenOfEmbedding(__half2float(hostEmbeds[static_cast<size_t>(token * hiddenSize)]));
        }
    }

    // The hidden state each token row carries (base output) or is fed (draft input), channel zero only.
    char const* const hiddenName = draft ? binding_names::kBaseModelHiddenStates : binding_names::kOutputHiddenStates;
    rt::Tensor* const hidden = map.get(hiddenName);
    bool const hiddenRows = hidden != nullptr && hidden->getDataType() == DataType::kHALF
        && hidden->getShape().getNumDims() >= 1 && hidden->getShape().volume() % tokens == 0
        && hidden->getShape().volume() / tokens > 0;
    int64_t const hiddenWidth = hiddenRows ? hidden->getShape().volume() / tokens : 0;
    if (hiddenRows && !draft)
    {
        std::vector<half> hostHidden(static_cast<size_t>(tokens * hiddenWidth), __float2half(0.0F));
        for (int64_t token = 0; token < tokens; ++token)
        {
            if (tokenIds[static_cast<size_t>(token)] >= 0)
            {
                hostHidden[static_cast<size_t>(token * hiddenWidth)] = __float2half(static_cast<float>(
                    markovNext(tokenIds[static_cast<size_t>(token)], hostPositions[static_cast<size_t>(token)])));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            hidden->rawPointer(), hostHidden.data(), hostHidden.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    }
    std::vector<half> fedHidden;
    bool const checksPairing = hiddenRows && draft && consumesTargetHidden;
    if (checksPairing)
    {
        fedHidden.resize(static_cast<size_t>(tokens * hiddenWidth));
        CUDA_CHECK(cudaMemcpyAsync(
            fedHidden.data(), hidden->rawPointer(), fedHidden.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<int32_t> winners(static_cast<size_t>(rows), -1);
    for (int64_t row = 0; row < rows; ++row)
    {
        int64_t const token = hostRows[static_cast<size_t>(row)];
        if (token < 0 || token >= tokens || hostPositions[static_cast<size_t>(token)] < 0)
        {
            continue; // padding: decodes to the tie-break token, which no proposal can match
        }
        int32_t const tokenId = tokenIds[static_cast<size_t>(token)];
        int32_t const position = hostPositions[static_cast<size_t>(token)];
        int32_t winner = markovNext(tokenId, draft ? position + 1 : position);
        bool const pairedWithItsHidden = !checksPairing
            || std::lround(__half2float(fedHidden[static_cast<size_t>(token * hiddenWidth)])) == tokenId;
        if (draft && ((tokenId + position) % 3 == 0 || !pairedWithItsHidden))
        {
            winner = winner == kFirstComputedToken ? winner + 1 : winner - 1;
        }
        winners[static_cast<size_t>(row)] = winner;
    }
    writeLogits(*logits, kVocabSize, winners, stream, scale);
}

//! The output the substitute base model gives a prompt, token by token: what every decoding path must reproduce.
inline std::vector<int32_t> markovContinuation(std::vector<int32_t> const& prompt, int64_t length)
{
    std::vector<int32_t> output;
    int32_t token = prompt.back();
    auto position = static_cast<int32_t>(prompt.size()) - 1;
    for (int64_t i = 0; i < length && token != 1; ++i)
    {
        token = markovNext(token, position++);
        output.push_back(token);
    }
    return output;
}

//! The generate budget of the session expectations: enough rounds for drafts to be accepted over varying lengths.
constexpr int64_t kSessionGenerateLength{14};

inline rt::LLMGenerationRequest makeTokenizedRequest(std::vector<int32_t> tokenIds, int64_t maxGenerateLength)
{
    rt::LLMGenerationRequest request = makeGreedyRequest("x", maxGenerateLength);
    request.preTokenizedInputIds = {std::move(tokenIds)};
    return request;
}

//! Drives a runtime session the way the engine's actor does: each admission runs its context step at once, decode
//! steps name every resident, and the caller's ledger advances by what each step reports it executed.
class SpecSessionDriver
{
public:
    explicit SpecSessionDriver(rt::RuntimeSession& session)
        : mSession(session)
    {
    }

    void admitAndPrefill(rt::LLMGenerationRequest const& request, rt::RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken = nullptr)
    {
        rt::AdmissionResult const admitted = mSession.admit(request, requestId, std::move(cancelToken));
        ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
        mLastPrefillStart = admitted.prefillStart;
        mResidents[admitted.ref.slot] = Resident{requestId, admitted.ref, admitted.prefillStart};
        rt::StepResult const result = mSession.execute(
            {rt::ScheduledSequenceDescriptor{rt::SequenceIdentity{requestId, admitted.ref}, rt::SequenceWork::kContext,
                admitted.promptLength - admitted.prefillStart, admitted.prefillStart, true}});
        ASSERT_TRUE(result.ok);
        commit(result);
    }

    void decode()
    {
        std::vector<rt::ScheduledSequenceDescriptor> plan;
        for (rt::ResidentRef const ref : mSession.residents())
        {
            Resident const& resident = mResidents.at(ref.slot);
            plan.push_back(rt::ScheduledSequenceDescriptor{
                rt::SequenceIdentity{resident.requestId, ref}, rt::SequenceWork::kDecode, 1, resident.committed, true});
        }
        rt::StepResult const result = mSession.execute(plan);
        ASSERT_TRUE(result.ok);
        commit(result);
    }

    void drain()
    {
        while (!mSession.idle())
        {
            ASSERT_NO_FATAL_FAILURE(decode());
        }
    }

    std::unordered_map<rt::RequestId, rt::LLMGenerationResponse> const& outcomes() const
    {
        return mOutcomes;
    }

    //! The prompt prefix the last admission reused from the context cache.
    int32_t lastPrefillStart() const noexcept
    {
        return mLastPrefillStart;
    }

private:
    struct Resident
    {
        rt::RequestId requestId{};
        rt::ResidentRef ref;
        int32_t committed{0};
    };

    void commit(rt::StepResult const& result)
    {
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
            mOutcomes.emplace(resident.requestId, mSession.materialize(snapshot, {}));
            mResidents.erase(ref.slot);
        }
    }

    rt::RuntimeSession& mSession;
    std::unordered_map<int32_t, Resident> mResidents;
    int32_t mLastPrefillStart{0};
    std::unordered_map<rt::RequestId, rt::LLMGenerationResponse> mOutcomes;
};

//! A speculative assembly whose engines compute writeMarkovLogits(), with the expectations a runtime session over it
//! must meet for every decoder.
class SpecSessionAssemblyTest : public SpecAssemblyTest
{
protected:
    //! A substitute engine computing writeMarkovLogits() on every forward.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeMarkovEngine(bool draft)
    {
        auto engine = makeEngine();
        struct Prepared
        {
            rt::TensorMap const* map{nullptr};
            bool consumesTargetHidden{false};
        };
        auto prepared = std::make_shared<Prepared>();
        bool const logProbabilities = draft && draftEmitsLogProbabilities();
        auto const defaultPrepare
            = prepareAction(logProbabilities ? FakeOutputScale::kLogProbabilities : FakeOutputScale::kLogits);
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([prepared, defaultPrepare](int32_t profile, rt::InferenceDims const& dims,
                               rt::TensorMap const& tensorMap, cudaStream_t stream) {
                prepared->map = &tensorMap;
                // A draft prefill, or the pass over a round's accepted span: kDraftingStep + 1 positions per
                // sequence, where the proposal chain runs kDraftingStep wide (see makeMtpDrafting).
                rt::ExecutionPhase const phase = rt::executionPhase(dims);
                prepared->consumesTargetHidden = phase == rt::ExecutionPhase::kContextPrefill
                    || phase == rt::ExecutionPhase::kContextChunk || dims.seqLen == dims.batch * kVerifySize;
                return defaultPrepare(profile, dims, tensorMap, stream);
            });
        ON_CALL(*engine, execute(::testing::_)).WillByDefault([prepared, draft](cudaStream_t stream) {
            writeMarkovLogits(*prepared->map, draft, prepared->consumesTargetHidden, stream);
            return true;
        });
        return engine;
    }

    //! The base engine's bindings beyond what makeEngine() exposes; a recurrent verify engine names its tree metadata.
    virtual void exposeBaseBindings(MockEngineExecutor& /*engine*/) {}

    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeMarkovBase()
    {
        auto engine = makeMarkovEngine(/*draft=*/false);
        exposeBaseBindings(*engine);
        return engine;
    }

    rt::LLMInferenceRuntime makeMarkovRuntime()
    {
        return makeRuntime(makeArtifacts(makeMarkovBase(), makeMarkovEngine(/*draft=*/true)));
    }

    rt::LLMGenerationResponse decodeBlocking(rt::LLMGenerationRequest const& request)
    {
        rt::LLMInferenceRuntime runtime = makeMarkovRuntime();
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        return response;
    }

    //! A sequence seated into a running batch decodes exactly as it does alone.
    //!
    //! Given two prompts and their solo decodes on the blocking path
    //! When the first runs through a session and the second is admitted while the first is mid-generation
    //! Then each sequence produces the substitute model's continuation of its prompt, the blocking path's response
    //!      and the rounds it took alone: seating the joiner's draft state moved neither its rounds nor the resident's
    void expectJoinerMatchesSolo(std::vector<int32_t> const& residentPrompt, std::vector<int32_t> const& joinerPrompt,
        int32_t residentRoundsBeforeTheJoiner)
    {
        auto const residentRequest = makeTokenizedRequest(residentPrompt, kSessionGenerateLength);
        auto const joinerRequest = makeTokenizedRequest(joinerPrompt, kSessionGenerateLength);
        rt::LLMGenerationResponse const residentBlocking = decodeBlocking(residentRequest);
        rt::LLMGenerationResponse const joinerBlocking = decodeBlocking(joinerRequest);
        ASSERT_EQ(residentBlocking.outputIds.size(), 1U);
        ASSERT_EQ(joinerBlocking.outputIds.size(), 1U);
        EXPECT_EQ(residentBlocking.outputIds.front(), markovContinuation(residentPrompt, kSessionGenerateLength));
        EXPECT_EQ(joinerBlocking.outputIds.front(), markovContinuation(joinerPrompt, kSessionGenerateLength));

        rt::LLMInferenceRuntime runtime = makeMarkovRuntime();
        ASSERT_TRUE(runtime.supportsSeatedAdmission());
        std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
        ASSERT_NE(session, nullptr);
        SpecSessionDriver driver(*session);
        driver.admitAndPrefill(residentRequest, 1);
        for (int32_t round = 0; round < residentRoundsBeforeTheJoiner; ++round)
        {
            driver.decode();
        }
        driver.admitAndPrefill(joinerRequest, 2);
        driver.drain();

        auto const sameAs
            = [](rt::LLMGenerationResponse const& joined, rt::LLMGenerationResponse const& expected, char const* what) {
                  EXPECT_EQ(joined.outputIds, expected.outputIds) << what;
                  EXPECT_EQ(joined.finishReasons, expected.finishReasons) << what;
                  EXPECT_EQ(joined.specVerifyCounts, expected.specVerifyCounts) << what;
              };
        ASSERT_EQ(driver.outcomes().count(1), 1U);
        ASSERT_EQ(driver.outcomes().count(2), 1U);
        sameAs(driver.outcomes().at(1), residentBlocking,
            "the resident's rounds moved when a sequence was seated beside it");
        sameAs(
            driver.outcomes().at(2), joinerBlocking, "the joiner's rounds moved when it was seated into a live batch");
    }

    //! A sequence admitted beside a resident but retired before its prompt ran -- cancelled between its admission and
    //! its context step -- leaves the batch without ever being seated, and the resident decodes as it does alone.
    void expectCancelledJoiner(std::vector<int32_t> const& residentPrompt)
    {
        auto const residentRequest = makeTokenizedRequest(residentPrompt, kSessionGenerateLength);
        rt::LLMGenerationResponse const residentBlocking = decodeBlocking(residentRequest);

        rt::LLMInferenceRuntime runtime = makeMarkovRuntime();
        std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
        ASSERT_NE(session, nullptr);
        SpecSessionDriver driver(*session);
        driver.admitAndPrefill(residentRequest, 1);
        driver.decode();
        auto const cancelled = std::make_shared<std::atomic<bool>>(true);
        driver.admitAndPrefill(makeTokenizedRequest({42, 43, 44}, kSessionGenerateLength), 2, cancelled);
        ASSERT_EQ(driver.outcomes().count(2), 1U) << "a cancelled admission leaves in its context step";
        EXPECT_EQ(driver.outcomes().at(2).finishReasons.front(), rt::FinishReason::kCancelled);
        driver.drain();

        ASSERT_EQ(driver.outcomes().count(1), 1U);
        EXPECT_EQ(driver.outcomes().at(1).outputIds, residentBlocking.outputIds);
        EXPECT_EQ(driver.outcomes().at(1).specVerifyCounts, residentBlocking.specVerifyCounts);
    }

    //! The seating expectations every speculative decoder meets: a seated joiner with a prompt, and with a
    //! one-position prompt (no draft prefill at all), decodes as it does alone; an admission retired unseated leaves
    //! the resident alone.
    void expectSeating()
    {
        std::vector<int32_t> const residentPrompt{5, 9, 17, 33, 21};
        ASSERT_NO_FATAL_FAILURE(
            expectJoinerMatchesSolo(residentPrompt, {42, 43, 44}, /*residentRoundsBeforeTheJoiner=*/2));
        ASSERT_NO_FATAL_FAILURE(expectJoinerMatchesSolo(residentPrompt, {42}, /*residentRoundsBeforeTheJoiner=*/1));
        ASSERT_NO_FATAL_FAILURE(expectCancelledJoiner(residentPrompt));
    }
};

} // namespace substitute_engine
