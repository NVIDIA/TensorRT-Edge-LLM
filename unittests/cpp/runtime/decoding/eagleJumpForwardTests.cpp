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

// Jump-forward requests on EAGLE deployments assembled around two substitute engines: the base engine's verify
// headroom decides whether a guided request the EAGLE decoder serves gets jump-forward, a request the registry routes
// to the vanilla fallback catches its forced tokens up on a context chunk, and a granted EAGLE request grafts the
// forced chain it plans into the next verify round.

#include "guidedTokenizerFixture.h"
#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! EAGLE-3's draft reads three base layers' hidden states; the base has to name that many layers.
constexpr int32_t kEagleHiddenStateLayers{3};
//! Verify tokens the jumpable deployment's engines hold beyond the verify width.
constexpr int32_t kVerifyHeadroom{8};

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kEos{1};
constexpr int32_t kH{'h' - 'b' + 3};
constexpr int32_t kE{'e' - 'b' + 3};
constexpr int32_t kI{'i' - 'b' + 3};
constexpr int32_t kL{'l' - 'b' + 3};
constexpr int32_t kO{'o' - 'b' + 3};
constexpr int32_t kQ{'q' - 'b' + 3};
constexpr int32_t kX{'x' - 'b' + 3};
constexpr int32_t kY{'y' - 'b' + 3};
constexpr int32_t kZ{'z' - 'b' + 3};

//! The jumpable deployment's derived limits: a wide round verifies V + headroom nodes and accepts headroom + 1.
constexpr int32_t kWideVerifySize{kVerifySize + kVerifyHeadroom};
constexpr int32_t kWideAcceptSize{kVerifyHeadroom + 1};
constexpr int32_t kNarrowAcceptSize{kDraftingStep + 1};

//! The tree-drafting deployment: top-2 over two steps, four verify nodes, the same headroom.
constexpr int32_t kTreeTopK{2};
constexpr int32_t kTreeSteps{2};
constexpr int32_t kTreeVerifySize{4};
constexpr int32_t kTreeWideVerifySize{kTreeVerifySize + kVerifyHeadroom};

Json makeEagleBaseConfig(int32_t maxVerifyTokens)
{
    Json config = makeSpecBaseConfig("eagle3", maxVerifyTokens);
    config["num_hidden_layers"] = kEagleHiddenStateLayers;
    Json layerIds = Json::array();
    for (int32_t layer = 0; layer < kEagleHiddenStateLayers; ++layer)
    {
        layerIds.push_back(layer);
    }
    config["eagle_hidden_state_layers"] = std::move(layerIds);
    return config;
}

Json makeEagleDraftConfig(int32_t maxDraftTokens)
{
    Json config = makeSpecDraftConfig("eagle3", maxDraftTokens);
    config["base_model_hidden_size"] = kEagleHiddenStateLayers * config["hidden_size"].get<int32_t>();
    return config;
}

//! An EAGLE deployment whose engine capacities the suite chooses. Requests run with the logger at DEBUG and both
//! log streams captured: a refusal is a warning, and a slot that ran with jump-forward gets a statistics line when
//! its request completes.
class EagleJumpForwardTestBase : public SpecAssemblyTest
{
protected:
    void SetUp() override
    {
        mSavedLevel = gLogger.getLevel();
        SpecAssemblyTest::SetUp();
    }

    void TearDown() override
    {
        SpecAssemblyTest::TearDown();
        gLogger.setLevel(mSavedLevel);
    }

    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }

    //! A greedy request whose grammar forces the literal `hello`, asking for jump-forward.
    static rt::LLMGenerationRequest makeHelloRequest()
    {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/8);
        request.requests.front().guidedDecoding
            = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hello", /*jumpForward=*/true};
        return request;
    }

    static std::vector<int32_t> hello()
    {
        return {kH, kE, kL, kL, kO, kEos};
    }

    //! One prepare() of a recording engine: the dims it asked for, the ragged query and past lengths it was handed,
    //! and the KV pages each row's page-table entry maps. A draft engine also gets the token behind each of its
    //! input embeddings, since an engine consumes embeddings, not ids.
    struct Prepared
    {
        int32_t profile{};
        rt::InferenceDims dims{};
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> pastLengths;
        std::vector<int32_t> mappedPages;
        std::vector<int32_t> embeddedTokens;
    };

    //! makeEngine() or makeLogProbDraftEngine(), recording every prepare() into `log`.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingEngine(std::vector<Prepared>& log, bool draft)
    {
        auto engine = draft ? makeLogProbDraftEngine() : makeEngine();
        FakeOutputScale const scale = draft ? FakeOutputScale::kLogProbabilities : FakeOutputScale::kLogits;
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([this, &log, draft, scale](int32_t profile, rt::InferenceDims const& dims,
                               rt::TensorMap const& map, cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = scale;
                mLastDims = dims;
                auto const batch = static_cast<int32_t>(dims.batch);
                Prepared prepared{profile, dims, readInt32Binding(map, binding_names::kQueryLengths, batch, stream),
                    readInt32Binding(map, binding_names::kPastLengths, batch, stream),
                    mappedPagesPerRow(map, batch, stream), {}};
                if (draft && profile == kDecodeProfile)
                {
                    prepared.embeddedTokens = decodeEmbeddedTokens(map, static_cast<int32_t>(dims.seqLen), stream);
                }
                log.push_back(std::move(prepared));
                return true;
            });
        return engine;
    }

    //! The runtime over a paged KV working set: with the context cache on, a sequence's page-table row maps only the
    //! pages its committed state and headroom need, so the headroom shows at the engine's bindings. Without it the
    //! page table is a static map of the whole capacity.
    rt::LLMInferenceRuntime makePagedRuntime(rt::ModelArtifacts artifacts)
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting();
        options.contextCacheConfig = {/*enabled=*/true, /*maxRecords=*/4};
        return rt::LLMInferenceRuntime{std::move(artifacts), std::move(options), mStream};
    }

    //! The K pages each of the first `rows` rows of the bound KV page table maps, i.e. its entries that are not the
    //! unused sentinel.
    static std::vector<int32_t> mappedPagesPerRow(rt::TensorMap const& map, int32_t rows, cudaStream_t stream)
    {
        auto* const table = map.get(binding_names::kKVPageTable);
        if (table == nullptr || rows <= 0)
        {
            return {};
        }
        // Rows are [row][K or V][logical page]; the V half mirrors the K half.
        auto const pagesPerRow = static_cast<int32_t>(table->getShape()[2]);
        auto const entries = readInt32Binding(map, binding_names::kKVPageTable, rows * 2 * pagesPerRow, stream);
        std::vector<int32_t> mapped(static_cast<size_t>(rows), 0);
        for (int32_t row = 0; row < rows; ++row)
        {
            for (int32_t page = 0; page < pagesPerRow; ++page)
            {
                mapped[static_cast<size_t>(row)]
                    += entries[static_cast<size_t>(row * 2 * pagesPerRow + page)] != rt::kUNUSED_PAGE_ENTRY ? 1 : 0;
            }
        }
        return mapped;
    }

    //! Pages that hold `tokens` KV entries.
    static int32_t pagesFor(int32_t tokens)
    {
        return (tokens + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    }

    //! Every prepare in `rounds` maps, for each row, exactly the pages of the row's past length plus
    //! `headroomOf(batch, row)` tokens: no fewer, which the engine would overrun, and no more, which would be paid
    //! for by every request.
    template <typename HeadroomOf>
    static void expectPagesForHeadroom(std::vector<Prepared const*> const& rounds, HeadroomOf headroomOf)
    {
        ASSERT_FALSE(rounds.empty());
        for (size_t round = 0; round < rounds.size(); ++round)
        {
            Prepared const& prepared = *rounds[round];
            auto const batch = static_cast<size_t>(prepared.dims.batch);
            ASSERT_EQ(prepared.pastLengths.size(), batch) << "round " << round;
            ASSERT_EQ(prepared.mappedPages.size(), batch) << "round " << round;
            for (size_t row = 0; row < batch; ++row)
            {
                EXPECT_EQ(prepared.mappedPages[row],
                    pagesFor(prepared.pastLengths[row] + headroomOf(prepared.dims.batch, static_cast<int32_t>(row))))
                    << "round " << round << " row " << row << " past " << prepared.pastLengths[row];
            }
        }
    }

    //! The base verify forwards: every decode-profile prepare of the base.
    static std::vector<Prepared const*> baseVerifyRounds(std::vector<Prepared> const& log)
    {
        std::vector<Prepared const*> rounds;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile)
            {
                rounds.push_back(&prepared);
            }
        }
        return rounds;
    }

    //! The draft proposal forwards: decode-profile prepares at the proposal width. The draft's past length equals
    //! the base's committed length there, after the accept step caught up.
    static std::vector<Prepared const*> draftProposalRounds(std::vector<Prepared> const& log)
    {
        std::vector<Prepared const*> rounds;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile
                && prepared.dims.seqLen == prepared.dims.batch * kDraftingStep * kDraftingTopK)
            {
                rounds.push_back(&prepared);
            }
        }
        return rounds;
    }

    //! The token whose embedding each of the first `rows` rows of inputs_embeds holds, -1 for none.
    static std::vector<int32_t> decodeEmbeddedTokens(rt::TensorMap const& map, int32_t rows, cudaStream_t stream)
    {
        auto* const embeds = map.get(binding_names::kInputsEmbeds);
        if (embeds == nullptr || rows <= 0)
        {
            return {};
        }
        auto const& shape = embeds->getShape();
        int64_t const hidden = shape[shape.getNumDims() - 1];
        std::vector<half> host(static_cast<size_t>(rows) * static_cast<size_t>(hidden));
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), embeds->rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<int32_t> tokens(static_cast<size_t>(rows), -1);
        for (int32_t row = 0; row < rows; ++row)
        {
            float const first = __half2float(host[static_cast<size_t>(row) * hidden]);
            float const second = __half2float(host[static_cast<size_t>(row) * hidden + 1]);
            for (int32_t token = 0; token < static_cast<int32_t>(kVocabSize); ++token)
            {
                if (static_cast<float>(fakeEmbeddingValue(token, 0)) == first
                    && static_cast<float>(fakeEmbeddingValue(token, 1)) == second)
                {
                    tokens[static_cast<size_t>(row)] = token;
                    break;
                }
            }
        }
        return tokens;
    }

    //! An execute() action laid out by the last prepared dims: the last slot's rows decode to `lastSlotToken`,
    //! every other slot's rows to `otherToken`. The same slot keeps its token when it moves to row zero or runs alone.
    auto emitByLastSlot(int32_t otherToken, int32_t lastSlotToken)
    {
        return [this, otherToken, lastSlotToken](cudaStream_t stream) {
            int64_t const rows = mLogits->getShape().volume() / kVocabSize;
            int64_t const batch = std::max<int64_t>(mLastDims.batch, 1);
            int64_t const rowsPerSlot = std::max<int64_t>(rows / batch, 1);
            std::vector<int32_t> tokens(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row)
            {
                tokens[static_cast<size_t>(row)] = row / rowsPerSlot == batch - 1 ? lastSlotToken : otherToken;
            }
            writeLogits(*mLogits, kVocabSize, tokens, stream, mLogitsScale);
            return true;
        };
    }

    //! An execute() action whose every row ranks `first` highest and `second` next, so a top-2 draft proposes both.
    auto emitRanked(int32_t first, int32_t second)
    {
        return [this, first, second](cudaStream_t stream) {
            int64_t const rows = mLogits->getShape().volume() / kVocabSize;
            std::vector<float> host(static_cast<size_t>(rows * kVocabSize));
            for (int64_t row = 0; row < rows; ++row)
            {
                double sum = 0.0;
                for (int64_t token = 0; token < kVocabSize; ++token)
                {
                    float const logit = token == second ? kPeakLogit - 0.5F : fakeLogit(token, first);
                    host[static_cast<size_t>(row * kVocabSize + token)] = logit;
                    sum += std::exp(static_cast<double>(logit));
                }
                if (mLogitsScale == FakeOutputScale::kLogProbabilities)
                {
                    auto const logSum = static_cast<float>(std::log(sum));
                    for (int64_t token = 0; token < kVocabSize; ++token)
                    {
                        host[static_cast<size_t>(row * kVocabSize + token)] -= logSum;
                    }
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(
                mLogits->rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return true;
        };
    }

    //! A greedy request whose grammar is `regex`, asking for jump-forward.
    static rt::LLMGenerationRequest makeRegexRequest(
        std::string const& regex, int64_t maxGenerateLength, std::string const& prompt = "a")
    {
        rt::LLMGenerationRequest request = makeGreedyRequest(prompt, maxGenerateLength);
        request.requests.front().guidedDecoding
            = rt::GuidedDecodingParams{rt::GuideType::kRegex, regex, /*jumpForward=*/true};
        return request;
    }

    //! The dims of every base verify forward, i.e. the decode-profile prepares of the base.
    static std::vector<rt::InferenceDims> verifyDims(std::vector<Prepared> const& log)
    {
        std::vector<rt::InferenceDims> dims;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile)
            {
                dims.push_back(prepared.dims);
            }
        }
        return dims;
    }

    //! The query lengths every base verify forward was handed.
    static std::vector<std::vector<int32_t>> verifyQueryLengths(std::vector<Prepared> const& log)
    {
        std::vector<std::vector<int32_t>> lengths;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile)
            {
                lengths.push_back(prepared.queryLengths);
            }
        }
        return lengths;
    }

    //! The draft's accept-step prepares: decode-profile forwards whose width is an accept width rather than the
    //! proposal width.
    static std::vector<Prepared> draftAcceptSteps(std::vector<Prepared> const& log)
    {
        std::vector<Prepared> steps;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile
                && prepared.dims.seqLen != prepared.dims.batch * kDraftingStep * kDraftingTopK)
            {
                steps.push_back(prepared);
            }
        }
        return steps;
    }

    struct Run
    {
        rt::LLMGenerationResponse response;
        std::string warnings; //!< Warnings go to stderr.
        std::string debugLog; //!< Debug and info lines go to stdout.
    };

    Run run(rt::LLMInferenceRuntime& runtime, rt::LLMGenerationRequest const& request)
    {
        gLogger.setLevel(logger::Severity::kVERBOSE);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        Run result;
        bool const handled = runtime.handleRequest(request, result.response, mStream);
        result.debugLog = testing::internal::GetCapturedStdout();
        result.warnings = testing::internal::GetCapturedStderr();
        EXPECT_TRUE(handled) << result.warnings;
        return result;
    }

    static size_t countOf(std::string const& text, std::string const& needle)
    {
        size_t count = 0;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
        {
            ++count;
        }
        return count;
    }

    static bool ranWithJumpForward(Run const& run)
    {
        return run.debugLog.find(kRanWithJumpForward) != std::string::npos;
    }

    static constexpr char const* kRefusal{"Jump-forward is not supported on"};
    //! The statistics line a request's completion logs for a slot that ran with jump-forward.
    static constexpr char const* kRanWithJumpForward{"jump-forward: jumps="};

    rt::InferenceDims mLastDims{};

private:
    logger::Severity mSavedLevel{};
};

//! The base verifies exactly the verify width: no headroom.
class EagleNoHeadroomJumpForwardTest : public EagleJumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir(
            "eagleNoHeadroomJumpForwardTests", makeEagleBaseConfig(kVerifySize), makeEagleDraftConfig(kVerifySize));
    }
};

//! Both engines hold kVerifyHeadroom tokens beyond the verify width.
class EagleJumpForwardTest : public EagleJumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("eagleJumpForwardTests", makeEagleBaseConfig(kVerifySize + kVerifyHeadroom),
            makeEagleDraftConfig(kVerifySize + kVerifyHeadroom));
    }
};

//! Tree drafting (top-2 over two steps, four verify nodes) with kVerifyHeadroom tokens beyond the verify width.
class EagleTreeJumpForwardTest : public EagleJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = kTreeTopK;
        drafting.draftingStep = kTreeSteps;
        drafting.verifyTokens = kTreeVerifySize;
        return drafting;
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("eagleTreeJumpForwardTests", makeEagleBaseConfig(kTreeWideVerifySize),
            makeEagleDraftConfig(kTreeWideVerifySize));
    }
};

// Given an EAGLE deployment whose base verifies no more tokens than the verify width
// When two guided requests ask for jump-forward
// Then both generate the grammar's text, one warning names the missing verify headroom, and neither ran with
//      jump-forward
TEST_F(EagleNoHeadroomJumpForwardTest, RefusesJumpForwardOnceWhenTheBaseHasNoVerifyHeadroom)
{
    auto runtime = makeRuntime(makeArtifacts());

    Run const first = run(runtime, makeHelloRequest());
    Run const second = run(runtime, makeHelloRequest());

    ASSERT_EQ(first.response.outputIds.size(), 1U);
    ASSERT_EQ(second.response.outputIds.size(), 1U);
    EXPECT_EQ(first.response.outputIds[0], hello());
    EXPECT_EQ(second.response.outputIds[0], hello());
    EXPECT_EQ(countOf(first.warnings, kRefusal), 1U) << first.warnings;
    EXPECT_NE(first.warnings.find("verify headroom"), std::string::npos) << first.warnings;
    EXPECT_EQ(countOf(second.warnings, kRefusal), 0U) << "a reason warns once per runtime";
    EXPECT_FALSE(ranWithJumpForward(first));
    EXPECT_FALSE(ranWithJumpForward(second));
}

// Given an EAGLE deployment whose base verifies eight tokens beyond the verify width
// When a greedy guided request asks for jump-forward, and then, on a fresh runtime, a request that disables
//      speculative decoding and one that samples non-greedily, both of which the registry routes to the vanilla
//      decoder, ask for it
// Then all three generate the grammar's text without a refusal and their slots ran with jump-forward; the two
//      fallback requests run no draft forward, and the base catches the sampled `e` and the forced `llo` up in one
//      context-chunk step behind the prompt and `h`
TEST_F(EagleJumpForwardTest, GrantsJumpForwardWithVerifyHeadroomAndToRequestsThatFallBackToVanilla)
{
    Run granted;
    {
        auto artifacts = makeArtifacts();
        ASSERT_EQ(artifacts.deployment.specConfig->maxJumpTokens, kVerifyHeadroom);
        auto runtime = makeRuntime(std::move(artifacts));
        granted = run(runtime, makeHelloRequest());
    }
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, /*draft=*/false);
    auto draftEngine = makeLogProbDraftEngine();
    EXPECT_CALL(*draftEngine, execute(::testing::_)).Times(0);
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest disabled = makeHelloRequest();
    disabled.disableSpecDecode = true;
    rt::LLMGenerationRequest sampled = makeHelloRequest();
    sampled.temperature = 0.7F;
    sampled.topK = 4;

    Run const first = run(runtime, disabled);
    std::vector<Prepared> const firstLog = std::exchange(baseLog, {});
    Run const second = run(runtime, sampled);
    std::vector<Prepared> const secondLog = std::exchange(baseLog, {});

    ASSERT_EQ(granted.response.outputIds.size(), 1U);
    EXPECT_EQ(granted.response.outputIds[0], hello());
    EXPECT_EQ(granted.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(countOf(granted.warnings, kRefusal), 0U) << granted.warnings;
    EXPECT_TRUE(ranWithJumpForward(granted));
    auto const expectCaughtUpOnAContextChunk = [](Run const& fallback, std::vector<Prepared> const& log) {
        ASSERT_EQ(fallback.response.outputIds.size(), 1U);
        EXPECT_EQ(fallback.response.outputIds[0], hello());
        EXPECT_EQ(fallback.response.finishReasons[0], rt::FinishReason::kEndId);
        EXPECT_EQ(countOf(fallback.warnings, kRefusal), 0U) << fallback.warnings;
        EXPECT_NE(fallback.debugLog.find("jump-forward: jumps=1 forced=3 rolledBack=0 chunkSteps=1"), std::string::npos)
            << fallback.debugLog;
        std::vector<Prepared> chunks;
        std::copy_if(log.begin(), log.end(), std::back_inserter(chunks), [](Prepared const& prepared) {
            return prepared.profile == kPrefillProfile
                && prepared.dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kContextChunk);
        });
        ASSERT_EQ(chunks.size(), 1U) << "one catch-up step";
        EXPECT_EQ(chunks[0].queryLengths, (std::vector<int32_t>{4})) << "the sampled e and the forced l, l, o";
        EXPECT_EQ(chunks[0].pastLengths, (std::vector<int32_t>{2})) << "the prompt `a` and the sampled `h` have KV";
    };
    expectCaughtUpOnAContextChunk(first, firstLog);
    expectCaughtUpOnAContextChunk(second, secondLog);
}

// Given a jumpable EAGLE deployment over a paged KV cache and a batch of a jump-forward slot and a free slot, both
//      asking for more tokens than their KV caches hold
// When the batch runs
// Then only the jump-forward slot reserves the grafting round: it stops by length that many tokens earlier, its
//      base row maps the pages of its committed state plus the grafting round's verify width while the free slot's
//      maps the plain width, and its draft row likewise maps the wide accept width where the free slot's maps the
//      proposal width, and it ran with jump-forward
TEST_F(EagleJumpForwardTest, OnlyAJumpForwardSlotReservesTheGraftingRoundHeadroom)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
    auto runtime = makePagedRuntime(makeArtifacts(std::move(baseEngine), makeRecordingEngine(draftLog, true)));
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/1000);
    batch.requests[0].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z]+", /*jumpForward=*/true};

    Run const batched = run(runtime, batch);

    ASSERT_EQ(batched.response.outputIds.size(), 2U);
    ASSERT_EQ(batched.response.inputTokenCounts.size(), 2U);
    EXPECT_EQ(batched.response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(batched.response.finishReasons[1], rt::FinishReason::kLength);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[0].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[0] - kWideVerifySize);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[1].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[1] - kVerifySize);
    EXPECT_TRUE(ranWithJumpForward(batched));
    // The jump slot is row 0 while both run; the free slot moves to row 0 once the jump slot has stopped.
    expectPagesForHeadroom(baseVerifyRounds(baseLog),
        [](int64_t batch, int32_t row) { return batch == 2 && row == 0 ? kWideVerifySize : kVerifySize; });
    expectPagesForHeadroom(draftProposalRounds(draftLog), [](int64_t batch, int32_t row) {
        return batch == 2 && row == 0 ? kWideAcceptSize - 1 : kDraftingStep * kDraftingTopK;
    });
}

// Given a jumpable EAGLE deployment, a grammar forcing `hello`, and a draft that proposes none of its letters
// When the request runs
// Then the chain planned after prefill is grafted into the first verify round: the base verifies the wide width with
//      V + 4 valid nodes, that one round accepts the four forced letters and the EOS the grammar leaves, the forced
//      letters count as neither drafts nor bonus, and no second verify round runs
TEST_F(EagleJumpForwardTest, GraftsAChainTheDraftDidNotProposeIntoTheFirstVerifyRound)
{
    std::vector<Prepared> baseLog;
    auto runtime = makeRuntime(makeArtifacts(makeRecordingEngine(baseLog, false), makeLogProbDraftEngine()));

    Run const on = run(runtime, makeHelloRequest());

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], hello());
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    auto const dims = verifyDims(baseLog);
    ASSERT_EQ(dims.size(), 1U) << "the whole word was accepted in one verify round";
    EXPECT_EQ(dims[0].batch, 1);
    EXPECT_EQ(dims[0].seqLen, kWideVerifySize);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kVerifySize + 4}}))
        << "the four inserted letters are the only valid nodes past the draft tree";
    EXPECT_NE(on.debugLog.find("generated=5, acceptedDrafts=0, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=4 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a jumpable EAGLE deployment, a grammar forcing `hello`, a draft that proposes e, l, l itself, and a base
//      whose prediction at every node is `x`
// When the request runs
// Then the forced nodes the draft proposed are accepted over the base's prediction, only the `o` the draft lacked is
//      inserted, and the text is `hello`
TEST_F(EagleJumpForwardTest, AcceptsAForcedChainTheBaseDisagreesWith)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
    // The draft prefill's logits seed the chain with `e`; each proposal step then appends `l`.
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kE})).WillRepeatedly(emit({kL}));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    Run const on = run(runtime, makeHelloRequest());

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], hello());
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kVerifySize + 1}}))
        << "e, l, l merged into the draft chain; only o was inserted";
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=4"), std::string::npos) << on.debugLog;
}

// Given a jumpable EAGLE deployment and the grammar `hello[a-z]+`, so the chain is consumed in the first round and
//      free letters follow
// When eight tokens are generated
// Then the first verify round is wide and the draft re-reads its path at the wide accept width, while every later
//      round verifies V nodes and the draft re-reads S + 1 tokens
TEST_F(EagleJumpForwardTest, ReturnsToTheDraftWidthOnceTheChainIsConsumed)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), makeRecordingEngine(draftLog, true)));

    Run const on = run(runtime, makeRegexRequest("hello[a-z]+", /*maxGenerateLength=*/8));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kX, kX, kX}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kLength);
    std::vector<int64_t> verifyWidths;
    for (auto const& dims : verifyDims(baseLog))
    {
        verifyWidths.push_back(dims.seqLen);
    }
    EXPECT_EQ(verifyWidths, (std::vector<int64_t>{kWideVerifySize, kVerifySize, kVerifySize}));
    std::vector<int64_t> acceptWidths;
    for (auto const& step : draftAcceptSteps(draftLog))
    {
        acceptWidths.push_back(step.dims.seqLen);
    }
    EXPECT_EQ(acceptWidths, (std::vector<int64_t>{kWideAcceptSize, kNarrowAcceptSize}));
}

// Given a jumpable EAGLE deployment with chain drafting whose decoding graphs were captured, and the grammar
//      `hello[a-z]+`
// When eight tokens are generated
// Then the base verify graph of each batch size is captured with the tree layout at the grafting width and the chain
//      layout at the plain width, the grafting round verifies as a tree and the later rounds as a chain, the text is
//      `hello` and three free letters, and four tokens are counted as forced
TEST_F(EagleJumpForwardTest, VerifiesTheGraftingRoundOfAChainDeploymentAsATree)
{
    using ::testing::_;
    using Verify = std::tuple<int64_t, int64_t, int64_t>; // batch, rows per slot, layout extent
    std::vector<Prepared> baseLog;
    std::vector<rt::InferenceDims> baseCaptures;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault(::testing::Return(true));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->verifyLayout, rt::SpecVerifyLayout::kChain);
    auto runtime = makeRuntime(std::move(artifacts));
    ASSERT_TRUE(runtime.captureDecodingCUDAGraph(mStream));
    baseLog.clear();

    Run const on = run(runtime, makeRegexRequest("hello[a-z]+", /*maxGenerateLength=*/8));

    int64_t const chain = rt::specVerifyLayoutExtent(rt::SpecVerifyLayout::kChain);
    int64_t const tree = rt::specVerifyLayoutExtent(rt::SpecVerifyLayout::kTree);
    std::vector<Verify> captures;
    for (auto const& dims : baseCaptures)
    {
        if (dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify))
        {
            captures.emplace_back(dims.batch, dims.seqLen / dims.batch, dims.specVerifyLayoutLen);
        }
    }
    EXPECT_EQ(captures,
        (std::vector<Verify>{
            {1, kVerifySize, chain}, {1, kWideVerifySize, tree}, {2, kVerifySize, chain}, {2, kWideVerifySize, tree}}));
    std::vector<Verify> rounds;
    for (auto const& dims : verifyDims(baseLog))
    {
        rounds.emplace_back(dims.batch, dims.seqLen, dims.specVerifyLayoutLen);
    }
    EXPECT_EQ(
        rounds, (std::vector<Verify>{{1, kWideVerifySize, tree}, {1, kVerifySize, chain}, {1, kVerifySize, chain}}));
    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kX, kX, kX}));
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=4 rolledBack=0"), std::string::npos) << on.debugLog;
}

// Given a jumpable EAGLE deployment, a batch of a `[a-z][a-z]hello` jump slot and a free slot, a draft that proposes
//      `x` at every step, and a base that predicts `q` on the free slot's verify rows and `x` elsewhere
// When the batch runs, and the free slot runs alone with the same base
// Then the wide verify round carries V + 5 valid nodes for the jump slot, whose chain hangs under its first draft
//      node, and V for the free slot, and the free slot decodes the same tokens in both runs
TEST_F(EagleJumpForwardTest, AChainlessSlotBesideAGraftingSlotDecodesWhatItDecodesAlone)
{
    using ::testing::_;
    constexpr int64_t kGenerateLength{8};
    auto const makeRuntimeEmittingByLastSlot = [&](std::vector<Prepared>& log) {
        auto baseEngine = makeRecordingEngine(log, false);
        ON_CALL(*baseEngine, execute(_)).WillByDefault(emitByLastSlot(kX, kQ));
        auto draftEngine = makeLogProbDraftEngine();
        ON_CALL(*draftEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
        return makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    };
    std::vector<Prepared> aloneLog;
    rt::LLMGenerationResponse alone;
    {
        auto runtime = makeRuntimeEmittingByLastSlot(aloneLog);
        ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("bcd", kGenerateLength), alone, mStream));
    }
    ASSERT_EQ(alone.outputIds.size(), 1U);
    ASSERT_EQ(alone.outputIds[0], std::vector<int32_t>(static_cast<size_t>(kGenerateLength), kQ));

    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "bcd"}, kGenerateLength);
    batch.requests[0].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z][a-z]hello", /*jumpForward=*/true};
    std::vector<Prepared> batchLog;
    auto runtime = makeRuntimeEmittingByLastSlot(batchLog);
    Run const batched = run(runtime, batch);

    ASSERT_EQ(batched.response.outputIds.size(), 2U);
    EXPECT_EQ(batched.response.outputIds[0], (std::vector<int32_t>{kX, kX, kH, kE, kL, kL, kO, kEos}));
    EXPECT_EQ(batched.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(batched.response.outputIds[1], alone.outputIds[0]) << "the padded rows beside the graft disturbed it";
    auto const lengths = verifyQueryLengths(batchLog);
    ASSERT_FALSE(lengths.empty());
    EXPECT_EQ(lengths[0], (std::vector<int32_t>{kVerifySize + 5, kVerifySize}));
    EXPECT_EQ(countOf(batched.warnings, kRefusal), 0U) << batched.warnings;
}

// Given a jumpable EAGLE deployment, the grammar `he[xq]lo`, a prefill that yields `h`, a draft chain e, x, x and a
//      base that predicts `x` after `e`
// When the request runs with two log-probabilities per token
// Then one wide round accepts e x l o and the EOS: the root's chain `e` reuses the draft's node, the chain `lo`
//      planned behind `x` hangs under it, the three forced positions carry synthetic log-probability rows while `x`
//      keeps its real row, `x` is the one accepted draft, and two chains are counted
TEST_F(EagleJumpForwardTest, GraftsARootChainAndAMidTreeChainInOneRound)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit({kX}));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kE})).WillRepeatedly(emit({kX}));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest request = makeRegexRequest("he[xq]lo", /*maxGenerateLength=*/8);
    request.numLogprobs = 2;

    Run const on = run(runtime, request);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kX, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kVerifySize + 2}}))
        << "only l and o were inserted; e was the draft's own node";
    ASSERT_EQ(on.response.logprobs.size(), 1U);
    auto const& rows = on.response.logprobs[0];
    ASSERT_EQ(rows.size(), 6U);
    EXPECT_EQ(rows[0].size(), 2U) << "h: the prefill's real row";
    ASSERT_EQ(rows[1].size(), 1U) << "e: forced";
    EXPECT_EQ(rows[1][0].tokenId, kE);
    EXPECT_EQ(rows[1][0].logprob, 0.0F);
    EXPECT_EQ(rows[2].size(), 2U) << "x: the base's own prediction keeps its row";
    ASSERT_EQ(rows[3].size(), 1U) << "l: forced";
    EXPECT_EQ(rows[3][0].tokenId, kL);
    ASSERT_EQ(rows[4].size(), 1U) << "o: forced";
    EXPECT_EQ(rows[4][0].tokenId, kO);
    EXPECT_EQ(rows[5].size(), 2U) << "EOS: the bonus prediction";
    EXPECT_NE(on.debugLog.find("generated=5, acceptedDrafts=1, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=2 forced=3 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
    EXPECT_NE(on.debugLog.find("grafted 2 chains (1 at root, 2 inserted nodes), accepted 3 forced in 2 chains"),
        std::string::npos)
        << on.debugLog;
}

// Given a jumpable EAGLE deployment and the grammar `[a-z]+`, which never forces text
// When a jump-forward request runs, and the same request without the flag
// Then every verify round keeps the plain width, nothing is counted, and both runs decode the same tokens
TEST_F(EagleJumpForwardTest, AGrammarThatForcesNothingKeepsEveryRoundNarrow)
{
    using ::testing::_;
    auto const makeRuntimeWith = [&](std::vector<Prepared>& log) {
        auto baseEngine = makeRecordingEngine(log, false);
        ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kWideVerifySize)));
        return makeRuntime(makeArtifacts(std::move(baseEngine), makeLogProbDraftEngine()));
    };
    std::vector<Prepared> offLog;
    rt::LLMGenerationResponse off;
    {
        auto runtime = makeRuntimeWith(offLog);
        rt::LLMGenerationRequest request = makeRegexRequest("[a-z]+", /*maxGenerateLength=*/8);
        request.requests.front().guidedDecoding->jumpForward = false;
        ASSERT_TRUE(runtime.handleRequest(request, off, mStream));
    }
    std::vector<Prepared> onLog;
    auto runtime = makeRuntimeWith(onLog);

    Run const on = run(runtime, makeRegexRequest("[a-z]+", /*maxGenerateLength=*/8));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    ASSERT_EQ(off.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0].size(), 8U);
    EXPECT_EQ(on.response.outputIds[0], off.outputIds[0]);
    EXPECT_TRUE(ranWithJumpForward(on));
    auto const dims = verifyDims(onLog);
    EXPECT_EQ(dims.size(), verifyDims(offLog).size());
    for (auto const& round : dims)
    {
        EXPECT_EQ(round.seqLen, kVerifySize);
    }
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=0 forced=0 rolledBack=0 chunkSteps=0 skips{}"), std::string::npos)
        << on.debugLog;
}

// Given a jumpable EAGLE deployment with chain drafting, the grammar `[xq][yz]lo`, a draft chain y, z, z under the
//      root `x`, and a base that predicts `y` at the root
// When the request runs
// Then the chain `lo` planned behind `y` hangs under it, one wide round accepts y l o and the EOS with `y` as the one
//      accepted draft, and `z` is neither grafted nor on the path
TEST_F(EagleJumpForwardTest, GraftsAChainUnderADraftNodeOfAChainDraft)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit({kY}));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kY})).WillRepeatedly(emit({kZ}));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    Run const on = run(runtime, makeRegexRequest("[xq][yz]lo", /*maxGenerateLength=*/8));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kY, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kVerifySize + 2}}));
    EXPECT_NE(on.debugLog.find("generated=4, acceptedDrafts=1, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("grafted 1 chains (0 at root, 2 inserted nodes), accepted 2 forced in 1 chains"),
        std::string::npos)
        << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=2"), std::string::npos) << on.debugLog;
}

// Given a tree-drafting jumpable EAGLE deployment, the grammar `[a-z][xq]hi`, a draft that proposes `x` and `q`
//      under the root and `z` under `x`, and a base that predicts `x`
// When the request runs with two log-probabilities per token
// Then the first round is wide with the chain `hi` grafted under both `x` and `q` (V + 4 nodes), it accepts x h i
//      and the EOS with the two forced positions carrying synthetic rows, and only x's chain counts as a jump
TEST_F(EagleTreeJumpForwardTest, GraftsChainsUnderBothDraftBranchesAndAcceptsTheBasesBranch)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emitRanked(kX, kQ)).WillRepeatedly(emitRanked(kZ, kE));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest request = makeRegexRequest("[a-z][xq]hi", /*maxGenerateLength=*/8);
    request.numLogprobs = 2;

    Run const on = run(runtime, request);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kX, kH, kI, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    auto const dims = verifyDims(baseLog);
    ASSERT_EQ(dims.size(), 1U);
    EXPECT_EQ(dims[0].seqLen, kTreeWideVerifySize);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kTreeVerifySize + 4}}));
    ASSERT_EQ(on.response.logprobs.size(), 1U);
    auto const& rows = on.response.logprobs[0];
    ASSERT_EQ(rows.size(), 5U);
    EXPECT_EQ(rows[1].size(), 2U) << "x: the draft's node, taken on the base's prediction";
    ASSERT_EQ(rows[2].size(), 1U) << "h: forced";
    EXPECT_EQ(rows[2][0].tokenId, kH);
    ASSERT_EQ(rows[3].size(), 1U) << "i: forced";
    EXPECT_EQ(rows[3][0].tokenId, kI);
    EXPECT_NE(on.debugLog.find("grafted 2 chains (0 at root, 4 inserted nodes), accepted 2 forced in 1 chains"),
        std::string::npos)
        << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=2"), std::string::npos) << on.debugLog;
}

// Given a tree-drafting jumpable EAGLE deployment, the grammar `[a-z][xq]hello`, a draft that proposes `x` and `q`
//      under the root, and eight rows of headroom
// When the request runs
// Then x's chain takes five rows and q's is cut to the three that remain, so the wide round is full, and the base's
//      branch is accepted whole
TEST_F(EagleTreeJumpForwardTest, CutsTheSecondChainToTheRowsTheFirstLeaves)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emitRanked(kX, kQ)).WillRepeatedly(emitRanked(kZ, kE));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    Run const on = run(runtime, makeRegexRequest("[a-z][xq]hello", /*maxGenerateLength=*/10));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kX, kH, kE, kL, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kTreeWideVerifySize}}))
        << "five rows for x's chain, the remaining three for q's";
    EXPECT_NE(on.debugLog.find("grafted 2 chains (0 at root, 8 inserted nodes), accepted 5 forced in 1 chains"),
        std::string::npos)
        << on.debugLog;
}

//! The candidate-head deployment: top-2 over three steps, five of the 11 draft-tree nodes verified, eight candidates
//! per draft row, the same headroom. The candidate head bounds a grafting round by the EAGLE tree limits.
constexpr int32_t kCandidateTopK{2};
constexpr int32_t kCandidateSteps{3};
constexpr int32_t kCandidateVerifySize{5};
constexpr int32_t kDraftCandidateCount{8};
constexpr int32_t kCandidateProposalSize{kCandidateSteps * kCandidateTopK};
constexpr int32_t kCandidateWideVerifySize{kCandidateVerifySize + kVerifyHeadroom};
constexpr int32_t kCandidateNarrowAcceptSize{kCandidateSteps + 1};

class EagleCandidateJumpForwardTest : public EagleJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = kCandidateTopK;
        drafting.draftingStep = kCandidateSteps;
        drafting.verifyTokens = kCandidateVerifySize;
        return drafting;
    }

    std::filesystem::path stageModelDir() override
    {
        Json draft = makeEagleDraftConfig(kCandidateWideVerifySize);
        draft["eagle3_config"]["candidate_head"]
            = Json{{"type", "low_rank"}, {"rank", 16}, {"num_candidates", kDraftCandidateCount}};
        return writeSpecModelDir(
            "eagleCandidateJumpForwardTests", makeEagleBaseConfig(kCandidateWideVerifySize), draft);
    }

    //! makeCandidateDraftEngine(), recording the profile and dims of every prepare() into `log`.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingCandidateDraftEngine(
        std::vector<Prepared>& log, std::vector<std::vector<int32_t>> winnersPerForward = {})
    {
        auto engine = makeCandidateDraftEngine(kDraftCandidateCount, std::move(winnersPerForward));
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(
                [this, &log](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                    mCandidateHead.prepare(dims, map);
                    mLastDims = dims;
                    log.push_back(Prepared{profile, dims, {}, {}, {}, {}});
                    return true;
                });
        return engine;
    }

    //! The candidate draft's proposal forwards: decode-profile prepares at the padded proposal width.
    static std::vector<rt::InferenceDims> candidateProposals(std::vector<Prepared> const& log)
    {
        std::vector<rt::InferenceDims> proposals;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile
                && prepared.dims.seqLen == prepared.dims.batch * kCandidateProposalSize)
            {
                proposals.push_back(prepared.dims);
            }
        }
        return proposals;
    }
};

// Given a jumpable EAGLE deployment with a candidate-head draft, a grammar forcing `hello`, and a draft whose
//      candidates are none of its letters
// When the request runs
// Then the deployment derives a grafting round of V + 8 nodes and 9 accepted tokens, the chain `ello` is grafted under
//      the root in the first verify round (V + 4 valid nodes of the wide width), that round accepts the four forced
//      letters and the EOS, the forced letters count as neither drafts nor bonus, and every draft proposal forward
//      selects B x topK candidate rows
TEST_F(EagleCandidateJumpForwardTest, GraftsAChainIntoACandidateDraftTree)
{
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    auto artifacts = makeArtifacts(makeRecordingEngine(baseLog, false), makeRecordingCandidateDraftEngine(draftLog));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kCandidateWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kVerifyHeadroom + 1);
    auto runtime = makeRuntime(std::move(artifacts));

    Run const on = run(runtime, makeHelloRequest());

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], hello());
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    auto const dims = verifyDims(baseLog);
    ASSERT_EQ(dims.size(), 1U) << "the whole word was accepted in one verify round";
    EXPECT_EQ(dims[0].seqLen, kCandidateWideVerifySize);
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{kCandidateVerifySize + 4}}));
    auto const proposals = candidateProposals(draftLog);
    ASSERT_EQ(proposals.size(), static_cast<size_t>(kCandidateSteps - 1));
    for (auto const& proposal : proposals)
    {
        EXPECT_EQ(proposal.selectLen, proposal.batch * kCandidateTopK);
    }
    EXPECT_NE(on.debugLog.find("generated=5, acceptedDrafts=0, proposedDrafts=4"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=4 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a jumpable EAGLE deployment with a candidate-head draft whose engines accept graph capture, a draft that leads
//      every candidate row with `x`, a base that predicts `x`, and the grammar `hello[a-z]+`
// When the decoding CUDA graphs are captured and then twelve tokens are generated
// Then for each batch size the base verify graph is captured at the plain and the grafting widths and the draft accept
//      graph at every plain accepted length and at the grafting path; the first round is wide, the later rounds are
//      narrow, every draft proposal forward selects B x topK candidate rows, and the round after the graft accepts the
//      draft's chain x x x
TEST_F(EagleCandidateJumpForwardTest, ProposesFromCandidatesAgainAfterTheGraftingRound)
{
    using ::testing::_;
    using Capture = std::pair<int64_t, int64_t>; // batch, rows per slot
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    std::vector<rt::InferenceDims> baseCaptures;
    std::vector<rt::InferenceDims> draftCaptures;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kCandidateWideVerifySize)));
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    auto draftEngine = makeRecordingCandidateDraftEngine(
        draftLog, {std::vector<int32_t>(static_cast<size_t>(kMaxBatchSize * kCandidateTopK), kX)});
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault([this, &draftCaptures](cudaStream_t) {
        draftCaptures.push_back(mLastDims);
        return true;
    });
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    EXPECT_TRUE(runtime.captureDecodingCUDAGraph(mStream));
    baseLog.clear();
    draftLog.clear();
    Run const on = run(runtime, makeRegexRequest("hello[a-z]+", /*maxGenerateLength=*/12));

    std::vector<Capture> verifyCaptures;
    for (auto const& dims : baseCaptures)
    {
        if (dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify))
        {
            verifyCaptures.emplace_back(dims.batch, dims.seqLen / dims.batch);
        }
    }
    std::vector<Capture> acceptCaptures;
    for (auto const& dims : draftCaptures)
    {
        if (dims.seqLen != dims.batch * kCandidateProposalSize)
        {
            acceptCaptures.emplace_back(dims.batch, dims.seqLen / dims.batch);
        }
    }
    EXPECT_EQ(verifyCaptures,
        (std::vector<Capture>{{1, kCandidateVerifySize}, {1, kCandidateWideVerifySize}, {2, kCandidateVerifySize},
            {2, kCandidateWideVerifySize}}));
    EXPECT_EQ(acceptCaptures,
        (std::vector<Capture>{{1, 1}, {1, 2}, {1, 3}, {1, kCandidateNarrowAcceptSize}, {1, kVerifyHeadroom + 1}, {2, 1},
            {2, 2}, {2, 3}, {2, kCandidateNarrowAcceptSize}, {2, kVerifyHeadroom + 1}}));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kX, kX, kX, kX, kX, kX, kX}));
    std::vector<int64_t> verifyWidths;
    for (auto const& dims : verifyDims(baseLog))
    {
        verifyWidths.push_back(dims.seqLen);
    }
    EXPECT_EQ(
        verifyWidths, (std::vector<int64_t>{kCandidateWideVerifySize, kCandidateVerifySize, kCandidateVerifySize}))
        << "h, then e l l o x, then the drafted x x x and the bonus, then the last two";
    auto const proposals = candidateProposals(draftLog);
    ASSERT_EQ(proposals.size(), 3U * (kCandidateSteps - 1));
    for (auto const& proposal : proposals)
    {
        EXPECT_EQ(proposal.selectLen, proposal.batch * kCandidateTopK);
    }
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=4"), std::string::npos) << on.debugLog;
}

//! The byte-level BPE vocabulary of the guided-decoding tests, staged as a jumpable EAGLE deployment: it has the
//! merged pieces (`",`) a rollback needs.
class EagleJumpForwardBpeTest : public EagleJumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const dir = makeScratchDir("eagleJumpForwardBpeTests");
        std::ofstream(dir / "base_config.json") << makeEagleBaseConfig(kVerifySize + kVerifyHeadroom).dump(2);
        std::ofstream(dir / "draft_config.json") << makeEagleDraftConfig(kVerifySize + kVerifyHeadroom).dump(2);
        guidedFixture::writeBpeTokenizer(dir);
        std::ofstream(dir / "chat_template.jinja") << "{% for message in messages %}{{ message.content }}{% endfor %}";
        return dir;
    }
};

// Given a jumpable EAGLE deployment over the BPE vocabulary, the grammar `"a+","b"`, and a base that predicts `"`
//      at every node
// When the request runs
// Then the chain `a` is grafted under the opening quote and `"b"` under the comma, while the closing quote's forced
//      text `,"b"`, which would respell the quote into `",`, is refused and counted: the token list keeps the quote
//      and the comma as separate tokens, nothing is rolled back, and the text is `"a","b"`
TEST_F(EagleJumpForwardBpeTest, ARespelledRootIsLeftAloneAndCounted)
{
    using namespace guidedFixture;
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kQUOTE, 2 * kWideVerifySize)));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), makeLogProbDraftEngine()));

    Run const on = run(runtime, makeRegexRequest("\"a+\",\"b\"", /*maxGenerateLength=*/8));

    std::vector<int32_t> const expected{kQUOTE, kA, kQUOTE, kCOMMA, kQUOTE, kB, kQUOTE, kEOS};
    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], expected);
    EXPECT_EQ(on.response.outputTexts[0], "\"a\",\"b\"");
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyQueryLengths(baseLog),
        (std::vector<std::vector<int32_t>>{{kVerifySize + 1}, {kVerifySize}, {kVerifySize + 3}}))
        << "a grafted, then the refused quote's narrow round, then \"b\" grafted";
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=2 forced=4 rolledBack=0 chunkSteps=2 skips{respell_refused=1}"),
        std::string::npos)
        << on.debugLog;
}

} // namespace
