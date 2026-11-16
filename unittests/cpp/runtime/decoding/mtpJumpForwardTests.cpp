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

// Jump-forward requests on MTP deployments assembled around two substitute engines: a chain draft takes the forced
// chain it plans in place of the draft after the chain's node, a tree draft takes forced chains under its nodes in a
// wider verify round, and the deployment's geometry decides how wide a grafting round is.

#include "fakeEngineExecutor.h"
#include "specSessionSupport.h"
#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

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

//! Chain drafting (kDraftingStep steps, kVerifySize rows) on engines whose capacities give a grafting round of
//! J = W = min(max_spec_verify_tokens, 16, max_spec_draft_tokens) rows.
constexpr int32_t kChainMaxVerifyTokens{12};
constexpr int32_t kChainMaxDraftTokens{16};
constexpr int32_t kChainWideVerifySize{12};

//! Tree drafting: top-2 over two steps, four verify rows, and both engines kVerifyHeadroom rows wider, so a grafting
//! round verifies V + 8 rows and accepts up to 9 tokens.
constexpr int32_t kTreeTopK{2};
constexpr int32_t kTreeSteps{2};
constexpr int32_t kTreeVerifySize{4};
constexpr int32_t kVerifyHeadroom{8};
constexpr int32_t kTreeWideVerifySize{kTreeVerifySize + kVerifyHeadroom};
constexpr int32_t kTreeWideAcceptSize{kVerifyHeadroom + 1};
constexpr int32_t kTreeNarrowAcceptSize{kTreeSteps + 1};

Json makeMtpDraftConfig(int32_t maxDraftTokens)
{
    Json config = makeSpecDraftConfig("mtp", maxDraftTokens);
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

//! An MTP base over one attention and one GDN layer: MTP context reuse, which pages the KV working set, needs a hybrid
//! base. GDN chunk replay uses 128-wide K/V heads and one K head for two V heads.
Json makeHybridMtpBaseConfig(int32_t maxVerifyTokens)
{
    Json config = makeSpecBaseConfig("mtp", maxVerifyTokens);
    config["num_attention_layers"] = 1;
    config["num_linear_attn_layers"] = 1;
    config["recurrent_state_num_heads"] = 2;
    config["recurrent_state_head_dim"] = 128;
    config["recurrent_state_size"] = 128;
    config["conv_dim"] = 512;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp32";
    config["conv_state_dtype"] = "fp16";
    return config;
}

//! An MTP deployment whose engine capacities the suite chooses. Requests run with the logger at DEBUG and both log
//! streams captured: a refusal is a warning, and a slot that ran with jump-forward gets a statistics line when its
//! request completes.
class MtpJumpForwardTestBase : public SpecAssemblyTest
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

    //! MTP's draft model ends in `F.log_softmax`.
    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
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

    //! One prepare() of a recording engine: the dims it asked for, the ragged query and past lengths and token
    //! positions it was handed, and the KV pages each row's page-table entry maps.
    struct Prepared
    {
        int32_t profile{};
        rt::InferenceDims dims{};
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> pastLengths;
        std::vector<int32_t> positions;
        std::vector<int32_t> mappedPages;
    };

    //! makeEngine() or makeLogProbDraftEngine(), recording every prepare() into `log`.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingEngine(std::vector<Prepared>& log, bool draft)
    {
        auto engine = draft ? makeLogProbDraftEngine() : makeEngine();
        FakeOutputScale const scale = draft ? FakeOutputScale::kLogProbabilities : FakeOutputScale::kLogits;
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([this, &log, scale](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map,
                               cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = scale;
                mLastDims = dims;
                auto const batch = static_cast<int32_t>(dims.batch);
                log.push_back(
                    Prepared{profile, dims, readInt32Binding(map, binding_names::kQueryLengths, batch, stream),
                        readInt32Binding(map, binding_names::kPastLengths, batch, stream),
                        readInt32Binding(map, binding_names::kPositions, static_cast<int32_t>(dims.seqLen), stream),
                        mappedPagesPerRow(map, batch, stream)});
                return true;
            });
        return engine;
    }

    //! The runtime over a paged KV working set: with the context cache on, a sequence's page-table row maps only the
    //! pages its committed state and headroom need, so the headroom shows at the engine's bindings. A hybrid base
    //! needs snapshot pools for its recurrent state and partial pages.
    rt::LLMInferenceRuntime makePagedRuntime(rt::ModelArtifacts artifacts)
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting();
        options.contextCacheConfig
            = {/*enabled=*/true, /*maxRecords=*/4, /*maxSnapshots=*/16, /*encoderEmbeddingCacheBudgetBytes=*/0};
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

    //! A fake engine's prepares as Prepared rows: profile, dims, past lengths and mapped pages. The engine must have
    //! recorded the past lengths and the KV page table.
    static std::vector<Prepared> preparedOf(FakeEngineExecutor const& engine)
    {
        auto const& rounds = engine.preparedRounds();
        auto const& pastLengths = engine.preparedBinding(binding_names::kPastLengths);
        auto const& tables = engine.preparedBinding(binding_names::kKVPageTable);
        rt::Tensor const* const table = engine.boundTensor(binding_names::kKVPageTable);
        auto const pagesPerRow = table == nullptr ? size_t{0} : static_cast<size_t>(table->getShape()[2]);
        std::vector<Prepared> log;
        for (size_t round = 0; round < rounds.size(); ++round)
        {
            Prepared prepared{rounds[round].profileIndex, rounds[round].dims};
            auto const batch = static_cast<size_t>(rounds[round].dims.batch);
            auto const& past = pastLengths[round];
            prepared.pastLengths.assign(
                past.begin(), past.begin() + static_cast<std::ptrdiff_t>(std::min(batch, past.size())));
            for (size_t row = 0; row < batch && (row + 1) * 2 * pagesPerRow <= tables[round].size(); ++row)
            {
                auto const first = tables[round].begin() + static_cast<std::ptrdiff_t>(row * 2 * pagesPerRow);
                prepared.mappedPages.push_back(
                    static_cast<int32_t>(std::count_if(first, first + static_cast<std::ptrdiff_t>(pagesPerRow),
                        [](int32_t entry) { return entry != rt::kUNUSED_PAGE_ENTRY; })));
            }
            log.push_back(std::move(prepared));
        }
        return log;
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

    //! Whether a draft decode-profile prepare is a proposal forward: the draft walks one chain, so a proposal binds
    //! draftSteps rows per slot, and an accept step binds an accept width.
    bool isDraftProposal(Prepared const& prepared) const
    {
        return prepared.profile == kDecodeProfile
            && prepared.dims.seqLen == prepared.dims.batch * *drafting().draftingStep;
    }

    //! The draft proposal forwards. The draft's past length equals the base's committed length there, after the
    //! accept step caught up.
    std::vector<Prepared const*> draftProposalRounds(std::vector<Prepared> const& log) const
    {
        std::vector<Prepared const*> rounds;
        for (auto const& prepared : log)
        {
            if (isDraftProposal(prepared))
            {
                rounds.push_back(&prepared);
            }
        }
        return rounds;
    }

    //! The rows per slot of each draft accept step: the accepted path of the round before it.
    std::vector<int64_t> draftAcceptWidths(std::vector<Prepared> const& log) const
    {
        std::vector<int64_t> widths;
        for (auto const& prepared : log)
        {
            if (prepared.profile == kDecodeProfile && !isDraftProposal(prepared))
            {
                widths.push_back(prepared.dims.seqLen / prepared.dims.batch);
            }
        }
        return widths;
    }

    //! The rows per slot of every base verify forward.
    static std::vector<int64_t> verifyWidths(std::vector<Prepared> const& log)
    {
        std::vector<int64_t> widths;
        for (auto const* round : baseVerifyRounds(log))
        {
            widths.push_back(round->dims.seqLen / round->dims.batch);
        }
        return widths;
    }

    //! The query lengths every base verify forward was handed.
    static std::vector<std::vector<int32_t>> verifyQueryLengths(std::vector<Prepared> const& log)
    {
        std::vector<std::vector<int32_t>> lengths;
        for (auto const* round : baseVerifyRounds(log))
        {
            lengths.push_back(round->queryLengths);
        }
        return lengths;
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

    //! An execute() action whose every row ranks `first` highest and `second` next, so a top-2 tree takes both.
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

rt::SpecDecodeDraftingConfig makeTreeDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = kTreeTopK;
    drafting.draftingStep = kTreeSteps;
    drafting.verifyTokens = kTreeVerifySize;
    return drafting;
}

//! Chain drafting whose grafting round verifies and accepts kChainWideVerifySize rows.
class MtpChainJumpForwardTest : public MtpJumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpChainJumpForwardTests", makeSpecBaseConfig("mtp", kChainMaxVerifyTokens),
            makeMtpDraftConfig(kChainMaxDraftTokens));
    }
};

//! The chain deployment over a hybrid base.
class MtpHybridChainJumpForwardTest : public MtpJumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpHybridChainJumpForwardTests", makeHybridMtpBaseConfig(kChainMaxVerifyTokens),
            makeMtpDraftConfig(kChainMaxDraftTokens));
    }
};

//! Tree drafting with kVerifyHeadroom rows of verify headroom on both engines.
class MtpTreeJumpForwardTest : public MtpJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeTreeDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpTreeJumpForwardTests", makeSpecBaseConfig("mtp", kTreeWideVerifySize),
            makeMtpDraftConfig(kTreeWideVerifySize));
    }
};

// Given a jumpable MTP chain deployment, the grammar `[xq][yz]hello`, a draft chain y, z, z under the root `x`, and a
//      base that predicts `y` at the root and `z` at every other row
// When the request runs with two log-probabilities per token
// Then the chain `hello` planned behind `y` replaces the draft after it: the one verify round holds the root, `y` and
//      the five forced tokens, it accepts y h e l l o and the EOS without either cut `z`, `y` is the round's one
//      accepted draft of the three proposed, and the five forced tokens count as one jump; each forced token's
//      log-probability row is its own token at 0, while x, y and the EOS keep their real top-2 rows
TEST_F(MtpChainJumpForwardTest, CutsTheDraftAfterTheGraftPoint)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeLogProbDraftEngine();
    std::vector<int32_t> baseRows(kChainWideVerifySize, kZ);
    baseRows[0] = kY;
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(baseRows));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kY})).WillRepeatedly(emit({kZ}));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest request = makeRegexRequest("[xq][yz]hello", /*maxGenerateLength=*/10);
    request.numLogprobs = 2;

    Run const on = run(runtime, request);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    std::vector<int32_t> const expected{kX, kY, kH, kE, kL, kL, kO, kEos};
    EXPECT_EQ(on.response.outputIds[0], expected);
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(verifyWidths(baseLog), (std::vector<int64_t>{kChainWideVerifySize}));
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{2 + 5}}))
        << "depth 1 + 1 + five forced tokens; the draft's z, z are cut";
    EXPECT_NE(on.debugLog.find("generated=7, acceptedDrafts=1, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=5 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
    ASSERT_EQ(on.response.logprobs.size(), 1U);
    auto const& rows = on.response.logprobs[0];
    ASSERT_EQ(rows.size(), expected.size());
    EXPECT_EQ(rows[0].size(), 2U) << "x: the prefill's real row";
    EXPECT_EQ(rows[1].size(), 2U) << "y: the base's own prediction keeps its row";
    for (size_t i = 2; i + 1 < rows.size(); ++i)
    {
        ASSERT_EQ(rows[i].size(), 1U) << "forced token " << i;
        EXPECT_EQ(rows[i][0].tokenId, expected[i]);
        EXPECT_EQ(rows[i][0].logprob, 0.0F);
    }
    EXPECT_EQ(rows.back().size(), 2U) << "EOS: the bonus prediction";
}

// Given a jumpable MTP tree deployment, the grammar `[a-z][xq]hi[a-z]+`, a draft that ranks `x` over `q` at depth one
//      and `z` over `e` at depth two, and a base that predicts `x` at every row
// When eight tokens are generated
// Then the first verify round is wide: `hi` is grafted under both `x` and `q`, the base verifies the grafting width
//      with V + 4 valid rows, each at its depth past the committed length, and accepts x h i x; the draft accept step
//      after it binds the wide accept width, and every later round verifies V rows and re-reads min(S + 1, V)
TEST_F(MtpTreeJumpForwardTest, GraftsMidTreeChainsIntoAWideRoundThatTheDraftRereads)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeRecordingEngine(draftLog, true);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emitRanked(kX, kQ)).WillRepeatedly(emitRanked(kZ, kE));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kTreeWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kTreeWideAcceptSize);
    auto runtime = makeRuntime(std::move(artifacts));

    Run const on = run(runtime, makeRegexRequest("[a-z][xq]hi[a-z]+", /*maxGenerateLength=*/8));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kX, kH, kI, kX, kX, kX, kX}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(verifyWidths(baseLog),
        (std::vector<int64_t>{kTreeWideVerifySize, kTreeVerifySize, kTreeVerifySize, kTreeVerifySize}));
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_FALSE(rounds.empty());
    EXPECT_EQ(rounds[0]->queryLengths, (std::vector<int32_t>{kTreeVerifySize + 4}));
    // Draft tree root, x, q, x>z; then h, i under x and h, i under q; the prompt `a` is the committed state.
    EXPECT_EQ(rounds[0]->positions, (std::vector<int32_t>{1, 2, 2, 3, 3, 4, 3, 4, -1, -1, -1, -1}));
    EXPECT_EQ(draftAcceptWidths(draftLog),
        (std::vector<int64_t>{kTreeWideAcceptSize, kTreeNarrowAcceptSize, kTreeNarrowAcceptSize}));
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=2"), std::string::npos) << on.debugLog;
}

// Given a jumpable MTP tree deployment whose engines accept graph capture
// When the decoding CUDA graphs are captured
// Then for each batch size the base verify graph is captured at the plain and the grafting widths, and the draft
//      graphs at the proposal width, at every plain accepted length and at the grafting round's accepted path
TEST_F(MtpTreeJumpForwardTest, CapturesTheGraftingRoundGraphs)
{
    using ::testing::_;
    using Capture = std::pair<int64_t, int64_t>; // batch, rows per slot
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    std::vector<rt::InferenceDims> baseCaptures;
    std::vector<rt::InferenceDims> draftCaptures;
    auto baseEngine = makeRecordingEngine(baseLog, false);
    auto draftEngine = makeRecordingEngine(draftLog, true);
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault([this, &draftCaptures](cudaStream_t) {
        draftCaptures.push_back(mLastDims);
        return true;
    });
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    EXPECT_TRUE(runtime.captureDecodingCUDAGraph(mStream));

    std::vector<Capture> verify;
    for (auto const& dims : baseCaptures)
    {
        if (dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify))
        {
            verify.emplace_back(dims.batch, dims.seqLen / dims.batch);
        }
    }
    std::vector<Capture> draft;
    for (auto const& dims : draftCaptures)
    {
        draft.emplace_back(dims.batch, dims.seqLen / dims.batch);
    }
    EXPECT_EQ(verify,
        (std::vector<Capture>{
            {1, kTreeVerifySize}, {1, kTreeWideVerifySize}, {2, kTreeVerifySize}, {2, kTreeWideVerifySize}}));
    // Per batch size: the proposal, then the accept step at 1..min(S + 1, V) and at the grafting path.
    EXPECT_EQ(draft,
        (std::vector<Capture>{{1, kTreeSteps}, {1, 1}, {1, 2}, {1, kTreeNarrowAcceptSize}, {1, kTreeWideAcceptSize},
            {2, kTreeSteps}, {2, 1}, {2, 2}, {2, kTreeNarrowAcceptSize}, {2, kTreeWideAcceptSize}}));
}

// Given a jumpable MTP chain deployment, a batch of a `[a-z][a-z]hello` slot and a `[a-z]+` slot that both ask for
//      jump-forward, a draft that proposes `x` at every step, and a base that predicts `q` on the second slot's rows
//      and `x` elsewhere
// When the batch runs, and the second slot runs alone with the same engines
// Then the wide verify round carries the first slot's root, `x` and the five forced tokens and the second slot's four
//      drafted rows, the first slot decodes `xxhello`, and the second decodes the same tokens in both runs
TEST_F(MtpChainJumpForwardTest, ASlotWithoutAChainBesideAGraftingSlotDecodesWhatItDecodesAlone)
{
    using ::testing::_;
    constexpr int64_t kGenerateLength{8};
    auto const makeRuntimeEmittingByLastSlot = [&](std::vector<Prepared>& log) {
        auto baseEngine = makeRecordingEngine(log, false);
        ON_CALL(*baseEngine, execute(_)).WillByDefault(emitByLastSlot(kX, kQ));
        auto draftEngine = makeLogProbDraftEngine();
        ON_CALL(*draftEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kChainWideVerifySize)));
        return makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    };
    std::vector<Prepared> aloneLog;
    rt::LLMGenerationResponse alone;
    {
        auto runtime = makeRuntimeEmittingByLastSlot(aloneLog);
        ASSERT_TRUE(runtime.handleRequest(makeRegexRequest("[a-z]+", kGenerateLength, "bcd"), alone, mStream));
    }
    ASSERT_EQ(alone.outputIds.size(), 1U);
    ASSERT_EQ(alone.outputIds[0], std::vector<int32_t>(static_cast<size_t>(kGenerateLength), kQ));

    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "bcd"}, kGenerateLength);
    batch.requests[0].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z][a-z]hello", /*jumpForward=*/true};
    batch.requests[1].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z]+", /*jumpForward=*/true};
    std::vector<Prepared> batchLog;
    auto runtime = makeRuntimeEmittingByLastSlot(batchLog);
    Run const batched = run(runtime, batch);

    ASSERT_EQ(batched.response.outputIds.size(), 2U);
    EXPECT_EQ(batched.response.outputIds[0], (std::vector<int32_t>{kX, kX, kH, kE, kL, kL, kO, kEos}));
    EXPECT_EQ(batched.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(batched.response.outputIds[1], alone.outputIds[0]) << "the padded rows beside the graft disturbed it";
    auto const lengths = verifyQueryLengths(batchLog);
    ASSERT_FALSE(lengths.empty());
    EXPECT_EQ(lengths[0], (std::vector<int32_t>{2 + 5, kVerifySize}));
    EXPECT_EQ(verifyWidths(batchLog).front(), kChainWideVerifySize);
    EXPECT_EQ(countOf(batched.warnings, kRefusal), 0U) << batched.warnings;
}

// Given a jumpable MTP chain deployment over a hybrid base and a paged KV cache, and a batch of a jump-forward slot and
//      a free slot, both asking for more tokens than their KV caches hold
// When the batch runs
// Then only the jump-forward slot reserves the grafting round: it stops by length that many tokens earlier, its base
//      row maps the pages of its committed state plus the grafting width while the free slot's maps the plain width,
//      its draft row maps the grafting round's accepted path but one where the free slot's maps the draft steps, and
//      it ran with jump-forward
TEST_F(MtpHybridChainJumpForwardTest, OnlyAJumpForwardSlotReservesTheGraftingRoundHeadroom)
{
    auto baseEngine = makeFakeEngine(makeHybridMtpBaseConfig(kChainMaxVerifyTokens));
    baseEngine->forceOutputTokens(agreedLogitsRows(kX, 2 * kChainWideVerifySize));
    for (auto const* name :
        {binding_names::kTreeParentIds, binding_names::kTreeDepths, binding_names::kSpecVerifyLayout})
    {
        baseEngine->exposeInt32Binding(name);
    }
    auto draftEngine = makeFakeEngine(makeMtpDraftConfig(kChainMaxDraftTokens));
    draftEngine->actAsDraftEngine();
    for (auto* engine : {baseEngine.get(), draftEngine.get()})
    {
        engine->recordBinding(binding_names::kPastLengths);
        engine->recordBinding(binding_names::kKVPageTable);
    }
    FakeEngineExecutor const& base = *baseEngine;
    FakeEngineExecutor const& draft = *draftEngine;
    auto runtime = makePagedRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/1000);
    batch.requests[0].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z]+", /*jumpForward=*/true};
    // The only commit policy hybrid speculative context reuse takes.
    batch.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;

    Run const batched = run(runtime, batch);
    std::vector<Prepared> const baseLog = preparedOf(base);
    std::vector<Prepared> const draftLog = preparedOf(draft);

    ASSERT_EQ(batched.response.outputIds.size(), 2U);
    ASSERT_EQ(batched.response.inputTokenCounts.size(), 2U);
    EXPECT_EQ(batched.response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(batched.response.finishReasons[1], rt::FinishReason::kLength);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[0].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[0] - kChainWideVerifySize);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[1].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[1] - kVerifySize);
    EXPECT_TRUE(ranWithJumpForward(batched));
    // The jump slot is row 0 while both run; the free slot moves to row 0 once the jump slot has stopped.
    expectPagesForHeadroom(baseVerifyRounds(baseLog),
        [](int64_t batch, int32_t row) { return batch == 2 && row == 0 ? kChainWideVerifySize : kVerifySize; });
    expectPagesForHeadroom(draftProposalRounds(draftLog),
        [](int64_t batch, int32_t row) { return batch == 2 && row == 0 ? kChainWideVerifySize - 1 : kDraftingStep; });
}

// Given a jumpable MTP chain deployment and a runtime session opened on it
// When a guided request with the jump-forward flag is admitted into the session and decodes to its end
// Then it runs without jump-forward: one refusal warning names in-flight batching, its completion logs no jump-forward
//      statistics, and it spells the grammar's text as the blocking path does without the flag
TEST_F(MtpChainJumpForwardTest, RefusesJumpForwardUnderInFlightBatching)
{
    using ::testing::_;
    std::vector<Prepared> referenceLog;
    std::vector<Prepared> sessionLog;
    auto const makeRuntimePredictingY = [&](std::vector<Prepared>& log) {
        auto baseEngine = makeRecordingEngine(log, false);
        auto draftEngine = makeLogProbDraftEngine();
        std::vector<int32_t> baseRows(kChainWideVerifySize, kZ);
        baseRows[0] = kY;
        ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(baseRows));
        ON_CALL(*draftEngine, execute(_)).WillByDefault(emit({kZ}));
        return makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    };
    rt::LLMGenerationRequest const request = makeRegexRequest("[xq][yz]hello", /*maxGenerateLength=*/10);

    rt::LLMGenerationRequest plain = request;
    plain.requests.front().guidedDecoding->jumpForward = false;
    auto referenceRuntime = makeRuntimePredictingY(referenceLog);
    Run const reference = run(referenceRuntime, plain);
    ASSERT_EQ(reference.response.outputIds.size(), 1U);
    ASSERT_EQ(reference.response.outputIds[0], (std::vector<int32_t>{kX, kY, kH, kE, kL, kL, kO, kEos}));

    auto runtime = makeRuntimePredictingY(sessionLog);
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SpecSessionDriver driver(*session);
    gLogger.setLevel(logger::Severity::kVERBOSE);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_FATAL_FAILURE(driver.admitAndPrefill(request, /*requestId=*/7));
    ASSERT_NO_FATAL_FAILURE(driver.drain());
    std::string const debugLog = testing::internal::GetCapturedStdout();
    std::string const warnings = testing::internal::GetCapturedStderr();

    ASSERT_EQ(driver.outcomes().count(7), 1U);
    rt::LLMGenerationResponse const& outcome = driver.outcomes().at(7);
    ASSERT_EQ(outcome.outputIds.size(), 1U);
    EXPECT_EQ(outcome.outputIds[0], reference.response.outputIds[0]);
    EXPECT_EQ(countOf(warnings, kRefusal), 1U) << warnings;
    EXPECT_NE(warnings.find("in-flight batching"), std::string::npos) << warnings;
    EXPECT_EQ(countOf(debugLog, kRanWithJumpForward), 0U) << debugLog;
}

// Given a jumpable MTP chain deployment, which keeps a non-greedy request and decodes it greedily, the grammar
//      `[xq][yz]hello`, a draft chain y, z, z under the root `x`, and a base that predicts `y` at the root and `z` at
//      every other row
// When a guided request that samples at temperature 0.7 with top-4 asks for jump-forward, and the same request greedy
// Then the sampled request warns once that its sampling is overridden to greedy, is not refused, and decodes what the
//      greedy request decodes with the same one jump of five forced tokens
TEST_F(MtpChainJumpForwardTest, GrantsJumpForwardToANonGreedyRequestItDecodesGreedily)
{
    using ::testing::_;
    auto const makeRuntimeForcingY = [&]() {
        auto baseEngine = makeEngine();
        auto draftEngine = makeLogProbDraftEngine();
        std::vector<int32_t> baseRows(kChainWideVerifySize, kZ);
        baseRows[0] = kY;
        ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(baseRows));
        EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kY})).WillRepeatedly(emit({kZ}));
        return makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    };
    rt::LLMGenerationRequest const greedy = makeRegexRequest("[xq][yz]hello", /*maxGenerateLength=*/10);
    rt::LLMGenerationRequest sampled = greedy;
    sampled.temperature = 0.7F;
    sampled.topK = 4;
    auto greedyRuntime = makeRuntimeForcingY();
    Run const reference = run(greedyRuntime, greedy);
    auto sampledRuntime = makeRuntimeForcingY();

    Run const on = run(sampledRuntime, sampled);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    ASSERT_EQ(reference.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], reference.response.outputIds[0]);
    EXPECT_EQ(countOf(on.warnings, "overriding sampling params to greedy"), 1U) << on.warnings;
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    constexpr char const* kOneJump{"jump-forward: jumps=1 forced=5 rolledBack=0 chunkSteps=1"};
    EXPECT_NE(reference.debugLog.find(kOneJump), std::string::npos) << reference.debugLog;
    EXPECT_NE(on.debugLog.find(kOneJump), std::string::npos) << on.debugLog;
}

} // namespace
