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

// Jump-forward requests on DFlash V1 deployments assembled around two substitute engines: a DDTree takes forced chains
// under its nodes in a wider verify round, a linear block takes the forced chain it plans in place of the draft after
// the chain's node, and the deployment's geometry decides whether a request gets it.

#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"

#include <algorithm>
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

//! One draft forward emits a block of kBlockSize rows: the root and kBlockSize - 1 proposals. A linear deployment
//! verifies the block itself.
constexpr int32_t kBlockSize{8};
constexpr int32_t kLinearVerifySize{kBlockSize};

//! DDTree drafting: a top-2 tree of four verify nodes on bases kVerifyHeadroom rows wider, so a grafting round accepts
//! up to one block and verifies V + kBlockSize - 1 rows.
constexpr int32_t kTreeTopK{2};
constexpr int32_t kTreeVerifySize{4};
constexpr int32_t kVerifyHeadroom{8};
constexpr int32_t kTreeWideVerifySize{kTreeVerifySize + kBlockSize - 1};

//! Both engines of a DFlash deployment declare the same dflash_config; the draft reads the hidden states of the two
//! target layers. Mask id 3 lies inside this deployment's 128-token vocabulary.
void addDFlashConfig(Json& config)
{
    config["dflash_config"]
        = Json{{"target_layer_ids", Json::array({0, 1})}, {"block_size", kBlockSize}, {"mask_token_id", 3}};
}

Json makeDFlashBaseConfig(int32_t maxVerifyTokens)
{
    Json config = makeSpecBaseConfig("dflash", maxVerifyTokens);
    addDFlashConfig(config);
    return config;
}

//! The DFlash base over one attention and one GDN layer. GDN chunk replay takes FP32 state with 128-wide K and V heads,
//! and conv_dim 512 gives one K head for two V heads.
Json makeHybridDFlashBaseConfig(int32_t maxVerifyTokens)
{
    Json config = makeDFlashBaseConfig(maxVerifyTokens);
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

Json makeDFlashDraftConfig()
{
    Json config = makeSpecDraftConfig("dflash", kBlockSize);
    addDFlashConfig(config);
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

rt::SpecDecodeDraftingConfig makeLinearDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.proposalTokens = kBlockSize - 1;
    drafting.verifyTokens = kLinearVerifySize;
    return drafting;
}

rt::SpecDecodeDraftingConfig makeTreeDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = kTreeTopK;
    drafting.proposalTokens = kBlockSize - 1;
    drafting.verifyTokens = kTreeVerifySize;
    return drafting;
}

//! A cached-block deployment whose engine capacities and drafting shape the suite chooses. Requests run with the logger
//! at DEBUG and both log streams captured: a refusal is a warning, and a slot that ran with jump-forward gets a
//! statistics line when its request completes.
class CachedBlockJumpForwardTestBase : public SpecAssemblyTest
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

    //! A greedy request whose grammar is `regex`, asking for jump-forward.
    static rt::LLMGenerationRequest makeRegexRequest(std::string const& regex, int64_t maxGenerateLength)
    {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", maxGenerateLength);
        request.requests.front().guidedDecoding
            = rt::GuidedDecodingParams{rt::GuideType::kRegex, regex, /*jumpForward=*/true};
        return request;
    }

    //! A greedy request whose grammar forces the literal `hello`, asking for jump-forward.
    static rt::LLMGenerationRequest makeHelloRequest()
    {
        return makeRegexRequest("hello", /*maxGenerateLength=*/8);
    }

    static std::vector<int32_t> hello()
    {
        return {kH, kE, kL, kL, kO, kEos};
    }

    //! The mid-tree scenario of the DDTree suites: after any first letter the grammar takes `x` or `q`, then forces
    //! `oil`. With a draft that ranks `x` over `q` at every depth the tree is root, x, q, x>x, so `oil` goes under both
    //! depth-1 nodes, and a base that predicts `x` takes the path through x and on after `oil`.
    static rt::LLMGenerationRequest makeMidTreeRequest()
    {
        return makeRegexRequest("[a-z][xq]oil[a-z]+", /*maxGenerateLength=*/9);
    }

    static std::vector<int32_t> midTreeText()
    {
        return {kX, kX, kO, kI, kL, kX, kX, kX, kX};
    }

    //! One prepare() of a recording engine: the dims it asked for, the ragged query lengths, the positions of its
    //! tokens, and the rows per slot of the first intermediate conv state bound to it (0 when none is).
    struct Prepared
    {
        rt::InferenceDims dims{};
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> positions;
        int64_t intermediateRows{};
    };

    //! makeEngine() recording every prepare() into `log`. A cached-block draft returns its head's output without
    //! `F.log_softmax`, so both engines write raw logits.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingEngine(std::vector<Prepared>& log)
    {
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([this, &log](int32_t /*profile*/, rt::InferenceDims const& dims, rt::TensorMap const& map,
                               cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = FakeOutputScale::kLogits;
                mLastDims = dims;
                auto const batch = static_cast<int32_t>(dims.batch);
                auto const tokens = static_cast<int32_t>(dims.seqLen);
                rt::Tensor const* const stash = map.get(binding_names::formatIntermediateConvStateName(0));
                log.push_back(Prepared{dims, readInt32Binding(map, binding_names::kQueryLengths, batch, stream),
                    readInt32Binding(map, binding_names::kPositions, tokens, stream),
                    stash != nullptr && stash->getShape().getNumDims() == 4 ? stash->getShape()[1] : 0});
                return true;
            });
        return engine;
    }

    //! Answers the binding queries a base that verifies recurrent state checks at assembly: INT32 tree metadata and
    //! the layout marker.
    static void exposeRecurrentVerifyBindings(MockEngineExecutor& engine)
    {
        using ::testing::Return;
        using ::testing::StrEq;
        for (auto const* name :
            {binding_names::kTreeParentIds, binding_names::kTreeDepths, binding_names::kSpecVerifyLayout})
        {
            ON_CALL(engine, hasIOTensor(StrEq(name))).WillByDefault(Return(true));
            ON_CALL(engine, getBindingDataType(StrEq(name))).WillByDefault(Return(trt_edgellm::DataType::kINT32));
        }
    }

    //! The base verify forwards.
    static std::vector<Prepared const*> baseVerifyRounds(std::vector<Prepared> const& log)
    {
        std::vector<Prepared const*> rounds;
        for (auto const& prepared : log)
        {
            if (prepared.dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify))
            {
                rounds.push_back(&prepared);
            }
        }
        return rounds;
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

    //! The target-hidden rows per slot each draft forward took: the prompt first, then the tokens each round accepted.
    static std::vector<int64_t> draftDeltas(std::vector<Prepared> const& log)
    {
        std::vector<int64_t> deltas;
        for (auto const& prepared : log)
        {
            deltas.push_back(prepared.dims.selectLen / prepared.dims.batch);
        }
        return deltas;
    }

    //! Whether a graph was captured for the shape `dims` asks for.
    static bool captured(std::vector<rt::InferenceDims> const& captures, rt::InferenceDims const& dims)
    {
        return std::any_of(captures.begin(), captures.end(), [&dims](rt::InferenceDims const& graph) {
            return graph.batch == dims.batch && graph.seqLen == dims.seqLen && graph.selectLen == dims.selectLen
                && graph.executionPhaseLen == dims.executionPhaseLen
                && graph.specVerifyLayoutLen == dims.specVerifyLayoutLen;
        });
    }

    //! An execute() action whose every row ranks `first` highest and `second` next, so a top-2 tree takes both.
    auto emitRanked(int32_t first, int32_t second)
    {
        return [this, first, second](cudaStream_t stream) {
            int64_t const rows = mLogits->getShape().volume() / kVocabSize;
            std::vector<float> host(static_cast<size_t>(rows * kVocabSize));
            for (int64_t row = 0; row < rows; ++row)
            {
                for (int64_t token = 0; token < kVocabSize; ++token)
                {
                    host[static_cast<size_t>(row * kVocabSize + token)]
                        = token == second ? kPeakLogit - 0.5F : fakeLogit(token, first);
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

//! Linear DFlash: the verify window is the drafted block.
class DFlashLinearJumpForwardTest : public CachedBlockJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeLinearDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir(
            "dflashLinearJumpForwardTests", makeDFlashBaseConfig(kLinearVerifySize), makeDFlashDraftConfig());
    }
};

//! DDTree DFlash on a base with kVerifyHeadroom rows of verify headroom.
class DFlashTreeJumpForwardTest : public CachedBlockJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeTreeDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("dflashTreeJumpForwardTests", makeDFlashBaseConfig(kTreeVerifySize + kVerifyHeadroom),
            makeDFlashDraftConfig());
    }
};

//! DDTree DFlash whose base verifies exactly the verify width: no headroom.
class DFlashTreeNoHeadroomJumpForwardTest : public CachedBlockJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeTreeDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir(
            "dflashTreeNoHeadroomJumpForwardTests", makeDFlashBaseConfig(kTreeVerifySize), makeDFlashDraftConfig());
    }
};

//! The DDTree DFlash deployment over a hybrid base.
class DFlashHybridTreeJumpForwardTest : public CachedBlockJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeTreeDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("dflashHybridTreeJumpForwardTests",
            makeHybridDFlashBaseConfig(kTreeVerifySize + kVerifyHeadroom), makeDFlashDraftConfig());
    }
};

// Given a jumpable DFlash DDTree deployment whose decoding graphs were captured, the grammar `[a-z][xq]oil[a-z]+`, a
//      draft that ranks `x` over `q` at every depth, and a base that predicts `x` at every row
// When nine tokens are generated
// Then the first verify round is wide: `oil` is grafted under both `x` and `q`, the base verifies the grafting width
//      with V + 6 valid rows, each at its depth past the committed length, on a shape whose graph was captured, and
//      accepts x o i l x, more than a plain round can, of which `x` is the one accepted draft of the V - 1 proposed;
//      the draft forward after it takes those five tokens as its delta, and the later round verifies V rows
TEST_F(DFlashTreeJumpForwardTest, GraftsMidTreeChainsIntoAWideRoundThatTheDraftTakesAsItsDelta)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    std::vector<rt::InferenceDims> baseCaptures;
    auto baseEngine = makeRecordingEngine(baseLog);
    auto draftEngine = makeRecordingEngine(draftLog);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emitRanked(kX, kQ));
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault(::testing::Return(true));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kTreeWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kBlockSize);
    auto runtime = makeRuntime(std::move(artifacts));
    ASSERT_TRUE(runtime.captureDecodingCUDAGraph(mStream));
    baseLog.clear();
    draftLog.clear();

    Run const on = run(runtime, makeMidTreeRequest());

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], midTreeText());
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(verifyWidths(baseLog), (std::vector<int64_t>{kTreeWideVerifySize, kTreeVerifySize}));
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_FALSE(rounds.empty());
    EXPECT_EQ(rounds[0]->queryLengths, (std::vector<int32_t>{kTreeVerifySize + 6}));
    // Draft tree root, x, q, x>x; then o, i, l under x and o, i, l under q; the prompt `a` is the committed state.
    EXPECT_EQ(rounds[0]->positions, (std::vector<int32_t>{1, 2, 2, 3, 3, 4, 5, 3, 4, 5, -1}));
    for (auto const* round : rounds)
    {
        EXPECT_TRUE(captured(baseCaptures, round->dims))
            << "no graph for the verify forward of " << round->dims.seqLen << " rows";
    }
    EXPECT_EQ(draftDeltas(draftLog), (std::vector<int64_t>{1, 5})) << "the prompt, then the grafting round's x o i l x";
    EXPECT_NE(on.debugLog.find("generated=5, acceptedDrafts=1, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=3 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a jumpable DFlash linear deployment, the grammar `[xq][yz]hello`, a draft block y, z, z, z, z, z, z under the
//      root `x`, and a base that predicts `y` at the root and `z` at every other row
// When the request runs
// Then the chain `hello` planned behind `y` replaces the draft after it in place: the one verify round keeps the block
//      width and the chain layout, holds the root, `y` and the five forced tokens as its valid rows, and accepts
//      y h e l l o and the EOS without any cut `z`; `y` is the round's one accepted draft of the seven proposed, and
//      the five forced tokens count as one jump
TEST_F(DFlashLinearJumpForwardTest, ReplacesTheDraftAfterTheGraftPointWithinTheBlock)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog);
    auto draftEngine = makeEngine();
    std::vector<int32_t> baseRows(kLinearVerifySize, kZ);
    baseRows[0] = kY;
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(baseRows));
    // Row 0 of a block predicts the root itself; the proposals start at row 1.
    std::vector<int32_t> draftRows(kBlockSize, kZ);
    draftRows[1] = kY;
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emit(draftRows));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kLinearVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->maxJumpTokens, kLinearVerifySize - 1);
    auto runtime = makeRuntime(std::move(artifacts));

    Run const on = run(runtime, makeRegexRequest("[xq][yz]hello", /*maxGenerateLength=*/10));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kY, kH, kE, kL, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    EXPECT_EQ(verifyWidths(baseLog), (std::vector<int64_t>{kLinearVerifySize}));
    EXPECT_EQ(verifyQueryLengths(baseLog), (std::vector<std::vector<int32_t>>{{2 + 5}}))
        << "depth 1 + 1 + five forced tokens; the draft's z's are cut";
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_EQ(rounds.size(), 1U);
    EXPECT_EQ(rounds[0]->dims.specVerifyLayoutLen, rt::specVerifyLayoutExtent(rt::SpecVerifyLayout::kChain));
    EXPECT_NE(on.debugLog.find("generated=7, acceptedDrafts=1, proposedDrafts=7"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=5 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a DFlash DDTree deployment whose base verifies no more rows than the verify width
// When two guided requests ask for jump-forward
// Then both generate the grammar's text, one warning names the missing verify headroom, and neither ran with
//      jump-forward
TEST_F(DFlashTreeNoHeadroomJumpForwardTest, RefusesJumpForwardOnceWhenTheBaseHasNoVerifyHeadroom)
{
    auto runtime = makeRuntime(makeArtifacts());

    Run const first = run(runtime, makeHelloRequest());
    Run const second = run(runtime, makeHelloRequest());

    ASSERT_EQ(first.response.outputIds.size(), 1U);
    ASSERT_EQ(second.response.outputIds.size(), 1U);
    EXPECT_EQ(first.response.outputIds[0], hello());
    EXPECT_EQ(second.response.outputIds[0], hello());
    EXPECT_EQ(countOf(first.warnings, kRefusal), 1U) << first.warnings;
    EXPECT_NE(first.warnings.find("speculative engines without verify headroom"), std::string::npos) << first.warnings;
    EXPECT_EQ(countOf(second.warnings, kRefusal), 0U) << "a reason warns once per runtime";
    EXPECT_FALSE(ranWithJumpForward(first));
    EXPECT_FALSE(ranWithJumpForward(second));
}

// Given a jumpable DFlash DDTree deployment over a hybrid base, the grammar `[a-z][xq]oil[a-z]+`, a draft that ranks
//      `x` over `q` at every depth, and a base that predicts `x` at every row
// When nine tokens are generated
// Then the wide verify round binds the GDN layer's intermediate state for every one of its grafting-width rows, the
//      plain round for its V rows, and the request decodes x x o i l x x x x
TEST_F(DFlashHybridTreeJumpForwardTest, SizesTheIntermediateStateForTheGraftingWidth)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog);
    exposeRecurrentVerifyBindings(*baseEngine);
    auto draftEngine = makeEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emitRanked(kX, kQ));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    Run const on = run(runtime, makeMidTreeRequest());

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], midTreeText());
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_EQ(rounds.size(), 2U);
    EXPECT_EQ(rounds[0]->dims.seqLen, kTreeWideVerifySize);
    EXPECT_EQ(rounds[0]->intermediateRows, kTreeWideVerifySize);
    EXPECT_EQ(rounds[1]->intermediateRows, kTreeVerifySize);
}

// Given a jumpable DFlash DDTree deployment, and a batch of a jump-forward slot and a free slot that both ask for more
//      tokens than the KV caches hold
// When the batch runs
// Then only the jump-forward slot reserves the grafting round: it stops by length with the grafting width of base KV
//      left free, while the free slot stops with one draft block of draft KV left free, the larger of its plain
//      reservations, and the jump-forward slot ran with jump-forward
TEST_F(DFlashTreeJumpForwardTest, OnlyAJumpForwardSlotReservesTheGraftingRoundHeadroom)
{
    using ::testing::_;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emitRanked(kX, kQ));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/1000);
    batch.requests[0].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z]+", /*jumpForward=*/true};

    Run const batched = run(runtime, batch);

    ASSERT_EQ(batched.response.outputIds.size(), 2U);
    ASSERT_EQ(batched.response.inputTokenCounts.size(), 2U);
    EXPECT_EQ(batched.response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(batched.response.finishReasons[1], rt::FinishReason::kLength);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[0].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[0] - kTreeWideVerifySize);
    EXPECT_EQ(static_cast<int64_t>(batched.response.outputIds[1].size()),
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[1] - kBlockSize);
    EXPECT_TRUE(ranWithJumpForward(batched));
}

} // namespace
