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

// Jump-forward requests on the DSpark drafter assembled around two substitute engines: a tree takes forced chains under
// its nodes in a wider verify round, a chain takes the forced chain it plans in place of the draft after the chain's
// node, and the request's sampling decides whether a request gets it.

#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"

#include <algorithm>
#include <cuda_fp16.h>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kA{0};
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

//! Rank of the low-rank Markov head. Its weights are zero, so each draft step proposes its backbone row's argmax.
constexpr int32_t kMarkovRank{8};

//! Tree drafting: a block of seven proposals and a top-2 tree of four verify nodes, on bases kVerifyHeadroom rows
//! wider, so a grafting round accepts up to the block and the bonus and verifies V + 7 rows.
constexpr int32_t kTreeBlockSize{7};
constexpr int32_t kTreeTopK{2};
constexpr int32_t kTreeVerifySize{4};
constexpr int32_t kVerifyHeadroom{8};
constexpr int32_t kTreeWideVerifySize{kTreeVerifySize + kTreeBlockSize};
constexpr int32_t kTreeWideAcceptSize{kTreeBlockSize + 1};
//! The plain tree round's draft block and accepted path, the larger of which a plain slot's draft KV keeps free.
constexpr int32_t kTreePlainDraftHeadroom{kTreeBlockSize};

//! Chain drafting: three proposals verified behind the root.
constexpr int32_t kChainBlockSize{4};
constexpr int32_t kChainVerifySize{kChainBlockSize};

//! A step's confidence is the sigmoid of the draft's first hidden channel (see writeHeadsSidecar): these make it
//! about 1 and about 0.
constexpr float kConfident{8.0F};
constexpr float kUnsure{-8.0F};

//! Both engines of a DSpark deployment declare the same dspark_config; the draft reads the hidden states of the two
//! target layers. Mask id 3 lies inside this deployment's 128-token vocabulary.
Json dsparkSection(int32_t blockSize, bool confidenceHead)
{
    return Json{{"target_layer_ids", Json::array({0, 1})}, {"block_size", blockSize}, {"mask_token_id", 3},
        {"markov_head_type", "vanilla"}, {"markov_rank", kMarkovRank}, {"enable_confidence_head", confidenceHead},
        {"confidence_head_with_markov", false}};
}

Json makeDSparkBase(int32_t maxVerifyTokens, int32_t blockSize, bool confidenceHead = false)
{
    Json config = makeSpecBaseConfig("dspark", maxVerifyTokens);
    config["dspark_config"] = dsparkSection(blockSize, confidenceHead);
    return config;
}

Json makeDSparkDraft(int32_t blockSize, bool confidenceHead = false)
{
    Json config = makeSpecDraftConfig("dspark", blockSize);
    config["dspark_config"] = dsparkSection(blockSize, confidenceHead);
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

//! Write the heads sidecar DSpark loads unconditionally: zero Markov matrices, and with a confidence head the weights
//! that make a step's confidence the sigmoid of the draft's first hidden channel.
void writeHeadsSidecar(
    std::filesystem::path const& dir, int32_t draftHiddenSize, bool confidenceHead, cudaStream_t stream)
{
    std::vector<rt::Tensor> heads;
    for (char const* name : {"markov_w1", "markov_w2"})
    {
        rt::Tensor matrix({kVocabSize, kMarkovRank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), stream));
        heads.push_back(std::move(matrix));
    }
    if (confidenceHead)
    {
        std::vector<half> weight(static_cast<size_t>(draftHiddenSize), __float2half(0.0F));
        weight[0] = __float2half(1.0F);
        rt::Tensor weightTensor(
            {draftHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "confidence_weight");
        CUDA_CHECK(cudaMemcpyAsync(
            weightTensor.rawPointer(), weight.data(), weight.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        rt::Tensor bias({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "confidence_bias");
        CUDA_CHECK(cudaMemsetAsync(bias.rawPointer(), 0, sizeof(half), stream));
        heads.push_back(std::move(weightTensor));
        heads.push_back(std::move(bias));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!rt::safetensors::saveSafetensors(dir / binding_names::kDSparkHeadsFileName, heads, stream))
    {
        throw std::runtime_error("failed to write the DSpark heads sidecar");
    }
}

std::filesystem::path stageDSparkModelDir(
    std::string const& suite, Json const& baseConfig, Json const& draftConfig, cudaStream_t stream)
{
    auto const dir = writeSpecModelDir(suite, baseConfig, draftConfig);
    writeHeadsSidecar(dir, draftConfig["hidden_size"].get<int32_t>(),
        draftConfig["dspark_config"]["enable_confidence_head"].get<bool>(), stream);
    return dir;
}

rt::SpecDecodeDraftingConfig makeTreeDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = kTreeTopK;
    drafting.proposalTokens = kTreeBlockSize;
    drafting.verifyTokens = kTreeVerifySize;
    return drafting;
}

rt::SpecDecodeDraftingConfig makeChainDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.proposalTokens = kChainBlockSize - 1;
    drafting.verifyTokens = kChainVerifySize;
    return drafting;
}

//! A DSpark deployment whose engine capacities and drafting shape the suite chooses. Requests run with the logger at
//! DEBUG and both log streams captured: a refusal is a warning, and a slot that ran with jump-forward gets a statistics
//! line when its request completes.
class DSparkJumpForwardTestBase : public SpecAssemblyTest
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

    //! The mid-tree scenario of the tree suites: after any first letter the grammar takes `x` or `q`, then forces
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

    //! One prepare() of a recording engine: the dims it asked for, and the ragged query lengths and token positions of
    //! its tokens.
    struct Prepared
    {
        int32_t profile{};
        rt::InferenceDims dims{};
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> positions;
    };

    //! makeEngine() recording every prepare() into `log`. A DSpark draft returns its backbone logits without
    //! `F.log_softmax`, so both engines write raw logits. The draft's hidden-state output is kept for
    //! emitWithConfidence.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingEngine(std::vector<Prepared>& log)
    {
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault([this, &log](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map,
                               cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = FakeOutputScale::kLogits;
                mDraftHidden = map.get(binding_names::kDSparkHiddenStates);
                mLastDims = dims;
                auto const batch = static_cast<int32_t>(dims.batch);
                auto const tokens = static_cast<int32_t>(dims.seqLen);
                log.push_back(
                    Prepared{profile, dims, readInt32Binding(map, binding_names::kQueryLengths, batch, stream),
                        readInt32Binding(map, binding_names::kPositions, tokens, stream)});
                return true;
            });
        return engine;
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

    //! A draft execute() action: row r proposes `tokens[r]` and carries `channel0[r]` in its first hidden channel,
    //! which decides the step's confidence. The rows are [slot, step] flattened.
    auto emitWithConfidence(std::vector<int32_t> tokens, std::vector<float> channel0)
    {
        return [this, tokens = std::move(tokens), channel0 = std::move(channel0)](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, tokens, stream, mLogitsScale);
            auto const& shape = mDraftHidden->getShape();
            int64_t const width = shape[shape.getNumDims() - 1];
            int64_t const rows = shape.volume() / width;
            std::vector<half> host(static_cast<size_t>(rows * width), __float2half(0.0F));
            for (int64_t row = 0; row < rows && row < static_cast<int64_t>(channel0.size()); ++row)
            {
                host[static_cast<size_t>(row * width)] = __float2half(channel0[static_cast<size_t>(row)]);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                mDraftHidden->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
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
    rt::Tensor* mDraftHidden{nullptr};

private:
    logger::Severity mSavedLevel{};
};

//! A DSpark tree on a base with kVerifyHeadroom rows of verify headroom.
class DSparkTreeJumpForwardTest : public DSparkJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeTreeDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return stageDSparkModelDir("dsparkTreeJumpForwardTests",
            makeDSparkBase(kTreeVerifySize + kVerifyHeadroom, kTreeBlockSize), makeDSparkDraft(kTreeBlockSize),
            mStream);
    }
};

//! A DSpark chain without a scheduler.
class DSparkChainJumpForwardTest : public DSparkJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeChainDrafting();
    }

    std::filesystem::path stageModelDir() override
    {
        return stageDSparkModelDir("dsparkChainJumpForwardTests", makeDSparkBase(kChainVerifySize, kChainBlockSize),
            makeDSparkDraft(kChainBlockSize), mStream);
    }
};

//! A DSpark chain whose confidence scheduler keeps the leading proposals whose survival stays at or above one half.
class DSparkScheduledChainJumpForwardTest : public DSparkJumpForwardTestBase
{
protected:
    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig drafting = makeChainDrafting();
        drafting.dsparkSchedulerMode = rt::DSparkSchedulerMode::kThreshold;
        drafting.dsparkConfidenceThreshold = 0.5F;
        return drafting;
    }

    std::filesystem::path stageModelDir() override
    {
        return stageDSparkModelDir("dsparkScheduledChainJumpForwardTests",
            makeDSparkBase(kChainVerifySize, kChainBlockSize, /*confidenceHead=*/true),
            makeDSparkDraft(kChainBlockSize, /*confidenceHead=*/true), mStream);
    }
};

// Given a jumpable DSpark tree deployment whose decoding graphs were captured, the grammar `[a-z][xq]oil[a-z]+`, a
// draft
//      that ranks `x` over `q` at every depth, and a base that predicts `x` at every row
// When nine tokens are generated
// Then the first verify round is wide: `oil` is grafted under both `x` and `q`, the base verifies the grafting width
//      with V + 6 valid rows, each at its depth past the committed length, on a shape whose graph was captured, and
//      accepts x o i l x, more than a plain round can, of which `x` is the one accepted draft of the V - 1 proposed;
//      the draft forward after it takes those five tokens as its delta on a captured shape, and the later round
//      verifies V rows
TEST_F(DSparkTreeJumpForwardTest, GraftsMidTreeChainsIntoAWideRoundThatTheDraftTakesAsItsDelta)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    std::vector<rt::InferenceDims> baseCaptures;
    std::vector<rt::InferenceDims> draftCaptures;
    auto baseEngine = makeRecordingEngine(baseLog);
    auto draftEngine = makeRecordingEngine(draftLog);
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(agreedLogitsRows(kX, 2 * kTreeWideVerifySize)));
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emitRanked(kX, kQ));
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault([this, &draftCaptures](cudaStream_t) {
        draftCaptures.push_back(mLastDims);
        return true;
    });
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kTreeWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kTreeWideAcceptSize);
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
    for (auto const& prepared : draftLog)
    {
        EXPECT_TRUE(prepared.profile != kDecodeProfile || captured(draftCaptures, prepared.dims))
            << "no graph for the draft forward of delta " << prepared.dims.selectLen;
    }
    EXPECT_NE(on.debugLog.find("generated=5, acceptedDrafts=1, proposedDrafts=3"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=3 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a jumpable DSpark chain deployment whose confidence scheduler keeps one proposal `y` under the root `x` for the
//      first slot of a batch, whose grammar is `[xq][yz]lo`, and two proposals `z z` for the second, whose grammar is
//      `[xq]l[a-z]+`, and a base that predicts `y` at the first slot's root, `a`, whose id the cut rows hold, at the
//      second slot's second row, and `z` at every other row
// When the batch runs
// Then the first verify round verifies the full proposal width, wider than either schedule, and each slot verifies its
//      grafted chain: the first slot's chain `lo` planned behind `y` replaces its draft in place and runs past its
//      schedule, so it verifies the root, `y` and `lo` and accepts y l o and the EOS; the second slot's forced `l` cuts
//      its draft after the root, so it verifies the root and `l` and accepts l and the bonus `a`, not the cut schedule
//      row; each slot counts one jump
TEST_F(DSparkScheduledChainJumpForwardTest, VerifiesEachSlotsGraftedChainInsteadOfItsSchedule)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<Prepared> draftLog;
    auto baseEngine = makeRecordingEngine(baseLog);
    auto draftEngine = makeRecordingEngine(draftLog);
    std::vector<int32_t> baseRows(2 * kChainVerifySize, kZ);
    baseRows[0] = kY;
    baseRows[kChainVerifySize + 1] = kA;
    ON_CALL(*baseEngine, execute(_)).WillByDefault(emit(baseRows));
    // Rows [slot, step]: the first slot proposes y, z, z and is sure of only the first step, the second proposes z, z,
    // z and is sure of the first two.
    ON_CALL(*draftEngine, execute(_))
        .WillByDefault(emitWithConfidence(
            {kY, kZ, kZ, kZ, kZ, kZ}, {kConfident, kUnsure, kUnsure, kConfident, kConfident, kUnsure}));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/6);
    batch.requests[0].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[xq][yz]lo", /*jumpForward=*/true};
    batch.requests[1].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[xq]l[a-z]+", /*jumpForward=*/true};

    Run const on = run(runtime, batch);

    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kY, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    ASSERT_GE(on.response.outputIds[1].size(), 3U);
    EXPECT_EQ(std::vector<int32_t>(on.response.outputIds[1].begin(), on.response.outputIds[1].begin() + 3),
        (std::vector<int32_t>{kX, kL, kA}));
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_FALSE(rounds.empty());
    EXPECT_EQ(rounds[0]->dims.seqLen, 2 * kChainVerifySize);
    EXPECT_EQ(rounds[0]->queryLengths, (std::vector<int32_t>{2 + 2, 1 + 1}))
        << "the root, y and the forced l, o; the second slot's root and forced l";
    EXPECT_NE(on.debugLog.find("SpecRound 0 batchIdx 1: generated=2,"), std::string::npos)
        << "the second slot accepts l and the bonus, not the cut row\n"
        << on.debugLog;
    EXPECT_NE(on.debugLog.find("Request 0 jump-forward: jumps=1 forced=2"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("Request 1 jump-forward: jumps=1 forced=1"), std::string::npos) << on.debugLog;
}

// Given a jumpable DSpark chain deployment, which keeps a non-greedy request and verifies its drafts by probability
// When a guided request that samples at temperature 0.7 asks for jump-forward
// Then it is refused with one warning naming non-greedy sampling, its rounds still verify the draft chain at the
//      plain width, the text is `hello`, and the slot did not run with jump-forward
TEST_F(DSparkChainJumpForwardTest, RefusesJumpForwardToANonGreedyRequestItKeeps)
{
    std::vector<Prepared> baseLog;
    auto runtime = makeRuntime(makeArtifacts(makeRecordingEngine(baseLog), makeEngine()));
    rt::LLMGenerationRequest sampled = makeHelloRequest();
    sampled.temperature = 0.7F;
    sampled.topK = 4;

    Run const on = run(runtime, sampled);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], hello());
    EXPECT_EQ(countOf(on.warnings, kRefusal), 1U) << on.warnings;
    EXPECT_NE(on.warnings.find("non-greedy sampling under speculative decoding"), std::string::npos) << on.warnings;
    auto const widths = verifyWidths(baseLog);
    EXPECT_FALSE(widths.empty()) << "the DSpark decoder served the request";
    EXPECT_EQ(widths, std::vector<int64_t>(widths.size(), kChainVerifySize));
    EXPECT_FALSE(ranWithJumpForward(on));
}

// Given a jumpable DSpark chain deployment, and a batch of a free slot that ends at its first token and a slot whose
//      grammar `[xq][yz]lo` forces `lo` after its draft `y` under the root `x`, whose first round is drafted before the
//      free slot leaves the batch
// When the batch runs
// Then the remaining slot's first verify round runs alone and grafts `lo` in place of the draft after `y`: it accepts
//      y l o and the EOS, and counts one jump
TEST_F(DSparkChainJumpForwardTest, GraftsTheFirstRoundAfterAPrefillFinishedSlotLeavesTheBatch)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    auto baseEngine = makeRecordingEngine(baseLog);
    auto draftEngine = makeEngine();
    // Prefill rows are per slot; the second slot's row is masked to `x`. Verify rows: `y` at the root, `z` elsewhere.
    EXPECT_CALL(*baseEngine, execute(_)).WillOnce(emit({kEos, kZ})).WillRepeatedly(emit({kY, kZ, kZ, kZ}));
    // Rows [slot, step]: the second slot proposes y, z, z.
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emit({kZ, kZ, kZ, kY, kZ, kZ}));
    // A managed KV request drafts the first round right after prefill and evicts the slots prefill finished.
    rt::LLMInferenceRuntime runtime{makeArtifacts(std::move(baseEngine), std::move(draftEngine)), mModelDir.string(),
        "", {}, drafting(), mStream, rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16}};
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/6);
    batch.requests[1].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[xq][yz]lo", /*jumpForward=*/true};

    Run const on = run(runtime, batch);

    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kEos}));
    EXPECT_EQ(on.response.outputIds[1], (std::vector<int32_t>{kX, kY, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[1], rt::FinishReason::kEndId);
    auto const rounds = baseVerifyRounds(baseLog);
    ASSERT_EQ(rounds.size(), 1U);
    EXPECT_EQ(rounds[0]->dims.batch, 1);
    EXPECT_NE(on.debugLog.find("Request 1 jump-forward: jumps=1 forced=2"), std::string::npos) << on.debugLog;
}

// Given a jumpable DSpark chain deployment, and a batch of a jump-forward slot that ends at its first token and a slot
//      without jump-forward whose grammar `[xq][yz]lo` follows its draft `y` under the root `x`, whose first round is
//      drafted before the jump-forward slot leaves the batch, and a base that predicts `y` at the root and `z` at every
//      other row
// When the batch runs
// Then the remaining slot's first round is still masked by its grammar: it accepts y and the forced l instead of the
//      base's `z`, its text is `xylo` and the EOS, and neither slot fails
TEST_F(DSparkChainJumpForwardTest, MasksTheFirstRoundAfterAJumpForwardSlotLeavesTheBatchAtPrefill)
{
    using ::testing::_;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    // Prefill rows are per slot; the second slot's row is masked to `x`.
    EXPECT_CALL(*baseEngine, execute(_)).WillOnce(emit({kEos, kZ})).WillRepeatedly(emit({kY, kZ, kZ, kZ}));
    // Rows [slot, step]: the second slot proposes y, z, z.
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emit({kZ, kZ, kZ, kY, kZ, kZ}));
    // A managed KV request drafts the first round right after prefill and evicts the slots prefill finished.
    rt::LLMInferenceRuntime runtime{makeArtifacts(std::move(baseEngine), std::move(draftEngine)), mModelDir.string(),
        "", {}, drafting(), mStream, rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16}};
    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "a"}, /*maxGenerateLength=*/6);
    batch.requests[0].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[a-z]*", /*jumpForward=*/true};
    batch.requests[1].guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[xq][yz]lo", /*jumpForward=*/false};

    Run const on = run(runtime, batch);

    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kEos}));
    EXPECT_EQ(on.response.outputIds[1], (std::vector<int32_t>{kX, kY, kL, kO, kEos}));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(on.response.finishReasons[1], rt::FinishReason::kEndId);
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
}

// Given a jumpable DSpark tree deployment, and a batch of a jump-forward slot and a free slot that both ask for more
//      tokens than the KV caches hold
// When the batch runs
// Then only the jump-forward slot reserves the grafting round: it stops by length with the grafting width of base KV
//      left free, while the free slot stops with one draft block of draft KV left free, the larger of its plain
//      reservations, and the jump-forward slot ran with jump-forward
TEST_F(DSparkTreeJumpForwardTest, OnlyAJumpForwardSlotReservesTheGraftingRoundHeadroom)
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
        kMtpKvCacheCapacity - batched.response.inputTokenCounts[1] - kTreePlainDraftHeadroom);
    EXPECT_TRUE(ranWithJumpForward(batched));
}

} // namespace
