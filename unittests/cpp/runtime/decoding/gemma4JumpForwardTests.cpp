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

// Jump-forward requests on Gemma4 MTP deployments assembled around two substitute engines: a chain draft takes the
// forced chain it plans in place of the draft after the chain's node, a tree draft takes forced chains under its nodes
// in a wider verify round, and the assistant's next round starts from the grafting round's last accepted row.

#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/logger.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kEos{1};
constexpr int32_t kE{'e' - 'b' + 3};
constexpr int32_t kH{'h' - 'b' + 3};
constexpr int32_t kI{'i' - 'b' + 3};
constexpr int32_t kL{'l' - 'b' + 3};
constexpr int32_t kO{'o' - 'b' + 3};
constexpr int32_t kQ{'q' - 'b' + 3};
constexpr int32_t kX{'x' - 'b' + 3};
constexpr int32_t kZ{'z' - 'b' + 3};

//! Chain drafting (kDraftingStep steps, kVerifySize rows) on a base that verifies 12 rows: a grafting round verifies
//! and accepts J = W = min(12, 16) rows, beyond the assistant's own kVerifySize-token capacity.
constexpr int32_t kChainMaxVerifyTokens{12};
constexpr int32_t kChainWideVerifySize{12};

//! Tree drafting: top-2 over two steps and four verify rows on a base kVerifyHeadroom rows wider, so a grafting round
//! verifies V + 8 rows and accepts up to 9 tokens.
constexpr int32_t kTreeTopK{2};
constexpr int32_t kTreeSteps{2};
constexpr int32_t kTreeVerifySize{4};
constexpr int32_t kVerifyHeadroom{8};
constexpr int32_t kTreeWideVerifySize{kTreeVerifySize + kVerifyHeadroom};
constexpr int32_t kTreeWideAcceptSize{kVerifyHeadroom + 1};

//! Per-layer embedding rows of the cached-PLE deployment: one FP16 value per layer and token.
constexpr int64_t kPleLayers{2};
constexpr int32_t kPleCacheSlots{32};

Json makeGemma4MtpBaseConfig(int32_t maxVerifyTokens)
{
    Json config = makeSpecBaseConfig("gemma4_mtp", maxVerifyTokens);
    config["model"] = "gemma4_text";
    return config;
}

//! The assistant reads the base hidden state unchanged, decodes the base vocabulary and shares the target's KV cache.
Json makeGemma4MtpDraftConfig(Json const& baseConfig)
{
    Json config = makeSpecDraftConfig("gemma4_mtp", kVerifySize);
    config["model"] = "gemma4_assistant";
    config["base_model_hidden_size"] = baseConfig["hidden_size"];
    config["vocab_size"] = baseConfig["vocab_size"];
    config["shares_target_kv"] = true;
    config["has_own_kv_cache"] = false;
    config["returns_feedback_hidden"] = true;
    config["constant_draft_positions"] = true;
    config["kv_sharing_map"] = Json::array({Json::object({{"assistant_layer", 0}, {"target_attention_layer", 0}})});
    return config;
}

//! A Gemma4 MTP deployment whose base verify capacity the suite chooses. Requests run with the logger at DEBUG and
//! both log streams captured: a refusal is a warning, and a slot that ran with jump-forward gets a statistics line when
//! its request completes.
class Gemma4JumpForwardTestBase : public SpecAssemblyTest
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

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }

    //! The runtime over a context cache: the base holds a paged working set, the assistant owns no KV cache.
    rt::LLMInferenceRuntime makePagedRuntime(rt::ModelArtifacts artifacts)
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting();
        options.contextCacheConfig
            = {/*enabled=*/true, /*maxRecords=*/4, /*maxSnapshots=*/0, /*encoderEmbeddingCacheBudgetBytes=*/0};
        return rt::LLMInferenceRuntime{std::move(artifacts), std::move(options), mStream};
    }

    //! A greedy batch of `slots` prompts `a` whose grammar is `regex`, asking for jump-forward.
    static rt::LLMGenerationRequest makeRegexRequest(std::string const& regex, int64_t maxGenerateLength, size_t slots)
    {
        rt::LLMGenerationRequest request = makeGreedyBatch(std::vector<std::string>(slots, "a"), maxGenerateLength);
        for (auto& slot : request.requests)
        {
            slot.guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, regex, /*jumpForward=*/true};
        }
        return request;
    }

    //! One prepare() of the base: the dims it asked for, and the ragged query lengths, token positions and PLE row map
    //! it was handed.
    struct Prepared
    {
        int32_t profile{};
        rt::InferenceDims dims{};
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> positions;
        std::vector<int32_t> pleRows;
    };

    //! The value base forward `forward` writes into every channel of its row `row` of hidden states: distinct per
    //! forward and row, and exact in FP16.
    static float hiddenMarker(int32_t forward, int64_t row)
    {
        return static_cast<float>(100 * forward + row + 1);
    }

    //! A base engine recording every prepare() into `log`. Each forward ranks `token` first on every row and writes
    //! hiddenMarker(forward, row) into every row of its hidden-state output.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeBase(std::vector<Prepared>& log, int32_t token)
    {
        using ::testing::_;
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this, &log](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map,
                               cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = FakeOutputScale::kLogits;
                mBaseHidden = map.get(binding_names::kOutputHiddenStates);
                mLastDims = dims;
                auto const batch = static_cast<int32_t>(dims.batch);
                auto const rows = static_cast<int32_t>(dims.seqLen);
                log.push_back(
                    Prepared{profile, dims, readInt32Binding(map, binding_names::kQueryLengths, batch, stream),
                        readInt32Binding(map, binding_names::kPositions, rows, stream),
                        readInt32Binding(map, "ple_row_map", rows, stream)});
                return true;
            });
        ON_CALL(*engine, execute(_)).WillByDefault([this, token](cudaStream_t stream) {
            int64_t const rows = mLogits->getShape().volume() / kVocabSize;
            writeLogits(*mLogits, kVocabSize, std::vector<int32_t>(static_cast<size_t>(rows), token), stream);
            writeHiddenMarkers(stream);
            ++mBaseForwards;
            return true;
        });
        return engine;
    }

    //! An assistant engine recording, at every prepare(), the first channel of each slot's hidden-state input: for
    //! a round's first assistant forward, the seed hidden state gathered from the base's last verify round.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeAssistant(std::vector<std::vector<float>>& seeds)
    {
        using ::testing::_;
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault(
                [this, &seeds](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) {
                    mLogits = map.get(binding_names::kLogits);
                    mLogitsScale = FakeOutputScale::kLogits;
                    mLastDims = dims;
                    seeds.push_back(firstChannels(map.get(binding_names::kBaseModelHiddenStates), dims.batch, stream));
                    return true;
                });
        return engine;
    }

    //! The tree assistant's execute() action: its first step of a round ranks `x` over `q`, its second `z` over `e`.
    auto emitTreeSteps()
    {
        return [this](cudaStream_t stream) {
            bool const firstStep = mAssistantForwards++ % kTreeSteps == 0;
            writeRanked(firstStep ? kX : kZ, firstStep ? kQ : kE, stream);
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

    //! The base verify forwards: every decode-profile prepare of the base.
    static std::vector<Prepared const*> verifyRounds(std::vector<Prepared> const& log)
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

    //! The rows of every base verify forward.
    static std::vector<int64_t> verifyRows(std::vector<Prepared> const& log)
    {
        std::vector<int64_t> rows;
        for (auto const* round : verifyRounds(log))
        {
            rows.push_back(round->dims.seqLen);
        }
        return rows;
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

    static constexpr char const* kRefusal{"Jump-forward is not supported on"};

    rt::Tensor* mBaseHidden{nullptr};
    rt::InferenceDims mLastDims{};
    int32_t mBaseForwards{0};
    int32_t mAssistantForwards{0};

private:
    void writeHiddenMarkers(cudaStream_t stream)
    {
        if (mBaseHidden == nullptr || mBaseHidden->getShape().getNumDims() == 0)
        {
            return;
        }
        auto const& shape = mBaseHidden->getShape();
        int64_t const width = shape[shape.getNumDims() - 1];
        int64_t const rows = shape.volume() / width;
        std::vector<half> host(static_cast<size_t>(shape.volume()));
        for (int64_t row = 0; row < rows; ++row)
        {
            std::fill_n(host.begin() + row * width, width, __float2half(hiddenMarker(mBaseForwards, row)));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mBaseHidden->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    //! Channel 0 of each of the first `batch` rows of an FP16 [batch, hidden] binding.
    static std::vector<float> firstChannels(rt::Tensor const* tensor, int64_t batch, cudaStream_t stream)
    {
        if (tensor == nullptr || batch <= 0 || tensor->getShape().volume() < batch)
        {
            return {};
        }
        int64_t const width = tensor->getShape().volume() / batch;
        std::vector<half> host(static_cast<size_t>(tensor->getShape().volume()));
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), tensor->rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> channels;
        for (int64_t row = 0; row < batch; ++row)
        {
            channels.push_back(__half2float(host[static_cast<size_t>(row * width)]));
        }
        return channels;
    }

    //! Every row ranks `first` highest and `second` next, so a top-2 tree takes both.
    void writeRanked(int32_t first, int32_t second, cudaStream_t stream)
    {
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
    }

    logger::Severity mSavedLevel{};
};

//! Chain drafting whose grafting round verifies and accepts kChainWideVerifySize rows.
class Gemma4ChainJumpForwardTest : public Gemma4JumpForwardTestBase
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const base = makeGemma4MtpBaseConfig(kChainMaxVerifyTokens);
        return writeSpecModelDir("gemma4ChainJumpForwardTests", base, makeGemma4MtpDraftConfig(base));
    }
};

//! Tree drafting with kVerifyHeadroom rows of base verify headroom.
class Gemma4TreeJumpForwardTest : public Gemma4JumpForwardTestBase
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
        auto const base = makeGemma4MtpBaseConfig(kTreeWideVerifySize);
        return writeSpecModelDir("gemma4TreeJumpForwardTests", base, makeGemma4MtpDraftConfig(base));
    }
};

// Given a jumpable Gemma4 MTP chain deployment, a batch of two slots whose grammar `hello[a-z]+` forces `ello` after
//      the prefill's `h`, an assistant drafting e, l, l, and a base that predicts `x` at every row
// When eight tokens are generated per slot
// Then the first verify round is wide: each slot verifies h e l l o in its J rows and accepts the four forced tokens
//      over the base's prediction, then the bonus `x`, as one jump; the next round's first assistant forward is seeded
//      with the hidden state of that round's row acceptLen - 1 = 4, the `o` that predicted `x`, taken from each slot's
//      own J rows; every later round verifies V rows
TEST_F(Gemma4ChainJumpForwardTest, SeedsTheNextDraftFromTheLastAcceptedRowOfTheGraftingRound)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<std::vector<float>> seeds;
    auto baseEngine = makeBase(baseLog, kX);
    auto draftEngine = makeAssistant(seeds);
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kE, kE})).WillRepeatedly(emit({kL, kL}));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kChainWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kChainWideVerifySize);
    auto runtime = makeRuntime(std::move(artifacts));

    Run const on = run(runtime, makeRegexRequest("hello[a-z]+", /*maxGenerateLength=*/8, /*slots=*/2));

    std::vector<int32_t> const text{kH, kE, kL, kL, kO, kX, kX, kX};
    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], text);
    EXPECT_EQ(on.response.outputIds[1], text);
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    EXPECT_EQ(verifyRows(baseLog), (std::vector<int64_t>{2 * kChainWideVerifySize, 2 * kVerifySize, 2 * kVerifySize}));
    auto const rounds = verifyRounds(baseLog);
    ASSERT_FALSE(rounds.empty());
    EXPECT_EQ(rounds[0]->queryLengths, (std::vector<int32_t>{5, 5})) << "h, then e l l reused from the draft and o";
    EXPECT_EQ(countOf(on.debugLog, "jump-forward: jumps=1 forced=4 rolledBack=0 chunkSteps=1"), 2U) << on.debugLog;
    // The prefill is the base's forward 0 and the grafting round its forward 1; each round runs kDraftingStep
    // assistant forwards.
    ASSERT_GT(seeds.size(), static_cast<size_t>(kDraftingStep));
    EXPECT_EQ(
        seeds[kDraftingStep], (std::vector<float>{hiddenMarker(1, 4), hiddenMarker(1, kChainWideVerifySize + 4)}));
}

// Given a jumpable Gemma4 MTP chain deployment over a context cache, whose assistant owns no KV cache, and the batch
//      of the test above
// When eight tokens are generated per slot
// Then the decode steps grow the base working set alone and the request decodes as it does without the cache: both
//      slots spell h e l l o x x x with one jump of four forced tokens
TEST_F(Gemma4ChainJumpForwardTest, GrowsOnlyTheBaseWorkingSetOfAContextCacheForAGraftingRound)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<std::vector<float>> seeds;
    auto baseEngine = makeBase(baseLog, kX);
    auto draftEngine = makeAssistant(seeds);
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kE, kE})).WillRepeatedly(emit({kL, kL}));
    auto runtime = makePagedRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    Run const on = run(runtime, makeRegexRequest("hello[a-z]+", /*maxGenerateLength=*/8, /*slots=*/2));

    std::vector<int32_t> const text{kH, kE, kL, kL, kO, kX, kX, kX};
    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], text);
    EXPECT_EQ(on.response.outputIds[1], text);
    EXPECT_EQ(countOf(on.debugLog, "jump-forward: jumps=1 forced=4 rolledBack=0 chunkSteps=1"), 2U) << on.debugLog;
}

// Given a jumpable Gemma4 MTP tree deployment whose decoding graphs were captured, a batch of two slots whose grammar
//      `[a-z][xq]hi[a-z]+` forces `hi` after either `x` or `q`, an assistant that ranks `x` over `q` at depth one and
//      `z` over `e` at depth two, and a base that predicts `x` at every row
// When eight tokens are generated per slot
// Then the first verify round is wide: `hi` is grafted under both `x` and `q`, and the base verifies B·J rows, on a
//      shape whose graph was captured, with V + 4 valid rows per slot, each at its depth past the committed length;
//      each slot accepts x h i x, two forced tokens in one jump, and the next round's first assistant forward is
//      seeded with the hidden state of the accepted `i`, the path's row acceptLen - 1, from each slot's own rows;
//      every later round verifies V rows
TEST_F(Gemma4TreeJumpForwardTest, GraftsMidTreeChainsIntoAWideRoundOfBTimesJRows)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<std::vector<float>> seeds;
    std::vector<rt::InferenceDims> baseCaptures;
    auto baseEngine = makeBase(baseLog, kX);
    auto draftEngine = makeAssistant(seeds);
    ON_CALL(*baseEngine, captureGraph(_)).WillByDefault([this, &baseCaptures](cudaStream_t) {
        baseCaptures.push_back(mLastDims);
        return true;
    });
    ON_CALL(*draftEngine, execute(_)).WillByDefault(emitTreeSteps());
    ON_CALL(*draftEngine, captureGraph(_)).WillByDefault(::testing::Return(true));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, kTreeWideVerifySize);
    ASSERT_EQ(artifacts.deployment.specConfig->jumpMaxPathLength, kTreeWideAcceptSize);
    auto runtime = makeRuntime(std::move(artifacts));
    ASSERT_TRUE(runtime.captureDecodingCUDAGraph(mStream));
    baseLog.clear();
    seeds.clear();

    Run const on = run(runtime, makeRegexRequest("[a-z][xq]hi[a-z]+", /*maxGenerateLength=*/8, /*slots=*/2));

    std::vector<int32_t> const text{kX, kX, kH, kI, kX, kX, kX, kX};
    ASSERT_EQ(on.response.outputIds.size(), 2U);
    EXPECT_EQ(on.response.outputIds[0], text);
    EXPECT_EQ(on.response.outputIds[1], text);
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    EXPECT_EQ(
        verifyRows(baseLog), (std::vector<int64_t>{2 * kTreeWideVerifySize, 2 * kTreeVerifySize, 2 * kTreeVerifySize}));
    auto const rounds = verifyRounds(baseLog);
    ASSERT_FALSE(rounds.empty());
    EXPECT_EQ(rounds[0]->queryLengths, (std::vector<int32_t>{kTreeVerifySize + 4, kTreeVerifySize + 4}));
    // Draft tree root, x, q, x>z; then h, i under x and h, i under q; the prompt `a` is the committed state.
    std::vector<int32_t> const slotPositions{1, 2, 2, 3, 3, 4, 3, 4, -1, -1, -1, -1};
    std::vector<int32_t> positions = slotPositions;
    positions.insert(positions.end(), slotPositions.begin(), slotPositions.end());
    EXPECT_EQ(rounds[0]->positions, positions);
    for (auto const* round : rounds)
    {
        bool const captured = std::any_of(baseCaptures.begin(), baseCaptures.end(), [round](auto const& graph) {
            return graph.batch == round->dims.batch && graph.seqLen == round->dims.seqLen
                && graph.selectLen == round->dims.selectLen && graph.executionPhaseLen == round->dims.executionPhaseLen
                && graph.specVerifyLayoutLen == round->dims.specVerifyLayoutLen;
        });
        EXPECT_TRUE(captured) << "no graph for the verify forward of " << round->dims.seqLen << " rows";
    }
    EXPECT_EQ(countOf(on.debugLog, "jump-forward: jumps=1 forced=2 rolledBack=0 chunkSteps=1"), 2U) << on.debugLog;
    // The accepted path is the root, x and the h, i grafted under x: verify rows 0, 1, 4, 5 of each slot.
    ASSERT_GT(seeds.size(), static_cast<size_t>(kTreeSteps));
    EXPECT_EQ(seeds[kTreeSteps], (std::vector<float>{hiddenMarker(1, 5), hiddenMarker(1, kTreeWideVerifySize + 5)}));
}

// Given a jumpable Gemma4 MTP chain deployment with cached per-layer embeddings, whose speculative PLE staging holds
//      max_spec_verify_tokens = 12 rows per slot, a grammar `hello` that forces `ello` after the prefill's `h`, and a
//      base that predicts `x` at every row
// When the request runs
// Then its grafting round verifies J = 12 rows, all the staging holds, and binds a PLE row map that covers every one:
//      h, e, l, l, o map to cache slots that only equal tokens share and the seven padding rows to the zero slot; the
//      text is `hello`
TEST_F(Gemma4ChainJumpForwardTest, StagesTheGraftingRoundsPerLayerEmbeddingsWithinTheVerifyCapacity)
{
    using ::testing::_;
    std::vector<Prepared> baseLog;
    std::vector<std::vector<float>> seeds;
    auto draftEngine = makeAssistant(seeds);
    EXPECT_CALL(*draftEngine, execute(_)).WillOnce(emit({kE})).WillRepeatedly(emit({kL}));
    auto artifacts = makeArtifacts(makeBase(baseLog, kX), std::move(draftEngine));
    ASSERT_EQ(artifacts.deployment.specConfig->jumpVerifyTokenCount, artifacts.deployment.specConfig->maxVerifyTokens);
    auto& base = artifacts.deployment.base;
    base.gemmaPleEnabled = true;
    base.gemmaPleLayerCount = kPleLayers;
    base.gemmaPleHiddenSize = 1;
    base.plePool = rt::PlePoolConfig{kVocabSize, kPleLayers, rt::PleTable::DType::kFP16, 1.0};
    base.plePool.source
        = {"weight", rt::PleTable::DType::kFP16, kVocabSize * kPleLayers * 2, 1.0, {{0, "003c"}}, "ple_table.bin", 0};
    base.plePool.checkpointDir = mModelDir.string();
    std::vector<uint16_t> const table(static_cast<size_t>(kVocabSize * kPleLayers), 0x3C00);
    {
        std::ofstream shard(mModelDir / "ple_table.bin", std::ios::binary);
        shard.write(
            reinterpret_cast<char const*>(table.data()), static_cast<std::streamsize>(table.size() * sizeof(uint16_t)));
        ASSERT_TRUE(shard.good());
    }
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.draftingConfig = drafting();
    options.pleConfig = {rt::PleResidency::kCACHED, kPleCacheSlots};
    rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};

    Run const on = run(runtime, makeRegexRequest("hello", /*maxGenerateLength=*/8, /*slots=*/1));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kEos}));
    auto const rounds = verifyRounds(baseLog);
    ASSERT_EQ(rounds.size(), 1U) << "the whole word in one grafting round";
    EXPECT_EQ(rounds[0]->dims.seqLen, kChainWideVerifySize);
    std::vector<int32_t> const& rows = rounds[0]->pleRows;
    ASSERT_EQ(rows.size(), static_cast<size_t>(kChainWideVerifySize));
    std::vector<int32_t> const tokens{kH, kE, kL, kL, kO};
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        EXPECT_NE(rows[i], 0) << "row " << i;
        for (size_t j = 0; j < i; ++j)
        {
            EXPECT_EQ(rows[i] == rows[j], tokens[i] == tokens[j]) << "rows " << j << " and " << i;
        }
    }
    EXPECT_TRUE(std::all_of(
        rows.begin() + static_cast<std::ptrdiff_t>(tokens.size()), rows.end(), [](int32_t slot) { return slot == 0; }));
}

} // namespace
