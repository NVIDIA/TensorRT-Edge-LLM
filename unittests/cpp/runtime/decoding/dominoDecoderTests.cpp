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

// Domino speculative decoding assembled around two substitute engines and a correction-head sidecar the test writes:
// the draft emits a candidate pool, the decoder replays the GRU correction along the chain, and the base verifies the
// corrected chain.

#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kEos{1};
constexpr int32_t kI{'i' - 'b' + 3};
constexpr int32_t kL{'l' - 'b' + 3};
constexpr int32_t kO{'o' - 'b' + 3};
constexpr int32_t kQ{'q' - 'b' + 3};
constexpr int32_t kV{'v' - 'b' + 3};
constexpr int32_t kW{'w' - 'b' + 3};
constexpr int32_t kX{'x' - 'b' + 3};
constexpr int32_t kY{'y' - 'b' + 3};
constexpr int32_t kZ{'z' - 'b' + 3};

//! One draft forward emits the anchor and three proposals; the GRU corrects every proposal after the first, so a round
//! runs two correction steps. Linear verification makes the block size the verify width too.
constexpr int32_t kDominoBlockSize{4};
constexpr int32_t kGruHiddenDim{2};
constexpr int32_t kProjDim{2};
constexpr int32_t kPoolSize{4};

// The chain the rollout derives from the draft outputs and the sidecar below. The draft's uncorrected pool logits put
// the first proposal on y and every later one on q. The sidecar's GRU replaces its state with (1, -1) after reading y
// and with (-1, 1) after reading v (update gate shut, candidate saturated, no recurrent weight), and projects the
// state by 4 * I; the draft's projected hidden is zero. The correction after y is SiLU(4, -4) ~ (3.93, -0.07), so v,
// whose codebook row reads the first component, outscores q's 1; after v it is ~ (-0.07, 3.93) and w wins. The chain
// is y v w, where the uncorrected argmax gives y q q and a state left at y gives y v v.
constexpr std::array<int32_t, kPoolSize> kPool{kQ, kY, kV, kW};
constexpr std::array<std::array<float, kPoolSize>, kDominoBlockSize - 1> kPoolLogits{
    {{0.0F, 2.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F, 0.0F}}};
constexpr std::array<std::array<float, kProjDim>, kPoolSize> kCodebook{
    {{0.0F, 0.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}}};

void addDominoConfig(Json& config)
{
    Json dflash;
    dflash["projector_type"] = "domino";
    // Both engines must name layers that exist in the base; the draft reads their hidden states.
    dflash["target_layer_ids"] = Json::array({0, 1});
    dflash["block_size"] = kDominoBlockSize;
    // The default is a real Qwen3 id, far outside this deployment's 128-token vocabulary.
    dflash["mask_token_id"] = 3;
    dflash["gru_hidden_dim"] = kGruHiddenDim;
    dflash["proj_dim"] = kProjDim;
    dflash["pool_size"] = kPoolSize;
    config["dflash_config"] = dflash;
}

Json makeDominoBaseConfig()
{
    Json config = makeSpecBaseConfig("domino", kDominoBlockSize);
    addDominoConfig(config);
    return config;
}

Json makeDominoDraftConfig()
{
    Json config = makeSpecDraftConfig("domino", kDominoBlockSize);
    addDominoConfig(config);
    // Concatenated hidden states from the two target layers named above.
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

//! The correction head described above, under the tensor names and shapes the decoder loads.
void writeDominoHeads(std::filesystem::path const& path, cudaStream_t stream)
{
    constexpr float kGateShut{-16.0F};
    constexpr float kSaturated{8.0F};
    constexpr float kStateGain{4.0F};
    std::vector<rt::Tensor> heads;
    heads.emplace_back(rt::Coords{kVocabSize, 3 * kGruHiddenDim}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF,
        "gru_input_table");
    heads.emplace_back(rt::Coords{3 * kGruHiddenDim, kGruHiddenDim}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF,
        "gru_weight_hh");
    heads.emplace_back(
        rt::Coords{kProjDim, kGruHiddenDim}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF, "proj_state_weight");

    // Input table rows are the reset, update and candidate gate inputs, kGruHiddenDim each.
    half* const table = heads[0].dataPointer<half>();
    for (int64_t token = 0; token < kVocabSize; ++token)
    {
        float const candidate = token == kY ? kSaturated : (token == kV ? -kSaturated : 0.0F);
        half* const row = table + token * 3 * kGruHiddenDim;
        for (int32_t g = 0; g < kGruHiddenDim; ++g)
        {
            row[g] = __float2half(0.0F);
            row[kGruHiddenDim + g] = __float2half(kGateShut);
            row[2 * kGruHiddenDim + g] = __float2half(g == 0 ? candidate : -candidate);
        }
    }
    std::fill_n(heads[1].dataPointer<half>(), 3 * kGruHiddenDim * kGruHiddenDim, __float2half(0.0F));
    half* const projection = heads[2].dataPointer<half>();
    for (int32_t r = 0; r < kProjDim; ++r)
    {
        for (int32_t g = 0; g < kGruHiddenDim; ++g)
        {
            projection[r * kGruHiddenDim + g] = __float2half(r == g ? kStateGain : 0.0F);
        }
    }
    EXPECT_TRUE(rt::safetensors::saveSafetensors(path, heads, stream));
}

template <typename T>
void upload(rt::Tensor& tensor, std::vector<T> const& values, cudaStream_t stream)
{
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.rawPointer(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}

//! The Domino draft engine's outputs, which take the place of logits: the candidate pool, its uncorrected logits per
//! proposal position, its codebook rows and the projected hidden state of every corrected position. Every forward
//! writes the fixture's pool to every bound row.
struct FakeDominoDraft
{
    rt::Tensor* poolIds{nullptr};
    rt::Tensor* poolLogits{nullptr};
    rt::Tensor* codebook{nullptr};
    rt::Tensor* zProj{nullptr};
    int64_t batch{0};

    //! Takes the bindings the runtime handed to prepare().
    void prepare(rt::InferenceDims const& dims, rt::TensorMap const& map)
    {
        poolIds = map.get(binding_names::kSpecPoolCandidateIds);
        poolLogits = map.get(binding_names::kSpecPoolBaseLogits);
        codebook = map.get(binding_names::kSpecPoolCodebook);
        zProj = map.get(binding_names::kSpecProposalZProj);
        batch = dims.batch;
    }

    //! Whether the runtime bound the four outputs at the engine's shapes; reports a test failure when not.
    bool boundAsEngine() const
    {
        struct Output
        {
            rt::Tensor const* tensor;
            char const* name;
            rt::Coords shape;
        };
        std::array<Output, 4> const outputs{{
            {poolIds, binding_names::kSpecPoolCandidateIds, {batch, kPoolSize}},
            {poolLogits, binding_names::kSpecPoolBaseLogits, {batch, kDominoBlockSize - 1, kPoolSize}},
            {codebook, binding_names::kSpecPoolCodebook, {batch, kPoolSize, kProjDim}},
            {zProj, binding_names::kSpecProposalZProj, {batch, kDominoBlockSize - 2, kProjDim}},
        }};
        for (auto const& output : outputs)
        {
            if (output.tensor == nullptr || output.tensor->getShape() != output.shape)
            {
                ADD_FAILURE() << "Domino draft output " << output.name << " is bound as "
                              << (output.tensor == nullptr ? std::string("nothing")
                                                           : output.tensor->getShape().formatString())
                              << ", expected " << output.shape.formatString();
                return false;
            }
        }
        return true;
    }

    //! One forward, or a failed forward when !boundAsEngine().
    bool write(cudaStream_t stream)
    {
        if (!boundAsEngine())
        {
            return false;
        }
        std::vector<int32_t> ids;
        std::vector<float> logits;
        std::vector<half> codes;
        for (int64_t row = 0; row < batch; ++row)
        {
            ids.insert(ids.end(), kPool.begin(), kPool.end());
            for (auto const& position : kPoolLogits)
            {
                logits.insert(logits.end(), position.begin(), position.end());
            }
            for (auto const& slot : kCodebook)
            {
                for (float const value : slot)
                {
                    codes.push_back(__float2half(value));
                }
            }
        }
        std::vector<half> const projectedHidden(
            static_cast<size_t>(batch * (kDominoBlockSize - 2) * kProjDim), __float2half(0.0F));
        upload(*poolIds, ids, stream);
        upload(*poolLogits, logits, stream);
        upload(*codebook, codes, stream);
        upload(*zProj, projectedHidden, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
};

//! The token each of the first `rows` rows of the bound `inputs_embeds` embeds, or -1 for a row that embeds none.
//! finishArtifacts() writes fakeEmbeddingValue(token, 0) into channel 0, which no two tokens of the vocabulary share.
std::vector<int32_t> embeddedTokens(rt::TensorMap const& map, int64_t rows, cudaStream_t stream)
{
    std::map<int32_t, int32_t> tokenOfValue;
    for (int32_t token = 0; token < static_cast<int32_t>(kVocabSize); ++token)
    {
        tokenOfValue.emplace(fakeEmbeddingValue(token, 0), token);
    }
    EXPECT_EQ(tokenOfValue.size(), static_cast<size_t>(kVocabSize)) << "channel 0 no longer identifies its token";
    rt::Tensor* const embeds = map.get(binding_names::kInputsEmbeds);
    if (embeds == nullptr)
    {
        ADD_FAILURE() << "the base forward bound no " << binding_names::kInputsEmbeds;
        return {};
    }
    auto const& shape = embeds->getShape();
    int64_t const hidden = shape[shape.getNumDims() - 1];
    std::vector<half> host(static_cast<size_t>(rows * hidden));
    CUDA_CHECK(
        cudaMemcpyAsync(host.data(), embeds->rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> tokens;
    for (int64_t row = 0; row < rows; ++row)
    {
        auto const it = tokenOfValue.find(static_cast<int32_t>(__half2float(host[static_cast<size_t>(row * hidden)])));
        tokens.push_back(it == tokenOfValue.end() ? -1 : it->second);
    }
    return tokens;
}

//! A Domino deployment over the tiny base: a linear chain of kDominoBlockSize verify rows and the sidecar above.
class DominoAssemblyTest : public SpecAssemblyTest
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

    std::filesystem::path stageModelDir() override
    {
        auto const dir = writeSpecModelDir("dominoAssemblyTests", makeDominoBaseConfig(), makeDominoDraftConfig());
        writeDominoHeads(dir / binding_names::kDominoHeadsFileName, mStream);
        return dir;
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig config{};
        config.draftTopK = 1;
        config.proposalTokens = kDominoBlockSize - 1;
        config.verifyTokens = kDominoBlockSize;
        return config;
    }

    //! A base whose prefill decodes to x and whose verify forwards decode their rows to `verifyRows`. It records the
    //! tokens each verify forward was handed in mVerifiedTokens.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeBaseEngine(std::vector<int32_t> verifyRows)
    {
        using ::testing::_;
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this](
                               int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                mLogitsScale = FakeOutputScale::kLogits;
                mBaseVerifying = dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify);
                if (mBaseVerifying)
                {
                    mVerifiedTokens.push_back(embeddedTokens(map, dims.seqLen, stream));
                }
                return true;
            });
        ON_CALL(*engine, execute(_)).WillByDefault([this, verifyRows = std::move(verifyRows)](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, mBaseVerifying ? verifyRows : std::vector<int32_t>{kX}, stream);
            return true;
        });
        return engine;
    }

    //! A draft engine that emits the fixture's candidate pool and, like the exported Domino draft, no logits.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeDraftEngine()
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::StrEq;
        auto engine = makeEngine();
        ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kLogits))).WillByDefault(Return(false));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mDraft.prepare(dims, map);
                return true;
            });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) { return mDraft.write(stream); });
        return engine;
    }

    struct Run
    {
        rt::LLMGenerationResponse response;
        std::string debugLog; //!< Debug and info lines go to stdout.
    };

    //! Runs `request` with the logger at DEBUG and stdout captured: every speculative round logs its accounting, and a
    //! slot that ran with jump-forward gets a statistics line when its request completes.
    Run run(rt::LLMInferenceRuntime& runtime, rt::LLMGenerationRequest const& request)
    {
        gLogger.setLevel(logger::Severity::kVERBOSE);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        Run result;
        bool const handled = runtime.handleRequest(request, result.response, mStream);
        result.debugLog = testing::internal::GetCapturedStdout();
        std::string const warnings = testing::internal::GetCapturedStderr();
        EXPECT_TRUE(handled) << warnings;
        return result;
    }

    FakeDominoDraft mDraft;
    std::vector<std::vector<int32_t>> mVerifiedTokens;
    bool mBaseVerifying{false};

private:
    logger::Severity mSavedLevel{};
};

// Given a Domino deployment whose correction head moves the second and third proposals off the draft's uncorrected
//      pool argmax q, to v and then w, and a base that agrees with that chain
// When a greedy request generates two blocks
// Then each round's one verify forward holds the root x and the corrected chain y v w, the base accepts the whole
//      block both times, and the request ends after two verify rounds
TEST_F(DominoAssemblyTest, VerifiesTheCorrectedChainOnceEveryRound)
{
    auto runtime = makeRuntime(makeArtifacts(makeBaseEngine({kY, kV, kW, kX}), makeDraftEngine()));

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", 2 * kDominoBlockSize + 1), response, mStream));

    EXPECT_EQ(mVerifiedTokens, (std::vector<std::vector<int32_t>>{{kX, kY, kV, kW}, {kX, kY, kV, kW}}));
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kX, kY, kV, kW, kX, kY, kV, kW, kX}));
    ASSERT_EQ(response.specVerifyCounts.size(), 1U);
    EXPECT_EQ(response.specVerifyCounts[0], 2);
}

// Given the Domino deployment, a greedy jump-forward request whose grammar `[xq][yz]oil` forces `oil` once y is
//      chosen, the rollout's chain y v w under the root x, and a base that predicts y at the root and z at every other
//      row
// When the request runs
// Then the forced chain replaces the rollout's chain after y in place: the first verify forward holds x y o i at the
//      block width and accepts y as a draft, o and i as forced tokens over the base's z, and l as the bonus, and the
//      next one holds the rollout's chain under l again; the first round's four appended tokens count one accepted
//      draft of the three proposed, the request counts one jump of two forced tokens, and the text is `xyoil` and the
//      EOS
TEST_F(DominoAssemblyTest, GraftsAForcedChainAfterTheRolloutInPlace)
{
    auto runtime = makeRuntime(makeArtifacts(makeBaseEngine({kY, kZ, kZ, kZ}), makeDraftEngine()));
    rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/10);
    request.requests.front().guidedDecoding
        = rt::GuidedDecodingParams{rt::GuideType::kRegex, "[xq][yz]oil", /*jumpForward=*/true};

    Run const on = run(runtime, request);

    EXPECT_EQ(mVerifiedTokens, (std::vector<std::vector<int32_t>>{{kX, kY, kO, kI}, {kL, kY, kV, kW}}));
    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kX, kY, kO, kI, kL, kEos}));
    EXPECT_NE(
        on.debugLog.find("SpecRound 0 batchIdx 0: generated=4, acceptedDrafts=1, proposedDrafts=3"), std::string::npos)
        << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=2 "), std::string::npos) << on.debugLog;
}

} // namespace
