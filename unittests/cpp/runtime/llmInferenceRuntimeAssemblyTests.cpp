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

//
// Assembling LLMInferenceRuntime around injected artifacts, with no serialized engine anywhere.
//
// The only substituted component is EngineExecutor. Everything else is the production code path: the parsed
// deployment config, the KV cache managers, the pipeline tensors, the tensor map, and the decoder registry.
//

#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRankRuntime.h"

#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/modelArtifacts.h"
#include "runtime/streaming.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <unistd.h>

#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

//! gmock falls back to a raw byte dump for types it cannot print, which buries the interesting part of a failed
//! expectation. Found by ADL, so these must sit in the namespace of the type they print.
namespace trt_edgellm
{
namespace rt
{
void PrintTo(Tensor const& tensor, std::ostream* os)
{
    *os << "Tensor{" << tensor.getName() << ", shape=" << tensor.getShape().formatString() << "}";
}

void PrintTo(FinishReason reason, std::ostream* os)
{
    switch (reason)
    {
    case FinishReason::kNotFinished: *os << "kNotFinished"; return;
    case FinishReason::kEndId: *os << "kEndId"; return;
    case FinishReason::kLength: *os << "kLength"; return;
    case FinishReason::kCancelled: *os << "kCancelled"; return;
    case FinishReason::kError: *os << "kError"; return;
    case FinishReason::kStopWords: *os << "kStopWords"; return;
    }
    *os << "FinishReason(" << static_cast<int32_t>(reason) << ")";
}

void PrintTo(InferenceDims const& dims, std::ostream* os)
{
    *os << "InferenceDims{batch=" << dims.batch << ", seqLen=" << dims.seqLen << ", kvLen=" << dims.kvLen
        << ", selectLen=" << dims.selectLen << "}";
}
} // namespace rt
} // namespace trt_edgellm

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

//! The same interface as a gmock double, for tests that want to state the calls they expect up front rather than
//! count them afterwards. Callers wrap it in NiceMock and give the assembly-time queries a default via ON_CALL.
class MockEngineExecutor : public rt::EngineExecutor
{
public:
    MOCK_METHOD(bool, prepare,
        (int32_t profileIndex, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream),
        (override));
    MOCK_METHOD(bool, execute, (cudaStream_t stream), (override));
    MOCK_METHOD(bool, captureGraph, (cudaStream_t stream), (override));
    MOCK_METHOD(int64_t, getRequiredContextMemorySize, (), (const, override));
    MOCK_METHOD(bool, setContextMemory, (rt::Tensor & sharedMem), (override));
    MOCK_METHOD(int32_t, getNumIOTensors, (), (const, override));
    MOCK_METHOD(char const*, getIOTensorName, (int32_t index), (const, override));
    MOCK_METHOD(bool, hasIOTensor, (char const* name), (const, override));
    MOCK_METHOD(nvinfer1::DataType, getBindingDataType, (char const* name), (const, override));
    MOCK_METHOD(nvinfer1::Dims, getProfileShape,
        (char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector), (const, override));
    MOCK_METHOD(void, setProfiler, (nvinfer1::IProfiler * profiler), (noexcept, override));
    //! Left without a default action on purpose: it returns a reference gmock cannot invent, so any call aborts the
    //! test. Nothing the runtime does should reach past the interface for the TRT engine.
    MOCK_METHOD(nvinfer1::ICudaEngine const&, getEngine, (), (const, noexcept, override));
};

//! Vocabulary of the tiny test deployment. Named because the tests choose token ids out of it.
constexpr int64_t kVocabSize{128};
constexpr int64_t kMaxBatchSize{2};

//! What greedy sampling returns for a row of zeroed logits: every entry ties, and the sampler keeps the highest
//! index. The tests below depend on this being stable, not on the tie-break rule itself.
constexpr int32_t kZeroLogitsToken{static_cast<int32_t>(kVocabSize) - 1};

//! Optimization-profile indices baked into every engine by llmBuilder: 0 is prefill, 1 is decode (and speculative
//! proposal / verification).
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

//! A deployment small enough that every derived allocation stays in the low megabytes.
Json makeTinyVanillaConfig()
{
    Json config;
    config["num_hidden_layers"] = 2;
    config["num_key_value_heads"] = 2;
    config["head_dim"] = 16;
    config["hidden_size"] = 64;
    config["vocab_size"] = kVocabSize;
    config["kv_cache_dtype"] = "fp16";
    config["spec_decode_type"] = "none";
    config["engine_role"] = "llm";

    Json builder;
    builder["max_batch_size"] = kMaxBatchSize;
    builder["max_input_len"] = 32;
    builder["max_kv_cache_capacity"] = 64;
    builder["max_kv_pool_pages"] = 4;
    builder["max_lora_rank"] = 0;
    builder["spec_base"] = false;
    config["builder_config"] = builder;
    return config;
}

//! The tokenizer trio every deployment needs. Tiny, but real: the runtime tokenizes and detokenizes for real.
void writeTokenizerFiles(std::filesystem::path const& dir)
{
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {"a": 0, "<eos>": 1, "<bos>": 2}, "merges": []},
  "added_tokens": [
    {"id": 1, "content": "<eos>"},
    {"id": 2, "content": "<bos>"}
  ],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";

    std::ofstream(dir / "tokenizer_config.json")
        << R"JSON({"eos_token": {"content": "<eos>"}, "bos_token": {"content": "<bos>"}})JSON";

    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {
    "system": {"prefix": "", "suffix": ""},
    "user": {"prefix": "", "suffix": ""},
    "assistant": {"prefix": "", "suffix": ""}
  },
  "generation_prompt": ""
})JSON";
}

//! Per-process, because /tmp is shared: two users running unitTest on the same machine would otherwise have one
//! SetUp remove_all the directory the other is reading, or fail outright on a directory owned by another uid.
std::filesystem::path freshModelDir(std::string const& name)
{
    auto const dir = std::filesystem::temp_directory_path() / (name + "." + std::to_string(getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

//! Write the files the artifacts are built from. None of them is an engine.
std::filesystem::path writeModelDir(Json const& config)
{
    auto const dir = freshModelDir("edgellm_runtime_assembly_test");
    std::ofstream(dir / "config.json") << config.dump(2);
    writeTokenizerFiles(dir);
    return dir;
}

//! Write the logits a forward pass would have produced.
//!
//! `tokensPerRow` names the argmax for each engine row; a row that is absent or negative stays zeroed, and the
//! sampler's tie-break then makes it decode to the same token every round.
void writeLogits(rt::Tensor& logits, int64_t vocabSize, std::vector<int32_t> const& tokensPerRow, cudaStream_t stream)
{
    auto const rows = logits.getShape().volume() / vocabSize;
    std::vector<float> host(static_cast<size_t>(logits.getShape().volume()), 0.0F);
    for (int64_t row = 0; row < rows && row < static_cast<int64_t>(tokensPerRow.size()); ++row)
    {
        if (tokensPerRow[static_cast<size_t>(row)] >= 0)
        {
            host[static_cast<size_t>(row * vocabSize + tokensPerRow[static_cast<size_t>(row)])] = 10.0F;
        }
    }
    CUDA_CHECK(
        cudaMemcpyAsync(logits.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

//! handleRequest documents that on success it repopulates the per-slot response vectors together, to matched
//! sizes. Checking that as a block keeps every test from having to re-derive it.
void expectResponseCoversEverySlot(rt::LLMGenerationResponse const& response, size_t slots)
{
    EXPECT_EQ(response.outputIds.size(), slots);
    EXPECT_EQ(response.outputTexts.size(), slots);
    EXPECT_EQ(response.finishReasons.size(), slots);
    EXPECT_EQ(response.inputTokenCounts.size(), slots);
}

//! Assemble the artifacts a vanilla text deployment needs around a caller-supplied engine.
rt::ModelArtifacts makeVanillaArtifacts(
    std::filesystem::path const& modelDir, std::unique_ptr<rt::EngineExecutor> executor, cudaStream_t stream)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment = rt::createDeploymentConfig(modelDir / "config.json", std::nullopt, std::nullopt);
    artifacts.baseExecutor = std::move(executor);

    // The model directory holds no sidecars and the config declares no checkpoint bindings, so this loads and
    // validates zero tensors. It still has to happen: assembly publishes the manager into the tensor map, and the
    // manager refuses that before it has been loaded and validated.
    artifacts.weights.load(modelDir, modelDir / "config.json", stream);
    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");

    artifacts.embedding.table = rt::Tensor({artifacts.deployment.base.vocabSize, artifacts.deployment.base.hiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "test::embedding");
    CUDA_CHECK(cudaMemsetAsync(artifacts.embedding.table.rawPointer(), 0,
        static_cast<size_t>(artifacts.embedding.table.getShape().volume()) * sizeof(half), stream));

    artifacts.tokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!artifacts.tokenizer->loadFromHF(modelDir.string()))
    {
        throw std::runtime_error("test tokenizer failed to load");
    }
    return artifacts;
}

//! A single-slot greedy request. Greedy keeps the sampled token a function of the logits the mock wrote.
rt::LLMGenerationRequest makeGreedyRequest(std::string const& prompt, int64_t maxGenerateLength)
{
    rt::LLMGenerationRequest request{};
    rt::LLMGenerationRequest::Request one;
    one.messages.push_back(rt::Message{"user", {rt::Message::MessageContent{"text", prompt}}});
    request.requests.push_back(std::move(one));
    request.temperature = 0.0F;
    request.topK = 1;
    request.topP = 1.0F;
    request.maxGenerateLength = maxGenerateLength;
    return request;
}

class RuntimeAssemblyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeModelDir(makeTinyVanillaConfig());
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    //! A mock engine with the assembly-time queries answered, so each test only states the calls it cares about.
    //!
    //! NiceMock, because assembly asks the engine a handful of questions that no test is about; leaving those to
    //! warn would bury the expectations that matter. Calls a test does declare are still checked strictly.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeEngine()
    {
        using ::testing::_;
        using ::testing::Return;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(kContextMemoryBytes));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(nvinfer1::DataType::kHALF));
        ON_CALL(*engine, captureGraph(_)).WillByDefault(Return(false));
        // The logits binding is only reachable through the tensor map the runtime hands to prepare().
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t stream) {
                mLogits = map.get(binding_names::kLogits);
                rt::Tensor* const pageTable = map.get(binding_names::kKVPageTable);
                int32_t firstPage{};
                CUDA_CHECK(cudaMemcpyAsync(
                    &firstPage, pageTable->rawPointer(), sizeof(firstPage), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                mFirstPagePerPrepare.push_back(firstPage);
                return true;
            });
        return engine;
    }

    //! An `execute()` action standing in for one forward pass. `tokensPerRow` names the argmax for each engine row;
    //! rows left out decode to whatever the sampler's tie-break picks out of the zeroed logits, which is stable.
    auto emit(std::vector<int32_t> tokensPerRow)
    {
        return [this, tokensPerRow = std::move(tokensPerRow)](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, tokensPerRow, stream);
            return true;
        };
    }

    static constexpr int64_t kContextMemoryBytes{4096};

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    //! Captured by `makeEngine`'s prepare() default; valid from the first prepare() until the runtime dies.
    rt::Tensor* mLogits{nullptr};
    std::vector<int32_t> mFirstPagePerPrepare;
};

TEST_F(RuntimeAssemblyTest, AssemblesWithoutAnyEngineFileOnDisk)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    ASSERT_FALSE(std::filesystem::exists(mModelDir / "llm.engine"));
    EXPECT_FALSE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "vanilla");
}

TEST_F(RuntimeAssemblyTest, SizesSharedContextMemoryFromTheExecutorItWasGiven)
{
    using ::testing::Ge;
    using ::testing::ResultOf;

    auto engine = makeEngine();
    auto& mock = *engine;

    // Assembly must ask the engine what it needs and hand back a buffer at least that large. Stated as a matcher on
    // the argument, so the expectation reads as the rule rather than as a value captured and compared later.
    EXPECT_CALL(mock, getRequiredContextMemorySize());
    EXPECT_CALL(mock,
        setContextMemory(
            ResultOf([](rt::Tensor const& memory) { return memory.getShape().volume(); }, Ge(kContextMemoryBytes))));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
}

TEST_F(RuntimeAssemblyTest, DrivesPrefillAndOneDecodeRoundPerGeneratedToken)
{
    using ::testing::_;
    using ::testing::AllOf;
    using ::testing::Each;
    using ::testing::SizeIs;

    constexpr int64_t kMaxGenerateLength{4};

    auto engine = makeEngine();
    auto& mock = *engine;

    // Prefill emits the first token, so N tokens cost one prefill forward plus N-1 decode forwards. Each round
    // switches the engine to the profile its shapes were built for.
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(kMaxGenerateLength - 1);
    EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength).WillRepeatedly(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    // The prompt is one character, and this tokenizer spends one token per character. Asserting the count the
    // runtime reports back makes that a checked fact rather than a comment the other tests lean on.
    EXPECT_EQ(response.inputTokenCounts[0], 1);
    // Every forward left the logits zeroed, so the whole completion is the tie-break token.
    EXPECT_THAT(response.outputIds[0], AllOf(SizeIs(kMaxGenerateLength), Each(kZeroLogitsToken)));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, StopsAtTheEosTokenTheEngineProduces)
{
    using ::testing::_;
    using ::testing::InSequence;

    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    // Prefill and the first decode round emit ordinary tokens; the second decode round emits EOS. Stating this as a
    // sequence means a fourth forward fails on its own — no counter, and the failure names the call that broke it.
    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).Times(2).WillRepeatedly(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId}));
    }

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/8);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), 3U);
    EXPECT_EQ(response.outputIds[0].back(), eosId);
    // The distinction the test is named for: stopping on the sampled token rather than on the length cap, which is
    // still 8 rounds away.
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
}

TEST_F(RuntimeAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    using ::testing::_;
    using ::testing::InSequence;

    constexpr int64_t kMaxGenerateLength{5};

    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    // On the second forward slot 0 hits EOS while slot 1 keeps going. Slot 1 then occupies engine row 0 for the
    // remaining rounds, which is the batch compaction under test: the later expectations name row 0 and would not
    // match if the survivor had stayed in row 1.
    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId, -1}));
        EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength - 2).WillRepeatedly(emit({}));
    }

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto request = makeGreedyRequest("a", kMaxGenerateLength);
    request.requests.push_back(request.requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 2);
    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_EQ(response.outputIds[0].size(), 2U);
    EXPECT_EQ(response.outputIds[0].back(), eosId);
    EXPECT_EQ(response.outputIds[1].size(), static_cast<size_t>(kMaxGenerateLength));
    EXPECT_NE(response.outputIds[1].back(), eosId);
    // The two slots left for different reasons, and the compacted slot still reports against its own index.
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, LogicalEvictionRemapsTheSurvivorAndTheNextRequestRestoresIdentity)
{
    using ::testing::_;
    using ::testing::InSequence;

    constexpr int64_t kMaxGenerateLength{4};
    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId, -1}));
        EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength - 2).WillRepeatedly(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId}));
    }

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto firstRequest = makeGreedyRequest("a", kMaxGenerateLength);
    firstRequest.requests.push_back(firstRequest.requests.front());
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));

    ASSERT_GE(mFirstPagePerPrepare.size(), 3U);
    EXPECT_NE(mFirstPagePerPrepare.back(), 0) << "logical compaction must preserve the old slot-1 physical page";

    size_t const beforeSecondRequest = mFirstPagePerPrepare.size();
    auto const secondRequest = makeGreedyRequest("a", /*maxGenerateLength=*/2);
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    ASSERT_GT(mFirstPagePerPrepare.size(), beforeSecondRequest);
    EXPECT_EQ(mFirstPagePerPrepare[beforeSecondRequest], 0)
        << "a new unmanaged request must restore the base page table to identity";
}

// --------------------------------------------------------------------------
// MTP speculative decoding: the same assembly with a second engine.
// --------------------------------------------------------------------------

//! Linear-chain MTP: the draft walks a chain of `kDraftingStep` tokens and the base verifies the root plus that
//! chain. `createDeploymentConfig` requires verifySize == draftingStep + 1 for the chain (topK == 1) mode.
constexpr int32_t kDraftingTopK{1};
constexpr int32_t kDraftingStep{3};
constexpr int32_t kVerifySize{kDraftingStep + 1};

//! MTP CUDA-graph capture simulates a 128-token resident prefix plus proposal headroom, so this fixture needs more
//! capacity than the tiny vanilla assembly configuration.
constexpr int64_t kMtpKvCacheCapacity{256};

//! Speculative engines have no cross-request page retention, so `requireMinimumActiveKVPool` demands the pool be
//! exactly the active working set. Derive it rather than hardcode, so a change to the page size stays consistent.
void sizeSpeculativeKvPool(Json& config)
{
    // eagleBaseCommitKVCacheAndAssembleHiddenState, which MTP reuses for accept and KV commit, is specialized for
    // HEAD_DIM in {64, 128, 256, 512}. The vanilla deployment's 16 would fail at the first verify round.
    config["head_dim"] = 64;
    config["hidden_size"] = 128;
    config["builder_config"]["max_kv_cache_capacity"] = kMtpKvCacheCapacity;
    config["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kMaxBatchSize, kMtpKvCacheCapacity);
}

Json makeMtpBaseConfig()
{
    Json config = makeTinyVanillaConfig();
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "base";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_verify_tree_size"] = kVerifySize;
    sizeSpeculativeKvPool(config);
    return config;
}

Json makeMtpDraftConfig()
{
    Json config = makeTinyVanillaConfig();
    config["num_hidden_layers"] = 1;
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "draft";
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    config["builder_config"].erase("spec_base");
    config["builder_config"]["max_draft_tree_size"] = kVerifySize;
    sizeSpeculativeKvPool(config);
    return config;
}

std::filesystem::path writeMtpModelDir()
{
    auto const dir = freshModelDir("edgellm_runtime_mtp_assembly_test");
    std::ofstream(dir / "base_config.json") << makeMtpBaseConfig().dump(2);
    std::ofstream(dir / "draft_config.json") << makeMtpDraftConfig().dump(2);
    writeTokenizerFiles(dir);
    return dir;
}

rt::SpecDecodeDraftingConfig makeMtpDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = kDraftingTopK;
    drafting.draftingStep = kDraftingStep;
    drafting.verifySize = kVerifySize;
    return drafting;
}

//! Assemble MTP artifacts around two caller-supplied engines.
rt::ModelArtifacts makeMtpArtifacts(std::filesystem::path const& modelDir,
    std::unique_ptr<rt::EngineExecutor> baseEngine, std::unique_ptr<rt::EngineExecutor> draftEngine,
    cudaStream_t stream)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment
        = rt::createDeploymentConfig(modelDir / "base_config.json", modelDir / "draft_config.json", makeMtpDrafting());
    artifacts.baseExecutor = std::move(baseEngine);
    artifacts.draftExecutor = std::move(draftEngine);

    artifacts.weights.load(modelDir, modelDir / "base_config.json", stream);
    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    artifacts.draftWeights.load(modelDir, modelDir / "draft_config.json", stream);
    artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, "draft");

    artifacts.embedding.table = rt::Tensor({artifacts.deployment.base.vocabSize, artifacts.deployment.base.hiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "test::embedding");
    CUDA_CHECK(cudaMemsetAsync(artifacts.embedding.table.rawPointer(), 0,
        static_cast<size_t>(artifacts.embedding.table.getShape().volume()) * sizeof(half), stream));

    artifacts.tokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!artifacts.tokenizer->loadFromHF(modelDir.string()))
    {
        throw std::runtime_error("test tokenizer failed to load");
    }
    return artifacts;
}

class MtpAssemblyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeMtpModelDir();
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    //! Same stubs as the vanilla fixture. Both engines bind `logits` to the one PipelineIO buffer, so whichever
    //! engine ran last is the one whose logits the sampler reads.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeEngine()
    {
        using ::testing::_;
        using ::testing::Return;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(nvinfer1::DataType::kHALF));
        ON_CALL(*engine, captureGraph(_)).WillByDefault(Return(false));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                return true;
            });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, {}, stream);
            return true;
        });
        return engine;
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    rt::Tensor* mLogits{nullptr};
};

TEST_F(MtpAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOnDisk)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeEngine(), makeEngine(), mStream);

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, makeMtpDrafting(), mStream};

    ASSERT_FALSE(std::filesystem::exists(mModelDir / "spec_base.engine"));
    ASSERT_FALSE(std::filesystem::exists(mModelDir / "spec_draft.engine"));
    EXPECT_TRUE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "mtp");
}

TEST_F(MtpAssemblyTest, RunsTheDraftChainThenOneBaseVerificationPerRound)
{
    using ::testing::_;
    using ::testing::AllOf;
    using ::testing::Each;
    using ::testing::Field;
    using ::testing::InSequence;
    using ::testing::SizeIs;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;

    // Ordered on the forwards, because that is where MTP's data dependency lies: the draft head consumes the target's
    // hidden state, so it cannot propose until the base has run, and the base cannot verify until the chain exists.
    // A round is therefore one base forward, `kDraftingStep` draft forwards, and one more base forward that verifies
    // all of them at once. Two target forwards for `kVerifySize` tokens is the speculative bargain.
    {
        InSequence forwards;
        EXPECT_CALL(base, execute(_));
        EXPECT_CALL(draft, execute(_)).Times(kDraftingStep);
        EXPECT_CALL(base, execute(_));
    }

    // Unordered on purpose. `prepare()` is binding setup, and nothing in the algorithm says when it has to happen
    // relative to the other engine's; hoisting or batching it is a refactor, not a behavior change. What each call
    // must carry is the profile it selects and the shapes it asks for.
    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _));
    EXPECT_CALL(draft, prepare(kPrefillProfile, _, _, _));
    EXPECT_CALL(draft, prepare(kDecodeProfile, _, _, _)).Times(kDraftingStep - 1);
    // The verification forward covers the root plus the whole proposed chain and asks for a logits row per position.
    // That `selectLen` is what separates speculative verification from an ordinary decode step.
    EXPECT_CALL(base,
        prepare(kDecodeProfile,
            AllOf(Field(&rt::InferenceDims::seqLen, kVerifySize), Field(&rt::InferenceDims::selectLen, kVerifySize)), _,
            _));

    auto artifacts = makeMtpArtifacts(mModelDir, std::move(baseEngine), std::move(draftEngine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, makeMtpDrafting(), mStream};

    // One round proposes `kDraftingStep` tokens on top of the root, so this length is reached without a second round.
    auto const request = makeGreedyRequest("a", kVerifySize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    // A speculative round returns the whole accepted chain at once, so the length cap is met exactly rather than
    // overshot, and the tokens are still the ones the substitute engines produced.
    EXPECT_THAT(response.outputIds[0], AllOf(SizeIs(kVerifySize), Each(kZeroLogitsToken)));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

} // namespace
