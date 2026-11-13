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

// Scaffolding for driving LLMInferenceRuntime with a substitute EngineExecutor and no serialized engine anywhere.
//
// The only component replaced is EngineExecutor. Everything else is the production path: the parsed deployment
// config, the KV cache managers, the pipeline tensors, the tensor map, and the decoder registry.
//
// Shared because the runtime suite and the per-decoder suites under cpp/runtime/decoding all assemble the same way
// and differ only in the files they stage and the expectations they state.

#include "runtime/llmInferenceRuntime.h"

#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/pagedKvTypes.h"
#include "common/safetensorsUtils.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/modelArtifacts.h"
#include "runtime/streaming.h"
#include "sampler/sampling.h"
#include "scratchDir.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
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
inline void PrintTo(Tensor const& tensor, std::ostream* os)
{
    *os << "Tensor{" << tensor.getName() << ", shape=" << tensor.getShape().formatString() << "}";
}

inline void PrintTo(FinishReason reason, std::ostream* os)
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

inline void PrintTo(InferenceDims const& dims, std::ostream* os)
{
    *os << "InferenceDims{batch=" << dims.batch << ", seqLen=" << dims.seqLen << ", kvLen=" << dims.kvLen
        << ", selectLen=" << dims.selectLen << "}";
}
} // namespace rt
} // namespace trt_edgellm

//! Everything the substitute-engine suites build on. Wrapped in a namespace rather than left at file scope so a
//! test file that includes this header does not silently acquire trt_edgellm's whole namespace with it.
namespace substitute_engine
{

using namespace trt_edgellm;
using Json = nlohmann::json;

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
    MOCK_METHOD(trt_edgellm::DataType, getBindingDataType, (char const* name), (const, override));
    MOCK_METHOD(nvinfer1::Dims, getProfileShape,
        (char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector), (const, override));
    MOCK_METHOD(nvinfer1::Dims, getTensorShape, (char const* name), (const, override));
    MOCK_METHOD(bool, dedicateCompactContext, (int32_t profileIndex, cudaStream_t stream), (override));
    //! Left without a default action on purpose: it returns a reference gmock cannot invent, so any call aborts the
    //! test. Nothing the runtime does should reach past the interface for the TRT engine.
    MOCK_METHOD(nvinfer1::ICudaEngine const&, getEngine, (), (const, noexcept, override));
};

//! Vocabulary of the tiny test deployment. Named because the tests choose token ids out of it.
constexpr int64_t kVocabSize{128};
constexpr int64_t kMaxBatchSize{2};

//! What a row decodes to when a test does not name a token for it.
//!
//! This used to be produced by leaving the row zeroed, which made all kVocabSize entries tie and left the
//! sampler's tie-break rule deciding the answer. Such a row is a bad fixture twice over: two engines "agreeing"
//! agree only because neither expressed a preference, and any ranking past the top entry is an arbitrary choice
//! among equals that no assertion can check. The row now carries a real distribution peaked here instead, so the
//! decoded token is unchanged and nothing that names this constant moves, while no fixture in these suites holds
//! a tie any more.
constexpr int32_t kDefaultRowToken{static_cast<int32_t>(kVocabSize) - 1};

//! An arbitrary, non-degenerate token to use as a "both sides genuinely agree" fixture value. Unlike
//! kDefaultRowToken (produced by an all-zero row, i.e. a literal tie across the whole vocab that only a
//! sampler-specific tie-break rule resolves), every row using this token has one unambiguous max entry -- real
//! LLM logits are essentially never tied, so a happy-path accept test should not depend on tie-break behavior to
//! reach full acceptance where a genuine argmax agreement will do.
constexpr int32_t kAgreedToken{42};

//! Optimization-profile indices baked into every engine by llmBuilder: 0 is prefill, 1 is decode (and speculative
//! proposal / verification).
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

//! A deployment small enough that every derived allocation stays in the low megabytes.
inline Json makeTinyVanillaConfig()
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
    builder["ragged_backend"] = "entry_padded_compatibility";
    config["builder_config"] = builder;
    return config;
}

//! A hybrid deployment: one attention layer and one recurrent layer, the shape NemotronH-style models take. Its
//! recurrent and convolution state lives in a pool with one row per resident slot.
inline Json makeTinyHybridConfig()
{
    Json config = makeTinyVanillaConfig();
    config["num_attention_layers"] = 1;
    config["num_linear_attn_layers"] = 1;
    config["recurrent_state_num_heads"] = 2;
    config["recurrent_state_head_dim"] = 3;
    config["recurrent_state_size"] = 4;
    config["conv_dim"] = 8;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp16";
    config["conv_state_dtype"] = "fp16";
    return config;
}

//! A prompt of `length` tokens whose content is fixed by `variant`.
//!
//! Two prompts of the same length and different variants differ at every single position (the per-variant shift is
//! coprime with the alphabet), so a sequence that read its neighbour's embeddings, or a cache hit that restored
//! another record's pages, changes the answer instead of only the token counts. A run of one repeated letter
//! cannot show either: it makes every prompt of a given length the same prompt.
inline std::string makePrompt(int64_t length, int32_t variant = 0)
{
    std::string prompt;
    prompt.reserve(static_cast<size_t>(std::max<int64_t>(length, 0)));
    for (int64_t index = 0; index < length; ++index)
    {
        prompt.push_back(static_cast<char>('a' + (index * 7 + variant * 11) % 26));
    }
    return prompt;
}

//! The ids writeTokenizerFiles()'s vocabulary assigns to `prompt`, for a test that has to hand the runtime
//! pre-tokenized input matching a prompt it also passes as text.
inline std::vector<int32_t> promptTokenIds(std::string const& prompt)
{
    std::vector<int32_t> ids;
    ids.reserve(prompt.size());
    for (char const letter : prompt)
    {
        ids.push_back(letter == 'a' ? 0 : letter - 'b' + 3);
    }
    return ids;
}

//! The tokenizer trio every deployment needs. Tiny, but real: the runtime tokenizes and detokenizes for real.
//!
//! Every lowercase letter is its own token, so a fixture can vary a prompt's *content* and not only its length.
//! With a single-letter vocabulary two prompts can differ only by how many tokens they carry, and a sequence that
//! read its neighbour's embeddings while counting its own tokens correctly would decode exactly as if nothing
//! were wrong.
//!
//! "a" keeps id 0 and the two special tokens keep ids 1 and 2, so the remaining letters start at 3: tests that
//! build a continuation prompt by concatenating the literal text of a generated token 0 still work.
inline void writeTokenizerFiles(std::filesystem::path const& dir)
{
    std::string vocab = R"("a": 0, "<eos>": 1, "<bos>": 2)";
    for (char letter = 'b'; letter <= 'z'; ++letter)
    {
        vocab += ", \"" + std::string(1, letter) + "\": " + std::to_string(letter - 'b' + 3);
    }
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {)JSON"
                                          << vocab << R"JSON(}, "merges": []},
  "added_tokens": [
    {"id": 1, "content": "<eos>"},
    {"id": 2, "content": "<bos>"}
  ],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";

    std::ofstream(dir / "tokenizer_config.json")
        << R"JSON({"eos_token": {"content": "<eos>"}, "bos_token": {"content": "<bos>"}})JSON";

    std::ofstream(dir / "chat_template.jinja") << "{% for message in messages %}{{ message.content }}{% endfor %}";
}

//! Stage a single-engine deployment: its config under the name the runtime looks for, plus the tokenizer trio.
//! None of the files written is an engine.
inline std::filesystem::path writeModelDir(std::string const& suite, Json const& config)
{
    auto const dir = makeScratchDir(suite);
    std::ofstream(dir / "config.json") << config.dump(2);
    writeTokenizerFiles(dir);
    return dir;
}

//! The logit a substitute engine gives the token it wants sampled.
//!
//! Small on purpose: at a large peak the winner takes essentially all of the mass and its log-probability rounds
//! to zero. At this value it takes a plausible share, which still decides an argmax outright while leaving each
//! step a log-probability with room to differ from the next.
constexpr float kPeakLogit{2.0F};

//! The logit a substitute engine gives `token` on a row whose argmax is `winner`.
//!
//! Deliberately not one spike over a flat background. Two things go wrong with that shape. Every non-winner
//! carries the same value, so any ranking past the top entry is a tie broken arbitrarily and a test reading one
//! cannot tell a correct order from any other. And a large peak takes essentially all the mass, which rounds the
//! winner's log-probability to zero and erases the difference between a confident step and an uncertain one for
//! anything downstream that accumulates them.
//!
//! The background is a permutation of the vocabulary -- the multiplier is odd and kVocabSize a power of two, so
//! the product modulo it is a bijection -- which gives every token its own value and leaves no two entries tied.
inline float fakeLogit(int64_t token, int32_t winner)
{
    constexpr float kBackgroundBase{-5.0F};
    constexpr float kBackgroundStep{0.03125F};
    if (token == static_cast<int64_t>(winner))
    {
        return kPeakLogit;
    }
    return kBackgroundBase + static_cast<float>((token * 37) % kVocabSize) * kBackgroundStep;
}

//! Whether an engine's output binding carries raw logits or log-probabilities.
//!
//! The two are not interchangeable and the runtime relies on the difference. Every draft model applies
//! `F.log_softmax` as its last operator before returning (see `modeling_qwen3_5_mtp.py`,
//! `modeling_nemotron_h_mtp.py`, `modeling_eagle3_draft.py`), so the tensor a draft engine writes into the
//! binding named "logits" already holds log-probabilities. Base models do not.
//!
//! Speculative decoding depends on it: a drafting step's values are accumulated down the chain, and only a sum
//! of log-probabilities is log P(path). Their non-positivity is also what keeps a node's score at or below its
//! parent's, which is the topological order `constructVerificationDraftTree` needs when it searches only
//! backwards for a node's ancestors. A substitute that wrote raw logits into a draft's binding would collapse
//! the verify tree -- and would do so silently, through code that is correct against a real engine.
enum class FakeOutputScale
{
    kLogits,           //!< A base engine: raw, unnormalised.
    kLogProbabilities, //!< A draft engine: what `F.log_softmax` leaves behind.
};

//! Normalising constant of the row `fakeLogit()` describes for `winner`.
inline float fakeRowLogSumExp(int64_t vocabSize, int32_t winner)
{
    double sum{0.0};
    for (int64_t token = 0; token < vocabSize; ++token)
    {
        sum += std::exp(static_cast<double>(fakeLogit(token, winner)));
    }
    return static_cast<float>(std::log(sum));
}

//! One entry of the row a substitute engine writes, on the scale its binding is documented to carry.
inline float fakeOutputValue(int64_t token, int32_t winner, int64_t vocabSize, FakeOutputScale scale)
{
    float const logit = fakeLogit(token, winner);
    return scale == FakeOutputScale::kLogits ? logit : logit - fakeRowLogSumExp(vocabSize, winner);
}

//! Write the logits a forward pass would have produced.
//!
//! `tokensPerRow` names the argmax for each engine row; a row that is absent or negative is peaked at
//! kDefaultRowToken instead, so it decodes to the same token every round without the row being a tie.
inline void writeLogits(rt::Tensor& logits, int64_t vocabSize, std::vector<int32_t> const& tokensPerRow,
    cudaStream_t stream, FakeOutputScale scale = FakeOutputScale::kLogits)
{
    auto const rows = logits.getShape().volume() / vocabSize;
    std::vector<float> host(static_cast<size_t>(logits.getShape().volume()), 0.0F);
    for (int64_t row = 0; row < rows; ++row)
    {
        int32_t const named
            = row < static_cast<int64_t>(tokensPerRow.size()) ? tokensPerRow[static_cast<size_t>(row)] : -1;
        int32_t const winner = named >= 0 ? named : kDefaultRowToken;
        for (int64_t token = 0; token < vocabSize; ++token)
        {
            host[static_cast<size_t>(row * vocabSize + token)] = fakeOutputValue(token, winner, vocabSize, scale);
        }
    }
    CUDA_CHECK(
        cudaMemcpyAsync(logits.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

//! One row of an EAGLE candidate-head draft output: the `count` ids with the highest logits, ordered by (logit desc, id
//! asc), and log_softmax over those `count` logits only. That is what the in-engine head writes: full-vocabulary ids,
//! and log-probabilities normalised over its candidates rather than over the vocabulary. The logits are fakeLogit()'s
//! for `winner`, except that the winner's is `peakLogit`: a lower peak gives a less confident row. A peak above the
//! highest background logit (-1.03125) keeps the winner first.
struct FakeCandidateRow
{
    std::vector<int32_t> ids;
    std::vector<float> logProbs;
};

inline FakeCandidateRow fakeCandidateRow(int32_t winner, int32_t count, float peakLogit = kPeakLogit)
{
    auto const logit
        = [winner, peakLogit](int32_t token) { return token == winner ? peakLogit : fakeLogit(token, winner); };
    std::vector<int32_t> order(static_cast<size_t>(kVocabSize));
    for (int32_t token = 0; token < static_cast<int32_t>(kVocabSize); ++token)
    {
        order[static_cast<size_t>(token)] = token;
    }
    std::stable_sort(
        order.begin(), order.end(), [&logit](int32_t lhs, int32_t rhs) { return logit(lhs) > logit(rhs); });
    order.resize(static_cast<size_t>(count));

    double sum{0.0};
    for (int32_t const token : order)
    {
        sum += std::exp(static_cast<double>(logit(token)));
    }
    float const logSumExp = static_cast<float>(std::log(sum));
    FakeCandidateRow row{order, std::vector<float>(order.size())};
    for (size_t i = 0; i < order.size(); ++i)
    {
        row.logProbs[i] = logit(order[i]) - logSumExp;
    }
    return row;
}

//! State of the candidate-writing draft substitute: the outputs it was bound to on its last prepare(), and the
//! leading id and its logit it writes per row on each forward.
struct FakeCandidateHead
{
    int32_t candidateCount{0};
    //! `winnersPerForward[forward][row]` and `peaksPerForward[forward][row]`; the last entry repeats. A missing or
    //! negative winner is kDefaultRowToken, and a missing peak is kPeakLogit.
    std::vector<std::vector<int32_t>> winnersPerForward;
    std::vector<std::vector<float>> peaksPerForward;
    rt::Tensor* supportIds{nullptr};
    rt::Tensor* supportLogProbs{nullptr};
    rt::InferenceDims preparedDims{};
    size_t forwards{0};

    //! Takes the bindings the runtime handed to prepare().
    void prepare(rt::InferenceDims const& dims, rt::TensorMap const& map)
    {
        supportIds = map.get(binding_names::kSpecProposalSupportIds);
        supportLogProbs = map.get(binding_names::kSpecProposalSupportLogProbs);
        preparedDims = dims;
    }

    //! Whether the runtime bound both support outputs as [selected rows, candidateCount], as the engine's are; reports
    //! a test failure when not.
    bool boundAsEngine() const
    {
        if (supportIds == nullptr || supportLogProbs == nullptr)
        {
            ADD_FAILURE() << "candidate draft substitute ran without bound support outputs";
            return false;
        }
        for (rt::Tensor const* const support : {supportIds, supportLogProbs})
        {
            auto const& shape = support->getShape();
            if (shape.getNumDims() != 2 || shape[0] != preparedDims.selectLen || shape[1] != candidateCount)
            {
                ADD_FAILURE() << "candidate draft output " << support->getName() << " is bound as "
                              << shape.formatString() << " for " << preparedDims.selectLen << " selected rows of "
                              << candidateCount;
                return false;
            }
        }
        return true;
    }

    //! One forward: writes `preparedDims.selectLen` candidate rows, or reports a failed forward when !boundAsEngine().
    bool write(cudaStream_t stream)
    {
        if (!boundAsEngine())
        {
            return false;
        }
        int64_t const rows = preparedDims.selectLen;
        size_t const elements = static_cast<size_t>(rows * candidateCount);
        std::vector<int32_t> ids(elements);
        std::vector<float> logProbs(elements);
        for (int64_t row = 0; row < rows; ++row)
        {
            int32_t const winner = valueFor(winnersPerForward, row, -1);
            float const peak = valueFor(peaksPerForward, row, kPeakLogit);
            auto const candidates = fakeCandidateRow(winner >= 0 ? winner : kDefaultRowToken, candidateCount, peak);
            std::copy(candidates.ids.begin(), candidates.ids.end(), ids.begin() + row * candidateCount);
            std::copy(candidates.logProbs.begin(), candidates.logProbs.end(), logProbs.begin() + row * candidateCount);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            supportIds->rawPointer(), ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(supportLogProbs->rawPointer(), logProbs.data(), logProbs.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++forwards;
        return true;
    }

private:
    template <typename T>
    T valueFor(std::vector<std::vector<T>> const& perForward, int64_t row, T fallback) const
    {
        if (perForward.empty())
        {
            return fallback;
        }
        auto const& rows = perForward[std::min(forwards, perForward.size() - 1)];
        return row < static_cast<int64_t>(rows.size()) ? rows[static_cast<size_t>(row)] : fallback;
    }
};

//! `tokensPerRow` for writeLogits() that broadcasts the same unambiguous winner to every row, regardless of how
//! many rows an individual forward pass actually has (writeLogits ignores entries past the real row count). Use
//! this in place of `{}` wherever a fixture wants every position to agree without relying on the all-zero
//! literal-tie behavior kDefaultRowToken depends on.
inline std::vector<int32_t> agreedLogitsRows(int32_t token, size_t maxRows = 8)
{
    return std::vector<int32_t>(maxRows, token);
}

//! handleRequest documents that on success it repopulates the per-slot response vectors together, to matched
//! sizes. Checking that as a block keeps every test from having to re-derive it.
inline void expectResponseCoversEverySlot(rt::LLMGenerationResponse const& response, size_t slots)
{
    EXPECT_EQ(response.outputIds.size(), slots);
    EXPECT_EQ(response.outputTexts.size(), slots);
    EXPECT_EQ(response.finishReasons.size(), slots);
    EXPECT_EQ(response.inputTokenCounts.size(), slots);
}

//! Verification rows for a batch of two where only slot 0 stops.
//!
//! The base writes [slot, verifyPosition] flattened, so slot 1 starts at `verifyTokenCount`. -1 leaves a row zeroed,
//! which decodes to the tie-break token, so the second slot keeps generating.
inline std::vector<int32_t> eosOnFirstSlotOnly(int32_t eosId, int32_t verifyTokenCount)
{
    std::vector<int32_t> rows(2U * static_cast<size_t>(verifyTokenCount), -1);
    std::fill(rows.begin(), rows.begin() + verifyTokenCount, eosId);
    return rows;
}

//! Modulus for every integer a substitute engine carries, in the embedding table and in the state it derives from
//! it. Prime, so any non-zero multiplier below it is invertible and a per-channel stride stays injective; below
//! 256, so a value survives a round trip through every KV dtype these substitutes support -- bf16 keeps only 8
//! mantissa bits and stops representing consecutive integers there, which would have a recomputed quantity compare
//! nearly-but-not-equal to a stored one.
constexpr int32_t kFakeModulus{251};

//! The first token id the substitute engines will ever produce on their own. Ids below it are reserved by the test
//! tokenizer (0 is "a", 1 is "<eos>", 2 is "<bos>"), and a computed token that happened to land on the EOS id would
//! end generation early in a test that was not asking about EOS at all -- deterministically, but only for some
//! prompt lengths. A test that wants EOS forces it.
constexpr int32_t kFirstComputedToken{3};

//! The embedding table entry for `tokenId`'s `channel`-th element.
//!
//! Distinct tokens must produce distinct embeddings, or an engine that derives its output from its input produces
//! the same output for every prompt and every test asserting two runs agree passes vacuously. Each channel applies
//! its own invertible stride, so reading the right value from the wrong channel is distinguishable too.
inline int32_t fakeEmbeddingValue(int64_t tokenId, int64_t channel)
{
    constexpr int32_t kChannelStrides[]{3, 29, 113, 197};
    auto const stride = static_cast<int64_t>(kChannelStrides[channel % 4]);
    return static_cast<int32_t>(((tokenId + 1) * stride + channel) % kFakeModulus);
}

//! The parts of an assembled deployment that do not vary with the decoding mode: an embedding table sized from the
//! parsed config, and the tokenizer staged alongside it.
inline void finishArtifacts(rt::ModelArtifacts& artifacts, std::filesystem::path const& modelDir, cudaStream_t stream)
{
    auto const vocabSize = artifacts.deployment.base.vocabSize;
    auto const hiddenSize = artifacts.deployment.base.hiddenSize;
    artifacts.embedding.table
        = rt::Tensor({vocabSize, hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "test::embedding");

    std::vector<half> table(static_cast<size_t>(vocabSize * hiddenSize));
    for (int64_t token = 0; token < vocabSize; ++token)
    {
        for (int64_t channel = 0; channel < hiddenSize; ++channel)
        {
            table[static_cast<size_t>(token * hiddenSize + channel)]
                = __float2half(static_cast<float>(fakeEmbeddingValue(token, channel)));
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(artifacts.embedding.table.rawPointer(), table.data(), table.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    artifacts.tokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!artifacts.tokenizer->loadFromHF(modelDir.string()))
    {
        throw std::runtime_error("test tokenizer failed to load");
    }
}

//! Assemble the artifacts a single-engine text deployment needs around a caller-supplied engine.
inline rt::ModelArtifacts makeVanillaArtifacts(
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

    finishArtifacts(artifacts, modelDir, stream);
    return artifacts;
}

//! A greedy batch, one slot per prompt in the order given. Greedy keeps the sampled token a function of the logits
//! the substitute engine wrote.
//!
//! The sampling knobs live on the batch, not on the individual requests, so they are stated once here. Building a
//! multi-slot batch by merging the `requests` vectors of several single-prompt calls would silently discard every
//! call's batch-level settings but the first.
//!
//! Spelled differently from the single-prompt `makeGreedyRequest` rather than overloading it: `{"a", "aa"}` is
//! ambiguous between `std::vector<std::string>` and `std::string`'s iterator-pair constructor, which a two-slot
//! batch -- the common case here -- would hit every time.
inline rt::LLMGenerationRequest makeGreedyBatch(std::vector<std::string> const& prompts, int64_t maxGenerateLength)
{
    rt::LLMGenerationRequest request{};
    for (auto const& prompt : prompts)
    {
        rt::LLMGenerationRequest::Request one;
        one.messages.push_back(rt::Message{"user", {rt::Message::MessageContent{"text", prompt}}});
        request.requests.push_back(std::move(one));
    }
    request.temperature = 0.0F;
    request.topK = 1;
    request.topP = 1.0F;
    request.maxGenerateLength = maxGenerateLength;
    return request;
}

//! A single-slot greedy request.
inline rt::LLMGenerationRequest makeGreedyRequest(std::string const& prompt, int64_t maxGenerateLength)
{
    return makeGreedyBatch({prompt}, maxGenerateLength);
}

//! Scaffolding every suite below needs: a stream, the mock engine, and the logits binding it writes through.
//!
//! Each deployment differs only in the files it stages and the artifacts it assembles, so those stay in the
//! derived fixtures and this holds what is genuinely identical.
class SubstituteEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! The prepare() action every substitute engine in these suites installs. The output scale is the only thing
    //! that varies between them, so it is a parameter rather than a second copy of the body: a substitute whose
    //! recording policy differs from its sibling's makes two engines in one deployment disagree about what a
    //! prepare() even counts as.
    //!
    //! Both bindings it reads are reachable only through the tensor map the runtime hands to prepare().
    auto prepareAction(FakeOutputScale scale)
    {
        return [this, scale](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t stream) {
            mLogits = map.get(binding_names::kLogits);
            mLogitsScale = scale;
            rt::Tensor* const pageTable = map.get(binding_names::kKVPageTable);
            if (pageTable == nullptr)
            {
                // A precondition of every deployment these suites assemble, not a model variant to tolerate:
                // buildTensorMap() binds the table unconditionally, outside the per-layer walk, so even a
                // deployment with no attention layer carries it. Reporting the failed prepare() unwinds the
                // generation loop, which stops on the first strategy step that returns false.
                ADD_FAILURE() << "substitute prepare(): deployment bound no " << binding_names::kKVPageTable;
                return false;
            }
            int32_t firstPage{};
            CUDA_CHECK(cudaMemcpyAsync(
                &firstPage, pageTable->rawPointer(), sizeof(firstPage), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            mFirstPagePerPrepare.push_back(firstPage);
            return true;
        };
    }

    //! A mock engine with the assembly-time queries answered, so each test only states the calls it cares about.
    //!
    //! NiceMock, because assembly asks the engine a handful of questions that no test is about; leaving those to
    //! warn would bury the expectations that matter. Calls a test does declare are still checked strictly.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeEngine()
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::StrEq;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(kContextMemoryBytes));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(trt_edgellm::DataType::kHALF));
        ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kLogits))).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kContextSequenceCountCarrier))).WillByDefault(Return(true));
        ON_CALL(*engine, getBindingDataType(StrEq(binding_names::kContextSequenceCountCarrier)))
            .WillByDefault(Return(trt_edgellm::DataType::kINT32));
        ON_CALL(*engine, captureGraph(_)).WillByDefault(Return(false));
        ON_CALL(*engine, prepare(_, _, _, _)).WillByDefault(prepareAction(FakeOutputScale::kLogits));
        // A forward that decodes to the tie-break token everywhere. Tests that care override it per call.
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, {}, stream, mLogitsScale);
            return true;
        });
        return engine;
    }

    //! `makeEngine()`, then marked as standing in for a draft model: its output binding carries log-probabilities
    //! (see FakeOutputScale). Only for decoders whose draft model actually ends in `F.log_softmax` -- EAGLE and
    //! MTP do, and their drafting steps accumulate the values, so the scale decides whether the verify tree comes
    //! out in topological order.
    //!
    //! The scale is recorded at prepare() rather than held per engine because these fixtures write through one
    //! logits pointer: the engine that just prepared is the one about to execute.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeLogProbDraftEngine()
    {
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(prepareAction(FakeOutputScale::kLogProbabilities));
        return engine;
    }

    //! A mock engine standing in for an EAGLE draft whose head is the in-engine candidate head: no `logits` output,
    //! and per selected row `candidateCount` full-vocabulary ids sorted by logit with log-probabilities normalised over
    //! those candidates (FakeCandidateRow). `winnersPerForward[forward][row]` names each row's leading candidate and
    //! `peaksPerForward[forward][row]` its logit; the last entry repeats. Its bindings answer startup validation the
    //! way a correctly built engine does. State lives in mCandidateHead.
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeCandidateDraftEngine(int32_t candidateCount,
        std::vector<std::vector<int32_t>> winnersPerForward = {}, std::vector<std::vector<float>> peaksPerForward = {})
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::StrEq;

        mCandidateHead = FakeCandidateHead{};
        mCandidateHead.candidateCount = candidateCount;
        mCandidateHead.winnersPerForward = std::move(winnersPerForward);
        mCandidateHead.peaksPerForward = std::move(peaksPerForward);

        auto engine = makeEngine();
        ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kLogits))).WillByDefault(Return(false));
        nvinfer1::Dims supportShape{};
        supportShape.nbDims = 2;
        supportShape.d[0] = -1;
        supportShape.d[1] = candidateCount;
        for (auto const* name : {binding_names::kSpecProposalSupportIds, binding_names::kSpecProposalSupportLogProbs})
        {
            ON_CALL(*engine, hasIOTensor(StrEq(name))).WillByDefault(Return(true));
            ON_CALL(*engine, getTensorShape(StrEq(name))).WillByDefault(Return(supportShape));
        }
        ON_CALL(*engine, getBindingDataType(StrEq(binding_names::kSpecProposalSupportIds)))
            .WillByDefault(Return(trt_edgellm::DataType::kINT32));
        ON_CALL(*engine, getBindingDataType(StrEq(binding_names::kSpecProposalSupportLogProbs)))
            .WillByDefault(Return(trt_edgellm::DataType::kFLOAT));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault([this](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mCandidateHead.prepare(dims, map);
                return true;
            });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            return mCandidateHead.write(stream);
        });
        return engine;
    }

    //! An `execute()` action standing in for one forward pass. `tokensPerRow` names the argmax for each engine row;
    //! rows left out decode to whatever the sampler's tie-break picks out of the zeroed logits, which is stable.
    //! For a speculative verification the rows are [slot, verifyPosition] flattened, so a batch of two with
    //! verifyTokenCount 4 has slot 1 starting at row 4.
    auto emit(std::vector<int32_t> tokensPerRow)
    {
        return [this, tokensPerRow = std::move(tokensPerRow)](cudaStream_t stream) {
            writeLogits(*mLogits, kVocabSize, tokensPerRow, stream, mLogitsScale);
            return true;
        };
    }

    //! Read the first `count` entries of an int32 binding out of the tensor map a prepare() was handed. Returns
    //! empty when the deployment does not bind that name at all.
    static std::vector<int32_t> readInt32Binding(
        rt::TensorMap const& map, char const* name, int32_t count, cudaStream_t stream)
    {
        auto* const tensor = map.get(name);
        if (tensor == nullptr || count <= 0 || tensor->getShape().volume() < count)
        {
            return {};
        }
        std::vector<int32_t> values(static_cast<size_t>(count));
        CUDA_CHECK(cudaMemcpyAsync(
            values.data(), tensor->rawPointer(), values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return values;
    }

    static constexpr int64_t kContextMemoryBytes{4096};

    cudaStream_t mStream{};
    //! Captured by `makeEngine`'s prepare() default; valid from the first prepare() until the runtime dies.
    rt::Tensor* mLogits{nullptr};
    FakeOutputScale mLogitsScale{FakeOutputScale::kLogits};
    std::vector<int32_t> mFirstPagePerPrepare;
    FakeCandidateHead mCandidateHead;
};

//! A `SubstituteEngineTest` whose deployment is staged on disk for the duration of one test.
//!
//! Every suite below stages a directory in SetUp and removes it in TearDown; what differs is only the files that go
//! into it, which is what `stageModelDir()` supplies.
class ModelDirTest : public SubstituteEngineTest
{
protected:
    void SetUp() override
    {
        SubstituteEngineTest::SetUp();
        mModelDir = stageModelDir();
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mModelDir);
        SubstituteEngineTest::TearDown();
    }

    virtual std::filesystem::path stageModelDir() = 0;

    //! The runtime under test, over already-assembled artifacts.
    //!
    //! Returned by value: a prvalue is constructed straight into the caller's storage, so this needs nothing of
    //! LLMInferenceRuntime that a test constructing it in place would not.
    rt::LLMInferenceRuntime makeRuntime(
        rt::ModelArtifacts artifacts, std::optional<rt::SpecDecodeDraftingConfig> const& drafting = std::nullopt)
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting;
        return rt::LLMInferenceRuntime{std::move(artifacts), std::move(options), mStream};
    }

    std::filesystem::path mModelDir;
};

//! Assembly must ask the engine how much scratch it needs and hand back a buffer at least that large. Stated as a
//! matcher on the argument, so the expectation reads as the rule rather than as a value captured and compared later.
inline void expectContextMemorySizedFromTheExecutor(MockEngineExecutor& engine, int64_t required)
{
    EXPECT_CALL(engine, getRequiredContextMemorySize());
    EXPECT_CALL(engine,
        setContextMemory(::testing::ResultOf(
            [](rt::Tensor const& memory) { return memory.getShape().volume(); }, ::testing::Ge(required))));
}

// --------------------------------------------------------------------------
// MTP speculative decoding: the same assembly with a second engine.
// --------------------------------------------------------------------------

//! Linear-chain MTP: the draft walks a chain of `kDraftingStep` tokens and the base verifies the root plus that
//! chain. `createDeploymentConfig` requires verifyTokenCount == draftSteps + 1 for the chain (topK == 1) mode.
constexpr int32_t kDraftingTopK{1};
constexpr int32_t kDraftingStep{3};
constexpr int32_t kVerifySize{kDraftingStep + 1};

//! MTP CUDA-graph capture simulates a 128-token resident prefix plus proposal headroom, so this fixture needs more
//! capacity than the tiny vanilla assembly configuration.
constexpr int64_t kMtpKvCacheCapacity{256};

//! Speculative engines have no cross-request page retention, so `requireMinimumActiveKVPool` demands the pool be
//! exactly the active working set. Derive it rather than hardcode, so a change to the page size stays consistent.
inline void sizeSpeculativeKvPool(Json& config)
{
    // commitSpecTargetKV, which MTP reuses for accept and KV commit, is specialized for
    // HEAD_DIM in {64, 128, 256, 512}. The vanilla deployment's 16 would fail at the first verify round.
    config["head_dim"] = 64;
    config["hidden_size"] = 128;
    config["builder_config"]["max_kv_cache_capacity"] = kMtpKvCacheCapacity;
    config["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kMaxBatchSize, kMtpKvCacheCapacity);
}

//! The half of a speculative config that is the same for every mode: which engine of the pair this is, and the
//! sizing the shared accept and KV-commit path requires. The mode-specific fields are added by the callers.
inline Json makeSpecBaseConfig(char const* specDecodeType, int32_t verifyTokenCount)
{
    Json config = makeTinyVanillaConfig();
    config["spec_decode_type"] = specDecodeType;
    config["engine_role"] = "base";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_spec_verify_tokens"] = verifyTokenCount;
    sizeSpeculativeKvPool(config);
    return config;
}

inline Json makeSpecDraftConfig(char const* specDecodeType, int32_t draftTreeSize)
{
    Json config = makeTinyVanillaConfig();
    // One layer, because nothing under test depends on the draft's depth and every mode's draft is the small half.
    config["num_hidden_layers"] = 1;
    config["spec_decode_type"] = specDecodeType;
    config["engine_role"] = "draft";
    config["builder_config"].erase("spec_base");
    config["builder_config"]["max_spec_draft_tokens"] = draftTreeSize;
    sizeSpeculativeKvPool(config);
    return config;
}

//! Stage a two-engine deployment under the names `createDeploymentConfig` reads. No engine and no sidecar; the
//! suites that need one write it themselves.
inline std::filesystem::path writeSpecModelDir(
    std::string const& suite, Json const& baseConfig, Json const& draftConfig)
{
    auto const dir = makeScratchDir(suite);
    std::ofstream(dir / "base_config.json") << baseConfig.dump(2);
    std::ofstream(dir / "draft_config.json") << draftConfig.dump(2);
    writeTokenizerFiles(dir);
    return dir;
}

//! Assemble a two-engine deployment around caller-supplied engines.
//!
//! Shared by every speculative mode: they differ in what their configs declare and in the drafting shape they are
//! given, and in nothing about how the artifacts are put together or validated.
inline rt::ModelArtifacts makeSpecArtifacts(std::filesystem::path const& modelDir,
    rt::SpecDecodeDraftingConfig const& drafting, std::unique_ptr<rt::EngineExecutor> baseEngine,
    std::unique_ptr<rt::EngineExecutor> draftEngine, cudaStream_t stream)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment
        = rt::createDeploymentConfig(modelDir / "base_config.json", modelDir / "draft_config.json", drafting);
    artifacts.baseExecutor = std::move(baseEngine);
    artifacts.draftExecutor = std::move(draftEngine);

    artifacts.weights.load(modelDir, modelDir / "base_config.json", stream);
    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    artifacts.draftWeights.load(modelDir, modelDir / "draft_config.json", stream);
    artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, "draft");

    finishArtifacts(artifacts, modelDir, stream);
    return artifacts;
}

//! Scaffolding shared by the two-engine suites, and the place the expectations they hold in common are stated.
//!
//! The modes below are separate decoders implementing one DecodingStrategy contract, so a per-decoder copy of an
//! expectation they all have to meet would be the same test written five times. Stating it once here and calling it
//! from each suite makes "these decoders agree" structural: where two of them disagree, one is wrong.
class SpecAssemblyTest : public ModelDirTest
{
protected:
    //! The drafting shape this mode declares. `createDeploymentConfig` cross-checks it against both configs, so it
    //! has to be the same value the suite staged its files for.
    virtual rt::SpecDecodeDraftingConfig drafting() const = 0;

    //! Whether this mode's draft model ends in `F.log_softmax`, which decides the scale its output binding
    //! carries (see FakeOutputScale). True for EAGLE and MTP (`modeling_eagle3_draft.py`,
    //! `modeling_nemotron_h_mtp.py`, `modeling_qwen3_5_mtp.py`); false for the rest, whose draft models return
    //! their head's output unnormalised.
    virtual bool draftEmitsLogProbabilities() const
    {
        return false;
    }

    rt::ModelArtifacts makeArtifacts(
        std::unique_ptr<rt::EngineExecutor> baseEngine, std::unique_ptr<rt::EngineExecutor> draftEngine)
    {
        return makeSpecArtifacts(mModelDir, drafting(), std::move(baseEngine), std::move(draftEngine), mStream);
    }

    rt::ModelArtifacts makeArtifacts()
    {
        return makeArtifacts(makeEngine(), draftEmitsLogProbabilities() ? makeLogProbDraftEngine() : makeEngine());
    }

    rt::LLMInferenceRuntime makeRuntime(rt::ModelArtifacts artifacts)
    {
        return ModelDirTest::makeRuntime(std::move(artifacts), drafting());
    }

    //! A draft engine that records the active sequence-to-resident-slot mapping for each prepare().
    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeRecordingDraftEngine()
    {
        auto engine = makeEngine();
        ON_CALL(*engine, prepare(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(
                [this](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) {
                    mLogits = map.get(binding_names::kLogits);
                    mLogitsScale = FakeOutputScale::kLogProbabilities;
                    auto stateIndices = readInt32Binding(map, binding_names::kStateIndices, dims.batch, stream);
                    if (!stateIndices.empty())
                    {
                        mDraftStateIndices.push_back(std::move(stateIndices));
                    }
                    return true;
                });
        return engine;
    }

    //! Verify logical batch compaction preserves the survivor's resident state row.
    void expectDraftStateFollowedTheSurvivor()
    {
        size_t compacted = 0;
        while (compacted < mDraftStateIndices.size() && mDraftStateIndices[compacted].size() != 1U)
        {
            ++compacted;
        }
        ASSERT_LT(compacted, mDraftStateIndices.size()) << "the draft engine never ran on the compacted batch";
        ASSERT_GT(compacted, 0U) << "the draft engine never ran on the full batch";

        auto const& paired = mDraftStateIndices[compacted - 1];
        ASSERT_EQ(paired.size(), 2U);
        ASSERT_NE(paired[0], paired[1]);
        EXPECT_EQ(mDraftStateIndices[compacted][0], paired[1])
            << "logical compaction changed the survivor's resident state row";
    }

    static constexpr char const* kShortPrompt{"a"};
    static constexpr char const* kLongPrompt{"aaaaaaaaaaaaaaaa"};

    std::vector<std::vector<int32_t>> mDraftStateIndices;

    //! Nothing on disk is an engine, and the pair still assembles into the strategy the base config names.
    //!
    //! Given a two-engine deployment whose directory holds no serialized engine
    //! When the runtime is assembled from the artifacts alone
    //! Then it reports a draft model and the strategy its base config names
    void expectAssembledFromArtifactsAlone(char const* strategyName)
    {
        auto runtime = makeRuntime(makeArtifacts());

        ASSERT_FALSE(std::filesystem::exists(mModelDir / "spec_base.engine"));
        ASSERT_FALSE(std::filesystem::exists(mModelDir / "spec_draft.engine"));
        EXPECT_TRUE(runtime.hasDraftModel());
        EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), strategyName);
    }

    //! One round of a chain-proposing decoder: one base forward, `draftSteps` draft forwards walking the chain,
    //! and one more base forward that verifies the root plus the whole chain at once. Two target forwards for
    //! `verifyTokenCount` tokens is the speculative bargain.
    //!
    //! Ordered on the forwards, because that is where the data dependency lies: the draft head consumes the target's
    //! hidden state, so it cannot propose until the base has run, and the base cannot verify until the chain exists.
    //!
    //! Given a chain-proposing speculative deployment
    //! When one round generates verifyTokenCount tokens
    //! Then the base runs once, the draft walks draftSteps proposals, and one base verification covers them all
    void expectChainProposalThenOneVerification()
    {
        using ::testing::_;
        using ::testing::AllOf;
        using ::testing::Each;
        using ::testing::Field;
        using ::testing::SizeIs;

        auto baseEngine = makeEngine();
        auto draftEngine = makeLogProbDraftEngine();
        auto& base = *baseEngine;
        auto& draft = *draftEngine;

        {
            ::testing::InSequence forwards;
            EXPECT_CALL(base, execute(_));
            EXPECT_CALL(draft, execute(_)).Times(kDraftingStep);
            EXPECT_CALL(base, execute(_));
        }

        // Unordered on purpose. `prepare()` is binding setup, and nothing in the algorithm says when it has to
        // happen relative to the other engine's; hoisting or batching it is a refactor, not a behavior change. What
        // each call must carry is the profile it selects and the shapes it asks for.
        EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _));
        EXPECT_CALL(draft, prepare(kPrefillProfile, _, _, _));
        EXPECT_CALL(draft, prepare(kDecodeProfile, _, _, _)).Times(kDraftingStep - 1);
        // The verification forward covers the root plus the whole proposed chain and asks for a logits row per
        // position. That `selectLen` is what separates speculative verification from an ordinary decode step.
        EXPECT_CALL(base,
            prepare(kDecodeProfile,
                AllOf(
                    Field(&rt::InferenceDims::seqLen, kVerifySize), Field(&rt::InferenceDims::selectLen, kVerifySize)),
                _, _));

        auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

        // One round proposes `kDraftingStep` tokens on top of the root, so this length is reached in a single round.
        auto const request = makeGreedyRequest("a", kVerifySize);
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        expectResponseCoversEverySlot(response, 1);
        ASSERT_EQ(response.outputIds.size(), 1U);
        // A speculative round returns the whole accepted chain at once, so the length cap is met exactly rather than
        // overshot, and the tokens are still the ones the substitute engines produced.
        EXPECT_THAT(response.outputIds[0], AllOf(SizeIs(kVerifySize), Each(kDefaultRowToken)));
        EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
    }

    //! Slot 0 stops on the first verification while slot 1 keeps going, so the survivor moves to engine row 0 and
    //! every per-slot structure the decoder owns has to move with it -- draft cache, host lengths, and whatever
    //! state the mode carries on top.
    //!
    //! Asserted through the runtime rather than end to end because a mismatch here is invisible in the text: the
    //! base re-verifies everything, so a stale draft slot leaves the output valid and only collapses the acceptance
    //! rate. `verifyWindow` is how many rows one slot occupies in the verification logits.
    //!
    //! Given a batch of two, holding different-length prompts, whose first slot stops on the first verification
    //! When the remaining rounds run
    //! Then the survivor keeps its resident state row while occupying active row 0
    void expectSurvivorKeepsDecodingAfterCompaction(int32_t verifyWindow)
    {
        using ::testing::_;

        int64_t const maxGenerateLength{5 * verifyWindow};

        auto baseEngine = makeEngine();
        auto& base = *baseEngine;

        auto artifacts = makeArtifacts(std::move(baseEngine), makeRecordingDraftEngine());
        auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

        {
            ::testing::InSequence seq;
            EXPECT_CALL(base, execute(_)).Times(2).WillRepeatedly(emit({}));
            EXPECT_CALL(base, execute(_)).WillOnce(emit(eosOnFirstSlotOnly(eosId, verifyWindow)));
            EXPECT_CALL(base, execute(_)).Times(::testing::AnyNumber()).WillRepeatedly(emit({}));
        }

        auto runtime = makeRuntime(std::move(artifacts));

        auto const request = makeGreedyBatch({kShortPrompt, kLongPrompt}, maxGenerateLength);
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        expectResponseCoversEverySlot(response, 2);
        ASSERT_EQ(response.outputIds.size(), 2U);
        EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
        EXPECT_EQ(response.outputIds[0].back(), eosId);
        // The survivor keeps decoding after the compaction, all the way to the length cap.
        EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kLength);
        EXPECT_EQ(response.outputIds[1].size(), static_cast<size_t>(maxGenerateLength));

        expectDraftStateFollowedTheSurvivor();
    }
};

inline rt::SpecDecodeDraftingConfig makeMtpDrafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = kDraftingTopK;
    drafting.draftingStep = kDraftingStep;
    drafting.verifyTokens = kVerifySize;
    return drafting;
}

} // namespace substitute_engine
