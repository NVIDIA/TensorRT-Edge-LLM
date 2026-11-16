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

//! N-gram history across a speculative verify step: hashing a whole chain, then re-seeding from the
//! accepted tokens, must give the row IDs that decoding those tokens one at a time gives.

#include "runtime/preprocess/qwen4ExpNgramPreprocessor.h"

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "scratchDir.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kNgramSize{3};
constexpr int32_t kHeadsPerOrder{2};
constexpr int32_t kHeads{(kNgramSize - 1) * kHeadsPerOrder};
constexpr int64_t kRowWidth{8};
constexpr int64_t kRowCount{420};
constexpr int32_t kMaxBatchSize{2};
constexpr int32_t kMaxSeqLen{16};
constexpr int32_t kEosTokenId{2};
constexpr int32_t kSlot{1};

//! Large moduli, so a wrong history shows up as different row IDs rather than a collision.
constexpr std::array<int64_t, kHeads> kHeadVocabSizes{101, 103, 107, 109};
constexpr std::array<int64_t, kHeads> kHeadOffsets{0, 101, 204, 311};
constexpr std::array<int64_t, kNgramSize> kLayerMultipliers{
    0x2545F4914F6CDD1DLL, 0x5851F42D4C957F2DLL, 0x14057B7EF767814FLL};

class Qwen4ExpNgramHistoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mDirectory = makeScratchDir("qwen4ExpNgramHistory");

        std::vector<Tensor> tensors;
        Tensor table({kRowCount, kRowWidth}, DeviceType::kCPU, trt_edgellm::DataType::kHALF, kTableKey);
        auto* rows = table.dataPointer<half>();
        for (int64_t index = 0; index < kRowCount * kRowWidth; ++index)
        {
            rows[index] = __float2half(static_cast<float>(index % 97));
        }
        tensors.push_back(std::move(table));
        tensors.push_back(int64Tensor(kMultipliersKey, {kLayerMultipliers.begin(), kLayerMultipliers.end()}));
        tensors.push_back(int64Tensor(kVocabSizesKey, {kHeadVocabSizes.begin(), kHeadVocabSizes.end()}));
        tensors.push_back(int64Tensor(kOffsetsKey, {kHeadOffsets.begin(), kHeadOffsets.end()}));
        ASSERT_TRUE(safetensors::saveSafetensors(mDirectory / "model.safetensors", tensors, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        std::filesystem::remove_all(mDirectory);
    }

    static Tensor int64Tensor(char const* name, std::vector<int64_t> const& values)
    {
        Tensor tensor({static_cast<int64_t>(values.size())}, DeviceType::kCPU, trt_edgellm::DataType::kINT64, name);
        std::memcpy(tensor.rawPointer(), values.data(), values.size() * sizeof(int64_t));
        return tensor;
    }

    LLMEngineConfig config() const
    {
        LLMEngineConfig cfg{};
        cfg.ngramLayers = {1};
        cfg.ngramEmbedDim = static_cast<int32_t>(kRowWidth) * kHeads;
        cfg.ngramConvKernel = 4;
        cfg.ngramConvDim = 16;
        cfg.ngramSize = kNgramSize;
        cfg.ngramHeadsPerOrder = kHeadsPerOrder;
        cfg.eosTokenIds = {kEosTokenId};
        cfg.plePool.rowCount = kRowCount;
        cfg.plePool.rowWidth = kRowWidth;
        cfg.plePool.dtype = PleTable::DType::kFP16;
        cfg.plePool.scale = 1.0;
        cfg.plePool.heads = kHeads;
        cfg.plePool.checkpointDir = mDirectory.string();
        cfg.plePool.layerMultipliersKey = kMultipliersKey;
        cfg.plePool.headVocabSizesKey = kVocabSizesKey;
        cfg.plePool.headOffsetsKey = kOffsetsKey;
        cfg.plePool.source.tensorKey = kTableKey;
        cfg.plePool.source.dtype = PleTable::DType::kFP16;
        cfg.plePool.source.bytes = kRowCount * kRowWidth * 2;
        cfg.plePool.source.scale = 1.0;
        cfg.plePool.source.samples = {{0, tableSample()}};
        return cfg;
    }

    //! The identity spot check PleTable::fromSource validates: the table's first bytes in hex.
    std::string tableSample() const
    {
        char const* const digits = "0123456789abcdef";
        std::string sample;
        for (int64_t index = 0; index < 4; ++index)
        {
            half const value = __float2half(static_cast<float>(index % 97));
            uint8_t bytes[sizeof(half)];
            std::memcpy(bytes, &value, sizeof(half));
            for (uint8_t byte : bytes)
            {
                sample.push_back(digits[byte >> 4]);
                sample.push_back(digits[byte & 15]);
            }
        }
        return sample;
    }

    std::unique_ptr<Qwen4ExpNgramPreprocessor> makePreprocessor()
    {
        PleRuntimeConfig pleConfig{};
        pleConfig.residency = PleResidency::kRESIDENT;
        auto preprocessor = std::make_unique<Qwen4ExpNgramPreprocessor>(
            config(), kMaxBatchSize, kMaxSeqLen, mTensorMap, mStream, pleConfig, mDirectory);
        preprocessor->resetHistory(kSlot);
        return preprocessor;
    }

    //! Hash one token for slot kSlot and return the row map it produced.
    std::vector<int32_t> step(Qwen4ExpNgramPreprocessor& preprocessor, int32_t token)
    {
        int32_t const slot = kSlot;
        preprocessor.embed(&token, /*hostQueryLengths=*/nullptr, &slot, /*numSequences=*/1, /*queryWidth=*/1, mStream);
        return readRowMap(preprocessor, /*tokens=*/1);
    }

    std::vector<int32_t> readRowMap(Qwen4ExpNgramPreprocessor const& preprocessor, int32_t tokens)
    {
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<int32_t> host(static_cast<size_t>(tokens) * kHeads);
        CUDA_CHECK(cudaMemcpy(
            host.data(), preprocessor.rowMap().rawPointer(), host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return host;
    }

    static constexpr char const* kTableKey{"ple.weight"};
    static constexpr char const* kMultipliersKey{"ple.layer_multipliers"};
    static constexpr char const* kVocabSizesKey{"ple.ngram_heads_vocab_sizes"};
    static constexpr char const* kOffsetsKey{"ple.ngram_heads_offsets"};

    cudaStream_t mStream{};
    std::filesystem::path mDirectory;
    TensorMap mTensorMap;
};

//! Tokens the sequence has already fed to the engine before the verify step.
std::vector<int32_t> const kConsumed{11, 23, 47};
//! The verify chain: the root token the previous round produced, then three drafts.
std::vector<int32_t> const kChain{52, 61, 70, 83};
//! The root of the round that follows.
constexpr int32_t kNextToken{97};

} // namespace

// Given a verify step that hashed a whole proposed chain, of which only a prefix was accepted,
// When the decoder re-seeds the slot's history from the fed and accepted tokens and hashes the next one,
// Then that token's row IDs match token-by-token decoding, for none, some and all drafts accepted, and
// at a sequence start, where a prefix shorter than the hash's reach brings the EOS padding back.
TEST_F(Qwen4ExpNgramHistoryTest, RewindingToTheAcceptedPrefixMatchesTokenByTokenDecoding)
{
    struct Case
    {
        std::vector<int32_t> fed;
        int32_t accepted;
    };
    std::vector<Case> const cases{
        {kConsumed, 0}, {kConsumed, 2}, {kConsumed, static_cast<int32_t>(kChain.size())}, {{}, 1}};
    for (auto const& [fed, accepted] : cases)
    {
        std::vector<int32_t> consumedAfter = fed;
        consumedAfter.insert(consumedAfter.end(), kChain.begin(), kChain.begin() + accepted);

        auto reference = makePreprocessor();
        for (int32_t token : consumedAfter)
        {
            step(*reference, token);
        }
        auto const expected = step(*reference, kNextToken);

        auto verified = makePreprocessor();
        for (int32_t token : fed)
        {
            step(*verified, token);
        }
        int32_t const slot = kSlot;
        verified->embed(kChain.data(), /*hostQueryLengths=*/nullptr, &slot, /*numSequences=*/1,
            static_cast<int32_t>(kChain.size()), mStream);
        verified->seedHistory(slot, consumedAfter.data(), static_cast<int32_t>(consumedAfter.size()));

        EXPECT_EQ(step(*verified, kNextToken), expected) << "fed=" << fed.size() << " accepted=" << accepted;
    }
}

// Given a verify tree whose branches share the root but not the tokens after it,
// When it is hashed with its parent links,
// Then each node's row IDs match token-by-token decoding of its root-to-node path, and the slot's
// history is left for the rewind, so the next step still continues from the consumed tokens.
TEST_F(Qwen4ExpNgramHistoryTest, TreeNodesHashLikeDecodingTheirOwnPath)
{
    // Node 0 is the root, nodes 1 and 2 are its children, node 3 continues node 2. Together they
    // cover a window from the history only, from history and ancestors, and from ancestors only.
    std::vector<int32_t> const tokens{52, 61, 70, 83};
    std::vector<int32_t> const parents{-1, 0, 0, 2};
    std::vector<std::vector<int32_t>> const paths{{52}, {52, 61}, {52, 70}, {52, 70, 83}};
    auto const nodes = static_cast<int32_t>(tokens.size());

    auto tree = makePreprocessor();
    for (int32_t token : kConsumed)
    {
        step(*tree, token);
    }
    int32_t const slot = kSlot;
    tree->embed(tokens.data(), /*hostQueryLengths=*/nullptr, &slot, /*numSequences=*/1, nodes, mStream, parents.data());
    auto const treeRows = readRowMap(*tree, nodes);

    for (int32_t node = 0; node < nodes; ++node)
    {
        auto reference = makePreprocessor();
        for (int32_t token : kConsumed)
        {
            step(*reference, token);
        }
        std::vector<int32_t> expected;
        for (int32_t token : paths[static_cast<size_t>(node)])
        {
            expected = step(*reference, token);
        }
        std::vector<int32_t> const actual(treeRows.begin() + node * kHeads, treeRows.begin() + (node + 1) * kHeads);
        EXPECT_EQ(actual, expected) << "node=" << node;
    }

    auto reference = makePreprocessor();
    for (int32_t token : kConsumed)
    {
        step(*reference, token);
    }
    EXPECT_EQ(step(*tree, kNextToken), step(*reference, kNextToken));
}
