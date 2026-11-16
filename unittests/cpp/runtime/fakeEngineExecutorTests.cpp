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

// Drives LLMInferenceRuntime through a real greedy decode with FakeEngineExecutor, the state-based substitute, in
// place of MockEngineExecutor: construction from config.json, the binding-name surface, and the KV/logits writes
// all have to hold together for a request to come back with the tokens the fake wrote.

#include "fakeEngineExecutor.h"

#include "runtime/llmInferenceRuntime.h"

#include <set>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

class FakeEngineExecutorTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("fakeEngineExecutorTests", makeTinyVanillaConfig());
    }
};

// Given a model directory holding a config and a tokenizer but no serialized engine
// When the runtime is assembled around a FakeEngineExecutor instead of the interaction-based mock
// Then it comes up as a working single-engine deployment, same as with MockEngineExecutor
TEST_F(FakeEngineExecutorTest, AssemblesWithoutAnyEngineFileOnDisk)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    EXPECT_FALSE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "vanilla");
}

// Everything else this substitute is good for rests on this one property. A substitute that answered the same way
// regardless of its input would decode identical text whether or not the runtime reused the right cache pages, so
// every test comparing two runs would pass vacuously.
//
// Given a FakeEngineExecutor with nothing forced
// When prompts differing in length, and prompts differing only in content, are decoded, and one is decoded twice
// Then every distinct prompt comes back with a distinct token stream, while the repeated prompt reproduces its own
// exactly
TEST_F(FakeEngineExecutorTest, DerivesItsOutputFromItsInputRatherThanReturningAFixture)
{
    auto const decode = [this](std::string const& prompt) {
        auto artifacts = makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream);
        auto runtime = makeRuntime(std::move(artifacts));
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(makeGreedyRequest(prompt, /*maxGenerateLength=*/4), response, mStream));
        return response.outputIds.empty() ? std::vector<int32_t>{} : response.outputIds[0];
    };

    auto const shortPrompt = decode(makePrompt(1));
    auto const longPrompt = decode(makePrompt(4));
    auto const shortPromptAgain = decode(makePrompt(1));

    ASSERT_EQ(shortPrompt.size(), 4U);
    EXPECT_NE(shortPrompt, longPrompt) << "the engine ignored how long its input was";
    EXPECT_EQ(shortPrompt, shortPromptAgain) << "the engine is not reproducible, so no two runs can be compared";

    // Length is the weaker half. A deployment that looked up the wrong row of the embedding table, or fed a
    // sequence its neighbour's tokens, still gets the token *count* right, so only two prompts of one length can
    // tell it apart -- and until the test tokenizer carried more than one letter, no fixture here could state one.
    EXPECT_NE(decode(makePrompt(6, /*variant=*/0)), decode(makePrompt(6, /*variant=*/1)))
        << "two prompts of the same length decoded alike: the engine is reading their length, not their content";
}

// Given a FakeEngineExecutor whose logits are set to always agree on a specific token
// When a greedy request runs to its length cap
// Then every decoded token is the one the fake wrote, proving execute() actually reaches and fills the logits
// binding the runtime bound rather than a fixture the test captured out of band
TEST_F(FakeEngineExecutorTest, DecodesWhateverTokenTheFakeWritesIntoLogits)
{
    auto engine = makeFakeEngine();
    engine->forceOutputTokens(agreedLogitsRows(kAgreedToken));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", /*maxGenerateLength=*/3), response, mStream));

    expectResponseCoversEverySlot(response, 1);
    EXPECT_THAT(response.outputIds[0], ::testing::AllOf(::testing::SizeIs(3U), ::testing::Each(kAgreedToken)));
}

// Given a batch of two whose first slot decodes EOS on round two
// When the batch compacts and the survivor keeps decoding
// Then the stopped slot reports kEndId at the round it actually hit EOS and the survivor reaches the length cap --
// exercising per-round tokens (forceOutputTokensPerRound) rather than one fixed pattern for the whole request, which is
// what a vanilla-mode compaction case actually needs
TEST_F(FakeEngineExecutorTest, SlotThatHitsEosStopsWhileTheOtherSlotKeepsDecoding)
{
    constexpr int64_t kMaxGenerateLength{4};

    auto engine = makeFakeEngine();
    auto* const rawEngine = engine.get();
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    // Both entries are two rows wide so the schedule stays correct regardless of which physical row the survivor
    // occupies before/after compaction; only the EOS round needs to pin slot 0 specifically.
    rawEngine->forceOutputTokensPerRound({
        agreedLogitsRows(kAgreedToken, 2), // round 0
        {eosId, kAgreedToken},             // round 1
        agreedLogitsRows(kAgreedToken, 2), // round 2 .. afterwards
    });

    auto runtime = makeRuntime(std::move(artifacts));
    // Batch size 2
    // 1st sentence "a"
    // 2nd sentence "aa"
    auto const request = makeGreedyBatch({"a", "aa"}, kMaxGenerateLength);

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 2);
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kAgreedToken, eosId}));
    EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kLength);
    EXPECT_THAT(response.outputIds[1],
        ::testing::AllOf(::testing::SizeIs(static_cast<size_t>(kMaxGenerateLength)), ::testing::Each(kAgreedToken)));
}

// The fake's KV write is what makes it a fake rather than a stub: it puts identifiable values where the runtime
// said this forward's tokens go. Without this, the generator, the dtype handling and the page-table addressing
// are all unexercised -- a constant-zero generator would pass every other test in the suite.
//
// Given a request whose prefill and decode rounds are each stored through the page table the runtime published
// When the KV pool is read back afterwards
// Then every token of every layer holds the value the generator produced for that (layer, position, half),
// resolved through the same page table rather than through a remembered address
TEST_F(FakeEngineExecutorTest, StoresEachTokenOfEachLayerWhereTheRuntimeSaidItGoes)
{
    constexpr int64_t kPromptLength{3};
    constexpr int64_t kGenerateLength{2};

    // The test supplies its own generator rather than asserting against slotKeyedStateValue(): comparing the pool
    // against the very function that filled it passes even when that function is a constant, which would make
    // every misplacement invisible. These values are distinct for every (layer, position, half) under test.
    auto const generator = [](FakeStateSlot const& slot) {
        return static_cast<double>(
            1 + slot.layerIdx * 16 + slot.position * 2 + (slot.kind == FakeStateKind::kValue ? 1 : 0));
    };

    auto engine = std::make_unique<FakeEngineExecutor>(
        FakeLayerComposition::fromConfigJson(makeTinyVanillaConfig()), generator);
    auto* const raw = engine.get();
    raw->recordBinding(binding_names::kKVPageTable);

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const numLayers = artifacts.deployment.base.numAttentionLayers;
    ASSERT_GT(numLayers, 1) << "a single-layer deployment could not catch a layer index dropped from the address";
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(
        runtime.handleRequest(makeGreedyRequest(makePrompt(kPromptLength), kGenerateLength), response, mStream));

    // The last forward's page table: every position this request touched is still mapped through it, since one
    // sequence keeps its pages for its lifetime.
    auto const& pageTable = raw->preparedBinding(binding_names::kKVPageTable).back();
    ASSERT_FALSE(pageTable.empty());

    rt::Tensor* const pool = raw->boundTensor(binding_names::formatKVCacheName(0, /*isPast=*/false));
    ASSERT_NE(pool, nullptr);
    auto const geometry = FakeKVGeometry::from(*pool, pageTable.size(), /*batch=*/1);

    // Prefill stored positions [0, kPromptLength); each decode round stored the one after it.
    int64_t const storedPositions = kPromptLength + kGenerateLength - 1;
    for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        rt::Tensor* const layerPool = raw->boundTensor(binding_names::formatKVCacheName(layerIdx, /*isPast=*/false));
        ASSERT_NE(layerPool, nullptr) << "layer " << layerIdx;
        for (int64_t position = 0; position < storedPositions; ++position)
        {
            for (auto kind : {FakeStateKind::kKey, FakeStateKind::kValue})
            {
                FakeStateSlot const slot{layerIdx, position, kind};
                EXPECT_DOUBLE_EQ(FakeKVWriter::read(*layerPool, geometry, pageTable, /*batchSlot=*/0, slot, mStream),
                    generator(slot))
                    << "layer " << layerIdx << " position " << position
                    << (kind == FakeStateKind::kValue ? " value" : " key");
            }
        }
    }
}

// The addressing this substitute does is otherwise only ever checked against itself: entries are written and read
// back through the same derivation, so a shared misreading of the layout round-trips perfectly. This compares the
// derivation against the runtime's own data instead -- the shape it bound the page table at, and the page ids it
// put in it.
//
// Given a live batch of two sequences
// When the geometry the fake derives from the bound tensors is compared with the page table the runtime published
// Then the derived pages-per-sequence matches the page table's own trailing dimension, each sequence's value pages
// are its key pages offset by the pool's page count, and the two sequences share no physical page
TEST_F(FakeEngineExecutorTest, DerivesTheSamePagedLayoutTheRuntimePublished)
{
    auto engine = makeFakeEngine();
    auto* const raw = engine.get();
    raw->recordBinding(binding_names::kKVPageTable);

    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(
        makeGreedyBatch({makePrompt(3, /*variant=*/0), makePrompt(9, /*variant=*/1)}, /*maxGenerateLength=*/2),
        response, mStream));
    ASSERT_EQ(response.outputIds.size(), 2U);

    rt::Tensor* const pageTableTensor = raw->boundTensor(binding_names::kKVPageTable);
    ASSERT_NE(pageTableTensor, nullptr);
    auto const pageTableShape = pageTableTensor->getShape();
    ASSERT_EQ(pageTableShape.getNumDims(), 3) << "expected a [batch, 2, pagesPerSequence] page table";
    int64_t const batch = pageTableShape[0];
    ASSERT_EQ(batch, 2) << "this check needs two live sequences to see them apart";
    ASSERT_EQ(pageTableShape[1], 2) << "expected separate key and value halves";

    rt::Tensor* const pool = raw->boundTensor(binding_names::formatKVCacheName(0, /*isPast=*/false));
    ASSERT_NE(pool, nullptr);
    auto const& pageTable = raw->preparedBinding(binding_names::kKVPageTable).back();
    auto const geometry = FakeKVGeometry::from(*pool, pageTable.size(), batch);

    EXPECT_EQ(geometry.pagesPerSequence, pageTableShape[2])
        << "the fake infers pages-per-sequence from the table's length; the runtime states it in the shape";

    std::set<int32_t> keyPages;
    for (int64_t slot = 0; slot < batch; ++slot)
    {
        for (int64_t page = 0; page < geometry.pagesPerSequence; ++page)
        {
            auto const keyIndex = geometry.pageTableIndex(slot, /*isValue=*/false, page);
            auto const valueIndex = geometry.pageTableIndex(slot, /*isValue=*/true, page);
            ASSERT_LT(valueIndex, pageTable.size());
            int32_t const keyPage = pageTable[keyIndex];
            EXPECT_EQ(pageTable[valueIndex], keyPage + static_cast<int32_t>(geometry.pages))
                << "slot " << slot << " logical page " << page
                << ": the value half is addressed as the key page plus the pool's page count";
            EXPECT_TRUE(keyPages.insert(keyPage).second)
                << "physical page " << keyPage << " is claimed by more than one sequence";
        }
    }
}

// slotKeyedStateValue() is what a test forcing state values reaches for when it wants entries it can name without
// running the engine's arithmetic, so its only real requirement is that a misplaced entry looks different from a
// correct one. Global injectivity is impossible -- the values are deliberately small enough to survive bf16 -- but
// a one-step error in any single coordinate has to change the number, since that is the shape of the mistakes
// worth catching.
//
// Given the slot-keyed KV value function
// When one of layer, position, or cache half moves by one and the others are held
// Then the value changes
TEST(SlotKeyedStateValueTest, SeparatesAOneStepErrorInEveryCoordinate)
{
    FakeStateSlot const base{1, 4, FakeStateKind::kKey};
    EXPECT_NE(
        slotKeyedStateValue(base), slotKeyedStateValue(FakeStateSlot{base.layerIdx + 1, base.position, base.kind}));
    EXPECT_NE(
        slotKeyedStateValue(base), slotKeyedStateValue(FakeStateSlot{base.layerIdx, base.position + 1, base.kind}));
    EXPECT_NE(slotKeyedStateValue(base),
        slotKeyedStateValue(FakeStateSlot{base.layerIdx, base.position, FakeStateKind::kValue}));
}

class FakeHybridEngineTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("fakeHybridEngineTests", makeTinyHybridConfig());
    }
};

// Whether a layer keeps a KV cache or a recurrent state is a property of the layer, not of the model: a hybrid
// decoder holds both, and one forward has to advance both. A substitute that picked one writer per model would
// leave half of a hybrid deployment's state untouched, and every hybrid test built on it would be asserting
// against a cache nothing ever wrote.
//
// Given a deployment with one attention layer and one recurrent layer
// When a request runs through it
// Then the attention layer's KV holds the entries for its tokens and the recurrent layer's state and conv rows
// hold the value for the last position the sequence reached, from the same forwards
TEST_F(FakeHybridEngineTest, WritesBothTheKvAndTheRecurrentStateOfAHybridDecoder)
{
    constexpr int64_t kPromptLength{3};
    constexpr int64_t kGenerateLength{2};

    auto const generator = [](FakeStateSlot const& slot) {
        return static_cast<double>(1 + slot.layerIdx * 32 + slot.position * 4 + static_cast<int32_t>(slot.kind));
    };

    auto engine
        = std::make_unique<FakeEngineExecutor>(FakeLayerComposition::fromConfigJson(makeTinyHybridConfig()), generator);
    auto* const raw = engine.get();
    raw->recordBinding(binding_names::kKVPageTable);
    raw->recordBinding(binding_names::kStateIndices);
    raw->recordStatePool(binding_names::formatRecurrentStateName(0, /*isPast=*/false));
    raw->recordStatePool(binding_names::formatConvStateName(0, /*isPast=*/false));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    ASSERT_EQ(artifacts.deployment.base.numAttentionLayers, 1);
    ASSERT_EQ(artifacts.deployment.base.numDecoderLayers, 2) << "the second layer must be the recurrent one";
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(
        runtime.handleRequest(makeGreedyRequest(makePrompt(kPromptLength), kGenerateLength), response, mStream));

    // The attention half: every token this request stored is where the page table says it is.
    auto const& pageTable = raw->preparedBinding(binding_names::kKVPageTable).back();
    rt::Tensor* const kvPool = raw->boundTensor(binding_names::formatKVCacheName(0, /*isPast=*/false));
    ASSERT_NE(kvPool, nullptr);
    auto const geometry = FakeKVGeometry::from(*kvPool, pageTable.size(), /*batch=*/1);
    int64_t const storedPositions = kPromptLength + kGenerateLength - 1;
    for (int64_t position = 0; position < storedPositions; ++position)
    {
        for (auto kind : {FakeStateKind::kKey, FakeStateKind::kValue})
        {
            FakeStateSlot const slot{0, position, kind};
            EXPECT_DOUBLE_EQ(
                FakeKVWriter::read(*kvPool, geometry, pageTable, /*batchSlot=*/0, slot, mStream), generator(slot))
                << "position " << position;
        }
    }

    // The recurrent half: one row per sequence carrying the last position it reached, rather than a per-token
    // history. Read at the start of the final forward, not after the request: the runtime releases a sequence's
    // resident row once it finishes, so what survives to the end is a cleared row either way. What matters is that
    // the row the last forward started from held what the forward before it left there.
    auto const& rounds = raw->preparedRounds();
    ASSERT_GE(rounds.size(), 2U) << "a single forward could not show the state carrying across a boundary";
    size_t const lastRound = rounds.size() - 1;
    int32_t const stateIndex = raw->preparedBinding(binding_names::kStateIndices)[lastRound].at(0);
    int64_t const positionBeforeLastForward
        = rounds[lastRound].dims.seqLen == 1 ? rounds[lastRound - 1].dims.seqLen - 1 : storedPositions - 1;
    for (auto const& [name, kind] :
        {std::pair{binding_names::formatRecurrentStateName(0, /*isPast=*/false), FakeStateKind::kRecurrent},
            std::pair{binding_names::formatConvStateName(0, /*isPast=*/false), FakeStateKind::kConv}})
    {
        auto const& snapshots = raw->preparedStatePool(name);
        ASSERT_EQ(snapshots.size(), rounds.size()) << name;
        ASSERT_GT(snapshots[lastRound].size(), static_cast<size_t>(stateIndex)) << name;
        EXPECT_DOUBLE_EQ(snapshots[lastRound][static_cast<size_t>(stateIndex)],
            generator(FakeStateSlot{0, positionBeforeLastForward, kind}))
            << name;
    }
}

// state_indices is how the runtime tells the engine which pool row a sequence owns, and rows get reassigned as
// sequences come and go. A substitute that wrote every sequence's state to the same row would look correct for a
// single request and silently merge two, so the routing needs a batch whose slots sit on different rows and are
// far enough apart in the sequence that their states cannot coincide.
//
// Given a hybrid deployment running two sequences of different lengths at once
// When their recurrent state is read back through the row indices the runtime assigned
// Then each row holds the value for its own sequence's progress, not the other's
TEST_F(FakeHybridEngineTest, KeepsEachSequencesRecurrentStateInItsOwnRow)
{
    constexpr int64_t kShortPrompt{2};
    constexpr int64_t kLongPrompt{4};

    auto const generator = [](FakeStateSlot const& slot) {
        return static_cast<double>(1 + slot.position * 4 + static_cast<int32_t>(slot.kind));
    };

    auto engine
        = std::make_unique<FakeEngineExecutor>(FakeLayerComposition::fromConfigJson(makeTinyHybridConfig()), generator);
    auto* const raw = engine.get();
    raw->recordBinding(binding_names::kStateIndices);
    raw->recordStatePool(binding_names::formatRecurrentStateName(0, /*isPast=*/false));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyBatch(
        {makePrompt(kShortPrompt, /*variant=*/0), makePrompt(kLongPrompt, /*variant=*/1)}, /*maxGenerateLength=*/2);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    // After the shared prefill, each sequence sits at the last position of its own prompt, so the round that
    // follows starts from two states that cannot be mistaken for one another.
    auto const& indices = raw->preparedBinding(binding_names::kStateIndices);
    auto const& snapshots = raw->preparedStatePool(binding_names::formatRecurrentStateName(0, /*isPast=*/false));
    ASSERT_GE(indices.size(), 2U);
    ASSERT_EQ(indices[1].size(), 2U) << "the second forward should still carry both sequences";
    ASSERT_NE(indices[1][0], indices[1][1])
        << "the two sequences must occupy different rows for this to prove anything";

    EXPECT_DOUBLE_EQ(snapshots[1][static_cast<size_t>(indices[1][0])],
        generator(FakeStateSlot{0, kShortPrompt - 1, FakeStateKind::kRecurrent}));
    EXPECT_DOUBLE_EQ(snapshots[1][static_cast<size_t>(indices[1][1])],
        generator(FakeStateSlot{0, kLongPrompt - 1, FakeStateKind::kRecurrent}));
}

} // namespace
