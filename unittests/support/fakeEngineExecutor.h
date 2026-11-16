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

// FakeEngineExecutor: a state-based substitute for EngineExecutor, complementing the interaction-based
// MockEngineExecutor in substituteEngine.h. Where MockEngineExecutor lets a test state the calls it expects up
// front, FakeEngineExecutor actually runs -- it answers binding queries from a declared layer topology and writes
// deterministic values into whatever the runtime binds, so tests built on it survive refactors that change call
// patterns but not behavior.
//
// Scope: the core decode path of attention, recurrent and hybrid decoders. Speculative-decode-only bindings are
// not covered. Layer counts and KV dtype are read directly from the deployment's config.json, the same JSON
// makeTinyVanillaConfig() produces, so a test only has to change the config to change what the engine exposes.

#include "fakeCachePools.h"
#include "fakeForwardLayout.h"
#include "fakeState.h"
#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace substitute_engine
{

using namespace trt_edgellm;

class FakeEngineExecutor : public rt::EngineExecutor
{
public:
    //! What the runtime asked for on one forward. These are the runtime's own decisions -- which optimization
    //! profile, how long a sequence to run -- reaching the engine as inputs, and `prepare()` is the only place
    //! they are visible, since `execute()` takes no arguments.
    struct PreparedRound
    {
        int32_t profileIndex{};
        rt::InferenceDims dims{};
    };

    explicit FakeEngineExecutor(FakeLayerComposition composition, FakeStateGenerator forcedStateValues = {})
        : mComposition(std::move(composition))
        , mForcedStateValues(std::move(forcedStateValues))
    {
        mBindingNames.push_back(binding_names::kInputsEmbeds);
        mBindingNames.push_back(binding_names::kPositions);
        mBindingNames.push_back(binding_names::kQueryStartOffsets);
        mBindingNames.push_back(binding_names::kQueryLengths);
        mBindingNames.push_back(binding_names::kPastLengths);
        mBindingNames.push_back(binding_names::kAttentionSequenceLengths);
        mBindingNames.push_back(binding_names::kStateIndices);
        mBindingNames.push_back(binding_names::kLogitsIndices);
        mBindingNames.push_back(binding_names::kLogits);
        mBindingNames.push_back(binding_names::kExecutionPhaseMarker);
        mBindingNames.push_back(binding_names::kContextSequenceCountCarrier);
        if (mComposition.attentionLayers > 0)
        {
            mBindingNames.push_back(binding_names::kKVCacheStartIndex);
            mBindingNames.push_back(binding_names::kKVPageTable);
        }
        for (int32_t layerIdx = 0; layerIdx < mComposition.attentionLayers; ++layerIdx)
        {
            mPastKvNames.push_back(binding_names::formatKVCacheName(layerIdx, /*isPast=*/true));
            mPresentKvNames.push_back(binding_names::formatKVCacheName(layerIdx, /*isPast=*/false));
            mBindingNames.push_back(mPastKvNames.back());
            mBindingNames.push_back(mPresentKvNames.back());
        }
        for (int32_t layerIdx = 0; layerIdx < mComposition.recurrentLayers; ++layerIdx)
        {
            mPresentRecurrentNames.push_back(binding_names::formatRecurrentStateName(layerIdx, /*isPast=*/false));
            mPresentConvNames.push_back(binding_names::formatConvStateName(layerIdx, /*isPast=*/false));
            mBindingNames.push_back(binding_names::formatRecurrentStateName(layerIdx, /*isPast=*/true));
            mBindingNames.push_back(mPresentRecurrentNames.back());
            mBindingNames.push_back(binding_names::formatConvStateName(layerIdx, /*isPast=*/true));
            mBindingNames.push_back(mPresentConvNames.back());
        }
        for (auto const& name : mBindingNames)
        {
            mBindingSet.insert(name);
        }
    }

    //! Override the token each physical logits row decodes to, on every forward.
    //!
    //! Left unset -- the default -- the engine decodes each row from what it computed, so the tokens that come back
    //! depend on the prompt and on every cached entry behind it. Forcing replaces only that last step: the forced
    //! token is still what carries forward into the next round's state, so a forced run stays internally consistent.
    //!
    //! `rows[i]` is the token for physical logits row `i`. A row past the end, or a negative id, is not overridden
    //! and keeps its computed token.
    void forceOutputTokens(std::vector<int32_t> rows)
    {
        forceOutputTokensPerRound({std::move(rows)});
    }

    //! Override tokens per round: `rounds[round][row]`, where the outer dimension is time (one entry per
    //! `execute()` call, the last entry repeating for every call past the end) and the inner dimension is the
    //! active batch position that round -- not a stable per-request identity, since compaction can shift a
    //! surviving request to a lower row after another slot finishes.
    //!
    //! Lets a test change what a later round decodes to -- slot 0 hitting EOS on round 3 while slot 1 keeps going --
    //! without gmock's `InSequence`/`WillOnce`/`WillRepeatedly` chain, since the fake tracks its own call count.
    void forceOutputTokensPerRound(std::vector<std::vector<int32_t>> rounds)
    {
        mForcedOutputTokens = std::move(rounds);
        mExecuteCount = 0;
    }

    //! State that this engine stands in for a draft model, whose output binding carries log-probabilities rather
    //! than raw logits -- every draft model ends in `F.log_softmax` (see FakeOutputScale). Speculative decoding
    //! accumulates those values down the chain, so an engine on the wrong scale collapses the verify tree through
    //! runtime code that is correct against a real engine.
    void actAsDraftEngine()
    {
        mOutputScale = FakeOutputScale::kLogProbabilities;
    }

    void exposeInt32Binding(std::string name)
    {
        if (mBindingSet.insert(name).second)
        {
            mBindingNames.push_back(std::move(name));
        }
    }

    //! Override what gets stored in each cache entry, keyed by which entry it is rather than by the sequence's
    //! content. For a test whose subject is *where* entries land; it gives entries the test can name without
    //! running the engine's arithmetic. Leaves the computed tokens alone.
    void forceStateValues(FakeStateGenerator generator)
    {
        mForcedStateValues = std::move(generator);
    }

    //! Snapshot `name`'s contents on every subsequent prepare(), for tests asserting on what the runtime decided
    //! rather than on what came back -- a reused KV page, a compacted state row. Opt-in because each recorded
    //! binding costs a device-to-host copy and a stream synchronization per forward.
    //!
    //! INT32 bindings only: the ones worth watching this way are indices and lengths, and reading a half or fp8
    //! binding as int32 would report plausible nonsense rather than fail.
    void recordBinding(std::string name)
    {
        mRecordedBindings.emplace(std::move(name), BindingRecord{});
    }

    //! Snapshot `name`'s floating-point contents on every subsequent prepare(), for a test asserting on an input the
    //! runtime prepared for the engine (a gathered RoPE table, say) rather than on a decision it reports as indices.
    void recordFloatBinding(std::string name)
    {
        mRecordedFloatBindings.emplace(std::move(name), std::vector<std::vector<double>>{});
    }

    //! One snapshot of `name` per prepare() call; a round where `name` was not bound leaves an empty entry.
    std::vector<std::vector<double>> const& preparedFloatBinding(std::string const& name) const
    {
        auto const it = mRecordedFloatBindings.find(name);
        if (it == mRecordedFloatBindings.end())
        {
            throw std::runtime_error(
                "FakeEngineExecutor::preparedFloatBinding(\"" + name + "\"): call recordFloatBinding first.");
        }
        return it->second;
    }

    //! Snapshot one value per row of a floating-point state pool on every subsequent prepare().
    //!
    //! Recurrent state, unlike the KV pool, is only observable while its sequence is live: the runtime releases the
    //! resident row when the sequence finishes, so a test reading the pool after handleRequest() returns finds it
    //! cleared. Recording at prepare() time catches what the previous forward left there, which is the property
    //! worth asserting -- that the state carried across the forward boundary.
    void recordStatePool(std::string name)
    {
        mRecordedStatePools.emplace(std::move(name), std::vector<std::vector<double>>{});
    }

    //! One snapshot per prepare() call: `preparedStatePool(name)[round][row]`.
    std::vector<std::vector<double>> const& preparedStatePool(std::string const& name) const
    {
        auto const it = mRecordedStatePools.find(name);
        if (it == mRecordedStatePools.end())
        {
            throw std::runtime_error("FakeEngineExecutor::preparedStatePool(\"" + name
                + "\"): this pool was never registered for recording. Call recordStatePool(\"" + name
                + "\") before running the request.");
        }
        return it->second;
    }

    //! How many forwards this engine has run. Separate from the forced-token cursor, which forceOutputTokensPerRound()
    //! rewinds, and from preparedRounds(), since nothing obliges the runtime to prepare once per execute -- hoisting
    //! or batching binding setup is a refactor. A test asking whether an engine did any work means this one.
    size_t forwardsRun() const noexcept
    {
        return mForwardsRun;
    }

    //! What every prepare() call so far was handed, in call order. Always collected -- it is host-side data the
    //! runtime already computed, so there is nothing to opt into.
    std::vector<PreparedRound> const& preparedRounds() const noexcept
    {
        return mPreparedRounds;
    }

    //! The tensor the runtime bound to `name` on the most recent prepare(), or null if it bound none. The pools
    //! behind the KV bindings outlive any single forward, so a test can read one back after the request finished.
    rt::Tensor* boundTensor(std::string const& name) const
    {
        return mMap == nullptr ? nullptr : mMap->get(name);
    }

    //! One snapshot of `name` per prepare() call, in the same order as `preparedRounds()`. A round where the
    //! deployment did not bind `name` leaves an empty entry, so indices stay aligned with `preparedRounds()`.
    std::vector<std::vector<int32_t>> const& preparedBinding(std::string const& name) const
    {
        auto const it = mRecordedBindings.find(name);
        if (it == mRecordedBindings.end())
        {
            throw std::runtime_error("FakeEngineExecutor::preparedBinding(\"" + name
                + "\"): this binding was never registered for recording, so nothing was captured. Call "
                  "recordBinding(\""
                + name + "\") before running the request.");
        }
        if (!it->second.everBound)
        {
            throw std::runtime_error("FakeEngineExecutor::preparedBinding(\"" + name
                + "\"): recording was registered, but the runtime never bound that name in any of the "
                + std::to_string(mPreparedRounds.size())
                + " forward(s) that ran. Check the spelling against binding_names, and that the configured layer "
                  "topology actually uses this binding.");
        }
        return it->second.rounds;
    }

    bool prepare(
        int32_t profileIndex, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) override
    {
        mMap = &map;
        mPreparedRounds.push_back(PreparedRound{profileIndex, dims});
        for (auto& [name, record] : mRecordedBindings)
        {
            record.rounds.push_back(snapshotInt32Binding(map, name, stream));
            record.everBound = record.everBound || !record.rounds.back().empty();
        }
        for (auto& [name, rounds] : mRecordedStatePools)
        {
            rounds.push_back(snapshotStatePool(map, name, stream));
        }
        for (auto& [name, rounds] : mRecordedFloatBindings)
        {
            rt::Tensor const* const tensor = map.get(name);
            rounds.push_back(tensor == nullptr ? std::vector<double>{} : readFloatBinding(*tensor, stream));
        }
        return true;
    }

    bool execute(cudaStream_t stream) override
    {
        if (mMap == nullptr)
        {
            throw std::runtime_error("FakeEngineExecutor::execute called before prepare");
        }
        runForward(stream);
        ++mExecuteCount;
        ++mForwardsRun;
        return true;
    }

    bool captureGraph(cudaStream_t /*stream*/) override
    {
        return false;
    }

    int64_t getRequiredContextMemorySize() const override
    {
        return kContextMemoryBytes;
    }

    bool setContextMemory(rt::Tensor& /*sharedMem*/) override
    {
        return true;
    }

    int32_t getNumIOTensors() const override
    {
        return static_cast<int32_t>(mBindingNames.size());
    }

    char const* getIOTensorName(int32_t index) const override
    {
        return mBindingNames.at(static_cast<size_t>(index)).c_str();
    }

    bool hasIOTensor(char const* name) const override
    {
        return mBindingSet.count(name) != 0;
    }

    trt_edgellm::DataType getBindingDataType(char const* name) const override
    {
        std::string const binding(name);
        if (binding_names::isKVCacheBinding(binding))
        {
            return mComposition.kvType;
        }
        if (binding == binding_names::kLogits)
        {
            return trt_edgellm::DataType::kFLOAT;
        }
        return trt_edgellm::DataType::kINT32;
    }

    //! Unreachable from the core decode path: every caller (llmEngineConfig.cpp's startup validation,
    //! LoRAManager weight loading) runs before or outside the prepare()/execute() cycle a substitute-engine test
    //! drives. Throwing keeps a deployment that does reach it from silently proceeding on an invented shape.
    nvinfer1::Dims getProfileShape(
        char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const override
    {
        if (mAdvertisePhase8 && std::string(name) == binding_names::kExecutionPhaseMarker
            && (profileIndex == 0 || profileIndex == 1) && selector == nvinfer1::OptProfileSelector::kMAX)
        {
            nvinfer1::Dims dims{};
            dims.nbDims = 1;
            dims.d[0] = profileIndex == 0 ? static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode)
                                          : mDecodeProfileMaxPhase;
            return dims;
        }
        throw std::runtime_error("FakeEngineExecutor::getProfileShape is not implemented");
    }

    //! Only startup validation of engine outputs asks for this, and no decoder this fake stands in for has an output
    //! shape that validation checks. Throwing keeps one that does from proceeding on an invented shape.
    nvinfer1::Dims getTensorShape(char const* /*name*/) const override
    {
        throw std::runtime_error("FakeEngineExecutor::getTensorShape is not implemented");
    }

    //! Report prefill and decode profiles that admit execution phase 8, as engines built with compact support do,
    //! or a decode profile that admits phases up to @p decodeProfileMaxPhase only. Opt-in: the runtime runs compact
    //! steps only on such an engine.
    void advertisePhase8Profile(
        int64_t decodeProfileMaxPhase = static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode))
    {
        mAdvertisePhase8 = true;
        mDecodeProfileMaxPhase = decodeProfileMaxPhase;
    }

    //! No real TRT engine backs this substitute; nothing exercised through substituteEngine.h's assembly path should
    //! reach past the EngineExecutor interface for it (matches MockEngineExecutor::getEngine, which is intentionally
    //! left without a default action).
    nvinfer1::ICudaEngine const& getEngine() const noexcept override
    {
        std::terminate();
    }

private:
    struct BindingRecord
    {
        std::vector<std::vector<int32_t>> rounds;
        //! Separates "bound, but empty this round" from "this name never appeared at all", which are different
        //! mistakes: the first is ordinary batch compaction, the second is a typo or a wrong topology.
        bool everBound{false};
    };

    //! Run one forward.
    //!
    //! Nothing here models an LLM. What it models is the *dependency structure* of one: a hidden value is seeded
    //! from the embedding the runtime looked up for each token, carried through every layer, and folded into the
    //! entry that layer stores -- and each attention layer reads the sequence's whole history back out of the KV
    //! pool first, through the page table the runtime just published. So the tokens that come back depend on every
    //! cached entry behind them, and a page that was reused when it should not have been, dropped when it should
    //! have been kept, or mapped to the wrong sequence changes the output. A substitute that ignored its inputs
    //! would decode the same text either way, which is the one thing these tests must be able to see.
    void runForward(cudaStream_t stream)
    {
        auto const layout = FakeForwardLayout::from(*mMap, stream);
        if (!layout.valid())
        {
            writeOutputLogits({}, stream);
            return;
        }
        auto hidden = seedHiddenFromEmbeddings(layout.physicalRows, stream);
        advanceAttentionLayers(hidden, layout, stream);
        advanceRecurrentLayers(hidden, layout, stream);
        writeOutputLogits(hidden, stream);
    }

    //! One hidden value per physical token row, from the embedding the runtime looked up for it.
    //!
    //! Distinct tokens must land on distinct hidden values or every prompt produces the same output; that property
    //! belongs to the embedding table (see fakeEmbeddingValue in substituteEngine.h) and is asserted separately.
    std::vector<FakeHidden> seedHiddenFromEmbeddings(size_t rows, cudaStream_t stream) const
    {
        std::vector<FakeHidden> hidden(rows);
        rt::Tensor* const embeds = mMap->get(binding_names::kInputsEmbeds);
        if (embeds == nullptr || rows == 0)
        {
            return hidden;
        }
        auto const host = readFloatBinding(*embeds, stream);
        auto const width = static_cast<int64_t>(host.size() / rows);
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t channel = 0; channel < kFakeHiddenChannels; ++channel)
            {
                auto const index = static_cast<size_t>(static_cast<int64_t>(row) * width) + channel;
                hidden[row][channel] = index < host.size() ? toFakeValue(host[index]) : 0;
            }
        }
        return hidden;
    }

    void advanceAttentionLayers(std::vector<FakeHidden>& hidden, FakeForwardLayout const& layout, cudaStream_t stream)
    {
        if (mComposition.attentionLayers == 0)
        {
            return;
        }
        auto const pageTable = snapshotInt32Binding(*mMap, binding_names::kKVPageTable, stream);
        if (pageTable.empty())
        {
            return; // A deployment that binds no page table has no paged KV to carry.
        }
        for (int32_t layerIdx = 0; layerIdx < mComposition.attentionLayers; ++layerIdx)
        {
            rt::Tensor* const pool = mMap->get(mPresentKvNames[static_cast<size_t>(layerIdx)]);
            if (pool == nullptr)
            {
                continue;
            }
            auto const geometry = FakeKVGeometry::from(*pool, pageTable.size(), layout.batch());
            for (int64_t slot = 0; slot < layout.batch(); ++slot)
            {
                advanceAttentionSlot(*pool, layerIdx, geometry, pageTable, layout, slot, hidden, stream);
            }
        }
    }

    //! Carry one sequence's hidden values through one attention layer, and store this forward's entries.
    //!
    //! The history fold runs over cache indices in increasing order and is re-read from the pool every time,
    //! which is what makes the result independent of how the runtime split the work: the same prompt prefilled
    //! whole, prefilled in chunks, or partly served from a reused cache folds the same entries in the same order.
    void advanceAttentionSlot(rt::Tensor& pool, int32_t layerIdx, FakeKVGeometry const& geometry,
        std::vector<int32_t> const& pageTable, FakeForwardLayout const& layout, int64_t slot,
        std::vector<FakeHidden>& hidden, cudaStream_t stream)
    {
        auto const rows = layout.rowsOf(slot);
        if (rows.first >= rows.second)
        {
            return; // This sequence contributed no token to this forward.
        }
        int64_t const past = layout.pastOf(slot);
        int32_t const committed = foldHistory(pool, geometry, pageTable, slot, past, stream);
        std::vector<int32_t> keysThisForward;
        for (int32_t row = rows.first; row < rows.second; ++row)
        {
            int32_t history = committed;
            for (int32_t earlier = rows.first; earlier < row; ++earlier)
            {
                if (layout.precedes(earlier, row))
                {
                    history = fakeMix(history, keysThisForward[static_cast<size_t>(earlier - rows.first)]);
                }
            }
            int64_t const cacheIndex = past + (row - rows.first);
            auto& value = hidden[static_cast<size_t>(row)];
            for (size_t channel = 0; channel < kFakeHiddenChannels; ++channel)
            {
                value[channel] = fakeMix(value[channel], fakeMix(history, static_cast<int32_t>(channel) + layerIdx));
            }
            int32_t const key = fakeFold(value);
            storeKVEntry(pool, layerIdx, geometry, pageTable, slot, cacheIndex, FakeStateKind::kKey, key, stream);
            storeKVEntry(pool, layerIdx, geometry, pageTable, slot, cacheIndex, FakeStateKind::kValue,
                fakeMix(key, static_cast<int32_t>(cacheIndex % kFakeModulus)), stream);
            keysThisForward.push_back(key);
        }
    }

    //! Fold every entry this sequence stored before `upTo`, read back through the page table.
    static int32_t foldHistory(rt::Tensor const& pool, FakeKVGeometry const& geometry,
        std::vector<int32_t> const& pageTable, int64_t slot, int64_t upTo, cudaStream_t stream)
    {
        int32_t folded{0};
        for (int64_t page = 0; page * geometry.tokensPerPage < upTo && page < geometry.pagesPerSequence; ++page)
        {
            auto const index = geometry.pageTableIndex(slot, /*isValue=*/false, page);
            if (index >= pageTable.size())
            {
                break;
            }
            auto const entries = FakeKVWriter::readPage(pool, geometry, pageTable[index], stream);
            int64_t const last = std::min<int64_t>(geometry.tokensPerPage, upTo - page * geometry.tokensPerPage);
            for (int64_t token = 0; token < last; ++token)
            {
                folded = fakeMix(folded, toFakeValue(entries[static_cast<size_t>(token)]));
            }
        }
        return folded;
    }

    void storeKVEntry(rt::Tensor& pool, int32_t layerIdx, FakeKVGeometry const& geometry,
        std::vector<int32_t> const& pageTable, int64_t slot, int64_t position, FakeStateKind kind, int32_t computed,
        cudaStream_t stream) const
    {
        FakeStateSlot const stateSlot{layerIdx, position, kind};
        FakeKVWriter::write(pool, geometry, pageTable, slot, stateSlot, storedValue(stateSlot, computed), stream);
    }

    //! Advance every recurrent layer's state for the sequences in this forward.
    //!
    //! A hybrid model reaches here and through advanceAttentionLayers() on the same forward: which cache a layer
    //! owns is a property of the layer, not of the model, so both run whenever the topology has both kinds.
    void advanceRecurrentLayers(std::vector<FakeHidden>& hidden, FakeForwardLayout const& layout, cudaStream_t stream)
    {
        if (mComposition.recurrentLayers == 0)
        {
            return;
        }
        auto const stateIndices = snapshotInt32Binding(*mMap, binding_names::kStateIndices, stream);
        if (stateIndices.empty())
        {
            return;
        }
        for (int32_t layerIdx = 0; layerIdx < mComposition.recurrentLayers; ++layerIdx)
        {
            for (size_t slot = 0; slot < stateIndices.size() && static_cast<int64_t>(slot) < layout.batch(); ++slot)
            {
                advanceRecurrentSlot(layerIdx, stateIndices[slot], layout, static_cast<int64_t>(slot), hidden, stream);
            }
        }
    }

    //! Fold this forward's tokens into one sequence's recurrent and convolution state, left to right.
    //!
    //! Both are read back before they are written, so a row `state_indices` points somewhere else -- a swapped or
    //! recycled slot -- carries the wrong history into the output rather than merely holding the wrong number.
    void advanceRecurrentSlot(int32_t layerIdx, int32_t stateIndex, FakeForwardLayout const& layout, int64_t slot,
        std::vector<FakeHidden>& hidden, cudaStream_t stream)
    {
        rt::Tensor* const recurrentPool = mMap->get(mPresentRecurrentNames[static_cast<size_t>(layerIdx)]);
        rt::Tensor* const convPool = mMap->get(mPresentConvNames[static_cast<size_t>(layerIdx)]);
        if (recurrentPool == nullptr || convPool == nullptr)
        {
            return;
        }
        auto const rows = layout.rowsOf(slot);
        if (rows.first >= rows.second)
        {
            return; // This sequence contributed no token to this forward, so its state does not move.
        }
        int64_t const last = layout.pastOf(slot) + (rows.second - rows.first) - 1;
        int32_t recurrent = toFakeValue(FakeRecurrentWriter::read(*recurrentPool, stateIndex, stream));
        int32_t convolution = toFakeValue(FakeRecurrentWriter::read(*convPool, stateIndex, stream));
        for (int32_t row = rows.first; row < rows.second; ++row)
        {
            auto& value = hidden[static_cast<size_t>(row)];
            recurrent = fakeMix(recurrent, fakeMix(fakeFold(value), layerIdx));
            convolution = fakeMix(convolution, recurrent);
            for (size_t channel = 0; channel < kFakeHiddenChannels; ++channel)
            {
                value[channel] = fakeMix(value[channel], recurrent);
            }
        }
        FakeStateSlot const recurrentSlot{layerIdx, last, FakeStateKind::kRecurrent};
        FakeStateSlot const convSlot{layerIdx, last, FakeStateKind::kConv};
        FakeRecurrentWriter::write(*recurrentPool, stateIndex, storedValue(recurrentSlot, recurrent), stream);
        FakeRecurrentWriter::write(*convPool, stateIndex, storedValue(convSlot, convolution), stream);
    }

    //! What actually gets stored for an entry: what the engine computed, unless a test forced the value.
    double storedValue(FakeStateSlot const& slot, int32_t computed) const
    {
        return mForcedStateValues ? mForcedStateValues(slot) : static_cast<double>(computed);
    }

    //! Write the logits this forward implies, honouring any forced tokens.
    //!
    //! `logits_indices` selects which physical token row each logits row reads from -- a prefill emits one logits
    //! row for the whole prompt, so the two are not the same index.
    void writeOutputLogits(std::vector<FakeHidden> const& hidden, cudaStream_t stream)
    {
        rt::Tensor* const logits = mMap->get(binding_names::kLogits);
        if (logits == nullptr)
        {
            return;
        }
        auto const volume = logits->getShape().volume();
        auto const rows = volume / kVocabSize;
        auto const sources = snapshotInt64Binding(*mMap, binding_names::kLogitsIndices, stream);
        auto const& forced = forcedTokensThisRound();

        std::vector<float> host(static_cast<size_t>(volume), 0.0F);
        for (int64_t row = 0; row < rows; ++row)
        {
            int32_t token = static_cast<size_t>(row) < forced.size() ? forced[static_cast<size_t>(row)] : -1;
            if (token < 0)
            {
                token = computedToken(hidden, sources, row);
            }
            // A row with nothing behind it is peaked at kDefaultRowToken rather than left flat: it decodes to
            // the same token either way, and a row of equal entries is a fixture no ranking assertion can check.
            int32_t const winner = token >= 0 ? token : kDefaultRowToken;
            for (int64_t entry = 0; entry < kVocabSize; ++entry)
            {
                host[static_cast<size_t>(row * kVocabSize + entry)]
                    = fakeOutputValue(entry, winner, kVocabSize, mOutputScale);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            logits->rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    //! The token a logits row decodes to, or -1 to leave the row zeroed -- which the sampler resolves to its stable
    //! tie-break token, the behaviour a row with nothing behind it had before this engine computed anything.
    static int32_t computedToken(
        std::vector<FakeHidden> const& hidden, std::vector<int64_t> const& sources, int64_t row)
    {
        auto const source = static_cast<size_t>(row) < sources.size()
            ? static_cast<size_t>(sources[static_cast<size_t>(row)])
            : static_cast<size_t>(row);
        if (source >= hidden.size())
        {
            return -1;
        }
        return kFirstComputedToken
            + fakeFold(hidden[source]) % (static_cast<int32_t>(kVocabSize) - kFirstComputedToken);
    }

    std::vector<int32_t> const& forcedTokensThisRound() const
    {
        static std::vector<int32_t> const kNone;
        if (mForcedOutputTokens.empty())
        {
            return kNone;
        }
        return mForcedOutputTokens[std::min(mExecuteCount, mForcedOutputTokens.size() - 1)];
    }

    //! One value per row of a state pool, taken from each row's first channel: the writer stores a single number
    //! across every channel of a row, so the first is representative.
    static std::vector<double> snapshotStatePool(rt::TensorMap const& map, std::string const& name, cudaStream_t stream)
    {
        rt::Tensor* const pool = map.get(name);
        if (pool == nullptr || pool->getShape().getNumDims() == 0)
        {
            return {};
        }
        std::vector<double> rows;
        for (int64_t row = 0; row < pool->getShape()[0]; ++row)
        {
            rows.push_back(FakeRecurrentWriter::read(*pool, static_cast<int32_t>(row), stream));
        }
        return rows;
    }

    //! A whole floating-point binding, converted to double. One copy, whatever the dtype: inputs_embeds is read in
    //! full on every forward, so a per-element transfer would dominate the suite's runtime.
    static std::vector<double> readFloatBinding(rt::Tensor const& tensor, cudaStream_t stream)
    {
        auto const volume = tensor.getShape().volume();
        switch (tensor.getDataType())
        {
        case trt_edgellm::DataType::kHALF: return convertBinding<half>(tensor, volume, stream, __half2float);
        case trt_edgellm::DataType::kBF16:
            return convertBinding<__nv_bfloat16>(tensor, volume, stream, __bfloat162float);
        case trt_edgellm::DataType::kFLOAT:
            return convertBinding<float>(tensor, volume, stream, [](float value) { return value; });
        default: throw std::runtime_error("FakeEngineExecutor: unsupported floating-point binding dtype");
        }
    }

    template <typename T, typename Convert>
    static std::vector<double> convertBinding(
        rt::Tensor const& tensor, int64_t volume, cudaStream_t stream, Convert convert)
    {
        std::vector<T> raw(static_cast<size_t>(volume));
        CUDA_CHECK(
            cudaMemcpyAsync(raw.data(), tensor.rawPointer(), raw.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> host(raw.size());
        std::transform(
            raw.begin(), raw.end(), host.begin(), [&](T value) { return static_cast<double>(convert(value)); });
        return host;
    }

    //! logits_indices is the one INT64 binding the decode path needs read back, so this stays separate from the
    //! INT32-only snapshotInt32Binding() rather than loosening its dtype check.
    static std::vector<int64_t> snapshotInt64Binding(
        rt::TensorMap const& map, std::string const& name, cudaStream_t stream)
    {
        rt::Tensor* const tensor = map.get(name);
        if (tensor == nullptr || tensor->getDataType() != trt_edgellm::DataType::kINT64)
        {
            return {};
        }
        std::vector<int64_t> host(static_cast<size_t>(tensor->getShape().volume()));
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), tensor->rawPointer(), host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }

    static std::vector<int32_t> snapshotInt32Binding(
        rt::TensorMap const& map, std::string const& name, cudaStream_t stream)
    {
        rt::Tensor* const tensor = map.get(name);
        if (tensor == nullptr)
        {
            return {};
        }
        if (tensor->getDataType() != trt_edgellm::DataType::kINT32)
        {
            throw std::runtime_error("FakeEngineExecutor::recordBinding(\"" + name
                + "\"): only INT32 bindings can be recorded, and this one is not. Reading it as INT32 would report "
                  "plausible-looking garbage instead of failing.");
        }
        auto const volume = tensor->getShape().volume();
        std::vector<int32_t> host(static_cast<size_t>(volume));
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), tensor->rawPointer(), host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }

    static constexpr int64_t kContextMemoryBytes{4096};

    FakeLayerComposition mComposition;
    FakeStateGenerator mForcedStateValues;
    FakeOutputScale mOutputScale{FakeOutputScale::kLogits};
    std::vector<std::string> mBindingNames;
    std::vector<std::string> mPastKvNames;
    std::vector<std::string> mPresentKvNames;
    std::vector<std::string> mPresentRecurrentNames;
    std::vector<std::string> mPresentConvNames;
    std::unordered_set<std::string> mBindingSet;
    rt::TensorMap const* mMap{nullptr};
    std::vector<std::vector<int32_t>> mForcedOutputTokens;
    size_t mExecuteCount{0};
    size_t mForwardsRun{0};
    std::vector<PreparedRound> mPreparedRounds;
    std::map<std::string, BindingRecord> mRecordedBindings;
    std::map<std::string, std::vector<std::vector<double>>> mRecordedStatePools;
    std::map<std::string, std::vector<std::vector<double>>> mRecordedFloatBindings;
    bool mAdvertisePhase8{false};
    int64_t mDecodeProfileMaxPhase{static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode)};
};

//! A fake engine for the deployment `config` describes, ready to hand to makeVanillaArtifacts(). Defaulted to the
//! tiny vanilla config, which is the one every text suite here stages; pass the config explicitly when the topology
//! under test differs from the staged one.
inline std::unique_ptr<FakeEngineExecutor> makeFakeEngine(Json const& config = makeTinyVanillaConfig())
{
    return std::make_unique<FakeEngineExecutor>(FakeLayerComposition::fromConfigJson(config));
}

} // namespace substitute_engine
