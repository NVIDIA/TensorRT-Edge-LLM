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

#include "runtime/recurrentStateManager.h"

#include "common/cudaMacros.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm::rt;
using trt_edgellm::DataType;

namespace
{

//! A GDN layer carrying an n-gram injection: three states of deliberately unequal row sizes and two
//! distinct element types, so a store that assumed one global geometry cannot pass.
std::vector<StateSpec> makeHeterogeneousSpecs()
{
    return {
        StateSpec{StateKind::kGdn, /*.absLayerIdx=*/1, Coords{2, 3, 4}, DataType::kFLOAT},
        StateSpec{StateKind::kConv, /*.absLayerIdx=*/1, Coords{5, 6}, DataType::kHALF},
        StateSpec{StateKind::kNgramConv, /*.absLayerIdx=*/1, Coords{7, 9}, DataType::kHALF},
    };
}

//! Distinct non-zero byte pattern per state, so a copy that reached the wrong slab is visible.
uint8_t patternFor(int32_t stateIdx, int32_t slot)
{
    return static_cast<uint8_t>(0x10 + 0x20 * stateIdx + slot);
}

void fillSlot(RecurrentStateManager& store, int32_t stateIdx, int32_t slot, uint8_t pattern, cudaStream_t stream)
{
    size_t const bytes = store.stateBytesPerSlot(stateIdx);
    CUDA_CHECK(
        cudaMemsetAsync(static_cast<uint8_t*>(store.state(stateIdx).rawPointer()) + static_cast<size_t>(slot) * bytes,
            pattern, bytes, stream));
}

std::vector<uint8_t> readSlot(RecurrentStateManager& store, int32_t stateIdx, int32_t slot)
{
    size_t const bytes = store.stateBytesPerSlot(stateIdx);
    std::vector<uint8_t> host(bytes);
    EXPECT_EQ(cudaMemcpy(host.data(),
                  static_cast<uint8_t const*>(store.state(stateIdx).rawPointer()) + static_cast<size_t>(slot) * bytes,
                  bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    return host;
}

bool allBytesAre(std::vector<uint8_t> const& bytes, uint8_t value)
{
    return !bytes.empty() && std::all_of(bytes.begin(), bytes.end(), [value](uint8_t byte) { return byte == value; });
}

} // namespace

// Given a pool whose states have unequal row shapes and unequal element types, when one slot's
// footprint is priced, then it is the sum of each state's own row bytes. Pool sizing runs against a
// device-memory budget before any allocation exists, so this is the only number a caller can size
// from; charging one global geometry for every state would undercount a heterogeneous model and
// overrun the declared ceiling.
TEST(RecurrentStateManagerTests, PricesEachStatesOwnRowIntoTheSlot)
{
    std::vector<StateSpec> const specs = makeHeterogeneousSpecs();
    size_t const expected = static_cast<size_t>(2 * 3 * 4) * sizeof(float) + static_cast<size_t>(5 * 6) * sizeof(half)
        + static_cast<size_t>(7 * 9) * sizeof(half);
    EXPECT_EQ(RecurrentStateManager::bytesPerSlot(specs), expected);
}

// Given a slot holding data in every state, when it is captured and then restored over a zeroed
// slot, then every state comes back byte for byte. This is the contract the system prompt cache
// rests on: a reused prefix that resumed from a partially restored recurrent state is silent, since
// the shapes still match and only the continuation diverges.
TEST(RecurrentStateManagerTests, CaptureAndRestoreRoundTripIsByteExact)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/3, "test", stream);
    int32_t constexpr kSlot{1};

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        fillSlot(store, stateIdx, kSlot, patternFor(stateIdx, kSlot), stream);
    }

    std::vector<Tensor> const saved = store.capture(kSlot, stream);
    store.clearSlot(kSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        ASSERT_TRUE(allBytesAre(readSlot(store, stateIdx, kSlot), 0U))
            << "clearSlot must zero state " << stateIdx << ", or the restore below proves nothing";
    }

    store.restore(saved, kSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        EXPECT_TRUE(allBytesAre(readSlot(store, stateIdx, kSlot), patternFor(stateIdx, kSlot)))
            << "state " << stateIdx << " did not round-trip";
    }
}

// Given every slot of every state holds a distinct pattern, when one slot is captured, zeroed and
// restored, then no other slot changes. Each state's row stride is its own, so a store that offset
// every slab by one shared per-slot size would write into a neighbouring sequence's state -- which
// corrupts an unrelated in-flight request rather than the one being restored.
TEST(RecurrentStateManagerTests, RestoreLeavesOtherSlotsAlone)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/3, "test", stream);
    int32_t constexpr kRestoredSlot{1};

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        for (int32_t slot = 0; slot < store.slotCount(); ++slot)
        {
            fillSlot(store, stateIdx, slot, patternFor(stateIdx, slot), stream);
        }
    }

    std::vector<Tensor> const saved = store.capture(kRestoredSlot, stream);
    store.clearSlot(kRestoredSlot, stream);
    store.restore(saved, kRestoredSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        for (int32_t slot = 0; slot < store.slotCount(); ++slot)
        {
            EXPECT_TRUE(allBytesAre(readSlot(store, stateIdx, slot), patternFor(stateIdx, slot)))
                << "state " << stateIdx << " slot " << slot << " changed";
        }
    }
}

// Given every slot of every state holds a distinct pattern, when one slot is cleared, then only that
// slot is zeroed. Retiring one resident row must not disturb the sequences still occupying the pool.
TEST(RecurrentStateManagerTests, ClearSlotLeavesOtherSlotsAlone)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/3, "test", stream);
    int32_t constexpr kClearedSlot{2};

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        for (int32_t slot = 0; slot < store.slotCount(); ++slot)
        {
            fillSlot(store, stateIdx, slot, patternFor(stateIdx, slot), stream);
        }
    }

    store.clearSlot(kClearedSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        EXPECT_TRUE(allBytesAre(readSlot(store, stateIdx, kClearedSlot), 0U))
            << "state " << stateIdx << " was not cleared";
        for (int32_t slot = 0; slot < kClearedSlot; ++slot)
        {
            EXPECT_TRUE(allBytesAre(readSlot(store, stateIdx, slot), patternFor(stateIdx, slot)))
                << "state " << stateIdx << " slot " << slot << " was cleared too";
        }
    }
}

// Given a checkpoint that covers only the leading states, when it is restored, then the states it
// omits are zeroed rather than left alone. A checkpoint predating a state's existence must resume
// that state from zero; inheriting the slot's previous occupant would leak one sequence's history
// into another.
TEST(RecurrentStateManagerTests, ShortCheckpointZeroesTheStatesItOmits)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/2, "test", stream);
    int32_t constexpr kSlot{0};

    for (int32_t stateIdx = 0; stateIdx < store.numStates(); ++stateIdx)
    {
        fillSlot(store, stateIdx, kSlot, patternFor(stateIdx, kSlot), stream);
    }

    std::vector<Tensor> saved = store.capture(kSlot, stream);
    saved.resize(1);
    store.restore(saved, kSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(allBytesAre(readSlot(store, 0, kSlot), patternFor(0, kSlot)));
    for (int32_t stateIdx = 1; stateIdx < store.numStates(); ++stateIdx)
    {
        EXPECT_TRUE(allBytesAre(readSlot(store, stateIdx, kSlot), 0U))
            << "state " << stateIdx << " kept the slot's previous contents";
    }
}

// Given a checkpoint whose trailing entry does not match its state's geometry, when it is restored,
// then the call throws and the slot is left exactly as it was. Validating entry by entry as it
// copies would leave the slot half-written -- a state set no sequence ever held, and one the caller
// cannot recover from by retrying.
TEST(RecurrentStateManagerTests, RejectsMismatchedCheckpointWithoutWritingAnything)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/2, "test", stream);
    int32_t constexpr kSlot{0};

    std::vector<Tensor> saved = store.capture(kSlot, stream);
    ASSERT_GE(saved.size(), 2U);
    CUDA_CHECK(cudaMemsetAsync(saved[0].rawPointer(), 0x7E, store.stateBytesPerSlot(0), stream));
    Coords const good = store.spec(1).shape;
    saved[1] = Tensor({1, good[0], good[1] + 1}, DeviceType::kGPU, store.spec(1).dataType, "drifted");

    store.clearSlot(kSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_THROW(store.restore(saved, kSlot, stream), std::runtime_error);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(allBytesAre(readSlot(store, 0, kSlot), 0U))
        << "the leading state was written before the trailing entry was rejected";
}

// Given a pool holding only an auxiliary n-gram convolution state and no mixer state at all, when a
// slot is captured and restored, then the state round-trips. A model may inject n-grams into
// attention-only layers, so the auxiliary states cannot be allocated or addressed off the presence
// of a recurrent mixer.
TEST(RecurrentStateManagerTests, AuxiliaryOnlyPoolRoundTripsWithoutAMixerState)
{
    cudaStream_t stream{};
    RecurrentStateManager store({StateSpec{StateKind::kNgramConv, /*.absLayerIdx=*/0, Coords{7, 9}, DataType::kHALF}},
        /*slotCount=*/2, "test", stream);
    int32_t constexpr kSlot{1};

    ASSERT_EQ(store.numStates(), 1);
    fillSlot(store, 0, kSlot, 0x4D, stream);
    std::vector<Tensor> const saved = store.capture(kSlot, stream);
    store.clearSlot(kSlot, stream);
    store.restore(saved, kSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(allBytesAre(readSlot(store, 0, kSlot), 0x4D));
}

// Given a live pool and a snapshot pool holding the same states at different slot counts, when one
// slot is copied across, then the destination slot matches the source and the destination's other
// slots do not change. The snapshot pool is sized by checkpoint budget and the live pool by batch
// size, so the two slot counts never agree and the copy cannot assume a shared row index space.
TEST(RecurrentStateManagerTests, CopiesOneSlotBetweenPoolsOfDifferentSlotCounts)
{
    cudaStream_t stream{};
    RecurrentStateManager live(makeHeterogeneousSpecs(), /*slotCount=*/2, "live", stream);
    RecurrentStateManager snapshots(makeHeterogeneousSpecs(), /*slotCount=*/5, "snapshots", stream);
    int32_t constexpr kLiveSlot{1};
    int32_t constexpr kSnapshotSlot{4};

    for (int32_t stateIdx = 0; stateIdx < live.numStates(); ++stateIdx)
    {
        fillSlot(live, stateIdx, kLiveSlot, patternFor(stateIdx, kLiveSlot), stream);
        fillSlot(snapshots, stateIdx, kSnapshotSlot - 1, 0x01, stream);
    }

    snapshots.copySlotFrom(live, kSnapshotSlot, kLiveSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t stateIdx = 0; stateIdx < snapshots.numStates(); ++stateIdx)
    {
        EXPECT_TRUE(allBytesAre(readSlot(snapshots, stateIdx, kSnapshotSlot), patternFor(stateIdx, kLiveSlot)))
            << "state " << stateIdx << " did not reach the snapshot slot";
        EXPECT_TRUE(allBytesAre(readSlot(snapshots, stateIdx, kSnapshotSlot - 1), 0x01))
            << "state " << stateIdx << " overwrote a neighbouring snapshot slot";
    }
}

// Given a pool that was configured with no slots, when any slot is addressed, then the call throws.
// The checkpoint pool is allocated from a budget that may price it to zero slots, and every
// lifecycle entry point must reject that pool rather than offset into an absent allocation.
TEST(RecurrentStateManagerTests, RejectsEverySlotOfAnUnallocatedPool)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/0, "test", stream);

    EXPECT_EQ(store.slotCount(), 0);
    EXPECT_EQ(store.numStates(), 3);
    EXPECT_EQ(store.bytesPerSlot(), RecurrentStateManager::bytesPerSlot(makeHeterogeneousSpecs()));
    EXPECT_THROW(store.clearSlot(0, stream), std::runtime_error);
    EXPECT_THROW(store.capture(0, stream), std::runtime_error);
    EXPECT_THROW(store.restore({}, 0, stream), std::runtime_error);
}

// Given a slot index past the end of the pool, when it is addressed, then the call throws rather
// than writing past the slab. Slot indices arrive from the resident-slot allocator, so an
// off-by-one there must surface here instead of silently corrupting the next state's storage.
TEST(RecurrentStateManagerTests, RejectsSlotIndexPastTheEndOfThePool)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/2, "test", stream);

    EXPECT_THROW(store.clearSlot(2, stream), std::runtime_error);
    EXPECT_THROW(store.clearSlot(-1, stream), std::runtime_error);
    EXPECT_THROW(store.capture(2, stream), std::runtime_error);
}

// Given a layer declared twice with the same state kind, when the pool is constructed, then it
// throws. The kind is the only discriminator a binding name carries, so two states sharing
// {kind, layer} would make `{kind}_state_{layer}` ambiguous and bind the engine to whichever the
// lookup happened to find first.
TEST(RecurrentStateManagerTests, RejectsTheSameKindDeclaredTwiceOnOneLayer)
{
    cudaStream_t stream{};
    std::vector<StateSpec> const duplicated{
        StateSpec{StateKind::kConv, /*.absLayerIdx=*/3, Coords{5, 6}, DataType::kHALF},
        StateSpec{StateKind::kConv, /*.absLayerIdx=*/3, Coords{5, 6}, DataType::kHALF},
    };
    EXPECT_THROW(RecurrentStateManager(duplicated, /*slotCount=*/1, "test", stream), std::runtime_error);

    std::vector<StateSpec> const distinctLayers{
        StateSpec{StateKind::kConv, /*.absLayerIdx=*/3, Coords{5, 6}, DataType::kHALF},
        StateSpec{StateKind::kConv, /*.absLayerIdx=*/4, Coords{5, 6}, DataType::kHALF},
    };
    EXPECT_NO_THROW(RecurrentStateManager(distinctLayers, /*slotCount=*/1, "test", stream));
}

// Given a state whose row shape has a zero extent, when the pool is constructed, then it throws.
// A zero-volume state would price at zero bytes per slot and leave `slot * bytesPerSlot` aliasing
// every slot onto the same address, so it has to be rejected at configuration rather than produce a
// pool that silently shares one row.
TEST(RecurrentStateManagerTests, RejectsAStateWithAnEmptyRow)
{
    cudaStream_t stream{};
    std::vector<StateSpec> const empty{StateSpec{StateKind::kConv, /*.absLayerIdx=*/0, Coords{5, 0}, DataType::kHALF}};
    EXPECT_THROW(RecurrentStateManager::bytesPerSlot(empty), std::runtime_error);
    EXPECT_THROW(RecurrentStateManager(empty, /*slotCount=*/1, "test", stream), std::runtime_error);
}

// Given a pool of states, when a state is looked up by its mixer lineage, then the lookup returns
// its pool position and reports absence for a layer that declares no such state. Consumers address
// states by {kind, absolute layer} straight from the engine binding name, with no local index space
// in between to validate against.
TEST(RecurrentStateManagerTests, FindsAStateByKindAndAbsoluteLayer)
{
    cudaStream_t stream{};
    RecurrentStateManager store(makeHeterogeneousSpecs(), /*slotCount=*/1, "test", stream);

    EXPECT_EQ(store.findState(StateKind::kGdn, 1), 0);
    EXPECT_EQ(store.findState(StateKind::kConv, 1), 1);
    EXPECT_EQ(store.findState(StateKind::kNgramConv, 1), 2);
    EXPECT_EQ(store.findState(StateKind::kConv, 0), -1);
    EXPECT_EQ(store.findState(StateKind::kMamba2, 1), -1);
}
