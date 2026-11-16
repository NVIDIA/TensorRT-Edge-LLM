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

#include "common/tensor.h"
#include "runtime/recurrentStateTypes.h" // StateKind, stateKindName, StateSpec

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Dense per-slot store for the recurrent axis: mixer recurrent and convolution states plus any
//! auxiliary convolution state a layer carries. Rows are resident slots, the index space
//! `state_indices` carries into the engine. Each state is one contiguous slab whose row stride is
//! its own per-slot byte size, which is what makes `slot * bytesPerSlot` addressing valid without
//! consulting tensor strides.
//!
//! State order is the order the specs were given in and never changes, because `capture` and
//! `restore` match a checkpoint to the pool positionally.
//!
//! The store carries no mixer-specific behaviour: it knows a state's geometry, not how the state
//! evolves or how speculative decoding commits it.
class RecurrentStateManager
{
public:
    //! Bytes one slot occupies across `specs`. Pool sizing runs against a byte budget before any
    //! allocation exists, so this is a property of the specs alone.
    static size_t bytesPerSlot(std::vector<StateSpec> const& specs);

    RecurrentStateManager() noexcept = default;

    //! Allocate one zeroed [slotCount] + shape slab per spec. No spec or no slot allocates nothing.
    //! @param specs Per-slot states in the order checkpoints will match them.
    //! @param slotCount Resident slot rows; may be 0, which leaves every slot index invalid.
    //! @param name Prefix for the slab tensor names, which surface in allocation diagnostics.
    //! @param stream CUDA stream for the zeroing memsets.
    //! @throws std::runtime_error if a spec has non-positive per-slot volume, names a negative
    //!         layer, or repeats a {kind, absLayerIdx} already claimed.
    RecurrentStateManager(
        std::vector<StateSpec> specs, int32_t slotCount, std::string const& name, cudaStream_t stream);

    RecurrentStateManager(RecurrentStateManager const&) = delete;
    RecurrentStateManager& operator=(RecurrentStateManager const&) = delete;
    RecurrentStateManager(RecurrentStateManager&&) noexcept = default;
    RecurrentStateManager& operator=(RecurrentStateManager&&) noexcept = default;

    //! @brief Number of resident slot rows.
    int32_t slotCount() const noexcept;

    //! @brief Number of per-slot states held.
    int32_t numStates() const noexcept;

    //! @brief The spec one state was allocated from.
    //! @param stateIdx Index in [0, numStates()).
    StateSpec const& spec(int32_t stateIdx) const;

    //! @brief Every spec, in pool order.
    std::vector<StateSpec> const& specs() const noexcept;

    //! @brief Storage for one state, shaped [slotCount] + spec.shape.
    //! @param stateIdx Index in [0, numStates()).
    Tensor& state(int32_t stateIdx);
    Tensor const& state(int32_t stateIdx) const;

    //! @brief Index of the state with the given lineage, or -1 when the layer declares none.
    int32_t findState(StateKind kind, int32_t absLayerIdx) const;

    //! @brief Bytes one slot occupies in one state.
    //! @param stateIdx Index in [0, numStates()).
    size_t stateBytesPerSlot(int32_t stateIdx) const;

    //! @brief Bytes one slot occupies across every state held.
    size_t bytesPerSlot() const;

    //! Zero every slot of every state.
    void clear(cudaStream_t stream);

    //! Zero one slot of every state, leaving every other slot untouched.
    void clearSlot(int32_t slot, cudaStream_t stream);

    //! Copy one slot out into freshly allocated [1] + shape tensors, one per state in pool order.
    std::vector<Tensor> capture(int32_t slot, cudaStream_t stream);

    //! Write a checkpoint back into one slot. States beyond `saved.size()` are zeroed rather than
    //! left alone, so a checkpoint taken before a state existed resumes that state from zero
    //! instead of from the previous occupant's data. Nothing is written until every entry has been
    //! validated against its spec.
    void restore(std::vector<Tensor> const& saved, int32_t slot, cudaStream_t stream);

    //! Copy one slot of every state from a pool holding the same states in the same order. Pairs a
    //! live pool with a snapshot pool of a different slot count.
    void copySlotFrom(
        RecurrentStateManager const& source, int32_t destinationSlot, int32_t sourceSlot, cudaStream_t stream);

private:
    void validateSlot(int32_t slot) const;

    std::vector<StateSpec> mSpecs;
    //! Bytes one slot occupies in each state, parallel to mSpecs.
    std::vector<size_t> mSlotBytes;
    //! One slab per state, parallel to mSpecs. Empty when nothing was allocated.
    std::vector<Tensor> mStates;
    int32_t mSlotCount{};
};

} // namespace rt
} // namespace trt_edgellm
