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

// The values a fake engine carries: which cache entry a number belongs to, and the arithmetic that produces it.
//
// Nothing here knows about the runtime. It is separate from the engine so that a test reading entries back, or a
// test supplying its own entry values, does not have to pull in the executor.

#include "substituteEngine.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>

namespace substitute_engine
{

using namespace trt_edgellm;

enum class FakeStateKind
{
    kKey,
    kValue,
    kRecurrent,
    kConv,
};

//! One entry of one layer's per-sequence cache, named the way a test reasons about it.
//!
//! Deliberately not the offset the value ends up at: which page a token lands on, and which pool row a sequence
//! occupies, are scheduling decisions. A test asking "does this sequence's state still hold its own values" has to
//! identify an entry by what it is rather than by where it was put.
struct FakeStateSlot
{
    //! Index among layers of this kind. Attention and recurrent layers are numbered separately, matching the
    //! binding names (see formatKVCacheName and formatRecurrentStateName in common/bindingNames.h).
    int32_t layerIdx{};
    //! For a KV entry, the token's absolute position. For a recurrent or conv entry -- one row per sequence, not
    //! per token -- the last position that sequence has consumed, which is what makes a stale or swapped row
    //! visible.
    int64_t position{};
    FakeStateKind kind{};
};

//! What the fake engine stores for one entry. Every channel of the entry gets this one number: misplacing an entry
//! across layers, positions or sequences is a scheduling bug a test can catch, while misplacing a channel within
//! one entry is kernel-internal and not something this substitute models.
//!
//! The engine derives this value itself. Supplying one overrides that, for a test whose subject is where entries
//! land rather than what they hold -- see FakeEngineExecutor::forceStateValues().
using FakeStateGenerator = std::function<double(FakeStateSlot const& slot)>;

//! A value keyed only by which entry it is, ignoring the sequence's content.
//!
//! Prime coefficients so that a one-step error in layer, position, or kind cannot alias another entry's value.
//! Useful as a forced generator when a test wants entries it can name without running the arithmetic.
inline double slotKeyedStateValue(FakeStateSlot const& slot)
{
    return static_cast<double>(
        (slot.layerIdx * 11 + slot.position * 3 + static_cast<int32_t>(slot.kind) * 17 + 1) % kFakeModulus);
}

//! Channels in the hidden value the fake engine carries from layer to layer.
//!
//! More than one so that a layer update cannot collapse two different histories onto the same hidden value as
//! readily as a single number would; few enough to stay cheap, since this runs on the host once per token per
//! layer.
inline constexpr size_t kFakeHiddenChannels{4};
using FakeHidden = std::array<int32_t, kFakeHiddenChannels>;

//! One mixing step of the fake engine's arithmetic.
//!
//! The requirement is not realism, it is sensitivity plus exact reproducibility: every input has to be able to
//! change the result, and the same inputs must give bit-identical results however the runtime chose to batch or
//! chunk the work, or the equivalence tests built on this engine compare nearly-equal numbers and prove nothing.
//! Integers under a prime modulus give both -- there is no rounding to reassociate, and the value neither grows
//! nor decays over the hundreds of steps a decode loop takes.
inline int32_t fakeMix(int32_t accumulated, int32_t incoming)
{
    auto const mixed = (static_cast<int64_t>(accumulated) * 31 + static_cast<int64_t>(incoming) * 7 + 1) % kFakeModulus;
    return static_cast<int32_t>(mixed < 0 ? mixed + kFakeModulus : mixed);
}

//! Collapse a hidden value into the single number an entry or a token id is derived from.
inline int32_t fakeFold(FakeHidden const& hidden)
{
    int32_t folded{0};
    for (auto const channel : hidden)
    {
        folded = fakeMix(folded, channel);
    }
    return folded;
}

//! Bring a value read back out of a pool into the engine's integer range. Rounds rather than truncates: the value
//! went out through fp16 or bf16, and the engine only ever stores integers this modulus keeps exact there.
inline int32_t toFakeValue(double value)
{
    auto const rounded = std::llround(value) % kFakeModulus;
    return static_cast<int32_t>(rounded < 0 ? rounded + kFakeModulus : rounded);
}

//! The paged KV pool's geometry, read off the bound tensors rather than restated from the config.
//!
//! The pool is `[2, pages, tokensPerPage, heads, channels]`, and a physical page id addresses it flat: id `p` is
//! `(p / pages, p % pages)`, which is the same `p * tokensPerPage` element stride either way. The page table is
//! `[batch, 2, pagesPerSequence]`, keys first and values second (see KVPageTable in cpp/runtime/state).

} // namespace substitute_engine
