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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Mixer lineage of a per-slot state. A kind occurs at most once per layer, so {kind, absLayerIdx}
//! names exactly one state and spells its engine binding as `{kind}_state_{absLayerIdx}`.
//!
//! This light-weight type vocabulary is shared by the recurrent-axis state store
//! (`RecurrentStateManager`), the per-kind spec-verify commit policy (`RecurrentCommitPolicy`) and
//! the engine config layer schema (`LayerConfig`). It lives in its own header so those files can
//! name a state's lineage without pulling in the full store, which depends on the commit policy.
enum class StateKind : int32_t
{
    kGdn,
    kMamba2,
    kConv,
    kNgramConv,
};

//! Schema spelling of a kind, shared with the `layers[].states[].kind` the frontends emit.
char const* stateKindName(StateKind kind);

//! One per-slot state. `shape` is a single slot's row, so the slab holding it is [slotCount] + shape
//! and a checkpoint of it is [1] + shape.
struct StateSpec
{
    StateKind kind{};
    int32_t absLayerIdx{};
    Coords shape;
    trt_edgellm::DataType dataType{trt_edgellm::DataType::kHALF};
};

} // namespace rt
} // namespace trt_edgellm
