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

#include "runtime/config/llmEngineConfig.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! The kinds of phase-aware layers whose plugins decide phase-8 support, in the order a gap reports them.
enum class CompactLayerKind : int32_t
{
    kAttention,
    kQsaAttention,
    kGatedDeltaNet,
    kMamba2Ssd,
    kCausalConv1d,
    kNgramConv,
};

//! A layer kind of an engine that cannot run phase 8, and why.
struct CompactLayerGap
{
    CompactLayerKind kind;
    std::string reason;
};

//! @brief The layer kinds of @p config that cannot execute compact (phase-8) steps on SM @p sm, in kind order.
//!
//! Each kind present in the engine is checked with the predicate its plugin applies when it reserves the phase-8
//! workspace and accepts a phase-8 enqueue (kernels/compactPhase/compactPhaseSupport.h), fed with the layer shapes
//! the engine config records. The reason names the shape, why it has no phase-8 path, and the SMs that have one.
//!
//! Inputs the config does not record are taken from what the export paths produce: no attention layer is a tree,
//! vision-block or attention-sink layer (speculative and vision-block deployments are refused before this check;
//! sinks are only exported for DSpark drafts), and a KV-shared bounded-SWA-capable layer receives the donor's
//! current K/V. A sliding-window layer that is not bounded-SWA-capable is checked as a causal one; the kernel
//! predicates accept the two masks alike.
std::vector<CompactLayerGap> compactLayerGaps(LLMEngineConfig const& config, int32_t sm);

//! @brief Why the layers of @p config cannot all execute compact (phase-8) steps on SM @p sm, or empty when they can:
//! every compactLayerGaps() entry, as one sentence naming the SM.
std::string compactExecutionGap(LLMEngineConfig const& config, int32_t sm);

} // namespace rt
} // namespace trt_edgellm
