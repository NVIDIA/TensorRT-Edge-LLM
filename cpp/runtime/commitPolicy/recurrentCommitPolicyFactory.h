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

#include "runtime/commitPolicy/recurrentCommitPolicy.h"
#include "runtime/recurrentStateTypes.h"

#include <memory>

namespace trt_edgellm
{
namespace rt
{

class TensorRegistry;

//! Build the commit policy for a layer's mixer kind over references to its live recurrent/conv slabs.
//! @throws std::runtime_error when the kind is not a recurrent mixer (only kMamba2/kGdn commit).
std::unique_ptr<RecurrentCommitPolicy> makeRecurrentCommitPolicy(StateKind mixerKind,
    RecurrentCommitPolicy::Config const& config, std::vector<rt::Tensor>& liveRecurrent,
    std::vector<rt::Tensor>& liveConv, cudaStream_t stream);

//! Declare the spec-verify intermediate-state output bindings a recurrent layer of the given mixer
//! kind emits during prefill/verification, under the layer's local recurrent-layer index (0-based
//! over recurrent layers only, as the builder profiles and runtime tensor map key these names). In an
//! interleaved hybrid the local index differs from the absolute layer index; using the local index is
//! what makes the declared output names match the tensors the runtime binds. mamba2 declares the four
//! replay-stash outputs; gdn declares the single full-state snapshot. Both declare the conv snapshot
//! when the layer carries a convolution. The engine sees no runtime mode flag: the kind alone selects
//! the output set, matching the commit path the policy of that kind runs.
void declareRecurrentSpecVerifyBindings(
    StateKind mixerKind, RecurrentCommitPolicy::Config const& config, int32_t recurrentLayerIdx, TensorRegistry& reg);

} // namespace rt
} // namespace trt_edgellm
