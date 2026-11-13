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

#include "runtime/commitPolicy/recurrentCommitPolicyFactory.h"
#include "common/bindingNames.h"
#include "runtime/commitPolicy/gdnConvCommitPolicy.h"
#include "runtime/commitPolicy/mamba2CommitPolicy.h"
#include "runtime/exec/tensorRegistry.h"

#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

std::unique_ptr<RecurrentCommitPolicy> makeRecurrentCommitPolicy(StateKind mixerKind,
    RecurrentCommitPolicy::Config const& config, std::vector<rt::Tensor>& liveRecurrent,
    std::vector<rt::Tensor>& liveConv, cudaStream_t stream)
{
    switch (mixerKind)
    {
    case StateKind::kMamba2: return std::make_unique<Mamba2CommitPolicy>(config, liveRecurrent, liveConv, stream);
    case StateKind::kGdn: return std::make_unique<GdnConvCommitPolicy>(config, liveRecurrent, liveConv, stream);
    case StateKind::kConv:
    case StateKind::kNgramConv: break;
    }
    throw std::runtime_error(std::string("makeRecurrentCommitPolicy: ") + stateKindName(mixerKind)
        + " is not a recurrent mixer; only gdn and mamba2 commit spec-verify state.");
}

void declareRecurrentSpecVerifyBindings(
    StateKind mixerKind, RecurrentCommitPolicy::Config const& config, int32_t recurrentLayerIdx, TensorRegistry& reg)
{
    auto addSpecVerifyTensor = [&](std::string name, trt_edgellm::DataType dtype, std::vector<ShapeDim> const& shape) {
        reg.addTensor({std::move(name), TensorIO::kOutput, dtype, shape});
    };

    switch (mixerKind)
    {
    case StateKind::kMamba2:
        // Minimal per-token replay inputs the verify pass stashes; the accepted recurrent state is
        // reconstructed from them rather than snapshotted.
        addSpecVerifyTensor(binding_names::formatReplayDaStateName(recurrentLayerIdx), trt_edgellm::DataType::kFLOAT,
            {sym(&InferenceDims::seqLen), fixed(config.recurrentStateNumHeads)});
        addSpecVerifyTensor(binding_names::formatReplayUStateName(recurrentLayerIdx), trt_edgellm::DataType::kFLOAT,
            {sym(&InferenceDims::seqLen), fixed(config.recurrentStateNumHeads), fixed(config.recurrentStateHeadDim)});
        addSpecVerifyTensor(binding_names::formatReplayBStateName(recurrentLayerIdx), trt_edgellm::DataType::kFLOAT,
            {sym(&InferenceDims::seqLen), fixed(config.recurrentStateNumGroups), fixed(config.recurrentStateSize)});
        addSpecVerifyTensor(binding_names::formatReplayDtStateName(recurrentLayerIdx), trt_edgellm::DataType::kFLOAT,
            {sym(&InferenceDims::seqLen), fixed(config.recurrentStateNumHeads)});
        break;
    case StateKind::kGdn:
        // The engine declares one full recurrent state per verify token; the plugin writes only the replay stash rows
        // (gdnTreeVerifyRowBytes) in place, and the runtime binds a buffer of that size.
        addSpecVerifyTensor(binding_names::formatIntermediateRecurrentStateName(recurrentLayerIdx),
            config.recurrentStateType,
            {sym(&InferenceDims::seqLen), fixed(config.recurrentStateNumHeads), fixed(config.recurrentStateHeadDim),
                fixed(config.recurrentStateSize)});
        break;
    case StateKind::kConv:
    case StateKind::kNgramConv:
        throw std::runtime_error(std::string("declareRecurrentSpecVerifyBindings: ") + stateKindName(mixerKind)
            + " is not a recurrent mixer.");
    }

    if (config.convDim > 0 && config.convKernel > 0)
    {
        addSpecVerifyTensor(binding_names::formatIntermediateConvStateName(recurrentLayerIdx), config.convStateType,
            {sym(&InferenceDims::seqLen), fixed(config.convDim), fixed(config.convKernel)});
    }
}

} // namespace rt
} // namespace trt_edgellm
