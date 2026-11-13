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
#include "multimodal/common/multimodalRunner.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/externalWeightManager.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime_api.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

struct DeploymentOptions;

//! Validate the fixed recurrent verification ABI independently of the selected layout.
void validateRecurrentVerifyMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor);

//! Everything LLMInferenceRuntime reads off disk before it can assemble itself: the parsed deployment
//! configuration, the engines, and the weight-shaped files that sit next to them.
//!
//! Loading is separated from assembly so a runtime can also be built from artifacts that no engine directory
//! produced — a substitute EngineExecutor plus a hand-written DeploymentConfig is a complete input.
struct ModelArtifacts
{
    DeploymentConfig deployment;
    std::unique_ptr<EngineExecutor> baseExecutor;
    std::unique_ptr<EngineExecutor> draftExecutor; //!< Null unless the deployment drafts speculatively.
    ExternalWeightManager weights;
    //! Externalized draft-engine weights, already validated against `draftExecutor`. Empty for vanilla deployments,
    //! and also empty when the draft was exported without `--externalize-weights`. The selected speculative decoder
    //! publishes it into its own tensor map.
    ExternalWeightManager draftWeights;
    EmbeddingData embedding;
    std::optional<Tensor> vocabMap; //!< Reduced-vocab mapping table; absent unless the base vocab is reduced.
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
    //! Encoder runners the caller assembled (a substitute encoder, for instance). Each one set takes the place of
    //! the runner DeploymentOptions::multimodalEngineDir would load for it.
    std::unique_ptr<MultimodalRunner> visionRunner;
    std::unique_ptr<MultimodalRunner> audioRunner;

    //! Checkpoint directories that supplied `weights`. Retained because encoder runners load from them separately.
    std::filesystem::path checkpointDir;
    std::filesystem::path draftCheckpointDir;

    //! Reject load-time options that contradict this already-loaded bundle. Omitted values inherit the bundle.
    void validateDeploymentOptions(DeploymentOptions const& options) const;

    //! Read one engine directory and validate each loaded engine against its parsed configuration.
    //!
    //! `draftCheckpointDir` must be empty for native MTP, whose draft weights are part of `checkpointDir`; the
    //! returned artifacts record it as equal to `checkpointDir` in that case.
    static ModelArtifacts loadFromEngineDir(std::filesystem::path const& engineDir,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::filesystem::path const& checkpointDir,
        std::filesystem::path const& draftCheckpointDir, bool contextReuseEnabled, cudaStream_t stream,
        std::optional<std::filesystem::path> const& runtimeCachePath = std::nullopt, bool offloadEmbedding = false,
        std::function<void()> const& checkCancelled = {});
};

} // namespace rt
} // namespace trt_edgellm
