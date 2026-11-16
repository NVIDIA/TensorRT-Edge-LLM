/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <getopt.h>
#include <optional>
#include <tensorrt_edgellm/runtime/config/deploymentOptions.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/preprocess/visualTokenPruner.h>
#include <vector>
namespace trt_edgellm
{
namespace exampleUtils
{
enum RuntimeOptionId : int
{
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    CHECKPOINT_DIR = 931,
    DRAFT_CHECKPOINT_DIR = 932,
    RTX_RUNTIME_CACHE_PATH = 945,
    SPEC_DECODE = 910,
    SPEC_DRAFT_TOP_K = 911,
    SPEC_DRAFT_STEP = 912,
    SPEC_PROPOSAL_TOKENS = 913,
    SPEC_VERIFY_TOKENS = 921,
    DSPARK_W2_BACKEND = 946,
    DSPARK_SCHEDULER = 923,
    DSPARK_CONFIDENCE_THRESHOLD = 924,
    DSPARK_MIN_PROPOSAL_LEN = 925,
    DSPARK_MAX_PROPOSAL_LEN = 926,
    ENABLE_CONTEXT_REUSE = 927,
    CONTEXT_CACHE_MAX_RECORDS = 928,
    CONTEXT_CACHE_MAX_SNAPSHOTS = 929,
    PLE_RESIDENCY = 942,
    PLE_CACHE_SLOTS = 943,
    PLE_FILE = 944,
    OFFLOAD_EMBEDDING = 949,
    VISUAL_PRUNE = 935,
    VISUAL_PRUNE_ALGO = 939,
    DART_REDUCTION_RATIO = 936,
    DART_PIVOT_IMAGE_TOKENS = 937,
    DART_PIVOT_TEXT_TOKENS = 938,
    ENCODER_CACHE_BUDGET_BYTES = 940,
};
struct SpecDecodeArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // For tree-based strategies this is the branching factor; for chain-style
    // strategies it is the number of candidates retained per draft step.
    int32_t draftTopK{10};
    bool draftTopKSet{false};

    // Number of iterative draft expansion rounds.
    int32_t draftStep{6};
    bool draftStepSet{false};

    // Usable block proposal positions, excluding the root.
    int32_t proposalTokens{0};
    bool proposalTokensSet{false};

    // Root-inclusive target verification budget.
    int32_t verifyTokens{60};
    bool verifyTokensSet{false};

    rt::DSparkW2Backend dsparkW2Backend{rt::DSparkW2Backend::kAuto};
    rt::DSparkSchedulerMode dsparkSchedulerMode{rt::DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0};
};

struct RuntimeArgs
{
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string checkpointDir{""};
    std::string draftCheckpointDir{""};
    std::string rtxRuntimeCachePath{""};
    SpecDecodeArgs specDecodeArgs;
    rt::ContextCacheConfig contextCacheConfig;
    rt::PleRuntimeConfig pleConfig{};
    bool offloadEmbedding{false};
    rt::VisualPrunerConfig visualPrunerConfig;
};

//! Select the same base config filename as ModelArtifacts for the requested decoding mode.
std::filesystem::path getBaseConfigPath(RuntimeArgs const& args);
//! Common deployment options, without a getopt sentinel.
std::vector<option> runtimeOptions();
//! nullopt means the option belongs to the caller; false means an invalid value.
std::optional<bool> parseRuntimeOption(int id, char const* value, RuntimeArgs& args);
//! Resolve speculative defaults from the engine contract, preserving explicit overrides.
bool applyEngineSpecDecodeDefaults(RuntimeArgs& args);
//! Construct the same deployment for inference and benchmark entry points.
rt::DeploymentOptions makeDeploymentOptions(
    RuntimeArgs const& args, std::unordered_map<std::string, std::string> const& loraWeightsMap = {});
} // namespace exampleUtils
} // namespace trt_edgellm
