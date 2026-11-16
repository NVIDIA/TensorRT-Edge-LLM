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

#include "runtimeArgs.h"
#include <charconv>
#include <iostream>
#include <string_view>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/runtime/config/llmEngineConfig.h>
#include <type_traits>
namespace trt_edgellm
{
namespace exampleUtils
{
namespace
{
template <typename IntegerType>
bool parseNonNegativeIntegerOption(char const* optionName, char const* value, IntegerType& output)
{
    static_assert(std::is_integral_v<IntegerType> && std::is_signed_v<IntegerType>);
    std::string_view const text{value == nullptr ? "" : value};
    IntegerType parsed{};
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed < 0)
    {
        LOG_ERROR("Invalid --%s value: %s (must be a non-negative integer)", optionName, text.data());
        return false;
    }
    output = parsed;
    return true;
}

} // namespace

std::filesystem::path getBaseConfigPath(RuntimeArgs const& args)
{
    return std::filesystem::path(args.engineDir) / (args.specDecodeArgs.enabled ? "base_config.json" : "config.json");
}

bool applyEngineSpecDecodeDefaults(RuntimeArgs& args)
{
    if (!args.specDecodeArgs.enabled)
    {
        return true;
    }

    try
    {
        rt::LLMEngineConfig const baseConfig = rt::parseEngineConfig(getBaseConfigPath(args));
        SpecDecodeArgs& specArgs = args.specDecodeArgs;
        switch (baseConfig.specDecodeType)
        {
        case rt::SpecDecodeMode::kDFlash:
        case rt::SpecDecodeMode::kJetSpec:
        case rt::SpecDecodeMode::kDomino:
        {
            if (!specArgs.draftTopKSet)
            {
                specArgs.draftTopK = 1;
            }
            if (!specArgs.draftStepSet)
            {
                specArgs.draftStep = 1;
            }
            break;
        }
        case rt::SpecDecodeMode::kDSpark:
            if (!specArgs.draftTopKSet)
            {
                specArgs.draftTopK = 1;
            }
            if (!specArgs.draftStepSet)
            {
                specArgs.draftStep = 1;
            }
            break;
        default:
            if (!specArgs.draftStepSet)
            {
                specArgs.draftStepSet = true;
            }
            if (specArgs.draftTopK > 1)
            {
                specArgs.verifyTokensSet = true;
            }
            break;
        }

        LOG_INFO("Spec decode engine mode: %s", rt::specDecodeModeName(baseConfig.specDecodeType));
        LOG_INFO("Resolved spec draft topK: %d", specArgs.draftTopK);
        LOG_INFO("Resolved spec draft step: %d", specArgs.draftStep);
        if (specArgs.proposalTokensSet)
        {
            LOG_INFO("Resolved spec proposal tokens: %d", specArgs.proposalTokens);
        }
        if (specArgs.verifyTokensSet)
        {
            LOG_INFO("Resolved spec verify tokens: %d", specArgs.verifyTokens);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to resolve speculative decoding defaults from engine config: %s", e.what());
        return false;
    }

    return true;
}

std::vector<option> runtimeOptions()
{
    return {
        {"context-cache-max-snapshots", required_argument, 0, RuntimeOptionId::CONTEXT_CACHE_MAX_SNAPSHOTS},
        {"contextCacheMaxSnapshots", required_argument, 0, RuntimeOptionId::CONTEXT_CACHE_MAX_SNAPSHOTS},
        {"engine-dir", required_argument, 0, RuntimeOptionId::ENGINE_DIR},
        {"engineDir", required_argument, 0, RuntimeOptionId::ENGINE_DIR},
        {"multimodal-engine-dir", required_argument, 0, RuntimeOptionId::MULTIMODAL_ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, RuntimeOptionId::MULTIMODAL_ENGINE_DIR},
        {"ple-file", required_argument, 0, RuntimeOptionId::PLE_FILE},
        {"pleFile", required_argument, 0, RuntimeOptionId::PLE_FILE},
        {"ple-residency", required_argument, 0, RuntimeOptionId::PLE_RESIDENCY},
        {"pleResidency", required_argument, 0, RuntimeOptionId::PLE_RESIDENCY},
        {"offload-embedding", no_argument, 0, RuntimeOptionId::OFFLOAD_EMBEDDING},
        {"offloadEmbedding", no_argument, 0, RuntimeOptionId::OFFLOAD_EMBEDDING},
        {"ple-cache-slots", required_argument, 0, RuntimeOptionId::PLE_CACHE_SLOTS},
        {"pleCacheSlots", required_argument, 0, RuntimeOptionId::PLE_CACHE_SLOTS},
        {"checkpoint-dir", required_argument, 0, RuntimeOptionId::CHECKPOINT_DIR},
        {"checkpointDir", required_argument, 0, RuntimeOptionId::CHECKPOINT_DIR},
        {"draft-checkpoint-dir", required_argument, 0, RuntimeOptionId::DRAFT_CHECKPOINT_DIR},
        {"draftCheckpointDir", required_argument, 0, RuntimeOptionId::DRAFT_CHECKPOINT_DIR},
#if IS_TRT_RTX
        {"rtx-runtime-cache-path", required_argument, 0, RuntimeOptionId::RTX_RUNTIME_CACHE_PATH},
        {"rtxRuntimeCachePath", required_argument, 0, RuntimeOptionId::RTX_RUNTIME_CACHE_PATH},
#endif
        {"spec-decode", no_argument, 0, RuntimeOptionId::SPEC_DECODE},
        {"specDecode", no_argument, 0, RuntimeOptionId::SPEC_DECODE},
        {"eagle", no_argument, 0, RuntimeOptionId::SPEC_DECODE},
        {"spec-draft-top-k", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_TOP_K},
        {"specDraftTopK", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_TOP_K},
        {"eagle-draft-top-k", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_TOP_K},
        {"eagleDraftTopK", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_TOP_K},
        {"spec-draft-step", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_STEP},
        {"specDraftStep", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_STEP},
        {"eagle-draft-step", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_STEP},
        {"eagleDraftStep", required_argument, 0, RuntimeOptionId::SPEC_DRAFT_STEP},
        {"spec-proposal-tokens", required_argument, 0, RuntimeOptionId::SPEC_PROPOSAL_TOKENS},
        {"specProposalTokens", required_argument, 0, RuntimeOptionId::SPEC_PROPOSAL_TOKENS},
        {"spec-verify-tokens", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"spec-verify-size", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"specVerifySize", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"spec-verify-tree-size", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"specVerifyTreeSize", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"eagle-verify-tree-size", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"eagleVerifyTreeSize", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"specVerifyTokens", required_argument, 0, RuntimeOptionId::SPEC_VERIFY_TOKENS},
        {"dspark-w2-backend", required_argument, 0, RuntimeOptionId::DSPARK_W2_BACKEND},
        {"dsparkW2Backend", required_argument, 0, RuntimeOptionId::DSPARK_W2_BACKEND},
        {"dspark-scheduler", required_argument, 0, RuntimeOptionId::DSPARK_SCHEDULER},
        {"dsparkScheduler", required_argument, 0, RuntimeOptionId::DSPARK_SCHEDULER},
        {"dspark-confidence-threshold", required_argument, 0, RuntimeOptionId::DSPARK_CONFIDENCE_THRESHOLD},
        {"dsparkConfidenceThreshold", required_argument, 0, RuntimeOptionId::DSPARK_CONFIDENCE_THRESHOLD},
        {"dspark-min-proposal-len", required_argument, 0, RuntimeOptionId::DSPARK_MIN_PROPOSAL_LEN},
        {"dsparkMinProposalLen", required_argument, 0, RuntimeOptionId::DSPARK_MIN_PROPOSAL_LEN},
        {"dspark-max-proposal-len", required_argument, 0, RuntimeOptionId::DSPARK_MAX_PROPOSAL_LEN},
        {"dsparkMaxProposalLen", required_argument, 0, RuntimeOptionId::DSPARK_MAX_PROPOSAL_LEN},
        {"enable-context-reuse", no_argument, 0, RuntimeOptionId::ENABLE_CONTEXT_REUSE},
        {"enableContextReuse", no_argument, 0, RuntimeOptionId::ENABLE_CONTEXT_REUSE},
        {"context-cache-max-records", required_argument, 0, RuntimeOptionId::CONTEXT_CACHE_MAX_RECORDS},
        {"contextCacheMaxRecords", required_argument, 0, RuntimeOptionId::CONTEXT_CACHE_MAX_RECORDS},
        {"visual-prune", no_argument, 0, RuntimeOptionId::VISUAL_PRUNE},
        {"visualPrune", no_argument, 0, RuntimeOptionId::VISUAL_PRUNE},
        {"visual-prune-algo", required_argument, 0, RuntimeOptionId::VISUAL_PRUNE_ALGO},
        {"visualPruneAlgo", required_argument, 0, RuntimeOptionId::VISUAL_PRUNE_ALGO},
        {"dart-reduction-ratio", required_argument, 0, RuntimeOptionId::DART_REDUCTION_RATIO},
        {"dartReductionRatio", required_argument, 0, RuntimeOptionId::DART_REDUCTION_RATIO},
        {"dart-pivot-image-tokens", required_argument, 0, RuntimeOptionId::DART_PIVOT_IMAGE_TOKENS},
        {"dartPivotImageTokens", required_argument, 0, RuntimeOptionId::DART_PIVOT_IMAGE_TOKENS},
        {"dart-pivot-text-tokens", required_argument, 0, RuntimeOptionId::DART_PIVOT_TEXT_TOKENS},
        {"dartPivotTextTokens", required_argument, 0, RuntimeOptionId::DART_PIVOT_TEXT_TOKENS},
        {"encoder-cache-budget-bytes", required_argument, 0, RuntimeOptionId::ENCODER_CACHE_BUDGET_BYTES},
        {"encoderCacheBudgetBytes", required_argument, 0, RuntimeOptionId::ENCODER_CACHE_BUDGET_BYTES},
    };
}
std::optional<bool> parseRuntimeOption(int id, char const* value, RuntimeArgs& args)
{
    switch (id)
    {
    case RuntimeOptionId::ENGINE_DIR: args.engineDir = value; break;
    case RuntimeOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = value; break;
    case RuntimeOptionId::PLE_FILE: args.pleConfig.filePath = value; break;
    case RuntimeOptionId::OFFLOAD_EMBEDDING: args.offloadEmbedding = true; break;
    case RuntimeOptionId::PLE_RESIDENCY:
    {
        std::string const policy(value);
        if (policy != "resident" && policy != "cached")
        {
            std::cerr << "--ple-residency must be resident or cached" << std::endl;
            return false;
        }
        args.pleConfig.residency = policy == "resident" ? rt::PleResidency::kRESIDENT : rt::PleResidency::kCACHED;
        break;
    }
    case RuntimeOptionId::PLE_CACHE_SLOTS:
    {
        if (!parseNonNegativeIntegerOption("--ple-cache-slots", value, args.pleConfig.cacheSlots))
        {
            return false;
        }
        break;
    }
    case RuntimeOptionId::CHECKPOINT_DIR: args.checkpointDir = value; break;
    case RuntimeOptionId::RTX_RUNTIME_CACHE_PATH: args.rtxRuntimeCachePath = value; break;
    case RuntimeOptionId::DRAFT_CHECKPOINT_DIR: args.draftCheckpointDir = value; break;
    case RuntimeOptionId::SPEC_DECODE: args.specDecodeArgs.enabled = true; break;
    case RuntimeOptionId::SPEC_DRAFT_TOP_K:
        try
        {
            args.specDecodeArgs.draftTopK = std::stoi(value);
            args.specDecodeArgs.draftTopKSet = true;
            if (args.specDecodeArgs.draftTopK <= 0)
            {
                LOG_ERROR("Invalid specDraftTopK value: %s (must be positive)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid specDraftTopK value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::SPEC_DRAFT_STEP:
        try
        {
            args.specDecodeArgs.draftStep = std::stoi(value);
            args.specDecodeArgs.draftStepSet = true;
            if (args.specDecodeArgs.draftStep <= 0)
            {
                LOG_ERROR("Invalid specDraftStep value: %s (must be positive)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid specDraftStep value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::SPEC_PROPOSAL_TOKENS:
        try
        {
            args.specDecodeArgs.proposalTokens = std::stoi(value);
            args.specDecodeArgs.proposalTokensSet = true;
            if (args.specDecodeArgs.proposalTokens <= 0)
            {
                LOG_ERROR("Invalid specProposalTokens value: %s (must be positive)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid specProposalTokens value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::SPEC_VERIFY_TOKENS:
        try
        {
            args.specDecodeArgs.verifyTokens = std::stoi(value);
            args.specDecodeArgs.verifyTokensSet = true;
            if (args.specDecodeArgs.verifyTokens <= 0)
            {
                LOG_ERROR("Invalid specVerifyTokens value: %s (must be positive)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid specVerifyTokens value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::DSPARK_W2_BACKEND:
        try
        {
            args.specDecodeArgs.dsparkW2Backend = rt::parseDSparkW2Backend(value);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dsparkW2Backend: %s", e.what());
            return false;
        }
        break;
    case RuntimeOptionId::DSPARK_SCHEDULER:
    {
        std::string const mode{value};
        if (mode == "off")
        {
            args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kOff;
        }
        else if (mode == "threshold")
        {
            args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kThreshold;
        }
        else if (mode == "sps")
        {
            args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kSPS;
        }
        else
        {
            LOG_ERROR("Invalid dsparkScheduler value: %s (expected off, threshold, or sps)", value);
            return false;
        }
        break;
    }
    case RuntimeOptionId::DSPARK_CONFIDENCE_THRESHOLD:
        try
        {
            args.specDecodeArgs.dsparkConfidenceThreshold = std::stof(value);
            if (args.specDecodeArgs.dsparkConfidenceThreshold < 0.0F
                || args.specDecodeArgs.dsparkConfidenceThreshold > 1.0F)
            {
                LOG_ERROR("Invalid dsparkConfidenceThreshold value: %s (must be in [0,1])", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dsparkConfidenceThreshold value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::DSPARK_MIN_PROPOSAL_LEN:
        try
        {
            args.specDecodeArgs.dsparkMinProposalLen = std::stoi(value);
            if (args.specDecodeArgs.dsparkMinProposalLen <= 0)
            {
                LOG_ERROR("Invalid dsparkMinProposalLen value: %s (must be positive)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dsparkMinProposalLen value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::DSPARK_MAX_PROPOSAL_LEN:
        try
        {
            args.specDecodeArgs.dsparkMaxProposalLen = std::stoi(value);
            if (args.specDecodeArgs.dsparkMaxProposalLen < 0)
            {
                LOG_ERROR("Invalid dsparkMaxProposalLen value: %s (must be non-negative)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dsparkMaxProposalLen value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::VISUAL_PRUNE: args.visualPrunerConfig.enabled = true; break;
    case RuntimeOptionId::VISUAL_PRUNE_ALGO: args.visualPrunerConfig.algorithm = value; break;
    case RuntimeOptionId::DART_REDUCTION_RATIO:
        try
        {
            args.visualPrunerConfig.reductionRatio = std::stof(value);
            if (args.visualPrunerConfig.reductionRatio <= 0.0F || args.visualPrunerConfig.reductionRatio >= 1.0F)
            {
                LOG_ERROR("Invalid dartReductionRatio value: %s (must be in (0, 1))", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dartReductionRatio value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::DART_PIVOT_IMAGE_TOKENS:
        try
        {
            args.visualPrunerConfig.pivotImageTokens = std::stoi(value);
            if (args.visualPrunerConfig.pivotImageTokens < 0)
            {
                LOG_ERROR("Invalid dartPivotImageTokens value: %s (must be non-negative)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dartPivotImageTokens value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::DART_PIVOT_TEXT_TOKENS:
        try
        {
            args.visualPrunerConfig.pivotTextTokens = std::stoi(value);
            if (args.visualPrunerConfig.pivotTextTokens < 0)
            {
                LOG_ERROR("Invalid dartPivotTextTokens value: %s (must be non-negative)", value);
                return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Invalid dartPivotTextTokens value: %s", value);
            return false;
        }
        break;
    case RuntimeOptionId::ENABLE_CONTEXT_REUSE: args.contextCacheConfig.enabled = true; break;
    case RuntimeOptionId::CONTEXT_CACHE_MAX_RECORDS:
        if (!parseNonNegativeIntegerOption("contextCacheMaxRecords", value, args.contextCacheConfig.maxRecords))
        {
            return false;
        }
        break;
    case RuntimeOptionId::CONTEXT_CACHE_MAX_SNAPSHOTS:
        if (!parseNonNegativeIntegerOption("contextCacheMaxSnapshots", value, args.contextCacheConfig.maxSnapshots))
        {
            return false;
        }
        break;
    case RuntimeOptionId::ENCODER_CACHE_BUDGET_BYTES:
        if (!parseNonNegativeIntegerOption(
                "encoderCacheBudgetBytes", value, args.contextCacheConfig.encoderEmbeddingCacheBudgetBytes))
        {
            return false;
        }
        break;
    default: return std::nullopt;
    }
    return true;
}
rt::DeploymentOptions makeDeploymentOptions(
    RuntimeArgs const& args, std::unordered_map<std::string, std::string> const& loraWeightsMap)
{
    if (args.engineDir.empty())
    {
        throw std::invalid_argument("--engine-dir is required");
    }
    if (!args.specDecodeArgs.enabled && !args.draftCheckpointDir.empty())
    {
        throw std::invalid_argument("--draft-checkpoint-dir requires --spec-decode");
    }
    rt::DeploymentOptions options{};
    options.engineDir = args.engineDir;
    options.multimodalEngineDir = args.multimodalEngineDir;
    options.loraWeightsMap = loraWeightsMap;
    options.contextCacheConfig = args.contextCacheConfig;
    options.pleConfig = args.pleConfig;
    options.offloadEmbedding = args.offloadEmbedding;
    options.checkpointDir = args.checkpointDir;
    options.draftCheckpointDir = args.draftCheckpointDir;
    if (!args.rtxRuntimeCachePath.empty())
    {
        options.runtimeCachePath = args.rtxRuntimeCachePath;
    }
    if (args.specDecodeArgs.enabled)
    {
        rt::SpecDecodeDraftingConfig config;
        config.draftTopK = args.specDecodeArgs.draftTopK;
        if (args.specDecodeArgs.draftStepSet)
        {
            config.draftingStep = args.specDecodeArgs.draftStep;
        }
        if (args.specDecodeArgs.proposalTokensSet)
        {
            config.proposalTokens = args.specDecodeArgs.proposalTokens;
        }
        if (args.specDecodeArgs.verifyTokensSet)
        {
            config.verifyTokens = args.specDecodeArgs.verifyTokens;
        }
        config.dsparkW2Backend = args.specDecodeArgs.dsparkW2Backend;
        config.dsparkSchedulerMode = args.specDecodeArgs.dsparkSchedulerMode;
        config.dsparkConfidenceThreshold = args.specDecodeArgs.dsparkConfidenceThreshold;
        config.dsparkMinProposalLen = args.specDecodeArgs.dsparkMinProposalLen;
        config.dsparkMaxProposalLen = args.specDecodeArgs.dsparkMaxProposalLen;
        options.draftingConfig = config;
    }
    return options;
}
} // namespace exampleUtils
} // namespace trt_edgellm
