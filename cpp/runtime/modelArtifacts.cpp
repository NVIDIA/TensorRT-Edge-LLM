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

#include "runtime/modelArtifacts.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

bool needsCachedBlockDraftDDTreeHybridBindings(DeploymentConfig const& deployment)
{
    return deployment.specConfig.has_value() && isCachedBlockDraftMode(deployment.specDecodeMode())
        && deployment.specConfig->draftingTopK > 1 && deployment.base.numLinearAttnLayers > 0;
}

void validateCachedBlockDraftTreeMetadataBindings(
    DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || !isCachedBlockDraftMode(deployment.specDecodeMode()))
    {
        return;
    }

    char const* modeName = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "JetSpec" : "DFlash";
    char const* treeBaseFlag
        = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "--jetspec-tree-base" : "--dflash-tree-base";
    char const* linearBaseFlag
        = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "--jetspec-base" : "--dflash-base";

    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const hasTreeMetadata = hasTreeParentIds || hasTreeDepths;
    bool const usesDDTree = deployment.specConfig->draftingTopK > 1;
    if (hasTreeMetadata)
    {
        ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
            std::string(modeName) + " tree-base engine must expose both INT32 tree metadata bindings '"
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == nvinfer1::DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == nvinfer1::DataType::kINT32,
            std::string(modeName) + " tree-base engine tree metadata bindings must be INT32: '"
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(usesDDTree,
            std::string(modeName) + " base engine was exported with " + treeBaseFlag
                + ", but runtime is configured for linear mode because specDraftTopK=1. "
                  "Use --specDraftTopK > 1 for DDTree, or re-export the base model with "
                + linearBaseFlag + ".");
    }

    if (!needsCachedBlockDraftDDTreeHybridBindings(deployment))
    {
        return;
    }

    ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
        std::string(modeName) + " DDTree hybrid base engine requires INT32 tree metadata bindings '"
            + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths
            + "'. Re-export the base model with " + treeBaseFlag + ", then rebuild spec_base.engine.");
}

void validateMtpTreeMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || deployment.specDecodeMode() != SpecDecodeMode::kMTP
        || deployment.base.numLinearAttnLayers == 0)
    {
        return;
    }

    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const usesDDTree = deployment.specConfig->draftingTopK > 1;
    ELLM_CHECK(hasTreeParentIds == hasTreeDepths,
        std::string("MTP tree-base engine must expose both INT32 tree metadata bindings '")
            + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
    if (hasTreeParentIds)
    {
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == nvinfer1::DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == nvinfer1::DataType::kINT32,
            std::string("MTP tree-base engine tree metadata bindings must be INT32: '") + binding_names::kTreeParentIds
                + "' and '" + binding_names::kTreeDepths + "'.");
    }
    ELLM_CHECK(usesDDTree == hasTreeParentIds,
        usesDDTree ? "Hybrid MTP DDTree requires a tree-base engine. Rebuild with --tree-base before using "
                     "--specDraftTopK > 1."
                   : "Hybrid MTP base engine was built with --tree-base, but runtime is configured for linear MTP. "
                     "Use --specDraftTopK > 1, or rebuild without --tree-base.");
}

//! Load the reduced-vocab mapping table and check it describes the vocab the base engine was built for.
Tensor loadVocabMap(std::filesystem::path const& engineDir, LLMEngineConfig const& baseConfig, cudaStream_t stream)
{
    LOG_INFO("Loading vocabulary mapping table for base model reduced vocab size: %d -> %d",
        baseConfig.reducedVocabSize, baseConfig.vocabSize);
    std::filesystem::path const vocabMapPath = engineDir / binding_names::kVocabMapFileName;

    std::vector<Tensor> vocabMapTensors;
    ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
        "Failed to load " + std::string(binding_names::kVocabMapFileName)
            + " from model directory: " + engineDir.string());

    check::check(vocabMapTensors.size() == 1,
        std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
    check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
    check::check(vocabMapTensors[0].getShape()[0] == baseConfig.reducedVocabSize,
        "vocab_map tensor length should match base model reduced vocab size");
    check::check(vocabMapTensors[0].getDataType() == nvinfer1::DataType::kINT32, "vocab_map tensor should be INT32");
    LOG_INFO("Base model vocabulary mapping table successfully loaded.");
    return std::move(vocabMapTensors[0]);
}

std::unique_ptr<tokenizer::Tokenizer> loadTokenizer(
    std::filesystem::path const& engineDir, LLMEngineConfig const& baseConfig)
{
    auto tok = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    ELLM_CHECK(
        tok->loadFromHF(engineDir.string()), "Failed to load tokenizer from model directory: " + engineDir.string());
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // Config-declared extras beyond the tokenizer files, e.g. Gemma4's eos_token_id: [1, 106].
    if (!baseConfig.eosTokenIds.empty())
    {
        std::vector<tokenizer::Rank> additionalEos(baseConfig.eosTokenIds.begin(), baseConfig.eosTokenIds.end());
        tok->setAdditionalEosIds(additionalEos);
        LOG_INFO("Loaded %zu EOS token IDs from config", additionalEos.size());
    }
    return tok;
}

} // namespace

ModelArtifacts ModelArtifacts::loadFromEngineDir(std::filesystem::path const& engineDir,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::filesystem::path const& checkpointDir,
    std::filesystem::path const& draftCheckpointDir, cudaStream_t stream)
{
    ModelArtifacts artifacts;
    artifacts.checkpointDir = checkpointDir;
    artifacts.draftCheckpointDir = draftCheckpointDir;

    std::filesystem::path const baseConfigPath
        = draftingConfig.has_value() ? engineDir / "base_config.json" : engineDir / "config.json";

    // Finish checkpoint reads and weight conversion before any engine can run.
    artifacts.weights.load(engineDir, baseConfigPath, stream, artifacts.checkpointDir);
    if (auto embedding = artifacts.weights.takeEmbedding())
    {
        artifacts.embedding.table = std::move(*embedding);
    }
    else
    {
        artifacts.embedding = loadEmbeddingTable(engineDir / "embedding.safetensors", stream);
    }
    artifacts.pleEmbedding = artifacts.weights.takePleEmbedding();

    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineDir / "draft_config.json"}
        : std::nullopt;

    artifacts.deployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);
    if (draftingConfig.has_value() && artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
    {
        ELLM_CHECK(artifacts.draftCheckpointDir.empty(),
            "Native MTP draft weights are part of --checkpointDir; do not pass --draftCheckpointDir.");
        artifacts.draftCheckpointDir = artifacts.checkpointDir;
    }

    std::filesystem::path const baseEnginePath = draftingConfig.has_value()
        ? engineDir / "spec_base.engine"
        : (artifacts.deployment.base.isDiffusionBackbone ? engineDir / "dllm.engine" : engineDir / "llm.engine");

    try
    {
        std::optional<int32_t> const specDecodeBaseOutputHiddenDim = artifacts.deployment.specConfig.has_value()
            ? std::optional<int32_t>{artifacts.deployment.specConfig->baseOutputHiddenDim}
            : std::nullopt;
        artifacts.baseExecutor
            = EngineExecutor::createForLLM(baseEnginePath, artifacts.deployment.base, specDecodeBaseOutputHiddenDim);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    validateAgainstEngine(artifacts.deployment.base, *artifacts.baseExecutor, "base");
    validateCachedBlockDraftTreeMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);
    validateMtpTreeMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);

    // Validate the draft engine ABI before its sidecar geometry is used to allocate physical cache resources.
    if (draftingConfig.has_value())
    {
        artifacts.draftExecutor = decoder_utils::loadDraftEngine(engineDir, artifacts.deployment);
        std::filesystem::path const draftConfig = engineDir / "draft_config.json";
        if (artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
        {
            // Native MTP draft weights ship inside the base checkpoint, so there is no separate target to fall back to.
            artifacts.draftWeights.load(engineDir, draftConfig, stream, artifacts.checkpointDir);
        }
        else
        {
            artifacts.draftWeights.load(
                engineDir, draftConfig, stream, artifacts.draftCheckpointDir, artifacts.checkpointDir);
        }
        // Name the mode in the label. Loading moved here from the decoders, and each of them used to pass its own
        // ("dflash_draft", "dspark_draft", ...); one call site would otherwise report every mismatch as plain "draft"
        // and leave the reader to work out which speculative mode produced it.
        std::string const draftLabel
            = std::string{specDecodeModeName(artifacts.deployment.specDecodeMode())} + " draft";
        artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, draftLabel);
    }

    if (artifacts.deployment.base.reducedVocabSize > 0)
    {
        artifacts.vocabMap = loadVocabMap(engineDir, artifacts.deployment.base, stream);
    }
    artifacts.tokenizer = loadTokenizer(engineDir, artifacts.deployment.base);

    return artifacts;
}

} // namespace rt
} // namespace trt_edgellm
