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

#include "builder/gr00tBuilder.h"

#include "builder/builderUtils.h"
#include "builder/visualBuilder.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/gr00tBindings.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <algorithm>
#include <system_error>
#include <utility>

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

constexpr int32_t kContractVersion = 1;

nvinfer1::Dims dimsFromJson(nlohmann::json const& shape)
{
    return builder::createDims(shape.get<std::vector<int64_t>>());
}

//! True when every extent of \p lhs is <= the matching extent of \p rhs.
bool dimsLessEqual(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs)
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < lhs.nbDims; ++i)
    {
        if (lhs.d[i] > rhs.d[i])
        {
            return false;
        }
    }
    return true;
}

bool copyTree(std::filesystem::path const& src, std::filesystem::path const& dst)
{
    std::error_code ec;
    std::filesystem::create_directories(dst.parent_path(), ec);
    std::filesystem::copy(
        src, dst, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
    {
        LOG_ERROR("Failed to stage %s to %s: %s", src.string().c_str(), dst.string().c_str(), ec.message().c_str());
        return false;
    }
    LOG_INFO("Staged %s", dst.string().c_str());
    return true;
}

//! The visual tower is the shared Qwen3-VL contract, so the shared builder compiles
//! it; its token budget is what the export recorded in builder_config.
bool buildVisual(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir)
{
    nlohmann::json config;
    if (!builder::loadJsonConfig((onnxDir / "config.json").string(), config))
    {
        return false;
    }
    builder::VisualBuilderConfig visualConfig
        = builder::VisualBuilderConfig::fromJson(config.value("builder_config", nlohmann::json::object()));
    builder::VisualBuilder visualBuilder(onnxDir, engineDir, visualConfig);
    if (!visualBuilder.build())
    {
        return false;
    }
    // VisualBuilder carries preprocessor_config.json itself; the processor config
    // beside it is part of the ONNX-less bundle layout.
    std::filesystem::path const processor = onnxDir / "processor_config.json";
    return !std::filesystem::exists(processor) || copyTree(processor, engineDir / "processor_config.json");
}

//! Copy the sidecars the runtime needs next to the engines: the token-embedding
//! table the host gathers from, the tokenizer with its chat template, and the
//! action component's constant tables and policy contract.
bool stageRuntimeArtifacts(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components)
{
    auto const built
        = [&](char const* name) { return std::find(components.begin(), components.end(), name) != components.end(); };
    if (built("vl_backbone"))
    {
        if (!copyTree(onnxRoot / "vl_backbone" / "embed_tokens.safetensors", engineRoot / "embed_tokens.safetensors")
            || !copyTree(onnxRoot / "tokenizer", engineRoot / "tokenizer"))
        {
            return false;
        }
    }
    if (built("action"))
    {
        for (char const* file : {"adanorm_modulation.safetensors", "action_time_bias.safetensors", "policy.json"})
        {
            if (!copyTree(onnxRoot / "action" / file, engineRoot / "action" / file))
            {
                return false;
            }
        }
    }
    return true;
}

//! Identity of a staged JSON file, or empty with \p error set.
BundleIdentity readIdentity(std::filesystem::path const& path, std::string const& what, std::string& error)
{
    nlohmann::json document;
    if (!builder::loadJsonConfig(path.string(), document))
    {
        error = "GR00T " + what + " is missing or unreadable: " + path.string();
        return {};
    }
    try
    {
        return requireIdentity(document, what);
    }
    catch (std::runtime_error const& e)
    {
        error = e.what();
        return {};
    }
}

} // namespace

std::vector<std::string> const& policyComponents()
{
    static std::vector<std::string> const components{"visual", "vl_backbone", "action"};
    return components;
}

bool verifyGr00tExportIds(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components)
{
    auto const selected
        = [&](char const* name) { return std::find(components.begin(), components.end(), name) != components.end(); };
    auto const filesOf = [](std::filesystem::path const& root, std::string const& component) {
        std::vector<std::pair<std::filesystem::path, std::string>> files{
            {root / component / "config.json", component + " config.json"}};
        if (component == "action")
        {
            files.emplace_back(root / "action" / "policy.json", "action/policy.json");
        }
        return files;
    };

    std::string error;
    BundleIdentity expected;
    std::string reference;
    for (auto const& component : components)
    {
        for (auto const& [path, what] : filesOf(onnxRoot, component))
        {
            BundleIdentity const found = readIdentity(path, "export " + what, error);
            if (found.exportId.empty())
            {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
            if (found.frontend != "onnx")
            {
                LOG_ERROR(
                    "GR00T export %s was written by the %s route; gr00t_policy_build takes "
                    "tensorrt-edgellm-export output",
                    what.c_str(), found.frontend.c_str());
                return false;
            }
            if (reference.empty())
            {
                expected = found;
                reference = what;
                continue;
            }
            if (found.exportId != expected.exportId)
            {
                LOG_ERROR(
                    "GR00T export %s is from export %s but %s is from %s; the components are from "
                    "different exports",
                    what.c_str(), found.exportId.c_str(), reference.c_str(), expected.exportId.c_str());
                return false;
            }
        }
    }

    // Components this build does not replace must already belong to the same export.
    for (auto const& component : policyComponents())
    {
        if (selected(component.c_str()))
        {
            continue;
        }
        for (auto const& [path, what] : filesOf(engineRoot, component))
        {
            if (!std::filesystem::exists(path))
            {
                continue;
            }
            BundleIdentity const staged = readIdentity(path, "staged " + what, error);
            if (staged.exportId.empty())
            {
                LOG_ERROR("%s", error.c_str());
                return false;
            }
            if (staged.exportId != expected.exportId || staged.frontend != expected.frontend)
            {
                LOG_ERROR(
                    "GR00T staged %s is from build %s (%s) but this build is %s (%s); rebuild every "
                    "component into %s or build into an empty directory",
                    what.c_str(), staged.exportId.c_str(), staged.frontend.c_str(), expected.exportId.c_str(),
                    expected.frontend.c_str(), engineRoot.string().c_str());
                return false;
            }
        }
    }
    LOG_INFO("GR00T export %s: %zu component(s) verified", expected.exportId.c_str(), components.size());
    return true;
}

bool buildGr00tPolicy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize)
{
    if (components.empty())
    {
        LOG_ERROR("GR00T build was given no components; an empty set would stage sidecars and report success");
        return false;
    }
    // The visual export carries a ViTAttentionPlugin node; the other two lower to trt:: ops.
    auto pluginHandles = loadEdgellmPluginLib();
    if (!pluginHandles && std::find(components.begin(), components.end(), "visual") != components.end())
    {
        LOG_ERROR(
            "GR00T visual carries a trt_edgellm plugin node; set EDGELLM_PLUGIN_PATH to libNvInfer_edgellm_plugin.so");
        return false;
    }
    if (!verifyGr00tExportIds(onnxRoot, engineRoot, components))
    {
        return false;
    }
    for (auto const& component : components)
    {
        std::filesystem::path const onnxDir = onnxRoot / component;
        bool const ok = component == "visual"
            ? buildVisual(onnxDir, engineRoot / component)
            : Gr00tBuilder(onnxDir, engineRoot / component, component, maxBatchSize).build();
        if (!ok)
        {
            LOG_ERROR("Failed to build GR00T %s engine.", component.c_str());
            return false;
        }
        LOG_INFO("GR00T %s engine built successfully.", component.c_str());
    }
    return stageRuntimeArtifacts(onnxRoot, engineRoot, components);
}

Gr00tBuilder::Gr00tBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
    std::string const& component, int32_t maxBatchSize)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mComponent(component)
    , mMaxBatchSize(maxBatchSize)
    , mEngineFilename(component + ".engine")
{
}

bool Gr00tBuilder::build()
{
    if (!parseConfig())
    {
        return false;
    }

    auto [trtBuilder, network] = builder::createBuilderAndNetwork();
    if (!trtBuilder || !network)
    {
        return false;
    }
    auto parser = builder::parseOnnxModel(network.get(), (mOnnxDir / mOnnxFilename).string());
    if (!parser)
    {
        return false;
    }
    LOG_DEBUG("%s", builder::printNetworkInfo(network.get(), ("GR00T " + mComponent).c_str()).c_str());

    auto config = builder::createBuilderConfig(trtBuilder.get());
    if (!config || !setupOptimizationProfile(*trtBuilder, *config, *network))
    {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(mEngineDir, ec);
    if (ec)
    {
        LOG_ERROR("Failed to create GR00T engine directory %s", mEngineDir.string().c_str());
        return false;
    }
    if (!builder::buildAndSerializeEngine(
            trtBuilder.get(), network.get(), config.get(), (mEngineDir / mEngineFilename).string()))
    {
        return false;
    }

    nlohmann::json builderConfig = mModelConfig.value("builder_config", nlohmann::json::object());
    builderConfig["max_batch_size"] = mMaxBatchSize;
    return builder::saveConfigWithBuilderInfo(mEngineDir, mModelConfig, builderConfig);
}

bool Gr00tBuilder::parseConfig()
{
    if (!builder::loadJsonConfig((mOnnxDir / "config.json").string(), mModelConfig))
    {
        return false;
    }
    version::checkVersion(mModelConfig.value(trt_edgellm::binding_names::kEdgellmVersion, ""));

    int32_t const contractVersion = mModelConfig.value("contract_version", 0);
    if (contractVersion != kContractVersion)
    {
        LOG_ERROR("GR00T %s was exported against contract version %d but this runtime implements %d; re-export it",
            mComponent.c_str(), contractVersion, kContractVersion);
        return false;
    }
    std::string const configComponent = mModelConfig.value("component", mComponent);
    if (configComponent != mComponent)
    {
        LOG_ERROR("GR00T component mismatch: CLI requested %s but config contains %s", mComponent.c_str(),
            configComponent.c_str());
        return false;
    }
    mOnnxFilename = mModelConfig.value("onnx_filename", mOnnxFilename);
    mEngineFilename = mModelConfig.value("engine_filename", mEngineFilename);
    return true;
}

bool Gr00tBuilder::inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const
{
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        if (network.getInput(i) != nullptr && name == network.getInput(i)->getName())
        {
            return true;
        }
    }
    return false;
}

bool Gr00tBuilder::setupOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network) const
{
    nlohmann::json const profileJson = mModelConfig.value("optimization_profile", nlohmann::json::object());
    nlohmann::json const batchAxis = mModelConfig.value("batch_axis", nlohmann::json::object());
    auto* profile = builder.createOptimizationProfile();
    bool ok = true;
    for (auto const& item : profileJson.items())
    {
        std::string const& name = item.key();
        if (!inputExists(network, name))
        {
            LOG_ERROR("GR00T %s profile names %s, which is not an ONNX input of this graph", mComponent.c_str(),
                name.c_str());
            return false;
        }
        nvinfer1::Dims const minDims = dimsFromJson(item.value().at("min"));
        nvinfer1::Dims optDims = dimsFromJson(item.value().at("opt"));
        nvinfer1::Dims maxDims = dimsFromJson(item.value().at("max"));
        if (mMaxBatchSize > 1 && batchAxis.contains(name) && !batchAxis.at(name).is_null())
        {
            int32_t const axis = batchAxis.at(name).get<int32_t>();
            if (axis < 0 || axis >= maxDims.nbDims)
            {
                LOG_ERROR("GR00T %s batch_axis %d for %s is out of range", mComponent.c_str(), axis, name.c_str());
                return false;
            }
            maxDims.d[axis] = std::max<int64_t>(maxDims.d[axis], mMaxBatchSize);
            optDims.d[axis] = maxDims.d[axis];
        }
        if (!dimsLessEqual(minDims, optDims) || !dimsLessEqual(optDims, maxDims))
        {
            LOG_ERROR("GR00T %s profile for %s violates min <= opt <= max", mComponent.c_str(), name.c_str());
            return false;
        }
        ok &= builder::setOptimizationProfile(profile, name.c_str(), minDims, optDims, maxDims);
    }
    if (!ok)
    {
        LOG_ERROR("Failed to set GR00T %s optimization profile", mComponent.c_str());
        return false;
    }
    LOG_DEBUG("%s", builder::printOptimizationProfile(profile, (mComponent + "_profile").c_str(), &network).c_str());
    config.addOptimizationProfile(profile);
    return true;
}

} // namespace gr00t
} // namespace trt_edgellm
