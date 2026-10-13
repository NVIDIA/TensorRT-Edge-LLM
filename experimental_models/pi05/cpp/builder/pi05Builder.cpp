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

#include "builder/pi05Builder.h"

#include "builder/builderUtils.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/pi05Bindings.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <algorithm>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

namespace
{

nvinfer1::Dims dimsFromJson(nlohmann::json const& shape)
{
    std::vector<int64_t> values = shape.get<std::vector<int64_t>>();
    return builder::createDims(values);
}

std::string defaultEngineFilename(std::string const& component)
{
    return component + ".engine";
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

//! Copy the sidecars the runtime needs next to the engines: the token-embedding
//! table the prefix gathers from on the device, the tokenizer, and the
//! normalization statistics a caller needs to map actions into robot units.
bool stageRuntimeArtifacts(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components)
{
    bool const builtPrefix = std::find(components.begin(), components.end(), "prefix") != components.end();
    if (builtPrefix)
    {
        std::filesystem::path const embedSrc = onnxRoot / "prefix" / "embed_tokens.safetensors";
        std::filesystem::path const embedDst = engineRoot / "embed_tokens.safetensors";
        if (!file_io::copyFile(embedSrc.string(), embedDst.string()))
        {
            LOG_ERROR("Failed to stage %s to %s", embedSrc.string().c_str(), embedDst.string().c_str());
            return false;
        }
        LOG_INFO("Staged token-embedding table: %s", embedDst.string().c_str());
    }

    std::filesystem::path const contractSrc = onnxRoot / "policy.json";
    if (!file_io::copyFile(contractSrc.string(), (engineRoot / "policy.json").string()))
    {
        LOG_ERROR("Failed to stage %s", contractSrc.string().c_str());
        return false;
    }
    LOG_INFO("Staged policy contract: %s", (engineRoot / "policy.json").string().c_str());

    for (char const* dir : {"text_tokenizer", "assets"})
    {
        std::filesystem::path const src = onnxRoot / dir;
        if (!std::filesystem::exists(src))
        {
            // text_tokenizer/ comes out of the export, so its absence is a broken export.
            // assets/ does not: the documented flow stages norm_stats.json after the export,
            // and the runtime refuses to load without it.
            if (std::string_view{dir} == "text_tokenizer")
            {
                LOG_ERROR("pi0.5 export has no %s/ at %s; re-run the export", dir, src.string().c_str());
                return false;
            }
            LOG_WARNING("pi0.5 export has no %s/; stage norm_stats.json under %s before running the policy", dir,
                (engineRoot / dir).string().c_str());
            continue;
        }
        std::error_code ec;
        std::filesystem::copy(src, engineRoot / dir,
            std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
        {
            LOG_ERROR("Failed to stage %s: %s", src.string().c_str(), ec.message().c_str());
            return false;
        }
        LOG_INFO("Staged %s", (engineRoot / dir).string().c_str());
    }
    return true;
}

//! Read ``export_id`` from a staged JSON file, or report why it cannot be trusted.
//! An empty return means \p what is unusable and \p error says so.
std::string readExportId(std::filesystem::path const& path, std::string const& what, std::string& error)
{
    nlohmann::json config;
    if (!builder::loadJsonConfig(path.string(), config))
    {
        error = "pi0.5 " + what + " is missing or unreadable: " + path.string();
        return {};
    }
    std::string const exportId = config.value(kExportIdKey, std::string{});
    if (exportId.empty())
    {
        error = "pi0.5 " + what + " carries no " + kExportIdKey + " (" + path.string()
            + "); re-export the checkpoint with tensorrt-edgellm-export so the components can be "
              "verified to come from one export";
    }
    return exportId;
}

//! The export every selected source component agrees on, or empty when they do not.
std::string sourceExportId(std::filesystem::path const& onnxRoot, std::vector<std::string> const& components)
{
    std::string error;
    std::string expected;
    std::string source;

    std::filesystem::path const contractPath = onnxRoot / "policy.json";
    if (std::filesystem::exists(contractPath))
    {
        expected = readExportId(contractPath, "policy.json", error);
        if (expected.empty())
        {
            LOG_ERROR("%s", error.c_str());
            return {};
        }
        source = "policy.json";
    }

    for (auto const& component : components)
    {
        std::string const componentId
            = readExportId(onnxRoot / component / "config.json", component + " config.json", error);
        if (componentId.empty())
        {
            LOG_ERROR("%s", error.c_str());
            return {};
        }
        if (expected.empty())
        {
            expected = componentId;
            source = component;
            continue;
        }
        if (componentId != expected)
        {
            LOG_ERROR("pi0.5 %s was exported as %s but %s says %s; the components are from different exports",
                component.c_str(), componentId.c_str(), source.c_str(), expected.c_str());
            return {};
        }
    }
    return expected;
}

//! Every artifact in \p engineRoot that declares an export identity, each with the
//! label its rejection is reported under: a component directory declares one in its
//! config.json, the staged contract declares one directly.
std::vector<std::pair<std::filesystem::path, std::string>> taggedArtifacts(std::filesystem::path const& engineRoot)
{
    std::vector<std::pair<std::filesystem::path, std::string>> tagged;
    std::error_code ec;
    for (auto const& entry : std::filesystem::directory_iterator(engineRoot, ec))
    {
        if (std::filesystem::exists(entry.path() / "config.json"))
        {
            tagged.emplace_back(entry.path() / "config.json", "staged " + entry.path().filename().string() + " engine");
        }
    }
    if (std::filesystem::exists(engineRoot / "policy.json"))
    {
        tagged.emplace_back(engineRoot / "policy.json", "staged policy.json");
    }
    return tagged;
}

//! A root with no entry, existing or not, has nothing of an older export that
//! could outlive this build.
bool engineRootIsEmpty(std::filesystem::path const& engineRoot)
{
    std::error_code ec;
    auto const entries = std::filesystem::directory_iterator(engineRoot, ec);
    return ec || entries == std::filesystem::directory_iterator{};
}

//! True when everything \p engineRoot already holds declares export \p expected.
//! A root declaring none is refused rather than treated as fresh: assets/ and
//! text_tokenizer/ carry no identity, so a leftover cannot be told apart.
bool engineRootIsExport(std::filesystem::path const& engineRoot, std::string const& expected)
{
    auto const tagged = taggedArtifacts(engineRoot);
    if (tagged.empty())
    {
        LOG_ERROR(
            "pi0.5 engine root %s is not empty but names no export; build into an empty directory so "
            "no untagged sidecar of an earlier export survives",
            engineRoot.string().c_str());
        return false;
    }
    for (auto const& [path, label] : tagged)
    {
        std::string error;
        std::string const staged = readExportId(path, label, error);
        if (staged.empty())
        {
            LOG_ERROR("%s", error.c_str());
            return false;
        }
        if (staged != expected)
        {
            LOG_ERROR(
                "pi0.5 %s is from export %s but this build is export %s; rebuilding into %s would keep "
                "that export's untagged assets/ and text_tokenizer/, whatever it replaces. "
                "Build into an empty directory",
                label.c_str(), staged.c_str(), expected.c_str(), engineRoot.string().c_str());
            return false;
        }
    }
    return true;
}

} // namespace

bool verifyPi05ExportIds(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components)
{
    // Part of what makes an export buildable, so it is checked here rather than with the
    // sidecars the engines are staged beside: Pi05Policy refuses to load without it.
    if (!std::filesystem::exists(onnxRoot / "policy.json"))
    {
        LOG_ERROR(
            "pi0.5 export has no policy.json at %s; re-run the export", (onnxRoot / "policy.json").string().c_str());
        return false;
    }
    std::string const expected = sourceExportId(onnxRoot, components);
    if (expected.empty())
    {
        return false;
    }
    if (!engineRootIsEmpty(engineRoot) && !engineRootIsExport(engineRoot, expected))
    {
        return false;
    }
    LOG_INFO("pi0.5 export %s: %zu component(s) verified", expected.c_str(), components.size());
    return true;
}

std::vector<std::string> policyComponents(std::filesystem::path const& onnxRoot)
{
    // Whether cond/ belongs to this export is a property of the action contract, not of
    // which directories survived: a hoisted export missing cond/ would otherwise build
    // and report success, and fail only when the runtime looks for the engine.
    std::vector<std::string> components{"visual", "prefix", "action"};
    nlohmann::json actionConfig;
    if (!builder::loadJsonConfig((onnxRoot / "action" / "config.json").string(), actionConfig))
    {
        LOG_ERROR("pi0.5 export has no readable action/config.json under %s", onnxRoot.string().c_str());
        return {};
    }
    // Top level, matching where the exporter writes it and where Pi05Runtime reads it.
    bool const hoisted = actionConfig.value("hoisted_adarms_cond", false);
    if (hoisted)
    {
        components.emplace_back("cond");
    }
    else if (std::filesystem::exists(onnxRoot / "cond"))
    {
        LOG_WARNING("pi0.5 export has a cond/ directory but action/config.json does not claim it; ignoring it");
    }
    return components;
}

bool buildPi05Policy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize)
{
    if (components.empty())
    {
        LOG_ERROR("pi0.5 build was given no components; an empty set would stage sidecars and report success");
        return false;
    }
    if (!verifyPi05ExportIds(onnxRoot, engineRoot, components))
    {
        return false;
    }
    for (auto const& component : components)
    {
        if (!std::filesystem::exists(onnxRoot / component))
        {
            LOG_ERROR("pi0.5 export declares component %s but %s does not exist; re-run the export", component.c_str(),
                (onnxRoot / component).string().c_str());
            return false;
        }
        Pi05Builder builder(onnxRoot / component, engineRoot / component, component, maxBatchSize);
        if (!builder.build())
        {
            LOG_ERROR("Failed to build pi0.5 %s engine.", component.c_str());
            return false;
        }
        LOG_INFO("pi0.5 %s engine built successfully.", component.c_str());
    }
    return stageRuntimeArtifacts(onnxRoot, engineRoot, components);
}

Pi05Builder::Pi05Builder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
    std::string const& component, int32_t maxBatchSize)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mComponent(component)
    , mMaxBatchSize(maxBatchSize)
    , mEngineFilename(defaultEngineFilename(component))
{
}

bool Pi05Builder::build()
{
    if (!parseConfig())
    {
        return false;
    }

    // pi0.5 graphs otherwise lower entirely to trt:: native ops, so a failed
    // load is benign -- except for an export that carries a plugin node.
    auto pluginHandles = loadEdgellmPluginLib();
    if (!pluginHandles && mComponent == "action")
    {
        LOG_ERROR(
            "pi0.5 %s carries a trt_edgellm plugin node; set EDGELLM_PLUGIN_PATH to "
            "libNvInfer_edgellm_plugin.so",
            mComponent.c_str());
        return false;
    }

    std::string const lunowudFlags = builder::applyCompileWorkarounds();
    if (!lunowudFlags.empty())
    {
        LOG_INFO("Using __LUNOWUD=%s", lunowudFlags.c_str());
    }
    auto [trtBuilder, network] = builder::createBuilderAndNetwork();
    if (!trtBuilder || !network)
    {
        return false;
    }

    std::filesystem::path const onnxPath = mOnnxDir / mOnnxFilename;
    auto parser = builder::parseOnnxModel(network.get(), onnxPath.string());
    if (!parser)
    {
        return false;
    }

    LOG_DEBUG("%s", builder::printNetworkInfo(network.get(), ("pi0.5 " + mComponent).c_str()).c_str());

    if (!validateProfileNames(*network))
    {
        return false;
    }

    auto config = builder::createBuilderConfig(trtBuilder.get());
    if (!config)
    {
        return false;
    }

    if (!setupOptimizationProfile(*trtBuilder, *config, *network))
    {
        return false;
    }

    if (!std::filesystem::exists(mEngineDir) && !std::filesystem::create_directories(mEngineDir))
    {
        LOG_ERROR("Failed to create pi0.5 engine directory %s", mEngineDir.string().c_str());
        return false;
    }

    std::filesystem::path const enginePath = mEngineDir / mEngineFilename;
    if (!builder::buildAndSerializeEngine(trtBuilder.get(), network.get(), config.get(), enginePath.string()))
    {
        return false;
    }

    nlohmann::json builderConfig = mModelConfig.value("builder_config", nlohmann::json::object());
    builderConfig["max_batch_size"] = mMaxBatchSize;
    return builder::saveConfigWithBuilderInfo(mEngineDir, mModelConfig, builderConfig);
}

bool Pi05Builder::parseConfig()
{
    std::filesystem::path const configPath = mOnnxDir / "config.json";
    if (!builder::loadJsonConfig(configPath.string(), mModelConfig))
    {
        return false;
    }

    std::string const modelVersion = mModelConfig.value(trt_edgellm::binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // A contract mismatch surfaces here rather than as a missing plugin creator deep
    // inside the ONNX parser, where the error points at the plugin library instead.
    int32_t const contractVersion = mModelConfig.value("contract_version", 0);
    if (contractVersion != kContractVersion)
    {
        LOG_ERROR(
            "pi0.5 %s was exported against contract version %d but this runtime implements %d; "
            "re-export it with tensorrt-edgellm-export",
            mComponent.c_str(), contractVersion, kContractVersion);
        return false;
    }

    std::string const configComponent = mModelConfig.value("component", mComponent);
    if (configComponent != mComponent)
    {
        LOG_ERROR("pi0.5 component mismatch: CLI requested %s but config contains %s", mComponent.c_str(),
            configComponent.c_str());
        return false;
    }
    mOnnxFilename = mModelConfig.value("onnx_filename", mOnnxFilename);
    mEngineFilename = mModelConfig.value("engine_filename", mEngineFilename);
    return true;
}

bool Pi05Builder::inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const
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

bool Pi05Builder::validateProfileNames(nvinfer1::INetworkDefinition const& network) const
{
    nlohmann::json const profileJson = mModelConfig.value("optimization_profile", nlohmann::json::object());
    bool ok = true;
    for (auto const& item : profileJson.items())
    {
        if (!inputExists(network, item.key()))
        {
            LOG_ERROR("pi0.5 %s profile names %s, which is not an ONNX input of this graph", mComponent.c_str(),
                item.key().c_str());
            ok = false;
        }
    }
    // The batch maps are consulted by name when the profile is widened, so a key that
    // names nothing in the profile would be silently ignored and leave that input at
    // its exported bound while the rest of the engine grew.
    nlohmann::json const batchAxis = mModelConfig.value("batch_axis", nlohmann::json::object());
    nlohmann::json const batchStride = mModelConfig.value("batch_stride", nlohmann::json::object());
    nlohmann::json const batchBias = mModelConfig.value("batch_bias", nlohmann::json::object());
    // Bound to named locals: items() holds a reference into its json, so iterating the
    // temporary that value() returns would read it after the init-statement destroys it.
    std::pair<char const*, nlohmann::json const&> const maps[]{
        {"batch_axis", batchAxis}, {"batch_stride", batchStride}, {"batch_bias", batchBias}};
    for (auto const& [map, entries] : maps)
    {
        for (auto const& item : entries.items())
        {
            if (!profileJson.contains(item.key()))
            {
                LOG_ERROR("pi0.5 %s %s names %s, which the optimization profile does not declare", mComponent.c_str(),
                    map, item.key().c_str());
                ok = false;
                continue;
            }
            // Stride and bias scale an axis; without one declared they are read and discarded.
            if (std::string(map) != "batch_axis" && !batchAxis.contains(item.key()))
            {
                LOG_ERROR(
                    "pi0.5 %s %s names %s, which declares no batch_axis", mComponent.c_str(), map, item.key().c_str());
                ok = false;
            }
        }
    }
    for (auto const& item : batchStride.items())
    {
        if (item.value().is_number_integer() && item.value().get<int64_t>() <= 0)
        {
            LOG_ERROR("pi0.5 %s batch_stride for %s is %ld; a request occupies at least one slot", mComponent.c_str(),
                item.key().c_str(), item.value().get<int64_t>());
            ok = false;
        }
    }
    return ok;
}

bool Pi05Builder::setupOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network) const
{
    nlohmann::json const profileJson = mModelConfig.value("optimization_profile", nlohmann::json::object());
    if (profileJson.empty())
    {
        LOG_INFO("pi0.5 %s has no dynamic optimization profile", mComponent.c_str());
        return true;
    }

    nlohmann::json const batchAxis = mModelConfig.value("batch_axis", nlohmann::json::object());
    // How many slots along the batch axis one request occupies. Only the paged K/V
    // pool needs more than one: it grows by a whole sequence's pages per request.
    nlohmann::json const batchStride = mModelConfig.value("batch_stride", nlohmann::json::object());
    // Constant slots the batch axis carries on top of the per-request ones, for a
    // prefix-sum input whose length is one past the request count.
    nlohmann::json const batchBias = mModelConfig.value("batch_bias", nlohmann::json::object());

    auto* profile = builder.createOptimizationProfile();
    bool ok = true;
    for (auto const& item : profileJson.items())
    {
        std::string const& name = item.key();
        nlohmann::json const& shape = item.value();
        nvinfer1::Dims const minDims = dimsFromJson(shape.at("min"));
        nvinfer1::Dims const optDims = dimsFromJson(shape.at("opt"));
        nvinfer1::Dims maxDims = dimsFromJson(shape.at("max"));

        if (!dimsLessEqual(minDims, optDims) || !dimsLessEqual(optDims, maxDims))
        {
            LOG_ERROR("pi0.5 %s profile for %s violates min <= opt <= max", mComponent.c_str(), name.c_str());
            return false;
        }

        // The contract names the batch axis per input; an input without one
        // (e.g. a per-request scalar) keeps its exported bound.
        if (mMaxBatchSize > 1 && batchAxis.contains(name) && !batchAxis.at(name).is_null())
        {
            int32_t const axis = batchAxis.at(name).get<int32_t>();
            if (axis < 0 || axis >= maxDims.nbDims)
            {
                LOG_ERROR("pi0.5 %s batch_axis %d for %s is out of range", mComponent.c_str(), axis, name.c_str());
                return false;
            }
            int64_t const stride = batchStride.value(name, 1);
            int64_t const bias = batchBias.value(name, 0);
            maxDims.d[axis] = std::max(maxDims.d[axis], static_cast<int64_t>(mMaxBatchSize) * stride + bias);
        }
        ok &= builder::setOptimizationProfile(profile, name.c_str(), minDims, optDims, maxDims);
    }

    if (!ok)
    {
        LOG_ERROR("Failed to set pi0.5 %s optimization profile", mComponent.c_str());
        return false;
    }

    LOG_DEBUG("%s", builder::printOptimizationProfile(profile, (mComponent + "_profile").c_str(), &network).c_str());
    config.addOptimizationProfile(profile);
    return true;
}

} // namespace pi05
} // namespace trt_edgellm
