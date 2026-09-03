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

#include "runtime/state/externalWeightManager.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/trtUtils.h"
#include "profiling/nvtx_wrapper.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/weight/checkpointReader.h"
#include "runtime/weight/checkpointWeightAssemble.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

using Json = nlohmann::json;

constexpr size_t kWeightAlignment = 256;
constexpr size_t kPackedNvfp4ArenaAlignment = 2U << 20;
constexpr char const* kStorageAliasField = "storage_alias_of";
constexpr char const* kRegisterPluginWeightSymbol = "edgellmRegisterFusedNvfp4WeightResource";
constexpr char const* kUnregisterPluginWeightSymbol = "edgellmUnregisterFusedNvfp4WeightResource";
using RegisterPluginWeightFn = bool (*)(int32_t, int32_t, void const*, void const*, int32_t, int32_t, int32_t);
using UnregisterPluginWeightFn = bool (*)(int32_t, int32_t);

size_t alignWeightBytes(size_t bytes)
{
    ELLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - (kWeightAlignment - 1), "Weight arena size overflow");
    return (bytes + kWeightAlignment - 1) & ~(kWeightAlignment - 1);
}

bool isPackedNvfp4Binding(Json const& binding)
{
    return binding.value("source_layout", "") == "nvfp4_packed";
}

bool isPluginResourceBinding(Json const& binding)
{
    bool const hasResourceId = binding.contains("plugin_resource_id");
    bool const hasResourceKind = binding.contains("plugin_resource_kind");
    ELLM_CHECK(hasResourceId == hasResourceKind,
        "Plugin resource bindings must define both plugin_resource_id and plugin_resource_kind");
    return hasResourceId;
}

bool registerPluginWeightResource(int32_t deviceId, int32_t resourceId, void const* weight, void const* scale,
    int32_t outFeatures, int32_t inFeatures, int32_t scaleCols)
{
    auto registerResource = reinterpret_cast<RegisterPluginWeightFn>(dlsym(RTLD_DEFAULT, kRegisterPluginWeightSymbol));
    ELLM_CHECK(registerResource != nullptr, std::string{kRegisterPluginWeightSymbol} + " was not found in the plugin");
    return registerResource(deviceId, resourceId, weight, scale, outFeatures, inFeatures, scaleCols);
}

bool unregisterPluginWeightResource(int32_t deviceId, int32_t resourceId) noexcept
{
    auto unregisterResource
        = reinterpret_cast<UnregisterPluginWeightFn>(dlsym(RTLD_DEFAULT, kUnregisterPluginWeightSymbol));
    return unregisterResource != nullptr && unregisterResource(deviceId, resourceId);
}

Coords bindingShape(Json const& binding)
{
    ELLM_CHECK(binding.contains("shape") && binding["shape"].is_array(),
        "Checkpoint binding is missing a static shape: " + binding.value("engine_name", ""));
    return Coords(binding["shape"].get<std::vector<int64_t>>());
}

Json readConfig(std::filesystem::path const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.is_open(), "Failed to open runtime config: " + path.string());
    try
    {
        return Json::parse(file);
    }
    catch (Json::parse_error const& error)
    {
        throw std::runtime_error("Failed to parse " + path.string() + ": " + error.what());
    }
}

bool isEmbeddingBinding(Json const& binding)
{
    return binding.value("role", "") == "embedding" || binding.value("engine_name", "") == "__embedding__";
}

bool isPleEmbeddingBinding(Json const& binding)
{
    return binding.value("role", "") == "ple_embedding" || binding.value("engine_name", "") == "__ple_embedding__";
}

size_t nonnegativeSize(Json const& value, std::string const& label);

uint8_t hexNibble(char value)
{
    if (value >= '0' && value <= '9')
    {
        return static_cast<uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f')
    {
        return static_cast<uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F')
    {
        return static_cast<uint8_t>(value - 'A' + 10);
    }
    throw std::runtime_error("Checkpoint identity contains non-hexadecimal sample data");
}

size_t decodedHexSize(std::string const& encoded)
{
    ELLM_CHECK(encoded.size() % 2 == 0, "Checkpoint identity sample must contain whole bytes");
    return encoded.size() / 2;
}

bool matchesHex(uint8_t const* actual, std::string const& encoded)
{
    for (size_t index = 0; index < encoded.size() / 2; ++index)
    {
        uint8_t const expected
            = static_cast<uint8_t>((hexNibble(encoded[index * 2]) << 4) | hexNibble(encoded[index * 2 + 1]));
        if (actual[index] != expected)
        {
            return false;
        }
    }
    return true;
}

void validateCheckpointIdentity(
    CheckpointReader const& checkpoint, Json const& identity, std::string const& source, bool validateContent)
{
    ELLM_CHECK(identity.is_object() && identity.contains("tensors") && identity["tensors"].is_object(),
        "Malformed " + source + " checkpoint identity");
    std::string const buildSource = identity.value("build_source", std::string{"<unknown>"});
    for (auto const& [name, expected] : identity["tensors"].items())
    {
        ELLM_CHECK(expected.is_object() && expected.contains("dtype") && expected["dtype"].is_string()
                && expected.contains("shape") && expected["shape"].is_array() && expected.contains("bytes")
                && expected.contains("samples") && expected["samples"].is_array(),
            "Malformed checkpoint identity for tensor " + name);

        CheckpointReader::View actual;
        ELLM_CHECK(checkpoint.findHost(name, actual),
            "Runtime " + source + " checkpoint does not match build provider " + buildSource + ": missing tensor "
                + name);
        nvinfer1::DataType const expectedType
            = safetensors::dataTypeFromString(expected["dtype"].get_ref<std::string const&>());
        Coords const expectedShape(expected["shape"].get<std::vector<int64_t>>());
        size_t const expectedBytes = nonnegativeSize(expected["bytes"], "checkpoint identity byte count");
        ELLM_CHECK(actual.dtype == expectedType && actual.shape == expectedShape && actual.bytes == expectedBytes,
            "Runtime " + source + " checkpoint does not match build provider " + buildSource
                + ": metadata differs for tensor " + name);

        if (!validateContent)
        {
            continue;
        }
        for (Json const& sample : expected["samples"])
        {
            ELLM_CHECK(sample.is_object() && sample.contains("offset") && sample.contains("data")
                    && sample["data"].is_string(),
                "Malformed checkpoint identity sample for tensor " + name);
            size_t const offset = nonnegativeSize(sample["offset"], "checkpoint identity sample offset");
            std::string const& encoded = sample["data"].get_ref<std::string const&>();
            size_t const sampleBytes = decodedHexSize(encoded);
            ELLM_CHECK(offset <= actual.bytes && sampleBytes <= actual.bytes - offset,
                "Checkpoint identity sample exceeds tensor " + name);
            ELLM_CHECK(matchesHex(actual.data + offset, encoded),
                "Runtime " + source + " checkpoint does not match build provider " + buildSource
                    + ": sampled content differs for tensor " + name);
        }
    }
}

std::filesystem::path normalizedPath(std::filesystem::path const& path)
{
    std::error_code error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(path, error);
    if (!error)
    {
        return normalized;
    }
    return std::filesystem::absolute(path).lexically_normal();
}

bool checkpointFilesUnchanged(std::filesystem::path const& directory, Json const& identity)
{
    if (!identity.contains("build_source") || !identity["build_source"].is_string()
        || normalizedPath(directory) != normalizedPath(identity["build_source"].get<std::string>()))
    {
        return false;
    }
    if (!identity.contains("files") || !identity["files"].is_object() || identity["files"].empty())
    {
        return false;
    }
    Json const& files = identity["files"];
    for (auto const& [filename, expected] : files.items())
    {
        std::filesystem::path const relative(filename);
        ELLM_CHECK(
            !relative.empty() && !relative.is_absolute(), "Checkpoint identity file must be relative: " + filename);
        for (auto const& part : relative)
        {
            ELLM_CHECK(part != "..", "Checkpoint identity file escapes its directory: " + filename);
        }
        ELLM_CHECK(expected.is_object() && expected.contains("bytes") && expected.contains("mtime_ns"),
            "Malformed checkpoint identity file: " + filename);

        struct stat metadata{};
        if (::stat((directory / relative).c_str(), &metadata) != 0 || !S_ISREG(metadata.st_mode))
        {
            return false;
        }
        uint64_t const bytes = static_cast<uint64_t>(metadata.st_size);
        uint64_t const mtimeNs = static_cast<uint64_t>(metadata.st_mtim.tv_sec) * 1000000000ULL
            + static_cast<uint64_t>(metadata.st_mtim.tv_nsec);
        if (bytes != expected["bytes"].get<uint64_t>() || mtimeNs != expected["mtime_ns"].get<uint64_t>())
        {
            return false;
        }
    }
    return true;
}

void validateStorageAlias(Json const& alias, Json const& target)
{
    std::string const aliasName = alias.at("engine_name").get<std::string>();
    std::string const targetName = target.at("engine_name").get<std::string>();
    ELLM_CHECK(alias.value("embedding_scale", 1.0f) == 1.0f,
        "Checkpoint storage alias cannot apply an embedding scale: " + aliasName);
    ELLM_CHECK(target.value("embedding_scale", 1.0f) == 1.0f,
        "Checkpoint storage alias target cannot apply an embedding scale: " + targetName);
    ELLM_CHECK(!target.contains(kStorageAliasField),
        "Checkpoint storage aliases cannot form chains: " + aliasName + " -> " + targetName);

    Json aliasContract = alias;
    Json targetContract = target;
    for (char const* field : {"engine_name", "role", "embedding_scale", kStorageAliasField})
    {
        aliasContract.erase(field);
        targetContract.erase(field);
    }
    aliasContract["checkpoint_source"] = alias.value("checkpoint_source", "component");
    targetContract["checkpoint_source"] = target.value("checkpoint_source", "component");
    aliasContract["source_layout"] = alias.value("source_layout", "plugin");
    targetContract["source_layout"] = target.value("source_layout", "plugin");
    ELLM_CHECK(aliasContract == targetContract,
        "Checkpoint storage alias has a different materialization contract: " + aliasName + " -> " + targetName);
}

void appendCheckpointKeys(Json const& binding, std::vector<std::string>& keys)
{
    if (!binding.contains("checkpoint_keys"))
    {
        return;
    }
    ELLM_CHECK(binding["checkpoint_keys"].is_array(), "checkpoint_keys must be an array");
    for (auto const& key : binding["checkpoint_keys"])
    {
        ELLM_CHECK(key.is_string(), "checkpoint_keys entries must be strings");
        keys.push_back(key.get<std::string>());
    }
}

size_t nonnegativeSize(Json const& value, std::string const& label)
{
    ELLM_CHECK(value.is_number_integer() || value.is_number_unsigned(), label + " must be an integer");
    if (value.is_number_integer())
    {
        ELLM_CHECK(value.get<int64_t>() >= 0, label + " must be nonnegative");
    }
    uint64_t const result = value.get<uint64_t>();
    ELLM_CHECK(result <= std::numeric_limits<size_t>::max(), label + " exceeds the host size limit");
    return static_cast<size_t>(result);
}

std::vector<CheckpointReader::TensorLocation> checkpointLocations(Json const& bindings, std::string const& source)
{
    std::unordered_map<std::string, CheckpointReader::TensorLocation> unique;
    for (auto const& binding : bindings)
    {
        if (binding.value("checkpoint_source", "component") != source)
        {
            continue;
        }
        Json const locations = binding.value("checkpoint_locations", Json::object());
        ELLM_CHECK(locations.is_object(), "checkpoint_locations must be an object");
        Json const keys = binding.value("checkpoint_keys", Json::array());
        ELLM_CHECK(keys.is_array(), "checkpoint_keys must be an array");

        for (auto const& [name, value] : locations.items())
        {
            ELLM_CHECK(std::any_of(keys.begin(), keys.end(),
                           [&name](Json const& key) { return key.is_string() && key.get<std::string>() == name; }),
                "checkpoint_locations contains an entry outside checkpoint_keys: " + name);
            ELLM_CHECK(value.is_object() && value.contains("file") && value["file"].is_string()
                    && value.contains("dtype") && value["dtype"].is_string() && value.contains("shape")
                    && value["shape"].is_array() && value.contains("offset") && value.contains("bytes"),
                "Malformed explicit checkpoint tensor location: " + name);

            CheckpointReader::TensorLocation location{
                name,
                value["file"].get<std::string>(),
                value["dtype"].get<std::string>(),
                Coords(value["shape"].get<std::vector<int64_t>>()),
                nonnegativeSize(value["offset"], "checkpoint offset"),
                nonnegativeSize(value["bytes"], "checkpoint byte count"),
            };
            auto const existing = unique.find(name);
            if (existing == unique.end())
            {
                unique.emplace(name, std::move(location));
            }
            else
            {
                CheckpointReader::TensorLocation const& previous = existing->second;
                ELLM_CHECK(previous.file == location.file && previous.dtype == location.dtype
                        && previous.shape == location.shape && previous.offset == location.offset
                        && previous.bytes == location.bytes,
                    "Conflicting explicit checkpoint tensor location: " + name);
            }
        }
    }

    std::vector<CheckpointReader::TensorLocation> result;
    result.reserve(unique.size());
    for (auto& [_, location] : unique)
    {
        result.push_back(std::move(location));
    }
    std::sort(result.begin(), result.end(), [](auto const& left, auto const& right) { return left.name < right.name; });
    return result;
}

void loadSidecars(std::filesystem::path const& engineDir, std::filesystem::path const& configPath, Json const& config,
    std::vector<Tensor>& weights, cudaStream_t stream)
{
    Json const files = config.value("external_weight_files", Json::array());
    ELLM_CHECK(files.is_array(), "external_weight_files must be an array in " + configPath.string());

    for (auto const& entry : files)
    {
        ELLM_CHECK(entry.is_object() && entry.contains("file") && entry["file"].is_string(),
            "Malformed external weight entry in " + configPath.string());
        std::filesystem::path const path = engineDir / entry["file"].get<std::string>();
        std::vector<Tensor> loaded;
        ELLM_CHECK(
            safetensors::loadSafetensors(path, loaded, stream), "Failed to load external weights: " + path.string());

        if (entry.contains("tensors"))
        {
            ELLM_CHECK(entry["tensors"].is_array(), "External weight tensor list must be an array");
            for (auto const& expected : entry["tensors"])
            {
                ELLM_CHECK(expected.is_string(), "External weight tensor names must be strings");
                std::string const name = expected.get<std::string>();
                ELLM_CHECK(std::any_of(loaded.begin(), loaded.end(),
                               [&name](Tensor const& tensor) { return tensor.getName() == name; }),
                    "External weight " + name + " is missing from " + path.string());
            }
        }

        weights.reserve(weights.size() + loaded.size());
        std::move(loaded.begin(), loaded.end(), std::back_inserter(weights));
    }
}

void validateTensor(nvinfer1::ICudaEngine const& engine, Tensor const& tensor)
{
    std::string const& name = tensor.getName();
    ELLM_CHECK(isEngineInput(engine, name), "External weight " + name + " is not an engine input");

    nvinfer1::DataType const expectedType = engine.getTensorDataType(name.c_str());
    ELLM_CHECK(tensor.getDataType() == expectedType,
        "External weight " + name + " has dtype " + std::to_string(static_cast<int32_t>(tensor.getDataType()))
            + "; engine expects " + std::to_string(static_cast<int32_t>(expectedType)));

    nvinfer1::Dims const expectedShape = engine.getTensorShape(name.c_str());
    ELLM_CHECK(!hasDynamicDims(expectedShape), "External weight input " + name + " must have a static shape");
    ELLM_CHECK(dimsEqual(expectedShape, tensor.getShape().getTRTDims()),
        "External weight " + name + " has shape " + tensor.getShape().formatString() + "; engine expects "
            + dimsToString(expectedShape));
}

} // namespace

ExternalWeightManager::ExternalWeightManager(ExternalWeightManager&& other) noexcept
{
    *this = std::move(other);
}

ExternalWeightManager& ExternalWeightManager::operator=(ExternalWeightManager&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    releasePluginResources();
    mWeightStorage = std::move(other.mWeightStorage);
    mWeights = std::move(other.mWeights);
    mPluginResourceTensors = std::move(other.mPluginResourceTensors);
    mPluginResourceRegistrations = std::move(other.mPluginResourceRegistrations);
    mEmbedding = std::move(other.mEmbedding);
    mPleEmbedding = std::move(other.mPleEmbedding);
    mLoaded = std::exchange(other.mLoaded, false);
    mValidated = std::exchange(other.mValidated, false);
    mRegistered = std::exchange(other.mRegistered, false);
    other.mPluginResourceRegistrations.clear();
    return *this;
}

ExternalWeightManager::~ExternalWeightManager() noexcept
{
    releasePluginResources();
}

void ExternalWeightManager::releasePluginResources() noexcept
{
    for (PluginResourceRegistration const& registration : mPluginResourceRegistrations)
    {
        if (!unregisterPluginWeightResource(registration.deviceId, registration.resourceId))
        {
            LOG_WARNING("Failed to unregister fused NVFP4 weight resource %d on device %d", registration.resourceId,
                registration.deviceId);
        }
    }
    mPluginResourceRegistrations.clear();
}

void ExternalWeightManager::load(std::filesystem::path const& engineDir, std::filesystem::path const& configPath,
    cudaStream_t stream, std::filesystem::path const& componentCheckpointDir,
    std::filesystem::path const& targetCheckpointDir, std::optional<int32_t> tpRank, std::optional<int32_t> tpSize)
{
    NVTX_SCOPED_RANGE(nvtx_model_weight_preparation, "MODEL_WEIGHT_PREPARATION", nvtx_colors::YELLOW);
    ELLM_CHECK(!mLoaded, "ExternalWeightManager::load called more than once");
    Json const config = readConfig(configPath);
    Json bindings = config.value("checkpoint_weight_bindings", Json::array());
    ELLM_CHECK(bindings.is_array(), "checkpoint_weight_bindings must be an array in " + configPath.string());
    ELLM_CHECK(
        tpRank.has_value() == tpSize.has_value(), "External weight TP rank and TP size must be provided together");
    ELLM_CHECK(!tpRank.has_value() || *tpRank >= 0, "External weight TP rank must be nonnegative");
    ELLM_CHECK(!tpSize.has_value() || *tpSize > 0, "External weight TP size must be positive");
    ELLM_CHECK(!tpRank.has_value() || !tpSize.has_value() || *tpRank < *tpSize,
        "External weight TP rank must be smaller than TP size");
    for (Json& binding : bindings)
    {
        ELLM_CHECK(binding.is_object(), "checkpoint_weight_bindings entries must be objects");
        if (tpRank.has_value())
        {
            binding["runtime_rank"] = *tpRank;
        }
        if (!binding.contains("tp_shard"))
        {
            continue;
        }
        ELLM_CHECK(tpRank.has_value(), "TP checkpoint binding requires a runtime TP rank");
        Json& shard = binding["tp_shard"];
        ELLM_CHECK(shard.is_object(), "tp_shard must be an object");
        int32_t const shardSize = shard.value("size", 0);
        ELLM_CHECK(shardSize > 1, "tp_shard.size must be greater than one");
        ELLM_CHECK(!tpSize.has_value() || shardSize == *tpSize, "tp_shard.size does not match the runtime TP size");
        shard["rank"] = *tpRank;
    }

    using Clock = std::chrono::steady_clock;
    auto const start = Clock::now();
    std::chrono::milliseconds checkpointIndexTime{0};
    std::chrono::milliseconds checkpointValidationTime{0};
    std::chrono::milliseconds checkpointRegistrationTime{0};
    std::chrono::milliseconds arenaAllocationTime{0};
    std::chrono::milliseconds transformTime{0};
    size_t weightArenaBytes = 0;
    size_t aliasedWeightBytes = 0;
    size_t peakRegisteredSourceBytes = 0;
    if (bindings.empty())
    {
        loadSidecars(engineDir, configPath, config, mWeights, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else
    {
        std::filesystem::path componentDir = componentCheckpointDir;
        if (componentDir.empty())
        {
            componentDir = config.value("checkpoint_dir", "");
        }
        ELLM_CHECK(!componentDir.empty(), "A component checkpoint directory is required for runtime weight loading");

        std::vector<std::string> componentKeys;
        std::vector<std::string> targetKeys;
        for (auto const& binding : bindings)
        {
            std::string const source = binding.value("checkpoint_source", "component");
            if (source == "component")
            {
                appendCheckpointKeys(binding, componentKeys);
            }
            else
            {
                ELLM_CHECK(source == "target", "Unsupported checkpoint_source: " + source);
                ELLM_CHECK(
                    !targetCheckpointDir.empty(), "A target checkpoint directory is required for target-model weights");
                appendCheckpointKeys(binding, targetKeys);
            }
        }

        bool const sharedTarget = !targetKeys.empty()
            && std::filesystem::absolute(componentDir).lexically_normal()
                == std::filesystem::absolute(targetCheckpointDir).lexically_normal();
        if (sharedTarget)
        {
            componentKeys.insert(componentKeys.end(), targetKeys.begin(), targetKeys.end());
        }

        std::vector<CheckpointReader::TensorLocation> componentLocations = checkpointLocations(bindings, "component");
        std::vector<CheckpointReader::TensorLocation> targetLocations = checkpointLocations(bindings, "target");
        if (sharedTarget)
        {
            componentLocations.insert(componentLocations.end(), std::make_move_iterator(targetLocations.begin()),
                std::make_move_iterator(targetLocations.end()));
            targetLocations.clear();
        }

        auto phaseStart = Clock::now();
        CheckpointReader component(componentDir, componentLocations);
        std::unique_ptr<CheckpointReader> target;
        if (!targetKeys.empty() && !sharedTarget)
        {
            target = std::make_unique<CheckpointReader>(targetCheckpointDir, targetLocations);
        }
        checkpointIndexTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart);

        ELLM_CHECK(config.contains("checkpoint_identity"),
            "Runtime config with checkpoint-backed weights has no checkpoint identity; rebuild the engine");
        Json const& identity = config["checkpoint_identity"];
        ELLM_CHECK(identity.is_object() && identity.value("version", 0) == 1 && identity.contains("sources")
                && identity["sources"].is_object(),
            "Malformed checkpoint_identity in " + configPath.string());
        Json const& sources = identity["sources"];
        phaseStart = Clock::now();
        if (!componentKeys.empty())
        {
            ELLM_CHECK(sources.contains("component"), "Runtime config has no identity for its component checkpoint");
            bool const unchanged = checkpointFilesUnchanged(componentDir, sources["component"]);
            validateCheckpointIdentity(component, sources["component"], "component", !unchanged);
        }
        if (!targetKeys.empty())
        {
            ELLM_CHECK(sources.contains("target"), "Runtime config has no identity for its target checkpoint");
            std::filesystem::path const& targetDir = sharedTarget ? componentDir : targetCheckpointDir;
            bool const unchanged = checkpointFilesUnchanged(targetDir, sources["target"]);
            validateCheckpointIdentity(sharedTarget ? component : *target, sources["target"], "target", !unchanged);
        }
        checkpointValidationTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart);

        auto checkpointFor = [&](Json const& binding) -> CheckpointReader& {
            std::string const source = binding.value("checkpoint_source", "component");
            if (source == "component")
            {
                return component;
            }
            ELLM_CHECK(source == "target", "Unsupported checkpoint_source: " + source);
            ELLM_CHECK(
                !targetCheckpointDir.empty(), "A target checkpoint directory is required for target-model weights");
            if (sharedTarget)
            {
                return component;
            }
            ELLM_CHECK(target != nullptr, "Target checkpoint reader was not initialized");
            return *target;
        };

        phaseStart = Clock::now();
        struct BindingLayout
        {
            Coords shape;
            nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
            size_t offset{0};
            bool materialize{true};
        };
        std::vector<BindingLayout> layouts(bindings.size());
        std::unordered_map<std::string, size_t> bindingIndices;
        size_t storageBytes = 0;
        for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
        {
            Json const& binding = bindings[bindingIndex];
            ELLM_CHECK(binding.is_object(), "checkpoint_weight_bindings entries must be objects");
            ELLM_CHECK(binding.contains("engine_name") && binding["engine_name"].is_string(),
                "Checkpoint binding is missing engine_name");
            std::string const name = binding["engine_name"].get<std::string>();
            ELLM_CHECK(
                bindingIndices.emplace(name, bindingIndex).second, "Duplicate checkpoint binding engine_name: " + name);
            Coords const shape = bindingShape(binding);
            ELLM_CHECK(binding.contains("dtype") && binding["dtype"].is_string(),
                "Checkpoint binding is missing its destination dtype: " + binding.value("engine_name", ""));
            nvinfer1::DataType const dtype
                = safetensors::dataTypeFromString(binding["dtype"].get_ref<std::string const&>());
            layouts[bindingIndex].shape = shape;
            layouts[bindingIndex].dtype = dtype;
        }
        bool const hasPackedNvfp4Weights = std::any_of(bindings.begin(), bindings.end(),
            [](Json const& binding) { return !binding.contains(kStorageAliasField) && isPackedNvfp4Binding(binding); });
        auto allocateBindings = [&](bool packedNvfp4) {
            for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
            {
                Json const& binding = bindings[bindingIndex];
                if (binding.contains(kStorageAliasField) || isPackedNvfp4Binding(binding) != packedNvfp4)
                {
                    continue;
                }
                BindingLayout& layout = layouts[bindingIndex];
                size_t const bytes = static_cast<size_t>(layout.shape.volume()) * utils::getTypeSize(layout.dtype);
                ELLM_CHECK(bytes > 0, "Checkpoint binding has an empty tensor: " + binding.value("engine_name", ""));
                storageBytes = alignWeightBytes(storageBytes);
                layout.offset = storageBytes;
                ELLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - storageBytes, "Weight arena size overflow");
                storageBytes += bytes;
            }
        };
        allocateBindings(/*packedNvfp4=*/true);
        allocateBindings(/*packedNvfp4=*/false);
        for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
        {
            Json const& binding = bindings[bindingIndex];
            if (!binding.contains(kStorageAliasField))
            {
                continue;
            }
            ELLM_CHECK(binding[kStorageAliasField].is_string(), "storage_alias_of must name an engine binding");
            std::string const targetName = binding[kStorageAliasField].get<std::string>();
            auto const target = bindingIndices.find(targetName);
            ELLM_CHECK(
                target != bindingIndices.end(), "Checkpoint storage alias names an unknown binding: " + targetName);
            ELLM_CHECK(target->second != bindingIndex, "Checkpoint binding cannot alias itself: " + targetName);
            validateStorageAlias(binding, bindings[target->second]);

            BindingLayout& layout = layouts[bindingIndex];
            BindingLayout const& targetLayout = layouts[target->second];
            ELLM_CHECK(layout.shape == targetLayout.shape && layout.dtype == targetLayout.dtype,
                "Checkpoint storage alias shape or dtype differs from target: "
                    + binding["engine_name"].get<std::string>());
            layout.offset = targetLayout.offset;
            layout.materialize = false;
            size_t const bytes = static_cast<size_t>(layout.shape.volume()) * utils::getTypeSize(layout.dtype);
            ELLM_CHECK(
                bytes <= std::numeric_limits<size_t>::max() - aliasedWeightBytes, "Aliased weight byte count overflow");
            aliasedWeightBytes += bytes;
        }
        storageBytes = alignWeightBytes(storageBytes);
        ELLM_CHECK(
            storageBytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), "Weight arena is too large");
        size_t const arenaAlignmentPadding = hasPackedNvfp4Weights ? kPackedNvfp4ArenaAlignment - 1 : 0;
        ELLM_CHECK(arenaAlignmentPadding <= std::numeric_limits<size_t>::max() - storageBytes,
            "Aligned weight arena size overflow");
        size_t const allocationBytes = storageBytes + arenaAlignmentPadding;
        mWeightStorage = Tensor(Coords{static_cast<int64_t>(allocationBytes)}, DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "ExternalWeightManager::weightArena");
        weightArenaBytes = allocationBytes;
        arenaAllocationTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart);

        phaseStart = Clock::now();
        auto* storage = static_cast<uint8_t*>(mWeightStorage.rawPointer());
        if (hasPackedNvfp4Weights)
        {
            uintptr_t const address = reinterpret_cast<uintptr_t>(storage);
            uintptr_t const alignedAddress
                = (address + kPackedNvfp4ArenaAlignment - 1) & ~(kPackedNvfp4ArenaAlignment - 1);
            storage = reinterpret_cast<uint8_t*>(alignedAddress);
        }
        std::vector<Tensor> preparedWeights;
        preparedWeights.reserve(bindings.size());
        for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
        {
            Json const& binding = bindings[bindingIndex];
            BindingLayout const& layout = layouts[bindingIndex];
            preparedWeights.emplace_back(storage + layout.offset, layout.shape, DeviceType::kGPU, layout.dtype,
                binding.at("engine_name").get<std::string>());
        }
        auto materializePhase = [&](CheckpointWeightPhase phase) {
            for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
            {
                Json const& binding = bindings[bindingIndex];
                if (!layouts[bindingIndex].materialize || checkpointWeightPhase(binding) != phase)
                {
                    continue;
                }
                CheckpointReader& checkpoint = checkpointFor(binding);
                std::vector<std::string> keys;
                appendCheckpointKeys(binding, keys);
                bool const managesSourceRegistration = checkpointWeightManagesSourceRegistration(binding);
                if (!managesSourceRegistration)
                {
                    checkpoint.registerTensors(keys);
                }
                peakRegisteredSourceBytes = std::max(peakRegisteredSourceBytes, component.peakRegisteredBytes());
                if (target != nullptr)
                {
                    peakRegisteredSourceBytes = std::max(peakRegisteredSourceBytes, target->peakRegisteredBytes());
                }
                try
                {
                    loadCheckpointWeight(checkpoint, binding, preparedWeights, preparedWeights[bindingIndex], stream);
                    if (!managesSourceRegistration)
                    {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    peakRegisteredSourceBytes = std::max(peakRegisteredSourceBytes, checkpoint.peakRegisteredBytes());
                }
                catch (...)
                {
                    // The source registration must outlive any transform already
                    // queued on this stream, even when a later launch fails.
                    cudaError_t const drainStatus = cudaStreamSynchronize(stream);
                    if (drainStatus != cudaSuccess)
                    {
                        LOG_WARNING(
                            "Failed to drain checkpoint weight transforms: %s", cudaGetErrorString(drainStatus));
                    }
                    checkpoint.unregisterTensors();
                    throw;
                }
                if (!managesSourceRegistration)
                {
                    checkpoint.unregisterTensors();
                    checkpoint.discardTensors(keys);
                }
            }
        };
        materializePhase(CheckpointWeightPhase::kPrerequisite);
        materializePhase(CheckpointWeightPhase::kWeight);
        ELLM_CHECK(component.registeredBytes() == 0,
            "Checkpoint source mappings remain registered after component weight materialization");
        ELLM_CHECK(target == nullptr || target->registeredBytes() == 0,
            "Checkpoint source mappings remain registered after target weight materialization");
        auto registrationTime = component.registrationTime();
        if (target != nullptr)
        {
            registrationTime += target->registrationTime();
        }
        checkpointRegistrationTime = std::chrono::duration_cast<std::chrono::milliseconds>(registrationTime);
        transformTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart);

        struct PluginResourceParts
        {
            Tensor const* weight{nullptr};
            Tensor const* scale{nullptr};
        };
        std::unordered_map<int32_t, PluginResourceParts> pluginResources;
        for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
        {
            Json const& binding = bindings[bindingIndex];
            if (!isPluginResourceBinding(binding))
            {
                continue;
            }
            int32_t const resourceId = binding.at("plugin_resource_id").get<int32_t>();
            ELLM_CHECK(resourceId >= 0, "plugin_resource_id must be nonnegative");
            std::string const kind = binding.at("plugin_resource_kind").get<std::string>();
            PluginResourceParts& parts = pluginResources[resourceId];
            if (kind == "weight")
            {
                ELLM_CHECK(parts.weight == nullptr, "Duplicate plugin weight resource id");
                parts.weight = &preparedWeights[bindingIndex];
            }
            else
            {
                ELLM_CHECK(kind == "scale", "Unsupported plugin resource kind: " + kind);
                ELLM_CHECK(parts.scale == nullptr, "Duplicate plugin scale resource id");
                parts.scale = &preparedWeights[bindingIndex];
            }
        }
        int32_t deviceId = -1;
        CUDA_CHECK(cudaGetDevice(&deviceId));
        mPluginResourceRegistrations.reserve(mPluginResourceRegistrations.size() + pluginResources.size());
        for (auto const& [resourceId, parts] : pluginResources)
        {
            ELLM_CHECK(parts.weight != nullptr && parts.scale != nullptr,
                "Fused NVFP4 plugin resource is missing its weight or scale");
            Coords const weightShape = parts.weight->getShape();
            Coords const scaleShape = parts.scale->getShape();
            ELLM_CHECK(parts.weight->getDataType() == nvinfer1::DataType::kUINT8,
                "Fused NVFP4 plugin weights must use packed UINT8 storage");
            ELLM_CHECK(parts.scale->getDataType() == nvinfer1::DataType::kFP8,
                "Fused NVFP4 plugin block scales must use FP8 storage");
            ELLM_CHECK(weightShape.getNumDims() == 2 && scaleShape.getNumDims() == 2 && weightShape[0] == scaleShape[0],
                "Fused NVFP4 plugin resource shapes are inconsistent");
            ELLM_CHECK(weightShape[0] <= std::numeric_limits<int32_t>::max()
                    && weightShape[1] <= std::numeric_limits<int32_t>::max() / 2
                    && scaleShape[1] <= std::numeric_limits<int32_t>::max(),
                "Fused NVFP4 plugin resource shape exceeds INT32 limits");
            int32_t const outFeatures = static_cast<int32_t>(weightShape[0]);
            int32_t const inFeatures = static_cast<int32_t>(weightShape[1] * 2);
            int32_t const scaleCols = static_cast<int32_t>(scaleShape[1]);
            ELLM_CHECK(inFeatures % 16 == 0 && scaleCols == inFeatures / 16,
                "Fused NVFP4 plugin resource weight and scale widths are inconsistent");
            ELLM_CHECK(registerPluginWeightResource(deviceId, resourceId, parts.weight->rawPointer(),
                           parts.scale->rawPointer(), outFeatures, inFeatures, scaleCols),
                "Fused NVFP4 plugin rejected external weight resource " + std::to_string(resourceId));
            mPluginResourceRegistrations.push_back(PluginResourceRegistration{deviceId, resourceId});
        }

        for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex)
        {
            Json const& binding = bindings[bindingIndex];
            BindingLayout const& layout = layouts[bindingIndex];
            Tensor tensor = std::move(preparedWeights[bindingIndex]);
            if (isPluginResourceBinding(binding))
            {
                mPluginResourceTensors.push_back(std::move(tensor));
            }
            else if (isEmbeddingBinding(binding))
            {
                ELLM_CHECK(!mEmbedding.has_value(), "Runtime config has multiple embedding bindings");
                if (binding.at("engine_name").get<std::string>() != "__embedding__")
                {
                    mWeights.emplace_back(storage + layout.offset, layout.shape, DeviceType::kGPU, layout.dtype,
                        binding.at("engine_name").get<std::string>());
                }
                mEmbedding = std::move(tensor);
            }
            else if (isPleEmbeddingBinding(binding))
            {
                ELLM_CHECK(!mPleEmbedding.has_value(), "Runtime config has multiple PLE embedding bindings");
                mPleEmbedding = std::move(tensor);
            }
            else
            {
                mWeights.push_back(std::move(tensor));
            }
        }
    }

    // This is the startup/inference boundary. Source-layout tensors and
    // checkpoint mappings are gone; only immutable engine inputs and plugin
    // resources remain.
    mLoaded = true;
    if (!mWeights.empty() || !mPluginResourceTensors.empty() || mEmbedding.has_value() || mPleEmbedding.has_value())
    {
        int32_t const modelWeightTensorCount = static_cast<int32_t>(mWeights.size() + mPluginResourceTensors.size());
        auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
        if (bindings.empty())
        {
            LOG_INFO("Loaded %d transformed model weight tensor(s)%s%s in %lld ms", modelWeightTensorCount,
                mEmbedding.has_value() ? " and the embedding table" : "",
                mPleEmbedding.has_value() ? " and the PLE table" : "", static_cast<long long>(elapsed.count()));
        }
        else
        {
            LOG_INFO(
                "Prepared %d model weight tensor(s)%s%s from checkpoint in %lld ms "
                "(index %lld ms, validate %lld ms, map/register %lld ms, peak mapped source %zu bytes, "
                "arena %zu bytes, aliases saved %zu bytes in %lld ms, transform %lld ms)",
                modelWeightTensorCount, mEmbedding.has_value() ? " and the embedding table" : "",
                mPleEmbedding.has_value() ? " and the PLE table" : "", static_cast<long long>(elapsed.count()),
                static_cast<long long>(checkpointIndexTime.count()),
                static_cast<long long>(checkpointValidationTime.count()),
                static_cast<long long>(checkpointRegistrationTime.count()), peakRegisteredSourceBytes, weightArenaBytes,
                aliasedWeightBytes, static_cast<long long>(arenaAllocationTime.count()),
                static_cast<long long>(transformTime.count()));
        }
    }
}

std::optional<Tensor> ExternalWeightManager::takeEmbedding()
{
    ELLM_CHECK(mLoaded, "takeEmbedding called before load");
    std::optional<Tensor> embedding = std::move(mEmbedding);
    mEmbedding.reset();
    return embedding;
}

std::optional<Tensor> ExternalWeightManager::takePleEmbedding()
{
    ELLM_CHECK(mLoaded, "takePleEmbedding called before load");
    std::optional<Tensor> embedding = std::move(mPleEmbedding);
    mPleEmbedding.reset();
    return embedding;
}

void ExternalWeightManager::validateAgainstEngine(EngineExecutor const& executor, std::string_view engineLabel)
{
    ELLM_CHECK(mLoaded, "ExternalWeightManager::validateAgainstEngine called before load");
    ELLM_CHECK(!mValidated, "ExternalWeightManager::validateAgainstEngine called more than once");
    for (auto const& tensor : mWeights)
    {
        validateTensor(executor.getEngine(), tensor);
    }
    mValidated = true;
    if (!mWeights.empty())
    {
        LOG_INFO("Validated %d external weight tensor(s) for %.*s", static_cast<int32_t>(mWeights.size()),
            static_cast<int32_t>(engineLabel.size()), engineLabel.data());
    }
}

void ExternalWeightManager::bindToContext(
    nvinfer1::ICudaEngine const& engine, nvinfer1::IExecutionContext& context, std::string_view engineLabel)
{
    ELLM_CHECK(mLoaded, "ExternalWeightManager::bindToContext called before load");
    ELLM_CHECK(!mValidated, "ExternalWeightManager::bindToContext called more than once");
    for (auto& tensor : mWeights)
    {
        validateTensor(engine, tensor);
        ELLM_CHECK(context.setInputTensorAddress(tensor.getName().c_str(), tensor.rawPointer()),
            "Failed to bind external weight " + tensor.getName());
    }
    mValidated = true;
    mRegistered = true;
    if (!mWeights.empty())
    {
        LOG_INFO("Bound %d external weight tensor(s) to %.*s", static_cast<int32_t>(mWeights.size()),
            static_cast<int32_t>(engineLabel.size()), engineLabel.data());
    }
}

void ExternalWeightManager::registerTensorMapEntries(TensorMap& map)
{
    ELLM_CHECK(mValidated, "registerTensorMapEntries called before weight validation");
    ELLM_CHECK(!mRegistered, "registerTensorMapEntries called more than once");
    for (auto& tensor : mWeights)
    {
        map.set(tensor.getName(), tensor);
    }
    mRegistered = true;
}

} // namespace rt
} // namespace trt_edgellm
