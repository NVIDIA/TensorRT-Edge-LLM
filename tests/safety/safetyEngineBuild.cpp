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

#include "safetyEngineTestUtils.h"

#include <NvInfer.h>
#include <array>
#include <fstream>
#include <memory>
#include <string>

namespace
{
using namespace nvinfer1;
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::engine_test;

class Logger final : public ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::fprintf(stderr, "TensorRT: %s\n", message);
        }
    }
};
Logger sLogger;

bool writeArtifact(char const* path, IHostMemory const* artifact)
{
    if (artifact == nullptr || artifact->size() == 0)
    {
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<char const*>(artifact->data()), static_cast<std::streamsize>(artifact->size()));
    output.close();
    return static_cast<bool>(output);
}

bool build(char const* outputPath, char const* remoteConfig)
{
    auto& registryRecorder = processRegistrationRecorder();
    auto* registry = nvinfer2::safe::getSafePluginRegistry(registryRecorder);
    if (registry == nullptr || !check(registerSafetyPlugins(*registry), "registerSafetyPlugins"))
    {
        return false;
    }
    std::unique_ptr<IBuilder> builder(createInferBuilder(sLogger));
    if (!builder)
    {
        return false;
    }
    uint32_t const flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(flags));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config)
    {
        return false;
    }
    std::array<ITensor*, 5> inputs{network->addInput("q", DataType::kHALF, Dims3{kTOKENS, kHEADS, kHEAD_SIZE}),
        network->addInput("k", DataType::kHALF, Dims3{kTOKENS, kHEADS, kHEAD_SIZE}),
        network->addInput("v", DataType::kHALF, Dims3{kTOKENS, kHEADS, kHEAD_SIZE}),
        network->addInput("sequence_ends", DataType::kINT32, Dims{1, {3}}),
        network->addInput("shape_carrier", DataType::kINT32, Dims{1, {3}})};
    for (auto* input : inputs)
    {
        if (input == nullptr)
        {
            return false;
        }
    }
    int32_t heads{kHEADS}, headSize{kHEAD_SIZE};
    float scale{kSCALE};
    PluginField const fields[]{{"num_heads", &heads, PluginFieldType::kINT32, 1},
        {"head_size", &headSize, PluginFieldType::kINT32, 1},
        {"attention_scale", &scale, PluginFieldType::kFLOAT32, 1}};
    PluginFieldCollection const collection{3, fields};
    auto& creator = safetyAttentionCreator(AttentionRole::kVISUAL);
    std::unique_ptr<IPluginV3> plugin(creator.createPlugin("safety_visual_smoke", &collection, TensorRTPhase::kBUILD));
    if (!plugin || plugin->getCapabilityInterface(PluginCapabilityType::kBUILD) == nullptr)
    {
        return false;
    }
    auto* layer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr)
    {
        return false;
    }
    layer->setName("safety_visual_smoke");
    layer->getOutput(0)->setName("output");
    network->markOutput(*layer->getOutput(0));
    config->setEngineCapability(EngineCapability::kSAFETY);
    config->setMaxAuxStreams(0);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 64ULL << 20);
    if (remoteConfig != nullptr)
    {
        if (!config->setRemoteAutoTuningConfig(remoteConfig))
        {
            return false;
        }
    }
    auto* profile = builder->createOptimizationProfile();
    if (profile == nullptr)
    {
        return false;
    }
    for (auto* input : inputs)
    {
        for (auto selector : {OptProfileSelector::kMIN, OptProfileSelector::kOPT, OptProfileSelector::kMAX})
        {
            if (!profile->setDimensions(input->getName(), selector, input->getDimensions()))
            {
                return false;
            }
        }
    }
    if (config->addOptimizationProfile(profile) < 0)
    {
        return false;
    }
    std::unique_ptr<ISafeSerializedNetworkArtifacts> artifacts(
        builder->buildSerializedSafeNetwork(*network, *config, false));
    if (!artifacts || registryRecorder.getNbErrors() != 0)
    {
        return false;
    }
    std::string const companionPath = std::string(outputPath) + ".so";
    if (!writeArtifact(companionPath.c_str(), artifacts->getCompanionSo()))
    {
        return false;
    }
    auto const* engine = artifacts->getSerializedNetwork();
    if (!engine || engine->size() == 0 || registryRecorder.getNbErrors() != 0)
    {
        return false;
    }
    if (!writeArtifact(outputPath, engine))
    {
        return false;
    }
    std::printf("Serialized real ViT Safety engine: %s (%zu bytes), TensorRT %d.%d.%d\n", outputPath, engine->size(),
        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::fprintf(stderr,
            "Usage: %s engine.plan [remote-auto-tuning-config]\n"
            "Builder, plugin artifacts and target runtime must use a matching SDK; this is not a VLM export test.\n",
            argv[0]);
        return 2;
    }
    {
        std::ofstream reset(argv[1], std::ios::binary | std::ios::trunc);
        reset.close();
        if (!reset)
        {
            std::fprintf(stderr, "Cannot initialize engine output artifact: %s\n", argv[1]);
            return 1;
        }
    }
    bool const remote = argc == 3;
    CUcontext context{};
    SafetyAttentionModules modules;
    SafetyAttentionModuleScope moduleScope(modules);
    if (!remote && !initializeModules(context, modules))
    {
        return 1;
    }
    std::fprintf(stderr, "%s\n",
        remote ? "Remote tuning: no local CUDA setup; the matching remote plugin endpoint must be provided by the "
                 "Safety builder workflow."
               : "Local GPU tuning: engine, plugin and target runtime must match.");
    try
    {
        if (!build(argv[1], argc == 3 ? argv[2] : nullptr))
        {
            std::fprintf(stderr, "Safety engine build failed; no runtime validation claimed.\n");
            return 1;
        }
    }
    catch (std::exception const& error)
    {
        std::fprintf(stderr, "Safety engine build failed: %s\n", error.what());
        return 1;
    }
    if (remote)
    {
        return 0;
    }
    return check(modules.release(), "SafetyAttentionModules::release") && check(cuCtxDestroy(context), "cuCtxDestroy")
        ? 0
        : 1;
}
