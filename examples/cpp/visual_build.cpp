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

#include <tensorrt_edgellm/builder/visualBuilder.h>
#include <tensorrt_edgellm/common/fileUtils.h>
#include <tensorrt_edgellm/common/logger.h>
#ifdef ENABLE_SAFETY_RUNTIME
#include <tensorrt_edgellm/builder/safety/remoteAutoTuningUtils.h>
#include <tensorrt_edgellm/builder/safety/safetyVisualBuilder.h>
#endif

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum VLMBuildOptionId : int
{
    HELP = 601,
    ONNX_DIR = 602,
    ENGINE_DIR = 603,
    DEBUG = 604,
    MIN_IMAGE_TOKENS = 605,
    MAX_IMAGE_TOKENS = 606,
    MAX_IMAGE_TOKENS_PER_IMAGE = 607,
    PROFILING_DETAILED = 608,
    SAFETY = 609,
    MSS_PROFILE = 610,
    REMOTE_AUTO_TUNING_CONFIG = 611,
    SAFETY_PLUGIN_LIBRARY = 612
};

struct CommonViTBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};
    int64_t maxImageTokensPerImage{512};
    bool profilingDetailed{false}; // Enable detailed profiling verbosity for layer info extraction
};

struct ViTBuildArgs : CommonViTBuildArgs
{
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
};

#ifdef ENABLE_SAFETY_RUNTIME
struct SafetyViTBuildArgs : CommonViTBuildArgs
{
    std::vector<safety::builder::SafetyVisualStaticShapeProfile> safetyProfiles;
    std::string remoteAutoTuningConfig;
    std::string safetyPluginLibrary;
};
#endif

#ifdef ENABLE_SAFETY_RUNTIME
bool parseSafetyVisualProfile(char const* value, safety::builder::SafetyVisualStaticShapeProfile& profile)
{
    try
    {
        size_t parsedCharacters{0};
        std::string const input{value != nullptr ? value : ""};
        profile.imageTokens = std::stoll(input, &parsedCharacters);
        return parsedCharacters == input.size() && profile.imageTokens > 0;
    }
    catch (std::exception const&)
    {
        return false;
    }
}
#endif

void printUsage(char const* programName)
{
    std::cerr
        << "Usage: " << programName
        << " [--help] <--onnx-dir str> <--engine-dir str> [--debug]"
           "[--min-image-tokens int] [--max-image-tokens int] [--max-image-tokens-per-image int] [--profiling-detailed]"
        << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr
        << "  --onnx-dir            Provide the directory containing the input onnx file for visual encoder. Required. "
        << std::endl;
    std::cerr << "  --engine-dir          Base output directory for multimodal engines. Required." << std::endl;
    std::cerr << "                       Note: Visual engine will be saved to <engineDir>/visual/" << std::endl;
    std::cerr << "  --debug              Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --min-image-tokens     Minimum image tokens. Default = 4" << std::endl;
    std::cerr << "  --max-image-tokens     Maximum image tokens. Default = 1024" << std::endl;
    std::cerr << "  --max-image-tokens-per-image     Maximum image tokens per image. Default = 512" << std::endl;
    std::cerr << "  --profiling-detailed  Enable detailed profiling verbosity to include ONNX op names. "
                 "Use for DLSim analysis."
              << std::endl;
#ifdef ENABLE_SAFETY_RUNTIME
    std::cerr << "Safety Builder options:" << std::endl
              << "  --safety                    Build a visual engine with TensorRT kSAFETY capability." << std::endl
              << "  --mss-profile <imageTokens>  Add a fixed MSS image-token profile." << std::endl
              << "  --remote-auto-tuning-config    TensorRT remote auto-tuning URI. May instead be supplied through "
                 "EDGELLM_REMOTE_AUTOTUNING_CONFIG. Omit for local GPU auto-tuning."
              << std::endl
              << "  --safety-plugin-library       Optional Safety ViT plugin library, or EDGELLM_SAFETY_PLUGIN_PATH. "
                 "Not needed for native-operator ONNX exports."
              << std::endl;
#endif
}

bool parseViTBuildArgs(CommonViTBuildArgs& commonArgs, ViTBuildArgs& args,
#ifdef ENABLE_SAFETY_RUNTIME
    SafetyViTBuildArgs& safetyArgs, bool& useSafety,
#endif
    int argc, char* argv[])
{
    static struct option vitOptions[] = {{"help", no_argument, 0, VLMBuildOptionId::HELP},
        {"onnx-dir", required_argument, 0, VLMBuildOptionId::ONNX_DIR},
        {"onnxDir", required_argument, 0, VLMBuildOptionId::ONNX_DIR},
        {"engine-dir", required_argument, 0, VLMBuildOptionId::ENGINE_DIR},
        {"engineDir", required_argument, 0, VLMBuildOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, VLMBuildOptionId::DEBUG},
        {"min-image-tokens", required_argument, 0, VLMBuildOptionId::MIN_IMAGE_TOKENS},
        {"minImageTokens", required_argument, 0, VLMBuildOptionId::MIN_IMAGE_TOKENS},
        {"max-image-tokens", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS},
        {"maxImageTokens", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS},
        {"max-image-tokens-per-image", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS_PER_IMAGE},
        {"maxImageTokensPerImage", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS_PER_IMAGE},
        {"profiling-detailed", no_argument, 0, VLMBuildOptionId::PROFILING_DETAILED},
        {"profilingDetailed", no_argument, 0, VLMBuildOptionId::PROFILING_DETAILED},
        {"safety", no_argument, 0, VLMBuildOptionId::SAFETY},
        {"mss-profile", required_argument, 0, VLMBuildOptionId::MSS_PROFILE},
        {"mssProfile", required_argument, 0, VLMBuildOptionId::MSS_PROFILE},
        {"remote-auto-tuning-config", required_argument, 0, VLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG},
        {"remoteAutoTuningConfig", required_argument, 0, VLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG},
        {"safety-plugin-library", required_argument, 0, VLMBuildOptionId::SAFETY_PLUGIN_LIBRARY},
        {"safetyPluginLibrary", required_argument, 0, VLMBuildOptionId::SAFETY_PLUGIN_LIBRARY}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", vitOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case VLMBuildOptionId::HELP: commonArgs.help = true; return true;
        case VLMBuildOptionId::ONNX_DIR:
            if (optarg)
            {
                commonArgs.onnxDir = optarg;
            }
            else
            {
                LOG_ERROR("--onnx-dir requires option argument.");
                return false;
            }
            break;
        case VLMBuildOptionId::ENGINE_DIR:
            if (optarg)
            {
                commonArgs.engineDir = optarg;
            }
            else
            {
                LOG_ERROR("--engine-dir requires option argument.");
                return false;
            }
            break;
        case VLMBuildOptionId::DEBUG: commonArgs.debug = true; break;
        case VLMBuildOptionId::MIN_IMAGE_TOKENS:
            if (optarg)
            {
                args.minImageTokens = std::stoi(optarg);
            }
            break;
        case VLMBuildOptionId::MAX_IMAGE_TOKENS:
            if (optarg)
            {
                args.maxImageTokens = std::stoi(optarg);
            }
            break;
        case VLMBuildOptionId::MAX_IMAGE_TOKENS_PER_IMAGE:
            if (optarg)
            {
                commonArgs.maxImageTokensPerImage = std::stoi(optarg);
            }
            break;
        case VLMBuildOptionId::PROFILING_DETAILED: commonArgs.profilingDetailed = true; break;
#ifdef ENABLE_SAFETY_RUNTIME
        case VLMBuildOptionId::SAFETY: useSafety = true; break;
        case VLMBuildOptionId::MSS_PROFILE:
        {
            safety::builder::SafetyVisualStaticShapeProfile profile;
            if (!parseSafetyVisualProfile(optarg, profile))
            {
                LOG_ERROR("--mss-profile expects a positive image-token count.");
                return false;
            }
            safetyArgs.safetyProfiles.push_back(profile);
            break;
        }
        case VLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG:
            if (optarg == nullptr)
            {
                return false;
            }
            safetyArgs.remoteAutoTuningConfig = optarg;
            break;
        case VLMBuildOptionId::SAFETY_PLUGIN_LIBRARY:
            if (optarg == nullptr)
            {
                return false;
            }
            safetyArgs.safetyPluginLibrary = optarg;
            break;
#else
        case VLMBuildOptionId::SAFETY:
        case VLMBuildOptionId::MSS_PROFILE:
        case VLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG:
        case VLMBuildOptionId::SAFETY_PLUGIN_LIBRARY:
            LOG_ERROR("Safety Builder is not enabled; configure with -DENABLE_SAFETY_RUNTIME=ON.");
            return false;
#endif
        default: LOG_ERROR("ERROR: Invalid Argument %c is %s", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    CommonViTBuildArgs commonArgs;
    ViTBuildArgs args;
#ifdef ENABLE_SAFETY_RUNTIME
    SafetyViTBuildArgs safetyArgs;
    bool useSafety{false};
#endif
    if ((argc < 2)
        || (!parseViTBuildArgs(commonArgs, args,
#ifdef ENABLE_SAFETY_RUNTIME
            safetyArgs, useSafety,
#endif
            argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    static_cast<CommonViTBuildArgs&>(args) = commonArgs;
#ifdef ENABLE_SAFETY_RUNTIME
    static_cast<CommonViTBuildArgs&>(safetyArgs) = commonArgs;
#endif
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(trt_edgellm::logger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(trt_edgellm::logger::Severity::kINFO);
    }

    // Validate input directory and required files
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

#ifdef ENABLE_SAFETY_RUNTIME
    if (!useSafety
        && (!safetyArgs.safetyProfiles.empty() || !safetyArgs.remoteAutoTuningConfig.empty()
            || !safetyArgs.safetyPluginLibrary.empty()))
    {
        LOG_ERROR("--mss-profile, --remote-auto-tuning-config, and --safety-plugin-library require --safety.");
        return EXIT_FAILURE;
    }
    if (useSafety)
    {
        if (safetyArgs.safetyProfiles.empty())
        {
            LOG_ERROR("--safety requires at least one --mss-profile.");
            return EXIT_FAILURE;
        }
        if (safetyArgs.remoteAutoTuningConfig.empty())
        {
            char const* const config = std::getenv("EDGELLM_REMOTE_AUTOTUNING_CONFIG");
            if (config != nullptr)
            {
                safetyArgs.remoteAutoTuningConfig = config;
            }
        }
        if (!safety::builder::validateSafetyAutoTuningConfig(safetyArgs.remoteAutoTuningConfig))
        {
            return EXIT_FAILURE;
        }
    }
#endif

    // The server loads <engineDir>/visual.engine, so land the engine under a
    // single "visual" segment whether the caller passes the base dir or the
    // visual subdir itself (avoids a visual/visual double nest).
    std::filesystem::path enginePath(args.engineDir);
    if (enginePath.filename().empty())
    {
        enginePath = enginePath.parent_path();
    }
    if (enginePath.filename() != "visual")
    {
        enginePath /= "visual";
    }
    std::string actualEngineDir = enginePath.string();

#ifdef ENABLE_SAFETY_RUNTIME
    if (useSafety)
    {
        safety::builder::SafetyVisualBuilderConfig safetyConfig;
        safetyConfig.profiles = safetyArgs.safetyProfiles;
        safetyConfig.remoteAutoTuningConfig = safetyArgs.remoteAutoTuningConfig;
        safetyConfig.safetyPluginLibrary = safetyArgs.safetyPluginLibrary;
        safetyConfig.maxImageTokensPerImage = safetyArgs.maxImageTokensPerImage;
        safetyConfig.profilingDetailed = safetyArgs.profilingDetailed;

        safety::builder::SafetyVisualBuilder safetyBuilder(
            safetyArgs.onnxDir, actualEngineDir, std::move(safetyConfig));
        if (!safetyBuilder.build())
        {
            LOG_ERROR("Failed to build the Safety visual engine.");
            return EXIT_FAILURE;
        }
        LOG_INFO("Safety visual engine built successfully.");
        return EXIT_SUCCESS;
    }
#endif

    // Create VisualBuilderConfig from args
    if (args.maxImageTokensPerImage < args.minImageTokens || args.maxImageTokensPerImage > args.maxImageTokens)
    {
        LOG_ERROR(
            "maxImageTokensPerImage must be greater than or equal to minImageTokens and less than or equal to "
            "maxImageTokens."
            "minImageTokens: %d, maxImageTokens: %d, maxImageTokensPerImage: %d",
            args.minImageTokens, args.maxImageTokens, args.maxImageTokensPerImage);
        return EXIT_FAILURE;
    }
    builder::VisualBuilderConfig config;
    config.minImageTokens = args.minImageTokens;
    config.maxImageTokens = args.maxImageTokens;
    config.maxImageTokensPerImage = args.maxImageTokensPerImage;
    config.profilingDetailed = args.profilingDetailed;

    // Create and run the builder
    builder::VisualBuilder visualBuilder(args.onnxDir, actualEngineDir, config);
    if (!visualBuilder.build())
    {
        LOG_ERROR("Failed to build Visual engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("Visual engine built successfully.");
    return EXIT_SUCCESS;
}
