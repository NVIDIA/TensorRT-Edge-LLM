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

#include "builder/visualBuilder.h"
#include "common/fileUtils.h"
#include "common/logger.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

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
    PROFILING_DETAILED = 608
};

struct ViTBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
    int64_t maxImageTokensPerImage{512};
    bool profilingDetailed{false}; // Enable detailed profiling verbosity for layer info extraction
};

void printUsage(char const* programName)
{
    std::cerr
        << "Usage: " << programName
        << " [--help] <--onnx-dir str> <--engine-dir str> [--debug]"
           "[--min-image-tokens int] [--max-image-tokens int] [--max-image-tokens-per-image int] [--profiling-detailed]"
        << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                        Display this help message" << std::endl;
    std::cerr << "  --onnx-dir                    Provide the directory containing the input onnx file for visual "
                 "encoder. Required. "
              << std::endl;
    std::cerr << "  --engine-dir                  Base output directory for multimodal engines. Required." << std::endl;
    std::cerr << "                                Note: Visual engine will be saved to <engineDir>/visual/"
              << std::endl;
    std::cerr << "  --debug                       Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --min-image-tokens            Minimum image tokens. Default = 4" << std::endl;
    std::cerr << "  --max-image-tokens            Maximum image tokens. Default = 1024" << std::endl;
    std::cerr << "  --max-image-tokens-per-image  Maximum image tokens per image. Default = 512" << std::endl;
    std::cerr << "  --profiling-detailed          Enable detailed profiling verbosity to include ONNX op names. "
                 "Use for DLSim analysis."
              << std::endl;
    std::cerr << "Legacy camelCase option spellings remain accepted.\n";
}

bool parseViTBuildArgs(ViTBuildArgs& args, int argc, char* argv[])
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
        {"profilingDetailed", no_argument, 0, VLMBuildOptionId::PROFILING_DETAILED}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", vitOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case VLMBuildOptionId::HELP: args.help = true; return true;
        case VLMBuildOptionId::ONNX_DIR:
            if (optarg)
            {
                args.onnxDir = optarg;
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
                args.engineDir = optarg;
            }
            else
            {
                LOG_ERROR("--engine-dir requires option argument.");
                return false;
            }
            break;
        case VLMBuildOptionId::DEBUG: args.debug = true; break;
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
                args.maxImageTokensPerImage = std::stoi(optarg);
            }
            break;
        case VLMBuildOptionId::PROFILING_DETAILED: args.profilingDetailed = true; break;
        default: LOG_ERROR("ERROR: Invalid Argument %c is %s", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    ViTBuildArgs args;
    if ((argc < 2) || (!parseViTBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
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
