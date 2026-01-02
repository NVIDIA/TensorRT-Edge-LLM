/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "builder/builder.h"
#include "common/fileUtils.h"
#include "common/logger.h"

#include <cstdlib>
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
    MAX_IMAGE_TOKENS_PER_IMAGE = 607
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
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--onnxDir str> <--engineDir str> [--debug]"
                 "[--minImageTokens int] [--maxImageTokens int] [--maxImageTokensPerImage int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help               Display this help message" << std::endl;
    std::cerr
        << "  --onnxDir            Provide the directory containing the input onnx file for visual encoder. Required. "
        << std::endl;
    std::cerr
        << "  --engineDir          Provide the output TensorRT engine directory path for visual encoder. Required. "
        << std::endl;
    std::cerr << "  --debug              Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --minImageTokens     Minimum image tokens. Default = 4" << std::endl;
    std::cerr << "  --maxImageTokens     Maximum image tokens. Default = 1024" << std::endl;
    std::cerr << "  --maxImageTokensPerImage     Maximum image tokens per image. Default = 512" << std::endl;
}

bool parseViTBuildArgs(ViTBuildArgs& args, int argc, char* argv[])
{
    static struct option vitOptions[] = {{"help", no_argument, 0, VLMBuildOptionId::HELP},
        {"onnxDir", required_argument, 0, VLMBuildOptionId::ONNX_DIR},
        {"engineDir", required_argument, 0, VLMBuildOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, VLMBuildOptionId::DEBUG},
        {"minImageTokens", required_argument, 0, VLMBuildOptionId::MIN_IMAGE_TOKENS},
        {"maxImageTokens", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS},
        {"maxImageTokensPerImage", required_argument, 0, VLMBuildOptionId::MAX_IMAGE_TOKENS_PER_IMAGE}, {0, 0, 0, 0}};

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
                LOG_ERROR("--onnxDir requires option argument.");
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
                LOG_ERROR("--engineDir requires option argument.");
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

    // Create and run the builder
    builder::VisualBuilder visualBuilder(args.onnxDir, args.engineDir, config);
    if (!visualBuilder.build())
    {
        LOG_ERROR("Failed to build Visual engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("Visual engine built successfully.");
    return EXIT_SUCCESS;
}
