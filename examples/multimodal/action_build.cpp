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

#include "builder/actionBuilder.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

enum ActionBuildOptionId : int
{
    HELP = 701,
    ONNX_DIR = 702,
    ENGINE_DIR = 703,
    DEBUG = 704,
    MAX_BATCH_SIZE = 705
};

struct ActionBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};
    int32_t maxBatchSize{4};
};

static void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--onnx-dir str> <--engine-dir str> [--debug] [--max-batch-size int] " << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help            Display this help message" << std::endl;
    std::cerr << "  --onnx-dir        Directory containing the action expert ONNX (model.onnx and config.json). "
                 "Required."
              << std::endl;
    std::cerr << "  --engine-dir      Output directory. Engine will be saved to <engineDir>/action/action.engine"
              << std::endl;
    std::cerr << "  --debug           Enable verbose logging" << std::endl;
    std::cerr << "  --max-batch-size  Maximum batch size for optimization profiles (default: 4)" << std::endl;
    std::cerr << "Legacy camelCase option spellings remain accepted.\n";
}

static bool parseActionBuildArgs(ActionBuildArgs& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, 0, ActionBuildOptionId::HELP},
        {"onnx-dir", required_argument, 0, ActionBuildOptionId::ONNX_DIR},
        {"onnxDir", required_argument, 0, ActionBuildOptionId::ONNX_DIR},
        {"engine-dir", required_argument, 0, ActionBuildOptionId::ENGINE_DIR},
        {"engineDir", required_argument, 0, ActionBuildOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, ActionBuildOptionId::DEBUG},
        {"max-batch-size", required_argument, 0, ActionBuildOptionId::MAX_BATCH_SIZE},
        {"maxBatchSize", required_argument, 0, ActionBuildOptionId::MAX_BATCH_SIZE}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case ActionBuildOptionId::HELP: args.help = true; return true;
        case ActionBuildOptionId::ONNX_DIR:
            if (optarg)
                args.onnxDir = optarg;
            else
            {
                LOG_ERROR("--onnx-dir requires an argument.");
                return false;
            }
            break;
        case ActionBuildOptionId::ENGINE_DIR:
            if (optarg)
                args.engineDir = optarg;
            else
            {
                LOG_ERROR("--engine-dir requires an argument.");
                return false;
            }
            break;
        case ActionBuildOptionId::DEBUG: args.debug = true; break;
        case ActionBuildOptionId::MAX_BATCH_SIZE:
            if (optarg)
                args.maxBatchSize = std::stoi(optarg);
            break;
        default: LOG_ERROR("Invalid argument"); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    ActionBuildArgs args;
    if ((argc < 2) || (!parseActionBuildArgs(args, argc, argv)))
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

    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    builder::ActionBuilderConfig config;
    config.maxBatchSize = args.maxBatchSize;

    std::string actualEngineDir = args.engineDir + "/action";

    builder::ActionBuilder actionBuilder(args.onnxDir, actualEngineDir, config);
    if (!actionBuilder.build())
    {
        LOG_ERROR("Failed to build Action engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("Action engine built successfully.");
    return EXIT_SUCCESS;
}
