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

// Build a Cosmos3 experimental component engine from its ONNX contract.

#include "builder/cosmos3Builder.h"
#include "common/logger.h"

#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{
enum OptionId : int
{
    HELP = 801,
    ONNX_DIR = 802,
    ENGINE_DIR = 803,
    COMPONENT = 804,
    MAX_BATCH_SIZE = 805,
    DEBUG = 806
};

struct Args
{
    std::string onnxDir;
    std::string engineDir;
    std::string component{"all"};
    int32_t maxBatchSize{1};
    bool help{false};
    bool debug{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnx-dir <dir> --engine-dir <dir> [--component <all|und_prefill|gen|vae_encoder>] "
                 "[--max-batch-size <int>] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help            Display this help message" << std::endl;
    std::cerr << "  --onnx-dir        Exporter output root (holds und_prefill/, gen/, vae_encoder/, "
                 "text_tokenizer/). Required."
              << std::endl;
    std::cerr << "  --engine-dir      Output TensorRT engine root; each component lands in <engineDir>/<component>, "
                 "and the tokenizer + token-embedding table are staged alongside. Required."
              << std::endl;
    std::cerr << "  --component       Component to build: all (default), und_prefill, gen, or vae_encoder."
              << std::endl;
    std::cerr << "  --max-batch-size  Maximum request batch the engine serves (widens the profile batch "
                 "axis). Default = 1"
              << std::endl;
    std::cerr << "  --debug           Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "Legacy camelCase option spellings remain accepted.\n";
}

bool isSupportedComponent(std::string const& component)
{
    return component == "all" || component == "und_prefill" || component == "gen" || component == "vae_encoder";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[]
        = {{"help", no_argument, nullptr, HELP}, {"onnx-dir", required_argument, nullptr, ONNX_DIR},
            {"onnxDir", required_argument, nullptr, ONNX_DIR}, {"engine-dir", required_argument, nullptr, ENGINE_DIR},
            {"engineDir", required_argument, nullptr, ENGINE_DIR}, {"component", required_argument, nullptr, COMPONENT},
            {"max-batch-size", required_argument, nullptr, MAX_BATCH_SIZE},
            {"maxBatchSize", required_argument, nullptr, MAX_BATCH_SIZE}, {"debug", no_argument, nullptr, DEBUG},
            {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: args.help = true; return true;
        case ONNX_DIR: args.onnxDir = optarg ? optarg : ""; break;
        case ENGINE_DIR: args.engineDir = optarg ? optarg : ""; break;
        case COMPONENT: args.component = optarg ? optarg : ""; break;
        case MAX_BATCH_SIZE: args.maxBatchSize = optarg ? std::stoi(optarg) : 1; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }
    return !args.onnxDir.empty() && !args.engineDir.empty() && isSupportedComponent(args.component)
        && args.maxBatchSize >= 1;
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kINFO);

    std::vector<std::string> const components
        = args.component == "all" ? cosmos3::policyComponents() : std::vector<std::string>{args.component};
    if (!cosmos3::buildCosmos3Policy(args.onnxDir, args.engineDir, components, args.maxBatchSize))
    {
        return EXIT_FAILURE;
    }
    LOG_INFO("Cosmos3 policy build complete: %s", args.engineDir.c_str());
    return EXIT_SUCCESS;
}
