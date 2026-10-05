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

// Build a pi0.5 experimental component engine from its ONNX contract.

#include "builder/pi05Builder.h"
#include "common/logger.h"

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{
enum OptionId : int
{
    HELP = 901,
    ONNX_DIR = 902,
    ENGINE_DIR = 903,
    COMPONENT = 904,
    MAX_BATCH_SIZE = 905,
    DEBUG = 906
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
              << " [--help] --onnxDir <dir> --engineDir <dir> [--component <all|visual|prefix|action|cond>] "
                 "[--maxBatchSize <int>] [--debug]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help          Display this help message" << std::endl;
    std::cerr << "  --onnxDir       Exporter output root (holds visual/, prefix/, action/, assets/). Required."
              << std::endl;
    std::cerr << "  --engineDir     Output TensorRT engine root; each component lands in <engineDir>/<component>, "
                 "and the token-embedding table, tokenizer and normalization assets are staged alongside. Required."
              << std::endl;
    std::cerr << "  --component     Component to build: all (default), visual, prefix, action, or cond "
                 "(hoisted-AdaRMS-modulation exports only)."
              << std::endl;
    std::cerr << "  --maxBatchSize  Maximum request batch the engine serves (widens the profile batch "
                 "axis). Default = 1"
              << std::endl;
    std::cerr << "  --debug         Use debug mode, which outputs more logs." << std::endl;
}

bool isSupportedComponent(std::string const& component)
{
    return component == "all" || component == "visual" || component == "prefix" || component == "action"
        || component == "cond";
}

//! A numeric option, or false. Reported through the usage path rather than thrown:
//! argument parsing runs before the try block below.
bool parseInt(char const* text, int32_t& out)
{
    if (text == nullptr)
    {
        return false;
    }
    char* end{nullptr};
    errno = 0;
    long const parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < std::numeric_limits<int32_t>::min()
        || parsed > std::numeric_limits<int32_t>::max())
    {
        std::cerr << "Not an integer: " << text << std::endl;
        return false;
    }
    out = static_cast<int32_t>(parsed);
    return true;
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[]
        = {{"help", no_argument, nullptr, HELP}, {"onnxDir", required_argument, nullptr, ONNX_DIR},
            {"engineDir", required_argument, nullptr, ENGINE_DIR}, {"component", required_argument, nullptr, COMPONENT},
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
        case MAX_BATCH_SIZE:
            if (!parseInt(optarg, args.maxBatchSize))
            {
                return false;
            }
            break;
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

    // A manifest that parses but carries a wrong field type reaches the JSON accessors
    // as an exception, not a parse error, so it needs a boundary here to be reported.
    try
    {
        std::vector<std::string> const components
            = args.component == "all" ? pi05::policyComponents(args.onnxDir) : std::vector<std::string>{args.component};
        if (!pi05::buildPi05Policy(args.onnxDir, args.engineDir, components, args.maxBatchSize))
        {
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("pi0.5 policy build failed: %s", error.what());
        return EXIT_FAILURE;
    }
    LOG_INFO("pi0.5 policy build complete: %s", args.engineDir.c_str());
    return EXIT_SUCCESS;
}
