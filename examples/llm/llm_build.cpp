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

#include "builder/llmBuilder.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/parallelArtifactNames.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMBuildOptionId : int
{
    HELP = 701,
    ONNX_DIR = 702,
    ENGINE_DIR = 703,
    MAX_INPUT_LEN = 704,
    MAX_KV_CACHE_CAPACITY = 705,
    DEBUG = 706,
    MAX_BATCH_SIZE = 707,
    MAX_LORA_RANK = 708,
    SPEC_DRAFT = 709,
    SPEC_BASE = 710,
    MAX_VERIFY_TREE_SIZE = 711,
    MAX_DRAFT_TREE_SIZE = 712,
    PROFILING_DETAILED = 713,
    MAX_KV_POOL_PAGES = 714,
    TP_SIZE = 715,
    TP_RANK = 716,
    LOCAL_DEVICE = 717
};

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    int64_t maxInputLen{1024};
    int64_t maxKVCacheCapacity{4096};
    int64_t maxKVPoolPages{0};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // Default to 0 means no LoRA
    bool specDraft{false};
    bool specBase{false};
    int64_t maxVerifyTreeSize{60};
    int64_t maxDraftTreeSize{60};
    bool profilingDetailed{false}; // Enable detailed profiling verbosity for layer info extraction
    int tpSize{1};
    int tpRank{0};
    int localDevice{0};
    bool tpSizeSet{false};
    bool tpRankSet{false};
    bool localDeviceSet{false};
};

namespace
{

std::string getInputConfigFileName(int tpSize, int tpRank)
{
    parallel_artifacts::RankArtifactContext const context{tpSize, tpRank};
    return parallel_artifacts::configFileName(context);
}

} // namespace

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnxDir <dir> --engineDir <dir> [--maxInputLen <int>] "
                 "[--maxKVCacheCapacity <int>] [--maxBatchSize <int>] [--debug] [--maxLoraRank <int>]"
                 " [--maxKVPoolPages <int>]"
                 " [--specDraft] [--specBase] [--maxVerifyTreeSize <int>] "
                 "[--maxDraftTreeSize <int>] [--profilingDetailed] [--tpSize <int> --tpRank <int>] "
                 "[--localDevice <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --onnxDir                 Provide the input ONNX directory path. Required. " << std::endl;
    std::cerr << "  --engineDir               Provide the output TensorRT engine directory path. Required. "
              << std::endl;
    std::cerr << "  --maxInputLen             Provide the maximum input length for the model. Default = 1024"
              << std::endl;
    std::cerr << "  --maxKVCacheCapacity      Provide the maximum KV cache capacity (sequence length). "
                 "Default = 4096"
              << std::endl;
    std::cerr << "  --maxKVPoolPages          Exact physical K-page count for the KV pool. Default = 0 "
                 "(minimum active pages)"
              << std::endl;
    std::cerr << "  --maxBatchSize            Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --debug                   Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank             Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    std::cerr
        << "  --specDraft               Build as speculative decoding draft model (EAGLE/MTP/DFlash/JetSpec/DSpark)"
        << std::endl;
    std::cerr
        << "  --specBase                Build as speculative decoding base model (EAGLE/MTP/DFlash/JetSpec/DSpark)"
        << std::endl;
    std::cerr << "  --maxVerifyTreeSize       Maximum input_ids tokens for base model verification. Default = 60"
              << std::endl;
    std::cerr << "  --maxDraftTreeSize        Maximum input_ids tokens for draft model generation. Default = 60"
              << std::endl;
    std::cerr << "  --profilingDetailed       Enable detailed profiling verbosity to include ONNX op names "
                 "in layer info. Use for DLSim analysis."
              << std::endl
              << "  --tpSize                  Tensor parallel world size for per-rank engine builds. "
                 "Must be paired with --tpRank."
              << std::endl
              << "  --tpRank                  Tensor parallel rank for per-rank engine builds. Must be in [0, tpSize)."
              << std::endl
              << "  --localDevice             Node-local CUDA device ordinal for an explicit TP build. Defaults to "
                 "tpRank."
              << std::endl
              << std::endl;
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, LLMBuildOptionId::HELP},
        {"onnxDir", required_argument, 0, LLMBuildOptionId::ONNX_DIR},
        {"engineDir", required_argument, 0, LLMBuildOptionId::ENGINE_DIR},
        {"maxInputLen", required_argument, 0, LLMBuildOptionId::MAX_INPUT_LEN},
        {"maxKVCacheCapacity", required_argument, 0, LLMBuildOptionId::MAX_KV_CACHE_CAPACITY},
        {"maxKVPoolPages", required_argument, 0, LLMBuildOptionId::MAX_KV_POOL_PAGES},
        {"debug", no_argument, 0, LLMBuildOptionId::DEBUG},
        {"maxBatchSize", required_argument, 0, LLMBuildOptionId::MAX_BATCH_SIZE},
        {"maxLoraRank", required_argument, 0, LLMBuildOptionId::MAX_LORA_RANK},
        {"specDraft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT},
        {"eagleDraft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT}, // deprecated alias
        {"specBase", no_argument, 0, LLMBuildOptionId::SPEC_BASE},
        {"eagleBase", no_argument, 0, LLMBuildOptionId::SPEC_BASE}, // deprecated alias
        {"maxVerifyTreeSize", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"maxDraftTreeSize", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"profilingDetailed", no_argument, 0, LLMBuildOptionId::PROFILING_DETAILED},
        {"tpSize", required_argument, 0, LLMBuildOptionId::TP_SIZE},
        {"tpRank", required_argument, 0, LLMBuildOptionId::TP_RANK},
        {"localDevice", required_argument, 0, LLMBuildOptionId::LOCAL_DEVICE}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", buildOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMBuildOptionId::HELP: args.help = true; return true;
        case LLMBuildOptionId::ONNX_DIR:
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
        case LLMBuildOptionId::ENGINE_DIR:
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
        case LLMBuildOptionId::MAX_INPUT_LEN:
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_KV_CACHE_CAPACITY:
            if (optarg)
            {
                args.maxKVCacheCapacity = std::stoll(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_KV_POOL_PAGES:
            if (optarg)
            {
                args.maxKVPoolPages = std::stoll(optarg);
            }
            break;
        case LLMBuildOptionId::DEBUG: args.debug = true; break;
        case LLMBuildOptionId::MAX_BATCH_SIZE:
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_LORA_RANK:
            if (optarg)
            {
                args.maxLoraRank = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::SPEC_DRAFT: args.specDraft = true; break;
        case LLMBuildOptionId::SPEC_BASE: args.specBase = true; break;
        case LLMBuildOptionId::MAX_VERIFY_TREE_SIZE:
            if (optarg)
            {
                args.maxVerifyTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_DRAFT_TREE_SIZE:
            if (optarg)
            {
                args.maxDraftTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::PROFILING_DETAILED: args.profilingDetailed = true; break;
        case LLMBuildOptionId::TP_SIZE:
            if (optarg)
            {
                args.tpSize = std::stoi(optarg);
                args.tpSizeSet = true;
            }
            break;
        case LLMBuildOptionId::TP_RANK:
            if (optarg)
            {
                args.tpRank = std::stoi(optarg);
                args.tpRankSet = true;
            }
            break;
        case LLMBuildOptionId::LOCAL_DEVICE:
            if (optarg)
            {
                args.localDevice = std::stoi(optarg);
                args.localDeviceSet = true;
            }
            break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    LLMBuildArgs args;
    if ((argc < 2) || (!parseLLMBuildArgs(args, argc, argv)))
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

    int tpSize = 1;
    int tpRank = 0;
    bool const useExplicitTpConfig = args.tpSizeSet || args.tpRankSet;

    if (args.tpSizeSet != args.tpRankSet)
    {
        LOG_ERROR("--tpSize and --tpRank must be specified together.");
        return EXIT_FAILURE;
    }
    if (useExplicitTpConfig)
    {
        if (args.tpSize < 1)
        {
            LOG_ERROR("--tpSize must be >= 1, got %d.", args.tpSize);
            return EXIT_FAILURE;
        }
        if (args.tpRank < 0 || args.tpRank >= args.tpSize)
        {
            LOG_ERROR("--tpRank must be in [0, tpSize), got tpRank=%d tpSize=%d.", args.tpRank, args.tpSize);
            return EXIT_FAILURE;
        }

        tpSize = args.tpSize;
        tpRank = args.tpRank;
    }

    if (args.localDeviceSet && !useExplicitTpConfig)
    {
        LOG_ERROR("--localDevice requires --tpSize and --tpRank.");
        return EXIT_FAILURE;
    }
    if (args.localDeviceSet && args.localDevice < 0)
    {
        LOG_ERROR("--localDevice must be >= 0, got %d.", args.localDevice);
        return EXIT_FAILURE;
    }

    if (tpSize > 1 && (args.specDraft || args.specBase))
    {
        LOG_ERROR("Tensor-parallel speculative decoding engine builds are not supported; build with --tpSize 1.");
        return EXIT_FAILURE;
    }

    if (tpSize > 1 || args.localDeviceSet)
    {
        int const localDevice = args.localDeviceSet ? args.localDevice : tpRank;
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (localDevice >= deviceCount)
        {
            LOG_ERROR("[TP rank %d/%d] CUDA device %d requested, but this host exposes only %d device(s).", tpRank,
                tpSize, localDevice, deviceCount);
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaSetDevice(localDevice));
        LOG_INFO("[TP rank %d/%d] Using node-local CUDA device %d.", tpRank, tpSize, localDevice);
    }

    // Validate input directory and required files
    std::string const configFileName = getInputConfigFileName(tpSize, tpRank);
    std::string const configPath = args.onnxDir + "/" + configFileName;
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("%s not found in onnx directory: %s", configFileName.c_str(), args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    // Create LLMBuilderConfig from args
    builder::LLMBuilderConfig config;
    config.maxInputLen = args.maxInputLen;
    config.maxKVCacheCapacity = args.maxKVCacheCapacity;
    config.maxKVPoolPages = args.maxKVPoolPages;
    config.maxBatchSize = args.maxBatchSize;
    config.maxLoraRank = args.maxLoraRank;
    config.specDraft = args.specDraft;
    config.specBase = args.specBase;
    config.maxVerifyTreeSize = args.maxVerifyTreeSize;
    config.maxDraftTreeSize = args.maxDraftTreeSize;
    config.profilingDetailed = args.profilingDetailed;
    config.tpSize = tpSize;
    config.tpRank = tpRank;

    // Create and run the builder
    builder::LLMBuilder llmBuilder(args.onnxDir, args.engineDir, config);
    if (!llmBuilder.build())
    {
        LOG_ERROR("Failed to build LLM engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("[TP rank %d/%d] LLM engine built successfully.", tpRank, tpSize);

    return EXIT_SUCCESS;
}
