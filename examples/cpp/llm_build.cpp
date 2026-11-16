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

#include <tensorrt_edgellm/builder/builderUtils.h>
#include <tensorrt_edgellm/builder/llmBuilder.h>
#include <tensorrt_edgellm/common/cudaUtils.h>
#include <tensorrt_edgellm/common/fileUtils.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/parallelArtifactNames.h>
#ifdef ENABLE_SAFETY_RUNTIME
#include <tensorrt_edgellm/builder/safety/remoteAutoTuningUtils.h>
#include <tensorrt_edgellm/builder/safety/safetyLLMBuilder.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    LOCAL_DEVICE = 717,
    SAFETY = 718,
    MSS_PROFILE = 719,
    REMOTE_AUTO_TUNING_CONFIG = 720,
    SAFETY_PLUGIN_LIBRARY = 721
};

struct CommonLLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    bool debug{false};
    bool profilingDetailed{false}; // Enable detailed profiling verbosity for layer info extraction
};

struct LLMBuildArgs : CommonLLMBuildArgs
{
    int64_t maxInputLen{1024};
    int64_t maxKVCacheCapacity{4096};
    int64_t maxKVPoolPages{0};
    bool maxKVPoolPagesSet{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // Default to 0 means no LoRA
    bool specDraft{false};
    bool specBase{false};
    int64_t maxSpecVerifyTokens{60};
    int64_t maxSpecDraftTokens{60};
    bool maxSpecDraftTokensSet{false};
    int tpSize{1};
    int tpRank{0};
    int localDevice{0};
    bool tpSizeSet{false};
    bool tpRankSet{false};
    bool localDeviceSet{false};
};

#ifdef ENABLE_SAFETY_RUNTIME
struct SafetyLLMBuildArgs : CommonLLMBuildArgs
{
    std::vector<safety::builder::SafetyLLMStaticShapeProfile> safetyProfiles;
    std::string remoteAutoTuningConfig;
    std::string safetyPluginLibrary;
};
#endif

namespace
{

std::string getInputConfigFileName(int tpSize, int tpRank)
{
    parallel_artifacts::RankArtifactContext const context{tpSize, tpRank};
    return parallel_artifacts::configFileName(context);
}

#ifdef ENABLE_SAFETY_RUNTIME
bool parseSafetyProfile(char const* value, safety::builder::SafetyLLMStaticShapeProfile& profile)
{
    try
    {
        std::vector<int64_t> dimensions;
        std::stringstream stream{value != nullptr ? value : ""};
        std::string token;
        while (std::getline(stream, token, ','))
        {
            size_t parsedCharacters{0};
            int64_t const dimension = std::stoll(token, &parsedCharacters);
            if (parsedCharacters != token.size())
            {
                return false;
            }
            dimensions.push_back(dimension);
        }
        if (dimensions.size() != 3 && dimensions.size() != 4)
        {
            return false;
        }
        profile.batchSize = dimensions[0];
        profile.inputSeqLen = dimensions[1];
        profile.kvCacheCapacity = dimensions[2];
        profile.imageTokens = dimensions.size() == 4 ? dimensions[3] : 0;
        return profile.batchSize > 0 && profile.inputSeqLen > 0 && profile.kvCacheCapacity > 0
            && profile.imageTokens >= 0;
    }
    catch (std::exception const&)
    {
        return false;
    }
}
#endif

} // namespace

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnx-dir <dir> --engine-dir <dir> [--max-input-len <int>] "
                 "[--max-kv-cache-capacity <int>] [--max-batch-size <int>] [--debug] [--max-lora-rank <int>]"
                 " [--max-kv-pool-pages <int>]"
                 " [--spec-draft] [--spec-base] [--max-spec-verify-tokens <int>] "
                 "[--max-spec-draft-tokens <int>] [--profiling-detailed] [--tp-size <int> --tp-rank <int>] "
                 "[--local-device <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --onnx-dir                 Provide the input ONNX directory path. Required. " << std::endl;
    std::cerr << "  --engine-dir               Provide the output TensorRT engine directory path. Required. "
              << std::endl;
    std::cerr << "  --max-input-len             Provide the maximum input length for the model. Default = 1024"
              << std::endl;
    std::cerr << "  --max-kv-cache-capacity      Provide the maximum KV cache capacity (sequence length). "
                 "Default = 4096"
              << std::endl;
    std::cerr << "  --max-kv-pool-pages          Exact physical K-page count for the KV pool. Default = 0 "
                 "(minimum active pages)"
              << std::endl;
    std::cerr << "  --max-batch-size            Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --debug                   Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --max-lora-rank             Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    std::cerr
        << "  --spec-draft               Build as speculative decoding draft model (EAGLE/MTP/DFlash/JetSpec/DSpark)"
        << std::endl;
    std::cerr
        << "  --spec-base                Build as speculative decoding base model (EAGLE/MTP/DFlash/JetSpec/DSpark)"
        << std::endl;
    std::cerr << "  --max-spec-verify-tokens       Maximum input_ids tokens for base model verification. Default = 60"
              << std::endl;
    std::cerr << "  --max-spec-draft-tokens        Maximum input_ids tokens for draft model generation. Default = 60 "
                 "(61 for DSpark with sample_from_anchor=false)"
              << std::endl;
    std::cerr << "  --profiling-detailed       Enable detailed profiling verbosity to include ONNX op names "
                 "in layer info. Use for DLSim analysis."
              << std::endl
              << "  --tp-size                  Tensor parallel world size for per-rank engine builds. "
                 "Must be paired with --tp-rank."
              << std::endl
              << "  --tp-rank                  Tensor parallel rank for per-rank engine builds. Must be in [0, tpSize)."
              << std::endl
              << "  --local-device             Node-local CUDA device ordinal for an explicit TP build. Defaults to "
                 "tpRank."
              << std::endl
              << std::endl;
#ifdef ENABLE_SAFETY_RUNTIME
    std::cerr << "Safety Builder options:" << std::endl
              << "  --safety                  Build an engine with TensorRT kSAFETY capability." << std::endl
              << "  --mss-profile B,S,KV[,I]   Add a fixed MSS profile; I is the image-token count for VLM."
              << std::endl
              << "  --remote-auto-tuning-config  TensorRT remote auto-tuning URI. May instead be supplied through "
                 "EDGELLM_REMOTE_AUTOTUNING_CONFIG. Omit for local GPU auto-tuning."
              << std::endl
              << "  --safety-plugin-library     Safety plugin library path. Defaults to EDGELLM_SAFETY_PLUGIN_PATH, "
                 "then build/libNvInfer_edgellm_plugin_safe.so."
              << std::endl;
#endif
}

bool parseLLMBuildArgs(CommonLLMBuildArgs& commonArgs, LLMBuildArgs& args,
#ifdef ENABLE_SAFETY_RUNTIME
    SafetyLLMBuildArgs& safetyArgs, bool& useSafety,
#endif
    int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, LLMBuildOptionId::HELP},
        {"onnx-dir", required_argument, 0, LLMBuildOptionId::ONNX_DIR},
        {"onnxDir", required_argument, 0, LLMBuildOptionId::ONNX_DIR},
        {"engine-dir", required_argument, 0, LLMBuildOptionId::ENGINE_DIR},
        {"engineDir", required_argument, 0, LLMBuildOptionId::ENGINE_DIR},
        {"max-input-len", required_argument, 0, LLMBuildOptionId::MAX_INPUT_LEN},
        {"maxInputLen", required_argument, 0, LLMBuildOptionId::MAX_INPUT_LEN},
        {"max-kv-cache-capacity", required_argument, 0, LLMBuildOptionId::MAX_KV_CACHE_CAPACITY},
        {"maxKVCacheCapacity", required_argument, 0, LLMBuildOptionId::MAX_KV_CACHE_CAPACITY},
        {"max-kv-pool-pages", required_argument, 0, LLMBuildOptionId::MAX_KV_POOL_PAGES},
        {"maxKVPoolPages", required_argument, 0, LLMBuildOptionId::MAX_KV_POOL_PAGES},
        {"debug", no_argument, 0, LLMBuildOptionId::DEBUG},
        {"max-batch-size", required_argument, 0, LLMBuildOptionId::MAX_BATCH_SIZE},
        {"maxBatchSize", required_argument, 0, LLMBuildOptionId::MAX_BATCH_SIZE},
        {"max-lora-rank", required_argument, 0, LLMBuildOptionId::MAX_LORA_RANK},
        {"maxLoraRank", required_argument, 0, LLMBuildOptionId::MAX_LORA_RANK},
        {"spec-draft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT},
        {"specDraft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT},
        {"eagle-draft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT},
        {"eagleDraft", no_argument, 0, LLMBuildOptionId::SPEC_DRAFT}, // deprecated alias
        {"spec-base", no_argument, 0, LLMBuildOptionId::SPEC_BASE},
        {"specBase", no_argument, 0, LLMBuildOptionId::SPEC_BASE},
        {"eagle-base", no_argument, 0, LLMBuildOptionId::SPEC_BASE},
        {"eagleBase", no_argument, 0, LLMBuildOptionId::SPEC_BASE}, // deprecated alias
        {"max-spec-verify-tokens", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"max-verify-tree-size", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"maxVerifyTreeSize", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"maxSpecVerifyTokens", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"max-spec-draft-tokens", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"max-draft-tree-size", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"maxDraftTreeSize", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"maxSpecDraftTokens", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"profiling-detailed", no_argument, 0, LLMBuildOptionId::PROFILING_DETAILED},
        {"profilingDetailed", no_argument, 0, LLMBuildOptionId::PROFILING_DETAILED},
        {"tp-size", required_argument, 0, LLMBuildOptionId::TP_SIZE},
        {"tpSize", required_argument, 0, LLMBuildOptionId::TP_SIZE},
        {"tp-rank", required_argument, 0, LLMBuildOptionId::TP_RANK},
        {"tpRank", required_argument, 0, LLMBuildOptionId::TP_RANK},
        {"local-device", required_argument, 0, LLMBuildOptionId::LOCAL_DEVICE},
        {"localDevice", required_argument, 0, LLMBuildOptionId::LOCAL_DEVICE},
        {"safety", no_argument, 0, LLMBuildOptionId::SAFETY},
        {"mss-profile", required_argument, 0, LLMBuildOptionId::MSS_PROFILE},
        {"mssProfile", required_argument, 0, LLMBuildOptionId::MSS_PROFILE},
        {"remote-auto-tuning-config", required_argument, 0, LLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG},
        {"remoteAutoTuningConfig", required_argument, 0, LLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG},
        {"safety-plugin-library", required_argument, 0, LLMBuildOptionId::SAFETY_PLUGIN_LIBRARY},
        {"safetyPluginLibrary", required_argument, 0, LLMBuildOptionId::SAFETY_PLUGIN_LIBRARY}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", buildOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMBuildOptionId::HELP: commonArgs.help = true; return true;
        case LLMBuildOptionId::ONNX_DIR:
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
        case LLMBuildOptionId::ENGINE_DIR:
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
                args.maxKVPoolPagesSet = true;
            }
            break;
        case LLMBuildOptionId::DEBUG: commonArgs.debug = true; break;
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
                args.maxSpecVerifyTokens = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_DRAFT_TREE_SIZE:
            if (optarg)
            {
                args.maxSpecDraftTokens = std::stoi(optarg);
                args.maxSpecDraftTokensSet = true;
            }
            break;
        case LLMBuildOptionId::PROFILING_DETAILED: commonArgs.profilingDetailed = true; break;
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
#ifdef ENABLE_SAFETY_RUNTIME
        case LLMBuildOptionId::SAFETY: useSafety = true; break;
        case LLMBuildOptionId::MSS_PROFILE:
        {
            safety::builder::SafetyLLMStaticShapeProfile profile;
            if (!parseSafetyProfile(optarg, profile))
            {
                LOG_ERROR("--mss-profile expects B,S,KV[,I] with positive B, S, and KV values.");
                return false;
            }
            safetyArgs.safetyProfiles.push_back(profile);
            break;
        }
        case LLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG:
            if (optarg == nullptr)
            {
                return false;
            }
            safetyArgs.remoteAutoTuningConfig = optarg;
            break;
        case LLMBuildOptionId::SAFETY_PLUGIN_LIBRARY:
            if (optarg == nullptr)
            {
                return false;
            }
            safetyArgs.safetyPluginLibrary = optarg;
            break;
#else
        case LLMBuildOptionId::SAFETY:
        case LLMBuildOptionId::MSS_PROFILE:
        case LLMBuildOptionId::REMOTE_AUTO_TUNING_CONFIG:
        case LLMBuildOptionId::SAFETY_PLUGIN_LIBRARY:
            LOG_ERROR("Safety Builder is not enabled; configure with -DENABLE_SAFETY_RUNTIME=ON.");
            return false;
#endif
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    CommonLLMBuildArgs commonArgs;
    LLMBuildArgs args;
#ifdef ENABLE_SAFETY_RUNTIME
    SafetyLLMBuildArgs safetyArgs;
    bool useSafety{false};
#endif
    if ((argc < 2)
        || (!parseLLMBuildArgs(commonArgs, args,
#ifdef ENABLE_SAFETY_RUNTIME
            safetyArgs, useSafety,
#endif
            argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    static_cast<CommonLLMBuildArgs&>(args) = commonArgs;
#ifdef ENABLE_SAFETY_RUNTIME
    static_cast<CommonLLMBuildArgs&>(safetyArgs) = commonArgs;
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

    int tpSize = 1;
    int tpRank = 0;
    bool const useExplicitTpConfig = args.tpSizeSet || args.tpRankSet;

    if (args.tpSizeSet != args.tpRankSet)
    {
        LOG_ERROR("--tp-size and --tp-rank must be specified together.");
        return EXIT_FAILURE;
    }
    if (useExplicitTpConfig)
    {
        if (args.tpSize < 1)
        {
            LOG_ERROR("--tp-size must be >= 1, got %d.", args.tpSize);
            return EXIT_FAILURE;
        }
        if (args.tpRank < 0 || args.tpRank >= args.tpSize)
        {
            LOG_ERROR("--tp-rank must be in [0, tpSize), got tpRank=%d tpSize=%d.", args.tpRank, args.tpSize);
            return EXIT_FAILURE;
        }

        tpSize = args.tpSize;
        tpRank = args.tpRank;
    }

    if (args.localDeviceSet && !useExplicitTpConfig)
    {
        LOG_ERROR("--local-device requires --tp-size and --tp-rank.");
        return EXIT_FAILURE;
    }
    if (args.localDeviceSet && args.localDevice < 0)
    {
        LOG_ERROR("--local-device must be >= 0, got %d.", args.localDevice);
        return EXIT_FAILURE;
    }

    if (tpSize > 1 && (args.specDraft || args.specBase))
    {
        LOG_ERROR("Tensor-parallel speculative decoding engine builds are not supported; build with --tp-size 1.");
        return EXIT_FAILURE;
    }

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
        if (args.maxKVPoolPagesSet)
        {
            LOG_ERROR("--max-kv-pool-pages is not supported with --safety; Safety uses contiguous KV cache.");
            return EXIT_FAILURE;
        }
        if (useExplicitTpConfig || args.localDeviceSet || args.specDraft || args.specBase || args.maxLoraRank != 0)
        {
            LOG_ERROR("The initial Safety Builder supports only single-rank vanilla LLM/VLM builds without LoRA.");
            return EXIT_FAILURE;
        }
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
    nlohmann::json modelConfig;
    if (!builder::loadJsonConfig(configPath, modelConfig))
    {
        return EXIT_FAILURE;
    }

#ifdef ENABLE_SAFETY_RUNTIME
    if (useSafety)
    {
        safety::builder::SafetyLLMBuilderConfig config;
        config.profiles = safetyArgs.safetyProfiles;
        config.remoteAutoTuningConfig = safetyArgs.remoteAutoTuningConfig;
        config.safetyPluginLibrary = safetyArgs.safetyPluginLibrary;
        config.profilingDetailed = safetyArgs.profilingDetailed;
        config.isVlm = std::any_of(config.profiles.begin(), config.profiles.end(),
            [](safety::builder::SafetyLLMStaticShapeProfile const& profile) { return profile.imageTokens > 0; });

        safety::builder::SafetyLLMBuilder safetyBuilder(safetyArgs.onnxDir, safetyArgs.engineDir, std::move(config));
        if (!safetyBuilder.build())
        {
            LOG_ERROR("Failed to build the Safety LLM engine.");
            return EXIT_FAILURE;
        }
        LOG_INFO("Safety LLM engine built successfully.");
        return EXIT_SUCCESS;
    }
#endif

    // Create LLMBuilderConfig from args
    builder::LLMBuilderConfig config;
    config.maxInputLen = args.maxInputLen;
    config.maxKVCacheCapacity = args.maxKVCacheCapacity;
    config.maxKVPoolPages = args.maxKVPoolPages;
    config.maxBatchSize = args.maxBatchSize;
    config.maxLoraRank = args.maxLoraRank;
    config.specDraft = args.specDraft;
    config.specBase = args.specBase;
    config.maxSpecVerifyTokens = args.maxSpecVerifyTokens;
    config.maxSpecDraftTokens
        = builder::resolveOfficialMaxSpecDraftTokens(modelConfig, args.maxSpecDraftTokens, args.maxSpecDraftTokensSet);
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
