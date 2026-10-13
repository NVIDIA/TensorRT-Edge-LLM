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

#include "benchLogger.h"
#include "benchRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "multimodal/common/multimodalRunner.h"
#include "profiling/layerProfiler.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/config/inferencePhase.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

// ==================== Type Definitions ====================

constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
// launchDFlashPrepareBaseVerifyInputs uses one CUDA thread per verify token.
constexpr int32_t kDFlashPrepareBaseVerifyMaxSize{1024};

enum ProfileBenchOptionId : int
{
    HELP = 801,
    ENGINE_DIR = 802,
    DEBUG = 803,
    BATCH_SIZE = 804,
    INPUT_LEN = 805,
    ITERATIONS = 806,
    WARMUP = 807,
    MODE = 812,
    REUSE_KV_LEN = 813,
    PAST_KV_LEN = 814,
    VERIFY_TREE_SIZE = 815,
    DRAFT_TREE_SIZE = 816,
    IMAGE_SIZE = 818,
    OSL = 827,
    OUTPUT_DIR = 828,
    SEED = 829,
    NO_CUDA_GRAPH = 830,
    EXTRACT_LAYER_INFO = 831,
    ACCEPT_RATE = 832,
    DRAFT_STEP = 833,
    PROFILE = 834,
    BLOCK_SIZE = 835,
    CANDIDATE_TOPK = 837,
    CHECKPOINT_DIR = 838,
    ACCEPT_LEN = 839,
};

struct ProfileBenchArgs
{
    bool help{false};
    std::string engineDir;
    std::string checkpointDir;
    bool debug{false};
    int32_t batchSize{1};
    int32_t inputLen{-1}; // Input sequence length per batch (required for prefill modes)
    int32_t iterations{10};
    int32_t warmup{3};
    bool noProfile{true};   // Layer profiling is disabled by default; --profile enables it.
    std::string outputDir;  // Directory to dump output CSV files (layer profiling and E2E timing)
    int32_t imageHeight{0}; // Image height in pixels (required for visual mode)
    int32_t imageWidth{0};  // Image width in pixels (required for visual mode)

    // Mode parameter - no default
    BenchMode mode{BenchMode::kNONE};

    // KV cache parameters - no defaults for required params
    int32_t reuseKVLen{0}; // For prefill: reused KV cache length per batch
    int32_t pastKVLen{-1}; // For decode/verify/draft: past KV cache length per batch (required)

    // Speculative decoding parameters - no defaults
    int32_t verifyTreeSize{-1}; // For spec_verify
    int32_t draftTreeSize{-1};  // For spec_draft_proposal/spec_draft_prefill
    int32_t acceptLen{-1};      // For spec_draft_accept (tokens caught up per pass; default: draftStep+1)

    int32_t osl{1};        // Output sequence length (LLM OSL per batch, default: 1)
    int32_t acceptRate{5}; // Avg accepted tokens per spec-decode iteration (default: 5)
    int32_t draftStep{6};  // Number of drafting steps per spec-decode iteration (default: 6)

    int32_t blockSize{-1}; // DFlash draft block size; -1 = read from engine
    // draftDeltaLen is derived from acceptRate (capped by blockSize) at DFlash setup time; not a CLI arg.
    int32_t draftDeltaLen{0};
    int32_t candidateTopK{-1}; // DDTree branching factor; required for isolated DDTree build

    // Random seed for reproducibility
    uint64_t seed{0};

    // CUDA graph is enabled by default for decode/EAGLE E2E timing.
    bool noCudaGraph{false};

    // Metadata extraction flags - disabled by default for performance.
    // Set via --extractLayerInfo <comma-separated list>.
    ExtractLayerInfo extractLayerInfo;

    BenchOutputParams toOutputParams() const
    {
        BenchOutputParams p;
        p.mode = mode;
        p.batchSize = batchSize;
        p.inputLen = inputLen;
        p.pastKVLen = pastKVLen;
        p.verifyTreeSize = verifyTreeSize;
        p.draftTreeSize = draftTreeSize;
        p.acceptLen = acceptLen;
        p.osl = osl;
        p.imageHeight = imageHeight;
        p.imageWidth = imageWidth;
        p.reuseKVLen = reuseKVLen;
        p.iterations = iterations;
        p.acceptRate = acceptRate;
        p.draftStep = draftStep;
        p.blockSize = blockSize;
        p.draftDeltaLen = draftDeltaLen;
        p.candidateTopK = candidateTopK;
        p.seed = seed;
        return p;
    }
};

// ==================== printUsage ====================

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir <dir> --mode <mode> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Required Options:" << std::endl;
    std::cerr << "  --engineDir               TensorRT engine directory path. Required." << std::endl;
    std::cerr << "  --mode                    Benchmarking mode. Required. One of:" << std::endl;
    std::cerr << "                              prefill           - LLM prefill phase" << std::endl;
    std::cerr << "                              decode            - LLM decode phase" << std::endl;
    std::cerr
        << "                              spec_verify       - Speculative decoding base model verification (EAGLE/MTP)"
        << std::endl;
    std::cerr << "                              spec_draft_accept - Speculative decoding draft accept-token"
              << " catch-up pass (EAGLE/MTP)" << std::endl;
    std::cerr << "                              spec_draft_proposal - Speculative decoding draft proposal (EAGLE/MTP)"
              << std::endl;
    std::cerr << "                              spec_draft_prefill - Speculative decoding draft prefill (EAGLE/MTP)"
              << std::endl;
    std::cerr << "                              visual            - Visual encoder" << std::endl;
    std::cerr << "                              dflash_draft_proposal    - DFlash draft engine, decode round"
              << std::endl;
    std::cerr << "                              dflash_draft_first_round - DFlash draft engine, first round after base "
                 "prefill"
              << std::endl;
    std::cerr << "                              dflash_verify            - DFlash target verify pass (linear or DDTree)"
              << std::endl;
    std::cerr
        << "                              dflash_ddtree_build      - DDTree build kernel in isolation (no executor)"
        << std::endl;
    std::cerr << std::endl;
    std::cerr << "Mode-Specific Required Options:" << std::endl;
    std::cerr << "  For prefill mode:" << std::endl;
    std::cerr << "    --inputLen              Input sequence length. Required." << std::endl;
    std::cerr << "    --reuseKVLen            Reused KV cache length. Optional, default=0." << std::endl;
    std::cerr << "  For decode mode:" << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_verify mode:" << std::endl;
    std::cerr << "    --verifyTreeSize        Verify tree size. Required." << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_draft_proposal mode:" << std::endl;
    std::cerr << "    --draftTreeSize         Draft proposal size for EAGLE/tree-MTP. Required." << std::endl;
    std::cerr << "                            Chain-MTP instead uses the production proposal geometry"
              << " draftStep*draftingTopK (with draftingTopK=1)." << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_draft_accept mode:" << std::endl;
    std::cerr << "    --acceptLen             Tokens caught up per accept pass (default: draftStep+1; pass"
              << " --draftStep matching your deployment's drafting_step)." << std::endl;
    std::cerr << "                            With osl>1, acceptRate must not exceed acceptLen." << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length per batch. Required." << std::endl;
    std::cerr << "    --verifyTreeSize        Optional; defaults to draftStep+1 (chain-MTP engines"
              << " require verifySize == draftStep+1)." << std::endl;
    std::cerr << "  For spec_draft_prefill mode:" << std::endl;
    std::cerr << "    --inputLen              Input sequence length. Required." << std::endl;
    std::cerr << "    --reuseKVLen            Reused KV cache length. Optional, default=0." << std::endl;
    std::cerr << "  For visual mode:" << std::endl;
    std::cerr << "    --engineDir             Visual encoder engine directory. Required." << std::endl;
    std::cerr << "    --imageSize             Image dimensions as HxW (e.g., 896x448). Required." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Common Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --checkpointDir           HF/ModelOpt checkpoint directory for runtime weight loading" << std::endl;
    std::cerr << "  --debug                   Use debug mode (verbose logging)" << std::endl;
    std::cerr << "  --batchSize               Batch size. Default = 1" << std::endl;
    std::cerr << "  --iterations              Number of profiling iterations (after warmup). Default = 10" << std::endl;
    std::cerr << "  --warmup                  Number of warmup iterations. Default = 3" << std::endl;
    std::cerr << "  --osl                     Output sequence length for decode E2E timing. Default = 1." << std::endl;
    std::cerr << "                            osl=1: E2E runs --iterations times." << std::endl;
    std::cerr << "                            osl>1: E2E runs full sequence decode once." << std::endl;
    std::cerr << "  --profile                 Enable per-layer profiling. Disabled by default." << std::endl;
    std::cerr << "  --outputDir               Directory to dump output CSV files (layer profiling and E2E timing)"
              << std::endl;
    std::cerr << "  --seed                    Random seed for reproducible data. Default = 0" << std::endl;
    std::cerr << "  --noCudaGraph             Disable CUDA graph capture for decode/EAGLE E2E timing." << std::endl;
    std::cerr << "                            Capture is enabled by default and falls back to non-graph on failure."
              << std::endl;
    std::cerr << "  --extractLayerInfo <opts>  Comma-separated list of layer info to extract:" << std::endl;
    std::cerr << "                              all, shapes, onnx_ops, tactics, data_types" << std::endl;
    std::cerr << "                            Implies --profile." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Speculative Decoding Options:" << std::endl;
    std::cerr << "  --acceptRate              Avg accepted tokens per spec-decode iteration (default: 5)." << std::endl;
    std::cerr << "                            verify_steps = ceil((osl-1) / acceptRate)." << std::endl;
    std::cerr << "                            Also feeds DFlash draft proposal deltaLen "
                 "(capped at blockSize)."
              << std::endl;
    std::cerr << "  --draftStep               Number of drafting steps per spec-decode iteration (default: 6)."
              << std::endl;
    std::cerr << "                            draft_calls = verify_steps * (draftStep-1)." << std::endl;
    std::cerr << std::endl;
    std::cerr << "DFlash Options:" << std::endl;
    std::cerr << "  --blockSize             DFlash draft block size. Default = engine's dflash_block_size."
              << std::endl;
    std::cerr << "  --candidateTopK         DDTree branching factor (for kDFLASH_DDTREE_BUILD). Required and must "
                 "be > 1."
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  # Prefill mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode prefill --inputLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Decode mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode decode --pastKVLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Spec-decode verify mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode spec_verify --verifyTreeSize 60 --pastKVLen 128"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Spec-decode draft accept-token catch-up mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode spec_draft_accept --pastKVLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Spec-decode draft proposal mode" << std::endl;
    std::cerr << "  " << programName
              << " --engineDir ./engines --mode spec_draft_proposal --draftTreeSize 60 --pastKVLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Visual encoder mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./visual_engines --mode visual --imageSize 896x448" << std::endl;
}

// ==================== parseArgs + validateArgs ====================

bool parseArgs(ProfileBenchArgs& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, 0, ProfileBenchOptionId::HELP},
        {"engineDir", required_argument, 0, ProfileBenchOptionId::ENGINE_DIR},
        {"checkpointDir", required_argument, 0, ProfileBenchOptionId::CHECKPOINT_DIR},
        {"debug", no_argument, 0, ProfileBenchOptionId::DEBUG},
        {"batchSize", required_argument, 0, ProfileBenchOptionId::BATCH_SIZE},
        {"inputLen", required_argument, 0, ProfileBenchOptionId::INPUT_LEN},
        {"iterations", required_argument, 0, ProfileBenchOptionId::ITERATIONS},
        {"warmup", required_argument, 0, ProfileBenchOptionId::WARMUP},
        {"mode", required_argument, 0, ProfileBenchOptionId::MODE},
        {"reuseKVLen", required_argument, 0, ProfileBenchOptionId::REUSE_KV_LEN},
        {"pastKVLen", required_argument, 0, ProfileBenchOptionId::PAST_KV_LEN},
        {"verifyTreeSize", required_argument, 0, ProfileBenchOptionId::VERIFY_TREE_SIZE},
        {"draftTreeSize", required_argument, 0, ProfileBenchOptionId::DRAFT_TREE_SIZE},
        {"acceptLen", required_argument, 0, ProfileBenchOptionId::ACCEPT_LEN},
        {"profile", no_argument, 0, ProfileBenchOptionId::PROFILE},
        {"outputDir", required_argument, 0, ProfileBenchOptionId::OUTPUT_DIR},
        {"osl", required_argument, 0, ProfileBenchOptionId::OSL},
        {"seed", required_argument, 0, ProfileBenchOptionId::SEED},
        {"imageSize", required_argument, 0, ProfileBenchOptionId::IMAGE_SIZE},
        {"noCudaGraph", no_argument, 0, ProfileBenchOptionId::NO_CUDA_GRAPH},
        {"extractLayerInfo", required_argument, 0, ProfileBenchOptionId::EXTRACT_LAYER_INFO},
        {"acceptRate", required_argument, 0, ProfileBenchOptionId::ACCEPT_RATE},
        {"draftStep", required_argument, 0, ProfileBenchOptionId::DRAFT_STEP},
        {"blockSize", required_argument, nullptr, ProfileBenchOptionId::BLOCK_SIZE},
        {"candidateTopK", required_argument, nullptr, ProfileBenchOptionId::CANDIDATE_TOPK}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        try
        {
            switch (opt)
            {
            case ProfileBenchOptionId::HELP: args.help = true; return true;
            case ProfileBenchOptionId::ENGINE_DIR: args.engineDir = optarg; break;
            case ProfileBenchOptionId::CHECKPOINT_DIR: args.checkpointDir = optarg; break;
            case ProfileBenchOptionId::DEBUG: args.debug = true; break;
            case ProfileBenchOptionId::BATCH_SIZE:
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::INPUT_LEN:
                args.inputLen = std::stoi(optarg);
                if (args.inputLen <= 0)
                {
                    LOG_ERROR("Invalid inputLen: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::ITERATIONS:
                args.iterations = std::stoi(optarg);
                if (args.iterations <= 0)
                {
                    LOG_ERROR("Invalid iterations: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::WARMUP:
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::MODE:
            {
                std::string modeStr = optarg;
                if (modeStr == "prefill")
                {
                    args.mode = BenchMode::kPREFILL;
                }
                else if (modeStr == "decode")
                {
                    args.mode = BenchMode::kDECODE;
                }
                else if (modeStr == "spec_verify" || modeStr == "eagle_verify")
                {
                    args.mode = BenchMode::kEAGLE_VERIFY;
                }
                else if (modeStr == "spec_draft_proposal" || modeStr == "eagle_draft_proposal")
                {
                    args.mode = BenchMode::kEAGLE_DRAFT_PROPOSAL;
                }
                else if (modeStr == "spec_draft_prefill" || modeStr == "eagle_draft_prefill")
                {
                    args.mode = BenchMode::kEAGLE_DRAFT_PREFILL;
                }
                else if (modeStr == "spec_draft_accept" || modeStr == "eagle_draft_accept")
                {
                    args.mode = BenchMode::kEAGLE_DRAFT_ACCEPT;
                }
                else if (modeStr == "visual")
                {
                    args.mode = BenchMode::kVISUAL;
                }
                else if (modeStr == "dflash_draft_proposal")
                {
                    args.mode = BenchMode::kDFLASH_DRAFT_PROPOSAL;
                }
                else if (modeStr == "dflash_draft_first_round")
                {
                    args.mode = BenchMode::kDFLASH_DRAFT_FIRST_ROUND;
                }
                else if (modeStr == "dflash_verify")
                {
                    args.mode = BenchMode::kDFLASH_VERIFY;
                }
                else if (modeStr == "dflash_ddtree_build")
                {
                    args.mode = BenchMode::kDFLASH_DDTREE_BUILD;
                }
                else
                {
                    LOG_ERROR("Invalid mode: %s", optarg);
                    return false;
                }
                break;
            }
            case ProfileBenchOptionId::REUSE_KV_LEN:
                args.reuseKVLen = std::stoi(optarg);
                if (args.reuseKVLen < 0)
                {
                    LOG_ERROR("Invalid reuseKVLen: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::PAST_KV_LEN:
                args.pastKVLen = std::stoi(optarg);
                if (args.pastKVLen < 0)
                {
                    LOG_ERROR("Invalid pastKVLen: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::VERIFY_TREE_SIZE:
                args.verifyTreeSize = std::stoi(optarg);
                if (args.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid verifyTreeSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::DRAFT_TREE_SIZE:
                args.draftTreeSize = std::stoi(optarg);
                if (args.draftTreeSize <= 0)
                {
                    LOG_ERROR("Invalid draftTreeSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::ACCEPT_LEN:
                args.acceptLen = std::stoi(optarg);
                if (args.acceptLen <= 0)
                {
                    LOG_ERROR("Invalid acceptLen: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::IMAGE_SIZE:
            {
                std::string sizeStr = optarg;
                size_t xPos = sizeStr.find('x');
                if (xPos == std::string::npos)
                    xPos = sizeStr.find('X');
                if (xPos != std::string::npos)
                {
                    args.imageHeight = std::stoi(sizeStr.substr(0, xPos));
                    args.imageWidth = std::stoi(sizeStr.substr(xPos + 1));
                }
                else
                {
                    args.imageHeight = std::stoi(sizeStr);
                    args.imageWidth = args.imageHeight;
                }
                if (args.imageWidth <= 0 || args.imageHeight <= 0)
                {
                    LOG_ERROR("Invalid imageSize: height and width must be positive");
                    return false;
                }
                break;
            }
            case ProfileBenchOptionId::PROFILE: args.noProfile = false; break;
            case ProfileBenchOptionId::OUTPUT_DIR: args.outputDir = optarg; break;
            case ProfileBenchOptionId::OSL:
                args.osl = std::stoi(optarg);
                if (args.osl < 1)
                {
                    LOG_ERROR("Invalid osl: must be >= 1");
                    return false;
                }
                break;
            case ProfileBenchOptionId::SEED: args.seed = std::stoull(optarg); break;
            case ProfileBenchOptionId::NO_CUDA_GRAPH: args.noCudaGraph = true; break;
            case ProfileBenchOptionId::EXTRACT_LAYER_INFO:
            {
                std::string val = optarg;
                std::istringstream stream(val);
                std::string token;
                while (std::getline(stream, token, ','))
                {
                    if (token == "all")
                    {
                        args.extractLayerInfo.shapes = true;
                        args.extractLayerInfo.onnxOps = true;
                        args.extractLayerInfo.tactics = true;
                        args.extractLayerInfo.dataTypes = true;
                    }
                    else if (token == "shapes")
                        args.extractLayerInfo.shapes = true;
                    else if (token == "onnx_ops")
                        args.extractLayerInfo.onnxOps = true;
                    else if (token == "tactics")
                        args.extractLayerInfo.tactics = true;
                    else if (token == "data_types")
                        args.extractLayerInfo.dataTypes = true;
                    else
                    {
                        LOG_ERROR(
                            "Unknown --extractLayerInfo value: '%s'. Valid: all,shapes,onnx_ops,tactics,data_types",
                            token.c_str());
                        return false;
                    }
                }
                break;
            }
            case ProfileBenchOptionId::ACCEPT_RATE: args.acceptRate = std::stoi(optarg); break;
            case ProfileBenchOptionId::DRAFT_STEP: args.draftStep = std::stoi(optarg); break;
            case ProfileBenchOptionId::BLOCK_SIZE:
                args.blockSize = std::stoi(optarg);
                ELLM_CHECK(args.blockSize > 0, "--blockSize must be positive");
                break;
            case ProfileBenchOptionId::CANDIDATE_TOPK:
                args.candidateTopK = std::stoi(optarg);
                ELLM_CHECK(args.candidateTopK > 0, "--candidateTopK must be positive");
                break;
            default: return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to parse argument: %s", e.what());
            return false;
        }
    }

    if (args.extractLayerInfo.any())
    {
        args.noProfile = false;
    }

    return true;
}

static bool isDFlashMode(BenchMode mode);
static bool isSpecDecodeMode(BenchMode mode);

bool validateArgs(ProfileBenchArgs const& args)
{
    if (args.engineDir.empty())
    {
        LOG_ERROR("--engineDir is required");
        return false;
    }

    if (args.mode == BenchMode::kNONE)
    {
        LOG_ERROR("--mode is required. Use --help for available modes.");
        return false;
    }

    bool const isSpecMode = isSpecDecodeMode(args.mode);
    if (isSpecMode && args.acceptRate <= 0)
    {
        LOG_ERROR("--acceptRate must be positive for speculative decoding modes");
        return false;
    }
    if (isSpecMode && args.draftStep < 1)
    {
        LOG_ERROR("--draftStep must be at least 1 for speculative decoding modes");
        return false;
    }
    if (args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT && args.osl > 1 && args.acceptRate > args.acceptLen)
    {
        LOG_ERROR(
            "--acceptRate (%d) cannot exceed --acceptLen (%d) for spec_draft_accept with --osl > 1; "
            "increase --acceptLen or reduce --acceptRate",
            args.acceptRate, args.acceptLen);
        return false;
    }

    if (args.mode == BenchMode::kDFLASH_DDTREE_BUILD && !args.noProfile)
    {
        LOG_ERROR(
            "--profile and --extractLayerInfo are not supported for dflash_ddtree_build because it does not "
            "execute a TensorRT engine");
        return false;
    }

    switch (args.mode)
    {
    case BenchMode::kPREFILL:
        if (args.inputLen < 0)
        {
            LOG_ERROR("--inputLen is required for prefill mode");
            return false;
        }
        break;
    case BenchMode::kDECODE:
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for decode mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_VERIFY:
        if (args.verifyTreeSize < 0)
        {
            LOG_ERROR("--verifyTreeSize is required for spec_verify mode");
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for spec_verify mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        if (args.draftTreeSize < 0)
        {
            LOG_ERROR("--draftTreeSize is required for spec_draft_proposal mode");
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for spec_draft_proposal mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_DRAFT_PREFILL:
        if (args.inputLen < 0)
        {
            LOG_ERROR("--inputLen is required for spec_draft_prefill mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT:
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for spec_draft_accept mode");
            return false;
        }
        break;
    case BenchMode::kVISUAL:
        if (args.imageHeight <= 0 || args.imageWidth <= 0)
        {
            LOG_ERROR("--imageSize is required for visual mode (e.g., --imageSize 896x448)");
            return false;
        }
        break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND:
        if (args.inputLen <= 0)
        {
            LOG_ERROR("--inputLen is required for dflash_draft_first_round mode");
            return false;
        }
        break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL:
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for dflash modes");
            return false;
        }
        break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        if (args.candidateTopK <= 1)
        {
            LOG_ERROR("--candidateTopK is required for dflash_ddtree_build mode and must be > 1");
            return false;
        }
        if (args.verifyTreeSize <= 0)
        {
            LOG_ERROR("--verifyTreeSize is required for dflash_ddtree_build mode");
            return false;
        }
        if (args.verifyTreeSize > kernel::kDDTreeMaxVerifySize)
        {
            LOG_ERROR("DFlash DDTree verify size (%d) exceeds the production kernel limit (%d)", args.verifyTreeSize,
                kernel::kDDTreeMaxVerifySize);
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for dflash_ddtree_build mode");
            return false;
        }
        break;
    case BenchMode::kDFLASH_VERIFY:
        if (args.verifyTreeSize <= 0)
        {
            LOG_ERROR("--verifyTreeSize is required for dflash_verify mode");
            return false;
        }
        if (args.verifyTreeSize > kDFlashPrepareBaseVerifyMaxSize)
        {
            LOG_ERROR("DFlash verify tree size (%d) exceeds the base verify metadata kernel limit (%d)",
                args.verifyTreeSize, kDFlashPrepareBaseVerifyMaxSize);
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for dflash_verify mode");
            return false;
        }
        break;
    default: LOG_ERROR("Unknown mode"); return false;
    }

    return true;
}

// ==================== Helper: detect engine paths ====================

static bool isDFlashMode(BenchMode mode)
{
    return mode == BenchMode::kDFLASH_DRAFT_PROPOSAL || mode == BenchMode::kDFLASH_DRAFT_FIRST_ROUND
        || mode == BenchMode::kDFLASH_VERIFY || mode == BenchMode::kDFLASH_DDTREE_BUILD;
}

static bool needsExecutor(BenchMode mode)
{
    // All modes except DDTree build execute a TensorRT engine.
    return mode != BenchMode::kDFLASH_DDTREE_BUILD;
}

bool isDraftEngineMode(BenchMode mode)
{
    return mode == BenchMode::kEAGLE_DRAFT_PROPOSAL || mode == BenchMode::kEAGLE_DRAFT_PREFILL
        || mode == BenchMode::kEAGLE_DRAFT_ACCEPT || mode == BenchMode::kDFLASH_DRAFT_PROPOSAL
        || mode == BenchMode::kDFLASH_DRAFT_FIRST_ROUND;
}

static bool isSpecDecodeMode(BenchMode mode)
{
    return mode == BenchMode::kEAGLE_VERIFY || mode == BenchMode::kEAGLE_DRAFT_PROPOSAL
        || mode == BenchMode::kEAGLE_DRAFT_PREFILL || mode == BenchMode::kEAGLE_DRAFT_ACCEPT || isDFlashMode(mode);
}

// ==================== main ====================

int main(int argc, char** argv)
{
    // ===== Phase 0: Parse & Setup =====
    ProfileBenchArgs args;
    if ((argc < 2) || (!parseArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    // Resolve the chain geometry before deployment creation and configuration logging.
    if (args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT)
    {
        int64_t const chainVerifySize = static_cast<int64_t>(args.draftStep) + 1;
        if ((args.acceptLen <= 0 || args.verifyTreeSize <= 0) && chainVerifySize > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR(
                "--draftStep is too large to derive the default acceptLen/verifyTreeSize; pass a value no "
                "greater than %d",
                std::numeric_limits<int32_t>::max() - 1);
            return EXIT_FAILURE;
        }
        if (args.acceptLen <= 0)
        {
            args.acceptLen = static_cast<int32_t>(chainVerifySize);
        }
        if (args.verifyTreeSize <= 0)
        {
            args.verifyTreeSize = static_cast<int32_t>(chainVerifySize);
        }
    }

    if (!validateArgs(args))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    LOG_INFO("=== LLM Profile Benchmark ===");
    LOG_INFO("Mode: %s", modeToString(args.mode).c_str());

    // Only load the edgellm plugin library when we actually create an EngineExecutor.
    // kDFLASH_DDTREE_BUILD times a bare CUDA kernel and doesn't need TRT plugins.
    std::unique_ptr<void, DlDeleter> pluginHandles;
    if (needsExecutor(args.mode))
    {
        pluginHandles = loadEdgellmPluginLib();
    }

    // DDTree build times a raw CUDA kernel — CUDA graph capture is neither meaningful
    // nor supported (no executor prepare/execute path), so force-disable it here.
    if (args.mode == BenchMode::kDFLASH_DDTREE_BUILD)
    {
        args.noCudaGraph = true;
        LOG_INFO("CUDA graph disabled for dflash_ddtree_build kernel timing");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (!args.noProfile)
    {
        layerProfiler::LayerProfiler::getInstance().setEnabled(true);
        LOG_INFO("Layer profiling enabled. E2E CUDA graph timing will be skipped.");
    }
    else
    {
        LOG_INFO("Layer profiling disabled from startup. E2E timing only.");
    }

    std::map<std::string, LayerMetadata> layerMetadata;

    if (!args.outputDir.empty())
    {
        std::filesystem::create_directories(args.outputDir);
        LOG_INFO("Output CSV files will be saved to: %s", args.outputDir.c_str());
    }

    std::vector<KernelTimes> timesPerIter;
    timesPerIter.reserve(args.iterations);
    float e2eTimeMsResult = 0.0f;
    OrderedLayerTimings layerTimings;

    // ===== Phase 1: Initialize Engine =====
    std::unique_ptr<rt::EngineExecutor> executor;
    std::unique_ptr<rt::SharedResources> resources;
    std::unique_ptr<rt::PipelineIO> io;
    rt::TensorMap tensorMap;
    std::unique_ptr<rt::StepPreparer> stepPreparer;
    std::unique_ptr<rt::DeepstackBinding> deepstack;
    rt::DeploymentConfig deployment;
    rt::Tensor contextMemory;
    rt::Tensor diffusionCanvasIds;
    rt::Tensor diffusionPrevSelfConditioningEmbeds;
    rt::Tensor diffusionNextSelfConditioningEmbeds;
    rt::Tensor diffusionSelfConditioningTemperature;
    bool useRaggedBindings{false};

    // Visual mode uses MultimodalRunner (unchanged from legacy)
    std::unique_ptr<rt::MultimodalRunner> visualRunner;
    std::unique_ptr<nvinfer1::ICudaEngine> standaloneEngine;
    int64_t imageTokens = 0;

    if (args.mode == BenchMode::kVISUAL)
    {
        try
        {
            int32_t maxSeqLen = 4096;
            for (auto const& siblingDir : {"llm", "base"})
            {
                auto llmConfigPath = std::filesystem::path(args.engineDir).parent_path() / siblingDir / "config.json";
                if (std::filesystem::exists(llmConfigPath))
                {
                    std::ifstream f(llmConfigPath);
                    if (f.is_open())
                    {
                        auto cfg = Json::parse(f);
                        if (cfg.contains("builder_config") && cfg["builder_config"].contains("max_kv_cache_capacity"))
                        {
                            maxSeqLen = cfg["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
                            LOG_INFO("Read maxSequenceLength=%d from %s", maxSeqLen, llmConfigPath.string().c_str());
                        }
                    }
                    break;
                }
            }
            visualRunner
                = rt::MultimodalRunner::create(args.engineDir, args.batchSize, maxSeqLen, stream, args.checkpointDir);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create MultimodalRunner: %s", e.what());
            return EXIT_FAILURE;
        }

        int64_t memSize = visualRunner->getRequiredContextMemorySize();
        contextMemory
            = rt::Tensor(rt::Coords{memSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "context_memory");
        visualRunner->setContextMemory(contextMemory);

        rt::LLMGenerationRequest dummyRequest;
        for (int32_t b = 0; b < args.batchSize; ++b)
        {
            rt::LLMGenerationRequest::Request req;
            rt::Tensor fakeImage({1, static_cast<int64_t>(args.imageHeight), static_cast<int64_t>(args.imageWidth), 3},
                rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "fake");
            std::memset(fakeImage.rawPointer(), 128, static_cast<size_t>(args.imageHeight) * args.imageWidth * 3);
            req.imageBuffers.emplace_back(std::move(fakeImage));
            req.imageBuffers.back().doResize = false;
            dummyRequest.requests.push_back(std::move(req));
        }

        std::vector<std::vector<int32_t>> unusedInputIds;
        rt::Tensor unusedRope;
        if (!visualRunner->preprocess(dummyRequest, unusedInputIds, nullptr, unusedRope, stream, true))
        {
            LOG_ERROR("Failed to prepare dummy visual inputs for %dx%d", args.imageHeight, args.imageWidth);
            return EXIT_FAILURE;
        }

        imageTokens = visualRunner->getOutputEmbedding().getShape()[0];
        LOG_INFO("Image Size: %dx%d -> %ld image tokens (batch=%d)", args.imageHeight, args.imageWidth, imageTokens,
            args.batchSize);

        // Only needed by the --extractLayerInfo path (used at Phase 4 below).
        if (args.extractLayerInfo.any())
        {
            standaloneEngine = loadStandaloneEngine(std::filesystem::path(args.engineDir) / "visual.engine");
        }
    }
    else
    {
        // --- Resolve engine/config paths ---
        std::filesystem::path const dir{args.engineDir};
        bool const hasSpecDecode = isSpecDecodeMode(args.mode);

        std::filesystem::path baseConfigPath = dir / "config.json";
        if (!std::filesystem::exists(baseConfigPath))
        {
            baseConfigPath = dir / "base_config.json";
        }

        std::optional<std::filesystem::path> draftConfigPath;
        std::optional<rt::SpecDecodeDraftingConfig> draftingConfig;
        try
        {
            if (hasSpecDecode)
            {
                draftConfigPath = dir / "draft_config.json";
                // Bench needs a SpecDecodeDraftingConfig to create DeploymentConfig.
                // Most isolated modes use their requested tensor width as a harmless
                // synthetic drafting shape. MTP proposal width is not branching,
                // however: it remains a linear chain regardless of draftTreeSize.
                rt::SpecDecodeDraftingConfig dc;
                if (isDFlashMode(args.mode))
                {
                    // DFlash configs use different fields (topK is candidate branching factor,
                    // step is fixed to 1) — leave whatever args provided (or 1s) so the factory
                    // accepts, then fold engine-derived values into args after the deployment
                    // is loaded (see DFlash defaults block below).
                    dc.draftingStep = 1;
                    dc.draftingTopK = args.candidateTopK > 0 ? args.candidateTopK : 1;
                    dc.verifySize = args.verifyTreeSize > 0 ? args.verifyTreeSize : 1;
                    dc.dflashBlockSize = args.blockSize > 0 ? args.blockSize : 0;
                }
                else
                {
                    rt::SpecDecodeMode const engineSpecMode = rt::parseEngineConfig(baseConfigPath).specDecodeType;
                    if (args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT)
                    {
                        ELLM_CHECK(
                            engineSpecMode == rt::SpecDecodeMode::kEAGLE || engineSpecMode == rt::SpecDecodeMode::kMTP,
                            "spec_draft_accept supports only spec_decode_type=eagle3 or mtp; engine config uses "
                                + std::string(rt::specDecodeModeName(engineSpecMode))
                                + ". Use the benchmark mode dedicated to that speculative family.");
                    }

                    if (engineSpecMode == rt::SpecDecodeMode::kMTP)
                    {
                        bool const explicitTree = args.verifyTreeSize > 0 && args.draftTreeSize > 1;
                        int64_t const chainVerifySize = static_cast<int64_t>(args.draftStep) + 1;
                        ELLM_CHECK(explicitTree || chainVerifySize <= std::numeric_limits<int32_t>::max(),
                            "--draftStep is too large to derive MTP chain verifySize; pass a value no greater than "
                                + std::to_string(std::numeric_limits<int32_t>::max() - 1));
                        dc.draftingTopK = explicitTree ? args.draftTreeSize : 1;
                        dc.draftingStep = args.draftStep;
                        dc.verifySize = explicitTree ? args.verifyTreeSize : static_cast<int32_t>(chainVerifySize);
                    }
                    else
                    {
                        dc.draftingTopK = std::max(args.draftTreeSize, 1);
                        dc.draftingStep = std::max(args.draftStep, 1);
                        dc.verifySize = std::max(args.verifyTreeSize, 1);
                    }
                }
                draftingConfig = dc;
            }

            // --- Parse configs and create DeploymentConfig ---
            deployment = rt::createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to parse engine configuration: %s", e.what());
            return EXIT_FAILURE;
        }
        if (deployment.base.isDiffusionBackbone && args.mode == BenchMode::kDECODE)
        {
            LOG_ERROR(
                "llm_bench --mode decode is not a valid DiffusionGemma serving benchmark because DiffusionGemma decode "
                "uses a denoise/sample/commit runtime state machine. Use llm_inference or add a dedicated "
                "diffusion_decode bench mode for end-to-end DG decode timing.");
            return EXIT_FAILURE;
        }

        if (args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT)
        {
            ELLM_CHECK(deployment.draft.has_value(), "spec_draft_accept requires a draft engine");
            ELLM_CHECK(args.acceptLen <= deployment.draft->maxDraftTreeSize,
                "spec_draft_accept acceptLen (" + std::to_string(args.acceptLen)
                    + ") exceeds the draft engine maxDraftTreeSize ("
                    + std::to_string(deployment.draft->maxDraftTreeSize) + ")");

            int64_t const decodeStepsForCapacity
                = args.osl > 1 ? (static_cast<int64_t>(args.osl) - 1 + args.acceptRate - 1) / args.acceptRate : 0;
            int64_t const requiredDraftCacheLength = static_cast<int64_t>(args.pastKVLen) + args.acceptLen
                + decodeStepsForCapacity * static_cast<int64_t>(args.acceptRate);
            ELLM_CHECK(requiredDraftCacheLength <= static_cast<int64_t>(deployment.draft->maxKVCacheCapacity),
                "spec_draft_accept requires pastKVLen + acceptLen + decodeSteps * acceptRate ("
                    + std::to_string(requiredDraftCacheLength) + ") to fit the draft engine maxKVCacheCapacity ("
                    + std::to_string(deployment.draft->maxKVCacheCapacity) + ")");
        }

        // --- DFlash: engine-config consistency + CLI-default resolution ---
        // Runs before EngineExecutor creation so the DFlash validation guard rails
        // reject mismatched deployments early, and so args.blockSize/draftDeltaLen/
        // candidateTopK reflect the values baked into the engine for the mode
        // dispatch below.
        if (isDFlashMode(args.mode))
        {
            if (deployment.specDecodeMode() != rt::SpecDecodeMode::kDFlash)
            {
                LOG_ERROR("DFlash benchmark mode '%s' requires engine configs with spec_decode_type=dflash",
                    modeToString(args.mode).c_str());
                return EXIT_FAILURE;
            }

            ELLM_CHECK(deployment.specConfig.has_value(),
                "DFlash mode requires a spec-decode deployment (base engine must be DFlash type)");
            if (args.blockSize <= 0)
                args.blockSize = deployment.specConfig->dflashBlockSize;
            // draftDeltaLen models accepted tokens per round (excluding the first): it comes from
            // acceptRate in production. Cap at blockSize because DFlash draft proposal cannot consume
            // more than one block per round.
            ELLM_CHECK(args.acceptRate > 0, "--acceptRate must be positive for DFlash modes");
            args.draftDeltaLen = std::min(args.acceptRate, args.blockSize);
            if (args.candidateTopK <= 0)
                args.candidateTopK = deployment.specConfig->draftingTopK; // whatever the factory resolved

            ELLM_CHECK(deployment.specConfig->draftingTopK > 0
                    && (deployment.specConfig->draftingTopK == 1
                        || deployment.specConfig->draftingTopK < deployment.specConfig->verifySize),
                "DFlash: candidateTopK (" + std::to_string(deployment.specConfig->draftingTopK)
                    + ") must be > 0 and (== 1 for linear mode, or < verifySize ("
                    + std::to_string(deployment.specConfig->verifySize) + ") for DDTree mode)");

            if (args.mode == BenchMode::kDFLASH_DDTREE_BUILD && deployment.draft.has_value()
                && deployment.draft->reducedVocabSize > 0)
            {
                ELLM_CHECK(false,
                    "DFlash DDTree build with reduced-vocab draft (vocab="
                        + std::to_string(deployment.draft->reducedVocabSize)
                        + ") not supported by llm_bench yet. Use a full-vocab engine or extend "
                          "llm_bench to load draft_vocab_map.safetensors and thread it into DDTreeBuildParams.");
            }

            if (args.mode == BenchMode::kDFLASH_VERIFY)
            {
                ELLM_CHECK(args.verifyTreeSize <= deployment.base.maxVerifyTreeSize,
                    "DFlash verify tree size (" + std::to_string(args.verifyTreeSize)
                        + ") exceeds the base engine maximum verify tree size ("
                        + std::to_string(deployment.base.maxVerifyTreeSize) + ")");
                int64_t const requiredBaseCacheLength
                    = static_cast<int64_t>(args.pastKVLen) + static_cast<int64_t>(args.verifyTreeSize);
                ELLM_CHECK(requiredBaseCacheLength <= static_cast<int64_t>(deployment.base.maxKVCacheCapacity),
                    "DFlash verify requires pastKVLen + verifyTreeSize (" + std::to_string(requiredBaseCacheLength)
                        + ") to fit the base KV cache maximum sequence length ("
                        + std::to_string(deployment.base.maxKVCacheCapacity) + ")");

                // The production DFlash configuration normalizes greedy verify
                // size to blockSize.  This isolated base-engine benchmark must
                // retain the requested, profile-valid shape so the shared verify
                // tensors are allocated for it (for example, verifyTreeSize=64).
                deployment.specConfig->verifySize = args.verifyTreeSize;
            }

            LOG_INFO("DFlash config: blockSize=%d draftDeltaLen=%d candidateTopK=%d verifySize=%d", args.blockSize,
                args.draftDeltaLen, args.candidateTopK, deployment.specConfig->verifySize);
        }

        // Everything from EngineExecutor creation through PipelineIO filling only
        // makes sense when we actually run a TRT engine — DDTree build times a
        // raw CUDA kernel with its own scratch (see DDTreeBuildScratch below).
        if (needsExecutor(args.mode))
        {
            // --- Determine which engine this mode operates on ---
            bool const useDraftEngine = isDraftEngineMode(args.mode);
            rt::LLMEngineConfig const& activeCfg = useDraftEngine ? *deployment.draft : deployment.base;

            // --- Create EngineExecutor ---
            std::filesystem::path enginePath;
            if (useDraftEngine)
            {
                enginePath = dir / "spec_draft.engine";
            }
            else if (deployment.base.isDiffusionBackbone)
            {
                enginePath = dir / "dllm.engine";
            }
            else
            {
                enginePath = dir / "llm.engine";
                if (!std::filesystem::exists(enginePath))
                {
                    enginePath = dir / "spec_base.engine";
                }
            }

            try
            {
                if (useDraftEngine)
                {
                    executor = rt::EngineExecutor::createForDraft(enginePath, deployment);
                }
                else
                {
                    std::optional<int32_t> specDecodeBaseOutputHiddenDim;
                    if (deployment.specConfig.has_value())
                    {
                        specDecodeBaseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
                    }
                    executor
                        = rt::EngineExecutor::createForLLM(enginePath, deployment.base, specDecodeBaseOutputHiddenDim);
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Failed to create EngineExecutor: %s", e.what());
                return EXIT_FAILURE;
            }

            rt::validateAgainstEngine(activeCfg, *executor, useDraftEngine ? "draft" : "base");
            useRaggedBindings = executor->hasIOTensor(binding_names::kQueryStartOffsets);

            if (layerProfiler::LayerProfiler::getInstance().isEnabled())
            {
                executor->setProfiler(&layerProfiler::LayerProfiler::getInstance());
            }

            LOG_INFO("EngineExecutor loaded from %s", enginePath.c_str());

            // --- Create SharedResources, PipelineIO, TensorMap ---
            // Use spec-decode factories when the deployment has a draft config, even if
            // the bench mode is prefill/decode. A spec-decode base engine has extra
            // bindings (attention_mask, attention_pos_id) that require PipelineIO to
            // allocate packedAttentionMask and specDecodePositionIds.
            bool const useSpecDecodeResources = deployment.draft.has_value();
            int32_t const maxBatch = deployment.maxRuntimeBatchSize();
            std::unordered_map<std::string, std::string> emptyLoraMap;

            if (useSpecDecodeResources)
            {
                resources = rt::SharedResources::createForSpecDecode(deployment, maxBatch, emptyLoraMap, stream);
                // The bench drives no Talker, but TensorRegistry::bindAll fails on any
                // engine I/O missing from the map, so the buffer must follow the engine.
                bool const hasTreeParentIds = executor->hasIOTensor(binding_names::kTreeParentIds);
                bool const hasTreeDepths = executor->hasIOTensor(binding_names::kTreeDepths);
                ELLM_CHECK(hasTreeParentIds == hasTreeDepths,
                    "Speculative engine must expose both tree_parent_ids and tree_depths, or neither.");
                io = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForSpecDecode(deployment, maxBatch, stream,
                    executor->hasIOTensor(binding_names::kAcceptHiddenStates), hasTreeParentIds && hasTreeDepths));
            }
            else
            {
                resources = rt::SharedResources::createForLLM(deployment.base, emptyLoraMap, stream);
                io = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(deployment.base, stream));
            }

            // Build TensorMap: kvCacheIndex=0 for base, 1 for draft
            if (useDraftEngine)
            {
                rt::buildTensorMapForSpecDecodeDraft(tensorMap, *io, *resources, *deployment.draft);
            }
            else
            {
                if (deployment.base.isDiffusionBackbone)
                {
                    rt::buildTensorMapForDiffusionBackbone(
                        tensorMap, *io, *resources, deployment.base, /*kvCacheIndex=*/0);
                }
                else
                {
                    rt::buildTensorMap(tensorMap, *io, *resources, deployment.base, /*kvCacheIndex=*/0);
                }
            }

            if (!useDraftEngine && deployment.base.isDiffusionBackbone && deployment.base.diffusionUnifiedConditioning)
            {
                int32_t maxConditioningSeqLen = deployment.base.maxSupportedInputLength;
                if (deployment.base.diffusionCanvasLength > maxConditioningSeqLen)
                {
                    maxConditioningSeqLen = deployment.base.diffusionCanvasLength;
                }
                diffusionCanvasIds = rt::Tensor({maxBatch, maxConditioningSeqLen}, rt::DeviceType::kGPU,
                    nvinfer1::DataType::kINT32, "llm_bench::diffusionCanvasIds");
                diffusionPrevSelfConditioningEmbeds
                    = rt::Tensor({maxBatch, maxConditioningSeqLen, deployment.base.hiddenSize}, rt::DeviceType::kGPU,
                        nvinfer1::DataType::kHALF, "llm_bench::diffusionPrevSelfConditioningEmbeds");
                diffusionNextSelfConditioningEmbeds
                    = rt::Tensor({maxBatch, maxConditioningSeqLen, deployment.base.hiddenSize}, rt::DeviceType::kGPU,
                        nvinfer1::DataType::kHALF, "llm_bench::diffusionNextSelfConditioningEmbeds");
                diffusionSelfConditioningTemperature = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
                    "llm_bench::diffusionSelfConditioningTemperature");

                CUDA_CHECK(cudaMemsetAsync(
                    diffusionCanvasIds.rawPointer(), 0, diffusionCanvasIds.getMemoryCapacity(), stream));
                CUDA_CHECK(cudaMemsetAsync(diffusionPrevSelfConditioningEmbeds.rawPointer(), 0,
                    diffusionPrevSelfConditioningEmbeds.getMemoryCapacity(), stream));
                CUDA_CHECK(cudaMemsetAsync(diffusionNextSelfConditioningEmbeds.rawPointer(), 0,
                    diffusionNextSelfConditioningEmbeds.getMemoryCapacity(), stream));
                float const diffusionTemperatureInit = 1.0F;
                CUDA_CHECK(cudaMemcpyAsync(diffusionSelfConditioningTemperature.rawPointer(), &diffusionTemperatureInit,
                    sizeof(float), cudaMemcpyHostToDevice, stream));

                tensorMap.set(binding_names::kCanvasIds, diffusionCanvasIds);
                tensorMap.set(binding_names::kPrevSelfConditioningEmbeds, diffusionPrevSelfConditioningEmbeds);
                tensorMap.set(binding_names::kNextSelfConditioningEmbeds, diffusionNextSelfConditioningEmbeds);
                tensorMap.set(binding_names::kSelfConditioningTemperature, diffusionSelfConditioningTemperature);
            }

            // --- Load externalized model weights ---
            std::filesystem::path const& activeConfigPath = useDraftEngine ? *draftConfigPath : baseConfigPath;
            resources->externalWeightManager->load(dir, activeConfigPath, stream, args.checkpointDir);
            resources->externalWeightManager->validateAgainstEngine(*executor, useDraftEngine ? "draft" : "base");
            resources->externalWeightManager->registerTensorMapEntries(tensorMap);

            // --- Context memory ---
            int64_t memSize = executor->getRequiredContextMemorySize();
            contextMemory
                = rt::Tensor(rt::Coords{memSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "context_memory");
            executor->setContextMemory(contextMemory);

            // --- StepPreparer (for prefill/decode metadata) ---
            stepPreparer = std::make_unique<rt::StepPreparer>(activeCfg);

            // --- DeepstackBinding (if applicable, base engine only) ---
            if (!useDraftEngine && !deployment.base.isDiffusionBackbone && deployment.base.numDeepstackFeatures > 0)
            {
                deepstack = std::make_unique<rt::DeepstackBinding>(io->deepstackEmbeds, resources->zeroBuffer);
            }

            // --- Log engine config ---
            LOG_INFO("Engine config:\n%s", rt::formatEngineConfig(activeCfg).c_str());

            // --- Standalone engine for layer metadata extraction ---
            // Only needed by the --extractLayerInfo path (used at Phase 4 below).
            // Skip the load otherwise: a second engine deserialization holds an
            // additional copy of the engine weights for the process lifetime and
            // can double peak GPU memory (~15 GB for Qwen3-8B), OOM'ing on
            // Jetson UMA and other memory-constrained targets.
            if (args.extractLayerInfo.any())
            {
                standaloneEngine = loadStandaloneEngine(enginePath);
            }

            // --- Fill PipelineIO tensors with random data ---
            nvinfer1::DataType const dtype = nvinfer1::DataType::kHALF;
            fillRandomData(io->inputsEmbeds, -1.0f, 1.0f, dtype, args.seed);

            if (!io->baseHiddenStates.isEmpty())
            {
                fillRandomData(io->baseHiddenStates, -1.0f, 1.0f, dtype, args.seed);
            }
            if (!io->draftHiddenStatesIn.isEmpty())
            {
                CUDA_CHECK(cudaMemsetAsync(
                    io->draftHiddenStatesIn.rawPointer(), 0, io->draftHiddenStatesIn.getMemoryCapacity(), stream));
            }
        }
    }

    logBenchConfig(args.toOutputParams(), imageTokens);

    // ===== Phase 2: Define step/reset/capture lambdas =====
    std::function<void()> resetState;
    std::function<bool()> step;
    std::string modeName;

    rt::Tensor reuseKVCacheLengths;
    if (args.mode != BenchMode::kVISUAL && needsExecutor(args.mode))
    {
        reuseKVCacheLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "reuse_kv_lengths");
    }

    std::vector<int32_t> reuseKVLenVec;
    std::vector<int32_t> pastKVLenVec;

    std::function<void(int32_t)> postStep = [](int32_t) {};
    std::function<bool()> captureGraph = []() { return false; };
    std::function<void()> prepareDraftAcceptInputs = []() {};
    rt::Tensor draftAcceptLengths;
    int32_t decodeSteps = 1;
    bool useSequentialE2E = false;

    int32_t const B = args.batchSize;
    bool const useDraftEngine = isDraftEngineMode(args.mode);
    int32_t const kvCacheIndex = useDraftEngine ? 1 : 0;
    rt::LLMEngineConfig const* activeCfg
        = needsExecutor(args.mode) ? &(useDraftEngine ? *deployment.draft : deployment.base) : nullptr;
    std::unique_ptr<rt::RaggedBatchBuilder> raggedBatchBuilder;
    rt::RaggedExecutionBatch raggedBatch;
    rt::ScheduledStep raggedStep;
    std::vector<int32_t> raggedTokens;
    int32_t raggedPastLength{0};
    if (useRaggedBindings)
    {
        rt::RaggedEngineContract const contract{rt::TokenLayoutBackend::kEntryPaddedCompatibility,
            activeCfg->maxNumSequences, activeCfg->maxQueryLength, activeCfg->maxPhysicalTokens,
            activeCfg->recurrentPoolRows, /*mixedStepSupported=*/false};
        raggedBatchBuilder = std::make_unique<rt::RaggedBatchBuilder>(contract);
        raggedBatchBuilder->reserve(raggedBatch);
    }
    auto prepareRaggedBenchBindings = [&](rt::SequenceWork work, int32_t queryLength) {
        check::check(useRaggedBindings && raggedBatchBuilder != nullptr, "Ragged benchmark state is not initialized");
        check::check(
            B > 0 && B <= activeCfg->maxNumSequences && queryLength > 0 && queryLength <= activeCfg->maxQueryLength,
            "Invalid ragged benchmark shape");
        int64_t const physicalTokens = static_cast<int64_t>(B) * queryLength;
        check::check(physicalTokens <= activeCfg->maxPhysicalTokens, "Ragged benchmark token count exceeds profile");
        raggedTokens.assign(static_cast<size_t>(physicalTokens), 0);
        raggedStep.id++;
        raggedStep.sequences.resize(static_cast<size_t>(B));
        for (int32_t batchIdx = 0; batchIdx < B; ++batchIdx)
        {
            raggedStep.sequences[static_cast<size_t>(batchIdx)] = rt::ScheduledSequence{
                rt::SequenceIdentity{static_cast<rt::RequestId>(batchIdx) + 1, rt::ResidentRef{batchIdx, 1}}, work,
                queryLength, raggedPastLength,
                rt::HostTokenRange{raggedTokens.data(), static_cast<int32_t>(raggedTokens.size()),
                    batchIdx * queryLength, queryLength},
                true};
        }
        raggedBatchBuilder->buildInto(raggedStep, raggedBatch);
        rt::prepareRaggedExecutionBindings(*io, *resources, *activeCfg, raggedBatch, kvCacheIndex, stream);
        check::check(io->inputsEmbeds.reshape({raggedBatch.shape.physicalTokens, activeCfg->hiddenSize}),
            "Ragged benchmark inputsEmbeds reshape failed");
        check::check(io->outputLogits.reshape({raggedBatch.shape.numLogits, activeCfg->outputVocabSize}),
            "Ragged benchmark outputLogits reshape failed");
    };
    // DDTree build owns its own scratch; DFlash draft modes share one scratch struct
    // whose Tensors must outlive the tensorMap (which stores raw pointers into them).
    std::unique_ptr<DDTreeBuildScratch> ddtreeScratch;
    DFlashDraftBenchScratch dflashDraftScratch;

    if (args.mode == BenchMode::kVISUAL)
    {
        modeName = "Visual Encoder";
        step = [&]() { return visualRunner->infer(stream); };
        resetState = []() {};
    }
    else if (args.mode == BenchMode::kPREFILL)
    {
        modeName = "Prefill";
        LOG_INFO("Prefill mode: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);

        // Reshape inputsEmbeds for this bench config
        check::check(io->inputsEmbeds.reshape(useRaggedBindings ? rt::Coords{B * args.inputLen, activeCfg->hiddenSize}
                                                                : rt::Coords{B, args.inputLen, activeCfg->hiddenSize}),
            "inputsEmbeds reshape failed");
        if (deployment.base.isDiffusionBackbone)
        {
            check::check(
                io->outputLogits.reshape({B, 1, deployment.base.outputVocabSize}), "outputLogits reshape failed");
            check::check(io->phaseIsEncoder.reshape({B}), "phaseIsEncoder reshape failed");
            check::check(io->hostPhaseIsEncoder.reshape({B}), "hostPhaseIsEncoder reshape failed");
            int32_t* hostPhase = io->hostPhaseIsEncoder.dataPointer<int32_t>();
            std::fill(hostPhase, hostPhase + B, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                io->phaseIsEncoder.rawPointer(), hostPhase, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            if (deployment.base.diffusionUnifiedConditioning)
            {
                check::check(diffusionCanvasIds.reshape({B, args.inputLen}), "diffusionCanvasIds reshape failed");
                check::check(
                    diffusionPrevSelfConditioningEmbeds.reshape({B, args.inputLen, deployment.base.hiddenSize}),
                    "diffusionPrevSelfConditioningEmbeds reshape failed");
                check::check(diffusionNextSelfConditioningEmbeds.reshape({B, 1, deployment.base.hiddenSize}),
                    "diffusionNextSelfConditioningEmbeds reshape failed");
            }
        }

        // Set context lengths on PipelineIO (host side, for StepPreparer)
        int32_t const contextLen = args.reuseKVLen + args.inputLen;
        int32_t* hostCtx = io->hostContextLengths.dataPointer<int32_t>();
        for (int32_t i = 0; i < B; ++i)
        {
            hostCtx[i] = contextLen;
        }

        reuseKVLenVec.assign(B, args.reuseKVLen);

        if (deepstack)
        {
            deepstack->useRealFeatures(tensorMap);
        }

        bool const kvCacheAllEmpty = (args.reuseKVLen == 0);
        rt::ExecutionPhase const phase = deployment.base.isDiffusionBackbone
            ? rt::ExecutionPhase::kDiffusionCommit
            : (kvCacheAllEmpty ? rt::ExecutionPhase::kContextPrefill : rt::ExecutionPhase::kContextChunk);
        auto const dims = deployment.base.prefillDims(B, args.inputLen, phase);

        resetState = [&]() {
            raggedPastLength = args.reuseKVLen;
            std::memcpy(reuseKVCacheLengths.rawPointer(), reuseKVLenVec.data(), reuseKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (useRaggedBindings)
            {
                prepareRaggedBenchBindings(rt::SequenceWork::kContext, args.inputLen);
            }
            stepPreparer->prepare(
                rt::InferencePhase::kPrefill, B, *resources->cacheManagers[kvCacheIndex], *io, stream);
            if (!executor->prepare(kPrefillProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
    }
    else if (args.mode == BenchMode::kDECODE)
    {
        modeName = "Decode";
        LOG_INFO("Decode mode: PastKVLen=%d", args.pastKVLen);

        int32_t const osl = args.osl;
        int32_t const decodeTokens = (osl > 1) ? (osl - 1) : 1;
        LOG_INFO("OSL=%d: will run %d decode steps for E2E timing", osl, decodeTokens);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        check::check(io->inputsEmbeds.reshape(useRaggedBindings ? rt::Coords{B, activeCfg->hiddenSize}
                                                                : rt::Coords{B, 1, activeCfg->hiddenSize}),
            "inputsEmbeds reshape failed");

        pastKVLenVec.assign(B, args.pastKVLen);

        auto const dims = deployment.base.decodeDims(B);

        resetState = [&]() {
            raggedPastLength = args.pastKVLen;
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (useRaggedBindings)
            {
                prepareRaggedBenchBindings(rt::SequenceWork::kDecode, /*queryLength=*/1);
            }
            stepPreparer->prepare(rt::InferencePhase::kDecode, B, *resources->cacheManagers[kvCacheIndex], *io, stream);
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (useRaggedBindings)
            {
                prepareRaggedBenchBindings(rt::SequenceWork::kDecode, /*queryLength=*/1);
            }
            stepPreparer->prepare(rt::InferencePhase::kDecode, B, *resources->cacheManagers[kvCacheIndex], *io, stream);
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };

        if (osl > 1)
        {
            useSequentialE2E = true;
            decodeSteps = decodeTokens;
            if (useRaggedBindings)
            {
                postStep = [&](int32_t) {
                    resources->cacheManagers[kvCacheIndex]->commitSequenceLength(/*increment=*/1, stream);
                    ++raggedPastLength;
                };
            }
        }
    }
    else if (args.mode == BenchMode::kEAGLE_VERIFY)
    {
        modeName = "Spec Verify";
        LOG_INFO("Spec Verify mode: VerifyTreeSize=%d, PastKVLen=%d", args.verifyTreeSize, args.pastKVLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        pastKVLenVec.assign(B, args.pastKVLen);

        check::check(io->inputsEmbeds.reshape({B, args.verifyTreeSize, deployment.base.hiddenSize}),
            "inputsEmbeds reshape failed");

        // selectTokenIndices: for verify, all positions are selected
        check::check(io->selectTokenIndices.reshape({B, args.verifyTreeSize}), "selectTokenIndices reshape failed");
        CUDA_CHECK(cudaMemsetAsync(
            io->selectTokenIndices.rawPointer(), 0, io->selectTokenIndices.getMemoryCapacity(), stream));

        // contextLengths: dummy values (past KV len + verify tree size)
        check::check(io->contextLengths.reshape({B}), "contextLengths reshape failed");
        {
            std::vector<int32_t> ctxVec(B, args.pastKVLen + args.verifyTreeSize);
            CUDA_CHECK(cudaMemcpyAsync(
                io->contextLengths.rawPointer(), ctxVec.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        if (deepstack)
        {
            deepstack->useZeroTarget(tensorMap);
        }

        auto const dims = deployment.base.specVerifyDims(B, args.verifyTreeSize);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };

        if (args.osl > 1)
        {
            useSequentialE2E = true;
            decodeSteps = (args.osl - 1 + args.acceptRate - 1) / args.acceptRate;
            postStep = [&](int32_t) {
                resources->cacheManagers[kvCacheIndex]->commitSequenceLength(args.acceptRate, stream);
            };
        }
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL)
    {
        modeName = "Spec Draft";
        int64_t proposalSize = args.draftTreeSize;
        int64_t selectLen = args.draftTreeSize;
        if (deployment.specDecodeMode() == rt::SpecDecodeMode::kMTP)
        {
            ELLM_CHECK(deployment.specConfig.has_value(), "MTP draft proposal requires drafting configuration");
            int32_t const effectiveDraftTopK
                = deployment.specConfig->draftingTopK > 1 ? 1 : deployment.specConfig->draftingTopK;
            proposalSize
                = static_cast<int64_t>(deployment.specConfig->draftingStep) * static_cast<int64_t>(effectiveDraftTopK);
            selectLen = effectiveDraftTopK;
        }
        ELLM_CHECK(proposalSize > 0 && selectLen > 0 && proposalSize <= std::numeric_limits<int32_t>::max()
                && static_cast<int64_t>(args.pastKVLen) + proposalSize <= std::numeric_limits<int32_t>::max(),
            "spec_draft_proposal production geometry exceeds the supported int32 tensor/context range");
        LOG_INFO("Spec Draft mode: RequestedDraftTreeSize=%d, ProposalSize=%ld, SelectLen=%ld, PastKVLen=%d",
            args.draftTreeSize, proposalSize, selectLen, args.pastKVLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        pastKVLenVec.assign(B, args.pastKVLen);

        int32_t const draftHiddenSize = deployment.draft->hiddenSize;
        check::check(io->inputsEmbeds.reshape({B, proposalSize, draftHiddenSize}), "inputsEmbeds reshape failed");

        check::check(io->selectTokenIndices.reshape({B, selectLen}), "selectTokenIndices reshape failed");
        CUDA_CHECK(cudaMemsetAsync(
            io->selectTokenIndices.rawPointer(), 0, io->selectTokenIndices.getMemoryCapacity(), stream));

        // contextLengths: dummy values
        check::check(io->contextLengths.reshape({B}), "contextLengths reshape failed");
        {
            std::vector<int32_t> ctxVec(B, static_cast<int32_t>(args.pastKVLen + proposalSize));
            CUDA_CHECK(cudaMemcpyAsync(
                io->contextLengths.rawPointer(), ctxVec.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        auto const dims = deployment.draft->proposalDims(B, proposalSize, selectLen);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };

        if (args.osl > 1)
        {
            useSequentialE2E = true;
            int32_t eagleIterations = (args.osl - 1 + args.acceptRate - 1) / args.acceptRate;
            decodeSteps = eagleIterations * (args.draftStep - 1);
            int32_t draftStepsPerIter = args.draftStep - 1;
            postStep = [&, draftStepsPerIter](int32_t t) {
                if ((t + 1) % draftStepsPerIter == 0)
                {
                    resources->cacheManagers[kvCacheIndex]->commitSequenceLength(args.acceptRate, stream);
                }
            };
        }
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT)
    {
        // One draft-engine execute advances state across the accepted token span.
        int32_t const acceptLen = args.acceptLen;
        modeName = "Spec Draft Accept";
        LOG_INFO("Spec Draft Accept mode: AcceptLen=%d, DraftStep=%d, PastKVLen=%d", acceptLen, args.draftStep,
            args.pastKVLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        pastKVLenVec.assign(B, args.pastKVLen);

        int32_t const draftHiddenSize = deployment.draft->hiddenSize;
        check::check(io->inputsEmbeds.reshape({B, acceptLen, draftHiddenSize}), "inputsEmbeds reshape failed");

        // Mirror the production accept pass metadata. For one-pass component
        // timing, every batch entry accepts the whole engine input. Sequential
        // timing advances by acceptRate, while acceptLen remains the fixed
        // engine input shape and upper bound.
        check::check(io->packedAttentionMask.reshape({B, acceptLen, static_cast<int64_t>(divUp(acceptLen, 32))}),
            "packedAttentionMask reshape failed");
        check::check(io->specDecodePositionIds.reshape({B, acceptLen}), "specDecodePositionIds reshape failed");
        check::check(io->selectTokenIndices.reshape({B, 1}), "selectTokenIndices reshape failed");
        check::check(io->contextLengths.reshape({B}), "contextLengths reshape failed");
        draftAcceptLengths = rt::Tensor({B}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "draft_accept_lengths");
        int32_t const acceptedTokensPerPass = args.osl > 1 ? args.acceptRate : acceptLen;
        fillInt32(draftAcceptLengths, acceptedTokensPerPass);
        prepareDraftAcceptInputs = [&]() {
            kernel::prepareEagleAcceptDecodeTokenInputs(resources->cacheManagers[kvCacheIndex]->getKVCacheLengths(),
                draftAcceptLengths, io->packedAttentionMask, io->specDecodePositionIds, io->selectTokenIndices,
                io->contextLengths, stream);
        };

        auto const dims = deployment.draft->acceptDims(B, acceptLen);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
            prepareDraftAcceptInputs();
        };
        step = [&, dims]() {
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };

        if (args.osl > 1)
        {
            useSequentialE2E = true;
            // One accept pass runs per speculative-decoding iteration.
            decodeSteps = (args.osl - 1 + args.acceptRate - 1) / args.acceptRate;
            postStep = [&](int32_t) {
                resources->cacheManagers[kvCacheIndex]->commitSequenceLength(args.acceptRate, stream);
                prepareDraftAcceptInputs();
            };
        }
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        modeName = "Spec Draft Prefill";
        LOG_INFO("Spec Draft Prefill mode: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);

        int32_t const draftHiddenSize = deployment.draft->hiddenSize;
        check::check(io->inputsEmbeds.reshape({B, args.inputLen, draftHiddenSize}), "inputsEmbeds reshape failed");

        // Set context lengths on PipelineIO (host side, for StepPreparer)
        int32_t const contextLen = args.reuseKVLen + args.inputLen;
        int32_t* hostCtx = io->hostContextLengths.dataPointer<int32_t>();
        for (int32_t i = 0; i < B; ++i)
        {
            hostCtx[i] = contextLen;
        }

        reuseKVLenVec.assign(B, args.reuseKVLen);

        bool const kvCacheAllEmpty = (args.reuseKVLen == 0);
        rt::ExecutionPhase const phase
            = kvCacheAllEmpty ? rt::ExecutionPhase::kContextPrefill : rt::ExecutionPhase::kContextChunk;
        auto const dims = deployment.draft->prefillDims(B, args.inputLen, phase);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), reuseKVLenVec.data(), reuseKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            stepPreparer->prepare(
                rt::InferencePhase::kPrefill, B, *resources->cacheManagers[kvCacheIndex], *io, stream);
            if (!executor->prepare(kPrefillProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
    }
    else if (args.mode == BenchMode::kDFLASH_DRAFT_PROPOSAL)
    {
        modeName = "DFlash Draft Proposal";
        LOG_INFO("DFlash Draft Proposal mode: BlockSize=%d, DraftDeltaLen=%d, PastKVLen=%d", args.blockSize,
            args.draftDeltaLen, args.pastKVLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        pastKVLenVec.assign(B, args.pastKVLen);

        DFlashDraftBenchParams const draftParams{/*.batchSize=*/B, /*.blockSize=*/args.blockSize,
            /*.deltaLen=*/args.draftDeltaLen, /*.pastKVLen=*/args.pastKVLen, /*.seed=*/args.seed};
        buildDFlashDraftTensorMap(
            deployment, draftParams, *resources->cacheManagers[kvCacheIndex], *io, dflashDraftScratch, tensorMap);
        fillDFlashDraftInputs(deployment, *io, *resources->cacheManagers[kvCacheIndex], *resources, dflashDraftScratch,
            draftParams, stream);

        auto const dims = deployment.draft->proposalDims(B, args.blockSize, args.draftDeltaLen);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };
    }
    else if (args.mode == BenchMode::kDFLASH_DRAFT_FIRST_ROUND)
    {
        modeName = "DFlash Draft First Round";
        LOG_INFO("DFlash Draft First Round mode: BlockSize=%d, InputLen=%d", args.blockSize, args.inputLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        pastKVLenVec.assign(B, 0);

        ELLM_CHECK(args.inputLen <= deployment.draft->maxSupportedInputLength,
            "DFlash draft first round --inputLen (" + std::to_string(args.inputLen)
                + ") exceeds the draft engine maximum supported input length ("
                + std::to_string(deployment.draft->maxSupportedInputLength) + ")");
        auto const& draftCacheConfig = resources->cacheManagers[kvCacheIndex]->getKVCacheManager().getConfig();
        int64_t const requiredDraftCacheLength
            = static_cast<int64_t>(args.inputLen) + static_cast<int64_t>(args.blockSize);
        ELLM_CHECK(requiredDraftCacheLength <= static_cast<int64_t>(draftCacheConfig.maxSequenceLength),
            "DFlash draft first round requires inputLen + blockSize (" + std::to_string(requiredDraftCacheLength)
                + ") to fit the draft KV cache maximum sequence length ("
                + std::to_string(draftCacheConfig.maxSequenceLength) + ")");

        DFlashDraftBenchParams const draftParams{/*.batchSize=*/B, /*.blockSize=*/args.blockSize,
            /*.deltaLen=*/args.inputLen, /*.pastKVLen=*/0, /*.seed=*/args.seed};
        buildDFlashDraftTensorMap(
            deployment, draftParams, *resources->cacheManagers[kvCacheIndex], *io, dflashDraftScratch, tensorMap);
        fillDFlashDraftInputs(deployment, *io, *resources->cacheManagers[kvCacheIndex], *resources, dflashDraftScratch,
            draftParams, stream);

        auto const dims = deployment.draft->proposalDims(B, args.blockSize, args.inputLen);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&, dims]() {
            if (!executor->prepare(kPrefillProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kPrefillProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };
    }
    else if (args.mode == BenchMode::kDFLASH_VERIFY)
    {
        modeName = "DFlash Verify";
        LOG_INFO("DFlash Verify mode: VerifyTreeSize=%d, PastKVLen=%d", args.verifyTreeSize, args.pastKVLen);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        check::check(B > 0 && B <= activeCfg->maxNumSequences && args.verifyTreeSize <= activeCfg->maxQueryLength
                && B <= activeCfg->recurrentPoolRows,
            "DFlash verify shape exceeds engine capacity");
        int64_t const physicalTokens64 = static_cast<int64_t>(B) * args.verifyTreeSize;
        check::check(
            physicalTokens64 <= activeCfg->maxPhysicalTokens && physicalTokens64 <= std::numeric_limits<int32_t>::max(),
            "DFlash verify token count exceeds engine capacity");
        int32_t const physicalTokens = static_cast<int32_t>(physicalTokens64);
        pastKVLenVec.assign(B, args.pastKVLen);

        check::check(io->inputsEmbeds.reshape({B, args.verifyTreeSize, deployment.base.hiddenSize}),
            "inputsEmbeds reshape failed");
        fillRandomData(io->inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, args.seed);

        check::check(io->packedAttentionMask.reshape(
                         {B, args.verifyTreeSize, static_cast<int64_t>(divUp(args.verifyTreeSize, 32))}),
            "packedAttentionMask reshape failed");
        check::check(
            io->specDecodePositionIds.reshape({B, args.verifyTreeSize}), "specDecodePositionIds reshape failed");
        check::check(io->selectTokenIndices.reshape({B, args.verifyTreeSize}), "selectTokenIndices reshape failed");
        check::check(io->contextLengths.reshape({B}), "contextLengths reshape failed");

        if (deepstack)
        {
            deepstack->useZeroTarget(tensorMap);
        }

        auto const dims = deployment.base.specVerifyDims(B, args.verifyTreeSize);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            resources->cacheManagers[kvCacheIndex]->resetForNewSequences(reuseKVCacheLengths, stream);
            if (useRaggedBindings)
            {
                raggedBatch.shape = {B, physicalTokens, physicalTokens, args.verifyTreeSize, 0, 0, physicalTokens};
                raggedBatch.positions.resize(static_cast<size_t>(physicalTokens));
                raggedBatch.queryStartOffsets.resize(static_cast<size_t>(B) + 1);
                raggedBatch.queryLengths.assign(static_cast<size_t>(B), args.verifyTreeSize);
                raggedBatch.pastLengths.assign(static_cast<size_t>(B), args.pastKVLen);
                raggedBatch.attentionSequenceLengths.assign(
                    static_cast<size_t>(B), args.pastKVLen + args.verifyTreeSize);
                raggedBatch.stateIndices.resize(static_cast<size_t>(B));
                raggedBatch.logitsIndices.resize(static_cast<size_t>(physicalTokens));
                for (int32_t batchIdx = 0; batchIdx < B; ++batchIdx)
                {
                    int32_t const start = batchIdx * args.verifyTreeSize;
                    raggedBatch.queryStartOffsets[static_cast<size_t>(batchIdx)] = start;
                    raggedBatch.stateIndices[static_cast<size_t>(batchIdx)] = batchIdx;
                    for (int32_t tokenIdx = 0; tokenIdx < args.verifyTreeSize; ++tokenIdx)
                    {
                        int32_t const token = start + tokenIdx;
                        raggedBatch.positions[static_cast<size_t>(token)] = args.pastKVLen + tokenIdx;
                        raggedBatch.logitsIndices[static_cast<size_t>(token)] = token;
                    }
                }
                raggedBatch.queryStartOffsets[static_cast<size_t>(B)] = physicalTokens;
                rt::prepareRaggedExecutionBindings(*io, *resources, *activeCfg, raggedBatch, kvCacheIndex, stream);
                check::check(io->inputsEmbeds.reshape({physicalTokens, activeCfg->hiddenSize}),
                    "Ragged DFlash verify inputsEmbeds reshape failed");
                check::check(io->outputLogits.reshape({physicalTokens, activeCfg->outputVocabSize}),
                    "Ragged DFlash verify outputLogits reshape failed");
            }
            kernel::launchDFlashPrepareBaseVerifyInputs(
                resources->cacheManagers[kvCacheIndex]->getKVCacheLengths().dataPointer<int32_t>(), args.verifyTreeSize,
                io->packedAttentionMask.dataPointer<int32_t>(), io->specDecodePositionIds.dataPointer<int32_t>(),
                io->selectTokenIndices.dataPointer<int64_t>(), io->contextLengths.dataPointer<int32_t>(), B, stream);
            if (!io->specTreeParentIds.isEmpty())
            {
                check::check(io->specTreeParentIds.reshape({B, args.verifyTreeSize}), "Tensor reshape failed");
                check::check(io->specTreeDepths.reshape({B, args.verifyTreeSize}), "Tensor reshape failed");
                kernel::launchDFlashBuildLinearTreeMetadata(io->specTreeParentIds.dataPointer<int32_t>(),
                    io->specTreeDepths.dataPointer<int32_t>(), B, args.verifyTreeSize, stream);
            }
        };

        step = [&, dims]() {
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->execute(stream);
        };
        captureGraph = [&, dims]() {
            resetState();
            if (!executor->prepare(kDecodeProfile, dims, tensorMap, stream))
                return false;
            return executor->captureGraph(stream);
        };
    }
    else if (args.mode == BenchMode::kDFLASH_DDTREE_BUILD)
    {
        ELLM_CHECK(deployment.draft.has_value(), "DFlash DDTree build requires a draft engine configuration");
        ELLM_CHECK(args.batchSize <= deployment.maxRuntimeBatchSize(),
            "DFlash DDTree batchSize (" + std::to_string(args.batchSize)
                + ") exceeds the deployment maximum batch size (" + std::to_string(deployment.maxRuntimeBatchSize())
                + ")");
        ELLM_CHECK(args.blockSize > 1, "DFlash DDTree build requires blockSize > 1");
        ELLM_CHECK(args.verifyTreeSize <= deployment.base.maxVerifyTreeSize,
            "DFlash DDTree verify size (" + std::to_string(args.verifyTreeSize)
                + ") exceeds the base configuration maximum (" + std::to_string(deployment.base.maxVerifyTreeSize)
                + ")");
        ELLM_CHECK(args.candidateTopK > 0 && args.candidateTopK <= kernel::kDDTreeMaxCandidateTopK,
            "DFlash DDTree candidateTopK must be in [1, " + std::to_string(kernel::kDDTreeMaxCandidateTopK) + "]");
        ELLM_CHECK(args.candidateTopK < args.verifyTreeSize,
            "DFlash DDTree candidateTopK must be less than verifyTreeSize because the root consumes one node");

        rt::LLMEngineConfig const& draftCfg = *deployment.draft;
        int32_t const draftVocabSize = draftCfg.outputVocabSize;
        ELLM_CHECK(draftCfg.reducedVocabSize == 0 && draftVocabSize == draftCfg.vocabSize,
            "DFlash DDTree build requires a full-vocabulary draft because no draft vocabulary map is loaded");
        ELLM_CHECK(args.candidateTopK <= draftVocabSize,
            "DFlash DDTree candidateTopK must not exceed the draft vocabulary size");

        int64_t const requiredBaseCacheLength
            = static_cast<int64_t>(args.pastKVLen) + static_cast<int64_t>(args.verifyTreeSize);
        ELLM_CHECK(requiredBaseCacheLength <= static_cast<int64_t>(deployment.base.maxKVCacheCapacity),
            "DFlash DDTree requires pastKVLen + verifyTreeSize (" + std::to_string(requiredBaseCacheLength)
                + ") to fit the base KV cache maximum sequence length ("
                + std::to_string(deployment.base.maxKVCacheCapacity) + ")");

        modeName = "DFlash DDTree Build";
        LOG_INFO(
            "DFlash DDTree Build mode: VerifyTreeSize=%d, CandidateTopK=%d, BlockSize=%d, DraftVocabSize=%d, "
            "PastKVLen=%d",
            args.verifyTreeSize, args.candidateTopK, args.blockSize, draftVocabSize, args.pastKVLen);
        LOG_INFO("CUDA graph disabled; DDTree build runs as an uncaptured production kernel call");

        ddtreeScratch = std::make_unique<DDTreeBuildScratch>(
            allocateDDTreeBuildScratch(B, args.blockSize, draftVocabSize, args.verifyTreeSize, args.candidateTopK));
        fillRandomData(ddtreeScratch->draftLogits, -1.0F, 1.0F, nvinfer1::DataType::kFLOAT, args.seed);
        fillInt32(ddtreeScratch->lastAcceptedTokens, 0);
        fillInt32(ddtreeScratch->baseKVCacheLengths, args.pastKVLen);

        resetState = []() {};
        step = [&]() {
            DDTreeBuildScratch& scratch = *ddtreeScratch;
            kernel::DDTreeBuildParams const buildParams{
                {scratch.draftLogits, scratch.lastAcceptedTokens, scratch.baseKVCacheLengths, nullptr},
                {scratch.treeTokenIds, scratch.treeDepths, scratch.treeParentIds, scratch.treeNodeScores,
                    scratch.validCounts, scratch.verifyTokenIds, scratch.specDecodePositionIds,
                    scratch.packedAttentionMask, scratch.verifyTreeMask, scratch.contextLengths,
                    scratch.selectTokenIndices},
                args.candidateTopK, scratch.workspace.rawPointer(),
                static_cast<size_t>(scratch.workspace.getMemoryCapacity()), stream};
            // Drain any pre-existing error state so the post-launch check reflects
            // only this launch, not a stale error from earlier CUDA calls.
            (void) cudaGetLastError();
            kernel::ddtreeBuild(buildParams);
            cudaError_t const launchStatus = cudaPeekAtLastError();
            if (launchStatus != cudaSuccess)
            {
                LOG_ERROR("DFlash DDTree build launch failed: %s", cudaGetErrorString(launchStatus));
                return false;
            }
            // Force a sync to catch mid-execution kernel failures; cudaPeekAtLastError
            // above only surfaces launch-config errors, not in-flight faults.
            cudaError_t const execStatus = cudaStreamSynchronize(stream);
            if (execStatus != cudaSuccess)
            {
                LOG_ERROR("DFlash DDTree build execution failed: %s", cudaGetErrorString(execStatus));
                return false;
            }
            return true;
        };
    }

    // ===== Phase 3: Warmup =====
    if (runWarmupLoop(modeName, args.warmup, resetState, step, stream) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    // ===== Phase 4: Extract Layer Metadata =====
    if (args.extractLayerInfo.any())
    {
        layerMetadata = extractLayerMetadata(standaloneEngine.get(), args.extractLayerInfo);
        standaloneEngine.reset();
    }

    // ===== Phase 5: Layer Profiling =====
    if (!args.noProfile)
    {
        if (runLayerProfilingLoop(modeName, args.iterations, !args.outputDir.empty(), resetState, step, layerMetadata,
                timesPerIter, layerTimings, stream)
            != EXIT_SUCCESS)
        {
            return EXIT_FAILURE;
        }

        if (!args.outputDir.empty() && !layerTimings.empty())
        {
            auto outParams = args.toOutputParams();
            writeLayerInfoCsv(
                layerTimings, buildLayerCsvPath(args.outputDir, outParams), outParams, imageTokens, layerMetadata);
        }

        logResultsSummary(args.toOutputParams(), timesPerIter, e2eTimeMsResult, imageTokens);
        if (!args.outputDir.empty())
        {
            LOG_INFO("Output CSV files saved to: %s", args.outputDir.c_str());
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_SUCCESS;
    }

    // ===== Phase 6: E2E Timing =====
    int32_t e2eNumTokens = 1;
    if (args.mode == BenchMode::kVISUAL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Visual Encoder", args.iterations, resetState, step, stream);
        e2eNumTokens = 1;
    }
    else if (args.mode == BenchMode::kPREFILL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Prefill", args.iterations, resetState, step, stream);
        e2eNumTokens = args.inputLen;
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Spec Draft Prefill", args.iterations, resetState, step, stream);
        e2eNumTokens = args.inputLen;
    }
    else if (useSequentialE2E)
    {
        int32_t const sequentialNumTokens = args.mode == BenchMode::kEAGLE_DRAFT_ACCEPT ? args.osl - 1 : decodeSteps;
        e2eTimeMsResult = runSequentialE2ETiming(modeName, decodeSteps, sequentialNumTokens, resetState, step, postStep,
            !args.noCudaGraph, captureGraph, stream);
        e2eNumTokens = sequentialNumTokens;
    }
    else
    {
        e2eTimeMsResult = runRepeatedE2ETiming(
            modeName, args.iterations, resetState, step, stream, !args.noCudaGraph, captureGraph);
        e2eNumTokens = 1;
    }

    if (e2eTimeMsResult < 0)
    {
        return EXIT_FAILURE;
    }

    if (!args.outputDir.empty())
    {
        auto outParams = args.toOutputParams();
        writeE2ECsv(buildE2ECsvPath(args.outputDir, outParams), outParams, e2eTimeMsResult, e2eNumTokens, imageTokens);
    }

    // ===== Phase 7: Results Summary =====
    logResultsSummary(args.toOutputParams(), args.noProfile ? std::vector<KernelTimes>{} : timesPerIter,
        e2eTimeMsResult, imageTokens);

    if (!args.outputDir.empty())
    {
        LOG_INFO("Output CSV files saved to: %s", args.outputDir.c_str());
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
