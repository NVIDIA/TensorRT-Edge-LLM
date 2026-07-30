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

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/trtUtils.h"
#include "common/utf8.h"
#include "memoryMonitor.h"
#include "multimodal/code2WavRunner.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    SPEC_DECODE = 910,
    SPEC_DRAFT_TOP_K = 911,
    SPEC_DRAFT_STEP = 912,
    SPEC_VERIFY_SIZE = 913,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915,
    ENABLE_AUDIO_OUTPUT = 916,
    TALKER_ENGINE_DIR = 917,
    CODE2WAV_ENGINE_DIR = 918,
    OUTPUT_AUDIO_DIR = 919,
    ENABLE_THINKER_TALKER_STREAMING = 920,
    DFLASH_BLOCK_SIZE = 921,
    NUM_LOGPROBS = 922,
    ENABLE_CONTEXT_REUSE = 923,
    CONTEXT_CACHE_MAX_RECORDS = 924,
    CONTEXT_CACHE_RECURRENT_SNAPSHOT_POOL_BYTES = 925,
    CONTEXT_CACHE_PARTIAL_KV_SNAPSHOT_POOL_BYTES = 926,
    CONTEXT_CACHE_MEDIA_ARTIFACT_POOL_BYTES = 927,
    CONTEXT_CACHE_MAX_MEDIA_ARTIFACTS = 928,
    CONTEXT_CACHE_PREFILL_STATE_ONLY = 929
};

// Struct to hold speculative decoding arguments (used by both EAGLE and MTP)
struct SpecDecodeArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // For tree-based strategies this is the branching factor; for chain-style
    // strategies it is the number of candidates retained per draft step.
    int32_t draftTopK{10};

    // Number of drafting steps to perform with the draft model.
    // Each step extends the current draft proposal.
    int32_t draftStep{6};

    // Number of proposal tokens to select for base model verification.
    int32_t verifySize{60};

    // DFlash-only draft horizon. 0 means infer from the engine config.
    int32_t dflashBlockSize{0};
};

struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    // Override parameters (only batchSize, maxGenerateLength, and numLogprobs can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    int32_t numLogprobs{-1};       // -1 means use value from input file
    SpecDecodeArgs specDecodeArgs;
    rt::ContextCacheConfig contextCacheConfig;
    //! Force PREFILL_STATE_ONLY commit policy on every request (required to exercise Hybrid+MTP context reuse from the
    //! CLI, which the server otherwise sets per request).
    bool contextCachePrefillStateOnly{false};

    // Qwen3-Omni audio output options
    bool enableAudioOutput{false};
    std::string talkerEngineDir{""};
    std::string code2wavEngineDir{""};
    std::string outputAudioDir{""};

    // Talker sampling params (read from input JSON, defaults match qwen3_tts_inference)
    float talkerTemperature{0.9f};
    int32_t talkerTopK{50};
    float talkerTopP{1.0f};
    float talkerRepetitionPenalty{1.05f};

    // Thinker-Talker streaming mode (single CUDA stream interleaved).
    // All fields below can be set either via CLI flag or the top-level
    // "streaming": { "enable", "codec_chunk_frames", "talker_prefill_threshold" }
    // block in the input JSON — JSON is the preferred path so scenarios are
    // self-describing; the CLI flag remains for ad-hoc runs.
    bool enableThinkerTalkerStreaming{false};
    int32_t codecChunkFrames{10};      //!< Vocode every N Talker frames during streaming (0 = disabled)
    int32_t talkerPrefillThreshold{4}; //!< Start Talker prefill after this many Thinker assistant tokens
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--specDecode] "
                 "[--specDraftTopK=<number>] [--specDraftStep=<number>] "
                 "[--specVerifySize=<number>] [--dflashBlockSize=<number>] [--enableContextReuse]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr
        << "  --numLogprobs             Number of top log-probabilities to return per token (0 = disabled, max 50)"
        << std::endl;
    std::cerr << "  --specDecode              Enable speculative decoding (EAGLE, MTP, or DFlash)" << std::endl;
    std::cerr << "  --specDraftTopK           Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            For DFlash: candidateTopK; 1 is linear, >1 enables branching DDTree"
              << std::endl;
    std::cerr << "  --specDraftStep           Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            DFlash requires this to be 1; use dflashBlockSize for proposal horizon"
              << std::endl;
    std::cerr << "  --specVerifySize          Number of proposal tokens for base verification (default: 60)"
              << std::endl;
    std::cerr << "  --dflashBlockSize         DFlash proposal block size; 0 means infer from engine config"
              << std::endl;
    std::cerr << "\nProduction Context Reuse Options:" << std::endl;
    std::cerr << "  --enableContextReuse      Enable process-local content-addressed context reuse" << std::endl;
    std::cerr << "  --contextCacheMaxRecords  Maximum retained context records (default: 1024)" << std::endl;
    std::cerr << "  --contextCacheRecurrentSnapshotPoolBytes" << std::endl;
    std::cerr << "                            Device byte budget for hybrid recurrent/conv snapshots (default: 0)"
              << std::endl;
    std::cerr << "  --contextCachePartialKVSnapshotPoolBytes" << std::endl;
    std::cerr << "                            Device byte budget for hybrid partial-KV snapshots (default: 0)"
              << std::endl;
    std::cerr << "  --contextCacheMediaArtifactPoolBytes" << std::endl;
    std::cerr << "                            Device byte limit for ViT/audio artifacts (default: 536870912)"
              << std::endl;
    std::cerr << "  --contextCacheMaxMediaArtifacts" << std::endl;
    std::cerr << "                            Maximum retained media artifacts (default: 1024)" << std::endl;
    std::cerr << "                            KV retention capacity is configured at build time with"
              << " --maxKVPoolPages" << std::endl;
    std::cerr << "                            Send full histories as sequential batches (batch_size=1)"
              << " for multi-turn reuse" << std::endl;
    std::cerr << "\nQwen3-Omni Audio Output Options:" << std::endl;
    std::cerr << "  --enableAudioOutput       Enable audio output from Thinker hidden states" << std::endl;
    std::cerr << "  --talkerEngineDir         Path to Talker engine directory" << std::endl;
    std::cerr << "  --code2wavEngineDir       Path to Code2Wav engine directory (optional)" << std::endl;
    std::cerr << "  --outputAudioDir          Directory to save generated audio (.wav) files" << std::endl;
}

namespace
{

template <typename IntegerType>
bool parseNonNegativeIntegerOption(char const* optionName, char const* value, IntegerType& output)
{
    static_assert(std::is_integral_v<IntegerType> && std::is_signed_v<IntegerType>);
    std::string_view const text{value == nullptr ? "" : value};
    IntegerType parsed{};
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed < 0)
    {
        LOG_ERROR("Invalid --%s value: %s (must be a non-negative integer)", optionName, text.data());
        return false;
    }
    output = parsed;
    return true;
}

} // namespace

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, LLMInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"specDecode", no_argument, 0, LLMInferenceOptionId::SPEC_DECODE},
        {"eagle", no_argument, 0, LLMInferenceOptionId::SPEC_DECODE}, // deprecated alias
        {"specDraftTopK", required_argument, 0, LLMInferenceOptionId::SPEC_DRAFT_TOP_K},
        {"eagleDraftTopK", required_argument, 0, LLMInferenceOptionId::SPEC_DRAFT_TOP_K}, // deprecated alias
        {"specDraftStep", required_argument, 0, LLMInferenceOptionId::SPEC_DRAFT_STEP},
        {"eagleDraftStep", required_argument, 0, LLMInferenceOptionId::SPEC_DRAFT_STEP}, // deprecated alias
        {"specVerifySize", required_argument, 0, LLMInferenceOptionId::SPEC_VERIFY_SIZE},
        {"specVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::SPEC_VERIFY_SIZE},
        {"eagleVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::SPEC_VERIFY_SIZE}, // deprecated alias
        {"dflashBlockSize", required_argument, 0, LLMInferenceOptionId::DFLASH_BLOCK_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH},
        {"numLogprobs", required_argument, 0, LLMInferenceOptionId::NUM_LOGPROBS},
        {"enableAudioOutput", no_argument, 0, LLMInferenceOptionId::ENABLE_AUDIO_OUTPUT},
        {"talkerEngineDir", required_argument, 0, LLMInferenceOptionId::TALKER_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, LLMInferenceOptionId::CODE2WAV_ENGINE_DIR},
        {"outputAudioDir", required_argument, 0, LLMInferenceOptionId::OUTPUT_AUDIO_DIR},
        {"enableThinkerTalkerStreaming", no_argument, 0, LLMInferenceOptionId::ENABLE_THINKER_TALKER_STREAMING},
        {"enableContextReuse", no_argument, 0, LLMInferenceOptionId::ENABLE_CONTEXT_REUSE},
        {"contextCacheMaxRecords", required_argument, 0, LLMInferenceOptionId::CONTEXT_CACHE_MAX_RECORDS},
        {"contextCacheRecurrentSnapshotPoolBytes", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_RECURRENT_SNAPSHOT_POOL_BYTES},
        {"contextCachePartialKVSnapshotPoolBytes", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_PARTIAL_KV_SNAPSHOT_POOL_BYTES},
        {"contextCacheMediaArtifactPoolBytes", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_MEDIA_ARTIFACT_POOL_BYTES},
        {"contextCacheMaxMediaArtifacts", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_MAX_MEDIA_ARTIFACTS},
        {"contextCachePrefillStateOnly", no_argument, 0, LLMInferenceOptionId::CONTEXT_CACHE_PREFILL_STATE_ONLY},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case LLMInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case LLMInferenceOptionId::DEBUG: args.debug = true; break;
        case LLMInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case LLMInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case LLMInferenceOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case LLMInferenceOptionId::SPEC_DECODE: args.specDecodeArgs.enabled = true; break;
        case LLMInferenceOptionId::SPEC_DRAFT_TOP_K:
            try
            {
                args.specDecodeArgs.draftTopK = std::stoi(optarg);
                if (args.specDecodeArgs.draftTopK <= 0)
                {
                    LOG_ERROR("Invalid specDraftTopK value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid specDraftTopK value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::SPEC_DRAFT_STEP:
            try
            {
                args.specDecodeArgs.draftStep = std::stoi(optarg);
                if (args.specDecodeArgs.draftStep <= 0)
                {
                    LOG_ERROR("Invalid specDraftStep value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid specDraftStep value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::SPEC_VERIFY_SIZE:
            try
            {
                args.specDecodeArgs.verifySize = std::stoi(optarg);
                if (args.specDecodeArgs.verifySize <= 0)
                {
                    LOG_ERROR("Invalid specVerifySize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid specVerifySize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DFLASH_BLOCK_SIZE:
            try
            {
                args.specDecodeArgs.dflashBlockSize = std::stoi(optarg);
                if (args.specDecodeArgs.dflashBlockSize < 0)
                {
                    LOG_ERROR("Invalid dflashBlockSize value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dflashBlockSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength <= 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::ENABLE_AUDIO_OUTPUT: args.enableAudioOutput = true; break;
        case LLMInferenceOptionId::TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case LLMInferenceOptionId::CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_AUDIO_DIR: args.outputAudioDir = optarg; break;
        case LLMInferenceOptionId::ENABLE_THINKER_TALKER_STREAMING: args.enableThinkerTalkerStreaming = true; break;
        case LLMInferenceOptionId::NUM_LOGPROBS:
            try
            {
                args.numLogprobs = std::stoi(optarg);
                if (args.numLogprobs < 0)
                {
                    LOG_ERROR("Invalid numLogprobs value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid numLogprobs value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::ENABLE_CONTEXT_REUSE: args.contextCacheConfig.enabled = true; break;
        case LLMInferenceOptionId::CONTEXT_CACHE_PREFILL_STATE_ONLY: args.contextCachePrefillStateOnly = true; break;
        case LLMInferenceOptionId::CONTEXT_CACHE_MAX_RECORDS:
            if (!parseNonNegativeIntegerOption("contextCacheMaxRecords", optarg, args.contextCacheConfig.maxRecords))
            {
                return false;
            }
            break;
        case LLMInferenceOptionId::CONTEXT_CACHE_RECURRENT_SNAPSHOT_POOL_BYTES:
            if (!parseNonNegativeIntegerOption("contextCacheRecurrentSnapshotPoolBytes", optarg,
                    args.contextCacheConfig.recurrentSnapshotPoolBytes))
            {
                return false;
            }
            break;
        case LLMInferenceOptionId::CONTEXT_CACHE_PARTIAL_KV_SNAPSHOT_POOL_BYTES:
            if (!parseNonNegativeIntegerOption("contextCachePartialKVSnapshotPoolBytes", optarg,
                    args.contextCacheConfig.partialKvSnapshotPoolBytes))
            {
                return false;
            }
            break;
        case LLMInferenceOptionId::CONTEXT_CACHE_MEDIA_ARTIFACT_POOL_BYTES:
            if (!parseNonNegativeIntegerOption(
                    "contextCacheMediaArtifactPoolBytes", optarg, args.contextCacheConfig.mediaArtifactPoolBytes))
            {
                return false;
            }
            break;
        case LLMInferenceOptionId::CONTEXT_CACHE_MAX_MEDIA_ARTIFACTS:
            if (!parseNonNegativeIntegerOption(
                    "contextCacheMaxMediaArtifacts", optarg, args.contextCacheConfig.maxMediaArtifacts))
            {
                return false;
            }
            break;
        default: return false;
        }
    }

    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
    }

    if (args.specDecodeArgs.enabled)
    {
        LOG_INFO("Speculative decoding enabled");
        LOG_INFO("Spec draft topK: %d", args.specDecodeArgs.draftTopK);
        LOG_INFO("Spec draft step: %d", args.specDecodeArgs.draftStep);
        LOG_INFO("Spec verify size: %d", args.specDecodeArgs.verifySize);
        LOG_INFO("DFlash block size: %d", args.specDecodeArgs.dflashBlockSize);
    }

    if (args.contextCacheConfig.enabled)
    {
        LOG_INFO("Production context reuse enabled");
        LOG_INFO(
            "Context cache config: maxRecords=%d recurrentSnapshotPoolBytes=%lld "
            "partialKVSnapshotPoolBytes=%lld mediaArtifactPoolBytes=%lld maxMediaArtifacts=%d",
            args.contextCacheConfig.maxRecords,
            static_cast<long long>(args.contextCacheConfig.recurrentSnapshotPoolBytes),
            static_cast<long long>(args.contextCacheConfig.partialKvSnapshotPoolBytes),
            static_cast<long long>(args.contextCacheConfig.mediaArtifactPoolBytes),
            args.contextCacheConfig.maxMediaArtifacts);
        if (args.warmup > 0)
        {
            LOG_INFO("Warmup requests will bypass context reuse and will not populate the cache");
        }
    }

    if (args.enableAudioOutput)
    {
        if (args.talkerEngineDir.empty())
        {
            LOG_ERROR("--talkerEngineDir is required when --enableAudioOutput is set");
            return false;
        }
        LOG_INFO("Audio output enabled");
        LOG_INFO("  Talker engine: %s", args.talkerEngineDir.c_str());
        if (!args.code2wavEngineDir.empty())
        {
            LOG_INFO("  Code2Wav engine: %s", args.code2wavEngineDir.c_str());
        }
        if (!args.outputAudioDir.empty())
        {
            LOG_INFO("  Audio output dir: %s", args.outputAudioDir.c_str());
        }
    }

    if (args.enableThinkerTalkerStreaming)
    {
        args.enableAudioOutput = true;
        if (args.talkerEngineDir.empty())
        {
            LOG_ERROR("--enableThinkerTalkerStreaming requires --talkerEngineDir");
            return false;
        }
        LOG_INFO("Thinker-Talker streaming enabled (single CUDA stream)");
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

// Thin wrapper around the shared parser in examples/utils/requestFileParser.h.
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1,
    int32_t numLogprobsOverride = -1, LLMInferenceArgs* argsOut = nullptr)
{
    auto result = exampleUtils::parseRequestFile(
        inputFilePath, batchSizeOverride, maxGenerateLengthOverride, numLogprobsOverride);

    if (argsOut != nullptr && argsOut->enableAudioOutput)
    {
        std::ifstream inputFileStream(inputFilePath);
        if (inputFileStream.is_open())
        {
            Json inputData = Json::parse(inputFileStream);
            argsOut->talkerTemperature = inputData.value("talker_temperature", 0.9f);
            argsOut->talkerTopK = inputData.value("talker_top_k", 50);
            argsOut->talkerTopP = inputData.value("talker_top_p", 1.0f);
            argsOut->talkerRepetitionPenalty = inputData.value("repetition_penalty", 1.05f);
            LOG_INFO("Talker params from JSON: temperature=%.2f, topK=%d, topP=%.2f, repetitionPenalty=%.2f",
                argsOut->talkerTemperature, argsOut->talkerTopK, argsOut->talkerTopP, argsOut->talkerRepetitionPenalty);

            // Thinker-Talker streaming config: top-level "streaming": {...} block.
            // CLI --enableThinkerTalkerStreaming takes precedence when set; JSON fills the rest.
            if (inputData.contains("streaming") && inputData["streaming"].is_object())
            {
                auto const& streamingCfg = inputData["streaming"];
                bool const jsonEnable = streamingCfg.value("enable", false);
                if (jsonEnable)
                {
                    argsOut->enableThinkerTalkerStreaming = true;
                }
                argsOut->codecChunkFrames = streamingCfg.value("codec_chunk_frames", argsOut->codecChunkFrames);
                argsOut->talkerPrefillThreshold
                    = streamingCfg.value("talker_prefill_threshold", argsOut->talkerPrefillThreshold);
                if (argsOut->enableThinkerTalkerStreaming)
                {
                    LOG_INFO(
                        "Thinker-Talker streaming from JSON: enable=true, codecChunkFrames=%d, "
                        "talkerPrefillThreshold=%d",
                        argsOut->codecChunkFrames, argsOut->talkerPrefillThreshold);
                }
            }
        }
    }

    return result;
}

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "llm_inference");
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    bool profilerEnabled = args.dumpProfile;
    MemoryMonitor memoryMonitor;
    // Start memory monitoring at the beginning if profiling is enabled
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    auto pluginHandles = loadEdgellmPluginLib();
    // load input file and parse to requests
    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    try
    {
        std::tie(loraWeightsMap, batchedRequests)
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength, args.numLogprobs, &args);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (args.contextCachePrefillStateOnly)
    {
        for (auto& request : batchedRequests)
        {
            request.contextCacheCommitPolicy = rt::CommitPolicy::kPrefillStateOnly;
        }
        LOG_INFO("Forced PREFILL_STATE_ONLY commit policy on %zu requests for Hybrid+MTP context reuse.",
            batchedRequests.size());
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    // Create unified runtime (handles both vanilla and speculative decoding modes)
    std::unique_ptr<rt::LLMInferenceRuntime> runtime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (args.specDecodeArgs.enabled)
    {
        rt::SpecDecodeDraftingConfig draftingConfig;
        draftingConfig.draftingTopK = args.specDecodeArgs.draftTopK;
        draftingConfig.draftingStep = args.specDecodeArgs.draftStep;
        draftingConfig.verifySize = args.specDecodeArgs.verifySize;
        draftingConfig.dflashBlockSize = args.specDecodeArgs.dflashBlockSize;
        try
        {
            runtime = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir,
                loraWeightsMap, draftingConfig, stream, args.contextCacheConfig);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime with speculative decoding: %s", e.what());
            return EXIT_FAILURE;
        }
    }
    else
    {
        // Standard vanilla-only mode (no draft model)
        try
        {
            runtime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream, args.contextCacheConfig);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("Failed to capture CUDA graph for decoding, proceeding with normal engine execution.");
    }

    // Initialize Qwen3-Omni audio output pipeline (TTS runtime + Code2Wav)
    std::unique_ptr<rt::Qwen3OmniTTSRuntime> ttsRuntime;
    std::unique_ptr<rt::Code2WavRunner> code2wavRunner;
    if (args.enableAudioOutput)
    {
        try
        {
            std::filesystem::path const codePredictorDir
                = std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor";
            ttsRuntime = std::make_unique<rt::Qwen3OmniTTSRuntime>(
                args.talkerEngineDir, codePredictorDir.string(), args.engineDir, stream);
            LOG_INFO("TTS runtime initialized for audio output");
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize TTS Runtime: %s", e.what());
            return EXIT_FAILURE;
        }

        // Code2Wav runner (optional — falls back to RVQ code output)
        std::filesystem::path const code2wavDir = args.code2wavEngineDir.empty()
            ? std::filesystem::path(args.talkerEngineDir).parent_path() / "code2wav"
            : std::filesystem::path(args.code2wavEngineDir);
        if (std::filesystem::exists(code2wavDir))
        {
            try
            {
                code2wavRunner = std::make_unique<rt::Code2WavRunner>(code2wavDir.string(), stream);
                LOG_INFO("Code2Wav runner initialized");
            }
            catch (std::exception const& e)
            {
                LOG_WARNING("Failed to initialize Code2Wav: %s. Will output RVQ codes only.", e.what());
            }
        }

        if (!ttsRuntime->captureDecodingCUDAGraph(stream))
        {
            LOG_WARNING("CUDA graph capture failed for TTS decoding, proceeding without.");
        }

        if (!args.outputAudioDir.empty())
        {
            std::filesystem::create_directories(args.outputAudioDir);
        }
    }

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        // Disable profiling for warmup runs
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = batchedRequests[0];
        // Warmup primes execution resources, not application-visible context state.
        bool const originalDisableContextReuse = firstRequest.disableContextReuse;
        firstRequest.disableContextReuse = true;

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = runtime->handleRequest(firstRequest, warmupResponse, stream);

            if (!requestStatus)
            {
                firstRequest.disableContextReuse = originalDisableContextReuse;
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
        firstRequest.disableContextReuse = originalDisableContextReuse;
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(true);
    }

    // Structure to collect all responses for JSON export
    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;
    // Index of the request in the input file's flat "requests" array. Batching packs
    // batchSize consecutive requests into one batched request, so downstream consumers
    // (e.g. calculate_wer_score.py) must receive the flat index, not the batch index.
    size_t flatRequestIdx = 0;

    // Process each request with progress indication
    LOG_INFO("Processing %zu batched requests...", batchedRequests.size());
    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        auto& request = batchedRequests[requestIdx];
        rt::LLMGenerationResponse response;

        // Show progress every 10% or every 100 requests, whichever is smaller
        size_t progressInterval = std::max(size_t(1), std::min(batchedRequests.size() / 10, size_t(100)));
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == batchedRequests.size() - 1)
        {
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, batchedRequests.size(),
                100.0 * (requestIdx + 1) / batchedRequests.size());
        }

        bool requestStatus = false;
        StreamingAudioWriter streamingWriter;

        cudaEvent_t e2eStart{nullptr}, e2eEnd{nullptr}, ttfpaEvent{nullptr};
        bool ttfpaRecorded = false;
        if (getProfilingEnabled() && ttsRuntime)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&e2eStart, cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&e2eEnd, cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&ttfpaEvent, cudaEventDefault));
            CUDA_CHECK(cudaEventRecord(e2eStart, stream));
        }

        // Streaming path: Thinker + Talker interleaved on the same CUDA stream
        rt::Qwen3OmniTTSRuntime::TalkerGenerationResponse streamingTalkerResp;
        if (args.enableThinkerTalkerStreaming && ttsRuntime)
        {
            rt::Qwen3OmniTTSRuntime::OmniGenerationRequest omniBaseReq;
            omniBaseReq.talkerTemperature = args.talkerTemperature;
            omniBaseReq.talkerTopK = args.talkerTopK;
            omniBaseReq.talkerTopP = args.talkerTopP;
            omniBaseReq.repetitionPenalty = args.talkerRepetitionPenalty;

            rt::Qwen3OmniTTSRuntime::ThinkerTalkerStreamingConfig streamCfg;
            streamCfg.talkerPrefillThreshold = args.talkerPrefillThreshold;

            if (!args.outputAudioDir.empty() && code2wavRunner)
            {
                std::string filename = format::fmtstr("audio_req%zu_batch0.wav", requestIdx);
                std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                streamingWriter.open(audioPath.string(), 24000);

                streamCfg.codecChunkFrames = args.codecChunkFrames;
                streamCfg.onAudioChunkReady
                    = [&](std::vector<std::vector<int32_t>> const& chunkCodes, bool /*isFinal*/) {
                          if (chunkCodes.empty() || !code2wavRunner)
                              return;
                          size_t const numFrames = chunkCodes.size();
                          size_t const numLayers = chunkCodes[0].size();
                          std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
                          for (size_t f = 0; f < numFrames; ++f)
                              for (size_t l = 0; l < numLayers; ++l)
                                  transposed[l][f] = chunkCodes[f][l];

                          rt::audioUtils::AudioData chunkAudio;
                          code2wavRunner->generateWaveform(transposed, chunkAudio, stream);
                          if (chunkAudio.hasWaveform)
                          {
                              streamingWriter.appendChunk(chunkAudio);
                              if (!ttfpaRecorded && ttfpaEvent)
                              {
                                  CUDA_CHECK(cudaEventRecord(ttfpaEvent, stream));
                                  ttfpaRecorded = true;
                              }
                          }
                      };
            }

            request.generateAudio = true;

            requestStatus = ttsRuntime->handleStreamingGeneration(
                *runtime, request, response, streamCfg, omniBaseReq, streamingTalkerResp, stream);

            if (requestStatus)
            {
                LOG_INFO("Request %zu: Thinker-Talker streaming generated %d audio frames", requestIdx,
                    streamingTalkerResp.numFramesPerSample.empty() ? 0 : streamingTalkerResp.numFramesPerSample[0]);
            }
        }
        else
        {
            // Sequential Omni: tell the runtime which hidden layer the Talker needs so it
            // registers mOutputHiddenStates under that layer in the portal (not layer 0).
            if (args.enableAudioOutput && ttsRuntime)
            {
                std::vector<int32_t> const requiredLayers = ttsRuntime->getThinkerHiddenLayerIndices();
                request.acceptHiddenLayer = (requiredLayers.size() >= 2) ? requiredLayers[1] : 14;
            }
            requestStatus = runtime->handleRequest(request, response, stream, args.enableAudioOutput);
        }

        // Qwen3-Omni audio generation: Code2Wav vocoding
        std::vector<rt::audioUtils::AudioData> audioOutputs;

        // Helper: transpose RVQ codes [frames][layers] → [layers][frames] and vocode
        auto vocodeAndSave = [&](std::vector<std::vector<int32_t>> const& framesCodes, size_t batchIdx) {
            if (framesCodes.empty() || framesCodes[0].empty() || !code2wavRunner)
                return;
            size_t const numFrames = framesCodes.size();
            size_t const numLayers = framesCodes[0].size();
            std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
            for (size_t f = 0; f < numFrames; ++f)
                for (size_t l = 0; l < numLayers; ++l)
                    transposed[l][f] = framesCodes[f][l];

            if (args.dumpOutput)
            {
                int32_t codeMin = transposed[0][0], codeMax = transposed[0][0];
                for (auto const& row : transposed)
                    for (int32_t c : row)
                    {
                        codeMin = std::min(codeMin, c);
                        codeMax = std::max(codeMax, c);
                    }
                LOG_INFO("Batch %zu: RVQ codes %zu frames x %zu layers, range [%d, %d]", batchIdx, numFrames, numLayers,
                    codeMin, codeMax);
            }

            rt::audioUtils::AudioData audioData;
            if (code2wavRunner->generateWaveform(transposed, audioData, stream) && audioData.hasWaveform)
            {
                if (!args.outputAudioDir.empty())
                {
                    std::string filename = format::fmtstr("audio_req%zu_batch%zu.wav", requestIdx, batchIdx);
                    std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                    saveAudioToWav(audioPath.string(), audioData);
                    LOG_INFO("Audio saved: %s", audioPath.string().c_str());
                }
                audioOutputs.push_back(std::move(audioData));
            }
        };

        // Streaming path: vocode the streaming RVQ codes
        if (requestStatus && args.enableThinkerTalkerStreaming && !streamingTalkerResp.batchRvqCodes.empty())
        {
            if (streamingWriter.totalSamplesWritten() > 0)
            {
                streamingWriter.finalize();
                LOG_INFO("Streaming audio written: %ld samples (%.2fs)", streamingWriter.totalSamplesWritten(),
                    static_cast<float>(streamingWriter.totalSamplesWritten()) / 24000.0f);
            }
            else
            {
                for (size_t batchIdx = 0; batchIdx < streamingTalkerResp.batchRvqCodes.size(); ++batchIdx)
                {
                    vocodeAndSave(streamingTalkerResp.batchRvqCodes[batchIdx], batchIdx);
                }
            }
        }

        // Non-streaming path: build batched Omni requests and call Talker once.
        // Fetch Thinker prefill embeddings / hidden states from the runtime portal
        // (see LLMInferenceRuntime::getBaseModelHiddenStates contract).
        rt::Tensor const* prefillEmbedsAll = runtime->getBaseModelHiddenStates(0);
        std::vector<int32_t> const requiredLayers
            = ttsRuntime ? ttsRuntime->getThinkerHiddenLayerIndices() : std::vector<int32_t>{};
        int32_t const acceptHiddenLayer = (requiredLayers.size() >= 2) ? requiredLayers[1] : 14;
        rt::Tensor const* hiddenStatesAll = runtime->getBaseModelHiddenStates(acceptHiddenLayer);
        auto const& thinkerInputTokenIds = runtime->getBaseModelInputTokenIds();

        if (requestStatus && !args.enableThinkerTalkerStreaming && args.enableAudioOutput && ttsRuntime
            && prefillEmbedsAll != nullptr && !prefillEmbedsAll->isEmpty())
        {
            int32_t const prefillLen = runtime->getBaseModelPrefillLength();
            int64_t const H = prefillEmbedsAll->getShape()[2];
            int64_t const batchStride = static_cast<int64_t>(prefillLen) * H;

            // Per-batch views into the [BS, prefillLen, H] tensors (must outlive omniRequests)
            size_t const batchSize = response.outputIds.size();
            std::vector<rt::Tensor> perBatchEmbedViews(batchSize);
            std::vector<rt::Tensor> perBatchHiddenViews(batchSize);

            std::vector<rt::Qwen3OmniTTSRuntime::OmniGenerationRequest> omniRequests;
            for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                rt::Qwen3OmniTTSRuntime::OmniGenerationRequest omniReq;
                if (batchIdx < thinkerInputTokenIds.size())
                {
                    omniReq.textTokenIds = thinkerInputTokenIds[batchIdx];
                    omniReq.textTokenIds.insert(omniReq.textTokenIds.end(), response.outputIds[batchIdx].begin(),
                        response.outputIds[batchIdx].end());
                }
                omniReq.talkerTemperature = args.talkerTemperature;
                omniReq.talkerTopK = args.talkerTopK;
                omniReq.talkerTopP = args.talkerTopP;
                omniReq.repetitionPenalty = args.talkerRepetitionPenalty;

                __half* embedBase = static_cast<__half*>(const_cast<void*>(prefillEmbedsAll->rawPointer()))
                    + static_cast<int64_t>(batchIdx) * batchStride;
                perBatchEmbedViews[batchIdx] = rt::Tensor(embedBase, rt::Coords{1, static_cast<int64_t>(prefillLen), H},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                omniReq.thinkerPrefillEmbeds = &perBatchEmbedViews[batchIdx];

                if (hiddenStatesAll != nullptr)
                {
                    __half* hiddenBase = static_cast<__half*>(const_cast<void*>(hiddenStatesAll->rawPointer()))
                        + static_cast<int64_t>(batchIdx) * batchStride;
                    perBatchHiddenViews[batchIdx]
                        = rt::Tensor(hiddenBase, rt::Coords{1, static_cast<int64_t>(prefillLen), H},
                            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                    omniReq.thinkerHiddenStates = &perBatchHiddenViews[batchIdx];
                }

                omniReq.prefillLength = batchIdx < thinkerInputTokenIds.size()
                    ? static_cast<int32_t>(thinkerInputTokenIds[batchIdx].size())
                    : 0;
                omniRequests.push_back(std::move(omniReq));
            }

            rt::Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResp;
            if (ttsRuntime->handleAudioGenerationFromThinker(omniRequests, talkerResp, stream))
            {
                for (size_t batchIdx = 0; batchIdx < talkerResp.batchRvqCodes.size(); ++batchIdx)
                {
                    vocodeAndSave(talkerResp.batchRvqCodes[batchIdx], batchIdx);
                }
            }
            else
            {
                LOG_WARNING("Batched audio generation failed for request %zu", requestIdx);
            }
        }

        if (e2eStart && e2eEnd && ttsRuntime)
        {
            CUDA_CHECK(cudaEventRecord(e2eEnd, stream));
            CUDA_CHECK(cudaEventSynchronize(e2eEnd));

            auto& latency = ttsRuntime->getMutableOmniLatencyMetrics();

            float e2eMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&e2eMs, e2eStart, e2eEnd));
            latency.endToEndMs = e2eMs;

            auto const& ttfaEnd = ttsRuntime->getTtfaEndEvent();
            if (ttfaEnd)
            {
                CUDA_CHECK(cudaEventSynchronize(ttfaEnd));
                float ttfaMs = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ttfaMs, e2eStart, ttfaEnd));
                latency.timeToFirstAudioCodeMs = ttfaMs;
            }

            if (ttfpaRecorded && ttfpaEvent)
            {
                CUDA_CHECK(cudaEventSynchronize(ttfpaEvent));
                float ttfpaMs = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ttfpaMs, e2eStart, ttfpaEvent));
                latency.timeToFirstPlayableAudioMs = ttfpaMs;
            }
            else
            {
                latency.timeToFirstPlayableAudioMs = e2eMs;
            }

            cudaEventDestroy(e2eStart);
            cudaEventDestroy(e2eEnd);
            if (ttfpaEvent)
            {
                cudaEventDestroy(ttfpaEvent);
            }
        }

        if (requestStatus)
        {
            if (args.dumpOutput)
            {
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    char const* reasonName = batchIdx < response.finishReasons.size()
                        ? rt::finishReasonName(response.finishReasons[batchIdx])
                        : "?";
                    LOG_INFO("Response for request %zu batch %zu [finish=%s]: %s", requestIdx, batchIdx, reasonName,
                        response.outputTexts[batchIdx].c_str());
                    if (batchIdx < audioOutputs.size() && audioOutputs[batchIdx].waveform
                        && !audioOutputs[batchIdx].waveform->isEmpty())
                    {
                        auto const& shape = audioOutputs[batchIdx].waveform->getShape();
                        int64_t samples = shape[shape.getNumDims() - 1];
                        LOG_INFO("  Audio: %ld samples (%.2fs)", samples,
                            static_cast<float>(samples) / audioOutputs[batchIdx].sampleRate);
                    }
                }
            }
        }
        else
        {
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // Add to JSON output with UTF-8 validation on output text
        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            bool const hasOutputText = requestStatus && batchIdx < response.outputTexts.size();
            std::string outputText = hasOutputText ? response.outputTexts[batchIdx] : errorMessage;
            auto const* formattedRequest
                = batchIdx < request.formattedRequests.size() ? &request.formattedRequests[batchIdx] : nullptr;
            // Validate UTF-8 for output text (inputs are always valid)
            // If invalid UTF-8 detected, error message is returned and original text is logged
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = flatRequestIdx++;
            responseJson["batch_idx"] = batchIdx;
            responseJson["finish_reason"] = (requestStatus && batchIdx < response.finishReasons.size())
                ? rt::finishReasonName(response.finishReasons[batchIdx])
                : "error";
            // Store messages for reference
            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : request.requests[batchIdx].messages)
            {
                nlohmann::json msgJson;
                msgJson["role"] = msg.role;
                msgJson["content"] = nlohmann::json::array();
                for (auto const& content : msg.contents)
                {
                    nlohmann::json contentJson;
                    contentJson["type"] = content.type;
                    if (content.type == "text")
                    {
                        contentJson["text"] = content.content;
                    }
                    else if (content.type == "image")
                    {
                        contentJson["image"] = content.content;
                    }
                    else if (content.type == "video")
                    {
                        contentJson["video"] = content.content;
                    }
                    msgJson["content"].push_back(contentJson);
                }
                messagesJson.push_back(msgJson);
            }
            responseJson["messages"] = messagesJson;
            // Store formatted prompts for reference
            responseJson["formatted_system_prompt"] = formattedRequest ? formattedRequest->formattedSystemPrompt : "";
            responseJson["formatted_complete_request"]
                = formattedRequest ? formattedRequest->formattedCompleteRequest : "";
            // Serialize logprobs if present: logprobs[step] = [{token_id, token, bytes, logprob}, ...]
            // `token` is the UTF-8-sanitized piece string (invalid bytes -> U+FFFD, required so
            // nlohmann::json::dump does not throw); `bytes` carries the raw token bytes losslessly.
            if (requestStatus && batchIdx < response.logprobs.size() && !response.logprobs[batchIdx].empty())
            {
                nlohmann::json logprobsJson = nlohmann::json::array();
                for (auto const& stepEntries : response.logprobs[batchIdx])
                {
                    nlohmann::json stepJson = nlohmann::json::array();
                    for (auto const& entry : stepEntries)
                    {
                        std::string pending;
                        std::string token = utf8::sanitizeUtf8Streaming(entry.piece, pending);
                        token += utf8::sanitizeUtf8Flush(pending);
                        nlohmann::json bytesJson = nlohmann::json::array();
                        for (unsigned char b : entry.piece)
                        {
                            bytesJson.push_back(static_cast<int>(b));
                        }
                        stepJson.push_back({{"token_id", entry.tokenId}, {"token", std::move(token)},
                            {"bytes", std::move(bytesJson)}, {"logprob", entry.logprob}});
                    }
                    logprobsJson.push_back(std::move(stepJson));
                }
                responseJson["logprobs"] = std::move(logprobsJson);
            }
            outputData["responses"].push_back(responseJson);
        }
    }

    // Final processing summary
    LOG_INFO("Processing complete: %zu/%zu batched requests successful", batchedRequests.size() - failedCount,
        batchedRequests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    if (profilerEnabled)
    {
        // Stop memory monitoring for examples
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Performance Summary ===" << std::endl;
        auto prefillMetrics = runtime->getPrefillMetrics();
        auto multimodalMetrics = runtime->getMultimodalMetrics();
        outputPrefillProfile(profileOutput, prefillMetrics);
        if (args.specDecodeArgs.enabled)
        {
            auto specDecodeGenerationMetrics = runtime->getSpecDecodeGenerationMetrics();
            outputSpecDecodeGenerationProfile(
                profileOutput, specDecodeGenerationMetrics, runtime->getSpeculativeDecodingStrategyName());
        }
        else
        {
            outputGenerationProfile(profileOutput, runtime->getGenerationMetrics());
        }
        outputMultimodalProfile(profileOutput, multimodalMetrics);
        if (ttsRuntime)
        {
            outputTalkerProfile(profileOutput, ttsRuntime->getMetrics());
            outputOmniProfile(profileOutput, ttsRuntime->getOmniTalkerMetrics(), ttsRuntime->getOmniLatencyMetrics());
        }
        outputMemoryProfile(profileOutput, memoryMonitor);
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // Export profile to JSON file
    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;

            // Add high-level metrics from unified runtime
            addJsonPrefillSummary(profileJson, runtime->getPrefillMetrics());
            if (args.specDecodeArgs.enabled)
            {
                addJsonSpecDecodeGenerationSummary(profileJson, runtime->getSpecDecodeGenerationMetrics(),
                    runtime->getSpeculativeDecodingStrategyName());
            }
            else
            {
                addJsonGenerationSummary(profileJson, runtime->getGenerationMetrics());
            }
            addJsonMultimodalSummary(profileJson, runtime->getMultimodalMetrics());
            if (ttsRuntime)
            {
                addJsonTalkerSummary(profileJson, ttsRuntime->getMetrics());
            }

            // Add detailed timing stages
            addJsonTimingStages(profileJson);

            if (ttsRuntime)
            {
                addJsonOmniStageExtensions(
                    profileJson, ttsRuntime->getOmniTalkerMetrics(), ttsRuntime->getOmniLatencyMetrics());
            }

            // Add memory usage
            addJsonMemorySummary(profileJson, memoryMonitor);

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2); // Pretty print with 2 space indentation
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    // Export to JSON file
    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4); // Pretty print with 4 spaces indentation
            outputFile.close();
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    // Return false if any request failed
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
