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
#include "multimodal/qwen3_omni/code2WavRunner.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
#include <mpi.h>
#endif

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
    DSPARK_SCHEDULER = 923,
    DSPARK_CONFIDENCE_THRESHOLD = 924,
    DSPARK_MIN_PROPOSAL_LEN = 925,
    DSPARK_MAX_PROPOSAL_LEN = 926,
    ENABLE_CONTEXT_REUSE = 927,
    CONTEXT_CACHE_MAX_RECORDS = 928,
    CONTEXT_CACHE_RECURRENT_SNAPSHOT_POOL_BYTES = 929,
    CONTEXT_CACHE_PARTIAL_KV_SNAPSHOT_POOL_BYTES = 930,
    CHECKPOINT_DIR = 931,
    DRAFT_CHECKPOINT_DIR = 932,
    TP_SIZE = 933,
    VISUAL_PRUNE = 935,
    DART_REDUCTION_RATIO = 936,
    DART_PIVOT_IMAGE_TOKENS = 937,
    DART_PIVOT_TEXT_TOKENS = 938,
    VISUAL_PRUNE_ALGO = 939,
    ENCODER_CACHE_BUDGET_BYTES = 940
};

// Struct to hold speculative decoding arguments (used by both EAGLE and MTP)
struct SpecDecodeArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // For tree-based strategies this is the branching factor; for chain-style
    // strategies it is the number of candidates retained per draft step.
    int32_t draftTopK{10};
    bool draftTopKSet{false};

    // Number of drafting steps to perform with the draft model.
    // Each step extends the current draft proposal.
    int32_t draftStep{6};
    bool draftStepSet{false};

    // Number of tokens in the base verification input.
    int32_t verifySize{60};
    bool verifySizeSet{false};

    // DFlash-only draft horizon. 0 means infer from the engine config.
    int32_t dflashBlockSize{0};

    rt::DSparkSchedulerMode dsparkSchedulerMode{rt::DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0};
};

struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string checkpointDir{""};
    std::string draftCheckpointDir{""};
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

    // Visual-token pruning (embedding-level, VLM prefill only; disabled by default).
    rt::VisualPrunerConfig visualPrunerConfig;

    // Thinker-Talker streaming mode (single CUDA stream interleaved).
    // All fields below can be set either via CLI flag or the top-level
    // "streaming": { "enable", "codec_chunk_frames", "talker_prefill_threshold" }
    // block in the input JSON — JSON is the preferred path so scenarios are
    // self-describing; the CLI flag remains for ad-hoc runs.
    bool enableThinkerTalkerStreaming{false};
    int32_t codecChunkFrames{10};      //!< Vocode every N Talker frames during streaming (0 = disabled)
    int32_t talkerPrefillThreshold{4}; //!< Start Talker prefill after this many Thinker assistant tokens

    int32_t tpSize{1};
};

namespace
{

std::filesystem::path getBaseConfigPath(std::string const& engineDir)
{
    std::filesystem::path const dir{engineDir};
    std::filesystem::path const configPath = dir / "config.json";
    if (std::filesystem::is_regular_file(configPath))
    {
        return configPath;
    }
    return dir / "base_config.json";
}

std::filesystem::path getDraftConfigPath(std::string const& engineDir)
{
    return std::filesystem::path{engineDir} / "draft_config.json";
}

int32_t maxVerifySizeOrDefault(rt::LLMEngineConfig const& config, int32_t fallback)
{
    return config.maxVerifyTreeSize > 0 ? config.maxVerifyTreeSize : fallback;
}

int32_t dsparkVerifySizeOrDefault(std::string const& engineDir)
{
    std::filesystem::path const draftConfigPath = getDraftConfigPath(engineDir);
    if (!std::filesystem::is_regular_file(draftConfigPath))
    {
        return 8;
    }

    rt::LLMEngineConfig const draftConfig = rt::parseDraftEngineConfig(draftConfigPath);
    return draftConfig.specDraftBlockSize > 0 ? draftConfig.specDraftBlockSize + 1 : 8;
}

int32_t cachedBlockDraftBlockSizeOrThrow(
    std::string const& engineDir, rt::LLMEngineConfig const& baseConfig, int32_t explicitBlockSize)
{
    if (explicitBlockSize > 0)
    {
        return explicitBlockSize;
    }

    std::filesystem::path const draftConfigPath = getDraftConfigPath(engineDir);
    if (std::filesystem::is_regular_file(draftConfigPath))
    {
        rt::LLMEngineConfig const draftConfig = rt::parseDraftEngineConfig(draftConfigPath);
        if (draftConfig.specDraftBlockSize > 0)
        {
            return draftConfig.specDraftBlockSize;
        }
    }
    if (baseConfig.specDraftBlockSize > 0)
    {
        return baseConfig.specDraftBlockSize;
    }

    throw std::runtime_error(
        "unable to resolve DFlash/JetSpec block size from CLI, draft_config.json, or base config.");
}

bool applyEngineSpecDecodeDefaults(LLMInferenceArgs& args)
{
    if (!args.specDecodeArgs.enabled)
    {
        return true;
    }

    try
    {
        rt::LLMEngineConfig const baseConfig = rt::parseEngineConfig(getBaseConfigPath(args.engineDir));
        SpecDecodeArgs& specArgs = args.specDecodeArgs;
        switch (baseConfig.specDecodeType)
        {
        case rt::SpecDecodeMode::kDFlash:
        case rt::SpecDecodeMode::kJetSpec:
        {
            if (!specArgs.draftTopKSet)
            {
                specArgs.draftTopK = 1;
            }
            if (!specArgs.draftStepSet)
            {
                specArgs.draftStep = 1;
            }
            int32_t const blockSize
                = cachedBlockDraftBlockSizeOrThrow(args.engineDir, baseConfig, specArgs.dflashBlockSize);
            if (specArgs.dflashBlockSize == 0)
            {
                specArgs.dflashBlockSize = blockSize;
            }
            if (!specArgs.verifySizeSet)
            {
                specArgs.verifySize = specArgs.draftTopK > 1 ? maxVerifySizeOrDefault(baseConfig, 128) : blockSize;
            }
            break;
        }
        case rt::SpecDecodeMode::kDSpark:
            if (!specArgs.draftTopKSet)
            {
                specArgs.draftTopK = 1;
            }
            if (!specArgs.draftStepSet)
            {
                specArgs.draftStep = 1;
            }
            if (!specArgs.verifySizeSet)
            {
                specArgs.verifySize = dsparkVerifySizeOrDefault(args.engineDir);
            }
            break;
        default: break;
        }

        bool const isCachedBlockDraft = baseConfig.specDecodeType == rt::SpecDecodeMode::kDFlash
            || baseConfig.specDecodeType == rt::SpecDecodeMode::kJetSpec;
        LOG_INFO("Spec decode engine mode: %s", rt::specDecodeModeName(baseConfig.specDecodeType));
        LOG_INFO("Resolved spec draft topK: %d", specArgs.draftTopK);
        LOG_INFO("Resolved spec draft step: %d", specArgs.draftStep);
        LOG_INFO("Resolved spec verify size: %d", specArgs.verifySize);
        if (isCachedBlockDraft)
        {
            LOG_INFO("Resolved DFlash/JetSpec block size: %d", specArgs.dflashBlockSize);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to resolve speculative decoding defaults from engine config: %s", e.what());
        return false;
    }

    return true;
}

} // namespace

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] "
                 "[--tpSize=<number>]";
    std::cerr << " [--specDecode] [--specDraftTopK=<number>] [--specDraftStep=<number>] "
                 "[--specVerifySize=<number>] [--dflashBlockSize=<number>|--jetspecBlockSize=<number>] "
                 "[--dsparkScheduler=off|threshold|sps] "
                 "[--dsparkConfidenceThreshold=<float>] "
                 "[--dsparkMinProposalLen=<number>] [--dsparkMaxProposalLen=<number>] "
                 "[--visualPrune] [--visualPruneAlgo=<name>] [--dartReductionRatio=<float>] "
                 "[--dartPivotImageTokens=<number>] [--dartPivotTextTokens=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --checkpointDir           HF/ModelOpt checkpoint dir (required for runtime weight loading)"
              << std::endl;
    std::cerr << "  --draftCheckpointDir      Separate checkpoint for paired speculative drafts" << std::endl;
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
    std::cerr << "  --specDecode              Enable speculative decoding (EAGLE, MTP, DFlash, JetSpec, or DSpark)"
              << std::endl;
    std::cerr << "  --specDraftTopK           Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            DFlash/JetSpec/DSpark default to 1 when omitted" << std::endl;
    std::cerr
        << "                            For DFlash/JetSpec: candidateTopK; 1 is linear, >1 enables branching DDTree"
        << std::endl;
    std::cerr << "  --specDraftStep           Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the current draft proposal; DFlash/JetSpec/DSpark "
                 "require this to be 1"
              << std::endl;
    std::cerr << "  --specVerifySize          Number of tokens in the base verification input (default: 60)"
              << std::endl;
    std::cerr
        << "                            DFlash/JetSpec linear default to block size; DDTree defaults to base budget"
        << std::endl;
    std::cerr << "  --dsparkScheduler         DSpark scheduler mode: off, threshold, or sps (default: off)"
              << std::endl;
    std::cerr << "  --dsparkConfidenceThreshold  DSpark threshold scheduler survival threshold in [0,1]" << std::endl;
    std::cerr << "  --dsparkMinProposalLen    DSpark scheduler minimum proposal length (default: 1)" << std::endl;
    std::cerr << "  --dsparkMaxProposalLen    DSpark scheduler maximum proposal length (default: full block)"
              << std::endl;
    std::cerr << "  --dflashBlockSize         DFlash/JetSpec proposal block size; 0 means infer from engine config"
              << std::endl;
    std::cerr << "\nContext Reuse Options:" << std::endl;
    std::cerr << "  --enableContextReuse      Enable process-local content-addressed context reuse" << std::endl;
    std::cerr << "  --contextCacheMaxRecords  Maximum retained context records (default: 1024)" << std::endl;
    std::cerr << "  --contextCacheRecurrentSnapshotPoolBytes" << std::endl;
    std::cerr << "                            Device byte budget for recurrent/conv snapshots (default: 0;"
              << " required for hybrid reuse)" << std::endl;
    std::cerr << "  --contextCachePartialKVSnapshotPoolBytes" << std::endl;
    std::cerr << "                            Device byte budget for partial-KV snapshots (default: 0;"
              << " required for hybrid attention reuse)" << std::endl;
    std::cerr << "                            KV retention capacity is configured at build time with"
              << " --maxKVPoolPages" << std::endl;
    std::cerr << "  --encoderCacheBudgetBytes  Device byte budget for encoder embedding cache (default: 256 MiB;"
              << " 0 disables)" << std::endl;
    std::cerr << "\nVisual-Token Pruning Options:" << std::endl;
    std::cerr << "  --visualPrune             Enable visual-token pruning (mRoPE VLM prefill, batch 1)" << std::endl;
    std::cerr << "  --visualPruneAlgo         Prune selection algorithm (default: dart)" << std::endl;
    std::cerr << "  --dartReductionRatio      Fraction of visual tokens to remove, in (0, 1) (default: 0.25)"
              << std::endl;
    std::cerr << "  --dartPivotImageTokens    Number of image pivot tokens for DART selection (default: 4)"
              << std::endl;
    std::cerr << "  --dartPivotTextTokens     Number of text pivot tokens for DART selection (default: 4)" << std::endl;
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
        {"checkpointDir", required_argument, 0, LLMInferenceOptionId::CHECKPOINT_DIR},
        {"draftCheckpointDir", required_argument, 0, LLMInferenceOptionId::DRAFT_CHECKPOINT_DIR},
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
        {"jetspecBlockSize", required_argument, 0, LLMInferenceOptionId::DFLASH_BLOCK_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH},
        {"numLogprobs", required_argument, 0, LLMInferenceOptionId::NUM_LOGPROBS},
        {"enableAudioOutput", no_argument, 0, LLMInferenceOptionId::ENABLE_AUDIO_OUTPUT},
        {"talkerEngineDir", required_argument, 0, LLMInferenceOptionId::TALKER_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, LLMInferenceOptionId::CODE2WAV_ENGINE_DIR},
        {"outputAudioDir", required_argument, 0, LLMInferenceOptionId::OUTPUT_AUDIO_DIR},
        {"enableThinkerTalkerStreaming", no_argument, 0, LLMInferenceOptionId::ENABLE_THINKER_TALKER_STREAMING},
        {"tpSize", required_argument, 0, LLMInferenceOptionId::TP_SIZE},
        {"dsparkScheduler", required_argument, 0, LLMInferenceOptionId::DSPARK_SCHEDULER},
        {"dsparkConfidenceThreshold", required_argument, 0, LLMInferenceOptionId::DSPARK_CONFIDENCE_THRESHOLD},
        {"dsparkMinProposalLen", required_argument, 0, LLMInferenceOptionId::DSPARK_MIN_PROPOSAL_LEN},
        {"dsparkMaxProposalLen", required_argument, 0, LLMInferenceOptionId::DSPARK_MAX_PROPOSAL_LEN},
        {"enableContextReuse", no_argument, 0, LLMInferenceOptionId::ENABLE_CONTEXT_REUSE},
        {"contextCacheMaxRecords", required_argument, 0, LLMInferenceOptionId::CONTEXT_CACHE_MAX_RECORDS},
        {"contextCacheRecurrentSnapshotPoolBytes", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_RECURRENT_SNAPSHOT_POOL_BYTES},
        {"contextCachePartialKVSnapshotPoolBytes", required_argument, 0,
            LLMInferenceOptionId::CONTEXT_CACHE_PARTIAL_KV_SNAPSHOT_POOL_BYTES},
        {"visualPrune", no_argument, 0, LLMInferenceOptionId::VISUAL_PRUNE},
        {"visualPruneAlgo", required_argument, 0, LLMInferenceOptionId::VISUAL_PRUNE_ALGO},
        {"dartReductionRatio", required_argument, 0, LLMInferenceOptionId::DART_REDUCTION_RATIO},
        {"dartPivotImageTokens", required_argument, 0, LLMInferenceOptionId::DART_PIVOT_IMAGE_TOKENS},
        {"dartPivotTextTokens", required_argument, 0, LLMInferenceOptionId::DART_PIVOT_TEXT_TOKENS},
        {"encoderCacheBudgetBytes", required_argument, 0, LLMInferenceOptionId::ENCODER_CACHE_BUDGET_BYTES},
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
        case LLMInferenceOptionId::CHECKPOINT_DIR: args.checkpointDir = optarg; break;
        case LLMInferenceOptionId::DRAFT_CHECKPOINT_DIR: args.draftCheckpointDir = optarg; break;
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
                args.specDecodeArgs.draftTopKSet = true;
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
                args.specDecodeArgs.draftStepSet = true;
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
                args.specDecodeArgs.verifySizeSet = true;
                if (args.specDecodeArgs.verifySize < 0)
                {
                    LOG_ERROR("Invalid specVerifySize value: %s (must be non-negative)", optarg);
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
                    LOG_ERROR("Invalid dflashBlockSize/jetspecBlockSize value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dflashBlockSize/jetspecBlockSize value: %s", optarg);
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
        case LLMInferenceOptionId::DSPARK_SCHEDULER:
        {
            std::string const mode{optarg};
            if (mode == "off")
            {
                args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kOff;
            }
            else if (mode == "threshold")
            {
                args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kThreshold;
            }
            else if (mode == "sps")
            {
                args.specDecodeArgs.dsparkSchedulerMode = rt::DSparkSchedulerMode::kSPS;
            }
            else
            {
                LOG_ERROR("Invalid dsparkScheduler value: %s (expected off, threshold, or sps)", optarg);
                return false;
            }
            break;
        }
        case LLMInferenceOptionId::DSPARK_CONFIDENCE_THRESHOLD:
            try
            {
                args.specDecodeArgs.dsparkConfidenceThreshold = std::stof(optarg);
                if (args.specDecodeArgs.dsparkConfidenceThreshold < 0.0F
                    || args.specDecodeArgs.dsparkConfidenceThreshold > 1.0F)
                {
                    LOG_ERROR("Invalid dsparkConfidenceThreshold value: %s (must be in [0,1])", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dsparkConfidenceThreshold value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DSPARK_MIN_PROPOSAL_LEN:
            try
            {
                args.specDecodeArgs.dsparkMinProposalLen = std::stoi(optarg);
                if (args.specDecodeArgs.dsparkMinProposalLen <= 0)
                {
                    LOG_ERROR("Invalid dsparkMinProposalLen value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dsparkMinProposalLen value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DSPARK_MAX_PROPOSAL_LEN:
            try
            {
                args.specDecodeArgs.dsparkMaxProposalLen = std::stoi(optarg);
                if (args.specDecodeArgs.dsparkMaxProposalLen < 0)
                {
                    LOG_ERROR("Invalid dsparkMaxProposalLen value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dsparkMaxProposalLen value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::VISUAL_PRUNE: args.visualPrunerConfig.enabled = true; break;
        case LLMInferenceOptionId::VISUAL_PRUNE_ALGO: args.visualPrunerConfig.algorithm = optarg; break;
        case LLMInferenceOptionId::DART_REDUCTION_RATIO:
            try
            {
                args.visualPrunerConfig.reductionRatio = std::stof(optarg);
                if (args.visualPrunerConfig.reductionRatio <= 0.0F || args.visualPrunerConfig.reductionRatio >= 1.0F)
                {
                    LOG_ERROR("Invalid dartReductionRatio value: %s (must be in (0, 1))", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dartReductionRatio value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DART_PIVOT_IMAGE_TOKENS:
            try
            {
                args.visualPrunerConfig.pivotImageTokens = std::stoi(optarg);
                if (args.visualPrunerConfig.pivotImageTokens < 0)
                {
                    LOG_ERROR("Invalid dartPivotImageTokens value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dartPivotImageTokens value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DART_PIVOT_TEXT_TOKENS:
            try
            {
                args.visualPrunerConfig.pivotTextTokens = std::stoi(optarg);
                if (args.visualPrunerConfig.pivotTextTokens < 0)
                {
                    LOG_ERROR("Invalid dartPivotTextTokens value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid dartPivotTextTokens value: %s", optarg);
                return false;
            }
            break;
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
        case LLMInferenceOptionId::TP_SIZE:
            try
            {
                args.tpSize = std::stoi(optarg);
                if (args.tpSize <= 0)
                {
                    LOG_ERROR("Invalid tpSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid tpSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::ENCODER_CACHE_BUDGET_BYTES:
            if (!parseNonNegativeIntegerOption(
                    "encoderCacheBudgetBytes", optarg, args.contextCacheConfig.encoderEmbeddingCacheBudgetBytes))
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
    if (args.tpSize > 1)
    {
        LOG_INFO("Tensor parallel launch requested: tpSize=%d", args.tpSize);
    }
    if (args.specDecodeArgs.enabled)
    {
        LOG_INFO("Speculative decoding enabled");
        LOG_INFO("DSpark scheduler mode: %d", static_cast<int32_t>(args.specDecodeArgs.dsparkSchedulerMode));
        LOG_INFO("DSpark confidence threshold: %.4f", args.specDecodeArgs.dsparkConfidenceThreshold);
        LOG_INFO("DSpark proposal length range: [%d, %d]", args.specDecodeArgs.dsparkMinProposalLen,
            args.specDecodeArgs.dsparkMaxProposalLen);
    }
    else if (!args.draftCheckpointDir.empty())
    {
        LOG_ERROR("--draftCheckpointDir requires --specDecode");
        return false;
    }

    if (args.contextCacheConfig.enabled)
    {
        LOG_INFO("Context reuse enabled");
        LOG_INFO("Context cache config: maxRecords=%d recurrentSnapshotPoolBytes=%lld partialKVSnapshotPoolBytes=%lld",
            args.contextCacheConfig.maxRecords,
            static_cast<long long>(args.contextCacheConfig.recurrentSnapshotPoolBytes),
            static_cast<long long>(args.contextCacheConfig.partialKvSnapshotPoolBytes));
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

#if defined(EDGELLM_ENABLE_MULTI_DEVICE)
namespace
{

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
[[noreturn]] void abortMpiWorld(int32_t processRank, int32_t worldSize, char const* phase, char const* detail)
{
    LOG_ERROR("[Rank %d/%d] Fatal %s failure: %s. Aborting the MPI world.", processRank, worldSize, phase, detail);
    int32_t const abortResult = MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    if (abortResult != MPI_SUCCESS)
    {
        LOG_ERROR("[Rank %d/%d] MPI_Abort failed with error %d.", processRank, worldSize, abortResult);
    }
    std::abort();
}
#endif

bool allProcessesSucceeded(bool localOk)
{
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    int32_t const localStatus = localOk ? 1 : 0;
    int32_t globalStatus = 0;
    MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalStatus != 0;
#else
    return localOk;
#endif
}

std::vector<RankMemorySummary> collectRankMemorySummaries(
    MemoryMonitor const& memoryMonitor, rt::ParallelLaunchMode launchMode, int32_t tpSize, int32_t processRank)
{
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (launchMode != rt::ParallelLaunchMode::kMpi || tpSize <= 1)
    {
        return {};
    }

    struct RankMemorySample
    {
        int32_t rank{0};
        int32_t device{0};
        size_t peakGpuMemoryBytes{0};
        size_t peakCpuMemoryBytes{0};
    };

    int32_t device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    RankMemorySample localSample{
        processRank, device, memoryMonitor.getPeakGpuMemory(), memoryMonitor.getPeakCpuMemory()};

    std::vector<RankMemorySample> rankSamples;
    if (processRank == 0)
    {
        rankSamples.resize(static_cast<size_t>(tpSize));
    }
    void* receiveBuffer = processRank == 0 ? static_cast<void*>(rankSamples.data()) : nullptr;
    MPI_Gather(&localSample, static_cast<int32_t>(sizeof(RankMemorySample)), MPI_BYTE, receiveBuffer,
        static_cast<int32_t>(sizeof(RankMemorySample)), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (processRank != 0)
    {
        return {};
    }

    std::vector<RankMemorySummary> rankMemory;
    rankMemory.reserve(rankSamples.size());
    for (RankMemorySample const& sample : rankSamples)
    {
        rankMemory.push_back(
            RankMemorySummary{sample.rank, sample.device, sample.peakGpuMemoryBytes, sample.peakCpuMemoryBytes});
    }
    return rankMemory;
#else
    (void) memoryMonitor;
    (void) launchMode;
    (void) tpSize;
    (void) processRank;
    return {};
#endif
}

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
std::vector<rt::ParallelBackendHandles> createMpiBackendHandles(int32_t tpSize, int32_t tpRank)
{
    std::vector<rt::ParallelBackendHandles> backendGroups;
    if (tpSize <= 1)
    {
        return backendGroups;
    }

    try
    {
        rt::NcclUniqueId uniqueId{};
        if (tpRank == 0)
        {
            rt::NcclCollectiveBackend::getUniqueId(uniqueId);
        }
        if (MPI_Bcast(&uniqueId, static_cast<int32_t>(sizeof(rt::NcclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD)
            != MPI_SUCCESS)
        {
            abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", "MPI failed to broadcast the NCCL unique ID");
        }

        void* ncclComm = nullptr;
        rt::NcclCollectiveBackend::initRank(&ncclComm, tpSize, uniqueId, tpRank);

        std::vector<void*> tensorHandles(static_cast<size_t>(tpSize), nullptr);
        tensorHandles[static_cast<size_t>(tpRank)] = ncclComm;
        backendGroups.push_back(rt::ParallelBackendHandles{rt::ParallelType::kTensor, std::move(tensorHandles), true});
    }
    catch (std::exception const& e)
    {
        abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", e.what());
    }
    catch (...)
    {
        abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", "unknown exception");
    }
    return backendGroups;
}
#endif

int runParallelInference(LLMInferenceArgs const& args,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::vector<rt::LLMGenerationRequest>& batchedRequests, rt::ParallelLaunchMode launchMode, int32_t processRank,
    int32_t processDevice, std::vector<rt::ParallelBackendHandles> backendHandles)
{
    if (args.enableAudioOutput || args.enableThinkerTalkerStreaming)
    {
        LOG_ERROR("Parallel inference does not support Qwen3-Omni audio output yet.");
        return EXIT_FAILURE;
    }
    if (args.specDecodeArgs.enabled)
    {
        LOG_ERROR("Tensor-parallel speculative decoding is not supported; omit --specDecode or run with --tpSize 1.");
        return EXIT_FAILURE;
    }
    if (args.contextCacheConfig.enabled)
    {
        LOG_ERROR(
            "Tensor-parallel context reuse is not supported yet; omit --enableContextReuse or run with --tpSize 1.");
        return EXIT_FAILURE;
    }

    rt::ParallelConfig parallelConfig{};
    parallelConfig.tensorParallelSize = args.tpSize;
    parallelConfig.launchMode = launchMode;

    MemoryMonitor memoryMonitor;
    if (args.dumpProfile)
    {
        memoryMonitor.start();
    }

    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        rt::LLMInferenceRuntime::ParallelExecutionConfig runtimeConfig{};
        runtimeConfig.parallelConfig = parallelConfig;
        runtimeConfig.checkpointDir = args.checkpointDir;
        runtimeConfig.draftCheckpointDir = args.draftCheckpointDir;
        if (launchMode == rt::ParallelLaunchMode::kMpi)
        {
            runtimeConfig.localRanks = {processRank};
            runtimeConfig.localRankDevices.emplace(processRank, processDevice);
            runtimeConfig.backendHandles = std::move(backendHandles);
        }
        runtime = std::make_unique<rt::LLMInferenceRuntime>(
            args.engineDir, args.multimodalEngineDir, loraWeightsMap, std::move(runtimeConfig));
        if (args.visualPrunerConfig.enabled)
        {
            runtime->setVisualPrunerConfig(args.visualPrunerConfig);
        }
        bool const captureOk = allProcessesSucceeded(runtime->captureDecodingCUDAGraph(nullptr));
        if (!captureOk)
        {
            LOG_WARNING("Failed to capture CUDA graph for one or more parallel ranks; using normal execution.");
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize parallel execution: %s", e.what());
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (launchMode == rt::ParallelLaunchMode::kMpi)
        {
            abortMpiWorld(processRank, args.tpSize, "parallel runtime initialization", e.what());
        }
#endif
        return EXIT_FAILURE;
    }

    bool const isOutputRank = runtime->ownsGlobalRank(0);
    for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
    {
        rt::LLMGenerationResponse warmupResponse;
        if (!allProcessesSucceeded(runtime->handleRequest(batchedRequests.front(), warmupResponse, nullptr)))
        {
            LOG_ERROR("Parallel warmup run %d/%d failed.", warmupRun + 1, args.warmup);
            return EXIT_FAILURE;
        }
    }

    nlohmann::json outputData;
    if (isOutputRank)
    {
        outputData["input_file"] = args.inputFile;
        outputData["tp_size"] = args.tpSize;
        outputData["launch_mode"] = rt::parallelLaunchModeName(launchMode);
        outputData["responses"] = nlohmann::json::array();
    }

    size_t failedCount = 0;
    int64_t totalGeneratedTokens = 0;
    auto const benchmarkStart = std::chrono::high_resolution_clock::now();

    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        setProfilingEnabled(args.dumpProfile);
        rt::LLMGenerationResponse response;
        bool const requestOk
            = allProcessesSucceeded(runtime->handleRequest(batchedRequests[requestIdx], response, nullptr));
        if (!requestOk)
        {
            ++failedCount;
            LOG_ERROR("Parallel request %zu failed.", requestIdx);
            continue;
        }
        if (!isOutputRank)
        {
            continue;
        }

        rt::LLMGenerationRequest const& rank0Request = batchedRequests[requestIdx];
        for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            bool const hasOutputText = requestOk && batchIdx < response.outputTexts.size();
            std::string const outputText = hasOutputText ? response.outputTexts[batchIdx] : "Request failed";
            auto const* formattedRequest = batchIdx < rank0Request.formattedRequests.size()
                ? &rank0Request.formattedRequests[batchIdx]
                : nullptr;

            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = requestIdx;
            responseJson["batch_idx"] = batchIdx;
            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : batchedRequests[requestIdx].requests[batchIdx].messages)
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
            responseJson["formatted_system_prompt"] = formattedRequest ? formattedRequest->formattedSystemPrompt : "";
            responseJson["formatted_complete_request"]
                = formattedRequest ? formattedRequest->formattedCompleteRequest : "";
            outputData["responses"].push_back(responseJson);

            if (requestOk && batchIdx < response.outputIds.size())
            {
                totalGeneratedTokens += static_cast<int64_t>(response.outputIds[batchIdx].size());
            }
            if (args.dumpOutput && hasOutputText)
            {
                LOG_INFO("[parallel %s] Response %zu batch %zu: %s", rt::parallelLaunchModeName(launchMode), requestIdx,
                    batchIdx, outputText.c_str());
            }
        }
    }

    auto const benchmarkEnd = std::chrono::high_resolution_clock::now();
    double const wallClockMs = std::chrono::duration<double, std::milli>(benchmarkEnd - benchmarkStart).count();
    double const wallClockTokPerSec
        = (wallClockMs > 0.0 && totalGeneratedTokens > 0) ? totalGeneratedTokens * 1000.0 / wallClockMs : 0.0;
    double const wallClockMsPerTok = totalGeneratedTokens > 0 ? wallClockMs / totalGeneratedTokens : 0.0;
    nlohmann::json wallClockData;
    if (isOutputRank)
    {
        wallClockData = {{"tp_size", args.tpSize}, {"generated_tokens", totalGeneratedTokens},
            {"total_time_ms", wallClockMs}, {"tokens_per_second", wallClockTokPerSec},
            {"ms_per_token", wallClockMsPerTok}, {"failed_requests", failedCount}};
    }

    std::vector<RankMemorySummary> rankMemory;
    if (args.dumpProfile)
    {
        setProfilingEnabled(false);
        memoryMonitor.stop();
        rankMemory = collectRankMemorySummaries(memoryMonitor, launchMode, args.tpSize, processRank);
    }

    if (args.dumpProfile && isOutputRank)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Parallel Execution Performance Summary (" << rt::parallelLaunchModeName(launchMode)
                      << ") ===" << std::endl;
        outputPrefillProfile(profileOutput, runtime->getPrefillMetrics());
        if (args.specDecodeArgs.enabled)
        {
            outputSpecDecodeGenerationProfile(profileOutput, runtime->getSpecDecodeGenerationMetrics(),
                runtime->getSpeculativeDecodingStrategyName());
        }
        else
        {
            outputGenerationProfile(profileOutput, runtime->getGenerationMetrics());
        }
        outputMultimodalProfile(profileOutput, runtime->getMultimodalMetrics());
        outputMemoryProfile(profileOutput, memoryMonitor);
        profileOutput << "=== Wall-Clock (end-to-end, prefill + generation) ===" << std::endl;
        profileOutput << "TP Size: " << args.tpSize << std::endl;
        profileOutput << "Total Generated Tokens: " << totalGeneratedTokens << std::endl;
        profileOutput << "Total Wall-Clock Time: " << std::fixed << std::setprecision(2) << wallClockMs << " ms"
                      << std::endl;
        profileOutput << "Wall-Clock Tokens/Second: " << std::fixed << std::setprecision(1) << wallClockTokPerSec
                      << std::endl;
        profileOutput << "Wall-Clock ms/Token: " << std::fixed << std::setprecision(3) << wallClockMsPerTok
                      << std::endl;
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    if (isOutputRank && !args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;
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
            addJsonTimingStages(profileJson);
            addJsonMemorySummary(
                profileJson, memoryMonitor, args.tpSize, rt::parallelLaunchModeName(launchMode), rankMemory);
            profileJson["wall_clock"] = wallClockData;

            std::ofstream profileFile(args.profileOutputFile);
            if (!profileFile.is_open())
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
            profileFile << profileJson.dump(2);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    if (isOutputRank)
    {
        try
        {
            std::ofstream outputFile(args.outputFile);
            if (!outputFile.is_open())
            {
                LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
                return EXIT_FAILURE;
            }
            outputFile << outputData.dump(2);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: %s", e.what());
            return EXIT_FAILURE;
        }

        LOG_INFO("Parallel %s complete: %zu/%zu requests successful, %.2f ms, %.2f tok/s, %.3f ms/token.",
            rt::parallelLaunchModeName(launchMode), batchedRequests.size() - failedCount, batchedRequests.size(),
            wallClockMs, wallClockTokPerSec, wallClockMsPerTok);
    }

    return failedCount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace
#endif

int main(int argc, char* argv[])
{
    int tpSize = 1;
    int tpRank = 0;
#if defined(EDGELLM_ENABLE_MULTI_DEVICE)
    int processDevice = 0;
#endif

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        LOG_ERROR("Failed to initialize MPI.");
        return EXIT_FAILURE;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &tpSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &tpRank);

    MPI_Comm localComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, tpRank, MPI_INFO_NULL, &localComm) != MPI_SUCCESS)
    {
        abortMpiWorld(tpRank, tpSize, "MPI device discovery", "MPI_Comm_split_type failed");
    }
    if (MPI_Comm_rank(localComm, &processDevice) != MPI_SUCCESS || MPI_Comm_free(&localComm) != MPI_SUCCESS)
    {
        abortMpiWorld(tpRank, tpSize, "MPI device discovery", "failed to resolve or release the node-local rank");
    }

    try
    {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (processDevice >= deviceCount)
        {
            throw std::runtime_error(
                format::fmtstr("Node-local rank %d requires CUDA device %d, but this host exposes only %d device(s).",
                    processDevice, processDevice, deviceCount));
        }
        CUDA_CHECK(cudaSetDevice(processDevice));
    }
    catch (std::exception const& e)
    {
        abortMpiWorld(tpRank, tpSize, "CUDA device selection", e.what());
    }
    if (tpSize > 1)
    {
        LOG_INFO("[Rank %d/%d] MPI initialized on node-local CUDA device %d.", tpRank, tpSize, processDevice);
    }
    else
    {
        LOG_INFO("MPI initialized with one process; threaded launch remains available for --tpSize > 1.");
    }
#endif

    NVTX_SCOPED_RANGE(nvtx_main, "llm_inference");
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (tpSize > 1)
        {
            abortMpiWorld(tpRank, tpSize, "argument parsing", "one rank rejected the command line");
        }
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        MPI_Finalize();
#endif
        return EXIT_SUCCESS;
    }
    if (!applyEngineSpecDecodeDefaults(args))
    {
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (tpSize > 1)
        {
            abortMpiWorld(tpRank, tpSize, "runtime option preparation", "failed to resolve engine defaults");
        }
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (tpSize > 1)
    {
        if (args.tpSize > 1 && args.tpSize != tpSize)
        {
            abortMpiWorld(tpRank, tpSize, "parallel configuration", "--tpSize does not match the MPI world size");
        }
        args.tpSize = tpSize;
    }
#endif
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
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (tpSize > 1)
        {
            abortMpiWorld(tpRank, tpSize, "input preparation", e.what());
        }
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (tpSize > 1)
        {
            abortMpiWorld(tpRank, tpSize, "input preparation", "no valid requests were parsed");
        }
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

#if defined(EDGELLM_ENABLE_MULTI_DEVICE)
    if (args.tpSize > 1)
    {
        rt::ParallelLaunchMode launchMode = rt::ParallelLaunchMode::kThread;
        int32_t processRank = 0;
        std::vector<rt::ParallelBackendHandles> backendHandles;
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        if (tpSize > 1)
        {
            launchMode = rt::ParallelLaunchMode::kMpi;
            processRank = tpRank;
            backendHandles = createMpiBackendHandles(args.tpSize, tpRank);
        }
#endif
        int const result = runParallelInference(
            args, loraWeightsMap, batchedRequests, launchMode, processRank, processDevice, std::move(backendHandles));
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        MPI_Finalize();
#endif
        return result;
    }
#elif !defined(EDGELLM_ENABLE_MULTI_DEVICE)
    if (args.tpSize > 1)
    {
        LOG_ERROR("--tpSize > 1 requires building llm_inference with ENABLE_MULTI_DEVICE=ON.");
        return EXIT_FAILURE;
    }
#endif

    bool profilerEnabled = args.dumpProfile;
    MemoryMonitor memoryMonitor;
    // Start memory monitoring at the beginning if profiling is enabled
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    // Create unified runtime (handles both vanilla and speculative decoding modes)
    std::unique_ptr<rt::LLMInferenceRuntime> runtime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto cleanupAfterStreamCreate = [&]() {
        runtime.reset();
        CUDA_CHECK(cudaStreamDestroy(stream));
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
        MPI_Finalize();
#endif
    };

    if (args.specDecodeArgs.enabled)
    {
        rt::SpecDecodeDraftingConfig draftingConfig;
        draftingConfig.draftingTopK = args.specDecodeArgs.draftTopK;
        draftingConfig.draftingStep = args.specDecodeArgs.draftStep;
        draftingConfig.verifySize = args.specDecodeArgs.verifySize;
        draftingConfig.dflashBlockSize = args.specDecodeArgs.dflashBlockSize;
        draftingConfig.dsparkSchedulerMode = args.specDecodeArgs.dsparkSchedulerMode;
        draftingConfig.dsparkConfidenceThreshold = args.specDecodeArgs.dsparkConfidenceThreshold;
        draftingConfig.dsparkMinProposalLen = args.specDecodeArgs.dsparkMinProposalLen;
        draftingConfig.dsparkMaxProposalLen = args.specDecodeArgs.dsparkMaxProposalLen;
        try
        {
            runtime
                = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir, loraWeightsMap,
                    draftingConfig, stream, args.contextCacheConfig, args.checkpointDir, args.draftCheckpointDir);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime with speculative decoding: %s", e.what());
            cleanupAfterStreamCreate();
            return EXIT_FAILURE;
        }
    }
    else
    {
        // Standard vanilla-only mode (no draft model)
        try
        {
            runtime = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir,
                loraWeightsMap, stream, args.contextCacheConfig, args.checkpointDir);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime: %s", e.what());
            cleanupAfterStreamCreate();
            return EXIT_FAILURE;
        }
    }

    if (args.visualPrunerConfig.enabled)
    {
        try
        {
            runtime->setVisualPrunerConfig(args.visualPrunerConfig);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to enable visual-token pruning: %s", e.what());
            cleanupAfterStreamCreate();
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
            ttsRuntime = std::make_unique<rt::Qwen3OmniTTSRuntime>(args.talkerEngineDir, codePredictorDir.string(),
                args.engineDir, /*cloneEncoderDir=*/"", stream, args.checkpointDir);
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
                code2wavRunner = std::make_unique<rt::Code2WavRunner>(code2wavDir.string(), stream, args.checkpointDir);
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
        rt::ContextCacheLookupPolicy const originalLookupPolicy = firstRequest.contextCacheLookupPolicy;
        if (args.contextCacheConfig.enabled)
        {
            firstRequest.contextCacheLookupPolicy = rt::ContextCacheLookupPolicy::kBypass;
        }

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = runtime->handleRequest(firstRequest, warmupResponse, stream, false);

            if (!requestStatus)
            {
                firstRequest.contextCacheLookupPolicy = originalLookupPolicy;
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                cleanupAfterStreamCreate();
                return EXIT_FAILURE;
            }
        }
        firstRequest.contextCacheLookupPolicy = originalLookupPolicy;
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

    // Wall-clock timing for end-to-end latency measurement (includes AllReduce, MPI sync, CPU overhead)
    auto benchmarkStart = std::chrono::high_resolution_clock::now();

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
            if (args.dumpOutput && tpRank == 0)
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

    auto benchmarkEnd = std::chrono::high_resolution_clock::now();

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

    if (args.dumpProfile && (tpRank == 0))
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Performance Summary ===" << std::endl;
        auto prefillMetrics = runtime->getPrefillMetrics();
        auto multimodalMetrics = runtime->getMultimodalMetrics();
        outputPrefillProfile(profileOutput, prefillMetrics);
        if (auto const contextCacheMetrics = runtime->getContextCacheMetrics(); contextCacheMetrics.has_value())
        {
            outputContextCacheProfile(profileOutput, *contextCacheMetrics);
        }
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

        {
            int64_t totalGeneratedTokens = args.specDecodeArgs.enabled
                ? runtime->getSpecDecodeGenerationMetrics().totalGeneratedTokens
                : runtime->getGenerationMetrics().generatedTokens;

            double wallClockMs = std::chrono::duration<double, std::milli>(benchmarkEnd - benchmarkStart).count();
            double wallClockTokPerSec
                = (wallClockMs > 0 && totalGeneratedTokens > 0) ? (totalGeneratedTokens * 1000.0 / wallClockMs) : 0;
            double wallClockMsPerTok = (totalGeneratedTokens > 0) ? (wallClockMs / totalGeneratedTokens) : 0;

            profileOutput << "=== Wall-Clock (end-to-end, prefill + generation) ===" << std::endl;
            profileOutput << "TP Size: " << tpSize << std::endl;
            profileOutput << "Total Generated Tokens: " << totalGeneratedTokens << std::endl;
            profileOutput << "Total Wall-Clock Time: " << std::fixed << std::setprecision(2) << wallClockMs << " ms"
                          << std::endl;
            profileOutput << "Wall-Clock Tokens/Second: " << std::fixed << std::setprecision(1) << wallClockTokPerSec
                          << std::endl;
            profileOutput << "Wall-Clock ms/Token: " << std::fixed << std::setprecision(3) << wallClockMsPerTok
                          << std::endl;
        }

        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // Only rank 0 writes output files to avoid race conditions in parallel mode.
    // tpRank is 0 for single-device, so this is always true in that case.
    if (tpRank == 0)
    {
        // Export profile to JSON file
        if (!args.profileOutputFile.empty())
        {
            try
            {
                nlohmann::json profileJson;

                // Add high-level metrics from unified runtime
                addJsonPrefillSummary(profileJson, runtime->getPrefillMetrics());
                if (auto const contextCacheMetrics = runtime->getContextCacheMetrics(); contextCacheMetrics.has_value())
                {
                    addJsonContextCacheSummary(profileJson, *contextCacheMetrics);
                }
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

                // Add detailed timing stages
                addJsonTimingStages(profileJson);

                // Add memory usage
                addJsonMemorySummary(profileJson, memoryMonitor);

                {
                    int64_t totalGeneratedTokens = args.specDecodeArgs.enabled
                        ? runtime->getSpecDecodeGenerationMetrics().totalGeneratedTokens
                        : runtime->getGenerationMetrics().generatedTokens;

                    double wallClockMs
                        = std::chrono::duration<double, std::milli>(benchmarkEnd - benchmarkStart).count();
                    double wallClockTokPerSec = (wallClockMs > 0 && totalGeneratedTokens > 0)
                        ? (totalGeneratedTokens * 1000.0 / wallClockMs)
                        : 0;
                    double wallClockMsPerTok = (totalGeneratedTokens > 0) ? (wallClockMs / totalGeneratedTokens) : 0;

                    profileJson["wall_clock"] = {{"tp_size", tpSize}, {"generated_tokens", totalGeneratedTokens},
                        {"total_time_ms", wallClockMs}, {"tokens_per_second", wallClockTokPerSec},
                        {"ms_per_token", wallClockMsPerTok}};
                }

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
                    cleanupAfterStreamCreate();
                    return EXIT_FAILURE;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Failed to write profile output file: %s", e.what());
                cleanupAfterStreamCreate();
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
                cleanupAfterStreamCreate();
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: %s", e.what());
            cleanupAfterStreamCreate();
            return EXIT_FAILURE;
        }
    }

    // Explicitly destroy the runtime before MPI teardown.
    // Runtime-owned TP communication must remain valid while TRT/plugin resources are released.
    runtime.reset();
    CUDA_CHECK(cudaStreamDestroy(stream));

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    MPI_Finalize();
#endif

    // Return false if any request failed
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
