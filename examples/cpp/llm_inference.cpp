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
#include "memoryMonitor.h"
#include "mpiUtils.h"
#include "profileFormatter.h"
#include "requestFileParser.h"
#include "runtimeArgs.h"
#include <algorithm>
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
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/inputLimits.h>
#include <tensorrt_edgellm/common/trtUtils.h>
#include <tensorrt_edgellm/common/utf8.h>
#include <tensorrt_edgellm/multimodal/qwen3_omni/code2WavRunner.h>
#include <tensorrt_edgellm/profiling/metrics.h>
#include <tensorrt_edgellm/profiling/nvtx_wrapper.h>
#include <tensorrt_edgellm/profiling/timer.h>
#include <tensorrt_edgellm/runtime/config/llmEngineConfig.h>
#include <tensorrt_edgellm/runtime/llmInferenceRuntime.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/multiDevice/ncclCollectiveBackend.h>
#include <tensorrt_edgellm/runtime/qwen3OmniTTSRuntime.h>
#include <tensorrt_edgellm/runtime/streaming.h>
#include <tensorrt_edgellm/tokenizer/tokenizer.h>
#include <tuple>
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
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915,
    ENABLE_AUDIO_OUTPUT = 916,
    TALKER_ENGINE_DIR = 917,
    CODE2WAV_ENGINE_DIR = 918,
    OUTPUT_AUDIO_DIR = 919,
    ENABLE_THINKER_TALKER_STREAMING = 920,
    NUM_LOGPROBS = 922,
    TP_SIZE = 933,
    CP_SPEC_VERIFY_SIZE = 941,
    JUMP_FORWARD = 947,
};

struct LLMInferenceArgs : exampleUtils::RuntimeArgs
{
    bool help{false};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    // Override parameters (only batchSize, maxGenerateLength, numLogprobs, and jumpForward can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    int32_t numLogprobs{-1};       // -1 means use value from input file
    bool jumpForward{false};       // Set jump_forward on every request that has guided_decoding
    int32_t cpSpecVerifySize{0};

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

    int32_t tpSize{1};
};

void printUsage(char const* programName)
{
    std::cerr
        << "Usage: " << programName
        << " [--help] [--engine-dir=<path to engine directory>] [--multimodal-engine-dir=<path to multimodal engine "
           "directory>] [--input-file=<path to input file>] [--output-file=<path to output file>] "
           "[--dump-profile] [--profile-output-file=<path to profile output file>] [--warmup=<number>] [--debug] "
           "[--dump-output] [--batch-size=<number>] [--max-generate-length=<number>] [--jump-forward] "
           "[--tp-size=<number>]";
    std::cerr << " [--spec-decode] [--spec-draft-top-k=<number>] [--spec-draft-step=<number>] "
                 "[--spec-proposal-tokens=<number>] [--spec-verify-tokens=<number>] "
                 "[--dspark-w2-backend=auto|fp16|fp8|w4a16] "
                 "[--dspark-scheduler=off|threshold|sps] "
                 "[--dspark-confidence-threshold=<float>] "
                 "[--dspark-min-proposal-len=<number>] [--dspark-max-proposal-len=<number>] "
                 "[--visual-prune] [--visual-prune-algo=<name>] [--dart-reduction-ratio=<float>] "
                 "[--dart-pivot-image-tokens=<number>] [--dart-pivot-text-tokens=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --input-file               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engine-dir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodal-engine-dir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --ple-file                Explicit PLE data file; overrides the checkpoint PLE source" << std::endl;
    std::cerr << "  --ple-residency           PLE storage policy: resident (default) or cached" << std::endl;
    std::cerr << "  --offload-embedding       Keep embedding artifact in pinned host memory (same CUDA lookup kernel)"
              << std::endl;
    std::cerr << "  --ple-cache-slots          Cached PLE slots including zero; chosen at runtime initialization"
              << std::endl;
    std::cerr << "                            Distinct valid rows across each execution batch must fit in slots - 1."
              << std::endl;
    std::cerr << "                            Increase slots or reduce the execution chunk/batch if it does not fit."
              << std::endl;
    std::cerr << "                            Smaller I/O batches do not relax this requirement." << std::endl;
    std::cerr << "  --checkpoint-dir           HF/ModelOpt checkpoint dir (required for runtime weight loading)"
              << std::endl;
    std::cerr << "  --draft-checkpoint-dir      Separate checkpoint for paired speculative drafts" << std::endl;
#if IS_TRT_RTX
    std::cerr << "  --rtx-runtime-cache-path     TRT-RTX runtime (JIT-kernel) cache file path (RTX only)." << std::endl;
    std::cerr << "                            Defaults to <engineDir>/runtime.rtxcache." << std::endl;
#endif
    std::cerr << "  --output-file              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dump-profile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profile-output-file       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dump-output              Dump inference output to console" << std::endl;
    std::cerr << "  --batch-size               Override batch size from input file" << std::endl;
    std::cerr << "  --max-generate-length       Override max generate length from input file" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr
        << "  --num-logprobs             Number of top log-probabilities to return per token (0 = disabled, max 50)"
        << std::endl;
    std::cerr << "  --jump-forward             Set jump_forward on every request that has guided_decoding" << std::endl;
    std::cerr
        << "  --spec-decode              Enable speculative decoding (EAGLE, MTP, DFlash, DFlash2, JetSpec, or DSpark)"
        << std::endl;
    std::cerr << "  --spec-draft-top-k           Number of tokens selected per drafting step (default: 10)"
              << std::endl;
    std::cerr << "                            DFlash/JetSpec/DSpark default to 1 when omitted" << std::endl;
    std::cerr
        << "                            For DFlash/JetSpec: candidateTopK; 1 is linear, >1 enables branching DDTree"
        << std::endl;
    std::cerr << "  --spec-draft-step           Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the current draft proposal; DFlash/JetSpec/DSpark "
                 "require this to be 1"
              << std::endl;
    std::cerr
        << "  --spec-proposal-tokens      Maximum draft proposal tokens; omitted values are inferred from the engines"
        << std::endl;
    std::cerr
        << "  --spec-verify-tokens        Number of tokens in the base verification input; omitted values are inferred"
        << std::endl;
    std::cerr << "  --dspark-w2-backend         DSpark W2 backend: auto, fp16, fp8, w4a16 (default: auto)" << std::endl;
    std::cerr << "  --dspark-scheduler         DSpark scheduler mode: off, threshold, or sps (default: off)"
              << std::endl;
    std::cerr << "  --dspark-confidence-threshold  DSpark threshold scheduler survival threshold in [0,1]" << std::endl;
    std::cerr << "  --dspark-min-proposal-len    DSpark scheduler minimum proposal length (default: 1)" << std::endl;
    std::cerr << "  --dspark-max-proposal-len    DSpark scheduler maximum proposal length (default: full block)"
              << std::endl;
    std::cerr << "\nContext Reuse Options:" << std::endl;
    std::cerr << "  --enable-context-reuse      Enable process-local content-addressed context reuse" << std::endl;
    std::cerr << "  --context-cache-max-records  Maximum retained context records (default: 1024)" << std::endl;
    std::cerr << "  --context-cache-max-snapshots  Total snapshots kept on a model with recurrent layers (default: 4 x"
              << " max batch size)" << std::endl;
    std::cerr << "                            KV retention capacity is configured at build time with"
              << " --max-kv-pool-pages" << std::endl;
    std::cerr << "  --encoder-cache-budget-bytes  Device byte budget for encoder embedding cache (default: 256 MiB;"
              << " 0 disables)" << std::endl;
    std::cerr << "\nVisual-Token Pruning Options:" << std::endl;
    std::cerr << "  --visual-prune             Enable visual-token pruning (mRoPE VLM prefill)" << std::endl;
    std::cerr << "  --visual-prune-algo         Prune selection algorithm (default: dart)" << std::endl;
    std::cerr << "  --dart-reduction-ratio      Fraction of visual tokens to remove, in (0, 1) (default: 0.25)"
              << std::endl;
    std::cerr << "  --dart-pivot-image-tokens    Number of image pivot tokens for DART selection (default: 4)"
              << std::endl;
    std::cerr << "  --dart-pivot-text-tokens     Number of text pivot tokens for DART selection (default: 4)"
              << std::endl;
    std::cerr << "\nQwen3-Omni Audio Output Options:" << std::endl;
    std::cerr << "  --enable-audio-output       Enable audio output from Thinker hidden states" << std::endl;
    std::cerr << "  --talker-engine-dir         Path to Talker engine directory" << std::endl;
    std::cerr << "  --code2wav-engine-dir       Path to Code2Wav engine directory (optional)" << std::endl;
    std::cerr << "  --output-audio-dir          Directory to save generated audio (.wav) files" << std::endl;
    std::cerr << "  --cp-spec-verify-size        CodePredictor speculative decoding verify window: 1 committed"
              << std::endl;
    std::cerr << "                            position plus N-1 drafted RVQ depths. 0 disables (default)." << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    std::vector<option> inferenceOptions = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"input-file", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},

        {"output-file", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"dump-profile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profile-output-file", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dump-output", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"cp-spec-verify-size", required_argument, 0, LLMInferenceOptionId::CP_SPEC_VERIFY_SIZE},
        {"cpSpecVerifySize", required_argument, 0, LLMInferenceOptionId::CP_SPEC_VERIFY_SIZE},
        // deprecated alias
        // deprecated alias
        // deprecated alias

        // deprecated alias

        {"batch-size", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"max-generate-length", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH},
        {"num-logprobs", required_argument, 0, LLMInferenceOptionId::NUM_LOGPROBS},
        {"numLogprobs", required_argument, 0, LLMInferenceOptionId::NUM_LOGPROBS},
        {"jump-forward", no_argument, 0, LLMInferenceOptionId::JUMP_FORWARD},
        {"jumpForward", no_argument, 0, LLMInferenceOptionId::JUMP_FORWARD},
        {"enable-audio-output", no_argument, 0, LLMInferenceOptionId::ENABLE_AUDIO_OUTPUT},
        {"enableAudioOutput", no_argument, 0, LLMInferenceOptionId::ENABLE_AUDIO_OUTPUT},
        {"talker-engine-dir", required_argument, 0, LLMInferenceOptionId::TALKER_ENGINE_DIR},
        {"talkerEngineDir", required_argument, 0, LLMInferenceOptionId::TALKER_ENGINE_DIR},
        {"code2wav-engine-dir", required_argument, 0, LLMInferenceOptionId::CODE2WAV_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, LLMInferenceOptionId::CODE2WAV_ENGINE_DIR},
        {"output-audio-dir", required_argument, 0, LLMInferenceOptionId::OUTPUT_AUDIO_DIR},
        {"outputAudioDir", required_argument, 0, LLMInferenceOptionId::OUTPUT_AUDIO_DIR},
        {"enable-thinker-talker-streaming", no_argument, 0, LLMInferenceOptionId::ENABLE_THINKER_TALKER_STREAMING},
        {"enableThinkerTalkerStreaming", no_argument, 0, LLMInferenceOptionId::ENABLE_THINKER_TALKER_STREAMING},
        {"tp-size", required_argument, 0, LLMInferenceOptionId::TP_SIZE},
        {"tpSize", required_argument, 0, LLMInferenceOptionId::TP_SIZE},

        {0, 0, 0, 0}};

    auto const runtimeOptions = exampleUtils::runtimeOptions();
    inferenceOptions.insert(inferenceOptions.end() - 1, runtimeOptions.begin(), runtimeOptions.end());
    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions.data(), nullptr)) != -1)
    {
        if (auto const parsed = exampleUtils::parseRuntimeOption(opt, optarg, args))
        {
            if (!*parsed)
            {
                return false;
            }
            continue;
        }
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
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
        case LLMInferenceOptionId::JUMP_FORWARD: args.jumpForward = true; break;
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
        case LLMInferenceOptionId::CP_SPEC_VERIFY_SIZE: args.cpSpecVerifySize = std::stoi(optarg); break;
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
        default: return false;
        }
    }

    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --input-file is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engine-dir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --output-file is required");
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
        LOG_ERROR("--draft-checkpoint-dir requires --spec-decode");
        return false;
    }

    if (args.contextCacheConfig.enabled)
    {
        LOG_INFO("Context reuse enabled");
        LOG_INFO("Context cache config: maxRecords=%d maxSnapshots=%d", args.contextCacheConfig.maxRecords,
            args.contextCacheConfig.maxSnapshots);
    }

    if (args.enableAudioOutput)
    {
        if (args.talkerEngineDir.empty())
        {
            LOG_ERROR("--talker-engine-dir is required when --enable-audio-output is set");
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
            LOG_ERROR("--enable-thinker-talker-streaming requires --talker-engine-dir");
            return false;
        }
        LOG_INFO("Thinker-Talker streaming enabled (single CUDA stream)");
    }

    if (args.debug)
    {
        gLogger.setLevel(trt_edgellm::logger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(trt_edgellm::logger::Severity::kINFO);
    }

    return true;
}

// Thin wrapper around the shared parser in common/requestFileParser.h.
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1,
    int32_t numLogprobsOverride = -1, bool jumpForwardOverride = false, LLMInferenceArgs* argsOut = nullptr)
{
    auto result = exampleUtils::parseRequestFile(
        inputFilePath, batchSizeOverride, maxGenerateLengthOverride, numLogprobsOverride, jumpForwardOverride);

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
            // CLI --enable-thinker-talker-streaming takes precedence when set; JSON fills the rest.
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

using example_utils::allProcessesSucceeded;
using example_utils::collectRankMemorySummaries;
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
using example_utils::abortMpiWorld;
using example_utils::createMpiBackendHandles;
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
        LOG_ERROR("Tensor-parallel speculative decoding is not supported; omit --spec-decode or run with --tp-size 1.");
        return EXIT_FAILURE;
    }
    if (args.contextCacheConfig.enabled)
    {
        LOG_ERROR(
            "Tensor-parallel context reuse is not supported yet; omit --enable-context-reuse or run with --tp-size 1.");
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
        auto options = exampleUtils::makeDeploymentOptions(args, loraWeightsMap);
        options.parallelConfig = parallelConfig;
        rt::RuntimeResources resources{};
        if (launchMode == rt::ParallelLaunchMode::kMpi)
        {
            resources.localRanks = {processRank};
            resources.localRankDevices.emplace(processRank, processDevice);
            resources.backendHandles = std::move(backendHandles);
        }
        runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(options), std::move(resources));
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
            if (batchIdx < response.outputIds.size())
            {
                responseJson["generated_token_count"] = response.outputIds[batchIdx].size();
            }
            if (batchIdx < response.specVerifyCounts.size())
            {
                responseJson["spec_verify_count"] = response.specVerifyCounts[batchIdx];
            }
            if (batchIdx < response.specAcceptanceLengths.size())
            {
                responseJson["spec_acceptance_length"] = response.specAcceptanceLengths[batchIdx];
            }
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
    example_utils::MpiProcessInfo mpiInfo;
    if (!example_utils::initializeMpiProcess(argc, argv, mpiInfo))
    {
        return EXIT_FAILURE;
    }
    int const tpSize = mpiInfo.worldSize;
    int const tpRank = mpiInfo.processRank;
#if defined(EDGELLM_ENABLE_MULTI_DEVICE)
    int const processDevice = mpiInfo.processDevice;
#endif
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (tpSize > 1)
    {
        LOG_INFO("[Rank %d/%d] MPI initialized on node-local CUDA device %d.", tpRank, tpSize, mpiInfo.processDevice);
    }
    else
    {
        LOG_INFO("MPI initialized with one process; threaded launch remains available for --tp-size > 1.");
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
    if (!exampleUtils::applyEngineSpecDecodeDefaults(args))
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
            abortMpiWorld(tpRank, tpSize, "parallel configuration", "--tp-size does not match the MPI world size");
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
        std::tie(loraWeightsMap, batchedRequests) = parseInputFile(
            args.inputFile, args.batchSize, args.maxGenerateLength, args.numLogprobs, args.jumpForward, &args);
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

#if defined(_WIN32)
    if (args.tpSize > 1)
    {
        LOG_ERROR("Multi-device inference is not supported on Windows.");
        return EXIT_FAILURE;
    }
#elif defined(EDGELLM_ENABLE_MULTI_DEVICE)
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
#else
    if (args.tpSize > 1)
    {
        LOG_ERROR("--tp-size > 1 requires building llm_inference with ENABLE_MULTI_DEVICE=ON.");
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

    try
    {
        auto options = exampleUtils::makeDeploymentOptions(args, loraWeightsMap);
        runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(options), stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize runtime: %s", e.what());
        cleanupAfterStreamCreate();
        return EXIT_FAILURE;
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
                args.engineDir, /*cloneEncoderDir=*/"", stream, args.checkpointDir, args.cpSpecVerifySize);
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
                    rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                omniReq.thinkerPrefillEmbeds = &perBatchEmbedViews[batchIdx];

                if (hiddenStatesAll != nullptr)
                {
                    __half* hiddenBase = static_cast<__half*>(const_cast<void*>(hiddenStatesAll->rawPointer()))
                        + static_cast<int64_t>(batchIdx) * batchStride;
                    perBatchHiddenViews[batchIdx]
                        = rt::Tensor(hiddenBase, rt::Coords{1, static_cast<int64_t>(prefillLen), H},
                            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
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
            if (batchIdx < response.outputIds.size())
            {
                responseJson["generated_token_count"] = response.outputIds[batchIdx].size();
            }
            if (batchIdx < response.specVerifyCounts.size())
            {
                responseJson["spec_verify_count"] = response.specVerifyCounts[batchIdx];
            }
            if (batchIdx < response.specAcceptanceLengths.size())
            {
                responseJson["spec_acceptance_length"] = response.specAcceptanceLengths[batchIdx];
            }
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
