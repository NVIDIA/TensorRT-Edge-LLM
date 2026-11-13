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

#include "audioWriter.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "requestFileParser.h"
#include "requestVocabulary.h"
#include "runtimeArgs.h"
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/safetensorsUtils.h>
#include <tensorrt_edgellm/common/trtUtils.h>
#include <tensorrt_edgellm/profiling/timer.h>
#include <tensorrt_edgellm/runtime/audioLoader.h>
#include <tensorrt_edgellm/runtime/imageUtils.h>
#include <tensorrt_edgellm/runtime/llmRunner.h>
#include <tensorrt_edgellm/runtime/multiStage/multiStageRuntime.h>
#include <tensorrt_edgellm/runtime/multiStage/route.h>
#include <tensorrt_edgellm/runtime/multiStage/stageRegistry.h>
#include <tensorrt_edgellm/runtime/state/sharedResources.h>
#include <tensorrt_edgellm/runtime/streaming.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ms = trt_edgellm::rt::multiStage;
using trt_edgellm::rt::AudioStreamChannel;
using trt_edgellm::rt::SharedResources;
using trt_edgellm::rt::StreamChannel;
using trt_edgellm::rt::VideoStreamChannel;

namespace
{

enum OptionId : int
{
    kHelp = 1200,
    kConfig = 1201,
    kPrompt = 1202,
    kAudio = 1203,
    kOutAudio = 1204,
    kVoice = 1205,
    kSampling = 1206,
    kInputFile = 1207,
    kOutputFile = 1208,
    kOutputAudioDir = 1209,
    kWarmup = 1210,
    kDumpProfile = 1211,
    kProfileOutputFile = 1212,
    kBatchSize = 1213,
    kMaxGenerateLength = 1214,
    kNumLogprobs = 1215,
    kDebug = 1216,
    kDumpOutput = 1217,
    kEnableAudioOutput = 1218,
    kCpSpecVerifySize = 1219,
    kChunkFrames = 1220,
    kWholeUtterance = 1221,
    kTokenizerDir = 1222,
    kValidateConfig = 1226,
    kStyle = 1227,
    kOutVideo = 1228,
    kOutputVideoDir = 1229,
    //! The other binaries' flags this one refuses, each with its reason.
    kNamedByGraph = 1223,
    kThinkerTalkerStreaming = 1224,
    kTensorParallel = 1225,
};

struct Args
{
    std::string configDir;
    //! Check the graph against the runners this binary links, then exit.
    bool validateConfig{false};
    std::string prompt;
    std::string voice;
    std::string sampling;
    std::string audioPath;
    std::string outAudioPath;
    std::string outVideoPath;
    std::string inputFile;
    std::string outputFile;
    std::string outputAudioDir;
    std::string outputVideoDir;
    //! Unmeasured runs of the first request. Engine, plugin and CuTe-DSL warmup and the lazy
    //! CUDA module load all land in the first request otherwise.
    int32_t warmup{0};
    //! Requests per cohort, at most. A request a cohort cannot seat leads a later one.
    int32_t batchSize{1};
    bool batchSizeGiven{false}; //!< Otherwise the input file's batch_size, as llm_inference reads it.
    //! Per-stage GPU timing. Off by default: the timers synchronise.
    bool dumpProfile{false};
    //! Where the profile lands. Naming it implies --dump-profile.
    std::string profileOutputFile;
    std::string style;
    std::vector<std::vector<int32_t>> speakerCodes;
    std::vector<std::string> framePaths;
    double fps{1.0};
    bool isVideo{false};
    std::string refAudioPath;
    bool help{false};
    //! llm_inference's deployment flags, applied to every llm.backbone stage.
    trt_edgellm::exampleUtils::RuntimeArgs runtime;
    std::set<int> runtimeFlags; //!< Which of them were passed.
    //! Batch mode: overrides of the input file's values, as llm_inference takes them. -1 keeps the file's.
    int64_t maxGenerateLength{-1};
    int32_t numLogprobs{-1};
    //! This request as llm_inference's parser read it. Null unless the backbone takes parsed requests.
    trt_edgellm::rt::LLMGenerationRequest const* parsedRequest{nullptr};
    bool debug{false};
    bool dumpOutput{false};
    //! Attach an audio sink with nowhere to write, so speech is still generated and timed.
    bool enableAudioOutput{false};
    //! Talker and vocoder overrides of the graph; -1, false or empty keeps the graph's.
    int32_t cpSpecVerifySize{-1};
    int32_t chunkFrames{-1};
    bool wholeUtterance{false};
    std::string tokenizerDir;
};

//! Backs every payload binding. The runtime borrows these until take(), so they have to
//! outlive the request rather than the expression that built it.
struct PayloadStorage
{
    std::string prompt;
    std::vector<int32_t> promptIds;
    std::string voice;
    std::string sampling;
    std::string style;
    std::vector<int32_t> speakerCodes;
    int64_t speakerCodeFrames{0};
    std::string vision;
    trt_edgellm::rt::imageUtils::ImageData pixels;
    trt_edgellm::rt::audio::AudioPCM audio;
    trt_edgellm::rt::audio::AudioPCM refAudio;
};

//! The rate LlmRunner tags request:audio with.
constexpr int32_t kAudioSampleRate = 16000;
//! What the reference voice encoders were trained at.
constexpr int32_t kRefAudioSampleRate = 24000;
//! What the vocoder emits.
constexpr double kAudioOutputSampleRate = 24000.0;

void usage(char const* program)
{
    std::cout << "Usage: " << program << " --config <dir> [--prompt <text>] [--audio <in.wav>] [--out-audio <out.wav>]"
              << "\n\n"
              << "  --config <dir>      Directory holding multistage_config.json and the stage artifacts.\n"
              << "  --validate-config   Check multistage_config.json against the runners this binary links, then\n"
              << "                      exit. Loads no plugin or engine.\n"
              << "  --prompt <text>     Text this request supplies as 'request:messages'.\n"
              << "  --audio <file>      Audio this request supplies as 'request:audio'.\n"
              << "  --voice <name>      Speaker this request supplies as 'request:voice'.\n"
              << "  --sampling <json>   Sampling defaults; an input file's own keys override them, e.g.\n"
              << "                      '{\"temperature\":0.7,\"talker_temperature\":0.9}'.\n"
              << "  --style <json>      Style this request supplies as 'request:style', e.g.\n"
              << "                      '{\"height\":768,\"width\":1344,\"num_frames\":124}'.\n"
              << "  --input-file <json> Batch mode: an llm_inference / qwen3_tts_inference input file.\n"
              << "  --output-file <json> Batch mode: predictions the accuracy harness reads.\n"
              << "  --output-audio-dir <d> Batch mode: writes audio_req<i>.wav and rvq_req<i>.safetensors into\n"
              << "                      this directory.\n"
              << "  --output-video-dir <d> Batch mode: writes video_req<i>.rgb into this directory.\n"
              << "  --warmup <n>        Batch mode: run the first request n times before measuring.\n"
              << "  --batch-size <n>    Batch mode: run up to n requests as one cohort, overriding the file's\n"
              << "                      batch_size. A request the cohort cannot seat leads a later one.\n"
              << "  --dump-profile      Batch mode: print GPU stage timings, peak memory and\n"
              << "                      wall clock as JSON.\n"
              << "  --profile-output-file <json>  Write the profile summary here; implies --dump-profile.\n"
              << "  --dump-output       Batch mode: print each request's output text.\n"
              << "  --out-audio <file>  Single request: WAV for the audio terminal.\n"
              << "  --out-video <file>  Single request: raw 8-bit RGB frames for the video terminal.\n"
              << "  --enable-audio-output Generate speech without writing it anywhere. A graph with no text\n"
              << "                      terminal always does; otherwise audio stages run only for a request\n"
              << "                      that has somewhere to send it.\n"
              << "  --max-generate-length <n> Batch mode: decode cap for every request, overriding the file.\n"
              << "  --num-logprobs <n>  Batch mode: top-n logprobs per generated token, overriding the file.\n"
              << "  --cp-spec-verify-size <n>  Code predictor verify window for every talker; 0 turns it off.\n"
              << "  --tokenizer-dir <d> Every talker's tokenizer_dir: where it looks up its special tokens and, with\n"
              << "                      no backbone, renders requests; a talker behind a backbone also reads\n"
              << "                      embedding.safetensors there.\n"
              << "  --chunk-frames <n>  Codec frames per vocoder window.\n"
              << "  --whole-utterance   Vocode each utterance once it is complete, in one pass when it fits\n"
              << "                      the vocoder engine.\n"
              << "  --debug             Verbose logging.\n"
              << "  llm_inference's deployment flags (--spec-decode and the --spec-*/--dspark-* settings,\n"
              << "  --draft-checkpoint-dir, --enable-context-reuse, --context-cache-*, --encoder-cache-budget-bytes,\n"
              << "  --visual-prune, --dart-*, --ple-*, --offload-embedding, --rtx-runtime-cache-path) apply to every\n"
              << "  llm.backbone stage, and --checkpoint-dir to every stage that loads weights; each overrides\n"
              << "  the graph. The camelCase spellings the other binaries take, such as --inputFile, work too.\n"
              << "  --help\n\n"
              << "The topology comes from multistage_config.json; which of its stages run follows from\n"
              << "the inputs the request carries and the sinks attached.\n";
}

//! Parses all of @p text as an integer of at least @p minimum, or says why not and returns false.
bool parseInteger(char const* flag, char const* text, int32_t minimum, int32_t& out)
{
    std::string_view const view{text};
    int32_t value{};
    auto const [end, error] = std::from_chars(view.data(), view.data() + view.size(), value);
    if (error != std::errc{} || end != view.data() + view.size() || value < minimum)
    {
        std::cerr << flag << " must be an integer >= " << minimum << ", got '" << view << "'\n";
        return false;
    }
    out = value;
    return true;
}

bool parseArgs(Args& args, int argc, char* argv[])
{
    // Hyphenated first; the camelCase twins keep scripts written for the other binaries working.
    std::vector<option> options = {{"help", no_argument, nullptr, OptionId::kHelp},
        {"config", required_argument, nullptr, OptionId::kConfig},
        {"validate-config", no_argument, nullptr, OptionId::kValidateConfig},
        {"prompt", required_argument, nullptr, OptionId::kPrompt},
        {"audio", required_argument, nullptr, OptionId::kAudio},
        {"voice", required_argument, nullptr, OptionId::kVoice},
        {"sampling", required_argument, nullptr, OptionId::kSampling},
        {"style", required_argument, nullptr, OptionId::kStyle},
        {"input-file", required_argument, nullptr, OptionId::kInputFile},
        {"inputFile", required_argument, nullptr, OptionId::kInputFile},
        {"output-file", required_argument, nullptr, OptionId::kOutputFile},
        {"outputFile", required_argument, nullptr, OptionId::kOutputFile},
        {"output-audio-dir", required_argument, nullptr, OptionId::kOutputAudioDir},
        {"outputAudioDir", required_argument, nullptr, OptionId::kOutputAudioDir},
        {"out-audio", required_argument, nullptr, OptionId::kOutAudio},
        {"out-video", required_argument, nullptr, OptionId::kOutVideo},
        {"output-video-dir", required_argument, nullptr, OptionId::kOutputVideoDir},
        {"outputVideoDir", required_argument, nullptr, OptionId::kOutputVideoDir},
        {"warmup", required_argument, nullptr, OptionId::kWarmup},
        {"batch-size", required_argument, nullptr, OptionId::kBatchSize},
        {"batchSize", required_argument, nullptr, OptionId::kBatchSize},
        {"dump-profile", no_argument, nullptr, OptionId::kDumpProfile},
        {"dumpProfile", no_argument, nullptr, OptionId::kDumpProfile},
        {"profile-output-file", required_argument, nullptr, OptionId::kProfileOutputFile},
        {"profileOutputFile", required_argument, nullptr, OptionId::kProfileOutputFile},
        {"max-generate-length", required_argument, nullptr, OptionId::kMaxGenerateLength},
        {"maxGenerateLength", required_argument, nullptr, OptionId::kMaxGenerateLength},
        {"num-logprobs", required_argument, nullptr, OptionId::kNumLogprobs},
        {"numLogprobs", required_argument, nullptr, OptionId::kNumLogprobs},
        {"debug", no_argument, nullptr, OptionId::kDebug}, {"dump-output", no_argument, nullptr, OptionId::kDumpOutput},
        {"dumpOutput", no_argument, nullptr, OptionId::kDumpOutput},
        {"enable-audio-output", no_argument, nullptr, OptionId::kEnableAudioOutput},
        {"enableAudioOutput", no_argument, nullptr, OptionId::kEnableAudioOutput},
        {"cp-spec-verify-size", required_argument, nullptr, OptionId::kCpSpecVerifySize},
        {"cpSpecVerifySize", required_argument, nullptr, OptionId::kCpSpecVerifySize},
        {"chunk-frames", required_argument, nullptr, OptionId::kChunkFrames},
        {"chunkFrames", required_argument, nullptr, OptionId::kChunkFrames},
        {"whole-utterance", no_argument, nullptr, OptionId::kWholeUtterance},
        {"wholeUtterance", no_argument, nullptr, OptionId::kWholeUtterance},
        {"tokenizer-dir", required_argument, nullptr, OptionId::kTokenizerDir},
        {"tokenizerDir", required_argument, nullptr, OptionId::kTokenizerDir},
        {"talker-engine-dir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"talkerEngineDir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"code2wav-engine-dir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"code2wavEngineDir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"clone-encoder-dir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"cloneEncoderDir", required_argument, nullptr, OptionId::kNamedByGraph},
        {"enable-thinker-talker-streaming", no_argument, nullptr, OptionId::kThinkerTalkerStreaming},
        {"enableThinkerTalkerStreaming", no_argument, nullptr, OptionId::kThinkerTalkerStreaming},
        {"tp-size", required_argument, nullptr, OptionId::kTensorParallel},
        {"tpSize", required_argument, nullptr, OptionId::kTensorParallel}};
    auto const runtimeOptions = trt_edgellm::exampleUtils::runtimeOptions();
    options.insert(options.end(), runtimeOptions.begin(), runtimeOptions.end());
    options.push_back({nullptr, 0, nullptr, 0});

    int opt = 0;
    int index = 0;
    while ((opt = getopt_long(argc, argv, "", options.data(), &index)) != -1)
    {
        if (auto const parsed = trt_edgellm::exampleUtils::parseRuntimeOption(opt, optarg, args.runtime))
        {
            if (!*parsed)
            {
                return false;
            }
            args.runtimeFlags.insert(opt);
            continue;
        }
        switch (opt)
        {
        case OptionId::kHelp: args.help = true; return true;
        case OptionId::kConfig: args.configDir = optarg; break;
        case OptionId::kValidateConfig: args.validateConfig = true; break;
        case OptionId::kPrompt: args.prompt = optarg; break;
        case OptionId::kAudio: args.audioPath = optarg; break;
        case OptionId::kVoice: args.voice = optarg; break;
        case OptionId::kSampling: args.sampling = optarg; break;
        case OptionId::kStyle: args.style = optarg; break;
        case OptionId::kInputFile: args.inputFile = optarg; break;
        case OptionId::kOutputFile: args.outputFile = optarg; break;
        case OptionId::kOutputAudioDir: args.outputAudioDir = optarg; break;
        case OptionId::kOutAudio: args.outAudioPath = optarg; break;
        case OptionId::kOutVideo: args.outVideoPath = optarg; break;
        case OptionId::kOutputVideoDir: args.outputVideoDir = optarg; break;
        case OptionId::kWarmup:
            if (!parseInteger("--warmup", optarg, 0, args.warmup))
            {
                return false;
            }
            break;
        case OptionId::kBatchSize:
            if (!parseInteger("--batch-size", optarg, 1, args.batchSize))
            {
                return false;
            }
            args.batchSizeGiven = true;
            break;
        case OptionId::kMaxGenerateLength:
        {
            int32_t value = 0;
            if (!parseInteger("--max-generate-length", optarg, 1, value))
            {
                return false;
            }
            args.maxGenerateLength = value;
            break;
        }
        case OptionId::kNumLogprobs:
            if (!parseInteger("--num-logprobs", optarg, 0, args.numLogprobs))
            {
                return false;
            }
            break;
        case OptionId::kDumpProfile: args.dumpProfile = true; break;
        case OptionId::kProfileOutputFile:
            args.profileOutputFile = optarg;
            args.dumpProfile = true;
            break;
        case OptionId::kDebug: args.debug = true; break;
        case OptionId::kDumpOutput: args.dumpOutput = true; break;
        case OptionId::kEnableAudioOutput: args.enableAudioOutput = true; break;
        case OptionId::kCpSpecVerifySize:
            if (!parseInteger("--cp-spec-verify-size", optarg, 0, args.cpSpecVerifySize))
            {
                return false;
            }
            break;
        case OptionId::kChunkFrames:
            if (!parseInteger("--chunk-frames", optarg, 1, args.chunkFrames))
            {
                return false;
            }
            break;
        case OptionId::kWholeUtterance: args.wholeUtterance = true; break;
        case OptionId::kTokenizerDir: args.tokenizerDir = optarg; break;
        case OptionId::kNamedByGraph:
            std::cerr << "--" << options[static_cast<size_t>(index)].name
                      << " does not apply; the graph names every engine directory\n";
            return false;
        case OptionId::kThinkerTalkerStreaming:
            std::cerr << "--" << options[static_cast<size_t>(index)].name
                      << ": the talker starts once the backbone has finished its reply\n";
            return false;
        case OptionId::kTensorParallel:
            std::cerr << "--" << options[static_cast<size_t>(index)].name
                      << ": tensor-parallel backbones are not supported\n";
            return false;
        default: return false;
        }
    }
    if (!args.runtime.engineDir.empty() || !args.runtime.multimodalEngineDir.empty())
    {
        std::cerr << "--engine-dir and --multimodal-engine-dir do not apply; the graph names every engine directory\n";
        return false;
    }
    return true;
}

namespace eu = trt_edgellm::exampleUtils;

//! The graph's spec_decode settings, overlaid with the flags passed and completed from the engine the
//! way llm_inference resolves its own defaults.
nlohmann::json specDecodeFor(ms::StageConfig const& stage, std::string const& configDir, Args const& args)
{
    auto spec = stage.config.value("spec_decode", nlohmann::json::object());
    eu::RuntimeArgs flags = args.runtime;
    auto& cli = flags.specDecodeArgs;
    auto const given = [&args](int id) { return args.runtimeFlags.count(id) != 0; };
    auto const take = [&](int id, char const* key, int32_t& field, bool& set) {
        if (!given(id) && spec.contains(key))
        {
            field = spec.at(key).get<int32_t>();
            set = true;
        }
    };
    take(eu::SPEC_DRAFT_TOP_K, "draft_top_k", cli.draftTopK, cli.draftTopKSet);
    take(eu::SPEC_DRAFT_STEP, "draft_step", cli.draftStep, cli.draftStepSet);
    take(eu::SPEC_PROPOSAL_TOKENS, "proposal_tokens", cli.proposalTokens, cli.proposalTokensSet);
    take(eu::SPEC_VERIFY_TOKENS, "verify_tokens", cli.verifyTokens, cli.verifyTokensSet);
    cli.enabled = true;
    flags.engineDir = (std::filesystem::path(configDir) / stage.artifact).string();
    if (!eu::applyEngineSpecDecodeDefaults(flags))
    {
        throw std::runtime_error("cannot resolve speculative decoding settings for stage '" + stage.id + "'");
    }
    // What llm_inference hands its runtime: the top-k always, each count only once it is set.
    spec["draft_top_k"] = cli.draftTopK;
    auto const keep = [&spec](char const* key, bool set, int32_t value) {
        if (set)
        {
            spec[key] = value;
        }
        else
        {
            spec.erase(key);
        }
    };
    keep("draft_step", cli.draftStepSet, cli.draftStep);
    keep("proposal_tokens", cli.proposalTokensSet, cli.proposalTokens);
    keep("verify_tokens", cli.verifyTokensSet, cli.verifyTokens);
    if (given(eu::DSPARK_W2_BACKEND))
    {
        spec["dspark_w2_backend"] = trt_edgellm::rt::dsparkW2BackendName(cli.dsparkW2Backend);
    }
    if (given(eu::DSPARK_SCHEDULER))
    {
        using Mode = trt_edgellm::rt::DSparkSchedulerMode;
        spec["dspark_scheduler"] = cli.dsparkSchedulerMode == Mode::kOff
            ? "off"
            : (cli.dsparkSchedulerMode == Mode::kThreshold ? "threshold" : "sps");
    }
    if (given(eu::DSPARK_CONFIDENCE_THRESHOLD))
    {
        spec["dspark_confidence_threshold"] = cli.dsparkConfidenceThreshold;
    }
    if (given(eu::DSPARK_MIN_PROPOSAL_LEN))
    {
        spec["dspark_min_proposal_len"] = cli.dsparkMinProposalLen;
    }
    if (given(eu::DSPARK_MAX_PROPOSAL_LEN))
    {
        spec["dspark_max_proposal_len"] = cli.dsparkMaxProposalLen;
    }
    return spec;
}

//! Whether any stage of @p config takes the request field @p field.
bool readsRequestField(ms::MultiStageConfig const& config, std::string const& field)
{
    for (auto const& stage : config.stages)
    {
        for (auto const& input : stage.inputs)
        {
            if (input.from == field)
            {
                return true;
            }
        }
    }
    return false;
}

constexpr char const* kBackboneRunner = "llm.backbone";
constexpr char const* kTalkerRunner = "qwen3_omni.talker";
constexpr char const* kVocoderRunner = "qwen3_omni.code2wav";

/*!
 * @brief The graph, with the command line's deployment flags written onto the stages they apply to.
 *
 * A flag that was not passed leaves the graph's setting as it is.
 *
 * @param reportCodes Ask every talker to report its codec frames, for the RVQ dump.
 */
ms::MultiStageConfig loadGraph(Args const& args, bool reportCodes = false)
{
    auto config = ms::MultiStageConfig::load(args.configDir);
    auto const& flags = args.runtime;
    auto const given = [&args](int id) { return args.runtimeFlags.count(id) != 0; };
    auto const absolute = [](std::string const& path) { return std::filesystem::absolute(path).string(); };
    for (auto& stage : config.stages)
    {
        auto& target = stage.config;
        // llm_inference and qwen3_tts_inference hand one checkpoint to every engine they load.
        auto const checkpoint = flags.checkpointDir.empty() ? std::string{} : absolute(flags.checkpointDir);
        if (stage.runner == kTalkerRunner)
        {
            if (!checkpoint.empty())
            {
                target["checkpoint"] = checkpoint;
                target["code_predictor_checkpoint"] = checkpoint;
            }
            if (args.cpSpecVerifySize >= 0)
            {
                target["cp_spec_verify_size"] = args.cpSpecVerifySize;
            }
            if (!args.tokenizerDir.empty())
            {
                target["tokenizer_dir"] = absolute(args.tokenizerDir);
            }
            if (reportCodes)
            {
                target["report_codes"] = true;
            }
            continue;
        }
        if (stage.runner == kVocoderRunner)
        {
            if (!checkpoint.empty())
            {
                target["checkpoint"] = checkpoint;
            }
            if (args.chunkFrames > 0)
            {
                target["chunk_frames"] = args.chunkFrames;
            }
            if (args.wholeUtterance)
            {
                target["whole_utterance"] = true;
            }
            continue;
        }
        if (stage.runner != kBackboneRunner)
        {
            continue;
        }
        if (!checkpoint.empty())
        {
            target["checkpoint"] = checkpoint;
        }
        if (!flags.draftCheckpointDir.empty())
        {
            target["draft_checkpoint"] = absolute(flags.draftCheckpointDir);
        }
        if (!flags.rtxRuntimeCachePath.empty())
        {
            target["runtime_cache"] = absolute(flags.rtxRuntimeCachePath);
        }
        if (flags.offloadEmbedding)
        {
            target["offload_embedding"] = true;
        }
        if (flags.specDecodeArgs.enabled || target.contains("spec_decode"))
        {
            target["spec_decode"] = specDecodeFor(stage, args.configDir, args);
        }

        auto const& cache = flags.contextCacheConfig;
        auto cacheJson = target.value("context_cache", nlohmann::json::object());
        if (given(eu::ENABLE_CONTEXT_REUSE))
        {
            cacheJson["enable_reuse"] = cache.enabled;
        }
        if (given(eu::CONTEXT_CACHE_MAX_RECORDS))
        {
            cacheJson["max_records"] = cache.maxRecords;
        }
        if (given(eu::CONTEXT_CACHE_MAX_SNAPSHOTS))
        {
            cacheJson["max_snapshots"] = cache.maxSnapshots;
        }
        if (given(eu::ENCODER_CACHE_BUDGET_BYTES))
        {
            cacheJson["encoder_cache_budget_bytes"] = cache.encoderEmbeddingCacheBudgetBytes;
        }
        if (!cacheJson.empty())
        {
            target["context_cache"] = cacheJson;
        }

        auto const& pruner = flags.visualPrunerConfig;
        if (pruner.enabled)
        {
            auto pruning = target.value("visual_pruning", nlohmann::json::object());
            pruning["algorithm"] = pruner.algorithm;
            pruning["reduction_ratio"] = pruner.reductionRatio;
            pruning["pivot_image_tokens"] = pruner.pivotImageTokens;
            pruning["pivot_text_tokens"] = pruner.pivotTextTokens;
            target["visual_pruning"] = pruning;
        }

        auto const& ple = flags.pleConfig;
        auto pleJson = target.value("ple", nlohmann::json::object());
        if (given(eu::PLE_FILE))
        {
            pleJson["file"] = absolute(ple.filePath);
        }
        if (given(eu::PLE_RESIDENCY))
        {
            pleJson["residency"] = ple.residency == trt_edgellm::rt::PleResidency::kRESIDENT ? "resident" : "cached";
        }
        if (given(eu::PLE_CACHE_SLOTS))
        {
            pleJson["cache_slots"] = ple.cacheSlots;
        }
        if (!pleJson.empty())
        {
            target["ple"] = pleJson;
        }
    }
    return config;
}

/*!
 * @brief What this request supplies, in the same key space the config's `from` fields use.
 *
 * Route prunes on these keys, so an input the request does not carry must not appear here.
 *
 * @param storage Keeps the decoded payloads alive; the bindings only borrow them.
 */
ms::PortBindings payloadFor(Args const& args, PayloadStorage& storage)
{
    ms::PortBindings payload;
    if (args.parsedRequest != nullptr)
    {
        // Carries its own media and decoding options; the prompt below still selects the route.
        payload.bind("request:llm_request",
            ms::PortValue{const_cast<trt_edgellm::rt::LLMGenerationRequest*>(args.parsedRequest), {1},
                ms::PortKind::kRequest, static_cast<int32_t>(sizeof(trt_edgellm::rt::LLMGenerationRequest)),
                /*complete=*/true});
    }
    if (!storage.promptIds.empty())
    {
        // Already tokenized: a graph whose entry stage has no tokenizer of its own needs the
        // chat template applied before the request reaches it.
        payload.bind("request:messages",
            ms::PortValue{storage.promptIds.data(), {static_cast<int64_t>(storage.promptIds.size())},
                ms::PortKind::kText, static_cast<int32_t>(sizeof(int32_t)), /*complete=*/true});
    }
    else if (!args.prompt.empty() || !args.audioPath.empty() || !args.framePaths.empty())
    {
        storage.prompt = args.prompt;
        if (storage.prompt.empty())
        {
            // Media with no text is a turn of its own; the backbone fills each item from its port.
            nlohmann::json content = nlohmann::json::array();
            if (!args.audioPath.empty())
            {
                content.push_back({{"type", "audio"}});
            }
            if (!args.framePaths.empty())
            {
                content.push_back({{"type", args.isVideo || args.framePaths.size() > 1 ? "video" : "image"}});
            }
            nlohmann::json const turn{{"role", "user"}, {"content", content}};
            storage.prompt = nlohmann::json{{"messages", nlohmann::json::array({turn})}}.dump();
        }
        payload.bind("request:messages",
            ms::PortValue{storage.prompt.data(), {static_cast<int64_t>(storage.prompt.size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
    }
    if (!args.voice.empty())
    {
        storage.voice = args.voice;
        payload.bind("request:voice",
            ms::PortValue{storage.voice.data(), {static_cast<int64_t>(storage.voice.size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
    }
    if (!args.sampling.empty())
    {
        storage.sampling = args.sampling;
        payload.bind("request:sampling",
            ms::PortValue{storage.sampling.data(), {static_cast<int64_t>(storage.sampling.size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
    }
    if (!args.style.empty())
    {
        storage.style = args.style;
        payload.bind("request:style",
            ms::PortValue{storage.style.data(), {static_cast<int64_t>(storage.style.size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
    }
    if (!args.speakerCodes.empty())
    {
        auto const perFrame = static_cast<int64_t>(args.speakerCodes.front().size());
        storage.speakerCodeFrames = static_cast<int64_t>(args.speakerCodes.size());
        storage.speakerCodes.clear();
        storage.speakerCodes.reserve(static_cast<size_t>(storage.speakerCodeFrames * perFrame));
        for (auto const& frame : args.speakerCodes)
        {
            if (static_cast<int64_t>(frame.size()) != perFrame)
            {
                throw std::runtime_error("speaker_codes frames must all have the same width");
            }
            storage.speakerCodes.insert(storage.speakerCodes.end(), frame.begin(), frame.end());
        }
        payload.bind("request:speaker_codes",
            ms::PortValue{storage.speakerCodes.data(), {storage.speakerCodeFrames, perFrame}, ms::PortKind::kCodec,
                static_cast<int32_t>(sizeof(int32_t)), /*complete=*/true});
    }
    if (args.parsedRequest == nullptr && !args.framePaths.empty())
    {
        namespace iu = trt_edgellm::rt::imageUtils;
        storage.pixels = args.framePaths.size() == 1 && !args.isVideo
            ? iu::loadRgbImageFromFile(args.framePaths.front())
            : iu::loadRgbVideoFromFrames(args.framePaths, args.fps);
        if (storage.pixels.buffer == nullptr)
        {
            throw std::runtime_error("failed to decode image: " + args.framePaths.front());
        }
        payload.bind("request:image",
            ms::PortValue{storage.pixels.buffer->rawPointer(),
                {storage.pixels.frames, storage.pixels.height, storage.pixels.width, 3}, ms::PortKind::kImage,
                /*elementBytes=*/1, /*complete=*/true});
        storage.vision
            = nlohmann::json{{"fps", args.fps}, {"is_video", args.isVideo || args.framePaths.size() > 1}}.dump();
        payload.bind("request:vision",
            ms::PortValue{storage.vision.data(), {static_cast<int64_t>(storage.vision.size())}, ms::PortKind::kText,
                /*elementBytes=*/1, /*complete=*/true});
    }
    if (!args.refAudioPath.empty())
    {
        if (!trt_edgellm::rt::audio::loadAudioFile(args.refAudioPath, kRefAudioSampleRate, storage.refAudio))
        {
            throw std::runtime_error("failed to load reference audio: " + args.refAudioPath);
        }
        payload.bind("request:ref_audio",
            ms::PortValue{storage.refAudio.samples->dataPointer<float>(), {storage.refAudio.numSamples()},
                ms::PortKind::kAudio, static_cast<int32_t>(sizeof(float)), /*complete=*/true});
    }
    if (args.parsedRequest == nullptr && !args.audioPath.empty())
    {
        if (!trt_edgellm::rt::audio::loadAudioFile(args.audioPath, kAudioSampleRate, storage.audio))
        {
            throw std::runtime_error("failed to load audio file: " + args.audioPath);
        }
        payload.bind("request:audio",
            ms::PortValue{storage.audio.samples->dataPointer<float>(), {storage.audio.numSamples()},
                ms::PortKind::kAudio, static_cast<int32_t>(sizeof(float)), /*complete=*/true});
    }
    return payload;
}

/*!
 * @brief Drains one text terminal until its producer is done.
 *
 * The terminal condition is sampled BEFORE the pop: a producer that pushes its last chunk
 * and then finishes in the window between an empty pop and the check would otherwise be
 * observed as finished while its chunk is still queued, and that chunk would be lost.
 * `stopped` is set only after run() returned, so by then no producer can still push.
 */
void drainText(StreamChannel& channel, std::atomic<bool> const& stopped, std::string& out, bool echo)
{
    while (true)
    {
        bool const terminal = channel.isFinished() || channel.isCancelled() || stopped.load(std::memory_order_acquire);
        auto chunk = channel.waitPop(std::chrono::milliseconds{20});
        if (chunk.has_value())
        {
            out += chunk->text;
            if (echo)
            {
                std::cout << chunk->text << std::flush;
            }
            continue;
        }
        if (terminal)
        {
            return;
        }
    }
}

//! Same sampling order as drainText, for the same reason. With no @p writer the samples are
//! counted and dropped. A failed write sets @p writeFailed.
void drainAudio(AudioStreamChannel& channel, std::atomic<bool> const& stopped, StreamingAudioWriter* writer,
    bool& writeFailed, int64_t& samples, int32_t& sampleRate, int32_t& numChannels,
    std::chrono::steady_clock::time_point submitted, double& firstAudioMs)
{
    while (true)
    {
        bool const terminal = channel.isFinished() || channel.isCancelled() || stopped.load(std::memory_order_acquire);
        auto chunk = channel.waitPop(std::chrono::milliseconds{20});
        if (chunk.has_value())
        {
            // The first window with samples in it, not the first push: the terminal window
            // is empty by construction and would report a latency that never happened.
            if (firstAudioMs <= 0.0 && !chunk->pcm16.empty())
            {
                firstAudioMs
                    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitted).count();
            }
            if (chunk->sampleRate > 0)
            {
                sampleRate = chunk->sampleRate;
                numChannels = chunk->numChannels;
                if (writer != nullptr)
                {
                    writer->setFormat(sampleRate, numChannels);
                }
            }
            samples += static_cast<int64_t>(chunk->pcm16.size() / sizeof(int16_t) / std::max(numChannels, 1));
            if (writer != nullptr && !writer->appendPcm16(chunk->pcm16))
            {
                writeFailed = true;
            }
            continue;
        }
        if (terminal)
        {
            // Here rather than after the join: the header patch and close are file I/O, and
            // on this thread they overlap the next member's stream instead of queueing behind it.
            if (writer != nullptr)
            {
                writer->finalize();
            }
            return;
        }
    }
}

//! What a video terminal produced.
struct VideoResult
{
    int64_t frames{0};
    int32_t height{0};
    int32_t width{0};
    int32_t fps{0};
};

/*!
 * @brief Drains one video terminal until its producer is done, appending raw RGB frames to @p file.
 *
 * Samples the terminal condition before the pop, as drainText() does.
 */
void drainVideo(VideoStreamChannel& channel, std::atomic<bool> const& stopped, std::ofstream* file, bool& writeFailed,
    VideoResult& video)
{
    while (true)
    {
        bool const terminal = channel.isFinished() || channel.isCancelled() || stopped.load(std::memory_order_acquire);
        auto chunk = channel.waitPop(std::chrono::milliseconds{20});
        if (chunk.has_value())
        {
            video.frames += chunk->numFrames;
            video.height = chunk->height;
            video.width = chunk->width;
            video.fps = chunk->fps;
            if (file != nullptr && !file->write(chunk->rgb8.data(), static_cast<std::streamsize>(chunk->rgb8.size())))
            {
                writeFailed = true;
            }
            continue;
        }
        if (terminal)
        {
            if (file != nullptr)
            {
                file->close();
            }
            return;
        }
    }
}

/*!
 * @brief Stops and joins every drain thread on any exit from its scope, throw included.
 *
 * The threads read channels and a writer that die with the enclosing frame, so unwinding
 * past a still-running drain is a use-after-free rather than a leak.
 */
class DrainThreads
{
public:
    ~DrainThreads()
    {
        mStop.store(true, std::memory_order_release);
        for (auto& thread : mThreads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
    }

    DrainThreads() = default;
    DrainThreads(DrainThreads const&) = delete;
    DrainThreads& operator=(DrainThreads const&) = delete;

    void add(std::thread thread)
    {
        mThreads.push_back(std::move(thread));
    }

    std::atomic<bool> const& stopFlag() const noexcept
    {
        return mStop;
    }

private:
    std::atomic<bool> mStop{false};
    std::vector<std::thread> mThreads;
};

//! One request of a cohort and where its outputs go.
struct CohortMember
{
    Args spec;
    std::vector<int32_t> promptIds;
    bool audio{false};     //!< Attach an audio sink.
    std::string wavPath;   //!< Where that audio is written; empty keeps it in memory only.
    bool video{false};     //!< Attach a video sink.
    std::string videoPath; //!< Where those frames are written; empty keeps them in memory only.
};

//! What one request produced. `error` says why when `ok` is false.
struct RequestResult
{
    std::string text;
    int64_t samples{0}; //!< Per channel.
    int32_t sampleRate{0};
    int32_t numChannels{1};
    double firstAudioMs{0.0};
    VideoResult video;
    bool ok{false};
    //! Refused at admission while another member already held a seat: the request may simply
    //! belong to another cohort, so this is no verdict on it.
    bool deferrable{false};
    std::string error;
    //! Keyed by stage id. Empty when no stage reported anything.
    nlohmann::json stages = nlohmann::json::object();
    //! "<terminal>:<reason>" per terminal the route reached.
    std::vector<std::string> finishReasons;
};

/*!
 * @brief Drive one cohort to completion on an already-loaded runtime.
 *
 * A member that cannot be seated, or whose media does not decode, is left out and the rest run.
 * Channels and WAV writers are per member: a StreamChannel serves one live request at a time.
 * A request that did not succeed leaves no WAV behind.
 */
std::vector<RequestResult> runCohort(ms::MultiStageRuntime& runtime, std::vector<CohortMember> const& members)
{
    auto const cohort = members.size();
    std::vector<RequestResult> results(cohort);

    std::vector<std::shared_ptr<StreamChannel>> textChannels(cohort);
    std::vector<std::unique_ptr<AudioStreamChannel>> audioChannels(cohort);
    std::vector<StreamingAudioWriter> audioWriters(cohort);
    std::vector<char> audioWriteFailed(cohort, 0);
    std::vector<std::unique_ptr<VideoStreamChannel>> videoChannels(cohort);
    std::vector<std::ofstream> videoFiles(cohort);
    std::vector<char> videoWriteFailed(cohort, 0);
    // Deque, not vector: every payload below borrows pointers into its own entry, and a
    // vector growing would move the storage out from under the runtime.
    std::deque<PayloadStorage> storage;
    std::vector<size_t> seated;

    auto const discardWav = [&](size_t member) {
        std::error_code ignored;
        if (!members[member].wavPath.empty())
        {
            audioWriters[member].finalize();
            std::filesystem::remove(members[member].wavPath, ignored);
        }
        if (!members[member].videoPath.empty())
        {
            videoFiles[member].close();
            std::filesystem::remove(members[member].videoPath, ignored);
        }
    };

    for (size_t member = 0; member < cohort; ++member)
    {
        auto const& entry = members[member];
        auto& result = results[member];
        textChannels[member] = StreamChannel::create();
        if (entry.audio)
        {
            if (!entry.wavPath.empty() && !audioWriters[member].open(entry.wavPath))
            {
                result.error = "cannot open '" + entry.wavPath + "' for writing";
                continue;
            }
            audioChannels[member] = std::make_unique<AudioStreamChannel>();
        }
        if (entry.video)
        {
            if (!entry.videoPath.empty())
            {
                videoFiles[member].open(entry.videoPath, std::ios::binary);
                if (!videoFiles[member].is_open())
                {
                    result.error = "cannot open '" + entry.videoPath + "' for writing";
                    discardWav(member);
                    continue;
                }
            }
            videoChannels[member] = std::make_unique<VideoStreamChannel>();
        }

        ms::Sinks sinks;
        sinks.text = textChannels[member];
        sinks.audio = audioChannels[member].get();
        sinks.video = videoChannels[member].get();

        storage.emplace_back();
        storage.back().promptIds = entry.promptIds;
        std::string error;
        ms::RequestSlot* slot = nullptr;
        try
        {
            slot = runtime.submit(payloadFor(entry.spec, storage.back()), sinks, error);
            result.deferrable = slot == nullptr && !seated.empty();
            error = "submit failed: " + error;
        }
        catch (std::exception const& payloadError)
        {
            // A media file that does not decode.
            error = payloadError.what();
        }
        if (slot == nullptr)
        {
            result.error = error;
            discardWav(member);
            continue;
        }
        seated.push_back(member);
    }
    if (seated.empty())
    {
        return results;
    }

    std::vector<ms::Response> responses;
    {
        // Only a lone member echoes to stdout; several would interleave mid-word.
        bool const echoText = seated.size() == 1;
        DrainThreads drains;
        auto const started = std::chrono::steady_clock::now();
        for (auto const member : seated)
        {
            drains.add(std::thread(
                [&, member] { drainText(*textChannels[member], drains.stopFlag(), results[member].text, echoText); }));
            if (audioChannels[member] != nullptr)
            {
                auto* writer = members[member].wavPath.empty() ? nullptr : &audioWriters[member];
                drains.add(std::thread([&, member, writer] {
                    bool failed = false;
                    drainAudio(*audioChannels[member], drains.stopFlag(), writer, failed, results[member].samples,
                        results[member].sampleRate, results[member].numChannels, started, results[member].firstAudioMs);
                    audioWriteFailed[member] = failed ? 1 : 0;
                }));
            }
            if (videoChannels[member] != nullptr)
            {
                auto* file = members[member].videoPath.empty() ? nullptr : &videoFiles[member];
                drains.add(std::thread([&, member, file] {
                    bool failed = false;
                    drainVideo(*videoChannels[member], drains.stopFlag(), file, failed, results[member].video);
                    videoWriteFailed[member] = failed ? 1 : 0;
                }));
            }
        }
        responses = runtime.run();
    }

    // Failed or not: seats left behind stay in the next cohort.
    runtime.take();

    for (size_t seat = 0; seat < seated.size(); ++seat)
    {
        auto const member = seated[seat];
        auto& result = results[member];
        auto const& response = responses[seat];
        result.stages = response.stageDiagnostics;
        result.finishReasons.reserve(response.terminalReasons.size());
        for (auto const& [terminal, reason] : response.terminalReasons)
        {
            result.finishReasons.push_back(terminal + ":" + ms::toString(reason));
        }

        if (!response.ok)
        {
            result.error = response.error;
        }
        else if (audioWriteFailed[member] != 0)
        {
            result.error = "writing '" + members[member].wavPath + "' failed";
        }
        else if (videoWriteFailed[member] != 0)
        {
            result.error = "writing '" + members[member].videoPath + "' failed";
        }
        else
        {
            result.ok = true;
            continue;
        }
        discardWav(member);
    }
    return results;
}

RequestResult runRequest(ms::MultiStageRuntime& runtime, CohortMember const& member)
{
    return runCohort(runtime, {member}).front();
}

//! What request files may set for the runners this binary links.
eu::RequestVocabulary const& requestVocabulary()
{
    static eu::RequestVocabulary const sVocabulary = eu::vocabularyOf(ms::builtinRunners());
    return sVocabulary;
}

//! One entry of the input file's `requests` array, folded onto the command-line defaults.
//! Shared by the measured loop and the warmup, so both build the same payload.
Args specFor(
    Args const& base, nlohmann::json const& document, nlohmann::json const& entry, std::vector<int32_t>& promptIds)
{
    Args spec = base;
    // A conversation travels on the same request field as a bare prompt, wrapped so the
    // backbone can tell the two apart without a flag PortValue has no room for.
    if (entry.contains("messages"))
    {
        nlohmann::json envelope{{"messages", entry.at("messages")}};
        for (auto const* control : {"apply_chat_template", "add_generation_prompt", "enable_thinking"})
        {
            if (entry.contains(control))
            {
                envelope[control] = entry.at(control);
            }
            else if (document.contains(control))
            {
                envelope[control] = document.at(control);
            }
        }
        spec.prompt = envelope.dump();
    }
    else
    {
        spec.prompt = entry.value("prompt", std::string{});
    }
    promptIds.clear();
    if (entry.contains("prompt_ids"))
    {
        promptIds = entry.at("prompt_ids").get<std::vector<int32_t>>();
    }
    auto const& vocabulary = requestVocabulary();
    auto const* voice = eu::findRequestField(vocabulary, "voice", document, entry);
    spec.voice = voice != nullptr ? voice->get<std::string>() : base.voice;
    spec.audioPath = entry.value("audio", base.audioPath);
    spec.refAudioPath = entry.value("ref_audio", document.value("ref_audio", base.refAudioPath));
    if (entry.contains("image"))
    {
        spec.framePaths = {entry.at("image").get<std::string>()};
        spec.isVideo = false;
    }
    else if (entry.contains("video"))
    {
        auto const& video = entry.at("video");
        spec.framePaths = video.at("frames").get<std::vector<std::string>>();
        spec.fps = video.value("fps", 1.0);
        spec.isVideo = true;
    }
    // A turn names its media inline, but the bytes travel on their own ports. Read after the
    // top-level keys so the turn's path wins.
    if (entry.contains("messages"))
    {
        for (auto const& turn : entry.at("messages"))
        {
            if (!turn.is_object() || !turn.contains("content") || !turn.at("content").is_array())
            {
                continue;
            }
            for (auto const& item : turn.at("content"))
            {
                if (!item.is_object() || !item.contains("type"))
                {
                    continue;
                }
                auto const type = item.at("type").get<std::string>();
                if (type == "audio" && item.contains("audio"))
                {
                    spec.audioPath = item.at("audio").get<std::string>();
                }
                else if (type == "image" && item.contains("image"))
                {
                    spec.framePaths = {item.at("image").get<std::string>()};
                    spec.isVideo = false;
                }
                else if (type == "video" && item.contains("frames"))
                {
                    spec.framePaths = item.at("frames").get<std::vector<std::string>>();
                    spec.fps = item.value("fps", 1.0);
                    spec.isVideo = true;
                }
            }
        }
    }

    auto sampling = eu::foldSampling(vocabulary, base.sampling, document, entry);
    if (base.maxGenerateLength > 0)
    {
        sampling["max_generate_length"] = base.maxGenerateLength;
    }
    spec.sampling = sampling.empty() ? std::string{} : sampling.dump();
    auto const style = eu::foldStyle(vocabulary, document, entry);
    if (!style.empty())
    {
        spec.style = style.dump();
    }
    if (auto const* codes = eu::findRequestField(vocabulary, "speaker_codes", document, entry))
    {
        spec.speakerCodes = codes->get<std::vector<std::vector<int32_t>>>();
    }
    return spec;
}

//! Whether @p config has a terminal of @p kind.
bool hasTerminal(ms::MultiStageConfig const& config, ms::PortKind kind)
{
    for (auto const& terminal : config.terminals)
    {
        auto const ref = ms::parseSourceRef(terminal);
        auto const* producer = config.findStage(ref.stageId);
        auto const* port = producer != nullptr ? producer->findPort(ref.portId) : nullptr;
        if (port != nullptr && port->kind == kind)
        {
            return true;
        }
    }
    return false;
}

/*!
 * @brief Whether a request gets an audio sink. False, after saying why, when audio was asked for
 * and the graph has no audio terminal: it would leave a valid, empty WAV and exit 0.
 *
 * A graph with nothing but audio to say speaks whether or not the audio is kept.
 */
bool attachesAudio(ms::MultiStageConfig const& config, bool requested, bool& attach)
{
    bool const audioTerminal = hasTerminal(config, ms::PortKind::kAudio);
    if (requested && !audioTerminal)
    {
        std::cerr << "audio output was requested but topology '" << config.label << "' emits no audio terminal\n";
        return false;
    }
    attach = audioTerminal && (requested || !hasTerminal(config, ms::PortKind::kText));
    return true;
}

//! Whether a request gets a video sink: only when video output was asked for. False, after saying why, when the
//! graph has no video terminal.
bool attachesVideo(ms::MultiStageConfig const& config, bool requested, bool& attach)
{
    if (requested && !hasTerminal(config, ms::PortKind::kImage))
    {
        std::cerr << "video output was requested but topology '" << config.label << "' emits no video terminal\n";
        return false;
    }
    attach = requested;
    return true;
}

//! llm_inference request-file keys only its own parser reads. A request passed field by field
//! would drop them, so a file that sets one is refused on that path.
constexpr std::array<char const*, 13> kParserOnlyKeys = {"sampling_seed", "spec_proposal_sampling", "logit_bias",
    "guided_decoding", "num_logprobs", "available_lora_weights", "lora_name", "context_cache_lookup_policy",
    "context_cache_commit_policy", "disable_spec_decode", "stop", "diffusion_config", "diffusion_max_denoising_steps"};

//! The first parser-only key in @p document or @p requests and where it sits; empty when none.
std::string parserOnlyKey(nlohmann::json const& document, nlohmann::json const& requests)
{
    auto const find = [](nlohmann::json const& object) -> char const* {
        for (auto const* key : kParserOnlyKeys)
        {
            if (object.is_object() && object.contains(key))
            {
                return key;
            }
        }
        return nullptr;
    };
    if (auto const* key = find(document))
    {
        return std::string("'") + key + "'";
    }
    for (size_t index = 0; index < requests.size(); ++index)
    {
        auto const& entry = requests.at(index);
        auto const where = " in request " + std::to_string(index);
        if (auto const* key = find(entry))
        {
            return std::string("'") + key + "'" + where;
        }
        if (!entry.is_object() || !entry.contains("messages") || !entry.at("messages").is_array())
        {
            continue;
        }
        for (auto const& turn : entry.at("messages"))
        {
            if (!turn.is_object() || !turn.contains("content") || !turn.at("content").is_array())
            {
                continue;
            }
            for (auto const& item : turn.at("content"))
            {
                if (item.is_object() && item.contains("do_resize"))
                {
                    return "'do_resize'" + where;
                }
            }
        }
    }
    return {};
}

//! What llm_inference writes as the output text of a request that failed; the accuracy scripts skip it.
constexpr char const* kFailedOutputText = "TensorRT Edge LLM cannot handle this request. Fails.";

//! A stream of this binary's own, for the file helpers that take one.
class OwnedStream
{
public:
    OwnedStream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }
    ~OwnedStream()
    {
        cudaStreamDestroy(mStream);
    }
    OwnedStream(OwnedStream const&) = delete;
    OwnedStream& operator=(OwnedStream const&) = delete;

    cudaStream_t get() const noexcept
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

//! qwen3_tts_inference's RVQ dump: one int32 [frames, codes per frame] tensor named rvq_codes.
bool writeRvqCodes(std::filesystem::path const& path, nlohmann::json const& frames, cudaStream_t stream)
{
    auto const numFrames = static_cast<int64_t>(frames.size());
    auto const perFrame = numFrames > 0 ? static_cast<int64_t>(frames.front().size()) : int64_t{0};
    std::vector<int32_t> flat;
    flat.reserve(static_cast<size_t>(numFrames * perFrame));
    for (auto const& frame : frames)
    {
        for (auto const& code : frame)
        {
            flat.push_back(code.get<int32_t>());
        }
    }
    std::vector<trt_edgellm::rt::Tensor> tensors;
    tensors.emplace_back(flat.data(), trt_edgellm::rt::Coords{numFrames, perFrame}, trt_edgellm::rt::DeviceType::kCPU,
        trt_edgellm::DataType::kINT32, "rvq_codes");
    return trt_edgellm::rt::safetensors::saveSafetensors(path, tensors, stream);
}

/*!
 * @brief Run every request in an input file and write what the accuracy harness reads.
 *
 * calculate_tts_score.py pairs each `responses` entry with audio_req<request_idx>.wav in
 * --output-audio-dir. A failed request keeps its entry so indices stay aligned with the input.
 */
int runBatch(Args const& args)
{
    std::ifstream input(args.inputFile);
    if (!input.is_open())
    {
        std::cerr << "cannot open '" << args.inputFile << "'\n";
        return 1;
    }
    auto const document = nlohmann::json::parse(input, nullptr, /*allow_exceptions=*/false);
    if (document.is_discarded())
    {
        std::cerr << "'" << args.inputFile << "' is not valid JSON\n";
        return 1;
    }
    auto const& requests = document.contains("requests") ? document.at("requests") : document;
    if (!requests.is_array())
    {
        std::cerr << "'" << args.inputFile << "' must hold a 'requests' array\n";
        return 1;
    }
    int32_t const cohortWidth
        = args.batchSizeGiven ? args.batchSize : (document.is_object() ? document.value("batch_size", 1) : 1);
    if (cohortWidth < 1)
    {
        std::cerr << "batch_size must be at least 1\n";
        return 1;
    }

    auto graph = loadGraph(args, /*reportCodes=*/!args.outputAudioDir.empty());
    // A backbone that takes parsed requests gets them from llm_inference's own parser, so media,
    // decoding options and LoRA names mean what they mean there. A file of bare prompts or
    // prompt_ids keeps the per-field inputs.
    std::vector<trt_edgellm::rt::LLMGenerationRequest> parsed;
    bool const everyEntryIsAConversation = std::all_of(requests.begin(), requests.end(),
        [](nlohmann::json const& entry) { return entry.is_object() && entry.contains("messages"); });
    if (readsRequestField(graph, "request:llm_request") && everyEntryIsAConversation)
    {
        auto [loraWeights, batches]
            = eu::parseRequestFile(args.inputFile, /*batchSizeOverride=*/1, args.maxGenerateLength, args.numLogprobs);
        if (batches.size() != requests.size())
        {
            std::cerr << "parsed " << batches.size() << " requests from '" << args.inputFile << "', expected "
                      << requests.size() << "\n";
            return 1;
        }
        parsed = std::move(batches);
        // llm_inference reads sampling only at the document level; a request's own `sampling`
        // object is this binary's extension, so it is laid on top. --max-generate-length still wins.
        for (size_t index = 0; index < parsed.size(); ++index)
        {
            auto const sampling = eu::foldSampling(requestVocabulary(), args.sampling, document, requests.at(index));
            auto& target = parsed[index];
            target.temperature = sampling.value("temperature", target.temperature);
            target.topP = sampling.value("top_p", target.topP);
            target.topK = sampling.value("top_k", target.topK);
            if (args.maxGenerateLength < 0)
            {
                target.maxGenerateLength = sampling.value("max_generate_length", target.maxGenerateLength);
            }
        }
        for (auto& stage : graph.stages)
        {
            for (auto const& [name, path] : loraWeights)
            {
                if (stage.runner == kBackboneRunner)
                {
                    stage.config["lora_adapters"][name] = std::filesystem::absolute(path).string();
                }
            }
        }
    }
    else
    {
        auto key = parserOnlyKey(document, requests);
        if (key.empty() && args.numLogprobs >= 0)
        {
            key = "--num-logprobs";
        }
        if (!key.empty())
        {
            std::cerr << key
                      << " is read only by llm_inference's request parser, which needs a backbone taking "
                         "'request:llm_request' and every request to be a conversation\n";
            return 1;
        }
    }

    SharedResources shared;
    ms::MultiStageRuntime runtime;
    runtime.load(std::move(graph), args.configDir, shared);

    bool attachAudio = false;
    if (!attachesAudio(runtime.config(), !args.outputAudioDir.empty() || args.enableAudioOutput, attachAudio))
    {
        return 1;
    }
    bool attachVideo = false;
    if (!attachesVideo(runtime.config(), !args.outputVideoDir.empty(), attachVideo))
    {
        return 1;
    }
    std::unique_ptr<OwnedStream> dumpStream;
    if (!args.outputAudioDir.empty())
    {
        std::filesystem::create_directories(args.outputAudioDir);
        dumpStream = std::make_unique<OwnedStream>();
    }
    if (attachVideo)
    {
        std::filesystem::create_directories(args.outputVideoDir);
    }
    auto const videoFileFor = [&args](size_t index) {
        return (std::filesystem::path(args.outputVideoDir) / ("video_req" + std::to_string(index) + ".rgb")).string();
    };

    auto const memberFor = [&](size_t index, bool keepAudio) {
        CohortMember member;
        member.spec = specFor(args, document, requests.at(index), member.promptIds);
        member.spec.parsedRequest = parsed.empty() ? nullptr : &parsed[index];
        member.audio = attachAudio;
        if (keepAudio && attachAudio && !args.outputAudioDir.empty())
        {
            member.wavPath
                = (std::filesystem::path(args.outputAudioDir) / ("audio_req" + std::to_string(index) + ".wav"))
                      .string();
        }
        member.video = attachVideo;
        if (keepAudio && attachVideo)
        {
            member.videoPath = videoFileFor(index);
        }
        return member;
    };

    for (int32_t run = 0; run < args.warmup && !requests.empty(); ++run)
    {
        auto const discarded = runRequest(runtime, memberFor(0, /*keepAudio=*/false));
        if (!discarded.ok)
        {
            std::cerr << "warmup run " << run << " failed: " << discarded.error << "\n";
            return 1;
        }
    }
    trt_edgellm::setProfilingEnabled(args.dumpProfile);
    MemoryMonitor memoryMonitor;
    if (args.dumpProfile)
    {
        memoryMonitor.start();
    }

    // llm_inference's per-response fields, from the backbone's report, and qwen3_tts_inference's
    // audio fields. A talker reports its codes only for the RVQ dump, so they leave the report here.
    auto const respond = [&](size_t index, RequestResult& result) {
        nlohmann::json response{{"request_idx", index}, {"batch_idx", 0}, {"output_text", kFailedOutputText}};
        if (!result.ok)
        {
            response["finish_reason"] = "error";
        }
        if (requests.at(index).contains("messages"))
        {
            response["messages"] = requests.at(index).at("messages");
        }
        bool sawBackbone = false;
        for (auto const& stage : runtime.config().stages)
        {
            if (!result.stages.contains(stage.id))
            {
                continue;
            }
            auto& report = result.stages.at(stage.id);
            if (stage.runner == kTalkerRunner && report.contains("rvq_codes"))
            {
                if (result.ok && dumpStream != nullptr && !report.at("rvq_codes").empty())
                {
                    auto const path = std::filesystem::path(args.outputAudioDir)
                        / ("rvq_req" + std::to_string(index) + ".safetensors");
                    if (writeRvqCodes(path, report.at("rvq_codes"), dumpStream->get()))
                    {
                        response["rvq_file"] = path.string();
                    }
                }
                report.erase("rvq_codes");
            }
            if (stage.runner != kBackboneRunner)
            {
                continue;
            }
            sawBackbone = true;
            if (result.ok)
            {
                response["output_text"] = report.value("output_text", result.text);
                response["finish_reason"] = report.value("finish_reason", std::string{});
            }
            response["generated_token_count"] = report.value("generated_tokens", int64_t{0});
            for (auto const* key : {"spec_verify_count", "spec_acceptance_length", "formatted_system_prompt",
                     "formatted_complete_request", "logprobs"})
            {
                if (report.contains(key))
                {
                    response[key] = report.at(key);
                }
            }
            report.erase("output_text");
        }
        if (result.ok && !sawBackbone)
        {
            // Speech from given text, which has no text of its own to report.
            response["output_text"] = result.text;
        }
        response["output_text"] = sanitizeUtf8ForJson(response.at("output_text").get<std::string>());
        if (result.ok && attachAudio)
        {
            if (!args.outputAudioDir.empty())
            {
                response["audio_file"]
                    = (std::filesystem::path(args.outputAudioDir) / ("audio_req" + std::to_string(index) + ".wav"))
                          .string();
            }
            double const sampleRate = result.sampleRate > 0 ? result.sampleRate : kAudioOutputSampleRate;
            response["audio_samples"] = result.samples;
            response["audio_sample_rate"] = static_cast<int32_t>(sampleRate);
            response["audio_channels"] = result.numChannels;
            response["audio_duration_ms"] = static_cast<int64_t>(1000.0 * result.samples / sampleRate);
            // Host wall clock from run() to the first non-empty PCM chunk received, not the
            // CUDA-event latency llm_inference reports under the same name in its profile.
            response["time_to_first_playable_audio_ms"] = result.firstAudioMs;
        }
        if (result.ok && attachVideo)
        {
            response["video_file"] = videoFileFor(index);
            response["video_frames"] = result.video.frames;
            response["video_height"] = result.video.height;
            response["video_width"] = result.video.width;
            response["video_fps"] = result.video.fps;
        }
        response["ok"] = result.ok;
        response["error"] = result.error;
        response["finish_reasons"] = result.finishReasons;
        response["stages"] = result.stages;
        return response;
    };

    std::vector<nlohmann::json> responses(requests.size());
    size_t failures = 0;
    int64_t generatedTokens = 0;
    // qwen3_tts_inference's talker summary: batched runs, utterances and codec frames.
    int64_t talkerRuns = 0;
    int64_t talkerOutputs = 0;
    int64_t talkerFrames = 0;
    std::deque<size_t> pending;
    for (size_t index = 0; index < requests.size(); ++index)
    {
        pending.push_back(index);
    }
    auto const measuredStart = std::chrono::steady_clock::now();
    while (!pending.empty())
    {
        std::vector<size_t> indices;
        std::vector<CohortMember> members;
        while (!pending.empty() && indices.size() < static_cast<size_t>(cohortWidth))
        {
            indices.push_back(pending.front());
            pending.pop_front();
            members.push_back(memberFor(indices.back(), /*keepAudio=*/true));
        }

        auto results = runCohort(runtime, members);
        // A member refused behind a seated one leads a later cohort instead, in file order.
        std::vector<size_t> deferred;
        bool talkerRan = false;
        for (size_t member = 0; member < indices.size(); ++member)
        {
            auto const index = indices[member];
            auto& result = results[member];
            if (result.deferrable)
            {
                std::cout << "request " << index << " waits for a later cohort: " << result.error << "\n";
                deferred.push_back(index);
                continue;
            }
            if (!result.ok)
            {
                ++failures;
                std::cerr << "request " << index << " failed: " << result.error << "\n";
            }
            std::cout << "request " << index << ": " << result.text.size() << " bytes, " << result.samples
                      << " samples, ttfa " << std::fixed << std::setprecision(1) << result.firstAudioMs << " ms\n";
            for (auto const& stage : result.stages)
            {
                generatedTokens += stage.value("generated_tokens", int64_t{0});
            }
            for (auto const& stage : runtime.config().stages)
            {
                if (stage.runner == kTalkerRunner && result.stages.contains(stage.id))
                {
                    talkerRan = true;
                    ++talkerOutputs;
                    talkerFrames += result.stages.at(stage.id).value("audio_frames", int64_t{0});
                }
            }
            responses[index] = respond(index, result);
            if (args.dumpOutput)
            {
                std::cout << "request " << index << " output: " << responses[index].at("output_text").get<std::string>()
                          << "\n";
            }
        }
        pending.insert(pending.begin(), deferred.begin(), deferred.end());
        talkerRuns += talkerRan ? 1 : 0;
    }
    auto const measuredMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - measuredStart).count();

    std::ofstream output(args.outputFile);
    if (!output.is_open())
    {
        std::cerr << "cannot write '" << args.outputFile << "'\n";
        return 1;
    }
    output << nlohmann::json{{"input_file", args.inputFile}, {"responses", responses}}.dump(2) << "\n";

    if (args.dumpProfile)
    {
        memoryMonitor.stop();
        nlohmann::json profile;
        // llm_inference's sections, from the backbone's runtime. A graph holds one backbone.
        for (auto const& stage : runtime.config().stages)
        {
            auto const* backbone = stage.runner == kBackboneRunner
                ? dynamic_cast<trt_edgellm::rt::LlmRunner const*>(runtime.stage(stage.id))
                : nullptr;
            if (backbone == nullptr)
            {
                continue;
            }
            auto const& llm = backbone->runtime();
            addJsonPrefillSummary(profile, llm.getPrefillMetrics());
            if (auto const cache = llm.getContextCacheMetrics(); cache.has_value())
            {
                addJsonContextCacheSummary(profile, *cache);
            }
            if (stage.config.contains("spec_decode"))
            {
                addJsonSpecDecodeGenerationSummary(
                    profile, llm.getSpecDecodeGenerationMetrics(), llm.getSpeculativeDecodingStrategyName());
            }
            else
            {
                addJsonGenerationSummary(profile, llm.getGenerationMetrics());
            }
            addJsonMultimodalSummary(profile, llm.getMultimodalMetrics());
            break;
        }
        if (talkerRuns > 0)
        {
            profile["talker"] = nlohmann::json{{"total_runs", talkerRuns}, {"total_audio_outputs", talkerOutputs},
                {"total_audio_codes", talkerFrames},
                {"average_codes_per_output",
                    talkerOutputs > 0 ? static_cast<float>(talkerFrames) / static_cast<float>(talkerOutputs) : 0.0F}};
        }
        // Same timing, memory and wall_clock fields as llm_inference's profile.
        addJsonTimingStages(profile);
        addJsonMemorySummary(profile, memoryMonitor);
        profile["wall_clock"]
            = nlohmann::json{{"tp_size", 1}, {"generated_tokens", generatedTokens}, {"total_time_ms", measuredMs},
                {"tokens_per_second", measuredMs > 0.0 ? generatedTokens * 1000.0 / measuredMs : 0.0},
                {"ms_per_token", generatedTokens > 0 ? measuredMs / static_cast<double>(generatedTokens) : 0.0}};
        profile["requests"] = responses;
        if (args.profileOutputFile.empty())
        {
            std::cout << profile.dump(2) << "\n";
        }
        else if (std::ofstream profileOut(args.profileOutputFile); profileOut.is_open())
        {
            profileOut << profile.dump(2) << "\n";
            std::cout << "profile -> " << args.profileOutputFile << "\n";
        }
        else
        {
            std::cerr << "cannot write '" << args.profileOutputFile << "'\n";
        }
    }
    std::cout << requests.size() - failures << "/" << requests.size() << " requests succeeded -> " << args.outputFile
              << "\n";
    return failures == 0 ? 0 : 1;
}

int runOnce(Args const& args)
{
    // Declared before the runtime so it outlives every stage that borrowed it.
    SharedResources shared;
    ms::MultiStageRuntime runtime;
    runtime.load(loadGraph(args), args.configDir, shared);

    CohortMember member;
    member.spec = args;
    member.wavPath = args.outAudioPath;
    member.videoPath = args.outVideoPath;
    if (!attachesAudio(runtime.config(), !args.outAudioPath.empty() || args.enableAudioOutput, member.audio)
        || !attachesVideo(runtime.config(), !args.outVideoPath.empty(), member.video))
    {
        return 1;
    }
    auto const result = runRequest(runtime, member);

    std::cout << "\nrequest (" << runtime.config().label << ")\n";
    if (hasTerminal(runtime.config(), ms::PortKind::kText))
    {
        std::cout << "  text (" << result.text.size() << " bytes): " << result.text << "\n";
    }
    if (member.audio)
    {
        double const sampleRate = result.sampleRate > 0 ? result.sampleRate : kAudioOutputSampleRate;
        std::cout << "  audio: " << result.samples << " samples x " << result.numChannels << " channels (" << std::fixed
                  << std::setprecision(2) << static_cast<double>(result.samples) / sampleRate << " s)";
        if (!args.outAudioPath.empty() && result.ok)
        {
            std::cout << " -> " << args.outAudioPath;
        }
        std::cout << "\n";
        if (result.firstAudioMs > 0.0)
        {
            std::cout << "  time to first audio: " << result.firstAudioMs << " ms\n";
        }
    }
    if (member.video)
    {
        std::cout << "  video: " << result.video.frames << " frames of " << result.video.width << "x"
                  << result.video.height << " RGB at " << result.video.fps << " fps";
        if (result.ok)
        {
            std::cout << " -> " << args.outVideoPath;
        }
        std::cout << "\n";
    }
    if (!result.ok)
    {
        std::cerr << "  error: " << result.error << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        usage(argv[0]);
        return 1;
    }
    if (args.help)
    {
        usage(argv[0]);
        return 0;
    }

    if (args.configDir.empty())
    {
        std::cerr << "--config is required\n";
        usage(argv[0]);
        return 1;
    }
    if (args.validateConfig)
    {
        try
        {
            // What MultiStageRuntime::load() checks before it builds a stage; no plugin, engine or device is touched.
            auto const config = ms::MultiStageConfig::load(args.configDir);
            ms::MultiStageRuntime::checkLoadable(config, ms::builtinRunners());
            std::cout << "stage graph '" << config.label << "' matches the runners of this binary\n";
            return 0;
        }
        catch (std::exception const& error)
        {
            std::cerr << error.what() << "\n";
            return 1;
        }
    }
    bool const batch = !args.inputFile.empty();
    if (batch && args.outputFile.empty())
    {
        std::cerr << "--input-file requires --output-file\n";
        return 1;
    }
    if (!batch && args.prompt.empty() && args.audioPath.empty())
    {
        std::cerr << "a request supplying neither --prompt nor --audio has no input for any stage\n";
        return 1;
    }
    if (args.debug)
    {
        trt_edgellm::gLogger.setLevel(trt_edgellm::logger::Severity::kVERBOSE);
    }
    if (!args.audioPath.empty() && !std::filesystem::exists(args.audioPath))
    {
        std::cerr << "--audio '" << args.audioPath << "' does not exist\n";
        return 1;
    }

    // Engines are deserialized inside load(); the plugin creators they name have to be
    // registered before that, and the handle has to outlive every engine that used them.
    auto pluginHandles = trt_edgellm::loadEdgellmPluginLib();

    try
    {
        return batch ? runBatch(args) : runOnce(args);
    }
    catch (std::exception const& error)
    {
        std::cerr << "multistage_inference failed: " << error.what() << "\n";
        return 1;
    }
}
