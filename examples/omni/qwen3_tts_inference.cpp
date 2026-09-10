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
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "multimodal/qwen3_omni/code2WavRunner.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

struct ParsedInput
{
    // One entry per request; each entry is the list of messages for that request.
    std::vector<std::vector<Message>> requests;
    // Per-request speaker name (parallel to requests). Falls back to top-level "speaker" default.
    std::vector<std::string> requestSpeakers;
    // [Qwen3-TTS] Per-request language name / instruction / voice-clone reference (Base ckpts).
    // Falls back to top-level "language" / "instruct" defaults.
    std::vector<std::string> requestLanguages;
    std::vector<std::string> requestInstructs;
    std::vector<std::string> requestRefAudios;
    std::vector<std::string> requestRefTexts;
    // [Qwen3-Omni Next] Per-request custom-voice codec codes / style / language / system instruct.
    std::vector<std::vector<std::vector<int32_t>>> requestPromptSpeakerCodes;
    std::vector<std::string> requestAssistantInstructs;
    std::vector<std::string> requestTalkerLanguages;
    std::vector<std::string> requestSystemInstructs;
    // (CodePredictor sampling uses the shared subtalker* fields above.)
    bool applyChatTemplate{true};
    bool addGenerationPrompt{true};
    bool enableThinking{false};
    float talkerTemperature{0.9f};
    int32_t talkerTopK{50};
    float talkerTopP{1.0f};
    float repetitionPenalty{1.05f};
    float subtalkerTemperature{0}; // 0 = runtime default (HF code_predictor.generate values)
    int32_t subtalkerTopK{0};
    float subtalkerTopP{0};
    std::string speakerName{""};
    std::string languageName{""};
    std::string instructText{""};
    std::string refAudioPath{""};
    std::string refText{""};
    int32_t maxAudioLength{4096};
    int32_t batchSize{1}; //!< Number of requests bundled per Talker/CodePredictor call.
};

ParsedInput parseInputFile(std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1)
{
    ParsedInput result;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s: %s", inputFilePath.string().c_str(), e.what()));
    }

    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size: %d", batchSize));
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("batch_size %d exceeds limit %d", batchSize, limits::security::kReasonableMaxBatchSize));
    result.batchSize = batchSize;

    result.applyChatTemplate = inputData.value("apply_chat_template", true);
    result.addGenerationPrompt = inputData.value("add_generation_prompt", true);
    result.enableThinking = inputData.value("enable_thinking", false);
    result.talkerTemperature = inputData.value("talker_temperature", 0.9f);
    result.talkerTopK = inputData.value("talker_top_k", 50);
    result.talkerTopP = inputData.value("talker_top_p", 1.0f);
    result.repetitionPenalty = inputData.value("repetition_penalty", 1.05f);
    result.subtalkerTemperature = inputData.value("subtalker_temperature", 0.0f);
    result.subtalkerTopK = inputData.value("subtalker_top_k", 0);
    result.subtalkerTopP = inputData.value("subtalker_top_p", 0.0f);
    result.speakerName = inputData.value("speaker", "");
    result.languageName = inputData.value("language", "");
    result.instructText = inputData.value("instruct", "");
    result.refAudioPath = inputData.value("ref_audio", "");
    result.refText = inputData.value("ref_text", "");
    result.maxAudioLength = inputData.value("max_audio_length", 4096);

    check::check(
        inputData.contains("requests") && inputData["requests"].is_array(), "'requests' array not found in input file");

    auto const& requestsArray = inputData["requests"];
    size_t const numRequests = requestsArray.size();

    for (size_t i = 0; i < numRequests; ++i)
    {
        auto const& requestItem = requestsArray[i];
        check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
            "Each request must contain a 'messages' array");

        std::string requestSpeaker = requestItem.value("speaker", result.speakerName);
        std::string requestLanguage = requestItem.value("language", result.languageName);
        std::string requestInstruct = requestItem.value("instruct", result.instructText);
        std::string requestRefAudio = requestItem.value("ref_audio", result.refAudioPath);
        std::string requestRefText = requestItem.value("ref_text", result.refText);

        auto const& messagesArray = requestItem["messages"];
        check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
            format::fmtstr("Too many messages in request %zu", i));

        std::vector<Message> messages;
        for (auto const& messageJson : messagesArray)
        {
            check::check(messageJson.contains("role") && messageJson.contains("content"),
                "Each message must have 'role' and 'content' fields");

            Message msg;
            msg.role = messageJson["role"].get<std::string>();

            auto const& contentJson = messageJson["content"];
            Message::MessageContent mc;
            mc.type = "text";
            if (contentJson.is_string())
            {
                mc.content = contentJson.get<std::string>();
            }
            else if (contentJson.is_array())
            {
                for (auto const& item : contentJson)
                {
                    check::check(item.contains("type") && item["type"] == "text", "Only 'text' content is supported");
                    mc.content += item["text"].get<std::string>();
                }
            }
            else
            {
                throw std::runtime_error("Message content must be a string or array");
            }
            check::check(mc.content.size() <= limits::security::kMaxMessageContentSizeBytes,
                format::fmtstr("Message content too large: %zu bytes", mc.content.size()));

            msg.contents.push_back(std::move(mc));
            messages.push_back(std::move(msg));
        }
        result.requests.push_back(std::move(messages));
        result.requestSpeakers.push_back(std::move(requestSpeaker));
        result.requestLanguages.push_back(std::move(requestLanguage));
        result.requestInstructs.push_back(std::move(requestInstruct));
        result.requestRefAudios.push_back(std::move(requestRefAudio));
        result.requestRefTexts.push_back(std::move(requestRefText));

        // [Qwen3-Omni Next] Optional custom voice: "prompt_speaker_codes": [[16 ints] per frame].
        std::vector<std::vector<int32_t>> promptCodes;
        if (requestItem.contains("prompt_speaker_codes"))
        {
            auto const& codesJson = requestItem["prompt_speaker_codes"];
            check::check(codesJson.is_array(), "'prompt_speaker_codes' must be an array of per-frame code arrays");
            for (auto const& frameJson : codesJson)
            {
                check::check(frameJson.is_array(), "each prompt_speaker_codes entry must be an array of ints");
                std::vector<int32_t> frame;
                frame.reserve(frameJson.size());
                for (auto const& v : frameJson)
                {
                    frame.push_back(v.get<int32_t>());
                }
                promptCodes.push_back(std::move(frame));
            }
        }
        result.requestPromptSpeakerCodes.push_back(std::move(promptCodes));
        result.requestAssistantInstructs.push_back(
            requestItem.value("assistant_instruct", inputData.value("assistant_instruct", "")));
        result.requestTalkerLanguages.push_back(
            requestItem.value("talker_language", inputData.value("talker_language", "")));
        result.requestSystemInstructs.push_back(
            requestItem.value("system_instruct", inputData.value("system_instruct", "")));
    }

    return result;
}

enum Qwen3TTSOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    TALKER_ENGINE_DIR = 903,
    CODE2WAV_ENGINE_DIR = 904,
    OUTPUT_FILE = 905,
    OUTPUT_AUDIO_DIR = 906,
    DEBUG = 907,
    DUMP_PROFILE = 908,
    PROFILE_OUTPUT_FILE = 909,
    DUMP_OUTPUT = 911,
    BATCH_SIZE = 912,
    TOKENIZER_DIR = 915,
    STREAMING = 916,
    CHUNK_FRAMES = 917,
    CLONE_ENCODER_DIR = 918,
    CHECKPOINT_DIR = 919
};

struct Qwen3TTSInferenceArgs
{
    bool help{false};
    std::string talkerEngineDir{""};
    std::string code2wavEngineDir{""};
    std::string cloneEncoderDir{""};
    std::string tokenizerDir{""};
    std::string checkpointDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string outputAudioDir{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    bool dumpOutput{false};
    int32_t batchSize{-1};
    bool streaming{false};  //!< Enable per-request streaming (RVQ chunks → vocoded inline)
    int32_t chunkFrames{0}; //!< Frames per streaming chunk; required when --streaming is set
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [OPTIONS]\n\n"
              << "Main Options:\n"
              << "  --help                       Display this help message\n"
              << "  --inputFile=<path>           Path to input JSON file (text messages only)\n"
              << "  --talkerEngineDir=<path>     Path to Talker engine directory\n"
              << "  --code2wavEngineDir=<path>   Path to Code2Wav engine directory\n"
              << "  --cloneEncoderDir=<path>     Voice-clone reference encoder engines (Base checkpoints)\n"
              << "  --tokenizerDir=<path>        Path to tokenizer directory\n"
              << "                               Defaults to --talkerEngineDir/../\n"
              << "  --checkpointDir=<path>       HF/ModelOpt checkpoint for external model weights\n"
              << "  --outputFile=<path>          Path to output JSON file\n"
              << "  --outputAudioDir=<path>      Directory to save generated audio (.wav) files\n\n"
              << "Performance Options:\n"
              << "  --batchSize=<number>         Override batch size from input file\n"
              << "  --streaming                  Enable streaming: RVQ chunks vocoded inline per request\n"
              << "  --chunkFrames=<N>            Required with --streaming; frames per chunk callback\n\n"
              << "Debug Options:\n"
              << "  --debug                      Enable verbose logging\n"
              << "  --dumpOutput                 Print inference output to console\n"
              << "  --dumpProfile                Print performance summary to console\n"
              << "  --profileOutputFile=<path>   Path to profile JSON output\n"
              << std::endl;
}

bool parseArgs(Qwen3TTSInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, Qwen3TTSOptionId::HELP},
        {"inputFile", required_argument, 0, Qwen3TTSOptionId::INPUT_FILE},
        {"talkerEngineDir", required_argument, 0, Qwen3TTSOptionId::TALKER_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, Qwen3TTSOptionId::CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, Qwen3TTSOptionId::TOKENIZER_DIR},
        {"cloneEncoderDir", required_argument, 0, Qwen3TTSOptionId::CLONE_ENCODER_DIR},
        {"checkpointDir", required_argument, 0, Qwen3TTSOptionId::CHECKPOINT_DIR},
        {"outputFile", required_argument, 0, Qwen3TTSOptionId::OUTPUT_FILE},
        {"outputAudioDir", required_argument, 0, Qwen3TTSOptionId::OUTPUT_AUDIO_DIR},
        {"debug", no_argument, 0, Qwen3TTSOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, Qwen3TTSOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, Qwen3TTSOptionId::PROFILE_OUTPUT_FILE},
        {"dumpOutput", no_argument, 0, Qwen3TTSOptionId::DUMP_OUTPUT},
        {"batchSize", required_argument, 0, Qwen3TTSOptionId::BATCH_SIZE},
        {"streaming", no_argument, 0, Qwen3TTSOptionId::STREAMING},
        {"chunkFrames", required_argument, 0, Qwen3TTSOptionId::CHUNK_FRAMES}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case Qwen3TTSOptionId::HELP: args.help = true; return true;
        case Qwen3TTSOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case Qwen3TTSOptionId::TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case Qwen3TTSOptionId::CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case Qwen3TTSOptionId::CLONE_ENCODER_DIR: args.cloneEncoderDir = optarg; break;
        case Qwen3TTSOptionId::TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case Qwen3TTSOptionId::CHECKPOINT_DIR: args.checkpointDir = optarg; break;
        case Qwen3TTSOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case Qwen3TTSOptionId::OUTPUT_AUDIO_DIR: args.outputAudioDir = optarg; break;
        case Qwen3TTSOptionId::DEBUG: args.debug = true; break;
        case Qwen3TTSOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case Qwen3TTSOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case Qwen3TTSOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case Qwen3TTSOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("batchSize must be positive, got: %s", optarg);
                    return false;
                }
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case Qwen3TTSOptionId::STREAMING: args.streaming = true; break;
        case Qwen3TTSOptionId::CHUNK_FRAMES:
            try
            {
                args.chunkFrames = std::stoi(optarg);
                if (args.chunkFrames <= 0)
                {
                    LOG_ERROR("chunkFrames must be positive, got: %s", optarg);
                    return false;
                }
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid chunkFrames value: %s", optarg);
                return false;
            }
            break;
        default: LOG_ERROR("Unknown option: %c", opt); return false;
        }
    }

    if (!args.help)
    {
        if (args.inputFile.empty())
        {
            LOG_ERROR("--inputFile is required");
            return false;
        }
        if (args.talkerEngineDir.empty())
        {
            LOG_ERROR("--talkerEngineDir is required");
            return false;
        }
        if (args.streaming && args.chunkFrames <= 0)
        {
            LOG_ERROR("--chunkFrames=<N> is required when --streaming is set");
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    Qwen3TTSInferenceArgs args;
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
    LOG_INFO("=== Qwen3-TTS Inference ===");

    auto pluginHandles = loadEdgellmPluginLib();

    LOG_INFO("Talker Engine:  %s", args.talkerEngineDir.c_str());
    if (!args.code2wavEngineDir.empty())
    {
        LOG_INFO("Code2Wav Engine: %s", args.code2wavEngineDir.c_str());
    }
    LOG_INFO("Input File:     %s", args.inputFile.c_str());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Initialize TTS Runtime (loads tokenizer, engines, and weights internally)
    std::unique_ptr<rt::Qwen3OmniTTSRuntime> ttsRuntime;
    try
    {
        std::filesystem::path const codePredictorDir
            = std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor";
        ttsRuntime = std::make_unique<rt::Qwen3OmniTTSRuntime>(args.talkerEngineDir, codePredictorDir.string(),
            args.tokenizerDir, args.cloneEncoderDir, stream, args.checkpointDir);
        LOG_INFO("TTS runtime initialized");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize TTS Runtime: %s", e.what());
        return EXIT_FAILURE;
    }

    // Initialize Code2Wav Runner (scoped block limits lifetime of code2wavDir)
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    std::filesystem::path const code2wavDir = args.code2wavEngineDir.empty()
        ? std::filesystem::path(args.talkerEngineDir).parent_path() / "code2wav"
        : std::filesystem::path(args.code2wavEngineDir);

    if (std::filesystem::exists(code2wavDir))
    {
        LOG_INFO("Initializing Code2Wav Runner from %s...", code2wavDir.string().c_str());
        try
        {
            code2wavRunner = std::make_unique<Code2WavRunner>(code2wavDir.string(), stream, args.checkpointDir);
            LOG_INFO("Code2Wav Runner initialized");
        }
        catch (std::exception const& e)
        {
            LOG_WARNING("Failed to initialize Code2Wav: %s. Will output RVQ codes only.", e.what());
        }
    }
    else
    {
        LOG_INFO("Code2Wav engine not found at %s. Will output RVQ codes only.", code2wavDir.string().c_str());
    }

    if (!args.outputAudioDir.empty())
    {
        std::filesystem::create_directories(args.outputAudioDir);
    }

    if (!ttsRuntime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for TTS decoding, proceeding without.");
    }

    bool const profilerEnabled = args.dumpProfile || !args.profileOutputFile.empty();
    MemoryMonitor memoryMonitor;
    if (profilerEnabled)
    {
        setProfilingEnabled(true);
        memoryMonitor.start();
    }

    LOG_INFO("Parsing input file...");
    auto input = parseInputFile(args.inputFile, args.batchSize);

    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    size_t failedCount = 0;

    // Determine how many requests to bundle into each Talker call.
    // input.batchSize comes from `--batchSize` (or `batch_size` in input JSON, default 1).
    size_t const ttsBatchSize = std::max<size_t>(1, static_cast<size_t>(input.batchSize));
    LOG_INFO("Processing %zu request(s) with ttsBatchSize=%zu...", input.requests.size(), ttsBatchSize);

    if (args.streaming)
    {
        check::check(
            code2wavRunner != nullptr, "--streaming requires a usable Code2Wav engine (chunks must be vocoded inline)");
        LOG_INFO("Streaming mode: chunkFrames=%d", args.chunkFrames);
    }

    // Per-batch streaming context for a single TTS group. Lives only during one
    // handleAudioGeneration call; cleared between groups.
    struct StreamingBatchCtx
    {
        std::vector<__half> pcmBuffer; //!< Accumulated FP16 PCM samples (chunk-by-chunk concat)
        int32_t numFramesEmitted{0};   //!< Codec frames vocoded so far
        int32_t numChunkCalls{0};      //!< Total callback invocations (including final flush)
        bool finalSeen{false};
        bool vocoderFailed{false};
        cudaEvent_t ttfpaEvent{nullptr}; //!< Recorded on first vocoded chunk emit (time-to-first-playable-audio)
        bool ttfpaRecorded{false};
    };

    for (size_t startIdx = 0; startIdx < input.requests.size(); startIdx += ttsBatchSize)
    {
        size_t const endIdx = std::min(startIdx + ttsBatchSize, input.requests.size());
        size_t const groupSize = endIdx - startIdx;

        // Allocate per-batch streaming state up front so callbacks capture stable references.
        std::vector<StreamingBatchCtx> streamCtxs(args.streaming ? groupSize : 0);
        cudaEvent_t e2eStart{nullptr};
        if (args.streaming)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&e2eStart, cudaEventDefault));
            CUDA_CHECK(cudaEventRecord(e2eStart, stream));
            for (auto& ctx : streamCtxs)
            {
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ttfpaEvent, cudaEventDefault));
            }
        }

        std::vector<rt::Qwen3OmniTTSRuntime::TalkerGenerationRequest> groupReqs(groupSize);
        for (size_t k = 0; k < groupSize; ++k)
        {
            size_t const requestIdx = startIdx + k;
            auto& talkerReq = groupReqs[k];
            talkerReq.talkerTemperature = input.talkerTemperature;
            talkerReq.talkerTopK = input.talkerTopK;
            talkerReq.talkerTopP = input.talkerTopP;
            talkerReq.repetitionPenalty = input.repetitionPenalty;
            talkerReq.subtalkerTemperature = input.subtalkerTemperature;
            talkerReq.subtalkerTopK = input.subtalkerTopK;
            talkerReq.subtalkerTopP = input.subtalkerTopP;
            talkerReq.applyChatTemplate = input.applyChatTemplate;
            talkerReq.addGenerationPrompt = input.addGenerationPrompt;
            talkerReq.enableThinking = input.enableThinking;
            talkerReq.speakerName = input.requestSpeakers[requestIdx];
            talkerReq.languageName = input.requestLanguages[requestIdx];
            talkerReq.instructText = input.requestInstructs[requestIdx];
            talkerReq.refAudioPath = input.requestRefAudios[requestIdx];
            talkerReq.refText = input.requestRefTexts[requestIdx];
            talkerReq.promptSpeakerCodes = input.requestPromptSpeakerCodes[requestIdx];
            talkerReq.assistantInstruct = input.requestAssistantInstructs[requestIdx];
            talkerReq.talkerLanguage = input.requestTalkerLanguages[requestIdx];
            talkerReq.systemInstruct = input.requestSystemInstructs[requestIdx];
            talkerReq.maxAudioLength = input.maxAudioLength;
            talkerReq.messages = input.requests[requestIdx];

            if (args.streaming)
            {
                talkerReq.streamingChunkFrames = args.chunkFrames;
                // Capture k (batch slot in group) + requestIdx (global request id) by value.
                talkerReq.onChunkReady = [&streamCtxs, &code2wavRunner, stream, k, requestIdx](
                                             std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal) {
                    auto& ctx = streamCtxs[k];
                    ++ctx.numChunkCalls;

                    if (!chunkRvqCodes.empty())
                    {
                        // Vocode this chunk independently. Code2Wav engine resets its KV per call so
                        // chunk boundaries lose a small amount of context; acceptable for streaming
                        // (RVQ codes still bit-exact vs. non-streaming).
                        size_t const numFrames = chunkRvqCodes.size();
                        size_t const numLayers = chunkRvqCodes[0].size();
                        std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
                        for (size_t f = 0; f < numFrames; ++f)
                        {
                            for (size_t l = 0; l < numLayers; ++l)
                            {
                                transposed[l][f] = chunkRvqCodes[f][l];
                            }
                        }
                        rt::audioUtils::AudioData chunkAudio;
                        if (!code2wavRunner->generateWaveform(transposed, chunkAudio, stream))
                        {
                            LOG_WARNING(
                                "Streaming vocode failed for request %zu chunk %d", requestIdx, ctx.numChunkCalls);
                            ctx.vocoderFailed = true;
                        }
                        else if (chunkAudio.waveform && !chunkAudio.waveform->isEmpty())
                        {
                            __half const* src = static_cast<__half const*>(chunkAudio.waveform->rawPointer());
                            int64_t const samples = chunkAudio.waveform->getShape()[1];
                            ctx.pcmBuffer.insert(ctx.pcmBuffer.end(), src, src + samples);
                            if (!ctx.ttfpaRecorded && ctx.ttfpaEvent)
                            {
                                CUDA_CHECK(cudaEventRecord(ctx.ttfpaEvent, stream));
                                ctx.ttfpaRecorded = true;
                            }
                        }
                        ctx.numFramesEmitted += static_cast<int32_t>(numFrames);
                    }
                    if (isFinal)
                    {
                        ctx.finalSeen = true;
                    }
                };
            }
        }

        rt::Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResp;
        bool const groupStatus = ttsRuntime->handleAudioGeneration(groupReqs, talkerResp, stream);

        if (!groupStatus)
        {
            LOG_WARNING("TTS generation failed for batch group [%zu, %zu)", startIdx, endIdx);
            hasFailedRequest = true;
            failedCount += groupSize;
        }

        for (size_t k = 0; k < groupSize; ++k)
        {
            size_t const requestIdx = startIdx + k;
            bool const requestStatus
                = groupStatus && k < talkerResp.batchRvqCodes.size() && !talkerResp.batchRvqCodes[k].empty();

            // Build audio output: from streaming pcmBuffer if --streaming, else vocode-once-at-end.
            rt::audioUtils::AudioData audioOutput;
            bool hasAudio = false;
            std::vector<std::vector<int32_t>> const& framesCodes = (k < talkerResp.batchRvqCodes.size())
                ? talkerResp.batchRvqCodes[k]
                : std::vector<std::vector<int32_t>>{};

            if (args.streaming && requestStatus)
            {
                auto const& ctx = streamCtxs[k];
                if (!ctx.finalSeen)
                {
                    LOG_WARNING("Streaming final flush missing for request %zu", requestIdx);
                }
                if (!ctx.vocoderFailed && !ctx.pcmBuffer.empty())
                {
                    int64_t const totalSamples = static_cast<int64_t>(ctx.pcmBuffer.size());
                    auto waveformTensor = std::make_shared<rt::Tensor>(
                        rt::Coords{1, totalSamples}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF, "streaming_pcm");
                    std::memcpy(waveformTensor->rawPointer(), ctx.pcmBuffer.data(), totalSamples * sizeof(__half));
                    audioOutput.waveform = waveformTensor;
                    audioOutput.sampleRate = code2wavRunner->getConfig().sampleRate;
                    audioOutput.hasWaveform = true;
                    hasAudio = true;
                    if (!args.outputAudioDir.empty())
                    {
                        std::string filename = format::fmtstr("audio_req%zu.wav", requestIdx);
                        std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                        if (!saveAudioToWav(audioPath.string(), audioOutput))
                        {
                            LOG_WARNING("Failed to save audio: %s", audioPath.string().c_str());
                        }
                    }

                    // Latency via CUDA events (same pattern as llm_inference.cpp Omni path).
                    // TTFC (time-to-first-codec) is shared across batches in a group — recorded by
                    // the runtime via mTtfaEnd at the joint first-sampling point of the bs=N prefill.
                    // TTFPA (time-to-first-playable-audio) is per-batch — recorded in the callback.
                    float ttfcMs = -1.0f;
                    float ttfpaMs = -1.0f;
                    auto const ttfaEnd = ttsRuntime->getTtfaEndEvent();
                    // e2eStart was queued before generation but never explicitly synced; in the
                    // current single-stream layout the later sync of ttfaEnd / ttfpaEvent implicitly
                    // serializes after e2eStart, so cudaEventElapsedTime is well-defined. Sync e2eStart
                    // explicitly anyway — defensive against future multi-stream layouts.
                    if (e2eStart)
                    {
                        CUDA_CHECK(cudaEventSynchronize(e2eStart));
                    }
                    if (e2eStart && ttfaEnd)
                    {
                        CUDA_CHECK(cudaEventSynchronize(ttfaEnd));
                        CUDA_CHECK(cudaEventElapsedTime(&ttfcMs, e2eStart, ttfaEnd));
                    }
                    if (e2eStart && ctx.ttfpaRecorded && ctx.ttfpaEvent)
                    {
                        CUDA_CHECK(cudaEventSynchronize(ctx.ttfpaEvent));
                        CUDA_CHECK(cudaEventElapsedTime(&ttfpaMs, e2eStart, ctx.ttfpaEvent));
                    }
                    LOG_INFO("[stream req %zu] chunks=%d frames=%d samples=%ld TTFC=%.1fms TTFPA=%.1fms", requestIdx,
                        ctx.numChunkCalls, ctx.numFramesEmitted, totalSamples, ttfcMs, ttfpaMs);

                    // For bs=1 groups, populate mOmniLatencyMetrics so downstream profilers see it
                    // (matches the contract llm_inference.cpp uses for Omni streaming).
                    if (groupSize == 1)
                    {
                        auto& latency = ttsRuntime->getMutableOmniLatencyMetrics();
                        if (ttfcMs >= 0.0f)
                            latency.timeToFirstAudioCodeMs = ttfcMs;
                        if (ttfpaMs >= 0.0f)
                            latency.timeToFirstPlayableAudioMs = ttfpaMs;
                    }
                }
            }
            else if (requestStatus && code2wavRunner && !framesCodes.empty())
            {
                size_t const numFrames = framesCodes.size();
                size_t const numLayers = framesCodes[0].size();
                std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
                for (size_t f = 0; f < numFrames; ++f)
                {
                    for (size_t l = 0; l < numLayers; ++l)
                    {
                        transposed[l][f] = framesCodes[f][l];
                    }
                }

                if (code2wavRunner->generateWaveform(transposed, audioOutput, stream))
                {
                    hasAudio = true;
                    if (!args.outputAudioDir.empty())
                    {
                        std::string filename = format::fmtstr("audio_req%zu.wav", requestIdx);
                        std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                        if (!saveAudioToWav(audioPath.string(), audioOutput))
                        {
                            LOG_WARNING("Failed to save audio: %s", audioPath.string().c_str());
                        }
                    }
                }
                else
                {
                    LOG_WARNING("Code2Wav failed for request %zu", requestIdx);
                }
            }

            if (args.dumpOutput && requestStatus && hasAudio)
            {
                int64_t samples = (!audioOutput.waveform || audioOutput.waveform->isEmpty())
                    ? 0
                    : audioOutput.waveform->getShape()[1];
                LOG_INFO("[%zu] Audio: %ld samples (%.2fs)", requestIdx, samples,
                    static_cast<float>(samples) / audioOutput.sampleRate);
            }

            // Build per-request JSON
            nlohmann::json responseJson;
            responseJson["request_idx"] = requestIdx;
            responseJson["output_text"] = requestStatus ? "" : "FAILED";

            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : input.requests[requestIdx])
            {
                nlohmann::json m;
                m["role"] = msg.role;
                m["content"] = msg.contents.empty() ? "" : msg.contents[0].content;
                messagesJson.push_back(std::move(m));
            }
            responseJson["messages"] = std::move(messagesJson);

            if (requestStatus && hasAudio && !args.outputAudioDir.empty())
            {
                std::string filename = format::fmtstr("audio_req%zu.wav", requestIdx);
                std::filesystem::path audioPath = std::filesystem::path(args.outputAudioDir) / filename;
                int64_t samples = (audioOutput.waveform && !audioOutput.waveform->isEmpty())
                    ? audioOutput.waveform->getShape()[1]
                    : 0;
                responseJson["audio_file"] = audioPath.string();
                responseJson["audio_samples"] = samples;
                responseJson["audio_sample_rate"] = audioOutput.sampleRate;
                responseJson["audio_duration_ms"] = static_cast<int64_t>(1000.0 * samples / audioOutput.sampleRate);
            }

            if (requestStatus && !framesCodes.empty() && !args.outputAudioDir.empty())
            {
                int64_t const numFrames = static_cast<int64_t>(framesCodes.size());
                int64_t const numCodes = framesCodes.empty() ? 0 : static_cast<int64_t>(framesCodes[0].size());
                std::vector<int32_t> flat;
                flat.reserve(numFrames * numCodes);
                for (auto const& frame : framesCodes)
                {
                    flat.insert(flat.end(), frame.begin(), frame.end());
                }
                std::vector<rt::Tensor> tensors;
                tensors.emplace_back(flat.data(), rt::Coords{numFrames, numCodes}, rt::DeviceType::kCPU,
                    nvinfer1::DataType::kINT32, "rvq_codes");
                std::string filename = format::fmtstr("rvq_req%zu.safetensors", requestIdx);
                std::filesystem::path stPath = std::filesystem::path(args.outputAudioDir) / filename;
                safetensors::saveSafetensors(stPath, tensors, stream);
                responseJson["rvq_file"] = stPath.string();
            }

            outputData["responses"].push_back(responseJson);
        }

        if (args.streaming)
        {
            for (auto& ctx : streamCtxs)
            {
                if (ctx.ttfpaEvent)
                    cudaEventDestroy(ctx.ttfpaEvent);
            }
            if (e2eStart)
                cudaEventDestroy(e2eStart);
        }
    }

    LOG_INFO("Done: %zu/%zu requests succeeded", input.requests.size() - failedCount, input.requests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("%zu request(s) failed", failedCount);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream ss;
        ss << "\n=== Performance Summary ===\n";
        outputTalkerProfile(ss, ttsRuntime->getMetrics());
        outputMemoryProfile(ss, memoryMonitor);
        ss << "===========================\n";
        LOG_INFO("%s", ss.str().c_str());
    }

    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;
            addJsonTalkerSummary(profileJson, ttsRuntime->getMetrics());
            addJsonTimingStages(profileJson);
            addJsonMemorySummary(profileJson, memoryMonitor);

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2);
                LOG_INFO("Profile saved to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile: %s", e.what());
        }
    }

    if (!args.outputFile.empty())
    {
        try
        {
            std::ofstream out(args.outputFile);
            if (out.is_open())
            {
                out << outputData.dump(2);
                LOG_INFO("Output saved to: %s", args.outputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: %s", e.what());
        }
    }

    cudaStreamDestroy(stream);
    LOG_INFO("=== Done ===");
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
