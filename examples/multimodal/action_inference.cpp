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

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum ActionInferenceOptionId : int
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
    BATCH_SIZE = 910,
    MAX_GENERATE_LENGTH = 911,
    NOISE_SEED = 912,
    CHECKPOINT_DIR = 913
};

struct ActionInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string checkpointDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    // Override parameters (only batchSize and maxGenerateLength can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    int32_t noiseSeed{5};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path>] [--multimodalEngineDir=<path>] [--inputFile=<path>] "
                 "[--checkpointDir=<path>] "
                 "[--outputFile=<path>] [--dumpProfile] [--profileOutputFile=<path>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--noiseSeed=<number>]"
              << std::endl;
    std::cerr << "Alpamayo VLM + action expert inference (no Eagle speculative decoding)." << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to LLM engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (required): visual/, action/, etc."
              << std::endl;
    std::cerr << "  --checkpointDir           HF/ModelOpt checkpoint dir (required for runtime weight loading)"
              << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (required)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr << "  --noiseSeed               Random seed for action diffusion initial noise trajectory (default: 5)"
              << std::endl;
}

bool parseActionInferenceArgs(
    ActionInferenceArgs& args, int argc, char* argv[]) // NOLINT(readability-function-cognitive-complexity)
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, ActionInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, ActionInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, ActionInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, ActionInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"checkpointDir", required_argument, 0, ActionInferenceOptionId::CHECKPOINT_DIR},
        {"outputFile", required_argument, 0, ActionInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, ActionInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, ActionInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, ActionInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, ActionInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, ActionInferenceOptionId::DUMP_OUTPUT},
        {"batchSize", required_argument, 0, ActionInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, ActionInferenceOptionId::MAX_GENERATE_LENGTH},
        {"noiseSeed", required_argument, 0, ActionInferenceOptionId::NOISE_SEED}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case ActionInferenceOptionId::HELP: args.help = true; return true;
        case ActionInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case ActionInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case ActionInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case ActionInferenceOptionId::CHECKPOINT_DIR: args.checkpointDir = optarg; break;
        case ActionInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case ActionInferenceOptionId::DEBUG: args.debug = true; break;
        case ActionInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case ActionInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case ActionInferenceOptionId::WARMUP:
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
        case ActionInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case ActionInferenceOptionId::BATCH_SIZE:
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
        case ActionInferenceOptionId::MAX_GENERATE_LENGTH:
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
        case ActionInferenceOptionId::NOISE_SEED:
            try
            {
                args.noiseSeed = std::stoi(optarg);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid noiseSeed value: %s", optarg);
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
    if (args.multimodalEngineDir.empty())
    {
        LOG_ERROR("ERROR: --multimodalEngineDir is required");
        return false;
    }
    LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());

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

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s with error: %s", inputFilePath.string().c_str(), e.what()));
    }

    // Extract global parameters
    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size value: %d (must be positive)", batchSize));

    // Enforce input limits (defined in cpp/common/inputLimits.h) to prevent DoS attacks and
    // excessive resource consumption. Requests exceeding these bounds are rejected early.
    // The actual engine-specific limit will be checked after the runtime is fully initialized.
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("Input rejected: batch_size %d exceeds limit %d. Limit defined in %s.", batchSize,
            limits::security::kReasonableMaxBatchSize, limits::kInputLimitsLocation));

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    // Default to 128 (config.tokens_per_future_traj) from
    // https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/main/config.json#L64
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 128);
    check::check(maxGenerateLength > 0,
        format::fmtstr(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength)));

    // Read apply_chat_template flag (defaults to true)
    bool applyChatTemplate = inputData.value("apply_chat_template", true);

    // Read add_generation_prompt flag (defaults to true)
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);

    // Read enable_thinking flag (defaults to false)
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            check::check(loraPath.is_string(), "LoRA weight path for '" + loraName + "' must be a string");
            check::check(loraWeightsMap.find(loraName) == loraWeightsMap.end(),
                "Lora weights with name " + loraName + " already exists");
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    // Parse requests and create batched requests
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requestsArray = inputData["requests"];
        size_t numRequests = requestsArray.size();

        // Process requests in batches according to batchSize
        for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
        {
            rt::LLMGenerationRequest batchRequest;
            batchRequest.temperature = temperature;
            batchRequest.topP = topP;
            batchRequest.topK = topK;
            batchRequest.maxGenerateLength = maxGenerateLength;
            batchRequest.applyChatTemplate = applyChatTemplate;
            batchRequest.addGenerationPrompt = addGenerationPrompt;
            batchRequest.enableThinking = enableThinking;

            // Track LoRA weights for validation
            std::string batchLoraWeightsName = "";
            bool firstInBatch = true;

            // Add requests to this batch (up to batchSize requests)
            size_t endIdx = std::min(startIdx + batchSize, numRequests);
            for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
            {
                auto const& requestItem = requestsArray[requestIdx];

                // Each request must be an object with "messages" key
                check::check(requestItem.is_object(), "Each request must be an object with 'messages' key");

                // These are request level property but currently we don't support the mechanism to group requests
                // manually in the input file. Thus, we adopt simply philosophy that we enable the property for all
                // requests in the batch if any request has set the property.
                bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
                if (saveSystemPromptKVCache)
                {
                    batchRequest.saveSystemPromptKVCache = true;
                }
                bool disableSpecDecode = requestItem.value("disable_spec_decode", false);
                if (disableSpecDecode)
                {
                    batchRequest.disableSpecDecode = true;
                }

                check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
                    "Each request object must contain a 'messages' array");

                auto const& messagesArray = requestItem["messages"];

                // Get per-conversation LoRA name if present
                std::string requestLoraName = "";
                if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
                {
                    requestLoraName = requestItem["lora_name"].get<std::string>();

                    // Validate that the LoRA name exists in available_lora_weights
                    check::check(
                        requestLoraName.empty() || loraWeightsMap.find(requestLoraName) != loraWeightsMap.end(),
                        "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
                }

                // Validate that all requests in this batch use the same LoRA weights
                if (firstInBatch)
                {
                    batchLoraWeightsName = requestLoraName;
                    firstInBatch = false;
                }
                else
                {
                    check::check(requestLoraName == batchLoraWeightsName,
                        "Different LoRA weights within the same batch are not supported");
                }

                // Parse messages into structured format
                std::vector<rt::Message> chatMessages;
                std::vector<rt::imageUtils::ImageData> imageBuffers;
                std::optional<std::vector<rt::PastTrajectoryPoint>> requestPastTrajectory;

                // Enforce message count limits
                check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
                    format::fmtstr(
                        "Input rejected: too many messages in request %zu: %zu (max: %zu). Limit defined in %s.",
                        requestIdx, messagesArray.size(), limits::security::kMaxMessagesPerRequest,
                        limits::kInputLimitsLocation));

                for (auto const& messageJson : messagesArray)
                {
                    check::check(messageJson.contains("role") && messageJson.contains("content"),
                        "Each message must have 'role' and 'content' fields");

                    rt::Message chatMsg;
                    chatMsg.role = messageJson["role"].get<std::string>();

                    auto const& contentJson = messageJson["content"];

                    // Support both string (simple text) and array (multimodal) formats
                    if (contentJson.is_string())
                    {
                        // Simple string format - treat as text content
                        std::string const& contentStr = contentJson.get<std::string>();

                        // Enforce content size limits
                        check::check(contentStr.size() <= limits::security::kMaxMessageContentSizeBytes,
                            format::fmtstr(
                                "Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
                                "Limit defined in %s.",
                                requestIdx, contentStr.size(), limits::security::kMaxMessageContentSizeBytes,
                                limits::kInputLimitsLocation));

                        rt::Message::MessageContent msgContent;
                        msgContent.type = "text";
                        msgContent.content = contentStr;
                        chatMsg.contents.push_back(msgContent);
                    }
                    else if (contentJson.is_array())
                    {
                        // Array format - supports multimodal content
                        // Enforce content item limits
                        check::check(contentJson.size() <= limits::security::kMaxContentItemsPerMessage,
                            format::fmtstr("Input rejected: too many content items in message %zu: %zu (max: %zu). "
                                           "Limit defined in %s.",
                                requestIdx, contentJson.size(), limits::security::kMaxContentItemsPerMessage,
                                limits::kInputLimitsLocation));

                        for (auto const& contentItemJson : contentJson)
                        {
                            check::check(
                                contentItemJson.contains("type"), "Each content item must have a 'type' field");

                            rt::Message::MessageContent msgContent;
                            msgContent.type = contentItemJson["type"].get<std::string>();

                            // Based on type, extract the appropriate field
                            if (msgContent.type == "text")
                            {
                                std::string const& textContent = contentItemJson["text"].get<std::string>();

                                // Enforce content size limits
                                check::check(textContent.size() <= limits::security::kMaxMessageContentSizeBytes,
                                    format::fmtstr(
                                        "Input rejected: message content too large in request %zu: %zu bytes "
                                        "(max: %zu). Limit defined in %s.",
                                        requestIdx, textContent.size(), limits::security::kMaxMessageContentSizeBytes,
                                        limits::kInputLimitsLocation));

                                msgContent.content = textContent;
                            }
                            else if (msgContent.type == "image")
                            {
                                msgContent.content = contentItemJson["image"].get<std::string>();
                                // TODO: Need to consider multi-turn conversation, and whether to load all images.
                                auto image = rt::imageUtils::loadRgbImageFromFile(msgContent.content);
                                if (image.buffer != nullptr)
                                {
                                    image.doResize = contentItemJson.value("do_resize", true);
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else if (msgContent.type == "trajectory")
                            {
                                check::check(
                                    contentItemJson.contains("trajectory") && contentItemJson["trajectory"].is_array(),
                                    "Content type 'trajectory' must have a 'trajectory' array of [x,y,z] points");
                                std::vector<rt::PastTrajectoryPoint> traj;
                                for (auto const& pt : contentItemJson["trajectory"])
                                {
                                    check::check(pt.is_array() && pt.size() == 3,
                                        "Each trajectory point must be a length-3 array [x, y, z]");
                                    traj.emplace_back(pt[0].get<float>(), pt[1].get<float>(), pt[2].get<float>());
                                }
                                requestPastTrajectory = std::move(traj);
                            }
                            else
                            {
                                throw std::runtime_error(
                                    format::fmtstr("Alpamayo action content type must be 'text', 'image', or "
                                                   "'trajectory', but got: %s",
                                        msgContent.type.c_str()));
                            }

                            chatMsg.contents.push_back(msgContent);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Message content must be a string or an array");
                    }

                    chatMessages.push_back(chatMsg);
                }

                // Create prompt structure with structured messages
                rt::LLMGenerationRequest::Request request;
                request.messages = std::move(chatMessages);
                request.imageBuffers = std::move(imageBuffers);
                request.pastTrajectory = std::move(requestPastTrajectory);
                batchRequest.requests.push_back(std::move(request));
            }

            // Set the LoRA weights name for this batch (all requests in this batch use the same LoRA weights)
            if (!batchLoraWeightsName.empty())
            {
                batchRequest.loraWeightsName = batchLoraWeightsName;
            }

            batchedRequests.push_back(std::move(batchRequest));
        }
    }
    else
    {
        throw std::runtime_error("'requests' array not found in input file");
    }

    return std::make_pair(std::move(loraWeightsMap), std::move(batchedRequests));
}

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "action_inference");
    ActionInferenceArgs args;
    if (!parseActionInferenceArgs(args, argc, argv))
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
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    // Create runtime
    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try
    {
        llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir,
            loraWeightsMap, stream, rt::ContextCacheConfig{}, args.checkpointDir);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
        return EXIT_FAILURE;
    }
    if (!llmInferenceRuntime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
    }
    LOG_INFO("Using noise seed %d for action diffusion trajectory", args.noiseSeed);
    llmInferenceRuntime->setActionNoiseSeed(args.noiseSeed);

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        // Disable profiling for warmup runs
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = batchedRequests[0];

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            if (!requestStatus)
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
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
    std::string const errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;
    // Index of the request in the input file's flat "requests" array. Batching packs
    // batchSize consecutive requests into one batched request, so downstream consumers
    // must receive the flat index, not the batch index.
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

        bool const requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);

        if (requestStatus)
        {
            // Display inference output to console if --dumpOutput is enabled
            if (args.dumpOutput)
            {
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                        response.outputTexts[batchIdx].c_str());
                }
            }
        }
        else
        {
            // Handle failed request - highlight failures
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // Add to JSON output with UTF-8 validation on output text
        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
            // Validate UTF-8 for output text (inputs are always valid)
            // If invalid UTF-8 detected, error message is returned and original text is logged
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = flatRequestIdx++;
            responseJson["batch_idx"] = batchIdx;
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
                    else if (content.type == "trajectory")
                    {
                        auto const& pastOpt = request.requests[batchIdx].pastTrajectory;
                        if (pastOpt.has_value() && !pastOpt->empty())
                        {
                            nlohmann::json trajArr = nlohmann::json::array();
                            for (auto const& pt : *pastOpt)
                            {
                                trajArr.push_back(
                                    nlohmann::json::array({std::get<0>(pt), std::get<1>(pt), std::get<2>(pt)}));
                            }
                            contentJson["trajectory"] = std::move(trajArr);
                        }
                    }
                    msgJson["content"].push_back(contentJson);
                }
                messagesJson.push_back(msgJson);
            }
            responseJson["messages"] = messagesJson;
            // Store formatted prompts for reference
            responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
            responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
            // Add output trajectory (action expert waypoints) when present
            if (batchIdx < response.outputTrajectories.size() && !response.outputTrajectories[batchIdx].empty())
            {
                nlohmann::json trajectoryJson = nlohmann::json::array();
                for (auto const& pt : response.outputTrajectories[batchIdx])
                {
                    trajectoryJson.push_back(nlohmann::json::array({pt.first, pt.second}));
                }
                responseJson["output_trajectory"] = trajectoryJson;
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
        auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
        outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
        outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
        outputMultimodalProfile(profileOutput, multimodalMetrics);
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
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();

            // Add high-level metrics
            addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
            addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
            addJsonMultimodalSummary(profileJson, multimodalMetrics);

            // Add detailed timing stages
            addJsonTimingStages(profileJson);

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
