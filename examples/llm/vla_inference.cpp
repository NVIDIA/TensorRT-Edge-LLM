/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Dedicated VLA (Vision-Language-Action) inference driver.
//
// Unlike `llm_inference` (which targets the paged-KV EngineExecutor runtime), this driver
// uses `VlaInferenceRuntime`, the LLMEngineRunner + ActionRunner based path that consumes
// the VLA engine binding layout (past_key_values_*, prefix_k/prefix_v) produced by the
// Test export harness for models such as PI0.5.

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/trtUtils.h"
#include "profileFormatter.h" // sanitizeUtf8ForJson
#include "profiling/nvtx_wrapper.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/vlaInferenceRuntime.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <cuda_runtime.h>
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

enum VlaInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_OUTPUT = 906,
    BATCH_SIZE = 907,
    MAX_GENERATE_LENGTH = 908,
    ACTION_NOISE_SEED = 909
};

struct VlaInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    bool debug{false};
    bool dumpOutput{false};
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    int32_t actionNoiseSeed{-1};   // -1 means leave runtime default
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --engineDir=<path> [--multimodalEngineDir=<path>] --inputFile=<path> "
                 "--outputFile=<path> [--debug] [--dumpOutput] [--batchSize=<number>] "
                 "[--maxGenerateLength=<number>] [--actionNoiseSeed=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to LLM engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal (vision) engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "  --actionNoiseSeed         Seed for action diffusion noise initialization (optional)" << std::endl;
}

bool parseVlaInferenceArgs(VlaInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, VlaInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, VlaInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, VlaInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, VlaInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, VlaInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, VlaInferenceOptionId::DEBUG},
        {"dumpOutput", no_argument, 0, VlaInferenceOptionId::DUMP_OUTPUT},
        {"batchSize", required_argument, 0, VlaInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, VlaInferenceOptionId::MAX_GENERATE_LENGTH},
        {"actionNoiseSeed", required_argument, 0, VlaInferenceOptionId::ACTION_NOISE_SEED}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case VlaInferenceOptionId::HELP: args.help = true; return true;
        case VlaInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case VlaInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case VlaInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case VlaInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case VlaInferenceOptionId::DEBUG: args.debug = true; break;
        case VlaInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case VlaInferenceOptionId::BATCH_SIZE:
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
        case VlaInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength < 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        case VlaInferenceOptionId::ACTION_NOISE_SEED:
            try
            {
                args.actionNoiseSeed = std::stoi(optarg);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid actionNoiseSeed value: %s", optarg);
                return false;
            }
            break;
        default: return false;
        }
    }

    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

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

namespace
{

std::vector<float> loadRobotStateBin(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check::check(file.is_open(), "Failed to open robot state file: " + path.string());
    auto const fileSize = static_cast<std::size_t>(file.tellg());
    check::check(fileSize % sizeof(float) == 0, "Robot state file size must be a multiple of 4 bytes: " + path.string());
    file.seekg(0);
    std::vector<float> values(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(fileSize));
    check::check(file.good(), "Failed to read robot state file: " + path.string());
    return values;
}

std::vector<float> flattenRobotStateJson(nlohmann::json const& stateJson)
{
    std::vector<float> values;
    if (stateJson.is_array())
    {
        if (!stateJson.empty() && stateJson.front().is_array())
        {
            for (auto const& row : stateJson)
            {
                check::check(row.is_array(), "robot_state nested arrays must contain numeric values");
                for (auto const& value : row)
                {
                    values.push_back(value.get<float>());
                }
            }
        }
        else
        {
            for (auto const& value : stateJson)
            {
                values.push_back(value.get<float>());
            }
        }
    }
    else
    {
        throw std::runtime_error("robot_state must be a JSON array");
    }
    return values;
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

    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size value: %d (must be positive)", batchSize));
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("Input rejected: batch_size %d exceeds limit %d. Limit defined in %s.", batchSize,
            limits::security::kReasonableMaxBatchSize, limits::kInputLimitsLocation));

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    check::check(maxGenerateLength >= 0,
        format::fmtstr("Invalid max_generate_length value: %lld (must be non-negative)",
            static_cast<long long>(maxGenerateLength)));

    bool applyChatTemplate = inputData.value("apply_chat_template", true);
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
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

    if (!(inputData.contains("requests") && inputData["requests"].is_array()))
    {
        throw std::runtime_error("'requests' array not found in input file");
    }

    auto& requestsArray = inputData["requests"];
    size_t numRequests = requestsArray.size();

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
        if (inputData.contains("embodiment_id"))
        {
            batchRequest.embodimentId = inputData["embodiment_id"].get<int64_t>();
        }
        if (inputData.contains("action_batch_size"))
        {
            batchRequest.actionBatchSize = inputData["action_batch_size"].get<int32_t>();
        }

        std::string batchLoraWeightsName = "";
        bool firstInBatch = true;

        size_t endIdx = std::min(startIdx + batchSize, numRequests);
        for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
        {
            auto const& requestItem = requestsArray[requestIdx];
            check::check(requestItem.is_object(), "Each request must be an object with 'messages' key");

            bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
            if (saveSystemPromptKVCache)
            {
                batchRequest.saveSystemPromptKVCache = true;
            }

            check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
                "Each request object must contain a 'messages' array");

            auto const& messagesArray = requestItem["messages"];

            std::string requestLoraName = "";
            if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
            {
                requestLoraName = requestItem["lora_name"].get<std::string>();
                check::check(requestLoraName.empty() || loraWeightsMap.find(requestLoraName) != loraWeightsMap.end(),
                    "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
            }

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

            std::vector<rt::Message> chatMessages;
            std::vector<rt::imageUtils::ImageData> imageBuffers;
            std::vector<rt::audioUtils::AudioData> audioBuffers;
            std::optional<std::vector<rt::PastTrajectoryPoint>> requestPastTrajectory;
            std::vector<float> requestRobotState;
            std::optional<int64_t> requestEmbodimentId;

            if (requestItem.contains("robot_state"))
            {
                if (requestItem["robot_state"].is_string())
                {
                    requestRobotState = loadRobotStateBin(requestItem["robot_state"].get<std::string>());
                }
                else
                {
                    requestRobotState = flattenRobotStateJson(requestItem["robot_state"]);
                }
            }
            if (requestItem.contains("embodiment_id"))
            {
                requestEmbodimentId = requestItem["embodiment_id"].get<int64_t>();
            }

            check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
                format::fmtstr("Input rejected: too many messages in request %zu: %zu (max: %zu). Limit defined in %s.",
                    requestIdx, messagesArray.size(), limits::security::kMaxMessagesPerRequest,
                    limits::kInputLimitsLocation));

            for (auto const& messageJson : messagesArray)
            {
                check::check(messageJson.contains("role") && messageJson.contains("content"),
                    "Each message must have 'role' and 'content' fields");

                rt::Message chatMsg;
                chatMsg.role = messageJson["role"].get<std::string>();

                auto const& contentJson = messageJson["content"];

                if (contentJson.is_string())
                {
                    std::string const& contentStr = contentJson.get<std::string>();
                    check::check(contentStr.size() <= limits::security::kMaxMessageContentSizeBytes,
                        format::fmtstr("Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
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
                    check::check(contentJson.size() <= limits::security::kMaxContentItemsPerMessage,
                        format::fmtstr("Input rejected: too many content items in message %zu: %zu (max: %zu). "
                                       "Limit defined in %s.",
                            requestIdx, contentJson.size(), limits::security::kMaxContentItemsPerMessage,
                            limits::kInputLimitsLocation));

                    for (auto const& contentItemJson : contentJson)
                    {
                        check::check(contentItemJson.contains("type"), "Each content item must have a 'type' field");

                        rt::Message::MessageContent msgContent;
                        msgContent.type = contentItemJson["type"].get<std::string>();

                        if (msgContent.type == "text")
                        {
                            std::string const& textContent = contentItemJson["text"].get<std::string>();
                            check::check(textContent.size() <= limits::security::kMaxMessageContentSizeBytes,
                                format::fmtstr(
                                    "Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
                                    "Limit defined in %s.",
                                    requestIdx, textContent.size(), limits::security::kMaxMessageContentSizeBytes,
                                    limits::kInputLimitsLocation));
                            msgContent.content = textContent;
                        }
                        else if (msgContent.type == "image")
                        {
                            msgContent.content = contentItemJson["image"].get<std::string>();
                            auto image = rt::imageUtils::loadImageFromFile(msgContent.content);
                            if (image.buffer != nullptr)
                            {
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
                        else if (msgContent.type == "state")
                        {
                            if (contentItemJson.contains("file"))
                            {
                                requestRobotState = loadRobotStateBin(contentItemJson["file"].get<std::string>());
                            }
                            else if (contentItemJson.contains("values"))
                            {
                                requestRobotState = flattenRobotStateJson(contentItemJson["values"]);
                            }
                            else
                            {
                                throw std::runtime_error("Content type 'state' requires 'file' or 'values'");
                            }
                        }
                        else if (msgContent.type == "embodiment_id")
                        {
                            requestEmbodimentId = contentItemJson["embodiment_id"].get<int64_t>();
                        }
                        else
                        {
                            throw std::runtime_error(
                                format::fmtstr("Content type must be 'text', 'image', 'trajectory', 'state', or "
                                               "'embodiment_id', but got: %s",
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

            rt::LLMGenerationRequest::Request request;
            request.messages = std::move(chatMessages);
            request.imageBuffers = std::move(imageBuffers);
            request.audioBuffers = std::move(audioBuffers);
            request.pastTrajectory = std::move(requestPastTrajectory);
            request.robotState = std::move(requestRobotState);
            request.embodimentId = requestEmbodimentId;
            batchRequest.requests.push_back(std::move(request));
        }

        if (!batchLoraWeightsName.empty())
        {
            batchRequest.loraWeightsName = batchLoraWeightsName;
        }

        batchedRequests.push_back(std::move(batchRequest));
    }

    return std::make_pair(std::move(loraWeightsMap), std::move(batchedRequests));
}

} // namespace

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "vla_inference");
    VlaInferenceArgs args;
    if (!parseVlaInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    auto pluginHandles = loadEdgellmPluginLib();

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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::unique_ptr<rt::VlaInferenceRuntime> runtime{nullptr};
    try
    {
        runtime = std::make_unique<rt::VlaInferenceRuntime>(
            args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize VlaInferenceRuntime: %s", e.what());
        return EXIT_FAILURE;
    }

    if (args.actionNoiseSeed >= 0)
    {
        runtime->setActionNoiseSeed(args.actionNoiseSeed);
    }

    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
    }

    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string const errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;

    LOG_INFO("Processing %zu batched requests...", batchedRequests.size());
    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        auto& request = batchedRequests[requestIdx];
        rt::LLMGenerationResponse response;

        bool requestStatus = runtime->handleRequest(request, response, stream);

        if (requestStatus)
        {
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
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            std::string outputText = (requestStatus && batchIdx < response.outputTexts.size())
                ? response.outputTexts[batchIdx]
                : errorMessage;
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = requestIdx;
            responseJson["batch_idx"] = batchIdx;

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
                    msgJson["content"].push_back(contentJson);
                }
                messagesJson.push_back(msgJson);
            }
            responseJson["messages"] = messagesJson;

            if (requestStatus && batchIdx < response.outputActions.size()
                && !response.outputActions[batchIdx].empty())
            {
                responseJson["actions"] = response.outputActions[batchIdx];
            }
            if (requestStatus && batchIdx < response.outputTrajectories.size()
                && !response.outputTrajectories[batchIdx].empty())
            {
                nlohmann::json trajJson = nlohmann::json::array();
                for (auto const& pt : response.outputTrajectories[batchIdx])
                {
                    trajJson.push_back(nlohmann::json::array({pt.first, pt.second}));
                }
                responseJson["trajectory"] = std::move(trajJson);
            }
            outputData["responses"].push_back(responseJson);
        }
    }

    LOG_INFO("Processing complete: %zu/%zu batched requests successful", batchedRequests.size() - failedCount,
        batchedRequests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4);
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

    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
