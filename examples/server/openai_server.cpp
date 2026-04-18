/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "common/trtUtils.h"
#include "openai_utils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>
#include <getopt.h>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Command line option IDs
enum ServerOptionId : int
{
    HELP = 1000,
    ENGINE_DIR = 1001,
    MULTIMODAL_ENGINE_DIR = 1002,
    PORT = 1003,
    HOST = 1004,
    LORA_WEIGHTS = 1005
};

struct ServerArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    int port{8080};
    std::string host{"0.0.0.0"};
    std::string loraWeightsFile{""};
};

void printUsage(const char* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --engineDir=<path> [--multimodalEngineDir=<path>] "
              << "[--port=<number>] [--host=<address>] [--loraWeights=<path>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                  Display this help message" << std::endl;
    std::cerr << "  --engineDir             Path to TensorRT engine directory (required)"
              << std::endl;
    std::cerr << "  --multimodalEngineDir   Path to multimodal engine directory (optional)"
              << std::endl;
    std::cerr << "  --port                  Server port (default: 8080)" << std::endl;
    std::cerr << "  --host                  Server host address (default: 0.0.0.0)"
              << std::endl;
    std::cerr << "  --loraWeights           Path to LoRA weights config JSON (optional)"
              << std::endl;
}

bool parseServerArgs(ServerArgs& args, int argc, char* argv[])
{
    static struct option serverOptions[] = {
        {"help", no_argument, 0, ServerOptionId::HELP},
        {"engineDir", required_argument, 0, ServerOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, ServerOptionId::MULTIMODAL_ENGINE_DIR},
        {"port", required_argument, 0, ServerOptionId::PORT},
        {"host", required_argument, 0, ServerOptionId::HOST},
        {"loraWeights", required_argument, 0, ServerOptionId::LORA_WEIGHTS},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", serverOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case ServerOptionId::HELP:
            args.help = true;
            return true;
        case ServerOptionId::ENGINE_DIR:
            args.engineDir = optarg;
            break;
        case ServerOptionId::MULTIMODAL_ENGINE_DIR:
            args.multimodalEngineDir = optarg;
            break;
        case ServerOptionId::PORT:
            try
            {
                args.port = std::stoi(optarg);
                if (args.port <= 0 || args.port > 65535)
                {
                    LOG_ERROR("Invalid port number: %s", optarg);
                    return false;
                }
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("Invalid port number: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::HOST:
            args.host = optarg;
            break;
        case ServerOptionId::LORA_WEIGHTS:
            args.loraWeightsFile = optarg;
            break;
        default:
            return false;
        }
    }

    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }

    return true;
}

// Load LoRA weights config from JSON file
std::unordered_map<std::string, std::string> loadLoraWeightsConfig(const std::string& filePath)
{
    std::unordered_map<std::string, std::string> loraWeightsMap;

    if (filePath.empty())
    {
        return loraWeightsMap;
    }

    std::ifstream configFile(filePath);
    if (!configFile.is_open())
    {
        LOG_WARNING("Failed to open LoRA weights config file: %s", filePath.c_str());
        return loraWeightsMap;
    }

    try
    {
        Json config = Json::parse(configFile);
        configFile.close();

        if (config.is_object())
        {
            for (const auto& [name, path] : config.items())
            {
                if (path.is_string())
                {
                    loraWeightsMap[name] = path.get<std::string>();
                    LOG_INFO("Registered LoRA weights '%s' -> '%s'",
                             name.c_str(), loraWeightsMap[name].c_str());
                }
            }
        }
    }
    catch (const Json::parse_error& e)
    {
        LOG_ERROR("Failed to parse LoRA weights config: %s", e.what());
    }

    return loraWeightsMap;
}

// Extract model ID from engine directory name
std::string getModelId(const std::string& engineDir)
{
    std::filesystem::path p(engineDir);
    return p.filename().string();
}

int main(int argc, char* argv[])
{
    ServerArgs args;
    if (!parseServerArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    // Load TensorRT plugins
    auto pluginHandles = loadEdgellmPluginLib();

    // Load LoRA weights config if provided
    auto loraWeightsMap = loadLoraWeightsConfig(args.loraWeightsFile);
    LOG_INFO("Loaded %zu LoRA weights configurations", loraWeightsMap.size());

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize LLM Inference Runtime
    LOG_INFO("Initializing LLM Inference Runtime with engine: %s", args.engineDir.c_str());
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        runtime = std::make_unique<rt::LLMInferenceRuntime>(
            args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
        return EXIT_FAILURE;
    }

    // Capture CUDA graph for decoding optimization
    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("Failed to capture CUDA graph, proceeding with normal execution");
    }

    // Get model ID for responses
    std::string modelId = getModelId(args.engineDir);

    LOG_INFO("Server initialized successfully");
    LOG_INFO("Model ID: %s", modelId.c_str());

    // Create HTTP server
    httplib::Server server;

    // Set server timeout
    server.set_read_timeout(300);  // 5 minutes for long generations
    server.set_write_timeout(300);

    // POST /v1/chat/completions - OpenAI API endpoint
    server.Post("/v1/chat/completions", [&runtime, &modelId, &stream](const httplib::Request& req, httplib::Response& res) {
        auto startTime = std::chrono::high_resolution_clock::now();
        LOG_INFO("Received request: /v1/chat/completions");

        // Parse JSON request body
        Json requestJson;
        try
        {
            requestJson = Json::parse(req.body);
        }
        catch (const Json::parse_error& e)
        {
            LOG_ERROR("Invalid JSON body: %s", e.what());
            res.status = 400;
            res.set_content(server::formatErrorResponse("Invalid JSON body", "invalid_request_error").dump(),
                           "application/json");
            return;
        }

        // Convert to LLMGenerationRequest
        rt::LLMGenerationRequest llmRequest;
        std::string errorMessage;
        if (!server::parseOpenAIRequest(requestJson, llmRequest, errorMessage))
        {
            LOG_ERROR("Request parsing failed: %s", errorMessage.c_str());
            res.status = 400;
            res.set_content(server::formatErrorResponse(errorMessage, "invalid_request_error").dump(),
                           "application/json");
            return;
        }

        // Log input prompt
        std::string inputPrompt;
        for (const auto& msg : llmRequest.requests[0].messages)
        {
            for (const auto& content : msg.contents)
            {
                if (content.type == "text")
                {
                    inputPrompt += "[" + msg.role + "] " + content.content + "\n";
                }
                else if (content.type == "image_url")
                {
                    inputPrompt += "[" + msg.role + "] <image>\n";
                }
            }
        }
        LOG_INFO("Input prompt:\n%s", inputPrompt.c_str());

        // Execute inference
        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        rt::LLMGenerationResponse llmResponse;
        bool success = runtime->handleRequest(llmRequest, llmResponse, stream);
        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        auto inferenceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEndTime - inferenceStartTime).count();

        if (!success || llmResponse.outputTexts.empty())
        {
            LOG_ERROR("Inference failed");
            res.status = 500;
            res.set_content(server::formatErrorResponse("Inference failed", "internal_error").dump(),
                           "application/json");
            return;
        }

        // Log output
        LOG_INFO("Output: %s", llmResponse.outputTexts[0].c_str());
        LOG_INFO("Inference time: %ld ms", inferenceDuration);

        // Format response in OpenAI API format
        Json responseJson = server::formatOpenAIResponse(llmResponse, modelId, llmRequest, nullptr);

        res.status = 200;
        res.set_content(responseJson.dump(), "application/json");

        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        LOG_INFO("Total request time: %ld ms", totalDuration);
    });

    // GET /v1/models - List available models
    server.Get("/v1/models", [&modelId](const httplib::Request& req, httplib::Response& res) {
        Json modelsJson;
        modelsJson["object"] = "list";

        Json modelData;
        modelData["id"] = modelId;
        modelData["object"] = "model";
        modelData["created"] = server::getCurrentTimestamp();
        modelData["owned_by"] = "nvidia";

        modelsJson["data"] = Json::array({modelData});

        res.status = 200;
        res.set_content(modelsJson.dump(), "application/json");
    });

    // Health check endpoint
    server.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        Json healthJson;
        healthJson["status"] = "healthy";
        res.status = 200;
        res.set_content(healthJson.dump(), "application/json");
    });

    // Handle 404 for other endpoints
    server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (res.status == 404)
        {
            res.set_content(server::formatErrorResponse("Not found", "not_found_error").dump(),
                           "application/json");
        }
    });

    LOG_INFO("Starting server on %s:%d", args.host.c_str(), args.port);
    LOG_INFO("Endpoints available:");
    LOG_INFO("  POST /v1/chat/completions");
    LOG_INFO("  GET  /v1/models");
    LOG_INFO("  GET  /health");

    if (!server.listen(args.host, args.port))
    {
        LOG_ERROR("Failed to start server on %s:%d", args.host.c_str(), args.port);
        return EXIT_FAILURE;
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
