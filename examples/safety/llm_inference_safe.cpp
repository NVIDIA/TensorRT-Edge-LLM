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

#include "safety_inference/runtime/safetyRuntime.h"
#include "safety_inference/runtime/safetyStatusUtils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace
{
namespace safety = trt_edgellm::safety;

struct Args
{
    safety::SafetyRuntimeArtifactBundle artifacts;
    safety::SafetyRuntimeOptions options;
    std::filesystem::path inputFile;
    std::filesystem::path outputFile;
    bool help{false};
};

Args parseArgs(int argc, char* argv[])
{
    option const options[]{{"engineDir", required_argument, nullptr, 'e'},
        {"multimodalEngineDir", required_argument, nullptr, 'v'}, {"inputFile", required_argument, nullptr, 'i'},
        {"outputFile", required_argument, nullptr, 'o'}, {"enableCudaGraphCapture", no_argument, nullptr, 'g'},
        {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
    Args args;
    for (int opt = getopt_long(argc, argv, "", options, nullptr); opt != -1;
        opt = getopt_long(argc, argv, "", options, nullptr))
    {
        switch (opt)
        {
        case 'e': args.artifacts.engineDir = optarg; break;
        case 'v': args.artifacts.multimodalEngineDir = optarg; break;
        case 'i': args.inputFile = optarg; break;
        case 'o': args.outputFile = optarg; break;
        case 'g': args.options.enableCudaGraphCapture = true; break;
        case 'h': args.help = true; break;
        default: throw std::invalid_argument("Invalid command-line option");
        }
    }
    if (!args.help
        && (optind != argc || args.artifacts.engineDir.empty() || args.artifacts.multimodalEngineDir.empty()
            || args.inputFile.empty() || args.outputFile.empty()))
    {
        throw std::invalid_argument("Required: --engineDir --multimodalEngineDir --inputFile --outputFile");
    }
    return args;
}

using ImageOwner = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;

void loadImage(std::filesystem::path const& path, safety::LLMGenerationRequest& request, ImageOwner& image)
{
    int width{}, height{}, channels{};
    constexpr int kRGB_CHANNELS{3};
    image.reset(stbi_load(path.c_str(), &width, &height, &channels, kRGB_CHANNELS));
    if (image == nullptr || width <= 0 || height <= 0
        || static_cast<std::size_t>(width)
            > std::numeric_limits<std::size_t>::max() / kRGB_CHANNELS / static_cast<std::size_t>(height))
    {
        throw std::invalid_argument("Cannot decode RGB image: " + path.string());
    }
    auto const bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * kRGB_CHANNELS;
    request.requests[0].imageBuffers.push_back({image.get(), bytes, width, height});
}

safety::Message parseMessage(nlohmann::json const& message, std::filesystem::path const& inputDirectory,
    safety::LLMGenerationRequest& request, ImageOwner& image)
{
    safety::Message parsed;
    parsed.role = message.at("role").get<std::string>();
    auto const& content = message.at("content");
    if (!content.is_array())
    {
        throw std::invalid_argument("Expected a content array");
    }
    for (auto const& item : content)
    {
        auto const type = item.at("type").get<std::string>();
        if (type == "text")
        {
            parsed.contents.push_back({type, item.at("text").get<std::string>()});
        }
        else if (type == "image" && image == nullptr)
        {
            loadImage(inputDirectory / item.at("image").get<std::string>(), request, image);
            parsed.contents.push_back({type, ""});
        }
        else
        {
            throw std::invalid_argument("Expected text content and exactly one image");
        }
    }
    return parsed;
}

void loadRequest(std::filesystem::path const& path, safety::LLMGenerationRequest& request, ImageOwner& image)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::invalid_argument("Cannot open input JSON");
    }
    auto const json = nlohmann::json::parse(input);
    for (auto const& entry : json.items())
    {
        if (entry.key() != "requests" && entry.key() != "max_generate_length")
        {
            throw std::invalid_argument("Unsupported request setting: " + entry.key());
        }
    }
    auto const& length = json.at("max_generate_length");
    if (!length.is_number_integer())
    {
        throw std::invalid_argument("max_generate_length must be an integer");
    }
    request.maxGenerateLength = length.get<int64_t>();
    auto const& requests = json.at("requests");
    if (request.maxGenerateLength <= 0 || !requests.is_array() || requests.size() != 1)
    {
        throw std::invalid_argument("Expected one request and a positive max_generate_length");
    }
    request.requests.resize(1);
    auto const& messages = requests[0].at("messages");
    if (!messages.is_array() || messages.empty())
    {
        throw std::invalid_argument("Expected nonempty messages");
    }
    for (auto const& message : messages)
    {
        request.requests[0].messages.push_back(parseMessage(message, path.parent_path(), request, image));
    }
    if (image == nullptr)
    {
        throw std::invalid_argument("Missing image content");
    }
}

void prepareResponse(safety::LLMGenerationRequest const& request, safety::LLMGenerationResponse& response)
{
    response.outputIds.resize(1);
    response.outputIds[0].reserve(static_cast<std::size_t>(request.maxGenerateLength));
    response.outputTexts.resize(1);
    constexpr std::size_t kOUTPUT_TEXT_BYTES{64 * 1024};
    response.outputTexts[0].reserve(kOUTPUT_TEXT_BYTES);
}

void writeResponse(std::filesystem::path const& path, safety::LLMGenerationResponse const& response)
{
    std::ofstream output(path);
    output << nlohmann::json{{"output_ids", response.outputIds}, {"output_texts", response.outputTexts}}.dump(2)
           << '\n';
    output.close();
    if (!output)
    {
        throw std::runtime_error("Cannot write output JSON");
    }
}

bool closeRuntime(std::unique_ptr<safety::SafetyRuntime>& runtime)
{
    if (runtime == nullptr)
    {
        return true;
    }
    if (!runtime->fatal())
    {
        auto const status = runtime->deinitialize();
        if (status.okStatus())
        {
            runtime.reset();
            return true;
        }
        std::cerr << "Teardown failed: " << status.message() << '\n';
    }
    auto& recorder = runtime->errorRecorder();
    for (int32_t index = 0; index < recorder.getNbErrors(); ++index)
    {
        std::cerr << "Runtime error: " << recorder.getErrorDesc(index) << '\n';
    }
    // This one-shot application exits normally after diagnostics. Keep graphs and their recorder alive until exit.
    // A long-lived application retains the owner for further queries and delegates recovery to its supervisor.
    (void) runtime.release();
    return false;
}
} // namespace

int main(int argc, char* argv[])
{
    std::unique_ptr<safety::SafetyRuntime> runtime;
    ImageOwner image{nullptr, &stbi_image_free};
    safety::LLMGenerationRequest request;
    safety::LLMGenerationResponse response;
    int result{EXIT_FAILURE};
    try
    {
        auto const args = parseArgs(argc, argv);
        if (args.help)
        {
            std::cout << "llm_inference_safe --engineDir DIR --multimodalEngineDir DIR\n"
                         "  --inputFile request.json --outputFile result.json [--enableCudaGraphCapture]\n";
            return EXIT_SUCCESS;
        }
        loadRequest(args.inputFile, request, image);
        prepareResponse(request, response);

        auto status = safety::fromCudaError(cudaSetDevice(0), "cudaSetDevice", true);
        if (!status.okStatus())
        {
            return EXIT_FAILURE;
        }
        runtime = std::make_unique<safety::SafetyRuntime>();
        status = runtime->initialize(args.artifacts, args.options);
        if (!status.okStatus())
        {
            std::cerr << "Initialization failed: " << status.message() << '\n';
        }
        else if (!runtime->handleRequest(request, response))
        {
            std::cerr << "Request failed: " << runtime->requestStatus().message() << '\n';
        }
        else if (closeRuntime(runtime))
        {
            writeResponse(args.outputFile, response);
            result = EXIT_SUCCESS;
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
    }
    catch (...)
    {
        std::cerr << "Unexpected application error\n";
    }
    if (!closeRuntime(runtime))
    {
        result = EXIT_FAILURE;
    }
    return result;
}
