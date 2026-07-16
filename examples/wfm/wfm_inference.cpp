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
#include "runtime/wfmInferenceRuntime.h"
#include "runtime/wfmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

enum WfmInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    NUM_INFERENCE_STEPS = 910,
    SEED = 911
};

struct WfmInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string inputFile;
    std::string outputFile;
    std::string profileOutputFile;
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    int32_t numInferenceSteps{-1}; //!< -1 = use value from input file / config.json
    int32_t seed{42};
};

struct WfmInputGlobals
{
    int32_t numInferenceSteps{0};
    int32_t seed{0};
};

struct WfmRequestSpec
{
    std::string prompt;
    bool generateSound{false};
    int32_t numInferenceSteps{0};
    int32_t seed{0};
    std::string pixelsFile;
    std::string waveformFile;
    std::string outputVideoFile;
    std::string outputWaveformFile;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path>] [--inputFile=<path>] [--outputFile=<path>] [--dumpProfile] "
                 "[--profileOutputFile=<path>] [--warmup=<number>] [--debug] [--dumpOutput] "
                 "[--numInferenceSteps=<number>] [--seed=<number>]"
              << std::endl;
    std::cerr << "Cosmos WFM video generation inference from an exported engine bundle." << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests (required)" << std::endl;
    std::cerr << "  --engineDir               Path to exported WFM engine directory (required)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (required)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output shapes to console" << std::endl;
    std::cerr << "  --numInferenceSteps       Override num_inference_steps from input file" << std::endl;
    std::cerr << "  --seed                    Default random seed when not set in input JSON (default: 42)" << std::endl;
}

bool parseWfmInferenceArgs(WfmInferenceArgs& args, int argc, char* argv[]) // NOLINT(readability-function-cognitive-complexity)
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, WfmInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, WfmInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, WfmInferenceOptionId::ENGINE_DIR},
        {"outputFile", required_argument, 0, WfmInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, WfmInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, WfmInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, WfmInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, WfmInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, WfmInferenceOptionId::DUMP_OUTPUT},
        {"numInferenceSteps", required_argument, 0, WfmInferenceOptionId::NUM_INFERENCE_STEPS},
        {"seed", required_argument, 0, WfmInferenceOptionId::SEED}, {0, 0, 0, 0}};

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case WfmInferenceOptionId::HELP: args.help = true; return true;
        case WfmInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case WfmInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case WfmInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case WfmInferenceOptionId::DEBUG: args.debug = true; break;
        case WfmInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case WfmInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case WfmInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case WfmInferenceOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case WfmInferenceOptionId::NUM_INFERENCE_STEPS:
            try
            {
                args.numInferenceSteps = std::stoi(optarg);
                if (args.numInferenceSteps <= 0)
                {
                    LOG_ERROR("Invalid numInferenceSteps value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid numInferenceSteps value: %s", optarg);
                return false;
            }
            break;
        case WfmInferenceOptionId::SEED:
            try
            {
                args.seed = std::stoi(optarg);
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid seed value: %s", optarg);
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

std::vector<half> loadFp16BinaryFile(std::filesystem::path const& path, std::size_t expectedElements)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check::check(file.is_open(), "Failed to open binary file: " + path.string());
    auto const fileSize = static_cast<std::size_t>(file.tellg());
    check::check(fileSize % sizeof(half) == 0,
        "Binary file size must be a multiple of 2 bytes (fp16): " + path.string());
    std::size_t const numElements = fileSize / sizeof(half);
    check::check(expectedElements == 0 || numElements == expectedElements,
        format::fmtstr("Binary file %s has %zu fp16 elements, expected %zu", path.string().c_str(), numElements,
            expectedElements));
    file.seekg(0);
    std::vector<half> values(numElements);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(fileSize));
    check::check(file.good(), "Failed to read binary file: " + path.string());
    return values;
}

void saveFp16BinaryFile(std::filesystem::path const& path, half const* data, std::size_t numElements)
{
    std::ofstream file(path, std::ios::binary);
    check::check(file.is_open(), "Failed to open output binary file: " + path.string());
    file.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(numElements * sizeof(half)));
    check::check(file.good(), "Failed to write output binary file: " + path.string());
}

std::shared_ptr<rt::Tensor> makeRandomPixels(
    rt::CosmosEngineConfig const& config, int32_t seed, cudaStream_t stream)
{
    auto pixels = std::make_shared<rt::Tensor>(
        rt::Coords({1, 3, config.numFrames, config.height, config.width}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "wfm_inference::input_pixels");

    std::vector<half> host(static_cast<std::size_t>(pixels->getShape().volume()));
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.F, 1.F);
    for (auto& value : host)
    {
        value = __float2half(dist(rng));
    }

    CUDA_CHECK(cudaMemcpyAsync(pixels->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return pixels;
}

std::shared_ptr<rt::Tensor> loadPixelsFromFile(
    std::filesystem::path const& path, rt::CosmosEngineConfig const& config, cudaStream_t stream)
{
    std::size_t const expectedElements
        = static_cast<std::size_t>(config.numFrames) * config.height * config.width * 3U;
    auto host = loadFp16BinaryFile(path, expectedElements);
    auto pixels = std::make_shared<rt::Tensor>(
        rt::Coords({1, 3, config.numFrames, config.height, config.width}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "wfm_inference::input_pixels");
    CUDA_CHECK(cudaMemcpyAsync(pixels->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return pixels;
}

std::shared_ptr<rt::Tensor> loadWaveformFromFile(
    std::filesystem::path const& path, rt::CosmosEngineConfig const& config, cudaStream_t stream, int64_t& numSamples)
{
    auto host = loadFp16BinaryFile(path, 0);
    numSamples = static_cast<int64_t>(host.size());
    check::check(numSamples > 0, "Waveform file is empty: " + path.string());

    auto waveform = std::make_shared<rt::Tensor>(
        rt::Coords({1, 1, numSamples}), rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wfm_inference::input_waveform");
    CUDA_CHECK(cudaMemcpyAsync(waveform->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return waveform;
}

void saveTensorToFp16File(std::filesystem::path const& path, rt::Tensor const& tensor, cudaStream_t stream)
{
    check::check(tensor.getDataType() == nvinfer1::DataType::kHALF, "saveTensorToFp16File only supports fp16 tensors");
    std::size_t const numElements = static_cast<std::size_t>(tensor.getShape().volume());
    std::vector<half> host(numElements);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.rawPointer(), numElements * sizeof(half), cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    saveFp16BinaryFile(path, host.data(), numElements);
}

int32_t resolveSeed(int32_t requestSeed, int32_t globalSeed, int32_t cliSeed)
{
    if (requestSeed != 0)
    {
        return requestSeed;
    }
    if (globalSeed != 0)
    {
        return globalSeed;
    }
    return cliSeed;
}

int32_t resolveNumInferenceSteps(int32_t requestSteps, int32_t globalSteps, int32_t cliSteps)
{
    if (requestSteps > 0)
    {
        return requestSteps;
    }
    if (globalSteps > 0)
    {
        return globalSteps;
    }
    if (cliSteps > 0)
    {
        return cliSteps;
    }
    return 0;
}

rt::WFMGenerationRequest buildWfmRequest(WfmRequestSpec const& spec, WfmInputGlobals const& globals,
    WfmInferenceArgs const& args, rt::CosmosEngineConfig const& config, cudaStream_t stream)
{
    rt::WFMGenerationRequest request{};
    request.prompt = spec.prompt;
    request.generateSound = spec.generateSound;
    request.numInferenceSteps
        = resolveNumInferenceSteps(spec.numInferenceSteps, globals.numInferenceSteps, args.numInferenceSteps);
    request.seed = resolveSeed(spec.seed, globals.seed, args.seed);

    if (!spec.pixelsFile.empty())
    {
        request.pixels.buffer = loadPixelsFromFile(spec.pixelsFile, config, stream);
    }
    else
    {
        request.pixels.buffer = makeRandomPixels(config, request.seed, stream);
    }
    request.pixels.batch = 1;
    request.pixels.channels = 3;
    request.pixels.numFrames = config.numFrames;
    request.pixels.height = config.height;
    request.pixels.width = config.width;

    if (!spec.waveformFile.empty())
    {
        int64_t numSamples{0};
        request.inputWaveform.buffer = loadWaveformFromFile(spec.waveformFile, config, stream, numSamples);
        request.inputWaveform.batch = 1;
        request.inputWaveform.sampleRate = config.sampleRate;
        request.inputWaveform.numSamples = numSamples;
    }

    return request;
}

std::pair<WfmInputGlobals, std::vector<WfmRequestSpec>> parseInputFile(std::filesystem::path const& inputFilePath)
{
    WfmInputGlobals globals;
    std::vector<WfmRequestSpec> requestSpecs;

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

    globals.numInferenceSteps = inputData.value("num_inference_steps", 0);
    globals.seed = inputData.value("seed", 0);

    check::check(inputData.contains("requests") && inputData["requests"].is_array(),
        "'requests' array not found in input file");

    auto const& requestsArray = inputData["requests"];
    for (size_t requestIdx = 0; requestIdx < requestsArray.size(); ++requestIdx)
    {
        auto const& requestItem = requestsArray[requestIdx];
        check::check(requestItem.is_object(), "Each request must be a JSON object");

        WfmRequestSpec spec;
        check::check(requestItem.contains("prompt") && requestItem["prompt"].is_string(),
            format::fmtstr("Request %zu must contain a string 'prompt' field", requestIdx));
        spec.prompt = requestItem["prompt"].get<std::string>();
        check::check(spec.prompt.size() <= limits::tokenizer::kMaxInputTextSizeBytes,
            format::fmtstr("Input rejected: prompt too large in request %zu: %zu bytes (max: %zu). Limit defined in "
                           "%s.",
                requestIdx, spec.prompt.size(), limits::tokenizer::kMaxInputTextSizeBytes,
                limits::kInputLimitsLocation));

        spec.generateSound = requestItem.value("generate_sound", false);
        spec.numInferenceSteps = requestItem.value("num_inference_steps", 0);
        spec.seed = requestItem.value("seed", 0);
        spec.pixelsFile = requestItem.value("pixels_file", "");
        spec.waveformFile = requestItem.value("waveform_file", "");
        spec.outputVideoFile = requestItem.value("output_video_file", "");
        spec.outputWaveformFile = requestItem.value("output_waveform_file", "");

        requestSpecs.push_back(std::move(spec));
    }

    if (requestSpecs.empty())
    {
        throw std::runtime_error("No requests found in input file");
    }

    return std::make_pair(std::move(globals), std::move(requestSpecs));
}

} // namespace

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "wfm_inference");

    WfmInferenceArgs args;
    if (!parseWfmInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    bool const profilerEnabled = args.dumpProfile || !args.profileOutputFile.empty();
    MemoryMonitor memoryMonitor;
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    auto pluginHandles = loadEdgellmPluginLib();

    WfmInputGlobals globals;
    std::vector<WfmRequestSpec> requestSpecs;
    try
    {
        std::tie(globals, requestSpecs) = parseInputFile(args.inputFile);
        LOG_INFO("Successfully parsed %zu requests from input file.", requestSpecs.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::unique_ptr<rt::WFMInferenceRuntime> wfmRuntime;
    try
    {
        wfmRuntime = std::make_unique<rt::WFMInferenceRuntime>(args.engineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize WFMInferenceRuntime: %s", e.what());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    auto const& config = wfmRuntime->getEngineConfig();

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto warmupRequest = buildWfmRequest(requestSpecs[0], globals, args, config, stream);

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::WFMGenerationResponse warmupResponse;
            bool const requestStatus = wfmRuntime->handleRequest(warmupRequest, warmupResponse, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!requestStatus)
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                CUDA_CHECK(cudaStreamDestroy(stream));
                return EXIT_FAILURE;
            }
        }
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(true);
        gTimer.reset();
    }

    Json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["engine_dir"] = args.engineDir;
    outputData["responses"] = Json::array();

    bool hasFailedRequest = false;
    size_t failedCount = 0;
    std::string const errorMessage = "TensorRT Edge LLM cannot handle this WFM request.";

    LOG_INFO("Processing %zu requests...", requestSpecs.size());
    for (size_t requestIdx = 0; requestIdx < requestSpecs.size(); ++requestIdx)
    {
        auto const& spec = requestSpecs[requestIdx];
        rt::WFMGenerationResponse response;

        size_t const progressInterval = std::max(size_t(1), std::min(requestSpecs.size() / 10, size_t(100)));
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == requestSpecs.size() - 1)
        {
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, requestSpecs.size(),
                100.0 * (requestIdx + 1) / requestSpecs.size());
        }

        rt::WFMGenerationRequest request = buildWfmRequest(spec, globals, args, config, stream);

        bool requestStatus = false;
        if (profilerEnabled)
        {
            TIME_STAGE("wfm_inference", stream);
            requestStatus = wfmRuntime->handleRequest(request, response, stream);
        }
        else
        {
            requestStatus = wfmRuntime->handleRequest(request, response, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Json responseJson;
        responseJson["request_idx"] = requestIdx;
        responseJson["prompt"] = sanitizeUtf8ForJson(spec.prompt);
        responseJson["generate_sound"] = spec.generateSound;
        responseJson["num_inference_steps"] = request.numInferenceSteps;
        responseJson["seed"] = request.seed;
        responseJson["success"] = requestStatus;

        if (requestStatus)
        {
            if (args.dumpOutput)
            {
                if (response.outputVideo.buffer)
                {
                    LOG_INFO("Request %zu output video shape: %s", requestIdx,
                        response.outputVideo.buffer->getShape().formatString().c_str());
                }
                if (response.outputWaveform.buffer)
                {
                    LOG_INFO("Request %zu output waveform shape: %s", requestIdx,
                        response.outputWaveform.buffer->getShape().formatString().c_str());
                }
            }

            if (!spec.outputVideoFile.empty() && response.outputVideo.buffer)
            {
                try
                {
                    saveTensorToFp16File(spec.outputVideoFile, *response.outputVideo.buffer, stream);
                    responseJson["output_video_file"] = spec.outputVideoFile;
                }
                catch (std::exception const& e)
                {
                    LOG_ERROR("Failed to write output video for request %zu: %s", requestIdx, e.what());
                    responseJson["output_video_file_error"] = e.what();
                    hasFailedRequest = true;
                    ++failedCount;
                }
            }
            else if (response.outputVideo.buffer)
            {
                responseJson["output_video_shape"] = response.outputVideo.buffer->getShape().formatString();
            }

            if (!spec.outputWaveformFile.empty() && response.outputWaveform.buffer)
            {
                try
                {
                    saveTensorToFp16File(spec.outputWaveformFile, *response.outputWaveform.buffer, stream);
                    responseJson["output_waveform_file"] = spec.outputWaveformFile;
                }
                catch (std::exception const& e)
                {
                    LOG_ERROR("Failed to write output waveform for request %zu: %s", requestIdx, e.what());
                    responseJson["output_waveform_file_error"] = e.what();
                    hasFailedRequest = true;
                    ++failedCount;
                }
            }
            else if (response.outputWaveform.buffer)
            {
                responseJson["output_waveform_shape"] = response.outputWaveform.buffer->getShape().formatString();
            }
        }
        else
        {
            hasFailedRequest = true;
            ++failedCount;
            responseJson["error"] = errorMessage;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        outputData["responses"].push_back(std::move(responseJson));
    }

    LOG_INFO("Processing complete: %zu/%zu requests successful", requestSpecs.size() - failedCount,
        requestSpecs.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu REQUESTS FAILED ***", failedCount);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== WFM Performance Summary ===" << std::endl;
        outputMemoryProfile(profileOutput, memoryMonitor);
        outputLayerProfiles(profileOutput, false);
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    if (!args.profileOutputFile.empty())
    {
        try
        {
            Json profileJson;
            addJsonTimingStages(profileJson);
            addJsonMemorySummary(profileJson, memoryMonitor);

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2);
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                CUDA_CHECK(cudaStreamDestroy(stream));
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }
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
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
