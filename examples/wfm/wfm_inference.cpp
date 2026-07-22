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

/*
 * This executable is the command-line "front end" for WFM inference.
 *
 * It does not implement the neural network algorithms itself. Instead, it:
 *   1. Reads command-line options and a JSON file describing one or more requests.
 *   2. Loads input pixels/audio into GPU tensors (or creates random input pixels).
 *   3. Calls WFMInferenceRuntime::handleRequest(), where encode/denoise/decode occur.
 *   4. Copies requested outputs back from the GPU and writes result metadata as JSON.
 *
 * Most failures in this file concern invalid input, missing files, CUDA transfers,
 * or output writing. Model-specific execution is implemented under cpp/runtime/.
 */

//! Numeric identifiers returned by getopt_long() for the supported CLI options.
//!
//! Values begin at 900 so that they do not collide with ordinary one-character
//! options such as 'h'. This program uses long options only (for example, --help).
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

//! Values collected from the command line.
//!
//! The brace initializers are defaults. For example, if --warmup is omitted,
//! warmup remains 0; if --seed is omitted, seed remains 42.
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

//! Default values read from the top level of the input JSON file.
//!
//! An individual request can override these values. Zero means "not specified"
//! for both fields in the resolution helpers below.
struct WfmInputGlobals
{
    int32_t numInferenceSteps{0};
    int32_t seed{0};
};

//! Lightweight CPU-side description of one item in the JSON "requests" array.
//!
//! This struct contains strings and scalar settings only. buildWfmRequest() later
//! converts it into a WFMGenerationRequest containing actual GPU tensors.
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

//! Print command-line syntax and a description of every accepted option.
//!
//! @param programName The executable name, normally argv[0].
//! @note Help is written to stderr so it is visible alongside validation errors.
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

//! Parse and validate command-line options.
//!
//! getopt_long() examines argv and returns one WfmInferenceOptionId at a time.
//! Options containing numbers arrive as text, so std::stoi() converts them.
//!
//! @param[out] args Receives all parsed values.
//! @param argc Number of command-line arguments.
//! @param argv Array of argument strings.
//! @return true if parsing succeeds (including --help), otherwise false.
//! @note This also selects INFO or VERBOSE logging after validation.
bool parseWfmInferenceArgs(WfmInferenceArgs& args, int argc, char* argv[]) // NOLINT(readability-function-cognitive-complexity)
{
    // Each entry maps a long option such as "--engineDir" to an enum value.
    // required_argument means the option must be followed by a value.
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
    // getopt_long() returns -1 after it has consumed all command-line options.
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

    // These three paths are necessary for every non-help invocation.
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

//! Read an entire raw FP16 binary file into CPU memory.
//!
//! A `half` occupies two bytes. The file has no header or shape metadata, so the
//! caller must know the intended tensor shape and optionally supply its element
//! count. This is not a PNG, MP4, WAV, or other container format.
//!
//! @param path File to read.
//! @param expectedElements Required number of FP16 values, or 0 to accept any size.
//! @return A CPU vector containing the file's FP16 values.
//! @throws std::runtime_error (through check::check) if validation or reading fails.
std::vector<half> loadFp16BinaryFile(std::filesystem::path const& path, std::size_t expectedElements)
{
    // ios::ate initially positions the read cursor at the end, allowing tellg()
    // to report the file's byte size without reading the file twice.
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

//! Write CPU FP16 values as a raw binary file.
//!
//! @param path Destination path.
//! @param data Pointer to the first FP16 value.
//! @param numElements Number of values to write.
//! @throws std::runtime_error (through check::check) if opening or writing fails.
void saveFp16BinaryFile(std::filesystem::path const& path, half const* data, std::size_t numElements)
{
    std::ofstream file(path, std::ios::binary);
    check::check(file.is_open(), "Failed to open output binary file: " + path.string());
    file.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(numElements * sizeof(half)));
    check::check(file.good(), "Failed to write output binary file: " + path.string());
}

//! Create reproducible random input pixels and copy them to a GPU tensor.
//!
//! The tensor uses NCTHW order:
//!   N = batch (1), C = color channels (3), T = frames, H = height, W = width.
//! Values are sampled uniformly from [-1, 1], which is the model's normalized
//! pixel range. Supplying the same seed and configuration produces the same data.
//!
//! @param config Engine dimensions used to determine the tensor shape.
//! @param seed Seed for the CPU pseudo-random number generator.
//! @param stream CUDA stream used for the host-to-device copy.
//! @return Shared ownership of an FP16 tensor allocated on the GPU.
std::shared_ptr<rt::Tensor> makeRandomPixels(
    rt::CosmosEngineConfig const& config, int32_t seed, cudaStream_t stream)
{
    // Tensor allocates device memory because DeviceType::kGPU is requested.
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

    // cudaMemcpyAsync schedules the copy. Synchronization keeps the temporary
    // CPU vector alive until the GPU has finished reading from it.
    CUDA_CHECK(cudaMemcpyAsync(pixels->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return pixels;
}

//! Load a raw FP16 pixel tensor from disk and copy it to the GPU.
//!
//! @param path Raw FP16 input file.
//! @param config Supplies the required frame count, height, and width.
//! @param stream CUDA stream used for the host-to-device copy.
//! @return GPU tensor with shape [1, 3, frames, height, width].
//! @note The binary file has no shape metadata; its element order must already
//! match the model's expected NCTHW layout.
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

//! Load a mono FP16 waveform from disk and copy it to the GPU.
//!
//! Unlike the pixel loader, the waveform loader accepts any non-empty length.
//! The number of samples is inferred directly from the file size.
//!
//! @param path Raw FP16 waveform file (not a WAV container).
//! @param config Engine configuration; currently unused by this helper.
//! @param stream CUDA stream used for the host-to-device copy.
//! @param[out] numSamples Receives the inferred waveform length.
//! @return GPU tensor with shape [batch=1, channel=1, samples].
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

//! Copy an FP16 tensor from GPU memory and save it as raw binary.
//!
//! @param path Destination file.
//! @param tensor Source GPU tensor. Other data types are rejected.
//! @param stream CUDA stream used for the device-to-host copy.
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

//! Select the effective random seed using the configured precedence.
//!
//! Priority is per-request JSON, then top-level JSON, then command line.
//! A JSON seed of zero is treated as "unset", not as a literal seed.
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

//! Select the effective denoising-step count using configured precedence.
//!
//! Priority is per-request JSON, then top-level JSON, then command line.
//! Returning zero tells WFMInferenceRuntime to use its engine configuration.
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

//! Convert one parsed JSON request into the runtime's GPU-backed request type.
//!
//! This is the bridge between configuration/I/O and model execution. It resolves
//! defaults, prepares a pixel tensor, fills dimensions expected by validation,
//! and optionally adds an input waveform.
//!
//! @param spec Per-request values parsed from JSON.
//! @param globals Top-level defaults parsed from JSON.
//! @param args Command-line defaults.
//! @param config Dimensions and sample rate expected by the exported engines.
//! @param stream CUDA stream used while preparing tensors.
//! @return A complete request suitable for WFMInferenceRuntime::handleRequest().
rt::WFMGenerationRequest buildWfmRequest(WfmRequestSpec const& spec, WfmInputGlobals const& globals,
    WfmInferenceArgs const& args, rt::CosmosEngineConfig const& config, cudaStream_t stream)
{
    rt::WFMGenerationRequest request{};
    request.prompt = spec.prompt;
    request.generateSound = spec.generateSound;
    request.numInferenceSteps
        = resolveNumInferenceSteps(spec.numInferenceSteps, globals.numInferenceSteps, args.numInferenceSteps);
    request.seed = resolveSeed(spec.seed, globals.seed, args.seed);

    // A supplied pixels_file conditions the request on that tensor. Otherwise,
    // random normalized pixels provide a deterministic starting input.
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

    // Audio is optional. Leaving inputWaveform.buffer empty tells the runtime
    // that this request has no waveform conditioning.
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

//! Parse the input JSON into global defaults and individual request specs.
//!
//! Expected structure:
//! {
//!   "num_inference_steps": 2,       // optional global default
//!   "seed": 42,                     // optional global default
//!   "requests": [
//!     {
//!       "prompt": "...",            // required
//!       "generate_sound": false,    // optional
//!       "pixels_file": "...",       // optional raw FP16 input
//!       "waveform_file": "...",     // optional raw FP16 input
//!       "output_video_file": "...", // optional raw FP16 output
//!       "output_waveform_file": "..." // optional raw FP16 output
//!     }
//!   ]
//! }
//!
//! @param inputFilePath JSON file to parse.
//! @return Pair containing top-level defaults and all request descriptions.
//! @throws std::runtime_error for malformed JSON or invalid required fields.
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
        // prompt is the only mandatory per-request field.
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

//! Program entry point: initialize resources, run all requests, and write results.
//!
//! The requests are processed sequentially on one CUDA stream. This makes tensor
//! lifetime and runtime-owned output buffers straightforward: each response is
//! consumed before the next request begins.
int main(int argc, char* argv[])
{
    // Add a top-level NVTX range so GPU profiling tools can identify this program.
    NVTX_SCOPED_RANGE(nvtx_main, "wfm_inference");

    // Phase 1: Parse and validate command-line configuration.
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

    // Profiling is enabled if either console or JSON profiling output was requested.
    bool const profilerEnabled = args.dumpProfile || !args.profileOutputFile.empty();
    MemoryMonitor memoryMonitor;
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    // TensorRT engines may depend on custom Edge-LLM layers. Loading the plugin
    // library before deserializing engines registers those layers with TensorRT.
    auto pluginHandles = loadEdgellmPluginLib();

    // Phase 2: Read lightweight request descriptions from JSON.
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

    // Phase 3: Create one CUDA stream. Operations submitted to the same stream
    // execute in order, which simplifies synchronization between pipeline stages.
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::unique_ptr<rt::WFMInferenceRuntime> wfmRuntime;
    try
    {
        // Construction loads config.json, packing data, tokenizer data, and the
        // TensorRT engines located beneath engineDir.
        wfmRuntime = std::make_unique<rt::WFMInferenceRuntime>(args.engineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize WFMInferenceRuntime: %s", e.what());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    auto const& config = wfmRuntime->getEngineConfig();

    // Phase 4 (optional): Warm up GPU kernels and engine state. Warmup uses the
    // first request repeatedly and is deliberately excluded from profiling.
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

    // Phase 5: Prepare the JSON document that will summarize every response.
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

        // Build input GPU tensors only when this request is about to run.
        rt::WFMGenerationRequest request = buildWfmRequest(spec, globals, args, config, stream);

        // handleRequest() is the core handoff. Internally the runtime validates
        // inputs, prepares text, encodes, denoises, and decodes video/audio.
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

        // Record settings and success independently of whether tensor files were
        // requested. This makes the output JSON useful as a batch manifest.
        Json responseJson;
        responseJson["request_idx"] = requestIdx;
        responseJson["prompt"] = sanitizeUtf8ForJson(spec.prompt);
        responseJson["generate_sound"] = spec.generateSound;
        responseJson["num_inference_steps"] = request.numInferenceSteps;
        responseJson["seed"] = request.seed;
        responseJson["success"] = requestStatus;

        if (requestStatus)
        {
            // --dumpOutput prints tensor shapes only; it does not print tensor data.
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

            // If an output path was provided, copy and save the video tensor.
            // Otherwise, preserve only its shape in the response JSON.
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

            // Audio output follows the same policy as video output. It exists only
            // when the selected bundle and request actually run the sound pipeline.
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

    // Phase 6: Stop measurement and report aggregate request status.
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

    // Profiling JSON is separate from the normal response JSON because it contains
    // timing stages and memory measurements rather than model outputs.
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

    // Phase 7: Always attempt to write the batch response manifest, including
    // entries for requests that failed.
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

    // Release the CUDA stream after all asynchronous work and output copies finish.
    CUDA_CHECK(cudaStreamDestroy(stream));
    // A partial batch failure produces a failing process exit code even though
    // successful request entries are still present in the output JSON.
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
