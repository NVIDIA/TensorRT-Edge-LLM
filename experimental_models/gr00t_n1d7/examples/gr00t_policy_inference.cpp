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

//! GR00T N1.7 policy inference CLI. Policy mode (--inputFile, or --image with --instruction/--state)
//! runs a real observation through Gr00tRuntime; canonical mode synthesizes the VL context and drives
//! the engines alone. Accuracy is scored by experimental_models/scripts/compare_actions.py.

#include "runtime/gr00tRuntime.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::gr00t;

namespace
{

//! Above every value getopt_long returns itself.
enum OptionId : int
{
    kHELP = 951,
    kENGINE_DIR,
    kINPUT_FILE,
    kIMAGE,
    kINSTRUCTION,
    kSTATE,
    kOUTPUT,
    kNOISE,
    kSEED,
    kITERATIONS,
    kWARMUP,
    kNO_CUDA_GRAPH,
    kBATCH,
    kVL_LEN,
    kIMAGE_LEN
};

//! Text tokens that precede the image span in a synthesized context.
constexpr int32_t kImageStart = 16;
//! Standard deviation of the synthesized embeddings, the scale of real inputs_embeds.
constexpr float kSyntheticScale = 0.02F;
//! Cosmos-Reason2-2B's rotary base.
constexpr float kRopeTheta = 5000000.0F;
constexpr double kMedian = 0.5;
constexpr int32_t kNbCanonicalEvents = 3; //!< before prefill, after prefill, after generate

struct Options
{
    std::string engineDir;
    std::string output;
    std::string inputFile;
    std::vector<std::string> imagePaths;
    std::string instruction{"pick up the object and place it in the basket"};
    std::vector<float> state;
    std::string noise;
    int32_t vlLen{160};
    int32_t imageLen{64};
    int32_t batch{1};
    int32_t seed{0};
    int32_t warmup{3};
    int32_t iterations{10};
    bool cudaGraph{true};
    bool help{false};
};

void printUsage(char const* program)
{
    std::cout << "Usage: " << program << " --engineDir DIR [options]\n"
              << "Policy mode:\n"
              << "  --inputFile FILE JSON request: task, state, and cameras keyed by contract name\n"
              << "  --image FILE     camera view, repeatable and in contract order (alternative to --inputFile)\n"
              << "  --instruction S  task text (default: a LIBERO-style instruction)\n"
              << "  --state V,V,...  robot-unit state in contract order (default: drawn in range)\n"
              << "Canonical mode (no --inputFile/--image):\n"
              << "  --vlLen N        VL context length (default 160)\n"
              << "  --imageLen N     image-token count (default 64)\n"
              << "  --batch N        request batch (default 1)\n"
              << "Both:\n"
              << "  --noise FILE     initial x_0 as raw float32 [horizon, action_dim], instead of --seed\n"
              << "  --seed N         noise seed (default 0)\n"
              << "  --iterations N   timed iterations (default 10)\n"
              << "  --warmup N       untimed iterations (default 3)\n"
              << "  --noCudaGraph    disable denoise-loop graph capture\n"
              << "  --output FILE    write the action chunk as JSON\n";
}

bool parseInt(char const* text, int32_t& out)
{
    if (text == nullptr)
    {
        return false;
    }
    char* end{nullptr};
    errno = 0;
    long const parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < std::numeric_limits<int32_t>::min()
        || parsed > std::numeric_limits<int32_t>::max())
    {
        std::cerr << "Not an integer: " << text << std::endl;
        return false;
    }
    out = static_cast<int32_t>(parsed);
    return true;
}

bool parseFloat(std::string const& text, float& out)
{
    char* end{nullptr};
    errno = 0;
    float const parsed = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0')
    {
        std::cerr << "Not a number: " << text << std::endl;
        return false;
    }
    out = parsed;
    return true;
}

bool parseArgs(int argc, char** argv, Options& options)
{
    static struct option const sLongOptions[]
        = {{"help", no_argument, nullptr, kHELP}, {"engineDir", required_argument, nullptr, kENGINE_DIR},
            {"inputFile", required_argument, nullptr, kINPUT_FILE}, {"image", required_argument, nullptr, kIMAGE},
            {"instruction", required_argument, nullptr, kINSTRUCTION}, {"state", required_argument, nullptr, kSTATE},
            {"output", required_argument, nullptr, kOUTPUT}, {"noise", required_argument, nullptr, kNOISE},
            {"seed", required_argument, nullptr, kSEED}, {"iterations", required_argument, nullptr, kITERATIONS},
            {"warmup", required_argument, nullptr, kWARMUP}, {"noCudaGraph", no_argument, nullptr, kNO_CUDA_GRAPH},
            {"batch", required_argument, nullptr, kBATCH}, {"vlLen", required_argument, nullptr, kVL_LEN},
            {"imageLen", required_argument, nullptr, kIMAGE_LEN}, {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", sLongOptions, nullptr)) != -1)
    {
        std::string const value = optarg != nullptr ? optarg : "";
        bool ok = true;
        switch (opt)
        {
        case kHELP: options.help = true; break;
        case kENGINE_DIR: options.engineDir = value; break;
        case kINPUT_FILE: options.inputFile = value; break;
        case kIMAGE: options.imagePaths.push_back(value); break;
        case kINSTRUCTION: options.instruction = value; break;
        case kSTATE:
        {
            std::stringstream values(value);
            for (std::string item; ok && std::getline(values, item, ',');)
            {
                float parsed{0.0F};
                ok = parseFloat(item, parsed);
                options.state.push_back(parsed);
            }
            break;
        }
        case kOUTPUT: options.output = value; break;
        case kNOISE: options.noise = value; break;
        case kSEED: ok = parseInt(optarg, options.seed); break;
        case kITERATIONS: ok = parseInt(optarg, options.iterations); break;
        case kWARMUP: ok = parseInt(optarg, options.warmup); break;
        case kNO_CUDA_GRAPH: options.cudaGraph = false; break;
        case kBATCH: ok = parseInt(optarg, options.batch); break;
        case kVL_LEN: ok = parseInt(optarg, options.vlLen); break;
        case kIMAGE_LEN: ok = parseInt(optarg, options.imageLen); break;
        default: ok = false; break;
        }
        if (!ok)
        {
            return false;
        }
    }
    if (options.help)
    {
        return true;
    }
    if (!options.inputFile.empty() && (!options.imagePaths.empty() || !options.state.empty()))
    {
        std::cerr << "--inputFile already carries the cameras and the state; drop --image/--state\n";
        return false;
    }
    return !options.engineDir.empty();
}

//! The policy-mode request: task, state, and cameras keyed by the contract's names.
Gr00tObservation readObservation(std::string const& path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("GR00T cannot open request file " + path);
    }
    nlohmann::json request;
    file >> request;
    for (char const* key : {"task", "state", "cameras"})
    {
        if (!request.contains(key))
        {
            throw std::runtime_error("GR00T request " + path + " has no \"" + key + "\"");
        }
    }
    Gr00tObservation observation;
    observation.task = request.at("task").get<std::string>();
    observation.state = request.at("state").get<std::vector<float>>();
    std::filesystem::path const base = std::filesystem::path(path).parent_path();
    for (auto const& [name, value] : request.at("cameras").items())
    {
        Gr00tCameraView view;
        view.name = name;
        std::filesystem::path const image = value.get<std::string>();
        view.imagePath = image.is_absolute() ? image.string() : (base / image).string();
        observation.cameras.push_back(std::move(view));
    }
    return observation;
}

//! Raw float32 x_0, [horizon, action_dim] or [batch, horizon, action_dim].
std::vector<float> readNoise(std::string const& path, size_t chunkElements, int32_t batch)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("GR00T cannot open noise file " + path);
    }
    auto const bytes = static_cast<size_t>(file.tellg());
    if (bytes % sizeof(float) != 0)
    {
        throw std::runtime_error("GR00T noise file " + path + " is not a whole number of float32 values");
    }
    file.seekg(0);
    std::vector<float> noise(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(noise.data()), static_cast<std::streamsize>(bytes));
    size_t const expected = chunkElements * static_cast<size_t>(batch);
    if (noise.size() == chunkElements && batch > 1)
    {
        std::vector<float> const one = noise;
        noise.clear();
        for (int32_t row = 0; row < batch; ++row)
        {
            noise.insert(noise.end(), one.begin(), one.end());
        }
    }
    if (noise.size() != expected)
    {
        throw std::runtime_error("GR00T noise file holds " + std::to_string(noise.size()) + " floats, expected "
            + std::to_string(chunkElements) + " or " + std::to_string(expected));
    }
    return noise;
}

//! Upload a host tensor to its device twin; setup only, so a blocking copy is acceptable.
void upload(rt::Tensor& device, rt::Tensor const& host)
{
    size_t const bytes = static_cast<size_t>(host.getShape().volume()) * rt::utils::getTypeSize(host.getDataType());
    CUDA_CHECK(cudaMemcpy(device.rawPointer(), host.rawPointer(), bytes, cudaMemcpyHostToDevice));
}

//! Fill a device FP16 tensor with N(0, scale) drawn on the host.
void fillNormal(rt::Tensor& device, rt::Tensor& host, std::mt19937& generator, float scale)
{
    std::normal_distribution<float> normal(0.0F, scale);
    size_t const count = static_cast<size_t>(host.getShape().volume());
    auto* values = host.dataPointer<half>();
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = __float2half(normal(generator));
    }
    upload(device, host);
}

//! Stage boundaries for canonical mode, on the caller's stream.
struct StageEvents
{
    StageEvents()
    {
        for (auto& event : events)
        {
            CUDA_CHECK(cudaEventCreate(&event));
        }
    }
    ~StageEvents() noexcept
    {
        for (auto& event : events)
        {
            cudaEventDestroy(event);
        }
    }
    StageEvents(StageEvents const&) = delete;
    StageEvents& operator=(StageEvents const&) = delete;

    cudaEvent_t events[kNbCanonicalEvents]{};
};

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t const index
        = std::min(values.size() - 1, static_cast<size_t>(fraction * static_cast<double>(values.size() - 1) + 0.5));
    return values[index];
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseArgs(argc, argv, options) || options.help)
    {
        printUsage(argv[0]);
        return options.help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    cudaStream_t stream{nullptr};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // The visual engine carries a ViTAttentionPlugin node, so the plugin library must be resident.
    auto pluginHandles = loadEdgellmPluginLib();
    if (!pluginHandles)
    {
        LOG_ERROR("GR00T visual.engine needs libNvInfer_edgellm_plugin.so; set EDGELLM_PLUGIN_PATH");
        return EXIT_FAILURE;
    }

    try
    {
        Gr00tRuntime runtime(options.engineDir, stream);
        Gr00tPolicyRunner& runner = runtime.runner();
        Gr00tPolicy const& policy = runtime.policy();
        auto const& config = runner.getConfig();
        runner.setUseCudaGraph(options.cudaGraph);
        runner.setNoiseSeed(options.seed);
        bool const policyMode = !options.inputFile.empty() || !options.imagePaths.empty();
        int32_t const b = policyMode ? 1 : options.batch;
        if (!options.noise.empty())
        {
            runner.setInitialNoise(readNoise(
                options.noise, static_cast<size_t>(config.actionHorizon) * static_cast<size_t>(config.actionDim), b));
        }

        // The observation: from the request file, from the flags, or drawn inside
        // the embodiment's own bounds so the engines only ever see valid inputs.
        std::mt19937 generator(static_cast<uint32_t>(options.seed));
        Gr00tObservation observation;
        if (!options.inputFile.empty())
        {
            observation = readObservation(options.inputFile);
        }
        else
        {
            observation.task = options.instruction;
            observation.state = options.state;
            for (auto const& path : options.imagePaths)
            {
                Gr00tCameraView view;
                view.imagePath = path;
                observation.cameras.push_back(std::move(view));
            }
        }
        if (observation.state.empty())
        {
            auto const& bounds = policy.state();
            observation.state.resize(static_cast<size_t>(bounds.totalDim()));
            std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
            for (size_t i = 0; i < observation.state.size(); ++i)
            {
                observation.state[i] = bounds.min[i] + uniform(generator) * (bounds.max[i] - bounds.min[i]);
            }
        }

        // Canonical mode: a synthesized VL context at the requested shape.
        int32_t s = options.vlLen;
        rt::Tensor embeds;
        std::vector<rt::Tensor> deepstack;
        rt::Tensor rope;
        rt::Tensor posIds;
        if (!policyMode)
        {
            if (kImageStart + options.imageLen > s)
            {
                LOG_ERROR("image span [%d, %d) does not fit in a VL context of %d", kImageStart,
                    kImageStart + options.imageLen, s);
                cudaStreamDestroy(stream);
                return EXIT_FAILURE;
            }
            embeds
                = rt::Tensor({b, s, config.hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "embeds");
            rt::Tensor embedsHost(
                {b, s, config.hiddenSize}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF, "embedsHost");
            fillNormal(embeds, embedsHost, generator, kSyntheticScale);

            deepstack.resize(static_cast<size_t>(config.numDeepstackFeatures));
            for (int32_t i = 0; i < config.numDeepstackFeatures; ++i)
            {
                deepstack[i] = rt::Tensor(
                    {b, s, config.hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "deepstack");
                rt::Tensor host(
                    {b, s, config.hiddenSize}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF, "deepstackHost");
                // Zero outside the image span: the graph adds the whole tensor.
                auto* values = host.dataPointer<half>();
                std::fill_n(values, static_cast<size_t>(host.getShape().volume()), __float2half(0.0F));
                std::normal_distribution<float> normal(0.0F, kSyntheticScale);
                for (int32_t row = 0; row < b; ++row)
                {
                    for (int32_t token = kImageStart; token < kImageStart + options.imageLen; ++token)
                    {
                        size_t const base = (static_cast<size_t>(row) * s + token) * config.hiddenSize;
                        for (int32_t c = 0; c < config.hiddenSize; ++c)
                        {
                            values[base + c] = __float2half(normal(generator));
                        }
                    }
                }
                upload(deepstack[i], host);
            }

            // Rotary table in token order with identity positions: mRoPE is a host-side
            // property of the VL layout, so the engine only ever sees per-token cos/sin.
            rope = rt::Tensor({b, s, config.headDim}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "rope");
            rt::Tensor ropeHost(
                {b, s, config.headDim}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "ropeHost");
            int32_t const ropeHalf = config.headDim / 2;
            auto* ropeValues = ropeHost.dataPointer<float>();
            for (int32_t row = 0; row < b; ++row)
            {
                for (int32_t token = 0; token < s; ++token)
                {
                    for (int32_t c = 0; c < ropeHalf; ++c)
                    {
                        float const frequency
                            = static_cast<float>(token) / std::pow(kRopeTheta, 2.0F * c / config.headDim);
                        size_t const base = (static_cast<size_t>(row) * s + token) * config.headDim;
                        ropeValues[base + c] = std::cos(frequency);
                        ropeValues[base + ropeHalf + c] = std::sin(frequency);
                    }
                }
            }
            upload(rope, ropeHost);

            posIds = rt::Tensor({b, s}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "posIds");
            rt::Tensor posIdsHost({b, s}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "posIdsHost");
            for (int32_t row = 0; row < b; ++row)
            {
                for (int32_t token = 0; token < s; ++token)
                {
                    posIdsHost.dataPointer<int32_t>()[static_cast<size_t>(row) * s + token] = row * s + token;
                }
            }
            upload(posIds, posIdsHost);

            std::vector<uint8_t> textMask(static_cast<size_t>(b) * s, 1);
            std::vector<uint8_t> imageMask(static_cast<size_t>(b) * s, 0);
            for (int32_t row = 0; row < b; ++row)
            {
                for (int32_t token = kImageStart; token < kImageStart + options.imageLen; ++token)
                {
                    textMask[static_cast<size_t>(row) * s + token] = 0;
                    imageMask[static_cast<size_t>(row) * s + token] = 1;
                }
            }

            // prefill() sizes the masks, so bind them once the batch and length are known.
            runner.prefill(embeds, deepstack, rope, posIds, b, s, stream);
            runner.setCrossMasks(textMask, imageMask, stream);
            std::vector<float> const normalized = policy.normalizeState(observation.state, config.stateDim);
            std::vector<float> state;
            for (int32_t row = 0; row < b; ++row)
            {
                state.insert(state.end(), normalized.begin(), normalized.end());
            }
            runner.setState(state, stream);
        }

        Gr00tActionChunk chunk;
        StageEvents stages;
        auto runOnce = [&]() {
            if (policyMode)
            {
                chunk = runtime.infer(observation);
                return;
            }
            CUDA_CHECK(cudaEventRecord(stages.events[0], stream));
            runner.prefill(embeds, deepstack, rope, posIds, b, s, stream);
            CUDA_CHECK(cudaEventRecord(stages.events[1], stream));
            chunk.actions = runner.generate(stream);
            CUDA_CHECK(cudaEventRecord(stages.events[2], stream));
            // generate() already drained the stream; this only completes the timestamp.
            CUDA_CHECK(cudaEventSynchronize(stages.events[2]));
            float backbone{0.0F};
            float action{0.0F};
            CUDA_CHECK(cudaEventElapsedTime(&backbone, stages.events[0], stages.events[1]));
            CUDA_CHECK(cudaEventElapsedTime(&action, stages.events[1], stages.events[2]));
            chunk.timings.observationMs = 0.0;
            chunk.timings.backboneMs = backbone;
            chunk.timings.actionMs = action;
            chunk.timings.policyMs = static_cast<double>(backbone) + action;
        };

        for (int32_t i = 0; i < options.warmup; ++i)
        {
            runOnce();
        }
        std::vector<double> observationTimes;
        std::vector<double> backboneTimes;
        std::vector<double> actionTimes;
        std::vector<double> totalTimes;
        for (int32_t i = 0; i < options.iterations; ++i)
        {
            runOnce();
            observationTimes.push_back(chunk.timings.observationMs);
            backboneTimes.push_back(chunk.timings.backboneMs);
            actionTimes.push_back(chunk.timings.actionMs);
            totalTimes.push_back(chunk.timings.policyMs);
        }
        if (policyMode)
        {
            s = static_cast<int32_t>(chunk.tokenIds.size());
        }
        else
        {
            // Canonical tensors carry no robot state of their own, so the drawn one converts them.
            chunk.modelHorizon = config.actionHorizon;
            chunk.modelActionDim = config.actionDim;
            chunk.horizon = policy.horizon();
            chunk.robotActionDim = policy.action().totalDim();
            chunk.robotActions = policy.unnormalizeAction(chunk.actions, config.actionHorizon, config.actionDim);
        }

        auto const median = [](std::vector<double> const& values) { return percentile(values, kMedian); };
        LOG_INFO("GR00T N1.7 batch %d, VL context %d, %d denoise steps, CUDA graph %s", b, s, config.numDenoiseSteps,
            options.cudaGraph ? "on" : "off");
        LOG_INFO("observation %.2f ms | vl_backbone %.2f ms | action %.2f ms | end-to-end %.2f ms (median of %d)",
            median(observationTimes), median(backboneTimes), median(actionTimes), median(totalTimes),
            options.iterations);
        LOG_INFO("end-to-end min %.2f ms, max %.2f ms, %.1f Hz", percentile(totalTimes, 0.0),
            percentile(totalTimes, 1.0), 1000.0 / median(totalTimes));
        LOG_INFO("Policy %s: %d actions x %d dims in robot units", policy.embodimentTag().c_str(), chunk.horizon,
            chunk.robotActionDim);

        if (!options.output.empty())
        {
            // actions is the engine's normalized, zero-padded chunk; robot_actions the
            // same chunk in robot units, sliced to the embodiment. The comparator reads either.
            nlohmann::json response;
            response["embodiment_tag"] = policy.embodimentTag();
            response["vl_len"] = s;
            response["batch"] = b;
            response["model_action_horizon"] = chunk.modelHorizon;
            response["model_action_dim"] = chunk.modelActionDim;
            response["action_horizon"] = chunk.horizon;
            response["action_dim"] = chunk.robotActionDim;
            response["action_keys"] = policy.action().keys;
            response["observation_ms_median"] = median(observationTimes);
            response["backbone_ms_median"] = median(backboneTimes);
            response["action_ms_median"] = median(actionTimes);
            response["e2e_ms_median"] = median(totalTimes);
            response["actions"] = chunk.actions;
            response["robot_actions"] = chunk.robotActions;
            response["robot_state"] = observation.state;
            std::ofstream(options.output) << response.dump(2) << "\n";
            LOG_INFO("Wrote %s", options.output.c_str());
        }
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("GR00T N1.7 policy inference failed: %s", error.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
