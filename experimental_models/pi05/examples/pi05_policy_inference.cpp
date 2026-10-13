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

// One binary for every pi0.5 engine bundle; configurations differ only by which
// bundle --engineDir names, each carrying its own policy.json. Policy mode
// (--inputFile) calls Pi05Policy::infer(); canonical tensor mode
// (--pixelValues/--tokens/--noise) calls Pi05Policy::inferTensors(), which is
// configuration-independent and is what the accuracy and profiling harnesses use.
// --iters times exactly those calls, so a timed run is a deployed run.

#include "common/logger.h"
#include "common/trtUtils.h"
#include "pi05_inference_io.h"
#include "runtime/pi05Policy.h"

#include <cerrno>
#include <cstdlib>
#include <cuda_runtime.h>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{
//! Above every value getopt_long returns itself.
enum OptionId : int
{
    HELP = 951,
    ENGINE_DIR,
    NUM_VIEWS,
    TOKENS,
    SEED,
    STEPS,
    OUTPUT,
    DEBUG,
    NOISE,
    PIXEL_VALUES,
    WARMUP,
    ITERS,
    CUDA_GRAPH,
    BATCH,
    INPUT_FILE
};

struct Args
{
    std::string engineDir;
    std::string outputFile;
    std::string tokens;
    std::string noiseFile;
    std::string pixelValuesFile;
    std::string inputFile; //!< JSON request: the policy-mode input
    int32_t numViews{0};   //!< 0 = take the contract's camera count
    int32_t batch{1};
    int32_t seed{0};
    int32_t warmup{0};
    int32_t iters{0};
    int32_t steps{0};
    bool cudaGraph{false};
    bool help{false};
    bool debug{false};
};

//! \param contract The loaded contract when --engineDir named one, so the camera names
//!        a request keys by are shown rather than described.
void printUsage(char const* programName, pi05::Pi05Contract const* contract)
{
    std::cerr << "Usage: " << programName << " [--help] --engineDir <dir> [options]\n"
              << "  --engineDir  Engine bundle: visual/, prefix/, action/, cond/, policy.json,\n"
                 "               text_tokenizer/, assets/, embed_tokens.safetensors. Required.\n"
                 "\n  Raw policy mode -- a real observation in, robot actions out:\n"
                 "  --inputFile  JSON request: task, state, and cameras keyed by contract name.\n";
    std::cerr << (contract != nullptr
            ? "               Cameras: " + pi05::cameraOrderSummary(contract->cameras) + "\n"
            : std::string("               Pass --engineDir with --help to list the camera names.\n"));
    std::cerr << "\n  Canonical tensor mode -- preprocessed tensors in, normalized [B, H, 32] out:\n"
                 "  --pixelValues  fp16 [views, 3, S, S] views, as a raw dump of the tensor's bytes.\n"
                 "  --tokens       Comma-separated token ids.\n"
                 "  --numViews     Views to synthesize when no --inputFile or --pixelValues is given.\n"
                 "\n  --output     .safetensors writes the raw normalized tensor, any other suffix the\n"
                 "               JSON response (stdout when omitted).\n"
                 "  --batch      Replicate the request (engines must be built for it); needs a\n"
                 "               .safetensors --output. Default = 1\n"
                 "  --seed       Seed the x_0 draw. Default = 0\n"
                 "  --noise      fp32 [B, horizon, action_dim] x_0, a raw dump. Overrides --seed.\n"
                 "  --steps      Override the denoise step count from the contract.\n"
                 "  --cudaGraph  Capture the denoise loop into a CUDA graph.\n"
                 "  --warmup     Warmup iterations before timing. Default = 0\n"
                 "  --iters      Timed iterations; reports per-stage and end-to-end latency.\n"
                 "  --debug      Verbose logging.\n";
}

//! A numeric option, or false. Reported through the usage path rather than thrown:
//! argument parsing runs before main's try block.
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

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[]
        = {{"help", no_argument, nullptr, HELP}, {"engineDir", required_argument, nullptr, ENGINE_DIR},
            {"numViews", required_argument, nullptr, NUM_VIEWS}, {"tokens", required_argument, nullptr, TOKENS},
            {"seed", required_argument, nullptr, SEED}, {"steps", required_argument, nullptr, STEPS},
            {"output", required_argument, nullptr, OUTPUT}, {"debug", no_argument, nullptr, DEBUG},
            {"noise", required_argument, nullptr, NOISE}, {"pixelValues", required_argument, nullptr, PIXEL_VALUES},
            {"warmup", required_argument, nullptr, WARMUP}, {"iters", required_argument, nullptr, ITERS},
            {"cudaGraph", no_argument, nullptr, CUDA_GRAPH}, {"batch", required_argument, nullptr, BATCH},
            {"inputFile", required_argument, nullptr, INPUT_FILE}, {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        std::string const value = optarg != nullptr ? optarg : "";
        bool ok = true;
        switch (opt)
        {
        case HELP: args.help = true; break;
        case ENGINE_DIR: args.engineDir = value; break;
        case NUM_VIEWS: ok = parseInt(optarg, args.numViews); break;
        case TOKENS: args.tokens = value; break;
        case SEED: ok = parseInt(optarg, args.seed); break;
        case STEPS: ok = parseInt(optarg, args.steps); break;
        case OUTPUT: args.outputFile = value; break;
        case DEBUG: args.debug = true; break;
        case NOISE: args.noiseFile = value; break;
        case PIXEL_VALUES: args.pixelValuesFile = value; break;
        case WARMUP: ok = parseInt(optarg, args.warmup); break;
        case ITERS: ok = parseInt(optarg, args.iters); break;
        case CUDA_GRAPH: args.cudaGraph = true; break;
        case BATCH: ok = parseInt(optarg, args.batch); break;
        case INPUT_FILE: args.inputFile = value; break;
        default: return false;
        }
        if (!ok)
        {
            return false;
        }
    }
    if (args.help)
    {
        // Parsed to the end rather than returned at --help, so --engineDir is known and
        // the usage text can list that bundle's camera order.
        return true;
    }
    // steps 0 means "take the contract's count"; the other two count real work, and a
    // negative one silently became something else.
    if (args.engineDir.empty() || args.numViews < 0 || args.batch < 1 || args.steps < 0 || args.warmup < 0
        || args.iters < 0)
    {
        return false;
    }
    // A JSON request carries only the observation; --noise/--seed select x_0 in either mode.
    if (!args.inputFile.empty() && (!args.pixelValuesFile.empty() || !args.tokens.empty() || args.numViews > 0))
    {
        std::cerr << "The two modes are alternatives: pass --inputFile, or --pixelValues/--tokens/--numViews."
                  << std::endl;
        return false;
    }
    if (args.batch > 1 && !pi05::isSafetensorsPath(args.outputFile))
    {
        // Checked here rather than at write time, which is after the whole request has run.
        std::cerr << "--batch > 1 needs a .safetensors --output; the JSON response records one request." << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    bool const parsed = parseArgs(args, argc, argv);
    if (!parsed || args.help)
    {
        std::unique_ptr<pi05::Pi05Policy> helpPolicy;
        try
        {
            if (args.help && pi05::Pi05Policy::available(args.engineDir))
            {
                helpPolicy = std::make_unique<pi05::Pi05Policy>(args.engineDir);
            }
        }
        catch (std::exception const&)
        {
            // Help must still print for a bundle whose contract does not load.
        }
        printUsage(argv[0], helpPolicy ? &helpPolicy->contract() : nullptr);
        return parsed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kINFO);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Only the action engine carries a plugin node; a failed load is benign otherwise.
    auto pluginHandles = loadEdgellmPluginLib();

    try
    {
        pi05::Pi05Policy policy(args.engineDir, stream);
        pi05::Pi05Runtime& runtime = policy.runtime();
        pi05::Pi05PolicyConfig const& cfg = runtime.getConfig();
        runtime.setNoiseSeed(args.seed);
        runtime.setUseCudaGraph(args.cudaGraph);
        if (args.steps > 0)
        {
            runtime.setNumDenoiseSteps(args.steps);
        }
        if (!args.noiseFile.empty())
        {
            runtime.setInitialNoise(
                pi05::readNoise(args.noiseFile, static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim, args.batch));
        }

        bool const rawPolicy = !args.inputFile.empty();
        int32_t const views = !rawPolicy && args.numViews > 0 ? args.numViews : policy.numCameras();
        pi05::Pi05Observation observation;
        rt::Tensor pixelValues;
        std::vector<int32_t> tokenIds;
        if (rawPolicy)
        {
            observation = pi05::readObservation(args.inputFile);
            observation.batch = args.batch;
            LOG_INFO("pi0.5 camera order: %s", pi05::cameraOrderSummary(policy.cameras()).c_str());
        }
        else
        {
            // The prefix carries no attention mask, so a wrong view count shifts every
            // language token's position rather than being masked out.
            policy.validateViewCount(views);
            tokenIds = pi05::readTokenIds(args.tokens);
            pixelValues = pi05::readPixelValues(
                args.pixelValuesFile, {views, 3, cfg.imageSize, cfg.imageSize}, args.seed + 1, stream);
        }
        auto once = [&] {
            return rawPolicy ? policy.infer(observation) : policy.inferTensors(pixelValues, tokenIds, args.batch);
        };

        std::optional<pi05::Pi05ActionChunk> last;
        for (int32_t i = 0; i < args.warmup; ++i)
        {
            last = once();
        }
        std::vector<pi05::Pi05Timings> rounds;
        for (int32_t i = 0; i < args.iters; ++i)
        {
            last = once();
            rounds.push_back(last->timings);
        }
        if (!last)
        {
            // Only when neither loop ran; a timed call must not follow an untimed one.
            last = once();
        }
        pi05::Pi05ActionChunk const& chunk = *last;

        // Derived, not the slot count: a request that omits an optional camera runs fewer
        // views than the contract names, and the prefix is what says how many.
        int32_t const activeViews
            = (runtime.getPrefixLen() - static_cast<int32_t>(chunk.tokenIds.size())) / cfg.numImageTokens;
        LOG_INFO("pi0.5 request: %d view(s) + %zu language tokens -> prefix %d", activeViews, chunk.tokenIds.size(),
            runtime.getPrefixLen());
        if (!chunk.prompt.empty())
        {
            LOG_INFO("pi0.5 prompt: %s", chunk.prompt.c_str());
        }
        if (!rounds.empty())
        {
            pi05::logLatency(rounds, args.warmup);
        }
        pi05::writeActionChunk(args.outputFile, chunk, policy.robotActionDim(), stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("pi0.5 policy inference failed: %s", e.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
