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

// End-to-end Cosmos3 policy inference from a raw image + text instruction(s).
//
// The pipeline runs entirely in-process:
//   (a) load + preprocess the image (aspect-preserve + reflection pad, normalize to
//       [-1,1] pixel_values; only clip frame 0 is the observation, remaining frames -1),
//   (b) tokenize the prompt(s) with the co-located tokenizer (text_tokenizer/tokenizer.json),
//   (c) apply embed_tokens with the core embeddingLookup kernel -> inputs_embeds [B,S,hidden],
//   (d) VAE encode -> UND prefill -> GEN diffusion loop -> action chunk,
//   (e) write the action chunk.
//
// Batching: --prompt may be given multiple times; the request batch B is the prompt count
// (the conditioning image is shared) and requires engines built with --maxBatchSize >= B.
// Batched prompts must tokenize to the SAME length: the UND prefill graph has no attention
// mask input, so padded positions would leak into the gen cross-attention.
//
// All shape parameters (clip F/H/W, hidden size, action dims) come from the exported
// component contracts under --engineDir; nothing is passed or hardcoded here.
//
// I/O convention (project-wide):
//   * The DEFAULT --output is a JSON action file (the inference-response INTERFACE).
//     For B == 1 the "action" field is [chunk][dim] (unchanged single-request layout);
//     for B > 1 it is [B][chunk][dim]. "shape" is always [B, chunk, dim].
//   * An --output <name>.safetensors writes the RAW action tensor for bulk tensor
//     recording under key "action" (float32, [B, chunk, dim]).
//
// The tokenizer + embed_tokens artifacts are looked up under the engine dir
// (co-located by the build step, like the core engine layout).

#include "common/logger.h"
#include "runtime/cosmos3Runtime.h"

#include "chatTemplate/chatTemplate.h"
#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace
{
enum OptionId : int
{
    HELP = 901,
    ENGINE_DIR = 902,
    IMAGE = 903,
    PROMPT = 904,
    DOMAIN = 905,
    OUTPUT = 906,
    STEPS = 907,
    SEED = 908,
    VIEW_POINT = 909,
    RAW_PROMPT = 911,
    PROMPT_FILE = 912,
    ITERS = 915,
    WARMUP = 916,
    CUDAGRAPH = 917,
    GUIDANCE = 918,
    VIDEO_SUBSAMPLE = 919,
    ALTERNATE_VSF = 920,
    ACTION_CHUNK = 921,
    VIDEO = 922,
    STATE = 923,
    REQUEST_STREAM = 924
};

struct Args
{
    std::string engineDir;
    std::string imagePath;               //!< PNG/JPG conditioning observation (shared across the batch)
    std::string videoPath;               //!< comma-separated observation frames; i2v conditions on the most recent one
    std::vector<std::string> prompts;    //!< text instructions; batch = prompt count
    std::vector<float> currentState;     //!< one clean current-state row for state-conditioned policy
    std::string domain{"droid_lerobot"}; //!< action domain
    std::string viewPoint{"ego_view"};   //!< camera viewpoint for the policy prompt framing
    bool rawPrompt{false};               //!< use --prompt / --promptFile verbatim (skip JSON build)
    std::string promptFile;              //!< read the instruction/JSON prompt from a file
    std::string output;                  //!< action chunk out (.json default, or .safetensors for raw tensor)
    int32_t steps{4};                    //!< diffusion steps
    int32_t seed{0};                     //!< initial-noise seed
    // Benchmark controls.
    int32_t iters{1};
    int32_t warmup{2};
    bool cudagraph{false};
    float guidance{1.0F};            //!< CFG scale; 1.0 = off (single conditional forward). 3.0 matches the reference.
    int32_t videoSubsampleFactor{1}; //!< 1 = regular path (default); 4 = optimized (fewer GEN video frames).
    int32_t actionChunk{0};          //!< 0 = engine canonical/max chunk (default); positive = clamped request.
    bool alternateVsf{false};        //!< test/benchmark only: alternate vsf 1<->videoSubsampleFactor per round.
    bool requestStream{false};       //!< read newline-delimited requests from stdin while keeping engines resident.
    bool help{false};
};

struct PixelClipContract
{
    std::vector<int64_t> shape;
    int64_t contentHeight{0};
    int64_t contentWidth{0};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --engineDir <dir> --image <path> --prompt \"<instruction>\" [--prompt ...] "
                 "[--output <action.json|.safetensors>] [--domain <name>] [--viewPoint <name>] [--steps <int>] [--seed "
                 "<int>] "
                 "[--iters <int>] [--warmup <int>] [--cudagraph]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help       Display this help message" << std::endl;
    std::cerr << "  --engineDir  Directory with und_prefill/, vae_encoder/, gen/ engines" << std::endl;
    std::cerr << "               (+ text_tokenizer/ and embed_tokens.safetensors). Required." << std::endl;
    std::cerr << "  --image      Conditioning observation (PNG/JPG), shared across the batch. Required unless --video."
              << std::endl;
    std::cerr << "  --video      Observation frames as a comma-separated PNG/JPG list (oldest..most-recent)."
              << std::endl;
    std::cerr << "               The policy conditions image-to-video (i2v): the most-recent frame is the single"
              << std::endl;
    std::cerr << "               conditioning observation; earlier frames are ignored. Mutually exclusive with --image."
              << std::endl;
    std::cerr << "  --prompt     Text instruction. Repeat for a batched request (batch = prompt count;" << std::endl;
    std::cerr << "               requires engines built with --maxBatchSize >= count). Required." << std::endl;
    std::cerr << "  --output     Action chunk output. DEFAULT is JSON (the inference-response" << std::endl;
    std::cerr << "               interface); a .safetensors suffix writes the raw action tensor." << std::endl;
    std::cerr << "  --domain     Action domain. Default = droid_lerobot" << std::endl;
    std::cerr << "  --rawPrompt  Use the prompt / --promptFile text verbatim as the instruction" << std::endl;
    std::cerr << "               (skip structured-JSON prompt construction)." << std::endl;
    std::cerr << "  --promptFile Read the instruction (or full JSON prompt) from a file." << std::endl;
    std::cerr << "  --viewPoint  Camera viewpoint for the policy prompt: ego_view (default)," << std::endl;
    std::cerr << "               third_person_view, wrist_view, or concat_view." << std::endl;
    std::cerr << "  --steps      Diffusion denoise steps. Default = 4" << std::endl;
    std::cerr << "  --seed       Initial-noise seed. Default = 0" << std::endl;
    std::cerr << "  --state      Comma-separated current model-space state (Policy-DROID: 8 values)." << std::endl;
    std::cerr << "  --iters      Benchmark iterations. Default = 1" << std::endl;
    std::cerr << "  --warmup     Benchmark warmup rounds. Default = 2" << std::endl;
    std::cerr << "  --cudagraph  Capture/replay the per-step GEN forward as a CUDA graph." << std::endl;
    std::cerr << "  --guidance   Classifier-free guidance scale (guidance-interval CFG). Default 1.0 = off;"
              << std::endl;
    std::cerr << "               3.0 matches the reference (CFG on the first step, interval [960,1001])." << std::endl;
    std::cerr << "  --video-subsample-factor  GEN video-subsample path: ANY integer >= 1 (1 = regular, default;"
              << std::endl;
    std::cerr << "               higher = more video subsample -> fewer GEN video tokens). One dynamic engine serves"
              << std::endl;
    std::cerr << "               every factor; a factor below the engine's built profile is clamped to its minimum."
              << std::endl;
    std::cerr << "               NOTE: vsf > 1 needs a subsample-trained checkpoint to be accuracy-correct; on a"
              << std::endl;
    std::cerr << "               regular checkpoint it is speed-representative only." << std::endl;
    std::cerr << "  --alternate-vsf  Test/benchmark only: flip vsf 1<->(--video-subsample-factor) every round."
              << std::endl;
    std::cerr << "               With --cudagraph, verifies the per-extent CUDA-graph cache captures each extent"
              << std::endl;
    std::cerr << "               once then only replays. Requires --video-subsample-factor > 1. Not for deployment."
              << std::endl;
    std::cerr << "  --action-chunk-size  Action-chunk length (0 = engine canonical/max, default). A positive"
              << std::endl;
    std::cerr << "               value is clamped into the engine's built action range; the output chunk follows it."
              << std::endl;
    std::cerr << "  --request-stream  Keep the TensorRT engines resident and read one JSON request per stdin line."
              << std::endl;
    std::cerr << "               Each request supplies image, prompt, state, output, and optional seed/steps."
              << std::endl;
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, nullptr, HELP},
        {"engineDir", required_argument, nullptr, ENGINE_DIR}, {"image", required_argument, nullptr, IMAGE},
        {"video", required_argument, nullptr, VIDEO}, {"prompt", required_argument, nullptr, PROMPT},
        {"domain", required_argument, nullptr, DOMAIN}, {"viewPoint", required_argument, nullptr, VIEW_POINT},
        {"rawPrompt", no_argument, nullptr, RAW_PROMPT}, {"promptFile", required_argument, nullptr, PROMPT_FILE},
        {"output", required_argument, nullptr, OUTPUT}, {"steps", required_argument, nullptr, STEPS},
        {"seed", required_argument, nullptr, SEED}, {"iters", required_argument, nullptr, ITERS},
        {"warmup", required_argument, nullptr, WARMUP}, {"cudagraph", no_argument, nullptr, CUDAGRAPH},
        {"guidance", required_argument, nullptr, GUIDANCE},
        {"video-subsample-factor", required_argument, nullptr, VIDEO_SUBSAMPLE},
        {"alternate-vsf", no_argument, nullptr, ALTERNATE_VSF},
        {"action-chunk-size", required_argument, nullptr, ACTION_CHUNK}, {"state", required_argument, nullptr, STATE},
        {"request-stream", no_argument, nullptr, REQUEST_STREAM}, {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: args.help = true; return true;
        case ENGINE_DIR: args.engineDir = optarg ? optarg : ""; break;
        case IMAGE: args.imagePath = optarg ? optarg : ""; break;
        case VIDEO: args.videoPath = optarg ? optarg : ""; break;
        case PROMPT:
            if (optarg != nullptr)
            {
                args.prompts.emplace_back(optarg);
            }
            break;
        case DOMAIN: args.domain = optarg ? optarg : args.domain; break;
        case VIEW_POINT: args.viewPoint = optarg ? optarg : args.viewPoint; break;
        case RAW_PROMPT: args.rawPrompt = true; break;
        case PROMPT_FILE: args.promptFile = optarg ? optarg : ""; break;
        case OUTPUT: args.output = optarg ? optarg : ""; break;
        case STEPS: args.steps = optarg ? std::stoi(optarg) : args.steps; break;
        case SEED: args.seed = optarg ? std::stoi(optarg) : args.seed; break;
        case ITERS: args.iters = optarg ? std::stoi(optarg) : args.iters; break;
        case WARMUP: args.warmup = optarg ? std::stoi(optarg) : args.warmup; break;
        case CUDAGRAPH: args.cudagraph = true; break;
        case GUIDANCE: args.guidance = optarg ? std::stof(optarg) : args.guidance; break;
        case VIDEO_SUBSAMPLE: args.videoSubsampleFactor = optarg ? std::stoi(optarg) : args.videoSubsampleFactor; break;
        case ALTERNATE_VSF: args.alternateVsf = true; break;
        case ACTION_CHUNK: args.actionChunk = optarg ? std::stoi(optarg) : args.actionChunk; break;
        case REQUEST_STREAM: args.requestStream = true; break;
        case STATE:
            if (optarg != nullptr)
            {
                std::string values(optarg);
                for (size_t start = 0; start <= values.size();)
                {
                    size_t const comma = values.find(',', start);
                    size_t const end = comma == std::string::npos ? values.size() : comma;
                    ELLM_CHECK(end > start, "--state contains an empty value");
                    args.currentState.push_back(std::stof(values.substr(start, end - start)));
                    if (comma == std::string::npos)
                    {
                        break;
                    }
                    start = comma + 1;
                }
            }
            break;
        default: return false;
        }
    }
    if (args.requestStream)
    {
        return !args.engineDir.empty();
    }
    // Exactly one conditioning source: --image (single frame) xor --video (frame list).
    bool const oneSource = args.imagePath.empty() != args.videoPath.empty();
    return !args.engineDir.empty() && oneSource && (!args.prompts.empty() || !args.promptFile.empty());
}

//! Read the [B,3,F,H,W] pixel-clip shape from the vae_encoder component contract.
PixelClipContract readPixelClipContract(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "vae_encoder" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open vae_encoder config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("optimization_profile") && j.at("optimization_profile").contains("pixel_values"),
        "vae_encoder config missing optimization_profile.pixel_values");
    auto shape = j.at("optimization_profile").at("pixel_values").at("opt").get<std::vector<int64_t>>();
    ELLM_CHECK(shape.size() == 5, "vae_encoder pixel_values profile must be [B,3,F,H,W]");
    ELLM_CHECK(j.contains("builder_config"), "vae_encoder config missing builder_config");
    auto const& builderConfig = j.at("builder_config");
    ELLM_CHECK(builderConfig.contains("content_height") && builderConfig.contains("content_width"),
        "vae_encoder builder_config missing baked content dimensions");
    PixelClipContract contract{std::move(shape), builderConfig.at("content_height").get<int64_t>(),
        builderConfig.at("content_width").get<int64_t>()};
    ELLM_CHECK(contract.contentHeight > 0 && contract.contentWidth > 0 && contract.contentHeight <= contract.shape[3]
            && contract.contentWidth <= contract.shape[4],
        "vae_encoder baked content dimensions must fit its pixel canvas");
    return contract;
}

//! Read the text hidden size from the und_prefill component contract.
int32_t readHiddenSize(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "und_prefill" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open und_prefill config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("hidden_size"), "und_prefill config missing hidden_size");
    return j.at("hidden_size").get<int32_t>();
}

//! Viewpoint framing sentences used in the structured policy prompt (mirrors the
//! reference action-dataset viewpoint templates).
std::string viewpointFraming(std::string const& viewPoint)
{
    if (viewPoint == "ego_view")
    {
        return "This video is captured from a first-person perspective looking at the scene.";
    }
    if (viewPoint == "third_person_view")
    {
        return "This video is captured from a third-person perspective looking towards the agent from the front.";
    }
    if (viewPoint == "wrist_view")
    {
        return "This video is captured from a wrist-mounted camera.";
    }
    if (viewPoint == "concat_view")
    {
        // The Policy-DROID RoboLab reference appends its _CONCAT_VIEW_DESCRIPTION to the canonical concat_view
        // template in cosmos_framework/scripts/action_policy_server_robolab.py. Keep that combined wording verbatim.
        return "This video contains concatenated views from multiple camera perspectives. The top row is from the "
               "wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective "
               "views of the scene from opposite sides, with the robot visible.";
    }
    ELLM_CHECK(false,
        "Unsupported --viewPoint '" + viewPoint
            + "'; expected ego_view, third_person_view, wrist_view, or concat_view.");
    return {};
}

//! Format integer seconds as M:SS for the policy prompt time range.
std::string formatTimeMSS(int32_t seconds)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d:%02d", seconds / 60, seconds % 60);
    return buf;
}

//! Canonical width,height aspect-ratio string for the known 480-class buckets;
//! reduced-fraction fallback otherwise.
std::string aspectRatioString(int32_t width, int32_t height)
{
    if (width == 736 && height == 544)
    {
        return "4,3";
    }
    if (width == 832 && height == 480)
    {
        return "16,9";
    }
    if (width == 480 && height == 832)
    {
        return "9,16";
    }
    if (width == 544 && height == 736)
    {
        return "3,4";
    }
    if (width == 640 && height == 640)
    {
        return "1,1";
    }
    int32_t const d = std::gcd(width, height);
    return std::to_string(width / d) + "," + std::to_string(height / d);
}

//! Build the structured JSON policy prompt the reference pipeline conditions the
//! UND tower on. Field order and ": " / ", " spacing must match the reference
//! Python json serialization exactly so the token stream is identical.
std::string buildPolicyPrompt(std::string const& instruction, std::string const& viewPoint, int32_t actionChunkSize,
    float fps, int32_t height, int32_t width)
{
    std::string sentence = instruction;
    while (!sentence.empty() && std::isspace(static_cast<unsigned char>(sentence.back())))
    {
        sentence.pop_back();
    }
    if (!sentence.empty() && sentence.back() != '.' && sentence.back() != '!' && sentence.back() != '?')
    {
        sentence += '.';
    }
    double const endSeconds = static_cast<double>(actionChunkSize) / fps;
    int32_t const timeEnd = static_cast<int32_t>(std::lround(endSeconds));
    int32_t const duration = static_cast<int32_t>(endSeconds); // truncated
    char fpsText[32];
    std::snprintf(fpsText, sizeof(fpsText), "%.1f", static_cast<double>(fps));

    std::string prompt = "{\"cinematography\": {\"framing\": ";
    prompt += Json(viewpointFraming(viewPoint)).dump();
    prompt += "}, \"actions\": [{\"time\": ";
    prompt += Json("0:00-" + formatTimeMSS(timeEnd)).dump();
    prompt += ", \"description\": ";
    prompt += Json(sentence).dump();
    prompt += "}], \"duration\": \"" + std::to_string(duration) + "s\", \"fps\": ";
    prompt += fpsText;
    prompt += ", \"resolution\": {\"H\": " + std::to_string(height) + ", \"W\": " + std::to_string(width) + "}";
    prompt += ", \"aspect_ratio\": " + Json(aspectRatioString(width, height)).dump() + "}";
    return prompt;
}

int64_t padCoordinate(int64_t coordinate, int64_t contentSize, int64_t targetSize)
{
    int64_t const padding = targetSize - contentSize;
    if (coordinate < contentSize)
    {
        return coordinate;
    }
    if (padding >= contentSize)
    {
        return contentSize - 1;
    }
    return 2 * contentSize - coordinate - 2;
}

//! Read action_chunk_size and fps from the GEN component contract.
std::pair<int32_t, float> readActionChunkAndFps(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "gen" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open gen config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("action_chunk_size") && j.contains("fps"), "gen config missing action_chunk_size / fps");
    return {j.at("action_chunk_size").get<int32_t>(), j.at("fps").get<float>()};
}

//! Write the RAW action tensor as a safetensors file (bulk tensor recording).
//! Single tensor "action" (float32, shape [B, chunk, dim]).
void writeActionSafetensors(std::string const& path, std::vector<float> const& action, int32_t batch, int32_t chunk,
    int32_t dim, cudaStream_t stream)
{
    std::vector<int64_t> const shape{batch, chunk, dim};
    rt::Tensor t(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "action");
    std::memcpy(t.rawPointer(), action.data(), action.size() * sizeof(float));
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(t));
    ELLM_CHECK(rt::safetensors::saveSafetensors(path, tensors, stream), "Failed to write safetensors: " + path);
    LOG_INFO("Wrote raw action tensor (%d,%d,%d) to %s (safetensors)", batch, chunk, dim, path.c_str());
}

//! Write the action chunk as the JSON inference-response interface (the DEFAULT path).
//! For batch == 1 "action" is [chunk][dim] (the single-request layout consumed by the
//! RoboLab wrapper); for batch > 1 it is [B][chunk][dim]. "shape" is always [B, chunk, dim].
void writeActionJson(std::string const& path, std::vector<float> const& action, int32_t batch, int32_t chunk,
    int32_t dim, std::string const& domain, int32_t steps, std::vector<std::string> const& prompts, bool finite,
    int32_t seqLen, int32_t videoSubsampleFactor)
{
    // droid_lerobot delivers the gripper (final action dim) in the dataset's
    // open/close convention; the raw model channel is inverted, so flip it back
    // at the JSON boundary. Mirrors the reference DROID action interface
    // (action_np[:, -1] = 1 - action_np[:, -1]); the raw .safetensors path stays raw.
    bool const invertGripper = (domain == "droid_lerobot") && dim > 0;
    auto chunkRows = [&](int32_t b) {
        Json rows = Json::array();
        for (int32_t c = 0; c < chunk; ++c)
        {
            Json row = Json::array();
            for (int32_t d = 0; d < dim; ++d)
            {
                float value = action[(static_cast<size_t>(b) * chunk + c) * dim + d];
                if (invertGripper && d == dim - 1)
                {
                    value = 1.0F - value;
                }
                row.push_back(value);
            }
            rows.push_back(std::move(row));
        }
        return rows;
    };

    Json j;
    if (batch == 1)
    {
        j["action"] = chunkRows(0);
        j["prompt"] = prompts.front();
    }
    else
    {
        Json batched = Json::array();
        for (int32_t b = 0; b < batch; ++b)
        {
            batched.push_back(chunkRows(b));
        }
        j["action"] = std::move(batched);
        j["prompt"] = prompts;
    }
    j["shape"] = {batch, chunk, dim};
    j["dtype"] = "float32";
    j["domain"] = domain;
    j["num_inference_steps"] = steps;
    j["finite"] = finite;
    j["meta"] = {{"seq_len", seqLen}, {"video_subsample_factor", videoSubsampleFactor}};

    std::ofstream of(path);
    of << j.dump(2) << "\n";
    LOG_INFO("Wrote action JSON (%d,%d,%d) to %s", batch, chunk, dim, path.c_str());
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
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
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    if (args.domain != "droid_lerobot")
    {
        LOG_WARNING("Domain '%s' is not the validated 'droid_lerobot' domain; running anyway.", args.domain.c_str());
    }

    cudaStream_t stream{nullptr};
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        LOG_ERROR("Failed to create CUDA stream");
        return EXIT_FAILURE;
    }

    // All shape parameters come from the exported component contracts.
    if (!args.promptFile.empty())
    {
        std::ifstream pf(args.promptFile);
        ELLM_CHECK(pf.is_open(), "Failed to open --promptFile: " + args.promptFile);
        std::stringstream ss;
        ss << pf.rdbuf();
        args.prompts.assign(1, ss.str());
    }
    PixelClipContract const clipContract = readPixelClipContract(args.engineDir);
    std::vector<int64_t> const& clipShape = clipContract.shape; // [B,3,F,H,W]
    int32_t const pixelFrames = static_cast<int32_t>(clipShape[2]);
    int32_t const clipH = static_cast<int32_t>(clipShape[3]);
    int32_t const clipW = static_cast<int32_t>(clipShape[4]);
    int32_t const hiddenSize = readHiddenSize(args.engineDir);
    auto const [actionChunkSize, fps] = readActionChunkAndFps(args.engineDir);
    size_t const hw = static_cast<size_t>(clipH) * clipW;

    // Engine deserialization dominates one-shot startup (tens of seconds on GB300).
    // Keep one runtime alive in --request-stream mode so closed-loop policy serving
    // pays this cost once instead of once per 32-action chunk.
    std::unique_ptr<cosmos3::Cosmos3Runtime> runtime;
    try
    {
        runtime = std::make_unique<cosmos3::Cosmos3Runtime>(args.engineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Cosmos3 policy initialization failed: %s", e.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // These artifacts are immutable across closed-loop requests. Loading the
    // tokenizer and embedding table inside runOnce makes every action chunk
    // dependent on filesystem latency and can turn an otherwise sub-second
    // request into a multi-second outlier under unrelated I/O pressure.
    fs::path tokDir = fs::path(args.engineDir) / "text_tokenizer";
    if (!fs::exists(tokDir / "tokenizer.json"))
    {
        tokDir = args.engineDir; // fall back to engine dir root
    }
    tokenizer::Tokenizer tok;
    chat_template::ChatTemplate chatTemplate;
    std::vector<rt::Tensor> embedTensors;
    fs::path const embedPath = fs::path(args.engineDir) / "embed_tokens.safetensors";
    try
    {
        ELLM_CHECK(tok.loadFromHF(tokDir), "Failed to load tokenizer from " + tokDir.string());
        auto const specialToken = [&tok](tokenizer::Rank id) {
            return id >= 0 ? tok.idToPiece(id, /*skipSpecialTokens=*/false) : std::string{};
        };
        ELLM_CHECK(chatTemplate.load(tokDir, specialToken(tok.getBosId()), specialToken(tok.getEosId())),
            "Failed to load chat template from " + tokDir.string());
        ELLM_CHECK(rt::safetensors::loadSafetensors(embedPath, embedTensors, stream),
            "Failed to load embed_tokens: " + embedPath.string());
        ELLM_CHECK(!embedTensors.empty(), "embed_tokens artifact has no tensors: " + embedPath.string());
        ELLM_CHECK(
            embedTensors.front().getShape().getNumDims() == 2 && embedTensors.front().getShape()[1] == hiddenSize,
            "embed_tokens must be [vocab, hidden] matching the und_prefill contract");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Cosmos3 policy static-artifact initialization failed: %s", e.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }
    rt::Tensor const& table = embedTensors.front();

    auto runOnce = [&](Args const& requestArgs) {
        Args const& args = requestArgs;
        int32_t const batch = static_cast<int32_t>(args.prompts.size());
        ELLM_CHECK(batch > 0, "Policy request must contain at least one prompt");
        // ------------------------------------------------------------------ //
        // (a) Conditioning frame -> pixel clip [B,3,F,H,W] in [-1,1].         //
        // ------------------------------------------------------------------ //
        // The policy conditions image-to-video (i2v): a single observation frame is the
        // conditioning input (only latent frame 0 is kept clean; the rest of the clip is
        // denoised). --video supplies an observation clip (oldest..most-recent); the policy
        // conditions on the most-recent frame, matching the reference i2v regime.
        std::string condFramePath = args.imagePath;
        if (condFramePath.empty())
        {
            std::vector<std::string> frames;
            for (size_t start = 0; start <= args.videoPath.size();)
            {
                size_t const comma = args.videoPath.find(',', start);
                size_t const end = (comma == std::string::npos) ? args.videoPath.size() : comma;
                if (end > start)
                {
                    frames.push_back(args.videoPath.substr(start, end - start));
                }
                if (comma == std::string::npos)
                {
                    break;
                }
                start = comma + 1;
            }
            ELLM_CHECK(!frames.empty(), "--video requires at least one frame path");
            condFramePath = frames.back();
            LOG_INFO("Video input: %zu observation frame(s); conditioning i2v on the most-recent frame %s.",
                frames.size(), condFramePath.c_str());
        }
        rt::imageUtils::ImageData const img = rt::imageUtils::loadRgbImageFromFile(condFramePath);
        // GEN consumes the VAE latent after a build-time crop to content_height/content_width. Fit the
        // input to those baked dimensions, and reject another aspect/size instead of silently pairing a
        // runtime image crop with a differently shaped latent graph.
        double const scale = std::min({static_cast<double>(clipContract.contentWidth) / static_cast<double>(img.width),
            static_cast<double>(clipContract.contentHeight) / static_cast<double>(img.height), 1.0});
        int64_t const contentW = static_cast<int64_t>(scale * static_cast<double>(img.width) + 0.5);
        int64_t const contentH = static_cast<int64_t>(scale * static_cast<double>(img.height) + 0.5);
        ELLM_CHECK(contentW == clipContract.contentWidth && contentH == clipContract.contentHeight,
            "Conditioning image geometry maps to " + std::to_string(contentW) + "x" + std::to_string(contentH)
                + ", but the engine was built for content " + std::to_string(clipContract.contentWidth) + "x"
                + std::to_string(clipContract.contentHeight)
                + "; use a matching input aspect/size or rebuild the engines");
        LOG_INFO(
            "Loaded conditioning frame %s (%ldx%ld), resizing aspect-preservingly to %ldx%ld, "
            "reflection-padding to %dx%d, and normalizing to pixel_values.",
            condFramePath.c_str(), img.width, img.height, contentW, contentH, clipW, clipH);
        rt::imageUtils::ImageData resized(rt::Tensor(
            {1, contentH, contentW, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "cosmos3::resized"));
        rt::imageUtils::ImageData const& content = rt::imageUtils::resizeImage(
            img, resized, contentW, contentH, rt::imageUtils::InterpolationMode::kBICUBIC);

        // HWC uint8 -> planar CHW float in [-1,1]. Only frame 0 is the observation; the remaining
        // frames stay at -1.0, matching the reference ActionTransformPipeline. Spatial padding is
        // bottom/right-only reflection padding.
        size_t const clipElems = static_cast<size_t>(3) * pixelFrames * hw;
        std::vector<float> clip(static_cast<size_t>(batch) * clipElems, -1.0F);
        unsigned char const* srcPixels = content.data();
        for (int32_t c = 0; c < 3; ++c)
        {
            float* frame0 = clip.data() + static_cast<size_t>(c) * pixelFrames * hw;
            for (int32_t y = 0; y < clipH; ++y)
            {
                int64_t const srcY = padCoordinate(y, contentH, clipH);
                for (int32_t x = 0; x < clipW; ++x)
                {
                    int64_t const srcX = padCoordinate(x, contentW, clipW);
                    size_t const dstIdx = static_cast<size_t>(y) * clipW + x;
                    size_t const srcIdx = (static_cast<size_t>(srcY) * contentW + srcX) * 3 + c;
                    frame0[dstIdx] = static_cast<float>(srcPixels[srcIdx]) / 127.5F - 1.0F;
                }
            }
        }
        for (int32_t b = 1; b < batch; ++b)
        {
            std::copy_n(clip.data(), clipElems, clip.data() + static_cast<size_t>(b) * clipElems);
        }
        rt::Tensor pixelTensor({batch, 3, pixelFrames, clipH, clipW}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "cosmos3::pixelValues");
        // One-time host->device setup copies below source from pageable std::vectors, so they use the
        // synchronous cudaMemcpy: cudaMemcpyAsync on pageable memory silently falls back to a sync copy.
        CUDA_CHECK(
            cudaMemcpy(pixelTensor.rawPointer(), clip.data(), clip.size() * sizeof(float), cudaMemcpyHostToDevice));

        // ------------------------------------------------------------------ //
        // (b) Prompt(s) -> token ids (co-located tokenizer).                  //
        // ------------------------------------------------------------------ //
        std::vector<std::vector<tokenizer::Rank>> ids;
        ids.reserve(args.prompts.size());
        for (std::string const& prompt : args.prompts)
        {
            // The reference policy pipeline wraps the instruction in the VLM chat template with a
            // generation prompt (tokenize_caption: user message + add_generation_prompt); the UND
            // tower must see the same token stream for the exported K/V to match.
            rt::LLMGenerationRequest::Request request;
            rt::Message message;
            message.role = "user";
            std::string const instruction = args.rawPrompt
                ? prompt
                : buildPolicyPrompt(prompt, args.viewPoint, actionChunkSize, fps, clipH, clipW);
            message.contents.push_back({"text", instruction});
            request.messages.push_back(std::move(message));
            rt::LLMGenerationRequest::FormattedRequest formatted;
            chat_template::ChatTemplate::Options options;
            options.enableThinking = true;
            ELLM_CHECK(chatTemplate.apply(request, formatted, options),
                "Failed to apply the chat template to prompt: " + prompt);
            ids.push_back(tok.encode(formatted.formattedCompleteRequest, /*addBos=*/false, /*addEos=*/false));
            ELLM_CHECK(!ids.back().empty(), "Prompt tokenized to zero tokens: " + prompt);
            // The reference terminates the UND span with EOS + the start-of-generation token
            // (<|vision_start|>); the GEN cross-attention attends to these positions.
            ids.back().push_back(tok.getEosId());
            std::vector<tokenizer::Rank> const sog = tok.encode("<|vision_start|>", /*addBos=*/false, /*addEos=*/false);
            ELLM_CHECK(sog.size() == 1, "Tokenizer must map <|vision_start|> to a single id");
            ids.back().push_back(sog.front());
            // The UND prefill graph has no attention mask input, so a batched request requires
            // equal-length prompts (padding would leak into the gen cross-attention).
            ELLM_CHECK(ids.back().size() == ids.front().size(),
                "Batched prompts must tokenize to the same length (got " + std::to_string(ids.front().size()) + " vs "
                    + std::to_string(ids.back().size()) + ")");
        }
        int32_t const seqLen = static_cast<int32_t>(ids.front().size());
        LOG_INFO("Tokenized %d prompt(s) to %d tokens each.", batch, seqLen);

        // ------------------------------------------------------------------ //
        // (c) embed_tokens via the core embeddingLookup kernel (all Tensors). //
        // ------------------------------------------------------------------ //
        std::vector<int32_t> idsHost;
        idsHost.reserve(static_cast<size_t>(batch) * seqLen);
        for (auto const& promptIds : ids)
        {
            idsHost.insert(idsHost.end(), promptIds.begin(), promptIds.end());
        }
        rt::Tensor idsTensor({batch, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "cosmos3::inputIds");
        CUDA_CHECK(cudaMemcpy(
            idsTensor.rawPointer(), idsHost.data(), idsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        rt::Tensor inputsEmbeds(
            {batch, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "cosmos3::inputsEmbeds");
        kernel::embeddingLookup(idsTensor, table, std::nullopt, inputsEmbeds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ------------------------------------------------------------------ //
        // (c') Unconditional (empty-prompt) embeddings for guidance-interval  //
        //      CFG. The reference drops the text caption for the uncond pass; //
        //      we tokenize the same chat-template scaffold with an empty      //
        //      instruction (broadcast across the batch).                      //
        // ------------------------------------------------------------------ //
        bool const useCfg = args.guidance != 1.0F;
        rt::Tensor uncondEmbeds;
        if (useCfg)
        {
            rt::LLMGenerationRequest::Request ureq;
            rt::Message umsg;
            umsg.role = "user";
            umsg.contents.push_back({"text", std::string{}});
            ureq.messages.push_back(std::move(umsg));
            rt::LLMGenerationRequest::FormattedRequest uformatted;
            chat_template::ChatTemplate::Options options;
            options.enableThinking = true;
            ELLM_CHECK(chatTemplate.apply(ureq, uformatted, options),
                "Failed to apply the chat template to the unconditional (empty) prompt");
            std::vector<tokenizer::Rank> uids
                = tok.encode(uformatted.formattedCompleteRequest, /*addBos=*/false, /*addEos=*/false);
            uids.push_back(tok.getEosId());
            std::vector<tokenizer::Rank> const usog
                = tok.encode("<|vision_start|>", /*addBos=*/false, /*addEos=*/false);
            ELLM_CHECK(usog.size() == 1, "Tokenizer must map <|vision_start|> to a single id");
            uids.push_back(usog.front());
            int32_t const uSeqLen = static_cast<int32_t>(uids.size());
            std::vector<int32_t> uIdsHost;
            uIdsHost.reserve(static_cast<size_t>(batch) * uSeqLen);
            for (int32_t b = 0; b < batch; ++b)
            {
                uIdsHost.insert(uIdsHost.end(), uids.begin(), uids.end());
            }
            rt::Tensor uIdsTensor(
                {batch, uSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "cosmos3::uncondIds");
            CUDA_CHECK(cudaMemcpy(
                uIdsTensor.rawPointer(), uIdsHost.data(), uIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            uncondEmbeds = rt::Tensor(
                {batch, uSeqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "cosmos3::uncondEmbeds");
            kernel::embeddingLookup(uIdsTensor, table, std::nullopt, uncondEmbeds, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            LOG_INFO(
                "Guidance-interval CFG on (guidance=%.2f): unconditional prompt = %d tokens.", args.guidance, uSeqLen);
        }

        // ------------------------------------------------------------------ //
        // (d) VAE encode -> UND prefill -> GEN diffusion loop -> action chunk //
        // ------------------------------------------------------------------ //
        auto const& policyConfig = runtime->policyConfig();
        if (policyConfig.stateRows > 0)
        {
            ELLM_CHECK(batch == 1, "State-conditioned CLI currently supports batch size 1");
            ELLM_CHECK(args.currentState.size() == static_cast<size_t>(policyConfig.rawActionDim),
                "--state must contain exactly raw_action_dim values for this engine");
        }
        runtime->setNoiseSeed(args.seed);
        runtime->setNumInferenceSteps(args.steps);
        runtime->setUseCudaGraph(args.cudagraph);
        runtime->setGuidance(args.guidance, /*intervalLo=*/960.0F, /*intervalHi=*/1001.0F);
        ELLM_CHECK(args.videoSubsampleFactor >= 1,
            "--video-subsample-factor must be >= 1 (1 = regular; higher = more subsample, clamped to the engine "
            "range)");
        runtime->setVideoSubsampleFactor(args.videoSubsampleFactor);
        ELLM_CHECK(args.actionChunk >= 0, "--action-chunk-size must be >= 0 (0 = engine canonical/max chunk)");
        runtime->setActionChunkSize(args.actionChunk);
        if (args.videoSubsampleFactor != 1)
        {
            LOG_WARNING(
                "Optimized video-subsample path (factor %d): speed-representative only; correct actions "
                "require a subsample-trained checkpoint.",
                args.videoSubsampleFactor);
        }
        rt::Tensor const* const uncondPtr = useCfg ? &uncondEmbeds : nullptr;

        // --alternate-vsf (test/benchmark only) flips the video-subsample factor 1<->videoSubsampleFactor
        // every round. With --cudagraph this exercises the per-extent graph cache: each extent captures once
        // (logged by the runner) and every later round of that extent replays without re-capturing, so
        // switching must not spike the latency.
        ELLM_CHECK(!args.alternateVsf || args.videoSubsampleFactor > 1,
            "--alternate-vsf requires --video-subsample-factor > 1 to alternate against");
        int32_t round = 0;
        auto vsfForRound = [&](int32_t k) -> int32_t { return (k % 2 == 0) ? 1 : args.videoSubsampleFactor; };

        std::vector<float> action;
        for (int32_t w = 0; w < args.warmup; ++w)
        {
            if (args.alternateVsf)
            {
                runtime->setVideoSubsampleFactor(vsfForRound(round++));
            }
            action = runtime->generatePolicy(pixelTensor, inputsEmbeds, uncondPtr, args.currentState, stream);
        }
        std::vector<double> walls;
        std::vector<double> wallsVsfLow;  //!< --alternate-vsf rounds at vsf == 1.
        std::vector<double> wallsVsfHigh; //!< --alternate-vsf rounds at vsf == videoSubsampleFactor.
        walls.reserve(args.iters);
        for (int32_t it = 0; it < args.iters; ++it)
        {
            int32_t const vsf = args.alternateVsf ? vsfForRound(round++) : args.videoSubsampleFactor;
            if (args.alternateVsf)
            {
                runtime->setVideoSubsampleFactor(vsf);
            }
            auto const t0 = std::chrono::high_resolution_clock::now();
            action = runtime->generatePolicy(pixelTensor, inputsEmbeds, uncondPtr, args.currentState, stream);
            cudaStreamSynchronize(stream);
            auto const t1 = std::chrono::high_resolution_clock::now();
            double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            walls.push_back(ms);
            if (args.alternateVsf)
            {
                (vsf == 1 ? wallsVsfLow : wallsVsfHigh).push_back(ms);
                LOG_INFO("alternate round %d: vsf %d -> %.2f ms", it, vsf, ms);
            }
        }
        auto median = [](std::vector<double> v) -> double {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : v[v.size() / 2];
        };
        if (!walls.empty())
        {
            std::vector<double> sorted = walls;
            std::sort(sorted.begin(), sorted.end());
            LOG_INFO(
                "Steady-state full policy round (batch %d): median %.2f ms over %d iters (min %.2f, max %.2f), "
                "warmup %d",
                batch, sorted[sorted.size() / 2], args.iters, sorted.front(), sorted.back(), args.warmup);
        }
        if (args.alternateVsf)
        {
            LOG_INFO("Alternating vsf medians: vsf=1 %.2f ms (%zu rounds), vsf=%d %.2f ms (%zu rounds)",
                median(wallsVsfLow), wallsVsfLow.size(), args.videoSubsampleFactor, median(wallsVsfHigh),
                wallsVsfHigh.size());
        }

        ELLM_CHECK(!action.empty(), "Policy generation returned no action.");

        // ------------------------------------------------------------------ //
        // (e) Emit the action chunk (B, chunk, dim).                          //
        // ------------------------------------------------------------------ //
        int32_t const rawActionDim = runtime->policyConfig().rawActionDim;
        // The returned action chunk follows the (possibly clamped) per-request action length, so derive
        // it from the flattened action size rather than the engine's canonical chunk.
        int32_t const chunk = static_cast<int32_t>(action.size() / (static_cast<size_t>(batch) * rawActionDim));
        bool finite = true;
        float amin = 0.0f, amax = 0.0f;
        for (size_t i = 0; i < action.size(); ++i)
        {
            float const v = action[i];
            if (!std::isfinite(v))
            {
                finite = false;
            }
            else
            {
                if (i == 0 || v < amin)
                {
                    amin = v;
                }
                if (i == 0 || v > amax)
                {
                    amax = v;
                }
            }
        }
        LOG_INFO("Action chunk (%zu values, %dx%dx%d): finite=%s min=%.5g max=%.5g", action.size(), batch, chunk,
            rawActionDim, finite ? "true" : "false", amin, amax);

        if (!args.output.empty())
        {
            bool const asSafetensors
                = args.output.size() >= 12 && args.output.substr(args.output.size() - 12) == ".safetensors";
            if (asSafetensors)
            {
                writeActionSafetensors(args.output, action, batch, chunk, rawActionDim, stream);
            }
            else
            {
                writeActionJson(args.output, action, batch, chunk, rawActionDim, args.domain, args.steps, args.prompts,
                    finite, seqLen, args.videoSubsampleFactor);
            }
        }
    };

    if (args.requestStream)
    {
        std::cout << "@@EDGELLM_READY {}" << std::endl;
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }
            Json response;
            try
            {
                Json const request = Json::parse(line);
                Args requestArgs = args;
                requestArgs.requestStream = false;
                requestArgs.imagePath = request.at("image").get<std::string>();
                requestArgs.videoPath.clear();
                requestArgs.prompts = {request.at("prompt").get<std::string>()};
                requestArgs.promptFile.clear();
                requestArgs.currentState = request.at("state").get<std::vector<float>>();
                requestArgs.output = request.at("output").get<std::string>();
                requestArgs.seed = request.value("seed", args.seed);
                requestArgs.steps = request.value("steps", args.steps);
                requestArgs.domain = request.value("domain", args.domain);
                requestArgs.viewPoint = request.value("viewpoint", args.viewPoint);
                requestArgs.guidance = request.value("guidance", args.guidance);
                requestArgs.warmup = 0;
                requestArgs.iters = 1;
                runOnce(requestArgs);
                response = {{"ok", true}, {"output", requestArgs.output}};
            }
            catch (std::exception const& e)
            {
                response = {{"ok", false}, {"error", e.what()}};
            }
            std::cout << "@@EDGELLM_RESPONSE " << response.dump() << std::endl;
        }
    }
    else
    {
        try
        {
            runOnce(args);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Cosmos3 policy inference failed: %s", e.what());
            cudaStreamDestroy(stream);
            return EXIT_FAILURE;
        }
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
