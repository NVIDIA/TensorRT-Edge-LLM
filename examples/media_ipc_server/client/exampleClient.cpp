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

#include "ipcClient.h"
#include "mediaSource.h"

#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace
{

using namespace edgellm::ipc;

constexpr int64_t kSamplingMaxGenerateLength = 256;
constexpr int32_t kSamplingNumLogprobs = 0;
constexpr int32_t kSamplingDiffusionMaxDenoisingSteps = 0;
constexpr uint32_t kSamplingFlags = kStream;

// default values for the options
constexpr int kDefaultIterations = 1;
constexpr float kDefaultTemperature = 0.8F;
constexpr char const* kDefaultPrompt = "Briefly introduce yourself.";
constexpr char const* kDefaultSystemPrompt = "You are a helpful multimodal assistant. Answer concisely.";

constexpr char const* kUsage
    = "usage: media_ipc_client [options]\n"
      "  --image SPEC           PATH:WxH:NV12_PL|NV12_BL:0|1 (0=NvSciBufObj, 1=NvBufSurface); no image if omitted\n"
      "  --audio PATH           f32le mono PCM raw file, 16 kHz; no audio if omitted\n"
      "  --prompt TEXT          user prompt\n"
      "  --systemPrompt TEXT    system prompt\n"
      "  --iterations N         number of requests to send (default 1)\n"
      "  --temperature X        sampling temperature (default 0.8)\n"
      "  --topP X               top-p sampling threshold (default 0.8)\n"
      "  --topK N               top-k sampling size (default 50)\n"
      "Connects to /tmp/edgellm-ipc.sock and prints each answer to stdout as it streams.\n";

option const kOptions[] = {
    {"image", required_argument, nullptr, 'i'},
    {"audio", required_argument, nullptr, 'a'},
    {"prompt", required_argument, nullptr, 'p'},
    {"systemPrompt", required_argument, nullptr, 'S'},
    {"iterations", required_argument, nullptr, 'n'},
    {"temperature", required_argument, nullptr, 't'},
    {"topP", required_argument, nullptr, 'P'},
    {"topK", required_argument, nullptr, 'k'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0},
};

struct ImageSpec
{
    std::string path;
    Resolution resolution;
    bool blockLinear;
    bool nvBufSurface; //!< BufferType 1; 0 is NvSciBufObj.
};

//! PATH:WxH:NV12_PL|NV12_BL:0|1, split from the right so PATH may hold ':'.
ImageSpec parseImageSpec(std::string spec)
{
    auto const popField = [&spec] {
        size_t const sep = spec.rfind(':');
        if (sep == std::string::npos)
        {
            throw std::runtime_error("image must be PATH:WxH:NV12_PL|NV12_BL:0|1");
        }
        std::string field = spec.substr(sep + 1U);
        spec.resize(sep);
        return field;
    };
    std::string const bufferType = popField();
    std::string const layout = popField();
    std::string const resolution = popField();

    ImageSpec parsed{spec, {}, layout == "NV12_BL", bufferType == "1"};
    if (parsed.path.empty() || (layout != "NV12_BL" && layout != "NV12_PL") || (bufferType != "0" && bufferType != "1"))
    {
        throw std::runtime_error("image must be PATH:WxH:NV12_PL|NV12_BL:0|1");
    }
#if !EDGELLM_IPC_HAS_NVBUFSURFACE
    if (parsed.nvBufSurface)
    {
        throw std::runtime_error("this binary was built without NvBufSurface; use BufferType 0 (NvSciBufObj)");
    }
#endif
    size_t const xSep = resolution.find('x');
    parsed.resolution.width = static_cast<uint32_t>(std::strtoul(resolution.c_str(), nullptr, 10));
    parsed.resolution.height = xSep == std::string::npos
        ? 0U
        : static_cast<uint32_t>(std::strtoul(resolution.c_str() + xSep + 1U, nullptr, 10));
    if (parsed.resolution.width == 0U || parsed.resolution.height == 0U)
    {
        throw std::runtime_error("resolution must be WxH with positive sizes, e.g. 1000x1000");
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv)
{
    std::optional<ImageSpec> imageInput;
    std::string audioPath;
    std::string prompt = kDefaultPrompt;
    std::string systemPrompt = kDefaultSystemPrompt;
    int iterations = kDefaultIterations;
    // LLM generation related parameters.
    SamplingParams params{};
    params.temperature = kDefaultTemperature;
    params.maxGenerateLength = kSamplingMaxGenerateLength;
    params.numLogprobs = kSamplingNumLogprobs;
    params.diffusionMaxDenoisingSteps = kSamplingDiffusionMaxDenoisingSteps;
    params.flags = kSamplingFlags;
    for (int option = 0; (option = getopt_long(argc, argv, "h", kOptions, nullptr)) != -1;)
    {
        switch (option)
        {
        case 'i': imageInput = parseImageSpec(optarg); break;
        case 'a': audioPath = optarg; break;
        case 'p': prompt = optarg; break;
        case 'S': systemPrompt = optarg; break;
        case 'n': iterations = std::stoi(optarg); break;
        case 't': params.temperature = std::stof(optarg); break;
        case 'P': params.topP = std::stof(optarg); break;
        case 'k': params.topK = std::stoll(optarg); break;
        case 'h': std::fputs(kUsage, stdout); return 0;
        default: std::fputs(kUsage, stderr); return 1;
        }
    }
    if (optind != argc || iterations < 1)
    {
        std::fputs(kUsage, stderr);
        return 1;
    }

    // ----------------------step 1 load NV12 frame and PCM data if given---------------------
    std::vector<uint8_t> const image
        = imageInput ? readFile<uint8_t>(imageInput->path, nv12Bytes(imageInput->resolution)) : std::vector<uint8_t>{};
    std::vector<float> const pcm = audioPath.empty() ? std::vector<float>{} : readFile<float>(audioPath);

    // ----------------------step 2 connect to the server and set up shared memory for the audio---------------------
    Client client(/*audioSamples=*/pcm.size());

    auto const sendFrames = [&](auto nvBuf /* NvBufSurface* or NvSciBufObj; null without an image */) {
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            // ----------------------step 3 copy the image into the nvBuf and the PCM into shared memory----------
            if (nvBuf != nullptr)
            {
                copyNV12ToNvBuf(nvBuf, imageInput->resolution, image.data());
            }
            copyPcmToShm(client.audioBuffer(), pcm.data(), pcm.size());

            // ----------------------step 4 send the request to the server---------------------
            client.request(params, nvBuf, systemPrompt, prompt);

            // ----------------------step 5 print the streamed answer until it finishes---------------------
            for (;;)
            {
                ResponseChunk const chunk = client.awaitResponses();
                std::fwrite(chunk.text.data(), 1, chunk.text.size(), stdout);
                std::fflush(stdout);
                if (chunk.finished)
                {
                    std::fputc('\n', stdout);
                    std::fflush(stdout);
                    std::fprintf(stderr,
                        "\nperf: requestId=%u client_round_trip_ms=%.2f server_inference_ms=%u isl=%u osl=%u\n\n",
                        chunk.requestId, static_cast<double>(chunk.latencyNs) / 1e6, chunk.serverInferenceMs,
                        chunk.inputTokens, chunk.outputTokens);
                    break;
                }
            }
        }

        if (nvBuf != nullptr)
        {
            destroyNvBuf(nvBuf);
        }
    };

    if (!imageInput)
    {
        sendFrames(NvSciBufObj{nullptr});
    }
#if EDGELLM_IPC_HAS_NVBUFSURFACE
    else if (imageInput->nvBufSurface)
    {
        sendFrames(allocateNvBufSurface(imageInput->resolution, imageInput->blockLinear));
    }
#endif
    else
    {
        sendFrames(allocateNvSciBuf(client.nvSciBufNeeds(), imageInput->resolution, imageInput->blockLinear));
    }

    std::fprintf(stderr, "%d requests sent and answered\n", iterations);
    return 0;
}
