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

//! System-2 replan benchmark for InternVLA-N1 over a simulated navigation episode.
//!
//! Each step issues the planner's request (InternNav prompt, `num_history` frames picked by
//! `unique(linspace(0, step - 1, num_history))` plus the current frame, latent queries appended) and
//! records its latency and z_latents. Running the same episode with different cache settings and
//! comparing the z dumps checks that a cache changes latency only.

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kBridgeLayer = 1;
constexpr int32_t kLatentDim = 768;
constexpr int32_t kNumQuery = 4;
constexpr char const* kImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";

std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback = "")
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool hasFlag(int argc, char** argv, char const* flag)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0)
        {
            return true;
        }
    }
    return false;
}

//! InternNav `s2_step` history selection: np.unique(np.linspace(0, step - 1, numHistory, dtype=int32)).
std::vector<int32_t> historyIds(int32_t step, int32_t numHistory)
{
    std::set<int32_t> ids;
    if (step == 0)
    {
        return {};
    }
    for (int32_t i = 0; i < numHistory; ++i)
    {
        double const v = numHistory == 1 ? 0.0 : static_cast<double>(step - 1) * i / (numHistory - 1);
        ids.insert(static_cast<int32_t>(v));
    }
    return {ids.begin(), ids.end()};
}

//! The InternNav prompt rendered through the Qwen2.5-VL chat template, latent queries appended.
std::string buildPrompt(std::string const& instruction, size_t numHistory)
{
    std::string user = "You are an autonomous navigation assistant. Your task is to " + instruction
        + " Where should you go next to stay on track? Please output the next waypoint's coordinates in the "
          "image. Please output STOP when you have successfully completed the task.";
    if (numHistory > 0)
    {
        user += " These are your historical observations:";
        for (size_t i = 0; i < numHistory; ++i)
        {
            user += kImagePlaceholder;
        }
        user += ".";
    }
    user += " you can see";
    user += kImagePlaceholder;
    user += ".";
    std::string text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + user
        + "<|im_end|>\n<|im_start|>assistant\n";
    for (int32_t i = 0; i < kNumQuery; ++i)
    {
        text += "<|latent_q" + std::to_string(i) + "|>";
    }
    return text;
}

std::vector<float> toHostFloat(rt::Tensor const& t, size_t count)
{
    std::vector<float> out(count);
    if (t.getDataType() == nvinfer1::DataType::kHALF)
    {
        std::vector<__half> raw(count);
        cudaMemcpy(raw.data(), t.rawPointer(), count * sizeof(__half), cudaMemcpyDefault);
        std::transform(raw.begin(), raw.end(), out.begin(), [](__half h) { return __half2float(h); });
    }
    else
    {
        cudaMemcpy(out.data(), t.rawPointer(), count * sizeof(float), cudaMemcpyDefault);
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const visDir = argOf(argc, argv, "--multimodalEngineDir");
    std::string const framesDir = argOf(argc, argv, "--framesDir");
    if (llmDir.empty() || visDir.empty() || framesDir.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --multimodalEngineDir DIR --framesDir DIR\n"
            "          [--steps 30] [--numHistory 8] [--instruction TEXT] [--zOut z.bin]\n"
            "          [--encoderCacheBudgetBytes N (0 disables)] [--enableContextReuse] [--repeat]\n"
            "  framesDir holds 000000.png, 000001.png, ... one per episode step.\n",
            argv[0]);
        return 1;
    }
    int32_t const steps = std::stoi(argOf(argc, argv, "--steps", "30"));
    int32_t const numHistory = std::stoi(argOf(argc, argv, "--numHistory", "8"));
    std::string const instruction = argOf(argc, argv, "--instruction",
        "walk past the sofa, turn left into the hallway and stop at the second door on the right");
    std::string const zOut = argOf(argc, argv, "--zOut");
    // Sends every request twice; the second send finds all of its images in the encoder cache, so the
    // latency gap is the vision cost and the z gap is any error the cache introduces.
    bool const repeat = hasFlag(argc, argv, "--repeat");

    rt::ContextCacheConfig cacheConfig;
    cacheConfig.enabled = hasFlag(argc, argv, "--enableContextReuse");
    cacheConfig.encoderEmbeddingCacheBudgetBytes
        = std::stoll(argOf(argc, argv, "--encoderCacheBudgetBytes", std::to_string(256LL * 1024 * 1024)));

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(llmDir, visDir, noLora, stream, cacheConfig);

    std::vector<rt::imageUtils::ImageData> frames;
    for (int32_t s = 0; s < steps; ++s)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "/%06d.png", s);
        frames.push_back(rt::imageUtils::loadRgbImageFromFile(framesDir + name));
    }

    std::ofstream zFile;
    if (!zOut.empty())
    {
        zFile.open(zOut, std::ios::binary);
    }
    std::vector<double> latencies;
    std::vector<double> repeatLatencies;
    std::printf(repeat ? "step  images  latency_ms  repeat_ms  max|dz|\n" : "step  images  latency_ms\n");
    for (int32_t step = 0; step < steps; ++step)
    {
        std::vector<int32_t> const history = historyIds(step, numHistory);
        rt::LLMGenerationRequest request;
        request.requests.resize(1);
        rt::Message msg;
        msg.role = "user";
        msg.contents.push_back({"text", buildPrompt(instruction, history.size())});
        request.requests[0].messages.push_back(std::move(msg));
        for (int32_t id : history)
        {
            request.requests[0].imageBuffers.push_back(frames[id]);
        }
        request.requests[0].imageBuffers.push_back(frames[step]);
        request.applyChatTemplate = false;
        request.maxGenerateLength = 1;
        request.acceptHiddenLayer = kBridgeLayer;
        // Only the latent-query rows are read, so the context cache may restore the prompt before them.
        request.hiddenCaptureTailTokens = kNumQuery;
        request.temperature = 1.0F;
        request.topP = 1.0F;
        request.topK = 1;

        auto const run = [&](double& ms, std::vector<float>& z) {
            rt::LLMGenerationResponse response;
            auto const t0 = std::chrono::steady_clock::now();
            bool const ok = runtime.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true);
            cudaStreamSynchronize(stream);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            rt::Tensor const* hidden = ok ? runtime.getBaseModelHiddenStates(kBridgeLayer) : nullptr;
            if (hidden == nullptr || hidden->isEmpty())
            {
                return false;
            }
            z = toHostFloat(*hidden, static_cast<size_t>(kNumQuery) * kLatentDim);
            return true;
        };
        double ms = 0;
        std::vector<float> z;
        if (!run(ms, z))
        {
            std::fprintf(stderr, "step %d: planner request failed\n", step);
            return 1;
        }
        if (zFile)
        {
            zFile.write(
                reinterpret_cast<char const*>(z.data()), static_cast<std::streamsize>(z.size() * sizeof(float)));
        }
        double repeatMs = 0;
        double maxAbsDiff = 0;
        if (repeat)
        {
            std::vector<float> z2;
            if (!run(repeatMs, z2))
            {
                std::fprintf(stderr, "step %d: repeated planner request failed\n", step);
                return 1;
            }
            for (size_t i = 0; i < z.size(); ++i)
            {
                maxAbsDiff = std::max(maxAbsDiff, static_cast<double>(std::fabs(z[i] - z2[i])));
            }
            repeatLatencies.push_back(repeatMs);
        }
        latencies.push_back(ms);
        if (repeat)
        {
            std::printf("%4d  %6zu  %10.1f  %10.1f  %.3g\n", step, history.size() + 1, ms, repeatMs, maxAbsDiff);
        }
        else
        {
            std::printf("%4d  %6zu  %10.1f\n", step, history.size() + 1, ms);
        }
    }

    // Step 0 carries one-time warm-up and a single image; summarize the steady state.
    std::vector<double> steady(latencies.begin() + std::min<size_t>(1, latencies.size()), latencies.end());
    if (!steady.empty())
    {
        std::sort(steady.begin(), steady.end());
        double sum = 0;
        for (double v : steady)
        {
            sum += v;
        }
        std::printf("steady state (steps 1..%d): mean %.1f ms  p50 %.1f ms  max %.1f ms\n", steps - 1,
            sum / steady.size(), steady[steady.size() / 2], steady.back());
    }
    if (repeatLatencies.size() > 1)
    {
        double sum = 0;
        for (size_t i = 1; i < repeatLatencies.size(); ++i)
        {
            sum += repeatLatencies[i];
        }
        std::printf("repeat (all images cached), steps 1..%d: mean %.1f ms\n", steps - 1,
            sum / static_cast<double>(repeatLatencies.size() - 1));
    }
    cudaStreamDestroy(stream);
    return 0;
}
