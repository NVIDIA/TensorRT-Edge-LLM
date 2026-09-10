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

//! Both InternVLA-N1 systems asynchronously in one resident process, one request per line.
//!
//! This is the deployment surface for the arrangement the model declares
//! (`system1: nextdit_async`): System 2 plans on a background thread while System 1 keeps
//! sampling from the newest plan available, and a trajectory request never waits for the
//! planner. `internvla_n1_dual_system_inference` demonstrates the same contract on tensors
//! read from files; this binary takes the fresh frames an agent produces at every step.
//!
//! One process is what makes the arrangement worth anything. CUDA orders streams within a
//! context, so System 1's priority stream only outranks the planner when the two share one;
//! across processes the GPU time-slices and the priority is invisible. Measured under a
//! competing load, a trajectory costs 73.3 ms in-process against 121 ms across processes.
//! The planner also waits on a condition variable rather than polling, so a plan is picked up
//! the moment it lands instead of at the next poll tick.
//!
//! Protocol, one JSON object per line in each direction:
//!   {"request":"text", "raw_text":"...", "images":[paths], "max_new_tokens":64}
//!       -> {"text":"..."}                       System-2 decision, synchronous.
//!   {"request":"replan", "raw_text":"...<|latent_q0..3|>", "images":[paths], "wait":false}
//!       -> {"queued":true}   Wakes the planner and returns immediately; "wait":true blocks
//!                            until the plan lands, which the first observation needs.
//!   {"request":"trajectory", "images_bytes":N, "noise_bytes":M, "num_frames":2}
//!       + N then M raw bytes on stdin
//!       -> {"trajectory_bytes":K, "staleness":S} + K raw bytes on stdout
//! A blank line or EOF on stdin shuts the server down.
//!
//! Tensors travel as raw float32 behind the header rather than as JSON arrays: a 2x3x224x224
//! frame window is 301,056 values, and parsing that as text cost 545 ms per step -- more than
//! every engine in the pipeline put together. The reply is binary for the same reason.

#include "action/internvlaN1System1Runner.h"
#include "runtime/internvlaN1DualSystem.h"

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::internvla_n1;
using json = nlohmann::json;

namespace
{
//! Any index but 0; the runtime parks the input embeddings at 0.
constexpr int32_t kBridgeLayer = 1;
constexpr int32_t kLatentDim = 768;
constexpr int32_t kNumQuery = 4;

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

std::vector<float> toHostFloat(rt::Tensor const& t)
{
    size_t const count = static_cast<size_t>(t.getShape().volume());
    std::vector<float> out(count);
    if (t.getDataType() == nvinfer1::DataType::kHALF)
    {
        std::vector<__half> raw(count);
        cudaMemcpy(raw.data(), t.rawPointer(), raw.size() * sizeof(__half), cudaMemcpyDefault);
        for (size_t i = 0; i < count; ++i)
        {
            out[i] = __half2float(raw[i]);
        }
    }
    else
    {
        cudaMemcpy(out.data(), t.rawPointer(), out.size() * sizeof(float), cudaMemcpyDefault);
    }
    return out;
}

rt::Tensor toDevice(std::vector<float> const& host, std::vector<int64_t> const& shape, char const* name)
{
    rt::Tensor tensor(rt::Coords(shape), rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, name);
    cudaMemcpy(tensor.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

//! \brief Build the [2, condLen, latentDim] conditioning tensor on the device.
//!
//! The layout is [null; memory + z_latents] with the null row first -- what classifier-free
//! guidance blends against, and what sampleTrajectory expects. Assembled on the GPU so the
//! memory tokens never leave it; only the four z_latents rows start on the host.
rt::Tensor makeConditioning(rt::Tensor const& memoryTokens, int32_t numMemory, std::vector<float> const& z,
    int32_t numQuery, cudaStream_t stream)
{
    int64_t const condLen = numMemory + numQuery;
    rt::Tensor cond(rt::Coords(std::vector<int64_t>{2, condLen, kLatentDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "cond");
    auto* base = static_cast<float*>(cond.rawPointer());
    // The null row stays zero, so one memset covers it and every gap.
    CUDA_CHECK(cudaMemsetAsync(base, 0, cond.getMemoryCapacity(), stream));
    float* real = base + static_cast<size_t>(condLen) * kLatentDim;
    CUDA_CHECK(cudaMemcpyAsync(real, memoryTokens.rawPointer(),
        static_cast<size_t>(numMemory) * kLatentDim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(real + static_cast<size_t>(numMemory) * kLatentDim, z.data(), z.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    return cond;
}

//! What the planner thread needs, published by a "replan" request.
struct PlanInput
{
    std::mutex mutex;
    std::string rawText;
    std::vector<std::string> imagePaths;
};

void fillRequest(
    rt::LLMGenerationRequest& request, std::string const& rawText, std::vector<std::string> const& imagePaths)
{
    request.requests.resize(1);
    rt::Message msg;
    msg.role = "user";
    msg.contents.push_back({"text", rawText});
    request.requests[0].messages.push_back(std::move(msg));
    // The caller has already templated the prompt; re-running the engine's template over it
    // would double the control tokens and change the prompt the model sees.
    request.applyChatTemplate = false;
    for (auto const& p : imagePaths)
    {
        auto image = rt::imageUtils::loadImageFromFile(p);
        if (image.buffer != nullptr)
        {
            request.requests[0].imageBuffers.push_back(std::move(image));
        }
    }
    // None of these have default initialisers; leaving topK uninitialised makes the sampler
    // size its workspace from stack garbage, reported as an 18-exabyte allocation.
    request.temperature = 1.0F;
    request.topP = 1.0F;
    request.topK = 1;
}
} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const actionDir = argOf(argc, argv, "--actionEngineDir");
    std::string const visDir = argOf(argc, argv, "--multimodalEngineDir");
    if (llmDir.empty() || actionDir.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --actionEngineDir DIR [--multimodalEngineDir DIR] "
            "[--numTrajs 32] [--steps 10] [--guidance 1.0]\n",
            argv[0]);
        return 2;
    }

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t s2Stream{};
    cudaStreamCreate(&s2Stream);
    // System 1 produces the control output, so it gets the priority stream. Effective here
    // precisely because System 2 shares the process.
    cudaStream_t const s1Stream = InternVLAN1System1Runner::makeControlStream();

    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(llmDir, visDir, noLora, s2Stream);

    InternVLAN1System1Runner::Config config;
    config.numSampleTrajs = std::stoi(argOf(argc, argv, "--numTrajs", "32"));
    config.numInferenceSteps = std::stoi(argOf(argc, argv, "--steps", "10"));
    config.guidanceScale = std::stof(argOf(argc, argv, "--guidance", "1.0"));
    InternVLAN1System1Runner s1(actionDir, config, s1Stream);

    PlanInput planInput;
    // Memory tokens are recomputed per trajectory request from that step's frames, so the
    // planner contributes only the z_latents half of the conditioning and the trajectory
    // handler concatenates the two. Keeping them apart is what lets a plan outlive the frames
    // it was computed from, which is the point of the asynchronous arrangement.
    std::mutex zMutex;
    std::vector<float> latestZ;

    InternVLAN1DualSystemState state(InternVLAN1DualSystemState::Mode::kPartialAsync);
    InternVLAN1DualSystemDriver driver(state, [&](int64_t observationIndex) -> InternVLAN1DualSystemState::Plan {
        std::string rawText;
        std::vector<std::string> images;
        {
            std::lock_guard<std::mutex> lock(planInput.mutex);
            rawText = planInput.rawText;
            images = planInput.imagePaths;
        }
        InternVLAN1DualSystemState::Plan plan;
        rt::LLMGenerationRequest request;
        fillRequest(request, rawText, images);
        // The bridge needs the prefill, not generated text; one token is enough.
        request.maxGenerateLength = 1;
        request.acceptHiddenLayer = kBridgeLayer;

        rt::LLMGenerationResponse response;
        if (!runtime.handleRequest(request, response, s2Stream, /*outputThinkerEmbeddings=*/true))
        {
            return plan;
        }
        rt::Tensor const* hidden = runtime.getBaseModelHiddenStates(kBridgeLayer);
        if (hidden == nullptr || hidden->isEmpty())
        {
            return plan;
        }
        auto const all = toHostFloat(*hidden);
        if (all.size() < static_cast<size_t>(kNumQuery) * kLatentDim)
        {
            return plan;
        }
        {
            std::lock_guard<std::mutex> lock(zMutex);
            latestZ.assign(all.begin(), all.begin() + static_cast<int64_t>(kNumQuery) * kLatentDim);
        }
        plan.condLen = kNumQuery;
        plan.latentDim = kLatentDim;
        plan.observationIndex = observationIndex;
        return plan;
    });

    std::printf("{\"ready\":true}\n");
    std::fflush(stdout);

    int64_t observationIndex = 0;
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            break;
        }
        json out;
        std::vector<float> trajectoryOut;
        try
        {
            auto const in = json::parse(line);
            std::string const kind = in.value("request", "text");

            if (kind == "text")
            {
                rt::LLMGenerationRequest request;
                fillRequest(
                    request, in.value("raw_text", std::string{}), in.value("images", std::vector<std::string>{}));
                request.maxGenerateLength = in.value("max_new_tokens", 64);
                rt::LLMGenerationResponse response;
                if (!runtime.handleRequest(request, response, s2Stream, /*outputThinkerEmbeddings=*/false))
                {
                    out["error"] = "handleRequest failed";
                }
                else
                {
                    out["text"] = response.outputTexts.empty() ? std::string{} : response.outputTexts[0];
                }
            }
            else if (kind == "replan")
            {
                {
                    std::lock_guard<std::mutex> lock(planInput.mutex);
                    planInput.rawText = in.value("raw_text", std::string{});
                    planInput.imagePaths = in.value("images", std::vector<std::string>{});
                }
                // Returns immediately: requestReplan notifies a condition variable, so the
                // planner wakes at once and the trajectory loop keeps sampling meanwhile.
                driver.requestReplan(observationIndex);
                // Except on the first observation, where there is no previous plan to steer
                // on and returning immediately only produces an error. `wait` blocks until
                // the plan lands, which is what internvla_n1_dual_system_inference does
                // before entering its loop. An agent sets it once and never again.
                if (in.value("wait", false))
                {
                    driver.waitIdle();
                }
                out["queued"] = true;
            }
            else if (kind == "trajectory")
            {
                // The payload sits immediately behind the header, so it must be consumed even
                // on a later error, or the stream desynchronizes for every later request.
                auto const readPayload = [](size_t nbytes) {
                    std::vector<float> buf(nbytes / sizeof(float));
                    std::cin.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(nbytes));
                    if (static_cast<size_t>(std::cin.gcount()) != nbytes)
                    {
                        throw std::runtime_error("short read on the request payload");
                    }
                    return buf;
                };
                auto const imagesHost = readPayload(in.at("images_bytes").get<size_t>());
                auto const noiseHost = readPayload(in.at("noise_bytes").get<size_t>());
                int32_t const numFrames = in.value("num_frames", 2);

                std::vector<float> z;
                {
                    std::lock_guard<std::mutex> lock(zMutex);
                    z = latestZ;
                }
                if (z.empty())
                {
                    // Nothing to steer on yet. Reporting it beats silently sampling against
                    // zeros, which would look like a working but very bad policy.
                    out["error"] = "no plan published yet";
                }
                else
                {
                    rt::Tensor const frames = toDevice(imagesHost, {numFrames, 3, 224, 224}, "frames");
                    rt::Tensor& memoryTokens = s1.encodeMemory(frames, s1Stream);
                    int32_t const numMemory = static_cast<int32_t>(memoryTokens.getShape().volume() / kLatentDim);
                    int32_t const numQuery = static_cast<int32_t>(z.size() / kLatentDim);
                    rt::Tensor const cond = makeConditioning(memoryTokens, numMemory, z, numQuery, s1Stream);
                    rt::Tensor const noise = toDevice(
                        noiseHost, {config.numSampleTrajs, config.predictStepNums, config.actionDim}, "noise");
                    rt::Tensor& traj = s1.sampleTrajectory(cond, noise, s1Stream);
                    cudaStreamSynchronize(s1Stream);
                    trajectoryOut = toHostFloat(traj);
                    out["trajectory_bytes"] = trajectoryOut.size() * sizeof(float);
                    // How many observations old the plan being steered by is. A caller that
                    // never sees this rise is running effectively synchronously.
                    out["staleness"] = state.stalenessAt(observationIndex);
                }
                ++observationIndex;
            }
            else
            {
                out["error"] = "unknown request kind: " + kind;
            }
        }
        catch (std::exception const& e)
        {
            out["error"] = e.what();
        }
        std::printf("%s\n", out.dump().c_str());
        if (!trajectoryOut.empty())
        {
            std::fwrite(trajectoryOut.data(), sizeof(float), trajectoryOut.size(), stdout);
        }
        std::fflush(stdout);
    }
    driver.stop();
    cudaStreamDestroy(s2Stream);
    cudaStreamDestroy(s1Stream);
    return 0;
}
