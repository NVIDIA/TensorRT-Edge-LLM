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

//! Both InternVLA-N1 systems in one process, running asynchronously.
//!
//! System 2 plans on a background thread; System 1 keeps producing trajectories from the newest
//! plan available rather than waiting for a fresh one. Running them in one process is what makes
//! the priority stream effective -- CUDA orders streams within a context, so across processes the
//! priority is invisible and the GPU simply time-slices.
//!
//! The bridge between them runs here on the host: the engine emits model-width hidden states, and
//! the final norm plus cond_projector turn the trajectory-query rows into z_latents. It is four
//! rows of arithmetic, so a plain host loop costs nothing next to a 623 ms plan and keeps the step
//! easy to check against the reference.
//!
//! The latent queries travel as real tokens. The export writes the four learned embeddings into
//! the embedding table's trailing padding rows and registers `<|latent_q0..3|>` as special
//! tokens, so a prompt ending with them puts the queries into the sequence through the runtime's
//! ordinary lookup -- the route Alpamayo uses for its trajectory tokens. The engine's graph folds
//! the final norm and cond_projector, so the hidden-states buffer's first
//! `n_query * latent_dim` elements *are* the z_latents; the rest of the buffer is the runtime's
//! model-width copy convention and is ignored.

#include "action/internvlaN1System1Runner.h"
#include "runtime/internvlaN1DualSystem.h"

#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Clock = std::chrono::steady_clock;

namespace
{

//! Any index but 0; the runtime parks the input embeddings at 0.
constexpr int32_t kBridgeLayer = 1;
constexpr int32_t kLatentDim = 768;

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

std::vector<float> readFloats(std::string const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<float> data(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const actionDir = argOf(argc, argv, "--actionEngineDir");
    std::string const framesPath = argOf(argc, argv, "--frames");
    std::string const noisePath = argOf(argc, argv, "--noise");
    std::string const outputPath = argOf(argc, argv, "--output");
    if (llmDir.empty() || actionDir.empty() || framesPath.empty() || noisePath.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --actionEngineDir DIR\n"
            "          --frames frames.bin --noise noise.bin\n"
            "          [--ticks 40] [--cadence 4] [--numFrames 2] [--prompt TEXT] [--output traj.bin]\n"
            "          [--numTrajs 32] [--steps 10] [--guidance 1.0]\n",
            argv[0]);
        return 2;
    }
    int32_t const ticks = std::stoi(argOf(argc, argv, "--ticks", "40"));
    int64_t const cadence = std::stoll(argOf(argc, argv, "--cadence", "4"));
    int32_t const numFrames = std::stoi(argOf(argc, argv, "--numFrames", "2"));
    std::string const userText = argOf(argc, argv, "--prompt",
        "You are an autonomous navigation assistant. Your task is to go to the kitchen. "
        "Where should you go next to stay on track?");
    // The queries must sit where the model was trained to find them: after the assistant
    // generation prompt. handleRequest's template always closes the user turn first, so the
    // ChatML is assembled here and the template is turned off for this request.
    std::string const prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                               "<|im_start|>user\n"
        + userText
        + "<|im_end|>\n<|im_start|>assistant\n"
          "<|latent_q0|><|latent_q1|><|latent_q2|><|latent_q3|>";

    auto const pluginHandles = loadEdgellmPluginLib();

    // System 1 owns the priority stream: it produces the control output, so it is the workload
    // that must not be starved. System 2 gets an ordinary one and is allowed to take longer.
    cudaStream_t s1Stream = internvla_n1::InternVLAN1System1Runner::makeControlStream();
    cudaStream_t s2Stream{};
    cudaStreamCreate(&s2Stream);

    std::printf("[1/4] loading System 2\n");
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(llmDir, "", noLora, s2Stream);

    std::printf("[2/4] loading System 1\n");
    internvla_n1::InternVLAN1System1Runner::Config config;
    config.numSampleTrajs = std::stoi(argOf(argc, argv, "--numTrajs", "32"));
    config.numInferenceSteps = std::stoi(argOf(argc, argv, "--steps", "10"));
    // 1.0 is what InternNav deploys: every generate_traj call site (realworld agent, policy,
    // habitat evaluator) leaves guidance_scale at its default. At 1.0 the CFG blend reduces to
    // the conditioned branch, so the value must match the reference exactly for trajectories to
    // be comparable -- a mismatch here is invisible until a reference comparison.
    config.guidanceScale = std::stof(argOf(argc, argv, "--guidance", "1.0"));
    internvla_n1::InternVLAN1System1Runner s1(actionDir, config, s1Stream);

    std::printf("[3/4] encoding the observation window\n");
    auto const frameHost = readFloats(framesPath);
    rt::Tensor frames(rt::Coords(std::vector<int64_t>{numFrames, 3, 224, 224}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "frames");
    cudaMemcpy(frames.rawPointer(), frameHost.data(), frameHost.size() * sizeof(float), cudaMemcpyHostToDevice);
    rt::Tensor& memoryTokens = s1.encodeMemory(frames, s1Stream);
    cudaStreamSynchronize(s1Stream);
    auto const memoryHost = toHostFloat(memoryTokens);
    int32_t const numMemory = static_cast<int32_t>(memoryHost.size() / kLatentDim);

    auto const noiseHost = readFloats(noisePath);
    rt::Tensor noise(rt::Coords(std::vector<int64_t>{config.numSampleTrajs, config.predictStepNums, config.actionDim}),
        rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "noise");
    cudaMemcpy(noise.rawPointer(), noiseHost.data(), noiseHost.size() * sizeof(float), cudaMemcpyHostToDevice);

    // The planner runs on the driver's thread. Everything it touches is either its own or
    // published atomically, so the trajectory loop never blocks on it.
    internvla_n1::InternVLAN1DualSystemState state(internvla_n1::InternVLAN1DualSystemState::Mode::kPartialAsync);
    std::atomic<int32_t> planCount{0};
    internvla_n1::InternVLAN1DualSystemDriver driver(
        state, [&](int64_t observationIndex) -> internvla_n1::InternVLAN1DualSystemState::Plan {
            rt::LLMGenerationRequest request;
            request.requests.resize(1);
            rt::Message message;
            message.role = "user";
            message.contents.push_back({"text", prompt});
            request.requests[0].messages.push_back(message);
            request.acceptHiddenLayer = kBridgeLayer;
            request.applyChatTemplate = false;
            // temperature, topP and topK have no default initializers in the struct. Leaving
            // them uninitialized makes the sampler compute a workspace from garbage; the
            // symptom is a size_t underflow reported as an 18-exabyte allocation.
            request.temperature = 1.0F;
            request.topP = 1.0F;
            request.topK = 1;
            // Also without a default initializer. The bridge needs the prefill, not the text, so
            // one token is enough and anything larger only spends time generating what is thrown
            // away.
            request.maxGenerateLength = 1;

            rt::LLMGenerationResponse response;
            internvla_n1::InternVLAN1DualSystemState::Plan plan;
            if (!runtime.handleRequest(request, response, s2Stream, /*outputThinkerEmbeddings=*/true))
            {
                return plan;
            }
            rt::Tensor const* hidden = runtime.getBaseModelHiddenStates(kBridgeLayer);
            if (hidden == nullptr || hidden->isEmpty())
            {
                return plan;
            }
            // The graph already applied the norm and cond_projector, so the first
            // numQuery * latent_dim elements of the buffer are the z_latents. The buffer's
            // reported shape is the runtime's model-width convention; only the prefix is real.
            int32_t const numQuery = 4;
            auto const all = toHostFloat(*hidden);
            std::vector<float> const z(all.begin(), all.begin() + static_cast<int64_t>(numQuery) * kLatentDim);

            // Conditioning is [null; memory ⧺ z_latents]; the null row is what classifier-free
            // guidance blends against.
            int32_t const condLen = numMemory + numQuery;
            plan.condLen = condLen;
            plan.latentDim = kLatentDim;
            plan.observationIndex = observationIndex;
            plan.conditioning.assign(static_cast<size_t>(2) * condLen * kLatentDim, 0.0F);
            float* real = plan.conditioning.data() + static_cast<size_t>(condLen) * kLatentDim;
            std::copy(memoryHost.begin(), memoryHost.end(), real);
            std::copy(z.begin(), z.end(), real + static_cast<size_t>(numMemory) * kLatentDim);
            return plan;
        });

    // Wait for the first plan before entering the loop. A robot cannot steer on nothing, and
    // without this the loop spins through every tick in microseconds while System 2 is still on
    // its first plan -- measuring the empty case rather than the real one.
    std::printf("[4/4] waiting for the first plan\n");
    auto const planStart = Clock::now();
    driver.requestReplan(0);
    driver.waitIdle();
    std::printf(
        "      first plan in %.0f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - planStart).count());

    std::printf("      running %d System-1 ticks, replanning every %ld\n", ticks, static_cast<long>(cadence));
    // Conditioning changes only when a new plan lands -- once every `cadence` ticks at most --
    // so upload it then rather than every tick. Re-uploading each tick costs a host-side copy of
    // the plan, a fresh device allocation and a transfer, all to move bytes that did not change.
    rt::Tensor cond;
    int64_t uploadedPlan = -1;
    int32_t uploads = 0;
    int32_t stalled = 0;
    int32_t ran = 0;
    double worstTick = 0.0;
    auto const start = Clock::now();
    for (int32_t tick = 0; tick < ticks; ++tick)
    {
        auto const tickStart = Clock::now();
        if (state.shouldReplan(tick, cadence, /*forced=*/false))
        {
            driver.requestReplan(tick);
        }
        // Ask for the index first: it is a scalar under the lock, where latest() copies the whole
        // conditioning vector. Only fetch the plan itself when it is one we have not uploaded.
        int64_t const staleness = state.stalenessAt(tick);
        bool const havePlan = staleness >= 0;
        if (havePlan && tick - staleness != uploadedPlan)
        {
            internvla_n1::InternVLAN1DualSystemState::Plan plan;
            if (state.latest(plan))
            {
                if (cond.isEmpty())
                {
                    cond = rt::Tensor(rt::Coords(std::vector<int64_t>{2, plan.condLen, plan.latentDim}),
                        rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cond");
                }
                cudaMemcpyAsync(cond.rawPointer(), plan.conditioning.data(), plan.conditioning.size() * sizeof(float),
                    cudaMemcpyHostToDevice, s1Stream);
                uploadedPlan = plan.observationIndex;
                ++uploads;
            }
        }
        if (!cond.isEmpty())
        {
            rt::Tensor& traj = s1.sampleTrajectory(cond, noise, s1Stream);
            ++ran;
            // Write the final tick's trajectory so a run can be compared against the reference.
            if (!outputPath.empty() && tick == ticks - 1)
            {
                cudaStreamSynchronize(s1Stream);
                auto const host = toHostFloat(traj);
                std::ofstream out(outputPath, std::ios::binary);
                out.write(reinterpret_cast<char const*>(host.data()),
                    static_cast<std::streamsize>(host.size() * sizeof(float)));
                std::printf("      wrote %s (%zu floats)\n", outputPath.c_str(), host.size());
            }
        }
        else
        {
            ++stalled; // no plan yet -- the head has nothing to steer with
        }

        worstTick = std::max(worstTick, std::chrono::duration<double, std::milli>(Clock::now() - tickStart).count());
    }
    double const total = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    driver.stop();

    std::printf("\n");
    std::printf("  System-1 ticks run      : %d (%d before the first plan landed)\n", ran, stalled);
    std::printf("  mean tick               : %.2f ms (%.1f Hz)\n", total / ticks, 1000.0 * ticks / total);
    std::printf("  worst tick              : %.2f ms\n", worstTick);
    std::printf("  plans completed         : %ld\n", static_cast<long>(driver.plansCompleted()));
    std::printf("  conditioning uploads    : %d (one per plan, not per tick)\n", uploads);
    std::printf("  final staleness         : %ld observations\n", static_cast<long>(state.stalenessAt(ticks - 1)));

    cudaStreamDestroy(s1Stream);
    cudaStreamDestroy(s2Stream);
    return 0;
}
