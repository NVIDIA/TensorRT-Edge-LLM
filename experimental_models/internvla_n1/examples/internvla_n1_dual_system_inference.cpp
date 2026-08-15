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
//! rows of arithmetic, so a plain host loop costs nothing next to a 646 ms plan and keeps the step
//! easy to check against the reference.

#include "action/internvlaN1System1Runner.h"
#include "runtime/internvlaN1DualSystem.h"

#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
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
constexpr int32_t kHiddenSize = 3584;
constexpr int32_t kLatentDim = 768;
constexpr float kNormEps = 1e-6F;

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

//! Host-side copy of the bridge weights, in the order the projection applies them.
struct Bridge
{
    std::vector<float> normWeight; //!< [hidden]
    std::vector<float> w0, b0;     //!< [latent, hidden], [latent]
    std::vector<float> w2, b2;     //!< [latent, latent], [latent]
};

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

//! Load cond_projector, latent_queries and the final norm weight from sidecar files.
bool loadBridge(std::string const& bridgePath, std::string const& normPath, Bridge& bridge, cudaStream_t stream)
{
    std::vector<rt::Tensor> tensors;
    if (!rt::safetensors::loadSafetensors(bridgePath, tensors, stream))
    {
        std::fprintf(stderr, "cannot read %s\n", bridgePath.c_str());
        return false;
    }
    for (auto const& t : tensors)
    {
        std::string const& name = t.getName();
        if (name.find("cond_projector.0.weight") != std::string::npos)
        {
            bridge.w0 = toHostFloat(t);
        }
        else if (name.find("cond_projector.0.bias") != std::string::npos)
        {
            bridge.b0 = toHostFloat(t);
        }
        else if (name.find("cond_projector.2.weight") != std::string::npos)
        {
            bridge.w2 = toHostFloat(t);
        }
        else if (name.find("cond_projector.2.bias") != std::string::npos)
        {
            bridge.b2 = toHostFloat(t);
        }
    }
    std::ifstream norm(normPath, std::ios::binary | std::ios::ate);
    if (!norm)
    {
        std::fprintf(stderr, "cannot read %s\n", normPath.c_str());
        return false;
    }
    bridge.normWeight.resize(static_cast<size_t>(norm.tellg()) / sizeof(float));
    norm.seekg(0);
    norm.read(reinterpret_cast<char*>(bridge.normWeight.data()),
        static_cast<std::streamsize>(bridge.normWeight.size() * sizeof(float)));
    return !bridge.w0.empty() && !bridge.w2.empty() && !bridge.normWeight.empty();
}

//! Pre-norm hidden states -> z_latents. RMSNorm, linear, tanh-GELU, linear.
std::vector<float> projectBridge(std::vector<float> const& rows, int32_t numQuery, Bridge const& bridge)
{
    std::vector<float> normed(rows.size());
    for (int32_t q = 0; q < numQuery; ++q)
    {
        float const* src = rows.data() + static_cast<size_t>(q) * kHiddenSize;
        double sum = 0.0;
        for (int32_t i = 0; i < kHiddenSize; ++i)
        {
            sum += static_cast<double>(src[i]) * src[i];
        }
        float const scale = 1.0F / std::sqrt(static_cast<float>(sum / kHiddenSize) + kNormEps);
        for (int32_t i = 0; i < kHiddenSize; ++i)
        {
            normed[static_cast<size_t>(q) * kHiddenSize + i] = src[i] * scale * bridge.normWeight[i];
        }
    }

    std::vector<float> mid(static_cast<size_t>(numQuery) * kLatentDim);
    for (int32_t q = 0; q < numQuery; ++q)
    {
        for (int32_t o = 0; o < kLatentDim; ++o)
        {
            float acc = bridge.b0.empty() ? 0.0F : bridge.b0[static_cast<size_t>(o)];
            float const* w = bridge.w0.data() + static_cast<size_t>(o) * kHiddenSize;
            float const* x = normed.data() + static_cast<size_t>(q) * kHiddenSize;
            for (int32_t i = 0; i < kHiddenSize; ++i)
            {
                acc += w[i] * x[i];
            }
            // GELU, tanh approximation -- the reference uses approximate="tanh".
            float const c = 0.7978845608F * (acc + 0.044715F * acc * acc * acc);
            mid[static_cast<size_t>(q) * kLatentDim + o] = 0.5F * acc * (1.0F + std::tanh(c));
        }
    }

    std::vector<float> out(static_cast<size_t>(numQuery) * kLatentDim);
    for (int32_t q = 0; q < numQuery; ++q)
    {
        for (int32_t o = 0; o < kLatentDim; ++o)
        {
            float acc = bridge.b2.empty() ? 0.0F : bridge.b2[static_cast<size_t>(o)];
            float const* w = bridge.w2.data() + static_cast<size_t>(o) * kLatentDim;
            float const* x = mid.data() + static_cast<size_t>(q) * kLatentDim;
            for (int32_t i = 0; i < kLatentDim; ++i)
            {
                acc += w[i] * x[i];
            }
            out[static_cast<size_t>(q) * kLatentDim + o] = acc;
        }
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
    std::string const bridgePath = argOf(argc, argv, "--bridge");
    std::string const normPath = argOf(argc, argv, "--normWeight");
    std::string const framesPath = argOf(argc, argv, "--frames");
    std::string const noisePath = argOf(argc, argv, "--noise");
    if (llmDir.empty() || actionDir.empty() || bridgePath.empty() || normPath.empty() || framesPath.empty()
        || noisePath.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --actionEngineDir DIR --bridge bridge.safetensors\n"
            "          --normWeight norm.bin --frames frames.bin --noise noise.bin\n"
            "          [--ticks 40] [--cadence 4] [--numFrames 2] [--prompt TEXT]\n",
            argv[0]);
        return 2;
    }
    int32_t const ticks = std::stoi(argOf(argc, argv, "--ticks", "40"));
    int64_t const cadence = std::stoll(argOf(argc, argv, "--cadence", "4"));
    int32_t const numFrames = std::stoi(argOf(argc, argv, "--numFrames", "2"));
    std::string const prompt = argOf(argc, argv, "--prompt",
        "You are an autonomous navigation assistant. Your task is to go to the kitchen. "
        "Where should you go next to stay on track?");

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
    internvla_n1::InternVLAN1System1Runner s1(actionDir, config, s1Stream);

    Bridge bridge;
    if (!loadBridge(bridgePath, normPath, bridge, s1Stream))
    {
        return 1;
    }

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
            // temperature, topP and topK have no default initializers in the struct. Leaving
            // them uninitialized makes the sampler compute a workspace from garbage; the
            // symptom is a size_t underflow reported as an 18-exabyte allocation.
            request.temperature = 1.0F;
            request.topP = 1.0F;
            request.topK = 1;

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
            auto const all = toHostFloat(*hidden);
            int32_t const numQuery = 4;
            std::vector<float> rows(all.end() - static_cast<int64_t>(numQuery) * kHiddenSize, all.end());
            auto const z = projectBridge(rows, numQuery, bridge);

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
            planCount.fetch_add(1);
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
            s1.sampleTrajectory(cond, noise, s1Stream);
            ++ran;
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
    std::printf("  plans completed         : %d\n", planCount.load());
    std::printf("  conditioning uploads    : %d (one per plan, not per tick)\n", uploads);
    std::printf("  final staleness         : %ld observations\n", static_cast<long>(state.stalenessAt(ticks - 1)));

    cudaStreamDestroy(s1Stream);
    cudaStreamDestroy(s2Stream);
    return 0;
}
