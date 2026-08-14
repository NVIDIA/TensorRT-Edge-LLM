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

//! Runs InternVLA-N1 System 1 on tensors supplied as raw float32 files, and writes the sampled
//! trajectories back the same way. Reading the inputs rather than generating them is what makes
//! the run comparable against a reference implementation bit for bit.

#include "action/internvlaN1System1Runner.h"

#include "common/tensor.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::internvla_n1;

namespace
{

std::vector<float> readFloats(std::string const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    auto const bytes = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

rt::Tensor toDevice(std::vector<float> const& host, std::vector<int64_t> const& shape, char const* name)
{
    rt::Tensor tensor(rt::Coords(shape), rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, name);
    cudaMemcpy(tensor.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

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

} // namespace

int main(int argc, char** argv)
{
    std::string const engineDir = argOf(argc, argv, "--engineDir");
    std::string const condPath = argOf(argc, argv, "--conditioning");
    std::string const noisePath = argOf(argc, argv, "--noise");
    std::string const outPath = argOf(argc, argv, "--output", "trajectory.bin");
    if (engineDir.empty() || condPath.empty() || noisePath.empty())
    {
        std::fprintf(stderr,
            "usage: %s --engineDir DIR --conditioning cond.bin --noise noise.bin "
            "[--output traj.bin] [--condLen 36] [--numTrajs 32] [--steps 10] [--guidance 1.0]\n",
            argv[0]);
        return 2;
    }

    InternVLAN1System1Runner::Config config;
    config.numSampleTrajs = std::stoi(argOf(argc, argv, "--numTrajs", "32"));
    config.numInferenceSteps = std::stoi(argOf(argc, argv, "--steps", "10"));
    config.guidanceScale = std::stof(argOf(argc, argv, "--guidance", "1.0"));
    int64_t const condLen = std::stoll(argOf(argc, argv, "--condLen", "36"));

    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    InternVLAN1System1Runner runner(engineDir, config, stream);

    auto const condHost = readFloats(condPath);
    auto const noiseHost = readFloats(noisePath);
    rt::Tensor const cond = toDevice(condHost, {2, condLen, 768}, "cond");
    rt::Tensor const noise
        = toDevice(noiseHost, {config.numSampleTrajs, config.predictStepNums, config.actionDim}, "noise");

    rt::Tensor& traj = runner.sampleTrajectory(cond, noise, stream);

    int const concurrent = std::stoi(argOf(argc, argv, "--concurrent", "0"));
    if (concurrent > 1)
    {
        // Same total work as N separate processes, but in one process on N streams. Without
        // MPS, separate processes get separate CUDA contexts and the GPU time-slices between
        // them; streams inside one context are the only arrangement that can genuinely
        // overlap. This measures whether that distinction buys anything here.
        int const iters = std::stoi(argOf(argc, argv, "--iters", "10"));
        bool const prioritize = std::stoi(argOf(argc, argv, "--prioritize", "0")) != 0;
        std::vector<std::thread> workers;
        std::vector<double> perWorker(static_cast<size_t>(concurrent), 0.0);
        auto const start = std::chrono::steady_clock::now();
        for (int w = 0; w < concurrent; ++w)
        {
            workers.emplace_back([&, w] {
                cudaStream_t own{};
                if (prioritize && w == 0)
                {
                    // System 1 drives control: if its rate collapses under a competing load the
                    // trajectories arrive too late to steer with. A high-priority stream lets
                    // its kernels jump the queue at kernel boundaries, which is the only lever
                    // CUDA offers short of partitioning SMs.
                    int least = 0;
                    int greatest = 0;
                    cudaDeviceGetStreamPriorityRange(&least, &greatest);
                    cudaStreamCreateWithPriority(&own, cudaStreamNonBlocking, greatest);
                }
                else
                {
                    cudaStreamCreate(&own);
                }
                InternVLAN1System1Runner local(engineDir, config, own);
                rt::Tensor const c = toDevice(condHost, {2, condLen, 768}, "cond");
                rt::Tensor const n
                    = toDevice(noiseHost, {config.numSampleTrajs, config.predictStepNums, config.actionDim}, "noise");
                local.sampleTrajectory(c, n, own);
                cudaStreamSynchronize(own);
                auto const t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i)
                {
                    local.sampleTrajectory(c, n, own);
                }
                cudaStreamSynchronize(own);
                perWorker[static_cast<size_t>(w)]
                    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()
                    / static_cast<double>(iters);
                cudaStreamDestroy(own);
            });
        }
        for (auto& t : workers)
        {
            t.join();
        }
        double const wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (int w = 0; w < concurrent; ++w)
        {
            std::printf("  stream %d%s: %.2f ms/trajectory\n", w, (prioritize && w == 0) ? " (high priority)" : "",
                perWorker[static_cast<size_t>(w)]);
        }
        std::printf("in-process x%d: %.1f trajectories/s aggregate (wall %.0f ms)\n", concurrent,
            1000.0 * concurrent * iters / wall, wall);
        cudaStreamDestroy(stream);
        return 0;
    }

    int const iters = std::stoi(argOf(argc, argv, "--iters", "0"));
    if (iters > 0)
    {
        // Warm up separately: the first call pays for lazy kernel selection and would otherwise
        // land inside the average.
        for (int i = 0; i < 3; ++i)
        {
            runner.sampleTrajectory(cond, noise, stream);
        }
        cudaStreamSynchronize(stream);
        auto const start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            runner.sampleTrajectory(cond, noise, stream);
        }
        cudaStreamSynchronize(stream);
        auto const elapsed = std::chrono::steady_clock::now() - start;
        double const ms = std::chrono::duration<double, std::milli>(elapsed).count() / static_cast<double>(iters);
        std::printf("trajectory: %.2f ms  (%.1f Hz, %d steps x %d samples, mean of %d)\n", ms, 1000.0 / ms,
            config.numInferenceSteps, config.numSampleTrajs, iters);
    }

    std::vector<float> out(static_cast<size_t>(traj.getShape().volume()));
    cudaMemcpy(out.data(), traj.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream sink(outPath, std::ios::binary);
    sink.write(reinterpret_cast<char const*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    std::printf("wrote %s (%zu floats)\n", outPath.c_str(), out.size());
    cudaStreamDestroy(stream);
    return 0;
}
