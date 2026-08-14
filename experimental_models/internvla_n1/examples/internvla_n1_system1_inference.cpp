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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
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

    std::vector<float> out(static_cast<size_t>(traj.getShape().volume()));
    cudaMemcpy(out.data(), traj.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream sink(outPath, std::ios::binary);
    sink.write(reinterpret_cast<char const*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    std::printf("wrote %s (%zu floats)\n", outPath.c_str(), out.size());
    cudaStreamDestroy(stream);
    return 0;
}
