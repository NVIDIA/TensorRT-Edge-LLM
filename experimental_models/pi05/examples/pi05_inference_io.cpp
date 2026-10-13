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

#include "pi05_inference_io.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

namespace trt_edgellm
{
namespace pi05
{

namespace
{

bool endsWith(std::string const& text, std::string const& suffix)
{
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//! Naming the suffix keeps the diagnostic on the mistake rather than on the byte count
//! a safetensors header happens to produce.
void requireRawDump(std::string const& path)
{
    ELLM_CHECK(!endsWith(path, ".safetensors"),
        path + " is a safetensors file; a canonical input is a raw dump of the tensor's bytes");
}

//! Byte count of the raw dump at \p path.
size_t payloadBytes(std::string const& path)
{
    requireRawDump(path);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    ELLM_CHECK(file.is_open(), "Failed to open " + path);
    return static_cast<size_t>(file.tellg());
}

//! Fill \p bytes bytes at \p out from the raw dump at \p path.
void readTensorFile(std::string const& path, void* out, size_t bytes)
{
    requireRawDump(path);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    ELLM_CHECK(file.is_open(), "Failed to open " + path);
    ELLM_CHECK(static_cast<size_t>(file.tellg()) == bytes,
        path + " is not a raw dump of the shape the engines were built for");
    file.seekg(0);
    file.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
}

//! One [horizon, dim] slice per timestep, the layout every harness reads back.
nlohmann::json actionRows(std::vector<float> const& flat, int32_t horizon, int32_t dim)
{
    nlohmann::json rows = nlohmann::json::array();
    for (int32_t t = 0; t < horizon; ++t)
    {
        auto const first = flat.begin() + static_cast<ptrdiff_t>(t) * dim;
        rows.push_back(nlohmann::json(std::vector<float>(first, first + dim)));
    }
    return rows;
}

} // namespace

bool isSafetensorsPath(std::string const& path)
{
    return endsWith(path, ".safetensors");
}

void logLatency(std::vector<Pi05Timings> const& rounds, int32_t warmup)
{
    auto report = [&rounds](char const* name, double (*pick)(Pi05Timings const&)) {
        std::vector<double> values;
        values.reserve(rounds.size());
        std::transform(rounds.begin(), rounds.end(), std::back_inserter(values), pick);
        std::sort(values.begin(), values.end());
        double const mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        LOG_INFO("  %-18s mean %8.2f ms   median %8.2f ms   min %8.2f ms", name, mean, values[values.size() / 2],
            values.front());
    };
    LOG_INFO("pi0.5 latency over %zu iterations (%d warmup):", rounds.size(), warmup);
    report("visual", [](Pi05Timings const& t) { return static_cast<double>(t.stages.visualMs); });
    report("assemble", [](Pi05Timings const& t) { return static_cast<double>(t.stages.assembleMs); });
    report("prefix", [](Pi05Timings const& t) { return static_cast<double>(t.stages.prefixMs); });
    report("action (N steps)", [](Pi05Timings const& t) { return static_cast<double>(t.stages.actionMs); });
    report("engine chain", [](Pi05Timings const& t) { return t.engineMs; });
    // Host work outside the engine chain, and zero in canonical tensor mode. Only the
    // decode is an artifact of taking file paths.
    report("obs decode", [](Pi05Timings const& t) { return t.observation.decodeMs; });
    report("obs resize", [](Pi05Timings const& t) { return t.observation.resizeMs; });
    report("obs rest", [](Pi05Timings const& t) { return t.observation.restMs; });
    report("observation", [](Pi05Timings const& t) { return t.observation.totalMs; });
    report("policy end-to-end", [](Pi05Timings const& t) { return t.policyMs; });
}

Pi05Observation readObservation(std::string const& path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("pi0.5 cannot open request file " + path);
    }
    nlohmann::json request;
    try
    {
        file >> request;
    }
    catch (nlohmann::json::exception const& e)
    {
        throw std::runtime_error("pi0.5 request " + path + " is not valid JSON: " + e.what());
    }
    for (char const* key : {"task", "cameras"})
    {
        if (!request.contains(key))
        {
            throw std::runtime_error("pi0.5 request " + path + " has no \"" + key + "\"");
        }
    }
    Pi05Observation observation;
    observation.task = request.at("task").get<std::string>();
    // A DROID client holds the two halves openpi concatenates rather than the joined
    // vector, so either spelling is read -- but a request carrying both has said the
    // state twice and there is no reason to prefer one.
    bool const hasJoint = request.contains("joint_position");
    bool const hasGripper = request.contains("gripper_position");
    bool const split = hasJoint && hasGripper;
    if (request.contains("state") == (hasJoint || hasGripper))
    {
        throw std::runtime_error("pi0.5 request " + path
            + " must carry either \"state\" or both \"joint_position\" and \"gripper_position\"");
    }
    if (hasJoint != hasGripper)
    {
        throw std::runtime_error("pi0.5 request " + path + " carries only \""
            + (hasJoint ? "joint_position" : "gripper_position") + "\"; the split spelling needs both");
    }
    if (split)
    {
        observation.state = request.at("joint_position").get<std::vector<float>>();
        std::vector<float> const gripper = request.at("gripper_position").get<std::vector<float>>();
        observation.state.insert(observation.state.end(), gripper.begin(), gripper.end());
    }
    else
    {
        observation.state = request.at("state").get<std::vector<float>>();
    }
    // Object rather than array: the contract places a view by name, so a request that
    // names its slots cannot be reordered into a different one.
    for (auto const& [name, imagePath] : request.at("cameras").items())
    {
        observation.cameras.push_back(Pi05CameraView{name, imagePath.get<std::string>()});
    }
    return observation;
}

std::vector<int32_t> readTokenIds(std::string const& spec)
{
    if (spec.empty())
    {
        // The five ids the openpi reference generator emits for its own canonical fixture, so a
        // --tokens-less run reproduces the numerical harness rather than an arbitrary prompt.
        return {2, 15, 27, 108, 4};
    }
    std::vector<int32_t> ids;
    std::stringstream parts(spec);
    for (std::string item; std::getline(parts, item, ',');)
    {
        if (item.empty())
        {
            continue;
        }
        ELLM_CHECK(item.find_first_not_of("0123456789") == std::string::npos,
            "pi0.5 token ids are a comma-separated list; '" + item + "' is not one");
        ids.push_back(std::stoi(item));
    }
    return ids;
}

rt::Tensor readPixelValues(
    std::string const& path, std::vector<int64_t> const& shape, int32_t seed, cudaStream_t stream)
{
    rt::Tensor device(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "pi05::pixelValues");
    rt::Tensor host(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF, "pi05::pixelValuesHost");
    auto const bytes = static_cast<size_t>(host.getMemoryCapacity());
    if (path.empty())
    {
        std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::generate_n(host.dataPointer<__half>(), host.getShape().volume(), [&] { return __float2half(dist(gen)); });
    }
    else
    {
        readTensorFile(path, host.rawPointer(), bytes);
        LOG_INFO("Loaded pixel values from %s", path.c_str());
    }
    CUDA_CHECK(cudaMemcpyAsync(device.rawPointer(), host.rawPointer(), bytes, cudaMemcpyHostToDevice, stream));
    // host is pinned and dies here, so the upload has to land before it does.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return device;
}

std::vector<float> readNoise(std::string const& path, size_t chunkElems, int32_t batch)
{
    size_t const supplied = payloadBytes(path) / sizeof(float);
    ELLM_CHECK(supplied == chunkElems || supplied == chunkElems * static_cast<size_t>(batch),
        path + " must hold one request's x_0 or the whole batch's");
    std::vector<float> noise(chunkElems * static_cast<size_t>(batch));
    readTensorFile(path, noise.data(), supplied * sizeof(float));
    for (size_t b = 1; supplied == chunkElems && b < static_cast<size_t>(batch); ++b)
    {
        std::copy_n(noise.begin(), chunkElems, noise.begin() + static_cast<ptrdiff_t>(b * chunkElems));
    }
    LOG_INFO("Using external initial noise: %zu floats", noise.size());
    return noise;
}

void writeActionChunk(std::string const& path, Pi05ActionChunk const& chunk, int32_t robotDim, cudaStream_t stream)
{
    if (isSafetensorsPath(path))
    {
        rt::Tensor raw({chunk.batch, chunk.horizon, chunk.modelActionDim}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kFLOAT, "actions");
        std::memcpy(raw.rawPointer(), chunk.normalizedActions.data(), chunk.normalizedActions.size() * sizeof(float));
        std::vector<rt::Tensor> tensors;
        tensors.push_back(std::move(raw));
        ELLM_CHECK(rt::safetensors::saveSafetensors(path, tensors, stream), "Failed to write " + path);
        LOG_INFO("Wrote the normalized chunk (%d,%d,%d) to %s", chunk.batch, chunk.horizon, chunk.modelActionDim,
            path.c_str());
        return;
    }

    // The JSON response records one request: both arrays are [horizon, dim], and the
    // robot-unit chunk exists for batch entry 0 alone. Use .safetensors for a batch.
    ELLM_CHECK(
        chunk.batch == 1, "pi0.5 JSON output holds one request's chunk; write a batched run to a .safetensors path");

    // "actions" stays the raw normalized, zero-padded chunk the engine emits;
    // "robot_actions" is the same chunk in robot units, sliced to the embodiment.
    nlohmann::json doc;
    doc["action_horizon"] = chunk.horizon;
    doc["action_dim"] = chunk.modelActionDim;
    doc["batch_size"] = chunk.batch;
    doc["normalized"] = true;
    doc["actions"] = actionRows(chunk.normalizedActions, chunk.horizon, chunk.modelActionDim);
    if (!chunk.prompt.empty())
    {
        // The request record, so an action chunk can be reproduced without
        // re-deriving the prompt.
        doc["prompt"] = chunk.prompt;
        doc["token_ids"] = chunk.tokenIds;
    }
    if (!chunk.robotActions.empty())
    {
        // Absent from a canonical-tensor run: converting to robot units needs the
        // request's own state, which those tensors do not carry.
        doc["robot_action_dim"] = robotDim;
        doc["robot_actions"] = actionRows(chunk.robotActions, chunk.horizon, robotDim);
    }
    if (path.empty())
    {
        std::cout << doc.dump(2) << "\n";
        return;
    }
    std::ofstream out(path);
    ELLM_CHECK(out.is_open(), "pi0.5 cannot open " + path + " for writing");
    out << doc.dump(2) << "\n";
    out.close();
    // Checked after close: a short write on a full disk surfaces only when the stream flushes,
    // and a silently truncated response reads as a valid but wrong action chunk.
    ELLM_CHECK(out.good(), "pi0.5 failed to write " + path);
    LOG_INFO("Wrote %s", path.c_str());
}

} // namespace pi05
} // namespace trt_edgellm
