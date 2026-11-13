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

#include "safetyEngineTestUtils.h"
#include "safetyTestLifecycle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
using namespace nvinfer1;
using namespace nvinfer2::safe;
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::engine_test;

Recorder sGraphRecorder;

struct Binding
{
    char const* name;
    DataType type;
    int32_t rank;
    std::array<int64_t, 3> dimensions;
    size_t bytes;
    void* host{};
    void* device{};
    bool seen{false};
};

bool bindGraph(ITRTGraph& graph, std::array<Binding, 6>& bindings)
{
    int64_t count{};
    if (!check(graph.getNbIOTensors(count), "getNbIOTensors") || count < 5 || count > 6)
    {
        return false;
    }
    for (int64_t index = 0; index < count; ++index)
    {
        AsciiChar const* name{};
        TensorDescriptor descriptor{};
        if (!check(graph.getIOTensorName(name, index), "getIOTensorName") || name == nullptr
            || !check(graph.getIOTensorDescriptor(descriptor, static_cast<int32_t>(index)), "getIOTensorDescriptor"))
        {
            return false;
        }
        Binding* binding{};
        for (auto& candidate : bindings)
        {
            if (std::strcmp(candidate.name, name) == 0)
            {
                binding = &candidate;
                break;
            }
        }
        if (binding == nullptr || binding->seen || descriptor.dataType != binding->type
            || descriptor.shape.nbDims != binding->rank)
        {
            return false;
        }
        binding->seen = true;
        int64_t stride = 1;
        for (int32_t d = binding->rank - 1; d >= 0; --d)
        {
            if (descriptor.shape.d[d] != binding->dimensions[d])
            {
                return false;
            }
            if (descriptor.sizeInBytes != 0 && descriptor.stride.d[d] != stride)
            {
                return false;
            }
            stride *= binding->dimensions[d];
        }
        bool const shapeOnly = std::strcmp(name, "shape_carrier") == 0;
        if (shapeOnly && descriptor.sizeInBytes == 0)
        {
            continue;
        }
        if ((descriptor.memPlacement != MemoryPlacement::kGPU && descriptor.memPlacement != MemoryPlacement::kNONE)
            || descriptor.sizeInBytes != binding->bytes || descriptor.vectorizedDim != -1)
        {
            return false;
        }
        if (!check(cudaMallocHost(&binding->host, binding->bytes), "cudaMallocHost(IO)")
            || !check(cudaMalloc(&binding->device, binding->bytes), "cudaMalloc(IO)"))
        {
            return false;
        }
        TypedArray const array = binding->type == DataType::kHALF
            ? TypedArray(static_cast<half_t*>(binding->device), binding->bytes)
            : TypedArray(static_cast<int32_t*>(binding->device), binding->bytes);
        if (!check(graph.setIOTensorAddress(name, array), "setIOTensorAddress"))
        {
            return false;
        }
    }
    for (auto const& binding : bindings)
    {
        if (std::strcmp(binding.name, "shape_carrier") != 0 && !binding.seen)
        {
            return false;
        }
    }
    return true;
}

//! Reject graph synchronization and pinned error-buffer failures before reading attention output.
bool executeAndCompare(ITRTGraph& graph, std::array<Binding, 6>& bindings, cudaStream_t stream,
    RuntimeErrorInformation const* errorBuffer, int32_t iteration, double& maximumError)
{
    for (int32_t which = 0; which < 3; ++which)
    {
        auto* values = static_cast<half*>(bindings[which].host);
        for (int32_t i = 0; i < kELEMENTS; ++i)
        {
            values[i] = __float2half(std::sin(i * (0.031F + which * 0.017F) + iteration * 0.2F));
        }
    }
    int32_t const ends[]{0, 2, kTOKENS};
    std::memcpy(bindings[3].host, ends, sizeof(ends));
    if (bindings[4].host != nullptr)
    {
        std::memset(bindings[4].host, 0, bindings[4].bytes);
    }
    for (size_t index = 0; index < bindings.size(); ++index)
    {
        auto const& binding = bindings[index];
        if (index == 5 || binding.host == nullptr)
        {
            continue;
        }
        if (!check(cudaMemcpyAsync(binding.device, binding.host, binding.bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(input)"))
        {
            return false;
        }
    }
    if (!check(graph.executeAsync(stream), "executeAsync") || !check(graph.sync(), "sync"))
    {
        return false;
    }
    if (errorBuffer->bitMask != 0 || sGraphRecorder.getNbErrors() != 0
        || processRegistrationRecorder().getNbErrors() != 0)
    {
        std::fprintf(stderr, "Graph error mask: %llu\n", static_cast<unsigned long long>(errorBuffer->bitMask));
        return false;
    }
    if (!check(cudaMemcpyAsync(bindings[5].host, bindings[5].device, bindings[5].bytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(output)")
        || !check(cudaStreamSynchronize(stream), "cudaStreamSynchronize"))
    {
        return false;
    }
    auto const* q = static_cast<half const*>(bindings[0].host);
    auto const* k = static_cast<half const*>(bindings[1].host);
    auto const* v = static_cast<half const*>(bindings[2].host);
    auto const* output = static_cast<half const*>(bindings[5].host);
    for (int32_t token = 0; token < kTOKENS; ++token)
    {
        int32_t const begin = token < 2 ? 0 : 2;
        int32_t const end = token < 2 ? 2 : kTOKENS;
        for (int32_t head = 0; head < kHEADS; ++head)
        {
            std::array<double, kTOKENS> weights{};
            double sum{};
            for (int32_t key = begin; key < end; ++key)
            {
                double score{};
                for (int32_t d = 0; d < kHEAD_SIZE; ++d)
                {
                    score += __half2float(q[(token * kHEADS + head) * kHEAD_SIZE + d])
                        * __half2float(k[(key * kHEADS + head) * kHEAD_SIZE + d]);
                }
                weights[key] = std::exp(score * kSCALE);
                sum += weights[key];
            }
            for (int32_t d = 0; d < kHEAD_SIZE; ++d)
            {
                double expected{};
                for (int32_t key = begin; key < end; ++key)
                {
                    expected += weights[key] * __half2float(v[(key * kHEADS + head) * kHEAD_SIZE + d]);
                }
                double const actual = __half2float(output[(token * kHEADS + head) * kHEAD_SIZE + d]);
                double const difference = std::abs(actual - expected / sum);
                if (!std::isfinite(actual) || difference > 0.003)
                {
                    std::fprintf(stderr, "Attention mismatch at iteration %d token %d head %d dim %d: error %.9g\n",
                        iteration, token, head, d, difference);
                    return false;
                }
                maximumError = std::max(maximumError, difference);
            }
        }
    }
    std::printf("Real Safety graph %s passed, max absolute error %.9g\n", iteration == 0 ? "warmup" : "post-warmup run",
        maximumError);
    return true;
}

bool smoke(std::vector<char> const& engine, char const* companionSoPath, double& maximumError)
{
    TestLifecycle lifecycle;
    CUcontext context{};
    SafetyAttentionModules modules;
    if (!lifecycle.initialize() || !initializeModules(context, modules))
    {
        return false;
    }
    ITRTGraph* graph{};
    if (!check(createSafetyGraph(modules, graph, engine.data(), engine.size(), sGraphRecorder, companionSoPath),
            "createSafetyGraph")
        || graph == nullptr)
    {
        return false;
    }
    int64_t profiles{};
    int32_t auxiliaryStreams{};
    if (!check(graph->getNbIOProfiles(profiles), "getNbIOProfiles") || profiles != 1
        || !check(graph->setIOProfile(0), "setIOProfile")
        || !check(graph->getNbAuxStreams(auxiliaryStreams), "getNbAuxStreams") || auxiliaryStreams != 0)
    {
        return false;
    }
    std::array<Binding, 6> bindings{{{"q", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, kELEMENTS * sizeof(half)},
        {"k", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, kELEMENTS * sizeof(half)},
        {"v", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, kELEMENTS * sizeof(half)},
        {"sequence_ends", DataType::kINT32, 1, {3, 0, 0}, 3 * sizeof(int32_t)},
        {"shape_carrier", DataType::kINT32, 1, {3, 0, 0}, 3 * sizeof(int32_t)},
        {"output", DataType::kHALF, 3, {kTOKENS, kHEADS, kHEAD_SIZE}, kELEMENTS * sizeof(half)}}};
    if (!bindGraph(*graph, bindings))
    {
        return false;
    }
    cudaStream_t stream{};
    // Borrowed pinned host storage remains valid until graph destruction; graph sync completes its device writes.
    RuntimeErrorInformation* errorBuffer{};
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")
        || !check(graph->getErrorBuffer(errorBuffer), "getErrorBuffer") || errorBuffer == nullptr)
    {
        return false;
    }
    // Any failure is terminal: process exit owns recovery, not graph/module retry or partial teardown.
    for (int32_t iteration = 0; iteration < 2; ++iteration)
    {
        if ((iteration == 1 && (!prepareForOperational() || modules.seal() != CUDA_SUCCESS || !lifecycle.operational()))
            || !executeAndCompare(*graph, bindings, stream, errorBuffer, iteration, maximumError))
        {
            return false;
        }
    }
    if (!lifecycle.deinitialize())
    {
        return false;
    }
    if (!check(destroyTRTGraph(graph), "destroyTRTGraph"))
    {
        return false;
    }
    if (sGraphRecorder.getNbErrors() != 0 || processRegistrationRecorder().getNbErrors() != 0)
    {
        return false;
    }
    if (!check(modules.release(), "SafetyAttentionModules::release"))
    {
        return false;
    }
    for (auto& binding : bindings)
    {
        if (binding.device != nullptr && !check(cudaFree(binding.device), "cudaFree(IO)"))
        {
            return false;
        }
        if (binding.host != nullptr && !check(cudaFreeHost(binding.host), "cudaFreeHost(IO)"))
        {
            return false;
        }
    }
    return check(cudaStreamDestroy(stream), "cudaStreamDestroy") && check(cuCtxDestroy(context), "cuCtxDestroy")
        && lifecycle.finish();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 && (argc != 4 || std::strcmp(argv[3], "--external-lifecycle") != 0))
    {
        std::fprintf(stderr,
            "Usage: %s matching-safety-engine.plan result.json [--external-lifecycle]\n"
            "Requires real matching libnvinfer_safe and admitted target kernel artifacts. No mocks or VLM export.\n",
            argv[0]);
        return finishTestProcess(2);
    }
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    if (argc != 4)
    {
        std::fprintf(stderr, "QNX Safety requires --external-lifecycle and an exclusive VM lifecycle coordinator.\n");
        return finishTestProcess(2);
    }
#endif
    try
    {
        std::ofstream initialReport(argv[2], std::ios::trunc);
        initialReport << "{\"success\":false}\n";
        initialReport.close();
        if (!initialReport)
        {
            return finishTestProcess(1);
        }
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        auto const bytes = input.tellg();
        if (!input || bytes <= 0 || bytes > (64LL << 20))
        {
            return finishTestProcess(1);
        }
        std::vector<char> engine(static_cast<size_t>(bytes));
        input.seekg(0);
        if (!input.read(engine.data(), bytes))
        {
            return finishTestProcess(1);
        }
        double maximumError{};
        std::string const companionPath = std::filesystem::absolute(std::string(argv[1]) + ".so").string();
        if (!smoke(engine, companionPath.c_str(), maximumError))
        {
            return finishTestProcess(1);
        }
        std::ofstream report(argv[2], std::ios::trunc);
        report << "{\"success\":true,\"real_safety_graph\":true,\"plugin\":\"trt_edgellm::ViTAttentionPlugin:1\","
                  "\"warmup_executions\":1,\"post_warmup_executions\":1,\"async_error_mask\":0,\"max_abs_error\":"
               << maximumError << ",\"model_export_validated\":false,\"trt_header_version\":\"" << NV_TENSORRT_MAJOR
               << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
               << "\",\"qnx_lifecycle_verified\":" << (TestLifecycle::kQNX_SAFETY ? "true" : "false") << "}\n";
        report.close();
        return finishTestProcess(report ? 0 : 1);
    }
    catch (std::exception const& error)
    {
        std::fprintf(stderr, "Safety engine smoke failed: %s\n", error.what());
        return finishTestProcess(1);
    }
}
