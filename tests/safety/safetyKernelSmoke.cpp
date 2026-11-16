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
#include <fstream>

namespace
{
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::engine_test;

//! Explicitly released storage: no CUDA cleanup from destructors or failed initialization.
template <typename T>
struct Storage
{
    T* host{};
    T* device{};
    size_t count{};

    bool allocate(size_t elements)
    {
        count = elements;
        if (!check(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(T)), "cudaMallocHost")
            || !check(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T)), "cudaMalloc"))
        {
            return false;
        }
        std::fill_n(host, count, T{});
        return true;
    }

    bool copy(cudaStream_t stream, bool upload)
    {
        return check(cudaMemcpyAsync(upload ? device : host, upload ? host : device, count * sizeof(T),
                         upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync");
    }

    bool release()
    {
        return check(cudaFree(device), "cudaFree") && check(cudaFreeHost(host), "cudaFreeHost");
    }
};

//! Zero Q/K yields uniform attention; distinct V rows test ragged boundaries, causal masking and contiguous cache
//! addressing.
struct KernelCase
{
    fmha::PreparedBindings bindings{};
    bool visual{};
    AttentionLaunch shape{};
    Storage<half> q, k, v, output, cache;
    Storage<float> rope;
    Storage<int32_t> metadata;
    Storage<uint64_t> errors;
    Storage<char> workspace;

    bool initialize(bool isVisual, int32_t headSize)
    {
        visual = isVisual;
        shape.tokens = visual ? 5 : 130;
        shape.qHeads = 2;
        shape.kvHeads = visual ? 2 : 1;
        shape.headSize = headSize;
        shape.capacity = 259;
        shape.ropeTokens = visual ? 3 : 256;
        shape.rotaryDim = 32;
        shape.sequences = 2;
        shape.scale = 0.125F;
        size_t const elements = shape.tokens * shape.qHeads * headSize;
        size_t const packed = shape.tokens * (shape.qHeads + 2 * shape.kvHeads) * headSize;
        if (!q.allocate(visual ? elements : packed) || !k.allocate(elements) || !v.allocate(elements)
            || !output.allocate(elements) || !cache.allocate(2 * shape.capacity * shape.kvHeads * headSize)
            || !rope.allocate(shape.ropeTokens * shape.rotaryDim) || !metadata.allocate(visual ? 3 : 2)
            || !errors.allocate(1) || !workspace.allocate(attentionWorkspaceBytes(shape)))
        {
            return false;
        }
        if (visual)
        {
            metadata.host[0] = 0;
            metadata.host[1] = 2;
            metadata.host[2] = shape.tokens;
        }
        else
        {
            // [live query length, start position].
            int32_t const values[]{shape.tokens, 0};
            std::copy_n(values, metadata.count, metadata.host);
            for (int32_t row = 0; row < shape.ropeTokens; ++row)
            {
                for (int32_t d = 0; d < shape.rotaryDim / 2; ++d)
                {
                    rope.host[row * shape.rotaryDim + d] = std::cos(row * 0.01F);
                    rope.host[row * shape.rotaryDim + shape.rotaryDim / 2 + d] = std::sin(row * 0.01F);
                }
            }
        }
        for (int32_t t = 0; t < shape.tokens; ++t)
        {
            auto* values = visual
                ? v.host + t * shape.qHeads * headSize
                : q.host + (t * (shape.qHeads + 2 * shape.kvHeads) + shape.qHeads + shape.kvHeads) * headSize;
            std::fill_n(values, (visual ? shape.qHeads : shape.kvHeads) * headSize, __float2half(t * 0.01F));
        }
        return true;
    }

    bool execute(SafetyAttentionModules& modules, cudaStream_t stream)
    {
        if (!q.copy(stream, true) || !k.copy(stream, true) || !v.copy(stream, true) || !cache.copy(stream, true)
            || !rope.copy(stream, true) || !metadata.copy(stream, true) || !errors.copy(stream, true))
        {
            return false;
        }
        auto const status = visual
            ? launchSafetyVisualAttention(modules, bindings, shape, q.device, k.device, v.device, metadata.device,
                  output.device, workspace.device, errors.device, stream)
            : launchSafetyLlmAttention(modules, bindings, shape, q.device, cache.device, metadata.device, rope.device,
                  metadata.device + 1, output.device, cache.device, workspace.device, errors.device, stream);
        if (!check(status, "attention launch") || !output.copy(stream, false) || !cache.copy(stream, false)
            || !errors.copy(stream, false) || !check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            || errors.host[0] != 0)
        {
            std::fprintf(stderr, "Attention execution failed; async mask=%llu\n",
                static_cast<unsigned long long>(errors.host[0]));
            return false;
        }
        for (int32_t t = 0; t < shape.tokens; ++t)
        {
            int32_t const begin = visual && t >= 2 ? 2 : 0;
            int32_t const end = visual ? (t < 2 ? 2 : shape.tokens) : t + 1;
            double expected{};
            for (int32_t key = begin; key < end; ++key)
            {
                expected += __half2float(__float2half(key * 0.01F)) / static_cast<double>(end - begin);
            }
            for (int32_t i = 0; i < shape.qHeads * shape.headSize; ++i)
            {
                double const actual = __half2float(output.host[t * shape.qHeads * shape.headSize + i]);
                if (!std::isfinite(actual) || std::abs(actual - expected) > 0.002)
                {
                    std::fprintf(stderr, "Output mismatch: visual=%d D=%d token=%d element=%d actual=%g expected=%g\n",
                        visual, shape.headSize, t, i, actual, expected);
                    return false;
                }
            }
        }
        if (!visual)
        {
            for (int32_t token = 0; token < shape.capacity; ++token)
            {
                float const expected = token < shape.tokens ? __half2float(__float2half(token * 0.01F)) : 0.F;
                for (int32_t d = 0; d < shape.headSize; ++d)
                {
                    int32_t const kOffset = token * shape.headSize + d;
                    int32_t const vOffset = shape.capacity * shape.headSize + kOffset;
                    if (__half2float(cache.host[kOffset]) != 0.F || __half2float(cache.host[vOffset]) != expected)
                    {
                        std::fprintf(stderr, "KV mismatch: D=%d token=%d element=%d\n", shape.headSize, token, d);
                        return false;
                    }
                }
            }
        }
        std::printf("PASS %s D%d\n", visual ? "visual-ragged" : "llm-contiguous", shape.headSize);
        return true;
    }

    bool release()
    {
        return q.release() && k.release() && v.release() && output.release() && cache.release() && rope.release()
            && metadata.release() && errors.release() && workspace.release();
    }
};

bool smoke()
{
    TestLifecycle lifecycle;
    CUcontext context{};
    SafetyAttentionModules modules;
    cudaStream_t stream{};
    std::array<KernelCase, 4> cases{};
    if (!lifecycle.initialize() || !initializeModules(context, modules)
        || !check(modules.prepare(), "SafetyAttentionModules::prepare")
        || !check(prepareAttentionKernels(), "prepareAttentionKernels")
        || !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
    {
        return false;
    }
    for (size_t i = 0; i < cases.size(); ++i)
    {
        if (!cases[i].initialize(i < 2, i % 2 == 0 ? 64 : 128))
        {
            return false;
        }
    }
    // Both passes use the same resources. No allocator/module setup is reachable in the second pass.
    for (int32_t pass = 0; pass < 2; ++pass)
    {
        if (pass == 1 && (!prepareForOperational() || modules.seal() != CUDA_SUCCESS || !lifecycle.operational()))
        {
            return false;
        }
        for (auto& test : cases)
        {
            if (!test.execute(modules, stream))
            {
                return false;
            }
        }
    }
    if (!lifecycle.deinitialize())
    {
        return false;
    }
    for (auto& test : cases)
    {
        if (!test.release())
        {
            return false;
        }
    }
    return check(modules.release(), "SafetyAttentionModules::release")
        && check(cudaStreamDestroy(stream), "cudaStreamDestroy") && check(cuCtxDestroy(context), "cuCtxDestroy")
        && lifecycle.finish();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && (argc != 3 || std::strcmp(argv[2], "--external-lifecycle") != 0))
    {
        std::fprintf(stderr, "Usage: %s result.json [--external-lifecycle]\n", argv[0]);
        return finishTestProcess(2);
    }
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    if (argc != 3)
    {
        std::fprintf(stderr, "QNX Safety requires --external-lifecycle and an exclusive VM lifecycle coordinator.\n");
        return finishTestProcess(2);
    }
#endif
    std::ofstream report(argv[1], std::ios::trunc);
    if (!report)
    {
        return finishTestProcess(1);
    }
    report << "{\"success\":false}\n";
    report.flush();
    bool const success = smoke();
    if (success)
    {
        report.close();
        report.open(argv[1], std::ios::trunc);
        report
            << "{\"success\":true,\"real_safety_graph\":false,\"variants\":4,"
               "\"warmup_executions\":4,\"post_warmup_executions\":4,\"async_error_mask\":0,\"qnx_lifecycle_verified\":"
            << (TestLifecycle::kQNX_SAFETY ? "true" : "false") << "}\n";
    }
    report.close();
    return finishTestProcess(success && report ? 0 : 1);
}
