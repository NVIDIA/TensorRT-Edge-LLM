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

#if defined(CUTE_DSL_SSD_SM12X_ENABLED)

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/cuteDslSSDRunner.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{
struct PackedSsdCudaResources
{
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};

    ~PackedSsdCudaResources()
    {
        if (executable != nullptr)
        {
            cudaGraphExecDestroy(executable);
        }
        if (graph != nullptr)
        {
            cudaGraphDestroy(graph);
        }
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }
};

using PackedSsdLengths = std::array<std::array<int32_t, 7>, 2>;

class PackedSsdRunnerTest : public testing::TestWithParam<std::tuple<std::array<int32_t, 2>, PackedSsdLengths>>
{
};

TEST_P(PackedSsdRunnerTest, RaggedStatePoolAndGraphReplay)
{
    // Given unequal packed lengths, empty sequences, and reordered nonzero state slots;
    // when the C++ runner executes and replays a CUDA Graph with refreshed metadata;
    // then output/final state match an independent FP32 recurrence and inactive slots remain bitwise unchanged.
    auto const& [shape, lengths] = GetParam();
    int32_t const dim = shape[0];
    int32_t const dstate = shape[1];
    int32_t const sm = getSMVersion();
    if (!CuteDslSSDRunner::canImplementPackedStatePool(dim, dstate, sm))
    {
        GTEST_SKIP() << "Packed SSD requires SM120/121 AOT modules";
    }
    constexpr int32_t heads{4}, groups{2}, slots{9}, batch{7};
    int32_t const tokens = std::accumulate(lengths[0].begin(), lengths[0].end(), int32_t{0});
    size_t const stateSize = static_cast<size_t>(slots) * heads * dim * dstate;
    std::mt19937 random(1280);
    std::normal_distribution<float> normal(0.F, 0.25F);
    auto randomHalf = [&](size_t count) {
        std::vector<half> values(count);
        for (auto& value : values)
        {
            value = __float2half(normal(random));
        }
        return values;
    };
    auto x = randomHalf(static_cast<size_t>(tokens) * heads * dim);
    auto dt = randomHalf(tokens * heads);
    auto b = randomHalf(tokens * groups * dstate);
    auto c = randomHalf(b.size());
    auto d = randomHalf(heads);
    auto initial = randomHalf(stateSize);
    std::vector<half> bias(heads, __float2half(-2.F));
    std::vector<float> a{-0.2F, -0.4F, -0.7F, -1.F};
    std::vector<int32_t> cu(batch + 1, 0);
    std::partial_sum(lengths[0].begin(), lengths[0].end(), cu.begin() + 1);
    std::vector<int32_t> indices{5, 1, 7, 0, 6, 4, 2};
    PackedSsdCudaResources resources;
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    auto const stream = resources.stream;
    std::vector<rt::Tensor> allocations;
    allocations.reserve(12);
    std::vector<rt::Tensor> hostAllocations;
    auto host = [&](void const* source, size_t bytes) {
        hostAllocations.emplace_back(
            rt::Coords{static_cast<int64_t>(bytes)}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8);
        void* result = hostAllocations.back().rawPointer();
        if (source != nullptr)
        {
            std::memcpy(result, source, bytes);
        }
        return result;
    };
    auto device = [&](void const* source, size_t bytes) {
        allocations.emplace_back(
            rt::Coords{static_cast<int64_t>(bytes)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
        void* result = allocations.back().rawPointer();
        if (source != nullptr)
        {
            CUDA_CHECK(cudaMemcpyAsync(result, host(source, bytes), bytes, cudaMemcpyHostToDevice, stream));
        }
        return result;
    };
    PackedSSDParams params{};
    params.x = device(x.data(), x.size() * sizeof(half));
    params.dt = device(dt.data(), dt.size() * sizeof(half));
    params.A = device(a.data(), a.size() * sizeof(float));
    params.B = device(b.data(), b.size() * sizeof(half));
    params.C = device(c.data(), c.size() * sizeof(half));
    params.D = device(d.data(), d.size() * sizeof(half));
    params.dtBias = device(bias.data(), bias.size() * sizeof(half));
    params.cuSeqlens = device(cu.data(), cu.size() * sizeof(int32_t));
    params.stateIndices = device(indices.data(), indices.size() * sizeof(int32_t));
    params.state = device(initial.data(), stateSize * sizeof(half));
    params.output = device(nullptr, x.size() * sizeof(half));
    params.workspaceSize = CuteDslSSDRunner::getPackedWorkspaceSize(tokens, heads);
    params.workspace = device(nullptr, params.workspaceSize);
    params.totalTokens = tokens;
    params.batch = batch;
    params.stateSlots = slots;
    params.nheads = heads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = groups;
    params.smVersion = sm;
    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.runPacked(params, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    int32_t const captureStatus = runner.runPacked(params, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &resources.graph));
    ASSERT_EQ(captureStatus, 0);
    CUDA_CHECK(instantiateCudaGraph(&resources.executable, resources.graph));

    for (int32_t replay = 0; replay < 2; ++replay)
    {
        SCOPED_TRACE(testing::Message() << "replay " << replay << ", total tokens " << tokens);
        std::partial_sum(lengths[replay].begin(), lengths[replay].end(), cu.begin() + 1);
        if (replay == 1)
        {
            std::rotate(indices.begin(), indices.begin() + 2, indices.end());
        }
        CUDA_CHECK(cudaMemcpyAsync(params.cuSeqlens, host(cu.data(), cu.size() * sizeof(int32_t)),
            cu.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(params.stateIndices, host(indices.data(), indices.size() * sizeof(int32_t)),
            indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(params.state, host(initial.data(), stateSize * sizeof(half)),
            stateSize * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(params.output, 0xFF, x.size() * sizeof(half), stream));
        CUDA_CHECK(cudaGraphLaunch(resources.executable, stream));
        auto* output = static_cast<half*>(host(nullptr, x.size() * sizeof(half)));
        auto* actual = static_cast<half*>(host(nullptr, stateSize * sizeof(half)));
        CUDA_CHECK(cudaMemcpyAsync(output, params.output, x.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(actual, params.state, stateSize * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> expectedState(stateSize), expectedOutput(x.size());
        std::transform(
            initial.begin(), initial.end(), expectedState.begin(), [](half value) { return __half2float(value); });
        std::array<bool, slots> touched{};
        for (int32_t seq = 0; seq < batch; ++seq)
        {
            int32_t const slot = indices[seq];
            touched[slot] = cu[seq] < cu[seq + 1];
            for (int32_t token = cu[seq]; token < cu[seq + 1]; ++token)
            {
                for (int32_t h = 0; h < heads; ++h)
                {
                    float const delta
                        = std::log1p(std::exp(__half2float(dt[token * heads + h]) + __half2float(bias[h])));
                    float const keep = std::exp(a[h] * delta);
                    for (int32_t channel = 0; channel < dim; ++channel)
                    {
                        size_t const xi = (static_cast<size_t>(token) * heads + h) * dim + channel;
                        float y = __half2float(d[h]) * __half2float(x[xi]);
                        for (int32_t n = 0; n < dstate; ++n)
                        {
                            size_t const si = ((static_cast<size_t>(slot) * heads + h) * dim + channel) * dstate + n;
                            size_t const bi = (static_cast<size_t>(token) * groups + h / (heads / groups)) * dstate + n;
                            expectedState[si]
                                = keep * expectedState[si] + delta * __half2float(x[xi]) * __half2float(b[bi]);
                            y += expectedState[si] * __half2float(c[bi]);
                        }
                        expectedOutput[xi] = y;
                    }
                }
            }
        }
        for (size_t i = 0; i < x.size(); ++i)
        {
            ASSERT_NEAR(__half2float(output[i]), expectedOutput[i], 0.004F + 0.005F * std::abs(expectedOutput[i]))
                << "output " << i;
        }
        for (size_t i = 0; i < stateSize; ++i)
        {
            if (touched[i / (heads * dim * dstate)])
            {
                ASSERT_NEAR(__half2float(actual[i]), expectedState[i], 0.002F + 0.005F * std::abs(expectedState[i]))
                    << "state " << i;
            }
            else
            {
                ASSERT_EQ(std::memcmp(&actual[i], &initial[i], sizeof(half)), 0) << "inactive state " << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Sm12x, PackedSsdRunnerTest,
    testing::Combine(
        testing::Values(std::array<int32_t, 2>{64, 64}, std::array<int32_t, 2>{64, 128}, std::array<int32_t, 2>{80, 64},
            std::array<int32_t, 2>{80, 128}, std::array<int32_t, 2>{128, 64}, std::array<int32_t, 2>{128, 128}),
        testing::Values(PackedSsdLengths{{{0, 1, 63, 64, 129, 0, 131}, {64, 0, 1, 127, 129, 67, 0}}},
            PackedSsdLengths{{{0, 63, 64, 65, 511, 512, 513}, {513, 0, 511, 64, 63, 65, 512}}},
            PackedSsdLengths{{{0, 129, 1025, 4097, 8193, 0, 32769}, {32769, 0, 129, 8193, 0, 1025, 4097}}})));
} // namespace
#endif // CUTE_DSL_SSD_SM12X_ENABLED
