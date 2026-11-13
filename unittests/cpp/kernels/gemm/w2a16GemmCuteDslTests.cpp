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

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/gemm/w2a16_cutedsl/cuteDslW2A16GemmRunner.h"
#include "kernels/weightsTransform/iq2_xxs/iq2XxsRepack.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
namespace
{
using rt::DeviceType;
using rt::Tensor;

struct Graph
{
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Graph()
    {
        if (exec)
        {
            cudaGraphExecDestroy(exec);
        }
        if (graph)
        {
            cudaGraphDestroy(graph);
        }
    }
};
std::ifstream openFixture(std::filesystem::path const& path, int32_t n, int32_t k, int32_t m)
{
    std::ifstream file(path, std::ios::binary);
    std::array<uint32_t, 4> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file
        || header
            != std::array<uint32_t, 4>{
                0x57324431, static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(m)})
    {
        throw std::runtime_error("Missing or incompatible dense fixture: " + path.string());
    }
    return file;
}
void readBytes(std::ifstream& file, void* dst, size_t bytes)
{
    file.read(static_cast<char*>(dst), bytes);
    if (!file)
    {
        throw std::runtime_error("Truncated dense fixture");
    }
}
void upload(std::ifstream& file, Tensor& dst, cudaStream_t stream)
{
    Tensor host({dst.getMemoryCapacity()}, DeviceType::kCPU, DataType::kINT8);
    readBytes(file, host.rawPointer(), dst.getMemoryCapacity());
    CUDA_CHECK(
        cudaMemcpyAsync(dst.rawPointer(), host.rawPointer(), dst.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
struct Case
{
    int32_t projection, n, k, m;
    bool boundary;
};
std::vector<Case> cases()
{
    std::vector<Case> result;
    std::array<std::array<int32_t, 2>, 13> const shapes{{{4096, 6144}, {6144, 4096}, {8, 256}, {136, 768}, {128, 4096},
        {256, 6144}, {1024, 6144}, {136, 4096}, {520, 6144}, {1048, 6144}, {512, 6144}, {1536, 6144}, {1480, 6144}}};
    for (int32_t i = 0; i < 13; ++i)
    {
        for (int32_t m : {1, 2, 3, 4, 8, 9, 16, 17, 128, 512, 2048, 8192})
        {
            if (i < 2 || (m != 512 && m != 2048))
            {
                result.push_back({i < 2 ? i : i - 2, shapes[i][0], shapes[i][1], m, i >= 2});
            }
        }
        if (i < 2)
        {
            for (int32_t m = 4096; m <= 32768; m += 2048)
            {
                if (m != 8192)
                {
                    result.push_back({i, shapes[i][0], shapes[i][1], m, false});
                }
            }
        }
    }
    return result;
}
class W2A16GemmFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }
    void TearDown() override
    {
        if (mStream)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
    }
    cudaStream_t mStream{};
};

class DenseGemmTest : public W2A16GemmFixture, public ::testing::WithParamInterface<Case>
{
protected:
    void SetUp() override
    {
        int device{};
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        if (!CuteDslW2A16GemmRunner::canImplement(
                1, 128, 256, props.major * 10 + props.minor, props.multiProcessorCount))
        {
            GTEST_SKIP() << "Requires SM110, SM120 or GB10";
        }
        auto const* root = std::getenv("EDGELLM_W2A16_GEMM_FIXTURES");
        if (!root)
        {
            GTEST_SKIP() << "Set EDGELLM_W2A16_GEMM_FIXTURES to provisioned fixtures";
        }
        mRoot = root;
        W2A16GemmFixture::SetUp();
    }
    std::filesystem::path mRoot;
};

TEST_P(DenseGemmTest, ReferenceAndChangedInputReplay)
{
    // Given real IQ2_XXS blocks, BF16 inputs and caller-owned output/workspace,
    // when dense GEMM runs repeatedly and a captured graph replays with changed input,
    // then every output is finite, has relative L2 <= 0.01 and cosine >= 0.999 against the independent reference.
    auto const c = GetParam();
    auto const dir = mRoot / (c.boundary ? "fixtures-boundary-full" : "fixtures-primary")
        / ("projection" + std::to_string(c.projection));
    Tensor values({int64_t{c.n} * c.k / 4}, DeviceType::kGPU, DataType::kINT8);
    Tensor bases({int64_t{c.n} * c.k / 256}, DeviceType::kGPU, DataType::kHALF);
    Tensor lookup({256, 8}, DeviceType::kGPU, DataType::kBF16);
    Tensor input({c.m, c.k}, DeviceType::kGPU, DataType::kBF16);
    Tensor output({c.m, c.n}, DeviceType::kGPU, DataType::kBF16);
    Tensor workspace;
    auto const workspaceBytes = CuteDslW2A16GemmRunner::getWorkspaceSize(c.m, c.n, c.k);
    if (workspaceBytes > 0)
    {
        workspace = Tensor({static_cast<int64_t>(workspaceBytes)}, DeviceType::kGPU, DataType::kINT8);
    }
    auto weights = openFixture(dir / "weights.bin", c.n, c.k, 0);
    for (auto* b : {&values, &bases, &lookup})
    {
        upload(weights, *b, mStream);
    }
    auto inputs = openFixture(dir / ("m" + std::to_string(c.m) + ".bin"), c.n, c.k, c.m);
    CuteDslW2A16GemmParams p{c.m, c.n, c.k, input.rawPointer(), values.rawPointer(), bases.rawPointer(),
        lookup.rawPointer(), output.rawPointer()};
    ASSERT_TRUE(CuteDslW2A16GemmRunner::ensureKernelModules(p, mStream));
    CuteDslW2A16GemmRunner runner;
    Graph graph;
    for (int variant = 0; variant < 2; ++variant)
    {
        upload(inputs, input, mStream);
        Tensor expected({c.m, c.n}, DeviceType::kCPU, DataType::kBF16);
        readBytes(inputs, expected.rawPointer(), output.getMemoryCapacity());
        for (int replay = 0; replay < 3; ++replay)
        {
            CUDA_CHECK(cudaMemsetAsync(output.rawPointer(), 0xff, output.getMemoryCapacity(), mStream));
            if (workspace.getMemoryCapacity())
            {
                CUDA_CHECK(cudaMemsetAsync(workspace.rawPointer(), 0xff, workspace.getMemoryCapacity(), mStream));
            }
            if (variant == 0)
            {
                ASSERT_EQ(runner.run(p, workspace.rawPointer(), workspace.getMemoryCapacity(), mStream), cudaSuccess);
            }
            else
            {
                CUDA_CHECK(cudaGraphLaunch(graph.exec, mStream));
            }
            Tensor actual({c.m, c.n}, DeviceType::kCPU, DataType::kBF16);
            CUDA_CHECK(cudaMemcpyAsync(
                actual.rawPointer(), output.rawPointer(), output.getMemoryCapacity(), cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            auto const* a = actual.dataPointer<__nv_bfloat16>();
            auto const* reference = expected.dataPointer<__nv_bfloat16>();
            double diff = 0, aa = 0, bb = 0, ab = 0;
            bool finite = true;
            for (size_t j = 0; j < static_cast<size_t>(c.m) * c.n; ++j)
            {
                double const x = __bfloat162float(a[j]), y = __bfloat162float(reference[j]);
                finite = finite && std::isfinite(x);
                diff += (x - y) * (x - y);
                aa += x * x;
                bb += y * y;
                ab += x * y;
            }
            double const rel = std::sqrt(diff / bb), cos = ab / std::sqrt(aa * bb);
            RecordProperty("relative_l2", std::to_string(rel));
            RecordProperty("cosine", std::to_string(cos));
            EXPECT_TRUE(finite);
            EXPECT_LE(rel, 0.01) << "input=" << variant << " repeat=" << replay;
            EXPECT_GE(cos, 0.999) << "input=" << variant << " repeat=" << replay;
        }
        if (variant == 0)
        {
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            auto const status = runner.run(p, workspace.rawPointer(), workspace.getMemoryCapacity(), mStream);
            CUDA_CHECK(cudaStreamEndCapture(mStream, &graph.graph));
            ASSERT_EQ(status, cudaSuccess);
            CUDA_CHECK(instantiateCudaGraph(&graph.exec, graph.graph));
        }
    }
}
INSTANTIATE_TEST_SUITE_P(Checkpoint, DenseGemmTest, ::testing::ValuesIn(cases()), [](auto const& info) {
    auto const c = info.param;
    return "N" + std::to_string(c.n) + "K" + std::to_string(c.k) + "M" + std::to_string(c.m);
});

TEST(W2A16GemmContract, InvalidShapes)
{
    // Given a shape or GPU outside the supported dense contract,
    // when admission is queried, then it is rejected without accessing device memory.
    for (auto const shape :
        std::vector<std::array<int32_t, 5>>{{0, 128, 256, 121, 48}, {32769, 128, 256, 121, 48}, {1, 0, 256, 121, 48},
            {1, 127, 256, 121, 48}, {1, 128, 255, 121, 48}, {1, 128, 0, 121, 48}, {1, 128, 256, 89, 48},
            {1, 128, 256, 110, 0}, {1, 128, 256, 120, 0}, {1, 128, 256, 121, 20}, {1, 4194304, 256, 121, 48}})
    {
        EXPECT_FALSE(CuteDslW2A16GemmRunner::canImplement(shape[0], shape[1], shape[2], shape[3], shape[4]));
    }
}
TEST_F(W2A16GemmFixture, TallTilesMatchSixteenRowTiles)
{
    // Given random IQ2_XXS payloads and BF16 inputs, when M selects a 64- or 128-row tile,
    // then every output matches running the same rows as independent 16-row GEMMs.
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (!CuteDslW2A16GemmRunner::canImplement(16, 128, 256, props.major * 10 + props.minor, props.multiProcessorCount))
    {
        GTEST_SKIP() << "Requires SM110, SM120 or GB10";
    }
    CuteDslW2A16GemmRunner runner;
    for (auto const [m, n, k] : std::array<std::array<int32_t, 3>, 5>{
             {{96, 1024, 4096}, {80, 136, 768}, {1040, 1024, 1024}, {1100, 5376, 512}, {1030, 136, 512}}})
    {
        std::vector<uint8_t> values(static_cast<size_t>(n) * k / 4);
        std::vector<__half> bases(static_cast<size_t>(k) / 256 * n);
        std::vector<__nv_bfloat16> lookup(256 * 8);
        std::vector<__nv_bfloat16> input(static_cast<size_t>(m) * k);
        uint32_t seed = static_cast<uint32_t>(m * 977 + n * 31 + k);
        auto next = [&seed]() {
            seed = seed * 1664525U + 1013904223U;
            return seed >> 8;
        };
        for (auto& v : values)
        {
            v = static_cast<uint8_t>(next());
        }
        for (auto& b : bases)
        {
            b = __float2half(static_cast<float>(next() % 1000) / 1.0e5F + 1.0e-4F);
        }
        for (auto& l : lookup)
        {
            l = __float2bfloat16(static_cast<float>(next() % 64));
        }
        for (auto& x : input)
        {
            x = __float2bfloat16(static_cast<float>(next() % 2001) / 1000.0F - 1.0F);
        }
        Tensor dValues({static_cast<int64_t>(values.size())}, DeviceType::kGPU, DataType::kINT8);
        Tensor dBases({static_cast<int64_t>(bases.size())}, DeviceType::kGPU, DataType::kHALF);
        Tensor dLookup({256, 8}, DeviceType::kGPU, DataType::kBF16);
        Tensor dInput({m, k}, DeviceType::kGPU, DataType::kBF16);
        Tensor tall({m, n}, DeviceType::kGPU, DataType::kBF16);
        Tensor chunked({m, n}, DeviceType::kGPU, DataType::kBF16);
        CUDA_CHECK(cudaMemcpy(dValues.rawPointer(), values.data(), values.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBases.rawPointer(), bases.data(), bases.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dLookup.rawPointer(), lookup.data(), lookup.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dInput.rawPointer(), input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        size_t workspaceBytes = CuteDslW2A16GemmRunner::getWorkspaceSize(m, n, k);
        for (int32_t rows = 1; rows <= 16; ++rows)
        {
            workspaceBytes = std::max(workspaceBytes, CuteDslW2A16GemmRunner::getWorkspaceSize(rows, n, k));
        }
        Tensor workspace(
            {std::max<int64_t>(static_cast<int64_t>(workspaceBytes), 256)}, DeviceType::kGPU, DataType::kINT8);
        auto const* in = static_cast<__nv_bfloat16 const*>(dInput.rawPointer());
        auto* tallOut = static_cast<__nv_bfloat16*>(tall.rawPointer());
        auto* chunkOut = static_cast<__nv_bfloat16*>(chunked.rawPointer());
        CuteDslW2A16GemmParams p{m, n, k, in, dValues.rawPointer(), dBases.rawPointer(), dLookup.rawPointer(), tallOut};
        // Modules load before any launch, as the runner contract requires.
        ASSERT_TRUE(CuteDslW2A16GemmRunner::ensureKernelModules(p, mStream));
        for (int32_t rows = 1; rows <= 16; ++rows)
        {
            CuteDslW2A16GemmParams q = p;
            q.m = rows;
            ASSERT_TRUE(CuteDslW2A16GemmRunner::ensureKernelModules(q, mStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        ASSERT_EQ(runner.run(p, workspace.rawPointer(), workspaceBytes, mStream), 0);
        for (int32_t row = 0; row < m; row += 16)
        {
            CuteDslW2A16GemmParams c{std::min(16, m - row), n, k, in + static_cast<size_t>(row) * k,
                dValues.rawPointer(), dBases.rawPointer(), dLookup.rawPointer(),
                chunkOut + static_cast<size_t>(row) * n};
            ASSERT_EQ(runner.run(c, workspace.rawPointer(), workspaceBytes, mStream), 0);
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<__nv_bfloat16> a(static_cast<size_t>(m) * n), b(a.size());
        CUDA_CHECK(cudaMemcpy(a.data(), tallOut, a.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), chunkOut, b.size() * 2, cudaMemcpyDeviceToHost));
        double diff = 0.0, ref = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double const x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
            ASSERT_TRUE(std::isfinite(x)) << "m=" << m << " index=" << i;
            diff += (x - y) * (x - y);
            ref += y * y;
        }
        EXPECT_GT(ref, 0.0) << "m=" << m;
        EXPECT_LE(std::sqrt(diff / ref), 1e-3) << "m=" << m << " n=" << n << " k=" << k;
    }
}

TEST(W2A16GemmContract, RuntimeCheckpointRepack)
{
    // Given raw IQ2_XXS checkpoint blocks [N, K/256, 66], when the runtime loader repacks them, then the payload is
    // [N/P, K/256, P, 64] (P = 128 if N % 128 == 0, else 8) and the FP16 bases are [K/256, N], as in
    // kernelSrcs/w2a16_gemm_cutedsl/prepare.py.
    for (auto const [n, kBlocks] : std::array<std::array<int32_t, 2>, 3>{{{256, 4}, {136, 3}, {5376, 2}}})
    {
        std::vector<uint8_t> raw(static_cast<size_t>(n) * kBlocks * kernel::kIq2XxsBlockBytes);
        uint32_t seed = static_cast<uint32_t>(n * 31 + kBlocks);
        for (auto& byte : raw)
        {
            seed = seed * 1664525U + 1013904223U;
            byte = static_cast<uint8_t>(seed >> 24);
        }
        int32_t const packing = n % 128 == 0 ? 128 : 8;
        std::vector<uint8_t> wantValues(static_cast<size_t>(n) * kBlocks * 64);
        std::vector<uint8_t> wantBases(static_cast<size_t>(n) * kBlocks * 2);
        for (int32_t row = 0; row < n; ++row)
        {
            for (int32_t k = 0; k < kBlocks; ++k)
            {
                uint8_t const* block
                    = raw.data() + (static_cast<size_t>(row) * kBlocks + k) * kernel::kIq2XxsBlockBytes;
                size_t const tile = (static_cast<size_t>(row / packing) * kBlocks + k) * packing + row % packing;
                std::copy(block + 2, block + 66, wantValues.begin() + tile * 64);
                std::copy(block, block + 2, wantBases.begin() + (static_cast<size_t>(k) * n + row) * 2);
            }
        }
        Tensor dRaw({static_cast<int64_t>(raw.size())}, DeviceType::kGPU, DataType::kINT8);
        Tensor dValues({static_cast<int64_t>(wantValues.size())}, DeviceType::kGPU, DataType::kINT8);
        Tensor dBases({static_cast<int64_t>(wantBases.size() / 2)}, DeviceType::kGPU, DataType::kHALF);
        CUDA_CHECK(cudaMemcpy(dRaw.rawPointer(), raw.data(), raw.size(), cudaMemcpyHostToDevice));
        ASSERT_EQ(kernel::launchIq2XxsGemmRepack(static_cast<uint8_t const*>(dRaw.rawPointer()),
                      static_cast<int8_t*>(dValues.rawPointer()), static_cast<uint16_t*>(dBases.rawPointer()), n,
                      kBlocks, nullptr),
            cudaSuccess);
        std::vector<uint8_t> gotValues(wantValues.size());
        std::vector<uint8_t> gotBases(wantBases.size());
        CUDA_CHECK(cudaMemcpy(gotValues.data(), dValues.rawPointer(), gotValues.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gotBases.data(), dBases.rawPointer(), gotBases.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(gotValues == wantValues) << "n=" << n << " kBlocks=" << kBlocks;
        EXPECT_TRUE(gotBases == wantBases) << "n=" << n << " kBlocks=" << kBlocks;
    }
}

TEST_F(W2A16GemmFixture, SmallRowsMatchDecodedReference)
{
    // Given random IQ2_XXS payloads, bases, magnitude tables and BF16 inputs with 1, 3 or 8 rows, both payload
    // packings and a partial 16-channel tile, when the runner multiplies them, then every output is within BF16
    // output rounding of an FP64 product over the decoded weights (relative L2 <= 2e-3).
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (!CuteDslW2A16GemmRunner::canImplement(1, 128, 256, props.major * 10 + props.minor, props.multiProcessorCount))
    {
        GTEST_SKIP() << "Requires SM110, SM120 or GB10";
    }
    CuteDslW2A16GemmRunner runner;
    for (auto const [n, k] : std::array<std::array<int32_t, 2>, 3>{{{5376, 4096}, {136, 768}, {1024, 1280}}})
    {
        int32_t const packing = n % 128 == 0 ? 128 : 8;
        int32_t const blocks = k / 256;
        std::vector<uint8_t> values(static_cast<size_t>(n) * k / 4);
        std::vector<__half> bases(static_cast<size_t>(blocks) * n);
        std::vector<__nv_bfloat16> lookup(256 * 8);
        uint32_t seed = static_cast<uint32_t>(n * 131 + k);
        auto next = [&seed]() {
            seed = seed * 1664525U + 1013904223U;
            return seed >> 8;
        };
        for (auto& v : values)
        {
            v = static_cast<uint8_t>(next());
        }
        for (auto& b : bases)
        {
            b = __float2half(static_cast<float>(next() % 1000) / 1.0e5F + 1.0e-4F);
        }
        for (auto& l : lookup)
        {
            l = __float2bfloat16(static_cast<float>(next() % 64));
        }
        // Decoded weights: record (channel, block) holds eight 8-byte sub-blocks of {grid bytes, 4x7 sign bits and a
        // 4-bit scale}; grid entry j covers K offsets sub * 32 + j * 8 + [0, 8).
        std::vector<double> weights(static_cast<size_t>(n) * k);
        for (int32_t ch = 0; ch < n; ++ch)
        {
            for (int32_t block = 0; block < blocks; ++block)
            {
                uint8_t const* record = values.data()
                    + ((static_cast<size_t>(ch / packing) * blocks + block) * packing + ch % packing) * 64;
                for (int32_t sub = 0; sub < 8; ++sub)
                {
                    uint32_t grids{}, signsScale{};
                    std::memcpy(&grids, record + sub * 8, 4);
                    std::memcpy(&signsScale, record + sub * 8 + 4, 4);
                    double const scale = static_cast<double>(__half2float(bases[static_cast<size_t>(block) * n + ch]))
                        * ((signsScale >> 28) + 0.5) * 0.25;
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        uint32_t const entry = (grids >> (8 * j)) & 0xFFU;
                        uint32_t signs = (signsScale >> (7 * j)) & 0x7FU;
                        signs |= (static_cast<uint32_t>(std::bitset<32>(signs).count()) & 1U) << 7;
                        for (int32_t e = 0; e < 8; ++e)
                        {
                            double const magnitude = __bfloat162float(lookup[entry * 8 + e]);
                            weights[static_cast<size_t>(ch) * k + block * 256 + sub * 32 + j * 8 + e]
                                = ((signs >> e) & 1U ? -magnitude : magnitude) * scale;
                        }
                    }
                }
            }
        }
        Tensor dValues({static_cast<int64_t>(values.size())}, DeviceType::kGPU, DataType::kINT8);
        Tensor dBases({static_cast<int64_t>(bases.size())}, DeviceType::kGPU, DataType::kHALF);
        Tensor dLookup({256, 8}, DeviceType::kGPU, DataType::kBF16);
        CUDA_CHECK(cudaMemcpy(dValues.rawPointer(), values.data(), values.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBases.rawPointer(), bases.data(), bases.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dLookup.rawPointer(), lookup.data(), lookup.size() * 2, cudaMemcpyHostToDevice));
        for (int32_t const m : {1, 3, 8})
        {
            std::vector<__nv_bfloat16> input(static_cast<size_t>(m) * k);
            for (auto& x : input)
            {
                x = __float2bfloat16(static_cast<float>(next() % 2001) / 1000.0F - 1.0F);
            }
            Tensor dInput({m, k}, DeviceType::kGPU, DataType::kBF16);
            Tensor dOutput({m, n}, DeviceType::kGPU, DataType::kBF16);
            CUDA_CHECK(cudaMemcpy(dInput.rawPointer(), input.data(), input.size() * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemsetAsync(dOutput.rawPointer(), 0xff, dOutput.getMemoryCapacity(), mStream));
            CuteDslW2A16GemmParams const p{m, n, k, dInput.rawPointer(), dValues.rawPointer(), dBases.rawPointer(),
                dLookup.rawPointer(), dOutput.rawPointer()};
            size_t const workspaceBytes = CuteDslW2A16GemmRunner::getWorkspaceSize(m, n, k);
            Tensor workspace(
                {std::max<int64_t>(static_cast<int64_t>(workspaceBytes), 256)}, DeviceType::kGPU, DataType::kINT8);
            ASSERT_TRUE(CuteDslW2A16GemmRunner::ensureKernelModules(p, mStream));
            ASSERT_EQ(runner.run(p, workspace.rawPointer(), workspaceBytes, mStream), 0);
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            std::vector<__nv_bfloat16> output(static_cast<size_t>(m) * n);
            CUDA_CHECK(cudaMemcpy(output.data(), dOutput.rawPointer(), output.size() * 2, cudaMemcpyDeviceToHost));
            double diff = 0.0, ref = 0.0;
            for (int32_t row = 0; row < m; ++row)
            {
                for (int32_t ch = 0; ch < n; ++ch)
                {
                    double expected = 0.0;
                    for (int32_t i = 0; i < k; ++i)
                    {
                        expected += static_cast<double>(__bfloat162float(input[static_cast<size_t>(row) * k + i]))
                            * weights[static_cast<size_t>(ch) * k + i];
                    }
                    double const actual = __bfloat162float(output[static_cast<size_t>(row) * n + ch]);
                    diff += (actual - expected) * (actual - expected);
                    ref += expected * expected;
                }
            }
            EXPECT_LE(std::sqrt(diff / ref), 2e-3) << "m=" << m << " n=" << n << " k=" << k;
        }
    }
}

TEST_F(W2A16GemmFixture, InvalidHostContracts)
{
    // Given null, misaligned, overlapping, or undersized buffers, or a default stream,
    // when dense GEMM is enqueued, then it returns cudaErrorInvalidValue before launching.
    Tensor storage({1 << 20}, DeviceType::kGPU, DataType::kINT8);
    auto* b = static_cast<char*>(storage.rawPointer());
    CuteDslW2A16GemmParams const good{16, 128, 4096, b, b + 131072, b + 262144, b + 278528, b + 294912};
    auto const size = CuteDslW2A16GemmRunner::getWorkspaceSize(good.m, good.n, good.k);
    CuteDslW2A16GemmRunner runner;
    for (int mode = 0; mode < 11; ++mode)
    {
        auto p = good;
        void* scratch = b + 524288;
        size_t bytes = size;
        auto s = mStream;
        switch (mode)
        {
        case 0: p.input = nullptr; break;
        case 1: p.values = nullptr; break;
        case 2: p.bases = nullptr; break;
        case 3: p.lookup = nullptr; break;
        case 4: p.output = nullptr; break;
        case 5: p.input = b + 1; break;
        case 6: p.output = b; break;
        case 7: scratch = b + 1; break;
        case 8: bytes = size - 1; break;
        case 9: scratch = p.output; break;
        case 10: s = nullptr; break;
        }
        EXPECT_EQ(runner.run(p, scratch, bytes, s), cudaErrorInvalidValue) << "mode=" << mode;
    }
}
} // namespace
#endif
