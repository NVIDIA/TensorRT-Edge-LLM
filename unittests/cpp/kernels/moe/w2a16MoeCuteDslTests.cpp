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

#if defined(CUTE_DSL_W2A16_MOE_ENABLED)

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/w2a16_cutedsl/cuteDslW2A16MoeRunner.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"
#include "kernels/weightsTransform/iq2_xxs/iq2XxsRepack.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{
using rt::DeviceType;
using rt::Tensor;
using trt_edgellm::DataType;

struct Geometry
{
    int32_t layer;
    int32_t experts;
    int32_t intermediate;
};

constexpr std::array<Geometry, 14> kGeometries{{{1, 256, 2048}, {3, 256, 1536}, {5, 256, 2560}, {10, 320, 2560},
    {14, 256, 2304}, {17, 384, 2816}, {21, 512, 2560}, {23, 448, 2816}, {28, 384, 2560}, {34, 448, 2560},
    {37, 320, 2816}, {72, 384, 2304}, {74, 448, 2304}, {76, 512, 2816}}};
constexpr std::array<Geometry, 12> kPerLayerGeometries{
    {{1, 240, 2048}, {3, 240, 1536}, {8, 304, 2304}, {10, 240, 2304}, {19, 304, 2048}, {21, 432, 2560}, {23, 432, 2304},
        {30, 304, 2560}, {37, 368, 2304}, {39, 368, 2048}, {45, 368, 2560}, {48, 496, 2560}}};
constexpr int32_t kHidden = 1024;
constexpr int32_t kTopK = 9;
constexpr uint32_t kFixtureMagic = 0x57324131;

struct Case
{
    Geometry geometry;
    int32_t tokens;
    std::string pattern;
};

std::ifstream openFixture(std::filesystem::path const& path, Geometry const& geometry, int32_t tokens)
{
    std::ifstream file(path, std::ios::binary);
    std::array<uint32_t, 4> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file
        || header
            != std::array<uint32_t, 4>{kFixtureMagic, static_cast<uint32_t>(geometry.experts),
                static_cast<uint32_t>(geometry.intermediate), static_cast<uint32_t>(tokens)})
    {
        throw std::runtime_error("Missing or incompatible W2A16 fixture: " + path.string());
    }
    return file;
}

void readHost(std::ifstream& file, Tensor& tensor)
{
    file.read(static_cast<char*>(tensor.rawPointer()), tensor.getMemoryCapacity());
    if (!file)
    {
        throw std::runtime_error("Truncated W2A16 fixture");
    }
}

void readDevice(std::ifstream& file, Tensor& tensor, cudaStream_t stream)
{
    Tensor host({tensor.getMemoryCapacity()}, DeviceType::kCPU, DataType::kINT8);
    readHost(file, host);
    CUDA_CHECK(cudaMemcpyAsync(
        tensor.rawPointer(), host.rawPointer(), tensor.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

struct Weights
{
    std::filesystem::path directory;
    Tensor q1, s1, q2, s2, lookup, unit;

    Weights(std::filesystem::path const& dir, Geometry const& g, cudaStream_t stream)
        : directory(dir)
        , q1({int64_t{g.experts} * g.intermediate * kHidden / 16}, DeviceType::kGPU, DataType::kINT32)
        , s1({int64_t{g.experts} * g.intermediate * kHidden / 512}, DeviceType::kGPU, DataType::kINT32)
        , q2({int64_t{g.experts} * g.intermediate * kHidden / 16}, DeviceType::kGPU, DataType::kINT32)
        , s2({int64_t{g.experts} * g.intermediate * kHidden / 512}, DeviceType::kGPU, DataType::kINT32)
        , lookup({2048}, DeviceType::kGPU, DataType::kINT8)
        , unit({g.experts}, DeviceType::kGPU, DataType::kFLOAT)
    {
        auto file = openFixture(dir / "weights.bin", g, 0);
        for (auto* tensor : {&q1, &s1, &q2, &s2, &lookup, &unit})
        {
            readDevice(file, *tensor, stream);
        }
    }
};

std::shared_ptr<Weights> loadWeights(std::filesystem::path const& directory, Geometry const& g, cudaStream_t stream)
{
    // Keep one geometry resident across its cases without retaining all checkpoint layers on the GPU.
    static std::shared_ptr<Weights> cached;
    if (!cached || cached->directory != directory)
    {
        cached = std::make_shared<Weights>(directory, g, stream);
    }
    return cached;
}

struct RunState
{
    int32_t const smVersion{getSMVersion()};
    int32_t const numSms{getDeviceMultiProcessorCount()};
    std::shared_ptr<Weights> weights;
    Tensor input, ids, routing, output, expected, workspace;
    CuteDslW2A16MoeParams params{};
    Case testCase;

    RunState(std::filesystem::path const& root, Case const& c, cudaStream_t stream)
        : weights(loadWeights(root / ("layer" + std::to_string(c.geometry.layer)), c.geometry, stream))
        , input({c.tokens, kHidden}, DeviceType::kGPU, DataType::kBF16)
        , ids({c.tokens, kTopK}, DeviceType::kGPU, DataType::kINT32)
        , routing({c.tokens, kTopK}, DeviceType::kGPU, DataType::kFLOAT)
        , output({c.tokens, kHidden}, DeviceType::kGPU, DataType::kBF16)
        , expected({c.tokens, kHidden}, DeviceType::kCPU, DataType::kBF16)
        , workspace({static_cast<int64_t>(CuteDslW2A16MoeRunner::getWorkspaceSize(
                        c.tokens, c.geometry.experts, c.geometry.intermediate, smVersion, numSms))},
              DeviceType::kGPU, DataType::kINT8)
        , testCase(c)
    {
        params = {c.tokens, c.geometry.experts, c.geometry.intermediate, input.rawPointer(), weights->q1.rawPointer(),
            weights->q2.rawPointer(), weights->s1.rawPointer(), weights->s2.rawPointer(),
            weights->unit.dataPointer<float>(), weights->lookup.rawPointer(), ids.dataPointer<int32_t>(),
            routing.dataPointer<float>(), output.rawPointer()};
        readInputs(c.pattern, stream);
        CUDA_CHECK(cudaMemsetAsync(output.rawPointer(), 0xff, output.getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(workspace.rawPointer(), 0xff, workspace.getMemoryCapacity(), stream));
    }

    void readInputs(std::string const& pattern, cudaStream_t stream)
    {
        auto file = openFixture(weights->directory / ("m" + std::to_string(testCase.tokens) + "_" + pattern + ".bin"),
            testCase.geometry, testCase.tokens);
        for (auto* tensor : {&input, &ids, &routing})
        {
            readDevice(file, *tensor, stream);
        }
        readHost(file, expected);
    }

    int32_t run(cudaStream_t stream)
    {
        return CuteDslW2A16MoeRunner{}.run(
            params, workspace.rawPointer(), workspace.getMemoryCapacity(), smVersion, numSms, stream);
    }

    std::vector<float> download(cudaStream_t stream)
    {
        Tensor host({testCase.tokens, kHidden}, DeviceType::kCPU, DataType::kBF16);
        CUDA_CHECK(cudaMemcpyAsync(
            host.rawPointer(), output.rawPointer(), output.getMemoryCapacity(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> values(int64_t{testCase.tokens} * kHidden);
        std::transform(host.dataPointer<__nv_bfloat16>(), host.dataPointer<__nv_bfloat16>() + values.size(),
            values.begin(), [](auto value) { return __bfloat162float(value); });
        return values;
    }

    void checkOutput(cudaStream_t stream, char const* phase = "output")
    {
        auto const actual = download(stream);
        double diff = 0, norm = 0, actualNorm = 0, dot = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            double const a = actual[i];
            double const b = __bfloat162float(expected.dataPointer<__nv_bfloat16>()[i]);
            diff += (a - b) * (a - b);
            norm += b * b;
            actualNorm += a * a;
            dot += a * b;
        }
        double const relativeL2 = std::sqrt(diff / norm);
        double const cosine = dot / std::sqrt(norm * actualNorm);
        printf("W2A16 layer%d M%d %s %s relative_l2=%.9g cosine=%.9g\n", testCase.geometry.layer, testCase.tokens,
            testCase.pattern.c_str(), phase, relativeL2, cosine);
        EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }));
        EXPECT_LE(relativeL2, 0.01);
        EXPECT_GE(cosine, 0.999);
    }
};

struct Graph
{
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    ~Graph()
    {
        if (executable)
        {
            cudaGraphExecDestroy(executable);
        }
        if (graph)
        {
            cudaGraphDestroy(graph);
        }
    }
    void endCapture(cudaStream_t stream)
    {
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    void launch(cudaStream_t stream)
    {
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
};

class W2A16Fixture : public testing::Test
{
protected:
    cudaStream_t mStream{};
    std::filesystem::path mData;
    char const* mDataVariable{"EDGE_LLM_W2A16_TEST_DATA"};
    void SetUp() override
    {
        int32_t device{};
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (!CuteDslW2A16MoeRunner::canImplement(
                1, 256, 2048, properties.major * 10 + properties.minor, properties.multiProcessorCount))
        {
            GTEST_SKIP() << "W2A16 MoE shape or device is not supported by the linked artifact";
        }
        char const* data = std::getenv(mDataVariable);
        if (data == nullptr)
        {
            GTEST_SKIP() << "Set " << mDataVariable << " to generated checkpoint fixtures";
        }
        mData = data;
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
};

class W2A16PreloadTest : public W2A16Fixture, public testing::WithParamInterface<Case>
{
};
class W2A16AccuracyTest : public W2A16Fixture, public testing::WithParamInterface<Case>
{
};
class W2A16GraphTest : public W2A16Fixture, public testing::WithParamInterface<Case>
{
};
class W2A16BoundaryTest : public W2A16Fixture, public testing::WithParamInterface<Case>
{
};
class W2A16BoltFixture : public W2A16Fixture
{
protected:
    W2A16BoltFixture()
    {
        mDataVariable = "EDGE_LLM_W2A16_PER_LAYER_TEST_DATA";
    }
};
class W2A16BoltSmallMTest : public W2A16BoltFixture, public testing::WithParamInterface<Case>
{
};
class W2A16BoltRouteChangeTest : public W2A16BoltFixture, public testing::WithParamInterface<Case>
{
};

constexpr std::array kArtifactSms{CUTE_DSL_W2A16_MOE_ARTIFACT_SMS};

// Given supported geometry, when hardware admission is queried, then SM101/SM110 UMMA accepts any positive SM count
// and fused targets require at least 48 SMs, provided the corresponding architecture is built.
TEST(CuteDslW2A16MoeTest, hardwareAdmission)
{
    auto const built
        = [](int32_t sm) { return std::find(kArtifactSms.begin(), kArtifactSms.end(), sm) != kArtifactSms.end(); };
    bool const kSm110Enabled = built(101) || built(110);
    bool const kSm120Enabled = built(120);
    bool const kSm121Enabled = built(121);
    for (int32_t const numSms : {1, 16, 20, 24, 48})
    {
        EXPECT_EQ(CuteDslW2A16MoeRunner::canImplement(1, 256, 2048, 101, numSms), kSm110Enabled);
        EXPECT_EQ(CuteDslW2A16MoeRunner::canImplement(1, 256, 2048, 110, numSms), kSm110Enabled);
    }
    for (int32_t const numSms : {48, 84, 170})
    {
        EXPECT_EQ(CuteDslW2A16MoeRunner::canImplement(1, 256, 2048, 120, numSms), kSm120Enabled);
        EXPECT_EQ(CuteDslW2A16MoeRunner::canImplement(1, 256, 2048, 121, numSms), kSm121Enabled);
        EXPECT_EQ(CuteDslW2A16MoeRunner::canImplement(1, 240, 2048, 121, numSms), kSm121Enabled);
    }
    for (auto const& hardware : std::array<std::array<int32_t, 2>, 8>{
             {{101, 0}, {101, -1}, {110, 0}, {110, -1}, {100, 48}, {120, 47}, {121, 47}, {121, 20}}})
    {
        EXPECT_FALSE(CuteDslW2A16MoeRunner::canImplement(1, 256, 2048, hardware[0], hardware[1]));
    }
}

// Repack raw checkpoint blocks with the runtime loader's transform and require the bytes the builder's packer wrote.
void expectRuntimeRepackMatches(
    std::filesystem::path const& directory, Geometry const& g, Weights const& packed, cudaStream_t stream)
{
    auto file = openFixture(directory / "raw.bin", g, 0);
    int32_t const blocks[2] = {kHidden / 256, g.intermediate / 256};
    int32_t const rows[2] = {g.intermediate, kHidden};
    Tensor const* expected[2][2] = {{&packed.q1, &packed.s1}, {&packed.q2, &packed.s2}};
    for (int32_t projection = 0; projection < 2; ++projection)
    {
        int64_t const expertBytes = int64_t{rows[projection]} * blocks[projection] * kernel::kIq2XxsBlockBytes;
        Tensor raw({g.experts * expertBytes}, DeviceType::kGPU, DataType::kINT8);
        readDevice(file, raw, stream);
        Tensor words(expected[projection][0]->getShape(), DeviceType::kGPU, DataType::kINT32);
        Tensor bases(expected[projection][1]->getShape(), DeviceType::kGPU, DataType::kINT32);
        int64_t const wordsPerExpert = words.getShape().volume() / g.experts;
        int64_t const basesPerExpert = bases.getShape().volume() / g.experts;
        for (int32_t begin = 0; begin < g.experts; begin += kernel::kCheckpointSourcesPerLaunch)
        {
            int32_t const count = std::min(kernel::kCheckpointSourcesPerLaunch, g.experts - begin);
            std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> sources{};
            for (int32_t local = 0; local < count; ++local)
            {
                sources[local] = static_cast<uint8_t const*>(raw.rawPointer()) + (begin + local) * expertBytes;
            }
            ASSERT_EQ(kernel::launchIq2XxsMoeRepackSourceBatch(sources.data(), count,
                          words.dataPointer<int32_t>() + begin * wordsPerExpert,
                          reinterpret_cast<uint16_t*>(bases.dataPointer<int32_t>() + begin * basesPerExpert),
                          rows[projection], blocks[projection], stream),
                cudaSuccess);
        }
        for (int32_t plane = 0; plane < 2; ++plane)
        {
            Tensor const& actual = plane == 0 ? words : bases;
            std::vector<uint8_t> got(actual.getMemoryCapacity());
            std::vector<uint8_t> want(actual.getMemoryCapacity());
            CUDA_CHECK(cudaMemcpyAsync(got.data(), actual.rawPointer(), got.size(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                want.data(), expected[projection][plane]->rawPointer(), want.size(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            EXPECT_TRUE(got == want) << "layer " << g.layer << " fc" << projection + 1
                                     << (plane ? " bases" : " payload");
        }
    }
}

// Given checkpoint fixtures whose layers differ in expert count and width, when the runtime repacks the raw expert
// blocks, then the bytes match the builder's packer; and when each E/I geometry runs decode and prefill, the output
// matches the independent checkpoint oracle.
TEST(CuteDslW2A16MoeTest, perLayerGeometries)
{
    char const* root = std::getenv("EDGE_LLM_W2A16_PER_LAYER_TEST_DATA");
    if (root == nullptr)
    {
        GTEST_SKIP() << "Set EDGE_LLM_W2A16_PER_LAYER_TEST_DATA to per-layer checkpoint fixtures";
    }
    if (!CuteDslW2A16MoeRunner::canImplement(1, 240, 2048, getSMVersion(), getDeviceMultiProcessorCount()))
    {
        GTEST_SKIP() << "Per-layer W2A16 MoE geometries are not supported on this GPU";
    }
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (auto const& geometry : kPerLayerGeometries)
    {
        std::filesystem::path const directory
            = std::filesystem::path(root) / ("layer" + std::to_string(geometry.layer));
        expectRuntimeRepackMatches(directory, geometry, *loadWeights(directory, geometry, stream), stream);
        for (int32_t tokens : {1, 16})
        {
            RunState state(root, {geometry, tokens, "balanced"}, stream);
            ASSERT_EQ(state.run(stream), 0);
            state.checkOutput(stream);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Given every admitted expert/intermediate geometry and devices with 20, 48, 148 and 192 SMs,
// When the workspace is sized for each token count from 1 to kMaxNumTokens,
// Then the size never shrinks as the token count grows, so the workspace reserved for a profile maximum covers every
// smaller token count the engine may run.
TEST(CuteDslW2A16MoeTest, workspaceIsNonDecreasingInTokens)
{
    for (int32_t const experts : {240, 256, 304, 320, 368, 384, 432, 448, 496, 512})
    {
        for (int32_t const inter : {1536, 2048, 2304, 2560, 2816})
        {
            if (CuteDslW2A16MoeRunner::getWorkspaceSize(1, experts, inter, kArtifactSms.front(), 148) == 0)
            {
                continue;
            }
            for (int32_t const sms : {20, 48, 148, 192})
            {
                size_t previous = 0;
                for (int32_t tokens = 1; tokens <= CuteDslW2A16MoeRunner::kMaxNumTokens; ++tokens)
                {
                    size_t const bytes
                        = CuteDslW2A16MoeRunner::getWorkspaceSize(tokens, experts, inter, kArtifactSms.front(), sms);
                    ASSERT_GE(bytes, previous)
                        << "E=" << experts << " I=" << inter << " sms=" << sms << " tokens=" << tokens;
                    previous = bytes;
                }
            }
        }
    }
}

// Given unsupported shapes, when workspace size is queried, then the runner rejects them with a zero size.
TEST(CuteDslW2A16MoeTest, invalidShapes)
{
    for (auto const& shape : std::array<std::array<int32_t, 3>, 6>{
             {{0, 256, 2048}, {-1, 256, 2048}, {8193, 256, 2048}, {1, 128, 2048}, {1, 512, 2048}, {1, 256, 1024}}})
    {
        EXPECT_EQ(CuteDslW2A16MoeRunner::getWorkspaceSize(shape[0], shape[1], shape[2], kArtifactSms.front(), 48), 0);
    }
}

// Given shape-only preload, when the first run is captured, then graph replay matches the independent checkpoint
// oracle.
TEST_P(W2A16PreloadTest, firstRunInGraph)
{
    auto const& c = GetParam();
    CuteDslW2A16MoeParams shape{};
    shape.numTokens = c.tokens;
    shape.numExperts = c.geometry.experts;
    shape.moeInterSize = c.geometry.intermediate;
    ASSERT_TRUE(
        CuteDslW2A16MoeRunner::ensureKernelModules(shape, getSMVersion(), getDeviceMultiProcessorCount(), mStream));
    RunState state(mData, c, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    Graph graph;
    CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
    int32_t const status = state.run(mStream);
    graph.endCapture(mStream);
    ASSERT_EQ(status, 0);
    graph.launch(mStream);
    state.checkOutput(mStream);
}

// Given each supported checkpoint geometry and routing distribution, when either mode runs, then its output matches the
// oracle.
TEST_P(W2A16AccuracyTest, checkpoint)
{
    RunState state(mData, GetParam(), mStream);
    ASSERT_EQ(state.run(mStream), 0);
    state.checkOutput(mStream);
}

// Given a captured graph, when inputs/routes change and routing weights become zero then are restored, then every
// replay uses the current data and resets workspace.
TEST_P(W2A16GraphTest, updatedInputsAndWorkspaceReuse)
{
    RunState state(mData, GetParam(), mStream);
    ASSERT_EQ(state.run(mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    Graph graph;
    CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
    int32_t const status = state.run(mStream);
    graph.endCapture(mStream);
    ASSERT_EQ(status, 0);
    state.readInputs("changed", mStream);
    graph.launch(mStream);
    state.checkOutput(mStream, "changed");
    CUDA_CHECK(cudaMemsetAsync(state.routing.rawPointer(), 0, state.routing.getMemoryCapacity(), mStream));
    graph.launch(mStream);
    auto const zero = state.download(mStream);
    EXPECT_TRUE(std::all_of(zero.begin(), zero.end(), [](float value) { return value == 0.0F; }));
    state.readInputs("changed", mStream);
    graph.launch(mStream);
    state.checkOutput(mStream, "reused");
}

// Given a Bolt checkpoint geometry and 1-9, 12 or 16 tokens whose routes are all distinct (balanced), shared by every
// token (hot; 9-16 tokens put more than 8 rows on one expert, so it spans two 8-token tiles) or mostly shared
// (imbalanced), when the runner executes, then the output matches the independent checkpoint oracle.
TEST_P(W2A16BoltSmallMTest, checkpoint)
{
    RunState state(mData, GetParam(), mStream);
    ASSERT_EQ(state.run(mStream), 0);
    state.checkOutput(mStream);
}

// Given a graph captured for M tokens whose routes are shared by every token, when it is replayed after the routes
// change to all distinct and then to mostly shared, then each replay matches the oracle for the current routes.
TEST_P(W2A16BoltRouteChangeTest, graphReplayRegroupsRoutes)
{
    RunState state(mData, GetParam(), mStream);
    ASSERT_EQ(state.run(mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    Graph graph;
    CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
    int32_t const status = state.run(mStream);
    graph.endCapture(mStream);
    ASSERT_EQ(status, 0);
    for (char const* pattern : {"balanced", "imbalanced"})
    {
        state.readInputs(pattern, mStream);
        CUDA_CHECK(cudaMemsetAsync(state.output.rawPointer(), 0xff, state.output.getMemoryCapacity(), mStream));
        graph.launch(mStream);
        state.checkOutput(mStream, pattern);
    }
}

// Given token counts through 8192, when valid or invalid workspace is supplied, then valid runs match the oracle and
// absent/short/misaligned workspace is rejected.
TEST_P(W2A16BoundaryTest, tokenCounts)
{
    RunState state(mData, GetParam(), mStream);
    CuteDslW2A16MoeRunner runner;
    auto const bytes = state.workspace.getMemoryCapacity();
    EXPECT_NE(runner.run(state.params, nullptr, 0, state.smVersion, state.numSms, mStream), 0);
    EXPECT_NE(
        runner.run(state.params, state.workspace.rawPointer(), bytes - 1, state.smVersion, state.numSms, mStream), 0);
    EXPECT_NE(runner.run(state.params, state.workspace.dataPointer<int8_t>() + 16, bytes, state.smVersion, state.numSms,
                  mStream),
        0);
    ASSERT_EQ(state.run(mStream), 0);
    state.checkOutput(mStream);
}

std::vector<Case> cases(std::vector<int32_t> const& tokens, std::vector<std::string> const& patterns,
    bool allGeometries, bool tileVariants = false)
{
    std::vector<Case> result;
    for (auto const& geometry : kGeometries)
    {
        auto tokenCounts = tokens;
        if (tileVariants)
        {
            for (int32_t const m : {geometry.experts / 2, geometry.experts, 2 * geometry.experts, 4 * geometry.experts,
                     6 * geometry.experts, 12 * geometry.experts})
            {
                tokenCounts.push_back(m);
            }
            std::sort(tokenCounts.begin(), tokenCounts.end());
            tokenCounts.erase(std::unique(tokenCounts.begin(), tokenCounts.end()), tokenCounts.end());
        }
        for (auto const m : tokenCounts)
        {
            for (auto const& pattern : patterns)
            {
                result.push_back({geometry, m, pattern});
            }
        }
        if (!allGeometries)
        {
            break;
        }
    }
    return result;
}

std::vector<Case> boltCases(std::vector<int32_t> const& tokens, std::vector<std::string> const& patterns)
{
    std::vector<Case> result;
    for (auto const& geometry : kPerLayerGeometries)
    {
        for (auto const m : tokens)
        {
            for (auto const& pattern : patterns)
            {
                result.push_back({geometry, m, pattern});
            }
        }
    }
    return result;
}

std::string caseName(testing::TestParamInfo<Case> const& info)
{
    return "Layer" + std::to_string(info.param.geometry.layer) + "M" + std::to_string(info.param.tokens) + "_"
        + info.param.pattern;
}

INSTANTIATE_TEST_SUITE_P(
    CuteDslW2A16Moe, W2A16PreloadTest, testing::ValuesIn(cases({8, 33}, {"hot"}, true, true)), caseName);
INSTANTIATE_TEST_SUITE_P(CuteDslW2A16Moe, W2A16AccuracyTest,
    testing::ValuesIn(cases({1, 8, 16, 128}, {"balanced", "hot", "imbalanced"}, true, true)), caseName);
INSTANTIATE_TEST_SUITE_P(
    CuteDslW2A16Moe, W2A16GraphTest, testing::ValuesIn(cases({8, 33}, {"hot"}, true, true)), caseName);
INSTANTIATE_TEST_SUITE_P(CuteDslW2A16Moe, W2A16BoltSmallMTest,
    testing::ValuesIn(boltCases({1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16}, {"balanced", "hot", "imbalanced"})), caseName);
INSTANTIATE_TEST_SUITE_P(
    CuteDslW2A16Moe, W2A16BoltRouteChangeTest, testing::ValuesIn(boltCases({2, 8, 12, 16}, {"hot"})), caseName);
INSTANTIATE_TEST_SUITE_P(CuteDslW2A16Moe, W2A16BoundaryTest,
    testing::ValuesIn(
        cases({2, 3, 4, 5, 6, 7, 9, 31, 32, 204, 205, 409, 410, 819, 820, 1228, 1229, 8192}, {"hot"}, false)),
    caseName);
} // namespace

#endif // CUTE_DSL_W2A16_MOE_ENABLED
