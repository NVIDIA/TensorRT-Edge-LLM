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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.h"
#include "kernels/speculative/dsparkKernels.h"
#include "testUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace trt_edgellm
{
namespace
{

template <typename T>
void uploadW2Test(rt::Tensor& device, std::vector<T> const& data, cudaStream_t stream)
{
    ASSERT_EQ(device.getShape().volume(), data.size());
    rt::Tensor host(device.getShape(), rt::DeviceType::kCPU, device.getDataType());
    std::copy(data.begin(), data.end(), static_cast<T*>(host.rawPointer()));
    CUDA_CHECK(cudaMemcpyAsync(
        device.rawPointer(), host.rawPointer(), data.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
std::vector<T> downloadW2Test(rt::Tensor const& device, cudaStream_t stream)
{
    rt::Tensor host(device.getShape(), rt::DeviceType::kCPU, device.getDataType());
    size_t const count = device.getShape().volume();
    CUDA_CHECK(
        cudaMemcpyAsync(host.rawPointer(), device.rawPointer(), count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const* values = static_cast<T const*>(host.rawPointer());
    return std::vector<T>(values, values + count);
}

void runW2Case(int32_t batch, int32_t proposals, int32_t offset, bool capture, bool ties, bool storeStacked = true,
    bool previouslyCompacted = false, bool poisonCorrection = false, int32_t vocab = 256, bool vocabularyTail = false)
{
    using namespace kernels;
    using rt::DeviceType;
    using rt::Tensor;
    using trt_edgellm::DataType;
    constexpr int32_t rank = 512;
    constexpr float global = 0.03125F;
    constexpr std::array<float, 16> fp4{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F, -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    constexpr std::array<uint8_t, 4> scaleCodes{0x20, 0x28, 0x30, 0x38};
    constexpr std::array<float, 4> scaleValues{0.125F, 0.25F, 0.5F, 1.0F};
    int32_t const stride = proposals + offset;
    SCOPED_TRACE(::testing::Message() << "B=" << batch << " P=" << proposals << " offset=" << offset
                                      << " graph=" << capture << " ties=" << ties);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Defer streamCleanup([&] { (void) cudaStreamDestroy(stream); });
    std::vector<half> hostW1(vocab * rank);
    std::vector<uint8_t> hostPacked(vocab * rank / 2), hostScales(vocab * rank / 16);
    std::vector<float> logicalW2(vocab * rank), hostLogits(batch * stride * vocab, -20.0F);
    std::vector<int32_t> previous(batch);
    for (int32_t n = 0; n < vocab; ++n)
    {
        for (int32_t k = 0; k < rank; ++k)
        {
            hostW1[n * rank + k] = __float2half(ties ? 0.0F : static_cast<float>((n * 11 + k * 3) % 13 - 6) / 128.0F);
            int32_t const code = (n * 7 + k * 5 + 1) % 16;
            hostPacked[(n * rank + k) / 2] |= static_cast<uint8_t>(code << (4 * (k % 2)));
            int32_t const scaleIndex = (n * 3 + k / 16) % 4;
            hostScales[n * rank / 16 + k / 16] = scaleCodes[scaleIndex];
            logicalW2[n * rank + k] = fp4[code] * scaleValues[scaleIndex];
        }
    }
    for (int32_t b = 0; b < batch; ++b)
    {
        previous[b] = 1 + b * 17;
        for (int32_t step = 0; step < proposals; ++step)
        {
            int32_t const winner = ties ? 4 : (vocabularyTail ? vocab - 1 : (b * 67 + step * 37 + 23) % vocab);
            hostLogits[(b * stride + step + offset) * vocab + winner] = 20.0F;
            if (ties)
                hostLogits[(b * stride + step + offset) * vocab + (vocabularyTail ? vocab - 1 : 11)] = 20.0F;
        }
    }
    Tensor w1({vocab, rank}, DeviceType::kGPU, DataType::kHALF);
    Tensor packed({vocab, rank / 2}, DeviceType::kGPU, DataType::kUINT8);
    Tensor scales({vocab, rank / 16}, DeviceType::kGPU, DataType::kUINT8);
    Tensor opaque({vocab / 128, rank / 64, 128, 32}, DeviceType::kGPU, DataType::kUINT8);
    Tensor opaqueScales({vocab / 128, rank / 64, 128, 4}, DeviceType::kGPU, DataType::kUINT8);
    Tensor globalScale({1}, DeviceType::kGPU, DataType::kFLOAT);
    Tensor logits({batch, stride, vocab}, DeviceType::kGPU, DataType::kFLOAT);
    Tensor firstPrevious({batch}, DeviceType::kGPU, DataType::kINT32);
    Tensor slots({batch, proposals}, DeviceType::kGPU, DataType::kINT64);
    Tensor activation({batch, rank}, DeviceType::kGPU, DataType::kHALF);
    Tensor correction({batch, vocab}, DeviceType::kGPU, DataType::kHALF);
    Tensor stacked({batch, proposals + 1, vocab}, DeviceType::kGPU, DataType::kFLOAT);
    uploadW2Test(w1, hostW1, stream);
    uploadW2Test(packed, hostPacked, stream);
    uploadW2Test(scales, hostScales, stream);
    uploadW2Test(globalScale, std::vector<float>{global}, stream);
    uploadW2Test(logits, hostLogits, stream);
    uploadW2Test(firstPrevious, previous, stream);
    uploadW2Test(stacked, std::vector<float>(batch * (proposals + 1) * vocab, 7.0F), stream);
    kernel::dsparkRepackW2Tcgen5(packed, scales, opaque, opaqueScales, vocab, rank, stream);
    CUDA_CHECK(Nvfp4A16BlackwellGemmRunner::prepare(Nvfp4A16BlackwellDtype::kFp16, batch, vocab, rank, stream));
    if (previouslyCompacted)
    {
        ASSERT_TRUE(stacked.reshape({1, proposals + 1, vocab}));
    }
    auto launch = [&] {
        CUDA_CHECK(
            cudaMemsetAsync(slots.rawPointer(), 0, static_cast<size_t>(batch) * proposals * sizeof(uint64_t), stream));
        for (int32_t step = 0; step < proposals; ++step)
        {
            if (poisonCorrection)
                CUDA_CHECK(cudaMemsetAsync(
                    correction.rawPointer(), 0xff, static_cast<size_t>(batch) * vocab * sizeof(half), stream));
            kernel::dsparkMarkovGreedyStepTcgen5(logits, w1, opaque, opaqueScales, globalScale, firstPrevious, slots,
                activation, correction, storeStacked ? &stacked : nullptr, step + 1, batch, step, proposals, vocab,
                rank, stream, stride, offset);
        }
    };
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    Defer graphCleanup([&] {
        if (executable != nullptr)
            (void) cudaGraphExecDestroy(executable);
        if (graph != nullptr)
            (void) cudaGraphDestroy(graph);
    });
    if (capture)
    {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    for (int32_t repeat = 0; repeat < 3; ++repeat)
    {
        if (capture)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        else
            launch();
    }
    auto const values = downloadW2Test<float>(stacked, stream);
    auto const winners = downloadW2Test<uint64_t>(slots, stream);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t n = 0; n < vocab; ++n)
            EXPECT_EQ(values[b * (proposals + 1) * vocab + n], 7.0F);
        int32_t prev = previous[b];
        for (int32_t step = 0; step < proposals; ++step)
        {
            int32_t bestToken = 0;
            float bestValue = -std::numeric_limits<float>::infinity();
            for (int32_t n = 0; n < vocab; ++n)
            {
                double dot = 0.0;
                for (int32_t k = 0; k < rank; ++k)
                    dot += static_cast<double>(__half2float(hostW1[prev * rank + k])) * logicalW2[n * rank + k];
                float const expected = hostLogits[(b * stride + step + offset) * vocab + n]
                    + __half2float(__float2half(static_cast<float>(dot * global)));
                if (storeStacked)
                    EXPECT_FLOAT_EQ(values[(b * (proposals + 1) + step + 1) * vocab + n], expected)
                        << "b=" << b << " step=" << step << " n=" << n;
                if (expected > bestValue)
                {
                    bestValue = expected;
                    bestToken = n;
                }
            }
            int32_t const actual
                = static_cast<int32_t>(0xffffffffU - static_cast<uint32_t>(winners[b * proposals + step]));
            EXPECT_EQ(actual, bestToken);
            prev = bestToken;
        }
    }
}

TEST(DSparkTcgen5W2, VocabularyTileTail)
{
    // Given a partial final vocabulary tile, when reducing winners, then padding cannot win.
    if (getSMVersion() != 110
        || !kernels::Nvfp4A16BlackwellGemmRunner::isSupported(110, kernels::Nvfp4A16BlackwellDtype::kFp16, 4, 128, 512)
        || !kernels::Nvfp4A16BlackwellGemmRunner::isSupported(
            110, kernels::Nvfp4A16BlackwellDtype::kFp16, 4, 1152, 512))
    {
        GTEST_SKIP() << "SM110 W4A16 AOT required";
    }
    for (int32_t vocab : {128, 1152})
    {
        for (bool graph : {false, true})
        {
            runW2Case(4, 3, 1, graph, false, true, false, true, vocab, true);
            runW2Case(2, 1, 0, graph, true, false, false, false, vocab, true);
        }
    }
}

TEST(DSparkTcgen5W2, SequentialGraphAndOffsets)
{
    // Given sequential proposals and offsets, when replayed with graphs, then winners match the reference.
    if (getSMVersion() != 110
        || !kernels::Nvfp4A16BlackwellGemmRunner::isSupported(110, kernels::Nvfp4A16BlackwellDtype::kFp16, 4, 256, 512))
        GTEST_SKIP() << "SM110 W4A16 AOT required";
    for (int32_t batch : {1, 2, 4})
        for (int32_t proposals : {1, 3, 8})
            for (int32_t offset : {0, 1})
                for (bool graph : {false, true})
                    runW2Case(batch, proposals, offset, graph, false);
    runW2Case(4, 3, 1, true, true);
    for (int32_t batch : {1, 2, 4})
        for (bool graph : {false, true})
            runW2Case(batch, 3, 1, graph, false, false);
}

TEST(DSparkTcgen5W2, OverwritesPoisonedCorrection)
{
    // Given poisoned scratch, when computing correction, then every logical output is overwritten.
    if (getSMVersion() != 110
        || !kernels::Nvfp4A16BlackwellGemmRunner::isSupported(110, kernels::Nvfp4A16BlackwellDtype::kFp16, 4, 256, 512))
        GTEST_SKIP() << "SM110 W4A16 AOT required";
    for (int32_t batch : {1, 2, 4})
        for (int32_t proposals : {3, 8})
            for (bool graph : {false, true})
                runW2Case(batch, proposals, 1, graph, false, true, false, true);
}

TEST(DSparkTcgen5W2, ReusesStackedCapacityAfterBatchCompaction)
{
    // Given compacted stacked storage, when batch grows, then retained capacity is safely reused.
    if (getSMVersion() != 110
        || !kernels::Nvfp4A16BlackwellGemmRunner::isSupported(110, kernels::Nvfp4A16BlackwellDtype::kFp16, 4, 256, 512))
    {
        GTEST_SKIP() << "SM110 W4A16 AOT required";
    }
    for (int32_t batch : {2, 4})
    {
        for (bool graph : {false, true})
        {
            runW2Case(batch, 8, 1, graph, false, true, true);
        }
    }
}

TEST(DSparkTcgen5, NativeW2Fallback)
{
    // Given all FP4 codes and finite E4M3 scales, when materializing fallback, then FP16 values match export.
    using rt::DeviceType;
    using rt::Tensor;
    using trt_edgellm::DataType;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Defer cleanup([&] { (void) cudaStreamDestroy(stream); });
    constexpr int32_t rows = 254;
    constexpr int32_t rank = 32;
    constexpr float global = 0.1F;
    std::vector<uint8_t> weights(rows * rank / 2), scales(rows * rank / 16);
    std::vector<half> expected(rows * rank);
    constexpr std::array<float, 16> values{0, 0.5F, 1, 1.5F, 2, 3, 4, 6, -0.0F, -0.5F, -1, -1.5F, -2, -3, -4, -6};
    for (int32_t row = 0; row < rows; ++row)
    {
        uint8_t const code = row < 127 ? row : row + 1;
        int32_t const exponent = (code >> 3) & 15;
        float scale = exponent == 0 ? static_cast<float>(code & 7) / 512.0F
                                    : (1.0F + static_cast<float>(code & 7) / 8.0F) * std::pow(2.0F, exponent - 7);
        scale = code & 128 ? -scale : scale;
        scales[row * 2] = scales[row * 2 + 1] = code;
        for (int32_t k = 0; k < rank; ++k)
        {
            if (k % 2 == 0)
            {
                weights[row * rank / 2 + k / 2] = (k % 16) | (((k + 1) % 16) << 4);
            }
            float const scaled = values[k % 16] * scale;
            expected[row * rank + k] = __float2half_rn(scaled * global);
        }
    }
    Tensor packed({rows, rank / 2}, DeviceType::kGPU, DataType::kUINT8);
    Tensor blocks({rows, rank / 16}, DeviceType::kGPU, DataType::kUINT8);
    Tensor multiplier({1}, DeviceType::kGPU, DataType::kFLOAT);
    Tensor dense({rows, rank}, DeviceType::kGPU, DataType::kHALF);
    uploadW2Test(packed, weights, stream);
    uploadW2Test(blocks, scales, stream);
    uploadW2Test(multiplier, std::vector<float>{global}, stream);
    kernel::dsparkDequantizeW2(packed, blocks, multiplier, dense, stream);
    auto const actual = downloadW2Test<half>(dense, stream);
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(__half_as_ushort(actual[i]), __half_as_ushort(expected[i])) << i;
    }
}

} // namespace
} // namespace trt_edgellm
