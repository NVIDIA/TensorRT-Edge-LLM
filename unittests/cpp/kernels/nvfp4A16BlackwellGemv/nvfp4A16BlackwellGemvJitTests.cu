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
#include "kernels/PluginJitKernels/pluginJitCompileCache.h"
#include "kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitCompiler.h"
#include "kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitRunner.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace
{

constexpr int32_t kTEST_N{2688};
constexpr int32_t kTEST_K{3712};
constexpr int32_t kN_TILE{128};
constexpr int32_t kK_TILE{64};
constexpr float kGLOBAL_SCALE{0.25F};
constexpr std::array<int32_t, 5> kSUPPORTED_M{1, 2, 4, 8, 16};
constexpr std::array<int32_t, 3> kSUPPORTED_SPLIT_K{1, 2, 4};
constexpr std::array<uint8_t, 4> kSCALE_CODES{0x20U, 0x28U, 0x30U, 0x38U};

Nvfp4A16BlackwellGemvJitKey makeKey(
    Nvfp4A16BlackwellGemvDataType const dataType, int32_t const n = kTEST_N, int32_t const k = kTEST_K)
{
    return {110, kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI, n, k, dataType, kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI};
}

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t const count)
    {
        CUDA_CHECK(cudaMalloc(&mPointer, count * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (mPointer != nullptr)
        {
            (void) cudaFree(mPointer);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    T* get() const noexcept
    {
        return mPointer;
    }

private:
    T* mPointer{};
};

template <typename T>
class PinnedBuffer
{
public:
    explicit PinnedBuffer(size_t const count)
        : mCount(count)
    {
        CUDA_CHECK(cudaMallocHost(&mPointer, count * sizeof(T)));
    }

    ~PinnedBuffer()
    {
        if (mPointer != nullptr)
        {
            (void) cudaFreeHost(mPointer);
        }
    }

    PinnedBuffer(PinnedBuffer const&) = delete;
    PinnedBuffer& operator=(PinnedBuffer const&) = delete;

    T* get() const noexcept
    {
        return mPointer;
    }

    size_t size() const noexcept
    {
        return mCount;
    }

private:
    T* mPointer{};
    size_t mCount{};
};

class NonBlockingStream
{
public:
    NonBlockingStream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~NonBlockingStream()
    {
        if (mStream != nullptr)
        {
            (void) cudaStreamDestroy(mStream);
        }
    }

    NonBlockingStream(NonBlockingStream const&) = delete;
    NonBlockingStream& operator=(NonBlockingStream const&) = delete;

    cudaStream_t get() const noexcept
    {
        return mStream;
    }

private:
    cudaStream_t mStream{};
};

bool isSm110()
{
    int32_t deviceCount{};
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        return false;
    }
    int32_t device{};
    CUDA_CHECK(cudaGetDevice(&device));
    int32_t major{};
    int32_t minor{};
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major * 10 + minor == 110;
}

float fp4Value(uint8_t const code)
{
    constexpr std::array<float, 8> kLEVELS{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    float const magnitude = kLEVELS[code & 7U];
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

float scaleValue(uint8_t const raw)
{
    switch (raw)
    {
    case 0x20U: return 0.125F;
    case 0x28U: return 0.25F;
    case 0x30U: return 0.5F;
    case 0x38U: return 1.0F;
    default: return 0.0F;
    }
}

uint8_t logicalCode(int32_t const n, int32_t const k)
{
    int32_t const nBlock = n / kN_TILE;
    int32_t const row = n % kN_TILE;
    int32_t const kBlock = k / kK_TILE;
    int32_t const kInBlock = k % kK_TILE;
    return static_cast<uint8_t>((nBlock * 3 + row * 7 + kBlock * 5 + kInBlock * 11 + 1) & 15);
}

uint8_t logicalScale(int32_t const n, int32_t const k)
{
    int32_t const nBlock = n / kN_TILE;
    int32_t const row = n % kN_TILE;
    int32_t const kBlock = k / kK_TILE;
    int32_t const scaleInBlock = (k % kK_TILE) / 16;
    return kSCALE_CODES[static_cast<size_t>((nBlock * 5 + row * 3 + kBlock * 7 + scaleInBlock) & 3)];
}

void fillPhysicalLayout(PinnedBuffer<uint8_t>& qweights, PinnedBuffer<uint8_t>& scales)
{
    int32_t const nBlocks = kTEST_N / kN_TILE;
    int32_t const kBlocks = kTEST_K / kK_TILE;
    for (int32_t nBlock = 0; nBlock < nBlocks; ++nBlock)
    {
        for (int32_t kBlock = 0; kBlock < kBlocks; ++kBlock)
        {
            for (int32_t row = 0; row < kN_TILE; ++row)
            {
                int32_t const n = nBlock * kN_TILE + row;
                size_t const rowTile = (static_cast<size_t>(nBlock) * kBlocks + kBlock) * kN_TILE + row;
                for (int32_t byte = 0; byte < kK_TILE / 2; ++byte)
                {
                    int32_t const evenK = kBlock * kK_TILE + byte * 2;
                    uint8_t const low = logicalCode(n, evenK);
                    uint8_t const high = logicalCode(n, evenK + 1);
                    qweights.get()[rowTile * (kK_TILE / 2) + byte] = static_cast<uint8_t>(low | (high << 4));
                }
                for (int32_t scale = 0; scale < kK_TILE / 16; ++scale)
                {
                    int32_t const k = kBlock * kK_TILE + scale * 16;
                    scales.get()[rowTile * (kK_TILE / 16) + scale] = logicalScale(n, k);
                }
            }
        }
    }
}

template <typename T>
T hostFromFloat(float value);

template <>
half hostFromFloat<half>(float const value)
{
    return __float2half_rn(value);
}

template <>
__nv_bfloat16 hostFromFloat<__nv_bfloat16>(float const value)
{
    return __float2bfloat16_rn(value);
}

template <typename T>
float hostToFloat(T value);

template <>
float hostToFloat<half>(half const value)
{
    return __half2float(value);
}

template <>
float hostToFloat<__nv_bfloat16>(__nv_bfloat16 const value)
{
    return __bfloat162float(value);
}

template <typename T>
void runConstantSpecializations(Nvfp4A16BlackwellGemvDataType const dataType, int32_t const splitK)
{
    Nvfp4A16BlackwellGemvJitKey const key = makeKey(dataType);
    Nvfp4A16BlackwellGemvJitRunner runner;
    runner.load(compileNvfp4A16BlackwellGemvJitKernel(key));
    size_t const qweightCount = static_cast<size_t>(kTEST_N) * kTEST_K / 2;
    size_t const scaleCount = static_cast<size_t>(kTEST_N) * kTEST_K / 16;
    constexpr int32_t kMAX_M{16};
    PinnedBuffer<uint8_t> hostQweights(qweightCount);
    PinnedBuffer<uint8_t> hostScales(scaleCount);
    std::fill_n(hostQweights.get(), hostQweights.size(), 0x22U);
    std::fill_n(hostScales.get(), hostScales.size(), 0x38U);
    PinnedBuffer<T> hostActivation(static_cast<size_t>(kMAX_M) * kTEST_K);
    PinnedBuffer<T> hostOutput(static_cast<size_t>(kMAX_M) * kTEST_N);
    PinnedBuffer<float> hostGlobalScale(1);
    hostGlobalScale.get()[0] = kGLOBAL_SCALE;

    DeviceBuffer<uint8_t> qweights(qweightCount);
    DeviceBuffer<uint8_t> scales(scaleCount);
    DeviceBuffer<T> activation(static_cast<size_t>(kMAX_M) * kTEST_K);
    DeviceBuffer<T> output(static_cast<size_t>(kMAX_M) * kTEST_N);
    DeviceBuffer<float> globalScale(1);
    DeviceBuffer<float> workspace(static_cast<size_t>(4) * kMAX_M * kTEST_N);
    NonBlockingStream stream;
    CUDA_CHECK(cudaMemcpyAsync(qweights.get(), hostQweights.get(), qweightCount, cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(scales.get(), hostScales.get(), scaleCount, cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(
        cudaMemcpyAsync(globalScale.get(), hostGlobalScale.get(), sizeof(float), cudaMemcpyHostToDevice, stream.get()));

    for (int32_t const m : kSUPPORTED_M)
    {
        for (int32_t row = 0; row < m; ++row)
        {
            T const value = hostFromFloat<T>(static_cast<float>(row + 1) / 1024.0F);
            std::fill_n(hostActivation.get() + static_cast<size_t>(row) * kTEST_K, kTEST_K, value);
        }
        CUDA_CHECK(cudaMemcpyAsync(activation.get(), hostActivation.get(), static_cast<size_t>(m) * kTEST_K * sizeof(T),
            cudaMemcpyHostToDevice, stream.get()));
        size_t const workspaceSize = getNvfp4A16BlackwellGemvJitWorkspaceSize(key, m, splitK);
        runner.launch(activation.get(), qweights.get(), scales.get(), globalScale.get(), output.get(),
            workspaceSize == 0 ? nullptr : workspace.get(), workspaceSize, m, splitK, stream.get());
        CUDA_CHECK(cudaMemcpyAsync(hostOutput.get(), output.get(), static_cast<size_t>(m) * kTEST_N * sizeof(T),
            cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        for (int32_t row = 0; row < m; ++row)
        {
            float const expected = hostToFloat(hostFromFloat<T>(
                hostToFloat(hostActivation.get()[static_cast<size_t>(row) * kTEST_K]) * kTEST_K * kGLOBAL_SCALE));
            for (int32_t column = 0; column < kTEST_N; ++column)
            {
                float const actual = hostToFloat(hostOutput.get()[static_cast<size_t>(row) * kTEST_N + column]);
                ASSERT_EQ(actual, expected) << "M=" << m << " row=" << row << " column=" << column;
            }
        }
    }
}

TEST(Nvfp4A16BlackwellGemvJitTest, SupportMatrix)
{
    constexpr std::array<std::array<int32_t, 2>, 6> kSHAPES{
        {{128, 64}, {256, 128}, {384, 192}, {3712, 2688}, {2688, 3712}, {131072, 2688}}};
    for (auto const& shape : kSHAPES)
    {
        for (int32_t const m : kSUPPORTED_M)
        {
            for (int32_t const splitK : kSUPPORTED_SPLIT_K)
            {
                EXPECT_TRUE(isNvfp4A16BlackwellGemvJitSupported(
                    makeKey(Nvfp4A16BlackwellGemvDataType::kHALF, shape[0], shape[1]), m, splitK));
                EXPECT_TRUE(isNvfp4A16BlackwellGemvJitSupported(
                    makeKey(Nvfp4A16BlackwellGemvDataType::kBF16, shape[0], shape[1]), m, splitK));
            }
        }
    }

    EXPECT_FALSE(isNvfp4A16BlackwellGemvJitSupported(makeKey(Nvfp4A16BlackwellGemvDataType::kHALF), 3, 1));
    EXPECT_FALSE(isNvfp4A16BlackwellGemvJitSupported(makeKey(Nvfp4A16BlackwellGemvDataType::kHALF, 129, 64), 1, 1));
    EXPECT_FALSE(isNvfp4A16BlackwellGemvJitSupported(makeKey(Nvfp4A16BlackwellGemvDataType::kHALF, 128, 65), 1, 1));
    EXPECT_FALSE(isNvfp4A16BlackwellGemvJitSupported(makeKey(Nvfp4A16BlackwellGemvDataType::kHALF), 1, 3));
    EXPECT_FALSE(isNvfp4A16BlackwellGemvJitSupported(makeKey(static_cast<Nvfp4A16BlackwellGemvDataType>(2)), 1, 1));
}

TEST(Nvfp4A16BlackwellGemvJitTest, GenericSemanticKeysRemainShapeSpecific)
{
    auto const first = makeKey(Nvfp4A16BlackwellGemvDataType::kHALF, 128, 64);
    auto const second = makeKey(Nvfp4A16BlackwellGemvDataType::kHALF, 256, 128);
    EXPECT_TRUE(canCompileNvfp4A16BlackwellGemvJitKernel(first));
    EXPECT_TRUE(canCompileNvfp4A16BlackwellGemvJitKernel(second));
    EXPECT_FALSE(first == second);

    Nvfp4A16BlackwellGemvJitKernel kernel;
    kernel.key = first;
    kernel.cubin = {0x7FU, 0x45U, 0x4CU, 0x46U, 0x01U};
    kernel.digest = computeNvfp4A16BlackwellGemvJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    auto const blob = serializeNvfp4A16BlackwellGemvJitKernel(kernel);
    auto const decoded = deserializeNvfp4A16BlackwellGemvJitKernel(blob.data(), blob.size());
    EXPECT_TRUE(decoded.key == first);
    EXPECT_FALSE(decoded.key == second);
}

TEST(Nvfp4A16BlackwellGemvJitTest, WorkspaceContract)
{
    Nvfp4A16BlackwellGemvJitKey const key = makeKey(Nvfp4A16BlackwellGemvDataType::kHALF);
    EXPECT_EQ(getNvfp4A16BlackwellGemvJitWorkspaceSize(key, 16, 1), 0U);
    EXPECT_EQ(
        getNvfp4A16BlackwellGemvJitWorkspaceSize(key, 16, 2), static_cast<size_t>(2) * 16 * kTEST_N * sizeof(float));
    EXPECT_EQ(
        getNvfp4A16BlackwellGemvJitWorkspaceSize(key, 16, 4), static_cast<size_t>(4) * 16 * kTEST_N * sizeof(float));
    EXPECT_EQ(getNvfp4A16BlackwellGemvJitWorkspaceSize(key, 3, 4), 0U);
}

TEST(Nvfp4A16BlackwellGemvJitTest, RejectsInvalidLaunchArguments)
{
    Nvfp4A16BlackwellGemvJitRunner runner;
    EXPECT_THROW(
        runner.launch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 1, nullptr), std::runtime_error);
}

TEST(Nvfp4A16BlackwellGemvJitTest, BundleRoundTripChecksDigest)
{
    Nvfp4A16BlackwellGemvJitKernel kernel;
    kernel.key = makeKey(Nvfp4A16BlackwellGemvDataType::kBF16);
    kernel.cubin = {0x7FU, 0x45U, 0x4CU, 0x46U, 0x01U};
    kernel.digest = computeNvfp4A16BlackwellGemvJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    std::vector<uint8_t> blob = serializeNvfp4A16BlackwellGemvJitKernel(kernel);
    Nvfp4A16BlackwellGemvJitKernel const decoded = deserializeNvfp4A16BlackwellGemvJitKernel(blob.data(), blob.size());
    EXPECT_TRUE(decoded.key == kernel.key);
    EXPECT_TRUE(decoded.digest == kernel.digest);
    EXPECT_EQ(decoded.cubin, kernel.cubin);

    blob.back() ^= 1U;
    EXPECT_THROW(deserializeNvfp4A16BlackwellGemvJitKernel(blob.data(), blob.size()), std::runtime_error);
}

TEST(Nvfp4A16BlackwellGemvJitTest, ConcurrentCompileCacheSharesAndRetries)
{
    struct Hash
    {
        size_t operator()(int const value) const noexcept
        {
            return static_cast<size_t>(value);
        }
    };
    PluginJitCompileCache<int, int, Hash> cache;
    std::atomic<int> compileCount{0};
    std::array<int, 8> results{};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < results.size(); ++index)
    {
        threads.emplace_back([&cache, &compileCount, &results, index] {
            results[index] = cache.getOrCompile(7, [&compileCount] {
                ++compileCount;
                return 42;
            });
        });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    EXPECT_EQ(compileCount.load(), 1);
    EXPECT_TRUE(std::all_of(results.begin(), results.end(), [](int const result) { return result == 42; }));

    EXPECT_THROW(cache.getOrCompile(9, []() -> int { throw std::runtime_error("expected"); }), std::runtime_error);
    EXPECT_EQ(cache.getOrCompile(9, [] { return 11; }), 11);
}

TEST(Nvfp4A16BlackwellGemvJitTest, DecodesBlackwellN128K64V1LayoutAndSplitK)
{
    if (!isSm110())
    {
        GTEST_SKIP() << "BLACKWELL_N128_K64_V1 execution requires SM110";
    }

    Nvfp4A16BlackwellGemvJitKey const key = makeKey(Nvfp4A16BlackwellGemvDataType::kHALF);
    Nvfp4A16BlackwellGemvJitRunner runner;
    runner.load(compileNvfp4A16BlackwellGemvJitKernel(key));

    size_t const qweightCount = static_cast<size_t>(kTEST_N) * kTEST_K / 2;
    size_t const scaleCount = static_cast<size_t>(kTEST_N) * kTEST_K / 16;
    PinnedBuffer<uint8_t> hostQweights(qweightCount);
    PinnedBuffer<uint8_t> hostScales(scaleCount);
    fillPhysicalLayout(hostQweights, hostScales);
    PinnedBuffer<half> hostActivation(kTEST_K);
    PinnedBuffer<half> hostOutput(kTEST_N);
    PinnedBuffer<float> hostGlobalScale(1);
    hostGlobalScale.get()[0] = kGLOBAL_SCALE;
    for (int32_t k = 0; k < kTEST_K; ++k)
    {
        hostActivation.get()[k] = __float2half_rn(static_cast<float>(k % 13 - 6) / 4096.0F);
    }

    DeviceBuffer<uint8_t> qweights(qweightCount);
    DeviceBuffer<uint8_t> scales(scaleCount);
    DeviceBuffer<half> activation(kTEST_K);
    DeviceBuffer<half> output(kTEST_N);
    DeviceBuffer<float> globalScale(1);
    DeviceBuffer<float> workspace(static_cast<size_t>(4) * kTEST_N);
    NonBlockingStream stream;
    CUDA_CHECK(cudaMemcpyAsync(qweights.get(), hostQweights.get(), qweightCount, cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(scales.get(), hostScales.get(), scaleCount, cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(activation.get(), hostActivation.get(), static_cast<size_t>(kTEST_K) * sizeof(half),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(
        cudaMemcpyAsync(globalScale.get(), hostGlobalScale.get(), sizeof(float), cudaMemcpyHostToDevice, stream.get()));

    for (int32_t const splitK : kSUPPORTED_SPLIT_K)
    {
        size_t const workspaceSize = getNvfp4A16BlackwellGemvJitWorkspaceSize(key, 1, splitK);
        runner.launch(activation.get(), qweights.get(), scales.get(), globalScale.get(), output.get(),
            workspaceSize == 0 ? nullptr : workspace.get(), workspaceSize, 1, splitK, stream.get());
        CUDA_CHECK(cudaMemcpyAsync(hostOutput.get(), output.get(), static_cast<size_t>(kTEST_N) * sizeof(half),
            cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        for (int32_t n = 0; n < kTEST_N; ++n)
        {
            double reference = 0.0;
            for (int32_t k = 0; k < kTEST_K; ++k)
            {
                reference = std::fma(static_cast<double>(__half2float(hostActivation.get()[k])),
                    static_cast<double>(fp4Value(logicalCode(n, k)) * scaleValue(logicalScale(n, k))), reference);
            }
            float const expected = __half2float(__float2half_rn(static_cast<float>(reference * kGLOBAL_SCALE)));
            float const actual = __half2float(hostOutput.get()[n]);
            ASSERT_NEAR(actual, expected, 0.02F) << "splitK=" << splitK << " n=" << n;
        }
    }

    runner.launch(
        activation.get(), qweights.get(), scales.get(), globalScale.get(), output.get(), nullptr, 0, 1, 1, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

TEST(Nvfp4A16BlackwellGemvJitTest, HalfSpecializationsReuseWeightsAcrossM)
{
    if (!isSm110())
    {
        GTEST_SKIP() << "NVFP4-A16 Blackwell GEMV execution requires SM110";
    }
    runConstantSpecializations<half>(Nvfp4A16BlackwellGemvDataType::kHALF, 1);
}

TEST(Nvfp4A16BlackwellGemvJitTest, Bf16SpecializationsReuseWeightsAcrossMWithSplitK)
{
    if (!isSm110())
    {
        GTEST_SKIP() << "NVFP4-A16 Blackwell GEMV execution requires SM110";
    }
    runConstantSpecializations<__nv_bfloat16>(Nvfp4A16BlackwellGemvDataType::kBF16, 4);
}

} // namespace
} // namespace trt_edgellm
