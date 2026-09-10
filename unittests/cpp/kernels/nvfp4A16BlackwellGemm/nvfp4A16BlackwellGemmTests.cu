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
#include "kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm::kernels
{
namespace
{

constexpr int32_t kNumTokens{8};
constexpr int32_t kNtile{128};
constexpr int32_t kKtile{64};
constexpr float kGlobalScale{0.25F};
constexpr std::array<uint8_t, 4> kScaleCodes{0x20U, 0x28U, 0x30U, 0x38U};

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count)
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

float fp4Value(uint8_t code)
{
    constexpr std::array<float, 8> kLevels{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    float const magnitude = kLevels[code & 7U];
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

float scaleValue(uint8_t raw)
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

uint8_t logicalWeightCode(int32_t n, int32_t k)
{
    return static_cast<uint8_t>((n * 7 + k * 11 + 1) & 15);
}

uint8_t logicalScaleCode(int32_t n, int32_t k)
{
    int32_t const k16 = k / 16;
    return kScaleCodes[static_cast<size_t>((n * 3 + k16) & 3)];
}

size_t qweightByteOffset(int32_t n, int32_t k, int32_t inFeatures)
{
    int32_t const kTiles = inFeatures / kKtile;
    int32_t const nTile = n / kNtile;
    int32_t const kTile = k / kKtile;
    int32_t const nWithinTile = n % kNtile;
    int32_t const byteWithinTile = (k % kKtile) / 2;
    return (((static_cast<size_t>(nTile) * kTiles + kTile) * kNtile + nWithinTile) * (kKtile / 2)) + byteWithinTile;
}

size_t scaleByteOffset(int32_t n, int32_t k, int32_t inFeatures)
{
    int32_t const kTiles = inFeatures / kKtile;
    int32_t const nTile = n / kNtile;
    int32_t const kTile = k / kKtile;
    int32_t const nWithinTile = n % kNtile;
    int32_t const groupWithinTile = (k % kKtile) / 16;
    return (((static_cast<size_t>(nTile) * kTiles + kTile) * kNtile + nWithinTile) * (kKtile / 16)) + groupWithinTile;
}

template <typename T>
T fromFloat(float value);

template <>
half fromFloat<half>(float value)
{
    return __float2half_rn(value);
}

template <>
__nv_bfloat16 fromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

template <typename T>
float toFloat(T value);

template <>
float toFloat<half>(half value)
{
    return __half2float(value);
}

template <>
float toFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
void runNonuniformK16ScaleCase(Nvfp4A16BlackwellDtype dtype, int32_t outFeatures, int32_t inFeatures, float tolerance)
{
    if (!isSm110())
    {
        GTEST_SKIP() << "NVFP4-A16 TCGen5 execution requires SM110";
    }
    if (!Nvfp4A16BlackwellGemmRunner::isSupported(110, dtype, kNumTokens, outFeatures, inFeatures))
    {
        GTEST_SKIP() << "matching nvfp4_a16_blackwell_gemm AOT variant is not linked";
    }

    std::vector<uint8_t> hostQweight(static_cast<size_t>(outFeatures) * inFeatures / 2);
    std::vector<uint8_t> hostScale(static_cast<size_t>(outFeatures) * inFeatures / 16);
    for (int32_t n = 0; n < outFeatures; ++n)
    {
        for (int32_t k = 0; k < inFeatures; k += 2)
        {
            uint8_t const low = logicalWeightCode(n, k);
            uint8_t const high = logicalWeightCode(n, k + 1);
            hostQweight[qweightByteOffset(n, k, inFeatures)] = static_cast<uint8_t>(low | (high << 4));
        }
        for (int32_t k = 0; k < inFeatures; k += 16)
        {
            hostScale[scaleByteOffset(n, k, inFeatures)] = logicalScaleCode(n, k);
        }
    }

    std::vector<T> hostActivation(static_cast<size_t>(kNumTokens) * inFeatures);
    std::vector<T> hostOutput(static_cast<size_t>(kNumTokens) * outFeatures);
    for (int32_t token = 0; token < kNumTokens; ++token)
    {
        for (int32_t k = 0; k < inFeatures; ++k)
        {
            hostActivation[static_cast<size_t>(token) * inFeatures + k]
                = fromFloat<T>(static_cast<float>((token + 1) * (k % 13 - 6)) / 256.0F);
        }
    }

    DeviceBuffer<uint8_t> qweight(hostQweight.size());
    DeviceBuffer<uint8_t> scale(hostScale.size());
    DeviceBuffer<T> activation(hostActivation.size());
    DeviceBuffer<T> output(hostOutput.size());
    DeviceBuffer<float> globalScale(1);
    NonBlockingStream stream;

    CUDA_CHECK(
        cudaMemcpyAsync(qweight.get(), hostQweight.data(), hostQweight.size(), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(scale.get(), hostScale.data(), hostScale.size(), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(activation.get(), hostActivation.data(), hostActivation.size() * sizeof(T),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(
        cudaMemcpyAsync(globalScale.get(), &kGlobalScale, sizeof(kGlobalScale), cudaMemcpyHostToDevice, stream.get()));

    ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::loadKernelModules(stream.get()), cudaSuccess);
    Nvfp4A16BlackwellGemmParams const params{activation.get(), qweight.get(), scale.get(), globalScale.get(),
        output.get(), kNumTokens, outFeatures, inFeatures, dtype};
    ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(params), 0U);
    ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::run(params, nullptr, 0, stream.get()), cudaSuccess);
    CUDA_CHECK(cudaMemcpyAsync(
        hostOutput.data(), output.get(), hostOutput.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    for (int32_t token = 0; token < kNumTokens; ++token)
    {
        // Validate every output column.  The scale TMA/transform partition can
        // permute only selected N positions within a 128-row tile, so a small
        // boundary sample is not a sufficient layout correctness gate.
        for (int32_t n = 0; n < outFeatures; ++n)
        {
            double reference{0.0};
            for (int32_t k = 0; k < inFeatures; ++k)
            {
                reference = std::fma(
                    static_cast<double>(toFloat(hostActivation[static_cast<size_t>(token) * inFeatures + k])),
                    static_cast<double>(fp4Value(logicalWeightCode(n, k)) * scaleValue(logicalScaleCode(n, k))),
                    reference);
            }
            float const expected = toFloat(fromFloat<T>(static_cast<float>(reference * kGlobalScale)));
            float const actual = toFloat(hostOutput[static_cast<size_t>(token) * outFeatures + n]);
            ASSERT_NEAR(actual, expected, tolerance) << "token=" << token << " n=" << n;
        }
    }
}

void runK16ScaleImpulseCase()
{
    constexpr int32_t kOutFeatures{3712};
    constexpr int32_t kInFeatures{2688};
    constexpr uint8_t kFp4OnePair{0x22U};
    constexpr uint8_t kE4m3One{0x38U};

    if (!isSm110())
    {
        GTEST_SKIP() << "NVFP4-A16 TCGen5 execution requires SM110";
    }
    if (!Nvfp4A16BlackwellGemmRunner::isSupported(
            110, Nvfp4A16BlackwellDtype::kFp16, kNumTokens, kOutFeatures, kInFeatures))
    {
        GTEST_SKIP() << "matching nvfp4_a16_blackwell_gemm AOT variant is not linked";
    }

    // Keep every logical weight at FP4 1.0. The four K16 groups use distinct
    // activation amplitudes so both broadcast and group-permutation bugs fail.
    std::vector<uint8_t> hostQweight(static_cast<size_t>(kOutFeatures) * kInFeatures / 2, kFp4OnePair);
    std::vector<uint8_t> hostScale(static_cast<size_t>(kOutFeatures) * kInFeatures / 16);
    std::vector<half> hostActivation(static_cast<size_t>(kNumTokens) * kInFeatures);
    std::vector<half> hostOutput(static_cast<size_t>(kNumTokens) * kOutFeatures);
    for (int32_t token = 0; token < kNumTokens; ++token)
    {
        for (int32_t k = 0; k < kInFeatures; ++k)
        {
            int32_t const groupWithinTile = (k % kKtile) / 16;
            float const value = static_cast<float>((token + 1) * (groupWithinTile + 1)) / 256.0F;
            hostActivation[static_cast<size_t>(token) * kInFeatures + k] = fromFloat<half>(value);
        }
    }

    DeviceBuffer<uint8_t> qweight(hostQweight.size());
    DeviceBuffer<uint8_t> scale(hostScale.size());
    DeviceBuffer<half> activation(hostActivation.size());
    DeviceBuffer<half> output(hostOutput.size());
    DeviceBuffer<float> globalScale(1);
    NonBlockingStream stream;

    CUDA_CHECK(
        cudaMemcpyAsync(qweight.get(), hostQweight.data(), hostQweight.size(), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(activation.get(), hostActivation.data(), hostActivation.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(
        cudaMemcpyAsync(globalScale.get(), &kGlobalScale, sizeof(kGlobalScale), cudaMemcpyHostToDevice, stream.get()));
    ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::loadKernelModules(stream.get()), cudaSuccess);

    Nvfp4A16BlackwellGemmParams const params{activation.get(), qweight.get(), scale.get(), globalScale.get(),
        output.get(), kNumTokens, kOutFeatures, kInFeatures, Nvfp4A16BlackwellDtype::kFp16};
    ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(params), 0U);

    std::array<int32_t, 6> const sampledRows{0, 1, 127, 128, kOutFeatures / 2, kOutFeatures - 1};
    for (int32_t activeGroup = 0; activeGroup < kKtile / 16; ++activeGroup)
    {
        SCOPED_TRACE(::testing::Message() << "active K16 group=" << activeGroup);
        std::fill(hostScale.begin(), hostScale.end(), 0U);
        for (int32_t n = 0; n < kOutFeatures; ++n)
        {
            for (int32_t kTile = 0; kTile < kInFeatures; kTile += kKtile)
            {
                int32_t const k = kTile + activeGroup * 16;
                hostScale[scaleByteOffset(n, k, kInFeatures)] = kE4m3One;
            }
        }

        CUDA_CHECK(
            cudaMemcpyAsync(scale.get(), hostScale.data(), hostScale.size(), cudaMemcpyHostToDevice, stream.get()));
        ASSERT_EQ(Nvfp4A16BlackwellGemmRunner::run(params, nullptr, 0, stream.get()), cudaSuccess);
        CUDA_CHECK(cudaMemcpyAsync(
            hostOutput.data(), output.get(), hostOutput.size() * sizeof(half), cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        for (int32_t token = 0; token < kNumTokens; ++token)
        {
            double reference{0.0};
            for (int32_t kTile = 0; kTile < kInFeatures; kTile += kKtile)
            {
                for (int32_t kWithinGroup = 0; kWithinGroup < 16; ++kWithinGroup)
                {
                    int32_t const k = kTile + activeGroup * 16 + kWithinGroup;
                    reference += toFloat(hostActivation[static_cast<size_t>(token) * kInFeatures + k]);
                }
            }
            float const expected = toFloat(fromFloat<half>(static_cast<float>(reference * kGlobalScale)));
            for (int32_t const n : sampledRows)
            {
                float const actual = toFloat(hostOutput[static_cast<size_t>(token) * kOutFeatures + n]);
                ASSERT_NEAR(actual, expected, 0.02F) << "token=" << token << " n=" << n;
            }
        }
    }
}

TEST(Nvfp4A16BlackwellGemmTest, HalfSharedUpNonuniformK16Scales)
{
    runNonuniformK16ScaleCase<half>(Nvfp4A16BlackwellDtype::kFp16, 3712, 2688, 0.02F);
}

TEST(Nvfp4A16BlackwellGemmTest, Bf16SharedDownNonuniformK16Scales)
{
    runNonuniformK16ScaleCase<__nv_bfloat16>(Nvfp4A16BlackwellDtype::kBf16, 2688, 3712, 0.04F);
}

TEST(Nvfp4A16BlackwellGemmTest, HalfSharedUpK16ScaleImpulse)
{
    runK16ScaleImpulseCase();
}

} // namespace
} // namespace trt_edgellm::kernels
