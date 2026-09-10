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

//! Execution coverage for the W4A16 NVFP4 Marlin MoE GEMM on SM12x
//! (DGX Spark / GB10 and GeForce Blackwell).
//!
//! The kernels themselves are SM80-generation and are compiled for every
//! architecture in CMAKE_CUDA_ARCHITECTURES, so no SM12x enablement work is
//! needed. What differs on SM12x is the E4M3 block-scale expansion:
//! nvfp4_scale.cuh keeps its packed BF16 conversion behind an SM110 guard, so
//! BF16 activations take the portable scalar fallback here while FP16 keeps the
//! native cvt. These tests pin down that both arms stay numerically correct.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moe_marlin/marlin/nvfp4_scale.cuh"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include "testUtils.h"

using namespace trt_edgellm;

namespace
{

//! Marlin's E2M1 magnitudes, indexed by the low three bits of a code.
constexpr float kFp4Levels[8]{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

constexpr int32_t kNumExperts{8};
constexpr int32_t kTopK{2};
constexpr int32_t kHiddenSize{256}; //!< GEMM K, must be a multiple of 16.
constexpr int32_t kOutDim{128};     //!< GEMM N, must be a multiple of 64 for the tile fixture.

//! Marlin packs one E4M3 block scale per 16-element K group.
constexpr int32_t kNvfp4GroupSize{16};

//! Un-adjusted per-expert global scale. The kernel receives it pre-multiplied
//! by the activation-specific exponent (see kFp16GlobalScaleExponent below).
constexpr float kGlobalScale{0.125F}; // 2^-3

//! Marlin's skip-flop E2M1 conversion expects the caller to fold these
//! exponents into the global scale (moeMarlin.h).
constexpr int32_t kFp16GlobalScaleExponent{7};
constexpr int32_t kBf16GlobalScaleExponent{119};

//! Raw E4M3 bytes used by the weight fixture. Chosen so the sign/exponent and
//! both mantissa branches of the scalar SM12x expansion are exercised:
//! 0x02 is subnormal, 0x0C and 0x3E carry a non-zero mantissa, 0x28/0x30 do not.
constexpr uint8_t kBlockScaleAlphabet[]{
    0x02, // 2^-8, subnormal
    0x0C, // 3 * 2^-7
    0x28, // 0.25
    0x30, // 0.5
    0x3E, // 1.75
};

bool isSupportedSm()
{
    int32_t const sm = getSMVersion();
    return sm == 120 || sm == 121;
}

//! Decode a raw E4M3 (e4m3fn) byte. Returns NaN for the two NaN encodings.
double e4m3ToDouble(uint8_t const raw)
{
    double const sign = (raw & 0x80U) != 0U ? -1.0 : 1.0;
    uint32_t const exponent = (raw >> 3) & 0xFU;
    uint32_t const mantissa = raw & 0x7U;
    if (exponent == 0xFU && mantissa == 0x7U)
    {
        return std::nan("");
    }
    if (exponent == 0U)
    {
        return sign * std::ldexp(static_cast<double>(mantissa) / 8.0, -6);
    }
    return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, static_cast<int32_t>(exponent) - 7);
}

//! A synthetic projection already laid out in Marlin order.
//!
//! Codes and block scales are constant inside each 16x64 Marlin tile, so the
//! Marlin nibble and block-scale permutations leave every tile unchanged. That
//! keeps this test independent of checkpoint repacking, which is what the
//! Python plugin fixture does too.
struct TileConstantProjection
{
    std::vector<int32_t> packedWeights;    //!< [E, K/16, 2*N] INT32 view of the packed E2M1 codes.
    std::vector<int8_t> packedBlockScales; //!< [E, K/16, N] raw E4M3 bytes.
    std::vector<uint8_t> codes;            //!< [E, N, K] un-permuted E2M1 codes, for the reference.
    std::vector<uint8_t> blockScales;      //!< [E, N, K/16] un-permuted E4M3 bytes, for the reference.
};

TileConstantProjection buildTileConstantProjection(
    int32_t const numExperts, int32_t const sizeN, int32_t const sizeK, std::mt19937& generator)
{
    int32_t const numNTiles = sizeN / 64;
    int32_t const numKTiles = sizeK / kNvfp4GroupSize;
    std::uniform_int_distribution<int32_t> codeDist(0, 15);
    std::uniform_int_distribution<size_t> scaleDist(0, std::size(kBlockScaleAlphabet) - 1);

    std::vector<uint8_t> tileCodes(static_cast<size_t>(numExperts) * numNTiles * numKTiles);
    std::vector<uint8_t> tileScales(tileCodes.size());
    for (size_t i = 0; i < tileCodes.size(); ++i)
    {
        tileCodes[i] = static_cast<uint8_t>(codeDist(generator));
        tileScales[i] = kBlockScaleAlphabet[scaleDist(generator)];
    }
    auto tileAt = [&](int32_t e, int32_t nTile, int32_t kTile) {
        return (static_cast<size_t>(e) * numNTiles + nTile) * numKTiles + kTile;
    };

    TileConstantProjection out;
    // Each packed INT32 covers 128 output columns' worth of one tile, and every
    // nibble in a tile holds the same code, so the word is the code byte
    // broadcast four times.
    out.packedWeights.resize(static_cast<size_t>(numExperts) * numKTiles * (2 * sizeN));
    for (int32_t e = 0; e < numExperts; ++e)
    {
        for (int32_t kTile = 0; kTile < numKTiles; ++kTile)
        {
            for (int32_t i = 0; i < 2 * sizeN; ++i)
            {
                uint8_t const code = tileCodes[tileAt(e, i / 128, kTile)];
                uint32_t const packedByte = static_cast<uint32_t>(code) | (static_cast<uint32_t>(code) << 4);
                out.packedWeights[(static_cast<size_t>(e) * numKTiles + kTile) * (2 * sizeN) + i]
                    = static_cast<int32_t>(packedByte * 0x01010101U);
            }
        }
    }

    out.packedBlockScales.resize(static_cast<size_t>(numExperts) * numKTiles * sizeN);
    for (int32_t e = 0; e < numExperts; ++e)
    {
        for (int32_t kTile = 0; kTile < numKTiles; ++kTile)
        {
            for (int32_t n = 0; n < sizeN; ++n)
            {
                out.packedBlockScales[(static_cast<size_t>(e) * numKTiles + kTile) * sizeN + n]
                    = static_cast<int8_t>(tileScales[tileAt(e, n / 64, kTile)]);
            }
        }
    }

    out.codes.resize(static_cast<size_t>(numExperts) * sizeN * sizeK);
    out.blockScales.resize(static_cast<size_t>(numExperts) * sizeN * numKTiles);
    for (int32_t e = 0; e < numExperts; ++e)
    {
        for (int32_t n = 0; n < sizeN; ++n)
        {
            for (int32_t k = 0; k < sizeK; ++k)
            {
                out.codes[(static_cast<size_t>(e) * sizeN + n) * sizeK + k]
                    = tileCodes[tileAt(e, n / 64, k / kNvfp4GroupSize)];
            }
            for (int32_t kTile = 0; kTile < numKTiles; ++kTile)
            {
                out.blockScales[(static_cast<size_t>(e) * sizeN + n) * numKTiles + kTile]
                    = tileScales[tileAt(e, n / 64, kTile)];
            }
        }
    }
    return out;
}

//! Routing arrays in the layout buildMarlinIndicesKernel produces: per-expert
//! contiguous runs of slot ids padded up to a multiple of moeBlockSize with the
//! sentinel numTokens*topK, one expert id per block, and routing weights
//! indexed by the un-padded slot id.
struct RoutingIndices
{
    std::vector<int32_t> sortedTokenIds;
    std::vector<int32_t> expertIds;
    std::vector<float> topkWeights;
    std::vector<int32_t> slotExpert; //!< Expert per slot, for the reference.
    int32_t numTokensPostPadded{};
};

RoutingIndices buildRoutingIndices(int32_t const numTokens, int32_t const topK, int32_t const numExperts,
    int32_t const moeBlockSize, std::mt19937& generator)
{
    int32_t const totalSlots = numTokens * topK;
    // Round-robin assignment keeps every expert busy and, for the prefill case,
    // yields both fully packed and partially padded blocks.
    std::vector<std::vector<int32_t>> slotsByExpert(numExperts);
    RoutingIndices out;
    out.slotExpert.resize(totalSlots);
    for (int32_t slot = 0; slot < totalSlots; ++slot)
    {
        int32_t const expert = slot % numExperts;
        out.slotExpert[slot] = expert;
        slotsByExpert[expert].push_back(slot);
    }

    // Capacity matches the plugin's conservative bound.
    int32_t const capacity = totalSlots + numExperts * (moeBlockSize - 1);
    out.sortedTokenIds.assign(capacity, totalSlots);
    out.expertIds.assign(static_cast<size_t>(divUp(capacity, moeBlockSize)), 0);

    int32_t cursor = 0;
    for (int32_t expert = 0; expert < numExperts; ++expert)
    {
        auto const& slots = slotsByExpert[expert];
        if (slots.empty())
        {
            continue;
        }
        int32_t const paddedCount
            = static_cast<int32_t>(divUp(static_cast<int32_t>(slots.size()), moeBlockSize)) * moeBlockSize;
        for (size_t i = 0; i < slots.size(); ++i)
        {
            out.sortedTokenIds[cursor + static_cast<int32_t>(i)] = slots[i];
        }
        for (int32_t b = 0; b < paddedCount / moeBlockSize; ++b)
        {
            out.expertIds[cursor / moeBlockSize + b] = expert;
        }
        cursor += paddedCount;
    }
    out.numTokensPostPadded = cursor;

    // Marlin reads topk_weights_ptr by un-padded slot id, so this array is
    // indexed by slot and sized to the padded capacity.
    std::uniform_real_distribution<float> weightDist(0.25F, 1.0F);
    out.topkWeights.assign(capacity, 0.0F);
    for (int32_t slot = 0; slot < totalSlots; ++slot)
    {
        out.topkWeights[slot] = weightDist(generator);
    }
    return out;
}

struct MarlinCase
{
    char const* name;
    int32_t numTokens;
    int64_t moeBlockSize;
    nvinfer1::DataType activationType;
    bool mulTopkWeights;
    uint32_t seed;
};

std::vector<MarlinCase> defaultCases()
{
    // Decode uses the 8-row Marlin tile and prefill the 32-row tile, matching
    // the plugin's block-size choice. The BF16 cases are the load-bearing ones
    // on SM12x: they are the only path that takes the scalar E4M3 -> BF16
    // fallback. mulTopkWeights mirrors the plugin's FC1 (false) and FC2 (true)
    // call sites.
    return {
        {"decode_t1_blk8_fp16", 1, 8, nvinfer1::DataType::kHALF, false, 0xDEADBEEFU},
        {"decode_t1_blk8_bf16", 1, 8, nvinfer1::DataType::kBF16, false, 0xDEADBEEFU},
        {"prefill_t64_blk32_fp16", 64, 32, nvinfer1::DataType::kHALF, false, 0x5EED1234U},
        {"prefill_t64_blk32_bf16", 64, 32, nvinfer1::DataType::kBF16, false, 0x5EED1234U},
        {"decode_t1_blk8_bf16_weighted", 1, 8, nvinfer1::DataType::kBF16, true, 0xC0FFEEU},
        {"prefill_t64_blk32_fp16_weighted", 64, 32, nvinfer1::DataType::kHALF, true, 0xC0FFEEU},
    };
}

//! Host-side view of one case: activations already rounded to the activation
//! type, the packed Marlin inputs, and the routing arrays.
struct CaseData
{
    MarlinCase cfg;
    std::vector<float> activations; //!< [numTokens, K], already rounded to cfg.activationType.
    TileConstantProjection weights;
    RoutingIndices routing;
};

float roundToActivationType(float const value, nvinfer1::DataType const type)
{
    return type == nvinfer1::DataType::kHALF ? __half2float(__float2half(value))
                                             : __bfloat162float(__float2bfloat16(value));
}

CaseData buildCase(MarlinCase const& cfg)
{
    std::mt19937 generator{cfg.seed};
    CaseData data;
    data.cfg = cfg;
    data.weights = buildTileConstantProjection(kNumExperts, kOutDim, kHiddenSize, generator);
    data.routing
        = buildRoutingIndices(cfg.numTokens, kTopK, kNumExperts, static_cast<int32_t>(cfg.moeBlockSize), generator);

    std::uniform_real_distribution<float> inputDist(-1.0F, 1.0F);
    data.activations.resize(static_cast<size_t>(cfg.numTokens) * kHiddenSize);
    for (auto& value : data.activations)
    {
        value = roundToActivationType(inputDist(generator), cfg.activationType);
    }
    return data;
}

//! Dequantize the original codes and accumulate the GEMM in double. The
//! adjusted global-scale exponent cancels against Marlin's skip-flop E2M1
//! conversion, so the reference uses the un-adjusted scale.
std::vector<double> computeReference(CaseData const& data)
{
    int32_t const totalSlots = data.cfg.numTokens * kTopK;
    int32_t const numKTiles = kHiddenSize / kNvfp4GroupSize;
    std::vector<double> reference(static_cast<size_t>(totalSlots) * kOutDim, 0.0);

    for (int32_t slot = 0; slot < totalSlots; ++slot)
    {
        int32_t const token = slot / kTopK;
        int32_t const expert = data.routing.slotExpert[slot];
        for (int32_t n = 0; n < kOutDim; ++n)
        {
            double accumulator = 0.0;
            for (int32_t k = 0; k < kHiddenSize; ++k)
            {
                uint8_t const code = data.weights.codes[(static_cast<size_t>(expert) * kOutDim + n) * kHiddenSize + k];
                double magnitude = kFp4Levels[code & 0x7U];
                if ((code & 0x8U) != 0U)
                {
                    magnitude = -magnitude;
                }
                uint8_t const scale
                    = data.weights
                          .blockScales[(static_cast<size_t>(expert) * kOutDim + n) * numKTiles + k / kNvfp4GroupSize];
                double const weight = magnitude * e4m3ToDouble(scale) * kGlobalScale;
                accumulator
                    += static_cast<double>(data.activations[static_cast<size_t>(token) * kHiddenSize + k]) * weight;
            }
            if (data.cfg.mulTopkWeights)
            {
                accumulator *= static_cast<double>(data.routing.topkWeights[slot]);
            }
            reference[static_cast<size_t>(slot) * kOutDim + n] = accumulator;
        }
    }
    return reference;
}

//! Run one case through moeNvfp4A16MarlinGemm and return the output as FP32.
std::vector<float> runCase(CaseData const& data)
{
    using namespace trt_edgellm::rt;

    MarlinCase const& cfg = data.cfg;
    int32_t const totalSlots = cfg.numTokens * kTopK;
    int32_t const numKTiles = kHiddenSize / kNvfp4GroupSize;
    int64_t const capacity = static_cast<int64_t>(data.routing.sortedTokenIds.size());

    int32_t device = 0;
    int32_t numSms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    int64_t const workspaceElements = kernel::getMoeMarlinWorkspaceSize(capacity, kOutDim, cfg.moeBlockSize, numSms);

    Tensor input({cfg.numTokens, kHiddenSize}, DeviceType::kGPU, cfg.activationType);
    Tensor output({totalSlots, kOutDim}, DeviceType::kGPU, cfg.activationType);
    Tensor weights({kNumExperts, numKTiles, 2 * kOutDim}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor blockScales({kNumExperts, numKTiles, kOutDim}, DeviceType::kGPU, nvinfer1::DataType::kINT8);
    Tensor globalScales({kNumExperts}, DeviceType::kGPU, cfg.activationType);
    Tensor sortedTokenIds({capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor expertIds(
        {static_cast<int64_t>(data.routing.expertIds.size())}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor numTokensPostPadded({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor topkWeights({capacity}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    Tensor workspace({workspaceElements}, DeviceType::kGPU, nvinfer1::DataType::kINT32);

    // The kernel skips padded rows, so start from a known-zero output.
    CUDA_CHECK(
        cudaMemset(output.rawPointer(), 0, output.getShape().volume() * utils::getTypeSize(output.getDataType())));

    if (cfg.activationType == nvinfer1::DataType::kHALF)
    {
        std::vector<half> hostInput(data.activations.size());
        for (size_t i = 0; i < hostInput.size(); ++i)
        {
            hostInput[i] = __float2half(data.activations[i]);
        }
        std::vector<half> hostGlobal(kNumExperts, __float2half(std::ldexp(kGlobalScale, kFp16GlobalScaleExponent)));
        CUDA_CHECK(
            cudaMemcpy(input.rawPointer(), hostInput.data(), hostInput.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            globalScales.rawPointer(), hostGlobal.data(), hostGlobal.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<__nv_bfloat16> hostInput(data.activations.size());
        for (size_t i = 0; i < hostInput.size(); ++i)
        {
            hostInput[i] = __float2bfloat16(data.activations[i]);
        }
        std::vector<__nv_bfloat16> hostGlobal(
            kNumExperts, __float2bfloat16(std::ldexp(kGlobalScale, kBf16GlobalScaleExponent)));
        CUDA_CHECK(cudaMemcpy(
            input.rawPointer(), hostInput.data(), hostInput.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(globalScales.rawPointer(), hostGlobal.data(), hostGlobal.size() * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(weights.rawPointer(), data.weights.packedWeights.data(),
        data.weights.packedWeights.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(blockScales.rawPointer(), data.weights.packedBlockScales.data(),
        data.weights.packedBlockScales.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sortedTokenIds.rawPointer(), data.routing.sortedTokenIds.data(),
        data.routing.sortedTokenIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(expertIds.rawPointer(), data.routing.expertIds.data(),
        data.routing.expertIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        numTokensPostPadded.rawPointer(), &data.routing.numTokensPostPadded, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(topkWeights.rawPointer(), data.routing.topkWeights.data(),
        data.routing.topkWeights.size() * sizeof(float), cudaMemcpyHostToDevice));

    kernel::moeNvfp4A16MarlinGemm(input, output, weights, blockScales, globalScales, sortedTokenIds, expertIds,
        numTokensPostPadded, topkWeights, workspace, cfg.moeBlockSize, kTopK, cfg.mulTopkWeights, /*stream=*/nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(static_cast<size_t>(totalSlots) * kOutDim);
    if (cfg.activationType == nvinfer1::DataType::kHALF)
    {
        std::vector<half> hostOutput(result.size());
        CUDA_CHECK(cudaMemcpy(
            hostOutput.data(), output.rawPointer(), hostOutput.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i] = __half2float(hostOutput[i]);
        }
    }
    else
    {
        std::vector<__nv_bfloat16> hostOutput(result.size());
        CUDA_CHECK(cudaMemcpy(
            hostOutput.data(), output.rawPointer(), hostOutput.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i] = __bfloat162float(hostOutput[i]);
        }
    }
    return result;
}

struct Summary
{
    double medianCosine;
    double relativeL2;
};

Summary summarize(
    std::vector<float> const& got, std::vector<double> const& reference, int32_t const numRows, int32_t const rowWidth)
{
    std::vector<double> cosines;
    cosines.reserve(numRows);
    double errorNorm2 = 0.0;
    double refNorm2 = 0.0;
    for (int32_t row = 0; row < numRows; ++row)
    {
        double dot = 0.0;
        double gotNorm = 0.0;
        double refNorm = 0.0;
        for (int32_t col = 0; col < rowWidth; ++col)
        {
            double const g = got[static_cast<size_t>(row) * rowWidth + col];
            double const r = reference[static_cast<size_t>(row) * rowWidth + col];
            dot += g * r;
            gotNorm += g * g;
            refNorm += r * r;
            errorNorm2 += (g - r) * (g - r);
            refNorm2 += r * r;
        }
        double const denominator = std::sqrt(gotNorm) * std::sqrt(refNorm);
        cosines.push_back(denominator > 0.0 ? dot / std::max(denominator, 1e-30) : 1.0);
    }
    std::sort(cosines.begin(), cosines.end());
    double const median = cosines.empty()
        ? 1.0
        : (cosines.size() % 2 == 0 ? 0.5 * (cosines[cosines.size() / 2 - 1] + cosines[cosines.size() / 2])
                                   : cosines[cosines.size() / 2]);
    return {median, std::sqrt(errorNorm2) / std::max(std::sqrt(refNorm2), 1e-30)};
}

//! Exercise the two E4M3 block-scale expanders over every packed byte pair.
__global__ void expandE4m3ScalesKernel(uint32_t* fp16Out, uint32_t* bf16Out)
{
    int32_t const index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 256)
    {
        return;
    }
    // Pack the code into both halves so the low and high lanes are both covered.
    uint16_t const packed = static_cast<uint16_t>(index) | static_cast<uint16_t>(index << 8);
    fp16Out[index] = kernel::nvfp4::e4m3x2ToFloat16x2Times128(packed);
    bf16Out[index] = kernel::nvfp4::e4m3x2ToBfloat16x2Times128(packed);
}

} // namespace

TEST(Nvfp4A16MoeMarlinSm12xTest, smoke)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm())
    {
        GTEST_SKIP()
            << "SM12x NVFP4 A16 Marlin MoE test requires Spark/GB10 or GeForce Blackwell (SM120/SM121), got SM=" << sm;
    }

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const data = buildCase(cfg);
        std::vector<float> const result = runCase(data);
        ASSERT_EQ(result.size(), static_cast<size_t>(cfg.numTokens) * kTopK * kOutDim);
        bool allZero = true;
        for (float const value : result)
        {
            ASSERT_TRUE(std::isfinite(value)) << "non-finite output in " << cfg.name;
            if (value != 0.0F)
            {
                allZero = false;
            }
        }
        ASSERT_FALSE(allZero) << "output is all-zero for " << cfg.name;
    }
}

TEST(Nvfp4A16MoeMarlinSm12xTest, accuracy)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm())
    {
        GTEST_SKIP()
            << "SM12x NVFP4 A16 Marlin MoE test requires Spark/GB10 or GeForce Blackwell (SM120/SM121), got SM=" << sm;
    }

    // The reference dequantizes the original E2M1 codes and accumulates in
    // double; the kernel accumulates in FP32 and stores one rounded activation
    // value per element, so the residual is dominated by that final rounding.
    // Measured on GB10 (SM121): cosine 1.0 for every case, relative L2 up to
    // 2.9e-4 in FP16 and 1.6e-3 in BF16. The bounds keep ~3x headroom.
    constexpr double kMinCosine = 0.999;
    constexpr double kMaxRelativeL2Fp16 = 1e-3;
    constexpr double kMaxRelativeL2Bf16 = 5e-3;

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const data = buildCase(cfg);
        std::vector<double> const reference = computeReference(data);
        std::vector<float> const result = runCase(data);
        ASSERT_EQ(result.size(), reference.size());

        Summary const summary = summarize(result, reference, cfg.numTokens * kTopK, kOutDim);
        double const maxRelativeL2
            = cfg.activationType == nvinfer1::DataType::kHALF ? kMaxRelativeL2Fp16 : kMaxRelativeL2Bf16;
        std::cout << "[" << cfg.name << "] median_cos=" << summary.medianCosine << " rel_l2=" << summary.relativeL2
                  << std::endl;
        EXPECT_GE(summary.medianCosine, kMinCosine)
            << "median cosine " << summary.medianCosine << " below " << kMinCosine << " for " << cfg.name;
        EXPECT_LE(summary.relativeL2, maxRelativeL2)
            << "relative L2 error " << summary.relativeL2 << " above " << maxRelativeL2 << " for " << cfg.name;
    }
}

//! On SM12x the BF16 expander compiles to the portable scalar fallback while
//! FP16 keeps the native cvt, so the two arms take different code paths here.
//! Both must reproduce value * 128 exactly: every finite E4M3 code scaled by
//! 128 stays exactly representable in FP16 and BF16.
TEST(Nvfp4A16MoeMarlinSm12xTest, e4m3BlockScaleExpansionIsExact)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm())
    {
        GTEST_SKIP()
            << "SM12x NVFP4 A16 Marlin MoE test requires Spark/GB10 or GeForce Blackwell (SM120/SM121), got SM=" << sm;
    }

    uint32_t* deviceFp16 = nullptr;
    uint32_t* deviceBf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceFp16, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&deviceBf16, 256 * sizeof(uint32_t)));
    expandE4m3ScalesKernel<<<1, 256>>>(deviceFp16, deviceBf16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> hostFp16(256);
    std::vector<uint32_t> hostBf16(256);
    CUDA_CHECK(cudaMemcpy(hostFp16.data(), deviceFp16, hostFp16.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostBf16.data(), deviceBf16, hostBf16.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceFp16));
    CUDA_CHECK(cudaFree(deviceBf16));

    for (int32_t code = 0; code < 256; ++code)
    {
        uint8_t const raw = static_cast<uint8_t>(code);
        double const decoded = e4m3ToDouble(raw);
        if (std::isnan(decoded))
        {
            // 0x7F and 0xFF are E4M3 NaN. They are not valid NVFP4 block-scale
            // encodings and the expanders are not required to propagate NaN.
            continue;
        }
        float const expected = static_cast<float>(decoded * 128.0);
        SCOPED_TRACE(::testing::Message() << "e4m3_code=0x" << std::hex << code);

        half const expectedHalf = __float2half(expected);
        uint16_t expectedHalfBits = 0;
        std::memcpy(&expectedHalfBits, &expectedHalf, sizeof(expectedHalfBits));
        EXPECT_EQ(static_cast<uint16_t>(hostFp16[code] & 0xFFFFU), expectedHalfBits) << "FP16 low lane";
        EXPECT_EQ(static_cast<uint16_t>(hostFp16[code] >> 16), expectedHalfBits) << "FP16 high lane";

        __nv_bfloat16 const expectedBf16 = __float2bfloat16(expected);
        uint16_t expectedBf16Bits = 0;
        std::memcpy(&expectedBf16Bits, &expectedBf16, sizeof(expectedBf16Bits));
        EXPECT_EQ(static_cast<uint16_t>(hostBf16[code] & 0xFFFFU), expectedBf16Bits) << "BF16 low lane";
        EXPECT_EQ(static_cast<uint16_t>(hostBf16[code] >> 16), expectedBf16Bits) << "BF16 high lane";
    }
}
