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

#include <cuda_runtime.h>

#if defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

#include <utility>
#include <vector>

namespace
{

struct LaunchConfigSnapshot
{
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    std::vector<cudaLaunchAttribute> attrs;
};

thread_local bool gCaptureLaunchConfigs{false};
thread_local std::vector<LaunchConfigSnapshot> gLaunchConfigs;

void beginLaunchConfigCapture()
{
    gLaunchConfigs.clear();
    gCaptureLaunchConfigs = true;
}

std::vector<LaunchConfigSnapshot> endLaunchConfigCapture()
{
    gCaptureLaunchConfigs = false;
    std::vector<LaunchConfigSnapshot> result = std::move(gLaunchConfigs);
    gLaunchConfigs.clear();
    return result;
}

void captureLaunchConfig(cudaLaunchConfig_t const* config)
{
    if (!gCaptureLaunchConfigs)
    {
        return;
    }

    LaunchConfigSnapshot snapshot{config->gridDim, config->blockDim, config->dynamicSmemBytes, {}};
    snapshot.attrs.assign(config->attrs, config->attrs + config->numAttrs);
    gLaunchConfigs.emplace_back(std::move(snapshot));
}

class ScopedLaunchConfigCapture
{
public:
    ScopedLaunchConfigCapture()
    {
        beginLaunchConfigCapture();
    }

    ScopedLaunchConfigCapture(ScopedLaunchConfigCapture const&) = delete;
    ScopedLaunchConfigCapture& operator=(ScopedLaunchConfigCapture const&) = delete;

    ~ScopedLaunchConfigCapture()
    {
        if (mActive)
        {
            (void) endLaunchConfigCapture();
        }
    }

    std::vector<LaunchConfigSnapshot> finish()
    {
        mActive = false;
        return endLaunchConfigCapture();
    }

private:
    bool mActive{true};
};

} // namespace

extern "C" cudaError_t __real__cudaLaunchKernelEx(
    cudaLaunchConfig_t const* config, cudaKernel_t kernel, void** kernelParams);

extern "C" cudaError_t __wrap__cudaLaunchKernelEx(
    cudaLaunchConfig_t const* config, cudaKernel_t kernel, void** kernelParams)
{
    captureLaunchConfig(config);
    return __real__cudaLaunchKernelEx(config, kernel, kernelParams);
}

#endif // defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#ifdef ENABLE_NVTX_PROFILING
#include <nvtx3.hpp>
#endif
#include <random>
#include <string>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.h"
#include "testUtils.h"

using namespace trt_edgellm;

namespace
{

// ---------------------------------------------------------------------------
// NVFP4 MoE plugin constants. These mirror the runner constants in
// cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.h so the C++
// reference reads back exactly what the kernel consumes.
// ---------------------------------------------------------------------------
constexpr int32_t kSfVecSize = 16;
constexpr int32_t kRowTile = 128;
constexpr int32_t kNumExperts = 128;
constexpr int32_t kTopK = 8;
constexpr int32_t kActSwiGLU = 2;
constexpr int32_t kActReLU2 = 4;
constexpr int32_t kActGeGLU = 5;
constexpr int32_t kIoDtypeFp16 = 1;
constexpr int32_t kBackendAuto = 0;
constexpr int32_t kGraphReplayCount = 8;
constexpr int32_t kOutputPoisonByte = 0xA5;

constexpr std::array<float, 16> kFp4Levels{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ---------------------------------------------------------------------------
// FP8 E4M3 helpers (match torch.float8_e4m3fn round-to-nearest-even encoding).
// ---------------------------------------------------------------------------
inline float fp8BytesToFloat(uint8_t raw)
{
    __nv_fp8_e4m3 fp8;
    fp8.__x = raw;
    return static_cast<float>(static_cast<__half>(fp8));
}

inline bool isSupportedSm(int32_t sm)
{
    return sm == 100 || sm == 101 || sm == 110;
}

inline uint8_t floatToFp8Byte(float value)
{
    __nv_fp8_e4m3 fp8(value);
    return fp8.__x;
}

// ---------------------------------------------------------------------------
// 6D NVFP4 scale-factor layout helpers. The swizzle matches the MMA atom
// layout consumed by the FC1/FC2 wrappers in
// cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.cpp so the
// reference can read the same FP8 SF bytes the kernel sees.
// ---------------------------------------------------------------------------
struct ScaleShape
{
    int32_t experts;
    int32_t mOuter;
    int32_t kOuter;
    int32_t inner0; // 32
    int32_t inner1; // 4
    int32_t inner2; // 4
    int64_t volume() const
    {
        return static_cast<int64_t>(experts) * mOuter * kOuter * inner0 * inner1 * inner2;
    }
};

inline ScaleShape scaleShape(int32_t rows, int32_t cols, int32_t experts = kNumExperts)
{
    int32_t const sfCols = cols / kSfVecSize;
    int32_t const mOuter = (rows + kRowTile - 1) / kRowTile;
    int32_t const kOuter = (sfCols + 3) / 4;
    return {experts, mOuter, kOuter, 32, 4, 4};
}

inline int64_t atomOffset(int32_t row, int32_t sfCol, int32_t sfCols)
{
    int64_t const innerK = sfCol % 4;
    int64_t const innerM = (row % kRowTile) / 32;
    int64_t const outerM = row % 32;
    int64_t const kTile = sfCol / 4;
    int64_t const numKTiles = (sfCols + 3) / 4;
    int64_t const mTile = row / kRowTile;
    return mTile * numKTiles * 512 + kTile * 512 + outerM * 16 + innerM * 4 + innerK;
}

// ---------------------------------------------------------------------------
// Deterministic test-input generators (uniform / N(0, sigma) PRNG, fixed seed
// so the C++ reference always reproduces the same bytes the kernel sees).
// ---------------------------------------------------------------------------
std::vector<uint8_t> makeQWeights(std::mt19937& rng, size_t numBytes)
{
    std::vector<uint8_t> out(numBytes);
    std::uniform_int_distribution<int32_t> dist(0, 255);
    for (size_t i = 0; i < numBytes; ++i)
    {
        out[i] = static_cast<uint8_t>(dist(rng));
    }
    return out;
}

std::vector<uint8_t> makeScaleTensorUniform(int32_t rows, int32_t cols, int32_t experts = kNumExperts)
{
    ScaleShape const shape = scaleShape(rows, cols, experts);
    std::vector<uint8_t> out(static_cast<size_t>(shape.volume()), 0);
    // Use the same FP8 byte value Python uses when non_uniform=False: 0.0078125.
    uint8_t const byteValue = floatToFp8Byte(0.0078125f);
    std::fill(out.begin(), out.end(), byteValue);
    return out;
}

// ---------------------------------------------------------------------------
// FP4 quantize / dequantize helpers (mirror python round_fp4,
// fp4_roundtrip_linear_sf_raw, dequant_weight).
// ---------------------------------------------------------------------------
inline uint8_t roundFp4(float value)
{
    uint8_t best = 0;
    float bestDist = std::fabs(value - kFp4Levels[0]);
    for (uint8_t i = 1; i < kFp4Levels.size(); ++i)
    {
        float const dist = std::fabs(value - kFp4Levels[i]);
        if (dist < bestDist)
        {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

// Roundtrip a [rows, cols] float buffer through NVFP4-with-linear-SF
// quantization and dequantization, returning the dequantized representation
// EXCLUDING the global scale (matches python fp4_roundtrip_linear_sf_raw).
std::vector<float> fp4RoundtripLinearSfRaw(float const* values, int32_t rows, int32_t cols, float globalScale)
{
    std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
    float const scale = std::max(globalScale, 1e-12f);
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t begin = 0; begin < cols; begin += kSfVecSize)
        {
            float const* block = values + static_cast<size_t>(row) * cols + begin;
            float vecMax = 0.0f;
            for (int32_t i = 0; i < kSfVecSize; ++i)
            {
                vecMax = std::max(vecMax, std::fabs(block[i]));
            }
            if (vecMax == 0.0f)
            {
                continue;
            }
            float const sfValue = (vecMax / 6.0f) / scale;
            float const sfBack = fp8BytesToFloat(floatToFp8Byte(sfValue));
            float const effectiveScale = sfBack * scale;
            if (effectiveScale <= 0.0f || !std::isfinite(effectiveScale))
            {
                continue;
            }
            for (int32_t i = 0; i < kSfVecSize; ++i)
            {
                uint8_t const code = roundFp4(block[i] / effectiveScale);
                out[static_cast<size_t>(row) * cols + begin + i] = kFp4Levels[code] * sfBack;
            }
        }
    }
    return out;
}

// Dequantize one expert's FP4 weight tile to a float [rows, cols] matrix,
// including the per-block FP8 SF and per-expert weight alpha (mirrors
// python dequant_weight).
std::vector<float> dequantWeight(
    uint8_t const* qWeights, uint8_t const* scale6d, int32_t expert, int32_t rows, int32_t cols, float alpha)
{
    int32_t const sfCols = cols / kSfVecSize;
    size_t const packedPerExpert = static_cast<size_t>(rows) * (cols / 2);
    // Per-expert SF byte count (expert-count independent): the volume for a
    // single expert in the 6D layout.
    size_t const sfPerExpert = static_cast<size_t>(scaleShape(rows, cols, /*experts=*/1).volume());
    uint8_t const* packed = qWeights + static_cast<size_t>(expert) * packedPerExpert;
    uint8_t const* sfBase = scale6d + static_cast<size_t>(expert) * sfPerExpert;

    std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t col = 0; col < cols; col += 2)
        {
            uint8_t const byte = packed[static_cast<size_t>(row) * (cols / 2) + col / 2];
            uint8_t const lo = byte & 0x0Fu;
            uint8_t const hi = (byte >> 4) & 0x0Fu;
            float const sfLo = fp8BytesToFloat(sfBase[atomOffset(row, col / kSfVecSize, sfCols)]);
            float const sfHi = fp8BytesToFloat(sfBase[atomOffset(row, (col + 1) / kSfVecSize, sfCols)]);
            out[static_cast<size_t>(row) * cols + col] = kFp4Levels[lo] * sfLo * alpha;
            out[static_cast<size_t>(row) * cols + col + 1] = kFp4Levels[hi] * sfHi * alpha;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Activations.
// ---------------------------------------------------------------------------
inline float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

inline float gelu(float x)
{
    // GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    float const inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// SwiGLU with 64-wide up/gate interleave used by the SM110 FC1 layout:
// reshape projection to [n1/128, 128], then up = chunk[:64], gate = chunk[64:].
std::vector<float> applySwigluInterleaved(float const* projection, int32_t n1)
{
    if (n1 % 128 != 0)
    {
        throw std::runtime_error("SwiGLU projection must be 128-interleaved");
    }
    int32_t const numChunks = n1 / 128;
    std::vector<float> out(static_cast<size_t>(numChunks) * 64, 0.0f);
    for (int32_t chunk = 0; chunk < numChunks; ++chunk)
    {
        float const* up = projection + static_cast<size_t>(chunk) * 128;
        float const* gate = up + 64;
        float* o = out.data() + static_cast<size_t>(chunk) * 64;
        for (int32_t i = 0; i < 64; ++i)
        {
            o[i] = up[i] * silu(gate[i]);
        }
    }
    return out;
}

// GeGLU with 64-wide up/gate interleave (same layout as SwiGLU, GELU activation on gate).
std::vector<float> applyGegluInterleaved(float const* projection, int32_t n1)
{
    if (n1 % 128 != 0)
    {
        throw std::runtime_error("GeGLU projection must be 128-interleaved");
    }
    int32_t const numChunks = n1 / 128;
    std::vector<float> out(static_cast<size_t>(numChunks) * 64, 0.0f);
    for (int32_t chunk = 0; chunk < numChunks; ++chunk)
    {
        float const* up = projection + static_cast<size_t>(chunk) * 128;
        float const* gate = up + 64;
        float* o = out.data() + static_cast<size_t>(chunk) * 64;
        for (int32_t i = 0; i < 64; ++i)
        {
            o[i] = up[i] * gelu(gate[i]);
        }
    }
    return out;
}

inline int32_t fc1InputN(int32_t intermediateSize, int32_t activationType)
{
    return (activationType == kActSwiGLU || activationType == kActGeGLU) ? 2 * intermediateSize : intermediateSize;
}

// ---------------------------------------------------------------------------
// Test case definition + deterministic generator.
// ---------------------------------------------------------------------------
struct MoeCase
{
    std::string name;
    int32_t numTokens;
    int32_t hiddenSize;
    int32_t intermediateSize;
    int32_t activationType;
    int32_t numExperts;
    uint64_t seed;
};

struct CaseData
{
    MoeCase config;
    int32_t n1; // FC1 output rows (per expert)
    // Inputs / metadata.
    std::vector<__half> hiddenFp16;      // [T, H]
    std::vector<int32_t> topkIds;        // [T, topK]
    std::vector<float> topkWeights;      // [T, topK]
    std::vector<uint8_t> fc1QWeights;    // [E, n1, H/2]
    std::vector<uint8_t> fc1Scale;       // 6D layout, bytes
    std::vector<uint8_t> fc2QWeights;    // [E, H, I/2]
    std::vector<uint8_t> fc2Scale;       // 6D layout, bytes
    std::vector<float> fc1Alpha;         // [E]
    std::vector<float> fc2Alpha;         // [E]
    std::vector<float> inputGlobalScale; // [E]
    std::vector<float> downInputScale;   // [E]
};

CaseData buildCase(MoeCase const& cfg)
{
    CaseData c{};
    c.config = cfg;
    c.n1 = fc1InputN(cfg.intermediateSize, cfg.activationType);
    int32_t const E = cfg.numExperts;

    std::mt19937 rng(cfg.seed);

    // Hidden states ~ N(0, 0.05) to match the python helper.
    c.hiddenFp16.resize(static_cast<size_t>(cfg.numTokens) * cfg.hiddenSize);
    std::vector<float> hiddenFp32(c.hiddenFp16.size());
    {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        float maxAbs = 0.0f;
        for (size_t i = 0; i < hiddenFp32.size(); ++i)
        {
            hiddenFp32[i] = dist(rng);
            c.hiddenFp16[i] = __float2half_rn(hiddenFp32[i]);
            maxAbs = std::max(maxAbs, std::fabs(hiddenFp32[i]));
        }
        // input_global_scale uses the python floor formula:
        // max(|hidden| / (448 * 6), 1e-12). Per-expert uniform (non_uniform=False).
        float const inputFloor = std::max(maxAbs / (448.0f * 6.0f), 1e-12f);
        c.inputGlobalScale.assign(E, inputFloor);
    }

    // Routing: each token gets kTopK distinct experts (random permutation),
    // uniform-random scores that are renormalized to sum=1 per token.
    c.topkIds.resize(static_cast<size_t>(cfg.numTokens) * kTopK);
    c.topkWeights.resize(static_cast<size_t>(cfg.numTokens) * kTopK);
    {
        std::vector<int32_t> pool(E);
        for (int32_t i = 0; i < E; ++i)
        {
            pool[i] = i;
        }
        std::uniform_real_distribution<float> wDist(0.05f, 1.0f);
        for (int32_t t = 0; t < cfg.numTokens; ++t)
        {
            std::shuffle(pool.begin(), pool.end(), rng);
            float wSum = 0.0f;
            for (int32_t k = 0; k < kTopK; ++k)
            {
                c.topkIds[static_cast<size_t>(t) * kTopK + k] = pool[k];
                float const w = wDist(rng);
                c.topkWeights[static_cast<size_t>(t) * kTopK + k] = w;
                wSum += w;
            }
            for (int32_t k = 0; k < kTopK; ++k)
            {
                c.topkWeights[static_cast<size_t>(t) * kTopK + k] /= wSum;
            }
        }
    }

    // Packed FP4 weights (random bytes covering all nibble combinations).
    c.fc1QWeights = makeQWeights(rng, static_cast<size_t>(E) * c.n1 * (cfg.hiddenSize / 2));
    c.fc2QWeights = makeQWeights(rng, static_cast<size_t>(E) * cfg.hiddenSize * (cfg.intermediateSize / 2));

    // Uniform per-block FP8 SF tensor (single representative byte).
    c.fc1Scale = makeScaleTensorUniform(c.n1, cfg.hiddenSize, E);
    c.fc2Scale = makeScaleTensorUniform(cfg.hiddenSize, cfg.intermediateSize, E);

    // Per-expert weight alphas and FC2 activation scale (non_uniform=False).
    c.fc1Alpha.assign(E, 0.85f);
    c.fc2Alpha.assign(E, 0.75f);
    c.downInputScale.assign(E, 1.0e-4f);
    return c;
}

// ---------------------------------------------------------------------------
// CPU reference: float[T, H]. For each (token, top-K slot) the path is
// FP4-quantize hidden -> FC1 (dequantized weight @ hidden) * input_global_scale
// -> SwiGLU or ReLU2 -> FP4-quantize activation -> FC2 -> router-weighted
// scatter-add. The FP4 / FP8 SF / alpha math matches the SM110 wrappers so the
// kernel and the reference agree up to FP4 rounding noise.
// ---------------------------------------------------------------------------
std::vector<float> computeReference(CaseData const& c)
{
    int32_t const T = c.config.numTokens;
    int32_t const H = c.config.hiddenSize;
    int32_t const I = c.config.intermediateSize;
    int32_t const n1 = c.n1;

    std::vector<float> hiddenFp32(static_cast<size_t>(T) * H);
    for (size_t i = 0; i < hiddenFp32.size(); ++i)
    {
        hiddenFp32[i] = __half2float(c.hiddenFp16[i]);
    }

    std::vector<float> output(static_cast<size_t>(T) * H, 0.0f);

    for (int32_t t = 0; t < T; ++t)
    {
        for (int32_t slot = 0; slot < kTopK; ++slot)
        {
            int32_t const expert = c.topkIds[static_cast<size_t>(t) * kTopK + slot];
            float const routerWeight = c.topkWeights[static_cast<size_t>(t) * kTopK + slot];
            if (routerWeight == 0.0f)
            {
                continue;
            }
            float const inputScale = c.inputGlobalScale[expert];
            float const downScale = c.downInputScale[expert];

            // FP4-quantize the hidden row for this expert.
            std::vector<float> hiddenRaw
                = fp4RoundtripLinearSfRaw(hiddenFp32.data() + static_cast<size_t>(t) * H, 1, H, inputScale);

            // FC1: projection[n1] = (fc1_weight @ hidden_raw) * input_global_scale.
            std::vector<float> fc1W
                = dequantWeight(c.fc1QWeights.data(), c.fc1Scale.data(), expert, n1, H, c.fc1Alpha[expert]);
            std::vector<float> projection(n1, 0.0f);
            for (int32_t row = 0; row < n1; ++row)
            {
                float acc = 0.0f;
                float const* wRow = fc1W.data() + static_cast<size_t>(row) * H;
                for (int32_t k = 0; k < H; ++k)
                {
                    acc += wRow[k] * hiddenRaw[k];
                }
                projection[row] = acc * inputScale;
            }

            // Activation: SwiGLU/GeGLU interleaved (64 up | 64 gate per 128), or ReLU2.
            std::vector<float> activated;
            if (c.config.activationType == kActSwiGLU)
            {
                activated = applySwigluInterleaved(projection.data(), n1);
            }
            else if (c.config.activationType == kActGeGLU)
            {
                activated = applyGegluInterleaved(projection.data(), n1);
            }
            else
            {
                // ReLU2: square(max(x, 0)).
                activated.resize(static_cast<size_t>(I));
                for (int32_t i = 0; i < I; ++i)
                {
                    float const v = std::max(projection[i], 0.0f);
                    activated[i] = v * v;
                }
            }

            // FC2: out += router_weight * down_scale * (fc2_weight @ activated_raw).
            std::vector<float> activatedRaw = fp4RoundtripLinearSfRaw(activated.data(), 1, I, downScale);
            std::vector<float> fc2W
                = dequantWeight(c.fc2QWeights.data(), c.fc2Scale.data(), expert, H, I, c.fc2Alpha[expert]);
            float* outRow = output.data() + static_cast<size_t>(t) * H;
            for (int32_t row = 0; row < H; ++row)
            {
                float acc = 0.0f;
                float const* wRow = fc2W.data() + static_cast<size_t>(row) * I;
                for (int32_t k = 0; k < I; ++k)
                {
                    acc += wRow[k] * activatedRaw[k];
                }
                outRow[row] += routerWeight * downScale * acc;
            }
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Runner driver: upload inputs, run the kernel, return FP32 output [T*H].
// ---------------------------------------------------------------------------
struct RunResult
{
    int32_t status;
    std::vector<uint16_t> outputFp16Bits; // [T*H]
    std::vector<float> outputFp32;        // [T*H]
    std::vector<uint16_t> outputPrefixGuardFp16Bits;
    std::vector<uint16_t> outputSuffixGuardFp16Bits;
};

enum class ExecutionMode
{
    kEager,
    kCudaGraph,
};

RunResult readOutput(int32_t status, void const* deviceOutput, size_t numElements)
{
    std::vector<__half> hostFp16(numElements);
    CUDA_CHECK(cudaMemcpy(hostFp16.data(), deviceOutput, hostFp16.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<uint16_t> hostFp16Bits(hostFp16.size());
    std::memcpy(hostFp16Bits.data(), hostFp16.data(), hostFp16Bits.size() * sizeof(uint16_t));

    std::vector<float> hostFp32(hostFp16.size());
    for (size_t i = 0; i < hostFp16.size(); ++i)
    {
        hostFp32[i] = __half2float(hostFp16[i]);
    }
    return {status, std::move(hostFp16Bits), std::move(hostFp32)};
}

RunResult runCase(CaseData const& c, bool enablePdl = false, ExecutionMode mode = ExecutionMode::kEager,
    char const* nvtxLabel = nullptr)
{
    using rt::Coords;
    using rt::DeviceType;
    int32_t const T = c.config.numTokens;
    int32_t const H = c.config.hiddenSize;
    int32_t const I = c.config.intermediateSize;
    int32_t const n1 = c.n1;
    int32_t const E = c.config.numExperts;
    int32_t const sfShapeOuterH = (n1 + kRowTile - 1) / kRowTile;
    int32_t const sfShapeOuterI = (H + kRowTile - 1) / kRowTile;
    (void) sfShapeOuterH;
    (void) sfShapeOuterI;

    // Allocate GPU buffers and upload.
    rt::Tensor hidden({T, H}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor topkIds({T, kTopK}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor topkWeights({T, kTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fc1Q({E, n1, H / 2}, DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor fc2Q({E, H, I / 2}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    auto const fc1ScaleShape = scaleShape(n1, H, E);
    auto const fc2ScaleShape = scaleShape(H, I, E);
    rt::Tensor fc1Scale({fc1ScaleShape.experts, fc1ScaleShape.mOuter, fc1ScaleShape.kOuter, fc1ScaleShape.inner0,
                            fc1ScaleShape.inner1, fc1ScaleShape.inner2},
        DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor fc2Scale({fc2ScaleShape.experts, fc2ScaleShape.mOuter, fc2ScaleShape.kOuter, fc2ScaleShape.inner0,
                            fc2ScaleShape.inner1, fc2ScaleShape.inner2},
        DeviceType::kGPU, nvinfer1::DataType::kINT8);

    rt::Tensor fc1Alpha({E}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fc2Alpha({E}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor inputScale({E}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor downScale({E}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    constexpr int32_t kOutputGuardElements = 16;
    size_t const outputElements = static_cast<size_t>(T) * H;
    size_t const outputPrefixElements = kOutputGuardElements;
    size_t const outputSuffixElements = kOutputGuardElements;
    void* outputAllocation = nullptr;
    CUDA_CHECK(
        cudaMalloc(&outputAllocation, (outputPrefixElements + outputElements + outputSuffixElements) * sizeof(__half)));
    auto outputGuard = Defer([&] { cudaFree(outputAllocation); });
    auto* const outputBase = static_cast<__half*>(outputAllocation);
    void* const outputPtr = outputBase + outputPrefixElements;

    CUDA_CHECK(cudaMemcpy(
        hidden.rawPointer(), c.hiddenFp16.data(), c.hiddenFp16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(topkIds.rawPointer(), c.topkIds.data(), c.topkIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        topkWeights.rawPointer(), c.topkWeights.data(), c.topkWeights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc1Q.rawPointer(), c.fc1QWeights.data(), c.fc1QWeights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc2Q.rawPointer(), c.fc2QWeights.data(), c.fc2QWeights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc1Scale.rawPointer(), c.fc1Scale.data(), c.fc1Scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc2Scale.rawPointer(), c.fc2Scale.data(), c.fc2Scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        fc1Alpha.rawPointer(), c.fc1Alpha.data(), c.fc1Alpha.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        fc2Alpha.rawPointer(), c.fc2Alpha.data(), c.fc2Alpha.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputScale.rawPointer(), c.inputGlobalScale.data(), c.inputGlobalScale.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(downScale.rawPointer(), c.downInputScale.data(), c.downInputScale.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputAllocation, kOutputPoisonByte,
        (outputPrefixElements + outputElements + outputSuffixElements) * sizeof(__half)));

    // Allocate runner workspace.
    size_t const workspaceBytes = CuteDslNvfp4MoeSm110Runner::getWorkspaceSize(T, T * kTopK, E, kTopK, H, I);
    EXPECT_GT(workspaceBytes, 0u);
    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));
    auto workspaceGuard = Defer([&] { cudaFree(workspace); });

    CuteDslNvfp4MoeSm110Params params{};
    params.numTokens = T;
    params.numExperts = E;
    params.topK = kTopK;
    params.hiddenSize = H;
    params.moeInterSize = I;
    params.hiddenStates = hidden.rawPointer();
    params.topkIds = topkIds.dataPointer<int32_t>();
    params.topkWeights = topkWeights.dataPointer<float>();
    params.fc1QWeights = fc1Q.rawPointer();
    params.fc1BlocksScale = fc1Scale.rawPointer();
    params.fc1Alpha = fc1Alpha.dataPointer<float>();
    params.fc2QWeights = fc2Q.rawPointer();
    params.fc2BlocksScale = fc2Scale.rawPointer();
    params.fc2Alpha = fc2Alpha.dataPointer<float>();
    params.inputGlobalScale = inputScale.dataPointer<float>();
    params.downInputScale = downScale.dataPointer<float>();
    params.output = outputPtr;
    params.activationType = c.config.activationType;
    params.enablePdl = enablePdl;

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto streamGuard = Defer([&] { cudaStreamDestroy(stream); });

    CuteDslNvfp4MoeSm110Runner runner{};
    auto readWithGuards = [&](int32_t status) {
        RunResult result = readOutput(status, outputPtr, outputElements);
        result.outputPrefixGuardFp16Bits.resize(outputPrefixElements);
        result.outputSuffixGuardFp16Bits.resize(outputSuffixElements);
        CUDA_CHECK(cudaMemcpy(result.outputPrefixGuardFp16Bits.data(), outputAllocation,
            outputPrefixElements * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(result.outputSuffixGuardFp16Bits.data(), outputBase + outputPrefixElements + outputElements,
                outputSuffixElements * sizeof(__half), cudaMemcpyDeviceToHost));
        return result;
    };
    if (mode == ExecutionMode::kEager)
    {
        auto launchAndRead = [&]() {
            int32_t const ret = runner.run(params, workspace, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return readWithGuards(ret);
        };
#ifdef ENABLE_NVTX_PROFILING
        if (nvtxLabel != nullptr)
        {
            nvtx3::scoped_range nvtxRange{nvtxLabel};
            return launchAndRead();
        }
#else
        (void) nvtxLabel;
#endif
        return launchAndRead();
    }

    // First-use module loading and cached device queries are not part of the capturable launch path.
    if (!CuteDslNvfp4MoeSm110Runner::ensureKernelModules(params, stream))
    {
        return {-1, {}, {}, {}, {}};
    }
    (void) getSMVersion();
    (void) getDeviceMultiProcessorCount();

    int32_t ret = runner.run(params, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ret != 0)
    {
        return readWithGuards(ret);
    }

    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ret = runner.run(params, workspace, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    auto graphGuard = Defer([&] {
        if (graph != nullptr)
        {
            cudaGraphDestroy(graph);
        }
    });
    if (ret != 0)
    {
        return readWithGuards(ret);
    }

    cudaGraphExec_t graphExec{};
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    auto graphExecGuard = Defer([&] {
        if (graphExec != nullptr)
        {
            cudaGraphExecDestroy(graphExec);
        }
    });

    RunResult firstReplay{};
    size_t const outputBytes = outputElements * sizeof(__half);
    for (int32_t replay = 0; replay < kGraphReplayCount; ++replay)
    {
        // Poison every graph-produced buffer so a missing setup or FC1 node
        // cannot pass by reusing values left by warmup or an earlier replay.
        CUDA_CHECK(cudaMemsetAsync(workspace, kOutputPoisonByte, workspaceBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(outputPtr, kOutputPoisonByte, outputBytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        RunResult replayResult = readWithGuards(ret);
        if (replay == 0)
        {
            firstReplay = std::move(replayResult);
        }
        else
        {
            EXPECT_EQ(replayResult.outputFp16Bits, firstReplay.outputFp16Bits)
                << "CUDA Graph output changed on replay " << replay;
        }
    }
    return firstReplay;
}

// ---------------------------------------------------------------------------
// Per-token median-cosine + magnitude-ratio summary. medianCosine catches sign
// / routing / activation regressions even if the magnitudes drift; magRatio
// catches per-block FP8 SF or per-expert alpha / global-scale regressions even
// if the directions stay aligned.
// ---------------------------------------------------------------------------
struct Summary
{
    double medianCosine;
    double magRatio;
    double maxAbs;
};

Summary summarize(std::vector<float> const& got, std::vector<float> const& ref, int32_t T, int32_t H)
{
    std::vector<double> cosines;
    cosines.reserve(static_cast<size_t>(T));
    double gotNorm2 = 0.0;
    double refNorm2 = 0.0;
    double maxAbs = 0.0;
    for (int32_t t = 0; t < T; ++t)
    {
        double dot = 0.0;
        double gn = 0.0;
        double rn = 0.0;
        for (int32_t h = 0; h < H; ++h)
        {
            float const g = got[static_cast<size_t>(t) * H + h];
            float const r = ref[static_cast<size_t>(t) * H + h];
            dot += static_cast<double>(g) * static_cast<double>(r);
            gn += static_cast<double>(g) * static_cast<double>(g);
            rn += static_cast<double>(r) * static_cast<double>(r);
            maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(g - r)));
        }
        gotNorm2 += gn;
        refNorm2 += rn;
        double const denom = std::sqrt(gn) * std::sqrt(rn);
        cosines.push_back(denom > 0.0 ? dot / std::max(denom, 1e-30) : 1.0);
    }
    std::sort(cosines.begin(), cosines.end());
    double const median = cosines.empty()
        ? 1.0
        : (cosines.size() % 2 == 0 ? 0.5 * (cosines[cosines.size() / 2 - 1] + cosines[cosines.size() / 2])
                                   : cosines[cosines.size() / 2]);
    double const mag = std::sqrt(refNorm2) > 0.0 ? std::sqrt(gotNorm2) / std::max(std::sqrt(refNorm2), 1e-30) : 0.0;
    return {median, mag, maxAbs};
}

void expectReferenceMatch(RunResult const& result, std::vector<float> const& ref, MoeCase const& cfg, char const* mode)
{
    constexpr double kMinCosine = 0.94;
    constexpr double kMinMagRatio = 0.25;
    constexpr double kMaxMagRatio = 3.00;

    ASSERT_EQ(result.status, 0) << mode << " runner returned non-zero status for " << cfg.name;
    ASSERT_EQ(result.outputFp32.size(), ref.size());
    Summary const s = summarize(result.outputFp32, ref, cfg.numTokens, cfg.hiddenSize);
    std::cout << "[" << cfg.name << "][" << mode << "] median_cos=" << s.medianCosine << " mag_ratio=" << s.magRatio
              << " max_abs=" << s.maxAbs << std::endl;
    EXPECT_GE(s.medianCosine, kMinCosine)
        << mode << " median cosine " << s.medianCosine << " below threshold " << kMinCosine << " for " << cfg.name;
    EXPECT_GE(s.magRatio, kMinMagRatio) << mode << " magnitude ratio below band [" << kMinMagRatio << ", "
                                        << kMaxMagRatio << "] for " << cfg.name;
    EXPECT_LE(s.magRatio, kMaxMagRatio) << mode << " magnitude ratio above band [" << kMinMagRatio << ", "
                                        << kMaxMagRatio << "] for " << cfg.name;
}

bool checkRequirements()
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        return false;
    }
    if (!CuteDslNvfp4MoeSm110Runner::canImplement(
            /*hiddenSize=*/1024, /*moeInterSize=*/768, kNumExperts, kTopK, sm, kActSwiGLU, kIoDtypeFp16, kBackendAuto))
    {
        return false;
    }
    return true;
}

std::vector<MoeCase> defaultCases()
{
    return {
        {/*name=*/"decode_h1024_i768_t1_swiglu_e128", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActSwiGLU, /*numExperts=*/128, /*seed=*/0xC0FFEEu},
        {/*name=*/"prefill_h1024_i768_t8_swiglu_e128", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActSwiGLU, /*numExperts=*/128, /*seed=*/0xDEADBEEFu},
        // ReLU2 path: fc1InputN = I (no doubling), reference uses square(max(x,0)).
        // Runner constraint moeInterSize % 64 == 0 and fc1InputN % kLevelTileN(128)
        // == 0 -> I=768 ok (768%128=0).
        {/*name=*/"decode_h1024_i768_t1_relu2_e128", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActReLU2, /*numExperts=*/128, /*seed=*/0xFEEDFACEu},
        {/*name=*/"prefill_h1024_i768_t8_relu2_e128", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActReLU2, /*numExperts=*/128, /*seed=*/0xBADCAFEu},
        // GeGLU path: same gated structure as SwiGLU (fc1InputN = 2*I) but with GELU activation on gate.
        {/*name=*/"decode_h1024_i768_t1_geglu_e128", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActGeGLU, /*numExperts=*/128, /*seed=*/0x6E610001u},
        {/*name=*/"prefill_h1024_i768_t8_geglu_e128", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActGeGLU, /*numExperts=*/128, /*seed=*/0x6E610008u},
        // E=256 coverage: the FC1/FC2 cubins are runtime-polymorphic in L, so the
        // same kernels must handle 256 experts. Decode exercises the fused
        // fp4BuildLayoutAndQuantizeRoutedLinearSFDecode path (kMaxDecodeExperts=256);
        // prefill exercises the general buildLayoutGpu path.
        {/*name=*/"decode_h1024_i768_t1_swiglu_e256", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActSwiGLU, /*numExperts=*/256, /*seed=*/0x5EED256u},
        {/*name=*/"decode_h1024_i768_t1_relu2_e256", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActReLU2, /*numExperts=*/256, /*seed=*/0xDEC0DE2u},
        {/*name=*/"prefill_h1024_i768_t8_swiglu_e256", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActSwiGLU, /*numExperts=*/256, /*seed=*/0xA11CE256u},
        {/*name=*/"prefill_h1024_i768_t8_relu2_e256", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActReLU2, /*numExperts=*/256, /*seed=*/0xB0B256u},
        {/*name=*/"decode_h1024_i768_t1_geglu_e256", /*numTokens=*/1, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActGeGLU, /*numExperts=*/256, /*seed=*/0x6E61256u},
        {/*name=*/"prefill_h1024_i768_t8_geglu_e256", /*numTokens=*/8, /*hiddenSize=*/1024,
            /*intermediateSize=*/768, /*activationType=*/kActGeGLU, /*numExperts=*/256, /*seed=*/0x6E68256u},
    };
}

std::vector<MoeCase> pdlCases()
{
    return defaultCases();
}

CaseData buildBitwiseStablePdlCase(MoeCase const& cfg)
{
    CaseData c = buildCase(cfg);
    // FC2 combines top-K rows with atomics. One nonzero route per token keeps
    // the bitwise check independent of otherwise valid atomic accumulation order.
    for (int32_t token = 0; token < cfg.numTokens; ++token)
    {
        float* weights = c.topkWeights.data() + static_cast<size_t>(token) * kTopK;
        std::fill_n(weights, kTopK, 0.0f);
        weights[0] = 1.0f;
    }
    return c;
}

CaseData buildZeroRouteCase(MoeCase const& cfg)
{
    CaseData c = buildCase(cfg);
    std::fill(c.topkWeights.begin(), c.topkWeights.end(), 0.0f);
    return c;
}

void expectOutputWasClearedAndGuardsIntact(RunResult const& result, MoeCase const& cfg)
{
    constexpr uint16_t kOutputPoisonFp16Bits{0xA5A5};
    ASSERT_EQ(result.status, 0) << "runner returned non-zero status for " << cfg.name;
    EXPECT_TRUE(std::all_of(
        result.outputFp16Bits.begin(), result.outputFp16Bits.end(), [](uint16_t bits) { return bits == 0; }))
        << "active output was not fully cleared for " << cfg.name;
    EXPECT_TRUE(std::all_of(result.outputPrefixGuardFp16Bits.begin(), result.outputPrefixGuardFp16Bits.end(),
        [kOutputPoisonFp16Bits](uint16_t bits) { return bits == kOutputPoisonFp16Bits; }))
        << "prefix guard was modified for " << cfg.name;
    EXPECT_TRUE(std::all_of(result.outputSuffixGuardFp16Bits.begin(), result.outputSuffixGuardFp16Bits.end(),
        [kOutputPoisonFp16Bits](uint16_t bits) { return bits == kOutputPoisonFp16Bits; }))
        << "suffix guard was modified for " << cfg.name;
}

#if defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

cudaLaunchAttribute const* findLaunchAttribute(LaunchConfigSnapshot const& config, cudaLaunchAttributeID const id)
{
    auto const it = std::find_if(
        config.attrs.begin(), config.attrs.end(), [id](cudaLaunchAttribute const& attr) { return attr.id == id; });
    return it == config.attrs.end() ? nullptr : &*it;
}

void expectSameNonPdlLaunchConfiguration(LaunchConfigSnapshot const& pdlOff, LaunchConfigSnapshot const& pdlOn)
{
    EXPECT_EQ(pdlOff.gridDim.x, pdlOn.gridDim.x);
    EXPECT_EQ(pdlOff.gridDim.y, pdlOn.gridDim.y);
    EXPECT_EQ(pdlOff.gridDim.z, pdlOn.gridDim.z);
    EXPECT_EQ(pdlOff.blockDim.x, pdlOn.blockDim.x);
    EXPECT_EQ(pdlOff.blockDim.y, pdlOn.blockDim.y);
    EXPECT_EQ(pdlOff.blockDim.z, pdlOn.blockDim.z);
    EXPECT_EQ(pdlOff.dynamicSmemBytes, pdlOn.dynamicSmemBytes);

    // CuTe DSL 4.7 emits exactly PSS, cluster, and cooperative attributes for
    // these wrappers. Require the two pre-existing attributes explicitly so a
    // wrapper that drops one cannot pass through equality alone.
    ASSERT_EQ(pdlOff.attrs.size(), 3u);
    ASSERT_EQ(pdlOn.attrs.size(), 3u);
    auto const* offCluster = findLaunchAttribute(pdlOff, cudaLaunchAttributeClusterDimension);
    auto const* onCluster = findLaunchAttribute(pdlOn, cudaLaunchAttributeClusterDimension);
    auto const* offCooperative = findLaunchAttribute(pdlOff, cudaLaunchAttributeCooperative);
    auto const* onCooperative = findLaunchAttribute(pdlOn, cudaLaunchAttributeCooperative);
    ASSERT_NE(offCluster, nullptr);
    ASSERT_NE(onCluster, nullptr);
    ASSERT_NE(offCooperative, nullptr);
    ASSERT_NE(onCooperative, nullptr);
    EXPECT_EQ(offCluster->val.clusterDim.x, 1u);
    EXPECT_EQ(offCluster->val.clusterDim.y, 1u);
    EXPECT_EQ(offCluster->val.clusterDim.z, 1u);
    EXPECT_EQ(onCluster->val.clusterDim.x, offCluster->val.clusterDim.x);
    EXPECT_EQ(onCluster->val.clusterDim.y, offCluster->val.clusterDim.y);
    EXPECT_EQ(onCluster->val.clusterDim.z, offCluster->val.clusterDim.z);
    EXPECT_EQ(offCooperative->val.cooperative, 0);
    EXPECT_EQ(onCooperative->val.cooperative, offCooperative->val.cooperative);
}

#endif // defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

} // namespace

TEST(CuteDslNvfp4MoeSm110Test, canImplementSupportedSms)
{
    for (int32_t const sm : {100, 101, 110})
    {
        // Supported expert counts {128, 256, 512} must pass on every supported SM.
        for (int32_t const e : {128, 256, 512})
        {
            EXPECT_TRUE(CuteDslNvfp4MoeSm110Runner::canImplement(
                /*hiddenSize=*/1024, /*moeInterSize=*/768, e, kTopK, sm, kActSwiGLU, kIoDtypeFp16, kBackendAuto))
                << "sm=" << sm << " e=" << e;
            EXPECT_TRUE(CuteDslNvfp4MoeSm110Runner::canImplement(
                /*hiddenSize=*/1024, /*moeInterSize=*/768, e, kTopK, sm, kActReLU2, kIoDtypeFp16, kBackendAuto))
                << "sm=" << sm << " e=" << e;
            EXPECT_TRUE(CuteDslNvfp4MoeSm110Runner::canImplement(
                /*hiddenSize=*/1024, /*moeInterSize=*/768, e, kTopK, sm, kActGeGLU, kIoDtypeFp16, kBackendAuto))
                << "sm=" << sm << " e=" << e;
        }
        // Expert counts outside the supported set are rejected (the cubin is
        // runtime-polymorphic, but the product contract is exactly {128, 256, 512}).
        for (int32_t const e : {64, 192, 1024})
        {
            EXPECT_FALSE(CuteDslNvfp4MoeSm110Runner::canImplement(
                /*hiddenSize=*/1024, /*moeInterSize=*/768, e, kTopK, sm, kActSwiGLU, kIoDtypeFp16, kBackendAuto))
                << "sm=" << sm << " e=" << e;
        }
        // top_k is accepted up to kMaxTopK (32, covers Nemotron-3-Super's top_k=22)
        // and rejected beyond it.
        EXPECT_TRUE(CuteDslNvfp4MoeSm110Runner::canImplement(
            /*hiddenSize=*/1024, /*moeInterSize=*/768, kNumExperts, /*topK=*/32, sm, kActReLU2, kIoDtypeFp16,
            kBackendAuto))
            << "sm=" << sm;
        EXPECT_FALSE(CuteDslNvfp4MoeSm110Runner::canImplement(
            /*hiddenSize=*/1024, /*moeInterSize=*/768, kNumExperts, /*topK=*/33, sm, kActReLU2, kIoDtypeFp16,
            kBackendAuto))
            << "sm=" << sm;
    }

    EXPECT_FALSE(CuteDslNvfp4MoeSm110Runner::canImplement(
        /*hiddenSize=*/1024, /*moeInterSize=*/768, kNumExperts, kTopK, /*smVersion=*/120, kActSwiGLU, kIoDtypeFp16,
        kBackendAuto));
}

TEST(CuteDslNvfp4MoeSm110Test, smoke)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL runner test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData c = buildCase(cfg);
        RunResult const result = runCase(c);
        ASSERT_EQ(result.status, 0) << "runner returned non-zero status for " << cfg.name;
        ASSERT_EQ(result.outputFp32.size(), static_cast<size_t>(cfg.numTokens) * cfg.hiddenSize);
        bool allZero = true;
        for (float v : result.outputFp32)
        {
            ASSERT_TRUE(std::isfinite(v)) << "non-finite output in " << cfg.name;
            if (v != 0.0f)
            {
                allZero = false;
            }
        }
        ASSERT_FALSE(allZero) << "output is all-zero for " << cfg.name;
    }
}

TEST(CuteDslNvfp4MoeSm110Test, accuracy)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL runner test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    // Loose-but-meaningful bands for NVFP4 MoE: cosine >= 0.94 catches sign /
    // routing / activation bugs; magnitude ratio in [0.25, 3.0] catches gross
    // scale drift (per-block FP8 SF, per-expert alpha, FC1/FC2 global scales)
    // without flagging the expected FP4-quantization noise floor.
    constexpr double kMinCosine = 0.94;
    constexpr double kMinMagRatio = 0.25;
    constexpr double kMaxMagRatio = 3.00;

    for (auto const& cfg : defaultCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData c = buildCase(cfg);
        std::vector<float> const ref = computeReference(c);
        RunResult const result = runCase(c);
        ASSERT_EQ(result.status, 0) << "runner returned non-zero status for " << cfg.name;
        ASSERT_EQ(result.outputFp32.size(), ref.size());

        Summary const s = summarize(result.outputFp32, ref, cfg.numTokens, cfg.hiddenSize);
        std::cout << "[" << cfg.name << "] median_cos=" << s.medianCosine << " mag_ratio=" << s.magRatio
                  << " max_abs=" << s.maxAbs << std::endl;
        EXPECT_GE(s.medianCosine, kMinCosine)
            << "median cosine " << s.medianCosine << " below threshold " << kMinCosine << " for " << cfg.name;
        EXPECT_GE(s.magRatio, kMinMagRatio)
            << "magnitude ratio below band [" << kMinMagRatio << ", " << kMaxMagRatio << "] for " << cfg.name;
        EXPECT_LE(s.magRatio, kMaxMagRatio)
            << "magnitude ratio above band [" << kMinMagRatio << ", " << kMaxMagRatio << "] for " << cfg.name;
    }
}

TEST(CuteDslNvfp4MoeSm110Test, pdlMatchesDisabledAndReference)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "NVFP4 MoE PDL test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    for (auto const& cfg : pdlCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const c = buildBitwiseStablePdlCase(cfg);
        std::vector<float> const ref = computeReference(c);
        RunResult const pdlOff = runCase(c, /*enablePdl=*/false);
        RunResult const pdlOn = runCase(c, /*enablePdl=*/true);

        expectReferenceMatch(pdlOff, ref, cfg, "pdl_off");
        expectReferenceMatch(pdlOn, ref, cfg, "pdl_on");
        EXPECT_EQ(pdlOn.outputFp16Bits, pdlOff.outputFp16Bits)
            << "PDL changed the bitwise FP16 output for " << cfg.name;
    }
}

TEST(CuteDslNvfp4MoeSm110Test, zeroRoutesClearPoisonedOutputAndPreserveGuards)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "NVFP4 MoE output-clear test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    std::vector<MoeCase> const cases{
        defaultCases()[0], // SwiGLU decode
        defaultCases()[2], // ReLU2 decode
        defaultCases()[4], // GeGLU decode
    };
    for (auto const& cfg : cases)
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const c = buildZeroRouteCase(cfg);
        for (bool const enablePdl : {false, true})
        {
            expectOutputWasClearedAndGuardsIntact(runCase(c, enablePdl), cfg);
            // The graph path instantiates the captured launch then replays it
            // eight times with a poisoned active output/workspace before each
            // replay; guards stay outside every runner-owned range.
            expectOutputWasClearedAndGuardsIntact(runCase(c, enablePdl, ExecutionMode::kCudaGraph), cfg);
        }
    }
}

#if defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

TEST(CuteDslNvfp4MoeSm110Test, generatedLaunchConfigurationUsesPdlAndPreservesExistingAttributes)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "NVFP4 MoE PDL launch-config test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    std::vector<MoeCase> launchCases{
        defaultCases()[0], // SwiGLU N128
        defaultCases()[2], // ReLU2 N128
        defaultCases()[4], // GeGLU N128
    };

    std::vector<std::string> checkedFc1Variants;
    for (auto const& cfg : launchCases)
    {
        checkedFc1Variants.push_back(cfg.name);
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const c = buildBitwiseStablePdlCase(cfg);

        ScopedLaunchConfigCapture offCapture;
        RunResult const pdlOff = runCase(c, /*enablePdl=*/false);
        auto const pdlOffConfigs = offCapture.finish();

        ScopedLaunchConfigCapture onCapture;
        RunResult const pdlOn = runCase(c, /*enablePdl=*/true);
        auto const pdlOnConfigs = onCapture.finish();

        ASSERT_EQ(pdlOff.status, 0);
        ASSERT_EQ(pdlOn.status, 0);
        // The runner has exactly two CuTe DSL launches: FC1 followed by FC2.
        ASSERT_EQ(pdlOffConfigs.size(), 2u);
        ASSERT_EQ(pdlOnConfigs.size(), pdlOffConfigs.size());

        for (size_t i = 0; i < pdlOffConfigs.size(); ++i)
        {
            auto const* offPdl
                = findLaunchAttribute(pdlOffConfigs[i], cudaLaunchAttributeProgrammaticStreamSerialization);
            auto const* onPdl
                = findLaunchAttribute(pdlOnConfigs[i], cudaLaunchAttributeProgrammaticStreamSerialization);

            // CuTe DSL 4.7 represents a disabled PDL policy as the PSS
            // attribute with value zero. CUDA treats that as the effective
            // no-PDL fallback; the enabled path carries value one.
            ASSERT_NE(offPdl, nullptr) << "launch=" << i;
            ASSERT_NE(onPdl, nullptr) << "launch=" << i;
            EXPECT_EQ(offPdl->val.programmaticStreamSerializationAllowed, 0) << "launch=" << i;
            EXPECT_EQ(onPdl->val.programmaticStreamSerializationAllowed, 1) << "launch=" << i;

            // Switching PDL must not replace cluster/cooperative metadata or
            // any other attribute emitted by the AOT wrapper.
            expectSameNonPdlLaunchConfiguration(pdlOffConfigs[i], pdlOnConfigs[i]);
        }
    }
    EXPECT_EQ(checkedFc1Variants.size(), 3u);
}

#endif // defined(EDGELLM_TEST_WRAP_CUTEDSL_LAUNCH) && CUDART_VERSION >= 12080

TEST(CuteDslNvfp4MoeSm110Test, DISABLED_pdlNsightProfile)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "NVFP4 MoE PDL Nsight test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    for (auto const& cfg : pdlCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const c = buildBitwiseStablePdlCase(cfg);
        std::string const offLabel = "nvfp4_moe_pdl/" + cfg.name + "/off";
        std::string const onLabel = "nvfp4_moe_pdl/" + cfg.name + "/on";
        RunResult const pdlOff = runCase(c, /*enablePdl=*/false, ExecutionMode::kEager, offLabel.c_str());
        RunResult const pdlOn = runCase(c, /*enablePdl=*/true, ExecutionMode::kEager, onLabel.c_str());
        ASSERT_EQ(pdlOff.status, 0);
        ASSERT_EQ(pdlOn.status, 0);
        EXPECT_EQ(pdlOn.outputFp16Bits, pdlOff.outputFp16Bits);
    }
}

TEST(CuteDslNvfp4MoeSm110Test, pdlCudaGraphReplayMatchesDisabledAndReference)
{
    int32_t const sm = getSMVersion();
    if (!isSupportedSm(sm))
    {
        GTEST_SKIP() << "NVFP4 MoE PDL CUDA Graph test requires SM100, SM101, or SM110, got SM=" << sm;
    }
    if (!checkRequirements())
    {
        GTEST_SKIP() << "SM100/101/110 NVFP4 MoE CuTeDSL canImplement returned false";
    }

    for (auto const& cfg : pdlCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << cfg.name);
        CaseData const c = buildBitwiseStablePdlCase(cfg);
        std::vector<float> const ref = computeReference(c);
        RunResult const pdlOff = runCase(c, /*enablePdl=*/false, ExecutionMode::kCudaGraph);
        RunResult const pdlOn = runCase(c, /*enablePdl=*/true, ExecutionMode::kCudaGraph);

        expectReferenceMatch(pdlOff, ref, cfg, "cuda_graph_pdl_off");
        expectReferenceMatch(pdlOn, ref, cfg, "cuda_graph_pdl_on");
        EXPECT_EQ(pdlOn.outputFp16Bits, pdlOff.outputFp16Bits)
            << "PDL changed the bitwise CUDA Graph output for " << cfg.name;
    }
}

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
