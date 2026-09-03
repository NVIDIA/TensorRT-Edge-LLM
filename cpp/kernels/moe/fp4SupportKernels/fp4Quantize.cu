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

// FP4 quantization: BF16 -> packed FP4 + swizzled FP8 E4M3 scale factors.
// SM100+ (Blackwell/Thor): hardware cvt.rn.satfinite.e2m1x2.f32.
// Pre-SM100: software E2M1 conversion fallback.

#include "fp4Quantize.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

#if SUPPORTS_FP8

namespace
{

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

constexpr int kEltsPerThread = 8; // 4 x bfloat162 = 128 bits
constexpr int kSfVecSize = 16;    // elements per SF block
constexpr int kThreadsPerSf = 2;  // kSfVecSize / kEltsPerThread

// Maximum number of experts the fused decode kernel
// (buildLayoutAndQuantizeRoutedToFp4LinearSfDecodeKernel) can scatter into.
// The kernel pre-allocates shared-memory `expertCounts`/`expertOffsets`/
// `scatterCounters` arrays of this length, and the host-side launcher rejects
// any call with `topK` or `localNumExperts` above this bound. Keep host and
// kernel in lockstep -- the kernel references this same constant.
// Must stay >= CuteDslNvfp4MoeSm110Runner::kMaxNumExperts (256) so the SM110
// runner's decode setup path can serve every supported expert count; at 256 the
// three int32 arrays use 3 KiB of static shared memory (well within limits).
constexpr int kMaxDecodeExperts = 512;

// Lower bound on the per-expert global scale factor used as a divisor in the
// FP4 routed quantization kernels (fmaxf(sfScales[expert], kSfScaleEpsilon)).
// Guards against zero-valued global scales and matches the numerical guard
// used in the corresponding host reference (see Python tests).
constexpr float kSfScaleEpsilon = 1.0e-20f;

// Resident-thread cap used to derive the per-SM block budget
// (numBlocksPerSM = max(1, kMaxResidentThreadsPerSm / blockSize)). This is the
// SM thread occupancy limit for the architectures we support (Hopper / Ada /
// Blackwell / Thor). Centralising it keeps the four launcher sites in sync.
constexpr int kMaxResidentThreadsPerSm = 2048;

// -------------------------------------------------------------------------
// Type traits for BF16 / FP16 generic quantization
// -------------------------------------------------------------------------

template <typename ScalarT>
struct QuantTraits;

template <>
struct QuantTraits<__nv_bfloat16>
{
    using Vec2 = __nv_bfloat162;
    static __device__ __forceinline__ float2 toFloat2(Vec2 v)
    {
        return __bfloat1622float2(v);
    }
    static __device__ __forceinline__ float toFloat(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }
};

template <>
struct QuantTraits<__half>
{
    using Vec2 = __half2;
    static __device__ __forceinline__ float2 toFloat2(Vec2 v)
    {
        return __half22float2(v);
    }
    static __device__ __forceinline__ float toFloat(__half v)
    {
        return __half2float(v);
    }
};

// -------------------------------------------------------------------------
// Vectorised load structure — 128-bit (8 values)
// -------------------------------------------------------------------------

template <typename Vec2T>
struct PackedVecT
{
    Vec2T elts[4]; // 4 x 4 bytes = 16 bytes = 128 bits
};

// -------------------------------------------------------------------------
// Device helpers
// -------------------------------------------------------------------------

/// Software FP4 E2M1 conversion for a single float value.
///
/// Returns 4-bit E2M1 encoding: [sign(1), exponent(2), mantissa(1)].
/// Bit-exact with hardware cvt.rn.satfinite.e2m1x2.f32 semantics:
///   NaN -> +6.0 (max positive), +/-Inf / out-of-range -> +/-6.0, round-to-nearest-even.
///
/// E2M1 representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
inline __device__ uint32_t floatToE2m1(float val)
{
    // satfinite: NaN -> max positive (matches Blackwell hardware behavior)
    if (__isnanf(val))
        return 7u;

    uint32_t sign = (__float_as_uint(val) >> 31) << 3; // bit 31 -> bit 3
    float aval = fabsf(val);

    // Round to nearest representable E2M1 value (round-to-nearest-even at midpoints).
    uint32_t bits;
    if (aval > 5.0f)
        bits = 7; // -> 6.0  (sat for +/-Inf)
    else if (aval >= 3.5f)
        bits = 6; // -> 4.0
    else if (aval > 2.5f)
        bits = 5; // -> 3.0
    else if (aval >= 1.75f)
        bits = 4; // -> 2.0
    else if (aval > 1.25f)
        bits = 3; // -> 1.5
    else if (aval >= 0.75f)
        bits = 2; // -> 1.0
    else if (aval > 0.25f)
        bits = 1; // -> 0.5
    else
        bits = 0; // -> 0.0

    return sign | bits;
}

/// Pack 8 float32 values into one uint32 of E2M1 nibbles.
///
/// SM100+ uses hardware cvt.rn.satfinite.e2m1x2.f32.
/// Older architectures use the software floatToE2m1 fallback.
inline __device__ uint32_t fp32VecToE2m1(float2 (&array)[4])
{
    // SM100 (1000), SM101 (1010), SM110 (1100).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1010 || __CUDA_ARCH__ == 1100)
    // Hardware path
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
    return val;
#else
    // Software fallback for pre-Blackwell GPUs
    uint32_t val = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        val |= floatToE2m1(array[i].x) << (i * 8);
        val |= floatToE2m1(array[i].y) << (i * 8 + 4);
    }
    return val;
#endif
}

/// Fast reciprocal approximation with flush-to-zero.
inline __device__ float rcpApproxFtz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

/// Atom-layout (128x4 swizzle) byte offset for scale factors.
inline __device__ int64_t getSfOutOffset128x4(int mIdx, int kIdx, int numRows, int numSfCols)
{
    int innerK = kIdx % 4;
    int innerM = (mIdx % 128) / 32;
    int outerM = mIdx % 32;
    int kTile = kIdx / 4;
    int numKTiles = (numSfCols + 3) / 4;
    int mTile = mIdx / 128;
    return (int64_t) mTile * numKTiles * 512 + (int64_t) kTile * 512 + outerM * 16 + innerM * 4 + innerK;
}

// -------------------------------------------------------------------------
// Main quantization kernel
// -------------------------------------------------------------------------

template <typename ScalarT>
__launch_bounds__(512, 4) __global__
    void quantizeToFp4Kernel(int32_t numRows, int32_t numCols, ScalarT const* __restrict__ input,
        float const* __restrict__ sfScale, uint32_t* __restrict__ outputFP4, uint8_t* __restrict__ outputSF)
{
    using Traits = QuantTraits<ScalarT>;
    using Vec2 = typename Traits::Vec2;
    using PVec = PackedVecT<Vec2>;

    int const numPaddedRows = ((numRows + 127) / 128) * 128;
    int const numColThreads = numCols / kEltsPerThread;
    int const numSfCols = numCols / kSfVecSize;
    int const numSfColThreads = numSfCols * kThreadsPerSf;
    // Forward-scale contract: callers pass the forward global SF (e.g. max|x|/(448*6));
    // the FP4 mapping needs its reciprocal, so compute it in-register once per thread.
    float const SFScaleInv = 1.0f / *sfScale;

    // Persistent row loop — blocks cycle through rows.
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRows; rowIdx += gridDim.x)
    {
        bool const isRowPadding = (rowIdx >= numRows);

        if (isRowPadding)
        {
            // Padding row: zero SF bytes only.
            for (int colIdx = threadIdx.x; colIdx < numSfColThreads; colIdx += blockDim.x)
            {
                if (colIdx % kThreadsPerSf == 0)
                {
                    int sfIdx = colIdx / kThreadsPerSf;
                    int64_t offset = getSfOutOffset128x4(rowIdx, sfIdx, numRows, numSfCols);
                    outputSF[offset] = 0;
                }
            }
        }
        else
        {
            // Data row — quantise.
            for (int colIdx = threadIdx.x; colIdx < numSfColThreads; colIdx += blockDim.x)
            {
                if (colIdx >= numColThreads)
                {
                    // Padding column beyond data: zero SF.
                    if (colIdx % kThreadsPerSf == 0)
                    {
                        int sfIdx = colIdx / kThreadsPerSf;
                        int64_t offset = getSfOutOffset128x4(rowIdx, sfIdx, numRows, numSfCols);
                        outputSF[offset] = 0;
                    }
                    continue;
                }

                // Vectorised 128-bit load.
                PVec vec = *reinterpret_cast<PVec const*>(
                    input + (int64_t) rowIdx * numCols + (int64_t) colIdx * kEltsPerThread);

                // Thread-local absolute max over 4 vec2 pairs.
                Vec2 localMax = __habs2(vec.elts[0]);
#pragma unroll
                for (int i = 1; i < 4; i++)
                    localMax = __hmax2(localMax, __habs2(vec.elts[i]));

                // 2-thread reduction via warp shuffle for 16-element block max.
                Vec2 otherMax = __shfl_xor_sync(0xFFFFFFFF, localMax, 1);
                localMax = __hmax2(localMax, otherMax);
                float vecMax = Traits::toFloat(__hmax(localMax.x, localMax.y));

                // Scale factor computation (FP8 E4M3FN).
                float SFValue = SFScaleInv * (vecMax * rcpApproxFtz(6.0f));
                __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(SFValue);
                float SFValue_back = float(sf_fp8);
                float outScale = (vecMax != 0.0f) ? rcpApproxFtz(SFValue_back * rcpApproxFtz(SFScaleInv)) : 0.0f;

                // Scale to FP4 range + E2M1 conversion.
                float2 fp2Vals[4];
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    fp2Vals[i] = Traits::toFloat2(vec.elts[i]);
                    fp2Vals[i].x *= outScale;
                    fp2Vals[i].y *= outScale;
                }
                uint32_t packed = fp32VecToE2m1(fp2Vals);

                // Write FP4 output.
                outputFP4[(int64_t) rowIdx * (numCols / kEltsPerThread) + colIdx] = packed;

                // Write SF byte (even-indexed thread only).
                if (colIdx % kThreadsPerSf == 0)
                {
                    int sfIdx = colIdx / kThreadsPerSf;
                    int64_t offset = getSfOutOffset128x4(rowIdx, sfIdx, numRows, numSfCols);
                    outputSF[offset] = sf_fp8.__x;
                }
            }
        }
    }
}

template <typename ScalarT>
__launch_bounds__(512, 4) __global__
    void quantizeToFp4LinearSfKernel(int32_t numRows, int32_t numCols, ScalarT const* __restrict__ input,
        float const* __restrict__ sfScale, uint32_t* __restrict__ outputFP4, uint8_t* __restrict__ outputSF)
{
    using Traits = QuantTraits<ScalarT>;
    using Vec2 = typename Traits::Vec2;
    using PVec = PackedVecT<Vec2>;

    int const numColThreads = numCols / kEltsPerThread;
    int const numSfCols = numCols / kSfVecSize;
    int const numSfColThreads = numSfCols * kThreadsPerSf;
    float const SFScaleInv = 1.0f / *sfScale;

    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int colIdx = threadIdx.x; colIdx < numSfColThreads; colIdx += blockDim.x)
        {
            PVec vec = *reinterpret_cast<PVec const*>(
                input + (int64_t) rowIdx * numCols + (int64_t) colIdx * kEltsPerThread);

            Vec2 localMax = __habs2(vec.elts[0]);
#pragma unroll
            for (int i = 1; i < 4; i++)
            {
                localMax = __hmax2(localMax, __habs2(vec.elts[i]));
            }

            Vec2 otherMax = __shfl_xor_sync(0xFFFFFFFF, localMax, 1);
            localMax = __hmax2(localMax, otherMax);
            float vecMax = Traits::toFloat(__hmax(localMax.x, localMax.y));

            float SFValue = SFScaleInv * (vecMax * rcpApproxFtz(6.0f));
            __nv_fp8_e4m3 sfFp8 = __nv_fp8_e4m3(SFValue);
            float SFValueBack = float(sfFp8);
            float outScale = (vecMax != 0.0f) ? rcpApproxFtz(SFValueBack * rcpApproxFtz(SFScaleInv)) : 0.0f;

            float2 fp2Vals[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                fp2Vals[i] = Traits::toFloat2(vec.elts[i]);
                fp2Vals[i].x *= outScale;
                fp2Vals[i].y *= outScale;
            }
            uint32_t const packed = fp32VecToE2m1(fp2Vals);
            outputFP4[(int64_t) rowIdx * numColThreads + colIdx] = packed;

            if (colIdx % kThreadsPerSf == 0)
            {
                int const sfIdx = colIdx / kThreadsPerSf;
                outputSF[(int64_t) rowIdx * numSfCols + sfIdx] = sfFp8.__x;
            }
        }
    }
}

template <typename ScalarT>
__launch_bounds__(512, 4) __global__ void quantizeRoutedToFp4LinearSfKernel(int32_t numRows, int32_t topK,
    int32_t numCols, ScalarT const* __restrict__ input, int32_t const* __restrict__ topkIds,
    float const* __restrict__ sfScales, uint32_t* __restrict__ outputFP4, uint8_t* __restrict__ outputSF)
{
    using Traits = QuantTraits<ScalarT>;
    using Vec2 = typename Traits::Vec2;
    using PVec = PackedVecT<Vec2>;

    int const numRoutedRows = numRows * topK;
    int const numColThreads = numCols / kEltsPerThread;
    int const numSfCols = numCols / kSfVecSize;
    int const numSfColThreads = numSfCols * kThreadsPerSf;

    for (int routedRowIdx = blockIdx.x; routedRowIdx < numRoutedRows; routedRowIdx += gridDim.x)
    {
        int const tokenIdx = routedRowIdx / topK;
        int const expert = topkIds[routedRowIdx];
        float const sfScale = fmaxf(sfScales[expert], kSfScaleEpsilon);
        float const sfScaleInv = 1.0f / sfScale;

        for (int colIdx = threadIdx.x; colIdx < numSfColThreads; colIdx += blockDim.x)
        {
            PVec vec = *reinterpret_cast<PVec const*>(
                input + (int64_t) tokenIdx * numCols + (int64_t) colIdx * kEltsPerThread);

            Vec2 localMax = __habs2(vec.elts[0]);
#pragma unroll
            for (int i = 1; i < 4; i++)
            {
                localMax = __hmax2(localMax, __habs2(vec.elts[i]));
            }

            Vec2 otherMax = __shfl_xor_sync(0xFFFFFFFF, localMax, 1);
            localMax = __hmax2(localMax, otherMax);
            float vecMax = Traits::toFloat(__hmax(localMax.x, localMax.y));

            float sfValue = sfScaleInv * (vecMax * rcpApproxFtz(6.0f));
            __nv_fp8_e4m3 sfFp8 = __nv_fp8_e4m3(sfValue);
            float sfValueBack = float(sfFp8);
            float outScale = (vecMax != 0.0f) ? rcpApproxFtz(sfValueBack * sfScale) : 0.0f;

            float2 fp2Vals[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                fp2Vals[i] = Traits::toFloat2(vec.elts[i]);
                fp2Vals[i].x *= outScale;
                fp2Vals[i].y *= outScale;
            }
            uint32_t const packed = fp32VecToE2m1(fp2Vals);
            outputFP4[(int64_t) routedRowIdx * numColThreads + colIdx] = packed;

            if (colIdx % kThreadsPerSf == 0)
            {
                int const sfIdx = colIdx / kThreadsPerSf;
                outputSF[(int64_t) routedRowIdx * numSfCols + sfIdx] = sfFp8.__x;
            }
        }
    }
}

template <typename ScalarT>
__launch_bounds__(512, 4) __global__ void buildLayoutAndQuantizeRoutedToFp4LinearSfDecodeKernel(int32_t topK,
    int32_t numCols, int32_t localNumExperts, int32_t tileSize, ScalarT const* __restrict__ input,
    int32_t const* __restrict__ topkIds, float const* __restrict__ sfScales, int32_t* __restrict__ permutedIdx,
    int32_t* __restrict__ tileGroupIdx, int32_t* __restrict__ tileMnLimit, int32_t* __restrict__ numNonExitingTiles,
    uint32_t* __restrict__ outputFP4, uint8_t* __restrict__ outputSF)
{
    using Traits = QuantTraits<ScalarT>;
    using Vec2 = typename Traits::Vec2;
    using PVec = PackedVecT<Vec2>;

    // Mirrors the file-scope `kMaxDecodeExperts` constant used by the host
    // launcher; the static asserts on `topK`/`localNumExperts` there are what
    // keep these shared arrays in bounds at runtime.
    __shared__ int32_t expertCounts[kMaxDecodeExperts];
    __shared__ int32_t expertOffsets[kMaxDecodeExperts];
    __shared__ int32_t scatterCounters[kMaxDecodeExperts];

    int const tid = threadIdx.x;
    if (blockIdx.x == 0)
    {
        for (int expert = tid; expert < localNumExperts; expert += blockDim.x)
        {
            expertCounts[expert] = 0;
            scatterCounters[expert] = 0;
        }
        __syncthreads();

        if (tid < topK)
        {
            int const expert = topkIds[tid];
            if (expert >= 0 && expert < localNumExperts)
            {
                atomicAdd_block(&expertCounts[expert], 1);
            }
        }
        __syncthreads();

        if (tid == 0)
        {
            int runningOffset = 0;
            int runningTiles = 0;
            for (int expert = 0; expert < localNumExperts; ++expert)
            {
                int const count = expertCounts[expert];
                expertOffsets[expert] = runningOffset;
                if (count > 0)
                {
                    tileGroupIdx[runningTiles] = expert;
                    tileMnLimit[runningTiles] = runningOffset + count;
                    runningOffset += tileSize;
                    ++runningTiles;
                }
            }
            numNonExitingTiles[0] = runningTiles;
        }
        __syncthreads();

        if (tid < topK)
        {
            int const expert = topkIds[tid];
            if (expert >= 0 && expert < localNumExperts)
            {
                int const pos = atomicAdd_block(&scatterCounters[expert], 1);
                permutedIdx[expertOffsets[expert] + pos] = tid;
            }
        }
        return;
    }

    int const numColThreads = numCols / kEltsPerThread;
    int const numSfCols = numCols / kSfVecSize;
    int const numSfColThreads = numSfCols * kThreadsPerSf;

    int const quantGridSize = gridDim.x - 1;
    for (int routedRowIdx = blockIdx.x - 1; routedRowIdx < topK; routedRowIdx += quantGridSize)
    {
        int const expert = topkIds[routedRowIdx];
        float const sfScale = fmaxf(sfScales[expert], kSfScaleEpsilon);
        float const sfScaleInv = 1.0f / sfScale;

        for (int colIdx = tid; colIdx < numSfColThreads; colIdx += blockDim.x)
        {
            PVec vec = *reinterpret_cast<PVec const*>(input + (int64_t) colIdx * kEltsPerThread);

            Vec2 localMax = __habs2(vec.elts[0]);
#pragma unroll
            for (int i = 1; i < 4; i++)
            {
                localMax = __hmax2(localMax, __habs2(vec.elts[i]));
            }

            Vec2 otherMax = __shfl_xor_sync(0xFFFFFFFF, localMax, 1);
            localMax = __hmax2(localMax, otherMax);
            float vecMax = Traits::toFloat(__hmax(localMax.x, localMax.y));

            float sfValue = sfScaleInv * (vecMax * rcpApproxFtz(6.0f));
            __nv_fp8_e4m3 sfFp8 = __nv_fp8_e4m3(sfValue);
            float sfValueBack = float(sfFp8);
            float outScale = (vecMax != 0.0f) ? rcpApproxFtz(sfValueBack * sfScale) : 0.0f;

            float2 fp2Vals[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                fp2Vals[i] = Traits::toFloat2(vec.elts[i]);
                fp2Vals[i].x *= outScale;
                fp2Vals[i].y *= outScale;
            }
            uint32_t const packed = fp32VecToE2m1(fp2Vals);
            outputFP4[(int64_t) routedRowIdx * numColThreads + colIdx] = packed;

            if (colIdx % kThreadsPerSf == 0)
            {
                int const sfIdx = colIdx / kThreadsPerSf;
                outputSF[(int64_t) routedRowIdx * numSfCols + sfIdx] = sfFp8.__x;
            }
        }
    }
}

} // namespace

// -------------------------------------------------------------------------
// Public dispatcher
// -------------------------------------------------------------------------

void fp4Quantize(rt::Tensor const& input, rt::Tensor const& globalSF, rt::Tensor& outputFP4, rt::Tensor& outputSF,
    cudaStream_t stream)
{
    int64_t const M = input.getShape()[0];
    int64_t const N = input.getShape()[1];
    int const numColThreads = static_cast<int>(N) / kEltsPerThread;
    int const blockSize = std::min(numColThreads, 512);
    int const numPaddedRows = static_cast<int>(divUp(M, 128) * 128);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));

    int const numBlocksPerSM = std::max(1, kMaxResidentThreadsPerSm / blockSize);
    int const gridSize = std::min(numPaddedRows, smCount * numBlocksPerSM);

    auto const* sfPtrFwd = static_cast<float const*>(globalSF.rawPointer());
    auto* fp4Ptr = static_cast<uint32_t*>(outputFP4.rawPointer());
    auto* sfPtr = static_cast<uint8_t*>(outputSF.rawPointer());

    if (input.getDataType() == nvinfer1::DataType::kBF16)
    {
        quantizeToFp4Kernel<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(N), static_cast<__nv_bfloat16 const*>(input.rawPointer()), sfPtrFwd, fp4Ptr, sfPtr);
    }
    else if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        quantizeToFp4Kernel<__half><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(N), static_cast<__half const*>(input.rawPointer()), sfPtrFwd, fp4Ptr, sfPtr);
    }
    else
    {
        throw std::runtime_error("fp4Quantize: unsupported input data type (expected kBF16 or kHALF).");
    }
}

void fp4QuantizeLinearSF(rt::Tensor const& input, rt::Tensor const& globalSF, rt::Tensor& outputFP4,
    rt::Tensor& outputSF, cudaStream_t stream)
{
    int64_t const M = input.getShape()[0];
    int64_t const N = input.getShape()[1];
    if (N <= 0 || N % kSfVecSize != 0)
    {
        throw std::runtime_error("fp4QuantizeLinearSF: input N must be a positive multiple of 16.");
    }
    if (M <= 0)
    {
        return;
    }

    int const numColThreads = static_cast<int>(N) / kEltsPerThread;
    int const blockSize = std::min(numColThreads, 512);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));

    int const numBlocksPerSM = std::max(1, kMaxResidentThreadsPerSm / blockSize);
    int const gridSize = std::min(static_cast<int>(M), smCount * numBlocksPerSM);

    auto const* sfPtrFwd = static_cast<float const*>(globalSF.rawPointer());
    auto* fp4Ptr = static_cast<uint32_t*>(outputFP4.rawPointer());
    auto* sfPtr = static_cast<uint8_t*>(outputSF.rawPointer());

    if (input.getDataType() == nvinfer1::DataType::kBF16)
    {
        quantizeToFp4LinearSfKernel<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(N), static_cast<__nv_bfloat16 const*>(input.rawPointer()), sfPtrFwd, fp4Ptr, sfPtr);
    }
    else if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        quantizeToFp4LinearSfKernel<__half><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(N), static_cast<__half const*>(input.rawPointer()), sfPtrFwd, fp4Ptr, sfPtr);
    }
    else
    {
        throw std::runtime_error("fp4QuantizeLinearSF: unsupported input data type (expected kBF16 or kHALF).");
    }
}

void fp4QuantizeRoutedLinearSF(rt::Tensor const& input, rt::Tensor const& topkIds, rt::Tensor const& expertGlobalSF,
    rt::Tensor& outputFP4, rt::Tensor& outputSF, cudaStream_t stream)
{
    int64_t const M = input.getShape()[0];
    int64_t const N = input.getShape()[1];
    int64_t const topK = topkIds.getShape()[1];
    if (N <= 0 || N % kSfVecSize != 0)
    {
        throw std::runtime_error("fp4QuantizeRoutedLinearSF: input N must be a positive multiple of 16.");
    }
    if (M <= 0 || topK <= 0)
    {
        return;
    }

    int const numColThreads = static_cast<int>(N) / kEltsPerThread;
    int const blockSize = std::min(numColThreads, 512);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));

    int const numBlocksPerSM = std::max(1, kMaxResidentThreadsPerSm / blockSize);
    int const numRoutedRows = static_cast<int>(M * topK);
    int const gridSize = std::min(numRoutedRows, smCount * numBlocksPerSM);

    auto const* topkPtr = static_cast<int32_t const*>(topkIds.rawPointer());
    auto const* sfPtrFwd = static_cast<float const*>(expertGlobalSF.rawPointer());
    auto* fp4Ptr = static_cast<uint32_t*>(outputFP4.rawPointer());
    auto* sfPtr = static_cast<uint8_t*>(outputSF.rawPointer());

    if (input.getDataType() == nvinfer1::DataType::kBF16)
    {
        quantizeRoutedToFp4LinearSfKernel<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(topK), static_cast<int32_t>(N), static_cast<__nv_bfloat16 const*>(input.rawPointer()),
            topkPtr, sfPtrFwd, fp4Ptr, sfPtr);
    }
    else if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        quantizeRoutedToFp4LinearSfKernel<__half><<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(M),
            static_cast<int32_t>(topK), static_cast<int32_t>(N), static_cast<__half const*>(input.rawPointer()),
            topkPtr, sfPtrFwd, fp4Ptr, sfPtr);
    }
    else
    {
        throw std::runtime_error("fp4QuantizeRoutedLinearSF: unsupported input data type (expected kBF16 or kHALF).");
    }
}

void fp4BuildLayoutAndQuantizeRoutedLinearSFDecode(rt::Tensor const& input, rt::Tensor const& topkIds,
    rt::Tensor const& expertGlobalSF, MoELayoutBuffers& layoutBuffers, rt::Tensor& outputFP4, rt::Tensor& outputSF,
    int32_t localNumExperts, int32_t tileSize, cudaStream_t stream)
{
    int64_t const M = input.getShape()[0];
    int64_t const N = input.getShape()[1];
    int64_t const topK = topkIds.getShape()[1];
    if (M != 1)
    {
        throw std::runtime_error("fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: input M must be 1.");
    }
    if (topK <= 0 || topK > kMaxDecodeExperts)
    {
        throw std::runtime_error(
            "fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: topK must be in (0, kMaxDecodeExperts].");
    }
    if (localNumExperts <= 0 || localNumExperts > kMaxDecodeExperts)
    {
        throw std::runtime_error(
            "fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: localNumExperts must be in (0, kMaxDecodeExperts].");
    }
    if (tileSize <= 0)
    {
        throw std::runtime_error("fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: tileSize must be positive.");
    }
    if (N <= 0 || N % kSfVecSize != 0)
    {
        throw std::runtime_error(
            "fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: input N must be a positive multiple of 16.");
    }
    if (layoutBuffers.tileIdxToGroupIdx.getShape()[0] < topK || layoutBuffers.tileIdxToMnLimit.getShape()[0] < topK
        || layoutBuffers.permutedIdxToExpandedIdx.getShape()[0] < topK * tileSize)
    {
        throw std::runtime_error("fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: layout buffers are too small.");
    }

    int const numColThreads = static_cast<int>(N) / kEltsPerThread;
    int const blockSize = std::min(numColThreads, 512);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    int const numBlocksPerSM = std::max(1, kMaxResidentThreadsPerSm / blockSize);
    int const quantGridSize = std::min(static_cast<int>(topK), smCount * numBlocksPerSM);
    int const gridSize = quantGridSize + 1;

    auto const* topkPtr = static_cast<int32_t const*>(topkIds.rawPointer());
    auto const* sfPtrFwd = static_cast<float const*>(expertGlobalSF.rawPointer());
    auto* fp4Ptr = static_cast<uint32_t*>(outputFP4.rawPointer());
    auto* sfPtr = static_cast<uint8_t*>(outputSF.rawPointer());
    auto* tileGroupPtr = layoutBuffers.tileIdxToGroupIdx.dataPointer<int32_t>();
    auto* tileLimitPtr = layoutBuffers.tileIdxToMnLimit.dataPointer<int32_t>();
    auto* permutedPtr = layoutBuffers.permutedIdxToExpandedIdx.dataPointer<int32_t>();
    auto* numTilesPtr = layoutBuffers.numNonExitingTiles.dataPointer<int32_t>();

    if (input.getDataType() == nvinfer1::DataType::kBF16)
    {
        buildLayoutAndQuantizeRoutedToFp4LinearSfDecodeKernel<__nv_bfloat16>
            <<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(topK), static_cast<int32_t>(N), localNumExperts,
                tileSize, static_cast<__nv_bfloat16 const*>(input.rawPointer()), topkPtr, sfPtrFwd, permutedPtr,
                tileGroupPtr, tileLimitPtr, numTilesPtr, fp4Ptr, sfPtr);
    }
    else if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        buildLayoutAndQuantizeRoutedToFp4LinearSfDecodeKernel<__half>
            <<<gridSize, blockSize, 0, stream>>>(static_cast<int32_t>(topK), static_cast<int32_t>(N), localNumExperts,
                tileSize, static_cast<__half const*>(input.rawPointer()), topkPtr, sfPtrFwd, permutedPtr, tileGroupPtr,
                tileLimitPtr, numTilesPtr, fp4Ptr, sfPtr);
    }
    else
    {
        throw std::runtime_error(
            "fp4BuildLayoutAndQuantizeRoutedLinearSFDecode: unsupported input data type (expected kBF16 or kHALF).");
    }
}

#else // !SUPPORTS_FP8

void fp4Quantize(rt::Tensor const& /*input*/, rt::Tensor const& /*globalSF*/, rt::Tensor& /*outputFP4*/,
    rt::Tensor& /*outputSF*/, cudaStream_t /*stream*/)
{
    throw std::runtime_error(
        "FP4 quantize emits FP8 E4M3 scale factors but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
}

void fp4QuantizeLinearSF(rt::Tensor const& /*input*/, rt::Tensor const& /*globalSF*/, rt::Tensor& /*outputFP4*/,
    rt::Tensor& /*outputSF*/, cudaStream_t /*stream*/)
{
    throw std::runtime_error(
        "FP4 quantize emits FP8 E4M3 scale factors but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
}

void fp4QuantizeRoutedLinearSF(rt::Tensor const& /*input*/, rt::Tensor const& /*topkIds*/,
    rt::Tensor const& /*expertGlobalSF*/, rt::Tensor& /*outputFP4*/, rt::Tensor& /*outputSF*/, cudaStream_t /*stream*/)
{
    throw std::runtime_error(
        "FP4 quantize emits FP8 E4M3 scale factors but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
}

void fp4BuildLayoutAndQuantizeRoutedLinearSFDecode(rt::Tensor const& /*input*/, rt::Tensor const& /*topkIds*/,
    rt::Tensor const& /*expertGlobalSF*/, MoELayoutBuffers& /*layoutBuffers*/, rt::Tensor& /*outputFP4*/,
    rt::Tensor& /*outputSF*/, int32_t /*localNumExperts*/, int32_t /*tileSize*/, cudaStream_t /*stream*/)
{
    throw std::runtime_error(
        "FP4 quantize emits FP8 E4M3 scale factors but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
}

#endif // SUPPORTS_FP8

} // namespace kernel
} // namespace trt_edgellm
