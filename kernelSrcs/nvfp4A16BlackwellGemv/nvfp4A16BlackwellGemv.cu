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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#if !defined(GEMV_N) || !defined(GEMV_K) || !defined(GEMV_DATA_TYPE) || !defined(GEMV_LAYOUT_ABI)                      \
    || !defined(GEMV_SOURCE_ABI)
#error "NVFP4-A16 Blackwell GEMV JIT configuration is incomplete"
#endif

#if GEMV_LAYOUT_ABI != 1
#error "Unsupported NVFP4-A16 Blackwell GEMV layout ABI"
#endif

#if GEMV_SOURCE_ABI != 4
#error "Unsupported NVFP4-A16 Blackwell GEMV source ABI"
#endif

namespace
{

constexpr int kWARP_SIZE{32};
constexpr int kWARPS_PER_BLOCK{8};
constexpr int kTHREADS_PER_BLOCK{kWARP_SIZE * kWARPS_PER_BLOCK};
constexpr int kN_TILE{128};
constexpr int kK_TILE{64};
constexpr int kROWS_PER_WARP{16};
constexpr int kPACKED_BYTES_PER_ROW_TILE{kK_TILE / 2};
constexpr int kSCALES_PER_ROW_TILE{kK_TILE / 16};
constexpr int kPACKED_BYTES_PER_THREAD{kPACKED_BYTES_PER_ROW_TILE / 2};
constexpr int kSMEM_HALF_STRIDE{kK_TILE / 2 + 4};
constexpr int kSMEM_ROW_STRIDE{kK_TILE + 4};

#if GEMV_DATA_TYPE == 0
using GemvType = half;
#elif GEMV_DATA_TYPE == 1
using GemvType = __nv_bfloat16;
#else
#error "Unsupported NVFP4-A16 Blackwell GEMV data type"
#endif

struct __align__(16) Uint4
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

__device__ __forceinline__ Uint4 loadGlobal128NoAllocate(void const* const pointer)
{
    Uint4 value{};
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer));
    return value;
}

__device__ __forceinline__ Uint4 loadGlobal128Retain(void const* const pointer)
{
    Uint4 value{};
    asm volatile("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer));
    return value;
}

__device__ __forceinline__ float2 fp4PairToFloat2(unsigned int const packedByte)
{
    __half2_raw const raw = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packedByte), __NV_E2M1);
    return __half22float2(static_cast<half2>(raw));
}

__device__ __forceinline__ float2 fp8PairToFloat2(unsigned short const packedBytes)
{
    __half2_raw const raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packedBytes), __NV_E4M3);
    return __half22float2(static_cast<half2>(raw));
}

template <typename T>
__device__ __forceinline__ float2 activationPairToFloat2(T const* pointer);

#if GEMV_DATA_TYPE == 0
template <>
__device__ __forceinline__ float2 activationPairToFloat2<half>(half const* const pointer)
{
    return __half22float2(*reinterpret_cast<half2 const*>(pointer));
}
#else
template <>
__device__ __forceinline__ float2 activationPairToFloat2<__nv_bfloat16>(__nv_bfloat16 const* const pointer)
{
    return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(pointer));
}
#endif

template <typename T>
__device__ __forceinline__ T outputFromFloat(float value);

#if GEMV_DATA_TYPE == 0
template <>
__device__ __forceinline__ half outputFromFloat<half>(float const value)
{
    return __float2half_rn(value);
}
#else
template <>
__device__ __forceinline__ __nv_bfloat16 outputFromFloat<__nv_bfloat16>(float const value)
{
    return __float2bfloat16_rn(value);
}
#endif

template <typename T>
__device__ __forceinline__ void storeOutputPair(T* pointer, float2 value);

#if GEMV_DATA_TYPE == 0
template <>
__device__ __forceinline__ void storeOutputPair<half>(half* const pointer, float2 const value)
{
    *reinterpret_cast<half2*>(pointer) = __floats2half2_rn(value.x, value.y);
}
#else
template <>
__device__ __forceinline__ void storeOutputPair<__nv_bfloat16>(__nv_bfloat16* const pointer, float2 const value)
{
    *reinterpret_cast<__nv_bfloat162*>(pointer) = __floats2bfloat162_rn(value.x, value.y);
}
#endif

template <int M, int SPLIT_K>
__device__ __forceinline__ void gemvBody(GemvType const* __restrict__ activation,
    unsigned char const* __restrict__ qweights, unsigned char const* __restrict__ blockScales,
    float const* __restrict__ globalScale, GemvType* __restrict__ output, float* __restrict__ partials,
    int const runtimeSplitK, float* const sharedActivation)
{
    static_assert(GEMV_N > 0);
    static_assert(GEMV_K > 0);
    static_assert(GEMV_N % kN_TILE == 0);
    static_assert(GEMV_K % kK_TILE == 0);
    static_assert(M == 1 || M == 2 || M == 4 || M == 8 || M == 16);
    // SPLIT_K == 0 deliberately keeps split-K as a runtime value. The M8/M16
    // split-K=1 path uses that form because it preserves the locked baseline's
    // peeled-first-tile/steady-state loop schedule on SM110.
    static_assert(SPLIT_K == 0 || SPLIT_K == 1 || SPLIT_K == 2 || SPLIT_K == 4);

    int const lane = static_cast<int>(threadIdx.x) & (kWARP_SIZE - 1);
    int const warp = static_cast<int>(threadIdx.x) / kWARP_SIZE;
    int const rowInTile = warp * kROWS_PER_WARP + lane / 2;
    int const kHalf = lane & 1;
    int const nBlock = static_cast<int>(blockIdx.x);
    int const outputColumn = nBlock * kN_TILE + rowInTile;
    constexpr int kK_BLOCKS{GEMV_K / kK_TILE};
    int const split = static_cast<int>(blockIdx.y);
    int const splitK = SPLIT_K == 0 ? runtimeSplitK : SPLIT_K;
    int const kBlockBegin = kK_BLOCKS * split / splitK;
    int const kBlockEnd = kK_BLOCKS * (split + 1) / splitK;
    float accumulators[M]{};
    GemvType const* activationKBlock = activation + static_cast<long long>(kBlockBegin) * kK_TILE;

    for (int kBlock = kBlockBegin; kBlock < kBlockEnd; ++kBlock)
    {
        constexpr int kELEMENTS_PER_VECTOR{static_cast<int>(sizeof(Uint4) / sizeof(GemvType))};
        constexpr int kVECTORS_PER_TILE{M * kK_TILE / kELEMENTS_PER_VECTOR};
        for (int vector = static_cast<int>(threadIdx.x); vector < kVECTORS_PER_TILE; vector += kTHREADS_PER_BLOCK)
        {
            int const activationRow = vector / (kK_TILE / kELEMENTS_PER_VECTOR);
            int const vectorInRow = vector % (kK_TILE / kELEMENTS_PER_VECTOR);
            GemvType const* const source = activationKBlock + static_cast<long long>(activationRow) * GEMV_K
                + vectorInRow * kELEMENTS_PER_VECTOR;
            Uint4 const packedActivation = loadGlobal128Retain(source);
            GemvType const* const packedElements = reinterpret_cast<GemvType const*>(&packedActivation);
            int const logicalK = vectorInRow * kELEMENTS_PER_VECTOR;
            int const stagedK = logicalK + (logicalK >= kK_TILE / 2 ? 4 : 0);
            float2* const stagedPairs
                = reinterpret_cast<float2*>(sharedActivation + activationRow * kSMEM_ROW_STRIDE + stagedK);
#pragma unroll
            for (int pair = 0; pair < kELEMENTS_PER_VECTOR / 2; ++pair)
            {
                stagedPairs[pair] = activationPairToFloat2(packedElements + pair * 2);
            }
        }
        __syncthreads();

        long long const rowTileIndex = (static_cast<long long>(nBlock) * kK_BLOCKS + kBlock) * kN_TILE + rowInTile;
        unsigned char const* const packedPointer
            = qweights + rowTileIndex * kPACKED_BYTES_PER_ROW_TILE + kHalf * kPACKED_BYTES_PER_THREAD;
        Uint4 const packedWeights = loadGlobal128NoAllocate(packedPointer);
        unsigned short const packedScales
            = *reinterpret_cast<unsigned short const*>(blockScales + rowTileIndex * kSCALES_PER_ROW_TILE + kHalf * 2);
        float2 const scales = fp8PairToFloat2(packedScales);
        unsigned int const words[4]{packedWeights.x, packedWeights.y, packedWeights.z, packedWeights.w};

#pragma unroll
        for (int word = 0; word < 4; ++word)
        {
#pragma unroll
            for (int byte = 0; byte < 4; ++byte)
            {
                int const pair = word * 4 + byte;
                unsigned int const packedByte = (words[word] >> (byte * 8)) & 0xFFU;
                float2 const weights = fp4PairToFloat2(packedByte);
                float const scale = pair < 8 ? scales.x : scales.y;
                int const kInTile = kHalf * kSMEM_HALF_STRIDE + pair * 2;
#pragma unroll
                for (int activationRow = 0; activationRow < M; ++activationRow)
                {
                    float2 const values = *reinterpret_cast<float2 const*>(
                        sharedActivation + activationRow * kSMEM_ROW_STRIDE + kInTile);
                    float const dot = fmaf(values.x, weights.x, values.y * weights.y);
                    accumulators[activationRow] = fmaf(dot, scale, accumulators[activationRow]);
                }
            }
        }
        activationKBlock += kK_TILE;
        __syncthreads();
    }

#pragma unroll
    for (int activationRow = 0; activationRow < M; ++activationRow)
    {
        accumulators[activationRow] += __shfl_xor_sync(0xFFFFFFFFU, accumulators[activationRow], 1, kWARP_SIZE);
        if (kHalf == 0)
        {
            long long const outputIndex = static_cast<long long>(activationRow) * GEMV_N + outputColumn;
            bool const directOutput = SPLIT_K == 0 ? splitK == 1 : SPLIT_K == 1;
            if (directOutput)
            {
                output[outputIndex] = outputFromFloat<GemvType>(accumulators[activationRow] * globalScale[0]);
            }
            else
            {
                long long const partialIndex
                    = (static_cast<long long>(split) * M + activationRow) * GEMV_N + outputColumn;
                partials[partialIndex] = accumulators[activationRow];
            }
        }
    }
}

template <int M, int SPLIT_K>
__device__ __forceinline__ void reduceBody(
    float const* __restrict__ partials, float const* __restrict__ globalScale, GemvType* __restrict__ output)
{
    long long const outputPair
        = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
    constexpr long long kOUTPUT_PAIRS{static_cast<long long>(M) * GEMV_N / 2};
    if (outputPair >= kOUTPUT_PAIRS)
    {
        return;
    }

    long long const outputIndex = outputPair * 2;
    float2 accumulator{};
#pragma unroll
    for (int split = 0; split < SPLIT_K; ++split)
    {
        float2 const values
            = *reinterpret_cast<float2 const*>(partials + static_cast<long long>(split) * M * GEMV_N + outputIndex);
        accumulator.x += values.x;
        accumulator.y += values.y;
    }
    float const scale = globalScale[0];
    storeOutputPair(output + outputIndex, make_float2(accumulator.x * scale, accumulator.y * scale));
}

} // namespace

#define DEFINE_GEMV_SPLIT_ENTRY(M, SPLIT_K)                                                                            \
    extern "C" __global__ __launch_bounds__(kTHREADS_PER_BLOCK, 1) void nvfp4_a16_blackwell_gemv_m##M##_sk##SPLIT_K(   \
        GemvType const* __restrict__ activation, unsigned char const* __restrict__ qweights,                           \
        unsigned char const* __restrict__ blockScales, float const* __restrict__ globalScale,                          \
        GemvType* __restrict__ output, float* __restrict__ partials)                                                   \
    {                                                                                                                  \
        __shared__ __align__(16) float sharedActivation[M * kSMEM_ROW_STRIDE];                                         \
        gemvBody<M, SPLIT_K>(                                                                                          \
            activation, qweights, blockScales, globalScale, output, partials, SPLIT_K, sharedActivation);              \
    }                                                                                                                  \
    extern "C" __global__ void nvfp4_a16_blackwell_gemv_reduce_m##M##_sk##SPLIT_K(                                     \
        float const* __restrict__ partials, float const* __restrict__ globalScale, GemvType* __restrict__ output)      \
    {                                                                                                                  \
        reduceBody<M, SPLIT_K>(partials, globalScale, output);                                                         \
    }

#define DEFINE_GEMV_RUNTIME_ENTRY(M)                                                                                   \
    extern "C" __global__ __launch_bounds__(kTHREADS_PER_BLOCK, 1) void nvfp4_a16_blackwell_gemv_m##M##_runtime(       \
        GemvType const* __restrict__ activation, unsigned char const* __restrict__ qweights,                           \
        unsigned char const* __restrict__ blockScales, float const* __restrict__ globalScale,                          \
        GemvType* __restrict__ output, float* __restrict__ partials, int splitK)                                       \
    {                                                                                                                  \
        __shared__ __align__(16) float sharedActivation[M * kSMEM_ROW_STRIDE];                                         \
        gemvBody<M, 0>(activation, qweights, blockScales, globalScale, output, partials, splitK, sharedActivation);    \
    }

#define DEFINE_GEMV_ENTRIES(M)                                                                                         \
    DEFINE_GEMV_SPLIT_ENTRY(M, 1)                                                                                      \
    DEFINE_GEMV_SPLIT_ENTRY(M, 2)                                                                                      \
    DEFINE_GEMV_SPLIT_ENTRY(M, 4)

DEFINE_GEMV_ENTRIES(1)
DEFINE_GEMV_ENTRIES(2)
DEFINE_GEMV_ENTRIES(4)
DEFINE_GEMV_ENTRIES(8)
DEFINE_GEMV_ENTRIES(16)
DEFINE_GEMV_RUNTIME_ENTRY(8)

extern "C" __global__ __maxnreg__(88) void nvfp4_a16_blackwell_gemv_m16_runtime(GemvType const* __restrict__ activation,
    unsigned char const* __restrict__ qweights, unsigned char const* __restrict__ blockScales,
    float const* __restrict__ globalScale, GemvType* __restrict__ output, float* __restrict__ partials, int splitK)
{
    __shared__ __align__(16) float sharedActivation[16 * kSMEM_ROW_STRIDE];
    gemvBody<16, 0>(activation, qweights, blockScales, globalScale, output, partials, splitK, sharedActivation);
}

#undef DEFINE_GEMV_RUNTIME_ENTRY
#undef DEFINE_GEMV_ENTRIES
#undef DEFINE_GEMV_SPLIT_ENTRY
