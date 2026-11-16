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

#include "scaleConvert.h"

#include "common/checkMacros.h"
#include "common/logger.h"

#include <limits>

namespace trt_edgellm
{
namespace kernels
{

namespace
{

int checkedLaunchElementCount(int64_t count, char const* label)
{
    if (count <= 0)
    {
        return 0;
    }
    if (count > std::numeric_limits<int>::max())
    {
        LOG_ERROR("%s: element count %lld exceeds int32 launch capacity", label, static_cast<long long>(count));
        return 0;
    }
    return static_cast<int>(count);
}

} // namespace

// FP32 → UE4M3 (unsigned FP8 E4M3) conversion for SfAtom NVFP4 GEMM.
// UE4M3 = float_ue4m3_t = unsigned FP8 with 4-bit exponent + 3-bit mantissa.
// This matches TRT's native block-scaled FP4 GEMM scale encoding.
// Scale factors are always positive, so we use fabsf() and cast via __nv_fp8_e4m3.
#if SUPPORTS_FP8
__device__ __forceinline__ uint8_t fp32ToUe4m3(float val)
{
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(fabsf(val));
    return *reinterpret_cast<uint8_t*>(&fp8);
}
#endif // SUPPORTS_FP8

// ---------------------------------------------------------------------------
// Fused FP32 → UE4M3 + SfAtom repack (single kernel, no memset needed)
//
// Iterates over ALL atom slots (including padding), so the output buffer is
// fully written without requiring a preceding cudaMemsetAsync.
// Valid slots: convert FP32 → UE4M3 and write.  Padding slots: write 0.
// ---------------------------------------------------------------------------

#if SUPPORTS_FP8
__global__ void fusedFp32ToSfAtomKernel(float const* __restrict__ input, uint8_t* __restrict__ output, int numRows,
    int numKBlocks, int numNAtoms, int numKAtoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSlots = numNAtoms * 128 * numKAtoms * 4;
    if (idx >= totalSlots)
        return;

    int kAtomSlots = numKAtoms * 4;
    int n = idx / kAtomSlots;
    int k_sf = idx % kAtomSlots;

    int atom_n = n / 128;
    int atom_k = k_sf / 4;
    int local_n = n % 128;
    int local_k = k_sf % 4;
    int intra = (local_n % 32) * 16 + (local_n / 32) * 4 + local_k;
    int phys = (atom_n * numKAtoms + atom_k) * 512 + intra;

    if (n < numRows && k_sf < numKBlocks)
    {
        output[phys] = fp32ToUe4m3(input[n * numKBlocks + k_sf]);
    }
    else
    {
        output[phys] = 0;
    }
}

void fusedFp32ToSfAtom(
    float const* fp32Scales, uint8_t* tiledOut, int32_t numRows, int32_t numKBlocks, cudaStream_t stream)
{
    int numNAtoms = (numRows + 127) / 128;
    int numKAtoms = (numKBlocks + 3) / 4;
    int totalSlots = checkedLaunchElementCount(
        static_cast<int64_t>(numNAtoms) * 128 * static_cast<int64_t>(numKAtoms) * 4, "fusedFp32ToSfAtom");
    if (totalSlots == 0)
        return;

    constexpr int kBlockSize = 256;
    int blocks = (totalSlots + kBlockSize - 1) / kBlockSize;

    fusedFp32ToSfAtomKernel<<<blocks, kBlockSize, 0, stream>>>(
        fp32Scales, tiledOut, numRows, numKBlocks, numNAtoms, numKAtoms);
    CUDA_CHECK(cudaGetLastError());
}
#else
void fusedFp32ToSfAtom(
    float const* fp32Scales, uint8_t* tiledOut, int32_t numRows, int32_t numKBlocks, cudaStream_t stream)
{
    (void) fp32Scales;
    (void) tiledOut;
    (void) numRows;
    (void) numKBlocks;
    (void) stream;
    LOG_ERROR("fusedFp32ToSfAtom requires CUDA_VERSION >= 11080 (cuda_fp8.h unavailable).");
}
#endif // SUPPORTS_FP8

// ---------------------------------------------------------------------------
// E4M3 block-scale repack into the SfAtom layout (single kernel, no memset)
// ---------------------------------------------------------------------------

#if SUPPORTS_FP8
__global__ void repackE4m3ScalesToSfAtomKernel(__nv_fp8_e4m3 const* __restrict__ fp8Scales,
    uint8_t* __restrict__ output, int numRows, int numKBlocks, int numNAtoms, int numKAtoms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSlots = numNAtoms * 128 * numKAtoms * 4;
    if (idx >= totalSlots)
        return;

    int kAtomSlots = numKAtoms * 4;
    int n = idx / kAtomSlots;
    int k_sf = idx % kAtomSlots;

    int atom_n = n / 128;
    int atom_k = k_sf / 4;
    int local_n = n % 128;
    int local_k = k_sf % 4;
    int intra = (local_n % 32) * 16 + (local_n / 32) * 4 + local_k;
    int phys = (atom_n * numKAtoms + atom_k) * 512 + intra;

    if (n < numRows && k_sf < numKBlocks)
    {
        // A raw byte copy. UE4M3 and E4M3 encode the non-negative values a
        // checkpoint stores identically, so no arithmetic belongs here.
        output[phys] = reinterpret_cast<uint8_t const*>(fp8Scales)[n * numKBlocks + k_sf];
    }
    else
    {
        output[phys] = 0;
    }
}

void repackE4m3ScalesToSfAtom(
    __nv_fp8_e4m3 const* fp8Scales, uint8_t* tiledOut, int32_t numRows, int32_t numKBlocks, cudaStream_t stream)
{
    int numNAtoms = (numRows + 127) / 128;
    int numKAtoms = (numKBlocks + 3) / 4;
    int totalSlots = checkedLaunchElementCount(
        static_cast<int64_t>(numNAtoms) * 128 * static_cast<int64_t>(numKAtoms) * 4, "repackE4m3ScalesToSfAtom");
    if (totalSlots == 0)
        return;

    constexpr int kBlockSize = 256;
    int blocks = (totalSlots + kBlockSize - 1) / kBlockSize;

    repackE4m3ScalesToSfAtomKernel<<<blocks, kBlockSize, 0, stream>>>(
        fp8Scales, tiledOut, numRows, numKBlocks, numNAtoms, numKAtoms);
    CUDA_CHECK(cudaGetLastError());
}
#endif // SUPPORTS_FP8

} // namespace kernels
} // namespace trt_edgellm
