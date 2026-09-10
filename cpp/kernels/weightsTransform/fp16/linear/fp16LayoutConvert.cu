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

#include "fp16LayoutConvert.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int kTile = 32;

template <typename T>
__global__ void copyKernel(T const* source, T* destination, int64_t count)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride)
    {
        destination[index] = source[index];
    }
}

__global__ void copy2DBytesKernel(uint8_t const* source, uint8_t* destination, int64_t count, size_t sourceRowBytes,
    size_t sourceColumnOffsetBytes, size_t destinationRowBytes)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride)
    {
        // Flatten the compact destination shard to one byte per thread. Each
        // destination row maps to one contiguous byte window in the wider source row.
        size_t const destinationRow = static_cast<size_t>(index) / destinationRowBytes;
        size_t const destinationColumnByte = static_cast<size_t>(index) % destinationRowBytes;
        destination[index] = source[destinationRow * sourceRowBytes + sourceColumnOffsetBytes + destinationColumnByte];
    }
}

template <typename Source>
__global__ void sliceToFp16Kernel(Source const* source, __half* destination, int64_t count, int64_t sourceColumns,
    int64_t sourceColumnOffset, int64_t destinationColumns)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride)
    {
        int64_t const row = index / destinationColumns;
        int64_t const column = index % destinationColumns;
        if constexpr (std::is_same_v<Source, __nv_bfloat16>)
        {
            destination[index]
                = __float2half(__bfloat162float(source[row * sourceColumns + sourceColumnOffset + column]));
        }
        else
        {
            destination[index] = __float2half(source[row * sourceColumns + sourceColumnOffset + column]);
        }
    }
}

__global__ void bf16ToFp16Kernel(__nv_bfloat16 const* src, __half* dst, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        dst[i] = __float2half(__bfloat162float(src[i]));
    }
}

__global__ void fp32ToFp16Kernel(float const* src, __half* dst, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void bf16ToFp32Kernel(__nv_bfloat16 const* src, float* dst, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        dst[i] = __bfloat162float(src[i]);
    }
}

__global__ void fp16ToFp32Kernel(__half const* src, float* dst, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        dst[i] = __half2float(src[i]);
    }
}

//! Tiled transpose for FP16 (and fused cast variants).
template <typename SrcT, bool kCastBf16, bool kCastFp32>
__global__ void transposeCastKernel(SrcT const* src, __half* dst, int32_t rows, int32_t cols)
{
    __shared__ __half tile[kTile][kTile + 1];

    int32_t const x = blockIdx.x * kTile + threadIdx.x;
    int32_t const y = blockIdx.y * kTile + threadIdx.y;

    if (x < cols && y < rows)
    {
        SrcT v = src[static_cast<int64_t>(y) * cols + x];
        if constexpr (kCastBf16)
        {
            tile[threadIdx.y][threadIdx.x]
                = __float2half(__bfloat162float(*reinterpret_cast<__nv_bfloat16 const*>(&v)));
        }
        else if constexpr (kCastFp32)
        {
            tile[threadIdx.y][threadIdx.x] = __float2half(*reinterpret_cast<float const*>(&v));
        }
        else
        {
            tile[threadIdx.y][threadIdx.x] = *reinterpret_cast<__half const*>(&v);
        }
    }
    __syncthreads();

    int32_t const tx = blockIdx.y * kTile + threadIdx.x;
    int32_t const ty = blockIdx.x * kTile + threadIdx.y;
    if (tx < rows && ty < cols)
    {
        dst[static_cast<int64_t>(ty) * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

inline int32_t grid1d(int64_t n, int32_t threads)
{
    int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535)
    {
        blocks = 65535;
    }
    return static_cast<int32_t>(blocks);
}

__global__ void scaleFp16Kernel(__half* data, int64_t n, float scale)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        data[i] = __float2half(__half2float(data[i]) * scale);
    }
}

__global__ void fillFp32Kernel(float* data, int64_t n, float value)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride)
    {
        data[i] = value;
    }
}

struct Fp32ValueBatch
{
    float values[512];
};

__global__ void writeFp32Kernel(Fp32ValueBatch values, float* output, int32_t count)
{
    int32_t const index = static_cast<int32_t>(threadIdx.x);
    if (index < count)
    {
        output[index] = values.values[index];
    }
}

} // namespace

cudaError_t launchCopyBytes(void const* source, void* destination, size_t bytes, cudaStream_t stream)
{
    if (bytes == 0)
    {
        return cudaSuccess;
    }
    if (source == nullptr || destination == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    int32_t constexpr threads = 256;
    auto const sourceAddress = reinterpret_cast<uintptr_t>(source);
    auto const destinationAddress = reinterpret_cast<uintptr_t>(destination);
    if (((sourceAddress | destinationAddress | bytes) & (alignof(uint4) - 1)) == 0)
    {
        int64_t const count = static_cast<int64_t>(bytes / sizeof(uint4));
        copyKernel<<<grid1d(count, threads), threads, 0, stream>>>(
            static_cast<uint4 const*>(source), static_cast<uint4*>(destination), count);
    }
    else if (((sourceAddress | destinationAddress | bytes) & (alignof(uint32_t) - 1)) == 0)
    {
        int64_t const count = static_cast<int64_t>(bytes / sizeof(uint32_t));
        copyKernel<<<grid1d(count, threads), threads, 0, stream>>>(
            static_cast<uint32_t const*>(source), static_cast<uint32_t*>(destination), count);
    }
    else
    {
        int64_t const count = static_cast<int64_t>(bytes);
        copyKernel<<<grid1d(count, threads), threads, 0, stream>>>(
            static_cast<uint8_t const*>(source), static_cast<uint8_t*>(destination), count);
    }
    return cudaGetLastError();
}

cudaError_t launchCopy2DBytes(void const* source, void* destination, int64_t rows, size_t sourceRowBytes,
    size_t sourceColumnOffsetBytes, size_t destinationRowBytes, cudaStream_t stream)
{
    if (rows <= 0 || destinationRowBytes == 0)
    {
        return cudaSuccess;
    }
    if (source == nullptr || destination == nullptr || sourceColumnOffsetBytes + destinationRowBytes > sourceRowBytes)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const count = rows * static_cast<int64_t>(destinationRowBytes);
    int32_t constexpr threads = 256;
    copy2DBytesKernel<<<grid1d(count, threads), threads, 0, stream>>>(static_cast<uint8_t const*>(source),
        static_cast<uint8_t*>(destination), count, sourceRowBytes, sourceColumnOffsetBytes, destinationRowBytes);
    return cudaGetLastError();
}

cudaError_t launchBf16SliceToFp16(void const* source, void* destination, int64_t rows, int64_t sourceColumns,
    int64_t sourceColumnOffset, int64_t destinationColumns, cudaStream_t stream)
{
    if (rows <= 0 || destinationColumns <= 0)
    {
        return cudaSuccess;
    }
    if (source == nullptr || destination == nullptr || sourceColumnOffset < 0
        || sourceColumnOffset + destinationColumns > sourceColumns)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const count = rows * destinationColumns;
    int32_t constexpr threads = 256;
    sliceToFp16Kernel<<<grid1d(count, threads), threads, 0, stream>>>(static_cast<__nv_bfloat16 const*>(source),
        static_cast<__half*>(destination), count, sourceColumns, sourceColumnOffset, destinationColumns);
    return cudaGetLastError();
}

cudaError_t launchFp32SliceToFp16(void const* source, void* destination, int64_t rows, int64_t sourceColumns,
    int64_t sourceColumnOffset, int64_t destinationColumns, cudaStream_t stream)
{
    if (rows <= 0 || destinationColumns <= 0)
    {
        return cudaSuccess;
    }
    if (source == nullptr || destination == nullptr || sourceColumnOffset < 0
        || sourceColumnOffset + destinationColumns > sourceColumns)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const count = rows * destinationColumns;
    int32_t constexpr threads = 256;
    sliceToFp16Kernel<<<grid1d(count, threads), threads, 0, stream>>>(static_cast<float const*>(source),
        static_cast<__half*>(destination), count, sourceColumns, sourceColumnOffset, destinationColumns);
    return cudaGetLastError();
}

cudaError_t launchBf16ToFp16(void const* dBf16, void* dFp16, int64_t n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    bf16ToFp16Kernel<<<grid1d(n, threads), threads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(dBf16), static_cast<__half*>(dFp16), n);
    return cudaGetLastError();
}

cudaError_t launchFp32ToFp16(void const* dFp32, void* dFp16, int64_t n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    fp32ToFp16Kernel<<<grid1d(n, threads), threads, 0, stream>>>(
        static_cast<float const*>(dFp32), static_cast<__half*>(dFp16), n);
    return cudaGetLastError();
}

cudaError_t launchBf16ToFp32(void const* dBf16, void* dFp32, int64_t n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    bf16ToFp32Kernel<<<grid1d(n, threads), threads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(dBf16), static_cast<float*>(dFp32), n);
    return cudaGetLastError();
}

cudaError_t launchFp16ToFp32(void const* dFp16, void* dFp32, int64_t n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    fp16ToFp32Kernel<<<grid1d(n, threads), threads, 0, stream>>>(
        static_cast<__half const*>(dFp16), static_cast<float*>(dFp32), n);
    return cudaGetLastError();
}

cudaError_t launchTransposeFp16(
    void const* dSrcRowsCols, void* dDstColsRows, int32_t rows, int32_t cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    dim3 block(kTile, kTile);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transposeCastKernel<__half, false, false><<<grid, block, 0, stream>>>(
        static_cast<__half const*>(dSrcRowsCols), static_cast<__half*>(dDstColsRows), rows, cols);
    return cudaGetLastError();
}

cudaError_t launchBf16TransposeToFp16(
    void const* dSrcBf16RowsCols, void* dDstFp16ColsRows, int32_t rows, int32_t cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    dim3 block(kTile, kTile);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transposeCastKernel<__nv_bfloat16, true, false><<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(dSrcBf16RowsCols), static_cast<__half*>(dDstFp16ColsRows), rows, cols);
    return cudaGetLastError();
}

cudaError_t launchFp32TransposeToFp16(
    void const* dSrcFp32RowsCols, void* dDstFp16ColsRows, int32_t rows, int32_t cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    dim3 block(kTile, kTile);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transposeCastKernel<float, false, true><<<grid, block, 0, stream>>>(
        static_cast<float const*>(dSrcFp32RowsCols), static_cast<__half*>(dDstFp16ColsRows), rows, cols);
    return cudaGetLastError();
}

cudaError_t launchScaleFp16(void* dFp16, int64_t n, float scale, cudaStream_t stream)
{
    if (n <= 0 || scale == 1.f)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    scaleFp16Kernel<<<grid1d(n, threads), threads, 0, stream>>>(static_cast<__half*>(dFp16), n, scale);
    return cudaGetLastError();
}

cudaError_t launchFillFp32(void* dFp32, int64_t n, float value, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    int32_t const threads = 256;
    fillFp32Kernel<<<grid1d(n, threads), threads, 0, stream>>>(static_cast<float*>(dFp32), n, value);
    return cudaGetLastError();
}

cudaError_t launchWriteFp32(float const* values, int32_t count, void* dFp32, cudaStream_t stream)
{
    if (values == nullptr || dFp32 == nullptr || count <= 0 || count > 512)
    {
        return cudaErrorInvalidValue;
    }
    Fp32ValueBatch batch{};
    for (int32_t index = 0; index < count; ++index)
    {
        batch.values[index] = values[index];
    }
    writeFp32Kernel<<<1, 512, 0, stream>>>(batch, static_cast<float*>(dFp32), count);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
