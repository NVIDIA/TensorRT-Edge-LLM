/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utilKernels.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace kernel
{

__global__ void calCuQCuKVSeqLensAndKVEndIdxsKernel(int32_t const* inputSeqLen, int32_t const* kvCacheStartIndices,
    int32_t* cuQSeqlen, int32_t* cuKVSeqLens, int32_t* kvCacheEndIndices, int32_t runtimeSeqLen, int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cuQSeqlen[0] = 0;
        cuKVSeqLens[0] = 0;

        int32_t runningCuSeqLen = 0;
        int32_t runningCuKvCacheLen = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            runningCuSeqLen += inputSeqLen[i];
            cuQSeqlen[i + 1] = runningCuSeqLen;

            int32_t kvCacheStartIdx = 0;
            if (kvCacheStartIndices != nullptr)
            {
                kvCacheStartIdx = kvCacheStartIndices[i];
            }

            runningCuKvCacheLen += (kvCacheStartIdx + inputSeqLen[i]);
            cuKVSeqLens[i + 1] = runningCuKvCacheLen;
            // To keep semantic consistency with the packed QKV layout for RoPE, use runtimeSeqLen here.
            kvCacheEndIndices[i] = kvCacheStartIdx + runtimeSeqLen;
        }
    }
}

// ===== kernel: produce [B, S, 2, H, D] (FMHA expected padded layout) =====
template <typename T>
__global__ void cvtKVLayoutBHSDToBSHDKernel(T const* __restrict__ src, // [B, 2, H, S, D]
    T* __restrict__ dst,                                               // [B, S, 2, H, D]
    int32_t B, int32_t S, int32_t H, int32_t D)
{
    // Thread mapping identical to paddedLayoutToCompactKernel but without cuSeqLens.
    //   x-dim: feature dimension  D
    //   y-dim: sequence/token     S
    //   z-dim: (batch, headPair)  batch * numHpBlocks + hpBlock

    uint32_t const token = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. S-1
    uint32_t const d = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. D-1

    // Decode batch index and head-pair tile from z-dimension
    uint32_t const numHpBlocks = (2 * H + blockDim.z - 1) / blockDim.z; // number of head-pair tiles per batch
    uint32_t const batch = blockIdx.z / numHpBlocks;                    // 0 .. B-1
    uint32_t const hpTile = blockIdx.z % numHpBlocks;                   // 0 .. numHpBlocks-1
    uint32_t const headPair = hpTile * blockDim.z + threadIdx.z;        // 0 .. 2*H-1

    if (batch >= B || headPair >= 2 * H || d >= D || token >= S)
        return;

    // ---------- flat indices ----------
    // src layout: [B, 2, H, S, D] -> ((((b * 2 + kv) * H + h) * S + token) * D + d)
    uint32_t const kv = headPair / H; // 0 = K, 1 = V
    uint32_t const h = headPair % H;  // head index
    size_t srcIdx = (((((size_t) batch * 2 + kv) * H + h) * S + token) * D + d);

    // dst layout: [B, S, 2, H, D] -> ((((b * S + token) * 2 + kv) * H + h) * D + d)
    size_t dstIdx = (((((size_t) batch * S + token) * 2 + kv) * H + h) * D + d);

    dst[dstIdx] = src[srcIdx];
}

void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs, int32_t const runtimeSeqLen,
    cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);

    // Perform necessary shape checks.
    check::check(cuQSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheEndIdxs.getShape()[0] == runtimeBatchSize, "kvCacheEndIdxs shall have shape [B].");

    if (!kvCacheStartIndices.isEmpty())
    {
        check::check(
            kvCacheStartIndices.getShape()[0] == runtimeBatchSize, "KVCacheStartIndices tensor shall have shape [B].");
    }
    else
    {
        // We rely on this nullptr behavior to indicate whether kvCacheStartIndices is available in the kernel.
        check::check(kvCacheStartIndices.rawPointer() == nullptr,
            "KVCacheStartIndices tensor shall be nullptr when it is empty.");
    }

    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), kvCacheEndIdxs.dataPointer<int32_t>(), runtimeSeqLen, runtimeBatchSize);
}

void cvtKVLayoutBHSDToBSHD(rt::Tensor const& src, rt::Tensor& dst, cudaStream_t stream)
{
    rt::Coords srcShape = src.getShape();
    int32_t const B = static_cast<int32_t>(srcShape[0]);
    int32_t const H = static_cast<int32_t>(srcShape[2]);
    int32_t const S = static_cast<int32_t>(srcShape[3]);
    int32_t const D = static_cast<int32_t>(srcShape[4]);

    // Perform necessary shape checks.
    rt::Coords dstShape = dst.getShape();
    check::check(src.getDataType() == nvinfer1::DataType::kHALF && dst.getDataType() == nvinfer1::DataType::kHALF,
        "Restrict input and output data types to FP16 for now.");
    check::check(srcShape[1] == 2 && dstShape[2] == 2, "Source and destination tensors separate KV respectively.");
    check::check(dstShape[0] == B && dstShape[1] == S && dstShape[3] == H && dstShape[4] == D,
        "Destination tensor shall have consistent shape of [B, S, 2, H, D].");

    // Block config with safe thread count (≤ 1024)
    uint32_t const tx = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    uint32_t const ty = 4; // token dimension per block
    uint32_t const tz = 1; // process one head-pair per thread in z
    dim3 block(tx, ty, tz);

    // Grid config
    uint32_t const hpTilesPerBatch = (2 * H + tz - 1) / tz; // z-blocks needed per batch for head-pairs
    dim3 grid((D + tx - 1) / tx,                            // x : feature dim
        (S + ty - 1) / ty,                                  // y : token dim
        hpTilesPerBatch * B);                               // z : (batch, headPair)

    cvtKVLayoutBHSDToBSHDKernel<half>
        <<<grid, block, 0, stream>>>(src.dataPointer<half>(), dst.dataPointer<half>(), B, S, H, D);
}

} // namespace kernel
} // namespace trt_edgellm
